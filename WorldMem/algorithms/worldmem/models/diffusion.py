from typing import Optional, Callable
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, extract
from .dit import DiT_models

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])


class Diffusion(nn.Module):
    # Special thanks to lucidrains for the implementation of the base Diffusion model
    # https://github.com/lucidrains/denoising-diffusion-pytorch

    def __init__(
        self,
        x_shape: torch.Size,
        reference_length: int,
        action_cond_dim: int,
        pose_cond_dim,
        is_causal: bool,
        cfg: DictConfig,
        is_dit: bool=False,
        use_plucker=False,
        relative_embedding=False,
        state_embed_only_on_qk=False,
        use_memory_attention=False,
        add_timestamp_embedding=False,
        ref_mode='sequential'
    ):
        super().__init__()
        self.cfg = cfg

        self.x_shape = x_shape
        self.action_cond_dim = action_cond_dim
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_schedule = cfg.beta_schedule
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.objective = cfg.objective
        self.use_fused_snr = cfg.use_fused_snr
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise
        self.arch = cfg.architecture
        self.stabilization_level = cfg.stabilization_level
        self.is_causal = is_causal
        self.is_dit = is_dit
        self.reference_length = reference_length
        self.pose_cond_dim = pose_cond_dim
        self.use_plucker = use_plucker
        self.relative_embedding = relative_embedding
        self.state_embed_only_on_qk = state_embed_only_on_qk
        self.use_memory_attention = use_memory_attention
        self.add_timestamp_embedding = add_timestamp_embedding
        self.ref_mode = ref_mode

        self._build_model()
        self._build_buffer()

    def _build_model(self):
        x_channel = self.x_shape[0]
        if self.is_dit:
            self.model = DiT_models["DiT-S/2"](action_cond_dim=self.action_cond_dim,
                                            pose_cond_dim=self.pose_cond_dim, reference_length=self.reference_length,
                                            use_plucker=self.use_plucker,
                                            relative_embedding=self.relative_embedding,
                                            state_embed_only_on_qk=self.state_embed_only_on_qk,
                                            use_memory_attention=self.use_memory_attention,
                                            add_timestamp_embedding=self.add_timestamp_embedding,
                                            ref_mode=self.ref_mode)
        else:
            raise NotImplementedError

    def _build_buffer(self):
        if self.beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif self.beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif self.beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {self.beta_schedule}")

        betas = beta_schedule_fn(self.timesteps, **self.schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting

        # register_buffer(
        #     "p2_loss_weight",
        #     (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
        #     ** -self.p2_loss_weight_gamma,
        # )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(self, x, t, action_cond=None, current_frame=None, 
        pose_cond=None, mode="training", reference_length=None, frame_idx=None):
        x = x.permute(1,0,2,3,4)
        action_cond = action_cond.permute(1,0,2)
        if pose_cond is not None and pose_cond[0] is not None:
            try:
                pose_cond = pose_cond.permute(1,0,2)
            except:
                pass
        t = t.permute(1,0)
        model_output = self.model(x, t, action_cond, current_frame=current_frame, pose_cond=pose_cond, 
            mode=mode, reference_length=reference_length, frame_idx=frame_idx)
        model_output = model_output.permute(1,0,2,3,4)
        x = x.permute(1,0,2,3,4)
        t = t.permute(1,0)        

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)


        return ModelPrediction(pred_noise, x_start, model_output)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, x, t, action_cond=None, pose_cond=None, reference_length=None):
        model_pred = self.model_predictions(x=x, t=t, action_cond=action_cond, 
            pose_cond=pose_cond, reference_length=reference_length,
            frame_idx=frame_idx)
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_t=x, t=t)

    def compute_loss_weights(self, noise_levels: torch.Tensor):

        snr = self.snr[noise_levels]
        clipped_snr = self.clipped_snr[noise_levels]
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        if not self.use_fused_snr:
            # min SNR reweighting
            match self.objective:
                case "pred_noise":
                    return clipped_snr / snr
                case "pred_x0":
                    return clipped_snr
                case "pred_v":
                    return clipped_snr / (snr + 1)

        cum_snr = torch.zeros_like(normalized_snr)
        for t in range(0, noise_levels.shape[0]):
            if t == 0:
                cum_snr[t] = normalized_clipped_snr[t]
            else:
                cum_snr[t] = self.cum_snr_decay * cum_snr[t - 1] + (1 - self.cum_snr_decay) * normalized_clipped_snr[t]

        cum_snr = F.pad(cum_snr[:-1], (0, 0, 1, 0), value=0.0)
        clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_clipped_snr)
        fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_snr)

        match self.objective:
            case "pred_noise":
                return clipped_fused_snr / fused_snr
            case "pred_x0":
                return clipped_fused_snr * self.snr_clip
            case "pred_v":
                return clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)
            case _:
                raise ValueError(f"unknown objective {self.objective}")

    def forward(
        self,
        x: torch.Tensor,
        action_cond: Optional[torch.Tensor],
        pose_cond,
        noise_levels: torch.Tensor,
        reference_length,
        frame_idx=None
    ):
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        
        noised_x = self.q_sample(x_start=x, t=noise_levels, noise=noise)

        model_pred = self.model_predictions(x=noised_x, t=noise_levels, action_cond=action_cond, 
                                    pose_cond=pose_cond,reference_length=reference_length, frame_idx=frame_idx)

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, noise_levels, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # 训练的时候每个frame随便给噪声
        loss = F.mse_loss(pred, target.detach(), reduction="none")
        loss_weight = self.compute_loss_weights(noise_levels)

        loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))

        loss = loss * loss_weight

        return x_pred, loss

    def sample_step(
        self,
        x: torch.Tensor,
        action_cond: Optional[torch.Tensor],
        pose_cond,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
        current_frame=None,
        mode="training",
        reference_length=None,
        frame_idx=None
    ):
        real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=x.device).long()

        # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
        curr_noise_level = real_steps[curr_noise_level]
        next_noise_level = real_steps[next_noise_level]

        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                action_cond=action_cond,
                pose_cond=pose_cond,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                guidance_fn=guidance_fn,
                current_frame=current_frame,
                mode=mode,
                reference_length=reference_length,
                frame_idx=frame_idx
            )

        # FIXME: temporary code for checking ddpm sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level) | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            action_cond=action_cond,
            pose_cond=pose_cond,
            curr_noise_level=curr_noise_level,
            guidance_fn=guidance_fn,
            reference_length=reference_length,
            frame_idx=frame_idx
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        action_cond: Optional[torch.Tensor],
        pose_cond,
        curr_noise_level: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
        reference_length=None,
        frame_idx=None,
    ):
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x),
        )
        x = torch.where(self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x)

        if guidance_fn is not None:
            raise NotImplementedError("Guidance function is not implemented for ddpm sampling yet.")

        else:
            model_mean, _, model_log_variance = self.p_mean_variance(
                x=x,
                t=clipped_curr_noise_level,
                action_cond=action_cond,
                pose_cond=pose_cond,
                reference_length=reference_length,
                frame_idx=frame_idx
            )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(self.add_shape_channels(curr_noise_level == -1), orig_x, x_pred)

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        action_cond: Optional[torch.Tensor],
        pose_cond,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        guidance_fn: Optional[Callable] = None,
        current_frame=None,
        mode="training",
        reference_length=None,
        frame_idx=None
    ):
        # convert noise level -1 to self.stabilization_level - 1
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.stabilization_level - 1, dtype=torch.long),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x),
        )
        x = torch.where(self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    t=clipped_curr_noise_level,
                    action_cond=action_cond,
                    pose_cond=pose_cond,
                    current_frame=current_frame,
                    mode=mode,
                    reference_length=reference_length,
                    frame_idx=frame_idx
                )

                guidance_loss = guidance_fn(model_pred.pred_x_start)
                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]

                pred_noise = model_pred.pred_noise + (1 - alpha_next).sqrt() * grad
                x_start = self.predict_start_from_noise(x, clipped_curr_noise_level, pred_noise)

        else:
            # print(clipped_curr_noise_level)
            model_pred = self.model_predictions(
                x=x,
                t=clipped_curr_noise_level,
                action_cond=action_cond,
                pose_cond=pose_cond,
                current_frame=current_frame,
                mode=mode,
                reference_length=reference_length,
                frame_idx=frame_idx
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )

        return x_pred

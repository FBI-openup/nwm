# Copyright (c) Meta Platforms, Inc. and affiliates.
# Hybrid CDiT model integrating WorldMem memory mechanisms
# 
# This implementation combines:
# - CDiT's precise conditional guidance through cross-attention
# - WorldMem's long-term memory capabilities through selective memory access
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Union, List
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from models import TimestepEmbedder, ActionEmbedder, modulate, FinalLayer


class MemoryBuffer:
    """
    Efficient memory buffer for storing and retrieving historical frames
    with spatial similarity-based selection.
    """
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.7):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.frames = []
        self.poses = []
        self.frame_indices = []
        
    def add_frame(self, frame_latent: torch.Tensor, pose: torch.Tensor, frame_idx: int):
        """Add a new frame to memory buffer"""
        self.frames.append(frame_latent.detach().cpu())
        self.poses.append(pose.detach().cpu())
        self.frame_indices.append(frame_idx)
        
        # Maintain buffer size
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
            self.poses.pop(0)
            self.frame_indices.pop(0)
    
    def get_relevant_frames(self, current_pose: torch.Tensor, k: int = 8) -> Optional[torch.Tensor]:
        """
        Retrieve k most spatially relevant frames based on enhanced pose similarity
        Optimized for computational efficiency
        """
        if len(self.frames) == 0:
            return None
            
        if len(self.frames) <= k:
            return torch.stack(self.frames).to(current_pose.device)
        
        # Vectorized similarity computation for efficiency
        current_pos = current_pose[:3].cpu()  # x, y, z (if available)
        memory_poses = torch.stack(self.poses)  # [N, pose_dim]
        
        # 1. Vectorized spatial distance computation
        if current_pose.shape[0] >= 2:  # At least x, y available
            pose_dims = min(3, memory_poses.shape[1])  # Use up to 3 dimensions
            spatial_dists = torch.norm(memory_poses[:, :pose_dims] - current_pos[:pose_dims], dim=1)
            spatial_sims = torch.exp(-spatial_dists / 10.0)
        else:
            spatial_sims = torch.ones(len(self.poses))
        
        # 2. Vectorized orientation similarity (if yaw available)
        # Note: Dataset only has yaw (no pitch), so yaw is at index 3 for 4D pose [x,y,z,yaw]
        if current_pose.shape[0] > 3 and memory_poses.shape[1] > 3:
            current_yaw = current_pose[3].cpu()
            yaw_diffs = torch.abs(current_yaw - memory_poses[:, 3])
            # Handle angle wrap-around efficiently
            yaw_diffs = torch.minimum(yaw_diffs, 2 * np.pi - yaw_diffs)
            angle_sims = torch.exp(-yaw_diffs / (np.pi / 4))
            
            # Combined similarity with weights
            similarities = 0.7 * spatial_sims + 0.3 * angle_sims
        else:
            similarities = spatial_sims
        
        # Select top-k (single operation, no loop)
        top_k_indices = torch.topk(similarities, min(k, len(similarities))).indices
        relevant_frames = [self.frames[i] for i in top_k_indices]
        return torch.stack(relevant_frames).to(current_pose.device)


class SelectiveMemoryAttention(nn.Module):
    """
    Selective memory attention module that can be optionally activated
    """
    def __init__(self, hidden_size: int, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Memory query/key/value projections
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism for optional activation
        self.activation_gate = nn.Parameter(torch.zeros(1))
        self.relevance_threshold = 0.1
        
    def compute_relevance(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Compute relevance scores between query and memory"""
        # Simplified relevance: cosine similarity
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        memory_norm = torch.nn.functional.normalize(memory, dim=-1)
        relevance = torch.sum(query_norm.unsqueeze(1) * memory_norm, dim=-1)
        return relevance
    
    def forward(self, x: torch.Tensor, memory_frames: Optional[torch.Tensor] = None, 
                activate_memory: bool = True) -> torch.Tensor:
        """
        Forward pass with optional memory activation
        
        Args:
            x: Current frame features [B, N, D]
            memory_frames: Memory buffer frames [B, M, N, D] 
            activate_memory: Whether to activate memory mechanism
        """
        if not activate_memory or memory_frames is None:
            return x * 0  # Return zero if memory not activated
        
        B, N, D = x.shape
        _, M, _, _ = memory_frames.shape
        
        # Reshape memory for attention
        memory_flat = memory_frames.view(B, M * N, D)
        
        # Compute relevance and filter irrelevant memory
        relevance = self.compute_relevance(x.mean(dim=1, keepdim=True), memory_flat.mean(dim=1, keepdim=True))
        
        if relevance.max() < self.relevance_threshold:
            return x * 0  # Skip memory if not relevant enough
        
        # Multi-head attention computation
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(memory_flat).view(B, M * N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(memory_flat).view(B, M * N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.to_out(out)
        
        # Apply activation gate (learnable parameter)
        return out * torch.sigmoid(self.activation_gate)


class HybridCDiTBlock(nn.Module):
    """
    Hybrid CDiT block that combines:
    1. CDiT's self-attention and cross-attention for precise conditional control
    2. Selective memory attention for long-term consistency
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, 
                 enable_memory: bool = True, **block_kwargs):
        super().__init__()
        self.enable_memory = enable_memory
        
        # CDiT core components (unchanged)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, 
                                         add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        
        # Memory components (new)
        if enable_memory:
            self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.memory_attn = SelectiveMemoryAttention(hidden_size, num_heads)
        
        # MLP
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # AdaLN modulation (extended for memory)
        num_modulations = 14 if enable_memory else 11
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_modulations * hidden_size, bias=True)
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, x_cond: torch.Tensor, 
                memory_frames: Optional[torch.Tensor] = None, 
                memory_activation_score: float = 0.0):
        """
        Forward pass with optional memory integration
        
        Args:
            x: Input features [B, N, D]
            c: Conditioning signal [B, D] 
            x_cond: Context frames [B, context_size, N, D]
            memory_frames: Memory buffer [B, M, N, D]
            memory_activation_score: Score determining memory activation strength
        """
        if self.enable_memory:
            modulations = self.adaLN_modulation(c).chunk(14, dim=1)
            (shift_msa, scale_msa, gate_msa, 
             shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x,
             shift_mem, scale_mem, gate_mem,
             shift_mlp, scale_mlp, gate_mlp) = modulations
        else:
            modulations = self.adaLN_modulation(c).chunk(11, dim=1)
            (shift_msa, scale_msa, gate_msa, 
             shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x,
             shift_mlp, scale_mlp, gate_mlp) = modulations
        
        # 1. Self-attention (CDiT standard)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # 2. Cross-attention with immediate context (CDiT core strength)
        x_cond_norm = modulate(self.norm_cond(x_cond.flatten(1, 2)), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(
            query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), 
            key=x_cond_norm, 
            value=x_cond_norm, 
            need_weights=False
        )[0]
        
        # 3. Selective memory attention (WorldMem enhancement)
        if self.enable_memory and memory_frames is not None:
            # Adaptive memory activation based on relevance score
            activate_memory = memory_activation_score > 0.3  # Threshold for activation
            
            memory_output = self.memory_attn(
                modulate(self.memory_norm(x), shift_mem, scale_mem),
                memory_frames=memory_frames,
                activate_memory=activate_memory
            )
            x = x + gate_mem.unsqueeze(1) * memory_output
        
        # 4. MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        
        return x


class HybridCDiT(nn.Module):
    """
    Hybrid CDiT model combining precise conditional control with selective long-term memory
    """
    def __init__(
        self,
        input_size: int = 32,
        context_size: int = 4,  # Will be 4 for XL, 3 for L/B
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        memory_enabled: bool = True,
        memory_layers: Optional[List[int]] = None,
        memory_buffer_size: int = 50
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.memory_enabled = memory_enabled
        
        # Default memory layers (activate in later layers)
        if memory_layers is None:
            memory_layers = list(range(depth // 2, depth))  # Second half of layers
        self.memory_layers = set(memory_layers)
        
        # Core CDiT components
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)
        
        # Positional embeddings
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, num_patches, hidden_size), 
            requires_grad=True
        )
        
        # Transformer blocks (hybrid)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            enable_memory_layer = memory_enabled and (i in self.memory_layers)
            self.blocks.append(
                HybridCDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                              enable_memory=enable_memory_layer)
            )
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # Memory management
        self.memory_buffer = MemoryBuffer(max_size=memory_buffer_size) if memory_enabled else None
        self.frame_counter = 0
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights (same as CDiT)"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # Initialize embedders
        for embedder in [self.y_embedder.x_emb, self.y_embedder.y_emb, self.y_embedder.angle_emb, 
                        self.t_embedder, self.time_embedder]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def update_memory(self, frame_latent: torch.Tensor, pose: torch.Tensor):
        """Update memory buffer with new frame"""
        if self.memory_buffer is not None:
            self.memory_buffer.add_frame(frame_latent, pose, self.frame_counter)
            self.frame_counter += 1
    
    def compute_memory_activation_score(self, current_pose: torch.Tensor, 
                                       action_magnitude: float) -> float:
        """
        Compute a score that determines when to activate memory
        Higher scores indicate more need for memory consultation
        """
        # Activate memory more when:
        # 1. Large camera movements (might revisit areas)
        # 2. Low action magnitude (might be looking around)
        # 3. When memory buffer has sufficient content
        
        movement_score = min(action_magnitude / 5.0, 1.0)  # Normalize action magnitude
        buffer_score = len(self.memory_buffer.frames) / self.memory_buffer.max_size if self.memory_buffer else 0
        
        # Combination heuristic (can be learned)
        activation_score = 0.3 * (1 - movement_score) + 0.7 * buffer_score
        return activation_score
    
    def unpatchify(self, x):
        """Unpatchify as in original CDiT"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, 
                x_cond: torch.Tensor, rel_t: torch.Tensor, 
                current_pose: Optional[torch.Tensor] = None,
                update_memory: bool = True):
        """
        Forward pass with optional memory integration
        
        Args:
            x: Input tensor [N, C, H, W]
            t: Timestep [N]
            y: Action conditions [N, 3]  
            x_cond: Context frames [N, context_size, C, H, W]
            rel_t: Relative time [N]
            current_pose: Current camera pose [N, 4] (x,y,z,yaw) - no pitch in dataset
            update_memory: Whether to update memory buffer
        """
        # Embed inputs (same as CDiT)
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) + self.pos_embed[:self.context_size]
        
        # Conditioning
        t_emb = self.t_embedder(t[..., None])
        y_emb = self.y_embedder(y)
        time_emb = self.time_embedder(rel_t[..., None])
        c = t_emb + time_emb + y_emb
        
        # Memory preparation
        memory_frames = None
        memory_activation_score = 0.0
        
        if self.memory_enabled and current_pose is not None:
            # Get relevant memory frames
            memory_frames = self.memory_buffer.get_relevant_frames(current_pose[0], k=8)
            if memory_frames is not None:
                memory_frames = memory_frames.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            
            # Compute memory activation score
            action_magnitude = torch.norm(y[0]).item()
            memory_activation_score = self.compute_memory_activation_score(current_pose[0], action_magnitude)
        
        # Transformer blocks with selective memory
        for i, block in enumerate(self.blocks):
            if i in self.memory_layers:
                x = block(x, c, x_cond, memory_frames, memory_activation_score)
            else:
                x = block(x, c, x_cond)  # Standard CDiT processing
        
        # Final processing
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        # Update memory if requested
        if update_memory and self.memory_enabled and current_pose is not None:
            self.update_memory(x.detach(), current_pose[0])
        
        return x


# Model configurations
def HybridCDiT_XL_2(**kwargs):
    return HybridCDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, 
                     context_size=4, **kwargs)

def HybridCDiT_L_2(**kwargs):
    return HybridCDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, 
                     context_size=3, **kwargs)

def HybridCDiT_B_2(**kwargs):
    return HybridCDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, 
                     context_size=3, **kwargs)

def HybridCDiT_S_2(**kwargs):
    return HybridCDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, 
                     context_size=3, **kwargs)

HybridCDiT_models = {
    'HybridCDiT-XL/2': HybridCDiT_XL_2,
    'HybridCDiT-L/2': HybridCDiT_L_2, 
    'HybridCDiT-B/2': HybridCDiT_B_2,
    'HybridCDiT-S/2': HybridCDiT_S_2
}

if __name__ == "__main__":
    # Test the hybrid model
    model = HybridCDiT_L_2(memory_enabled=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randn(batch_size, 3)
    x_cond = torch.randn(batch_size, 3, 4, 32, 32)  # context_size=3 for L model
    rel_t = torch.randn(batch_size)
    current_pose = torch.randn(batch_size, 4)  # [x, y, z, yaw] - no pitch
    
    with torch.no_grad():
        output = model(x, t, y, x_cond, rel_t, current_pose)
        print(f"Output shape: {output.shape}")

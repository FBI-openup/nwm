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
    with action-based behavioral similarity selection.
    """
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.7):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.frames = []
        self.poses = []
        self.actions = []  # Store actions for behavioral similarity
        self.frame_indices = []
        
    def add_frame(self, frame_latent: torch.Tensor, pose: torch.Tensor, action: torch.Tensor = None, frame_idx: int = 0):
        """Add a new frame to memory buffer with associated action - keep on GPU for cluster training"""
        # On L40s GPU cluster, all data must remain on GPU as CPU storage is not available
        self.frames.append(frame_latent.detach())
        self.poses.append(pose.detach())
        if action is not None:
            self.actions.append(action.detach())
        else:
            # Placeholder if no action provided
            if len(self.actions) > 0:
                self.actions.append(torch.zeros_like(self.actions[0]))
            else:
                self.actions.append(torch.zeros(3, device=frame_latent.device))  # Default [delta_x, delta_y, delta_yaw] on same device
        self.frame_indices.append(frame_idx)
        
        # Maintain buffer size
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
            self.poses.pop(0)
            self.actions.pop(0)
            self.frame_indices.pop(0)
    
    def get_relevant_frames(self, current_pose: torch.Tensor, target_action: torch.Tensor = None, k: int = 8) -> Optional[torch.Tensor]:
        """
        Retrieve k most relevant frames based on spatial proximity and action intent
        Focus on behavioral relevance rather than pose similarity
        """
        if len(self.frames) == 0:
            return None
            
        if len(self.frames) <= k:
            return torch.stack(self.frames).to(current_pose.device)
        
        # Vectorized similarity computation for efficiency - pure GPU operations for cluster training
        current_pos = current_pose[:3]  # Keep on GPU for cluster compatibility
        memory_poses = torch.stack(self.poses).to(current_pose.device)  # Ensure same device
        
        # 1. Spatial distance computation (primary factor)
        if current_pose.shape[0] >= 2:  # At least x, y available
            pose_dims = min(3, memory_poses.shape[1])  # Use up to 3 dimensions
            spatial_dists = torch.norm(memory_poses[:, :pose_dims] - current_pos[:pose_dims], dim=1)
            spatial_sims = torch.exp(-spatial_dists / 10.0)
        else:
            spatial_sims = torch.ones(len(self.poses), device=current_pose.device)
        
        # 2. Action-based similarity (if target action provided)
        # Focus on BEHAVIORAL CATEGORIES rather than exact action matching
        if target_action is not None and hasattr(self, 'actions') and len(self.actions) > 0:
            target_action_gpu = target_action.to(current_pose.device)  # Keep on GPU for cluster training
            memory_actions = torch.stack(self.actions).to(current_pose.device)  # Ensure on GPU
            
            # Behavioral similarity based on movement categories
            action_sims = self._compute_behavioral_similarity(target_action_gpu, memory_actions)
            
            # Combined similarity: prioritize spatial proximity, consider action patterns
            similarities = 0.8 * spatial_sims + 0.2 * action_sims
        else:
            # Fall back to spatial similarity only
            similarities = spatial_sims
        
        # Select top-k (single operation, no loop)
        top_k_indices = torch.topk(similarities, min(k, len(similarities))).indices
        relevant_frames = [self.frames[i] for i in top_k_indices]
        return torch.stack(relevant_frames).to(current_pose.device)
    
    def _compute_behavioral_similarity(self, target_action: torch.Tensor, memory_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute behavioral similarity based on action semantics rather than simple numerical distance
        Considers turning directionality and movement pattern classification for cluster GPU training
        
        Args:
            target_action: [delta_x, delta_y, delta_yaw] target action
            memory_actions: [N, 3] historical action set
            
        Returns:
            torch.Tensor: [N] behavioral similarity scores
        """
        # Extract movement components
        target_linear = target_action[:2]  # [delta_x, delta_y]
        target_yaw = target_action[2]      # delta_yaw
        
        memory_linear = memory_actions[:, :2]  # [N, 2]
        memory_yaw = memory_actions[:, 2]      # [N]
        
        # 1. Linear movement similarity (displacement vector)
        target_linear_norm = torch.norm(target_linear)
        memory_linear_norm = torch.norm(memory_linear, dim=1)
        
        # 归一化后的线性移动阈值
        LINEAR_THRESHOLD = 0.307  # 原 0.1 meter，归一化后
        
        # Movement magnitude similarity
        magnitude_sims = torch.exp(-torch.abs(target_linear_norm - memory_linear_norm) / 2.0)
        
        # Movement direction similarity (only compute for non-zero movements)
        direction_sims = torch.ones_like(magnitude_sims)
        if target_linear_norm > LINEAR_THRESHOLD:  # Significant linear movement (归一化后)
            target_direction = target_linear / target_linear_norm
            valid_memory = memory_linear_norm > LINEAR_THRESHOLD
            if valid_memory.any():
                memory_directions = memory_linear[valid_memory] / memory_linear_norm[valid_memory].unsqueeze(1)
                dot_products = torch.mm(memory_directions, target_direction.unsqueeze(1)).squeeze()
                direction_sims[valid_memory] = (dot_products + 1) / 2  # Map from [-1,1] to [0,1]
        
        # 2. Rotation behavior classification similarity
        yaw_sims = self._compute_rotation_similarity(target_yaw, memory_yaw)
        
        # 3. Combined behavioral similarity
        # Weights: linear movement(40%) + direction(30%) + rotation behavior(30%)
        behavioral_sims = 0.4 * magnitude_sims + 0.3 * direction_sims + 0.3 * yaw_sims
        
        return behavioral_sims
    
    def _compute_rotation_similarity(self, target_yaw: torch.Tensor, memory_yaw: torch.Tensor) -> torch.Tensor:
        """
        Rotation behavior semantic similarity computation - turning and straight movement have different scoring criteria
        Ensure turning actions have significantly lower similarity scores than straight movement to avoid confusion
        
        Behavior classification (归一化后的阈值):
        - Straight: |yaw| < 0.032 (原0.1 rad ~6°) -> high similarity baseline (0.9-1.0)
        - Minor adjustment: 0.032 <= |yaw| < 0.096 (原0.3 rad ~17°) -> medium similarity (0.6-0.8)
        - Turn: 0.096 <= |yaw| < 0.318 (原1.0 rad ~57°) -> low similarity (0.3-0.6)
        - Sharp turn: |yaw| >= 0.318 (原>1.0 rad ~57°+) -> lowest similarity (0.1-0.4)
        
        注意：yaw现在已归一化到[-1,1]范围，阈值相应调整
        """
        device = target_yaw.device
        
        # 归一化后的行为分类阈值
        STRAIGHT_THRESHOLD = 0.032    # 原 0.1 rad
        MINOR_THRESHOLD = 0.096       # 原 0.3 rad  
        TURN_THRESHOLD = 0.318        # 原 1.0 rad
        
        # Rotation direction classification
        target_direction = torch.sign(target_yaw)  # -1, 0, 1
        memory_direction = torch.sign(memory_yaw)   # [N]
        
        # Rotation magnitude classification
        def categorize_rotation(yaw_abs):
            """Classify rotation magnitude using normalized thresholds"""
            return torch.where(yaw_abs < STRAIGHT_THRESHOLD, 0,      # straight
                   torch.where(yaw_abs < MINOR_THRESHOLD, 1,         # minor adjustment
                   torch.where(yaw_abs < TURN_THRESHOLD, 2, 3)))     # turn / sharp turn
        
        target_abs = torch.abs(target_yaw)
        memory_abs = torch.abs(memory_yaw)
        
        target_category = categorize_rotation(target_abs)
        memory_categories = categorize_rotation(memory_abs)
        
        # Compute similarity - different behavior categories have different scoring baselines
        direction_match = (target_direction == memory_direction).float()
        category_match = (target_category == memory_categories).float()
        
        # Initialize similarity scores
        yaw_sims = torch.zeros_like(memory_yaw, device=device)
        
        # Category-wise processing, ensure clear scoring differences between turning and straight movement
        
        # 1. Straight category (target is straight movement)
        if target_category == 0:
            # When going straight, prioritize other straight actions
            straight_mask = (memory_categories == 0)
            yaw_sims[straight_mask] = 1.0  # Perfect match for straight movements
            
            # Give lower scores to minor adjustments
            minor_adjust_mask = (memory_categories == 1)
            yaw_sims[minor_adjust_mask] = 0.3
            
            # Give very low scores to turns
            turn_mask = (memory_categories >= 2)
            yaw_sims[turn_mask] = 0.1
            
        # 2. Turning category (target is turning)
        else:
            # Perfect match: same direction same category
            perfect_match = (direction_match == 1) & (category_match == 1)
            # Set different base scores based on category, turning categories have lower base scores
            base_scores = torch.tensor([0.9, 0.7, 0.5, 0.3], device=device)  # straight, minor, turn, sharp
            yaw_sims[perfect_match] = base_scores[target_category]
            
            # Same direction different category: give medium scores
            direction_only = (direction_match == 1) & (category_match == 0)
            if direction_only.any():
                # Decay based on angle difference
                yaw_diff = torch.abs(target_abs - memory_abs[direction_only])
                angle_similarity = torch.exp(-yaw_diff / 0.5)
                # Apply category penalty
                category_penalty = base_scores[target_category] * 0.7
                yaw_sims[direction_only] = angle_similarity * category_penalty
            
            # Opposite direction: extremely low scores
            opposite_direction = (direction_match == 0) & (target_category > 0) & (memory_categories > 0)
            yaw_sims[opposite_direction] = 0.05
            
            # Turning vs straight: low scores
            turning_vs_straight = (target_category > 0) & (memory_categories == 0)
            yaw_sims[turning_vs_straight] = 0.1
        
        return yaw_sims


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
    
    def update_memory(self, frame_latent: torch.Tensor, pose: torch.Tensor, action: torch.Tensor = None):
        """Update memory buffer with new frame and associated action"""
        if self.memory_buffer is not None:
            self.memory_buffer.add_frame(frame_latent, pose, action, self.frame_counter)
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
        
        # Memory preparation - only use during inference, skip during training
        memory_frames = None
        memory_activation_score = 0.0
        
        # Only use memory buffer during inference phase (not training mode) on GPU cluster
        if self.memory_enabled and current_pose is not None and not self.training:
            # Get relevant memory frames based on intended action (behavioral similarity)
            target_action = y[0] if y is not None else None  # Current target action
            memory_frames = self.memory_buffer.get_relevant_frames(
                current_pose[0], target_action=target_action, k=8
            )
            if memory_frames is not None:
                # memory_frames shape: [k, C, H, W], need to expand to [batch_size, k, C, H, W]
                if memory_frames.dim() == 4:  # [k, C, H, W]
                    memory_frames = memory_frames.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
                elif memory_frames.dim() == 3:  # [k, H, W] single channel case
                    memory_frames = memory_frames.unsqueeze(0).unsqueeze(2).expand(x.shape[0], -1, x.shape[1], -1, -1)
            
            # Compute memory activation score
            action_magnitude = torch.norm(y[0]).item()
            memory_activation_score = self.compute_memory_activation_score(current_pose[0], action_magnitude)
        
        # Transformer blocks with selective memory
        for i, block in enumerate(self.blocks):
            if i in self.memory_layers:
                # Use memory for inference in specified layers
                x = block(x, c, x_cond, memory_frames, memory_activation_score)
            else:
                # Standard CDiT processing (no memory usage for inference)
                x = block(x, c, x_cond)
        
        # Final processing
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        # Update memory during inference for all layers (storage happens regardless of layer)
        # Key design: 
        # - Layers 0-15 (non-memory): Store to memory but don't use memory for inference
        # - Later layers (memory): Both store to memory and use memory for inference
        # This ensures continuous memory building even when only using CDiT
        if update_memory and self.memory_enabled and current_pose is not None and not self.training:
            current_action = y[0] if y is not None else None  # Store the action that led to this frame
            self.update_memory(x.detach(), current_pose[0], current_action)
        
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

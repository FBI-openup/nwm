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
    智能记忆缓存系统，基于打分机制存储和检索关键帧
    支持动态评分调整和多因素综合评估
    """
    def __init__(self, max_size: int = 40, min_score_threshold: float = 0.3):
        # === 可配置评分参数 (方便调参) ===
        self.SCORING_CONFIG = {
            # 初始分数权重
            'turn_action_weight': 2.0,        # 转弯动作重要性
            'sharp_turn_weight': 3.0,         # 急转弯额外加权
            'view_clarity_weight': 1.5,       # 视野开阔度权重
            'spatial_diversity_weight': 1.8,  # 空间分布多样性
            'angle_diversity_weight': 1.2,    # 角度多样性
            
            # 动态调整参数
            'usage_boost': 0.2,               # 每次使用的分数提升
            'decay_rate': 0.05,               # 未使用时的衰减率
            'max_score': 5.0,                 # 最高分数上限
            'min_survival_score': 0.1,        # 保留的最低分数
            
            # 行为分类阈值 (归一化后)
            'significant_turn_threshold': 0.25,  # 重要转弯阈值
            'sharp_turn_threshold': 0.45,        # 急转弯阈值
            'linear_motion_threshold': 0.2,      # 线性运动阈值
            
            # 空间分布参数
            'min_spatial_distance': 3.0,         # 最小空间间距
            'min_angle_difference': 1.0,         # 最小角度差异(弧度)
        }
        
        self.max_size = max_size
        self.min_score_threshold = min_score_threshold
        
        # 存储结构
        self.frames = []
        self.poses = []
        self.actions = []
        self.frame_indices = []
        self.scores = []           # 每个记忆的当前分数
        self.usage_counts = []     # 使用次数统计
        self.last_used = []        # 最后使用时间
        
    def add_frame(self, frame_latent: torch.Tensor, pose: torch.Tensor, action: torch.Tensor = None, frame_idx: int = 0):
        """智能添加帧到记忆缓存，基于综合评分决定是否存储"""
        # 计算新帧的初始分数
        initial_score = self.compute_frame_score(pose, action, frame_idx)
        
        # 如果分数过低，直接丢弃
        if initial_score < self.min_score_threshold:
            return False
        
        # 如果缓存未满，直接添加
        if len(self.frames) < self.max_size:
            self.frames.append(frame_latent.detach())
            self.poses.append(pose.detach())
            if action is not None:
                self.actions.append(action.detach())
            else:
                self.actions.append(torch.zeros(3, device=frame_latent.device))
            self.frame_indices.append(frame_idx)
            self.scores.append(initial_score)
            self.usage_counts.append(0)
            self.last_used.append(frame_idx)
            return True
        
        # 缓存已满，需要替换最低分的记忆
        min_score_idx = self.scores.index(min(self.scores))
        min_score = self.scores[min_score_idx]
        
        # 如果新帧分数更高，替换掉最低分的记忆
        if initial_score > min_score:
            self.frames[min_score_idx] = frame_latent.detach()
            self.poses[min_score_idx] = pose.detach()
            if action is not None:
                self.actions[min_score_idx] = action.detach()
            else:
                self.actions[min_score_idx] = torch.zeros(3, device=frame_latent.device)
            self.frame_indices[min_score_idx] = frame_idx
            self.scores[min_score_idx] = initial_score
            self.usage_counts[min_score_idx] = 0
            self.last_used[min_score_idx] = frame_idx
            return True
        
        return False
    
    def compute_frame_score(self, pose: torch.Tensor, action: torch.Tensor = None, frame_idx: int = 0) -> float:
        """
        计算帧的综合评分，决定是否值得存储到记忆中
        
        Args:
            pose: 当前位置 [x, y, z, yaw]
            action: 当前动作 [delta_x, delta_y, delta_yaw] (归一化)
            frame_idx: 帧索引
            
        Returns:
            float: 综合评分 (0-5分)
        """
        score = 0.0
        config = self.SCORING_CONFIG
        
        # 1. 动作重要性评分
        if action is not None:
            turn_magnitude = torch.abs(action[2]).item()  # |delta_yaw|
            linear_magnitude = torch.norm(action[:2]).item()
            
            # 转弯动作评分
            if turn_magnitude >= config['sharp_turn_threshold']:
                score += config['sharp_turn_weight']  # 急转弯 +3分
            elif turn_magnitude >= config['significant_turn_threshold']:
                score += config['turn_action_weight']  # 重要转弯 +2分
            
            # 复杂动作评分 (转弯+移动)
            if turn_magnitude >= 0.15 and linear_magnitude >= config['linear_motion_threshold']:
                score += 1.0  # 复杂机动 +1分
        
        # 2. 空间多样性评分
        if len(self.poses) > 0:
            current_pos = pose[:3]
            stored_poses = torch.stack(self.poses)
            distances = torch.norm(stored_poses[:, :3] - current_pos, dim=1)
            min_distance = torch.min(distances).item()
            
            # 如果距离现有记忆足够远，增加分数
            if min_distance >= config['min_spatial_distance']:
                score += config['spatial_diversity_weight'] * min(min_distance / 10.0, 1.0)
        else:
            score += 2.0  # 第一帧自动获得高分
        
        # 3. 角度多样性评分
        if len(self.poses) > 0 and pose.shape[0] > 3:
            current_yaw = pose[3].item()
            stored_poses = torch.stack(self.poses)
            if stored_poses.shape[1] > 3:
                stored_yaws = stored_poses[:, 3]
                yaw_differences = torch.abs(stored_yaws - current_yaw)
                # 处理角度环绕
                yaw_differences = torch.minimum(yaw_differences, 2 * torch.pi - yaw_differences)
                min_yaw_diff = torch.min(yaw_differences).item()
                
                if min_yaw_diff >= config['min_angle_difference']:
                    score += config['angle_diversity_weight'] * min(min_yaw_diff / torch.pi, 1.0)
        
        # 4. 视野清晰度评分 (基于位置特征)
        # 简化版本：基于高度和开放程度估算
        if pose.shape[0] >= 3:
            height = pose[2].item()
            # 假设更高的位置有更好的视野
            clarity_score = min(height / 5.0, 1.0) * config['view_clarity_weight']
            score += clarity_score
        
        # 限制分数范围
        return min(score, config['max_score'])
    
    def update_usage_scores(self, used_indices: List[int], current_frame_idx: int):
        """
        更新使用过的记忆分数，实现动态评分调整
        
        Args:
            used_indices: 本次推理中使用的记忆索引列表
            current_frame_idx: 当前帧索引
        """
        config = self.SCORING_CONFIG
        
        # 提升使用过的记忆分数
        for idx in used_indices:
            if 0 <= idx < len(self.scores):
                self.scores[idx] = min(
                    self.scores[idx] + config['usage_boost'],
                    config['max_score']
                )
                self.usage_counts[idx] += 1
                self.last_used[idx] = current_frame_idx
        
        # 对未使用的记忆进行衰减
        for i in range(len(self.scores)):
            if i not in used_indices:
                frames_since_use = current_frame_idx - self.last_used[i]
                # 基于时间的衰减
                decay = config['decay_rate'] * min(frames_since_use / 100.0, 1.0)
                self.scores[i] = max(
                    self.scores[i] - decay,
                    config['min_survival_score']
                )
    
    def should_store_frame(self, pose: torch.Tensor, action: torch.Tensor = None, 
                          frame_idx: int = 0, min_distance: float = 5.0) -> bool:
        """
        判断是否应该将当前帧存储到记忆buffer中
        现在基于综合评分系统
        """
        score = self.compute_frame_score(pose, action, frame_idx)
        return score >= self.min_score_threshold
    
    def get_relevant_frames(self, current_pose: torch.Tensor, target_action: torch.Tensor = None, k: int = 8) -> Optional[torch.Tensor]:
        """
        基于动作相似性和记忆分数检索最相关的帧
        重点关注：相似动作 -> 相似变化 -> 寻找最近几帧
        
        Args:
            current_pose: 当前位置
            target_action: 目标动作 (当前要执行的动作)
            k: 返回的帧数量
        """
        if len(self.frames) == 0:
            return None
            
        if len(self.frames) <= k:
            return torch.stack(self.frames).to(current_pose.device)
        
        # 记录使用的索引，用于后续分数更新
        used_indices = []
        
        # 核心策略：相似的动作应该产生相似的变化
        if target_action is not None:
            # 1. 动作相似性评分 (主要因素)
            memory_actions = torch.stack(self.actions).to(current_pose.device)
            action_similarities = self._compute_action_similarity_for_retrieval(
                target_action.to(current_pose.device), memory_actions
            )
            
            # 2. 记忆分数加权
            memory_scores = torch.tensor(self.scores, device=current_pose.device)
            # 归一化分数到[0,1]
            norm_scores = memory_scores / self.SCORING_CONFIG['max_score']
            
            # 3. 空间相关性（次要因素）
            current_pos = current_pose[:3]
            memory_poses = torch.stack(self.poses).to(current_pose.device)
            spatial_dists = torch.norm(memory_poses[:, :3] - current_pos, dim=1)
            spatial_similarities = torch.exp(-spatial_dists / 8.0)  # 降低空间权重
            
            # 综合评分：动作相似性60% + 记忆分数30% + 空间相关性10%
            final_similarities = (0.6 * action_similarities + 
                                0.3 * norm_scores + 
                                0.1 * spatial_similarities)
        else:
            # 没有目标动作时，主要基于记忆分数和空间相关性
            memory_scores = torch.tensor(self.scores, device=current_pose.device)
            norm_scores = memory_scores / self.SCORING_CONFIG['max_score']
            
            current_pos = current_pose[:3]
            memory_poses = torch.stack(self.poses).to(current_pose.device)
            spatial_dists = torch.norm(memory_poses[:, :3] - current_pos, dim=1)
            spatial_similarities = torch.exp(-spatial_dists / 5.0)
            
            final_similarities = 0.7 * norm_scores + 0.3 * spatial_similarities
        
        # 选择top-k
        top_k_indices = torch.topk(final_similarities, min(k, len(final_similarities))).indices
        used_indices = top_k_indices.tolist()
        
        # 更新使用统计（如果有frame_counter信息）
        if hasattr(self, 'current_frame_idx'):
            self.update_usage_scores(used_indices, self.current_frame_idx)
        
        relevant_frames = [self.frames[i] for i in top_k_indices]
        return torch.stack(relevant_frames).to(current_pose.device)
    
    def _compute_action_similarity_for_retrieval(self, target_action: torch.Tensor, memory_actions: torch.Tensor) -> torch.Tensor:
        """
        专门为检索设计的动作相似性计算
        重点：相似动作应该找到最近的相似执行案例
        """
        target_linear = target_action[:2]
        target_yaw = target_action[2]
        
        memory_linear = memory_actions[:, :2]
        memory_yaw = memory_actions[:, 2]
        
        # 1. 线性运动相似性
        target_linear_norm = torch.norm(target_linear)
        memory_linear_norm = torch.norm(memory_linear, dim=1)
        
        # 运动幅度相似性
        magnitude_diff = torch.abs(target_linear_norm - memory_linear_norm)
        magnitude_sims = torch.exp(-magnitude_diff / 0.3)  # 更严格的幅度匹配
        
        # 运动方向相似性
        direction_sims = torch.ones_like(magnitude_sims)
        if target_linear_norm > 0.1:  # 有明显线性运动
            target_direction = target_linear / target_linear_norm
            valid_memory = memory_linear_norm > 0.1
            if valid_memory.any():
                memory_directions = memory_linear[valid_memory] / memory_linear_norm[valid_memory].unsqueeze(1)
                dot_products = torch.mm(memory_directions, target_direction.unsqueeze(1)).squeeze()
                direction_sims[valid_memory] = torch.clamp((dot_products + 1) / 2, 0, 1)
        
        # 2. 转向行为相似性 - 更精确匹配
        yaw_sims = self._compute_precise_rotation_similarity(target_yaw, memory_yaw)
        
        # 3. 综合相似性：对于检索，我们更关注精确匹配
        # 线性运动50% + 方向25% + 转向25%
        action_similarities = 0.5 * magnitude_sims + 0.25 * direction_sims + 0.25 * yaw_sims
        
        return action_similarities
    
    def _compute_precise_rotation_similarity(self, target_yaw: torch.Tensor, memory_yaw: torch.Tensor) -> torch.Tensor:
        """
        精确的转向相似性计算，用于动作检索
        重点：相同类型的转向动作应该获得高相似性
        """
        config = self.SCORING_CONFIG
        
        target_abs = torch.abs(target_yaw)
        memory_abs = torch.abs(memory_yaw)
        
        # 转向类型分类
        def classify_turn(yaw_abs):
            if yaw_abs < 0.1:  # 直行
                return 0
            elif yaw_abs < config['significant_turn_threshold']:  # 微调
                return 1
            elif yaw_abs < config['sharp_turn_threshold']:  # 转弯
                return 2
            else:  # 急转弯
                return 3
        
        target_class = classify_turn(target_abs)
        memory_classes = torch.tensor([classify_turn(y) for y in memory_abs], device=memory_yaw.device)
        
        # 方向匹配
        target_direction = torch.sign(target_yaw)
        memory_directions = torch.sign(memory_yaw)
        direction_match = (target_direction == memory_directions).float()
        
        # 类型匹配
        class_match = (target_class == memory_classes).float()
        
        # 角度差异
        yaw_diff = torch.abs(target_yaw - memory_yaw)
        angle_similarity = torch.exp(-yaw_diff / 0.2)  # 更严格的角度匹配
        
        # 综合评分：完全匹配 > 同类型同方向 > 角度相近
        similarities = (0.5 * class_match * direction_match +  # 完全匹配
                       0.3 * class_match +                    # 同类型
                       0.2 * angle_similarity)                # 角度相近
        
        return similarities
    
    def get_memory_stats(self) -> dict:
        """获取记忆缓存的统计信息，用于调试和监控"""
        if len(self.frames) == 0:
            return {"empty": True}
        
        return {
            "total_memories": len(self.frames),
            "max_capacity": self.max_size,
            "average_score": sum(self.scores) / len(self.scores),
            "highest_score": max(self.scores),
            "lowest_score": min(self.scores),
            "total_usage": sum(self.usage_counts),
            "most_used_count": max(self.usage_counts) if self.usage_counts else 0,
            "scoring_config": self.SCORING_CONFIG
        }
    
    def reset_memory(self):
        """重置记忆缓存"""
        self.frames.clear()
        self.poses.clear()
        self.actions.clear()
        self.frame_indices.clear()
        self.scores.clear()
        self.usage_counts.clear()
        self.last_used.clear()
    

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
        """智能更新记忆缓存，基于评分系统决定存储"""
        if self.memory_buffer is not None:
            # 使用智能存储机制
            stored = self.memory_buffer.add_frame(frame_latent, pose, action, self.frame_counter)
            # 更新当前帧索引，用于分数衰减计算
            self.memory_buffer.current_frame_idx = self.frame_counter
            self.frame_counter += 1
            return stored
        return False
    
    def get_memory_stats(self):
        """获取记忆系统的统计信息"""
        if self.memory_buffer is not None:
            return self.memory_buffer.get_memory_stats()
        return {"memory_disabled": True}
    
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

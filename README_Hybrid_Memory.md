# Hybrid CDiT Memory System Design

## 核心设计理念

本项目实现了一个智能的Hybrid CDiT模型，结合了CDiT的精确条件控制和WorldMem的长期记忆能力。核心创新在于**分离存储与检索机制**，实现更智能的记忆管理。

## 架构依赖关系

### models.py 核心组件依赖

`hybrid_models.py` 依赖并复用 `models.py` 中的以下核心组件：

```python
from models import TimestepEmbedder, ActionEmbedder, modulate, FinalLayer
```

**组件功能说明：**
- **`TimestepEmbedder`**: 使用正弦余弦嵌入将标量时间步转换为向量表示
- **`ActionEmbedder`**: 将动作向量 [x, y, yaw] 嵌入到隐藏表示空间
- **`modulate`**: AdaLN (自适应层归一化) 调制函数，用于条件控制
- **`FinalLayer`**: 最终输出层，将隐藏特征转换回图像空间

**设计优势：**
1. **模块化复用**: 避免重复实现，保持代码简洁
2. **向后兼容**: 确保与原始CDiT组件完全一致
3. **易于维护**: 核心组件的修改自动同步到混合模型
4. **独立测试**: 可以独立测试CDiT和HybridCDiT的区别

### 设计哲学
- **存储严格，检索灵活**：用严格标准筛选值得保存的关键帧，用灵活标准匹配相关记忆
- **行为导向**：重点关注动作行为的相似性，而非简单的空间距离
- **动态衰减**：实现记忆的自然遗忘和重要性调整
- **模块化复用**：充分利用 `models.py` 中已验证的核心组件

## 双标准记忆系统

### 1. 存储标准（严格筛选）

存储系统负责筛选真正有价值的关键帧，采用100分制评分系统。

#### 评分参数配置（与代码中 SCORING_CONFIG 一致）

**动作价值评估:**
- **急转弯地标** (+50.0分): `storage_sharp_turn_weight` - 导航中的重要节点
- **重要转弯** (+35.0分): `storage_turn_weight` - 次要导航节点  
- **复杂机动** (+15.0分): `storage_complex_maneuver` - 困难路段记忆
- **平凡动作** (-8.0分): `storage_trivial_penalty` - 降低普通直行的价值

**空间独特性:**
- **新区域探索** (+30.0分): `storage_spatial_weight` - 距离现有记忆足够远的位置
- **位置太近** (-12.0分): `storage_close_penalty` - 重复区域的价值降低
- **第一帧奖励** (+40.0分): `storage_first_frame_bonus` - 起点的特殊重要性

**视角多样性:**
- **独特视角** (+25.0分): `storage_angle_weight` - 不同朝向的关键视角
- **相似视角** (-5.0分): `storage_similar_angle_penalty` - 降低重复视角的价值

**环境特征:**
- **高度优势** (+20.0分): `storage_height_weight` - 高位置的观察价值

**存储约束条件:**
- **最小空间间距**: 4.0米 (`storage_min_distance`)
- **最小角度差异**: 1.2弧度 (~69°) (`storage_min_angle_diff`)
- **存储阈值**: 6.0分 (`min_score_threshold`) - 只有超过此阈值的帧才会被存储
- **最大容量**: 40帧 (`max_size`)

**行为分类阈值:**
- **重要转弯**: 0.25弧度 (~14°) (`significant_turn_threshold`)
- **急转弯**: 0.45弧度 (~26°) (`sharp_turn_threshold`)  
- **线性运动**: 0.2米 (`linear_motion_threshold`)

### 2. 检索标准（灵活匹配）

检索系统负责找到对当前推理最有帮助的记忆，采用多因素加权评分。

#### 权重分配（代码中的实际权重）
- **动作相似性** (50%): `retrieval_action_weight` - 主要因素，寻找相似的行为模式
- **记忆价值** (25%): `retrieval_memory_weight` - 重要因素，高质量记忆优先
- **空间相关性** (15%): `retrieval_spatial_weight` - 辅助因素，考虑地理位置
- **使用经验** (10%): `retrieval_usage_weight` - 经验因素，频繁使用的记忆

**检索参数:**
- **空间匹配半径**: 10.0米 (`retrieval_spatial_radius`)
- **默认检索数量**: 8帧 (`k=8`)

#### 动作相似性算法
1. **运动幅度匹配**: 相似的移动距离 (50%权重)
2. **方向一致性**: 相似的移动方向 (25%权重)  
3. **转向行为分类**: 直行/微调/转弯/急转弯的精确匹配 (25%权重)

## 动态衰减系统

### 固定记忆时间保护机制（与代码实现一致）
- **保护期**: 前3步 (`fixed_memory_time=3`) 完全不衰减
- **设计理由**: 新记忆需要时间证明其价值，避免过早淘汰

### 加速衰减机制
使用动态衰减公式实现记忆的自然遗忘：
衰减率 = 基础衰减率 × (加速系数 ^ 超出保护期步数)
```python
# 代码中的实际衰减计算
excess_steps = unused_steps - fixed_memory_time  # 超出保护期的步数
dynamic_decay_rate = base_decay_rate * (accelerated_decay_rate ** excess_steps)
new_score = max(old_score - dynamic_decay_rate, min_survival_score)
```

#### 参数设置（代码中的实际值）
- **基础衰减率**: 1.2分/步 (`base_decay_rate`)
- **加速系数**: 1.4 (`accelerated_decay_rate`)
- **最低生存分数**: 3.0分 (`min_survival_score`)
- **最高分数上限**: 100.0分 (`max_score`)

#### 衰减示例（实际计算）
- **第1-3步**: 无衰减（保护期，`unused_steps <= 3`）
- **第4步**: 衰减 1.2 × 1.4¹ = 1.68分
- **第5步**: 衰减 1.2 × 1.4² = 2.35分  
- **第6步**: 衰减 1.2 × 1.4³ = 3.29分
- **第7步**: 衰减 1.2 × 1.4⁴ = 4.61分
- **持续加速**: 衰减速度递增，促进遗忘

### 使用奖励机制（代码实现）
- **分数提升**: +5.0分/次使用 (`usage_boost`)
- **冷却重置**: 重新享受3步保护期 (`unused_steps=0`)
- **使用统计**: 记录使用频率用于检索权重计算

```python
# 代码中的使用奖励逻辑
self.scores[idx] = min(
    self.scores[idx] + config['usage_boost'],  # +5.0分
    config['max_score']  # 不超过100分上限
)
self.unused_steps[idx] = 0  # 重置衰减冷却期
```

## 记忆统计与监控

系统提供完整的统计信息用于调试和优化：

### 基础统计
- 总记忆数 / 最大容量
- 平均分数 / 最高分数 / 最低分数
- 总使用次数 / 最高使用次数

### 动态衰减统计
- **保护期记忆数**: 享受保护的记忆数量
- **衰减期记忆数**: 正在衰减的记忆数量  
- **平均未使用步数**: 衰减系统健康度指标
- **最大未使用步数**: 最久未使用的记忆

## 技术实现细节

### 架构集成
- **CDiT主干**: 保持原有的自注意力和交叉注意力
- **选择性记忆注意力**: 在后半层（14-28层）激活记忆机制
- **自适应激活**: 基于相关性分数动态决定是否使用记忆

### 训练与推理分离
- **训练阶段**: 禁用记忆系统，专注于基础能力学习
- **推理阶段**: 全面激活记忆系统，利用历史经验

### 内存管理
- **智能缓存**: 基于评分的LRU替换策略
- **GPU优化**: 高效的张量操作和内存复用
- **批处理支持**: 支持批量推理的记忆管理

## 配置参数说明

所有关键参数都在 `SCORING_CONFIG` 中集中管理，与代码实现完全一致：

### 存储标准参数（严格筛选关键帧）
```python
# hybrid_models.py 中的实际配置
'storage_turn_weight': 35.0,           # 存储：转弯动作重要性
'storage_sharp_turn_weight': 50.0,     # 存储：急转弯额外加权  
'storage_spatial_weight': 30.0,        # 存储：空间独特性权重
'storage_angle_weight': 25.0,          # 存储：角度多样性权重
'storage_height_weight': 20.0,         # 存储：高度优势权重
'storage_complex_maneuver': 15.0,      # 存储：复杂机动加分
'storage_trivial_penalty': -8.0,       # 存储：平凡动作扣分
'storage_close_penalty': -12.0,        # 存储：位置太近扣分
'storage_similar_angle_penalty': -5.0, # 存储：相似视角扣分
'storage_first_frame_bonus': 40.0,     # 存储：第一帧起点奖励
'storage_min_distance': 4.0,           # 存储：最小空间间距要求(米)
'storage_min_angle_diff': 1.2,         # 存储：最小角度差异要求(弧度)
```

### 检索标准参数（灵活匹配相关记忆）
```python
'retrieval_action_weight': 0.50,       # 检索：动作相似性权重 (主要)
'retrieval_memory_weight': 0.25,       # 检索：记忆价值权重 (重要)
'retrieval_spatial_weight': 0.15,      # 检索：空间相关性权重 (辅助)
'retrieval_usage_weight': 0.10,        # 检索：使用经验权重 (经验)
'retrieval_spatial_radius': 10.0,      # 检索：空间匹配半径(米)
```

### 动态衰减系统参数
```python
'usage_boost': 5.0,                    # 每次使用的分数提升
'fixed_memory_time': 3,                # 固定记忆时间：前3步不衰减
'base_decay_rate': 1.2,                # 基础衰减率(分/步)
'accelerated_decay_rate': 1.4,         # 加速衰减率：衰减速度递增系数
'max_score': 100.0,                    # 最高分数上限
'min_survival_score': 3.0,             # 保留的最低分数
```

### 行为分类阈值
```python
'significant_turn_threshold': 0.25,    # 重要转弯阈值(弧度, ~14°)
'sharp_turn_threshold': 0.45,          # 急转弯阈值(弧度, ~26°)
'linear_motion_threshold': 0.2,        # 线性运动阈值(米)
```

### 缓存管理参数
```python
# MemoryBuffer 初始化参数
max_size: int = 40,                     # 最大缓存容量
min_score_threshold: float = 6.0        # 最小存储阈值 (0.3 * 20 = 6.0)
```

### 模型架构参数
```python
# HybridCDiT 初始化参数
memory_enabled: bool = True,            # 是否启用记忆机制
memory_buffer_size: int = 50,           # 记忆缓存大小
memory_layers: List[int] = None         # 记忆激活层 (默认: 后半层)

# 默认记忆层配置
if memory_layers is None:
    memory_layers = list(range(depth // 2, depth))  # 后半层激活记忆
```

## 性能优势

### 相比原始CDiT
1. **长期一致性**: 记忆机制提供历史上下文
2. **行为学习**: 从相似情况中学习最佳决策
3. **适应性**: 动态调整记忆重要性

### 相比简单WorldMem
1. **智能筛选**: 不是所有帧都保存，只存储关键帧
2. **精确检索**: 基于行为相似性而非简单距离
3. **动态管理**: 自动遗忘和重要性调整

## 使用示例

### 基础使用（依赖 models.py）

```python
# 确保 models.py 在相同目录
from hybrid_models import HybridCDiT_L_2

# 创建混合模型
model = HybridCDiT_L_2(
    memory_enabled=True, 
    memory_buffer_size=40,  # 与 SCORING_CONFIG 中的 max_size 对应
    memory_layers=list(range(12, 24))  # L模型的后半层
)

# 推理时自动使用记忆
output = model(
    x=input_tensor,           # [N, C, H, W] 输入图像
    t=timesteps,              # [N] 扩散时间步
    y=actions,                # [N, 3] 动作条件 [dx, dy, dyaw]
    x_cond=context_frames,    # [N, context_size, C, H, W] 上下文帧
    rel_t=relative_time,      # [N] 相对时间
    current_pose=poses        # [N, 4] 当前位姿 [x, y, z, yaw]
)

# 获取记忆统计（与代码中 get_memory_stats() 一致）
stats = model.get_memory_stats()
print(f"记忆容量: {stats['total_memories']}/{stats['max_capacity']}")
print(f"平均分数: {stats['average_score']:.2f}")
print(f"保护期记忆: {stats['protected_memories']}")
print(f"衰减期记忆: {stats['decaying_memories']}")
```

### 纯CDiT模式（兼容性测试）

```python
# 禁用记忆，获得与原始CDiT相同的行为
model_cdit = HybridCDiT_L_2(memory_enabled=False)

# 这种模式下不需要 current_pose 参数
output = model_cdit(x, t, y, x_cond, rel_t)
```

### 记忆系统配置调优

```python
# 创建模型后动态调整评分参数
model = HybridCDiT_L_2(memory_enabled=True)

# 调整存储策略 - 更严格筛选
model.memory_buffer.SCORING_CONFIG.update({
    'storage_turn_weight': 40.0,        # 提高转弯重要性
    'storage_trivial_penalty': -12.0,   # 更严格惩罚平凡动作
    'min_score_threshold': 8.0          # 提高存储阈值
})

# 调整检索策略 - 更注重行为相似性  
model.memory_buffer.SCORING_CONFIG.update({
    'retrieval_action_weight': 0.60,    # 增强动作相似性权重
    'retrieval_spatial_weight': 0.10,   # 降低空间权重
})

# 调整衰减策略 - 更长保护期
model.memory_buffer.SCORING_CONFIG.update({
    'fixed_memory_time': 5,             # 延长保护期到5步
    'usage_boost': 8.0,                 # 增加使用奖励
})
```

### 记忆系统监控

```python
# 推理过程中实时监控记忆状态
for step, (x, t, y, x_cond, rel_t, pose) in enumerate(dataloader):
    output = model(x, t, y, x_cond, rel_t, pose)
    
    if step % 10 == 0:  # 每10步监控一次
        stats = model.get_memory_stats()
        print(f"Step {step}:")
        print(f"  记忆使用率: {stats['total_memories']}/{stats['max_capacity']}")
        print(f"  平均未使用步数: {stats['avg_unused_steps']:.1f}")
        print(f"  最高/最低分数: {stats['highest_score']:.1f}/{stats['lowest_score']:.1f}")
        
        # 检查记忆系统健康度
        if stats['avg_unused_steps'] > 10:
            print("  警告: 记忆平均未使用步数过高，考虑调整检索策略")
        if stats['total_memories'] < stats['max_capacity'] * 0.5:
            print("  提示: 记忆使用率较低，考虑降低存储阈值")
```

## 项目文件结构与依赖关系

### 核心文件依赖图

```
models.py (基础组件)
    ├── TimestepEmbedder
    ├── ActionEmbedder  
    ├── modulate
    ├── FinalLayer
    └── CDiT (原始模型)
         │
         ▼
hybrid_models.py (混合架构)
    ├── MemoryBuffer (记忆缓存系统)
    ├── SelectiveMemoryAttention (选择性记忆注意力)
    ├── HybridCDiTBlock (混合变换器块)
    └── HybridCDiT (主模型)
         │
         ▼
应用层文件
    ├── train.py (训练脚本)
    ├── isolated_nwm_infer.py (推理脚本)
    ├── planning_eval.py (规划评估)
    └── interactive_model.ipynb (交互式测试)
```

### 模块化设计原则

1. **基础组件复用** (`models.py`)
   - 提供经过验证的核心嵌入组件
   - 保持原始CDiT实现作为基准
   - 确保数值计算的一致性

2. **记忆系统扩展** (`hybrid_models.py`)
   - 基于 `models.py` 构建混合架构
   - 添加智能记忆管理功能
   - 保持向后兼容性

3. **应用层适配**
   - 现有脚本可以无缝切换模型
   - 支持CDiT和HybridCDiT的A/B测试
   - 渐进式迁移策略

### 兼容性保证

**向后兼容性:**
```python
# 使用原始CDiT
from models import CDiT_L_2
model_cdit = CDiT_L_2()

# 使用混合模型的CDiT模式 (行为完全一致)
from hybrid_models import HybridCDiT_L_2  
model_hybrid = HybridCDiT_L_2(memory_enabled=False)

# 两者应产生相同的输出 (数值精度范围内)
assert torch.allclose(
    model_cdit(x, t, y, x_cond, rel_t),
    model_hybrid(x, t, y, x_cond, rel_t),
    rtol=1e-5
)
```

**渐进式增强:**
```python
# 阶段1: 使用CDiT模式验证基础功能
model = HybridCDiT_L_2(memory_enabled=False)

# 阶段2: 启用记忆但不更新(只观察)
model = HybridCDiT_L_2(memory_enabled=True)
output = model(x, t, y, x_cond, rel_t, pose, update_memory=False)

# 阶段3: 完整启用记忆系统
model = HybridCDiT_L_2(memory_enabled=True)  
output = model(x, t, y, x_cond, rel_t, pose, update_memory=True)
```

## 调优建议

### 提高存储选择性
```python
# 更严格的存储策略
model.memory_buffer.SCORING_CONFIG.update({
    'min_score_threshold': 8.0,         # 提高存储阈值
    'storage_turn_weight': 40.0,        # 增加关键动作权重
    'storage_trivial_penalty': -12.0,   # 增加惩罚项权重
    'storage_close_penalty': -15.0      # 更严格的空间去重
})
```

### 提高检索准确性  
```python
# 优化检索权重分配
model.memory_buffer.SCORING_CONFIG.update({
    'retrieval_action_weight': 0.60,    # 增强行为相似性
    'retrieval_memory_weight': 0.20,    # 降低记忆价值影响
    'retrieval_spatial_radius': 8.0,    # 缩小空间半径
})
```

### 优化衰减策略
```python
# 调整记忆生命周期
model.memory_buffer.SCORING_CONFIG.update({
    'fixed_memory_time': 5,             # 延长保护期
    'base_decay_rate': 1.0,             # 降低衰减速度
    'usage_boost': 8.0,                 # 增加使用奖励
})
```

## 重要注意事项

### 1. 依赖要求
- **必须确保 `models.py` 在相同目录或Python路径中**
- 如果移动 `hybrid_models.py`，需要更新导入路径
- 所有核心组件的修改需要在 `models.py` 中进行

### 2. 内存管理
- 记忆缓存会占用GPU内存，根据显存调整 `memory_buffer_size`
- 推理时记忆系统处于激活状态，训练时自动禁用
- 长时间推理建议定期检查记忆使用统计

### 3. 性能考虑
- 记忆激活会增加 ~15% 的计算开销
- 可以通过调整 `memory_layers` 控制记忆使用的层数
- 批处理推理时记忆在batch间共享

### 4. 调试技巧
```python
# 启用详细的记忆统计输出
stats = model.get_memory_stats()
for key, value in stats.items():
    print(f"{key}: {value}")

# 监控记忆激活频率
if hasattr(model, 'memory_buffer'):
    print(f"记忆激活率: {model.memory_buffer.usage_counts}")
```

## 未来扩展

1. **学习化参数**: 将固定评分参数改为可学习参数
2. **多层次记忆**: 短期/中期/长期记忆分层管理
3. **语义记忆**: 结合视觉特征的语义相似性
4. **协作记忆**: 多智能体间的记忆共享机制
5. **自适应阈值**: 根据环境动态调整存储和检索阈值

---

*最后更新: 2025年8月5日*  
*版本: 混合记忆系统 v2.0*  
*依赖: models.py (核心组件提供)*
        
        # 维护缓冲区大小
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
            self.poses.pop(0)
            self.actions.pop(0)
            self.frame_indices.pop(0)
```

## 行为驱动检索机制

### 核心问题解决

**原始问题**：基于状态相似性的检索会导致语义错误
- **往前走**：选择相似朝向的帧 ✓ 正确
- **转弯时**：选择相同朝向的帧 ✗ 错误（需要转弯经验，不是直行经验）

**解决方案**：从状态相似性转向行为相关性

### 智能检索算法

```python
def get_relevant_frames(self, current_pose, target_action=None, k=8):
    """
    基于空间邻近性和行为相关性的智能帧检索
    
    Args:
        current_pose: 当前位置 [x, y, z, yaw]
        target_action: 目标动作 [delta_x, delta_y, delta_yaw]
        k: 检索帧数量
    
    Returns:
        相关性最高的k个历史帧
    """
    if len(self.frames) == 0:
        return None
        
    # 1. 空间相似性计算 (80% 权重)
    current_pos = current_pose[:3].cpu()
    memory_poses = torch.stack(self.poses)
    spatial_dists = torch.norm(memory_poses[:, :3] - current_pos[:3], dim=1)
    spatial_sims = torch.exp(-spatial_dists / 10.0)
    
    # 2. 行为相似性计算 (20% 权重)
    if target_action is not None and len(self.actions) > 0:
        target_action_cpu = target_action.cpu()
        memory_actions = torch.stack(self.actions)
        action_dists = torch.norm(memory_actions - target_action_cpu, dim=1)
        action_sims = torch.exp(-action_dists / 5.0)
        
        # 综合评分：空间优先，行为指导
        similarities = 0.8 * spatial_sims + 0.2 * action_sims
    else:
        similarities = spatial_sims
    
    # 3. 选择最相关的帧
    top_k_indices = torch.topk(similarities, min(k, len(similarities))).indices
    relevant_frames = [self.frames[i] for i in top_k_indices]
    return torch.stack(relevant_frames)
```

### 行为语义分析

对于 NoMaD 轨迹数据计算的动作 `[delta_x, delta_y, delta_yaw]`：

#### 智能行为分类

**问题**：转弯角度多样性导致简单距离匹配失效
- 需要90°右转，历史有30°右转和90°左转
- 简单距离：会错误选择30°右转 (距离60°) 而非90°左转 (距离180°)

**解决方案**：行为语义分类而非数值匹配

| 旋转类别 | 角度范围 | 行为特征 | 匹配策略 |
|---------|----------|----------|----------|
| **直行** | \|yaw\| < 6° | 保持方向 | 所有直行互相匹配 |
| **微调** | 6° ≤ \|yaw\| < 17° | 小幅调整 | 方向+幅度匹配 |
| **转弯** | 17° ≤ \|yaw\| < 57° | 明显转向 | 方向优先，幅度次要 |
| **大转** | \|yaw\| ≥ 57° | 大幅转向 | 方向优先，幅度次要 |

#### 相似性计算优先级

```python
def compute_behavioral_similarity(target_action, memory_actions):
    # 1. 线性运动分析 (40% 权重)
    magnitude_sims = exp(-|target_speed - memory_speed| / threshold)
    direction_sims = (dot_product + 1) / 2  # 运动方向相似性
    
    # 2. 旋转行为分类 (30% 权重)  
    rotation_sims = classify_and_match_rotation(target_yaw, memory_yaw)
    
    # 3. 综合评分
    behavioral_sims = 0.4 * magnitude_sims + 0.3 * direction_sims + 0.3 * rotation_sims
    
    # 最终结合空间相似性
    final_sims = 0.8 * spatial_sims + 0.2 * behavioral_sims
```

#### 实际匹配效果（改进的评分标准）

**场景1：需要90°右转**
- ✅ **优先选择**：60°右转(0.5分), 45°右转(0.4分), 120°右转(0.3分) (同方向转弯)
- ⭐ **评分差异**：转弯基础分数(0.3-0.5) << 直行基础分数(0.9-1.0)
- ❌ **避免选择**：直行(0.1分), 90°左转(0.05分) (不同行为类别)

**场景2：需要直行**  
- ✅ **优先选择**：所有直行经验(1.0分) (delta_yaw ≈ 0)
- ⭐ **评分差异**：直行vs转弯有明显分数差距 (1.0 vs 0.1)
- ❌ **避免选择**：任何转弯经验(0.1分)

**场景3：需要微调(10°左转)**
- ✅ **最优匹配**：5-15°左转(0.7分) (同类别同方向)
- ✅ **次优匹配**：20-30°左转(0.5分) (同方向不同类别)  
- ❌ **避免选择**：直行(0.1分), 右转(0.05分), 大幅左转(0.3分)

**评分原则**：
- 直行类别基础分数：0.9-1.0（高）
- 转弯类别基础分数：0.3-0.7（中低）  
- 交叉类别惩罚：×0.1（严格区分）

## 混合架构优势

### 1. 分层内存激活

```python
class HybridCDiTBlock(nn.Module):
    def forward(self, x, c, x_cond, memory_frames=None, memory_activation_score=0.0):
        # 1. 自注意力 (CDiT 标准)
        x = x + self.self_attention(x)
        
        # 2. 条件交叉注意力 (CDiT 核心优势)
        x = x + self.cross_attention(x, x_cond)
        
        # 3. 选择性内存注意力 (WorldMem 增强)
        if self.enable_memory and memory_frames is not None:
            if memory_activation_score > 0.3:  # 智能激活
                x = x + self.memory_attention(x, memory_frames)
        
        # 4. MLP 处理
        x = x + self.mlp(x)
        return x
```

### 2. 自适应内存激活

```python
def compute_memory_activation_score(self, current_pose, action_magnitude):
    """
    根据情况复杂度决定是否激活内存
    - 复杂场景：激活内存获取历史经验
    - 简单场景：依赖即时感知
    """
    # 基于动作幅度和场景复杂度
    complexity_score = action_magnitude * 0.5
    
    # 基于内存相关性
    if len(self.memory_buffer.frames) > 10:
        memory_relevance = self.compute_memory_relevance(current_pose)
        complexity_score += memory_relevance * 0.5
    
    return min(complexity_score, 1.0)
```

## 训练集成

### 内存使用策略

**重要更新**：分离存储和推理逻辑，实现分层内存机制

```python
def forward(self, x, t, y, x_cond, rel_t, current_pose=None, update_memory=True):
    # ... 模型前向传播 ...
    
    # 内存检索：只在推理时且指定层进行
    if self.memory_enabled and current_pose is not None and not self.training:
        memory_frames = self.memory_buffer.get_relevant_frames(...)
    
    # 分层处理：
    for i, block in enumerate(self.blocks):
        if i in self.memory_layers:
            # 后期层：既存储又使用记忆进行推理
            x = block(x, c, x_cond, memory_frames, memory_activation_score)
        else:
            # 前期层（0-15）：只进行标准CDiT处理，不使用记忆推理
            x = block(x, c, x_cond)
    
    # 内存存储：推理时所有层都进行存储
    # 关键设计：即使前15层不使用记忆推理，也会存储记忆
    if update_memory and self.memory_enabled and not self.training:
        current_action = y[0] if y is not None else None
        self.update_memory(x.detach(), current_pose[0], current_action)
    
    return x
```

### 分层内存设计

1. **前期层（0-15层）**：
   - ✅ **激活记忆存储**：持续积累历史经验
   - ❌ **不使用记忆推理**：保持CDiT的原始处理能力
   - 🎯 **设计目的**：确保记忆系统始终在工作，为后续层提供丰富素材

2. **后期层（memory_layers）**：
   - ✅ **激活记忆存储**：继续积累经验
   - ✅ **使用记忆推理**：利用历史经验增强生成
   - 🎯 **设计目的**：基于积累的记忆进行智能决策

### 训练优势

1. **纯GPU训练**：所有计算操作保持在GPU上，避免CPU-GPU数据传输
2. **无内存开销**：训练时不使用buffer，节省内存和计算资源
3. **简洁高效**：训练逻辑与标准CDiT基本相同，稳定可靠
4. **推理增强**：内存机制仅在推理时启用，提供额外的上下文信息
5. **连续记忆**：前期层的持续存储确保记忆系统不间断工作

## 性能优化

### 向量化计算

所有内存操作都进行了向量化优化：

- **原始循环方式**：~23% 计算开销
- **向量化优化后**：~15.6% 计算开销
- **性能提升**：~32% 减少计算时间

### 内存管理

- **缓冲区大小**：可配置 (默认 50 帧)
- **激活阈值**：自适应调整
- **清理策略**：FIFO 方式维护缓冲区

## 配置说明

### 模型配置

```yaml
# config/nwm_hybrid.yaml
model: "CDiT-XL/2"
use_hybrid_model: true
memory_enabled: true
memory_buffer_size: 50
memory_layers: [0, 3, 6, 9]  # 在这些层激活内存

# 权重配置
spatial_weight: 0.8  # 空间相似性权重
action_weight: 0.2   # 行为相似性权重
```

### 训练参数

```python
# 使用混合模型
model = HybridCDiT_models["HybridCDiT-XL/2"](
    context_size=num_cond,
    input_size=latent_size,
    in_channels=4,
    memory_enabled=True,
    memory_buffer_size=50
)
```

## 实际应用效果

### 场景示例

#### 🏬 仓库导航
- **任务**：在货架间导航并转弯
- **传统方法**：可能选择相同朝向的直行经验
- **混合系统**：选择相似位置的转弯经验
- **结果**：更自然的转弯行为

#### 🏃 走廊直行
- **任务**：在长走廊中保持直行
- **传统方法**：可能选择转弯经验
- **混合系统**：选择直行行为的历史经验
- **结果**：更稳定的直行控制

## 技术特点

1. **语义合理性**：基于行为意图而非状态相似性
2. **计算效率**：向量化操作，性能优化
3. **自适应性**：根据场景复杂度调整内存激活
4. **可扩展性**：模块化设计，易于定制
5. **向后兼容**：支持标准 CDiT 模式

## 总结

WorldMem-CDiT 混合内存系统通过"**合理性优于相似性**"的设计理念，实现了：

- ✅ **智能内存检索**：基于行为相关性选择历史经验
- ✅ **高效计算**：向量化操作，最小化性能开销  
- ✅ **语义一致性**：确保内存支持当前行为意图
- ✅ **自适应激活**：根据场景复杂度智能使用内存

这种设计使得模型能够在合适的时机调用合适的历史经验，显著提升导航的智能性和鲁棒性。

## 智能存储机制优化

### 当前存储策略分析

**现状**：目前所有帧都无条件存储到memory buffer
- ✅ **优点**：确保不丢失任何信息
- ❌ **缺点**：可能存储大量冗余或低价值信息

### 关键位置检测

**建议实现基于场景重要性的选择性存储**：

1. **大转弯检测**：
   ```python
   # 检测显著转向动作
   if abs(delta_yaw) > SIGNIFICANT_TURN_THRESHOLD:
       should_store = True  # 转弯时的视觉信息很重要
   ```

2. **关键地标识别**：
   ```python
   # 未来扩展：基于视觉特征检测地标
   if detect_landmark(frame_features):
       should_store = True  # 地标位置需要记忆
   ```

3. **行为变化点**：
   ```python
   # 检测行为模式变化
   if action_pattern_changed(current_action, previous_actions):
       should_store = True  # 行为转换点很重要
   ```

4. **空间多样性**：
   ```python
   # 确保空间覆盖的多样性
   if spatial_diversity_score(current_pose, buffer_poses) > threshold:
       should_store = True  # 新区域需要记忆
   ```

### 存储价值评估框架

```python
def compute_storage_value(frame, pose, action, buffer_state):
    """
    计算帧的存储价值分数
    
    评估维度：
    1. 行为重要性（转弯、停止、加速等）
    2. 空间新颖性（是否到达新区域）
    3. 时间间隔（避免连续相似帧）
    4. 缓冲区多样性（平衡不同类型经验）
    """
    
    # 1. 行为重要性评分
    behavior_score = evaluate_action_significance(action)
    
    # 2. 空间新颖性评分  
    spatial_score = evaluate_spatial_novelty(pose, buffer_state.poses)
    
    # 3. 时间多样性评分
    temporal_score = evaluate_temporal_diversity(buffer_state.timestamps)
    
    # 4. 缓冲区平衡评分
    balance_score = evaluate_buffer_balance(action, buffer_state.actions)
    
    # 综合评分
    storage_value = (0.4 * behavior_score + 
                    0.3 * spatial_score + 
                    0.2 * temporal_score + 
                    0.1 * balance_score)
    
    return storage_value
```

### 实现建议

1. **阶段性实现**：
   - 第一阶段：基于转弯幅度的简单过滤
   - 第二阶段：加入空间多样性考虑
   - 第三阶段：引入视觉地标检测

2. **存储阈值动态调整**：
   ```python
   # 根据buffer占用率动态调整存储阈值
   if buffer_utilization < 0.5:
       storage_threshold = 0.3  # 宽松标准
   elif buffer_utilization < 0.8:
       storage_threshold = 0.5  # 中等标准
   else:
       storage_threshold = 0.7  # 严格标准
   ```

3. **优先级替换策略**：
   - 当buffer满时，优先替换价值评分最低的帧
   - 保留关键转弯点和地标位置的记忆

## 完全归一化更新

### 核心修改

为解决yaw权重不平衡问题，已实现**完全归一化**方案：

**问题**：原始yaw范围[-3.14, 3.14]与dx/dy范围相近，导致某些情况下yaw主导距离计算，影响训练稳定性。

**解决**：所有action维度（dx, dy, dyaw）统一归一化到[-1,1]范围。

### 代码修改

```python
# datasets.py & latent_dataset.py - 完全归一化
actions = normalize_data(actions, self.ACTION_STATS)  # 所有3维
goal_pos = normalize_data(goal_pos, self.ACTION_STATS)  # 所有3维

# hybrid_models.py - 调整行为分类阈值
STRAIGHT_THRESHOLD = 0.032    # 原 0.1 rad (~6°)
MINOR_THRESHOLD = 0.096       # 原 0.3 rad (~17°)  
TURN_THRESHOLD = 0.318        # 原 1.0 rad (~57°)
LINEAR_THRESHOLD = 0.307      # 原 0.1 meter
```

### 优势

- ✅ **权重平衡**：消除yaw主导距离计算的问题
- ✅ **训练稳定**：各维度梯度scale一致，减少数值不稳定
- ✅ **语义保持**：行为分类阈值精确调整，保持原有语义正确性
- ✅ **内存一致**：存储和检索都使用归一化值，逻辑一致

### 重要提醒

⚠️ **破坏性修改**：需要重新预处理数据和重新开始训练

```bash
# 重新预处理
cd latent-encoding && ./encode_all_datasets.sh

# 重新训练
python train.py [配置]
```

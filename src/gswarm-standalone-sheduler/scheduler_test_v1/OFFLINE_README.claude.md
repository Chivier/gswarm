## 离线工作流调度策略文档（优化版）

## 1. 背景与目标

### 1.1 问题背景
在AI工作流推理场景中，用户提交的工作流（Workflow）由多个节点（Node）组成，每个节点代表对特定AI模型的一次调用。在离线批处理场景下，我们需要处理大量已积累的工作流请求，目标是以最短的总时间（makespan）完成所有请求的处理。

### 1.2 核心挑战
- **模型切换开销**：在GPU上切换模型需要卸载当前模型并加载新模型，这个过程耗时较长（通常需要数秒到数十秒）
- **依赖约束**：工作流中的节点存在依赖关系，必须按照拓扑顺序执行
- **资源限制**：GPU数量有限，某些大模型可能需要多个GPU
- **负载均衡**：需要合理分配任务到各个GPU，避免资源闲置

### 1.3 优化目标
1. **最小化总执行时间（makespan）**：从开始执行到所有请求完成的总时间
2. **最大化GPU利用率**：确保所有GPU都得到充分利用，避免负载不均
3. **减少模型切换次数**：在保证负载均衡的前提下，尽量减少模型切换

## 2. 调度策略设计

### 2.1 核心思想
采用**优先级队列 + 动态负载均衡**的策略：

1. **细粒度调度**：不使用大批次，而是逐个节点进行调度，提高并行度
2. **动态负载均衡**：使用评分机制动态选择最佳GPU，避免某些GPU过载
3. **依赖驱动执行**：使用优先级队列管理就绪节点，确保依赖正确处理
4. **模型亲和性**：在负载均衡的前提下，优先使用已加载相应模型的GPU

### 2.2 算法流程

#### 阶段1：初始化
```
1. 创建所有节点的执行对象
2. 计算每个节点的拓扑层级
3. 将无依赖的节点加入就绪队列
4. 将有依赖的节点加入待处理集合
```

#### 阶段2：动态调度
```
while 就绪队列不为空 或 待处理集合不为空:
    1. 如果就绪队列为空，检查待处理节点是否有新的就绪
    2. 从就绪队列取出优先级最高的节点
    3. 计算节点的依赖就绪时间
    4. 为节点选择最佳GPU（考虑负载均衡）
    5. 调度节点到选定的GPU
    6. 更新GPU状态和完成时间表
    7. 检查是否有新节点变为就绪状态
```

#### 阶段3：GPU选择算法
```python
def find_best_gpu(node, ready_time):
    best_score = infinity
    best_gpu = None
    
    for gpu in all_gpus:
        # 计算GPU可用时间
        gpu_available = gpu.available_at
        
        # 计算模型切换时间
        switch_time = get_switch_time(gpu.current_model, node.model)
        
        # 任务完成时间
        completion_time = max(gpu_available, ready_time) + switch_time + node.exec_time
        
        # 负载均衡因子（GPU总忙碌时间）
        load_factor = gpu.total_busy_time * 0.1
        
        # 综合评分：完成时间为主，负载均衡为辅
        score = completion_time + load_factor
        
        if score < best_score:
            best_score = score
            best_gpu = gpu
    
    return best_gpu
```

### 2.3 关键优化技术

#### 2.3.1 优先级队列管理
- **优先级计算**：节点优先级 = 拓扑层级 + 随机小量（避免同层级节点总是相同顺序）
- **动态更新**：完成一个节点后立即检查其依赖节点是否就绪
- **死锁检测**：如果就绪队列为空但仍有待处理节点，检查是否存在循环依赖

#### 2.3.2 负载均衡评分
- **多因素评分**：综合考虑任务完成时间和GPU历史负载
- **动态权重**：根据当前系统状态调整负载均衡的权重
- **实时调整**：每次调度都重新评估所有GPU的状态

#### 2.3.3 模型切换优化
- **局部贪心**：在不影响负载均衡的前提下，优先选择已加载模型的GPU
- **切换成本精确计算**：基于模型大小和PCIe带宽计算准确的切换时间
- **切换统计**：记录总切换次数和时间，用于后续优化

#### 2.3.4 多GPU模型处理
- **连续GPU分配**：为需要多GPU的模型寻找连续可用的GPU组
- **同步调度**：确保多GPU任务在所有相关GPU上同时开始
- **资源预留**：避免多GPU任务被单GPU任务打断

## 3. 实现细节

### 3.1 数据结构设计

#### NodeExecution
```python
@dataclass
class NodeExecution:
    request_id: str          # 请求ID
    workflow_id: str         # 工作流ID
    node_id: str            # 节点ID
    model_name: str         # 模型名称
    status: str             # 状态
    level: int              # 拓扑层级
    estimated_time: float   # 预估执行时间
    start_time: float       # 实际开始时间
    end_time: float         # 实际结束时间
    gpu_id: int            # 分配的GPU
```

#### GPUState
```python
@dataclass
class GPUState:
    gpu_id: int             # GPU编号
    current_model: str      # 当前加载的模型
    available_at: float     # 可用时间
    total_busy_time: float  # 总忙碌时间（用于负载均衡）
```

#### ScheduledTask
```python
@dataclass
class ScheduledTask:
    node: NodeExecution     # 节点
    gpu_id: int            # GPU编号
    start_time: float      # 开始时间
    end_time: float        # 结束时间
    switch_time: float     # 模型切换时间
```

### 3.2 调度示例
```
时间轴示例：
GPU 0: [Model A: Node1] -> [Switch to B] -> [Model B: Node3] -> ...
GPU 1: [Model B: Node2] -> [Model B: Node4] -> [Switch to C] -> ...
GPU 2: [Model C: Node5] -> [Model C: Node6] -> ...

通过动态负载均衡，确保所有GPU的利用率接近。
```

## 1. 背景与目标

### 1.1 问题背景
在AI工作流推理场景中，用户提交的工作流（Workflow）由多个节点（Node）组成，每个节点代表对特定AI模型的一次调用。在离线批处理场景下，我们需要处理大量已积累的工作流请求，目标是以最短的总时间（makespan）完成所有请求的处理。

### 1.2 核心挑战
- **模型切换开销**：在GPU上切换模型需要卸载当前模型并加载新模型，这个过程耗时较长（通常需要数秒到数十秒）
- **依赖约束**：工作流中的节点存在依赖关系，必须按照拓扑顺序执行
- **资源限制**：GPU数量有限，某些大模型可能需要多个GPU
- **负载均衡**：需要合理分配任务到各个GPU，避免资源闲置

### 1.3 优化目标
1. **最小化总执行时间（makespan）**：从开始执行到所有请求完成的总时间
2. **最小化模型切换次数**：减少GPU上的模型切换，降低切换开销
3. **最大化GPU利用率**：确保GPU资源得到充分利用

## 2. 调度策略设计

### 2.1 核心思想
利用离线场景的全局信息优势，通过以下策略优化调度：

1. **全局批处理**：将所有工作流的所有节点统一考虑，而非逐个处理
2. **模型分组**：将使用相同模型的节点分组成批次（batch），同一批次内的节点可以连续执行
3. **拓扑排序**：在满足依赖关系的前提下，尽可能将相同模型的节点安排在一起
4. **贪心分配**：使用贪心算法将批次分配到GPU，优先考虑已加载相应模型的GPU

### 2.2 算法流程

#### 阶段1：节点创建与分析
```
1. 遍历所有工作流请求
2. 为每个请求的每个节点创建执行对象
3. 分析节点间的依赖关系
4. 计算每个节点的拓扑层级（topological level）
```

#### 阶段2：批次生成
```
1. 按模型类型对节点进行分组
2. 在每个模型组内，按拓扑层级进一步分组
3. 生成批次列表，每个批次包含：
   - 相同模型类型
   - 相同或相近拓扑层级
   - 满足依赖约束的节点集合
4. 对批次进行排序：
   - 首先按最小拓扑层级排序（保证依赖顺序）
   - 同层级内按预期执行时间降序排序（大任务优先）
```

#### 阶段3：GPU分配
```
1. 初始化GPU状态（空闲时间、当前模型等）
2. 对每个批次：
   a. 计算批次的依赖就绪时间
   b. 评估每个可用GPU的调度成本：
      - GPU空闲时间
      - 模型切换时间（如果需要）
   c. 选择总成本最小的GPU
   d. 更新GPU状态和节点完成时间
```

#### 阶段4：执行调度
```
1. 按照计算出的调度方案执行
2. 记录实际执行时间和模型切换次数
3. 处理完成事件并更新状态
```

### 2.3 关键优化技术

#### 2.3.1 智能批次划分
- **依赖感知分组**：确保同一批次内的节点不存在相互依赖
- **层级优先**：优先将同一拓扑层级的节点分为一组
- **批次大小优化**：平衡批次大小，避免过小（增加切换）或过大（降低并行度）

#### 2.3.2 批次内串行执行
- **GPU独占原则**：同一时刻一个GPU只能执行一个任务
- **批次内串行**：同一批次内的节点按顺序串行执行，避免资源冲突
- **依赖检查**：每个节点开始前检查其特定依赖是否满足
- **时间累积**：批次的总时间是所有节点执行时间的累加

#### 2.3.3 模型亲和性调度
- **模型复用**：优先将批次分配到已加载相应模型的GPU
- **切换成本计算**：基于模型大小和PCIe带宽计算精确的切换时间
- **预加载策略**：在GPU即将空闲时提前加载下一个模型

#### 2.3.4 负载均衡
- **动态分配**：根据实时GPU状态动态分配批次
- **工作窃取**：当某GPU提前完成时，可以从其他GPU窃取任务
- **多GPU模型处理**：对于需要多GPU的大模型，协调多个GPU的调度

## 3. 实现细节

### 3.1 数据结构设计

#### NodeExecution
```python
@dataclass
class NodeExecution:
    request_id: str          # 请求ID
    workflow_id: str         # 工作流ID
    node_id: str            # 节点ID
    model_name: str         # 模型名称
    status: str             # 状态：pending/ready/running/completed
    level: int              # 拓扑层级
    estimated_time: float   # 预估执行时间
    start_time: float       # 实际开始时间
    end_time: float         # 实际结束时间
    dependencies_completed: Set[str]  # 已完成的依赖
```

#### ModelBatch
```python
@dataclass
class ModelBatch:
    model_name: str         # 模型名称
    nodes: List[NodeExecution]  # 批次中的节点
    start_time: float       # 批次开始时间
    end_time: float         # 批次结束时间
```

### 3.2 批次执行逻辑
```python
# 批次内节点串行执行
current_time = batch.start_time
for node in batch.nodes:
    # 检查节点特定的依赖
    node_deps_ready = check_dependencies(node)
    
    # 节点开始时间 = max(GPU空闲时间, 依赖就绪时间)
    node.start_time = max(current_time, node_deps_ready)
    node.end_time = node.start_time + node.estimated_time
    
    # 更新时间轴
    current_time = node.end_time

# 批次结束时间 = 最后一个节点的结束时间
batch.end_time = current_time
```

### 3.2 模型切换时间计算
```python
def _get_model_switch_time(from_model: str, to_model: str) -> float:
    if from_model == to_model:
        return 0.0
    
    # 计算数据传输时间
    from_size = models[from_model].memory_gb
    to_size = models[to_model].memory_gb
    transfer_time = (from_size + to_size) / PCIE_BANDWIDTH_GB_S
    
    # 加上固定开销
    overhead = 2.0  # 秒
    return transfer_time + overhead
```

### 3.3 依赖处理
- 使用拓扑排序确保执行顺序
- 维护节点完成时间表，用于计算后续节点的最早开始时间
- 批次调度时检查所有节点的依赖是否满足

## 4. 性能分析

### 4.1 时间复杂度
- 节点创建：O(N)，N为总节点数
- 优先级队列操作：O(N log N)，每个节点入队出队一次
- GPU选择：O(G)，G为GPU数量
- 总体复杂度：O(N log N + N×G)

### 4.2 空间复杂度
- O(N)用于存储节点和执行状态
- O(N)用于优先级队列和待处理集合

### 4.3 优化效果
相比基础调度策略，本优化版本的优势：
1. **更高的GPU利用率**：通过动态负载均衡，所有GPU的利用率都接近平均水平
2. **更短的makespan**：避免了某些GPU空闲而其他GPU过载的情况
3. **灵活的调度**：细粒度调度允许更好地适应不同的工作负载
4. **更好的扩展性**：算法复杂度与GPU数量线性相关，适合大规模GPU集群

### 4.4 对比分析
| 指标 | 批处理策略 | 动态负载均衡策略 |
|------|------------|------------------|
| GPU利用率差异 | 高（可能50%差异） | 低（通常<10%差异） |
| 调度开销 | 低 | 中等 |
| 适应性 | 差 | 好 |
| makespan | 较长 | 较短 |

## 5. 扩展与优化方向

### 5.1 抢占式调度
- 支持高优先级任务抢占低优先级任务
- 实现任务的暂停和恢复机制

### 5.2 预测优化
- 基于历史数据预测任务执行时间
- 使用机器学习优化调度决策

### 5.3 能效优化
- 考虑GPU功耗，实现能效最优调度
- 支持动态电压频率调整（DVFS）

### 5.4 容错机制
- 任务失败自动重试
- GPU故障时的任务迁移
- 检查点机制支持长任务

### 5.5 混合调度
- 同时支持在线和离线任务
- 实现QoS保证机制

## 6. 总结

本优化版离线调度策略通过以下创新实现了更好的性能：

1. **细粒度调度**：逐节点调度而非大批次，提高了并行度和GPU利用率
2. **动态负载均衡**：使用评分机制确保所有GPU负载均衡，避免资源浪费
3. **优先级管理**：通过优先级队列和依赖追踪，确保高效的执行顺序
4. **灵活性**：能够适应各种工作负载模式，包括不均匀的任务分布

该策略特别适合：
- GPU数量较多的场景（避免部分GPU空闲）
- 任务执行时间差异较大的场景
- 需要最小化总执行时间的离线批处理场景

通过实施这些优化，可以显著提升系统的整体吞吐量和资源利用率，减少任务完成时间。
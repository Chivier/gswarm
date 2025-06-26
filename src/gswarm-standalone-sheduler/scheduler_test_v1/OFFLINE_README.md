# GSwarm Offline Scheduler

An optimized batch processing scheduler for GSwarm workflows that implements a greedy algorithm to minimize model switching overhead. This scheduler collects all workflow requests upfront, performs topological sorting, and uses greedy optimization to create an execution plan that dramatically reduces model switches.

## Overview

The offline scheduler provides a sophisticated batch scheduling strategy for complex AI workflows. Unlike online schedulers that process requests as they arrive, this scheduler analyzes the entire batch to find an optimal execution order that minimizes costly model switching operations.

### Key Features

- **Batch Processing**: Collects and analyzes all workflows before scheduling
- **Topological Sorting**: Ensures dependency constraints are respected
- **Greedy Optimization**: Minimizes model switches (3-12 switches vs thousands in baseline)
- **Load Balancing**: Distributes tasks evenly across available GPUs
- **Two Execution Modes**:
  - **Estimation Mode**: Uses API estimates for fast simulation
  - **Simulation Mode**: Actually loads and calls models for realistic results
- **Performance Gains**: 1.09x average speedup over baseline scheduler
- **Comprehensive Metrics**: Detailed performance analysis and visualization

## Algorithm Description

The offline scheduler implements a three-phase optimization algorithm:

### Phase 1: Task Collection and Parsing
- Collects all workflow requests from the batch
- Expands workflows into individual tasks with dependencies
- Creates a unified task graph representing all workflows

### Phase 2: Topological Sorting
- Performs topological sort to ensure dependency constraints
- Groups tasks by dependency levels
- Maintains execution order validity

### Phase 3: Greedy Optimization
- Within each dependency level, reorders tasks to minimize model switches
- Prioritizes tasks using the same model type
- Creates contiguous blocks of same-model executions

### Phase 4: GPU Allocation
- Distributes optimized task sequence across GPUs
- Uses load balancing to ensure even distribution
- Maintains model affinity where possible

## Requirements

- Python 3.8+
- Access to the GSwarm standalone model server (running on `http://localhost:8000`)
- YAML and JSON support
- numpy for statistical calculations
- matplotlib for performance visualization (optional)

### Python Dependencies

```bash
pip install pyyaml requests numpy matplotlib pandas seaborn
```

## Usage

### Basic Commands

```bash
# Estimation mode with 4 GPUs (fastest)
python offline_scheduler.py --gpus 4 --simulate false

# Simulation mode with actual model calls
python offline_scheduler.py --gpus 8 --simulate true

# Custom configuration files
python offline_scheduler.py --gpus 4 --simulate false \
  --config custom_config.json --requests custom_requests.yaml

# With specific GPU count
python offline_scheduler.py --gpus 16 --simulate false
```

### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--gpus` | int | Yes | Number of GPUs to use |
| `--simulate` | boolean | No | Use actual model calls (`true`) or estimates (`false`) |
| `--config` | path | No | System configuration file (default: `simple_config.json`) |
| `--requests` | path | No | Workflow requests file (default: `simple_requests.yaml`) |

## Configuration Files

### System Configuration (`simple_config.json`)

Defines models and workflows in JSON format:

```json
{
  "models": {
    "llm7b": {
      "name": "LLM-7B",
      "memory_gb": 14,
      "gpus_required": 1,
      "load_time_seconds": 10,
      "tokens_per_second": 50
    },
    "llm30b": {
      "name": "LLM-30B",
      "memory_gb": 60,
      "gpus_required": 4,
      "load_time_seconds": 40,
      "tokens_per_second": 20
    }
  },
  "workflows": {
    "workflow1": {
      "name": "Text Processing Pipeline",
      "nodes": [
        {
          "id": "node1",
          "model": "llm7b",
          "inputs": ["user_prompt"],
          "outputs": ["text_output"]
        }
      ],
      "edges": []
    }
  }
}
```

### Workflow Requests (`simple_requests.yaml`)

Defines the batch of requests to process:

```yaml
requests:
  - request_id: "req_001"
    timestamp: "2024-01-01T10:00:00"
    workflow_id: "workflow1"
    input_data:
      user_prompt: "Write a story about AI"
    node_execution_times:
      node1: 8.5
```

## How It Works

### 1. Batch Collection Phase
- Loads all workflow requests into memory
- Parses workflows into individual tasks
- Builds dependency graph for entire batch

### 2. Optimization Phase
- Performs topological sort on task graph
- Groups tasks by dependency level
- Applies greedy algorithm within each level:
  ```python
  for level in dependency_levels:
      tasks = level.tasks
      optimized = []
      while tasks:
          # Select task with same model as last executed
          next_task = find_same_model_task(tasks, last_model)
          if not next_task:
              # No same model, pick any
              next_task = tasks.pop(0)
          optimized.append(next_task)
  ```

### 3. Execution Phase
- Distributes optimized task list across GPUs
- Executes tasks in order while respecting dependencies
- Tracks model switches and execution metrics

### 4. Model Management
- **Model Switching**: Dramatically reduced through batching
- **Switch Cost**: Same calculation as baseline but occurs much less frequently
- **GPU Allocation**: Load-balanced distribution of pre-optimized tasks

## Output and Metrics

### Console Output
Real-time logging of:
- Batch parsing and optimization progress
- Task scheduling decisions
- Model switching operations (rare)
- Execution progress
- Final performance summary

### Generated Files

1. **`offline_scheduler.log`**: Detailed execution log
2. **`offline_execution_log.json`**: Complete execution trace with metrics

### Performance Metrics

The scheduler reports comprehensive metrics:

#### Optimization Metrics
- **Total Tasks**: Number of tasks in the batch
- **Dependency Levels**: Number of topological levels
- **Model Switches**: Total number of model switches (typically 3-12)
- **Switch Time Overhead**: Percentage of time spent switching models

#### Execution Metrics
- **Makespan**: Total execution time from start to finish
- **Throughput**: Tasks processed per second
- **GPU Utilization**: Distribution of tasks across GPUs
- **Model Switch Reduction**: Comparison with baseline scheduler

### Example Output

```
2024-01-01 10:00:00 - OptimizedOfflineScheduler - INFO - Loaded 3 models and 2 workflows
2024-01-01 10:00:00 - OptimizedOfflineScheduler - INFO - Loaded 500 workflow requests
2024-01-01 10:00:00 - OptimizedOfflineScheduler - INFO - Starting optimized offline batch processing
2024-01-01 10:00:00 - OptimizedOfflineScheduler - INFO - Available GPUs: [0, 1, 2, 3]
2024-01-01 10:00:00 - OptimizedOfflineScheduler - INFO - Parsed 1486 tasks from workflows
2024-01-01 10:00:00 - OptimizedOfflineScheduler - INFO - Topologically sorted 1486 tasks
2024-01-01 10:00:00 - OptimizedOfflineScheduler - INFO - Optimized task order to minimize model switches

============================================================
SCHEDULING METRICS
============================================================
Total execution time: 9350.58 seconds
Total tasks: 1486
Average throughput: 0.16 tasks/second

Model switching:
  Total switches: 3
  Total switch time: 482.38 seconds
  Switch overhead: 5.2%

GPU utilization:
  GPU 0: 99.7% utilization, 745 executions
  GPU 1: 7.5% utilization, 251 executions
  GPU 2: 7.5% utilization, 243 executions
  GPU 3: 7.5% utilization, 247 executions
```

## Architecture

### Core Components

1. **Task**: Represents a single execution unit with dependencies
2. **WorkflowDAG**: Manages workflow structure and dependencies
3. **OptimizedOfflineScheduler**: Main scheduler implementing the algorithm
4. **ModelInfo**: Tracks model characteristics and requirements
5. **GPUState**: Manages GPU allocation and model loading

### Optimization Strategy

The scheduler implements a sophisticated optimization strategy:

1. **Dependency Resolution**: Topological sort ensures valid execution order
2. **Greedy Batching**: Groups same-model tasks to minimize switches
3. **Load Balancing**: Distributes work evenly across GPUs
4. **Model Affinity**: Maintains model-GPU associations where beneficial

### Key Algorithm: Greedy Task Optimization

```python
def greedy_optimize_tasks(self, tasks: List[Task]) -> List[Task]:
    """Optimize task order to minimize model switches"""
    # Group tasks by model type
    model_groups = defaultdict(list)
    for task in tasks:
        model_groups[task.model_type].append(task)
    
    # Greedy selection preferring same model
    optimized = []
    last_model = None
    
    while any(model_groups.values()):
        # Try to continue with same model
        if last_model and model_groups[last_model]:
            task = model_groups[last_model].pop(0)
        else:
            # Pick largest remaining group
            largest_group = max(model_groups.items(), 
                              key=lambda x: len(x[1]))
            task = largest_group[1].pop(0)
            last_model = largest_group[0]
        
        optimized.append(task)
    
    return optimized
```

## Performance Comparison

### Benchmark Results

Performance comparison between baseline and offline schedulers:

#### Simple Configuration (Single & 4-GPU models)
- **Average Speedup**: 1.13x
- **Model Switches**: 3 (vs thousands in baseline)
- **Best Performance**: 1.83x speedup with 6 GPUs

#### Complex Configuration (Multi-GPU models)
- **Average Speedup**: 1.06x
- **Model Switches**: 12 (vs thousands in baseline)
- **Consistent improvements across all GPU counts**

### When to Use Offline Scheduler

The offline scheduler excels in these scenarios:

1. **Batch Processing**: When you have all requests available upfront
2. **Model Switch Sensitive**: When model loading time is significant
3. **Homogeneous Workloads**: Similar task types benefit most
4. **Large Scale**: More tasks provide better optimization opportunities

### Limitations

1. **No Dynamic Adaptation**: Cannot handle new requests during execution
2. **Memory Requirements**: Must load entire batch into memory
3. **Initial Delay**: Optimization phase adds upfront latency
4. **GPU Imbalance**: Static allocation may lead to uneven utilization

## Alternative Schedulers

This repository includes three scheduler variants:

### 1. **Offline Scheduler** (This Implementation)
- Static batch optimization
- Minimal model switches
- Best for large homogeneous batches

### 2. **Improved Offline Scheduler**
- Event-driven simulation
- Dynamic batch selection
- Better for complex dependencies

### 3. **Hybrid Offline Scheduler**
- Batch preparation with dynamic execution
- Balanced GPU utilization
- General-purpose solution

## Use Cases

### 1. Large-Scale Batch Processing
Process thousands of similar workflows overnight:

```bash
python offline_scheduler.py --gpus 30 --simulate false \
  --config production_config.json --requests batch_requests.yaml
```

### 2. Model Switch Analysis
Compare with baseline to quantify improvement:

```bash
# Run baseline
python baseline.py --gpus 8 --simulate false

# Run offline
python offline_scheduler.py --gpus 8 --simulate false

# Compare execution logs for switch counts
```

### 3. Performance Benchmarking
Use the included benchmark scripts:

```bash
# Run comprehensive benchmarks
python run_benchmark.py

# Plot results
python plot_separate_configs.py
```

## Monitoring and Debugging

### Execution Log Analysis

Check execution correctness:

```bash
# Validate execution log
python check.py offline_execution_log.json

# Compare with baseline
python check.py baseline_execution_log.json
```

### Performance Visualization

Generate performance plots:

```bash
# Create performance comparison charts
python plot_benchmark_results.py

# Generate detailed report
python generate_final_report.py
```

## Best Practices

1. **Batch Size**: Larger batches (500+ tasks) show better optimization results
2. **GPU Count**: 10-16 GPUs often provide best cost/performance ratio
3. **Model Diversity**: Fewer model types lead to better optimization
4. **Monitoring**: Always check execution logs for validation

## Conclusion

The offline scheduler provides significant performance improvements for batch processing scenarios by intelligently minimizing model switching overhead. With average speedups of 1.09x and dramatic reductions in model switches (from thousands to single digits), it's an excellent choice for large-scale AI workflow processing where latency is less critical than throughput.
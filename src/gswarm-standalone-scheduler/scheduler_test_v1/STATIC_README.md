# GSwarm Static Deployment Scheduler

A static deployment scheduler for GSwarm workflows that implements persistent model loading strategy. This scheduler eliminates model switching overhead by permanently assigning models to GPUs and optimizing for minimal cross-server communication.

## Overview

The static scheduler provides a zero-switching deployment strategy for AI workflows. Unlike dynamic schedulers that load and unload models, this scheduler pre-assigns models to GPUs based on workflow patterns and maintains these assignments throughout execution.

### Key Features

- **Zero Model Switching**: Each GPU permanently hosts specific models
- **Server-Aware Placement**: Groups GPUs into servers (default: 4 GPUs/server)
- **Workflow Locality Optimization**: Attempts to complete workflows within single servers
- **Pattern-Based Assignment**: Analyzes workflow patterns to optimize model placement
- **Cross-Server Communication Tracking**: Monitors and minimizes inter-server data transfers
- **Comprehensive Metrics**: Tracks server efficiency, GPU utilization, and communication costs

## Algorithm Strategy

The static scheduler implements a multi-phase optimization approach:

### Phase 1: Workflow Pattern Analysis
- Analyzes model usage frequency across all workflows
- Tracks model co-occurrence patterns
- Identifies workflow frequencies

### Phase 2: Model-to-GPU Assignment
- Assigns models to GPUs based on usage patterns
- Groups frequently co-occurring models on same server
- Handles multi-GPU models with consecutive GPU allocation

### Phase 3: Server Organization
- Groups GPUs into logical servers
- Optimizes for workflow locality
- Minimizes cross-server communication

### Phase 4: Runtime Scheduling
- Routes requests to appropriate GPUs based on model requirements
- No model switching - immediate execution when GPU available
- Tracks cross-server communications

## Requirements

- Python 3.8+
- YAML and JSON support
- numpy for statistical calculations

### Python Dependencies

```bash
pip install pyyaml numpy
```

## Usage

### Basic Commands

```bash
# Basic static deployment with 8 GPUs (2 servers of 4 GPUs each)
python static_scheduler.py --gpus 8 --gpus-per-server 4

# Custom server configuration (8 GPUs per server)
python static_scheduler.py --gpus 16 --gpus-per-server 8

# Online mode with streaming requests
python static_scheduler.py --gpus 8 --mode online

# Custom configuration files
python static_scheduler.py --gpus 8 --config custom_config.json --requests custom_requests.yaml
```

### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--gpus` | integer | Yes | Total number of available GPUs |
| `--gpus-per-server` | integer | No | GPUs per server (default: 4) |
| `--simulate` | boolean | No | Use actual model calls (`true`) or estimates (`false`, default) |
| `--mode` | string | No | Scheduling mode: `offline` (default) or `online` |
| `--config` | path | No | System configuration file (default: `simple_config.json`) |
| `--requests` | path | No | Workflow requests file (default: `simple_requests.yaml`) |

## How It Works

### 1. Pattern Analysis Phase
```python
# Analyzes workflow patterns
model_usage = {
    "llm7b": 180,      # High usage
    "llm30b": 100,     # Medium usage
    "stable_diffusion": 80  # Medium usage
}

model_cooccurrence = {
    ("llm7b", "stable_diffusion"): 60,  # Frequently used together
    ("llm7b", "llm30b"): 40
}
```

### 2. Model Assignment Phase
```python
# Example assignment for 8 GPUs (2 servers)
Server 0 (GPUs 0-3):
  GPU 0: llm7b (high usage, dedicated)
  GPU 1: stable_diffusion (pairs with llm7b)
  GPU 2-3: llm30b (multi-GPU model)

Server 1 (GPUs 4-7):
  GPU 4: llm7b (replica for load balancing)
  GPU 5: clip_model
  GPU 6-7: mixtral_8x7b (multi-GPU model)
```

### 3. Runtime Execution
- No model switching delays
- Workflows preferentially scheduled within single server
- Cross-server communication tracked and minimized

## Output and Metrics

### Console Output
Real-time logging with:
- Model-to-GPU assignments
- Server configurations
- Request processing events
- Server efficiency metrics
- Cross-server communication counts

### Generated Files

1. **`static_scheduler.log`**: Detailed execution log
2. **`static_execution_log.json`**: Complete execution trace with static deployment metrics

### Performance Metrics

#### Static Deployment Specific Metrics

**1. Server Efficiency (SE)**
```
SE = (Intra-server workflows / Total workflows) × 100%
```
Higher is better - indicates workflows completing within single servers.

**2. Cross-Server Communication (CSC)**
```
CSC = Total number of edges crossing server boundaries
```
Lower is better - indicates better workflow locality.

**3. GPU Utilization Balance**
```
Utilization Variance = Var(GPU_utilizations)
```
Lower variance indicates better load balancing.

#### Standard Metrics
- **Makespan**: Total execution time
- **Throughput**: Tasks processed per second
- **Request Latency**: Average, P99, min, max
- **Model Switches**: Always 0 in static deployment
- **Switch Overhead**: Always 0% in static deployment

### Example Output

```
============================================================
EXECUTION METRICS
============================================================
Total execution time (makespan): 1234.56 seconds

Static deployment metrics:
  Server efficiency: 85.3%
  Intra-server workflows: 853
  Inter-server workflows: 147
  Cross-server communications: 294

Request completion times:
  Average: 45.67s
  Median: 42.30s
  P99: 78.90s
  Min: 12.34s
  Max: 89.01s

GPU utilization:
  GPU 0 (Server 0): 234 executions, 1100.50s busy time (89.2% utilization), Models: {'llm7b'}
  GPU 1 (Server 0): 189 executions, 980.30s busy time (79.4% utilization), Models: {'stable_diffusion'}
  GPU 2 (Server 0): 156 executions, 890.20s busy time (72.1% utilization), Models: {'llm30b'}
  GPU 3 (Server 0): 156 executions, 890.20s busy time (72.1% utilization), Models: {'llm30b'}

Overall GPU statistics:
  Average utilization: 78.2%
  Utilization variance: 64.5

Model switching:
  Total switches: 0
  Total switch time: 0.00 seconds
  Switch overhead: 0.0%
```

## Execution Log Format

The `static_execution_log.json` includes additional static deployment information:

```json
{
  "summary": {
    "total_requests": 1000,
    "completed_requests": 1000,
    "total_nodes_executed": 3456,
    "mode": "offline",
    "simulate": false,
    "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
    "makespan": 1234.56,
    "total_model_switches": 0,
    "total_switch_time": 0,
    "server_efficiency": 85.3,
    "intra_server_workflows": 853,
    "inter_server_workflows": 147,
    "cross_server_communications": 294,
    "gpus_per_server": 4
  },
  "gpu_assignments": {
    "0": {
      "server_id": 0,
      "assigned_models": ["llm7b"],
      "utilization": 89.2
    }
  },
  "server_assignments": {
    "0": {
      "gpu_ids": [0, 1, 2, 3],
      "models": ["llm7b", "stable_diffusion", "llm30b"]
    }
  },
  "executions": [...]
}
```

## Architecture

### Core Components

1. **StaticGPUState**: Extended GPU state tracking assigned models and server
2. **ServerInfo**: Tracks server configuration and hosted models
3. **ModelAffinity**: Analyzes model co-occurrence patterns
4. **StaticScheduler**: Main scheduler implementing static deployment

### Key Algorithms

#### Model Assignment Algorithm
```python
# Simplified assignment logic
1. Sort models by usage frequency
2. For each model:
   - If multi-GPU: find server with enough consecutive free GPUs
   - If single-GPU: round-robin assignment with affinity consideration
3. Update server model sets
```

#### Workflow Routing
```python
# Check if workflow can complete within single server
def can_complete_locally(workflow, server):
    required_models = get_workflow_models(workflow)
    return required_models.issubset(server.models)
```

## Use Cases

### 1. High-Throughput Production
When you have predictable workload patterns and need maximum throughput:

```bash
python static_scheduler.py --gpus 32 --gpus-per-server 8
```

### 2. Multi-Server Deployment
For distributed deployments across multiple physical servers:

```bash
python static_scheduler.py --gpus 16 --gpus-per-server 4
```

### 3. Specialized Model Hosting
When certain models should be co-located for performance:

```bash
# Configure model groups in the assignment phase
python static_scheduler.py --gpus 8 --config specialized_config.json
```

## Advantages and Trade-offs

### Advantages
- **Zero switching overhead**: No time wasted loading/unloading models
- **Predictable performance**: Consistent response times
- **Better locality**: Workflows often complete within single server
- **Simplified operations**: Fixed model placement

### Trade-offs
- **Less flexibility**: Cannot adapt to changing workload patterns
- **Potential underutilization**: Some GPUs may be idle if workload imbalanced
- **Requires good planning**: Initial assignment crucial for performance
- **May need more GPUs**: To ensure all models are available

## Comparison with Other Schedulers

| Metric | Baseline | Offline | Static |
|--------|----------|---------|---------|
| Model Switches | High (1000s) | Low (3-12) | Zero |
| Switch Overhead | 20-30% | 1-5% | 0% |
| Flexibility | High | Medium | Low |
| Predictability | Low | Medium | High |
| Setup Complexity | Low | Medium | High |
| Best For | Dynamic workloads | Batch processing | Production systems |

## Best Practices

1. **Analyze Workload Patterns**: Run pattern analysis on representative workloads before deployment
2. **Plan Server Capacity**: Ensure each server can host commonly co-occurring models
3. **Monitor Utilization**: Track GPU utilization to identify imbalances
4. **Periodic Rebalancing**: Periodically re-analyze patterns and adjust assignments
5. **Replica Strategy**: Add replicas of high-usage models across servers

## Future Enhancements

1. **Adaptive Replication**: Dynamically adjust model replicas based on load
2. **Smart Routing**: Predict optimal server for incoming requests
3. **Elastic Scaling**: Add/remove model instances based on queue lengths
4. **ML-Based Optimization**: Use ML to predict optimal model placements
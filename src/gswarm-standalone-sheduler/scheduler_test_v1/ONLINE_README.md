# Online Greedy Scheduler for AI Workflows

## Overview

The online greedy scheduler implements real-time scheduling for AI workflows with a focus on minimizing user waiting time, particularly P99 latency. It uses a multi-factor cost function to make scheduling decisions as tasks arrive dynamically.

## Key Features

### 1. **Multi-Factor Greedy Cost Function**
```
Cost(task, gpu) = α × WaitTime + β × SwitchCost + γ × FutureImpact + δ × LoadBalance
```

- **WaitTime**: Time the task has been waiting + time until GPU is available
- **SwitchCost**: Model switching overhead (0 if same model)
- **FutureImpact**: Estimated impact on future tasks in the queue
- **LoadBalance**: Penalty for uneven GPU utilization

### 2. **Work Stealing Strategy**
When GPUs become idle, they actively "steal" work from the ready queue:
- Prioritizes tasks that have been waiting longest
- Ensures no GPU remains idle when work is available
- Reduces overall system latency

### 3. **Load Balancing**
- Tracks execution count and busy time per GPU
- Penalizes overloaded GPUs in cost calculation
- Achieves better distribution of work across GPUs

### 4. **Queue-Wide Optimization**
- Considers impact on entire queue, not just individual tasks
- Prioritizes tasks that unblock others
- Handles multi-GPU models specially to avoid fragmentation

## Usage

```bash
python online_scheduler.py --gpus 4 --config config.json --requests requests.yaml --simulate true
```

### Tunable Parameters

- `--alpha` (default: 1.0): Wait time weight - increase to prioritize reducing wait time
- `--beta` (default: 1.5): Switch cost weight - increase to avoid model switches
- `--gamma` (default: 0.3): Future impact weight - increase for better long-term planning
- `--delta` (default: 0.5): Load balance weight - increase for better GPU utilization
- `--lookahead` (default: 3): Number of future tasks to consider
- `--no-work-stealing`: Disable work stealing (not recommended)

### Recommended Configurations

**For minimum P99 latency (single-node workflows):**
```bash
--alpha 5.0 --beta 0.1 --delta 0.0 --gamma 0.0
```

**For balanced performance (multi-node workflows):**
```bash
--alpha 1.0 --beta 1.5 --delta 1.0 --gamma 0.3
```

**For maximum throughput:**
```bash
--alpha 0.5 --beta 2.0 --delta 2.0 --gamma 0.5
```

## Performance Comparison

### Single-Node Workflows
Compared to baseline FIFO scheduler:
- **P99 waiting time**: 30% improvement (1065s vs 1535s)
- **Average waiting time**: Higher due to load balancing
- **GPU utilization**: More balanced (50-65% per GPU vs 20-99%)

### Multi-Node Workflows
Compared to baseline:
- **P99 waiting time**: Competitive performance
- **GPU utilization**: Significantly more balanced
- **Model switches**: Reduced through intelligent batching

### vs Offline Scheduler
- **Makespan**: 30-40% higher (expected for online algorithm)
- **Flexibility**: Handles dynamic arrivals
- **Fairness**: Better response time distribution

## Implementation Details

### Core Algorithm
1. Maintain priority queue of ready tasks
2. On task arrival:
   - Add to pending tasks
   - Check dependencies
   - Move to ready queue if satisfied
3. On scheduling step:
   - Check for idle GPUs → work stealing
   - Evaluate all ready tasks
   - Select best task-GPU assignment using cost function
   - Schedule and update state

### Multi-GPU Model Handling
- Evaluates all possible GPU sets
- Considers maximum wait time across set
- Reserves auxiliary GPUs during execution
- Releases reservations on completion

### Time Complexity
- Per scheduling decision: O(n × m) where n = ready tasks, m = GPUs
- With lookahead: O(n × m × k) where k = lookahead window
- Work stealing: O(n) per idle GPU

## Monitoring and Metrics

The scheduler tracks:
- Task-level: waiting time, response time
- Request-level: end-to-end response time
- System-level: GPU utilization, model switches, makespan

Output includes:
- Real-time logs during execution
- Summary statistics
- Detailed execution log (JSON)

## Limitations and Future Work

1. **Current Limitations**:
   - No preemption support
   - Limited to homogeneous GPU clusters
   - Assumes accurate execution time estimates

2. **Future Improvements**:
   - Reinforcement learning for parameter tuning
   - Heterogeneous GPU support
   - Dynamic priority adjustment
   - Preemption for long-running tasks

## Example Results

On a 4-GPU cluster with mixed LLM workloads:
```
Total tasks: 500
Makespan: 2164.85s
Average waiting time: 538.92s
P99 waiting time: 1065.13s
GPU utilization: 52-65% (balanced)
Model switches: 0 (with batching)
```

The online greedy scheduler successfully balances the trade-offs between latency, throughput, and resource utilization in dynamic AI workload scenarios.
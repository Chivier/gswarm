# GSwarm Scheduler Guide

This guide explains the different scheduling strategies available in GSwarm for optimizing AI workflow execution across distributed GPU resources.

## Overview

GSwarm provides four different scheduling strategies, each optimized for different scenarios:

1. **Baseline Scheduler**: Ray-like one-by-one execution
2. **Offline Scheduler**: Batch processing with optimization
3. **Online Scheduler**: Real-time scheduling with P99 optimization
4. **Static Scheduler**: Fixed model deployment without switching

## Device Specification

GSwarm uses a precise device notation format to identify specific GPUs across distributed nodes:

```
client:device_type:device_id
```

### Examples:
- `node1:cuda:0` - First CUDA GPU on node1
- `worker2:cuda:3` - Fourth CUDA GPU on worker2
- `localhost:cuda:0` - First CUDA GPU on localhost

### Using Device Format in Schedulers

```python
# New device format (recommended)
scheduler = BaselineScheduler(
    devices=["node1:cuda:0", "node1:cuda:1", "node2:cuda:0"],
    models=models
)

# Legacy format (still supported)
scheduler = BaselineScheduler(
    gpus=[0, 1, 2],  # Assumes localhost
    models=models
)
```

## Scheduler Components

### Base Classes

```python
from gswarm.scheduler import (
    SchedulerBase,
    SchedulingStrategy,
    ModelInfo,
    Workflow,
    Request,
    ScheduledTask,
    ExecutionMetrics
)
```

### Model Information

Define model configurations:

```python
models = {
    "gpt-4": ModelInfo(
        name="gpt-4",
        memory_gb=16.0,
        gpus_required=1,
        load_time_seconds=5.0,
        tokens_per_second=50.0
    ),
    "stable-diffusion": ModelInfo(
        name="stable-diffusion",
        memory_gb=8.0,
        gpus_required=1,
        load_time_seconds=3.0,
        inference_time_mean=2.5
    )
}
```

### Workflow Definition

Create workflows with nodes and dependencies:

```python
from gswarm.scheduler import WorkflowNode, WorkflowEdge, Workflow

# Define nodes
nodes = [
    WorkflowNode(
        id="prompt",
        model="gpt-4",
        inputs=["user_input"],
        outputs=["generated_prompt"]
    ),
    WorkflowNode(
        id="image",
        model="stable-diffusion",
        inputs=["generated_prompt"],
        outputs=["image"]
    )
]

# Define edges (dependencies)
edges = [
    WorkflowEdge(from_node="prompt", to_node="image")
]

# Create workflow
workflow = Workflow(
    id="text2image",
    name="Text to Image Pipeline",
    nodes=nodes,
    edges=edges
)
```

## Using Different Schedulers

### Baseline Scheduler

Best for: Simple workflows, debugging, and when model switching cost is not critical.

```python
from gswarm.scheduler import BaselineScheduler

# Initialize scheduler with new device format
scheduler = BaselineScheduler(
    devices=["node1:cuda:0", "node1:cuda:1", "node2:cuda:0", "node2:cuda:1"],
    models=models,
    simulate=False
)

# Or use legacy format (still supported)
# scheduler = BaselineScheduler(
#     gpus=[0, 1, 2, 3],
#     models=models,
#     simulate=False
# )

# Add workflow
scheduler.add_workflow(workflow)

# Add request
request = Request(
    id="req_001",
    workflow_id="text2image",
    arrival_time=0.0,
    priority=1
)
scheduler.add_request(request)

# Get next task to execute
task = scheduler.get_next_task()
if task:
    print(f"Execute {task.model_name} on device {task.device}")
    # Legacy gpu_id property still available for compatibility
    print(f"  (GPU ID: {task.gpu_id})")
    
# Mark task as completed
scheduler.complete_task(task)
```

### Offline Scheduler

Best for: Batch processing, when all requests are known in advance.

```python
from gswarm.scheduler import OfflineScheduler

# Initialize scheduler with distributed devices
scheduler = OfflineScheduler(
    devices=["compute1:cuda:0", "compute1:cuda:1", "compute2:cuda:0", "compute2:cuda:1"],
    models=models
)

# Add workflows
scheduler.add_workflow(workflow)

# Create batch of requests
requests = [
    Request(id=f"req_{i}", workflow_id="text2image", arrival_time=0.0)
    for i in range(10)
]

# Schedule all at once
scheduled_tasks = scheduler.schedule(requests)

# Tasks are optimally ordered to minimize model switching
for task in scheduled_tasks:
    print(f"Time {task.scheduled_time}: {task.model_name} on {task.device}")
```

### Online Scheduler

Best for: Real-time systems, minimizing P99 latency.

```python
from gswarm.scheduler import OnlineScheduler

# Initialize scheduler with device format
scheduler = OnlineScheduler(
    devices=["worker1:cuda:0", "worker1:cuda:1", "worker2:cuda:0", "worker2:cuda:1"],
    models=models
)

# Add workflow
scheduler.add_workflow(workflow)

# Simulate incoming requests
for i in range(10):
    request = Request(
        id=f"req_{i}",
        workflow_id="text2image",
        arrival_time=i * 0.5,  # Requests arrive every 0.5 seconds
        priority=1
    )
    
    # Advance time
    scheduler.advance_time(request.arrival_time)
    
    # Add request
    scheduler.add_request(request)
    
    # Try to schedule tasks
    while True:
        task = scheduler.get_next_task()
        if task is None:
            break
        print(f"Schedule {task.model_name} for request {task.request_id}")
```

### Static Scheduler

Best for: Production deployments with known workload patterns.

```python
from gswarm.scheduler import StaticScheduler

# Define server configuration with device format
servers = {
    0: ["server1:cuda:0", "server1:cuda:1"],  # Server 0 devices
    1: ["server2:cuda:0", "server2:cuda:1"]   # Server 1 devices
}

# Define model assignments using device strings
model_assignments = {
    "gpt-4": ["server1:cuda:0", "server2:cuda:0"],           # GPT-4 on these devices
    "stable-diffusion": ["server1:cuda:1", "server2:cuda:1"]  # SD on these devices
}

# Initialize scheduler with device format
scheduler = StaticScheduler(
    devices=["server1:cuda:0", "server1:cuda:1", "server2:cuda:0", "server2:cuda:1"],
    models=models,
    servers=servers,
    model_assignments=model_assignments
)

# Add workflow
scheduler.add_workflow(workflow)

# Schedule requests
requests = [Request(id=f"req_{i}", workflow_id="text2image", arrival_time=0.0) for i in range(5)]
scheduled_tasks = scheduler.schedule(requests)

# No model switching overhead in static deployment
for task in scheduled_tasks:
    print(f"Task {task.node_id} runs on {task.device} (pre-loaded with {task.model_name})")
```

## Performance Metrics

All schedulers track performance metrics:

```python
# Get metrics
metrics = scheduler.get_metrics()

print(f"Completed requests: {metrics.completed_requests}")
print(f"Average latency: {metrics.average_request_latency:.2f}s")
print(f"P99 latency: {metrics.p99_latency:.2f}s")
print(f"Model switches: {metrics.model_switch_count}")
print(f"Total model load time: {metrics.total_model_load_time:.2f}s")

# Get GPU utilization
utilization = scheduler.get_gpu_utilization()
for gpu_id, util in utilization.items():
    print(f"GPU {gpu_id}: {util:.1f}% utilized")

# Get device-specific info
for i, device in enumerate(scheduler.devices):
    print(f"Device {device}: {utilization.get(i, 0):.1f}% utilized")
```

## Choosing the Right Scheduler

| Scheduler | Best For | Key Features |
|-----------|----------|--------------|
| Baseline | Simple workflows, debugging | Easy to understand, fair scheduling |
| Offline | Batch jobs, known workload | Minimizes model switching, optimizes makespan |
| Online | Real-time systems | Low P99 latency, prevents starvation |
| Static | Production with fixed models | No switching overhead, predictable performance |

## Advanced Features

### Custom Priority

Set request priorities:

```python
high_priority_request = Request(
    id="urgent_001",
    workflow_id="text2image",
    arrival_time=0.0,
    priority=0  # Lower value = higher priority
)
```

### Deadline Support

Some schedulers support deadlines:

```python
request_with_deadline = Request(
    id="deadline_001",
    workflow_id="text2image",
    arrival_time=0.0,
    deadline=10.0  # Must complete within 10 seconds
)
```

### Multi-Server Support

The static scheduler supports multi-server deployments:

```python
# Check server assignments
server_info = scheduler.get_server_info()
for server_id, info in server_info.items():
    print(f"Server {server_id}: Devices {info['devices']}, Models: {info['models']}")
```

## Integration with Cost Models

Schedulers automatically use cost models for better time estimation:

```python
# The scheduler will use cost models internally
# No additional configuration needed if cost models are trained
scheduler = OnlineScheduler(
    devices=["node1:cuda:0", "node1:cuda:1"],
    models=models
)

# Cost models automatically handle device format
```

## Best Practices

1. **Choose the Right Scheduler**: Match scheduler to your workload pattern
2. **Use Device Format**: Specify `client:device:id` for distributed clarity
3. **Provide Accurate Model Info**: Better estimates lead to better scheduling
4. **Monitor Metrics**: Track performance to identify bottlenecks
5. **Update Cost Models**: Keep cost predictions accurate with real data
6. **Consider Model Placement**: For static scheduler, analyze workflow patterns
7. **Legacy Support**: Both `gpus=[0,1]` and `devices=["node:cuda:0"]` work

## Troubleshooting

### High Model Switching

If seeing excessive model switching:
- Use offline scheduler for batch workloads
- Consider static deployment for production
- Group similar requests together

### High P99 Latency

For real-time systems with high P99:
- Use online scheduler
- Tune aging factor and weights
- Consider over-provisioning GPUs

### Poor GPU Utilization

If GPUs are underutilized:
- Check if model loading time dominates
- Consider static deployment
- Batch more requests together
# Quick Start

## Gswarm Profiler

### 1. Start Host and Client (Worker)

Start host node:
```bash
gswarm host start --port 8090 --http-port 8091 --model-port 9010
```

Start worker node:
```bash
# Connect to host with gRPC
gswarm client connect <host_ip>:8090
```

### 2. Start Profiling

Start profiling via CLI (with auto-discovery):

```bash
gswarm profiler start --name <task_name>
```

Or specify host and metrics:

```bash
gswarm profiler start --name <task_name> --host localhost:50051 --report-metrics gpu_utilization --report-metrics gpu_memory
```

Parameters:
- `name`: Name of your profiling task (auto-generated if not provided)
- `host`: gRPC host address (auto-discovered if not specified)
- `report-metrics`: Specific metrics to collect (can be specified multiple times)

Available metrics:
- `gpu_utilization` - GPU utilization percentage
- `gpu_memory` - GPU memory usage
- `gpu_dram_bandwidth` - DRAM bandwidth utilization
- `gpu_bubble` - GPU bubble metrics for performance analysis

### 3. Stop Profiling

Stop a specific profiling session:
```bash
gswarm profiler stop --name <task_name>
```

Or stop all active sessions:
```bash
gswarm profiler stop
```

### 4. Check Status

Get profiler status:
```bash
gswarm profiler status
```

Read cluster metrics:
```bash
gswarm profiler read --output cluster_metrics.json
```

### 5. Retrieve and Analyze Results

Profiling data is saved in the gswarm working directory as `<task_name>.json`.

Analyze the collected data:
```bash
gswarm profiler analyze <task_name>.json --plot analysis.pdf
```

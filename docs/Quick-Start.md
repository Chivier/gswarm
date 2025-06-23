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

Send POST request to host HTTP API:

```bash
curl -X POST http://localhost:8091/profiling/start \
    -H "Content-Type: application/json" \
    -d '{"name": "<task_name>", "report_metrics": ["gpu_utilization", "gpu_memory", "gpu_dram_bandwidth", "gpu_bubble"]}'
```

Parameters:
- `name`: Name of your profiling task
- `report_metrics`: Metrics to be collected (reference: `src/gswarm/profiler/utils.py`)

Alternatively, start profiling via CLI:

```bash
gswarm profiler start --name <task_name> --report-metrics <metrics>
```

### 3. Stop Profiling

Via HTTP API:
```bash
curl -X POST http://localhost:8091/profiling/stop \
    -H "Content-Type: application/json" \
    -d '{"name": "<task_name>"}'
```

Or via CLI:
```bash
gswarm profiler stop --name <task_name>
```

### 4. Retrieve Results

Profiling data is saved in the gswarm working directory as `<task_name>.json`.
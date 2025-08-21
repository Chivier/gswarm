# gRPC-based Profiling Design

## Overview

The gswarm profiler uses gRPC for communication and supports:
- Multiple concurrent profiling sessions with unique names
- Real-time GPU metrics collection (utilization, memory, bandwidth)
- Session management and recovery
- Advanced metrics like DRAM bandwidth and NVLink monitoring

## API Design

### Session Management

Each profiling session is identified by a unique name and maintains its own:
- Output file (`<session_name>.json`)
- Frame counter
- Start/end timestamps
- Accumulated statistics

### Command Line Interface

The profiler is accessed through the `gswarm profiler` subcommand:

```bash
# Start first session with auto-discovery
gswarm profiler start --name training_epoch_1

# Start overlapping session with specific host
gswarm profiler start --name memory_intensive_phase --host localhost:50051

# Start session with custom metrics
gswarm profiler start --name optimization_step --report-metrics gpu_utilization --report-metrics gpu_memory

# Stop specific session
gswarm profiler stop --name training_epoch_1

# Stop all active sessions
gswarm profiler stop

# Get profiler status
gswarm profiler status

# Read cluster metrics
gswarm profiler read --output cluster_metrics.json
```

## Use Cases

### 1. Overlapping Performance Analysis

Monitor different phases of your workload:

```bash
# Start monitoring entire training
gswarm profiler start --name full_training

# Later, start monitoring specific epoch
gswarm profiler start --name epoch_5

# Stop epoch monitoring but continue full training
gswarm profiler stop --name epoch_5

# Eventually stop full training
gswarm profiler stop --name full_training
```

### 2. A/B Performance Testing

Compare different configurations:

```bash
# Start baseline profiling
gswarm profiler start --name baseline_config

# Run baseline workload...

# Start optimized profiling (can overlap)
gswarm profiler start --name optimized_config

# Run optimized workload...

# Stop both
gswarm profiler stop
```

### 3. Debugging Performance Issues

Focus on specific problematic regions:

```bash
# Normal profiling
gswarm profiler start --name normal_operation

# When issue detected, start detailed profiling with additional metrics
gswarm profiler start --name performance_issue_debug --report-metrics gpu_bubble --report-metrics gpu_dram_bandwidth

# Stop debug session after issue
gswarm profiler stop --name performance_issue_debug

# Analyze the collected data
gswarm profiler analyze performance_issue_debug.json --plot debug_analysis.pdf
```

## gRPC API and CLI Commands

### Available Commands

#### Profiling Control
- `gswarm profiler start` - Start a profiling session
  - `--name` - Session name (auto-generated if not provided)
  - `--host` - gRPC host address (auto-discovered if not specified)
  - `--freq` - Override sampling frequency
  - `--report-metrics` - Specific metrics to collect
- `gswarm profiler stop` - Stop profiling session(s)
  - `--name` - Specific session to stop (all if not specified)
- `gswarm profiler status` - Get profiler status
- `gswarm profiler sessions` - List all profiling sessions

#### Cluster Monitoring
- `gswarm profiler read` - Read cluster or node status with metrics
  - `--node` - Read specific node status
  - `--output` - Export to JSON file

#### Advanced Features
- `gswarm profiler analyze` - Analyze profiling data and generate plots
- `gswarm profiler recover` - Recover crashed profiling sessions
- `gswarm profiler enable-bandwidth` - Enable DRAM bandwidth profiling
- `gswarm profiler disable-bandwidth` - Disable DRAM bandwidth profiling
- `gswarm profiler enable-nvlink` - Enable NVLink profiling
- `gswarm profiler disable-nvlink` - Disable NVLink profiling

### Available Metrics

The profiler can report the following metrics:
- `gpu_utilization` - GPU utilization percentage
- `gpu_memory` - GPU memory usage
- `gpu_dram_bandwidth` - DRAM bandwidth utilization
- `gpu_bubble` - GPU bubble metrics for performance analysis

## Implementation Details

### Session Isolation

- Each session maintains independent data collection
- Sessions share the same underlying metrics stream
- No interference between concurrent sessions

### Resource Efficiency

- Single metrics collection stream serves all sessions
- Minimal overhead for additional sessions
- Automatic cleanup of completed sessions

### Data Output

Each session produces a separate JSON file with:
- Session metadata (name, start/end time)
- Collected frames
- Summary statistics

## Best Practices

1. **Use Descriptive Names**: Choose session names that clearly indicate what's being profiled
2. **Manage Overlaps**: Be aware of overlapping sessions to avoid confusion
3. **Clean Up**: Stop sessions when done to free resources
4. **Organize Output**: Consider using directories for output files when running many sessions

## Example Workflows

### Command Line Workflow

```bash
# Start main profiling session
gswarm profiler start --name main_workflow

# Check status
gswarm profiler status

# Start detailed profiling for specific phase
gswarm profiler start --name data_loading_phase --report-metrics gpu_memory --report-metrics gpu_utilization

# Monitor cluster status in real-time
gswarm profiler read --output cluster_status.json

# Stop data loading profiling
gswarm profiler stop --name data_loading_phase

# Start training phase profiling with bandwidth monitoring
gswarm profiler enable-bandwidth
gswarm profiler start --name training_phase

# Stop all profiling
gswarm profiler stop

# Analyze results
gswarm profiler analyze main_workflow.json --plot main_analysis.pdf
gswarm profiler analyze training_phase.json --plot training_analysis.pdf
```

### Python Integration Examples

#### Starting and Stopping Profiling

```python
import asyncio
import grpc
from gswarm.profiler import profiler_pb2, profiler_pb2_grpc

async def profile_workflow():
    # Connect to profiler service
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
    
    # Start main profiling
    start_response = await stub.StartProfiling(
        profiler_pb2.StartProfilingRequest(
            name="main_workflow",
            report_metrics=["gpu_utilization", "gpu_memory"]
        )
    )
    
    if start_response.success:
        print(f"Profiling started: {start_response.message}")
        print(f"Output file: {start_response.output_file}")
    
    # Run your workload...
    await asyncio.sleep(60)
    
    # Stop profiling
    stop_response = await stub.StopProfiling(
        profiler_pb2.StopProfilingRequest(name="main_workflow")
    )
    
    if stop_response.success:
        print(f"Profiling stopped: {stop_response.message}")
    
    await channel.close()

# Run the profiling
asyncio.run(profile_workflow())
```

#### Fetching Profiler Status

```python
import asyncio
import grpc
from gswarm.profiler import profiler_pb2, profiler_pb2_grpc

async def get_profiler_status():
    # Connect to profiler service
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
    
    # Get status
    status_response = await stub.GetStatus(profiler_pb2.Empty())
    
    print("Profiler Status:")
    print(f"  Frequency: {status_response.freq}ms")
    print(f"  Bandwidth Profiling: {'Enabled' if status_response.enable_bandwidth_profiling else 'Disabled'}")
    print(f"  NVLink Profiling: {'Enabled' if status_response.enable_nvlink_profiling else 'Disabled'}")
    print(f"  Is Profiling: {'Yes' if status_response.is_profiling else 'No'}")
    if status_response.output_filename:
        print(f"  Current Session: {status_response.output_filename}")
    print(f"  Connected Clients: {len(status_response.connected_clients)}")
    
    for client in status_response.connected_clients:
        print(f"    - {client}")
    
    await channel.close()

# Get status
asyncio.run(get_profiler_status())
```

#### Reading Cluster Metrics

```python
import asyncio
import grpc
import json
from gswarm.profiler import profiler_pb2, profiler_pb2_grpc

async def read_cluster_metrics():
    # Connect to profiler service
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
    
    # Read cluster status
    response = await stub.ReadClusterStatus(
        profiler_pb2.ReadClusterStatusRequest()
    )
    
    if response.success:
        metrics = {
            "cluster_id": response.cluster_id,
            "nodes": []
        }
        
        for node in response.nodes:
            node_data = {
                "node_id": node.node_id,
                "gpus": []
            }
            
            for gpu in node.gpus:
                gpu_data = {
                    "gpu_id": gpu.gpu_id,
                    "device_type": gpu.device_type,
                    "utilization": gpu.utilization,
                    "memory_used": gpu.memory_used,
                    "memory_total": gpu.memory_total,
                    "dram_bandwidth": gpu.dram_bandwidth,
                    "nvlink_bandwidth": gpu.nvlink_bandwidth
                }
                node_data["gpus"].append(gpu_data)
                
                print(f"Node {node.node_id} - GPU {gpu.gpu_id}:")
                print(f"  Device: {gpu.device_type}")
                print(f"  Utilization: {gpu.utilization:.1f}%")
                print(f"  Memory: {gpu.memory_used:.0f}/{gpu.memory_total:.0f} MB")
                print(f"  DRAM Bandwidth: {gpu.dram_bandwidth:.1f}%")
            
            metrics["nodes"].append(node_data)
        
        # Save to JSON
        with open("cluster_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        print(f"Failed to read cluster status: {response.message}")
    
    await channel.close()

# Read metrics
asyncio.run(read_cluster_metrics())
```

This will generate output files:
- `main_workflow.json`: Complete profiling data
- `data_loading_phase.json`: Focused on data loading with memory metrics
- `training_phase.json`: Training phase with bandwidth monitoring
- `cluster_metrics.json`: Real-time cluster GPU metrics

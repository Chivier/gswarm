# gswarm-profiler

A multi-node multi-GPU profiling tool for monitoring and analyzing GPU performance across distributed systems.

## Introduction

gswarm-profiler is a distributed GPU monitoring and profiling tool designed to collect performance metrics from multiple nodes in a GPU cluster. It uses a head-client architecture where a central head node coordinates data collection from multiple client nodes, enabling unified profiling of GPU utilization, memory usage, and bandwidth metrics across your entire GPU infrastructure.

Key features:
- Monitor GPU utilization and memory usage across multiple machines
- Track PCIe bandwidth (GPU-DRAM) and NVLink (GPU-GPU) connections
- Configurable sampling frequency
- JSON output format for easy integration with analysis tools
- Simple command-line interface
- Built on nvitop for accurate GPU metrics
- HTTP API for web-based control panels
- Support for multiple concurrent profiling sessions
- Fault tolerance with automatic reconnection and data persistence
- Session recovery after crashes
- Client-side buffering for network interruptions

Ideal for debugging performance issues in distributed deep learning workloads, cluster monitoring, and resource optimization in multi-node GPU environments.

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPUs with installed drivers

### Installing gswarm-profiler

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gswarm-profiler.git
cd gswarm-profiler
```

2. Install the package:

```bash
pip install .
```

## Usage

### Starting the Head Node

The head node acts as a central server that collects data from all client nodes. Start it on your primary machine:

```bash
# Start with both gRPC (port 8090) and HTTP API (port 8091)
gsprof start --port 8090 --http-port 8091 --enable-bandwidth --freq 1000

# Start with gRPC only (no HTTP API)
gsprof start --port 8090 --http-port 0 --enable-bandwidth --freq 1000

# Start with custom HTTP port
gsprof start --port 8090 --http-port 8080 --enable-bandwidth --freq 1000
```

Parameters:
- `--port`: gRPC port to listen on (default: 8090)
- `--http-port`: HTTP API port (default: 8091, set to 0 to disable)
- `--enable-bandwidth`: Enable GPU-DRAM bandwidth profiling
- `--enable-nvlink`: Enable NVLink bandwidth profiling
- `--freq`: Sampling frequency in milliseconds (default: 500)

### Why use HTTP API?

The HTTP API is useful for:
- Monitoring and controlling profiling sessions from a web interface
- Integrating with existing monitoring systems
- Running profiling sessions from a remote machine

GRPC is used for high-performance metric streaming, while HTTP API is used for control panel integration and session management.

### Connecting Client Nodes

On each machine with GPUs that you want to monitor, run:

```bash
# Standard client (backward compatible)
gsprof connect <head-node-ip>:8090

# Resilient client with automatic reconnection and buffering
gsprof connect <head-node-ip>:8090 --resilient
```

The `--resilient` flag enables:
- Automatic reconnection with exponential backoff
- Local metric buffering during network interruptions
- Data persistence on client shutdown
- Seamless recovery from network failures

Replace `<head-node-ip>` with the IP address or hostname of your head node.

### Controlling Profiling

#### Using CLI Commands (via gRPC)

```bash
# Start profiling
gsprof profile <head-node-ip>:8090 --name my_experiment

# Check status
gsprof status <head-node-ip>:8090

# Stop profiling (all sessions)
gsprof stop <head-node-ip>:8090

# Stop specific session
gsprof stop <head-node-ip>:8090 --name my_experiment
```

#### Using HTTP API Commands

```bash
# Start profiling
gsprof http-profile <head-node-ip>:8091 --name my_experiment

# Check status
gsprof http-status <head-node-ip>:8091

# Stop profiling (all sessions)
gsprof http-stop <head-node-ip>:8091

# Stop specific session
gsprof http-stop <head-node-ip>:8091 --name my_experiment
```

#### Using Direct HTTP Requests

```bash
# Get status
curl http://localhost:8091/status

# Start profiling
curl -X POST http://localhost:8091/profiling/start \
  -H "Content-Type: application/json" \
  -d '{"name": "my_experiment"}'

# Stop specific profiling session
curl -X POST http://localhost:8091/profiling/stop \
  -H "Content-Type: application/json" \
  -d '{"name": "my_experiment"}'

# Stop all profiling sessions
curl -X POST http://localhost:8091/profiling/stop \
  -H "Content-Type: application/json" \
  -d '{}'

# Get connected clients info
curl http://localhost:8091/clients

# Get latest metrics
curl http://localhost:8091/metrics/latest

# Get all profiling sessions
curl http://localhost:8091/profiling/sessions
```

### HTTP API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/status` | GET | Get profiler status with active sessions |
| `/profiling/start` | POST | Start profiling session |
| `/profiling/stop` | POST | Stop profiling session(s) |
| `/profiling/sessions` | GET | List all profiling sessions |
| `/clients` | GET | Get connected clients and GPU info |
| `/metrics/latest` | GET | Get latest metrics from all clients |

### Multiple Concurrent Sessions

gswarm-profiler supports running multiple profiling sessions simultaneously:

```bash
# Start multiple overlapping sessions
gsprof http-profile localhost:8091 --name training_epoch_1
gsprof http-profile localhost:8091 --name memory_analysis
gsprof http-profile localhost:8091 --name gpu_utilization_test

# Stop specific session
gsprof http-stop localhost:8091 --name training_epoch_1

# Check all sessions
curl http://localhost:8091/profiling/sessions

# Stop all remaining sessions
gsprof http-stop localhost:8091
```

### Fault Tolerance and Recovery

#### Session Recovery

gswarm-profiler automatically persists profiling data to disk, enabling recovery after crashes:

```bash
# List recoverable sessions
gsprof recover --list

# Resume a specific session
gsprof recover --session-id <session-id>

# Resume session by name
gsprof recover --name my_experiment

# Export recovered session data
gsprof recover --export --name my_experiment
```

#### Data Persistence

All profiling data is immediately written to disk in `.gswarm_profiler_data/` directory:
- Frames are stored in append-only JSONL format
- Session state is periodically checkpointed
- Automatic recovery on head node restart

#### Client Resilience

The resilient client provides:
- **Automatic Reconnection**: Exponential backoff from 1s to 60s
- **Data Buffering**: Up to 10,000 frames buffered locally
- **Zero Data Loss**: Buffered data sent after reconnection
- **Graceful Degradation**: Continues collecting even when disconnected

Example resilient client behavior:
```bash
# Client automatically handles network interruptions
gsprof connect master-node:8090 --resilient

# On shutdown, buffered data is saved to:
# buffered_metrics_<hostname>_<timestamp>.pkl
```

#### Health Monitoring

The head node automatically monitors client health:
- Clients marked as unhealthy after 30s of no data
- Profiling continues with available clients
- Data marked with client availability information

### Output Format

The output JSON file contains profiling data frames, each with:

- `session_name`: Name of the profiling session
- `start_time`: Session start timestamp
- `end_time`: Session end timestamp
- `frames`: Array of profiling frames containing:
  - `frame_id`: Sequential identifier for the data frame
  - `time`: Timestamp of data collection
  - `gpu_id`: Array of unique GPU identifiers (format: `hostname:device_idx:device_name`)
  - `gpu_util`: Array of GPU utilization percentages (0-100)
  - `gpu_memory`: Array of GPU memory utilization percentages (0-100)
  - `dram_bandwidth`: Array of DRAM bandwidth measurements in GB/s (if bandwidth profiling enabled)
  - `dram_bandwidth_rx`: Array of DRAM RX bandwidth in GB/s
  - `dram_bandwidth_tx`: Array of DRAM TX bandwidth in GB/s
  - `gpu_bandwidth`: Array of GPU-to-GPU bandwidth measurements (if NVLink profiling enabled)
- `summary_by_device`: Aggregated statistics per GPU device

## Example Workflows

### Basic Profiling Workflow

1. Start the head node on your primary machine:
```bash
gsprof start --port 8090 --http-port 8091 --enable-bandwidth --freq 1000
```

2. Connect client nodes from each machine in your cluster:
```bash
# On machine 1 (with resilient client)
gsprof connect master-node:8090 --resilient

# On machine 2 (with resilient client)
gsprof connect master-node:8090 --resilient

# ... and so on
```

3. Start profiling before running your workload:
```bash
gsprof http-profile master-node:8091 --name training_run_1
```

4. Run your distributed GPU workload

5. Stop profiling after your workload completes:
```bash
gsprof http-stop master-node:8091 --name training_run_1
```

6. Analyze the resulting JSON file (`training_run_1.json`)

### Fault-Tolerant Long-Running Profiling

For long-running experiments where reliability is critical:

```bash
# Start head node
gsprof start --port 8090 --http-port 8091 --enable-bandwidth

# Connect resilient clients
# On each client machine:
gsprof connect master-node:8090 --resilient

# Start profiling session
gsprof http-profile master-node:8091 --name week_long_training

# If head node crashes:
# 1. Restart head node
gsprof start --port 8090 --http-port 8091 --enable-bandwidth

# 2. Clients will automatically reconnect (resilient mode)

# 3. List and resume the session
gsprof recover --list
gsprof recover --name week_long_training

# Continue profiling...
```

### Web Dashboard Integration

The HTTP API makes it easy to integrate with web dashboards:

```javascript
// Check profiler status
fetch('http://profiler-host:8091/status')
  .then(res => res.json())
  .then(data => console.log(data));

// Start profiling
fetch('http://profiler-host:8091/profiling/start', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({name: 'experiment_1'})
})
  .then(res => res.json())
  .then(data => console.log(data));

// Get all sessions (including recovered ones)
fetch('http://profiler-host:8091/profiling/sessions')
  .then(res => res.json())
  .then(data => console.log(data));

// Stop specific session
fetch('http://profiler-host:8091/profiling/stop', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({name: 'experiment_1'})
})
  .then(res => res.json())
  .then(data => console.log(data));
```

### Multiple Session Example

Monitor different aspects of your workload simultaneously:

```bash
# Start overall monitoring
gsprof http-profile master-node:8091 --name full_training

# Start monitoring specific phase
gsprof http-profile master-node:8091 --name data_loading_phase

# ... after data loading completes
gsprof http-stop master-node:8091 --name data_loading_phase

# Start monitoring training phase
gsprof http-profile master-node:8091 --name training_phase

# ... after training
gsprof http-stop master-node:8091 --name training_phase

# Stop overall monitoring
gsprof http-stop master-node:8091 --name full_training
```

This generates separate files for each phase:
- `full_training.json`: Complete profiling data
- `data_loading_phase.json`: Data loading metrics only
- `training_phase.json`: Training metrics only

### Analyzing Results

```bash
# View profiling statistics
gsprof stat --data training_run_1.json --plot training_run_1.pdf

# Analyze recovered session
gsprof recover --export --name my_experiment
gsprof stat --data my_experiment.json --plot my_experiment.pdf
```

TODO: add a figure demo here

## Architecture

gswarm-profiler uses a distributed architecture with fault tolerance:
- **Head Node**: Central server that coordinates profiling and aggregates data
  - Persistent storage for immediate data durability
  - Session management for concurrent profiling
  - Automatic client health monitoring
- **Client Nodes**: Run on each GPU machine, collect and stream metrics
  - Resilient mode with automatic reconnection
  - Local buffering during network interruptions
  - Graceful handling of head node failures
- **Communication**: 
  - gRPC for client-head communication (high-performance metric streaming)
  - HTTP REST API for control panel integration and session management
- **Storage**:
  - File-based persistence in `.gswarm_profiler_data/`
  - Append-only frame storage for crash safety
  - Periodic session state checkpointing

## Documentation

For detailed documentation:
- [gRPC Protocol Design](docs/grpc-protocol-design.md)
- [Multiple Sessions Guide](docs/multiple-sessions.md)
- [Fault Tolerance Architecture](docs/fault-tolerance.md)

## Troubleshooting

### Client Connection Issues
- Ensure head node is accessible from client machines
- Check firewall rules for gRPC port (default: 8090)
- Use `--resilient` flag for automatic reconnection

### Data Recovery
- Check `.gswarm_profiler_data/` directory for persisted data
- Use `gsprof recover --list` to see available sessions
- Export recovered data with `gsprof recover --export`

### Performance Considerations
- Resilient clients buffer up to 10,000 frames in memory
- Adjust `--freq` parameter based on monitoring needs
- Monitor disk space for long-running sessions

## Some other notes

Why don't we use `wandb`?

Hmm, after I finished the first version of gswarm-profiler, I realized I could implement an integration with `wandb` in a few hours. But anyway, I don't want to use `wandb` for the stupid logging system. to use `wandb` for the stupid logging system.
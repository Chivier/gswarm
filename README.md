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

Ideal for debugging performance issues in distributed deep learning workloads, cluster monitoring, and resource optimization in multi-node GPU environments.

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPUs with installed drivers
- uv package manager

### Installing uv

If you don't have uv installed, you can install it by following the instructions at [uv's official documentation](https://github.com/astral-sh/uv), or simply run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing gswarm-profiler

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gswarm-profiler.git
cd gswarm-profiler
```

2. Install dependencies using uv:
```bash
uv sync
```

## Usage

### Starting the Head Node

The head node acts as a central server that collects data from all client nodes. Start it on your primary machine:

```bash
uv run gswarm-profiler start --port 8090 --enable-bandwidth --freq 1000
```

Parameters:
- `--port`: HTTP port to listen on (default: 8090)
- `--enable-bandwidth`: Enable GPU-DRAM and GPU-GPU bandwidth profiling
- `--freq`: Sampling frequency in milliseconds (default: 500)

### Connecting Client Nodes

On each machine with GPUs that you want to monitor, run:

```bash
uv run gswarm-profiler connect <head-node-ip>:8090
```

Replace `<head-node-ip>` with the IP address or hostname of your head node.

### Controlling Profiling

To start collecting profiling data:

```bash
curl -X POST http://localhost:8090/start
```

To stop profiling and save the collected data:

```bash
curl -X POST http://localhost:8090/stop
```

The profiling results will be saved in JSON format in the current directory.

### Output Format

The output JSON file contains profiling data frames, each with:

- `frame_id`: Sequential identifier for the data frame
- `time`: Timestamp of data collection
- `gpu_id`: Array of unique GPU identifiers
- `gpu_util`: Array of GPU utilization percentages (0-1)
- `gpu_memory`: Array of GPU memory utilization percentages (0-1)
- `dram_bandwidth`: Array of DRAM bandwidth measurements in GB/s (if bandwidth profiling enabled)
- `gpu_bandwidth`: Array of GPU-to-GPU bandwidth measurements (if bandwidth profiling enabled)

## Example Workflow

1. Start the head node on your primary machine:
```bash
uv run gswarm-profiler start --port 8090 --enable-bandwidth --freq 1000
```

2. Connect client nodes from each machine in your cluster:
```bash
# On machine 1
uv run gswarm-profiler connect master-node:8090

# On machine 2
uv run gswarm-profiler connect master-node:8090

# ... and so on
```

3. Start profiling before running your workload:
```bash
curl -X POST http://master-node:8090/start
```

If you want to save the profiling data to a specific file, you can use the following command:

```bash
curl -X POST http://master-node:8090/start/my_profiling_data
```
Then the profiling data will be saved to `my_profiling_data.json` in the current directory.

4. Run your distributed GPU workload

5. Stop profiling after your workload completes:
```bash
curl -X POST http://master-node:8090/stop
```

6. Analyze the resulting JSON file using your preferred data analysis tools

## License

MIT License

Copyright (c) 2023 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

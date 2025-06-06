# gswarm

A comprehensive distributed GPU cluster management system combining profiling, model storage, and orchestration capabilities.

## Overview

gswarm is an integrated platform for managing GPU clusters, providing:

- **GPU Profiling**: Multi-node GPU monitoring and performance analysis
- **Model Management**: Distributed model storage, deployment, and serving
- **Data Pooling**: Efficient data management across nodes
- **Task Orchestration**: Queue-based asynchronous task execution

The system uses a host-client architecture where a central host node coordinates operations across multiple client nodes, enabling unified management of your entire GPU infrastructure.

## Key Features

### Profiling Capabilities
- Monitor GPU utilization and memory usage across multiple machines
- Track PCIe bandwidth (GPU-DRAM) and NVLink (GPU-GPU) connections
- Configurable sampling frequency with JSON output
- Built on nvitop for accurate GPU metrics
- Fault tolerance with automatic reconnection
- Session recovery after crashes

### Model Management
- Distributed model storage across disk, DRAM, and GPU memory
- Automatic model deployment and serving
- Cross-node model transfer and replication
- Support for multiple model frameworks (vLLM, Transformers, TGI)
- Real-time model status tracking

### Data Pool System
- Distributed data chunk management
- Automatic data migration between devices
- Reference counting and garbage collection
- Transparent cross-node data access
- Support for model inputs/outputs chaining

### Task Queue System
- Asynchronous task execution with priorities
- Dependency management and resource conflict detection
- Parallel execution of independent tasks
- Automatic retry with exponential backoff

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPUs with installed drivers
- Network connectivity between cluster nodes

### Installing gswarm

```bash
# Clone the repository
git clone https://github.com/yourusername/gswarm.git
cd gswarm

# Install the package
pip install .
```

## Quick Start

### 1. Start the Host Node

```bash
# Start host with both profiling and model management
gswarm host start --port 8090 --http-port 8091 --model-port 9010
```

### 2. Connect Client Nodes

On each GPU machine:

```bash
# Connect client with resilient mode
gswarm client connect <host-ip>:8090 --resilient
```

### 3. Profile GPU Usage

```bash
# Start profiling
gswarm profiler start --name training_run

# Check status
gswarm profiler status

# Stop profiling
gswarm profiler stop --name training_run
```

### 4. Manage Models

```bash
# List available models
gswarm model list

# Download a model (on host node)
gswarm model download llama-7b --source huggingface --url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct --node node1 --type llm
# or use hf:// format
gswarm model download llama-7b --source hf://meta-llama/Llama-3.1-8B-Instruct --node node1 --type llm

# Download a model (on client node, if node-id is not specified, it will download local)
gswarm model download llama-7b --source huggingface --url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct --type llm
# or use hf:// format
gswarm model download llama-7b --source hf://meta-llama/Llama-3.1-8B-Instruct --type llm

# Deploy model to GPU (on client node)
gswarm model move llama-7b --from disk --to gpu0 
# if i am on host, i must specify the node id
gswarm model move llama-7b --from disk --to gpu0 --node node1

# Start model serving (on client node)
# each model requires different method to implement serving, this is provideed in model/instance/xxx.py
# xxx is the model type, we use type to support different model inference methods
gswarm model serve llama-7b --device gpu0 --port 8080
# if i am on host, i must specify the node id
gswarm model serve llama-7b --device gpu0 --port 8080 --node node1

# Check model status
gswarm model status llama-7b
gswarm model status llama-7b --node node1
```

### 5. Manage Data

```bash
# Create data chunk
gswarm data create --source s3://bucket/data --device dram

# List data chunks
gswarm data list

# Transfer data to another node
gswarm data transfer chunk-123 --to node2:dram
```

## Architecture

### System Components

1. **Host Node**: Central coordinator
   - Model registry management
   - Task orchestration
   - Global resource tracking
   - API gateway

2. **Client Nodes**: Worker nodes
   - Local model storage
   - Model serving
   - GPU profiling
   - Task execution
   - Data pool management

3. **Communication**:
   - gRPC for high-performance metric streaming
   - HTTP REST API for control and management
   - WebSocket for real-time updates

### Port Configuration

Default ports used by gswarm:
- **gRPC Server**: 8090 (profiling metrics)
- **HTTP API**: 8091 (control panel)
- **Model API**: 9010 (model management)
- **Model Services**: 8080+ (dynamic allocation)

## CLI Reference

### Host Commands

```bash
# Host management
gswarm host start [--port PORT] [--http-port HTTP_PORT]
gswarm host stop
gswarm host status

# System overview
gswarm status              # Overall system status
gswarm nodes               # List all nodes
gswarm health              # Health check
```

### Profiler Commands

```bash
# Profiling operations
gswarm profiler start [--name NAME] [--freq FREQ]
gswarm profiler stop [--name NAME]
gswarm profiler status
gswarm profiler sessions   # List all sessions
gswarm profiler recover    # Recover crashed sessions

# Analysis
gswarm profiler analyze --data <file.json> --plot <output.pdf>
```

### Model Commands

```bash
# Model management
gswarm model list [--location LOCATION]
gswarm model info <model_name>
gswarm model register <model_name> --type TYPE --source URL

# Model operations
gswarm model download <model_name> [--device DEVICE]
gswarm model move <model_name> --from SOURCE --to DEST [--keep-source]
gswarm model copy <model_name> --from SOURCE --to DEST
gswarm model delete <model_name> --device DEVICE

# Model serving
gswarm model serve <model_name> --device DEVICE [--port PORT]
gswarm model stop <model_name>
gswarm model services      # List all running services
```

### Data Commands

```bash
# Data pool management
gswarm data list [--device DEVICE]
gswarm data create --source SOURCE --device DEVICE
gswarm data info <chunk_id>
gswarm data move <chunk_id> --to DEVICE
gswarm data transfer <chunk_id> --to NODE:DEVICE
gswarm data delete <chunk_id>
```

### Queue Commands

```bash
# Task queue management
gswarm queue status
gswarm queue tasks [--status STATUS]
gswarm queue cancel <task_id>
gswarm queue history [--limit N]
```

## API Reference

### Model Management APIs

```bash
# List models
GET /api/v1/models

# Get model info
GET /api/v1/models/{model_name}

# Register model
POST /api/v1/models

# Download model
POST /api/v1/models/{model_name}/download

# Move model
POST /api/v1/models/{model_name}/move

# Start serving
POST /api/v1/services

# Get service status
GET /api/v1/services/{service_id}/status
```

### Data Pool APIs

```bash
# List data chunks
GET /api/v1/data

# Create data chunk
POST /api/v1/data

# Get chunk info
GET /api/v1/data/{chunk_id}

# Move data
POST /api/v1/data/{chunk_id}/move

# Transfer data
POST /api/v1/data/{chunk_id}/transfer
```

### Queue APIs

```bash
# Get queue status
GET /api/v1/queue

# Get task details
GET /api/v1/queue/tasks/{task_id}

# Cancel task
POST /api/v1/queue/tasks/{task_id}/cancel

# Get history
GET /api/v1/queue/history
```

## Configuration

### Config File Location

`~/.gswarm/config.yaml`

### Example Configuration

```yaml
cluster:
  host: "master.cluster.local"
  port: 8090
  
profiling:
  default_frequency: 1000
  enable_bandwidth: true
  enable_nvlink: false
  
models:
  storage_path: "/data/models"
  cache_size: "100GB"
  
queue:
  max_concurrent_tasks: 4
  task_timeout: 3600
  retry_count: 3
  
nodes:
  - name: "node1"
    address: "192.168.1.101"
    capabilities:
      gpus: ["gpu0", "gpu1"]
      storage:
        disk: 1000000000000
        dram: 64000000000
        
  - name: "node2"
    address: "192.168.1.102"
    capabilities:
      gpus: ["gpu0"]
      storage:
        disk: 500000000000
        dram: 32000000000
```

## Example Workflows

### Distributed Model Deployment

```yaml
name: "distributed-deployment"
description: "Deploy model across multiple nodes"

actions:
  # Download model to primary node
  - action_id: "download"
    action_type: "download"
    model_name: "llama-7b"
    target_device: "node1:disk"
    
  # Replicate to other nodes
  - action_id: "replicate_node2"
    action_type: "copy"
    model_name: "llama-7b"
    source_device: "node1:disk"
    target_device: "node2:disk"
    dependencies: ["download"]
    
  # Load models to GPUs
  - action_id: "load_gpu_node1"
    action_type: "move"
    model_name: "llama-7b"
    source_device: "node1:disk"
    target_device: "node1:gpu0"
    dependencies: ["download"]
    
  - action_id: "load_gpu_node2"
    action_type: "move"
    model_name: "llama-7b"
    source_device: "node2:disk"
    target_device: "node2:gpu0"
    dependencies: ["replicate_node2"]
    
  # Start services
  - action_id: "serve_node1"
    action_type: "serve"
    model_name: "llama-7b"
    device: "node1:gpu0"
    port: 8080
    dependencies: ["load_gpu_node1"]
    
  - action_id: "serve_node2"
    action_type: "serve"
    model_name: "llama-7b"
    device: "node2:gpu0"
    port: 8081
    dependencies: ["load_gpu_node2"]
```

### Data Pipeline with Model Chaining

```yaml
name: "ml-pipeline"
description: "Process data through multiple models"

actions:
  # Prepare input data
  - action_id: "load_data"
    action_type: "data_create"
    source: "s3://bucket/input"
    target_device: "node1:dram"
    
  # First model processing
  - action_id: "model1_process"
    action_type: "inference"
    model_name: "preprocessor"
    input_data: "${load_data.chunk_id}"
    output_device: "node1:dram"
    dependencies: ["load_data"]
    
  # Transfer intermediate data
  - action_id: "transfer_data"
    action_type: "data_transfer"
    data_id: "${model1_process.output}"
    target_device: "node2:dram"
    dependencies: ["model1_process"]
    
  # Second model processing
  - action_id: "model2_process"
    action_type: "inference"
    model_name: "classifier"
    input_data: "${transfer_data.chunk_id}"
    output_device: "node2:dram"
    dependencies: ["transfer_data"]
```

## Monitoring and Troubleshooting

### Health Checks

```bash
# System health
gswarm health

# Node-specific health
gswarm node status node1

# Service health
gswarm model service-health llama-7b
```

### Logs

Logs are stored in `~/.gswarm/logs/`:
- `host.log`: Host node logs
- `client-<node>.log`: Client node logs
- `profiler.log`: Profiling session logs
- `model.log`: Model operation logs

### Common Issues

1. **Connection Issues**
   - Check firewall rules for ports 8090-8091, 9010-9011
   - Verify network connectivity between nodes
   - Use `--resilient` flag for automatic reconnection

2. **Model Download Failures**
   - Check internet connectivity
   - Verify HuggingFace token if needed
   - Check disk space on target device

3. **GPU Memory Issues**
   - Monitor GPU memory with `gswarm profiler`
   - Use model quantization for large models
   - Distribute model across multiple GPUs

4. **Task Queue Blockage**
   - Check task dependencies with `gswarm queue tasks`
   - Look for resource conflicts
   - Cancel stuck tasks with `gswarm queue cancel`

## Migration from Legacy Components

If you're migrating from separate `gswarm-profiler` and `gswarm-model`:

1. **Backup existing data**:
   ```bash
   cp -r ~/.gswarm_profiler_data ~/.gswarm_profiler_data.backup
   cp -r ~/.gswarm_model_data ~/.gswarm_model_data.backup
   ```

2. **Update CLI commands**:
   - `gsprof` → `gswarm profiler`
   - `gsmodel` → `gswarm model`

3. **Update API endpoints**:
   - Model APIs now use `/api/v1/` prefix
   - Same ports are maintained for compatibility

See the [Migration Guide](docs/Migration-Guide.md) for detailed instructions.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_profiler.py
pytest tests/test_model.py
pytest tests/test_queue.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Documentation

- [Architecture Overview](docs/Architecture.md)
- [API Reference](docs/API-Reference.md)
- [Migration Guide](docs/Migration-Guide.md)
- [Model Storage Design](docs/Model-Storage-design.md)
- [Profiler Protocol Design](docs/grpc-protocol-design.md)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built on [nvitop](https://github.com/XuehaiPan/nvitop) for GPU monitoring
- Inspired by distributed computing frameworks
- Thanks to all contributors

## Roadmap

- [ ] Kubernetes operator for cluster deployment
- [ ] Web UI for cluster management
- [ ] Advanced scheduling algorithms
- [ ] Model optimization toolkit
- [ ] Integration with popular ML frameworks
- [ ] Multi-cloud support

For more information, see the [documentation](docs/).
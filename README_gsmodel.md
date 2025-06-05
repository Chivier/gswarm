# GSwarm Model Management System (FastAPI Edition)

A simplified distributed model storage and management system for GPU clusters, now using FastAPI instead of gRPC for better simplicity and easier debugging.

## 🚀 What's New in v0.3.0

- **No More gRPC**: Completely replaced gRPC with simple REST APIs using FastAPI
- **Simplified Architecture**: Removed protobuf dependencies and complex RPC calls
- **Better Debugging**: HTTP requests are much easier to debug than gRPC
- **Fewer Dependencies**: No need for grpcio, grpcio-tools, or protobuf compilation
- **Same Features**: All core functionality remains the same, just simpler

## Features

### 🚀 Core Capabilities
- **Distributed Model Registry**: Central coordination of model locations and availability
- **Multi-Storage Support**: Manage models across disk, RAM, and GPU memory
- **Job Workflows**: JSON/YAML-based pipeline definitions for complex model operations
- **REST API**: Simple HTTP API for all operations
- **Smart GPU Detection**: Automatic detection of available GPUs using nvitop/pynvml

### 📦 Model Operations
- **Download**: Fetch models from web sources (HuggingFace, etc.)
- **Move/Copy**: Transfer models between storage devices
- **Serve**: Start model inference services
- **Health Checks**: Monitor service availability
- **Location Tracking**: Track model storage across the cluster

## Architecture

The system follows a simple client-server architecture:

- **Head Node**: FastAPI server that manages the model registry and coordinates operations
- **Client Nodes**: Worker nodes that connect via REST API to store and serve models
- **Device Naming**: Standardized naming convention (`node:storage_type[:index]`)

### Device Types
- `disk`: Persistent storage (SSD/HDD)
- `dram`: System memory (RAM)
- `gpu0`, `gpu1`: GPU memory

## Installation

### Prerequisites
- Python 3.8+
- FastAPI and related dependencies
- NVIDIA drivers (for GPU support)

### Install Dependencies

```bash
# Install the project with all dependencies
pip install -e .

# Or install specific dependencies
pip install fastapi uvicorn pydantic pyyaml typer loguru aiofiles requests
```

## Quick Start

### 1. Start Head Node

```bash
# Start the head node server
gsmodel head --port 8100

# The head node provides a REST API at http://localhost:8100
```

### 2. Connect Client Nodes

```bash
# Connect a client node to the head node
gsmodel client http://localhost:8100 --node-id worker1

# On another machine
gsmodel client http://head_node_ip:8100 --node-id worker2
```

### 3. Register a Model

```bash
# Register a model in the system
gsmodel register llama-7b llm \
  --source-url https://huggingface.co/meta-llama/Llama-2-7b-hf
```

### 4. Manage Models

```bash
# List all models
gsmodel list

# Download a model
gsmodel download llama-7b https://huggingface.co/meta-llama/Llama-2-7b-hf node1:disk

# Serve a model
gsmodel serve llama-7b node1:gpu0 8080
```

## Command Line Interface

### Head Node Management

```bash
# Start head node
gsmodel head [--host HOST] [--port PORT]

# Default: http://0.0.0.0:8100
gsmodel head
```

### Client Node Management

```bash
# Connect client node
gsmodel client HEAD_URL [--node-id NODE_ID]

# Example
gsmodel client http://192.168.1.100:8100 --node-id gpu-worker-01
```

### Model Management

```bash
# Register model
gsmodel register MODEL_NAME MODEL_TYPE [--source-url URL] [--metadata-file FILE]

# List all models
gsmodel list [--head-url URL]

# Download model
gsmodel download MODEL_NAME SOURCE_URL DEVICE [--head-url URL]

# Serve model
gsmodel serve MODEL_NAME DEVICE PORT [--head-url URL]
```

### Job Management

```bash
# Create job from file
gsmodel job JOB_FILE [--head-url URL]

# Job file can be JSON or YAML
gsmodel job workflow.yaml
```

## REST API Reference

The head node provides a simple REST API:

### Model Management

```bash
# List models
GET /models

# Get model details
GET /models/{model_name}

# Register model
POST /models
{
    "name": "llama-7b",
    "type": "llm",
    "source_url": "https://...",
    "metadata": {}
}

# Delete model
DELETE /models/{model_name}
```

### Model Operations

```bash
# Download model
POST /download
{
    "model_name": "llama-7b",
    "source_url": "https://...",
    "target_device": "node1:disk"
}

# Move model
POST /move
{
    "model_name": "llama-7b",
    "source_device": "node1:disk",
    "target_device": "node1:gpu0",
    "keep_source": false
}

# Serve model
POST /serve
{
    "model_name": "llama-7b",
    "device": "node1:gpu0",
    "port": 8080,
    "config": {}
}

# Stop serving
POST /stop/{model_name}/{device}
```

### Node Management

```bash
# Register node
POST /nodes
{
    "node_id": "worker1",
    "hostname": "gpu-server-01",
    "ip_address": "192.168.1.101",
    "storage_devices": {},
    "gpu_count": 2
}

# List nodes
GET /nodes

# Node heartbeat
POST /nodes/{node_id}/heartbeat
```

### Job Management

```bash
# Create job
POST /jobs
{
    "name": "deployment",
    "description": "Deploy model",
    "actions": [...]
}

# Get job status
GET /jobs/{job_id}
```

### System Status

```bash
# Health check
GET /health

# Root endpoint
GET /
```

## Job Definition Examples

### Simple Deployment (JSON)

```json
{
    "name": "deploy-llama",
    "description": "Deploy Llama model",
    "actions": [
        {
            "action": "download",
            "model": "llama-7b",
            "source": "https://huggingface.co/meta-llama/Llama-2-7b-hf",
            "target": "node1:disk"
        },
        {
            "action": "move",
            "model": "llama-7b",
            "from": "node1:disk",
            "to": "node1:gpu0"
        },
        {
            "action": "serve",
            "model": "llama-7b",
            "device": "node1:gpu0",
            "port": 8080
        }
    ]
}
```

### Multi-Model Deployment (YAML)

```yaml
name: multi-model-deployment
description: Deploy multiple models

actions:
  - action: download
    model: llama-7b
    source: https://huggingface.co/meta-llama/Llama-2-7b-hf
    target: node1:disk
    
  - action: download
    model: stable-diffusion
    source: https://huggingface.co/stabilityai/stable-diffusion-2-1
    target: node2:disk
    
  - action: serve
    model: llama-7b
    device: node1:gpu0
    port: 8080
    
  - action: serve
    model: stable-diffusion
    device: node2:gpu0
    port: 8081
```

## Migration from gRPC Version

If you're migrating from the gRPC version:

1. **Remove gRPC files**: Delete `*.proto`, `*_pb2.py`, `*_pb2_grpc.py` files
2. **Update imports**: Change from gRPC clients to the new REST client
3. **Update CLI calls**: The CLI commands are similar but simplified
4. **Update API calls**: Replace gRPC calls with REST API calls

### Key Differences

| Feature | gRPC Version | FastAPI Version |
|---------|--------------|-----------------|
| Protocol | gRPC/Protobuf | REST/JSON |
| Dependencies | grpcio, protobuf | fastapi, requests |
| Debugging | Complex | Simple (HTTP) |
| Setup | Requires protobuf compilation | No compilation needed |
| Performance | Slightly faster | Fast enough |

## Development

### Project Structure

```
src/gswarm_model/
├── __init__.py           # Package initialization
├── __main__.py           # Module entry point
├── cli.py                # Command line interface
├── models.py             # Data models and schemas
├── head.py               # Head node implementation
├── client.py             # Client node implementation
├── http_api.py           # REST API implementation
├── model.proto           # gRPC protocol definition
├── generate_grpc.py      # gRPC code generation
├── model_pb2.py          # Generated protobuf (auto)
└── model_pb2_grpc.py     # Generated gRPC stubs (auto)
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Related Projects

- **gswarm-profiler**: Multi-node GPU profiling system
- **gswarm-scheduler**: Distributed job scheduling system
- **gswarm-storage**: Distributed storage management

---

For more information and examples, visit the [documentation](docs/) or check the [examples](examples/) directory. 
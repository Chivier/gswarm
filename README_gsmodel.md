# GSwarm Model Management System

A distributed model storage and management system for GPU clusters, designed to efficiently manage model distribution, storage, and serving across multiple nodes.

## Features

### 🚀 Core Capabilities
- **Distributed Model Registry**: Central coordination of model locations and availability
- **Multi-Storage Support**: Manage models across disk, RAM, and GPU memory
- **Job Workflows**: YAML-based pipeline definitions for complex model operations
- **REST API**: Complete HTTP API for integration and automation
- **gRPC Communication**: High-performance inter-node communication
- **Automatic Persistence**: State persistence and recovery

### 📦 Model Operations
- **Download**: Fetch models from web sources (HuggingFace, etc.)
- **Move/Copy**: Transfer models between storage devices
- **Serve**: Start model inference services
- **Health Checks**: Monitor service availability
- **Location Tracking**: Track model storage across the cluster

### 🎯 Workflow Management
- **YAML Pipelines**: Define complex multi-step operations
- **Dependency Management**: Automatic dependency resolution
- **Parallel Execution**: Concurrent action execution
- **Progress Tracking**: Real-time job status monitoring

## Architecture

The system follows a head-client architecture:

- **Head Node**: Central coordinator managing model registry and orchestrating operations
- **Client Nodes**: Worker nodes that store, serve, and execute models
- **Device Naming**: Standardized naming convention (`node:storage_type[:index]`)

### Device Types
- `web`: External web sources (HuggingFace Hub)
- `disk`: Persistent storage (SSD/HDD)
- `dram`: System memory (RAM)
- `gpu0`, `gpu1`: GPU memory

## Installation

### Prerequisites
- Python 3.8+
- gRPC tools for protocol buffer generation

### Install Dependencies

```bash
# Install the project with all dependencies
pip install -e .

# Or install specific dependencies
pip install grpcio grpcio-tools fastapi uvicorn pydantic pyyaml typer loguru aiofiles requests
```

### Generate gRPC Files

```bash
# Generate protocol buffer files
gsmodel generate-grpc
```

## Quick Start

### 1. Start Head Node

```bash
# Start the head node with both gRPC and HTTP APIs
gsmodel start --host 0.0.0.0 --port 8090 --http-port 8080

# Or run in background
gsmodel start --background --port 8090 --http-port 8080
```

The head node provides:
- **gRPC server**: `localhost:8090` (for client nodes)
- **HTTP API**: `http://localhost:8080` (for management)

### 2. Connect Client Nodes

```bash
# Connect a client node to the head node
gsmodel connect localhost:8090 --node-id worker1

# On another machine
gsmodel connect head_node_ip:8090 --node-id worker2
```

### 3. Register a Model

```bash
# Register a model in the system
gsmodel register llama-7b \
  --type llm \
  --url https://huggingface.co/meta-llama/Llama-2-7b-hf \
  --desc "Llama 2 7B language model"
```

### 4. Run a Workflow

```bash
# Generate example workflows
gsmodel example --output examples/

# Execute a workflow
gsmodel job examples/simple-deployment.yaml --wait
```

## Command Line Interface

### Head Node Management

```bash
# Start head node
gsmodel start [--host HOST] [--port PORT] [--http-port HTTP_PORT] [--background]

# Start with custom data directory
gsmodel start --data-dir /path/to/data
```

### Client Node Management

```bash
# Connect client node
gsmodel connect HEAD_ADDRESS [--node-id NODE_ID]

# Example
gsmodel connect 192.168.1.100:8090 --node-id gpu-worker-01
```

### Model Management

```bash
# Register model
gsmodel register MODEL_NAME [--type TYPE] [--url URL] [--desc DESCRIPTION]

# List all models
gsmodel list [--verbose]

# Get model details
gsmodel info MODEL_NAME

# Unregister model
gsmodel unregister MODEL_NAME [--force]
```

### Job Management

```bash
# Create job from YAML
gsmodel job workflow.yaml [--wait]

# List all jobs
gsmodel jobs [--status STATUS]

# Get job status
gsmodel job-status JOB_ID [--verbose]

# Generate example workflows
gsmodel example [--output DIR]
```

### System Monitoring

```bash
# System status
gsmodel status

# List connected nodes
gsmodel nodes [--verbose]
```

## Workflow Examples

### Simple Model Deployment

```yaml
name: "llama-deployment-pipeline"
description: "Download and serve Llama model"

actions:
  - action_id: "download_llama"
    action_type: "download"
    model_name: "llama-7b-chat"
    source_url: "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
    devices: ["node1:disk"]
    dependencies: []

  - action_id: "move_to_gpu"
    action_type: "move"
    model_name: "llama-7b-chat"
    devices: ["node1:disk", "node1:gpu0"]  # from, to
    keep_source: true
    dependencies: ["download_llama"]

  - action_id: "serve_model"
    action_type: "serve"
    model_name: "llama-7b-chat"
    port: 8080
    devices: ["node1:gpu0"]
    dependencies: ["move_to_gpu"]

  - action_id: "health_check"
    action_type: "health_check"
    target_url: "http://node1:8080/health"
    devices: []
    dependencies: ["serve_model"]
```

### Multi-Model Deployment

```yaml
name: "multi-model-inference"
description: "Deploy multiple models for inference"

actions:
  # Download models in parallel
  - action_id: "download_llama"
    action_type: "download"
    model_name: "llama-7b"
    source_url: "https://huggingface.co/meta-llama/Llama-2-7b-hf"
    devices: ["node1:disk"]
    dependencies: []

  - action_id: "download_diffusion"
    action_type: "download"
    model_name: "stable-diffusion-xl"
    source_url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
    devices: ["node2:disk"]
    dependencies: []

  # Load to different GPUs
  - action_id: "load_llama_gpu"
    action_type: "move"
    model_name: "llama-7b"
    devices: ["node1:disk", "node1:gpu0"]
    dependencies: ["download_llama"]

  - action_id: "load_diffusion_gpu"
    action_type: "move"
    model_name: "stable-diffusion-xl"
    devices: ["node2:disk", "node2:gpu0"]
    dependencies: ["download_diffusion"]

  # Serve both models
  - action_id: "serve_llama"
    action_type: "serve"
    model_name: "llama-7b"
    port: 8080
    devices: ["node1:gpu0"]
    dependencies: ["load_llama_gpu"]

  - action_id: "serve_diffusion"
    action_type: "serve"
    model_name: "stable-diffusion-xl"
    port: 8081
    devices: ["node2:gpu0"]
    dependencies: ["load_diffusion_gpu"]
```

## HTTP API

The head node provides a comprehensive REST API:

### Model Management

```bash
# List models
curl http://localhost:8080/models

# Get model details
curl http://localhost:8080/models/llama-7b

# Register model
curl -X POST http://localhost:8080/models/llama-7b/register \
  -H "Content-Type: application/json" \
  -d '{"model_type": "llm", "metadata": {"description": "Llama model"}}'

# Get model locations
curl http://localhost:8080/models/llama-7b/locations

# Get active services
curl http://localhost:8080/models/llama-7b/services
```

### Job Management

```bash
# Create job
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d @job_definition.json

# Upload YAML workflow
curl -X POST http://localhost:8080/jobs/from-yaml \
  -F "file=@workflow.yaml"

# Get job status
curl http://localhost:8080/jobs/JOB_ID/status

# List jobs
curl http://localhost:8080/jobs
```

### System Status

```bash
# System overview
curl http://localhost:8080/status

# Connected nodes
curl http://localhost:8080/nodes

# Health check
curl http://localhost:8080/health
```

## Configuration

### Environment Variables

```bash
# Head node configuration
export GSWARM_MODEL_HOST=0.0.0.0
export GSWARM_MODEL_PORT=8090
export GSWARM_MODEL_HTTP_PORT=8080
export GSWARM_MODEL_DATA_DIR=/data/gswarm_model

# Client node configuration
export GSWARM_MODEL_HEAD_ADDRESS=head_node:8090
export GSWARM_MODEL_NODE_ID=worker-gpu-01
```

### Data Directory Structure

```
.gswarm_model_data/
├── registry.json          # Persistent model and node registry
├── jobs/                  # Job execution logs
│   ├── job_123.json
│   └── job_456.json
└── logs/                  # System logs
    ├── head.log
    └── client.log
```

## Device Naming Convention

The system uses a standardized device naming format:

- **Format**: `<node_identifier>:<storage_type>[:<index>]`
- **Examples**:
  - `web` - External web source
  - `node1:disk` - Disk storage on node1
  - `node1:dram` - RAM on node1
  - `node1:gpu0` - GPU 0 on node1
  - `192.168.1.100:gpu1` - GPU 1 on specific IP

## Action Types

### Supported Actions

- **download**: Download model from web source
- **move**: Move model between devices (removes from source)
- **copy**: Copy model between devices (keeps source)
- **serve**: Start model inference service
- **stop_serve**: Stop model service
- **delete**: Remove model from storage
- **health_check**: Check service health

### Action Configuration

Each action supports various configuration options:

```yaml
- action_id: "custom_serve"
  action_type: "serve"
  model_name: "llama-7b"
  port: 8080
  devices: ["node1:gpu0"]
  config:
    max_batch_size: 32
    timeout: 30
    framework: "vllm"
  dependencies: ["load_model"]
```

## Monitoring and Troubleshooting

### Logs

```bash
# View head node logs
tail -f .gswarm_model_data/logs/head.log

# View client logs
tail -f /var/log/gswarm_model/client.log
```

### Common Issues

1. **gRPC Files Missing**:
   ```bash
   gsmodel generate-grpc
   ```

2. **Port Already in Use**:
   ```bash
   # Check what's using the port
   lsof -i :8090
   
   # Use different port
   gsmodel start --port 8091
   ```

3. **Client Connection Failed**:
   - Check head node is running
   - Verify network connectivity
   - Check firewall settings

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
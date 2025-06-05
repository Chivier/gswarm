# Model Storage and Management System Design

## Overview

This document outlines the design for a distributed model storage and management system integrated with the gswarm-profiler. The system enables efficient model distribution, storage management, and service orchestration across multiple nodes in a GPU cluster.

## Architecture

### System Components

1. **Host Node**: Central coordinator that maintains global model registry and orchestrates model operations
2. **Client Nodes**: Worker nodes that store, serve, and execute models
3. **Model Registry**: Distributed database tracking model locations and availability
4. **Service Manager**: Handles model serving and API endpoints

### Data Structures

#### Host Node Model Registry

```python
{
    "model_name": {
        "model_name": str,                    # Unique model identifier
        "model_type": str,                    # e.g., "llm", "diffusion", "vision"
        "model_size": int,                    # Size in bytes
        "model_hash": str,                    # Content hash for integrity
        "stored_locations": [str],            # List of device_name
        "available_services": {               # Active serving endpoints
            "device_name": "http://node:port",
            ...
        },
        "metadata": {                         # model metadata (optional)
            "description": str,
            "tags": [str],
            "created_at": str,
            "updated_at": str,
            "requirements": {                 # Resource requirements
                "min_memory": int,
                "min_vram": int,
                "gpu_arch": [str]
            }
        }
    }
}
```

#### Client Node Model Registry

```python
{
    "model_name": {
        "model_name": str,
        "stored_locations": [str],            # Local storage locations
        "status": str,                        # "available", "downloading", "moving", "serving"
        "service_port": int,                  # Port if serving (None if not)
        "last_accessed": str,                 # ISO timestamp
        "local_path": str,                    # Filesystem path
        "size": int,                          # Actual size on disk
        "integrity_hash": str                 # Verification hash
    }
}
```

### Device Naming Convention

Device names follow the pattern: `<node_identifier>:<storage_type>[:<index>]`

**Storage Types:**
- `web`: External web source (e.g., HuggingFace Hub)
- `disk`: Persistent storage (SSD/HDD)
- `dram`: System memory (RAM)
- `gpu<index>`: GPU memory (e.g., gpu0, gpu1)

**Examples:**
- `web`: HuggingFace or other web repositories
- `node1:disk`: Disk storage on node1
- `node1:dram`: RAM on node1
- `node1:gpu0`: GPU 0 memory on node1
- `192.168.1.100:gpu1`: GPU 1 on node with IP 192.168.1.100

## API Design

### Host Node APIs

#### Model Registry Management

**GET `/models`**
- Description: List all registered models
- Response: Array of model summaries
```json
{
    "models": [
        {
            "model_name": "llama-7b",
            "model_type": "llm",
            "size": 13968179200,
            "locations": ["node1:disk", "node2:gpu0"],
            "services": ["http://node1:8080", "http://node2:8081"],
            "status": "available"
        }
    ],
    "total_count": 1
}
```

**GET `/models/{model_name}`**
- Description: Get detailed model information
- Response: Complete model metadata and location information

**POST `/models/{model_name}/register`**
- Description: Register a new model in the system
- Request Body:
```json
{
    "model_type": "llm",
    "source_url": "https://huggingface.co/model/repo",
    "metadata": {
        "description": "7B parameter language model",
        "tags": ["llm", "chat"],
        "requirements": {
            "min_memory": 16000000000,
            "min_vram": 8000000000
        }
    }
}
```

**DELETE `/models/{model_name}`**
- Description: Unregister model and cleanup all instances

#### Model Location Management

**GET `/models/{model_name}/locations`**
- Description: Get all storage locations for a model
- Response: List of device locations and their status

**POST `/models/{model_name}/locations/{device_name}`**
- Description: Track a new storage location for a model

**DELETE `/models/{model_name}/locations/{device_name}`**
- Description: Remove a storage location record

#### Service Management

**GET `/models/{model_name}/services`**
- Description: Get all active service endpoints for a model
- Response: List of service URLs and their status

**POST `/models/{model_name}/services`**
- Description: Request model serving on specified nodes
- Request Body:
```json
{
    "target_nodes": ["node1", "node2"],
    "port": 8080,
    "config": {
        "max_batch_size": 32,
        "timeout": 30
    }
}
```

**DELETE `/models/{model_name}/services/{node_id}`**
- Description: Stop model service on specified node

#### System Status

**GET `/status`**
- Description: Get system-wide model management status
- Response: Storage utilization, active services, node health

**GET `/nodes`**
- Description: List all connected nodes and their capabilities
- Response: Node specifications, available storage, current load

**GET `/health`**
- Description: Health check endpoint
- Response: System health indicators

### Client Node APIs

#### Model Storage Management

**GET `/models`**
- Description: List locally stored models
- Response: Local model inventory

**POST `/models/{model_name}/download`**
- Description: Download model from source or another node
- Request Body:
```json
{
    "source": "web|node_id:device",
    "target_device": "disk|dram|gpu0",
    "source_url": "https://huggingface.co/...",  // if source=web
    "priority": "high|normal|low"
}
```

**POST `/models/{model_name}/move`**
- Description: Move model between local storage devices
- Request Body:
```json
{
    "from_device": "disk",
    "to_device": "gpu0",
    "keep_source": false
}
```

**DELETE `/models/{model_name}`**
- Description: Remove model from local storage
- Query Parameters: `device` - specific device to remove from

#### Model Serving

**POST `/models/{model_name}/serve`**
- Description: Start serving a model
- Request Body:
```json
{
    "port": 8080,
    "device": "gpu0",
    "config": {
        "max_batch_size": 32,
        "timeout": 30,
        "framework": "vllm|transformers|tgi"
    }
}
```

**DELETE `/models/{model_name}/serve`**
- Description: Stop serving a model

**GET `/models/{model_name}/serve/status`**
- Description: Get serving status and metrics

#### Resource Management

**GET `/storage`**
- Description: Get storage utilization across all devices
- Response:
```json
{
    "disk": {"total": 1000000000, "used": 500000000, "available": 500000000},
    "dram": {"total": 64000000000, "used": 32000000000, "available": 32000000000},
    "gpu0": {"total": 24000000000, "used": 8000000000, "available": 16000000000}
}
```

#### More device details can be read from gswarm-profiler

## Additional APIs Needed

### Job Management

**POST `/jobs/{job_id}/create`**
- Description: Create complex model management job

**GET `/jobs/{job_id}/status`**
- Description: Get job execution status

**POST `/jobs/{job_id}/cancel`**
- Description: Cancel running job



## Model Execution YAML Schema

Simple action sequence with dependencies:

```yaml
# Model execution workflow
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
    devices: ["node1:disk", "node1:gpu0"]  # src, dst
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
    devices: []  # no devices needed for health check
    dependencies: ["serve_model"]
```

### More Examples

#### Multi-Model Pipeline

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

### Distributed Model (Multi-GPU)

```yaml
name: "distributed-llama-70b"
description: "Deploy large model across multiple GPUs"

actions:
  - action_id: "download_model"
    action_type: "download"
    model_name: "llama-70b"
    source_url: "https://huggingface.co/meta-llama/Llama-2-70b-hf"
    devices: ["node1:disk"]
    dependencies: []

  # Load to multiple GPUs
  - action_id: "load_gpu0"
    action_type: "move"
    model_name: "llama-70b"
    devices: ["node1:disk", "node1:gpu0", "node1:gpu1", "node1:gpu2"]
    dependencies: ["download_model"]

  # Serve across multiple GPUs
  - action_id: "serve_distributed"
    action_type: "serve"
    model_name: "llama-70b"
    port: 8080
    devices: ["node1:gpu0", "node1:gpu1", "node1:gpu2"]
    dependencies: ["load_gpu0"]
```

### Model Migration Pipeline

```yaml
name: "model-migration"
description: "Move model between nodes"

actions:
  # Copy model from node1 to node2
  - action_id: "transfer_model"
    action_type: "copy"  # copy between nodes
    model_name: "llama-7b"
    devices: ["node1:disk", "node2:disk"]  # src, dst
    dependencies: []

  # Stop service on old node
  - action_id: "stop_old_service"
    action_type: "stop_serve"
    model_name: "llama-7b"
    devices: ["node1:gpu0"]
    dependencies: ["transfer_model"]

  # Start service on new node
  - action_id: "start_new_service"
    action_type: "serve"
    model_name: "llama-7b"
    port: 8080
    devices: ["node2:gpu0"]
    dependencies: ["transfer_model"]

  # Cleanup old model
  - action_id: "cleanup_old"
    action_type: "delete"
    model_name: "llama-7b"
    devices: ["node1:disk", "node1:gpu0"]
    dependencies: ["stop_old_service"]
```

## Implementation Notes

### Error Handling

All APIs should return consistent error formats
Implement retry mechanisms for network operations
Graceful degradation when nodes are unavailable

### Performance Considerations

Implement caching for frequently accessed model metadata
Use streaming for large model transfers
Support resume for interrupted downloads

This design provides a comprehensive foundation for distributed model management with room for future extensions and optimizations.
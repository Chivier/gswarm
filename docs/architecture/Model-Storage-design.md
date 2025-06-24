# Model Storage and Management System Design

## Overview

This document outlines the design for a distributed model storage and management system integrated with gswarm. The system enables efficient model distribution, storage management, data pooling, and service orchestration across multiple nodes in a GPU cluster.

## Architecture

### System Components

1. **Host Node**: Central coordinator that maintains global model registry and orchestrates model operations
2. **Client Nodes**: Worker nodes that store, serve, and execute models with message queue and data pool capabilities
3. **Model Registry**: Distributed database tracking model locations and availability
4. **Service Manager**: Handles model serving and API endpoints
5. **Message Queue**: Asynchronous task execution system for client nodes
6. **Data Pool**: Distributed data management system for model inputs/outputs

### Status

#### Host Node Model Registry

```python
{
    "model_status": [{
        "model_name": str,                    # Unique model identifier
        "model_type": str,                    # e.g., "llm", "diffusion", "vision"
        "model_size": int,                    # Size in bytes
        "stored_locations": [str],            # List of device_name
        "available_services": {               # Active serving endpoints
            "device_name": "http://node:port",
            ...
        },
    }],
    "node_list": ["node_name1", "node_name2", ...]
}
```

#### Client Node Model Registry

```python
{
    "model_status": [{
        "model_name": str,                   # Unique model identifier
        "model_type": str,                   # e.g., "llm", "diffusion", "vision"
        "model_size": int,                   # Size in bytes
        "model_status": str,                 # "ready", "downloading", "moving", "unavailable", "queued"
        "stored_locations": [str],           # List of device_name
        "available_services": [str],         # Active serving endpoints
    }],
    "task_queue": {                          # Message queue status
        "pending": int,                      # Number of pending tasks
        "running": int,                      # Currently executing tasks
        "completed": int                     # Completed tasks in current session
    },
    "data_pool": {                           # Data pool status
        "chunks": int,                       # Number of data chunks
        "total_size": int,                   # Total size in bytes
        "locations": dict                    # Device location summary
    }
}
```

### Device Naming Convention

Device names follow the pattern: `<node_identifier>:<storage_type>[:<index>]`

**Model Status:**
- `ready`: Model is ready to be used
- `downloading`: Model is being downloaded from web
- `moving`: Model is being moved to another device
- `unavailable`: Model is unavailable
- `queued`: Operation is queued for execution

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

## Client Node Message Queue System

### Queue Architecture

```python
{
    "queue_config": {
        "max_concurrent_tasks": int,         # Maximum parallel tasks
        "priority_levels": ["critical", "high", "normal", "low"],
        "resource_tracking": bool            # Enable resource conflict detection
    },
    "task": {
        "task_id": str,                      # Unique task identifier
        "task_type": str,                    # download, move, serve, etc.
        "priority": str,                     # Task priority level
        "dependencies": [str],               # List of task_ids this depends on
        "resources": {                       # Resources this task needs
            "devices": [str],                # Device requirements
            "models": [str],                 # Model requirements
            "exclusive": bool                # Requires exclusive access
        },
        "status": str,                       # pending, running, completed, failed
        "created_at": float,                 # Timestamp
        "started_at": float,                 # Optional
        "completed_at": float                # Optional
    }
}
```

### Task Scheduling Rules

1. Tasks are executed based on priority and dependencies
2. Resource conflicts are automatically detected and queued
3. Independent tasks without resource conflicts can run in parallel
4. Failed tasks can be retried with exponential backoff

## Data Pool System

### Data Pool Architecture

```python
{
    "data_pool": {
        "chunk_id": str,                     # Unique chunk identifier
        "chunk_type": str,                   # input, output, intermediate
        "size": int,                         # Size in bytes
        "locations": [{                      # Where chunk is stored
            "device": str,                   # Device name
            "path": str,                     # Storage path
            "status": str                    # available, moving, deleted
        }],
        "metadata": {
            "created_by": str,               # Model/service that created it
            "created_at": float,             # Timestamp
            "last_accessed": float,          # Last access time
            "access_count": int,             # Number of accesses
            "checksum": str,                 # Data integrity check
            "format": str                    # Data format (tensor, json, etc.)
        },
        "references": [str]                  # Models/services using this chunk
    }
}
```

### Data Pool Features

1. Automatic data migration between devices
2. Reference counting for garbage collection
3. Data deduplication with checksums
4. Transparent access across nodes
5. Streaming support for large data transfers

## API Design

### Host Node APIs

#### Model Management

**GET `/api/v1/models`**
- Description: List all registered models with filtering options
- Query Parameters:
  - `type`: Filter by model type
  - `location`: Filter by storage location
  - `status`: Filter by model status
- Response:
```json
{
    "models": [{
        "model_name": "llama-7b",
        "model_type": "llm",
        "model_size": 13968179200,
        "locations": ["node1:disk", "node2:gpu0"],
        "services": {
            "node1": "http://node1:8080",
            "node2": "http://node2:8081"
        },
        "status": "ready"
    }],
    "total": 1,
    "page": 1,
    "per_page": 20
}
```

**GET `/api/v1/models/{model_name}`**
- Description: Get detailed model information including all metadata

**POST `/api/v1/models`**
- Description: Register a new model in the system
- Request Body:
```json
{
    "model_name": "llama-7b",
    "model_type": "llm",
    "model_size": 13968179200,
    "source_url": "https://huggingface.co/model/repo",
    "metadata": {
        "framework": "pytorch",
        "precision": "fp16"
    }
}
```

**DELETE `/api/v1/models/{model_name}`**
- Description: Unregister model and cleanup all instances
- Query Parameters:
  - `force`: Force deletion even if services are running

#### Location Management

**GET `/api/v1/models/{model_name}/locations`**
- Description: Get all storage locations for a model

**POST `/api/v1/models/{model_name}/locations`**
- Description: Add a new storage location
- Request Body:
```json
{
    "device": "node1:gpu0",
    "status": "ready"
}
```

**DELETE `/api/v1/models/{model_name}/locations/{device}`**
- Description: Remove a storage location

#### Service Management

**GET `/api/v1/services`**
- Description: List all active services across the cluster

**POST `/api/v1/services`**
- Description: Deploy model service on specified nodes
- Request Body:
```json
{
    "model_name": "llama-7b",
    "nodes": ["node1", "node2"],
    "config": {
        "port": 8080,
        "replicas": 2,
        "load_balancing": "round_robin",
        "max_batch_size": 32,
        "timeout": 30
    }
}
```

**DELETE `/api/v1/services/{service_id}`**
- Description: Stop and remove a service

#### System Management

**GET `/api/v1/nodes`**
- Description: List all nodes with their capabilities and status

**GET `/api/v1/health`**
- Description: System health check with detailed status

**GET `/api/v1/metrics`**
- Description: System-wide metrics and statistics

### Client Node APIs

#### Model Operations

**GET `/api/v1/models`**
- Description: List locally stored models

**POST `/api/v1/models/{model_name}/download`**
- Description: Download model from source
- Request Body:
```json
{
    "source_url": "https://huggingface.co/...",
    "target_device": "disk",
    "priority": "normal",
    "verify_checksum": true
}
```
- Returns:
```json
{
    "task_id": "download-llama-7b-1234",
    "status": "queued",
    "position": 3,
    "estimated_time": 300
}
```

**POST `/api/v1/models/{model_name}/move`**
- Description: Move model between storage devices
- Request Body:
```json
{
    "source_device": "disk",
    "target_device": "gpu0",
    "keep_source": false,
    "priority": "high",
    "compression": "none",
    "verify_checksum": true
}
```
- Move Operation Details:
  - Status updates: `queued` → `moving` → `verifying` → `ready`
  - Progress tracking via task API
  - Automatic rollback on failure
  - Support for cross-node moves with `node:device` format

**POST `/api/v1/models/{model_name}/copy`**
- Description: Copy model to another location (including cross-node)
- Request Body:
```json
{
    "source_device": "node1:disk",
    "target_device": "node2:disk",
    "priority": "normal",
    "bandwidth_limit": 1000000000  // bytes per second, optional
}
```

**DELETE `/api/v1/models/{model_name}`**
- Description: Remove model from specified device
- Query Parameters:
  - `device`: Target device to remove from
  - `force`: Force removal even if in use

#### Model Serving

**POST `/api/v1/services`**
- Description: Start serving a model
- Request Body:
```json
{
    "model_name": "llama-7b",
    "device": "gpu0",
    "port": 8080,
    "config": {
        "framework": "vllm",
        "max_batch_size": 32,
        "max_sequence_length": 2048,
        "gpu_memory_fraction": 0.9
    }
}
```

**DELETE `/api/v1/services/{service_id}`**
- Description: Stop a running service

**GET `/api/v1/services/{service_id}/status`**
- Description: Get service status and metrics

#### Message Queue Management

**GET `/api/v1/queue`**
- Description: Get queue status and pending tasks

**GET `/api/v1/queue/tasks/{task_id}`**
- Description: Get specific task details

**POST `/api/v1/queue/tasks/{task_id}/cancel`**
- Description: Cancel a pending or running task

**GET `/api/v1/queue/history`**
- Description: Get task execution history
- Query Parameters:
  - `limit`: Number of records
  - `status`: Filter by status
  - `since`: Timestamp filter

#### Data Pool Management

**GET `/api/v1/data`**
- Description: List data chunks in the pool

**POST `/api/v1/data`**
- Description: Create a new data chunk
- Request Body:
```json
{
    "data": "base64_encoded_or_url",
    "type": "input",
    "format": "tensor",
    "device": "dram",
    "metadata": {
        "shape": [1, 512, 768],
        "dtype": "float16"
    }
}
```

**GET `/api/v1/data/{chunk_id}`**
- Description: Get data chunk information

**POST `/api/v1/data/{chunk_id}/move`**
- Description: Move data chunk between devices
- Request Body:
```json
{
    "target_device": "gpu0",
    "priority": "high"
}
```

**POST `/api/v1/data/{chunk_id}/transfer`**
- Description: Transfer data chunk to another node
- Request Body:
```json
{
    "target_node": "node2",
    "target_device": "dram",
    "delete_source": false
}
```

**DELETE `/api/v1/data/{chunk_id}`**
- Description: Remove data chunk from pool

#### Resource Monitoring

**GET `/api/v1/resources`**
- Description: Get current resource utilization
- Response:
```json
{
    "storage": {
        "disk": {"total": 1000000000000, "used": 500000000000, "available": 500000000000},
        "dram": {"total": 64000000000, "used": 32000000000, "available": 32000000000},
        "gpu0": {"total": 24000000000, "used": 8000000000, "available": 16000000000}
    },
    "compute": {
        "cpu": {"cores": 32, "usage": 45.5},
        "gpu0": {"usage": 78.2, "temperature": 65}
    },
    "network": {
        "bandwidth": {"in": 125000000, "out": 100000000}
    }
}
```

## Workflow Examples

### Model Deployment with Data Pipeline

```yaml
name: "model-deployment-with-data"
description: "Deploy model and setup data pipeline"

actions:
  # Download model
  - action_id: "download_model"
    action_type: "download"
    model_name: "llama-7b"
    source_url: "https://huggingface.co/meta-llama/Llama-2-7b"
    target_device: "node1:disk"
    priority: "high"
    dependencies: []

  # Prepare input data
  - action_id: "prepare_data"
    action_type: "data_create"
    data_type: "input"
    source: "s3://bucket/input-data"
    target_device: "node1:dram"
    dependencies: []

  # Load model to GPU
  - action_id: "load_model"
    action_type: "move"
    model_name: "llama-7b"
    source_device: "node1:disk"
    target_device: "node1:gpu0"
    keep_source: true
    dependencies: ["download_model"]

  # Start model service
  - action_id: "start_service"
    action_type: "serve"
    model_name: "llama-7b"
    device: "node1:gpu0"
    port: 8080
    dependencies: ["load_model"]

  # Process data
  - action_id: "process_data"
    action_type: "inference"
    service_url: "http://node1:8080"
    input_data_id: "${prepare_data.output.chunk_id}"
    output_device: "node1:dram"
    dependencies: ["start_service", "prepare_data"]

  # Transfer results to next node
  - action_id: "transfer_results"
    action_type: "data_transfer"
    data_id: "${process_data.output.chunk_id}"
    target_node: "node2"
    target_device: "node2:dram"
    dependencies: ["process_data"]
```

### Multi-Stage Pipeline with Queue Management

```yaml
name: "multi-stage-pipeline"
description: "Complex pipeline with parallel execution"

queue_config:
  max_concurrent_tasks: 4
  enable_resource_tracking: true

actions:
  # Stage 1: Parallel model downloads
  - action_id: "download_llm"
    action_type: "download"
    model_name: "llama-7b"
    target_device: "node1:disk"
    priority: "high"
    dependencies: []

  - action_id: "download_vision"
    action_type: "download"
    model_name: "clip-vit-large"
    target_device: "node2:disk"
    priority: "high"
    dependencies: []

  # Stage 2: Data preparation (can run in parallel with downloads)
  - action_id: "prepare_text_data"
    action_type: "data_create"
    source: "s3://bucket/text-data"
    target_device: "node1:dram"
    dependencies: []

  - action_id: "prepare_image_data"
    action_type: "data_create"
    source: "s3://bucket/image-data"
    target_device: "node2:dram"
    dependencies: []

  # Stage 3: Model deployment
  - action_id: "deploy_llm"
    action_type: "serve"
    model_name: "llama-7b"
    device: "node1:gpu0"
    dependencies: ["download_llm"]

  - action_id: "deploy_vision"
    action_type: "serve"
    model_name: "clip-vit-large"
    device: "node2:gpu0"
    dependencies: ["download_vision"]

  # Stage 4: Processing pipeline
  - action_id: "process_text"
    action_type: "inference"
    service: "${deploy_llm.service_url}"
    input: "${prepare_text_data.chunk_id}"
    output_device: "node1:dram"
    dependencies: ["deploy_llm", "prepare_text_data"]

  - action_id: "process_images"
    action_type: "inference"
    service: "${deploy_vision.service_url}"
    input: "${prepare_image_data.chunk_id}"
    output_device: "node2:dram"
    dependencies: ["deploy_vision", "prepare_image_data"]

  # Stage 5: Combine results
  - action_id: "combine_results"
    action_type: "data_merge"
    inputs: ["${process_text.output}", "${process_images.output}"]
    output_device: "node1:dram"
    dependencies: ["process_text", "process_images"]
```

## Error Handling and Recovery

### Consistent Error Response Format

```json
{
    "error": {
        "code": "RESOURCE_UNAVAILABLE",
        "message": "Insufficient GPU memory on device gpu0",
        "details": {
            "required": 16000000000,
            "available": 8000000000,
            "device": "node1:gpu0"
        },
        "timestamp": 1234567890.123,
        "request_id": "req-123-456"
    }
}
```

### Error Codes

- `RESOURCE_UNAVAILABLE`: Insufficient resources
- `MODEL_NOT_FOUND`: Model does not exist
- `TASK_FAILED`: Task execution failed
- `DEPENDENCY_FAILED`: Required dependency failed
- `NETWORK_ERROR`: Network communication error
- `INVALID_REQUEST`: Invalid request parameters
- `CONFLICT`: Resource conflict or race condition

### Recovery Mechanisms

1. **Automatic Retry**: Failed tasks retry with exponential backoff
2. **Checkpoint Recovery**: Long operations save progress for resume
3. **Rollback**: Failed moves/copies automatically rollback
4. **Health Monitoring**: Automatic node failure detection and rerouting

## Performance Optimizations

### Caching Strategy

1. **Model Metadata Cache**: Frequently accessed model info cached in memory
2. **Data Chunk Index**: Fast lookup for data pool chunks
3. **Service Discovery Cache**: Active service endpoints cached with TTL

### Transfer Optimizations

1. **Compression**: Optional compression for network transfers
2. **Chunked Transfers**: Large files transferred in resumable chunks
3. **Parallel Streams**: Multiple connections for faster transfers
4. **Bandwidth Management**: Configurable limits to prevent saturation

### Resource Management

1. **Predictive Loading**: Pre-load models based on usage patterns
2. **Smart Eviction**: LRU-based cache eviction with priorities
3. **Resource Pooling**: Reuse connections and buffers
4. **Batch Operations**: Group similar operations for efficiency

## Integration with gswarm

### Unified CLI

The project has been restructured to provide a single `gswarm` CLI with subcommands:

```bash
# Profiler functionality
gswarm profiler start
gswarm profiler status
gswarm profiler report

# Model management functionality
gswarm model list
gswarm model download llama-7b
gswarm model serve llama-7b --device gpu0

# Data pool management
gswarm data list
gswarm data create --source s3://bucket/data
gswarm data transfer chunk-123 --to node2

# Queue management
gswarm queue status
gswarm queue tasks
gswarm queue cancel task-456
```

### Consistent Node Naming

All components use the same node naming convention:
- Node names are consistent across profiler and model components
- Device names follow the unified pattern
- Service discovery uses the same registry

### Shared Configuration

```yaml
# ~/.gswarm/config.yaml
cluster:
  host: "master.cluster.local"
  port: 8000
  
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

## Implementation Status

### Completed
- ✅ Unified project structure
- ✅ Consistent API design
- ✅ Message queue architecture
- ✅ Data pool design
- ✅ Enhanced move operations

### In Progress
- 🔄 Task queue implementation
- 🔄 Data pool implementation
- 🔄 Cross-node transfers

### Planned
- ⏳ Web UI dashboard
- ⏳ Kubernetes operator
- ⏳ Multi-cloud support

This design provides a robust foundation for distributed model and data management with enhanced queue processing and data pooling capabilities.
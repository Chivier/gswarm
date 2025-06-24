# gswarm API Reference

## Overview

gswarm provides both REST and gRPC APIs for different components. This document provides a comprehensive reference for all available APIs.

## Base URLs

- **Profiler gRPC**: `grpc://host:8090`
- **Profiler HTTP**: `http://host:8091`
- **Model Host API**: `http://host:9010`
- **Model Client API**: `http://host:9011`

## REST API Conventions

### Request Format
- Content-Type: `application/json`
- Accept: `application/json`

### Response Format
```json
{
    "success": true,
    "data": { ... },
    "error": null
}
```

### Error Response
```json
{
    "success": false,
    "data": null,
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": { ... }
    }
}
```

## Model Management APIs (Host)

### Model Registry

#### List Models
```http
GET /api/v1/models
```

Query Parameters:
- `type` (string): Filter by model type
- `location` (string): Filter by storage location
- `status` (string): Filter by model status
- `page` (int): Page number (default: 1)
- `per_page` (int): Items per page (default: 20)

Response:
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

#### Get Model Details
```http
GET /api/v1/models/{model_name}
```

Response:
```json
{
    "model_name": "llama-7b",
    "model_type": "llm",
    "model_size": 13968179200,
    "locations": ["node1:disk", "node2:gpu0"],
    "services": {
        "node1": "http://node1:8080"
    },
    "status": "ready",
    "metadata": {
        "framework": "pytorch",
        "precision": "fp16",
        "created_at": "2024-01-01T00:00:00Z"
    }
}
```

#### Register Model
```http
POST /api/v1/models
```

Request Body:
```json
{
    "model_name": "llama-7b",
    "model_type": "llm",
    "model_size": 13968179200,
    "source_url": "https://huggingface.co/meta-llama/Llama-2-7b",
    "metadata": {
        "framework": "pytorch",
        "precision": "fp16"
    }
}
```

#### Delete Model
```http
DELETE /api/v1/models/{model_name}
```

Query Parameters:
- `force` (boolean): Force deletion even if services are running

### Location Management

#### Get Model Locations
```http
GET /api/v1/models/{model_name}/locations
```

#### Add Location
```http
POST /api/v1/models/{model_name}/locations
```

Request Body:
```json
{
    "device": "node1:gpu0",
    "status": "ready"
}
```

#### Remove Location
```http
DELETE /api/v1/models/{model_name}/locations/{device}
```

### Service Management

#### List Services
```http
GET /api/v1/services
```

Response:
```json
{
    "services": [{
        "service_id": "llama-7b-node1-8080",
        "model_name": "llama-7b",
        "node": "node1",
        "device": "gpu0",
        "port": 8080,
        "status": "running",
        "url": "http://node1:8080",
        "created_at": "2024-01-01T00:00:00Z"
    }]
}
```

#### Deploy Service
```http
POST /api/v1/services
```

Request Body:
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

#### Stop Service
```http
DELETE /api/v1/services/{service_id}
```

### System Management

#### List Nodes
```http
GET /api/v1/nodes
```

Response:
```json
{
    "nodes": [{
        "node_id": "node1",
        "hostname": "gpu-server-01",
        "ip_address": "192.168.1.101",
        "status": "online",
        "last_heartbeat": "2024-01-01T00:00:00Z",
        "capabilities": {
            "gpus": ["gpu0", "gpu1"],
            "storage": {
                "disk": 1000000000000,
                "dram": 64000000000
            }
        }
    }]
}
```

#### Health Check
```http
GET /api/v1/health
```

#### System Metrics
```http
GET /api/v1/metrics
```

## Model Operations APIs (Client)

### Model Storage

#### List Local Models
```http
GET /api/v1/models
```

#### Download Model
```http
POST /api/v1/models/{model_name}/download
```

Request Body:
```json
{
    "source_url": "https://huggingface.co/...",
    "target_device": "disk",
    "priority": "normal",
    "verify_checksum": true
}
```

Response:
```json
{
    "task_id": "download-llama-7b-1234",
    "status": "queued",
    "position": 3,
    "estimated_time": 300
}
```

#### Move Model
```http
POST /api/v1/models/{model_name}/move
```

Request Body:
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

#### Copy Model
```http
POST /api/v1/models/{model_name}/copy
```

Request Body:
```json
{
    "source_device": "node1:disk",
    "target_device": "node2:disk",
    "priority": "normal",
    "bandwidth_limit": 1000000000
}
```

#### Delete Model
```http
DELETE /api/v1/models/{model_name}
```

Query Parameters:
- `device` (string): Target device to remove from
- `force` (boolean): Force removal even if in use

### Model Serving

#### Start Service
```http
POST /api/v1/services
```

Request Body:
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

#### Stop Service
```http
DELETE /api/v1/services/{service_id}
```

#### Get Service Status
```http
GET /api/v1/services/{service_id}/status
```

### Resource Management

#### Get Storage Status
```http
GET /api/v1/resources
```

Response:
```json
{
    "storage": {
        "disk": {
            "total": 1000000000000,
            "used": 500000000000,
            "available": 500000000000
        },
        "dram": {
            "total": 64000000000,
            "used": 32000000000,
            "available": 32000000000
        },
        "gpu0": {
            "total": 24000000000,
            "used": 8000000000,
            "available": 16000000000
        }
    },
    "compute": {
        "cpu": {
            "cores": 32,
            "usage": 45.5
        },
        "gpu0": {
            "usage": 78.2,
            "temperature": 65
        }
    }
}
```

## Data Pool APIs

### Data Management

#### List Data Chunks
```http
GET /api/v1/data
```

Query Parameters:
- `device` (string): Filter by device
- `type` (string): Filter by chunk type

Response:
```json
{
    "chunks": [{
        "chunk_id": "chunk-a1b2c3d4e5f6",
        "chunk_type": "input",
        "size": 1048576,
        "format": "tensor",
        "locations": [
            {"device": "node1:dram", "status": "available"}
        ],
        "metadata": {
            "created_by": "llama-7b",
            "created_at": "2024-01-01T00:00:00Z",
            "access_count": 5
        }
    }]
}
```

#### Create Data Chunk
```http
POST /api/v1/data
```

Request Body:
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

#### Get Chunk Info
```http
GET /api/v1/data/{chunk_id}
```

#### Move Data Chunk
```http
POST /api/v1/data/{chunk_id}/move
```

Request Body:
```json
{
    "target_device": "gpu0",
    "priority": "high"
}
```

#### Transfer Data Chunk
```http
POST /api/v1/data/{chunk_id}/transfer
```

Request Body:
```json
{
    "target_node": "node2",
    "target_device": "dram",
    "delete_source": false
}
```

#### Delete Data Chunk
```http
DELETE /api/v1/data/{chunk_id}
```

Query Parameters:
- `force` (boolean): Force deletion even if referenced

## Task Queue APIs

### Queue Management

#### Get Queue Status
```http
GET /api/v1/queue
```

Response:
```json
{
    "pending": 5,
    "running": 2,
    "completed": 150,
    "config": {
        "max_concurrent_tasks": 4,
        "priority_levels": ["critical", "high", "normal", "low"],
        "resource_tracking": true
    }
}
```

#### List Tasks
```http
GET /api/v1/queue/tasks
```

Query Parameters:
- `status` (string): Filter by task status
- `limit` (int): Maximum tasks to return

Response:
```json
{
    "tasks": [{
        "task_id": "download-llama-7b-1234",
        "task_type": "download",
        "priority": "normal",
        "status": "running",
        "dependencies": [],
        "resources": {
            "devices": ["node1:disk"],
            "models": ["llama-7b"]
        },
        "created_at": 1704067200.0,
        "started_at": 1704067210.0,
        "progress": 45
    }]
}
```

#### Get Task Details
```http
GET /api/v1/queue/tasks/{task_id}
```

#### Cancel Task
```http
POST /api/v1/queue/tasks/{task_id}/cancel
```

#### Get Task History
```http
GET /api/v1/queue/history
```

Query Parameters:
- `limit` (int): Number of records
- `since` (string): Timestamp filter
- `status` (string): Filter by final status

## Profiler APIs

### gRPC APIs (protobuf)

See [grpc-protocol-design.md](grpc-protocol-design.md) for detailed gRPC API documentation.

### HTTP Control APIs

#### Get Status
```http
GET /status
```

Response:
```json
{
    "is_profiling": true,
    "active_sessions": [
        {
            "name": "training_run_1",
            "start_time": "2024-01-01T00:00:00Z",
            "frames_collected": 1500
        }
    ],
    "connected_clients": 3,
    "enable_bandwidth_profiling": true,
    "enable_nvlink_profiling": false,
    "sampling_frequency_ms": 1000
}
```

#### Start Profiling
```http
POST /profiling/start
```

Request Body:
```json
{
    "name": "my_experiment"
}
```

#### Stop Profiling
```http
POST /profiling/stop
```

Request Body:
```json
{
    "name": "my_experiment"
}
```

#### List Sessions
```http
GET /profiling/sessions
```

#### Get Connected Clients
```http
GET /clients
```

Response:
```json
{
    "clients": [
        {
            "client_id": "node1",
            "hostname": "gpu-server-01",
            "gpus": [
                {
                    "device_id": 0,
                    "name": "NVIDIA A100",
                    "memory_total": 40960,
                    "compute_capability": "8.0"
                }
            ],
            "last_update": "2024-01-01T00:00:00Z",
            "status": "healthy"
        }
    ]
}
```

#### Get Latest Metrics
```http
GET /metrics/latest
```

## Error Codes

### Common Error Codes

| Code | Description |
|------|-------------|
| `RESOURCE_UNAVAILABLE` | Insufficient resources (memory, storage, etc.) |
| `MODEL_NOT_FOUND` | Model does not exist in registry |
| `TASK_FAILED` | Task execution failed |
| `DEPENDENCY_FAILED` | Required dependency task failed |
| `NETWORK_ERROR` | Network communication error |
| `INVALID_REQUEST` | Invalid request parameters |
| `CONFLICT` | Resource conflict or race condition |
| `NOT_IMPLEMENTED` | Feature not yet implemented |
| `INTERNAL_ERROR` | Internal server error |

### HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 201 | Created |
| 204 | No Content (successful deletion) |
| 400 | Bad Request |
| 404 | Not Found |
| 409 | Conflict |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## Rate Limiting

Currently, no rate limiting is implemented. Future versions will include:
- Per-client rate limits
- Endpoint-specific limits
- Burst allowances

## Authentication

Currently, no authentication is required. Future versions will support:
- API key authentication
- JWT tokens
- Role-based access control

## Versioning

All APIs are versioned with `/api/v1/` prefix. When breaking changes are introduced:
- New version will be `/api/v2/`
- Old version supported for 6 months
- Deprecation warnings in headers

## WebSocket APIs (Future)

Planned WebSocket endpoints for real-time updates:
- `/ws/profiling`: Real-time profiling metrics
- `/ws/tasks`: Task status updates
- `/ws/models`: Model status changes 
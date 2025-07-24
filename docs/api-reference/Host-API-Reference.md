# gswarm API Reference

## Overview

gswarm provides a REST API for its model management and profiler components. This document provides a comprehensive reference for all available APIs.

## Base URLs

- **Profiler HTTP**: `http://host:8091`
- **Model Manager API**: `http://host:8100` (Note: Port updated from 9010 to 8100 as per latest code)

## REST API Conventions

### Request Format
- Content-Type: `application/json`
- Accept: `application/json`

### Response Format
```json
{
    "success": true,
    "message": "...",
    "data": { ... }
}
```

### Error Response
```json
{
    "detail": "Error message"
}
```

---

## Model Management APIs (FastAPI Head)

These endpoints are served by the Model Manager on port `8100`.

### System Management

#### Root
```http
GET /
```
- **Description**: Root endpoint for basic connectivity check.
- **Response**: `{"message": "GSwarm Model Manager API", "version": "0.5.0", "config_loaded": true}`

#### Health Check
```http
GET /health
```
- **Description**: Provides health status and memory usage.
- **Response**: `{"status": "healthy", "timestamp": "...", "memory_usage": {...}}`

#### Get Configuration
```http
GET /config
```
- **Description**: Returns the current server configuration.
- **Response**: `{"model_cache_dir": "...", "dram_cache_dir": "...", "model_manager_port": 8100}`

### Model Registry

#### List Models
```http
GET /models
```
- **Description**: Lists all registered models with their status.
- **Response**: `{"models": [...], "count": ...}`

#### Get Model Details
```http
GET /models/{model_name}
```
- **Description**: Get detailed information for a specific model.
- **Response**: A JSON object with model details, checkpoints, and serving instances.

#### Register Model
```http
POST /models
```
- **Description**: Register a new model in the system.
- **Request Body**: `RegisterModelRequest`
  ```json
  {
      "name": "llama-7b",
      "type": "llm",
      "metadata": {"author": "meta"}
  }
  ```

#### Discover Models
```http
POST /discover
```
- **Description**: Manually triggers a scan of cache directories to discover and register models.

#### Validate All Models
```http
POST /models/validate
```
- **Description**: Validates the cache paths for all registered models and updates their statuses.

#### Validate Single Model
```http
GET /models/{model_name}/validate
```
- **Description**: Validates the cache path for a single model.

### Model Operations

#### Download Model
```http
POST /download
```
- **Description**: Downloads a model from a source URL to disk or DRAM.
- **Request Body**: `DownloadRequest`
  ```json
  {
      "model_name": "llama-7b",
      "source_url": "hf://meta-llama/Llama-2-7b-hf",
      "target_device": "disk"
  }
  ```

#### Copy Model
```http
POST /copy
```
- **Description**: Copies a model between storage devices (disk, DRAM). If the target is a GPU, it will trigger a serve operation.
- **Request Body**: `CopyRequest`
  ```json
  {
      "model_name": "llama-7b",
      "source_device": "disk",
      "target_device": "dram",
      "keep_source": false
  }
  ```

### Model Serving

#### Serve Model
```http
POST /serve
```
- **Description**: Starts a vLLM server to serve a model on a specified GPU.
- **Request Body**: `ServeRequest`
  ```json
  {
      "model_name": "llama-7b",
      "source_device": "disk",
      "gpu_device": "gpu0",
      "port": 8080,
      "config": {"gpu_memory_utilization": 0.9}
  }
  ```

#### Stop Serving Instance
```http
POST /stop_serve
```
- **Description**: Stops a specific model serving instance.
- **Request Body**: `StopServeRequest`
  ```json
  {
      "model_name": "llama-7b",
      "instance_id": "..."
  }
  ```

#### List All Serving Instances
```http
GET /serving
```
- **Description**: Lists all active serving instances across all models.

#### Get Model Serving Instances
```http
GET /serving/{model_name}
```
- **Description**: Lists all serving instances for a specific model.

### In-Memory Storage Management

#### Get All Memory Models
```http
GET /memory/models
```
- **Description**: Lists all models currently loaded in DRAM or on a GPU.

#### Get DRAM Models
```http
GET /memory/dram
```
- **Description**: Lists models loaded in DRAM.

#### Get GPU Models
```http
GET /memory/gpu
```
- **Description**: Lists model instances loaded on GPUs.

#### Unload DRAM Model
```http
DELETE /memory/dram/{model_name}
```
- **Description**: Unloads a model from DRAM, freeing up memory.

---

## Profiler APIs (HTTP Control)

These endpoints are served by the Profiler on port `8091`.

#### Get Status
```http
GET /status
```
- **Description**: Retrieves the current status of the profiler, including connected clients and active sessions.

#### Start Profiling
```http
POST /profiling/start
```
- **Description**: Starts a new profiling session.
- **Request Body**:
  ```json
  {
      "name": "my_experiment",
      "report_metrics": ["gpu_utilization", "gpu_memory"]
  }
  ```

#### Stop Profiling
```http
POST /profiling/stop
```
- **Description**: Stops the currently active profiling session.

#### Get Connected Clients
```http
GET /clients
```
- **Description**: Returns a list of all clients connected to the profiler.

#### Get Latest Metrics
```http
GET /metrics/latest
```
- **Description**: Fetches the most recent metrics payload from all connected clients. This is useful for real-time monitoring.
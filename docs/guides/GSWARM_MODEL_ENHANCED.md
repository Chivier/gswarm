# GSwarm Model System - Enhanced Features

This document describes the enhanced features of the GSwarm model management system.

## Key Enhancements

### 1. Fixed Disk Path
- **Path**: `~/.cache/gswarm/models`
- All models are now stored in a consistent location
- Simplified path management and discovery

### 2. Enhanced Configuration System
- **Location**: `~/.gswarm.conf` (YAML format)
- Separated into `host` and `client` sections
- Supports running host and client on the same machine

### 3. Model Variable Storage
- **DRAM Models**: Actual model objects stored in memory
- **GPU Models**: Inference endpoint variables stored
- Efficient memory management and variable access

### 4. Improved HuggingFace Integration
- Uses HuggingFace Hub API for better downloads
- Automatic model discovery from existing HF cache
- Support for `hf://` URL format

### 5. Startup Model Discovery
- Automatically scans and registers existing models
- Supports both GSwarm and HuggingFace caches
- Configurable via `auto_discover_models` setting

## Configuration

### Sample Configuration File (`~/.gswarm.conf`)

```yaml
host:
  host: "0.0.0.0"
  port: 8100
  model_cache_dir: "~/.cache/gswarm/models"
  huggingface_cache_dir: "~/.cache/huggingface"
  auto_discover_models: true
  cleanup_on_shutdown: true
  
  storage_devices:
    disk:
      enabled: true
      path: "~/.cache/gswarm/models"
    dram:
      enabled: true
      path: "/dev/shm/gswarm_models"
      max_size_gb: 64

client:
  host_url: "http://localhost:8100"
  model_cache_dir: "~/.cache/gswarm/models"
  dram_cache_size: 16
  gpu_memory_fraction: 0.9
  default_gpu_memory_utilization: 0.90
  max_concurrent_requests: 256
```

## New API Endpoints

### Model Memory Management

- `GET /memory/models` - Get all models in memory (DRAM + GPU)
- `GET /memory/dram` - Get models loaded in DRAM with details
- `GET /memory/gpu` - Get GPU inference instances
- `DELETE /memory/dram/{model_name}` - Unload model from DRAM

### Configuration and Discovery

- `GET /config` - Get current configuration
- `POST /discover` - Manually trigger model discovery
- `GET /health` - Enhanced health check with memory usage

## Usage Examples

### 1. Download Model to Fixed Path

```bash
# Download using HF Hub API
curl -X POST "http://localhost:8100/download" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama2-7b",
    "source_url": "hf://meta-llama/Llama-2-7b-hf",
    "target_device": "disk"
  }'
```

### 2. Load Model to DRAM with Variables

```bash
# Copy from disk to DRAM (loads model variables)
curl -X POST "http://localhost:8100/copy" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama2-7b",
    "source_device": "disk",
    "target_device": "dram"
  }'
```

### 3. Check Memory Usage

```bash
# Get memory status
curl http://localhost:8100/memory/models

# Response includes:
# {
#   "dram_models": ["llama2-7b"],
#   "gpu_models": ["instance-123"],
#   "memory_usage": {
#     "dram_models": 1,
#     "gpu_models": 1,
#     "gpu_memory": {...}
#   }
# }
```

### 4. Serve Model from DRAM

```bash
# Start serving (uses DRAM variables for faster loading)
curl -X POST "http://localhost:8100/serve" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama2-7b",
    "source_device": "dram",
    "gpu_device": "gpu0",
    "port": 8080
  }'
```

### 5. Save Model Back to Disk

```bash
# Save DRAM model back to disk
curl -X POST "http://localhost:8100/copy" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama2-7b",
    "source_device": "dram",
    "target_device": "disk",
    "keep_source": false
  }'
```

## Model Variable Storage

### DRAM Models
- Models loaded to DRAM are stored as actual Python objects
- Supports HuggingFace transformers and safetensors
- Enables faster GPU transfers and preprocessing

### GPU Models
- Inference endpoints stored as variables
- Includes server references and configuration
- Allows direct access to inference objects

### Memory Management
- Automatic cleanup on shutdown (configurable)
- Manual unloading via API endpoints
- Memory usage tracking and reporting

## Directory Structure

```
~/.cache/gswarm/
├── models/                 # Fixed model storage
│   ├── llama2-7b/         # Model files
│   └── mistral-7b/        # Model files
└── profiler/              # Profiler data

/dev/shm/gswarm_models/    # DRAM cache
├── llama2-7b/             # Models loaded to DRAM
└── mistral-7b/            # Models loaded to DRAM

~/.cache/huggingface/      # Scanned for existing models
└── hub/
    └── models--*/         # HF cached models
```

## Migration from Previous Version

1. **Configuration**: The system will auto-create `~/.gswarm.conf` with defaults
2. **Model Discovery**: Existing models in HF cache will be auto-discovered
3. **Path Migration**: Models will be copied to the fixed path as needed
4. **API Compatibility**: All existing endpoints remain functional

## Benefits

1. **Consistent Storage**: Fixed path eliminates configuration complexity
2. **Faster Loading**: Model variables in DRAM enable rapid GPU transfers
3. **Better Discovery**: Automatic scanning of existing model caches
4. **Enhanced Monitoring**: Detailed memory usage and model state tracking
5. **Improved HF Integration**: Better download reliability and progress tracking 
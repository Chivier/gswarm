# GSwarm Standalone Model Server

A standalone HTTP server for managing and serving machine learning models, with support for LLM and Stable Diffusion models.

## Features

- **Download**: Download models from Hugging Face to local cache
- **Load**: Load models to DRAM for faster access
- **Serve**: Create serving instances on CPU or GPU devices
- **Call**: Send inference requests to model instances
- **Offload**: Move models between storage locations and stop serving

## Installation

1. **Install dependencies**:
```bash
cd src/gswarm-standalone-sheduler/standalone_model_server
pip install fastapi uvicorn torch transformers loguru pydantic
```

2. **Optional: Install diffusers for Stable Diffusion support**:
```bash
pip install diffusers accelerate
```

## Quick Start

### 1. Start the Server

```bash
python serve.py --host localhost --port 8000
```

The server will start on `http://localhost:8000`

### 2. Basic Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Download a Model
```bash
curl -X POST "http://localhost:8000/standalone/download/microsoft/DialoGPT-medium?source=huggingface"
```

#### Load Model to DRAM
```bash
curl -X POST "http://localhost:8000/standalone/load/microsoft/DialoGPT-medium?target=dram"
```

#### Serve Model
```bash
curl -X POST "http://localhost:8000/standalone/serve/microsoft/DialoGPT-medium?device=cpu"
```

Example response:
```json
{
  "success": true,
  "message": "Successfully started serving microsoft/DialoGPT-medium",
  "data": {
    "instance_id": "550e8400-e29b-41d4-a716-446655440000",
    "model_name": "microsoft/DialoGPT-medium",
    "device": "cpu",
    "endpoint": "/standalone/call/550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2024-01-15T10:30:45.123456"
  }
}
```

#### Call Model (Inference)
```bash
curl -X POST "http://localhost:8000/standalone/call/550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "550e8400-e29b-41d4-a716-446655440000",
    "data": {
      "prompt": "Hello, how are you?",
      "max_length": 50,
      "temperature": 0.7
    }
  }'
```

## API Reference

### Base URL
All endpoints are prefixed with the base URL: `http://localhost:8000`

### Endpoints

#### 1. Download Model
**POST** `/standalone/download/{model_name}`

Download a model from Hugging Face to local disk cache.

**Parameters:**
- `model_name` (path): Model name from Hugging Face (e.g., `microsoft/DialoGPT-medium`)
- `source` (query, optional): Source platform (`huggingface`, default)

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/download/gpt2?source=huggingface"
```

#### 2. Load Model to DRAM
**POST** `/standalone/load/{model_name}`

Load a model from disk to DRAM for faster serving.

**Parameters:**
- `model_name` (path): Model name
- `target` (query): Target location (`dram`)

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/load/gpt2?target=dram"
```

#### 3. Serve Model
**POST** `/standalone/serve/{model_name}`

Create a serving instance for a model on specified device.

**Parameters:**
- `model_name` (path): Model name
- `device` (query, optional): Device to serve on (`cpu`, `cuda:0`, `cuda:1`, etc.)

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/serve/gpt2?device=cuda:0"
```

#### 4. Offload Model
**POST** `/standalone/offload/{model_name}`

Stop serving instances and move model to specified storage.

**Parameters:**
- `model_name` (path): Model name
- `target` (query): Target storage (`disk` or `dram`)

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/offload/gpt2?target=disk"
```

#### 5. Call Model (Inference)
**POST** `/standalone/call/{instance_id}`

Send inference request to a serving model instance.

**Request Body:**
```json
{
  "instance_id": "string",
  "data": {
    "prompt": "string",
    "max_length": 100,
    "temperature": 0.7
  }
}
```

**For LLM models:**
- `prompt` (required): Input text
- `max_length` (optional): Maximum output length
- `temperature` (optional): Sampling temperature

**For Diffusion models:**
- `prompt` (required): Image description
- `steps` (optional): Number of inference steps
- `guidance_scale` (optional): Guidance scale

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/call/YOUR_INSTANCE_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "YOUR_INSTANCE_ID",
    "data": {
      "prompt": "The future of AI is",
      "max_length": 50
    }
  }'
```

#### 6. Get Status
**GET** `/standalone/status`

Get server status and model information.

**Example:**
```bash
curl http://localhost:8000/standalone/status
```

#### 7. Stop Instance
**DELETE** `/standalone/instance/{instance_id}`

Stop a specific serving instance.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/standalone/instance/YOUR_INSTANCE_ID"
```

## Supported Models

### Language Models (LLM)
- GPT-2, GPT-Neo, GPT-J
- LLaMA, Mistral, Gemma
- DialoGPT, BlenderBot
- Any Hugging Face transformers model

### Diffusion Models
- Stable Diffusion v1.4, v1.5, v2.0
- FLUX models
- Any diffusers-compatible model

## Configuration

### Model Cache Directory
Models are stored in: `~/.cache/gswarm/models/`

### Device Selection
- `cpu`: Use CPU for inference
- `cuda:0`, `cuda:1`, etc.: Use specific GPU
- Auto-detection: Leave device parameter empty for automatic selection

## Example Workflows

### Complete LLM Workflow
```bash
# 1. Download model
curl -X POST "http://localhost:8000/standalone/download/gpt2"

# 2. Load to DRAM
curl -X POST "http://localhost:8000/standalone/load/gpt2?target=dram"

# 3. Start serving
curl -X POST "http://localhost:8000/standalone/serve/gpt2?device=cuda:0"

# Response will contain instance_id, use it for inference
# 4. Inference
curl -X POST "http://localhost:8000/standalone/call/YOUR_INSTANCE_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "YOUR_INSTANCE_ID",
    "data": {
      "prompt": "Artificial intelligence will",
      "max_length": 100,
      "temperature": 0.8
    }
  }'

# 5. Offload when done
curl -X POST "http://localhost:8000/standalone/offload/gpt2?target=disk"
```

### Stable Diffusion Workflow
```bash
# 1. Download Stable Diffusion model
curl -X POST "http://localhost:8000/standalone/download/runwayml/stable-diffusion-v1-5"

# 2. Load to DRAM
curl -X POST "http://localhost:8000/standalone/load/runwayml/stable-diffusion-v1-5?target=dram"

# 3. Start serving
curl -X POST "http://localhost:8000/standalone/serve/runwayml/stable-diffusion-v1-5?device=cuda:0"

# 4. Generate image
curl -X POST "http://localhost:8000/standalone/call/YOUR_INSTANCE_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "YOUR_INSTANCE_ID",
    "data": {
      "prompt": "A beautiful sunset over mountains",
      "steps": 20,
      "guidance_scale": 7.5
    }
  }'
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Use smaller models or offload models not in use
   - Serve on CPU instead of GPU

2. **Model not found**
   - Ensure model is downloaded first
   - Check model name spelling

3. **Permission errors**
   - Check write permissions for `~/.cache/gswarm/models/`

### Logs
Server logs are saved to `standalone_server.log` in the current directory.

### Health Check
Monitor server health: `curl http://localhost:8000/health`

## Advanced Usage

### Multiple GPU Setup
```bash
# Serve same model on different GPUs
curl -X POST "http://localhost:8000/standalone/serve/gpt2?device=cuda:0"
curl -X POST "http://localhost:8000/standalone/serve/gpt2?device=cuda:1"
```

### Custom Server Configuration
```bash
# Start server on different host/port
python serve.py --host 0.0.0.0 --port 9000
```

## Integration

This standalone server can be integrated with:
- Load balancers for scaling
- Monitoring systems via health endpoints
- CI/CD pipelines for automated model deployment
- Other GSwarm components

## Support

For issues and questions, check the server logs and ensure all dependencies are installed correctly. 
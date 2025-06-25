# GSwarm Standalone Model Server

A standalone HTTP server for managing and serving machine learning models, with support for LLM and Stable Diffusion models.

## Features

- **Download**: Download models from Hugging Face to local cache
- **Load**: Load models to DRAM for faster access
- **Serve**: Create serving instances on CPU or GPU devices with automatic port allocation
- **Call**: Send inference requests to model instances
- **Offload**: Move models between storage locations and stop serving
- **Multi-Instance**: Support multiple instances of the same model on one device
- **Port Management**: Automatic random port allocation (10000-20000 range)

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
curl -X POST "http://localhost:8000/standalone/download" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "microsoft/DialoGPT-medium", "source": "huggingface"}'
```

#### Load Model to DRAM
```bash
curl -X POST "http://localhost:8000/standalone/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "microsoft/DialoGPT-medium", "target": "dram"}'
```

#### Serve Model
```bash
curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "microsoft/DialoGPT-medium", "device": "cpu"}'
```

Example response:
```json
{
  "success": true,
  "message": "Successfully started serving microsoft/DialoGPT-medium",
  "data": {
    "instance_id": "a3k9m2",
    "model_name": "microsoft/DialoGPT-medium",
    "device": "cpu",
    "port": 15432,
    "url": "http://localhost:15432",
    "endpoint": "/standalone/call/a3k9m2",
    "created_at": "2024-01-15T10:30:45.123456"
  }
}
```

#### Call Model (Inference)
```bash
curl -X POST "http://localhost:8000/standalone/call/a3k9m2" \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "a3k9m2",
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
**POST** `/standalone/download`

Download a model from Hugging Face to local disk cache.

**Request Body:**
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "source": "huggingface"
}
```

**Parameters:**
- `model_name` (string, required): Model name from Hugging Face
- `source` (string, optional): Source platform (`huggingface`, default)

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/download" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "source": "huggingface"}'
```

#### 2. Load Model to DRAM
**POST** `/standalone/load`

Load a model from disk to DRAM for faster serving.

**Request Body:**
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "target": "dram"
}
```

**Parameters:**
- `model_name` (string, required): Model name
- `target` (string, optional): Target location (`dram`, default)

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "target": "dram"}'
```

#### 3. Serve Model
**POST** `/standalone/serve`

Create a serving instance for a model on specified device. Each instance gets a unique 6-character ID and random port (10000-20000).

**Request Body:**
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "device": "cuda:0"
}
```

**Parameters:**
- `model_name` (string, required): Model name
- `device` (string, required): Device to serve on (`cpu`, `cuda:0`, `cuda:1`, etc.)

**Response includes:**
- `instance_id`: Short 6-character unique identifier (e.g., `a3k9m2`)
- `port`: Allocated random port (10000-20000 range)
- `url`: Full URL for the instance
- `endpoint`: API endpoint path for calling this instance

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "device": "cuda:0"}'
```

#### 4. Offload Model
**POST** `/standalone/offload`

Stop serving instances and move model to specified storage.

**Request Body:**
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "target": "disk"
}
```

**Parameters:**
- `model_name` (string, required): Model name
- `target` (string, required): Target storage (`disk` or `dram`)

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/offload" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "target": "disk"}'
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

**Response includes:**
- `port`: Instance port number for reference

**Example:**
```bash
curl -X POST "http://localhost:8000/standalone/call/a3k9m2" \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "a3k9m2",
    "data": {
      "prompt": "The future of AI is",
      "max_length": 50
    }
  }'
```

#### 6. Get Status
**GET** `/standalone/status`

Get server status and model information.

**Response includes:**
- List of serving instances with their ports and URLs
- Used ports tracking

**Example:**
```bash
curl http://localhost:8000/standalone/status
```

**Example Response:**
```json
{
  "models_on_disk": ["gpt2"],
  "models_in_dram": ["gpt2"],
  "serving_instances": [
    {
      "instance_id": "a3k9m2",
      "model_name": "gpt2",
      "device": "cuda:0",
      "port": 15432,
      "url": "http://localhost:15432",
      "created_at": "2024-01-15T10:30:45.123456"
    }
  ],
  "cache_directory": "/home/user/.cache/gswarm/models",
  "used_ports": [15432]
}
```

#### 7. Stop Instance
**DELETE** `/standalone/instance/{instance_id}`

Stop a specific serving instance and release its port.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/standalone/instance/a3k9m2"
```

## Multi-Instance Support

### Running Multiple Instances on Same GPU
You can now run multiple instances of the same model on a single GPU device:

```bash
# Create first instance
curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "device": "cuda:0"}'
# Returns: {"instance_id": "a3k9m2", "port": 15432, ...}

# Create second instance of same model on same GPU
curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "device": "cuda:0"}'
# Returns: {"instance_id": "x7z2p5", "port": 18901, ...}
```

Each instance gets:
- Unique 6-character ID
- Unique port (10000-20000 range)
- Shared model weights (efficient memory usage)
- Independent inference endpoints

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

### Port Management
- Automatic allocation: Ports are randomly selected from 10000-20000 range
- Conflict prevention: Duplicate ports are automatically avoided
- Cleanup: Ports are released when instances are stopped

## Example Workflows

### Complete LLM Workflow
```bash
# 1. Download model
curl -X POST "http://localhost:8000/standalone/download" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "microsoft/DialoGPT-medium", "source": "huggingface"}'

# 2. Load to DRAM
curl -X POST "http://localhost:8000/standalone/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "microsoft/DialoGPT-medium", "target": "dram"}'

# 3. Start serving
curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "microsoft/DialoGPT-medium", "device": "cuda:0"}'

# Response will contain instance_id and port, use them for inference
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
curl -X POST "http://localhost:8000/standalone/offload" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "microsoft/DialoGPT-medium", "target": "disk"}'
```

### Multi-Instance Deployment
```bash
# Load model to DRAM once
curl -X POST "http://localhost:8000/standalone/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "target": "dram"}'

# Create multiple serving instances
for i in {1..3}; do
  curl -X POST "http://localhost:8000/standalone/serve" \
    -H "Content-Type: application/json" \
    -d '{"model_name": "gpt2", "device": "cuda:0"}'
done

# Check status to see all instances
curl http://localhost:8000/standalone/status
```

### Stable Diffusion Workflow
```bash
# 1. Download Stable Diffusion model
curl -X POST "http://localhost:8000/standalone/download" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "runwayml/stable-diffusion-v1-5", "source": "huggingface"}'

# 2. Load to DRAM
curl -X POST "http://localhost:8000/standalone/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "runwayml/stable-diffusion-v1-5", "target": "dram"}'

# 3. Start serving
curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "runwayml/stable-diffusion-v1-5", "device": "cuda:0"}'

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
   - Stop unnecessary instances to free GPU memory

2. **Model not found**
   - Ensure model is downloaded first
   - Check model name spelling

3. **Permission errors**
   - Check write permissions for `~/.cache/gswarm/models/`

4. **Port conflicts**
   - The server automatically handles port allocation
   - If all ports (10000-20000) are occupied, stop unused instances

5. **Instance ID not found**
   - Check status endpoint to see active instances
   - Use the correct 6-character instance ID from serve response

### Logs
Server logs are saved to `standalone_server.log` in the current directory.

### Health Check
Monitor server health: `curl http://localhost:8000/health`

## Advanced Usage

### Multiple GPU Setup
```bash
# Serve same model on different GPUs
curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "device": "cuda:0"}'

curl -X POST "http://localhost:8000/standalone/serve" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "device": "cuda:1"}'
```

### Custom Server Configuration
```bash
# Start server on different host/port
python serve.py --host 0.0.0.0 --port 9000
```

### Load Balancing
Use the status endpoint to distribute requests across multiple instances:

```bash
# Get all active instances
curl http://localhost:8000/standalone/status

# Send requests to different instances based on availability
```

## Integration

This standalone server can be integrated with:
- Load balancers for scaling across multiple instances
- Monitoring systems via health endpoints  
- CI/CD pipelines for automated model deployment
- Other GSwarm components
- Container orchestration (Docker, Kubernetes)

## Support

For issues and questions, check the server logs and ensure all dependencies are installed correctly. 
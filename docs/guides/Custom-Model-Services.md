# Custom Model Services Guide

GSwarm provides a flexible framework for creating custom model services that can be managed through the unified model management system. This guide explains how to create, deploy, and manage custom model services.

## Overview

The custom model service framework allows you to:
- Create services with custom start, stop, and inference logic
- Automatic port allocation and management
- Unified CLI for service lifecycle management
- Support for both simple and batch inference
- Health check and monitoring endpoints

## Architecture

The framework consists of several key components:

1. **Base Service Classes** (`base_service.py`)
   - `ModelService`: Base class for all services
   - `BatchModelService`: Extended class with batch inference support
   - `ServiceConfig`: Configuration dataclass

2. **Service Manager** (`service_manager.py`)
   - Handles service registration and lifecycle
   - Manages port allocation
   - Tracks service state persistently

3. **Service Runner** (`service_runner.py`)
   - Executes services as separate processes
   - Dynamically loads service classes

4. **CLI Integration** (`cli.py`)
   - Commands for service management

## Creating a Custom Service

### 1. Basic Service Implementation

Here's a minimal example of a custom service:

```python
from gswarm.model.base_service import ModelService, ServiceConfig
import asyncio

class MyCustomService(ModelService):
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.model = None
    
    async def start(self) -> None:
        """Initialize your model here"""
        # Load model from self.config.model_path
        self.model = load_my_model(self.config.model_path)
        
    async def stop(self) -> None:
        """Clean up resources"""
        self.model = None
        
    async def inference(self, inputs: Any, parameters: Dict[str, Any]) -> Any:
        """Perform inference"""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Process inputs and return results
        result = self.model.predict(inputs)
        return {"prediction": result}
```

### 2. Service with Batch Support

For services that can handle batch inference efficiently:

```python
from gswarm.model.base_service import BatchModelService, ServiceConfig

class MyBatchService(BatchModelService):
    async def inference_batch(self, inputs_list: List[Any], parameters_list: List[Dict[str, Any]]) -> List[Any]:
        """Process multiple inputs in a single batch"""
        # Efficient batch processing
        results = self.model.batch_predict(inputs_list)
        return [{"prediction": r} for r in results]
```

## Service Configuration

Services are configured using the `ServiceConfig` dataclass:

```python
config = ServiceConfig(
    name="my-service",
    model_path="/path/to/model",
    host="0.0.0.0",
    port=8080,  # Optional, auto-assigned if None
    device="cuda:0",
    max_batch_size=32,
    timeout=30.0,
    extra_args={
        "custom_param": "value",
        "another_param": 123
    }
)
```

## Using the CLI

### Register a Service

```bash
# Using built-in service types
gswarm model service-register my-llm --type vllm --model /path/to/llama-7b --device cuda:0

# Using custom service from module
gswarm model service-register my-custom --type mymodule.MyService --model /path/to/model

# Using service from file
gswarm model service-register my-service --type /path/to/service.py:MyServiceClass --model /path/to/model

# With extra arguments
gswarm model service-register my-service --type simple --model dummy.pkl --args '{"feature_dim": 20}'
```

### Start a Service

```bash
gswarm model service-start my-service
```

### Check Service Status

```bash
# List all services
gswarm model service-list

# Get detailed status
gswarm model service-status my-service
```

### Stop a Service

```bash
# Graceful stop
gswarm model service-stop my-service

# Force stop
gswarm model service-stop my-service --force
```

### Unregister a Service

```bash
# Service must be stopped first
gswarm model service-unregister my-service
```

## API Endpoints

Every service automatically provides these endpoints:

### Health Check
```bash
GET /
```
Returns service health status.

### Inference
```bash
POST /inference
Content-Type: application/json

{
    "inputs": {...},
    "parameters": {...}
}
```

### Batch Inference (if supported)
```bash
POST /inference_batch
Content-Type: application/json

[
    {"inputs": {...}, "parameters": {...}},
    {"inputs": {...}, "parameters": {...}}
]
```

### Status
```bash
GET /status
```
Returns detailed service status.

### Stop
```bash
GET /stop
```
Gracefully stops the service.

## Example: vLLM Service

Here's a complete example of a vLLM service implementation:

```python
from gswarm.model.base_service import BatchModelService, ServiceConfig
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio

class VLLMService(BatchModelService):
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.engine = None
        
    async def start(self) -> None:
        engine_args = AsyncEngineArgs(
            model=self.config.model_path,
            dtype="auto",
            max_model_len=self.config.extra_args.get("max_model_len", 2048),
            device=self.config.device,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
    async def stop(self) -> None:
        self.engine = None
        
    async def inference(self, inputs: Any, parameters: Dict[str, Any]) -> Any:
        prompt = inputs if isinstance(inputs, str) else inputs["prompt"]
        
        sampling_params = SamplingParams(
            temperature=parameters.get("temperature", 0.7),
            max_tokens=parameters.get("max_tokens", 100),
            top_p=parameters.get("top_p", 0.9)
        )
        
        request_id = f"req-{asyncio.get_event_loop().time()}"
        results = []
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            results.append(output)
        
        final_output = results[-1]
        return {
            "text": final_output.outputs[0].text,
            "tokens": len(final_output.outputs[0].token_ids)
        }
```

## Advanced Features

### Custom Status Information

Override the `get_status` method to provide custom status information:

```python
async def get_status(self) -> Dict[str, Any]:
    status = await super().get_status()
    status.update({
        "model_loaded": self.model is not None,
        "model_size": self.get_model_size(),
        "requests_processed": self.request_count
    })
    return status
```

### Dynamic Port Allocation

If you don't specify a port, the service manager will automatically allocate one:

```bash
gswarm model service-register auto-port-service --type simple --model dummy.pkl
# Output: Service 'auto-port-service' registered on port 8237
```

### Service Discovery

Services can discover each other through the service manager:

```python
from gswarm.model.service_manager import get_service_manager

manager = get_service_manager()
services = manager.list_services()
for service in services:
    if service.status == "running":
        print(f"{service.name} is available at port {service.config.port}")
```

## Best Practices

1. **Resource Management**: Always clean up resources in the `stop()` method
2. **Error Handling**: Provide meaningful error messages in inference methods
3. **Logging**: Use loguru for consistent logging across services
4. **Health Checks**: Implement proper health check logic if needed
5. **Batch Processing**: Implement batch inference for better throughput
6. **Configuration**: Use `extra_args` for service-specific configuration

## Troubleshooting

### Service Won't Start
- Check if the port is already in use
- Verify the model path exists
- Check logs for initialization errors

### Service Crashes
- Check memory usage for large models
- Verify CUDA availability for GPU services
- Check service logs in `~/.gswarm/services/`

### Port Conflicts
- Use automatic port allocation
- Check `service-list` for used ports
- Manually specify a different port

## Integration with GSwarm

Custom services integrate seamlessly with the GSwarm ecosystem:
- Services are managed alongside traditional model deployments
- Can be used with the scheduler for task distribution
- Supports the same monitoring and logging infrastructure
- Compatible with the predictor for performance estimation
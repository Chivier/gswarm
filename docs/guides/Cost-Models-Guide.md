# GSwarm Cost Models Guide

This guide explains how to use the cost prediction models integrated into GSwarm for estimating model inference times across distributed GPU resources.

## Overview

GSwarm includes cost models for predicting inference times for different types of AI models:
- **LLM Cost Model**: Predicts token generation time for language models
- **SD Cost Model**: Predicts image generation time for Stable Diffusion models

These models help the scheduler make better decisions about task placement and timing in distributed environments.

## Device Notation Format

GSwarm uses a precise device notation format to identify specific GPUs across distributed nodes:

```
client:device_type:device_id
```

### Examples:
- `node1:cuda:0` - First CUDA GPU on node1
- `worker2:cuda:3` - Fourth CUDA GPU on worker2
- `localhost:cuda:0` - First CUDA GPU on localhost

### Legacy Format Support
For backward compatibility, the legacy format `cuda:0` is still supported and will be interpreted as `localhost:cuda:0`.

## Quick Start - Unified API

The simplest way to use cost prediction is through the unified API:

```python
from gswarm.model import CostModel

# Create a cost model (auto-detects model type)
cost_model = CostModel()

# Predict LLM inference time with new device format
time = cost_model.predict("gpt-4", "node1:cuda:0", {"prompt": "Hello world"})

# Predict diffusion model time on specific node
time = cost_model.predict("stable-diffusion", "node2:cuda:1", {"height": 512, "width": 512})

# Update with actual execution time
cost_model.update("gpt-4", "node1:cuda:0", {"prompt": "Hi", "output": "Hello"}, actual_time=0.5)
```

Or use the even simpler predictor interface:

```python
from gswarm.predictor import predictor

# Direct prediction with new device format
time = predictor.predict("llama-2-7b", "worker1:cuda:0", {"prompt": "Explain AI"})

# Batch prediction with mixed device formats
requests = [
    {"model_name": "gpt-4", "device": "node1:cuda:0", "inputs": {"prompt": "Q1"}},
    {"model_name": "stable-diffusion", "device": "node2:cuda:1", "inputs": {"height": 512, "width": 512}},
    {"model_name": "llama-2", "device": "cuda:2", "inputs": {"prompt": "Q2"}},  # Legacy format still works
]
times = predictor.batch_predict(requests)
```

## Using Cost Models

### Import Options

```python
# Option 1: Unified CostModel (Recommended)
from gswarm.model import CostModel

# Option 2: Simple Predictor
from gswarm.predictor import Predictor, predictor

# Option 3: Direct model classes (for advanced use)
from gswarm.model import LLMCostModel, SDCostModel, get_estimation_cost, update_predictor
```

### Unified CostModel API

The `CostModel` class provides a simple, unified interface:

```python
# Initialize with auto-detection
cost_model = CostModel()  # Auto-detects model type from name

# Or specify model type
cost_model = CostModel(model_type="llm")  # Force LLM type
cost_model = CostModel(model_type="diffusion")  # Force diffusion type

# Predict using new device format
time = cost_model.predict(
    model_name="gpt-4",
    device="compute1:cuda:0",  # Specific node and GPU
    inputs={"prompt": "What is AI?"}
)

# Update with real data
cost_model.update(
    model_name="gpt-4",
    device="compute1:cuda:0",
    inputs={"prompt": "What is AI?", "output": "AI is..."},
    actual_time=0.8
)
```

### LLM Cost Prediction

For language models, you can predict inference time based on prompt length:

```python
# Initialize the cost model
llm_model = LLMCostModel()

# Predict based on prompt length (number of tokens)
output_tokens = llm_model.predict(prompt_length=100)

# Or predict based on actual prompt text
output_tokens = llm_model.predict_str("What is the meaning of life?")
```

### Stable Diffusion Cost Prediction

For image generation models, prediction is based on image dimensions:

```python
# Initialize the cost model
sd_model = SDCostModel()

# Predict generation time
time_seconds = sd_model.predict(
    model_name="stable-diffusion-v1-5",
    device_name="cuda:0",
    height=512,
    width=512
)
```

### Simple Predictor Interface

The simplest way to use cost prediction:

```python
from gswarm.predictor import Predictor

# Create predictor
predictor = Predictor()

# Single prediction with new device format
time = predictor.predict("gpt-4", "node1:cuda:0", {"prompt": "Hello"})

# Batch prediction with distributed devices
times = predictor.batch_predict([
    {"model_name": "gpt-4", "device": "node1:cuda:0", "inputs": {"prompt": "Q1"}},
    {"model_name": "llama-2", "device": "node2:cuda:1", "inputs": {"prompt": "Q2"}},
])

# Update model with specific device
predictor.update("gpt-4", "node1:cuda:0", {"prompt": "Hi", "output": "Hello"}, actual_time=0.5)
```

### Using the Legacy Interface

The `get_estimation_cost` function provides backward compatibility:

```python
# For LLM models
llm_features = [{"prompt": "Hello, world!"}]
cost = get_estimation_cost(
    model_type="llm",
    model_name="gpt-4",
    device="cuda:0",
    data_features=llm_features
)

# For diffusion models
sd_features = [{"height": 1024, "width": 1024}]
cost = get_estimation_cost(
    model_type="diffusion",
    model_name="stable-diffusion-xl",
    device="cuda:0",
    data_features=sd_features
)
```

## Updating Models with Real Data

The cost models can be updated with actual execution times to improve accuracy:

```python
# Update LLM model with actual data
llm_features = [{
    "prompt": "Explain quantum computing",
    "output": "Quantum computing is... [actual output]"
}]
update_predictor(
    model_type="llm",
    model_name="gpt-4",
    device="cuda:0",
    data_features=llm_features
)

# Update diffusion model with actual timing
sd_features = [{
    "height": 512,
    "width": 512,
    "processing_time": 2.5  # actual seconds
}]
update_predictor(
    model_type="diffusion",
    model_name="stable-diffusion-v1-5",
    device="cuda:0",
    data_features=sd_features
)
```

## Model Storage

Trained models are automatically saved to disk for reuse:
- Default location: `~/.gswarm/`
- LLM model: `~/.gswarm/llm_cost_model.pkl`
- SD models: `~/.gswarm/{model_name}_{normalized_device}_predictor.pkl`

Note: Device names are normalized for storage. For example, both `cuda:0` and `localhost:cuda:0` will be stored as `localhost_cuda_0`.

## Integration with Predictor Module

The cost models are integrated with the predictor module for seamless usage:

```python
from gswarm.predictor import predict_inference, predictor

# Method 1: Using the global predictor with new device format
time = predictor.predict("stable-diffusion-v1-5", "worker1:cuda:0", {"height": 512, "width": 512})

# Method 2: Using predict_inference function
result = predict_inference(
    model_name="stable-diffusion-v1-5",
    device_name="worker2:cuda:1",  # Specific node and GPU
    inputs={"height": 512, "width": 512, "prompt": "A beautiful landscape"}
)
print(f"Estimated time: {result.prediction['inference_time']} seconds")

# Method 3: Creating your own predictor
from gswarm.predictor import Predictor
my_predictor = Predictor(model_type="diffusion")  # Force diffusion type
time = my_predictor.predict("custom-sd-model", "node3:cuda:2", {"height": 1024, "width": 1024})
```

## API Design Philosophy

The cost model API is designed with simplicity in mind:

1. **Unified Interface**: Single `CostModel` class handles all model types
2. **Auto-Detection**: Automatically detects model type from name
3. **Simple Parameters**: Just `model_name`, `device`, and `inputs`
4. **No Explicit Types**: No need to specify "llm" or "diffusion" in most cases

Example of the simplicity:
```python
# Old way (still supported)
cost = get_estimation_cost("llm", "gpt-4", "cuda:0", [{"prompt": "Hello"}])

# New way with device format (recommended)
cost = cost_model.predict("gpt-4", "node1:cuda:0", {"prompt": "Hello"})

# Even simpler with specific device
cost = predictor.predict("gpt-4", "worker1:cuda:0", {"prompt": "Hello"})
```

## Device Utilities

GSwarm provides utilities for working with device notations:

```python
from gswarm.utils import parse_device, format_device, normalize_device, is_same_device

# Parse device string
device_info = parse_device("node1:cuda:0")
print(f"Client: {device_info.client}")  # "node1"
print(f"Type: {device_info.device_type}")  # "cuda"
print(f"ID: {device_info.device_id}")  # 0

# Format device string
device_str = format_device("worker2", "cuda", 3)
print(device_str)  # "worker2:cuda:3"

# Normalize device (adds default client if missing)
normalized = normalize_device("cuda:0")
print(normalized)  # "localhost:cuda:0"

# Compare devices
same = is_same_device("cuda:0", "localhost:cuda:0")
print(same)  # True
```

## Best Practices

1. **Use the Unified API**: Prefer `CostModel` or `Predictor` over direct model classes
2. **Use New Device Format**: Specify `client:device:id` for clarity in distributed environments
3. **Initial Training**: The models work best when trained on your specific hardware and workload
4. **Regular Updates**: Update models with actual execution times for better accuracy
5. **Model-Specific Features**: Use appropriate features for each model type
6. **Device Specificity**: Models are device-specific - train separately for different GPUs
7. **Batch Operations**: Use batch predictions for multiple requests to improve efficiency

## Troubleshooting

### Model Not Found

If you get a "model not trained" error, ensure the model has been initialized:

```python
# Force model initialization
llm_model = LLMCostModel()
llm_model.initialize_model()
```

### Inaccurate Predictions

If predictions are inaccurate:
1. Collect more training data from actual executions
2. Update the model with real timing data
3. Check that you're using the correct device name

### Missing Dependencies

The LLM model requires `tiktoken` for accurate tokenization. Install it with:

```bash
pip install tiktoken
```

Without it, the model falls back to word-based approximation.

## Migration from Legacy Device Format

If you have existing code using the legacy format, you can migrate gradually:

```python
# Legacy code (still works)
time = predictor.predict("gpt-4", "cuda:0", inputs)

# New format (recommended for clarity)
time = predictor.predict("gpt-4", "localhost:cuda:0", inputs)

# Or specify your actual node
time = predictor.predict("gpt-4", "compute-node-1:cuda:0", inputs)
```

Key points for migration:
1. The system automatically handles both formats
2. Legacy `cuda:0` is interpreted as `localhost:cuda:0`
3. New format is recommended for distributed deployments
4. No code changes required - both formats work seamlessly
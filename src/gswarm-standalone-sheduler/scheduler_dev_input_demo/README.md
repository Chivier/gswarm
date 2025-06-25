# Scheduler Development Input Demo

This directory contains a data generator for creating synthetic workload data to test and develop scheduling algorithms.

## Overview

The `model_seq_gen.py` script generates realistic AI/ML workload data with:
- **Model definitions** with GPU requirements and performance characteristics
- **Workflow definitions** representing multi-node AI pipelines
- **Request sequences** with Poisson-distributed arrival times

## Generated Files

Running `python model_seq_gen.py` creates the following files:

### System Configuration Files
- `system_config.yaml` / `system_config.json` - Model and workflow definitions

### Request Files  
- `workflow_requests.yaml` / `workflow_requests.json` - Generated workflow requests

## Data Format

### Models
Each model definition includes:
```yaml
models:
  llm7b:
    name: "LLM-7B"
    memory_gb: 14
    gpus_required: 1
    load_time_seconds: 10
    tokens_per_second: 50
    token_mean: 512
    token_std: 128
```

**Available Models:**
- `llm7b` - 7B parameter language model (1 GPU)
- `llm30b` - 30B parameter language model (4 GPUs) 
- `stable_diffusion` - Image generation model (1 GPU)

### Workflows
Workflows define multi-node AI pipelines:
```yaml
workflows:
  workflow1:
    name: "LLM to Image Generation"
    nodes:
      - id: "node1"
        model: "llm7b"
        inputs: ["user_prompt"]
        outputs: ["image_prompt"]
    edges:
      - from: "node1"
        to: "node2"
```

**Available Workflows:**
- `workflow1` - LLM → Stable Diffusion pipeline
- `workflow2` - LLM fork-and-merge pattern

### Requests
Each request includes:
```yaml
requests:
  - request_id: "req_0001"
    timestamp: "2024-01-01T10:00:00.000000"
    workflow_id: "workflow1"
    input_data:
      user_prompt: "Sample prompt for request 1"
    node_execution_times:
      node1: 10.24  # Pre-generated execution time in seconds
    node_configs:
      node2:
        width: 512
        height: 768
```

## Key Features

- **Poisson Arrivals**: Requests arrive with realistic inter-arrival times
- **Pre-computed Execution Times**: Based on token distributions and model performance
- **Resource Requirements**: GPU memory and count requirements per model
- **Dynamic Configurations**: Node-specific runtime configurations

## Usage

1. **Generate Data:**
   ```bash
   python model_seq_gen.py
   ```

2. **Customize Generation:**
   - Modify `num_requests` and `duration_minutes` in `main()` function
   - Adjust model parameters in `MODELS` dictionary
   - Add new workflows in `WORKFLOWS` dictionary

## Output Statistics

The generator provides statistics including:
- Total requests generated
- Workflow distribution percentages  
- Estimated GPU-hours by model type

This data is designed for testing scheduling algorithms with realistic AI/ML workload characteristics. 
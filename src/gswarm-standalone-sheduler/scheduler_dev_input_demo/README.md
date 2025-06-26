# Scheduler Development Input Demo

This directory contains a data generator for creating synthetic workload data to test and develop scheduling algorithms.

## Overview

The `model_seq_gen.py` script generates realistic AI/ML workload data with:
- **Model definitions** with GPU requirements and performance characteristics
- **Workflow definitions** representing multi-node AI pipelines
- **Request sequences** with Poisson-distributed arrival times
- **Configurable input** via external JSON configuration files

## Prerequisites

The generator requires a **configuration file** that defines models and workflows. You need to provide this configuration file (JSON format) when running the generator.

## Generated Files

Running `python model_seq_gen.py --config <config_file>` creates the following files:

### Request Files  
- `<output_prefix>_requests.yaml` / `<output_prefix>_requests.json` - Generated workflow requests
- `<output_prefix>_stats.json` - Generation statistics and summary

## Configuration File Format

The generator loads model and workflow definitions from a JSON configuration file:

### Models
Each model definition supports multiple types and inheritance:
```json
{
  "models": {
    "llm7b": {
      "name": "LLM-7B",
      "type": "llm",
      "memory_gb": 14,
      "gpus_required": 1,
      "tokens_per_second": 50,
      "token_mean": 512,
      "token_std": 128
    },
    "stable_diffusion": {
      "name": "Stable Diffusion",
      "type": "image_generation",
      "memory_gb": 8,
      "gpus_required": 1,
      "inference_time_mean": 2.0,
      "inference_time_std": 0.5
    },
    "embedding_model": {
      "name": "Text Embedding",
      "type": "embedding", 
      "memory_gb": 2,
      "gpus_required": 1,
      "inference_time_mean": 0.5,
      "inference_time_std": 0.1
    }
  }
}
```

**Supported Model Types:**
- `llm` - Language models with token-based execution times
- `image_generation` - Image generation models with fixed inference times  
- `embedding` - Embedding models with fast, predictable inference

**Base Model Support:**
Models can inherit from base models using the `base_model` field for shared configurations.

### Workflows
Workflows define multi-node AI pipelines:
```json
{
  "workflows": {
    "workflow1": {
      "name": "LLM to Image Generation",
      "nodes": [
        {
          "id": "node1",
          "model": "llm7b",
          "inputs": ["user_prompt"],
          "outputs": ["image_prompt"]
        },
        {
          "id": "node2", 
          "model": "stable_diffusion",
          "inputs": ["image_prompt"],
          "outputs": ["generated_image"],
          "config_options": ["width", "height"]
        }
      ],
      "edges": [
        {
          "from": "node1",
          "to": "node2"
        }
      ]
    }
  }
}
```

### Generated Requests
Each request includes enhanced metadata:
```yaml
requests:
  - request_id: "req_0001"
    timestamp: "2024-01-01T10:00:00.000000"
    workflow_id: "workflow1"
    input_data:
      user_prompt: "Sample prompt for request 1"
    node_execution_times:
      node1: 10.24  # Pre-generated execution time in seconds
      node2: 2.15
    node_configs:
      node2:
        width: 512
        height: 768
    metadata:
      base_models_used: ["llm7b", "stable_diffusion"]
      total_estimated_time: 12.39
```

## Key Features

- **External Configuration**: Models and workflows defined in separate JSON files
- **Poisson Arrivals**: Requests arrive with realistic inter-arrival times
- **Multi-Model Types**: Support for LLM, image generation, and embedding models
- **Base Model Inheritance**: Models can inherit configurations from base models
- **Pre-computed Execution Times**: Based on model type and performance characteristics
- **Resource Requirements**: GPU memory and count requirements per model
- **Dynamic Configurations**: Node-specific runtime configurations
- **Enhanced Metadata**: Tracks base model usage and execution estimates

## Usage

### Basic Usage
```bash
python model_seq_gen.py --config test_baseline.json
```

### Advanced Usage with Parameters
```bash
python model_seq_gen.py \
  --config my_config.json \
  --num-requests 1000 \
  --duration 60 \
  --output-prefix my_test \
  --format yaml \
  --seed 42
```

### Command Line Arguments
- `--config`: Path to configuration JSON file (required)
- `--num-requests`: Number of requests to generate (default: 500)
- `--duration`: Duration in minutes over which requests arrive (default: 20)
- `--output-prefix`: Prefix for output files (default: "generated")
- `--format`: Output format - "json", "yaml", or "both" (default: "both")
- `--seed`: Random seed for reproducible results

### Customization
To customize the generation:
1. **Create/modify configuration file** with your models and workflows
2. **Adjust command line parameters** for request count and timing
3. **Set random seed** for reproducible test scenarios

## Output Statistics

The generator provides comprehensive statistics including:
- Total requests generated
- Workflow distribution percentages  
- Estimated GPU-hours by model type
- Base model usage counts
- Generation metadata and timestamps

Statistics are saved to `<output_prefix>_stats.json` and printed to console.

## Example Configuration Files

The generator works with any JSON configuration file that follows the schema above. Create your own configuration files to test different:
- Model performance characteristics
- Workflow complexity patterns  
- Resource requirement scenarios
- Mixed workload compositions

This data is designed for testing scheduling algorithms with realistic AI/ML workload characteristics. 
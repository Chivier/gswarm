# GSwarm Examples

This directory contains various examples demonstrating how to use GSwarm for different AI/ML workloads.

## Table of Contents

1. [OCR Processing Example](#ocr-processing-example)
2. [ComfyUI Workflows](#comfyui-workflows)
3. [Data API Examples](#data-api-examples)
4. [Model Management Examples](#model-management-examples)
5. [KV Storage Examples](#kv-storage-examples)
6. [Task Queue Examples](#task-queue-examples)

## OCR Processing Example

Located in `ocr-test/` directory.

Demonstrates how to configure and use PaddleOCR for text detection and recognition tasks.

### Key Files:
- `paddle_ocr_config.json`: Configuration file defining the OCR workflow with text detection and recognition models
- Models defined:
  - `paddle_ocr_v4_detector`: Text detection model
  - `paddle_ocr_v4_recognizer`: Text recognition model

### Workflow:
1. Text detection using PaddleOCR v4 detector
2. Text recognition using PaddleOCR v4 recognizer
3. Results combined to provide complete OCR output

## ComfyUI Workflows

Located in `comfyui-test/` directory.

Contains examples of converting ComfyUI workflows to GSwarm scheduler configurations.

### Key Files:
- `comfyui_to_config.py`: Converts ComfyUI workflow JSON files to GSwarm config format
- `comfyui_batch_converter.py`: Batch converts multiple ComfyUI workflows to individual config files
- `configs/`: Directory containing converted workflow configurations

### Supported Workflows:
1. CRM_Comfy_3D.json
2. CosXL_Edit_ArtGallery_1.0.json
3. FLUX.1_DEV_1.0.json
4. FLUX.1_SCHNELL_1.0.json
5. HUNYUAN_VIDEO_1.0.json
6. LivePortrait_Animals_1.0.json
7. SD3_BASE_1.0.json
8. SD3_Medium_Qwen2.json
9. SD3_Medium_肖像大师中文版.json
10. SD3是否内置文本编码器的对比.json
11. SDXS-512-0.9.json
12. Sketch_to_3D.json
13. Stable_Cascade_Canny_ControlNet.json
14. Stable_Cascade_ImagePrompt_Mix.json
15. Stable_Cascade_ImagePrompt_Standard.json
16. Stable_Cascade_Img2Img.json
17. Stable_Cascade_Inpainting_ControlNet.json

### Usage:
```bash
# Convert a single workflow
python comfyui_to_config.py --workflow path/to/workflow.json --output config.json

# Batch convert workflows
python comfyui_batch_converter.py --workflow-dir workflows/ --output-dir configs/
```

## Data API Examples

Located in `data_api_example.py`.

Demonstrates usage of GSwarm's data module API for both KV storage and data pool operations.

### Key Features Demonstrated:
1. KV storage operations (read/write with persistence options)
2. Complex data types (numpy arrays, nested structures)
3. Storage statistics and memory management
4. Data pool operations via HTTP API
5. Data chunk creation, movement, and transfer
6. Distributed operations

### Example Operations:
```python
# KV storage
client.write("user:123", {"name": "Alice", "role": "admin"}, persist=True)
user_data = client.read("user:123")

# Data pool via HTTP API
response = requests.post(f"{base_url}/api/v1/data", json=chunk_data)
response = requests.post(f"{base_url}/api/v1/data/{chunk_id}/move", json=move_data)
```

## Model Management Examples

Located in `model_management_example.py`.

Shows how to use the unified data API for complete model lifecycle management.

### Key Features Demonstrated:
1. Model registration with metadata
2. Model weight storage as data chunks
3. Model loading to specific devices
4. Session management for inference
5. Resource monitoring and statistics

### Example Usage:
```python
# Register a model
model_manager.register_model("llama-7b", model_info)

# Store model weights
chunk_id = model_manager.store_model_weights("llama-7b", "/models/llama-7b/pytorch_model.bin")

# Load model to GPU
model_manager.load_model_to_device(chunk_id, "gpu:0")
```

## KV Storage Examples

Located in `kv_storage_example.py`.

Demonstrates the unified KV storage system that combines both key-value storage and data pool functionality.

### Key Features Demonstrated:
1. Starting the unified data server
2. Writing and reading various data types
3. Persistence options for data
4. Storage statistics
5. Integration between KV storage and data pool

### Example Usage:
```python
# Write data with persistence
data_server.write("user:123", {"name": "Alice", "age": 30}, persist=True)

# Read data
user_data = data_server.read("user:123")

# Check storage stats
stats = data_server.get_stats()
```

## Task Queue Examples

Located in `queue_example.py`.

Demonstrates how to use the task queue manager with schedulers.

### Key Features Demonstrated:
1. Creating tasks with model information, inputs, and dependencies
2. Adding tasks to GPU-specific queues
3. Sorting tasks with custom priority functions
4. Retrieving tasks for execution

### Example Usage:
```python
# Create a task
task = Task(
    uuid=None,
    datetime=datetime.now(),
    model_name="model_a",
    input={"text": "Hello world"},
    tag="nlp_task",
    dependencies=set(),
    timeout=10.0
)

# Add to queue
manager.add_task("cuda:0", task)

# Sort with custom function
manager.sort_queue("cuda:0", priority_function)

# Get next task
next_task = manager.get_next_task("cuda:0")
```

## Benchmark Scenarios

### Scenario 1: OCR Model
- Dataset: Document images with text
- Baseline: Ray
- Evaluation: GPU utilization metrics

### Scenario 2: ComfyUI Test
- Dataset: Various image generation workflows
- Baseline: Ray, ComfyIO
- Evaluation: Workflow execution efficiency

### Scenario 3: PD Separation
- Dataset: LongBench v2
- Baseline: Ray, vLLM, SGLang
- Evaluation: Performance on long context processing

### Scenario 4: LLM Multi-Agent Workflow
- Dataset: Multi-agent conversation scenarios
- Baseline: Ray, Camel, LangChain
- Evaluation: Coordination and communication efficiency

Each scenario includes evaluations for Online, Offline, and Static scheduling approaches with GPU utilization metrics.
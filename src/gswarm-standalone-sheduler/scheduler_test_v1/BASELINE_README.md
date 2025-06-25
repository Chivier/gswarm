# GSwarm Baseline Scheduler

A baseline scheduler for GSwarm workflows that implements a Ray-like scheduling strategy. This scheduler processes workflow requests by maintaining a queue of ready-to-execute nodes and assigning them to available GPUs.

## Overview

The baseline scheduler provides a simple yet effective scheduling strategy for complex AI workflows involving multiple models. It supports both estimation-based scheduling (fast simulation) and actual model execution (realistic testing).

### Key Features

- **Ray-like Scheduling**: Maintains a queue of ready nodes and assigns them to available GPUs
- **Dependency Management**: Handles complex workflow dependencies between nodes
- **Two Execution Modes**:
  - **Estimation Mode**: Uses API estimates for fast simulation
  - **Simulation Mode**: Actually loads and calls models for realistic results
- **Two Scheduling Modes**:
  - **Offline Mode**: Batch processing of all requests
  - **Online Mode**: Streaming requests with realistic arrival times
- **GPU Resource Management**: Intelligent GPU allocation and model switching
- **Comprehensive Metrics**: Detailed performance analysis and logging

## Requirements

- Python 3.8+
- Access to the GSwarm standalone model server (running on `http://localhost:8000`)
- YAML and JSON support
- numpy for statistical calculations

### Python Dependencies

```bash
pip install pyyaml requests numpy
```

## Usage

### Basic Commands

```bash
# Estimation mode with offline scheduling (fastest)
python baseline.py --gpus 4 --simulate false --mode offline

# Simulation mode with actual model calls
python baseline.py --gpus 4 --simulate true --mode offline

# Online mode for latency analysis
python baseline.py --gpus 4 --simulate false --mode online

# Custom configuration files
python baseline.py --gpus 4 --simulate false --mode offline \
  --config custom_config.yaml --requests custom_requests.yaml
```

### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--gpus` | string | Yes | Number of GPUs (e.g., `4`) |
| `--simulate` | boolean | No | Use actual model calls (`true`) or estimates (`false`) |
| `--mode` | string | No | Scheduling mode: `offline` or `online` |
| `--config` | path | No | System configuration file (default: `system_config.yaml`) |
| `--requests` | path | No | Workflow requests file (default: `workflow_requests.yaml`) |

## Configuration Files

### System Configuration (`system_config.yaml`)

Defines models and workflows:

```yaml
models:
  llm7b:
    name: "Llama-7B"
    memory_gb: 14.0
    gpus_required: 1
    load_time_seconds: 30.0
    tokens_per_second: 50.0
  
  llm30b:
    name: "Llama-30B"
    memory_gb: 60.0
    gpus_required: 4
    load_time_seconds: 120.0
    tokens_per_second: 20.0

workflows:
  text_generation:
    name: "Text Generation Pipeline"
    nodes:
      - id: "generate"
        model: "llm7b"
        inputs: ["user_prompt"]
        outputs: ["generated_text"]
    edges: []
  
  multimodal_workflow:
    name: "Multimodal Processing"
    nodes:
      - id: "text_analysis"
        model: "llm7b"
        inputs: ["user_prompt"]
        outputs: ["analysis"]
      - id: "image_generation"
        model: "stable_diffusion"
        inputs: ["analysis"]
        outputs: ["image"]
    edges:
      - from: "text_analysis"
        to: "image_generation"
```

### Workflow Requests (`workflow_requests.yaml`)

Defines the requests to process:

```yaml
requests:
  - request_id: "req_001"
    timestamp: "2024-01-01T10:00:00"
    workflow_id: "text_generation"
    input_data:
      user_prompt: "Write a story about AI"
    node_configs:
      generate:
        max_length: 100
        temperature: 0.7
    node_execution_times:
      generate: 8.5
  
  - request_id: "req_002"
    timestamp: "2024-01-01T10:01:00"
    workflow_id: "multimodal_workflow"
    input_data:
      user_prompt: "Create an image of a futuristic city"
    node_configs:
      text_analysis:
        max_length: 50
      image_generation:
        steps: 20
        guidance_scale: 7.5
    node_execution_times:
      text_analysis: 5.2
      image_generation: 15.8
```

## How It Works

### 1. Initialization
- Loads system configuration (models and workflows)
- Loads workflow requests from file
- Initializes GPU state tracking

### 2. Request Processing
- Creates node executions for each workflow
- Identifies dependencies between nodes
- Adds ready nodes (no dependencies) to the execution queue

### 3. Scheduling Loop
- Continuously processes the ready node queue
- Finds available GPUs for each node's required model
- Handles model switching with calculated transfer costs
- Executes nodes and updates dependent nodes when complete

### 4. Model Management
- **Model Switching Cost**: Calculated based on model sizes and PCIe bandwidth
  ```
  switch_time = (old_model_size + new_model_size) / PCIe_bandwidth + overhead
  ```
- **GPU Allocation**: Prioritizes GPUs already loaded with the required model
- **Multi-GPU Support**: Handles models requiring multiple consecutive GPUs

## Output and Metrics

### Console Output
Real-time logging of:
- Request processing
- Node scheduling decisions
- Model switching operations
- Execution completions
- Final performance metrics

### Generated Files

1. **`baseline_scheduler.log`**: Detailed execution log
2. **`baseline_execution_log.json`**: Complete execution trace in JSON format

### Performance Metrics

The scheduler reports comprehensive metrics:

#### Summary Metrics
- **Total Execution Time**: Time from first node start to last node completion
- **Total Requests Processed**: Number of workflow requests completed
- **Total Nodes Executed**: Number of individual node executions

#### Latency Metrics (especially important for online mode)
- **Average Request Completion Time**
- **Median Request Completion Time**
- **P99 Latency**: 99th percentile completion time
- **Min/Max Request Times**

#### Resource Utilization
- **GPU Utilization**: Per-GPU execution counts and busy time
- **Model Execution Counts**: Number of times each model was used
- **Model Switch Statistics**: Frequency of model changes per GPU

### Example Output

```
2024-01-01 10:00:00 - Starting baseline scheduler with 5 requests
2024-01-01 10:00:00 - Mode: offline, Simulate: false
2024-01-01 10:00:00 - Available GPUs: [0, 1, 2, 3]
2024-01-01 10:00:01 - Scheduling node generate of request req_001 on GPU 2
2024-01-01 10:00:09 - Completed node generate of request req_001 on GPU 2 (execution time: 8.50s)

============================================================
EXECUTION METRICS
============================================================
Total execution time: 45.30 seconds

Request completion times:
  Average: 12.45s
  Median: 11.20s
  P99: 18.90s
  Min: 8.50s
  Max: 19.40s

GPU utilization:
  GPU 2: 3 executions, 25.40s busy time
  GPU 3: 2 executions, 18.60s busy time
  GPU 4: 1 executions, 15.80s busy time

Model execution counts:
  llm7b: 4 executions
  stable_diffusion: 2 executions
```

## Architecture

### Core Components

1. **BaselineScheduler**: Main scheduler class
2. **GPUState**: Tracks GPU allocation and current model
3. **NodeExecution**: Represents a single node execution
4. **Workflow/WorkflowNode**: Defines workflow structure
5. **Request**: Represents a workflow request

### Scheduling Strategy

The scheduler implements a simple but effective strategy:

1. **Ready Queue**: Maintains nodes ready for execution
2. **GPU Assignment**: Assigns nodes to available GPUs
3. **Model Affinity**: Prefers GPUs already loaded with required model
4. **Dependency Tracking**: Updates ready nodes when dependencies complete

### Model Switching Logic

```python
def _get_model_switch_time(self, from_model, to_model):
    if from_model == to_model:
        return 0.0
    
    # Calculate transfer time based on model sizes and PCIe bandwidth
    transfer_time = (from_size + to_size) / PCIE_BANDWIDTH_GB_S
    return transfer_time + overhead
```

## Use Cases

### 1. Performance Baseline
Use the scheduler to establish baseline performance metrics for your workflow configurations:

```bash
python baseline.py --gpus 3 --simulate false --mode offline
```

### 2. Latency Analysis
Analyze request completion times in streaming scenarios:

```bash
python baseline.py --gpus 3 --simulate false --mode online
```

### 3. Resource Planning
Understand GPU utilization patterns for capacity planning:

```bash
python baseline.py --gpus 3 --simulate true --mode offline
```

## How to check the execution log

- Overlap Detection: Checks that no two executions on the same GPU overlap in time
- Data Validation: Validates JSON structure and required fields
- Detailed Reporting: Shows conflicts with specific time ranges and durations

```bash
# Check the default file
python check.py

# Check a specific file
python check.py my_execution_log.json

# Save conflicts to a report file
python check.py --save-report

# Quiet mode (only show pass/fail)
python check.py --quiet
```
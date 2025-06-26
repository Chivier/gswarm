# GSwarm Baseline Scheduler

A baseline scheduler for GSwarm workflows that implements a Ray-like scheduling strategy using discrete event simulation. This scheduler processes workflow requests by maintaining a queue of ready-to-execute nodes and assigning them to available GPUs.

## Overview

The baseline scheduler provides a simple yet effective scheduling strategy for complex AI workflows involving multiple models. It uses a discrete event simulation approach to accurately model the execution timeline, including model switching costs and GPU availability.

### Key Features

- **Discrete Event Simulation**: Uses an event-driven architecture for accurate timing and scheduling
- **Ray-like Scheduling**: Maintains a queue of ready nodes and assigns them to available GPUs
- **Dependency Management**: Handles complex workflow dependencies between nodes
- **Two Execution Modes**:
  - **Estimation Mode**: Uses API estimates for fast simulation (default)
  - **Simulation Mode**: Actually loads and calls models for realistic results
- **Two Scheduling Modes**:
  - **Offline Mode**: Batch processing of all requests at time 0
  - **Online Mode**: Streaming requests with realistic arrival times
- **Multi-GPU Model Support**: Properly handles models requiring multiple GPUs
- **GPU Resource Management**: Intelligent GPU allocation with model switching cost calculation
- **Comprehensive Metrics**: Detailed performance analysis and logging with simulation timestamps

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
  --config custom_config.json --requests custom_requests.yaml
```

### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--gpus` | integer | Yes | Number of available GPUs (e.g., 4 for GPUs 0-3) |
| `--simulate` | boolean | No | Use actual model calls (`true`) or estimates (`false`, default) |
| `--mode` | string | No | Scheduling mode: `offline` (default) or `online` |
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
    # Optional fields for more accurate estimation:
    token_mean: 50.0
    token_std: 5.0
    inference_time_mean: 8.5
    inference_time_std: 1.2
  
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
        config_options: ["max_length", "temperature"]  # Optional
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
- Initializes GPU state tracking and event queue
- Downloads and loads models if in simulation mode

### 2. Event-Driven Simulation
The scheduler uses discrete event simulation with two main event types:
- **request_arrival**: A new workflow request arrives for processing
- **node_complete**: A node execution completes, potentially making other nodes ready

### 3. Request Processing
- Creates node executions for each workflow
- Identifies dependencies between nodes
- Adds ready nodes (no dependencies) to the execution queue
- Schedules initial request arrival events based on mode (offline: all at time 0, online: based on timestamps)

### 4. Scheduling Loop
The main simulation loop:
1. Pops the next event from the priority queue
2. Updates simulation time to the event timestamp
3. Processes the event:
   - For request arrivals: Creates node executions and adds ready nodes to queue
   - For node completions: Updates GPU states and checks for newly ready nodes
4. Attempts to schedule any ready nodes on available GPUs
5. Continues until all events are processed

### 5. GPU Allocation and Scheduling
- **Single-GPU models**: Finds the best available GPU (preferring same model)
- **Multi-GPU models**: Allocates consecutive GPUs when all are available
- Calculates actual start time based on GPU availability and model switching
- Creates completion events for scheduled nodes

### 6. Model Management
- **Model Switching Cost**: Calculated based on model sizes and PCIe bandwidth
  ```
  switch_time = (old_model_size + new_model_size) / PCIe_bandwidth + overhead
  ```
- **GPU Allocation**: Prioritizes GPUs already loaded with the required model
- **Multi-GPU Support**: Handles models requiring multiple consecutive GPUs with proper state tracking

## Output and Metrics

### Console Output
Real-time logging with simulation timestamps:
- Request processing events
- Node scheduling decisions
- Model switching operations
- Execution completions
- Final performance metrics

Example log entries:
```
[0.00s] Processing request req_001 (workflow: text_generation)
[0.00s] Scheduled node generate of request req_001 on GPU(s) [0] (start: 0.00s, end: 8.50s)
[8.50s] Completed node generate of request req_001 on GPU(s) [0]
[8.50s] Request req_001 completed
```

### Generated Files

1. **`baseline_scheduler.log`**: Detailed execution log with timestamps
2. **`baseline_execution_log.json`**: Complete execution trace in JSON format

### Performance Metrics

The scheduler reports comprehensive metrics:

#### Summary Metrics
- **Total Execution Time (Makespan)**: Maximum completion time across all nodes
- **Total Requests Processed**: Number of workflow requests completed
- **Total Nodes Executed**: Number of individual node executions
- **Simulation Mode**: Whether actual models were called

#### Latency Metrics (especially important for online mode)
- **Average Request Completion Time**
- **Median Request Completion Time**
- **P99 Latency**: 99th percentile completion time
- **Min/Max Request Times**

#### Resource Utilization
- **GPU Utilization**: 
  - Number of executions per GPU
  - Total busy time (including model switching)
  - Utilization percentage
- **Model Execution Counts**: Number of times each model was used
- **Model Switch Statistics**: Implicit in busy time calculations

### Example Output

```
2024-01-01 10:00:00 - Starting baseline scheduler with 5 requests
2024-01-01 10:00:00 - Mode: offline, Simulate: false
2024-01-01 10:00:00 - Available GPUs: [0, 1, 2, 3]
2024-01-01 10:00:00 - Starting discrete event simulation...
[0.00s] Processing request req_001 (workflow: text_generation)
[0.00s] Scheduled node generate of request req_001 on GPU(s) [0] (start: 0.00s, end: 8.50s)
[8.50s] Completed node generate of request req_001 on GPU(s) [0]
Simulation completed at time 45.30s

============================================================
EXECUTION METRICS
============================================================
Total execution time (makespan): 45.30 seconds

Request completion times:
  Average: 12.45s
  Median: 11.20s
  P99: 18.90s
  Min: 8.50s
  Max: 19.40s

GPU utilization:
  GPU 0: 3 executions, 25.40s busy time (56.1% utilization)
  GPU 1: 2 executions, 18.60s busy time (41.1% utilization)
  GPU 2: 1 executions, 15.80s busy time (34.9% utilization)

Model execution counts:
  llm7b: 4 executions
  stable_diffusion: 2 executions

Detailed execution log written to baseline_execution_log.json
```

## Architecture

### Core Components

1. **BaselineScheduler**: Main scheduler class with discrete event simulation logic
2. **Event**: Event dataclass for the simulation queue (timestamp, type, data)
3. **GPUState**: Tracks GPU allocation, current model, and availability time
4. **NodeExecution**: Represents a single node execution with timing information
5. **Workflow/WorkflowNode**: Defines workflow structure and dependencies
6. **Request**: Represents a workflow request with configurations

### Discrete Event Simulation

The scheduler uses a priority queue (heap) to manage events:

```python
@dataclass
class Event:
    timestamp: float
    event_type: str  # "node_complete", "request_arrival"
    data: Any
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp
```

### Scheduling Strategy

The scheduler implements a simple but effective strategy:

1. **Event Queue**: Manages all future events in chronological order
2. **Ready Queue**: Maintains nodes ready for execution
3. **GPU Assignment**: Assigns nodes to available GPUs with model affinity
4. **Dependency Tracking**: Updates ready nodes when dependencies complete
5. **Simulation Time**: Advances based on event timestamps

### GPU Allocation Logic

```python
def _find_available_gpu(self, model_name, current_time):
    # Single-GPU: Find best available GPU
    # Multi-GPU: Find consecutive available GPUs
    # Returns GPU IDs when allocation is possible
```

## Use Cases

### 1. Performance Baseline
Establish baseline performance metrics for your workflow configurations:

```bash
python baseline.py --gpus 4 --simulate false --mode offline
```

### 2. Latency Analysis
Analyze request completion times in streaming scenarios:

```bash
python baseline.py --gpus 4 --simulate false --mode online
```

### 3. Resource Planning
Understand GPU utilization patterns with actual model execution:

```bash
python baseline.py --gpus 4 --simulate true --mode offline
```

### 4. Scalability Testing
Test with different numbers of GPUs to understand scaling behavior:

```bash
# Test with 2, 4, and 8 GPUs
python baseline.py --gpus 2 --simulate false --mode offline
python baseline.py --gpus 4 --simulate false --mode offline
python baseline.py --gpus 8 --simulate false --mode offline
```

## Execution Log Analysis

The generated `baseline_execution_log.json` contains detailed execution information:

```json
{
  "summary": {
    "total_requests": 5,
    "total_nodes_executed": 12,
    "mode": "offline",
    "simulate": false,
    "gpus": [0, 1, 2, 3],
    "makespan": 45.3
  },
  "executions": [
    {
      "request_id": "req_001",
      "workflow_id": "text_generation",
      "node_id": "generate",
      "model_name": "llm7b",
      "gpu_id": 0,
      "start_time": 0.0,
      "end_time": 8.5,
      "execution_time": 8.5,
      "estimated_time": 8.5
    }
  ]
}
```

### Checking Execution Correctness

Use the provided check script to validate execution logs:

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

The checker validates:
- No overlapping executions on the same GPU
- Proper JSON structure and required fields
- Consistency between start/end times and execution times
# Multiple Concurrent Profiling Sessions

## Overview

gswarm-profiler supports running multiple profiling sessions concurrently. This allows you to:
- Monitor different intervals of your workload with unique names
- Overlap profiling sessions to capture different aspects of performance
- Maintain separate output files for each session

## API Design

### Session Management

Each profiling session is identified by a unique name and maintains its own:
- Output file (`<session_name>.json`)
- Frame counter
- Start/end timestamps
- Accumulated statistics

### Starting Sessions

```bash
# Start first session
gsprof http-profile localhost:8091 --name training_epoch_1

# Start overlapping session
gsprof http-profile localhost:8091 --name memory_intensive_phase

# Start third session
gsprof http-profile localhost:8091 --name optimization_step
```

### Stopping Sessions

```bash
# Stop specific session
gsprof http-stop localhost:8091 --name training_epoch_1

# Stop all active sessions
gsprof http-stop localhost:8091
```

## Use Cases

### 1. Overlapping Performance Analysis

Monitor different phases of your workload:

```bash
# Start monitoring entire training
gsprof http-profile localhost:8091 --name full_training

# Later, start monitoring specific epoch
gsprof http-profile localhost:8091 --name epoch_5

# Stop epoch monitoring but continue full training
gsprof http-stop localhost:8091 --name epoch_5

# Eventually stop full training
gsprof http-stop localhost:8091 --name full_training
```

### 2. A/B Performance Testing

Compare different configurations:

```bash
# Start baseline profiling
gsprof http-profile localhost:8091 --name baseline_config

# Run baseline workload...

# Start optimized profiling (can overlap)
gsprof http-profile localhost:8091 --name optimized_config

# Run optimized workload...

# Stop both
gsprof http-stop localhost:8091
```

### 3. Debugging Performance Issues

Focus on specific problematic regions:

```bash
# Normal profiling
gsprof http-profile localhost:8091 --name normal_operation

# When issue detected, start detailed profiling
gsprof http-profile localhost:8091 --name performance_issue_debug

# Stop debug session after issue
gsprof http-stop localhost:8091 --name performance_issue_debug
```

## HTTP API Endpoints

### Get All Sessions

```bash
curl http://localhost:8091/profiling/sessions
```

Response:
```json
{
  "total_sessions": 3,
  "active_sessions": 2,
  "sessions": [
    {
      "name": "training_epoch_1",
      "is_active": true,
      "start_time": "2024-01-15T10:30:00",
      "frame_count": 150,
      "output_file": "training_epoch_1.json"
    },
    {
      "name": "memory_test",
      "is_active": false,
      "start_time": "2024-01-15T10:25:00",
      "frame_count": 300,
      "output_file": "memory_test.json"
    }
  ]
}
```

### Status with Sessions

```bash
curl http://localhost:8091/status
```

The status endpoint now includes active session information.

## Implementation Details

### Session Isolation

- Each session maintains independent data collection
- Sessions share the same underlying metrics stream
- No interference between concurrent sessions

### Resource Efficiency

- Single metrics collection stream serves all sessions
- Minimal overhead for additional sessions
- Automatic cleanup of completed sessions

### Data Output

Each session produces a separate JSON file with:
- Session metadata (name, start/end time)
- Collected frames
- Summary statistics

## Best Practices

1. **Use Descriptive Names**: Choose session names that clearly indicate what's being profiled
2. **Manage Overlaps**: Be aware of overlapping sessions to avoid confusion
3. **Clean Up**: Stop sessions when done to free resources
4. **Organize Output**: Consider using directories for output files when running many sessions

## Example Workflow

```python
import requests
import time

API_URL = "http://localhost:8091"

# Start main profiling
requests.post(f"{API_URL}/profiling/start", json={"name": "main_workflow"})

# Start detailed profiling for specific phase
time.sleep(30)
requests.post(f"{API_URL}/profiling/start", json={"name": "data_loading_phase"})

# Run data loading...
time.sleep(60)

# Stop data loading profiling
requests.post(f"{API_URL}/profiling/stop", json={"name": "data_loading_phase"})

# Continue with training...
time.sleep(120)

# Start profiling for training phase
requests.post(f"{API_URL}/profiling/start", json={"name": "training_phase"})

# Run training...
time.sleep(300)

# Stop all profiling
requests.post(f"{API_URL}/profiling/stop", json={})
```

This will generate three files:
- `main_workflow.json`: Complete profiling data
- `data_loading_phase.json`: Focused on data loading
- `training_phase.json`: Focused on training 
# Adding Custom Metrics to the Profiler

The profiler allows you to extend its functionality by adding custom metrics. This guide explains how to implement your own profiling metrics.

## Prerequisites

- Familiarity with Python programming
- Understanding of the metric data structure used by the profiler

## Implementation Steps

### Step 1: Create Your Metric Function

Navigate to `src/gswarm/profiler/extra_metrics.py` and define a new function following this naming convention:

```python
def get_<your_metric_name>_metrics():
    # Your implementation here
    pass
```

### Step 2: Implement the Metric Logic

Write the logic to collect and process your desired metrics. Ensure your function:
- Performs the necessary measurements or calculations
- Handles any potential errors gracefully
- Returns data in the expected format

### Step 3: Return Metrics as Dictionary

Your function must return metrics in dictionary format:

```python
def get_<your_metric_name>_metrics():
    # Implementation logic
    metrics = {
        "metric_1": value_1,
        "metric_2": value_2,
        # Add more metrics as needed
    }
    return metrics
```

## Examples

For reference implementations and code examples, see the existing metric functions in `src/gswarm/profiler/extra_metrics.py`.

## Best Practices

- Use descriptive metric names
- Include appropriate error handling
- Document your metric functions with docstrings
- Test your metrics thoroughly before deployment
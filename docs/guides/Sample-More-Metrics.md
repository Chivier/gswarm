# Sampling Additional Metrics Beyond Default Configuration

The gswarm profiler supports extending its monitoring capabilities by incorporating additional GPU metrics beyond the standard collection. This feature enables comprehensive performance analysis and detailed system monitoring through configurable metric selection.

## Implementation

To specify additional metrics for sampling, utilize the `--extra-metrics` parameter when establishing a client connection:

```bash
gswarm client connect <head_ip:head_port> --resilient --extra-metrics "gpu_clock,sm_clock"
```

### Parameter Specification

- `--extra-metrics`: Accepts a comma-separated string of metric identifiers
- Multiple metrics can be specified simultaneously without spaces between entries
- Metric names must correspond to supported nvitop device properties

## Supported Metrics Reference

For a comprehensive list of available metrics and their specifications, consult the official nvitop documentation:
[nvitop Device API Reference](https://nvitop.readthedocs.io/en/latest/api/device.html#nvitop.Device)

### Common Additional Metrics

- `gpu_clock`: GPU core clock frequency
- `sm_clock`: Streaming Multiprocessor clock frequency
- `memory_clock`: Memory subsystem clock frequency
- `temperature`: Device temperature readings
- `power_draw`: Current power consumption
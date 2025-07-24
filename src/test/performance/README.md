# Performance Tests

This directory contains performance and stress tests for GSwarm components.

## Test Files

### `test_stress.py`
Comprehensive performance and stress tests.

```bash
python test_stress.py
```

## Test Categories

### 1. Predictor Throughput
Tests prediction performance under load:
- Single-threaded vs multi-threaded
- 1000+ concurrent predictions
- Mixed LLM and SD requests
- Throughput measurements

### 2. Scheduler Scalability
Tests scheduler performance with large workloads:
- 100+ requests
- Complex workflows (12+ nodes)
- Different scheduling strategies
- Tasks per second metrics

### 3. Device Scaling
Tests system performance with increasing devices:
- 4 to 64 devices
- Scaling efficiency
- Throughput vs device count
- Linear scaling analysis

### 4. Memory Efficiency
Tests memory usage patterns:
- 100+ predictor instances
- Memory per predictor
- Garbage collection efficiency
- Memory leak detection

### 5. Concurrent Updates
Tests thread safety and concurrent operations:
- 8 concurrent threads
- 800+ model updates
- Update rate measurements
- Model consistency verification

## Running Performance Tests

### Basic Run
```bash
python test_stress.py
```

### Individual Tests
```bash
# Test only predictor throughput
python -m unittest test_stress.TestPerformance.test_predictor_throughput

# Test only scheduler scalability
python -m unittest test_stress.TestPerformance.test_scheduler_scalability
```

### Performance Monitoring
```bash
# With memory profiling
python -m memory_profiler test_stress.py

# With CPU profiling
python -m cProfile -o profile.stats test_stress.py
python -m pstats profile.stats
```

## Performance Baselines

Expected performance on modern hardware:

### Predictor Throughput
- Single-threaded: ~500 predictions/second
- Multi-threaded (4): ~1500 predictions/second
- Speedup: 3-4x

### Scheduler Performance
- Baseline: ~1000 tasks/second
- Offline: ~2000 tasks/second
- Online: ~800 tasks/second

### Memory Usage
- Per predictor: ~0.5 MB
- 100 predictors: ~50 MB
- Release efficiency: >90%

## Stress Test Scenarios

### High Load Test
```python
# Simulate 10,000 requests
num_requests = 10000
devices = [f"node{i}:cuda:{j}" for i in range(10) for j in range(8)]
```

### Extended Duration Test
```python
# Run for 1 hour
duration = 3600  # seconds
start_time = time.time()
while time.time() - start_time < duration:
    # Continuous load...
```

### Resource Exhaustion Test
```python
# Test with limited resources
devices = ["node1:cuda:0"]  # Single GPU
requests = 1000  # Many requests
```

## Performance Tuning

### Optimization Tips
1. **Batch Operations**: Use batch predictions for better throughput
2. **Device Distribution**: Spread load across multiple devices
3. **Model Caching**: Reuse model instances when possible
4. **Connection Pooling**: For API tests, use connection pools

### Profiling Commands
```bash
# Line profiling
kernprof -l -v test_stress.py

# Memory profiling
mprof run test_stress.py
mprof plot

# CPU profiling visualization
python -m cProfile -o profile.stats test_stress.py
snakeviz profile.stats
```
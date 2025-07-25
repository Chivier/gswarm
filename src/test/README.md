# GSwarm Test Suite

This directory contains a comprehensive test suite for the GSwarm distributed GPU cluster management system, organized into specialized subdirectories for different types of testing.

## Test Organization

```
src/test/
├── basic/         # Basic functionality tests (host/client startup)
├── profiler/      # GSwarm profiler tests
├── model/         # Model download and serving tests
├── prediction/    # Model execution time prediction tests
├── data/          # Data handling and management tests
├── unit/          # Unit tests for individual components
├── gsmodel_data/  # Test data for gsmodel
├── gsmodel_test/  # Test configurations for gsmodel
└── scripts/       # Test automation scripts
    ├── llm/       # LLM-specific test scripts
    └── run_all_tests.sh
```

## Quick Start

### Run All New Tests

```bash
# Run the complete new test suite
cd src/test
./run_new_tests.sh
```

### Run Individual Test Categories

```bash
# 1. Basic functionality test (host/client startup)
python basic/test_basic_functionality.py

# 2. Profiler test
python profiler/test_profiler.py

# 3. Model serving test (requires GPU and may take time)
python model/test_model_serve.py

# 4. Prediction test
python prediction/test_prediction.py

# 5. Data handling test
python data/test_data_handling.py
```

### Run Unit Tests

```bash
# Run all unit tests
cd unit
python -m unittest discover
```

## Test Categories

### Basic Functionality (`basic/`)
Tests for core GSwarm operations:
- `test_basic_functionality.py` - Host and client startup/shutdown
- Verifies process management and graceful termination
- Tests basic connectivity between host and client

### Profiler Tests (`profiler/`)
GPU and system profiling functionality:
- `test_profiler.py` - Tests profiler read operations
- Validates profiler output formats (text/JSON)
- Checks GPU metrics collection

### Model Tests (`model/`)
Model management and serving:
- `test_model_serve.py` - Model download and serving
- Tests model listing and deployment
- Requires GPU for full functionality

### Prediction Tests (`prediction/`)
Model execution time prediction:
- `test_prediction.py` - Performance prediction for various configurations
- Tests prediction across different GPU types
- Validates batch size and token count impact

### Data Tests (`data/`)
Data management functionality:
- `test_data_handling.py` - Upload, download, and sync operations
- Tests dataset listing and retrieval
- Validates data synchronization between nodes

### Unit Tests (`unit/`)
Focused tests for individual components:
- `test_device_utils.py` - Device format parsing and utilities
- `test_cost_models.py` - Cost prediction models
- `test_schedulers.py` - Scheduler implementations

## Device Format Testing

The test suite validates the new device notation format throughout:

```python
# New format (recommended)
"node1:cuda:0"      # GPU 0 on node1
"worker2:cuda:3"    # GPU 3 on worker2

# Legacy format (supported)
"cuda:0"            # Interpreted as "localhost:cuda:0"
```

## Running All Tests

### Complete New Test Suite

```bash
# From src/test directory
./run_new_tests.sh
```

This will run all new tests in sequence:
1. Basic functionality tests
2. Profiler tests
3. Model serving tests
4. Prediction tests
5. Data handling tests

### Running Legacy Tests

```bash
# Unit tests only
cd unit && python -m unittest discover && cd ..

# All legacy tests (if available)
./scripts/run_all_tests.sh
```

### Test Coverage

```bash
# Install coverage tool
pip install coverage

# Run with coverage
coverage run -m unittest discover
coverage report
coverage html  # View in browser
```

## Multi-Node Testing

### Using New Tests

The new test suite automatically handles host/client startup for each test. To test in a multi-node environment:

```bash
# Set environment variables for remote host
export GSWARM_HOST_URL=http://host-ip:8095
export GSWARM_HTTP_PORT=8096
export GSWARM_MODEL_PORT=9010

# Run tests
./run_new_tests.sh
```

### Manual Multi-Node Setup

```bash
# Host machine
gswarm host start --port 8095 --http-port 8096 --model-port 9010

# GPU Node 1
gswarm client connect host-ip:8095 --resilient

# GPU Node 2
gswarm client connect host-ip:8095 --resilient
```

## Performance Benchmarks

Expected performance on modern hardware:

| Test | Metric | Expected |
|------|--------|----------|
| Predictor Throughput | Single-threaded | ~500/sec |
| Predictor Throughput | Multi-threaded (4) | ~1500/sec |
| Scheduler | Baseline | ~1000 tasks/sec |
| Scheduler | Offline | ~2000 tasks/sec |
| Memory | Per predictor | ~0.5 MB |

## Environment Variables

### New Test Suite
- `GSWARM_HOST_URL`: Host URL (default: http://localhost:8095)
- `GSWARM_HTTP_PORT`: HTTP API port (default: 8096)
- `GSWARM_MODEL_PORT`: Model serving port (default: 9010)
- `TEST_VERBOSE`: Enable verbose output

### Legacy Tests
- `SKIP_API_TESTS`: Skip API tests (default: true)
- `HOST_PORT`: Host HTTP port (default: 8080)
- `CLIENT_NAME`: Client identifier
- `GPU_COUNT`: Number of GPUs

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check if ports 8095/8096/9010 are in use
2. **GPU detection**: Ensure nvidia-smi is available (model tests may fail without GPU)
3. **Process cleanup**: Tests automatically clean up processes, but check for orphaned gswarm processes
4. **Model download**: Large model downloads may timeout on slow connections
5. **Network issues**: Verify connectivity between nodes for multi-node testing

### Debug Mode

```bash
# Enable debug logging for tests
export TEST_VERBOSE=1
./run_new_tests.sh

# Run individual test with verbose output
python -v basic/test_basic_functionality.py

# Check GSwarm logs (if using default paths)
tail -f ~/.gswarm/host/logs/host.log
tail -f ~/.gswarm/client/logs/client.log

# Monitor running processes
ps aux | grep gswarm
```

## Contributing

When adding new tests:

1. Place in appropriate subdirectory
2. Follow naming convention: `test_*.py`
3. Include docstrings and comments
4. Update relevant README
5. Ensure tests are idempotent
6. Add to CI/CD pipeline
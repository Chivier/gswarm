# GSwarm Test Suite

This directory contains a comprehensive test suite for the GSwarm distributed GPU cluster management system, organized into specialized subdirectories for different types of testing.

## Test Organization

```
src/test/
├── deployment/     # Infrastructure deployment scripts
├── unit/          # Unit tests for individual components
├── integration/   # Integration tests for component interactions
├── api/           # REST API endpoint tests
├── performance/   # Performance and stress tests
└── ...           # Other test resources
```

## Quick Start

### 1. Deploy Infrastructure

```bash
# Start host
cd deployment
./start_host.sh

# Start client(s)
./start_client.sh

# Monitor clients
./connect_client.sh monitor
```

### 2. Run Tests

```bash
# Run all unit tests
cd unit
python -m unittest discover

# Run integration tests
cd integration
python test_gswarm.py

# Run API tests (requires running host)
cd api
SKIP_API_TESTS=false python test_rest_api.py

# Run performance tests
cd performance
python test_stress.py
```

## Test Categories

### Deployment (`deployment/`)
Infrastructure deployment and management scripts:
- `start_host.sh` - Start GSwarm host service
- `start_client.sh` - Start and register clients
- `connect_client.sh` - Client connection management

### Unit Tests (`unit/`)
Focused tests for individual components:
- `test_device_utils.py` - Device format parsing and utilities
- `test_cost_models.py` - Cost prediction models
- `test_schedulers.py` - Scheduler implementations

### Integration Tests (`integration/`)
Tests for component interactions:
- `test_gswarm.py` - Comprehensive component testing
- `test_end_to_end.py` - Complete workflow simulations

### API Tests (`api/`)
REST API endpoint verification:
- `test_rest_api.py` - All API endpoints including device format

### Performance Tests (`performance/`)
Stress and scalability testing:
- `test_stress.py` - Throughput, scaling, and concurrency tests

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

### Complete Test Suite

```bash
# From src/test directory
./run_all_tests.sh  # (script to be created)

# Or manually:
cd unit && python -m unittest discover && cd ..
cd integration && python test_gswarm.py && cd ..
cd api && SKIP_API_TESTS=false python test_rest_api.py && cd ..
cd performance && python test_stress.py && cd ..
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

### Setup Multiple Nodes

```bash
# Host machine
cd deployment
./start_host.sh -d

# GPU Node 1
CLIENT_NAME=gpu-node-1 HOST_URL=http://host-ip:8080 ./start_client.sh -d

# GPU Node 2
CLIENT_NAME=gpu-node-2 HOST_URL=http://host-ip:8080 ./start_client.sh -d

# Monitor
HOST_URL=http://host-ip:8080 ./connect_client.sh monitor
```

### Run Distributed Tests

```bash
cd integration
GSWARM_HOST_URL=http://host-ip:8080 python test_end_to_end.py
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

### Common
- `GSWARM_HOST_URL`: Host URL (default: http://localhost:8080)
- `SKIP_API_TESTS`: Skip API tests (default: true)
- `TEST_VERBOSE`: Enable verbose output

### Deployment
- `HOST_PORT`: Host HTTP port (default: 8080)
- `CLIENT_NAME`: Client identifier
- `GPU_COUNT`: Number of GPUs

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check if ports 8080/8081 are in use
2. **GPU detection**: Ensure nvidia-smi is available
3. **Network issues**: Verify connectivity between nodes
4. **API timeouts**: Increase timeout values in tests

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG ./start_host.sh

# Verbose test output
python test_gswarm.py -v

# Check logs
tail -f ~/.gswarm/host/logs/host.log
tail -f ~/.gswarm/client/logs/client.log
```

## Contributing

When adding new tests:

1. Place in appropriate subdirectory
2. Follow naming convention: `test_*.py`
3. Include docstrings and comments
4. Update relevant README
5. Ensure tests are idempotent
6. Add to CI/CD pipeline
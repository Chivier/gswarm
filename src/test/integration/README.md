# Integration Tests

This directory contains integration tests that verify the interaction between multiple GSwarm components.

## Test Files

### `test_gswarm.py`
Comprehensive test script that tests all components together.

```bash
# Run all tests
python test_gswarm.py

# Run specific test
python test_gswarm.py --test device
python test_gswarm.py --test cost
python test_gswarm.py --test scheduler

# With custom host
python test_gswarm.py --host http://192.168.1.10:8080

# Verbose mode
python test_gswarm.py -v
```

Available tests:
- `device`: Device utilities
- `cost`: Cost models
- `predictor`: Predictor functionality
- `scheduler`: All schedulers
- `api`: Host API endpoints
- `e2e`: End-to-end workflow

### `test_end_to_end.py`
End-to-end integration tests for complete workflows.

```bash
# Run all integration tests
python test_end_to_end.py

# Skip API tests (default)
python test_end_to_end.py

# Include API tests
SKIP_API_TESTS=false python test_end_to_end.py
```

Tests include:
- Complete workflow simulation
- Cost prediction accuracy
- API workflow submission
- Scheduler comparison
- Multi-modal workflows

## Test Environment Setup

### Basic Setup
```bash
# Terminal 1: Start host
cd ../deployment
./start_host.sh

# Terminal 2: Start client
cd ../deployment
./start_client.sh

# Terminal 3: Run tests
cd ../integration
python test_gswarm.py
```

### Multi-Client Setup
```bash
# Start multiple clients
for i in {1..4}; do
    CLIENT_NAME=client-$i CLIENT_PORT=$((8080+$i)) ../deployment/start_client.sh -d
done

# Run integration tests
python test_end_to_end.py
```

## Test Output

Tests generate detailed reports:
- Colored console output
- Pass/fail status
- Performance metrics
- JSON report: `test_report_YYYYMMDD_HHMMSS.json`

## Environment Variables

- `GSWARM_HOST_URL`: Host URL (default: http://localhost:8080)
- `SKIP_API_TESTS`: Skip API tests (default: true)
- `TEST_VERBOSE`: Enable verbose output
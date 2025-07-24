# API Tests

This directory contains tests for GSwarm REST API endpoints.

## Test Files

### `test_rest_api.py`
Comprehensive REST API endpoint tests.

```bash
# API tests are skipped by default
python test_rest_api.py

# Run API tests (requires running host)
SKIP_API_TESTS=false python test_rest_api.py

# With custom host URL
SKIP_API_TESTS=false GSWARM_HOST_URL=http://192.168.1.10:8080 python test_rest_api.py
```

## API Endpoints Tested

### Core Endpoints
- `/health` - Health check
- `/api/v1/clients` - Client management
- `/api/v1/workflows` - Workflow submission
- `/api/v1/models` - Model management
- `/api/v1/scheduler` - Scheduler info

### Client Operations
- `POST /api/v1/clients/register` - Register new client
- `GET /api/v1/clients` - List all clients
- `GET /api/v1/clients/{id}` - Get client info
- `POST /api/v1/clients/{id}/reconnect` - Reconnect client
- `POST /api/v1/clients/{id}/disconnect` - Disconnect client

### Workflow Operations
- `POST /api/v1/workflows` - Submit workflow
- `GET /api/v1/workflows/{id}` - Get workflow status
- `POST /api/v1/workflows/batch` - Batch submission
- `DELETE /api/v1/workflows/{id}` - Cancel workflow

### Model Operations
- `GET /api/v1/models` - List models
- `POST /api/v1/models` - Register model
- `GET /api/v1/models/{id}` - Get model info
- `POST /api/v1/predict/cost` - Cost prediction

## Test Features

### Device Format Testing
Tests verify proper handling of new device format:
- Client registration with `node:cuda:id` format
- Device listing and info
- Workflow submission with device preferences

### Error Handling
- 404 for non-existent resources
- 400 for invalid requests
- Proper error messages

### Batch Operations
- Batch workflow submission
- Batch cost predictions
- Concurrent request handling

## Running Tests

### Prerequisites
1. Start GSwarm host:
   ```bash
   cd ../deployment
   ./start_host.sh
   ```

2. Optionally start clients:
   ```bash
   ./start_client.sh
   ```

### Run Tests
```bash
cd ../api
SKIP_API_TESTS=false python test_rest_api.py
```

### Test Specific Endpoints
```bash
# Test only client endpoints
SKIP_API_TESTS=false python -m unittest test_rest_api.TestGSwarmAPI.test_client_registration

# Test only workflow endpoints
SKIP_API_TESTS=false python -m unittest test_rest_api.TestGSwarmAPI.test_workflow_submission
```

## Mock API Server

For testing without a real host, you can use a mock server:

```python
# Example mock server setup
from unittest.mock import Mock, patch

@patch('requests.get')
@patch('requests.post')
def test_with_mock(mock_post, mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"status": "healthy"}
    # Run tests...
```
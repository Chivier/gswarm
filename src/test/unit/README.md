# Unit Tests

This directory contains unit tests for individual GSwarm components.

## Test Files

### `test_device_utils.py`
Tests device parsing, formatting, and normalization utilities.

```bash
python test_device_utils.py
```

Tests include:
- Parsing new device format (`node1:cuda:0`)
- Legacy format compatibility (`cuda:0`)
- Device formatting and normalization
- Device comparison functions
- Edge cases and error handling

### `test_cost_models.py`
Tests cost prediction models for LLM and Stable Diffusion.

```bash
python test_cost_models.py
```

Tests include:
- LLM cost model predictions
- Stable Diffusion cost model predictions
- Unified CostModel interface
- Model type auto-detection
- Model updates and persistence
- Legacy API compatibility

### `test_schedulers.py`
Tests all scheduler implementations.

```bash
python test_schedulers.py
```

Tests include:
- Baseline scheduler (Ray-like)
- Offline scheduler (batch optimization)
- Online scheduler (P99 optimization)
- Static scheduler (fixed deployment)
- Workflow dependency handling
- Priority scheduling
- Device format compatibility

## Running All Unit Tests

```bash
# Run all unit tests
python -m unittest discover -s . -p "test_*.py"

# Run with verbose output
python -m unittest discover -s . -p "test_*.py" -v

# Run specific test class
python -m unittest test_device_utils.TestDeviceUtils

# Run specific test method
python -m unittest test_device_utils.TestDeviceUtils.test_parse_new_format
```

## Test Coverage

To measure test coverage:

```bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run -m unittest discover -s . -p "test_*.py"

# Generate report
coverage report
coverage html  # Creates htmlcov/index.html
```
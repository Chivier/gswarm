# LLM Serving Tests

This directory contains test scripts for vLLM-based LLM serving in the gswarm system.

## Files

- `test_llm_serving.sh` - Main test script for LLM inference
- `serve_and_test.sh` - Complete workflow script (serve + test + cleanup)

## Prerequisites

1. **Install vLLM**:
   ```bash
   pip install vllm
   ```

2. **Have a model downloaded**:
   ```bash
   gswarm model download llama-7b --source hf://meta-llama/Llama-3.1-8B-Instruct --type llm
   ```

3. **Install dependencies**:
   ```bash
   sudo apt-get install jq curl  # On Ubuntu/Debian
   ```

## Usage

### Quick Test (All-in-one)
```bash
# Serve llama-7b and run tests
./serve_and_test.sh llama-7b gpu0 8080
```

### Manual Testing
```bash
# 1. Serve the model
gswarm model serve llama-7b --device gpu0 --port 8080

# 2. Run tests
./test_llm_serving.sh http://localhost 8080 9010

# 3. Cleanup
gswarm model stop llama-7b --device gpu0
```

### Test Different Models
```bash
./serve_and_test.sh my-model gpu1 8081
```

## What the Tests Do

1. **Management API Tests**:
   - Health check
   - List models
   - List running servers

2. **LLM Inference Tests**:
   - List available models from vLLM
   - Text completion
   - Chat completion (if supported)

3. **Performance Tests**:
   - Multiple requests with timing
   - Throughput measurement

4. **GPU Monitoring**:
   - GPU utilization before/during serving
   - Memory usage
   - Running processes

## Example Output

```bash
[INFO] Starting serve and test for model: llama-7b
[SUCCESS] Model llama-7b found
[INFO] Model status:
Model: llama-7b
Type: llm
Status: ready
✅ 
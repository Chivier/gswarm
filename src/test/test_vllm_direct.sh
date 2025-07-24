#!/bin/bash

# Direct vLLM test script without gswarm API dependency
# Usage: ./test_vllm_direct.sh [model_path]

set -e

# Configuration
MODEL_PATH=${1:-"/home/hyq/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"}
PORT=${2:-"8080"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if vLLM is installed
check_vllm() {
    log_info "Checking vLLM installation..."
    
    if python -c "import vllm" 2>/dev/null; then
        log_success "vLLM is installed"
        return 0
    else
        log_error "vLLM not found. Please install it with: pip install vllm"
        return 1
    fi
}

# Start vLLM server
start_vllm_server() {
    log_info "Starting vLLM server with model: $MODEL_PATH"
    log_info "Port: $PORT"
    
    # Kill any existing process on the port
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_info "Killing existing process on port $PORT..."
        kill $(lsof -Pi :$PORT -sTCP:LISTEN -t) 2>/dev/null || true
        sleep 2
    fi
    
    # Ensure we're using the correct Python
    if [ -f ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python"
    fi
    
    # Start vLLM server in background
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $PORT \
        --dtype auto \
        --max-model-len 2048 \
        > vllm_server.log 2>&1 &
    
    SERVER_PID=$!
    log_info "vLLM server started with PID: $SERVER_PID"
    
    # Save PID for cleanup
    echo $SERVER_PID > vllm_server.pid
    
    # Wait for server to be ready
    log_info "Waiting for server to initialize (this may take a minute)..."
    sleep 30
    
    # Check if server is running
    if ! ps -p $SERVER_PID > /dev/null; then
        log_error "vLLM server failed to start. Check vllm_server.log for details"
        tail -20 vllm_server.log
        return 1
    fi
    
    return 0
}

# Test inference
test_inference() {
    log_info "Testing inference..."
    
    # Test 1: Check models endpoint
    log_info "Checking models endpoint..."
    models=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null)
    if [ $? -eq 0 ]; then
        log_success "Models endpoint accessible"
        echo "$models" | jq '.' 2>/dev/null || echo "$models"
    else
        log_error "Cannot reach models endpoint"
        return 1
    fi
    
    # Test 2: Simple completion
    log_info "Testing text completion..."
    completion_request='{
        "model": "'$MODEL_PATH'",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.7
    }'
    
    response=$(curl -s -X POST "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$completion_request" 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$response" | jq -e '.choices[0].text' >/dev/null 2>&1; then
        log_success "Text completion successful!"
        echo "Response: $(echo "$response" | jq -r '.choices[0].text')"
    else
        log_error "Text completion failed"
        echo "Response: $response"
        return 1
    fi
    
    return 0
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    if [ -f vllm_server.pid ]; then
        PID=$(cat vllm_server.pid)
        if ps -p $PID > /dev/null 2>&1; then
            log_info "Stopping vLLM server (PID: $PID)..."
            kill $PID 2>/dev/null || true
        fi
        rm -f vllm_server.pid
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting direct vLLM test..."
    
    # Activate virtual environment if needed
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Check vLLM installation
    if ! check_vllm; then
        exit 1
    fi
    
    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model path not found: $MODEL_PATH"
        log_info "Please download the model first or specify a valid path"
        exit 1
    fi
    
    # Start server
    if ! start_vllm_server; then
        exit 1
    fi
    
    # Run tests
    if test_inference; then
        log_success "All tests passed!"
    else
        log_error "Some tests failed"
        exit 1
    fi
}

# Run main
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
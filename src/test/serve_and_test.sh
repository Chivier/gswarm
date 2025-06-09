#!/bin/bash

# Complete test script that serves a model and tests it
# Usage: ./serve_and_test.sh [model_name]

set -e

MODEL_NAME=${1:-"llama-7b"}
DEVICE=${2:-"gpu0"}
PORT=${3:-"8080"}

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

# Check if model exists
check_model() {
    log_info "Checking if model $MODEL_NAME exists..."
    
    if gswarm model list | grep -q "$MODEL_NAME"; then
        log_success "Model $MODEL_NAME found"
        return 0
    else
        log_error "Model $MODEL_NAME not found. Please download it first:"
        echo "  gswarm model download $MODEL_NAME --source hf://meta-llama/Llama-3.1-8B-Instruct --type llm"
        return 1
    fi
}

# Serve the model
serve_model() {
    log_info "Serving model $MODEL_NAME on device $DEVICE port $PORT..."
    
    if gswarm model serve "$MODEL_NAME" --device "$DEVICE" --port "$PORT"; then
        log_success "Model serving command executed"
        
        # Wait a bit for server to start
        log_info "Waiting for vLLM server to initialize..."
        sleep 10
        
        return 0
    else
        log_error "Failed to serve model"
        return 1
    fi
}

# Stop the model (cleanup)
stop_model() {
    log_info "Stopping model $MODEL_NAME on device $DEVICE..."
    gswarm model stop "$MODEL_NAME" --device "$DEVICE" || true
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    stop_model
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting serve and test for model: $MODEL_NAME"
    
    # Check if model exists
    if ! check_model; then
        exit 1
    fi
    
    # Show model status
    log_info "Model status:"
    gswarm model status "$MODEL_NAME"
    
    # Serve the model
    if ! serve_model; then
        exit 1
    fi
    
    # Show servers
    log_info "Active servers:"
    curl -s http://localhost:9010/servers | jq . || echo "Could not get server list"
    
    # Run tests
    log_info "Running inference tests..."
    ./test_llm_serving.sh http://localhost "$PORT" 9010
    
    log_success "Test completed successfully!"
}

# Check dependencies
check_deps() {
    for cmd in gswarm curl jq; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
}

# Run main
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_deps
    main "$@"
fi 
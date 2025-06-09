#!/bin/bash

# Test script for LLM serving with vLLM
# Usage: ./test_llm_serving.sh [base_url] [port]

set -e

# Configuration
BASE_URL=${1:-"http://localhost"}
MODEL_PORT=${2:-"8080"}
MANAGEMENT_PORT=${3:-"9010"}

MODEL_API_URL="${BASE_URL}:${MANAGEMENT_PORT}"
LLM_API_URL="${BASE_URL}:${MODEL_PORT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to wait for server to be ready
wait_for_server() {
    local url=$1
    local timeout=${2:-60}
    local interval=2
    local elapsed=0
    
    log_info "Waiting for server at $url to be ready..."
    
    while [ $elapsed -lt $timeout ]; do
        if curl -s "$url/health" >/dev/null 2>&1 || curl -s "$url/v1/models" >/dev/null 2>&1; then
            log_success "Server is ready!"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        echo -n "."
    done
    
    log_error "Server did not become ready within $timeout seconds"
    return 1
}

# Function to test management API
test_management_api() {
    log_info "Testing management API at $MODEL_API_URL"
    
    # Test health
    log_info "Checking management API health..."
    response=$(curl -s "$MODEL_API_URL/health")
    echo "Health response: $response"
    
    # List models
    log_info "Listing models..."
    models=$(curl -s "$MODEL_API_URL/models" | jq -r '.models[] | .name' 2>/dev/null || echo "")
    if [ -n "$models" ]; then
        log_success "Found models:"
        echo "$models"
    else
        log_warning "No models found"
    fi
    
    # List servers
    log_info "Listing running servers..."
    servers=$(curl -s "$MODEL_API_URL/servers" 2>/dev/null)
    echo "Servers: $servers"
}

# Function to test LLM inference
test_llm_inference() {
    local model_name=${1:-"llama-7b"}
    
    log_info "Testing LLM inference for model: $model_name"
    
    # Test 1: List available models
    log_info "Getting available models from vLLM server..."
    models_response=$(curl -s "$LLM_API_URL/v1/models" 2>/dev/null)
    if [ $? -eq 0 ]; then
        log_success "Models endpoint accessible"
        echo "Models: $models_response"
    else
        log_error "Could not reach models endpoint"
        return 1
    fi
    
    # Test 2: Simple completion
    log_info "Testing text completion..."
    completion_request='{
        "model": "'$model_name'",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }'
    
    completion_response=$(curl -s -X POST "$LLM_API_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$completion_request" 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$completion_response" | jq -e '.choices[0].text' >/dev/null 2>&1; then
        log_success "Text completion successful!"
        echo "Response: $(echo "$completion_response" | jq -r '.choices[0].text')"
    else
        log_warning "Text completion failed or returned unexpected format"
        echo "Response: $completion_response"
    fi
    
    # Test 3: Chat completion (if supported)
    log_info "Testing chat completion..."
    chat_request='{
        "model": "'$model_name'",
        "messages": [
            {"role": "user", "content": "Hello! How are you?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
    
    chat_response=$(curl -s -X POST "$LLM_API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$chat_request" 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$chat_response" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
        log_success "Chat completion successful!"
        echo "Response: $(echo "$chat_response" | jq -r '.choices[0].message.content')"
    else
        log_warning "Chat completion failed (may not be supported by this model)"
        echo "Response: $chat_response"
    fi
}

# Function to performance test
performance_test() {
    local model_name=${1:-"llama-7b"}
    local num_requests=${2:-5}
    
    log_info "Running performance test with $num_requests requests..."
    
    start_time=$(date +%s)
    
    for i in $(seq 1 $num_requests); do
        log_info "Request $i/$num_requests"
        
        request='{
            "model": "'$model_name'",
            "prompt": "Generate a short story about",
            "max_tokens": 50,
            "temperature": 0.7
        }'
        
        response=$(curl -s -X POST "$LLM_API_URL/v1/completions" \
            -H "Content-Type: application/json" \
            -d "$request" 2>/dev/null)
        
        if echo "$response" | jq -e '.choices[0].text' >/dev/null 2>&1; then
            echo "✓ Request $i completed"
        else
            echo "✗ Request $i failed"
        fi
    done
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    log_success "Performance test completed in ${duration}s"
    log_info "Average time per request: $((duration * 1000 / num_requests))ms"
}

# Function to monitor GPU usage
monitor_gpu() {
    log_info "GPU monitoring (requires nvidia-smi)..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "GPU utilization:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        
        log_info "GPU processes:"
        nvidia-smi --query-compute-apps=pid,process_name,gpu_name,used_memory --format=csv,noheader
    else
        log_warning "nvidia-smi not available, skipping GPU monitoring"
    fi
}

# Main test execution
main() {
    log_info "Starting LLM serving tests..."
    log_info "Management API: $MODEL_API_URL"
    log_info "LLM API: $LLM_API_URL"
    
    echo "============================================"
    
    # Test management API
    test_management_api
    
    echo "============================================"
    
    # Monitor GPU before serving
    log_info "GPU status before serving:"
    monitor_gpu
    
    echo "============================================"
    
    # Wait for LLM server to be ready
    if wait_for_server "$LLM_API_URL" 120; then
        # Test LLM inference
        test_llm_inference "llama-7b"
        
        echo "============================================"
        
        # Monitor GPU after serving
        log_info "GPU status during serving:"
        monitor_gpu
        
        echo "============================================"
        
        # Performance test
        read -p "Run performance test? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            performance_test "llama-7b" 3
        fi
    else
        log_error "LLM server not ready, skipping inference tests"
    fi
    
    echo "============================================"
    log_info "Test complete!"
}

# Command line options
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [base_url] [llm_port] [management_port]"
        echo
        echo "Examples:"
        echo "  $0                                    # Use defaults (localhost:8080, localhost:9010)"
        echo "  $0 http://192.168.1.100              # Custom host"
        echo "  $0 http://localhost 8081 9011        # Custom ports"
        echo
        echo "Environment variables:"
        echo "  MODEL_NAME    - Model name to test (default: llama-7b)"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac 
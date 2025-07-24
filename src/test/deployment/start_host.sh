#!/bin/bash
# GSwarm Host Start Script
# This script starts the GSwarm host service with proper configuration

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
HOST_PORT=${HOST_PORT:-8080}
GRPC_PORT=${GRPC_PORT:-50051}
DATA_DIR=${DATA_DIR:-"$HOME/.gswarm/host"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
MODEL_STORAGE_DIR=${MODEL_STORAGE_DIR:-"$HOME/.gswarm/models"}

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if gswarm host is installed
if ! command -v gswarm-host &> /dev/null; then
    print_error "gswarm-host command not found. Please install GSwarm first."
    exit 1
fi

# Create necessary directories
print_info "Creating data directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_STORAGE_DIR"
mkdir -p "$DATA_DIR/logs"

# Check if host is already running
if lsof -Pi :$HOST_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port $HOST_PORT is already in use. Host might be already running."
    echo "Do you want to kill the existing process? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        print_info "Stopping existing host..."
        pkill -f "gswarm-host" || true
        sleep 2
    else
        print_error "Cannot start host while port is in use."
        exit 1
    fi
fi

# Generate host configuration
print_info "Generating host configuration..."
cat > "$DATA_DIR/host_config.yaml" <<EOF
# GSwarm Host Configuration
host:
  address: "0.0.0.0"
  port: $HOST_PORT
  grpc_port: $GRPC_PORT

storage:
  data_dir: "$DATA_DIR"
  model_dir: "$MODEL_STORAGE_DIR"
  
logging:
  level: "$LOG_LEVEL"
  file: "$DATA_DIR/logs/host.log"
  
scheduler:
  default_strategy: "baseline"
  enable_cost_prediction: true
  
monitoring:
  enable_metrics: true
  metrics_port: 9090
EOF

# Start the host
print_info "Starting GSwarm host..."
print_info "Configuration:"
print_info "  - HTTP Port: $HOST_PORT"
print_info "  - gRPC Port: $GRPC_PORT"
print_info "  - Data Directory: $DATA_DIR"
print_info "  - Model Storage: $MODEL_STORAGE_DIR"
print_info "  - Log Level: $LOG_LEVEL"

# Export environment variables
export GSWARM_DATA_DIR="$DATA_DIR"
export GSWARM_MODEL_DIR="$MODEL_STORAGE_DIR"
export GSWARM_LOG_LEVEL="$LOG_LEVEL"

# Run the host
if [[ "$1" == "-d" ]] || [[ "$1" == "--daemon" ]]; then
    print_info "Starting host in daemon mode..."
    nohup gswarm-host serve \
        --config "$DATA_DIR/host_config.yaml" \
        --port $HOST_PORT \
        --grpc-port $GRPC_PORT \
        > "$DATA_DIR/logs/host.out" 2>&1 &
    
    HOST_PID=$!
    echo $HOST_PID > "$DATA_DIR/host.pid"
    
    print_info "Host started with PID: $HOST_PID"
    print_info "Logs: $DATA_DIR/logs/host.out"
    print_info "To stop: kill \$(cat $DATA_DIR/host.pid)"
else
    print_info "Starting host in foreground mode..."
    print_info "Press Ctrl+C to stop"
    
    gswarm-host serve \
        --config "$DATA_DIR/host_config.yaml" \
        --port $HOST_PORT \
        --grpc-port $GRPC_PORT
fi
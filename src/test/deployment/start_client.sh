#!/bin/bash
# GSwarm Client Start Script
# This script starts a GSwarm client and connects it to the host

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
HOST_URL=${HOST_URL:-"http://localhost:8080"}
CLIENT_NAME=${CLIENT_NAME:-"client-$(hostname)-$$"}
CLIENT_PORT=${CLIENT_PORT:-8081}
GPU_COUNT=${GPU_COUNT:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)}
DATA_DIR=${DATA_DIR:-"$HOME/.gswarm/client"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

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

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check if gswarm client is installed
if ! command -v gswarm-client &> /dev/null; then
    print_error "gswarm-client command not found. Please install GSwarm first."
    exit 1
fi

# Check GPU availability
if [ "$GPU_COUNT" -eq 0 ]; then
    print_warning "No GPUs detected. Client will run in CPU-only mode."
    DEVICE_LIST="cpu:0"
else
    print_info "Detected $GPU_COUNT GPU(s)"
    DEVICE_LIST=""
    for i in $(seq 0 $((GPU_COUNT-1))); do
        if [ -z "$DEVICE_LIST" ]; then
            DEVICE_LIST="cuda:$i"
        else
            DEVICE_LIST="$DEVICE_LIST,cuda:$i"
        fi
    done
fi

# Create necessary directories
print_info "Creating client directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/logs"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/cache"

# Check if client is already running on this port
if lsof -Pi :$CLIENT_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port $CLIENT_PORT is already in use."
    CLIENT_PORT=$((CLIENT_PORT + RANDOM % 1000))
    print_info "Using alternative port: $CLIENT_PORT"
fi

# Generate client configuration
print_info "Generating client configuration..."
cat > "$DATA_DIR/client_config.yaml" <<EOF
# GSwarm Client Configuration
client:
  name: "$CLIENT_NAME"
  port: $CLIENT_PORT
  host_url: "$HOST_URL"
  
resources:
  devices: [$DEVICE_LIST]
  memory_limit_gb: 32
  
storage:
  data_dir: "$DATA_DIR"
  model_cache_dir: "$DATA_DIR/models"
  cache_dir: "$DATA_DIR/cache"
  
logging:
  level: "$LOG_LEVEL"
  file: "$DATA_DIR/logs/client.log"
  
monitoring:
  enable_metrics: true
  report_interval: 30
  
features:
  enable_model_caching: true
  enable_compression: true
  max_concurrent_tasks: 4
EOF

# Function to register client with host
register_client() {
    print_info "Registering client with host..."
    
    # Wait for host to be ready
    MAX_RETRIES=10
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s -f "$HOST_URL/health" > /dev/null 2>&1; then
            print_info "Host is ready at $HOST_URL"
            break
        fi
        
        print_warning "Host not ready, retrying in 2 seconds..."
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "Failed to connect to host at $HOST_URL"
        exit 1
    fi
    
    # Prepare device list in new format
    DEVICE_JSON="["
    IFS=',' read -ra DEVICES <<< "$DEVICE_LIST"
    for i in "${!DEVICES[@]}"; do
        device="${DEVICES[$i]}"
        if [[ "$device" == cuda:* ]]; then
            device_id="${device#cuda:}"
            device_formatted="${CLIENT_NAME}:cuda:${device_id}"
        else
            device_formatted="${CLIENT_NAME}:${device}"
        fi
        
        if [ $i -gt 0 ]; then
            DEVICE_JSON+=","
        fi
        DEVICE_JSON+="\"$device_formatted\""
    done
    DEVICE_JSON+="]"
    
    # Register client
    RESPONSE=$(curl -s -X POST "$HOST_URL/api/v1/clients/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"client_id\": \"$CLIENT_NAME\",
            \"address\": \"$(hostname -I | awk '{print $1}'):$CLIENT_PORT\",
            \"devices\": $DEVICE_JSON,
            \"capabilities\": {
                \"gpu_count\": $GPU_COUNT,
                \"supports_llm\": true,
                \"supports_diffusion\": true
            }
        }")
    
    if echo "$RESPONSE" | grep -q "error"; then
        print_error "Failed to register client: $RESPONSE"
        return 1
    else
        print_info "Client registered successfully"
        print_debug "Response: $RESPONSE"
        return 0
    fi
}

# Start the client
print_info "Starting GSwarm client..."
print_info "Configuration:"
print_info "  - Client Name: $CLIENT_NAME"
print_info "  - Client Port: $CLIENT_PORT"
print_info "  - Host URL: $HOST_URL"
print_info "  - Devices: $DEVICE_LIST"
print_info "  - Data Directory: $DATA_DIR"

# Export environment variables
export GSWARM_CLIENT_NAME="$CLIENT_NAME"
export GSWARM_CLIENT_PORT="$CLIENT_PORT"
export GSWARM_HOST_URL="$HOST_URL"
export GSWARM_DATA_DIR="$DATA_DIR"
export GSWARM_LOG_LEVEL="$LOG_LEVEL"

# Run the client
if [[ "$1" == "-d" ]] || [[ "$1" == "--daemon" ]]; then
    print_info "Starting client in daemon mode..."
    
    nohup gswarm-client serve \
        --config "$DATA_DIR/client_config.yaml" \
        --name "$CLIENT_NAME" \
        --port $CLIENT_PORT \
        --host "$HOST_URL" \
        > "$DATA_DIR/logs/client.out" 2>&1 &
    
    CLIENT_PID=$!
    echo $CLIENT_PID > "$DATA_DIR/client.pid"
    
    print_info "Client started with PID: $CLIENT_PID"
    print_info "Logs: $DATA_DIR/logs/client.out"
    
    # Wait a bit for client to start
    sleep 3
    
    # Register with host
    if register_client; then
        print_info "Client is ready and connected to host"
        print_info "To stop: kill \$(cat $DATA_DIR/client.pid)"
    else
        print_error "Failed to register client, stopping..."
        kill $CLIENT_PID
        exit 1
    fi
else
    print_info "Starting client in foreground mode..."
    print_info "Press Ctrl+C to stop"
    
    # Register first in background
    (sleep 3 && register_client) &
    
    gswarm-client serve \
        --config "$DATA_DIR/client_config.yaml" \
        --name "$CLIENT_NAME" \
        --port $CLIENT_PORT \
        --host "$HOST_URL"
fi
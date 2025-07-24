#!/bin/bash
# GSwarm Client Connection Script
# This script connects an existing client to a host or manages client connections

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
HOST_URL=${HOST_URL:-"http://localhost:8080"}
ACTION=${1:-"status"}

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

# Function to check host health
check_host() {
    if curl -s -f "$HOST_URL/health" > /dev/null 2>&1; then
        print_info "Host is healthy at $HOST_URL"
        return 0
    else
        print_error "Host is not reachable at $HOST_URL"
        return 1
    fi
}

# Function to list all clients
list_clients() {
    print_info "Fetching client list from $HOST_URL..."
    
    RESPONSE=$(curl -s "$HOST_URL/api/v1/clients")
    
    if [ -z "$RESPONSE" ]; then
        print_error "Failed to get client list"
        return 1
    fi
    
    echo "$RESPONSE" | jq -r '.clients[] | "\(.id)\t\(.status)\t\(.devices | join(","))\t\(.address)"' | \
        awk 'BEGIN {printf "%-30s %-10s %-40s %-20s\n", "CLIENT ID", "STATUS", "DEVICES", "ADDRESS"} 
             {printf "%-30s %-10s %-40s %-20s\n", $1, $2, $3, $4}'
}

# Function to get client status
get_client_status() {
    local client_id=$1
    
    if [ -z "$client_id" ]; then
        print_error "Client ID required"
        return 1
    fi
    
    print_info "Getting status for client: $client_id"
    
    RESPONSE=$(curl -s "$HOST_URL/api/v1/clients/$client_id")
    
    if echo "$RESPONSE" | grep -q "not found"; then
        print_error "Client $client_id not found"
        return 1
    fi
    
    echo "$RESPONSE" | jq .
}

# Function to reconnect a client
reconnect_client() {
    local client_id=$1
    
    if [ -z "$client_id" ]; then
        print_error "Client ID required"
        return 1
    fi
    
    print_info "Reconnecting client: $client_id"
    
    RESPONSE=$(curl -s -X POST "$HOST_URL/api/v1/clients/$client_id/reconnect")
    
    if echo "$RESPONSE" | grep -q "error"; then
        print_error "Failed to reconnect: $RESPONSE"
        return 1
    else
        print_info "Reconnection initiated"
        echo "$RESPONSE" | jq .
    fi
}

# Function to disconnect a client
disconnect_client() {
    local client_id=$1
    
    if [ -z "$client_id" ]; then
        print_error "Client ID required"
        return 1
    fi
    
    print_info "Disconnecting client: $client_id"
    
    RESPONSE=$(curl -s -X POST "$HOST_URL/api/v1/clients/$client_id/disconnect")
    
    if echo "$RESPONSE" | grep -q "error"; then
        print_error "Failed to disconnect: $RESPONSE"
        return 1
    else
        print_info "Client disconnected"
        echo "$RESPONSE" | jq .
    fi
}

# Function to test client connectivity
test_client() {
    local client_id=$1
    
    if [ -z "$client_id" ]; then
        print_error "Client ID required"
        return 1
    fi
    
    print_info "Testing connectivity for client: $client_id"
    
    # Send a test task
    RESPONSE=$(curl -s -X POST "$HOST_URL/api/v1/test/client/$client_id" \
        -H "Content-Type: application/json" \
        -d '{
            "test_type": "ping",
            "timeout": 5
        }')
    
    if echo "$RESPONSE" | grep -q "success"; then
        print_info "Client is responsive"
        echo "$RESPONSE" | jq .
    else
        print_error "Client test failed"
        echo "$RESPONSE" | jq .
        return 1
    fi
}

# Function to monitor all clients
monitor_clients() {
    print_info "Monitoring clients (press Ctrl+C to stop)..."
    
    while true; do
        clear
        echo "=== GSwarm Client Monitor ==="
        echo "Host: $HOST_URL"
        echo "Time: $(date)"
        echo ""
        
        RESPONSE=$(curl -s "$HOST_URL/api/v1/clients")
        
        if [ ! -z "$RESPONSE" ]; then
            echo "$RESPONSE" | jq -r '.clients[] | "\(.id)\t\(.status)\t\(.last_heartbeat)\t\(.active_tasks)\t\(.devices | length)"' | \
                awk 'BEGIN {printf "%-30s %-10s %-20s %-15s %-10s\n", "CLIENT ID", "STATUS", "LAST HEARTBEAT", "ACTIVE TASKS", "DEVICES"} 
                     {printf "%-30s %-10s %-20s %-15s %-10s\n", $1, $2, $3, $4, $5}'
        fi
        
        sleep 5
    done
}

# Main script logic
case "$ACTION" in
    "status")
        check_host && list_clients
        ;;
    "list")
        list_clients
        ;;
    "info")
        get_client_status "$2"
        ;;
    "reconnect")
        reconnect_client "$2"
        ;;
    "disconnect")
        disconnect_client "$2"
        ;;
    "test")
        test_client "$2"
        ;;
    "monitor")
        monitor_clients
        ;;
    "help"|"-h"|"--help")
        echo "GSwarm Client Connection Manager"
        echo ""
        echo "Usage: $0 [action] [arguments]"
        echo ""
        echo "Actions:"
        echo "  status              - Check host health and list all clients (default)"
        echo "  list                - List all connected clients"
        echo "  info <client_id>    - Get detailed information about a specific client"
        echo "  reconnect <id>      - Reconnect a disconnected client"
        echo "  disconnect <id>     - Disconnect a client"
        echo "  test <client_id>    - Test client connectivity"
        echo "  monitor             - Monitor all clients in real-time"
        echo ""
        echo "Environment Variables:"
        echo "  HOST_URL            - GSwarm host URL (default: http://localhost:8080)"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 info client-node1-1234"
        echo "  $0 test client-gpu-server"
        echo "  HOST_URL=http://192.168.1.10:8080 $0 monitor"
        ;;
    *)
        print_error "Unknown action: $ACTION"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
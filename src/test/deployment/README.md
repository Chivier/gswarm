# Deployment Tests

This directory contains scripts for deploying and managing GSwarm infrastructure.

## Scripts

### `start_host.sh`
Starts the GSwarm host service with configurable options.

```bash
# Start in foreground
./start_host.sh

# Start as daemon
./start_host.sh -d

# With custom settings
HOST_PORT=9090 ./start_host.sh
```

### `start_client.sh`
Starts a GSwarm client and registers it with the host.

```bash
# Start with auto-detected GPUs
./start_client.sh

# Start with custom name and host
CLIENT_NAME=gpu-server-1 HOST_URL=http://192.168.1.10:8080 ./start_client.sh -d
```

### `connect_client.sh`
Manages client connections and monitoring.

```bash
# List all clients
./connect_client.sh list

# Monitor clients
./connect_client.sh monitor

# Test specific client
./connect_client.sh test client-name
```

## Multi-Node Deployment Example

```bash
# On host machine
./start_host.sh -d

# On GPU node 1
CLIENT_NAME=gpu-node-1 HOST_URL=http://host-ip:8080 ./start_client.sh -d

# On GPU node 2  
CLIENT_NAME=gpu-node-2 HOST_URL=http://host-ip:8080 ./start_client.sh -d

# Monitor all
HOST_URL=http://host-ip:8080 ./connect_client.sh monitor
```

## Environment Variables

### Host Configuration
- `HOST_PORT`: HTTP API port (default: 8080)
- `GRPC_PORT`: gRPC service port (default: 50051)
- `DATA_DIR`: Data directory (default: ~/.gswarm/host)
- `MODEL_STORAGE_DIR`: Model storage (default: ~/.gswarm/models)

### Client Configuration
- `HOST_URL`: GSwarm host URL (default: http://localhost:8080)
- `CLIENT_NAME`: Unique client name (default: auto-generated)
- `CLIENT_PORT`: Client service port (default: 8081)
- `GPU_COUNT`: Number of GPUs (default: auto-detected)
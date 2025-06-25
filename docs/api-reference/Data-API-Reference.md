# Data API Reference

The Data API provides comprehensive data pool and KV storage management operations for the gswarm system.

## Overview

The data module supports two main types of operations:

1. **KV Storage Operations**: Direct key-value storage with persistent and volatile options
2. **Data Pool Operations**: Managed data chunks with device placement and transfer capabilities

**Note**: Both KV storage and data pool operations are now unified on a single server running on port 9015.

## KV Storage Commands

### Start Server

Start the unified data storage server that handles both KV storage and data pool operations.

```bash
gswarm data start [OPTIONS]
```

**Options:**
- `--host TEXT`: Server host (default: 0.0.0.0)
- `--port INTEGER`: Server port (default: 9015)
- `--max-memory TEXT`: Maximum memory size (default: 16GB)

**Example:**
```bash
gswarm data start --host localhost --port 9015 --max-memory 16GB
```

### Write Key-Value

Write a key-value pair to storage.

```bash
gswarm data write KEY VALUE [OPTIONS]
```

**Arguments:**
- `KEY`: Key to write (required)
- `VALUE`: Value to write (required)

**Options:**
- `--persist/--volatile`: Whether to persist data (default: persist)
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data write "user:123" "john_doe" --persist
```

### Read Key

Read value by key from storage.

```bash
gswarm data read KEY [OPTIONS]
```

**Arguments:**
- `KEY`: Key to read (required)

**Options:**
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data read "user:123"
```

### Release Key

Remove key from storage.

```bash
gswarm data release KEY [OPTIONS]
```

**Arguments:**
- `KEY`: Key to release (required)

**Options:**
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data release "user:123"
```

### Send Key

Send key to another KV storage server.

```bash
gswarm data send KEY TARGET [OPTIONS]
```

**Arguments:**
- `KEY`: Key to send (required)
- `TARGET`: Target URL (e.g., localhost:9016) (required)

**Options:**
- `--host TEXT`: Source server address (default: localhost:9015)

**Example:**
```bash
gswarm data send "user:123" "localhost:9016"
```

### Storage Statistics

Show storage statistics.

```bash
gswarm data stats [OPTIONS]
```

**Options:**
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data stats
```

**Output includes:**
- Memory usage (current/max/percentage)
- Key counts (total/persistent/volatile)
- System memory information

## Data Pool Commands

### List Data Chunks

List data chunks in the pool.

```bash
gswarm data list [OPTIONS]
```

**Options:**
- `--device, -d TEXT`: Filter by device
- `--type, -t TEXT`: Filter by data type
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data list --device gpu --type input
```

### Create Data Chunk

Create a new data chunk.

```bash
gswarm data create [OPTIONS]
```

**Options:**
- `--source, -s TEXT`: Data source (URL or path) (required)
- `--device, -d TEXT`: Target device (default: dram)
- `--type, -t TEXT`: Data type (input/output/intermediate) (default: input)
- `--format, -f TEXT`: Data format (default: tensor)
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data create --source "/path/to/data.bin" --device gpu --type input --format tensor
```

### Get Chunk Information

Get detailed information about a data chunk.

```bash
gswarm data info CHUNK_ID [OPTIONS]
```

**Arguments:**
- `CHUNK_ID`: Data chunk ID (required)

**Options:**
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data info "chunk_abc123"
```

### Move Data Chunk

Move data chunk between devices.

```bash
gswarm data move CHUNK_ID [OPTIONS]
```

**Arguments:**
- `CHUNK_ID`: Data chunk ID (required)

**Options:**
- `--to, -t TEXT`: Target device (required)
- `--priority, -p TEXT`: Priority (default: normal)
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data move "chunk_abc123" --to gpu --priority high
```

### Transfer Data Chunk

Transfer data chunk to another node.

```bash
gswarm data transfer CHUNK_ID [OPTIONS]
```

**Arguments:**
- `CHUNK_ID`: Data chunk ID (required)

**Options:**
- `--to, -t TEXT`: Target node:device (required)
- `--delete-source`: Delete source after transfer
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data transfer "chunk_abc123" --to "node2:gpu" --delete-source
```

### Delete Data Chunk

Delete data chunk from pool.

```bash
gswarm data delete CHUNK_ID [OPTIONS]
```

**Arguments:**
- `CHUNK_ID`: Data chunk ID (required)

**Options:**
- `--force, -f`: Force deletion even if referenced
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data delete "chunk_abc123" --force
```

## Memory Size Format

Memory size can be specified with the following suffixes:
- `B`: Bytes
- `KB`: Kilobytes (1024 bytes)
- `MB`: Megabytes (1024² bytes)
- `GB`: Gigabytes (1024³ bytes)
- `TB`: Terabytes (1024⁴ bytes)

**Examples:**
- `16GB`
- `1TB`
- `512MB`
- `1024`

## Error Handling

All commands include comprehensive error handling with descriptive messages. Common error scenarios include:

- Network connectivity issues
- Invalid chunk IDs
- Insufficient memory
- Permission errors
- Data format incompatibilities

## HTTP API Endpoints

The commands interact with the following HTTP API endpoints on the unified server (Port 9015):

### KV Storage API
- `POST /write`: Write key-value pair (JSON compatible values)
- `POST /write_extended`: Write key-value pair (supports complex types via pickle)
- `GET /read/{key}`: Read value by key (JSON compatible response)
- `GET /read_extended/{key}`: Read value by key (supports complex types)
- `DELETE /release/{key}`: Release key
- `POST /send`: Send key to another server
- `GET /stats`: Get storage statistics
- `POST /set_max_memory/{max_size}`: Set maximum memory size

### Data Pool API
- `GET /api/v1/data`: List data chunks
- `POST /api/v1/data`: Create data chunk
- `GET /api/v1/data/{chunk_id}`: Get chunk information
- `POST /api/v1/data/{chunk_id}/move`: Move chunk between devices
- `POST /api/v1/data/{chunk_id}/transfer`: Transfer chunk to another node
- `DELETE /api/v1/data/{chunk_id}`: Delete data chunk

## Python API Usage

### Basic KV Storage Operations

```python
from gswarm.data import DataServer, get_storage, start_server

# Using DataServer client (recommended for remote access)
client = DataServer("localhost:9015")

# Write data
client.write("user:123", {"name": "john", "age": 30}, persist=True)
client.write("temp:cache", [1, 2, 3, 4, 5], persist=False)

# Read data
user_data = client.read("user:123")
cache_data = client.read("temp:cache")

# Get statistics
stats = client.get_stats()
print(f"Memory usage: {stats['usage_percent']:.1f}%")

# Send data to another server
success = client.send("user:123", "localhost:9016")

# Release data
client.release("temp:cache")
```

### Advanced KV Storage with Complex Types

```python
import numpy as np
import torch

# Store complex data types (automatically uses extended API)
client = DataServer("localhost:9015", use_extended_api=True)

# Store numpy arrays
data = np.random.rand(1000, 1000)
client.write("model:weights", data, persist=True)

# Store PyTorch tensors
tensor = torch.randn(512, 768)
client.write("model:embeddings", tensor, persist=True)

# Store custom objects
model_config = {
    "name": "llama-7b",
    "layers": 32,
    "hidden_size": 4096,
    "weights": data,
    "tensor_data": tensor
}
client.write("model:config", model_config, persist=True)

# Read back the data
retrieved_config = client.read("model:config")
```

### Direct Storage Access (for embedded use)

```python
from gswarm.data import get_storage, set_max_memory

# Set memory limit (16GB)
set_max_memory(16 * 1024 * 1024 * 1024)

# Get storage instance
storage = get_storage()

# Direct operations
storage.write("key1", "value1", persist=True)
value = storage.read("key1")
stats = storage.get_stats()
storage.release("key1")
```

### Data Pool Operations via HTTP

```python
import requests

base_url = "http://localhost:9015"

# List data chunks
response = requests.get(f"{base_url}/api/v1/data")
chunks = response.json()["chunks"]

# Create a data chunk
chunk_data = {
    "source": "/path/to/data.bin",
    "device": "gpu",
    "type": "input",
    "format": "tensor"
}
response = requests.post(f"{base_url}/api/v1/data", json=chunk_data)
chunk_id = response.json()["chunk_id"]

# Get chunk information
response = requests.get(f"{base_url}/api/v1/data/{chunk_id}")
chunk_info = response.json()

# Move chunk to different device
move_data = {"target_device": "dram", "priority": "high"}
response = requests.post(f"{base_url}/api/v1/data/{chunk_id}/move", json=move_data)

# Transfer chunk to another node
transfer_data = {
    "target_node": "node2",
    "target_device": "gpu",
    "delete_source": False
}
response = requests.post(f"{base_url}/api/v1/data/{chunk_id}/transfer", json=transfer_data)

# Delete chunk
response = requests.delete(f"{base_url}/api/v1/data/{chunk_id}")
```

### Starting the Server Programmatically

```python
from gswarm.data import start_server
import threading

# Start server in a separate thread
def run_server():
    start_server(host="0.0.0.0", port=9015, max_mem_size=16*1024*1024*1024)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Now use the client
client = DataServer("localhost:9015")
client.write("test", "Hello, World!")
```

### Memory Management

```python
# Monitor memory usage
client = DataServer("localhost:9015")

def check_memory_usage():
    stats = client.get_stats()
    usage_percent = stats['usage_percent']
    
    if usage_percent > 80:
        print(f"Warning: Memory usage at {usage_percent:.1f}%")
        
    print(f"Memory: {stats['current_size'] / (1024**3):.2f} GB / {stats['max_size'] / (1024**3):.2f} GB")
    print(f"Keys: {stats['total_keys']} ({stats['persistent_keys']} persistent, {stats['volatile_keys']} volatile)")

# Set memory limit dynamically
client.set_max_memory(32 * 1024 * 1024 * 1024)  # 32GB
```

## Configuration

Default configurations:
- **Unified Server**: `localhost:9015` (both KV storage and data pool)
- **Default device**: `dram`
- **Default memory limit**: `16GB`
- **Default data type**: `input`
- **Default format**: `tensor`
- **Extended API**: Enabled by default for complex data types

## Migration Notes

If you were previously using separate ports (9011 for data pool, 9015 for KV storage):

1. **Update your scripts**: Change all `localhost:9011` references to `localhost:9015`
2. **Single server**: You now only need to start one server that handles both APIs
3. **Backward compatibility**: All existing commands and API calls continue to work
4. **Performance**: Unified server reduces resource usage and improves data locality

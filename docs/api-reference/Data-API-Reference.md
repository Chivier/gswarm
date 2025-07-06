# Data API Reference

The Data API provides comprehensive data pool and KV storage management operations for the gswarm system with support for multiple storage locations including DRAM, GPU memory, and disk.

## Overview

The data module supports two main types of operations:

1. **KV Storage Operations**: Direct key-value storage with multi-location support (DRAM, pinned memory, GPU, disk)
2. **Data Pool Operations**: Managed data chunks with device placement and transfer capabilities

**Note**: Both KV storage and data pool operations are now unified on a single server running on port 9015.

### Storage Locations

The enhanced data module supports the following storage locations:

- **dram**: Regular system memory (default)
- **pinned_dram**: Pinned memory for faster GPU transfers
- **device:X**: GPU memory on device X (e.g., device:0, device:1)
- **disk**: Persistent disk storage

**Important**: Only data stored in DRAM can be read directly. Data in other locations must be moved to DRAM first using the move API.

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

Write a key-value pair to a specific storage location.

```bash
gswarm data write KEY VALUE [OPTIONS]
```

**Arguments:**
- `KEY`: Key to write (required)
- `VALUE`: Value to write (required)

**Options:**
- `--location, -l TEXT`: Storage location (default: dram)
  - Options: dram, pinned_dram, device:X, disk
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
# Write to DRAM (default)
gswarm data write "user:123" "john_doe"

# Write to GPU device 0
gswarm data write "model:weights" "large_tensor" --location device:0

# Write to disk for persistence
gswarm data write "backup:data" "important_data" --location disk
```

### Read Key

Read value by key from DRAM storage.

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

**Note**: Reading is only allowed from DRAM. If the data is in another location (GPU, disk, etc.), you must first move it to DRAM using the move command.

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

Send key to another KV storage server with optional NVLink GPU-to-GPU support.

```bash
gswarm data send KEY TARGET [OPTIONS]
```

**Arguments:**
- `KEY`: Key to send (required)
- `TARGET`: Target destination (required)
  - URL format: `localhost:9016` for standard transfers
  - GPU format: `node2:device:0` for GPU-to-GPU transfers

**Options:**
- `--source, -s TEXT`: Source location for GPU transfers (e.g., device:0)
- `--host TEXT`: Source server address (default: localhost:9015)

**Examples:**
```bash
# Standard transfer
gswarm data send "user:123" "localhost:9016"

# GPU-to-GPU transfer with NVLink (if available)
gswarm data send "model:weights" "node2:device:1" --source device:0
```

### Send GPU Data (Direct GPU-to-GPU)

Optimized command for GPU-to-GPU transfers.

```bash
gswarm data send-gpu KEY SOURCE_DEVICE TARGET_NODE TARGET_DEVICE [OPTIONS]
```

**Arguments:**
- `KEY`: Key to send (required)
- `SOURCE_DEVICE`: Source GPU device (e.g., device:0) (required)
- `TARGET_NODE`: Target node name (required)
- `TARGET_DEVICE`: Target GPU device (e.g., device:1) (required)

**Options:**
- `--host TEXT`: Source server address (default: localhost:9015)

**Example:**
```bash
# Direct GPU-to-GPU transfer
gswarm data send-gpu "model:weights" device:0 node2 device:1
```

**Features:**
- Automatic NVLink detection and usage
- Falls back to standard transfer if NVLink unavailable
- Shows transfer method and timing information

### Move Data

Move data between storage locations with optimized transfer methods.

```bash
gswarm data move-data KEY DESTINATION [OPTIONS]
```

**Arguments:**
- `KEY`: Key to move (required)
- `DESTINATION`: Target location (required)
  - Options: dram, pinned_dram, device:X, disk

**Options:**
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
# Move from GPU to DRAM for reading
gswarm data move-data "model:weights" dram

# Move from DRAM to GPU device 1
gswarm data move-data "input:batch" device:1

# Archive to disk
gswarm data move-data "results:final" disk
```

**Features:**
- Async operations for non-GPU transfers
- CUDA async memcpy for GPU operations
- Progress tracking and status updates
- Read pointer support for PD separation

### Get Location

Get the current location(s) of data.

```bash
gswarm data get-location KEY [OPTIONS]
```

**Arguments:**
- `KEY`: Key to query (required)

**Options:**
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data get-location "user:123"
```

**Output includes:**
- Primary location
- All locations where data exists
- Copy status (complete/copying/error)
- Read pointers for optimization

### List All Locations

List locations of all data in the system.

```bash
gswarm data list-locations [OPTIONS]
```

**Options:**
- `--host TEXT`: Server address (default: localhost:9015)

**Example:**
```bash
gswarm data list-locations
```

**Output includes:**
- All keys and their locations
- Size information
- Copy status for each location

### Storage Statistics

Show enhanced storage statistics including multi-location usage.

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
- DRAM usage (current/max/percentage)
- Pinned memory usage
- GPU memory usage per device
- Disk storage usage
- Key counts by location
- Active move operations
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
- `POST /write`: Write key-value pair to specified location (JSON compatible values)
- `POST /write_extended`: Write key-value pair to specified location (supports complex types via pickle)
- `GET /read/{key}`: Read value by key from DRAM (JSON compatible response)
- `GET /read_extended/{key}`: Read value by key from DRAM (supports complex types)
- `DELETE /release/{key}`: Release key from all locations
- `POST /send`: Send key to another server (legacy)
- `POST /send_extended`: Send key with GPU-to-GPU and NVLink support
- `GET /stats`: Get enhanced storage statistics
- `POST /set_max_memory/{max_size}`: Set maximum DRAM memory size
- `POST /move`: Move data between storage locations
- `GET /location/{key}`: Get location information for a key
- `GET /locations`: List all data locations in the system

### Data Pool API
- `GET /api/v1/data`: List data chunks
- `POST /api/v1/data`: Create data chunk
- `GET /api/v1/data/{chunk_id}`: Get chunk information
- `POST /api/v1/data/{chunk_id}/move`: Move chunk between devices
- `POST /api/v1/data/{chunk_id}/transfer`: Transfer chunk to another node
- `DELETE /api/v1/data/{chunk_id}`: Delete data chunk

## Python API Usage

### Basic KV Storage Operations with Multi-Location Support

```python
from gswarm.data import DataServer, get_storage, start_server

# Using DataServer client (recommended for remote access)
client = DataServer("localhost:9015")

# Write data to different locations
client.write("user:123", {"name": "john", "age": 30}, location="dram")
client.write("model:weights", large_tensor, location="device:0")
client.write("backup:data", important_data, location="disk")

# Move data to DRAM for reading
client.move("model:weights", "dram")

# Read data (only from DRAM)
user_data = client.read("user:123")
model_weights = client.read("model:weights")  # Now available after move

# Get location information
location_info = client.get_location("model:weights")
print(f"Data is in: {location_info['location']}")

# List all data locations
all_locations = client.list_locations()
for key, locations in all_locations.items():
    print(f"{key}: {[loc['location'] for loc in locations]}")

# Get enhanced statistics
stats = client.get_stats()
print(f"DRAM usage: {stats['dram_usage_percent']:.1f}%")
print(f"GPU 0 usage: {stats['gpu_stats'].get('device:0', {}).get('used', 0) / 1e9:.2f} GB")

# Send data to another server
success = client.send("user:123", "localhost:9016")

# GPU-to-GPU transfer with NVLink
success = client.send("model:weights", "node2:device:1", source_location="device:0")

# Release data from all locations
client.release("temp:cache")
```

### Advanced Multi-Location Operations

```python
import asyncio
from gswarm.data import DataServer

client = DataServer("localhost:9015")

# Async move operations for better performance
async def move_data_async():
    # Move multiple datasets concurrently
    tasks = [
        client.move("dataset1", "device:0"),
        client.move("dataset2", "device:1"),
        client.move("dataset3", "pinned_dram")
    ]
    
    # Wait for all moves to complete
    results = await asyncio.gather(*tasks)
    return results

# Run async moves
results = asyncio.run(move_data_async())

# PD Separation Optimization Example
# Process A writes data to GPU
client.write("shared:tensor", tensor_data, location="device:0")

# Process B moves data to its GPU
client.move("shared:tensor", "device:1")

# Get read pointer for optimized access
location_info = client.get_location("shared:tensor")
read_pointer = location_info['locations'][0].get('read_pointer')
print(f"Optimized read pointer: {read_pointer}")
```

### Advanced KV Storage with Complex Types and GPU

```python
import numpy as np
import torch

# Store complex data types with location control
client = DataServer("localhost:9015", use_extended_api=True)

# Store numpy arrays in pinned memory for faster GPU transfer
data = np.random.rand(1000, 1000)
client.write("model:weights", data, location="pinned_dram")

# Store PyTorch tensors directly on GPU
tensor = torch.randn(512, 768).cuda()
client.write("model:embeddings", tensor, location="device:0")

# Store large dataset on disk
large_dataset = np.random.rand(10000, 10000)
client.write("dataset:train", large_dataset, location="disk")

# Move data as needed
# Move weights to GPU for processing
client.move("model:weights", "device:0")

# Move embeddings to DRAM for CPU processing
client.move("model:embeddings", "dram")

# Now can read the embeddings
embeddings = client.read("model:embeddings")

# Store custom objects with optimal placement
model_config = {
    "name": "llama-7b",
    "layers": 32,
    "hidden_size": 4096,
    "device_placement": "device:0"
}
client.write("model:config", model_config, location="dram")
```

### NVLink GPU-to-GPU Transfers

```python
from gswarm.data import DataServer
import time

client = DataServer("localhost:9015")

# Store large model weights on GPU 0
model_weights = load_model_weights()  # Large tensor
client.write("model:llama-weights", model_weights, location="device:0")

# Direct GPU-to-GPU transfer to another node
# If NVLink is available, this bypasses CPU/DRAM completely
start_time = time.time()
result = client.send("model:llama-weights", "node2:device:1", source_location="device:0")
transfer_time = time.time() - start_time

print(f"Transfer completed in {transfer_time:.3f}s")
print(f"Method used: {result.get('method', 'unknown')}")

# Multi-node GPU cluster example
cluster_nodes = ["node1", "node2", "node3", "node4"]

# Distribute model shards across GPUs
for i, shard in enumerate(model_shards):
    source_device = f"device:{i % 2}"  # Alternate between GPU 0 and 1
    target_node = cluster_nodes[i % len(cluster_nodes)]
    target_device = f"device:{i % 4}"  # Distribute across 4 GPUs per node
    
    # Write shard to local GPU
    client.write(f"shard:{i}", shard, location=source_device)
    
    # Transfer to target node's GPU
    client.send(f"shard:{i}", f"{target_node}:{target_device}", source_location=source_device)

# Check NVLink availability
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        for j in range(i + 1, torch.cuda.device_count()):
            if torch.cuda.can_device_access_peer(i, j):
                print(f"NVLink available between GPU {i} and GPU {j}")
```

### Direct Storage Access with Multi-Location Support

```python
from gswarm.data import get_storage, set_max_memory
import asyncio

# Set DRAM limit (16GB)
set_max_memory(16 * 1024 * 1024 * 1024)

# Get storage instance
storage = get_storage()

# Direct operations with location control
storage.write("key1", "value1", location="dram")
storage.write("key2", large_array, location="device:0")
storage.write("key3", archive_data, location="disk")

# Move data to DRAM for reading
async def move_and_read():
    await storage.move_async("key2", "dram")
    value = storage.read("key2")
    return value

value = asyncio.run(move_and_read())

# Get location information
location = storage.get_location("key2")
all_locations = storage.list_locations()

# Enhanced statistics
stats = storage.get_stats()
print(f"DRAM: {stats['dram_keys']} keys, {stats['dram_usage_percent']:.1f}% used")
print(f"GPU devices: {stats['gpu_stats']}")
print(f"Disk: {stats['disk_keys']} keys, {stats['disk_usage'] / 1e9:.2f} GB")

# Clean up
storage.release("key1")  # Removes from all locations
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
- **Default location**: `dram`
- **Default DRAM limit**: `16GB`
- **Default data type**: `input`
- **Default format**: `tensor`
- **Extended API**: Enabled by default for complex data types
- **Disk storage path**: `/tmp/gswarm_data`
- **Move executor threads**: `4`

### Storage Location Capabilities

| Location | Read Value | Write | Move From | Move To | Persistence |
|----------|------------|-------|-----------|---------|-------------|
| dram | ✓ | ✓ | ✓ | ✓ | Until restart |
| pinned_dram | ✗ | ✓ | ✓ | ✓ | Until restart |
| device:X | ✗ | ✓ | ✓ | ✓ | Until restart |
| disk | ✗ | ✓ | ✓ | ✓ | Persistent |

### Performance Optimization Tips

1. **Use pinned_dram** for data that will be frequently moved to/from GPU
2. **Batch move operations** using async API for better throughput
3. **Use read pointers** for PD separation to avoid redundant copies
4. **Monitor GPU memory** usage to avoid out-of-memory errors
5. **Archive to disk** for data not actively being processed
6. **Use NVLink** for direct GPU-to-GPU transfers when available

### NVLink Support

**Benefits:**
- Direct GPU-to-GPU transfers without CPU involvement
- Significantly reduced latency for large model weights
- Automatic fallback to standard transfer if unavailable
- Support for multi-node GPU clusters

**Requirements:**
- NVIDIA GPUs with NVLink connectivity
- PyTorch with CUDA support
- Proper P2P (peer-to-peer) access between GPUs

**Transfer Methods:**
- `nvlink`: Direct GPU-to-GPU via NVLink (fastest)
- `standard`: Traditional CPU-mediated transfer (fallback)

## Migration Notes

### From Previous Version

If you were previously using the older version without multi-location support:

1. **Write operations**: Replace `persist=True/False` with `location="dram"` or other locations
2. **Read operations**: Ensure data is in DRAM before reading (use move if needed)
3. **Storage stats**: Update code to use new stats structure with location-specific metrics

### Example Migration

```python
# Old code
client.write("key", value, persist=True)

# New code
client.write("key", value, location="dram")  # or "disk" for persistence
```

### Unified Server Notes

If you were previously using separate ports (9011 for data pool, 9015 for KV storage):

1. **Update your scripts**: Change all `localhost:9011` references to `localhost:9015`
2. **Single server**: You now only need to start one server that handles both APIs
3. **Backward compatibility**: Most existing commands continue to work with minor adjustments
4. **Performance**: Unified server with multi-location support provides better resource utilization

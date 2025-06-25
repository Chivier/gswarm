"""
Data API Usage Examples

This example demonstrates how to use the gswarm data module API for both
KV storage and data pool operations.
"""

import time
import numpy as np
import threading
from typing import Any, Dict, List
from gswarm.data import DataServer, get_storage, start_server


def example_kv_storage():
    """Demonstrate KV storage operations"""
    print("=== KV Storage Examples ===")
    
    # Create a client connection
    client = DataServer("localhost:9015")
    
    # Basic operations
    print("\n1. Basic KV Operations:")
    client.write("user:123", {"name": "Alice", "role": "admin"}, persist=True)
    client.write("session:abc", "temporary_token", persist=False)
    
    user_data = client.read("user:123")
    session_data = client.read("session:abc")
    
    print(f"User data: {user_data}")
    print(f"Session data: {session_data}")
    
    # Complex data types
    print("\n2. Complex Data Types:")
    # Store numpy array
    large_array = np.random.rand(1000, 1000)
    client.write("model:weights", large_array, persist=True)
    
    # Store nested data structure
    config = {
        "model_name": "llama-7b",
        "parameters": {
            "layers": 32,
            "hidden_size": 4096,
            "vocab_size": 32000
        },
        "weights": large_array,
        "metadata": {
            "created_at": time.time(),
            "version": "1.0"
        }
    }
    client.write("model:config", config, persist=True)
    
    # Read back complex data
    retrieved_config = client.read("model:config")
    print(f"Model config retrieved: {retrieved_config['model_name']}")
    print(f"Weights shape: {retrieved_config['weights'].shape}")
    
    # Statistics
    print("\n3. Storage Statistics:")
    stats = client.get_stats()
    print(f"Memory usage: {stats['current_size'] / (1024**3):.2f} GB / {stats['max_size'] / (1024**3):.2f} GB ({stats['usage_percent']:.1f}%)")
    print(f"Total keys: {stats['total_keys']} (persistent: {stats['persistent_keys']}, volatile: {stats['volatile_keys']})")
    
    # Cleanup
    print("\n4. Cleanup:")
    client.release("session:abc")
    client.release("model:weights")
    client.release("model:config")
    client.release("user:123")
    print("All test data released")


def example_data_pool():
    """Demonstrate data pool operations using HTTP API"""
    print("\n=== Data Pool Examples ===")
    
    import requests
    base_url = "http://localhost:9015"
    
    try:
        # List existing chunks
        print("\n1. List Data Chunks:")
        response = requests.get(f"{base_url}/api/v1/data")
        response.raise_for_status()
        chunks = response.json()["chunks"]
        print(f"Found {len(chunks)} existing chunks")
        
        # Create a new chunk
        print("\n2. Create Data Chunk:")
        chunk_data = {
            "source": "/tmp/example_data.bin",
            "device": "dram",
            "type": "input",
            "format": "tensor"
        }
        response = requests.post(f"{base_url}/api/v1/data", json=chunk_data)
        response.raise_for_status()
        result = response.json()
        chunk_id = result["chunk_id"]
        print(f"Created chunk: {chunk_id}")
        
        # Get chunk information
        print("\n3. Get Chunk Info:")
        response = requests.get(f"{base_url}/api/v1/data/{chunk_id}")
        response.raise_for_status()
        chunk_info = response.json()
        print(f"Chunk {chunk_id}:")
        print(f"  Type: {chunk_info['chunk_type']}")
        print(f"  Size: {chunk_info['size'] / 1e6:.2f} MB")
        print(f"  Locations: {[loc['device'] for loc in chunk_info['locations']]}")
        
        # Move chunk to different device
        print("\n4. Move Chunk:")
        move_data = {"target_device": "gpu", "priority": "normal"}
        response = requests.post(f"{base_url}/api/v1/data/{chunk_id}/move", json=move_data)
        response.raise_for_status()
        move_result = response.json()
        print(f"Move operation started: {move_result.get('task_id')}")
        
        # Transfer chunk to another node
        print("\n5. Transfer Chunk:")
        transfer_data = {
            "target_node": "node2",
            "target_device": "dram",
            "delete_source": False
        }
        response = requests.post(f"{base_url}/api/v1/data/{chunk_id}/transfer", json=transfer_data)
        response.raise_for_status()
        transfer_result = response.json()
        print(f"Transfer operation started: {transfer_result.get('task_id')}")
        
        # List chunks again to see changes
        print("\n6. List Updated Chunks:")
        response = requests.get(f"{base_url}/api/v1/data")
        response.raise_for_status()
        chunks = response.json()["chunks"]
        for chunk in chunks:
            if chunk["chunk_id"] == chunk_id:
                print(f"Chunk {chunk_id} locations: {[loc['device'] for loc in chunk['locations']]}")
        
        # Delete chunk
        print("\n7. Delete Chunk:")
        response = requests.delete(f"{base_url}/api/v1/data/{chunk_id}")
        response.raise_for_status()
        delete_result = response.json()
        print(f"Chunk deleted: {delete_result['message']}")
        
    except requests.RequestException as e:
        print(f"HTTP request failed: {e}")
    except Exception as e:
        print(f"Error: {e}")


def example_memory_management():
    """Demonstrate memory management features"""
    print("\n=== Memory Management Examples ===")
    
    client = DataServer("localhost:9015")
    
    # Set memory limit
    print("\n1. Set Memory Limit:")
    success = client.set_max_memory(8 * 1024 * 1024 * 1024)  # 8GB
    print(f"Memory limit set: {success}")
    
    # Fill memory with data
    print("\n2. Fill Memory:")
    data_chunks = []
    for i in range(10):
        # Create 100MB chunks
        chunk = np.random.rand(100, 100, 100).astype(np.float32)
        key = f"large_data:{i}"
        
        # Persistent data (won't be evicted)
        if i < 5:
            success = client.write(key, chunk, persist=True)
            print(f"Stored persistent chunk {i}: {success}")
        else:
            # Volatile data (can be evicted)
            success = client.write(key, chunk, persist=False)
            print(f"Stored volatile chunk {i}: {success}")
        
        data_chunks.append(key)
        
        # Check memory usage
        stats = client.get_stats()
        print(f"  Memory usage: {stats['usage_percent']:.1f}%")
    
    # Monitor eviction
    print("\n3. Check Data Availability:")
    for key in data_chunks:
        value = client.read(key)
        available = value is not None
        print(f"{key}: {'Available' if available else 'Evicted'}")
    
    # Cleanup
    print("\n4. Cleanup:")
    for key in data_chunks:
        client.release(key)
    
    # Reset memory limit
    client.set_max_memory(16 * 1024 * 1024 * 1024)  # 16GB
    print("Memory limit reset to 16GB")


def example_distributed_operations():
    """Demonstrate distributed data operations"""
    print("\n=== Distributed Operations Examples ===")
    
    client = DataServer("localhost:9015")
    
    print("\n1. Store Data for Distribution:")
    # Store some data
    user_profile = {
        "user_id": "user_456",
        "preferences": {"theme": "dark", "language": "en"},
        "activity_log": list(range(1000))  # Simulate large data
    }
    client.write("user:profile:456", user_profile, persist=True)
    
    # Send to another server (this would fail if target doesn't exist)
    print("\n2. Send Data to Another Server:")
    try:
        success = client.send("user:profile:456", "localhost:9016")
        print(f"Data sent to remote server: {success}")
    except Exception as e:
        print(f"Send operation failed (expected if target server not running): {e}")
    
    # Simulate replication to multiple nodes
    targets = ["localhost:9016", "localhost:9017", "localhost:9018"]
    print(f"\n3. Replicate to Multiple Nodes:")
    for target in targets:
        try:
            success = client.send("user:profile:456", target)
            print(f"  -> {target}: {'Success' if success else 'Failed'}")
        except Exception as e:
            print(f"  -> {target}: Failed ({e})")
    
    # Cleanup
    client.release("user:profile:456")
    print("Distributed operations example completed")


def start_server_if_needed():
    """Start the data server if it's not already running"""
    import socket
    
    # Check if server is already running
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 9015))
    sock.close()
    
    if result != 0:
        print("Starting data server...")
        def run_server():
            start_server(host="0.0.0.0", port=9015, max_mem_size=16*1024*1024*1024)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        print("Data server started")
    else:
        print("Data server already running")


def main():
    """Run all examples"""
    print("gswarm Data API Examples")
    print("=" * 40)
    
    # Start server if needed
    start_server_if_needed()
    
    try:
        # Run examples
        example_kv_storage()
        example_data_pool()
        example_memory_management()
        example_distributed_operations()
        
        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
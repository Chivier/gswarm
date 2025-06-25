"""Example usage of the unified KV storage and data pool system"""

import time
import threading
import requests
from gswarm.data import DataServer, start_server


def start_server_in_background():
    """Start the unified data server in a background thread"""

    def run_server():
        # Now starts both KV storage and data pool APIs on port 9015
        start_server(host="localhost", port=9015, max_mem_size=1024 * 1024 * 1024)  # 1GB for demo

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Give server time to start


def main():
    # Start unified server (in production you would run this separately)
    print("Starting unified data server (KV storage + data pool)...")
    start_server_in_background()

    # Create client connection - now handles both KV and data pool operations
    data_url = "localhost:9015"
    data_server = DataServer(data_url)

    print("\n=== KV Storage Operations ===")

    # Write various types of data
    print("Writing data...")

    # Simple values
    data_server.write("user:123", {"name": "Alice", "age": 30}, persist=True)
    data_server.write("session:abc", {"token": "xyz789", "expires": 1234567890}, persist=False)
    data_server.write("config:app", {"debug": True, "max_connections": 100}, persist=True)

    # Complex nested data
    model_config = {
        "model_name": "llama-7b",
        "parameters": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9},
        "device_config": {"primary": "gpu:0", "fallback": "cpu"},
    }
    data_server.write("model:llama-7b", model_config, persist=True)

    # Read data
    print("\nReading data...")
    user_data = data_server.read("user:123")
    print(f"User data: {user_data}")

    session_data = data_server.read("session:abc")
    print(f"Session data: {session_data}")

    model_data = data_server.read("model:llama-7b")
    print(f"Model config: {model_data['model_name']} with {model_data['parameters']['max_tokens']} tokens")

    # Check stats
    print("\nStorage stats:")
    stats = data_server.get_stats()
    if stats:
        print(f"Memory usage: {stats['usage_percent']:.1f}%")
        print(f"Total keys: {stats['total_keys']}")
        print(f"Persistent keys: {stats['persistent_keys']}")
        print(f"Volatile keys: {stats['volatile_keys']}")
        print(f"Storage: {stats['current_size'] / (1024 * 1024):.2f} MB")

    print("\n=== Data Pool Operations ===")

    # Now demonstrate data pool operations using the same server
    base_url = f"http://{data_url}"

    # Create some data chunks
    print("Creating data chunks...")

    chunk_requests = [
        {"source": "/data/training_batch_1.tensor", "device": "dram", "type": "input", "format": "tensor"},
        {"source": "/models/weights.bin", "device": "gpu", "type": "intermediate", "format": "binary"},
    ]

    created_chunks = []
    for i, chunk_data in enumerate(chunk_requests):
        try:
            response = requests.post(f"{base_url}/api/v1/data", json=chunk_data)
            if response.status_code == 200:
                result = response.json()
                chunk_id = result["chunk_id"]
                created_chunks.append(chunk_id)
                print(f"Created chunk {i + 1}: {chunk_id}")
        except Exception as e:
            print(f"Error creating chunk {i + 1}: {e}")

    # List all chunks
    print(f"\nListing data chunks...")
    try:
        response = requests.get(f"{base_url}/api/v1/data")
        chunks = response.json()["chunks"]
        print(f"Found {len(chunks)} total chunks")

        for chunk in chunks[:3]:  # Show first 3
            print(
                f"  - {chunk['chunk_id']}: {chunk['chunk_type']} on {chunk['locations'][0]['device'] if chunk['locations'] else 'unknown'}"
            )
    except Exception as e:
        print(f"Error listing chunks: {e}")

    # Get detailed info for first created chunk
    if created_chunks:
        chunk_id = created_chunks[0]
        print(f"\nGetting details for chunk {chunk_id}...")
        try:
            response = requests.get(f"{base_url}/api/v1/data/{chunk_id}")
            if response.status_code == 200:
                chunk = response.json()
                print(f"  Type: {chunk['chunk_type']}")
                print(f"  Size: {chunk['size'] / (1024 * 1024):.2f} MB")
                print(f"  Format: {chunk['format']}")
        except Exception as e:
            print(f"Error getting chunk details: {e}")

    print("\n=== Integration Example ===")

    # Store data pool chunk IDs in KV storage for easy reference
    if created_chunks:
        chunk_registry = {
            "training_chunks": created_chunks,
            "created_at": time.time(),
            "total_size_mb": len(created_chunks) * 1.0,  # Mock size
        }
        data_server.write("registry:chunks", chunk_registry, persist=True)
        print("Stored chunk registry in KV storage")

        # Retrieve and use
        registry = data_server.read("registry:chunks")
        print(f"Retrieved registry with {len(registry['training_chunks'])} chunks")

    # Test sending data to another server (would need another server running)
    # data_server.send("user:123", "localhost:9016")

    # Cleanup
    print("\nCleaning up...")

    # Release KV data
    data_server.release("session:abc")
    print("Released volatile session data")

    # Delete data chunks
    for chunk_id in created_chunks:
        try:
            response = requests.delete(f"{base_url}/api/v1/data/{chunk_id}")
            if response.status_code == 200:
                print(f"Deleted chunk: {chunk_id}")
        except Exception as e:
            print(f"Error deleting chunk {chunk_id}: {e}")

    # Final stats
    final_stats = data_server.get_stats()
    if final_stats:
        print(f"\nFinal memory usage: {final_stats['usage_percent']:.1f}%")

    print("\n=== Example completed ===")
    print("✅ Unified data server provides both KV storage and data pool APIs")
    print("🚀 All operations now available on single port 9015")


if __name__ == "__main__":
    main()

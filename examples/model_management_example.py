"""
Enhanced model management example using the unified data API.
Demonstrates how to use both KV storage and data pool for model lifecycle management.
"""

import time
import threading
import requests
import json
from typing import Dict, List, Optional
from gswarm.data import DataServer, start_server


class ModelManager:
    """Model management using unified data API"""

    def __init__(self, data_server: DataServer, base_url: str):
        self.data_server = data_server
        self.base_url = base_url

    def register_model(self, model_name: str, model_info: Dict) -> bool:
        """Register model metadata in KV storage"""
        key = f"model:registry:{model_name}"
        return self.data_server.write(key, model_info, persist=True)

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information from KV storage"""
        key = f"model:registry:{model_name}"
        return self.data_server.read(key)

    def store_model_weights(self, model_name: str, source_path: str, device: str = "disk") -> Optional[str]:
        """Store model weights as data chunk"""
        chunk_data = {
            "source": source_path,
            "device": device,
            "type": "intermediate",
            "format": "binary",
            "metadata": {"model_name": model_name, "content_type": "model_weights"},
        }

        try:
            response = requests.post(f"{self.base_url}/api/v1/data", json=chunk_data)
            if response.status_code == 200:
                return response.json()["chunk_id"]
        except Exception as e:
            print(f"Error storing model weights: {e}")
        return None

    def load_model_to_device(self, chunk_id: str, target_device: str) -> bool:
        """Move model weights to specific device"""
        move_data = {"target_device": target_device, "priority": "high"}

        try:
            response = requests.post(f"{self.base_url}/api/v1/data/{chunk_id}/move", json=move_data)
            return response.status_code == 200
        except Exception as e:
            print(f"Error moving model to device: {e}")
        return False

    def list_models(self) -> List[Dict]:
        """List all registered models"""
        # This is a simplified implementation
        # In practice, you might want to maintain a model index
        models = []

        # Get model info from KV storage
        # We'll look for keys with pattern "model:registry:*"
        # Note: This is a demonstration - real implementation would use a proper index

        return models

    def create_model_session(self, model_name: str, session_config: Dict) -> str:
        """Create a model inference session"""
        session_id = f"session_{int(time.time())}_{model_name}"
        session_data = {
            "model_name": model_name,
            "config": session_config,
            "created_at": time.time(),
            "status": "active",
        }

        key = f"session:{session_id}"
        if self.data_server.write(key, session_data, persist=False):  # Volatile session
            return session_id
        return None


def start_server_in_background():
    """Start the unified data server"""

    def run_server():
        start_server(host="localhost", port=9015, max_mem_size=2 * 1024 * 1024 * 1024)  # 2GB

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)


def main():
    print("🤖 Model Management with Unified Data API Example")
    print("=" * 60)

    # Start server
    print("Starting data server...")
    start_server_in_background()

    # Initialize clients
    data_server = DataServer("localhost:9015")
    model_manager = ModelManager(data_server, "http://localhost:9015")

    print("\n=== Model Registration ===")

    # Register some models
    models_to_register = [
        {
            "name": "llama-7b",
            "info": {
                "model_type": "llm",
                "size_gb": 13.5,
                "parameters": "7B",
                "architecture": "transformer",
                "supported_devices": ["cpu", "gpu"],
                "memory_requirements": {"minimum_gpu_memory": "8GB", "recommended_gpu_memory": "16GB"},
                "capabilities": ["text_generation", "chat", "completion"],
            },
        },
        {
            "name": "stable-diffusion-xl",
            "info": {
                "model_type": "diffusion",
                "size_gb": 6.9,
                "parameters": "3.5B",
                "architecture": "unet",
                "supported_devices": ["gpu"],
                "memory_requirements": {"minimum_gpu_memory": "8GB", "recommended_gpu_memory": "12GB"},
                "capabilities": ["image_generation", "inpainting"],
            },
        },
        {
            "name": "whisper-large",
            "info": {
                "model_type": "speech",
                "size_gb": 2.9,
                "parameters": "1.5B",
                "architecture": "transformer",
                "supported_devices": ["cpu", "gpu"],
                "memory_requirements": {"minimum_gpu_memory": "4GB", "recommended_gpu_memory": "8GB"},
                "capabilities": ["speech_to_text", "translation"],
            },
        },
    ]

    registered_models = []
    for model in models_to_register:
        success = model_manager.register_model(model["name"], model["info"])
        if success:
            registered_models.append(model["name"])
            print(f"✅ Registered {model['name']} ({model['info']['size_gb']} GB)")
        else:
            print(f"❌ Failed to register {model['name']}")

    print("\n=== Model Weight Storage ===")

    # Store model weights in data pool
    model_chunks = {}
    for model_name in registered_models[:2]:  # Store weights for first 2 models
        source_path = f"/models/{model_name}/pytorch_model.bin"
        chunk_id = model_manager.store_model_weights(model_name, source_path, "disk")

        if chunk_id:
            model_chunks[model_name] = chunk_id
            print(f"📦 Stored weights for {model_name}: {chunk_id}")

            # Update model registry with chunk ID
            model_info = model_manager.get_model_info(model_name)
            if model_info:
                model_info["weight_chunk_id"] = chunk_id
                model_manager.register_model(model_name, model_info)
                print(f"   Updated registry with chunk ID")

    print("\n=== Model Loading Simulation ===")

    # Simulate loading models to different devices
    if "llama-7b" in model_chunks:
        chunk_id = model_chunks["llama-7b"]

        print(f"🚚 Loading llama-7b to GPU...")
        success = model_manager.load_model_to_device(chunk_id, "gpu:0")
        if success:
            print("   ✅ Model loaded to GPU successfully")

        # Get updated chunk info
        try:
            response = requests.get(f"http://localhost:9015/api/v1/data/{chunk_id}")
            if response.status_code == 200:
                chunk = response.json()
                devices = [loc["device"] for loc in chunk["locations"]]
                print(f"   📍 Model now available on: {', '.join(devices)}")
        except Exception as e:
            print(f"   Error getting chunk info: {e}")

    print("\n=== Session Management ===")

    # Create inference sessions
    sessions = []

    session_configs = [
        {"model": "llama-7b", "config": {"max_tokens": 2048, "temperature": 0.7, "top_p": 0.9, "device": "gpu:0"}},
        {"model": "whisper-large", "config": {"language": "en", "task": "transcribe", "device": "cpu"}},
    ]

    for session_cfg in session_configs:
        session_id = model_manager.create_model_session(session_cfg["model"], session_cfg["config"])
        if session_id:
            sessions.append(session_id)
            print(f"🎯 Created session for {session_cfg['model']}: {session_id}")

    print("\n=== Resource Monitoring ===")

    # Check storage usage
    stats = data_server.get_stats()
    if stats:
        print(f"💾 Storage usage: {stats['usage_percent']:.1f}%")
        print(
            f"📊 Total keys: {stats['total_keys']} (persistent: {stats['persistent_keys']}, volatile: {stats['volatile_keys']})"
        )
        print(f"💿 Memory: {stats['current_size'] / (1024 * 1024):.2f} MB used")

    # List data chunks (models)
    try:
        response = requests.get("http://localhost:9015/api/v1/data")
        chunks = response.json()["chunks"]
        print(f"📦 Data chunks: {len(chunks)} total")

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if metadata.get("content_type") == "model_weights":
                model_name = metadata.get("model_name", "unknown")
                size_mb = chunk["size"] / (1024 * 1024)
                devices = [loc["device"] for loc in chunk["locations"]]
                print(f"   🤖 {model_name}: {size_mb:.1f} MB on {', '.join(devices)}")
    except Exception as e:
        print(f"Error listing chunks: {e}")

    print("\n=== Model Information Retrieval ===")

    # Demonstrate retrieving model information
    for model_name in registered_models:
        info = model_manager.get_model_info(model_name)
        if info:
            print(f"🔍 {model_name}:")
            print(f"   Type: {info['model_type']}")
            print(f"   Size: {info['size_gb']} GB")
            print(f"   Capabilities: {', '.join(info['capabilities'])}")
            if "weight_chunk_id" in info:
                print(f"   Weights: {info['weight_chunk_id']}")

    print("\n=== Cleanup ===")

    # Clean up sessions (they're volatile, so will be cleaned automatically)
    for session_id in sessions:
        data_server.release(session_id)
        print(f"🧹 Cleaned up session: {session_id}")

    # Optionally clean up model chunks
    cleanup_chunks = input("\nDelete model weight chunks? (y/N): ").lower().strip() == "y"
    if cleanup_chunks:
        for model_name, chunk_id in model_chunks.items():
            try:
                response = requests.delete(f"http://localhost:9015/api/v1/data/{chunk_id}")
                if response.status_code == 200:
                    print(f"🗑️  Deleted weights for {model_name}")
            except Exception as e:
                print(f"Error deleting chunk for {model_name}: {e}")

    print("\n" + "=" * 60)
    print("✅ Model management example completed!")
    print("💡 This example shows how to use the unified data API for:")
    print("   • Model metadata storage (KV storage)")
    print("   • Model weight storage (data pool)")
    print("   • Device placement and loading")
    print("   • Session management")
    print("   • Resource monitoring")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Replica Model Serving Load Test Example

This script demonstrates:
1. Downloading 3 different models
2. Loading them to specific GPUs
3. Setting up multiple replicas across different GPUs
4. Continuous load testing to keep GPUs busy
"""

import requests
import json
import time
import random
import threading
import signal
import sys
import torch
from typing import List, Dict, Any
from datetime import datetime, timedelta
import concurrent.futures
from dataclasses import dataclass
from cost import get_estimation_cost

# Configuration
SERVER_URL = "http://localhost:8000"
LOAD_TEST_DURATION = 300  # 5 minutes in seconds
REQUEST_INTERVAL_RANGE = (0.1, 2.0)  # Random interval between requests (seconds)

# Detect available GPUs
def get_available_gpus():
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []

AVAILABLE_GPUS = get_available_gpus()
print(f"Available GPUs: {AVAILABLE_GPUS}")

# Model configuration
MODELS = {
    "A": "gpt2",  # Small and fast
    "B": "distilgpt2",  # Distilled version
    "C": "microsoft/DialoGPT-small"  # Another small model
}

# Dynamic replica configuration based on available GPUs
def create_replica_config():
    if len(AVAILABLE_GPUS) == 0:
        print("❌ No GPUs available! This example requires CUDA GPUs.")
        sys.exit(1)
    elif len(AVAILABLE_GPUS) == 1:
        # Single GPU setup
        return {
            "A": [{"device": f"cuda:{AVAILABLE_GPUS[0]}", "count": 2}],  
            "B": [{"device": f"cuda:{AVAILABLE_GPUS[0]}", "count": 1}],  
            "C": [{"device": f"cuda:{AVAILABLE_GPUS[0]}", "count": 1}]   
        }
    elif len(AVAILABLE_GPUS) == 2:
        # Dual GPU setup
        return {
            "A": [{"device": f"cuda:{AVAILABLE_GPUS[0]}", "count": 2}],  
            "B": [{"device": f"cuda:{AVAILABLE_GPUS[1]}", "count": 1}],  
            "C": [{"device": f"cuda:{AVAILABLE_GPUS[1]}", "count": 1}]   
        }
    else:
        # Multi-GPU setup (3+ GPUs)
        return {
            "A": [{"device": f"cuda:{AVAILABLE_GPUS[0]}", "count": 2}],  
            "B": [{"device": f"cuda:{AVAILABLE_GPUS[1]}", "count": 1}],  
            "C": [{"device": f"cuda:{AVAILABLE_GPUS[2]}", "count": 1}]   
        }

REPLICAS = create_replica_config()

# Print configuration
print("Configuration:")
for model_key, model_name in MODELS.items():
    print(f"  Model {model_key}: {model_name}")
for model_key, replica_configs in REPLICAS.items():
    for config in replica_configs:
        print(f"  Model {model_key} replicas: {config['count']} on {config['device']}")

@dataclass
class Instance:
    """Represents a model serving instance"""
    instance_id: str
    model_name: str
    device: str
    port: int
    url: str
    endpoint: str

class ModelServerClient:
    """Client for interacting with the standalone model server"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.instances: List[Instance] = []
        self.stop_load_test = False
        
    def check_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def download_model(self, model_name: str) -> bool:
        """Download a model from HuggingFace"""
        print(f"📥 Downloading model: {model_name}")
        try:
            response = requests.post(
                f"{self.server_url}/standalone/download",
                json={"model_name": model_name, "source": "huggingface"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Downloaded {model_name}: {result['message']}")
                return True
            else:
                print(f"❌ Failed to download {model_name}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Download error for {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str, device: str = None) -> bool:
        """Load model to DRAM on specific device"""
        device_info = f" on {device}" if device else ""
        print(f"🧠 Loading model to DRAM: {model_name}{device_info}")
        try:
            payload = {"model_name": model_name, "target": "dram"}
            if device:
                payload["device"] = device
                
            response = requests.post(
                f"{self.server_url}/standalone/load",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                actual_device = result.get('data', {}).get('device', 'unknown')
                print(f"✅ Loaded {model_name} on {actual_device}: {result['message']}")
                return True
            else:
                print(f"❌ Failed to load {model_name}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Load error for {model_name}: {e}")
            return False
    
    def serve_model(self, model_name: str, device: str) -> Instance:
        """Create a serving instance"""
        print(f"🚀 Starting serving instance: {model_name} on {device}")
        try:
            response = requests.post(
                f"{self.server_url}/standalone/serve",
                json={"model_name": model_name, "device": device}
            )
            
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                instance = Instance(
                    instance_id=data['instance_id'],
                    model_name=data['model_name'],
                    device=data['device'],
                    port=data['port'],
                    url=data['url'],
                    endpoint=data['endpoint']
                )
                
                self.instances.append(instance)
                print(f"✅ Started instance {instance.instance_id} for {model_name} on {device}:{instance.port}")
                return instance
                
            else:
                print(f"❌ Failed to serve {model_name}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Serve error for {model_name}: {e}")
            return None
    
    def call_model(self, instance: Instance, prompt: str) -> Dict[str, Any]:
        """Send inference request to model instance"""
        try:
            response = requests.post(
                f"{self.server_url}{instance.endpoint}",
                json={
                    "instance_id": instance.instance_id,
                    "data": {
                        "prompt": prompt,
                        "max_length": random.randint(30, 100),
                        "temperature": random.uniform(0.7, 1.0)
                    }
                },
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                device_used = result.get('data', {}).get('device_used', instance.device)
                return {
                    "success": True,
                    "response": result.get('data', {}).get('response', ''),
                    "instance_id": instance.instance_id,
                    "model": instance.model_name,
                    "device": device_used
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "instance_id": instance.instance_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "instance_id": instance.instance_id
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        try:
            response = requests.get(f"{self.server_url}/standalone/status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

def generate_random_prompts() -> List[str]:
    """Generate random prompts for testing"""
    prompts = [
        "The future of artificial intelligence",
        "What is machine learning",
        "Explain deep learning",
        "How does a neural network work",
        "What are transformers in AI",
        "Tell me about GPUs",
        "What is distributed computing",
        "How do language models work",
        "What is natural language processing",
        "Explain computer vision",
        "What is reinforcement learning",
        "How does backpropagation work",
        "What are embeddings",
        "Tell me about attention mechanisms",
        "What is transfer learning",
        "How do CNNs work",
        "What is gradient descent",
        "Explain overfitting",
        "What is regularization",
        "How do RNNs work"
    ]
    return prompts

def load_test_worker(client: ModelServerClient, worker_id: int, prompts: List[str]):
    """Worker function for load testing"""
    print(f"🔄 Load test worker {worker_id} started")
    request_count = 0
    success_count = 0
    
    while not client.stop_load_test:
        try:
            # Select random instance and prompt
            if not client.instances:
                time.sleep(1)
                continue
                
            instance = random.choice(client.instances)
            prompt = random.choice(prompts)
            
            # Send request
            start_time = time.time()
            result = client.call_model(instance, prompt)
            duration = time.time() - start_time
            
            request_count += 1
            if result['success']:
                success_count += 1
                print(f"⚡ Worker-{worker_id} Request-{request_count}: {instance.model_name}[{instance.instance_id}] "
                      f"on {instance.device} ({duration:.2f}s) - SUCCESS")
            else:
                print(f"❌ Worker-{worker_id} Request-{request_count}: {instance.model_name}[{instance.instance_id}] "
                      f"on {instance.device} ({duration:.2f}s) - FAILED: {result['error']}")
            
            # Random delay between requests
            delay = random.uniform(*REQUEST_INTERVAL_RANGE)
            time.sleep(delay)
            
        except Exception as e:
            print(f"❌ Worker-{worker_id} error: {e}")
            time.sleep(1)
    
    print(f"🔄 Load test worker {worker_id} finished: {success_count}/{request_count} successful")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal, stopping load test...")
    sys.exit(0)

def main():
    """Main execution function"""
    print("🚀 GSwarm Multi-Replica Load Test Example")
    print("=" * 60)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize client
    client = ModelServerClient(SERVER_URL)
    
    # Step 1: Health check
    print("\n1️⃣ Checking server health...")
    if not client.check_health():
        print("❌ Server is not healthy. Please start the server first:")
        print("   python serve.py --host localhost --port 8000")
        return
    print("✅ Server is healthy")
    
    # Step 2: Download models
    print("\n2️⃣ Downloading models...")
    for model_key, model_name in MODELS.items():
        print(f"\nDownloading Model {model_key}: {model_name}")
        if not client.download_model(model_name):
            print(f"❌ Failed to download {model_name}")
            return
    
    # Step 3: Load models to DRAM on specific devices
    print("\n3️⃣ Loading models to DRAM on specific GPUs...")
    model_to_device = {}
    
    for model_key, replica_configs in REPLICAS.items():
        model_name = MODELS[model_key]
        # Use the device from the first replica config for loading
        target_device = replica_configs[0]["device"]
        model_to_device[model_key] = target_device
        
        print(f"\nLoading Model {model_key} ({model_name}) to {target_device}")
        if not client.load_model(model_name, target_device):
            print(f"❌ Failed to load {model_name} to {target_device}")
            return
    
    # Step 4: Create serving instances
    print("\n4️⃣ Creating serving instances...")
    total_instances = 0
    
    for model_key, replica_configs in REPLICAS.items():
        model_name = MODELS[model_key]
        print(f"\nSetting up replicas for Model {model_key} ({model_name}):")
        
        for config in replica_configs:
            device = config["device"]
            count = config["count"]
            
            for i in range(count):
                instance = client.serve_model(model_name, device)
                if instance:
                    total_instances += 1
                    print(f"  ✅ Replica {i+1}/{count} on {device}: {instance.instance_id}")
                else:
                    print(f"  ❌ Failed to create replica {i+1}/{count} on {device}")
    
    print(f"\n✅ Successfully created {total_instances} serving instances")
    
    # Step 5: Display status
    print("\n5️⃣ Current server status:")
    status = client.get_status()
    if "error" not in status:
        print(f"  Models on disk: {len(status['models_on_disk'])}")
        print(f"  Models in DRAM: {len(status['models_in_dram'])}")
        print(f"  Serving instances: {len(status['serving_instances'])}")
        print(f"  Used ports: {status['used_ports']}")
        
        print("\n  Active instances:")
        for instance_info in status['serving_instances']:
            print(f"    {instance_info['instance_id']}: {instance_info['model_name']} "
                  f"on {instance_info['device']}:{instance_info['port']}")
    
    # Step 6: Wait for user confirmation
    print(f"\n6️⃣ Setup complete! Ready to start load testing.")
    print("   Current setup:")
    for model_key, replica_configs in REPLICAS.items():
        model_name = MODELS[model_key]
        for config in replica_configs:
            print(f"   - Model {model_key} ({model_name}): {config['count']} replica(s) on {config['device']}")
    
    print(f"\n⏸️  Type 'Y' and press Enter to start the {LOAD_TEST_DURATION//60}-minute load test...")
    
    user_input = input().strip().upper()
    if user_input != 'Y':
        print("❌ Load test cancelled.")
        return
    
    # Step 7: Start load testing
    print(f"\n7️⃣ Starting load test for {LOAD_TEST_DURATION} seconds...")
    print(f"   Request interval: {REQUEST_INTERVAL_RANGE[0]}-{REQUEST_INTERVAL_RANGE[1]} seconds")
    print(f"   Total instances: {len(client.instances)}")
    
    # Generate prompts
    prompts = generate_random_prompts()
    
    # Start load test workers
    num_workers = min(8, len(client.instances) * 2)  # 2 workers per instance, max 8
    print(f"   Starting {num_workers} load test workers...")
    
    start_time = time.time()
    end_time = start_time + LOAD_TEST_DURATION
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit worker tasks
        futures = []
        for i in range(num_workers):
            future = executor.submit(load_test_worker, client, i+1, prompts)
            futures.append(future)
        
        # Monitor progress
        try:
            while time.time() < end_time:
                remaining = end_time - time.time()
                print(f"⏱️  Load test running... {remaining:.0f} seconds remaining")
                time.sleep(30)  # Update every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Load test interrupted by user")
        
        # Stop load test
        client.stop_load_test = True
        
        # Wait for workers to finish
        print("🔄 Waiting for workers to finish...")
        concurrent.futures.wait(futures, timeout=10)
    
    total_duration = time.time() - start_time
    print(f"\n✅ Load test completed after {total_duration:.1f} seconds")
    
    # Final status
    print("\n8️⃣ Final server status:")
    final_status = client.get_status()
    if "error" not in final_status:
        print(f"  Active instances: {len(final_status['serving_instances'])}")
        print(f"  Used ports: {final_status['used_ports']}")
    
    print("\n🎉 Multi-replica load test example completed!")
    print("   Check GPU utilization with: nvidia-smi")
    print("   Check server logs: standalone_server.log")

if __name__ == "__main__":
    main() 
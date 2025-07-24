#!/usr/bin/env python3
"""
Test script for custom model services
"""

import asyncio
import requests
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from gswarm.model.service_manager import ServiceManager, ServiceConfig
from gswarm.model.examples.simple_service import SimpleMLService


async def test_service_lifecycle():
    """Test basic service lifecycle"""
    print("=== Testing Service Lifecycle ===")
    
    # Create service manager
    manager = ServiceManager(port_range=(8100, 8200))
    
    # Register a service
    config = ServiceConfig(
        name="test-simple",
        model_path="dummy.pkl",
        device="cpu",
        extra_args={"feature_dim": 5}
    )
    
    print(f"1. Registering service...")
    service_info = manager.register_service("test-simple", "simple", config)
    print(f"   ✓ Service registered on port {service_info.config.port}")
    
    # Start the service
    print(f"2. Starting service...")
    await manager.start_service("test-simple")
    print(f"   ✓ Service started with PID {service_info.pid}")
    
    # Wait for service to be ready
    time.sleep(3)
    
    # Test the service
    print(f"3. Testing service endpoints...")
    base_url = f"http://localhost:{service_info.config.port}"
    
    # Health check
    response = requests.get(f"{base_url}/")
    print(f"   Health check: {response.json()}")
    
    # Status
    response = requests.get(f"{base_url}/status")
    print(f"   Status: {response.json()}")
    
    # Inference
    inference_data = {
        "inputs": [1.0, 2.0, 3.0, 4.0, 5.0],
        "parameters": {"normalize": True, "activation": "sigmoid"}
    }
    response = requests.post(f"{base_url}/inference", json=inference_data)
    print(f"   Inference result: {response.json()}")
    
    # Stop the service
    print(f"4. Stopping service...")
    await manager.stop_service("test-simple")
    print(f"   ✓ Service stopped")
    
    # Unregister
    print(f"5. Unregistering service...")
    manager.unregister_service("test-simple")
    print(f"   ✓ Service unregistered")
    
    print("\n✅ Service lifecycle test completed successfully!")


async def test_multiple_services():
    """Test multiple services running concurrently"""
    print("\n=== Testing Multiple Services ===")
    
    manager = ServiceManager(port_range=(8200, 8300))
    
    # Register multiple services
    services = []
    for i in range(3):
        config = ServiceConfig(
            name=f"service-{i}",
            model_path="dummy.pkl",
            device="cpu",
            extra_args={"feature_dim": 10}
        )
        service_info = manager.register_service(f"service-{i}", "simple", config)
        services.append(service_info)
        print(f"Registered service-{i} on port {service_info.config.port}")
    
    # Start all services
    for service in services:
        await manager.start_service(service.name)
        print(f"Started {service.name}")
    
    # Wait for services to be ready
    time.sleep(3)
    
    # Test all services
    for service in services:
        base_url = f"http://localhost:{service.config.port}"
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        print(f"{service.name} is healthy")
    
    # Stop all services
    for service in services:
        await manager.stop_service(service.name)
        manager.unregister_service(service.name)
        print(f"Stopped and unregistered {service.name}")
    
    print("\n✅ Multiple services test completed successfully!")


async def test_cli_commands():
    """Test CLI commands"""
    print("\n=== Testing CLI Commands ===")
    
    import subprocess
    
    # Test service registration via CLI
    cmd = [
        "python", "-m", "gswarm.model.cli",
        "service-register", "cli-test",
        "--type", "simple",
        "--model", "dummy.pkl",
        "--device", "cpu",
        "--args", '{"feature_dim": 8}'
    ]
    
    print("1. Registering service via CLI...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"   Output: {result.stdout.strip()}")
    
    # List services
    cmd = ["python", "-m", "gswarm.model.cli", "service-list"]
    print("\n2. Listing services...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    # Get status
    cmd = ["python", "-m", "gswarm.model.cli", "service-status", "cli-test"]
    print("3. Getting service status...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    # Cleanup
    cmd = ["python", "-m", "gswarm.model.cli", "service-unregister", "cli-test"]
    subprocess.run(cmd, capture_output=True)
    print("\n✅ CLI test completed!")


async def main():
    """Run all tests"""
    print("Starting Custom Model Service Tests\n")
    
    try:
        await test_service_lifecycle()
        await test_multiple_services()
        await test_cli_commands()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
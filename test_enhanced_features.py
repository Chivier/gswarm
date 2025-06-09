#!/usr/bin/env python3
"""
Test script for GSwarm enhanced model management features.
This script demonstrates the new functionality including:
- Configuration loading
- Model discovery
- DRAM/GPU variable storage
- Enhanced HuggingFace integration
"""

import asyncio
import json
import requests
import time
from pathlib import Path

# Test configuration
API_BASE = "http://localhost:8100"
TEST_MODEL = "microsoft/DialoGPT-small"  # Small model for testing


def test_api_connection():
    """Test basic API connectivity"""
    print("🔍 Testing API connection...")
    try:
        response = requests.get(f"{API_BASE}/")
        response.raise_for_status()
        data = response.json()
        print(f"✅ API connected: {data['message']} v{data.get('version', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


def test_configuration():
    """Test configuration endpoint"""
    print("\n🔍 Testing configuration...")
    try:
        response = requests.get(f"{API_BASE}/config")
        response.raise_for_status()
        config = response.json()
        
        print(f"✅ Configuration loaded:")
        print(f"   📁 Model cache: {config['model_cache_dir']}")
        print(f"   📁 DRAM cache: {config['dram_cache_dir']}")
        print(f"   🔎 Auto-discover: {config['auto_discover_models']}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_health_check():
    """Test enhanced health check with memory usage"""
    print("\n🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        response.raise_for_status()
        health = response.json()
        
        print(f"✅ Health check passed:")
        print(f"   📊 Status: {health['status']}")
        print(f"   📊 DRAM models: {health['memory_usage']['dram_models']}")
        print(f"   📊 GPU models: {health['memory_usage']['gpu_models']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_model_discovery():
    """Test model discovery functionality"""
    print("\n🔍 Testing model discovery...")
    try:
        response = requests.post(f"{API_BASE}/discover")
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Model discovery completed:")
        print(f"   📚 Total models: {result['data']['total_models']}")
        print(f"   🧠 DRAM models: {result['data']['dram_models']}")
        print(f"   🖥️  GPU models: {result['data']['gpu_models']}")
        return True
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        return False


def register_test_model():
    """Register a test model"""
    print(f"\n🔍 Registering test model: {TEST_MODEL}...")
    try:
        data = {
            "name": TEST_MODEL,
            "type": "llm",
            "metadata": {"test": True, "source": "huggingface"}
        }
        response = requests.post(f"{API_BASE}/models", json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["success"]:
            print(f"✅ Model registered: {TEST_MODEL}")
        else:
            print(f"⚠️  Model already exists: {TEST_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return False


def test_download_model():
    """Test downloading a model using HF integration"""
    print(f"\n🔍 Testing model download: {TEST_MODEL}...")
    try:
        data = {
            "model_name": TEST_MODEL,
            "source_url": f"hf://{TEST_MODEL}",
            "target_device": "disk"
        }
        response = requests.post(f"{API_BASE}/download", json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["success"]:
            print(f"✅ Download started: {result['message']}")
            return True
        else:
            print(f"❌ Download failed: {result['message']}")
            return False
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return False


def wait_for_download():
    """Wait for download to complete"""
    print("\n⏳ Waiting for download to complete...")
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{API_BASE}/models/{TEST_MODEL}")
            if response.status_code == 200:
                model = response.json()
                if model["status"] == "ready":
                    print("✅ Download completed!")
                    return True
                elif model["status"] == "error":
                    print("❌ Download failed!")
                    return False
                else:
                    print(f"⏳ Status: {model['status']}")
            
            time.sleep(10)
        except Exception as e:
            print(f"⚠️  Error checking status: {e}")
            time.sleep(10)
    
    print("❌ Download timeout!")
    return False


def test_dram_loading():
    """Test loading model to DRAM with variables"""
    print(f"\n🔍 Testing DRAM loading: {TEST_MODEL}...")
    try:
        data = {
            "model_name": TEST_MODEL,
            "source_device": "disk",
            "target_device": "dram"
        }
        response = requests.post(f"{API_BASE}/copy", json=data)
        response.raise_for_status()
        result = response.json()
        
        if result["success"]:
            print(f"✅ DRAM loading started: {result['message']}")
            
            # Wait a bit for the operation to complete
            time.sleep(5)
            
            # Check DRAM models
            response = requests.get(f"{API_BASE}/memory/dram")
            response.raise_for_status()
            dram_models = response.json()
            
            print(f"✅ DRAM models: {dram_models['count']} total")
            for model in dram_models['dram_models']:
                print(f"   🧠 {model['model_name']}: {model['type']} (loaded: {model['loaded']})")
            
            return True
        else:
            print(f"❌ DRAM loading failed: {result['message']}")
            return False
    except Exception as e:
        print(f"❌ DRAM loading test failed: {e}")
        return False


def test_memory_endpoints():
    """Test memory management endpoints"""
    print("\n🔍 Testing memory endpoints...")
    try:
        # Test memory overview
        response = requests.get(f"{API_BASE}/memory/models")
        response.raise_for_status()
        memory = response.json()
        
        print(f"✅ Memory overview:")
        print(f"   🧠 DRAM models: {memory['total_dram_models']}")
        print(f"   🖥️  GPU models: {memory['total_gpu_models']}")
        
        # Test DRAM details
        response = requests.get(f"{API_BASE}/memory/dram")
        response.raise_for_status()
        dram = response.json()
        
        print(f"✅ DRAM details: {dram['count']} models")
        
        # Test GPU details
        response = requests.get(f"{API_BASE}/memory/gpu")
        response.raise_for_status()
        gpu = response.json()
        
        print(f"✅ GPU details: {gpu['count']} instances")
        return True
    except Exception as e:
        print(f"❌ Memory endpoints test failed: {e}")
        return False


def test_model_list():
    """Test enhanced model listing"""
    print("\n🔍 Testing enhanced model listing...")
    try:
        response = requests.get(f"{API_BASE}/models")
        response.raise_for_status()
        models = response.json()
        
        print(f"✅ Models overview: {models['count']} total")
        for model in models['models']:
            print(f"   📚 {model['name']}:")
            print(f"      📱 Type: {model['type']}")
            print(f"      📦 Status: {model['status']}")
            print(f"      🧠 DRAM loaded: {model['dram_loaded']}")
            print(f"      🖥️  Serving instances: {model['serving_instances']}")
        return True
    except Exception as e:
        print(f"❌ Model listing test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 GSwarm Enhanced Features Test Suite")
    print("=" * 50)
    
    tests = [
        test_api_connection,
        test_configuration,
        test_health_check,
        test_model_discovery,
        register_test_model,
        test_download_model,
        wait_for_download,
        test_dram_loading,
        test_memory_endpoints,
        test_model_list,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! GSwarm enhanced features are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the API server and configuration.")


if __name__ == "__main__":
    main() 
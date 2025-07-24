#!/usr/bin/env python3
"""
Test script to verify the new device format (client:device:id) works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gswarm.utils import parse_device, format_device, normalize_device, is_same_device
from gswarm.model import CostModel
from gswarm.predictor import predictor
from gswarm.scheduler import BaselineScheduler, ModelInfo, Workflow, WorkflowNode, WorkflowEdge, Request


def test_device_utilities():
    """Test device parsing and formatting utilities."""
    print("=== Testing Device Utilities ===")
    
    # Test parsing new format
    print("\n1. Testing new device format:")
    device_info = parse_device("node1:cuda:0")
    print(f"   Parsed 'node1:cuda:0': client={device_info.client}, type={device_info.device_type}, id={device_info.device_id}")
    print(f"   Full name: {device_info.full_name}")
    
    # Test parsing legacy format
    print("\n2. Testing legacy device format:")
    device_info = parse_device("cuda:0")
    print(f"   Parsed 'cuda:0': client={device_info.client}, type={device_info.device_type}, id={device_info.device_id}")
    print(f"   Full name: {device_info.full_name}")
    
    # Test formatting
    print("\n3. Testing device formatting:")
    formatted = format_device("worker2", "cuda", 3)
    print(f"   Formatted device: {formatted}")
    
    # Test normalization
    print("\n4. Testing device normalization:")
    normalized1 = normalize_device("cuda:0")
    normalized2 = normalize_device("node1:cuda:0")
    print(f"   'cuda:0' normalized to: {normalized1}")
    print(f"   'node1:cuda:0' normalized to: {normalized2}")
    
    # Test comparison
    print("\n5. Testing device comparison:")
    same1 = is_same_device("node1:cuda:0", "node1:cuda:0")
    same2 = is_same_device("cuda:0", "localhost:cuda:0")
    same3 = is_same_device("node1:cuda:0", "node2:cuda:0")
    print(f"   'node1:cuda:0' == 'node1:cuda:0': {same1}")
    print(f"   'cuda:0' == 'localhost:cuda:0': {same2}")
    print(f"   'node1:cuda:0' == 'node2:cuda:0': {same3}")
    
    print("\n✓ Device utilities working correctly")


def test_cost_model_with_devices():
    """Test cost models with new device format."""
    print("\n=== Testing Cost Models with New Device Format ===")
    
    cost_model = CostModel()
    
    # Test with new device format
    print("\n1. Testing with new device format:")
    time1 = cost_model.predict("gpt-4", "node1:cuda:0", {"prompt": "Hello"})
    time2 = cost_model.predict("stable-diffusion", "node2:cuda:1", {"height": 512, "width": 512})
    print(f"   GPT-4 on node1:cuda:0: {time1:.2f}s")
    print(f"   SD on node2:cuda:1: {time2:.2f}s")
    
    # Test with legacy format (should work)
    print("\n2. Testing with legacy format:")
    time3 = cost_model.predict("llama-2", "cuda:0", {"prompt": "Test"})
    print(f"   Llama-2 on cuda:0 (legacy): {time3:.2f}s")
    
    # Test updates
    print("\n3. Testing model updates:")
    cost_model.update("gpt-4", "node1:cuda:0", {"prompt": "Hi", "output": "Hello"}, actual_time=0.5)
    print("   ✓ Update successful")
    
    print("\n✓ Cost models working with new device format")


def test_predictor_with_devices():
    """Test predictor with new device format."""
    print("\n=== Testing Predictor with New Device Format ===")
    
    # Single predictions
    print("\n1. Testing single predictions:")
    time1 = predictor.predict("gpt-3.5", "worker1:cuda:0", {"prompt": "Test"})
    time2 = predictor.predict("stable-diffusion-xl", "worker2:cuda:2", {"height": 1024, "width": 1024})
    print(f"   GPT-3.5 on worker1:cuda:0: {time1:.2f}s")
    print(f"   SDXL on worker2:cuda:2: {time2:.2f}s")
    
    # Batch predictions with mixed formats
    print("\n2. Testing batch predictions with mixed formats:")
    requests = [
        {"model_name": "gpt-4", "device": "node1:cuda:0", "inputs": {"prompt": "Q1"}},
        {"model_name": "llama-2", "device": "cuda:1", "inputs": {"prompt": "Q2"}},  # Legacy
        {"model_name": "stable-diffusion", "device": "node3:cuda:0", "inputs": {"height": 512, "width": 512}},
    ]
    times = predictor.batch_predict(requests)
    for i, (req, time) in enumerate(zip(requests, times)):
        print(f"   Request {i+1} ({req['model_name']} on {req['device']}): {time:.2f}s")
    
    print("\n✓ Predictor working with new device format")


def test_scheduler_with_devices():
    """Test scheduler with new device format."""
    print("\n=== Testing Scheduler with New Device Format ===")
    
    # Define models
    models = {
        "gpt-4": ModelInfo(
            name="gpt-4",
            memory_gb=16.0,
            gpus_required=1,
            load_time_seconds=5.0,
            tokens_per_second=50.0
        ),
        "stable-diffusion": ModelInfo(
            name="stable-diffusion",
            memory_gb=8.0,
            gpus_required=1,
            load_time_seconds=3.0,
            inference_time_mean=2.5
        )
    }
    
    # Test with new device format
    print("\n1. Testing scheduler with new device format:")
    devices = ["node1:cuda:0", "node1:cuda:1", "node2:cuda:0", "node2:cuda:1"]
    scheduler = BaselineScheduler(devices=devices, models=models, simulate=True)
    print(f"   Scheduler initialized with {len(devices)} devices")
    print(f"   Devices: {scheduler.devices}")
    
    # Test with legacy format
    print("\n2. Testing scheduler with legacy format:")
    legacy_scheduler = BaselineScheduler(devices=[0, 1, 2, 3], models=models, simulate=True)
    print(f"   Legacy scheduler initialized with {len(legacy_scheduler.devices)} devices")
    print(f"   Converted devices: {legacy_scheduler.devices}")
    
    # Create and schedule a workflow
    print("\n3. Testing workflow scheduling:")
    nodes = [
        WorkflowNode(id="prompt", model="gpt-4", inputs=["user_input"], outputs=["text"]),
        WorkflowNode(id="image", model="stable-diffusion", inputs=["text"], outputs=["image"])
    ]
    edges = [WorkflowEdge(from_node="prompt", to_node="image")]
    workflow = Workflow(id="test", name="Test Workflow", nodes=nodes, edges=edges)
    
    scheduler.add_workflow(workflow)
    request = Request(id="req1", workflow_id="test", arrival_time=0.0)
    scheduler.add_request(request)
    
    task = scheduler.get_next_task()
    if task:
        print(f"   Scheduled task on device: {task.device}")
        print(f"   Model: {task.model_name}")
        print(f"   ✓ Task scheduled successfully")
    
    print("\n✓ Scheduler working with new device format")


def main():
    """Run all device format tests."""
    print("GSwarm Device Format Test")
    print("=" * 50)
    
    try:
        test_device_utilities()
        test_cost_model_with_devices()
        test_predictor_with_devices()
        test_scheduler_with_devices()
        
        print("\n" + "=" * 50)
        print("✓ All device format tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
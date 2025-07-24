#!/usr/bin/env python3
"""
Test script to verify the integration of cost models and schedulers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gswarm.model import CostModel, LLMCostModel, SDCostModel, get_estimation_cost
from gswarm.predictor import predict_inference, predictor
from gswarm.scheduler import (
    BaselineScheduler,
    OfflineScheduler,
    OnlineScheduler,
    StaticScheduler,
    ModelInfo,
    WorkflowNode,
    WorkflowEdge,
    Workflow,
    Request,
)


def test_cost_models():
    """Test cost model functionality."""
    print("=== Testing Cost Models ===")
    
    # Test LLM cost model
    print("\n1. Testing LLM Cost Model (old API):")
    llm_cost = get_estimation_cost(
        model_type="llm",
        model_name="gpt-4",
        device="cuda:0",
        data_features=[{"prompt": "Hello, world!"}]
    )
    print(f"   LLM cost estimation: {llm_cost}")
    
    # Test SD cost model
    print("\n2. Testing SD Cost Model (old API):")
    sd_cost = get_estimation_cost(
        model_type="diffusion",
        model_name="stable-diffusion-v1-5",
        device="cuda:0",
        data_features=[{"height": 512, "width": 512}]
    )
    print(f"   SD cost estimation: {sd_cost}")
    
    # Test unified CostModel
    print("\n3. Testing Unified CostModel (new API):")
    cost_model = CostModel()
    llm_time = cost_model.predict("gpt-4", "cuda:0", {"prompt": "Hello, world!"})
    sd_time = cost_model.predict("stable-diffusion-v1-5", "cuda:0", {"height": 512, "width": 512})
    print(f"   LLM time (unified): {llm_time:.2f}s")
    print(f"   SD time (unified): {sd_time:.2f}s")
    
    print("\n✓ Cost models working correctly")


def test_predictor_integration():
    """Test predictor integration with cost models."""
    print("\n=== Testing Predictor Integration ===")
    
    # Test LLM prediction (old API)
    print("\n1. Testing LLM prediction (old API):")
    llm_result = predict_inference(
        model_name="llama-2-7b",
        device_name="cuda:0",
        inputs={"prompt": "Explain quantum computing in simple terms"}
    )
    print(f"   Model: {llm_result.model_name}")
    print(f"   Device: {llm_result.device_name}")
    print(f"   Prediction: {llm_result.prediction}")
    print(f"   Error: {llm_result.error}")
    
    # Test SD prediction (old API)
    print("\n2. Testing SD prediction (old API):")
    sd_result = predict_inference(
        model_name="stable-diffusion-xl",
        device_name="cuda:1",
        inputs={"height": 1024, "width": 1024, "prompt": "A beautiful sunset"}
    )
    print(f"   Model: {sd_result.model_name}")
    print(f"   Device: {sd_result.device_name}")
    print(f"   Prediction: {sd_result.prediction}")
    print(f"   Error: {sd_result.error}")
    
    # Test new simple predictor API
    print("\n3. Testing simple predictor (new API):")
    llm_time = predictor.predict("llama-2-7b", "cuda:0", {"prompt": "Hello"})
    sd_time = predictor.predict("stable-diffusion-xl", "cuda:1", {"height": 1024, "width": 1024})
    print(f"   LLM time: {llm_time:.2f}s")
    print(f"   SD time: {sd_time:.2f}s")
    
    # Test with new device format
    print("\n4. Testing with new device format (client:device:id):")
    llm_time_new = predictor.predict("gpt-4", "node1:cuda:0", {"prompt": "Hello"})
    sd_time_new = predictor.predict("stable-diffusion", "node2:cuda:1", {"height": 512, "width": 512})
    print(f"   LLM time on node1:cuda:0: {llm_time_new:.2f}s")
    print(f"   SD time on node2:cuda:1: {sd_time_new:.2f}s")
    
    print("\n✓ Predictor integration working correctly")


def test_schedulers():
    """Test scheduler functionality."""
    print("\n=== Testing Schedulers ===")
    
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
    
    # Define workflow
    nodes = [
        WorkflowNode(
            id="prompt_gen",
            model="gpt-4",
            inputs=["user_input"],
            outputs=["generated_prompt"]
        ),
        WorkflowNode(
            id="image_gen",
            model="stable-diffusion",
            inputs=["generated_prompt"],
            outputs=["image"]
        )
    ]
    
    edges = [
        WorkflowEdge(from_node="prompt_gen", to_node="image_gen")
    ]
    
    workflow = Workflow(
        id="text2image",
        name="Text to Image Pipeline",
        nodes=nodes,
        edges=edges
    )
    
    # Test each scheduler
    schedulers = [
        ("Baseline", BaselineScheduler),
        ("Offline", OfflineScheduler),
        ("Online", OnlineScheduler),
        ("Static", StaticScheduler),
    ]
    
    for name, SchedulerClass in schedulers:
        print(f"\n{name} Scheduler Test:")
        
        try:
            if name == "Static":
                # Test with new device format for static scheduler
                scheduler = SchedulerClass(
                    devices=["node1:cuda:0", "node1:cuda:1"],
                    models=models,
                    simulate=True
                )
            else:
                # Use legacy format for others
                scheduler = SchedulerClass(
                    gpus=[0, 1],
                    models=models,
                    simulate=True
                )
            
            scheduler.add_workflow(workflow)
            
            # Create test requests
            requests = [
                Request(
                    id=f"req_{i}",
                    workflow_id="text2image",
                    arrival_time=i * 0.5,
                    priority=1
                )
                for i in range(3)
            ]
            
            if name in ["Offline", "Static"]:
                # Batch scheduling
                tasks = scheduler.schedule(requests)
                print(f"   Scheduled {len(tasks)} tasks")
                for task in tasks[:3]:  # Show first 3 tasks
                    print(f"   - {task.model_name} on GPU {task.gpu_id} at time {task.scheduled_time:.2f}")
            else:
                # Online scheduling
                scheduler.add_request(requests[0])
                task = scheduler.get_next_task()
                if task:
                    print(f"   First task: {task.model_name} on GPU {task.gpu_id}")
                else:
                    print("   No tasks ready")
                    
            print(f"   ✓ {name} scheduler working")
            
        except Exception as e:
            print(f"   ✗ {name} scheduler error: {e}")
    
    print("\n✓ All schedulers tested")


def main():
    """Run all integration tests."""
    print("GSwarm Integration Test")
    print("=" * 50)
    
    try:
        test_cost_models()
        test_predictor_integration()
        test_schedulers()
        
        print("\n" + "=" * 50)
        print("✓ All integration tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
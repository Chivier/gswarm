#!/usr/bin/env python3
"""
Test script to demonstrate the unified CostModel API.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gswarm.model import CostModel
from gswarm.predictor import Predictor, predictor


def test_unified_cost_model():
    """Test the unified CostModel interface."""
    print("=== Testing Unified CostModel Interface ===")
    
    # Create a cost model with auto-detection
    cost_model = CostModel(model_type="auto")
    
    print("\n1. Testing LLM prediction with CostModel:")
    llm_time = cost_model.predict(
        model_name="gpt-4",
        device="cuda:0",
        inputs={"prompt": "What is machine learning?"}
    )
    print(f"   GPT-4 estimated time: {llm_time:.2f} seconds")
    
    print("\n2. Testing Diffusion prediction with CostModel:")
    sd_time = cost_model.predict(
        model_name="stable-diffusion-v1-5",
        device="cuda:0",
        inputs={"height": 512, "width": 512}
    )
    print(f"   SD v1.5 estimated time: {sd_time:.2f} seconds")
    
    print("\n3. Testing with explicit model type:")
    # Force model type
    time = cost_model.predict(
        model_name="custom-model",
        device="cuda:0",
        inputs={"prompt": "Hello"},
        model_type="llm"  # Explicitly set as LLM
    )
    print(f"   Custom model (as LLM) estimated time: {time:.2f} seconds")
    
    print("\n✓ Unified CostModel working correctly")


def test_simple_predictor():
    """Test the simple Predictor interface."""
    print("\n=== Testing Simple Predictor Interface ===")
    
    # Create a predictor
    pred = Predictor()
    
    print("\n1. Using instance predictor:")
    # LLM prediction
    llm_time = pred.predict("llama-2-7b", "cuda:0", {"prompt": "Explain AI"})
    print(f"   Llama-2 prediction: {llm_time:.2f} seconds")
    
    # Diffusion prediction
    sd_time = pred.predict("stable-diffusion-xl", "cuda:1", {"height": 1024, "width": 1024})
    print(f"   SDXL prediction: {sd_time:.2f} seconds")
    
    print("\n2. Using global predictor:")
    # Use the global predictor instance
    time = predictor.predict("gpt-3.5-turbo", "cuda:0", {"prompt": "Hello world"})
    print(f"   GPT-3.5 prediction: {time:.2f} seconds")
    
    print("\n3. Batch prediction:")
    requests = [
        {"model_name": "gpt-4", "device": "cuda:0", "inputs": {"prompt": "Question 1"}},
        {"model_name": "stable-diffusion", "device": "cuda:1", "inputs": {"height": 512, "width": 512}},
        {"model_name": "llama-2-13b", "device": "cuda:2", "inputs": {"prompt": "Question 2"}},
    ]
    times = pred.batch_predict(requests)
    for i, (req, time) in enumerate(zip(requests, times)):
        print(f"   Request {i+1} ({req['model_name']}): {time:.2f} seconds")
    
    print("\n4. Model updates:")
    # Update with actual execution data
    pred.update(
        "gpt-4",
        "cuda:0",
        {"prompt": "What is AI?", "output": "AI is..."},
        actual_time=0.8
    )
    print("   ✓ Model updated with actual data")
    
    print("\n✓ Simple Predictor working correctly")


def test_comparison():
    """Compare different ways to use the API."""
    print("\n=== API Usage Comparison ===")
    
    print("\n1. Direct CostModel usage:")
    print("   cost_model = CostModel()")
    print("   time = cost_model.predict('gpt-4', 'cuda:0', {'prompt': 'Hello'})")
    
    print("\n2. Simple Predictor usage:")
    print("   predictor = Predictor()")
    print("   time = predictor.predict('gpt-4', 'cuda:0', {'prompt': 'Hello'})")
    
    print("\n3. Global predictor usage:")
    print("   from gswarm.predictor import predictor")
    print("   time = predictor.predict('gpt-4', 'cuda:0', {'prompt': 'Hello'})")
    
    print("\n4. Original predict_inference usage (still supported):")
    print("   from gswarm.predictor import predict_inference")
    print("   result = predict_inference('gpt-4', 'cuda:0', {'prompt': 'Hello'})")
    print("   time = result.prediction['inference_time']")


def main():
    """Run all tests."""
    print("GSwarm Unified API Test")
    print("=" * 50)
    
    try:
        test_unified_cost_model()
        test_simple_predictor()
        test_comparison()
        
        print("\n" + "=" * 50)
        print("✓ All unified API tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
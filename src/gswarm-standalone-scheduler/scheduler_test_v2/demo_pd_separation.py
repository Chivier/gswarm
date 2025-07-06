#!/usr/bin/env python3
"""
Demo script to showcase PD-separated inference concept using mock engine
"""

import asyncio
import time
import sys
import os
from typing import List, Dict

# Add gswarm to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from llm_pd_server_lite import PDConfig, PDScheduler, EngineType


async def demo_pd_separation():
    """Demonstrate PD-separated inference with visual output"""
    
    print("\n" + "="*80)
    print("PD-SEPARATED INFERENCE DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how Prefill-Decode separation works:")
    print("- Prefill nodes: Process input prompts to generate KV cache")
    print("- Decode nodes: Generate output tokens using KV cache")
    print("- KV cache is transferred via gswarm data module")
    print("-"*80)
    
    # Configuration
    config = PDConfig(
        model_name="mock-model",
        pd_ratio=(2, 3),  # 2 prefill nodes, 3 decode nodes
        engine_type=EngineType.MOCK,
        data_server_url="localhost:9015"
    )
    
    # Create scheduler
    print(f"\n1. Creating scheduler with {config.pd_ratio[0]} prefill and {config.pd_ratio[1]} decode nodes...")
    scheduler = PDScheduler(config)
    await scheduler.initialize()
    
    # Start scheduler
    scheduler_task = asyncio.create_task(scheduler.run())
    
    # Test prompts
    test_cases = [
        {
            "prompt": "What is artificial intelligence?",
            "expected_prefill_ms": 50,
            "expected_decode_ms": 500,
        },
        {
            "prompt": "Explain how neural networks learn from data.",
            "expected_prefill_ms": 80,
            "expected_decode_ms": 500,
        },
        {
            "prompt": "The future of AI is",
            "expected_prefill_ms": 30,
            "expected_decode_ms": 500,
        },
    ]
    
    print(f"\n2. Submitting {len(test_cases)} test requests...")
    
    # Submit requests
    request_info = []
    for i, test in enumerate(test_cases):
        request_id = await scheduler.submit_request({
            "prompt": test["prompt"],
            "max_tokens": 50,
            "temperature": 0.7
        })
        
        request_info.append({
            "id": request_id,
            "prompt": test["prompt"],
            "submit_time": time.time()
        })
        
        print(f"   [{i+1}] Submitted: {test['prompt'][:40]}... (ID: {request_id})")
        await asyncio.sleep(0.1)  # Small delay between submissions
    
    print(f"\n3. Processing requests through PD pipeline...")
    print("   [Prefill] -> [KV Cache Transfer] -> [Decode]")
    
    # Collect results
    results = []
    for info in request_info:
        result = await scheduler.get_result(info["id"], timeout=10)
        if result:
            result["submit_time"] = info["submit_time"]
            results.append(result)
            
            # Show progress
            print(f"\n   ✓ Request {info['id']}:")
            print(f"     Prompt: {result['prompt'][:50]}...")
            print(f"     Prefill: {result['prefill_time']*1000:.1f}ms")
            print(f"     Decode: {result['decode_time']*1000:.1f}ms")
            print(f"     Total: {result['total_time']*1000:.1f}ms")
            print(f"     Tokens: {result['tokens_generated']}")
    
    # Show summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    if results:
        avg_prefill = sum(r['prefill_time'] for r in results) / len(results) * 1000
        avg_decode = sum(r['decode_time'] for r in results) / len(results) * 1000
        avg_total = sum(r['total_time'] for r in results) / len(results) * 1000
        
        print(f"\nAverage Times (ms):")
        print(f"  Prefill:  {avg_prefill:6.1f} ms")
        print(f"  Decode:   {avg_decode:6.1f} ms")
        print(f"  Total:    {avg_total:6.1f} ms")
        
        print(f"\nThroughput:")
        total_time = max(r['total_time'] + r['submit_time'] - results[0]['submit_time'] for r in results)
        throughput = len(results) / total_time
        print(f"  Requests/sec: {throughput:.2f}")
        
        # Show PD separation benefits
        print(f"\nPD Separation Benefits:")
        print(f"  - Prefill nodes specialized for prompt processing")
        print(f"  - Decode nodes optimized for token generation")
        print(f"  - KV cache sharing enables better resource utilization")
        print(f"  - {config.pd_ratio[1]/config.pd_ratio[0]:.1f}x more decode capacity for long generations")
    
    # Cleanup
    scheduler_task.cancel()
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)


async def compare_pd_vs_traditional():
    """Compare PD-separated vs traditional inference"""
    
    print("\n" + "="*80)
    print("COMPARING PD-SEPARATED VS TRADITIONAL INFERENCE")
    print("="*80)
    
    # Test configuration
    test_prompts = [
        "Explain machine learning",
        "What is deep learning?",
        "How do transformers work?",
        "Describe neural networks",
        "What is backpropagation?",
    ]
    
    # PD-separated configuration
    pd_config = PDConfig(
        model_name="mock-model",
        pd_ratio=(2, 4),  # 2 prefill, 4 decode
        engine_type=EngineType.MOCK,
        data_server_url="localhost:9015"
    )
    
    # Traditional configuration (balanced)
    trad_config = PDConfig(
        model_name="mock-model",
        pd_ratio=(3, 3),  # Equal split
        engine_type=EngineType.MOCK,
        data_server_url="localhost:9015"
    )
    
    print("\n1. Testing PD-Separated (2 prefill, 4 decode)...")
    pd_scheduler = PDScheduler(pd_config)
    await pd_scheduler.initialize()
    pd_task = asyncio.create_task(pd_scheduler.run())
    
    # Submit requests
    pd_start = time.time()
    pd_requests = []
    for prompt in test_prompts:
        req_id = await pd_scheduler.submit_request({
            "prompt": prompt,
            "max_tokens": 100,  # Longer generation
            "temperature": 0.7
        })
        pd_requests.append(req_id)
    
    # Collect results
    pd_results = []
    for req_id in pd_requests:
        result = await pd_scheduler.get_result(req_id)
        if result:
            pd_results.append(result)
    
    pd_total_time = time.time() - pd_start
    pd_task.cancel()
    
    print("\n2. Testing Traditional (3 prefill, 3 decode)...")
    trad_scheduler = PDScheduler(trad_config)
    await trad_scheduler.initialize()
    trad_task = asyncio.create_task(trad_scheduler.run())
    
    # Submit requests
    trad_start = time.time()
    trad_requests = []
    for prompt in test_prompts:
        req_id = await trad_scheduler.submit_request({
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        })
        trad_requests.append(req_id)
    
    # Collect results
    trad_results = []
    for req_id in trad_requests:
        result = await trad_scheduler.get_result(req_id)
        if result:
            trad_results.append(result)
    
    trad_total_time = time.time() - trad_start
    trad_task.cancel()
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  PD-Separated: 2 prefill, 4 decode nodes")
    print(f"  Traditional:  3 prefill, 3 decode nodes")
    print(f"  Total nodes:  6 (same for both)")
    
    print(f"\nPerformance:")
    print(f"  PD-Separated:")
    print(f"    Total time: {pd_total_time:.2f}s")
    print(f"    Throughput: {len(pd_results)/pd_total_time:.2f} req/s")
    
    print(f"  Traditional:")
    print(f"    Total time: {trad_total_time:.2f}s")
    print(f"    Throughput: {len(trad_results)/trad_total_time:.2f} req/s")
    
    improvement = ((len(pd_results)/pd_total_time) - (len(trad_results)/trad_total_time)) / (len(trad_results)/trad_total_time) * 100
    
    print(f"\nImprovement: {improvement:+.1f}%")
    
    print(f"\nKey Insights:")
    print(f"  - PD separation is better for generation-heavy workloads")
    print(f"  - More decode nodes handle token generation efficiently")
    print(f"  - KV cache sharing reduces redundant computation")
    
    print("\n" + "="*80)


async def main():
    """Main demo entry point"""
    
    # Check if gswarm data server is running
    try:
        from gswarm.data import DataServer
        client = DataServer("localhost:9015")
        # Try a simple operation
        client.write("test", "test", location="dram")
        print("✓ Gswarm data server is running")
    except:
        print("! Gswarm data server not running. Please start it with:")
        print("  python -m gswarm.data start --host 0.0.0.0 --port 9015")
        return
    
    # Run demos
    print("\nSelect demo:")
    print("1. Basic PD-separation demo")
    print("2. Compare PD vs Traditional")
    print("3. Run both demos")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        await demo_pd_separation()
    elif choice == "2":
        await compare_pd_vs_traditional()
    elif choice == "3":
        await demo_pd_separation()
        await compare_pd_vs_traditional()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
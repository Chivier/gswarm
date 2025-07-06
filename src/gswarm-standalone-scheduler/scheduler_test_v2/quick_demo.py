#!/usr/bin/env python3
"""
Quick demo of PD-separated inference using mock engine (no GPU required)
"""

import asyncio
import sys
import os

# Add gswarm to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from llm_pd_server_lite import PDConfig, PDScheduler, EngineType

async def main():
    print("\n" + "="*60)
    print("PD-SEPARATED INFERENCE DEMO (Mock Engine)")
    print("="*60)
    
    # Use mock engine - no GPU required
    config = PDConfig(
        model_name="mock-model",
        pd_ratio=(2, 3),  # 2 prefill nodes, 3 decode nodes
        engine_type=EngineType.MOCK,
        data_server_url="localhost:9015"
    )
    
    print(f"\nConfiguration:")
    print(f"- Engine: Mock (no GPU required)")
    print(f"- Prefill nodes: {config.pd_ratio[0]}")
    print(f"- Decode nodes: {config.pd_ratio[1]}")
    
    # Initialize scheduler
    print("\nInitializing scheduler...")
    scheduler = PDScheduler(config)
    await scheduler.initialize()
    
    # Start scheduler
    scheduler_task = asyncio.create_task(scheduler.run())
    
    # Test prompts
    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "How does AI work?",
    ]
    
    print(f"\nSubmitting {len(prompts)} test requests...")
    
    # Submit and track requests
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[Request {i+1}] {prompt}")
        
        # Submit request
        request_id = await scheduler.submit_request({
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7
        })
        
        # Get result
        result = await scheduler.get_result(request_id, timeout=10)
        
        if result:
            results.append(result)
            print(f"✓ Completed in {result['total_time']:.3f}s")
            print(f"  - Prefill: {result['prefill_time']*1000:.1f}ms")
            print(f"  - Decode: {result['decode_time']*1000:.1f}ms")
            print(f"  - Tokens: {result['tokens_generated']}")
    
    # Summary
    if results:
        print("\n" + "-"*60)
        print("SUMMARY")
        print("-"*60)
        
        avg_prefill = sum(r['prefill_time'] for r in results) / len(results) * 1000
        avg_decode = sum(r['decode_time'] for r in results) / len(results) * 1000
        avg_total = sum(r['total_time'] for r in results) / len(results) * 1000
        
        print(f"Average times:")
        print(f"  Prefill: {avg_prefill:.1f}ms")
        print(f"  Decode:  {avg_decode:.1f}ms")
        print(f"  Total:   {avg_total:.1f}ms")
        
        print(f"\nPD-separation benefits:")
        print(f"  ✓ Specialized prefill nodes for prompt processing")
        print(f"  ✓ Dedicated decode nodes for generation")
        print(f"  ✓ KV cache sharing via gswarm")
        print(f"  ✓ Better resource utilization")
    
    # Cleanup
    scheduler_task.cancel()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

if __name__ == "__main__":
    # Check if gswarm data server is running
    import requests
    try:
        response = requests.get("http://localhost:9015/stats")
        response.raise_for_status()
        print("✓ Gswarm data server is running")
    except:
        print("ERROR: Gswarm data server not running!")
        print("Please start it with:")
        print("  python -m gswarm.data start --host 0.0.0.0 --port 9015")
        sys.exit(1)
    
    # Run demo
    asyncio.run(main())
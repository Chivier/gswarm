#!/usr/bin/env python3
"""
Simple PD-separated inference demo with built-in storage (no external dependencies)
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json

# Simple in-memory storage to replace gswarm for demo
class SimpleStorage:
    def __init__(self):
        self.data = {}
        
    def write(self, key: str, value: Any, location: str = "dram"):
        """Store data with location info"""
        self.data[key] = {
            "value": value,
            "location": location,
            "timestamp": time.time()
        }
        return True
        
    def read(self, key: str) -> Optional[Any]:
        """Read data if in DRAM"""
        if key in self.data:
            entry = self.data[key]
            if entry["location"] == "dram":
                return entry["value"]
            else:
                print(f"Warning: Data '{key}' is in {entry['location']}, not DRAM")
        return None
        
    def move(self, key: str, destination: str):
        """Move data to different location"""
        if key in self.data:
            self.data[key]["location"] = destination
            return True
        return False
        
    def get_location(self, key: str) -> Optional[str]:
        """Get current location of data"""
        if key in self.data:
            return self.data[key]["location"]
        return None


@dataclass
class PDConfig:
    """Configuration for PD-separated inference"""
    pd_ratio: Tuple[int, int]  # (prefill_nodes, decode_nodes)
    max_batch_size: int = 8


class MockPDEngine:
    """Simplified mock engine for demo"""
    
    def __init__(self, role: str, storage: SimpleStorage):
        self.role = role
        self.storage = storage
        
    async def process_request(self, request: Dict) -> Dict:
        """Process request based on role"""
        if self.role == "prefill":
            return await self._process_prefill(request)
        else:
            return await self._process_decode(request)
            
    async def _process_prefill(self, request: Dict) -> Dict:
        """Mock prefill processing"""
        request_id = request["request_id"]
        prompt = request["prompt"]
        prompt_len = len(prompt.split())
        
        # Simulate prefill time
        prefill_time = prompt_len * 0.002  # 2ms per token
        await asyncio.sleep(prefill_time)
        
        # Create mock KV cache (simplified)
        kv_cache = {
            "prompt_len": prompt_len,
            "hidden_states": f"mock_hidden_states_{prompt_len}",
            "attention_scores": f"mock_attention_{prompt_len}"
        }
        
        # Store in our simple storage
        cache_key = f"kv_cache:{request_id}"
        metadata_key = f"kv_metadata:{request_id}"
        
        self.storage.write(cache_key, kv_cache, location="device:0")
        self.storage.write(metadata_key, {
            "request_id": request_id,
            "seq_len": prompt_len,
            "timestamp": time.time()
        }, location="dram")
        
        return {
            "request_id": request_id,
            "status": "prefill_complete",
            "prefill_time": prefill_time,
            "cache_key": cache_key,
            "metadata_key": metadata_key,
            "seq_len": prompt_len
        }
        
    async def _process_decode(self, request: Dict) -> Dict:
        """Mock decode processing"""
        request_id = request["request_id"]
        cache_key = request["cache_key"]
        metadata_key = request["metadata_key"]
        max_tokens = request.get("max_tokens", 50)
        
        # Simulate KV cache retrieval
        self.storage.move(metadata_key, "dram")
        metadata = self.storage.read(metadata_key)
        
        # Simulate decode time
        decode_time = max_tokens * 0.01  # 10ms per token
        await asyncio.sleep(decode_time)
        
        # Generate mock output
        generated_text = f"[Generated {max_tokens} tokens based on prompt]"
        
        return {
            "request_id": request_id,
            "status": "decode_complete",
            "decode_time": decode_time,
            "generated_text": generated_text,
            "tokens_generated": max_tokens
        }


class SimplePDScheduler:
    """Simplified scheduler for demo"""
    
    def __init__(self, config: PDConfig):
        self.config = config
        self.storage = SimpleStorage()
        self.prefill_engines = []
        self.decode_engines = []
        self.request_queue = asyncio.Queue()
        self.pending_decodes = asyncio.Queue()
        self.results = {}
        
    async def initialize(self):
        """Initialize engines"""
        prefill_count, decode_count = self.config.pd_ratio
        
        # Create engines
        for i in range(prefill_count):
            engine = MockPDEngine("prefill", self.storage)
            self.prefill_engines.append(engine)
            
        for i in range(decode_count):
            engine = MockPDEngine("decode", self.storage)
            self.decode_engines.append(engine)
            
        print(f"✓ Initialized {prefill_count} prefill and {decode_count} decode engines")
        
    async def submit_request(self, prompt: str, max_tokens: int = 50) -> str:
        """Submit request and return ID"""
        request_id = f"req_{int(time.time() * 1000000)}"
        request = {
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens
        }
        await self.request_queue.put(request)
        return request_id
        
    async def run_workers(self):
        """Run all workers"""
        workers = []
        
        # Start prefill workers
        for i, engine in enumerate(self.prefill_engines):
            workers.append(asyncio.create_task(self._prefill_worker(engine, i)))
            
        # Start decode workers
        for i, engine in enumerate(self.decode_engines):
            workers.append(asyncio.create_task(self._decode_worker(engine, i)))
            
        # Run for demo duration
        await asyncio.sleep(10)
        
        # Cancel workers
        for w in workers:
            w.cancel()
            
    async def _prefill_worker(self, engine, worker_id: int):
        """Prefill worker loop"""
        while True:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
                print(f"  [Prefill-{worker_id}] Processing: {request['prompt'][:30]}...")
                
                result = await engine.process_request(request)
                
                # Queue for decode
                decode_request = {
                    "request_id": result["request_id"],
                    "cache_key": result["cache_key"],
                    "metadata_key": result["metadata_key"],
                    "max_tokens": request["max_tokens"],
                    "original_prompt": request["prompt"],
                    "prefill_time": result["prefill_time"]
                }
                
                await self.pending_decodes.put(decode_request)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"  [Prefill-{worker_id}] Error: {e}")
                
    async def _decode_worker(self, engine, worker_id: int):
        """Decode worker loop"""
        while True:
            try:
                decode_request = await asyncio.wait_for(self.pending_decodes.get(), timeout=0.1)
                print(f"  [Decode-{worker_id}] Generating tokens...")
                
                result = await engine.process_request(decode_request)
                
                # Store result
                self.results[result["request_id"]] = {
                    "prompt": decode_request["original_prompt"],
                    "generated_text": result["generated_text"],
                    "prefill_time": decode_request["prefill_time"],
                    "decode_time": result["decode_time"],
                    "total_time": decode_request["prefill_time"] + result["decode_time"],
                    "tokens": result["tokens_generated"]
                }
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"  [Decode-{worker_id}] Error: {e}")


async def run_demo():
    """Run the complete demo"""
    print("\n" + "="*70)
    print("PD-SEPARATED INFERENCE DEMO")
    print("="*70)
    print("\nThis demo shows Prefill-Decode separation in action:")
    print("- Prefill nodes process prompts → generate KV cache")
    print("- Decode nodes use KV cache → generate output tokens")
    print("- Storage system manages data transfer between phases")
    print("-"*70)
    
    # Test different configurations
    configs = [
        ((2, 4), "Generation-Heavy: 2 prefill, 4 decode"),
        ((3, 3), "Balanced: 3 prefill, 3 decode"),
        ((4, 2), "Prompt-Heavy: 4 prefill, 2 decode"),
    ]
    
    test_prompts = [
        "What is machine learning?",
        "Explain neural networks in simple terms",
        "How does deep learning work?",
        "What are transformers in AI?",
        "Describe the attention mechanism",
    ]
    
    for pd_ratio, description in configs:
        print(f"\n\nTesting Configuration: {description}")
        print("="*50)
        
        # Create scheduler
        config = PDConfig(pd_ratio=pd_ratio)
        scheduler = SimplePDScheduler(config)
        await scheduler.initialize()
        
        # Start workers
        worker_task = asyncio.create_task(scheduler.run_workers())
        
        # Submit requests
        print("\nSubmitting requests:")
        request_ids = []
        start_time = time.time()
        
        for prompt in test_prompts:
            req_id = await scheduler.submit_request(prompt, max_tokens=50)
            request_ids.append(req_id)
            print(f"  → {prompt[:40]}...")
            await asyncio.sleep(0.05)  # Small delay between submissions
        
        print("\nProcessing:")
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Collect results
        completed = len(scheduler.results)
        total_time = time.time() - start_time
        
        if scheduler.results:
            avg_prefill = sum(r["prefill_time"] for r in scheduler.results.values()) / len(scheduler.results)
            avg_decode = sum(r["decode_time"] for r in scheduler.results.values()) / len(scheduler.results)
            avg_total = sum(r["total_time"] for r in scheduler.results.values()) / len(scheduler.results)
            throughput = completed / total_time
            
            print(f"\nResults:")
            print(f"  Completed: {completed}/{len(test_prompts)} requests")
            print(f"  Avg Prefill: {avg_prefill*1000:.1f}ms")
            print(f"  Avg Decode: {avg_decode*1000:.1f}ms")
            print(f"  Avg Total: {avg_total*1000:.1f}ms")
            print(f"  Throughput: {throughput:.2f} req/s")
            
            # Show efficiency metrics
            prefill_efficiency = avg_prefill / avg_total * 100
            decode_efficiency = avg_decode / avg_total * 100
            
            print(f"\nTime Distribution:")
            print(f"  Prefill: {prefill_efficiency:.1f}%")
            print(f"  Decode: {decode_efficiency:.1f}%")
            
            if pd_ratio[0] < pd_ratio[1]:
                print(f"  → Good for generation-heavy workloads ✓")
            elif pd_ratio[0] > pd_ratio[1]:
                print(f"  → Good for prompt-heavy workloads ✓")
            else:
                print(f"  → Balanced for mixed workloads ✓")
        
        # Cleanup
        worker_task.cancel()
        await asyncio.sleep(0.5)
    
    print("\n\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("\n1. PD-Separation Benefits:")
    print("   - Specialized resource allocation")
    print("   - Better hardware utilization")
    print("   - Flexible scaling based on workload")
    
    print("\n2. Configuration Guidelines:")
    print("   - More decode nodes: Long text generation (stories, articles)")
    print("   - More prefill nodes: Many short queries (Q&A, chat)")
    print("   - Balanced: General-purpose serving")
    
    print("\n3. Real-World Optimizations:")
    print("   - GPU memory for KV cache storage")
    print("   - NVLink for fast GPU-GPU transfers")
    print("   - Dynamic scheduling based on queue depth")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    print("\nStarting PD-Separated Inference Demo...")
    print("(No external dependencies required)")
    
    asyncio.run(run_demo())
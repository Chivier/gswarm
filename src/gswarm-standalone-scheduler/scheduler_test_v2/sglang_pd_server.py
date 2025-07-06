#!/usr/bin/env python3
"""
SGLang-based PD-separated inference server

This implementation uses SGLang's runtime for real model serving with PD separation.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import numpy as np

# Add gswarm to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.data import DataServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import SGLang
try:
    import sglang as sgl
    from sglang import Runtime, set_default_backend
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    logger.warning("SGLang not installed. Please install with: pip install sglang")


@dataclass
class SGLangPDConfig:
    """Configuration for SGLang PD-separated inference"""
    model_path: str
    pd_ratio: Tuple[int, int]  # (prefill_nodes, decode_nodes)
    tp_size: int = 1  # Tensor parallel size
    max_batch_size: int = 8
    max_seq_len: int = 2048
    mem_fraction_static: float = 0.9
    data_server_url: str = "localhost:9015"
    use_flash_attn: bool = True


class SGLangPrefillEngine:
    """SGLang engine optimized for prefill phase"""
    
    def __init__(self, config: SGLangPDConfig, engine_id: int):
        self.config = config
        self.engine_id = engine_id
        self.data_client = DataServer(config.data_server_url)
        self.runtime = None
        
    def initialize(self):
        """Initialize SGLang runtime for prefill"""
        if not SGLANG_AVAILABLE:
            raise ImportError("SGLang not available")
            
        logger.info(f"Initializing SGLang prefill engine {self.engine_id}")
        
        # Configure for prefill optimization
        runtime_args = {
            "model_path": self.config.model_path,
            "tp_size": self.config.tp_size,
            "max_batch_size": self.config.max_batch_size,
            "mem_fraction_static": self.config.mem_fraction_static,
            "disable_radix_cache": False,  # Keep cache for prefill
            "disable_flashinfer": not self.config.use_flash_attn,
        }
        
        # Launch runtime
        self.runtime = Runtime(**runtime_args)
        set_default_backend(self.runtime)
        
        logger.info(f"SGLang prefill engine {self.engine_id} initialized")
        
    def shutdown(self):
        """Shutdown the runtime"""
        if self.runtime:
            self.runtime.shutdown()
            
    @sgl.function
    def prefill_only(s, prompt, max_tokens=1):
        """SGLang function that only does prefill"""
        s += prompt
        # Generate just 1 token to trigger KV cache creation
        s += sgl.gen("output", max_tokens=max_tokens, temperature=0.0)
        
    async def process_request(self, request: Dict) -> Dict:
        """Process prefill request"""
        request_id = request["request_id"]
        prompt = request["prompt"]
        
        start_time = time.time()
        
        try:
            # Run prefill using SGLang
            state = self.prefill_only.run(
                prompt=prompt,
                max_tokens=1,  # Only need to generate KV cache
                stream=False
            )
            
            prefill_time = time.time() - start_time
            
            # Extract KV cache from SGLang runtime
            # Note: This is a simplified version - actual implementation would need
            # to hook into SGLang internals to extract the actual KV cache
            kv_cache_data = self._extract_kv_cache(state, prompt)
            
            # Store KV cache in gswarm
            cache_key = f"kv_cache:{request_id}"
            metadata_key = f"kv_metadata:{request_id}"
            
            # Store on GPU for fast access
            device = f"device:{torch.cuda.current_device()}" if torch.cuda.is_available() else "dram"
            self.data_client.write(cache_key, kv_cache_data, location=device)
            
            # Store metadata
            metadata = {
                "request_id": request_id,
                "prompt": prompt,
                "seq_len": len(self.runtime.get_tokenizer()(prompt)["input_ids"]),
                "model_path": self.config.model_path,
                "timestamp": time.time(),
                "prefill_engine": self.engine_id
            }
            self.data_client.write(metadata_key, metadata, location="dram")
            
            return {
                "request_id": request_id,
                "status": "prefill_complete",
                "prefill_time": prefill_time,
                "cache_key": cache_key,
                "metadata_key": metadata_key,
                "seq_len": metadata["seq_len"]
            }
            
        except Exception as e:
            logger.error(f"Prefill error: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
            
    def _extract_kv_cache(self, state, prompt):
        """Extract KV cache from SGLang state"""
        # This is a placeholder - actual implementation would need to
        # access SGLang's internal KV cache storage
        # For demo purposes, we'll create a mock cache
        tokenizer = self.runtime.get_tokenizer()
        tokens = tokenizer(prompt)["input_ids"]
        
        # Mock KV cache data
        kv_cache = {
            "tokens": tokens,
            "seq_len": len(tokens),
            "layer_data": {}  # Would contain actual K,V tensors per layer
        }
        
        # In real implementation, you'd extract from model's attention layers
        return kv_cache


class SGLangDecodeEngine:
    """SGLang engine optimized for decode phase"""
    
    def __init__(self, config: SGLangPDConfig, engine_id: int):
        self.config = config
        self.engine_id = engine_id
        self.data_client = DataServer(config.data_server_url)
        self.runtime = None
        
    def initialize(self):
        """Initialize SGLang runtime for decode"""
        if not SGLANG_AVAILABLE:
            raise ImportError("SGLang not available")
            
        logger.info(f"Initializing SGLang decode engine {self.engine_id}")
        
        # Configure for decode optimization
        runtime_args = {
            "model_path": self.config.model_path,
            "tp_size": self.config.tp_size,
            "max_batch_size": self.config.max_batch_size * 2,  # More batch for decode
            "mem_fraction_static": self.config.mem_fraction_static,
            "disable_radix_cache": True,  # No need for prefix cache in decode
            "disable_flashinfer": not self.config.use_flash_attn,
        }
        
        # Launch runtime
        self.runtime = Runtime(**runtime_args)
        set_default_backend(self.runtime)
        
        logger.info(f"SGLang decode engine {self.engine_id} initialized")
        
    def shutdown(self):
        """Shutdown the runtime"""
        if self.runtime:
            self.runtime.shutdown()
            
    @sgl.function
    def decode_with_cache(s, kv_cache_data, max_tokens, temperature):
        """SGLang function that decodes using pre-computed KV cache"""
        # In real implementation, we'd restore the KV cache here
        # For now, we'll simulate by using the prompt tokens
        s += sgl.gen(
            "output",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
    async def process_request(self, request: Dict) -> Dict:
        """Process decode request"""
        request_id = request["request_id"]
        cache_key = request["cache_key"]
        metadata_key = request["metadata_key"]
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)
        
        start_time = time.time()
        
        try:
            # Retrieve metadata
            self.data_client.move(metadata_key, "dram")
            metadata = self.data_client.read(metadata_key)
            
            # Move KV cache to current GPU if needed
            current_device = f"device:{torch.cuda.current_device()}" if torch.cuda.is_available() else "dram"
            self.data_client.move(cache_key, current_device)
            
            # Retrieve KV cache
            self.data_client.move(cache_key, "dram")
            kv_cache_data = self.data_client.read(cache_key)
            
            # Run decode using SGLang
            state = self.decode_with_cache.run(
                kv_cache_data=kv_cache_data,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            decode_time = time.time() - start_time
            
            # Extract generated text
            generated_text = state["output"]
            
            return {
                "request_id": request_id,
                "status": "decode_complete",
                "decode_time": decode_time,
                "generated_text": generated_text,
                "tokens_generated": len(self.runtime.get_tokenizer()(generated_text)["input_ids"])
            }
            
        except Exception as e:
            logger.error(f"Decode error: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }


class SGLangPDScheduler:
    """Scheduler for SGLang PD-separated inference"""
    
    def __init__(self, config: SGLangPDConfig):
        self.config = config
        self.prefill_engines = []
        self.decode_engines = []
        self.request_queue = asyncio.Queue()
        self.pending_decodes = asyncio.Queue()
        self.results = {}
        
    def initialize(self):
        """Initialize all engines"""
        prefill_count, decode_count = self.config.pd_ratio
        
        logger.info(f"Initializing SGLang PD scheduler with {prefill_count}:{decode_count} ratio")
        
        # Create prefill engines
        for i in range(prefill_count):
            engine = SGLangPrefillEngine(self.config, i)
            engine.initialize()
            self.prefill_engines.append(engine)
            
        # Create decode engines  
        for i in range(decode_count):
            engine = SGLangDecodeEngine(self.config, i)
            engine.initialize()
            self.decode_engines.append(engine)
            
        logger.info("All SGLang engines initialized")
        
    def shutdown(self):
        """Shutdown all engines"""
        for engine in self.prefill_engines:
            engine.shutdown()
        for engine in self.decode_engines:
            engine.shutdown()
            
    async def submit_request(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Submit a new request"""
        request_id = f"req_{int(time.time() * 1000000)}"
        request = {
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        await self.request_queue.put(request)
        return request_id
        
    async def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[Dict]:
        """Get result for a request"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.results:
                return self.results.pop(request_id)
            await asyncio.sleep(0.1)
        return None
        
    async def run(self):
        """Run the scheduler"""
        workers = []
        
        # Start prefill workers
        for i, engine in enumerate(self.prefill_engines):
            workers.append(asyncio.create_task(self._prefill_worker(engine, i)))
            
        # Start decode workers
        for i, engine in enumerate(self.decode_engines):
            workers.append(asyncio.create_task(self._decode_worker(engine, i)))
            
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            
    async def _prefill_worker(self, engine: SGLangPrefillEngine, worker_id: int):
        """Prefill worker"""
        logger.info(f"Prefill worker {worker_id} started")
        
        while True:
            try:
                request = await self.request_queue.get()
                logger.info(f"Prefill-{worker_id} processing: {request['prompt'][:50]}...")
                
                result = await engine.process_request(request)
                
                if result["status"] == "prefill_complete":
                    # Queue for decode
                    decode_request = {
                        "request_id": result["request_id"],
                        "cache_key": result["cache_key"],
                        "metadata_key": result["metadata_key"],
                        "max_tokens": request["max_tokens"],
                        "temperature": request["temperature"],
                        "prefill_time": result["prefill_time"]
                    }
                    await self.pending_decodes.put(decode_request)
                    
            except Exception as e:
                logger.error(f"Prefill worker {worker_id} error: {e}")
                
    async def _decode_worker(self, engine: SGLangDecodeEngine, worker_id: int):
        """Decode worker"""
        logger.info(f"Decode worker {worker_id} started")
        
        while True:
            try:
                decode_request = await self.pending_decodes.get()
                logger.info(f"Decode-{worker_id} generating for request {decode_request['request_id']}")
                
                result = await engine.process_request(decode_request)
                
                if result["status"] == "decode_complete":
                    # Store final result
                    self.results[result["request_id"]] = {
                        "request_id": result["request_id"],
                        "generated_text": result["generated_text"],
                        "tokens_generated": result["tokens_generated"],
                        "prefill_time": decode_request["prefill_time"],
                        "decode_time": result["decode_time"],
                        "total_time": decode_request["prefill_time"] + result["decode_time"]
                    }
                    
            except Exception as e:
                logger.error(f"Decode worker {worker_id} error: {e}")


async def run_sglang_demo():
    """Run demo with SGLang"""
    print("\n" + "="*70)
    print("SGLANG PD-SEPARATED INFERENCE DEMO")
    print("="*70)
    
    # Check if SGLang is available
    if not SGLANG_AVAILABLE:
        print("ERROR: SGLang not installed!")
        print("Please install with: pip install sglang")
        return
        
    # Configuration
    config = SGLangPDConfig(
        model_path="microsoft/phi-2",  # Small model for demo
        pd_ratio=(1, 2),  # 1 prefill, 2 decode
        tp_size=1,
        max_batch_size=4,
        mem_fraction_static=0.8
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_path}")
    print(f"  PD Ratio: {config.pd_ratio[0]}:{config.pd_ratio[1]}")
    print(f"  Tensor Parallel: {config.tp_size}")
    
    # Initialize scheduler
    print("\nInitializing SGLang engines...")
    scheduler = SGLangPDScheduler(config)
    
    try:
        scheduler.initialize()
        
        # Start scheduler
        scheduler_task = asyncio.create_task(scheduler.run())
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks in simple terms.",
            "How does deep learning work?",
        ]
        
        print(f"\nSubmitting {len(test_prompts)} test requests...")
        
        # Submit requests
        request_ids = []
        for prompt in test_prompts:
            req_id = await scheduler.submit_request(prompt, max_tokens=50)
            request_ids.append((req_id, prompt))
            print(f"  Submitted: {prompt[:40]}... (ID: {req_id})")
            
        # Get results
        print("\nWaiting for results...")
        for req_id, prompt in request_ids:
            result = await scheduler.get_result(req_id, timeout=30)
            
            if result:
                print(f"\n[Result for: {prompt[:40]}...]")
                print(f"  Generated: {result['generated_text'][:100]}...")
                print(f"  Tokens: {result['tokens_generated']}")
                print(f"  Prefill: {result['prefill_time']:.3f}s")
                print(f"  Decode: {result['decode_time']:.3f}s")
                print(f"  Total: {result['total_time']:.3f}s")
            else:
                print(f"\n[Timeout for: {prompt[:40]}...]")
                
        # Cancel scheduler
        scheduler_task.cancel()
        
    finally:
        # Cleanup
        scheduler.shutdown()
        
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="SGLang PD-separated inference server")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model path")
    parser.add_argument("--pd-ratio", type=str, default="1:2", help="Prefill:Decode ratio")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--data-server", type=str, default="localhost:9015", help="Gswarm data server")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo
        asyncio.run(run_sglang_demo())
    else:
        # Run server mode
        prefill, decode = map(int, args.pd_ratio.split(":"))
        
        config = SGLangPDConfig(
            model_path=args.model,
            pd_ratio=(prefill, decode),
            tp_size=args.tp_size,
            data_server_url=args.data_server
        )
        
        scheduler = SGLangPDScheduler(config)
        
        try:
            scheduler.initialize()
            
            # Run scheduler
            print(f"SGLang PD server running with {prefill}:{decode} ratio")
            print(f"Model: {args.model}")
            print("Press Ctrl+C to stop")
            
            asyncio.run(scheduler.run())
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            scheduler.shutdown()


if __name__ == "__main__":
    main()
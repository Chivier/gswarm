#!/usr/bin/env python3
"""
LLM Prefill-Decode Separated Inference Server (Lite Version)

This server implements PD-separated inference with smaller models and memory optimization.
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
from enum import Enum
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Add gswarm to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.data import DataServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EngineType(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    MOCK = "mock"  # For testing without actual models


@dataclass
class PDConfig:
    """Configuration for Prefill-Decode separation"""
    model_name: str
    pd_ratio: Tuple[int, int]  # (prefill_nodes, decode_nodes)
    engine_type: EngineType
    tensor_parallel_size: int = 1
    max_batch_size: int = 8  # Reduced for memory
    max_seq_len: int = 2048  # Reduced for memory
    kv_cache_dtype: str = "float16"
    data_server_url: str = "localhost:9015"
    gpu_memory_fraction: float = 0.9  # Reserve some memory
    quantization: Optional[str] = None  # Support quantization


class MockPDEngine:
    """Mock engine for testing without loading actual models"""
    
    def __init__(self, config: PDConfig, role: str):
        self.config = config
        self.role = role
        self.data_client = DataServer(config.data_server_url)
        logger.info(f"Initialized Mock engine for {role} role")
        
    async def initialize(self):
        """Initialize mock engine"""
        logger.info(f"Mock {self.role} engine ready")
        
    async def process_request(self, request: Dict) -> Dict:
        """Process request with mock timing"""
        if self.role == "prefill":
            return await self._mock_prefill(request)
        else:
            return await self._mock_decode(request)
            
    async def _mock_prefill(self, request: Dict) -> Dict:
        """Mock prefill phase"""
        request_id = request["request_id"]
        prompt = request["prompt"]
        prompt_len = len(prompt.split())
        
        # Simulate prefill time (roughly 1ms per token)
        prefill_time = prompt_len * 0.001
        await asyncio.sleep(prefill_time)
        
        # Create mock KV cache
        num_layers = 32
        num_heads = 32
        head_dim = 128
        seq_len = prompt_len
        
        # Create smaller mock tensors to save memory
        kv_cache = []
        for layer in range(num_layers):
            # Use uint8 to save memory in mock
            k_cache = np.random.randint(0, 255, (1, num_heads, seq_len, head_dim), dtype=np.uint8)
            v_cache = np.random.randint(0, 255, (1, num_heads, seq_len, head_dim), dtype=np.uint8)
            kv_cache.append((k_cache, v_cache))
        
        # Store in gswarm
        cache_key = f"kv_cache:{request_id}"
        metadata_key = f"kv_metadata:{request_id}"
        
        # Store on GPU if available, otherwise DRAM
        if torch.cuda.is_available():
            location = f"device:{torch.cuda.current_device()}"
        else:
            location = "dram"
            
        self.data_client.write(cache_key, kv_cache, location=location)
        
        metadata = {
            "request_id": request_id,
            "seq_len": seq_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": "uint8",
            "device": location,
            "timestamp": time.time()
        }
        
        self.data_client.write(metadata_key, metadata, location="dram")
        
        return {
            "request_id": request_id,
            "status": "prefill_complete",
            "prefill_time": prefill_time,
            "cache_key": cache_key,
            "metadata_key": metadata_key,
            "seq_len": seq_len
        }
        
    async def _mock_decode(self, request: Dict) -> Dict:
        """Mock decode phase"""
        request_id = request["request_id"]
        cache_key = request["cache_key"]
        metadata_key = request["metadata_key"]
        max_tokens = request.get("max_tokens", 100)
        
        # Retrieve metadata
        self.data_client.move(metadata_key, "dram")
        metadata = self.data_client.read(metadata_key)
        
        # Simulate decode time (roughly 10ms per token)
        decode_time = max_tokens * 0.01
        await asyncio.sleep(decode_time)
        
        # Generate mock text
        generated_text = " ".join([f"token_{i}" for i in range(max_tokens)])
        
        return {
            "request_id": request_id,
            "status": "decode_complete",
            "decode_time": decode_time,
            "generated_text": generated_text,
            "tokens_generated": max_tokens
        }


class VLLMPDEngineLite:
    """Lightweight vLLM-based PD inference engine"""
    
    def __init__(self, config: PDConfig, role: str):
        self.config = config
        self.role = role
        self.data_client = DataServer(config.data_server_url)
        self.engine = None
        
    async def initialize(self):
        """Initialize vLLM engine with memory constraints"""
        try:
            from vllm import LLM, SamplingParams
            
            # Common configuration
            engine_config = {
                "model": self.config.model_name,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "dtype": self.config.kv_cache_dtype,
                "max_num_seqs": self.config.max_batch_size,
                "gpu_memory_utilization": self.config.gpu_memory_fraction,
                "max_model_len": self.config.max_seq_len,
            }
            
            # Add quantization if specified
            if self.config.quantization:
                engine_config["quantization"] = self.config.quantization
            
            # Role-specific configuration
            if self.role == "prefill":
                engine_config["enforce_eager"] = True  # Disable CUDA graphs
                engine_config["enable_prefix_caching"] = False
            else:
                engine_config["enforce_eager"] = False  # Enable CUDA graphs
                engine_config["enable_prefix_caching"] = True
                
            self.engine = LLM(**engine_config)
            logger.info(f"Initialized vLLM engine for {self.role} role")
            
        except ImportError:
            logger.error("vLLM not installed. Falling back to mock engine.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
            
    async def process_request(self, request: Dict) -> Dict:
        """Process request based on role"""
        # Similar to MockPDEngine but with actual vLLM calls
        if self.role == "prefill":
            return await self._process_prefill(request)
        else:
            return await self._process_decode(request)
            
    # ... (implement _process_prefill and _process_decode similar to original)


def create_engine(config: PDConfig, role: str):
    """Factory function to create appropriate engine"""
    if config.engine_type == EngineType.MOCK:
        return MockPDEngine(config, role)
    elif config.engine_type == EngineType.VLLM:
        try:
            return VLLMPDEngineLite(config, role)
        except:
            logger.warning("Failed to create vLLM engine, falling back to mock")
            return MockPDEngine(config, role)
    else:
        # SGLang implementation
        return MockPDEngine(config, role)


class PDScheduler:
    """Scheduler for PD-separated inference"""
    
    def __init__(self, config: PDConfig):
        self.config = config
        self.prefill_engines = []
        self.decode_engines = []
        self.request_queue = asyncio.Queue()
        self.pending_decodes = asyncio.Queue()
        self.results = {}
        
    async def initialize(self):
        """Initialize prefill and decode engines"""
        prefill_count, decode_count = self.config.pd_ratio
        
        logger.info(f"Initializing {prefill_count} prefill and {decode_count} decode engines")
        
        # Create prefill engines
        for i in range(prefill_count):
            engine = create_engine(self.config, "prefill")
            await engine.initialize()
            self.prefill_engines.append(engine)
            
        # Create decode engines
        for i in range(decode_count):
            engine = create_engine(self.config, "decode")
            await engine.initialize()
            self.decode_engines.append(engine)
            
        logger.info("All engines initialized successfully")
        
    async def submit_request(self, request: Dict) -> str:
        """Submit a new inference request"""
        request_id = f"req_{int(time.time() * 1000000)}"
        request["request_id"] = request_id
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
        """Main scheduler loop"""
        # Start workers
        workers = []
        
        # Prefill workers
        for i, engine in enumerate(self.prefill_engines):
            workers.append(asyncio.create_task(self._prefill_worker(engine, i)))
            
        # Decode workers
        for i, engine in enumerate(self.decode_engines):
            workers.append(asyncio.create_task(self._decode_worker(engine, i)))
            
        # Result aggregator
        workers.append(asyncio.create_task(self._result_aggregator()))
        
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            # Cancel all workers
            for w in workers:
                w.cancel()
                
    async def _prefill_worker(self, engine, worker_id: int):
        """Worker for processing prefill requests"""
        logger.info(f"Prefill worker {worker_id} started")
        
        while True:
            try:
                request = await self.request_queue.get()
                logger.info(f"Prefill worker {worker_id} processing request {request['request_id']}")
                
                result = await engine.process_request(request)
                
                # Create decode request
                decode_request = {
                    "request_id": result["request_id"],
                    "cache_key": result["cache_key"],
                    "metadata_key": result["metadata_key"],
                    "max_tokens": request.get("max_tokens", 100),
                    "temperature": request.get("temperature", 0.7),
                    "prefill_time": result["prefill_time"],
                    "original_request": request
                }
                
                await self.pending_decodes.put(decode_request)
                
            except Exception as e:
                logger.error(f"Prefill worker {worker_id} error: {e}")
                await asyncio.sleep(1)
                
    async def _decode_worker(self, engine, worker_id: int):
        """Worker for processing decode requests"""
        logger.info(f"Decode worker {worker_id} started")
        
        while True:
            try:
                decode_request = await self.pending_decodes.get()
                logger.info(f"Decode worker {worker_id} processing request {decode_request['request_id']}")
                
                result = await engine.process_request(decode_request)
                
                # Combine results
                final_result = {
                    "request_id": result["request_id"],
                    "prompt": decode_request["original_request"]["prompt"],
                    "generated_text": result["generated_text"],
                    "tokens_generated": result["tokens_generated"],
                    "prefill_time": decode_request["prefill_time"],
                    "decode_time": result["decode_time"],
                    "total_time": decode_request["prefill_time"] + result["decode_time"]
                }
                
                self.results[result["request_id"]] = final_result
                
            except Exception as e:
                logger.error(f"Decode worker {worker_id} error: {e}")
                await asyncio.sleep(1)
                
    async def _result_aggregator(self):
        """Aggregate and log results"""
        while True:
            await asyncio.sleep(5)
            if self.results:
                logger.info(f"Completed requests: {len(self.results)}")


async def run_interactive_demo(scheduler: PDScheduler):
    """Run an interactive demo"""
    print("\n" + "="*60)
    print("PD-Separated Inference Demo")
    print("="*60)
    print("Enter prompts to generate text (type 'quit' to exit)")
    print("-"*60)
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
            
        if not prompt:
            continue
            
        # Submit request
        request_id = await scheduler.submit_request({
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7
        })
        
        print(f"Request submitted (ID: {request_id})")
        print("Waiting for response...")
        
        # Get result
        result = await scheduler.get_result(request_id, timeout=30)
        
        if result:
            print(f"\nGenerated text: {result['generated_text']}")
            print(f"Prefill time: {result['prefill_time']:.3f}s")
            print(f"Decode time: {result['decode_time']:.3f}s")
            print(f"Total time: {result['total_time']:.3f}s")
        else:
            print("Request timed out")


async def main():
    parser = argparse.ArgumentParser(description="PD-separated LLM inference server (Lite)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", 
                      help="Model name or path (default: Llama-2-7b)")
    parser.add_argument("--pd-ratio", type=str, default="2:3", 
                      help="Prefill:Decode ratio (e.g., 2:3)")
    parser.add_argument("--engine", type=str, choices=["vllm", "sglang", "mock"], 
                      default="mock", help="Inference engine")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--data-server", type=str, default="localhost:9015", 
                      help="Gswarm data server URL")
    parser.add_argument("--max-batch-size", type=int, default=8, 
                      help="Maximum batch size")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9, 
                      help="GPU memory fraction to use")
    parser.add_argument("--quantization", type=str, choices=["awq", "gptq", "squeezellm"], 
                      help="Quantization method")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    
    args = parser.parse_args()
    
    # Parse PD ratio
    prefill, decode = map(int, args.pd_ratio.split(":"))
    
    # Create configuration
    config = PDConfig(
        model_name=args.model,
        pd_ratio=(prefill, decode),
        engine_type=EngineType(args.engine),
        tensor_parallel_size=args.tp_size,
        data_server_url=args.data_server,
        max_batch_size=args.max_batch_size,
        gpu_memory_fraction=args.gpu_memory_fraction,
        quantization=args.quantization
    )
    
    # Create and initialize scheduler
    scheduler = PDScheduler(config)
    await scheduler.initialize()
    
    # Start scheduler in background
    scheduler_task = asyncio.create_task(scheduler.run())
    
    if args.demo:
        # Run interactive demo
        await run_interactive_demo(scheduler)
    else:
        # Submit test requests
        test_prompts = [
            "What is machine learning?",
            "Explain quantum computing.",
            "Write a haiku about AI.",
            "What are the benefits of exercise?",
            "How does photosynthesis work?",
        ]
        
        request_ids = []
        for prompt in test_prompts:
            request_id = await scheduler.submit_request({
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.7
            })
            request_ids.append(request_id)
            logger.info(f"Submitted request {request_id}: {prompt[:30]}...")
        
        # Wait for results
        for request_id in request_ids:
            result = await scheduler.get_result(request_id)
            if result:
                logger.info(f"Completed {request_id}: {result['tokens_generated']} tokens in {result['total_time']:.2f}s")
        
        # Keep running
        await scheduler_task


if __name__ == "__main__":
    asyncio.run(main())
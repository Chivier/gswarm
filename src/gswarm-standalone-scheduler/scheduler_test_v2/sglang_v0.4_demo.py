#!/usr/bin/env python3
"""
SGLang v0.4+ Demo for PD-Separated Inference
Updated for the latest SGLang API (December 2024)
"""

import os
import sys
import time
import json
import asyncio
import requests
from typing import Dict, List, Optional, Tuple
import subprocess

# Check dependencies
try:
    import orjson  # Required dependency
except ImportError:
    print("ERROR: orjson not installed. Please run: pip install orjson")
    sys.exit(1)


class SGLangServerManager:
    """Manager for SGLang server instances"""
    
    def __init__(self):
        self.processes = []
        
    def launch_server(self, model_path: str, port: int, 
                     tp_size: int = 1, dp_size: int = 1,
                     mem_fraction: float = 0.9,
                     disable_radix_cache: bool = False,
                     grammar_backend: str = "xgrammar") -> subprocess.Popen:
        """Launch an SGLang server with specified configuration"""
        
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--tp", str(tp_size),  # Tensor parallelism
            "--mem-fraction-static", str(mem_fraction),
            "--host", "0.0.0.0"
        ]
        
        # Add data parallelism if specified
        if dp_size > 1:
            cmd.extend(["--dp", str(dp_size)])
            
        # Disable radix cache for prefill servers
        if disable_radix_cache:
            cmd.append("--disable-radix-cache")
            
        # Use xgrammar for structured outputs
        if grammar_backend:
            cmd.extend(["--grammar-backend", grammar_backend])
        
        print(f"Launching SGLang server on port {port}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Launch server
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.processes.append(process)
        
        # Wait for server to be ready
        if self._wait_for_server(port):
            print(f"✓ Server started on port {port}")
            return process
        else:
            raise RuntimeError(f"Failed to start server on port {port}")
            
    def _wait_for_server(self, port: int, timeout: int = 60) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check health endpoint
                response = requests.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
            
        return False
        
    def shutdown_all(self):
        """Shutdown all managed servers"""
        print("\nShutting down all servers...")
        for proc in self.processes:
            proc.terminate()
            proc.wait()
        print("✓ All servers stopped")


class PDSeparatedClient:
    """Client for PD-separated inference using SGLang v0.4+"""
    
    def __init__(self, prefill_urls: List[str], decode_urls: List[str]):
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        self.prefill_idx = 0
        self.decode_idx = 0
        
    def _get_next_prefill_url(self) -> str:
        """Round-robin selection of prefill server"""
        url = self.prefill_urls[self.prefill_idx]
        self.prefill_idx = (self.prefill_idx + 1) % len(self.prefill_urls)
        return url
        
    def _get_next_decode_url(self) -> str:
        """Round-robin selection of decode server"""
        url = self.decode_urls[self.decode_idx]
        self.decode_idx = (self.decode_idx + 1) % len(self.decode_urls)
        return url
        
    async def process_request_native(self, prompt: str, max_tokens: int = 100) -> Dict:
        """Process request using native SGLang API"""
        
        # Phase 1: Prefill (generate just 1 token to build KV cache)
        prefill_url = self._get_next_prefill_url()
        prefill_start = time.time()
        
        prefill_response = requests.post(
            f"{prefill_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0
                }
            }
        )
        
        prefill_time = time.time() - prefill_start
        prefill_result = prefill_response.json()
        
        # Phase 2: Decode (continue generation)
        # In real implementation, we'd transfer KV cache
        # For demo, we simulate by continuing from prompt
        decode_url = self._get_next_decode_url()
        decode_start = time.time()
        
        decode_response = requests.post(
            f"{decode_url}/generate",
            json={
                "text": prompt + prefill_result["text"],
                "sampling_params": {
                    "max_new_tokens": max_tokens - 1,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        )
        
        decode_time = time.time() - decode_start
        decode_result = decode_response.json()
        
        return {
            "prompt": prompt,
            "generated_text": prefill_result["text"] + decode_result["text"],
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "prefill_server": prefill_url,
            "decode_server": decode_url
        }
        
    async def process_request_openai(self, prompt: str, max_tokens: int = 100) -> Dict:
        """Process request using OpenAI-compatible API"""
        
        # Phase 1: Prefill
        prefill_url = self._get_next_prefill_url()
        prefill_start = time.time()
        
        prefill_response = requests.post(
            f"{prefill_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0
            }
        )
        
        prefill_time = time.time() - prefill_start
        prefill_result = prefill_response.json()
        first_token = prefill_result["choices"][0]["message"]["content"]
        
        # Phase 2: Decode
        decode_url = self._get_next_decode_url()
        decode_start = time.time()
        
        decode_response = requests.post(
            f"{decode_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt + first_token}],
                "max_tokens": max_tokens - 1,
                "temperature": 0.7
            }
        )
        
        decode_time = time.time() - decode_start
        decode_result = decode_response.json()
        
        return {
            "prompt": prompt,
            "generated_text": first_token + decode_result["choices"][0]["message"]["content"],
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "prefill_server": prefill_url,
            "decode_server": decode_url
        }


async def run_benchmark(client: PDSeparatedClient, prompts: List[str], use_openai_api: bool = False):
    """Run benchmark with multiple prompts"""
    
    print(f"\nRunning benchmark with {len(prompts)} prompts...")
    print(f"API: {'OpenAI-compatible' if use_openai_api else 'Native SGLang'}")
    print("-" * 70)
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Request {i+1}/{len(prompts)}]")
        print(f"Prompt: {prompt[:50]}...")
        
        if use_openai_api:
            result = await client.process_request_openai(prompt, max_tokens=50)
        else:
            result = await client.process_request_native(prompt, max_tokens=50)
            
        results.append(result)
        
        print(f"Generated: {result['generated_text'][:80]}...")
        print(f"Prefill: {result['prefill_time']:.3f}s ({result['prefill_server']})")
        print(f"Decode: {result['decode_time']:.3f}s ({result['decode_server']})")
        print(f"Total: {result['total_time']:.3f}s")
        
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    avg_prefill = sum(r["prefill_time"] for r in results) / len(results)
    avg_decode = sum(r["decode_time"] for r in results) / len(results)
    avg_total = sum(r["total_time"] for r in results) / len(results)
    
    print(f"\nAverage times for {len(results)} requests:")
    print(f"  Prefill: {avg_prefill:.3f}s ({(avg_prefill/avg_total)*100:.1f}%)")
    print(f"  Decode:  {avg_decode:.3f}s ({(avg_decode/avg_total)*100:.1f}%)")
    print(f"  Total:   {avg_total:.3f}s")
    
    throughput = len(results) / sum(r["total_time"] for r in results)
    print(f"\nThroughput: {throughput:.2f} requests/second")
    
    return results


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang v0.4+ PD-Separated Inference Demo")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", 
                      help="Model to use")
    parser.add_argument("--prefill-ports", type=str, default="30000,30001",
                      help="Comma-separated prefill server ports")
    parser.add_argument("--decode-ports", type=str, default="30002,30003,30004",
                      help="Comma-separated decode server ports")
    parser.add_argument("--tp", type=int, default=1,
                      help="Tensor parallelism size")
    parser.add_argument("--dp", type=int, default=1,
                      help="Data parallelism size")
    parser.add_argument("--openai-api", action="store_true",
                      help="Use OpenAI-compatible API")
    parser.add_argument("--skip-launch", action="store_true",
                      help="Skip server launch (assume already running)")
    
    args = parser.parse_args()
    
    # Parse ports
    prefill_ports = [int(p) for p in args.prefill_ports.split(",")]
    decode_ports = [int(p) for p in args.decode_ports.split(",")]
    
    print("\n" + "=" * 70)
    print("SGLANG v0.4+ PD-SEPARATED INFERENCE DEMO")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Prefill servers: {len(prefill_ports)} (ports: {prefill_ports})")
    print(f"  Decode servers: {len(decode_ports)} (ports: {decode_ports})")
    print(f"  Tensor Parallel: {args.tp}")
    print(f"  Data Parallel: {args.dp}")
    
    # Server manager
    manager = SGLangServerManager()
    
    try:
        if not args.skip_launch:
            # Launch prefill servers
            print("\nLaunching prefill servers...")
            for port in prefill_ports:
                manager.launch_server(
                    model_path=args.model,
                    port=port,
                    tp_size=args.tp,
                    dp_size=args.dp,
                    mem_fraction=0.3,  # Less memory for prefill
                    disable_radix_cache=True  # Disable for variable-length prefill
                )
                time.sleep(5)  # Stagger launches
                
            # Launch decode servers
            print("\nLaunching decode servers...")
            for port in decode_ports:
                manager.launch_server(
                    model_path=args.model,
                    port=port,
                    tp_size=args.tp,
                    dp_size=args.dp,
                    mem_fraction=0.5,  # More memory for decode
                    disable_radix_cache=False  # Enable for decode
                )
                time.sleep(5)
                
        # Create client
        prefill_urls = [f"http://localhost:{p}" for p in prefill_ports]
        decode_urls = [f"http://localhost:{p}" for p in decode_ports]
        client = PDSeparatedClient(prefill_urls, decode_urls)
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks in simple terms.",
            "How does deep learning work?",
            "What are the benefits of artificial intelligence?",
            "Describe the transformer architecture.",
            "What is natural language processing?",
        ]
        
        # Run benchmark
        asyncio.run(run_benchmark(client, test_prompts, use_openai_api=args.openai_api))
        
        print("\n" + "=" * 70)
        print("KEY INSIGHTS")
        print("=" * 70)
        print("\n1. PD-Separation with SGLang v0.4+:")
        print("   - Zero-overhead batch scheduler")
        print("   - Cache-aware load balancing")
        print("   - Fast structured outputs with xgrammar")
        
        print("\n2. Performance Optimizations:")
        print("   - RadixAttention for KV cache sharing")
        print("   - FlashInfer for efficient attention")
        print("   - Data parallelism for DeepSeek models")
        
        print("\n3. Deployment Benefits:")
        print("   - Independent scaling of prefill/decode")
        print("   - Better resource utilization")
        print("   - Support for both native and OpenAI APIs")
        
    finally:
        if not args.skip_launch:
            manager.shutdown_all()
            
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
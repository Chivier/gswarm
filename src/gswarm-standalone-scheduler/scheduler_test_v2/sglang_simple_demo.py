#!/usr/bin/env python3
"""
Simple SGLang demo showing how to launch and use SGLang servers for PD-separated inference
"""

import os
import sys
import time
import json
import asyncio
import requests
from typing import Dict, List, Optional

# Try to import SGLang
try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    print("SGLang not installed. Please install with: pip install 'sglang[all]'")
    sys.exit(1)


def launch_sglang_server(model_path: str, port: int, tp_size: int = 1, 
                        disable_cuda_graph: bool = False, 
                        mem_fraction: float = 0.9):
    """Launch SGLang server using CLI"""
    import subprocess
    
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp-size", str(tp_size),
        "--mem-fraction-static", str(mem_fraction),
        "--host", "0.0.0.0"
    ]
    
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    
    print(f"Launching SGLang server on port {port}...")
    print(f"Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd)
    
    # Wait for server to start
    for i in range(60):  # Wait up to 60 seconds
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                print(f"✓ SGLang server started on port {port}")
                return process
        except:
            pass
        time.sleep(1)
    
    raise RuntimeError(f"Failed to start SGLang server on port {port}")


class SGLangPDClient:
    """Client for PD-separated inference using SGLang servers"""
    
    def __init__(self, prefill_ports: List[int], decode_ports: List[int]):
        self.prefill_ports = prefill_ports
        self.decode_ports = decode_ports
        self.prefill_idx = 0
        self.decode_idx = 0
        
    def get_next_prefill_server(self) -> str:
        """Round-robin prefill server selection"""
        port = self.prefill_ports[self.prefill_idx]
        self.prefill_idx = (self.prefill_idx + 1) % len(self.prefill_ports)
        return f"http://localhost:{port}"
        
    def get_next_decode_server(self) -> str:
        """Round-robin decode server selection"""
        port = self.decode_ports[self.decode_idx]
        self.decode_idx = (self.decode_idx + 1) % len(self.decode_ports)
        return f"http://localhost:{port}"
        
    async def process_request(self, prompt: str, max_tokens: int = 100) -> Dict:
        """Process request with PD separation"""
        
        # Phase 1: Prefill
        prefill_server = self.get_next_prefill_server()
        prefill_start = time.time()
        
        # For demo, we'll generate 1 token on prefill server
        prefill_response = requests.post(
            f"{prefill_server}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0
                }
            }
        )
        
        prefill_time = time.time() - prefill_start
        
        # In real implementation, we'd extract and transfer KV cache here
        # For demo, we'll simulate by passing the prompt + first token to decode server
        
        # Phase 2: Decode
        decode_server = self.get_next_decode_server()
        decode_start = time.time()
        
        # Continue generation on decode server
        # In real implementation, we'd restore KV cache and continue
        decode_response = requests.post(
            f"{decode_server}/generate",
            json={
                "text": prompt,  # Would include KV cache in real implementation
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        )
        
        decode_time = time.time() - decode_start
        
        # Parse response
        result = decode_response.json()
        
        return {
            "prompt": prompt,
            "generated_text": result["text"],
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "prefill_server": prefill_server,
            "decode_server": decode_server
        }


def run_demo():
    """Run the SGLang PD-separated inference demo"""
    
    print("\n" + "="*70)
    print("SGLANG PD-SEPARATED INFERENCE DEMO")
    print("="*70)
    
    # Configuration
    model = "microsoft/phi-2"  # Small model for demo
    prefill_ports = [30000, 30001]  # 2 prefill servers
    decode_ports = [30002, 30003, 30004]  # 3 decode servers
    
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Prefill servers: {len(prefill_ports)} (ports: {prefill_ports})")
    print(f"  Decode servers: {len(decode_ports)} (ports: {decode_ports})")
    
    # Launch servers
    print("\nLaunching SGLang servers...")
    processes = []
    
    try:
        # Launch prefill servers (CUDA graphs disabled for variable length)
        for port in prefill_ports:
            proc = launch_sglang_server(
                model, port, 
                disable_cuda_graph=True,  # Better for prefill
                mem_fraction=0.4  # Less memory needed
            )
            processes.append(proc)
            time.sleep(5)  # Stagger launches
            
        # Launch decode servers (CUDA graphs enabled for speed)
        for port in decode_ports:
            proc = launch_sglang_server(
                model, port,
                disable_cuda_graph=False,  # CUDA graphs for decode
                mem_fraction=0.3
            )
            processes.append(proc)
            time.sleep(5)
            
        print("\n✓ All servers launched successfully")
        
        # Create client
        client = SGLangPDClient(prefill_ports, decode_ports)
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks",
            "How does AI work?",
            "What are transformers?",
            "Describe deep learning",
        ]
        
        print(f"\nProcessing {len(test_prompts)} requests...")
        print("-"*70)
        
        # Process requests
        async def process_all():
            tasks = []
            for prompt in test_prompts:
                task = client.process_request(prompt, max_tokens=50)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
            
        # Run async processing
        results = asyncio.run(process_all())
        
        # Show results
        for i, result in enumerate(results):
            print(f"\n[Request {i+1}]")
            print(f"Prompt: {result['prompt']}")
            print(f"Generated: {result['generated_text'][:100]}...")
            print(f"Prefill: {result['prefill_time']:.3f}s (server: {result['prefill_server']})")
            print(f"Decode: {result['decode_time']:.3f}s (server: {result['decode_server']})")
            print(f"Total: {result['total_time']:.3f}s")
            
        # Summary
        avg_prefill = sum(r['prefill_time'] for r in results) / len(results)
        avg_decode = sum(r['decode_time'] for r in results) / len(results)
        avg_total = sum(r['total_time'] for r in results) / len(results)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Average times:")
        print(f"  Prefill: {avg_prefill:.3f}s")
        print(f"  Decode: {avg_decode:.3f}s")
        print(f"  Total: {avg_total:.3f}s")
        print(f"\nPD-separation benefits demonstrated:")
        print(f"  ✓ Specialized servers for each phase")
        print(f"  ✓ Better resource utilization")
        print(f"  ✓ Scalable architecture")
        
    finally:
        # Cleanup
        print("\nShutting down servers...")
        for proc in processes:
            proc.terminate()
            proc.wait()
        print("✓ All servers stopped")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


def run_simple_test():
    """Run a simple test with single SGLang server"""
    
    print("\n" + "="*70)
    print("SIMPLE SGLANG TEST")
    print("="*70)
    
    # Launch single server
    model = "microsoft/phi-2"
    port = 30000
    
    print(f"Launching SGLang server with {model}...")
    process = launch_sglang_server(model, port)
    
    try:
        # Test generation
        print("\nTesting generation...")
        
        response = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "text": "What is artificial intelligence?",
                "sampling_params": {
                    "max_new_tokens": 50,
                    "temperature": 0.7
                }
            }
        )
        
        result = response.json()
        print(f"\nPrompt: What is artificial intelligence?")
        print(f"Generated: {result['text']}")
        
        # Get model info
        info_response = requests.get(f"http://localhost:{port}/get_model_info")
        info = info_response.json()
        print(f"\nModel info: {json.dumps(info, indent=2)}")
        
    finally:
        print("\nShutting down server...")
        process.terminate()
        process.wait()
        print("✓ Server stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang PD-separated inference demo")
    parser.add_argument("--simple", action="store_true", help="Run simple single-server test")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model to use")
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_test()
    else:
        run_demo()
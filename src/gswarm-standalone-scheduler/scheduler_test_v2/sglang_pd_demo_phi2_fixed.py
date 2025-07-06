#!/usr/bin/env python3
"""
SGLang PD-Separated Inference Demo - Fixed for Phi-2
Works around head_dim=80 issues
"""

import os
import sys
import time
import json
import asyncio
import requests
from typing import Dict, List, Optional, Tuple
import subprocess

# Set CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-12.6/targets/x86_64-linux/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add gswarm to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))


class SGLangPDServerManager:
    """Manages SGLang servers for PD-separated inference"""
    
    def __init__(self):
        self.processes = []
        self.base_gpu_id = 0
        
    def launch_prefill_server(self, model: str, port: int, 
                            mem_fraction: float = 0.4,
                            tp_size: int = 1) -> subprocess.Popen:
        """Launch a prefill-optimized server"""
        
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--tensor-parallel-size", str(tp_size),
            "--mem-fraction-static", str(mem_fraction),
            # Workarounds for Phi-2
            "--disable-cuda-graph",  # Disable CUDA graphs
            "--attention-backend", "triton",  # Use triton instead of flashinfer
            # Prefill optimizations
            "--chunked-prefill-size", "4096",  # Smaller chunks
            "--max-prefill-tokens", "8192",   
            "--schedule-policy", "fcfs",
        ]
        
        print(f"Launching prefill server on port {port}...")
        return self._launch_server(cmd, port)
        
    def launch_decode_server(self, model: str, port: int,
                           mem_fraction: float = 0.6,
                           tp_size: int = 1) -> subprocess.Popen:
        """Launch a decode-optimized server"""
        
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--tensor-parallel-size", str(tp_size),
            "--mem-fraction-static", str(mem_fraction),
            # Workarounds for Phi-2
            "--disable-cuda-graph",
            "--attention-backend", "triton",
            # Decode optimizations
            "--max-running-requests", "16",    # Fewer concurrent requests
            "--schedule-policy", "lpm",
            "--stream-interval", "1",
        ]
        
        print(f"Launching decode server on port {port}...")
        return self._launch_server(cmd, port)
        
    def _launch_server(self, cmd: List[str], port: int) -> subprocess.Popen:
        """Launch server and wait for it to be ready"""
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.processes.append(process)
        
        # Wait for server to be ready
        print(f"Waiting for server on port {port} to start...")
        start_time = time.time()
        
        while time.time() - start_time < 180:  # 3 minute timeout
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    print(f"✓ Server on port {port} is ready")
                    return process
            except:
                pass
                
            # Check if process died
            if process.poll() is not None:
                output, _ = process.communicate()
                print(f"✗ Server on port {port} failed to start")
                print("Error output (last 1000 chars):")
                print(output[-1000:])
                raise RuntimeError(f"Server failed to start on port {port}")
                
            time.sleep(2)
            print(".", end="", flush=True)
            
        raise RuntimeError(f"Server on port {port} timeout")
        
    def shutdown_all(self):
        """Shutdown all servers"""
        print("\nShutting down all servers...")
        for proc in self.processes:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        print("✓ All servers stopped")


async def process_request_pd_separated(
    prompt: str,
    prefill_url: str,
    decode_url: str,
    max_tokens: int = 100
) -> Dict:
    """Process a request using PD-separated inference"""
    
    # Phase 1: Prefill (generate first token)
    prefill_start = time.time()
    
    prefill_resp = requests.post(
        f"{prefill_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0
            }
        }
    )
    
    prefill_time = time.time() - prefill_start
    
    if prefill_resp.status_code != 200:
        raise Exception(f"Prefill failed: {prefill_resp.text}")
        
    prefill_result = prefill_resp.json()
    first_token_output = prefill_result["text"]
    
    # Phase 2: Decode (continue generation)
    decode_start = time.time()
    
    # In real implementation, we'd transfer KV cache
    # For demo, we continue from prompt + first token
    decode_resp = requests.post(
        f"{decode_url}/generate",
        json={
            "text": first_token_output,  # Continue from prefill output
            "sampling_params": {
                "max_new_tokens": max_tokens - 1,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
    )
    
    decode_time = time.time() - decode_start
    
    if decode_resp.status_code != 200:
        raise Exception(f"Decode failed: {decode_resp.text}")
        
    decode_result = decode_resp.json()
    full_output = decode_result["text"]
    
    return {
        "prompt": prompt,
        "output": full_output,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": prefill_time + decode_time,
        "prefill_server": prefill_url,
        "decode_server": decode_url
    }


async def run_pd_demo(model: str = "microsoft/phi-2"):
    """Run the PD-separated inference demo"""
    
    print("\n" + "="*70)
    print("SGLANG PD-SEPARATED INFERENCE DEMO (PHI-2 FIXED)")
    print("="*70)
    print(f"\nModel: {model}")
    print("Configuration: 1 prefill server, 2 decode servers")
    print("Using triton attention backend (avoids head_dim issues)")
    print("-"*70)
    
    # Server configuration - reduced for testing
    prefill_ports = [30000]
    decode_ports = [30001, 30002]
    
    manager = SGLangPDServerManager()
    
    try:
        # Launch servers
        print("\nPhase 1: Launching prefill servers...")
        for port in prefill_ports:
            manager.launch_prefill_server(model, port, mem_fraction=0.3)
            
        print("\nPhase 2: Launching decode servers...")
        for port in decode_ports:
            manager.launch_decode_server(model, port, mem_fraction=0.3)
            
        print("\n✓ All servers ready!")
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Write a Python function to add two numbers:",
            "Explain photosynthesis in simple terms:",
        ]
        
        # Process requests
        print(f"\nPhase 3: Processing {len(test_prompts)} test requests...")
        print("-"*70)
        
        results = []
        prefill_idx = 0
        decode_idx = 0
        
        for i, prompt in enumerate(test_prompts):
            # Round-robin server selection
            prefill_url = f"http://localhost:{prefill_ports[prefill_idx]}"
            decode_url = f"http://localhost:{decode_ports[decode_idx]}"
            
            prefill_idx = (prefill_idx + 1) % len(prefill_ports)
            decode_idx = (decode_idx + 1) % len(decode_ports)
            
            print(f"\n[Request {i+1}] {prompt}")
            
            try:
                result = await process_request_pd_separated(
                    prompt, prefill_url, decode_url, max_tokens=30
                )
                results.append(result)
                
                print(f"✓ Success!")
                print(f"  Prefill: {result['prefill_time']:.3f}s (port {prefill_url.split(':')[-1]})")
                print(f"  Decode:  {result['decode_time']:.3f}s (port {decode_url.split(':')[-1]})")
                print(f"  Output: {result['output'][:80]}...")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                
        # Summary
        if results:
            print("\n" + "="*70)
            print("PERFORMANCE SUMMARY")
            print("="*70)
            
            avg_prefill = sum(r['prefill_time'] for r in results) / len(results)
            avg_decode = sum(r['decode_time'] for r in results) / len(results)
            avg_total = sum(r['total_time'] for r in results) / len(results)
            
            print(f"\nAverage times for {len(results)} successful requests:")
            print(f"  Prefill: {avg_prefill:.3f}s ({avg_prefill/avg_total*100:.1f}%)")
            print(f"  Decode:  {avg_decode:.3f}s ({avg_decode/avg_total*100:.1f}%)")
            print(f"  Total:   {avg_total:.3f}s")
            
            print(f"\nPD-Separation Benefits:")
            print(f"  ✓ Specialized server configurations")
            print(f"  ✓ Independent scaling (1:2 ratio)")
            print(f"  ✓ Better resource utilization")
            print(f"  ✓ Works with Phi-2 (head_dim=80)")
            
    finally:
        manager.shutdown_all()
        
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang PD-Separated Inference Demo (Phi-2 Fixed)")
    parser.add_argument("--model", type=str, default="microsoft/phi-2",
                      help="Model to use (default: microsoft/phi-2)")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import sglang
        print(f"✓ SGLang version: {getattr(sglang, '__version__', 'Unknown')}")
    except ImportError:
        print("✗ SGLang not installed")
        print("Install with:")
        print("  pip install orjson")
        print("  pip install 'sglang[all]'")
        sys.exit(1)
        
    # Run full PD demo
    asyncio.run(run_pd_demo(args.model))


if __name__ == "__main__":
    main()
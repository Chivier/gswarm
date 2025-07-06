#!/usr/bin/env python3
"""
Minimal SGLang PD-Separated Demo
Shows the concept with just 2 servers
"""

import os
import sys
import time
import requests
import subprocess

# Set CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-12.6/targets/x86_64-linux/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

def launch_server(port: int, server_type: str, model: str = "microsoft/phi-2"):
    """Launch a single SGLang server"""
    
    # Base command
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--mem-fraction-static", "0.3",  # Low memory to fit both
        "--disable-cuda-graph",  # For Phi-2 compatibility
        "--attention-backend", "triton",
    ]
    
    # Add type-specific optimizations
    if server_type == "prefill":
        cmd.extend([
            "--chunked-prefill-size", "4096",
            "--schedule-policy", "fcfs"
        ])
    else:  # decode
        cmd.extend([
            "--max-running-requests", "8",
            "--schedule-policy", "lpm"
        ])
    
    print(f"Launching {server_type} server on port {port}...")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Wait for server
    start_time = time.time()
    while time.time() - start_time < 120:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
            if resp.status_code == 200:
                print(f"✓ {server_type.capitalize()} server ready on port {port}")
                return process
        except:
            pass
            
        if process.poll() is not None:
            output, _ = process.communicate()
            print(f"✗ {server_type.capitalize()} server failed")
            print("Last output:", output[-500:])
            return None
            
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n✗ {server_type.capitalize()} server timeout")
    process.terminate()
    return None

def test_pd_inference():
    """Test PD-separated inference"""
    
    print("\n" + "="*70)
    print("TESTING PD-SEPARATED INFERENCE")
    print("="*70)
    
    # Test prompt
    prompt = "What is artificial intelligence? Explain in one sentence:"
    print(f"\nPrompt: {prompt}")
    
    # Phase 1: Prefill (get first token)
    print("\n[Phase 1] Prefill...")
    start = time.time()
    
    resp1 = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0
            }
        }
    )
    
    prefill_time = time.time() - start
    
    if resp1.status_code != 200:
        print(f"✗ Prefill failed: {resp1.status_code}")
        return
        
    result1 = resp1.json()
    prompt_with_first = result1["text"]
    print(f"✓ Prefill done in {prefill_time:.3f}s")
    print(f"  First token output: '{prompt_with_first}'")
    
    # Phase 2: Decode (continue generation)
    print("\n[Phase 2] Decode...")
    start = time.time()
    
    resp2 = requests.post(
        "http://localhost:30001/generate",
        json={
            "text": prompt_with_first,
            "sampling_params": {
                "max_new_tokens": 30,
                "temperature": 0.7
            }
        }
    )
    
    decode_time = time.time() - start
    
    if resp2.status_code != 200:
        print(f"✗ Decode failed: {resp2.status_code}")
        return
        
    result2 = resp2.json()
    final_output = result2["text"]
    print(f"✓ Decode done in {decode_time:.3f}s")
    print(f"  Final output: '{final_output}'")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Prefill time: {prefill_time:.3f}s")
    print(f"Decode time:  {decode_time:.3f}s")
    print(f"Total time:   {prefill_time + decode_time:.3f}s")
    print(f"\nThis demonstrates PD-separation where:")
    print("- Prefill server handles prompt processing")
    print("- Decode server handles token generation")
    print("- In production, KV cache would be transferred between servers")

def main():
    print("="*70)
    print("MINIMAL SGLANG PD-SEPARATED DEMO")
    print("="*70)
    
    # Check SGLang
    try:
        import sglang
        print(f"✓ SGLang version: {sglang.__version__}")
    except ImportError:
        print("✗ SGLang not installed")
        sys.exit(1)
    
    # Launch servers
    servers = []
    
    print("\nLaunching servers...")
    
    # Prefill server
    p1 = launch_server(30000, "prefill")
    if p1:
        servers.append(p1)
    else:
        print("Failed to launch prefill server")
        sys.exit(1)
    
    # Decode server
    p2 = launch_server(30001, "decode")
    if p2:
        servers.append(p2)
    else:
        print("Failed to launch decode server")
        for s in servers:
            s.terminate()
        sys.exit(1)
    
    try:
        # Wait for initialization
        time.sleep(5)
        
        # Test PD inference
        test_pd_inference()
        
        # Compare with single server
        print("\n" + "="*70)
        print("COMPARISON WITH SINGLE SERVER")
        print("="*70)
        print("\nFor comparison, a single server would:")
        print("- Handle both prefill and decode")
        print("- Cannot optimize separately for each phase")
        print("- Limited scaling options")
        
    finally:
        # Cleanup
        print("\nShutting down servers...")
        for s in servers:
            s.terminate()
            try:
                s.wait(timeout=5)
            except subprocess.TimeoutExpired:
                s.kill()
                s.wait()
        print("✓ All servers stopped")
    
    print("\n✓ Demo completed successfully!")

if __name__ == "__main__":
    main()
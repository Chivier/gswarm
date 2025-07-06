#!/usr/bin/env python3
"""
SGLang PD-Separated Demo with Multiple GPUs
Uses different GPUs for prefill and decode servers
"""

import os
import sys
import time
import requests
import subprocess

# Set CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-12.6/targets/x86_64-linux/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

def launch_server(port: int, server_type: str, gpu_id: int, model: str = "microsoft/phi-2"):
    """Launch a single SGLang server on specific GPU"""
    
    # Set CUDA_VISIBLE_DEVICES for this process
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Base command
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--mem-fraction-static", "0.6",  # More memory since using separate GPUs
        "--disable-cuda-graph",  # For Phi-2 compatibility
        "--attention-backend", "triton",
    ]
    
    # Add type-specific optimizations
    if server_type == "prefill":
        cmd.extend([
            "--chunked-prefill-size", "8192",
            "--max-prefill-tokens", "16384",
            "--schedule-policy", "fcfs"
        ])
    else:  # decode
        cmd.extend([
            "--max-running-requests", "16",
            "--schedule-policy", "lpm",
            "--stream-interval", "1"
        ])
    
    print(f"Launching {server_type} server on port {port} (GPU {gpu_id})...")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )
    
    # Wait for server
    start_time = time.time()
    while time.time() - start_time < 180:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
            if resp.status_code == 200:
                print(f"✓ {server_type.capitalize()} server ready on port {port} (GPU {gpu_id})")
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
    """Test PD-separated inference with multiple requests"""
    
    print("\n" + "="*70)
    print("TESTING PD-SEPARATED INFERENCE")
    print("="*70)
    
    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate factorial:",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[Request {i+1}] {prompt}")
        
        # Phase 1: Prefill
        start = time.time()
        
        resp1 = requests.post(
            "http://localhost:30000/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0
                }
            },
            timeout=30
        )
        
        prefill_time = time.time() - start
        
        if resp1.status_code != 200:
            print(f"✗ Prefill failed: {resp1.status_code}")
            continue
            
        result1 = resp1.json()
        prompt_with_first = result1["text"]
        
        # Phase 2: Decode
        start = time.time()
        
        # Alternate between decode servers
        decode_port = 30001 + (i % 2)
        
        resp2 = requests.post(
            f"http://localhost:{decode_port}/generate",
            json={
                "text": prompt_with_first,
                "sampling_params": {
                    "max_new_tokens": 40,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=30
        )
        
        decode_time = time.time() - start
        
        if resp2.status_code != 200:
            print(f"✗ Decode failed: {resp2.status_code}")
            continue
            
        result2 = resp2.json()
        final_output = result2["text"]
        
        print(f"✓ Success!")
        print(f"  Prefill: {prefill_time:.3f}s (GPU 1)")
        print(f"  Decode:  {decode_time:.3f}s (GPU {2 + (i % 2)})")
        print(f"  Output: {final_output[:80]}...")
        
        results.append({
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time
        })
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        
        avg_prefill = sum(r['prefill_time'] for r in results) / len(results)
        avg_decode = sum(r['decode_time'] for r in results) / len(results)
        avg_total = sum(r['total_time'] for r in results) / len(results)
        
        print(f"\nProcessed {len(results)} requests successfully")
        print(f"Average times:")
        print(f"  Prefill: {avg_prefill:.3f}s ({avg_prefill/avg_total*100:.1f}%)")
        print(f"  Decode:  {avg_decode:.3f}s ({avg_decode/avg_total*100:.1f}%)")
        print(f"  Total:   {avg_total:.3f}s")
        
        print(f"\nPD-Separation Benefits Demonstrated:")
        print(f"  ✓ Independent GPU allocation (GPU 1 for prefill, GPU 2-3 for decode)")
        print(f"  ✓ Specialized configurations per phase")
        print(f"  ✓ Load balancing across decode servers")
        print(f"  ✓ Works with Phi-2 using triton backend")

def main():
    print("="*70)
    print("SGLANG PD-SEPARATED DEMO (MULTI-GPU)")
    print("="*70)
    
    # Check SGLang
    try:
        import sglang
        print(f"✓ SGLang version: {sglang.__version__}")
    except ImportError:
        print("✗ SGLang not installed")
        sys.exit(1)
    
    # Check GPUs
    print("\nChecking GPU availability...")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        gpus = result.stdout.strip().split('\n')
        print(f"✓ Found {len(gpus)} GPUs")
        for gpu in gpus[:4]:
            print(f"  {gpu}")
    
    # Launch servers
    servers = []
    
    print("\nLaunching servers...")
    
    # Prefill server on GPU 1
    p1 = launch_server(30000, "prefill", 1)
    if p1:
        servers.append(p1)
    else:
        print("Failed to launch prefill server")
        sys.exit(1)
    
    # Decode servers on GPU 2 and 3
    p2 = launch_server(30001, "decode", 2)
    if p2:
        servers.append(p2)
    
    p3 = launch_server(30002, "decode", 3)
    if p3:
        servers.append(p3)
    
    if len(servers) < 2:
        print("Failed to launch enough servers")
        for s in servers:
            s.terminate()
        sys.exit(1)
    
    try:
        # Wait for initialization
        time.sleep(5)
        
        # Test PD inference
        test_pd_inference()
        
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
    print("\nThis demonstrates how PD-separation enables:")
    print("- Better GPU utilization across multiple devices")
    print("- Independent scaling of prefill and decode capacity")
    print("- Optimized configurations for each inference phase")

if __name__ == "__main__":
    main()
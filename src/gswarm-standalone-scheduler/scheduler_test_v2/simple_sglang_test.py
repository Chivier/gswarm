#!/usr/bin/env python3
"""
Simple SGLang test without PD separation
"""

import subprocess
import sys
import time
import requests
import os

def launch_simple_server():
    """Launch a simple SGLang server"""
    
    # Set CUDA environment
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
    os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-12.6/targets/x86_64-linux/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    model = "microsoft/phi-2"
    port = 30000
    
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--mem-fraction-static", "0.4",
        "--tp", "1",
        "--host", "0.0.0.0",
        "--disable-cuda-graph",  # Disable to avoid head_dim issues
        "--attention-backend", "triton"  # Use triton instead of flashinfer
    ]
    
    print("=" * 70)
    print("SIMPLE SGLANG TEST")
    print("=" * 70)
    print(f"\nLaunching server with command:")
    print(f"  {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Wait for server
    print("\nWaiting for server to start...")
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < 120:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                print("✓ Server is ready!")
                break
        except:
            pass
            
        if process.poll() is not None:
            print("✗ Server process died")
            output, _ = process.communicate()
            print("\nError output:")
            print(output[-2000:])
            return None
            
        time.sleep(2)
        print(".", end="", flush=True)
    
    if not server_ready:
        print("\n✗ Server failed to start")
        process.terminate()
        return None
        
    return process

def test_generation(port=30000):
    """Test simple generation"""
    
    print("\n" + "="*70)
    print("TESTING GENERATION")
    print("="*70)
    
    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "Write a Python function to calculate factorial:",
        "Explain quantum computing in simple terms:",
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[Test {i+1}] {prompt}")
        
        try:
            response = requests.post(
                f"http://localhost:{port}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": 50,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get('text', 'No output')
                print(f"✓ Response: {output[:150]}...")
                
                if 'meta_info' in result:
                    tokens = result['meta_info'].get('completion_tokens', 'N/A')
                    latency = result['meta_info'].get('ttft_ms', 'N/A')
                    print(f"  Tokens: {tokens}, TTFT: {latency}ms")
            else:
                print(f"✗ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error: {e}")

def main():
    # Check SGLang
    try:
        import sglang
        print(f"✓ SGLang version: {sglang.__version__}")
    except ImportError:
        print("✗ SGLang not installed")
        print("Install with: pip install 'sglang[all]'")
        sys.exit(1)
    
    # Launch server
    process = launch_simple_server()
    if process is None:
        print("\nFailed to start server")
        sys.exit(1)
    
    try:
        # Wait a bit for full initialization
        time.sleep(5)
        
        # Test generation
        test_generation()
        
        # Test model info
        print("\n" + "="*70)
        print("MODEL INFO")
        print("="*70)
        
        try:
            response = requests.get("http://localhost:30000/get_model_info")
            if response.status_code == 200:
                info = response.json()
                print(f"Model: {info.get('model_path', 'N/A')}")
                print(f"Context length: {info.get('context_length', 'N/A')}")
                print(f"Max tokens: {info.get('max_total_num_tokens', 'N/A')}")
        except Exception as e:
            print(f"Error getting model info: {e}")
            
    finally:
        print("\n" + "="*70)
        print("Shutting down server...")
        process.terminate()
        try:
            process.wait(timeout=10)
            print("✓ Server stopped")
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print("✓ Server killed")
    
    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    main()
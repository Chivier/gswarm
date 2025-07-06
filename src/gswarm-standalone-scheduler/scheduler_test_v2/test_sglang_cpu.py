#!/usr/bin/env python3
"""
Test SGLang with CPU (no GPU required)
"""

import sys
import time
import requests

def test_cpu_server():
    """Test SGLang server on CPU"""
    import subprocess
    
    print("Testing SGLang on CPU (no GPU required)...")
    print("=" * 70)
    
    # Use a very small model for CPU testing
    model = "microsoft/phi-2"  # 2.7B model
    port = 30000
    
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--device", "cpu",  # Force CPU
        "--mem-fraction-static", "0.8"
    ]
    
    print(f"Launching CPU server...")
    print(f"Command: {' '.join(cmd)}")
    print("\nNOTE: CPU inference will be slow. This is just for testing.")
    print("Downloading model if needed (first run only)...")
    
    # Launch server
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Wait for server
    print("\nWaiting for server to start (may take 2-3 minutes on CPU)...")
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < 300:  # 5 minute timeout for CPU
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                print("\n✓ Server started successfully on CPU")
                break
        except:
            pass
            
        if process.poll() is not None:
            print("\n✗ Server process died")
            output, _ = process.communicate()
            print("Error output:")
            print(output[-1000:])
            return None
            
        time.sleep(2)
        print(".", end="", flush=True)
    
    if not server_ready:
        print("\n✗ Server failed to start")
        process.terminate()
        return None
        
    return process

def test_simple_generation(port=30000):
    """Test simple generation on CPU"""
    print("\nTesting generation (will be slow on CPU)...")
    
    try:
        response = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "text": "Hello, my name is",
                "sampling_params": {
                    "max_new_tokens": 10,  # Very short for CPU
                    "temperature": 0.7
                }
            },
            timeout=60  # Long timeout for CPU
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Generation successful")
            print(f"Input: Hello, my name is")
            print(f"Output: {result.get('text', 'No output')}")
        else:
            print(f"✗ Generation failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    print("=" * 70)
    print("SGLANG CPU TEST (No GPU Required)")
    print("=" * 70)
    
    # Check if SGLang is installed
    try:
        import sglang
        print(f"✓ SGLang version: {getattr(sglang, '__version__', 'Unknown')}")
    except ImportError:
        print("✗ SGLang not installed")
        print("Install with: pip install 'sglang[all]'")
        sys.exit(1)
    
    # Launch CPU server
    process = test_cpu_server()
    if process is None:
        print("\nCPU test failed. Try:")
        print("1. Install all dependencies: pip install 'sglang[all]' orjson")
        print("2. Check available memory")
        print("3. Try a smaller model")
        sys.exit(1)
    
    try:
        # Test generation
        test_simple_generation()
        
    finally:
        print("\nShutting down server...")
        process.terminate()
        process.wait()
        print("✓ Server stopped")
    
    print("\n" + "=" * 70)
    print("CPU test completed!")
    print("For better performance, use GPU with:")
    print("python test_sglang_basic_fixed.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
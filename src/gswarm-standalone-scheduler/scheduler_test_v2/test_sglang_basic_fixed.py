#!/usr/bin/env python3
"""
Basic SGLang test to verify installation and functionality (Fixed version)
"""

import subprocess
import sys
import time
import requests
import json

def test_sglang_import():
    """Test if SGLang can be imported"""
    print("1. Testing SGLang import...")
    try:
        import sglang
        print(f"✓ SGLang imported successfully (version: {sglang.__version__ if hasattr(sglang, '__version__') else 'unknown'})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import SGLang: {e}")
        print("\nPlease install SGLang with:")
        print("  pip install orjson")
        print("  pip install 'sglang[all]'")
        return False

def test_launch_server():
    """Test launching SGLang server"""
    print("\n2. Testing SGLang server launch...")
    
    model = "microsoft/phi-2"  # Small model for testing
    port = 30000
    
    # Correct command without invalid arguments
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--mem-fraction-static", "0.5",
        "--tp", "1",  # Tensor parallelism
        "--host", "0.0.0.0"
    ]
    
    print(f"Launching server with command:")
    print(f"  {' '.join(cmd)}")
    
    # Launch server
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Wait for server to start
    print("Waiting for server to start (this may take 30-60 seconds)...")
    start_time = time.time()
    server_ready = False
    last_output = []
    
    while time.time() - start_time < 120:  # 120 second timeout
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                print("\n✓ Server started successfully")
                break
        except:
            pass
        
        # Check if process has died
        if process.poll() is not None:
            print("\n✗ Server process died")
            # Print output
            output, _ = process.communicate()
            print("\nServer output:")
            print("-" * 70)
            # Print last 2000 chars to see the actual error
            print(output[-2000:] if len(output) > 2000 else output)
            print("-" * 70)
            return None
            
        # Capture some output while waiting
        try:
            line = process.stdout.readline()
            if line:
                last_output.append(line.strip())
                if len(last_output) > 10:
                    last_output.pop(0)
        except:
            pass
            
        time.sleep(1)
        print(".", end="", flush=True)
    
    if not server_ready:
        print("\n✗ Server failed to start within timeout")
        print("\nLast server output:")
        for line in last_output:
            print(f"  {line}")
        process.terminate()
        process.wait()
        return None
        
    return process

def test_generation(port=30000):
    """Test text generation"""
    print("\n3. Testing text generation...")
    
    # Test native API
    print("Testing native API...")
    try:
        response = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "text": "What is artificial intelligence? Answer in one sentence:",
                "sampling_params": {
                    "max_new_tokens": 30,
                    "temperature": 0.7
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Native API generation successful")
            print(f"Input: What is artificial intelligence? Answer in one sentence:")
            print(f"Output: {result.get('text', 'No text in response')[:200]}")
            if 'meta_info' in result:
                print(f"Tokens generated: {result['meta_info'].get('completion_tokens', 'N/A')}")
        else:
            print(f"✗ Native API failed: {response.status_code}")
            print(response.text[:500])
    except Exception as e:
        print(f"✗ Native API error: {e}")
    
    # Test OpenAI API
    print("\nTesting OpenAI-compatible API...")
    try:
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "What is machine learning? Answer in one sentence:"}
                ],
                "max_tokens": 30,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("✓ OpenAI API generation successful")
            print(f"Input: What is machine learning? Answer in one sentence:")
            print(f"Output: {content[:200]}")
            print(f"Tokens used: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
        else:
            print(f"✗ OpenAI API failed: {response.status_code}")
            print(response.text[:500])
    except Exception as e:
        print(f"✗ OpenAI API error: {e}")

def check_gpu():
    """Check GPU availability"""
    print("\n4. Checking GPU availability...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ GPU(s) detected:")
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    name, total, free = parts
                    print(f"  - {name}: {free} free / {total} total")
        else:
            print("✗ No GPU detected or nvidia-smi not available")
    except:
        print("✗ Could not check GPU status")

def main():
    print("=" * 70)
    print("SGLANG BASIC FUNCTIONALITY TEST")
    print("=" * 70)
    
    # Test import
    if not test_sglang_import():
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Test server launch
    process = test_launch_server()
    if process is None:
        print("\nServer launch failed. Common issues:")
        print("1. Insufficient GPU memory - try with CPU:")
        print("   python -m sglang.launch_server --model-path microsoft/phi-2 --port 30000 --device cpu")
        print("2. Model download - first run may take time to download")
        print("3. Port already in use - check with: lsof -i :30000")
        print("4. Missing dependencies - check error output above")
        sys.exit(1)
    
    try:
        # Wait a bit for server to fully initialize
        print("\nWaiting for server to fully initialize...")
        time.sleep(5)
        
        # Test generation
        test_generation()
        
        # Get model info
        print("\n5. Getting model info...")
        try:
            response = requests.get(f"http://localhost:30000/get_model_info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print("✓ Model info retrieved:")
                print(f"  Model: {info.get('model_path', 'N/A')}")
                print(f"  Max tokens: {info.get('max_total_num_tokens', 'N/A')}")
                print(f"  Served model: {info.get('served_model', 'N/A')}")
            else:
                print("✗ Failed to get model info")
        except Exception as e:
            print(f"✗ Error getting model info: {e}")
            
    finally:
        # Cleanup
        print("\nShutting down server...")
        process.terminate()
        # Give it time to shutdown gracefully
        try:
            process.wait(timeout=10)
            print("✓ Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("⚠ Server didn't stop gracefully, killing...")
            process.kill()
            process.wait()
            print("✓ Server killed")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
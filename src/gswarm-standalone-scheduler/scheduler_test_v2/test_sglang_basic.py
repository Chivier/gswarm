#!/usr/bin/env python3
"""
Basic SGLang test to verify installation and functionality
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
    
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--mem-fraction-static", "0.5",
        "--max-new-tokens", "100"
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
    print("Waiting for server to start...")
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < 60:  # 60 second timeout
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                server_ready = True
                print("✓ Server started successfully")
                break
        except:
            pass
        
        # Check if process has died
        if process.poll() is not None:
            print("✗ Server process died")
            # Print last few lines of output
            output, _ = process.communicate()
            print("Server output:")
            print(output[-1000:])  # Last 1000 chars
            return None
            
        time.sleep(1)
        print(".", end="", flush=True)
    
    if not server_ready:
        print("\n✗ Server failed to start within timeout")
        process.terminate()
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
            print(f"Generated text: {result['text'][:100]}...")
        else:
            print(f"✗ Native API failed: {response.status_code}")
            print(response.text)
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
            print(f"Generated text: {content[:100]}...")
        else:
            print(f"✗ OpenAI API failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"✗ OpenAI API error: {e}")

def main():
    print("=" * 70)
    print("SGLANG BASIC FUNCTIONALITY TEST")
    print("=" * 70)
    
    # Test import
    if not test_sglang_import():
        sys.exit(1)
    
    # Test server launch
    process = test_launch_server()
    if process is None:
        print("\nServer launch failed. Common issues:")
        print("- Insufficient GPU memory (try a smaller model)")
        print("- Missing dependencies (run install_sglang.sh)")
        print("- Port already in use")
        sys.exit(1)
    
    try:
        # Test generation
        test_generation()
        
        # Get model info
        print("\n4. Getting model info...")
        try:
            response = requests.get("http://localhost:30000/get_model_info")
            if response.status_code == 200:
                info = response.json()
                print("✓ Model info retrieved")
                print(json.dumps(info, indent=2))
            else:
                print("✗ Failed to get model info")
        except Exception as e:
            print(f"✗ Error getting model info: {e}")
            
    finally:
        # Cleanup
        print("\nShutting down server...")
        process.terminate()
        process.wait()
        print("✓ Server stopped")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
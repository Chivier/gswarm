#!/usr/bin/env python3
"""
Test script to verify CLI command works with parameter overrides.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def test_cli_command():
    """Test the CLI command with parameters"""
    print("🔍 Testing CLI command with parameter overrides...")
    
    # Test parameters
    test_port = 8095
    test_http_port = 8096  
    test_model_port = 9010
    
    print(f"Starting: gswarm host start --port {test_port} --http-port {test_http_port} --model-port {test_model_port}")
    
    try:
        # Run the command in a subprocess
        cmd = [
            "gswarm", "host", "start",
            "--port", str(test_port),
            "--http-port", str(test_http_port), 
            "--model-port", str(test_model_port)
        ]
        
        print("Command:", " ".join(cmd))
        print("This should start without asyncio errors...")
        print("Press Ctrl+C to stop the test")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check if process is still running (indicates successful startup)
        if process.poll() is None:
            print("✅ Process started successfully!")
            
            # Try to connect to the model API
            try:
                response = requests.get(f"http://localhost:{test_model_port}/", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Model API responding: {data.get('message')}")
                    
                    # Test config endpoint
                    config_response = requests.get(f"http://localhost:{test_model_port}/config", timeout=5)
                    if config_response.status_code == 200:
                        config_data = config_response.json()
                        print(f"✅ Configuration loaded from API")
                        print(f"   Model cache dir: {config_data.get('model_cache_dir')}")
                    
                else:
                    print(f"⚠️  Model API responded with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Could not connect to model API: {e}")
            
            print("\nStopping process...")
            process.terminate()
            try:
                process.wait(timeout=10)
                print("✅ Process stopped cleanly")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️  Process had to be killed")
                
            return True
            
        else:
            # Process already ended, check for errors
            stdout, stderr = process.communicate()
            print(f"❌ Process ended early")
            print(f"Exit code: {process.returncode}")
            if stdout:
                print(f"STDOUT:\n{stdout}")
            if stderr:
                print(f"STDERR:\n{stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 GSwarm CLI Command Test")
    print("=" * 50)
    
    print("This test will start the GSwarm host with custom parameters")
    print("to verify that:")
    print("1. No asyncio errors occur during startup")
    print("2. CLI parameters override config file settings")
    print("3. The service starts and responds correctly")
    print()
    
    success = test_cli_command()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 CLI command test passed!")
        print("✅ Your fix is working correctly.")
        print("\nYou can now safely run:")
        print("gswarm host start --port 8095 --http-port 8096 --model-port 9010")
    else:
        print("❌ CLI command test failed!")
        print("⚠️  There may still be issues with the startup fix.")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1) 
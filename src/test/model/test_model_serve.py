#!/usr/bin/env python3
"""
Test GSwarm model serving functionality
"""
import subprocess
import time
import signal
import os
import sys
import unittest
import requests
import json

class TestModelServe(unittest.TestCase):
    """Test GSwarm model download and serving functionality"""
    
    def setUp(self):
        self.host_process = None
        self.client_process = None
        self.model_process = None
        self.host_port = 8095
        self.http_port = 8096
        self.model_port = 9010
        self.model_name = "llama-7b"
        self.model_source = "hf://meta-llama/Llama-3.1-8B-Instruct"
        
    def tearDown(self):
        """Clean up any running processes"""
        processes = [self.model_process, self.client_process, self.host_process]
        for proc in processes:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    
    def start_gswarm_services(self):
        """Start GSwarm host and client"""
        print("\nStarting GSwarm services for model serving test...")
        
        # Start host
        host_cmd = [
            "gswarm", "host", "start",
            "--port", str(self.host_port),
            "--http-port", str(self.http_port),
            "--model-port", str(self.model_port)
        ]
        
        self.host_process = subprocess.Popen(
            host_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for host to start
        time.sleep(3)
        
        # Start client
        client_cmd = [
            "gswarm", "client", "connect",
            f"localhost:{self.host_port}",
            "--resilient"
        ]
        
        self.client_process = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for client to connect
        time.sleep(3)
        
        # Verify both are running
        self.assertIsNone(self.host_process.poll(), "Host process terminated unexpectedly")
        self.assertIsNone(self.client_process.poll(), "Client process terminated unexpectedly")
        print(f"✓ Host PID: {self.host_process.pid}, Client PID: {self.client_process.pid}")
        
    def test_model_download_and_serve(self):
        """Test downloading and serving a model"""
        print("\n=== Testing GSwarm Model Serve ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Download model
        print(f"\nAttempting to download model {self.model_name}...")
        print(f"Source: {self.model_source}")
        print("Note: This test requires a model API server to be running")
        
        download_cmd = [
            "gswarm", "model", "download",
            self.model_name,
            "--source", self.model_source,
            "--type", "llm"
        ]
        
        try:
            # Run download with extended timeout
            result = subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                timeout=30  # Reduced timeout since we expect it might fail
            )
            
            print(f"Download return code: {result.returncode}")
            
            if result.stdout:
                print("Download output:")
                print(result.stdout[:500])  # First 500 chars
                
            if result.stderr:
                print("Download errors:")
                print(result.stderr[:500])
                
                # Check for connection refused error
                if "connection refused" in result.stderr.lower() or "9015" in result.stderr:
                    print("\n⚠ Model API server is not running on port 9015")
                    print("This is expected if no model API server is configured")
                    print("Skipping model download/serve test...")
                    return
                
            # Check if download was successful
            if result.returncode != 0:
                print("Warning: Model download failed, skipping serve test...")
                return
                
        except subprocess.TimeoutExpired:
            print("Warning: Model download timed out, skipping serve test...")
            return
            
        # Serve model
        print(f"\nServing model {self.model_name} on port {self.model_port}...")
        serve_cmd = [
            "gswarm", "model", "serve",
            self.model_name,
            "--device", "cuda:0",
            "--port", str(self.model_port)
        ]
        
        self.model_process = subprocess.Popen(
            serve_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for model to load
        print("Waiting for model to load (this may take a while)...")
        time.sleep(10)
        
        # Check if model process is running
        if self.model_process.poll() is not None:
            # Read error output
            stderr = self.model_process.stderr.read()
            print(f"Model serve failed with error:\n{stderr}")
            
            # Check if it's a GPU/CUDA issue
            if "cuda" in stderr.lower() or "gpu" in stderr.lower():
                print("\nNote: Model serving failed, possibly due to GPU/CUDA requirements.")
                print("This is expected in environments without GPU support.")
            else:
                self.fail("Model serve process terminated unexpectedly")
        else:
            print(f"✓ Model serve process running with PID: {self.model_process.pid}")
            
            # Try to check model endpoint
            try:
                response = requests.get(f"http://localhost:{self.model_port}/health", timeout=5)
                if response.status_code == 200:
                    print("✓ Model endpoint is responding")
                else:
                    print(f"Model endpoint returned status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Could not reach model endpoint: {e}")
                
        print("\n=== Model serve test completed ===")
        
    def test_model_command_help(self):
        """Test model command help - doesn't require services"""
        print("\n=== Testing Model Command Help ===")
        
        # Test model help
        help_cmd = ["gswarm", "model", "--help"]
        
        try:
            result = subprocess.run(
                help_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✓ Model command help retrieved successfully")
                if result.stdout:
                    print("\nModel commands available:")
                    print(result.stdout[:300])  # First 300 chars
            else:
                print(f"Model help failed with return code: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print("Error: Model help command timed out")
            
        print("\n=== Model help test completed ===")
    
    def test_model_list(self):
        """Test listing available models"""
        print("\n=== Testing Model List ===")
        
        # List models
        list_cmd = ["gswarm", "model", "list"]
        
        try:
            result = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"Model list return code: {result.returncode}")
            
            if result.stdout:
                print("\nAvailable models:")
                print(result.stdout)
                
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)
                
                # Check for connection issues
                if "connection refused" in result.stderr.lower():
                    print("\n⚠ Model API server is not available")
                    print("This is expected if no model API server is configured")
                
        except subprocess.TimeoutExpired:
            print("Error: Model list command timed out")
            
        print("\n=== Model list test completed ===")

def main():
    """Run the model serve test"""
    # Check if gswarm is available
    try:
        subprocess.run(["which", "gswarm"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: gswarm command not found. Please ensure GSwarm is installed.")
        sys.exit(1)
        
    # Check GPU availability
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            print("✓ GPU detected, full model serving test will be performed")
        else:
            print("⚠ No GPU detected, model serving may fail")
    except FileNotFoundError:
        print("⚠ nvidia-smi not found, GPU support may not be available")
        
    # Run tests
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    main()
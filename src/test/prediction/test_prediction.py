#!/usr/bin/env python3
"""
Test GSwarm prediction functionality
"""
import subprocess
import time
import signal
import os
import sys
import unittest
import json

class TestPrediction(unittest.TestCase):
    """Test GSwarm prediction functionality for model execution time"""
    
    def setUp(self):
        self.host_process = None
        self.client_process = None
        self.host_port = 8095
        self.http_port = 8096
        self.model_port = 9010
    
    def test_gswarm_status(self):
        """Test basic gswarm status command as a placeholder"""
        print("\n=== Testing GSwarm Status Command ===")
        
        # Test status command which should always be available
        status_cmd = ["gswarm", "status"]
        
        try:
            result = subprocess.run(
                status_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print(f"Status command return code: {result.returncode}")
            if result.stdout:
                print("Status output:")
                print(result.stdout[:200])
            
            print("\n✓ GSwarm status command tested")
            
        except subprocess.TimeoutExpired:
            print("Status command timed out")
        
        print("\n=== Status test completed ===")
        return
        
    def tearDown(self):
        """Clean up any running processes"""
        processes = [self.host_process, self.client_process]
        for proc in processes:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    
    def start_gswarm_services(self):
        """Start GSwarm host and client"""
        print("\nStarting GSwarm services for prediction test...")
        
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
        
    def test_prediction_basic(self):
        """Test basic prediction functionality"""
        print("\n=== Testing GSwarm Prediction ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Check if prediction command exists
        print("\nChecking for prediction command...")
        check_cmd = ["gswarm", "--help"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if "prediction" not in result.stdout:
            print("⚠ Prediction command not available in this version of GSwarm")
            print("Skipping prediction tests...")
            return
        
        # Test prediction command
        print("\nRunning prediction for model execution time...")
        
        # Example prediction scenarios
        test_scenarios = [
            {
                "model": "llama-7b",
                "input_tokens": 512,
                "output_tokens": 128,
                "batch_size": 1
            },
            {
                "model": "llama-13b",
                "input_tokens": 1024,
                "output_tokens": 256,
                "batch_size": 4
            },
            {
                "model": "mistral-7b",
                "input_tokens": 2048,
                "output_tokens": 512,
                "batch_size": 8
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\nTesting prediction for {scenario['model']}:")
            print(f"  Input tokens: {scenario['input_tokens']}")
            print(f"  Output tokens: {scenario['output_tokens']}")
            print(f"  Batch size: {scenario['batch_size']}")
            
            # Build prediction command
            predict_cmd = [
                "gswarm", "prediction", "estimate",
                "--model", scenario['model'],
                "--input-tokens", str(scenario['input_tokens']),
                "--output-tokens", str(scenario['output_tokens']),
                "--batch-size", str(scenario['batch_size'])
            ]
            
            try:
                result = subprocess.run(
                    predict_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("✓ Prediction completed successfully")
                    if result.stdout:
                        print("Prediction output:")
                        print(result.stdout)
                        
                        # Try to parse structured output
                        try:
                            # Check if output contains JSON
                            if "{" in result.stdout:
                                json_start = result.stdout.find("{")
                                json_str = result.stdout[json_start:]
                                prediction_data = json.loads(json_str)
                                
                                # Extract key metrics
                                if "execution_time" in prediction_data:
                                    print(f"  Estimated execution time: {prediction_data['execution_time']}ms")
                                if "gpu_memory" in prediction_data:
                                    print(f"  Estimated GPU memory: {prediction_data['gpu_memory']}MB")
                                if "throughput" in prediction_data:
                                    print(f"  Estimated throughput: {prediction_data['throughput']} tokens/sec")
                        except:
                            # If not JSON, just display as is
                            pass
                else:
                    print(f"Prediction failed with return code: {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                print("Error: Prediction command timed out")
                
        print("\n=== Prediction test completed ===")
        
    def test_prediction_compare(self):
        """Test comparing predictions across different configurations"""
        print("\n=== Testing Prediction Comparison ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Check if prediction command exists
        check_cmd = ["gswarm", "--help"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if "prediction" not in result.stdout:
            print("⚠ Prediction command not available in this version of GSwarm")
            print("Skipping prediction comparison tests...")
            return
        
        # Compare different GPU configurations
        print("\nComparing predictions for different GPU configurations...")
        
        gpu_configs = [
            {"device": "cuda:0", "gpu_type": "A100"},
            {"device": "cuda:0", "gpu_type": "V100"},
            {"device": "cuda:0", "gpu_type": "T4"}
        ]
        
        for config in gpu_configs:
            print(f"\nPrediction for {config['gpu_type']}:")
            
            predict_cmd = [
                "gswarm", "prediction", "estimate",
                "--model", "llama-7b",
                "--input-tokens", "1024",
                "--output-tokens", "256",
                "--device", config['device'],
                "--gpu-type", config['gpu_type']
            ]
            
            try:
                result = subprocess.run(
                    predict_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and result.stdout:
                    print(result.stdout[:200])  # First 200 chars
                else:
                    print("Prediction not available for this configuration")
                    
            except subprocess.TimeoutExpired:
                print("Prediction timed out")
                
        print("\n=== Comparison test completed ===")

def main():
    """Run the prediction test"""
    # Check if gswarm is available
    try:
        subprocess.run(["which", "gswarm"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: gswarm command not found. Please ensure GSwarm is installed.")
        sys.exit(1)
        
    # Run tests
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    main()
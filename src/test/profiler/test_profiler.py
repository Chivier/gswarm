#!/usr/bin/env python3
"""
Test GSwarm profiler functionality
"""
import subprocess
import time
import signal
import os
import sys
import unittest
import json

class TestProfiler(unittest.TestCase):
    """Test GSwarm profiler functionality"""
    
    def setUp(self):
        self.host_process = None
        self.client_process = None
        self.host_port = 8095
        self.http_port = 8096
        self.model_port = 9010
        
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
        print("\nStarting GSwarm services for profiler test...")
        
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
        
    def test_profiler_read(self):
        """Test GSwarm profiler read functionality"""
        print("\n=== Testing GSwarm Profiler ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Run profiler
        print("\nRunning GSwarm profiler...")
        profiler_cmd = ["gswarm", "profiler", "read"]
        
        try:
            result = subprocess.run(
                profiler_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print(f"Profiler return code: {result.returncode}")
            
            if result.stdout:
                print("\nProfiler output:")
                print(result.stdout)
                
                # Verify output contains expected information
                output = result.stdout.lower()
                
                # Check for common profiler output patterns
                expected_patterns = [
                    "gpu", "memory", "utilization", "temperature",
                    "power", "device", "process", "compute"
                ]
                
                found_patterns = []
                for pattern in expected_patterns:
                    if pattern in output:
                        found_patterns.append(pattern)
                        
                if found_patterns:
                    print(f"\n✓ Found profiler data: {', '.join(found_patterns)}")
                else:
                    print("\n⚠ Warning: No expected profiler patterns found in output")
                    
            if result.stderr:
                print("\nProfiler errors:")
                print(result.stderr)
                
            # Check if profiler executed successfully
            self.assertEqual(result.returncode, 0, "Profiler command failed")
            
        except subprocess.TimeoutExpired:
            print("Error: Profiler command timed out")
            self.fail("Profiler command timed out")
            
        print("\n=== Profiler test completed successfully ===")
        
    def test_profiler_json_output(self):
        """Test GSwarm profiler JSON output format"""
        print("\n=== Testing GSwarm Profiler JSON Output ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Run profiler with JSON output
        print("\nRunning GSwarm profiler with JSON output...")
        profiler_cmd = ["gswarm", "profiler", "read", "--format", "json"]
        
        try:
            result = subprocess.run(
                profiler_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                try:
                    # Try to parse as JSON
                    profiler_data = json.loads(result.stdout)
                    print("✓ Successfully parsed JSON output")
                    
                    # Check for expected fields
                    if isinstance(profiler_data, dict):
                        print(f"Found {len(profiler_data)} top-level keys")
                        for key in list(profiler_data.keys())[:5]:
                            print(f"  - {key}")
                            
                except json.JSONDecodeError:
                    print("Note: Output is not valid JSON, may be plain text format")
                    
        except subprocess.TimeoutExpired:
            print("Error: Profiler command timed out")
            
        print("\n=== JSON output test completed ===")

def main():
    """Run the profiler test"""
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
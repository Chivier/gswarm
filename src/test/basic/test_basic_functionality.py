#!/usr/bin/env python3
"""
Test basic GSwarm functionality - starting host and client processes
"""
import subprocess
import time
import signal
import os
import sys
import unittest
import psutil

class TestBasicFunctionality(unittest.TestCase):
    """Test basic GSwarm host and client startup/shutdown"""
    
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
                    
    def test_start_host_and_client(self):
        """Test starting GSwarm host and client processes"""
        print("\n=== Testing Basic GSwarm Functionality ===")
        
        # Start host
        print(f"\nStarting GSwarm host on port {self.host_port}...")
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
        
        # Check if host is running
        self.assertIsNone(self.host_process.poll(), "Host process terminated unexpectedly")
        print(f"Host PID: {self.host_process.pid}")
        
        # Read host output
        try:
            host_stdout = self.host_process.stdout.read(100)
            if host_stdout:
                print(f"Host output: {host_stdout}")
        except:
            pass
            
        # Start client
        print(f"\nStarting GSwarm client connecting to localhost:{self.host_port}...")
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
        
        # Check if client is running
        self.assertIsNone(self.client_process.poll(), "Client process terminated unexpectedly")
        print(f"Client PID: {self.client_process.pid}")
        
        # Read client output
        try:
            client_stdout = self.client_process.stdout.read(100)
            if client_stdout:
                print(f"Client output: {client_stdout}")
        except:
            pass
            
        # Verify processes are still running
        print("\nVerifying processes are running...")
        host_running = psutil.pid_exists(self.host_process.pid)
        client_running = psutil.pid_exists(self.client_process.pid)
        
        self.assertTrue(host_running, "Host process is not running")
        self.assertTrue(client_running, "Client process is not running")
        print("✓ Both processes are running successfully")
        
        # Test graceful shutdown
        print("\nTesting graceful shutdown...")
        
        # Kill client first
        print(f"Terminating client (PID: {self.client_process.pid})...")
        self.client_process.terminate()
        self.client_process.wait(timeout=5)
        print("✓ Client terminated successfully")
        
        # Kill host
        print(f"Terminating host (PID: {self.host_process.pid})...")
        self.host_process.terminate()
        self.host_process.wait(timeout=5)
        print("✓ Host terminated successfully")
        
        print("\n=== Basic functionality test completed successfully ===")

def main():
    """Run the basic functionality test"""
    # Check if gswarm is available by checking if we can find it
    try:
        which_result = subprocess.run(["which", "gswarm"], capture_output=True, text=True, check=True)
        print(f"Using gswarm at: {which_result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("Error: gswarm command not found. Please ensure GSwarm is installed.")
        sys.exit(1)
        
    # Run tests
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    main()
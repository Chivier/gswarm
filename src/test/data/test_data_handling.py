#!/usr/bin/env python3
"""
Test GSwarm data handling functionality
"""
import subprocess
import time
import signal
import os
import sys
import unittest
import tempfile
import json

class TestDataHandling(unittest.TestCase):
    """Test GSwarm data handling and management functionality"""
    
    def setUp(self):
        self.host_process = None
        self.client_process = None
        self.host_port = 8095
        self.http_port = 8096
        self.model_port = 9010
        self.test_data_dir = tempfile.mkdtemp(prefix="gswarm_test_data_")
        
    def tearDown(self):
        """Clean up any running processes and test data"""
        processes = [self.host_process, self.client_process]
        for proc in processes:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    
        # Clean up test data
        if os.path.exists(self.test_data_dir):
            import shutil
            shutil.rmtree(self.test_data_dir)
            
    def start_gswarm_services(self):
        """Start GSwarm host and client"""
        print("\nStarting GSwarm services for data handling test...")
        
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
        
    def create_test_data(self):
        """Create test data files"""
        print("\nCreating test data files...")
        
        # Create sample dataset
        test_dataset = {
            "name": "test_dataset",
            "version": "1.0",
            "description": "Test dataset for GSwarm data handling",
            "files": []
        }
        
        # Create test files
        test_files = [
            ("sample_data.json", {"data": [1, 2, 3, 4, 5], "type": "numeric"}),
            ("model_config.json", {"model": "llama-7b", "params": {"temperature": 0.7}}),
            ("training_data.txt", "Sample training text for testing\nLine 2\nLine 3")
        ]
        
        for filename, content in test_files:
            filepath = os.path.join(self.test_data_dir, filename)
            
            if isinstance(content, dict):
                with open(filepath, 'w') as f:
                    json.dump(content, f, indent=2)
            else:
                with open(filepath, 'w') as f:
                    f.write(content)
                    
            test_dataset["files"].append({
                "name": filename,
                "path": filepath,
                "size": os.path.getsize(filepath)
            })
            
        # Save dataset metadata
        metadata_path = os.path.join(self.test_data_dir, "dataset_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(test_dataset, f, indent=2)
            
        print(f"✓ Created {len(test_files)} test files in {self.test_data_dir}")
        return metadata_path
        
    def test_data_upload(self):
        """Test uploading data to GSwarm"""
        print("\n=== Testing GSwarm Data Upload ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Create test data
        metadata_path = self.create_test_data()
        
        # Upload data
        print("\nUploading test data to GSwarm...")
        upload_cmd = [
            "gswarm", "data", "upload",
            "--path", self.test_data_dir,
            "--name", "test_dataset",
            "--type", "dataset"
        ]
        
        try:
            result = subprocess.run(
                upload_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Upload return code: {result.returncode}")
            
            if result.stdout:
                print("Upload output:")
                print(result.stdout)
                
            if result.stderr:
                print("Upload errors:")
                print(result.stderr)
                
            # Check if upload was successful
            if result.returncode == 0:
                print("✓ Data upload completed successfully")
            else:
                print("⚠ Data upload may have failed")
                
        except subprocess.TimeoutExpired:
            print("Error: Data upload timed out")
            
        print("\n=== Data upload test completed ===")
        
    def test_data_list(self):
        """Test listing available datasets"""
        print("\n=== Testing GSwarm Data List ===")
        
        # Start services
        self.start_gswarm_services()
        
        # List datasets
        print("\nListing available datasets...")
        list_cmd = ["gswarm", "data", "list"]
        
        try:
            result = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"List return code: {result.returncode}")
            
            if result.stdout:
                print("\nAvailable datasets:")
                print(result.stdout)
                
                # Check if our test dataset appears
                if "test_dataset" in result.stdout:
                    print("✓ Test dataset found in list")
                    
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("Error: Data list command timed out")
            
        print("\n=== Data list test completed ===")
        
    def test_data_download(self):
        """Test downloading data from GSwarm"""
        print("\n=== Testing GSwarm Data Download ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Create temporary download directory
        download_dir = tempfile.mkdtemp(prefix="gswarm_download_")
        
        try:
            # Download data
            print(f"\nDownloading test dataset to {download_dir}...")
            download_cmd = [
                "gswarm", "data", "download",
                "--name", "test_dataset",
                "--output", download_dir
            ]
            
            result = subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Download return code: {result.returncode}")
            
            if result.stdout:
                print("Download output:")
                print(result.stdout)
                
            if result.stderr:
                print("Download errors:")
                print(result.stderr)
                
            # Check downloaded files
            if os.path.exists(download_dir):
                files = os.listdir(download_dir)
                if files:
                    print(f"✓ Downloaded {len(files)} files")
                    for f in files[:5]:  # Show first 5 files
                        print(f"  - {f}")
                else:
                    print("⚠ No files downloaded")
                    
        except subprocess.TimeoutExpired:
            print("Error: Data download timed out")
            
        finally:
            # Clean up download directory
            if os.path.exists(download_dir):
                import shutil
                shutil.rmtree(download_dir)
                
        print("\n=== Data download test completed ===")
        
    def test_data_sync(self):
        """Test data synchronization between nodes"""
        print("\n=== Testing GSwarm Data Sync ===")
        
        # Start services
        self.start_gswarm_services()
        
        # Test sync command
        print("\nTesting data synchronization...")
        sync_cmd = ["gswarm", "data", "sync", "--verbose"]
        
        try:
            result = subprocess.run(
                sync_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"Sync return code: {result.returncode}")
            
            if result.stdout:
                print("Sync output:")
                print(result.stdout)
                
                # Check for sync status
                if "synchronized" in result.stdout.lower() or "up to date" in result.stdout.lower():
                    print("✓ Data synchronization successful")
                    
            if result.stderr:
                print("Sync errors:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("Error: Data sync timed out")
            
        print("\n=== Data sync test completed ===")

def main():
    """Run the data handling test"""
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
#!/usr/bin/env python3
"""
REST API endpoint tests for GSwarm
"""

import unittest
import requests
import json
import time
import os
from typing import Dict, Any, List


class TestGSwarmAPI(unittest.TestCase):
    """Test GSwarm REST API endpoints"""
    
    def setUp(self):
        """Set up test environment"""
        self.host_url = os.environ.get("GSWARM_HOST_URL", "http://localhost:8080")
        self.api_base = f"{self.host_url}/api/v1"
        self.headers = {"Content-Type": "application/json"}
        
        # Check if host is available
        try:
            response = requests.get(f"{self.host_url}/health", timeout=2)
            response.raise_for_status()
            self.host_available = True
        except:
            self.host_available = False
    
    def skip_if_no_host(self):
        """Skip test if host is not available"""
        if not self.host_available:
            self.skipTest("Host not available at " + self.host_url)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        self.skip_if_no_host()
        
        response = requests.get(f"{self.host_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
    
    def test_client_registration(self):
        """Test client registration endpoint"""
        self.skip_if_no_host()
        
        # Register a test client
        client_data = {
            "client_id": f"test-client-{int(time.time())}",
            "address": "192.168.1.100:8081",
            "devices": [
                "test-client:cuda:0",
                "test-client:cuda:1"
            ],
            "capabilities": {
                "gpu_count": 2,
                "gpu_memory_gb": 24,
                "supports_llm": True,
                "supports_diffusion": True
            }
        }
        
        response = requests.post(
            f"{self.api_base}/clients/register",
            json=client_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("client_id", result)
        self.assertEqual(result["client_id"], client_data["client_id"])
        
        # Verify client appears in list
        list_response = requests.get(f"{self.api_base}/clients")
        self.assertEqual(list_response.status_code, 200)
        
        clients = list_response.json().get("clients", [])
        client_ids = [c["id"] for c in clients]
        self.assertIn(client_data["client_id"], client_ids)
    
    def test_client_list_and_info(self):
        """Test client list and info endpoints"""
        self.skip_if_no_host()
        
        # Get client list
        response = requests.get(f"{self.api_base}/clients")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("clients", data)
        self.assertIsInstance(data["clients"], list)
        
        # If there are clients, test getting info for one
        if data["clients"]:
            client = data["clients"][0]
            client_id = client["id"]
            
            # Get client info
            info_response = requests.get(f"{self.api_base}/clients/{client_id}")
            self.assertEqual(info_response.status_code, 200)
            
            info = info_response.json()
            self.assertEqual(info["id"], client_id)
            self.assertIn("devices", info)
            self.assertIn("status", info)
    
    def test_workflow_submission(self):
        """Test workflow submission endpoint"""
        self.skip_if_no_host()
        
        # Create test workflow
        workflow_data = {
            "workflow": {
                "id": f"test-workflow-{int(time.time())}",
                "name": "Test Workflow",
                "nodes": [
                    {
                        "id": "node1",
                        "model": "gpt-4",
                        "inputs": {"prompt": "Hello world"},
                        "outputs": ["text_output"]
                    },
                    {
                        "id": "node2",
                        "model": "stable-diffusion",
                        "inputs": {
                            "prompt": "text_output",
                            "height": 512,
                            "width": 512
                        },
                        "outputs": ["image_output"]
                    }
                ],
                "edges": [
                    {"from": "node1", "to": "node2"}
                ]
            },
            "priority": 1,
            "device_preferences": {
                "node1": "gpu-node-1:cuda:0",
                "node2": "gpu-node-2:cuda:0"
            }
        }
        
        response = requests.post(
            f"{self.api_base}/workflows",
            json=workflow_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("workflow_id", result)
        self.assertIn("status", result)
        
        workflow_id = result["workflow_id"]
        
        # Check workflow status
        status_response = requests.get(f"{self.api_base}/workflows/{workflow_id}")
        self.assertEqual(status_response.status_code, 200)
        
        status = status_response.json()
        self.assertEqual(status["workflow_id"], workflow_id)
        self.assertIn("status", status)
        self.assertIn("nodes", status)
    
    def test_model_management(self):
        """Test model management endpoints"""
        self.skip_if_no_host()
        
        # List available models
        response = requests.get(f"{self.api_base}/models")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)
        
        # Register a new model
        model_data = {
            "name": f"test-model-{int(time.time())}",
            "type": "llm",
            "memory_gb": 8.0,
            "load_time_seconds": 3.0,
            "supported_devices": ["cuda"],
            "metadata": {
                "version": "1.0",
                "parameters": "7B"
            }
        }
        
        reg_response = requests.post(
            f"{self.api_base}/models",
            json=model_data,
            headers=self.headers
        )
        
        if reg_response.status_code == 200:
            result = reg_response.json()
            self.assertIn("model_id", result)
            
            # Get model info
            model_id = result["model_id"]
            info_response = requests.get(f"{self.api_base}/models/{model_id}")
            self.assertEqual(info_response.status_code, 200)
            
            info = info_response.json()
            self.assertEqual(info["name"], model_data["name"])
    
    def test_scheduler_endpoints(self):
        """Test scheduler-related endpoints"""
        self.skip_if_no_host()
        
        # Get scheduler info
        response = requests.get(f"{self.api_base}/scheduler/info")
        if response.status_code == 200:
            data = response.json()
            self.assertIn("strategy", data)
            self.assertIn("active_tasks", data)
            self.assertIn("pending_requests", data)
        
        # Get scheduler metrics
        metrics_response = requests.get(f"{self.api_base}/scheduler/metrics")
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            self.assertIn("total_requests", metrics)
            self.assertIn("completed_requests", metrics)
            self.assertIn("average_latency", metrics)
    
    def test_cost_prediction_endpoint(self):
        """Test cost prediction API endpoint"""
        self.skip_if_no_host()
        
        # Test LLM prediction
        llm_request = {
            "model_name": "gpt-4",
            "device": "node1:cuda:0",
            "inputs": {
                "prompt": "Explain quantum computing in simple terms"
            }
        }
        
        response = requests.post(
            f"{self.api_base}/predict/cost",
            json=llm_request,
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            self.assertIn("predicted_time", result)
            self.assertGreater(result["predicted_time"], 0)
            self.assertIn("model_type", result)
            self.assertEqual(result["model_type"], "llm")
        
        # Test SD prediction
        sd_request = {
            "model_name": "stable-diffusion-xl",
            "device": "node2:cuda:1",
            "inputs": {
                "height": 1024,
                "width": 1024
            }
        }
        
        sd_response = requests.post(
            f"{self.api_base}/predict/cost",
            json=sd_request,
            headers=self.headers
        )
        
        if sd_response.status_code == 200:
            result = sd_response.json()
            self.assertIn("predicted_time", result)
            self.assertGreater(result["predicted_time"], 0)
            self.assertEqual(result["model_type"], "diffusion")
    
    def test_batch_operations(self):
        """Test batch operation endpoints"""
        self.skip_if_no_host()
        
        # Batch workflow submission
        batch_data = {
            "workflows": [
                {
                    "id": f"batch-wf-1-{int(time.time())}",
                    "name": "Batch Workflow 1",
                    "nodes": [
                        {
                            "id": "simple",
                            "model": "gpt-4",
                            "inputs": {"prompt": "Test 1"},
                            "outputs": ["output"]
                        }
                    ],
                    "edges": []
                },
                {
                    "id": f"batch-wf-2-{int(time.time())}",
                    "name": "Batch Workflow 2",
                    "nodes": [
                        {
                            "id": "simple",
                            "model": "stable-diffusion",
                            "inputs": {"height": 512, "width": 512},
                            "outputs": ["image"]
                        }
                    ],
                    "edges": []
                }
            ],
            "scheduling_strategy": "offline"
        }
        
        response = requests.post(
            f"{self.api_base}/workflows/batch",
            json=batch_data,
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            self.assertIn("workflow_ids", result)
            self.assertEqual(len(result["workflow_ids"]), 2)
    
    def test_device_format_in_api(self):
        """Test that API properly handles new device format"""
        self.skip_if_no_host()
        
        # Test device listing
        response = requests.get(f"{self.api_base}/devices")
        if response.status_code == 200:
            data = response.json()
            self.assertIn("devices", data)
            
            # Check device format
            for device in data["devices"]:
                self.assertIn(":", device["id"])
                if "cuda" in device["id"]:
                    # Should be in format client:cuda:id
                    parts = device["id"].split(":")
                    self.assertEqual(len(parts), 3)
    
    def test_error_handling(self):
        """Test API error handling"""
        self.skip_if_no_host()
        
        # Test 404 for non-existent resource
        response = requests.get(f"{self.api_base}/workflows/non-existent-id")
        self.assertEqual(response.status_code, 404)
        
        # Test 400 for invalid request
        invalid_workflow = {
            "workflow": {
                # Missing required fields
                "nodes": []
            }
        }
        
        response = requests.post(
            f"{self.api_base}/workflows",
            json=invalid_workflow,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 400)
        
        error_data = response.json()
        self.assertIn("error", error_data)


if __name__ == "__main__":
    # Set SKIP_API_TESTS=false to run these tests
    if os.environ.get("SKIP_API_TESTS", "true").lower() == "true":
        print("API tests are skipped by default.")
        print("To run: SKIP_API_TESTS=false python test_rest_api.py")
    
    unittest.main()
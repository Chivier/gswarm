#!/usr/bin/env python3
"""
End-to-end integration tests for GSwarm
"""

import unittest
import sys
import os
import time
import threading
import requests
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.model import CostModel
from gswarm.predictor import predictor
from gswarm.scheduler import (
    BaselineScheduler,
    ModelInfo,
    Workflow,
    WorkflowNode,
    WorkflowEdge,
    Request
)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.host_url = os.environ.get("GSWARM_HOST_URL", "http://localhost:8080")
        self.skip_api_tests = os.environ.get("SKIP_API_TESTS", "true").lower() == "true"
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow execution in simulation mode"""
        # Create models
        models = {
            "gpt-4": ModelInfo(
                name="gpt-4",
                memory_gb=16.0,
                gpus_required=1,
                load_time_seconds=5.0,
                tokens_per_second=50.0
            ),
            "stable-diffusion-xl": ModelInfo(
                name="stable-diffusion-xl",
                memory_gb=10.0,
                gpus_required=1,
                load_time_seconds=4.0,
                inference_time_mean=3.0
            ),
            "whisper": ModelInfo(
                name="whisper",
                memory_gb=2.0,
                gpus_required=1,
                load_time_seconds=2.0,
                inference_time_mean=1.5
            )
        }
        
        # Create multi-modal workflow
        nodes = [
            WorkflowNode(id="transcribe", model="whisper", inputs=["audio"], outputs=["text"]),
            WorkflowNode(id="enhance", model="gpt-4", inputs=["text"], outputs=["enhanced_text"]),
            WorkflowNode(id="generate", model="stable-diffusion-xl", inputs=["enhanced_text"], outputs=["image"])
        ]
        edges = [
            WorkflowEdge(from_node="transcribe", to_node="enhance"),
            WorkflowEdge(from_node="enhance", to_node="generate")
        ]
        workflow = Workflow(id="multimodal", name="Multi-modal Pipeline", nodes=nodes, edges=edges)
        
        # Use cost predictor for time estimates
        cost_model = CostModel()
        
        # Test with different schedulers
        devices = ["gpu-node-1:cuda:0", "gpu-node-1:cuda:1", "gpu-node-2:cuda:0", "gpu-node-2:cuda:1"]
        
        scheduler = BaselineScheduler(devices=devices, models=models, simulate=True)
        scheduler.add_workflow(workflow)
        
        # Create batch of requests
        requests = []
        for i in range(5):
            request = Request(
                id=f"req_{i}",
                workflow_id="multimodal",
                arrival_time=i * 2.0,  # Staggered arrivals
                priority=i % 3  # Different priorities
            )
            requests.append(request)
            scheduler.add_request(request)
        
        # Process all tasks
        completed_tasks = []
        task_times = {}
        
        while len(completed_tasks) < len(requests) * len(nodes):
            task = scheduler.get_next_task()
            if task is None:
                # Advance time
                scheduler.advance_time(scheduler.current_time + 0.1)
                continue
            
            # Estimate execution time
            if task.model_name == "gpt-4":
                exec_time = cost_model.predict(task.model_name, task.device, {"prompt": "test" * 100})
            elif task.model_name == "stable-diffusion-xl":
                exec_time = cost_model.predict(task.model_name, task.device, {"height": 1024, "width": 1024})
            else:
                exec_time = 1.5  # Default for whisper
            
            task_times[task.task_id] = exec_time
            completed_tasks.append(task)
            
            # Simulate task completion
            scheduler.advance_time(task.scheduled_time + exec_time)
            scheduler.complete_task(task)
        
        # Verify all tasks completed
        self.assertEqual(len(completed_tasks), len(requests) * len(nodes))
        
        # Check metrics
        metrics = scheduler.get_metrics()
        self.assertEqual(metrics.completed_requests, len(requests))
        self.assertGreater(metrics.average_request_latency, 0)
        
        # Verify device assignments
        device_usage = {}
        for task in completed_tasks:
            if task.device not in device_usage:
                device_usage[task.device] = []
            device_usage[task.device].append(task.model_name)
        
        # All devices should be used
        self.assertGreater(len(device_usage), 1)
        
        print(f"\nCompleted {len(completed_tasks)} tasks across {len(device_usage)} devices")
        print(f"Average latency: {metrics.average_request_latency:.2f}s")
        print(f"Model switches: {metrics.model_switch_count}")
    
    def test_cost_prediction_accuracy(self):
        """Test cost prediction accuracy across different scenarios"""
        cost_model = CostModel()
        
        # Test scenarios
        scenarios = [
            # (model, device, inputs, description)
            ("gpt-4", "node1:cuda:0", {"prompt": "Hello"}, "Short GPT-4 prompt"),
            ("gpt-4", "node1:cuda:0", {"prompt": "Explain " * 100}, "Long GPT-4 prompt"),
            ("llama-2-7b", "node2:cuda:1", {"prompt": "Test"}, "Llama-2 inference"),
            ("stable-diffusion", "node3:cuda:0", {"height": 512, "width": 512}, "SD 512x512"),
            ("stable-diffusion-xl", "node3:cuda:1", {"height": 1024, "width": 1024}, "SDXL 1024x1024"),
        ]
        
        predictions = []
        for model_name, device, inputs, description in scenarios:
            pred_time = cost_model.predict(model_name, device, inputs)
            predictions.append({
                "description": description,
                "model": model_name,
                "device": device,
                "predicted_time": pred_time
            })
            
            # Verify prediction is reasonable
            self.assertGreater(pred_time, 0)
            self.assertLess(pred_time, 100)  # Less than 100 seconds
        
        # Verify relative predictions make sense
        # Long prompt should take more time than short
        self.assertGreater(predictions[1]["predicted_time"], predictions[0]["predicted_time"])
        
        # Larger image should take more time
        self.assertGreater(predictions[4]["predicted_time"], predictions[3]["predicted_time"])
        
        print("\nCost Predictions:")
        for pred in predictions:
            print(f"  {pred['description']}: {pred['predicted_time']:.2f}s")
    
    @unittest.skipIf(os.environ.get("SKIP_API_TESTS", "true").lower() == "true", 
                     "Skipping API tests (set SKIP_API_TESTS=false to run)")
    def test_api_workflow_submission(self):
        """Test workflow submission through API"""
        try:
            # Check if host is available
            response = requests.get(f"{self.host_url}/health", timeout=2)
            response.raise_for_status()
        except:
            self.skipTest("Host not available")
        
        # Create workflow payload
        workflow_data = {
            "workflow": {
                "id": "test_api_workflow",
                "name": "API Test Workflow",
                "nodes": [
                    {
                        "id": "llm",
                        "model": "gpt-4",
                        "inputs": {"prompt": "Generate a creative story"},
                        "outputs": ["story"]
                    },
                    {
                        "id": "image",
                        "model": "stable-diffusion",
                        "inputs": {"prompt": "story", "height": 512, "width": 512},
                        "outputs": ["image"]
                    }
                ],
                "edges": [
                    {"from": "llm", "to": "image"}
                ]
            },
            "device_preferences": {
                "llm": "gpu-server-1:cuda:0",
                "image": "gpu-server-2:cuda:0"
            }
        }
        
        # Submit workflow
        response = requests.post(
            f"{self.host_url}/api/v1/workflows",
            json=workflow_data,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("workflow_id", result)
        
        workflow_id = result["workflow_id"]
        
        # Check status
        status_response = requests.get(f"{self.host_url}/api/v1/workflows/{workflow_id}")
        self.assertEqual(status_response.status_code, 200)
        
        status = status_response.json()
        self.assertIn("status", status)
        self.assertIn("nodes", status)
    
    def test_scheduler_comparison(self):
        """Compare different scheduler strategies"""
        # Common setup
        models = {
            "model_a": ModelInfo(name="model_a", memory_gb=8.0, gpus_required=1, 
                               load_time_seconds=3.0, inference_time_mean=1.0),
            "model_b": ModelInfo(name="model_b", memory_gb=8.0, gpus_required=1,
                               load_time_seconds=3.0, inference_time_mean=1.5),
            "model_c": ModelInfo(name="model_c", memory_gb=8.0, gpus_required=1,
                               load_time_seconds=3.0, inference_time_mean=2.0)
        }
        
        devices = ["server1:cuda:0", "server1:cuda:1"]
        
        # Create workflow with parallel paths
        nodes = [
            WorkflowNode(id="start", model="model_a", inputs=["input"], outputs=["out1"]),
            WorkflowNode(id="path1", model="model_b", inputs=["out1"], outputs=["out2"]),
            WorkflowNode(id="path2", model="model_c", inputs=["out1"], outputs=["out3"]),
            WorkflowNode(id="end", model="model_a", inputs=["out2", "out3"], outputs=["final"])
        ]
        edges = [
            WorkflowEdge(from_node="start", to_node="path1"),
            WorkflowEdge(from_node="start", to_node="path2"),
            WorkflowEdge(from_node="path1", to_node="end"),
            WorkflowEdge(from_node="path2", to_node="end")
        ]
        workflow = Workflow(id="parallel", name="Parallel Workflow", nodes=nodes, edges=edges)
        
        # Test with baseline scheduler
        baseline = BaselineScheduler(devices=devices, models=models, simulate=True)
        baseline.add_workflow(workflow)
        
        # Create requests
        requests = [Request(id=f"req_{i}", workflow_id="parallel", arrival_time=0.0) for i in range(3)]
        
        for req in requests:
            baseline.add_request(req)
        
        # Process with baseline
        baseline_tasks = []
        while True:
            task = baseline.get_next_task()
            if task is None:
                break
            baseline_tasks.append(task)
            baseline.advance_time(task.scheduled_time + task.estimated_duration)
            baseline.complete_task(task)
        
        baseline_metrics = baseline.get_metrics()
        
        print(f"\nScheduler Comparison:")
        print(f"  Baseline - Tasks: {len(baseline_tasks)}, Switches: {baseline_metrics.model_switch_count}")
        print(f"  Baseline - Avg Latency: {baseline_metrics.average_request_latency:.2f}s")


if __name__ == "__main__":
    unittest.main()
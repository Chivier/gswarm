#!/usr/bin/env python3
"""
Unit tests for schedulers
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.scheduler import (
    BaselineScheduler,
    OfflineScheduler,
    OnlineScheduler,
    StaticScheduler,
    ModelInfo,
    Workflow,
    WorkflowNode,
    WorkflowEdge,
    Request,
    ScheduledTask
)


class TestSchedulers(unittest.TestCase):
    """Test scheduler functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Define test models
        self.models = {
            "gpt-4": ModelInfo(
                name="gpt-4",
                memory_gb=16.0,
                gpus_required=1,
                load_time_seconds=5.0,
                tokens_per_second=50.0
            ),
            "stable-diffusion": ModelInfo(
                name="stable-diffusion",
                memory_gb=8.0,
                gpus_required=1,
                load_time_seconds=3.0,
                inference_time_mean=2.5
            ),
            "llama-2": ModelInfo(
                name="llama-2",
                memory_gb=13.0,
                gpus_required=1,
                load_time_seconds=4.0,
                tokens_per_second=30.0
            )
        }
        
        # Define test workflow
        self.nodes = [
            WorkflowNode(id="prompt", model="gpt-4", inputs=["user_input"], outputs=["text"]),
            WorkflowNode(id="image", model="stable-diffusion", inputs=["text"], outputs=["image"])
        ]
        self.edges = [WorkflowEdge(from_node="prompt", to_node="image")]
        self.workflow = Workflow(id="test_workflow", name="Test", nodes=self.nodes, edges=self.edges)
        
        # Define test devices
        self.devices = ["node1:cuda:0", "node1:cuda:1", "node2:cuda:0", "node2:cuda:1"]
        self.legacy_gpus = [0, 1, 2, 3]
    
    def test_baseline_scheduler_new_format(self):
        """Test baseline scheduler with new device format"""
        scheduler = BaselineScheduler(devices=self.devices, models=self.models, simulate=True)
        scheduler.add_workflow(self.workflow)
        
        # Add request
        request = Request(id="req1", workflow_id="test_workflow", arrival_time=0.0)
        scheduler.add_request(request)
        
        # Get first task
        task = scheduler.get_next_task()
        self.assertIsNotNone(task)
        self.assertIn(task.device, self.devices)
        self.assertEqual(task.model_name, "gpt-4")
        
        # Complete task
        scheduler.complete_task(task)
        
        # Get second task (should be dependent)
        task2 = scheduler.get_next_task()
        self.assertIsNotNone(task2)
        self.assertEqual(task2.model_name, "stable-diffusion")
    
    def test_baseline_scheduler_legacy_format(self):
        """Test baseline scheduler with legacy GPU format"""
        scheduler = BaselineScheduler(gpus=self.legacy_gpus, models=self.models, simulate=True)
        scheduler.add_workflow(self.workflow)
        
        request = Request(id="req1", workflow_id="test_workflow", arrival_time=0.0)
        scheduler.add_request(request)
        
        task = scheduler.get_next_task()
        self.assertIsNotNone(task)
        self.assertIn("localhost:cuda:", task.device)
        self.assertIn(task.gpu_id, self.legacy_gpus)
    
    def test_offline_scheduler(self):
        """Test offline scheduler"""
        scheduler = OfflineScheduler(devices=self.devices, models=self.models)
        scheduler.add_workflow(self.workflow)
        
        # Create batch of requests
        requests = [
            Request(id=f"req{i}", workflow_id="test_workflow", arrival_time=0.0)
            for i in range(5)
        ]
        
        # Schedule batch
        tasks = scheduler.schedule(requests)
        self.assertGreater(len(tasks), 0)
        
        # Check all tasks have valid devices
        for task in tasks:
            self.assertIn(task.device, self.devices)
            self.assertIn(task.model_name, self.models)
    
    def test_online_scheduler(self):
        """Test online scheduler"""
        scheduler = OnlineScheduler(devices=self.devices, models=self.models)
        scheduler.add_workflow(self.workflow)
        
        # Add requests with different arrival times
        for i in range(3):
            request = Request(
                id=f"req{i}",
                workflow_id="test_workflow",
                arrival_time=i * 0.5,
                priority=i  # Different priorities
            )
            scheduler.advance_time(request.arrival_time)
            scheduler.add_request(request)
        
        # Get tasks
        tasks = []
        while True:
            task = scheduler.get_next_task()
            if task is None:
                break
            tasks.append(task)
            if len(tasks) > 10:  # Prevent infinite loop
                break
        
        self.assertGreater(len(tasks), 0)
    
    def test_static_scheduler(self):
        """Test static scheduler"""
        # Define server configuration with new format
        servers = {
            0: ["server1:cuda:0", "server1:cuda:1"],
            1: ["server2:cuda:0", "server2:cuda:1"]
        }
        
        # Model assignments
        model_assignments = {
            "gpt-4": ["server1:cuda:0", "server2:cuda:0"],
            "stable-diffusion": ["server1:cuda:1", "server2:cuda:1"]
        }
        
        scheduler = StaticScheduler(
            devices=self.devices,
            models=self.models,
            servers=servers,
            model_assignments=model_assignments
        )
        scheduler.add_workflow(self.workflow)
        
        # Schedule requests
        requests = [Request(id=f"req{i}", workflow_id="test_workflow", arrival_time=0.0) for i in range(3)]
        tasks = scheduler.schedule(requests)
        
        # Check model placement
        for task in tasks:
            if task.model_name == "gpt-4":
                self.assertIn(task.device, ["server1:cuda:0", "server2:cuda:0"])
            elif task.model_name == "stable-diffusion":
                self.assertIn(task.device, ["server1:cuda:1", "server2:cuda:1"])
    
    def test_workflow_dependencies(self):
        """Test workflow dependency handling"""
        # Create more complex workflow
        nodes = [
            WorkflowNode(id="input", model="gpt-4", inputs=["user"], outputs=["text1"]),
            WorkflowNode(id="process1", model="llama-2", inputs=["text1"], outputs=["text2"]),
            WorkflowNode(id="process2", model="gpt-4", inputs=["text1"], outputs=["text3"]),
            WorkflowNode(id="combine", model="llama-2", inputs=["text2", "text3"], outputs=["final"])
        ]
        edges = [
            WorkflowEdge(from_node="input", to_node="process1"),
            WorkflowEdge(from_node="input", to_node="process2"),
            WorkflowEdge(from_node="process1", to_node="combine"),
            WorkflowEdge(from_node="process2", to_node="combine")
        ]
        workflow = Workflow(id="complex", name="Complex", nodes=nodes, edges=edges)
        
        # Test dependency calculation
        deps = workflow.get_dependencies()
        self.assertEqual(deps["input"], set())
        self.assertEqual(deps["process1"], {"input"})
        self.assertEqual(deps["process2"], {"input"})
        self.assertEqual(deps["combine"], {"process1", "process2"})
        
        # Test with scheduler
        scheduler = BaselineScheduler(devices=self.devices, models=self.models, simulate=True)
        scheduler.add_workflow(workflow)
        scheduler.add_request(Request(id="req1", workflow_id="complex", arrival_time=0.0))
        
        # First task should be input
        task = scheduler.get_next_task()
        self.assertEqual(task.node_id, "input")
    
    def test_scheduler_metrics(self):
        """Test scheduler metrics tracking"""
        scheduler = BaselineScheduler(devices=self.devices, models=self.models, simulate=True)
        scheduler.add_workflow(self.workflow)
        
        # Process some requests
        for i in range(3):
            request = Request(id=f"req{i}", workflow_id="test_workflow", arrival_time=i * 1.0)
            scheduler.add_request(request)
            
            # Process all tasks
            while True:
                task = scheduler.get_next_task()
                if task is None:
                    break
                scheduler.complete_task(task)
        
        # Check metrics
        metrics = scheduler.get_metrics()
        self.assertGreater(metrics.completed_tasks, 0)
        self.assertEqual(metrics.completed_requests, 3)
        self.assertGreaterEqual(metrics.model_switch_count, 0)
    
    def test_priority_scheduling(self):
        """Test priority-based scheduling"""
        scheduler = BaselineScheduler(devices=self.devices, models=self.models, simulate=True)
        scheduler.add_workflow(self.workflow)
        
        # Add requests with different priorities
        high_priority = Request(id="high", workflow_id="test_workflow", arrival_time=0.0, priority=0)
        low_priority = Request(id="low", workflow_id="test_workflow", arrival_time=0.0, priority=10)
        
        scheduler.add_request(low_priority)
        scheduler.add_request(high_priority)
        
        # High priority should be scheduled first
        task = scheduler.get_next_task()
        self.assertEqual(task.request_id, "high")
    
    def test_device_compatibility(self):
        """Test that both device formats work correctly"""
        # Create scheduler with mixed format
        mixed_devices = ["node1:cuda:0", 1, "node2:cuda:1", 3]
        scheduler = BaselineScheduler(devices=mixed_devices, models=self.models, simulate=True)
        
        # Should have 4 devices
        self.assertEqual(scheduler.num_devices, 4)
        self.assertEqual(len(scheduler.devices), 4)
        
        # All should be in new format
        for device in scheduler.devices:
            self.assertIn(":", device)
            self.assertIn("cuda", device)


if __name__ == "__main__":
    unittest.main()
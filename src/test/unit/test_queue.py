"""
Unit tests for the Task Queue Manager
"""

import unittest
from datetime import datetime, timedelta
from gswarm.queue.manager import TaskQueueManager, Task


class TestTaskQueueManager(unittest.TestCase):
    
    def setUp(self):
        self.manager = TaskQueueManager()
        self.sample_task = Task(
            uuid=None,
            datetime=datetime.now(),
            model_name="test_model",
            input={"test": "data"},
            tag="test_tag",
            dependencies=set(),
            timeout=30.0,
            others={"key": "value"}
        )
    
    def test_add_task(self):
        """Test adding a task to a queue."""
        self.manager.add_task("cuda:0", self.sample_task)
        self.assertEqual(self.manager.get_queue_size("cuda:0"), 1)
    
    def test_remove_task(self):
        """Test removing a task from a queue."""
        self.manager.add_task("cuda:0", self.sample_task)
        removed_task = self.manager.remove_task("cuda:0", self.sample_task.uuid)
        self.assertEqual(removed_task.uuid, self.sample_task.uuid)
        self.assertEqual(self.manager.get_queue_size("cuda:0"), 0)
    
    def test_get_task(self):
        """Test retrieving a specific task."""
        self.manager.add_task("cuda:0", self.sample_task)
        retrieved_task = self.manager.get_task("cuda:0", self.sample_task.uuid)
        self.assertEqual(retrieved_task.uuid, self.sample_task.uuid)
    
    def test_get_all_tasks(self):
        """Test retrieving all tasks from a queue."""
        task1 = Task(
            uuid=None,
            datetime=datetime.now(),
            model_name="model1",
            input={},
            tag="",
            dependencies=set(),
            timeout=10.0
        )
        
        task2 = Task(
            uuid=None,
            datetime=datetime.now(),
            model_name="model2",
            input={},
            tag="",
            dependencies=set(),
            timeout=20.0
        )
        
        self.manager.add_task("cuda:0", task1)
        self.manager.add_task("cuda:0", task2)
        
        tasks = self.manager.get_all_tasks("cuda:0")
        self.assertEqual(len(tasks), 2)
        self.assertIn(task1.uuid, [t.uuid for t in tasks])
        self.assertIn(task2.uuid, [t.uuid for t in tasks])
    
    def test_sort_tasks(self):
        """Test sorting tasks in a queue."""
        # Create tasks with different timeouts
        task1 = Task(
            uuid=None,
            datetime=datetime.now(),
            model_name="model1",
            input={},
            tag="",
            dependencies=set(),
            timeout=10.0  # Shortest timeout
        )
        
        task2 = Task(
            uuid=None,
            datetime=datetime.now(),
            model_name="model2",
            input={},
            tag="",
            dependencies=set(),
            timeout=30.0  # Longest timeout
        )
        
        self.manager.add_task("cuda:0", task2)  # Add in reverse order
        self.manager.add_task("cuda:0", task1)
        
        # Define a simple priority function based on timeout
        def timeout_priority(tasks, device_id):
            return tasks[0].timeout  # Lower timeout = higher priority
        
        # Sort the queue
        self.manager.sort_queue("cuda:0", timeout_priority)
        
        # Get the next task - should be the one with shortest timeout
        next_task = self.manager.get_next_task("cuda:0")
        self.assertEqual(next_task.uuid, task1.uuid)
    
    def test_get_next_task_empty_queue(self):
        """Test getting next task from an empty queue."""
        next_task = self.manager.get_next_task("cuda:0")
        self.assertIsNone(next_task)
    
    def test_get_all_device_ids(self):
        """Test getting all device IDs with queues."""
        self.manager.add_task("cuda:0", self.sample_task)
        self.manager.add_task("cuda:1", self.sample_task)
        
        devices = self.manager.get_all_device_ids()
        self.assertIn("cuda:0", devices)
        self.assertIn("cuda:1", devices)
        self.assertEqual(len(devices), 2)


if __name__ == '__main__':
    unittest.main()
"""
Task Queue Manager for GSwarm Schedulers
Manages task queues for each GPU device with support for adding, removing, and sorting tasks.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Callable, Optional, Any
import uuid


@dataclass
class Task:
    """Represents a task in the queue."""
    uuid: str
    datetime: datetime
    model_name: str
    input: Any
    tag: str
    dependencies: Set[str]
    timeout: float
    others: Dict[str, Any] = field(default_factory=dict)
    
    # For priority queue sorting
    priority: float = 0.0
    
    def __post_init__(self):
        if self.uuid is None:
            self.uuid = str(uuid.uuid4())


class TaskQueue:
    """Queue for tasks assigned to a specific GPU."""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.tasks: List[Task] = []
        self.task_dict: Dict[str, Task] = {}  # For O(1) lookup by UUID
        self.sorted_tasks: List[Task] = []  # Cache for sorted tasks
        self._is_sorted = False
    
    def add_task(self, task: Task) -> None:
        """Add a task to the queue."""
        self.tasks.append(task)
        self.task_dict[task.uuid] = task
        self._is_sorted = False
    
    def remove_task(self, task_uuid: str) -> Optional[Task]:
        """Remove a task from the queue by UUID."""
        if task_uuid not in self.task_dict:
            return None
            
        task = self.task_dict[task_uuid]
        self.tasks.remove(task)
        del self.task_dict[task_uuid]
        self._is_sorted = False
        return task
    
    def get_task(self, task_uuid: str) -> Optional[Task]:
        """Get a task by UUID."""
        return self.task_dict.get(task_uuid)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in the queue."""
        return list(self.tasks)
    
    def sort_tasks(self, value_function: Callable[[List[Task], str, Any], float], *args, **kwargs) -> None:
        """
        Sort tasks based on a value function.
        
        Args:
            value_function: A function that takes (tasks, device_id, *args, **kwargs) and returns a float score
            *args, **kwargs: Additional arguments to pass to the value function
        """
        if not self.tasks:
            return
            
        # Calculate priority for each task
        for task in self.tasks:
            task.priority = value_function([task], self.device_id, *args, **kwargs)
        
        # Sort tasks by priority (lower value = higher priority)
        self.tasks.sort(key=lambda t: t.priority)
        self._is_sorted = True
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task (lowest priority value)."""
        if not self.tasks:
            return None
            
        if not self._is_sorted:
            # Sort by UUID as default if not explicitly sorted
            self.tasks.sort(key=lambda t: t.uuid)
            
        return self.tasks[0] if self.tasks else None
    
    def size(self) -> int:
        """Get the number of tasks in the queue."""
        return len(self.tasks)


class TaskQueueManager:
    """Manages task queues for multiple GPU devices."""
    
    def __init__(self):
        self.queues: Dict[str, TaskQueue] = {}
    
    def get_queue(self, device_id: str) -> TaskQueue:
        """Get or create a queue for a specific device."""
        if device_id not in self.queues:
            self.queues[device_id] = TaskQueue(device_id)
        return self.queues[device_id]
    
    def add_task(self, device_id: str, task: Task) -> None:
        """Add a task to a specific device queue."""
        queue = self.get_queue(device_id)
        queue.add_task(task)
    
    def remove_task(self, device_id: str, task_uuid: str) -> Optional[Task]:
        """Remove a task from a specific device queue."""
        queue = self.get_queue(device_id)
        return queue.remove_task(task_uuid)
    
    def get_task(self, device_id: str, task_uuid: str) -> Optional[Task]:
        """Get a task from a specific device queue."""
        queue = self.get_queue(device_id)
        return queue.get_task(task_uuid)
    
    def get_all_tasks(self, device_id: str) -> List[Task]:
        """Get all tasks from a specific device queue."""
        queue = self.get_queue(device_id)
        return queue.get_all_tasks()
    
    def sort_queue(self, device_id: str, value_function: Callable, *args, **kwargs) -> None:
        """
        Sort tasks in a specific device queue.
        
        Args:
            device_id: The device identifier
            value_function: A function that evaluates task priority
            *args, **kwargs: Additional arguments for the value function
        """
        queue = self.get_queue(device_id)
        queue.sort_tasks(value_function, *args, **kwargs)
    
    def get_next_task(self, device_id: str) -> Optional[Task]:
        """Get the next task from a specific device queue."""
        queue = self.get_queue(device_id)
        return queue.get_next_task()
    
    def get_queue_size(self, device_id: str) -> int:
        """Get the number of tasks in a specific device queue."""
        queue = self.get_queue(device_id)
        return queue.size()
    
    def get_all_device_ids(self) -> List[str]:
        """Get all device IDs that have queues."""
        return list(self.queues.keys())
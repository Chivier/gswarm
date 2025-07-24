"""
Example usage of the Task Queue Manager with schedulers
"""

from datetime import datetime
from typing import List
from gswarm.queue.manager import TaskQueueManager, Task


def example_priority_function(tasks: List[Task], device_id: str, **kwargs) -> float:
    """
    Example value function that prioritizes tasks based on:
    1. Task timeout (sooner timeouts get higher priority)
    2. Task age (older tasks get higher priority)
    """
    if not tasks:
        return 0.0
        
    task = tasks[0]  # We're evaluating one task at a time
    
    # Calculate priority based on timeout (sooner = higher priority)
    timeout_priority = task.timeout
    
    # Calculate priority based on age (older = higher priority)
    age = (datetime.now() - task.datetime).total_seconds()
    age_priority = -age  # Negative because we want older tasks to have lower values
    
    # Combine with weights
    return timeout_priority * 0.7 + age_priority * 0.3


def demo():
    # Create a queue manager
    manager = TaskQueueManager()
    
    # Create some sample tasks
    task1 = Task(
        uuid=None,
        datetime=datetime.now(),
        model_name="model_a",
        input={"text": "Hello world"},
        tag="nlp_task",
        dependencies=set(),
        timeout=10.0,
        others={"user_id": "123"}
    )
    
    task2 = Task(
        uuid=None,
        datetime=datetime.now(),
        model_name="model_b",
        input={"image": "image_data"},
        tag="vision_task",
        dependencies={"task1"},
        timeout=5.0,
        others={"user_id": "456"}
    )
    
    task3 = Task(
        uuid=None,
        datetime=datetime.now(),
        model_name="model_c",
        input={"audio": "audio_data"},
        tag="speech_task",
        dependencies=set(),
        timeout=15.0,
        others={"user_id": "789"}
    )
    
    # Add tasks to GPU queues
    manager.add_task("cuda:0", task1)
    manager.add_task("cuda:0", task2)
    manager.add_task("cuda:1", task3)
    
    # Sort the queues using our priority function
    manager.sort_queue("cuda:0", example_priority_function)
    manager.sort_queue("cuda:1", example_priority_function)
    
    # Get next task from each queue
    next_task_gpu0 = manager.get_next_task("cuda:0")
    next_task_gpu1 = manager.get_next_task("cuda:1")
    
    print(f"Next task for GPU 0: {next_task_gpu0.uuid if next_task_gpu0 else 'None'}")
    print(f"Next task for GPU 1: {next_task_gpu1.uuid if next_task_gpu1 else 'None'}")
    
    # List all tasks
    print("\nAll tasks in GPU 0 queue:")
    for task in manager.get_all_tasks("cuda:0"):
        print(f"  {task.uuid}: {task.model_name}")
        
    print("\nAll tasks in GPU 1 queue:")
    for task in manager.get_all_tasks("cuda:1"):
        print(f"  {task.uuid}: {task.model_name}")


if __name__ == "__main__":
    demo()
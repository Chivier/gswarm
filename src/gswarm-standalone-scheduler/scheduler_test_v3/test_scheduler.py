#!/usr/bin/env python3
"""
Test script demonstrating the complete OCR model scheduling system
"""

import time
import random
import threading
from datetime import datetime

from task import Task, TaskStatus
from scheduler import HybridScheduler
from scheduler_types import SchedulerConfig
from ocr_model_predictor import OCRModelPredictor
from ocr_model import ModelInstance


def simulate_model_execution(model_instance: ModelInstance, task: Task, scheduler: HybridScheduler):
    """Simulate model execution for a task"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting task {task.task_id[:8]}...")
    
    # Mark task as running
    task.mark_running()
    
    # Simulate execution with some variance
    base_duration = task.estimated_duration
    actual_duration = base_duration * random.uniform(0.8, 1.2)
    
    # Simulate possible failure (5% chance)
    if random.random() < 0.05:
        time.sleep(actual_duration * 0.3)  # Fail partway through
        error = "Simulated model error"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Task {task.task_id[:8]} failed: {error}")
        scheduler.handle_task_failure(task.task_id, error)
        return
    
    # Simulate execution
    time.sleep(actual_duration)
    
    # Generate result
    result = {
        'predictions': [f'result_{i}' for i in range(random.randint(1, 5))],
        'confidence': random.uniform(0.85, 0.99),
        'processing_time': actual_duration
    }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Task {task.task_id[:8]} completed in {actual_duration:.2f}s")
    scheduler.handle_task_completion(task.task_id, result)


def model_worker(scheduler: HybridScheduler, model_name: str, device_id: str):
    """Worker thread that processes tasks for a specific model"""
    print(f"Worker started for {model_name} on {device_id}")
    
    # Create model instance
    model = ModelInstance(model_name, device_id)
    
    while True:
        # Get available queues for this model
        queue_ids = scheduler.model_queues.get(model_name, [])
        
        task_found = False
        for queue_id in queue_ids:
            queue = scheduler.queues.get(queue_id)
            if queue:
                # Get next task
                task = queue.get_next_task()
                if task:
                    task_found = True
                    simulate_model_execution(model, task, scheduler)
                    break
        
        if not task_found:
            time.sleep(0.5)  # Wait before checking again


def main():
    print("=== OCR Model Scheduler Test ===\n")
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = OCRModelPredictor(
        retrain_threshold=20,
        max_logs=1000
    )
    
    # Configure scheduler
    config = SchedulerConfig(
        max_queues_per_model=3,
        task_timeout_check_interval=5.0,
        queue_rebalance_interval=15.0,
        enable_preemption=True,
        max_retry_count=2
    )
    
    # Initialize scheduler
    print("Initializing scheduler...")
    scheduler = HybridScheduler(predictor, config)
    
    # Add some initial training data to predictor
    print("Adding sample training data...")
    sample_models = ['CRNN', 'CRAFT', 'DBNet']
    for _ in range(10):
        model_name = random.choice(sample_models)
        input_data = {
            'model_name': model_name,
            'params': {
                'batch_size': random.choice([16, 32, 64]),
                'image_width': random.choice([640, 1280]),
                'image_height': random.choice([480, 720])
            },
            'device': random.choice(['cuda:0', 'cuda:1', 'cpu'])
        }
        duration = random.uniform(10, 60)
        predictor.add_record(input_data, duration)
    
    # Create queues for different models
    print("\nCreating queues...")
    for model_name in sample_models:
        queue_id = scheduler.add_queue(model_name)
        print(f"Created queue {queue_id}")
    
    # Start worker threads
    print("\nStarting worker threads...")
    workers = []
    for i, model_name in enumerate(sample_models):
        device = f"cuda:{i % 2}"
        worker = threading.Thread(
            target=model_worker,
            args=(scheduler, model_name, device),
            daemon=True
        )
        worker.start()
        workers.append(worker)
    
    # Submit various tasks
    print("\nSubmitting tasks...")
    tasks = []
    
    # Online tasks (with timeout)
    for i in range(5):
        model_name = random.choice(sample_models)
        task = Task(
            model_name=model_name,
            input_data={
                'batch_size': 32,
                'image_path': f'/data/image_{i}.jpg',
                'device': 'cuda:0'
            },
            timeout=random.uniform(30, 120),  # 30-120 second timeout
            priority=random.randint(1, 5)
        )
        queue_id = scheduler.submit_task(task)
        tasks.append(task)
        print(f"Submitted online task {task.task_id[:8]} to queue {queue_id}")
    
    # Offline tasks (no timeout)
    for i in range(10):
        model_name = random.choice(sample_models)
        task = Task(
            model_name=model_name,
            input_data={
                'batch_size': 64,
                'dataset_path': f'/data/dataset_{i}',
                'device': 'cuda:1'
            },
            timeout=None,  # Offline task
            priority=random.randint(0, 3)
        )
        queue_id = scheduler.submit_task(task)
        tasks.append(task)
        print(f"Submitted offline task {task.task_id[:8]} to queue {queue_id}")
    
    # Task with dependencies
    print("\nSubmitting tasks with dependencies...")
    parent_task = Task(
        model_name='CRAFT',
        input_data={'stage': 'detection'},
        timeout=60
    )
    parent_queue = scheduler.submit_task(parent_task)
    print(f"Submitted parent task {parent_task.task_id[:8]}")
    
    child_task = Task(
        model_name='CRNN',
        input_data={'stage': 'recognition'},
        dependency_task_ids=[parent_task.task_id],
        timeout=60
    )
    child_queue = scheduler.submit_task(child_task)
    print(f"Submitted child task {child_task.task_id[:8]} (depends on {parent_task.task_id[:8]})")
    
    # Monitor progress
    print("\n=== Monitoring Progress ===")
    start_time = time.time()
    
    try:
        while time.time() - start_time < 180:  # Run for 3 minutes
            time.sleep(10)
            
            # Get metrics
            metrics = scheduler.get_scheduler_metrics()
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scheduler Status:")
            print(f"  Total submitted: {metrics['total_tasks']['submitted']}")
            print(f"  Completed: {metrics['total_tasks']['completed']}")
            print(f"  Failed: {metrics['total_tasks']['failed']}")
            print(f"  Timeout: {metrics['total_tasks']['timeout']}")
            print(f"  Active queues: {metrics['queues']['active']}")
            print(f"  Success rate: {metrics['performance']['success_rate']:.1f}%")
            print(f"  Avg wait time (online): {metrics['performance']['avg_wait_time_online']:.1f}s")
            print(f"  Avg queue utilization: {metrics['performance']['avg_queue_utilization']:.1f}%")
            
            # Show queue status
            print("\n  Queue Status:")
            for queue in scheduler.queues.values():
                status = queue.get_queue_status()
                print(f"    {queue.queue_id}: {status['online_tasks']} online, "
                      f"{status['offline_tasks']} offline, "
                      f"running: {status['running_task'] or 'None'}")
            
            # Check if all tasks completed
            all_completed = all(
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]
                for task in tasks + [parent_task, child_task]
            )
            
            if all_completed:
                print("\nAll tasks completed!")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Final summary
    print("\n=== Final Summary ===")
    final_metrics = scheduler.get_scheduler_metrics()
    
    print(f"Total tasks processed: {final_metrics['total_tasks']['completed']}")
    print(f"Total failures: {final_metrics['total_tasks']['failed']}")
    print(f"Total timeouts: {final_metrics['total_tasks']['timeout']}")
    print(f"Success rate: {final_metrics['performance']['success_rate']:.1f}%")
    
    # Show individual task results
    print("\nTask Results:")
    for task in tasks[:5]:  # Show first 5 tasks
        status = "✓" if task.status == TaskStatus.COMPLETED else "✗"
        wait_time = task.get_wait_time()
        duration = task.get_actual_duration()
        
        print(f"  {status} Task {task.task_id[:8]}: "
              f"status={task.status.value}, "
              f"wait={wait_time:.1f}s, "
              f"duration={duration:.1f}s" if duration else "pending")
    
    # Shutdown
    print("\nShutting down scheduler...")
    scheduler.shutdown()
    print("Test completed!")


if __name__ == "__main__":
    main()
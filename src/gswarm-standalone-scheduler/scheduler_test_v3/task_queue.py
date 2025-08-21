from enum import Enum
from typing import Dict, List, Optional, Tuple
import heapq
import time
from collections import deque

from .task import Task, TaskStatus
from .ocr_model_predictor import OCRModelPredictor


class SchedulingMethod(Enum):
    """Task scheduling methods"""
    FIFO = "fifo"
    DEADLINE_AWARE = "deadline_aware"


class TaskQueue:
    """
    Task queue implementation with hybrid scheduling support
    
    Supports both FIFO and deadline-aware scheduling for online/offline tasks
    """
    
    def __init__(self, queue_id: str, model_name: str, predictor: OCRModelPredictor):
        """
        Initialize task queue
        
        Args:
            queue_id: Unique queue identifier
            model_name: OCR model name this queue serves
            predictor: Model predictor for time estimation
        """
        self.queue_id = queue_id
        self.model_name = model_name
        self.predictor = predictor
        
        # Separate queues for online and offline tasks
        self.online_tasks: List[Task] = []  # Priority queue for online tasks
        self.offline_tasks: deque[Task] = deque()  # FIFO queue for offline tasks
        
        # Task lookup for quick access
        self.task_map: Dict[str, Task] = {}
        
        # Queue statistics
        self.total_processed = 0
        self.total_failed = 0
        self.total_timeout = 0
        
        # Currently running task
        self.running_task: Optional[Task] = None
        
        # Scheduling method
        self.scheduling_method = SchedulingMethod.DEADLINE_AWARE
    
    def add_task(self, task: Task, scheduling_method: SchedulingMethod = SchedulingMethod.DEADLINE_AWARE) -> None:
        """
        Add task to queue with specified scheduling method
        
        Args:
            task: Task to add
            scheduling_method: Scheduling method to use
        """
        if task.task_id in self.task_map:
            raise ValueError(f"Task {task.task_id} already exists in queue")
        
        # Set scheduling method
        self.scheduling_method = scheduling_method
        
        # Estimate task duration
        task.estimated_duration = self.predictor.predict_task_duration(
            task.model_name,
            task.input_data
        )
        
        # Update task status
        task.status = TaskStatus.QUEUED
        task.assigned_queue_id = self.queue_id
        
        # Add to appropriate queue
        if task.is_online_task():
            self._add_online_task(task)
        else:
            self.offline_tasks.append(task)
        
        # Add to task map
        self.task_map[task.task_id] = task
        
        # Reorder if using deadline-aware scheduling
        if self.scheduling_method == SchedulingMethod.DEADLINE_AWARE:
            self.reorder_tasks()
    
    def _add_online_task(self, task: Task) -> None:
        """Add online task to priority queue"""
        if self.scheduling_method == SchedulingMethod.FIFO:
            # For FIFO, use negative start time as priority
            priority = -task.start_time
        else:
            # For deadline-aware, calculate urgency score
            priority = self._calculate_task_urgency(task)
        
        # Add to heap (min-heap, so lower priority value = higher priority)
        heapq.heappush(self.online_tasks, (priority, task.task_id, task))
    
    def _calculate_task_urgency(self, task: Task) -> float:
        """
        Calculate urgency score for deadline-aware scheduling
        Lower score = more urgent
        """
        if task.timeout is None:
            return float('inf')  # Offline tasks have lowest urgency
        
        # Calculate slack time
        remaining_time = task.get_remaining_time()
        if remaining_time is None or remaining_time <= 0:
            return -float('inf')  # Already timed out
        
        # Estimate wait time based on queue position
        wait_time = self._estimate_position_wait_time(task)
        
        # Slack = remaining_time - estimated_duration - wait_time
        slack = remaining_time - task.estimated_duration - wait_time
        
        # Combine slack with priority (lower slack = more urgent)
        # Negative priority means higher priority
        urgency = slack - (task.priority * 10)
        
        return urgency
    
    def _estimate_position_wait_time(self, new_task: Task) -> float:
        """Estimate wait time for a task based on queue position"""
        wait_time = 0.0
        
        # Account for running task
        if self.running_task:
            elapsed = time.time() - self.running_task.actual_start_time
            remaining = max(0, self.running_task.estimated_duration - elapsed)
            wait_time += remaining
        
        # Don't calculate full queue wait for urgency calculation
        # Just use a rough estimate based on queue size
        online_count = len(self.online_tasks)
        avg_duration = 30.0  # Default average duration
        
        # Get better estimate from predictor stats
        stats = self.predictor.get_model_performance_stats(self.model_name)
        if stats['total_runs'] > 0:
            avg_duration = stats['average_time']
        
        # Rough estimate: half the online tasks will run before this one
        wait_time += (online_count / 2) * avg_duration
        
        return wait_time
    
    def remove_task(self, task_id: str) -> Optional[Task]:
        """
        Remove task from queue
        
        Args:
            task_id: Task ID to remove
            
        Returns:
            Removed task or None if not found
        """
        task = self.task_map.pop(task_id, None)
        if task is None:
            return None
        
        # Remove from appropriate queue
        if task.is_online_task():
            # Rebuild heap without the task
            self.online_tasks = [(p, tid, t) for p, tid, t in self.online_tasks if tid != task_id]
            heapq.heapify(self.online_tasks)
        else:
            # Remove from deque
            self.offline_tasks = deque(t for t in self.offline_tasks if t.task_id != task_id)
        
        return task
    
    def get_next_task(self) -> Optional[Task]:
        """
        Get next task to execute
        
        Returns:
            Next task or None if queue is empty
        """
        # Check for timeout tasks first
        self._check_timeout_tasks()
        
        # Always prioritize online tasks
        while self.online_tasks:
            _, task_id, task = heapq.heappop(self.online_tasks)
            
            # Skip if task was already removed
            if task_id not in self.task_map:
                continue
            
            # Check if task already timed out
            if task.is_timeout():
                task.mark_timeout()
                self.total_timeout += 1
                continue
            
            self.running_task = task
            return task
        
        # Then check offline tasks
        while self.offline_tasks:
            task = self.offline_tasks.popleft()
            
            # Skip if task was already removed
            if task.task_id not in self.task_map:
                continue
            
            self.running_task = task
            return task
        
        return None
    
    def _check_timeout_tasks(self) -> None:
        """Check and mark timeout tasks"""
        # Check online tasks for timeouts
        timeout_tasks = []
        
        for priority, task_id, task in self.online_tasks:
            if task.is_timeout() and task.status == TaskStatus.QUEUED:
                timeout_tasks.append(task_id)
        
        # Mark timeout tasks
        for task_id in timeout_tasks:
            task = self.task_map.get(task_id)
            if task:
                task.mark_timeout()
                self.total_timeout += 1
    
    def estimate_time_cost(self, online_only: bool = True) -> float:
        """
        Estimate total execution time for queued tasks
        
        Args:
            online_only: If True, only count online tasks
            
        Returns:
            Estimated total time in seconds
        """
        total_time = 0.0
        
        # Account for running task
        if self.running_task:
            elapsed = time.time() - self.running_task.actual_start_time
            remaining = max(0, self.running_task.estimated_duration - elapsed)
            total_time += remaining
        
        # Add online tasks
        for _, _, task in self.online_tasks:
            if task.status == TaskStatus.QUEUED:
                total_time += task.estimated_duration
        
        # Add offline tasks if requested
        if not online_only:
            for task in self.offline_tasks:
                if task.status == TaskStatus.QUEUED:
                    total_time += task.estimated_duration
        
        return total_time
    
    def estimate_task_wait_time(self, task: Task) -> float:
        """
        Estimate wait time for a specific task
        
        Args:
            task: Task to estimate wait time for
            
        Returns:
            Estimated wait time in seconds
        """
        if task.task_id not in self.task_map:
            return 0.0
        
        wait_time = 0.0
        
        # Account for running task
        if self.running_task and self.running_task.task_id != task.task_id:
            elapsed = time.time() - self.running_task.actual_start_time
            remaining = max(0, self.running_task.estimated_duration - elapsed)
            wait_time += remaining
        
        # Calculate wait time based on tasks ahead
        if task.is_online_task():
            # For online tasks, consider other online tasks with higher priority
            task_urgency = self._calculate_task_urgency(task)
            
            for priority, _, other_task in self.online_tasks:
                if other_task.task_id == task.task_id:
                    break
                if priority < task_urgency:  # Higher priority (lower value)
                    wait_time += other_task.estimated_duration
        else:
            # For offline tasks, all online tasks go first
            for _, _, online_task in self.online_tasks:
                wait_time += online_task.estimated_duration
            
            # Then count offline tasks ahead
            for offline_task in self.offline_tasks:
                if offline_task.task_id == task.task_id:
                    break
                wait_time += offline_task.estimated_duration
        
        return wait_time
    
    def get_queue_status(self) -> Dict:
        """Get comprehensive queue status"""
        online_count = len([t for _, _, t in self.online_tasks if t.status == TaskStatus.QUEUED])
        offline_count = len([t for t in self.offline_tasks if t.status == TaskStatus.QUEUED])
        
        # Calculate average task duration
        stats = self.predictor.get_model_performance_stats(self.model_name)
        avg_duration = stats['average_time'] if stats['total_runs'] > 0 else 0
        
        # Calculate utilization
        total_tasks = online_count + offline_count
        if self.running_task:
            total_tasks += 1
        
        return {
            'queue_id': self.queue_id,
            'model_name': self.model_name,
            'online_tasks': online_count,
            'offline_tasks': offline_count,
            'running_task': self.running_task.task_id if self.running_task else None,
            'total_tasks': total_tasks,
            'estimated_wait_time_online': self.estimate_time_cost(online_only=True),
            'estimated_wait_time_total': self.estimate_time_cost(online_only=False),
            'average_task_duration': avg_duration,
            'total_processed': self.total_processed,
            'total_failed': self.total_failed,
            'total_timeout': self.total_timeout,
            'scheduling_method': self.scheduling_method.value
        }
    
    def reorder_tasks(self) -> None:
        """Reorder tasks based on current scheduling method and urgency"""
        if self.scheduling_method != SchedulingMethod.DEADLINE_AWARE:
            return
        
        # Rebuild online task heap with updated urgencies
        new_heap = []
        for _, task_id, task in self.online_tasks:
            if task.status == TaskStatus.QUEUED:
                urgency = self._calculate_task_urgency(task)
                heapq.heappush(new_heap, (urgency, task_id, task))
        
        self.online_tasks = new_heap
    
    def mark_task_completed(self, task_id: str, result: Any = None) -> None:
        """Mark task as completed and update statistics"""
        task = self.task_map.get(task_id)
        if task:
            task.mark_completed(result)
            self.total_processed += 1
            
            if task == self.running_task:
                self.running_task = None
    
    def mark_task_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed and update statistics"""
        task = self.task_map.get(task_id)
        if task:
            task.mark_failed(error)
            self.total_failed += 1
            
            if task == self.running_task:
                self.running_task = None
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in queue"""
        return list(self.task_map.values())
    
    def clear_completed_tasks(self) -> int:
        """
        Remove completed, failed, and timeout tasks from memory
        
        Returns:
            Number of tasks cleared
        """
        completed_statuses = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT}
        tasks_to_remove = [
            task_id for task_id, task in self.task_map.items()
            if task.status in completed_statuses
        ]
        
        for task_id in tasks_to_remove:
            self.task_map.pop(task_id, None)
        
        return len(tasks_to_remove)
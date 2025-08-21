from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional
import uuid
import time


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    WAITING_DEPENDENCY = "waiting_dependency"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """
    Task definition for the scheduling system
    
    Attributes:
        task_id: Unique identifier for the task (auto-generated UUID)
        model_name: Name of the OCR model to run
        input_data: Input data for the model
        dependency_task_ids: List of task IDs that must complete before this task
        timeout: Timeout in seconds (None for offline tasks)
        start_time: Timestamp when the task was created
        priority: Task priority (higher values = higher priority)
        status: Current task status
        estimated_duration: Estimated execution time in seconds
        actual_start_time: Timestamp when task actually started executing
        actual_end_time: Timestamp when task completed
        result: Task execution result
        error: Error message if task failed
    """
    model_name: str
    input_data: Any
    dependency_task_ids: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    priority: int = 0
    
    # Auto-generated fields
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    
    # Status fields
    status: TaskStatus = TaskStatus.PENDING
    estimated_duration: Optional[float] = None
    actual_start_time: Optional[float] = None
    actual_end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Queue assignment
    assigned_queue_id: Optional[str] = None
    
    def is_online_task(self) -> bool:
        """Check if this is an online task (has timeout)"""
        return self.timeout is not None
    
    def is_offline_task(self) -> bool:
        """Check if this is an offline task (no timeout)"""
        return self.timeout is None
    
    def get_remaining_time(self) -> Optional[float]:
        """
        Get remaining time for online task
        Returns None for offline tasks or if timeout already passed
        """
        if self.timeout is None:
            return None
        
        elapsed = time.time() - self.start_time
        remaining = self.timeout - elapsed
        
        return remaining if remaining > 0 else 0
    
    def is_timeout(self) -> bool:
        """Check if task has timed out"""
        if self.timeout is None:
            return False
        
        return time.time() - self.start_time > self.timeout
    
    def get_actual_duration(self) -> Optional[float]:
        """Get actual execution duration if task completed"""
        if self.actual_start_time is None or self.actual_end_time is None:
            return None
        
        return self.actual_end_time - self.actual_start_time
    
    def get_wait_time(self) -> Optional[float]:
        """Get time spent waiting before execution"""
        if self.actual_start_time is None:
            return None
        
        return self.actual_start_time - self.start_time
    
    def mark_running(self) -> None:
        """Mark task as running"""
        self.status = TaskStatus.RUNNING
        self.actual_start_time = time.time()
    
    def mark_completed(self, result: Any = None) -> None:
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.actual_end_time = time.time()
        self.result = result
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error"""
        self.status = TaskStatus.FAILED
        self.actual_end_time = time.time()
        self.error = error
    
    def mark_timeout(self) -> None:
        """Mark task as timed out"""
        self.status = TaskStatus.TIMEOUT
        self.actual_end_time = time.time()
        self.error = f"Task timed out after {self.timeout} seconds"
    
    def can_execute(self, completed_task_ids: set) -> bool:
        """
        Check if task can be executed based on dependencies
        
        Args:
            completed_task_ids: Set of completed task IDs
            
        Returns:
            True if all dependencies are satisfied
        """
        if not self.dependency_task_ids:
            return True
        
        return all(dep_id in completed_task_ids for dep_id in self.dependency_task_ids)
    
    def to_dict(self) -> dict:
        """Convert task to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'model_name': self.model_name,
            'input_data': self.input_data,
            'dependency_task_ids': self.dependency_task_ids,
            'timeout': self.timeout,
            'priority': self.priority,
            'start_time': self.start_time,
            'status': self.status.value,
            'estimated_duration': self.estimated_duration,
            'actual_start_time': self.actual_start_time,
            'actual_end_time': self.actual_end_time,
            'assigned_queue_id': self.assigned_queue_id,
            'error': self.error
        }
    
    def __repr__(self) -> str:
        return (f"Task(id={self.task_id[:8]}..., model={self.model_name}, "
                f"status={self.status.value}, priority={self.priority}, "
                f"online={self.is_online_task()})")
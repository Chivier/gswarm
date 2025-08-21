from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueueMetrics:
    """Metrics for a single task queue"""
    queue_id: str
    model_name: str
    total_tasks: int = 0
    online_tasks: int = 0
    offline_tasks: int = 0
    running_tasks: int = 0
    estimated_wait_time_online: float = 0.0
    estimated_wait_time_offline: float = 0.0
    average_task_duration: float = 0.0
    queue_utilization: float = 0.0
    total_processed: int = 0
    total_failed: int = 0
    total_timeout: int = 0
    
    def calculate_utilization(self, time_window: float = 3600.0) -> float:
        """
        Calculate queue utilization over a time window
        
        Args:
            time_window: Time window in seconds (default: 1 hour)
            
        Returns:
            Utilization percentage (0-100)
        """
        if time_window <= 0 or self.average_task_duration <= 0:
            return 0.0
        
        # Estimate busy time based on processed tasks
        busy_time = self.total_processed * self.average_task_duration
        
        # Calculate utilization percentage
        utilization = (busy_time / time_window) * 100
        
        # Cap at 100%
        return min(utilization, 100.0)


@dataclass
class SchedulerConfig:
    """Configuration for the hybrid scheduler"""
    # Queue management
    max_queues_per_model: int = 5
    min_queue_utilization: float = 20.0  # Minimum utilization before considering queue removal
    max_queue_utilization: float = 80.0  # Maximum utilization before adding new queue
    
    # Rebalancing
    queue_rebalance_interval: float = 60.0  # Seconds between rebalance checks
    enable_auto_scaling: bool = True  # Auto add/remove queues based on load
    
    # Task management
    task_timeout_check_interval: float = 10.0  # Seconds between timeout checks
    enable_preemption: bool = True  # Allow online tasks to preempt offline tasks
    max_retry_count: int = 3  # Maximum retries for failed tasks
    retry_delay_base: float = 5.0  # Base delay for retries (exponential backoff)
    
    # Performance tuning
    predictor_update_batch_size: int = 10  # Update predictor after N completed tasks
    task_cleanup_interval: float = 300.0  # Seconds between cleaning completed tasks
    max_completed_tasks_retention: int = 1000  # Max completed tasks to keep in memory
    
    # Scheduling policies
    online_task_preemption_threshold: float = 0.8  # Preempt if online task uses >80% of timeout
    load_balancing_threshold: float = 0.2  # Rebalance if queue load differs by >20%
    
    # Monitoring
    enable_metrics_collection: bool = True
    metrics_collection_interval: float = 30.0  # Seconds between metrics snapshots
    
    def validate(self) -> None:
        """Validate configuration values"""
        if self.max_queues_per_model < 1:
            raise ValueError("max_queues_per_model must be at least 1")
        
        if self.min_queue_utilization >= self.max_queue_utilization:
            raise ValueError("min_queue_utilization must be less than max_queue_utilization")
        
        if self.online_task_preemption_threshold <= 0 or self.online_task_preemption_threshold > 1:
            raise ValueError("online_task_preemption_threshold must be between 0 and 1")
        
        if self.load_balancing_threshold <= 0 or self.load_balancing_threshold > 1:
            raise ValueError("load_balancing_threshold must be between 0 and 1")


@dataclass
class SchedulerMetrics:
    """Overall scheduler metrics"""
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tasks_timeout: int = 0
    total_tasks_retried: int = 0
    
    # Queue metrics
    active_queues: int = 0
    total_queues_created: int = 0
    total_queues_removed: int = 0
    
    # Performance metrics
    average_wait_time_online: float = 0.0
    average_wait_time_offline: float = 0.0
    average_execution_time: float = 0.0
    average_queue_utilization: float = 0.0
    
    # Model-specific metrics
    model_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    def update_model_metrics(self, model_name: str, metrics: Dict) -> None:
        """Update metrics for a specific model"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = {
                'total_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'timeout_tasks': 0,
                'average_duration': 0.0,
                'queue_count': 0
            }
        
        self.model_metrics[model_name].update(metrics)
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.total_tasks_completed + self.total_tasks_failed + self.total_tasks_timeout
        if total == 0:
            return 0.0
        
        return (self.total_tasks_completed / total) * 100
    
    def get_timeout_rate(self) -> float:
        """Calculate timeout rate for online tasks"""
        if self.total_tasks_submitted == 0:
            return 0.0
        
        return (self.total_tasks_timeout / self.total_tasks_submitted) * 100
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for reporting"""
        return {
            'total_tasks': {
                'submitted': self.total_tasks_submitted,
                'completed': self.total_tasks_completed,
                'failed': self.total_tasks_failed,
                'timeout': self.total_tasks_timeout,
                'retried': self.total_tasks_retried
            },
            'queues': {
                'active': self.active_queues,
                'total_created': self.total_queues_created,
                'total_removed': self.total_queues_removed
            },
            'performance': {
                'avg_wait_time_online': self.average_wait_time_online,
                'avg_wait_time_offline': self.average_wait_time_offline,
                'avg_execution_time': self.average_execution_time,
                'avg_queue_utilization': self.average_queue_utilization,
                'success_rate': self.get_success_rate(),
                'timeout_rate': self.get_timeout_rate()
            },
            'models': self.model_metrics
        }


@dataclass
class TaskRetryInfo:
    """Information for task retry logic"""
    task_id: str
    retry_count: int = 0
    last_error: Optional[str] = None
    next_retry_time: Optional[float] = None
    
    def should_retry(self, max_retries: int) -> bool:
        """Check if task should be retried"""
        return self.retry_count < max_retries
    
    def calculate_next_retry_time(self, base_delay: float) -> float:
        """Calculate next retry time with exponential backoff"""
        import time
        delay = base_delay * (2 ** self.retry_count)
        self.next_retry_time = time.time() + delay
        return self.next_retry_time
import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import uuid
import logging

from .task import Task, TaskStatus
from .task_queue import TaskQueue, SchedulingMethod
from .ocr_model_predictor import OCRModelPredictor
from .scheduler_types import (
    QueueMetrics, SchedulerConfig, SchedulerMetrics, TaskRetryInfo
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridScheduler:
    """
    Hybrid scheduler for managing OCR model tasks with online/offline support
    
    Features:
    - Dynamic queue management per model
    - Deadline-aware scheduling for online tasks
    - Load balancing across queues
    - Automatic retry with exponential backoff
    - Preemption support for urgent online tasks
    """
    
    def __init__(self, predictor: OCRModelPredictor, config: Optional[SchedulerConfig] = None):
        """
        Initialize the hybrid scheduler
        
        Args:
            predictor: Model predictor for time estimation
            config: Scheduler configuration (uses defaults if None)
        """
        self.predictor = predictor
        self.config = config or SchedulerConfig()
        self.config.validate()
        
        # Queue management
        self.queues: Dict[str, TaskQueue] = {}  # queue_id -> TaskQueue
        self.model_queues: Dict[str, List[str]] = defaultdict(list)  # model_name -> [queue_ids]
        
        # Task tracking
        self.all_tasks: Dict[str, Task] = {}  # task_id -> Task
        self.completed_task_ids: Set[str] = set()  # For dependency tracking
        self.retry_info: Dict[str, TaskRetryInfo] = {}  # task_id -> TaskRetryInfo
        
        # Metrics
        self.metrics = SchedulerMetrics()
        
        # Threading
        self.lock = threading.RLock()
        self.running = False
        self.threads = []
        
        # Start background threads
        self._start_background_threads()
    
    def _start_background_threads(self) -> None:
        """Start background maintenance threads"""
        self.running = True
        
        # Timeout checker thread
        if self.config.task_timeout_check_interval > 0:
            timeout_thread = threading.Thread(
                target=self._timeout_checker_loop,
                daemon=True
            )
            timeout_thread.start()
            self.threads.append(timeout_thread)
        
        # Queue rebalancer thread
        if self.config.queue_rebalance_interval > 0:
            rebalance_thread = threading.Thread(
                target=self._rebalancer_loop,
                daemon=True
            )
            rebalance_thread.start()
            self.threads.append(rebalance_thread)
        
        # Task cleaner thread
        if self.config.task_cleanup_interval > 0:
            cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            cleanup_thread.start()
            self.threads.append(cleanup_thread)
    
    def _timeout_checker_loop(self) -> None:
        """Background thread to check for timeout tasks"""
        while self.running:
            try:
                self.reschedule_timeout_tasks()
                time.sleep(self.config.task_timeout_check_interval)
            except Exception as e:
                logger.error(f"Error in timeout checker: {e}")
    
    def _rebalancer_loop(self) -> None:
        """Background thread to rebalance queues"""
        while self.running:
            try:
                time.sleep(self.config.queue_rebalance_interval)
                if self.config.enable_auto_scaling:
                    self._auto_scale_queues()
                self._rebalance_queues()
            except Exception as e:
                logger.error(f"Error in rebalancer: {e}")
    
    def _cleanup_loop(self) -> None:
        """Background thread to clean completed tasks"""
        while self.running:
            try:
                time.sleep(self.config.task_cleanup_interval)
                self._cleanup_completed_tasks()
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
    
    def add_queue(self, model_name: str) -> str:
        """
        Create a new queue for a specific model
        
        Args:
            model_name: OCR model name
            
        Returns:
            Queue ID of the created queue
        """
        with self.lock:
            # Check queue limit
            if len(self.model_queues[model_name]) >= self.config.max_queues_per_model:
                raise ValueError(
                    f"Maximum queues ({self.config.max_queues_per_model}) "
                    f"reached for model {model_name}"
                )
            
            # Create new queue
            queue_id = f"{model_name}_{uuid.uuid4().hex[:8]}"
            queue = TaskQueue(queue_id, model_name, self.predictor)
            
            # Register queue
            self.queues[queue_id] = queue
            self.model_queues[model_name].append(queue_id)
            
            # Update metrics
            self.metrics.total_queues_created += 1
            self.metrics.active_queues += 1
            
            logger.info(f"Created queue {queue_id} for model {model_name}")
            return queue_id
    
    def remove_queue(self, queue_id: str) -> bool:
        """
        Remove a queue
        
        Args:
            queue_id: Queue ID to remove
            
        Returns:
            True if removed, False if not found or has tasks
        """
        with self.lock:
            queue = self.queues.get(queue_id)
            if not queue:
                return False
            
            # Don't remove if queue has tasks
            if queue.get_queue_status()['total_tasks'] > 0:
                logger.warning(f"Cannot remove queue {queue_id} with active tasks")
                return False
            
            # Remove queue
            del self.queues[queue_id]
            self.model_queues[queue.model_name].remove(queue_id)
            
            # Update metrics
            self.metrics.active_queues -= 1
            self.metrics.total_queues_removed += 1
            
            logger.info(f"Removed queue {queue_id}")
            return True
    
    def submit_task(self, task: Task) -> str:
        """
        Submit a task to the optimal queue
        
        Args:
            task: Task to submit
            
        Returns:
            Queue ID where task was submitted
        """
        with self.lock:
            # Validate task
            if task.task_id in self.all_tasks:
                raise ValueError(f"Task {task.task_id} already exists")
            
            # Check dependencies
            if not self.check_dependencies(task):
                task.status = TaskStatus.WAITING_DEPENDENCY
            
            # Find optimal queue
            queue_id = self.find_optimal_queue(task)
            
            # Add task to queue
            queue = self.queues[queue_id]
            queue.add_task(task)
            
            # Track task
            self.all_tasks[task.task_id] = task
            
            # Update metrics
            self.metrics.total_tasks_submitted += 1
            
            logger.info(
                f"Submitted task {task.task_id} to queue {queue_id} "
                f"(online={task.is_online_task()})"
            )
            
            return queue_id
    
    def find_optimal_queue(self, task: Task) -> str:
        """
        Find the optimal queue for a task
        
        Args:
            task: Task to place
            
        Returns:
            Queue ID of the optimal queue
        """
        with self.lock:
            # Ensure at least one queue exists for the model
            if task.model_name not in self.model_queues or not self.model_queues[task.model_name]:
                queue_id = self.add_queue(task.model_name)
                return queue_id
            
            # Get all queues for this model
            queue_ids = self.model_queues[task.model_name]
            
            if task.is_online_task():
                # For online tasks: find queue with earliest completion time
                best_queue_id = None
                min_completion_time = float('inf')
                
                for queue_id in queue_ids:
                    queue = self.queues[queue_id]
                    
                    # Estimate when this task would complete in this queue
                    wait_time = queue.estimate_task_wait_time(task)
                    completion_time = wait_time + task.estimated_duration
                    
                    # Check if task can meet deadline in this queue
                    remaining_time = task.get_remaining_time()
                    if remaining_time and completion_time < remaining_time:
                        if completion_time < min_completion_time:
                            min_completion_time = completion_time
                            best_queue_id = queue_id
                
                # If no queue can meet deadline, choose least loaded
                if best_queue_id is None:
                    logger.warning(
                        f"Task {task.task_id} may not meet deadline in any queue"
                    )
                    best_queue_id = self._find_least_loaded_queue(queue_ids)
                
                return best_queue_id
            
            else:
                # For offline tasks: find least loaded queue
                return self._find_least_loaded_queue(queue_ids)
    
    def _find_least_loaded_queue(self, queue_ids: List[str]) -> str:
        """Find queue with lowest load"""
        min_load = float('inf')
        best_queue_id = queue_ids[0]
        
        for queue_id in queue_ids:
            queue = self.queues[queue_id]
            load = queue.estimate_time_cost(online_only=False)
            
            if load < min_load:
                min_load = load
                best_queue_id = queue_id
        
        return best_queue_id
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get current status of a task"""
        with self.lock:
            task = self.all_tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            return task.status
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        with self.lock:
            task = self.all_tasks.get(task_id)
            if not task:
                return False
            
            # Can't cancel completed tasks
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]:
                return False
            
            # Remove from queue
            if task.assigned_queue_id:
                queue = self.queues.get(task.assigned_queue_id)
                if queue:
                    queue.remove_task(task_id)
            
            # Mark as failed
            task.mark_failed("Cancelled by user")
            
            logger.info(f"Cancelled task {task_id}")
            return True
    
    def get_scheduler_metrics(self) -> Dict:
        """Get comprehensive scheduler metrics"""
        with self.lock:
            # Update queue counts
            self.metrics.active_queues = len(self.queues)
            
            # Calculate average metrics
            total_wait_online = 0
            total_wait_offline = 0
            total_utilization = 0
            queue_count = 0
            
            for queue in self.queues.values():
                status = queue.get_queue_status()
                total_wait_online += status['estimated_wait_time_online']
                total_wait_offline += status['estimated_wait_time_total'] - status['estimated_wait_time_online']
                
                metrics = QueueMetrics(
                    queue_id=queue.queue_id,
                    model_name=queue.model_name,
                    **status
                )
                total_utilization += metrics.calculate_utilization()
                queue_count += 1
            
            if queue_count > 0:
                self.metrics.average_wait_time_online = total_wait_online / queue_count
                self.metrics.average_wait_time_offline = total_wait_offline / queue_count
                self.metrics.average_queue_utilization = total_utilization / queue_count
            
            return self.metrics.to_dict()
    
    def handle_task_completion(self, task_id: str, result: Any) -> None:
        """Handle successful task completion"""
        with self.lock:
            task = self.all_tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for completion")
                return
            
            # Update task
            queue = self.queues.get(task.assigned_queue_id)
            if queue:
                queue.mark_task_completed(task_id, result)
            
            # Track completion
            self.completed_task_ids.add(task_id)
            self.metrics.total_tasks_completed += 1
            
            # Update predictor
            if task.actual_start_time and task.actual_end_time:
                duration = task.actual_end_time - task.actual_start_time
                self.predictor.update_performance_history(
                    task_id,
                    duration,
                    {
                        'model_name': task.model_name,
                        'params': task.input_data,
                        'device': task.input_data.get('device', 'cpu')
                    }
                )
            
            # Check dependent tasks
            self._check_dependent_tasks()
            
            logger.info(f"Task {task_id} completed successfully")
    
    def handle_task_failure(self, task_id: str, error: str) -> None:
        """Handle task failure with retry logic"""
        with self.lock:
            task = self.all_tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for failure handling")
                return
            
            # Update task
            queue = self.queues.get(task.assigned_queue_id)
            if queue:
                queue.mark_task_failed(task_id, error)
            
            # Check retry
            retry_info = self.retry_info.get(task_id, TaskRetryInfo(task_id))
            retry_info.retry_count += 1
            retry_info.last_error = error
            
            if retry_info.should_retry(self.config.max_retry_count):
                # Calculate retry time
                retry_time = retry_info.calculate_next_retry_time(self.config.retry_delay_base)
                self.retry_info[task_id] = retry_info
                
                # Schedule retry
                logger.info(
                    f"Task {task_id} failed (attempt {retry_info.retry_count}), "
                    f"will retry at {time.ctime(retry_time)}"
                )
                
                # Reset task for retry
                task.status = TaskStatus.PENDING
                task.error = None
                self.metrics.total_tasks_retried += 1
                
                # Resubmit after delay
                threading.Timer(
                    retry_time - time.time(),
                    lambda: self.submit_task(task)
                ).start()
            else:
                # Max retries exceeded
                self.metrics.total_tasks_failed += 1
                logger.error(f"Task {task_id} failed after {retry_info.retry_count} attempts: {error}")
    
    def check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        return task.can_execute(self.completed_task_ids)
    
    def _check_dependent_tasks(self) -> None:
        """Check and update tasks waiting on dependencies"""
        for task in self.all_tasks.values():
            if task.status == TaskStatus.WAITING_DEPENDENCY:
                if self.check_dependencies(task):
                    task.status = TaskStatus.PENDING
                    self.submit_task(task)
    
    def reschedule_timeout_tasks(self) -> List[str]:
        """Check and reschedule timeout tasks"""
        timeout_task_ids = []
        
        with self.lock:
            for queue in self.queues.values():
                # Check timeout tasks in queue
                for task in queue.get_all_tasks():
                    if task.is_timeout() and task.status in [TaskStatus.QUEUED, TaskStatus.PENDING]:
                        timeout_task_ids.append(task.task_id)
                        task.mark_timeout()
                        self.metrics.total_tasks_timeout += 1
                        
                        # Try to reschedule if within retry limit
                        retry_info = self.retry_info.get(task.task_id, TaskRetryInfo(task.task_id))
                        if retry_info.should_retry(self.config.max_retry_count):
                            # Create new task with extended timeout
                            new_task = Task(
                                model_name=task.model_name,
                                input_data=task.input_data,
                                dependency_task_ids=task.dependency_task_ids,
                                timeout=task.timeout * 1.5 if task.timeout else None,
                                priority=task.priority + 1  # Increase priority
                            )
                            self.submit_task(new_task)
                            logger.info(f"Rescheduled timeout task {task.task_id} as {new_task.task_id}")
        
        return timeout_task_ids
    
    def _auto_scale_queues(self) -> None:
        """Automatically scale queues based on load"""
        for model_name, queue_ids in self.model_queues.items():
            if not queue_ids:
                continue
            
            # Calculate average utilization
            total_utilization = 0
            for queue_id in queue_ids:
                queue = self.queues[queue_id]
                metrics = QueueMetrics(
                    queue_id=queue.queue_id,
                    model_name=queue.model_name,
                    **queue.get_queue_status()
                )
                total_utilization += metrics.calculate_utilization()
            
            avg_utilization = total_utilization / len(queue_ids)
            
            # Scale up if overloaded
            if avg_utilization > self.config.max_queue_utilization:
                if len(queue_ids) < self.config.max_queues_per_model:
                    self.add_queue(model_name)
                    logger.info(f"Scaled up: added queue for {model_name} (util: {avg_utilization:.1f}%)")
            
            # Scale down if underutilized
            elif avg_utilization < self.config.min_queue_utilization and len(queue_ids) > 1:
                # Find least loaded queue
                min_tasks = float('inf')
                remove_queue_id = None
                
                for queue_id in queue_ids:
                    queue = self.queues[queue_id]
                    task_count = queue.get_queue_status()['total_tasks']
                    if task_count < min_tasks:
                        min_tasks = task_count
                        remove_queue_id = queue_id
                
                if remove_queue_id and min_tasks == 0:
                    self.remove_queue(remove_queue_id)
                    logger.info(f"Scaled down: removed queue {remove_queue_id} (util: {avg_utilization:.1f}%)")
    
    def _rebalance_queues(self) -> None:
        """Rebalance tasks across queues"""
        for model_name, queue_ids in self.model_queues.items():
            if len(queue_ids) < 2:
                continue
            
            # Calculate load for each queue
            queue_loads = []
            for queue_id in queue_ids:
                queue = self.queues[queue_id]
                load = queue.estimate_time_cost(online_only=False)
                queue_loads.append((queue_id, load))
            
            # Sort by load
            queue_loads.sort(key=lambda x: x[1])
            
            # Check if rebalancing needed
            min_load = queue_loads[0][1]
            max_load = queue_loads[-1][1]
            
            if max_load > 0 and (max_load - min_load) / max_load > self.config.load_balancing_threshold:
                # Move some offline tasks from highest to lowest loaded queue
                high_queue = self.queues[queue_loads[-1][0]]
                low_queue = self.queues[queue_loads[0][0]]
                
                # Get offline tasks from high load queue
                tasks_to_move = []
                for task in high_queue.offline_tasks:
                    if task.status == TaskStatus.QUEUED:
                        tasks_to_move.append(task)
                        if len(tasks_to_move) >= 2:  # Move at most 2 tasks at a time
                            break
                
                # Move tasks
                for task in tasks_to_move:
                    high_queue.remove_task(task.task_id)
                    low_queue.add_task(task)
                    logger.info(f"Rebalanced task {task.task_id} from {high_queue.queue_id} to {low_queue.queue_id}")
    
    def _cleanup_completed_tasks(self) -> None:
        """Clean up old completed tasks from memory"""
        with self.lock:
            # Get completed tasks sorted by completion time
            completed_tasks = [
                task for task in self.all_tasks.values()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]
            ]
            
            # Sort by actual end time (oldest first)
            completed_tasks.sort(key=lambda t: t.actual_end_time or float('inf'))
            
            # Keep only recent completed tasks
            if len(completed_tasks) > self.config.max_completed_tasks_retention:
                tasks_to_remove = completed_tasks[:-self.config.max_completed_tasks_retention]
                
                for task in tasks_to_remove:
                    # Remove from all tracking
                    del self.all_tasks[task.task_id]
                    self.completed_task_ids.discard(task.task_id)
                    self.retry_info.pop(task.task_id, None)
                    
                    # Remove from queue
                    if task.assigned_queue_id:
                        queue = self.queues.get(task.assigned_queue_id)
                        if queue:
                            queue.clear_completed_tasks()
                
                logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
    
    def shutdown(self) -> None:
        """Shutdown the scheduler and all background threads"""
        logger.info("Shutting down scheduler...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        logger.info("Scheduler shutdown complete")
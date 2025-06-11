"""Task queue manager implementation"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from loguru import logger
from collections import deque
import heapq


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Task:
    task_id: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    dependencies: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority.value < other.priority.value


class TaskQueueManager:
    """Manages asynchronous task execution with priorities and dependencies"""

    def __init__(self, max_concurrent_tasks: int = 4):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, Task] = {}
        self.pending_queue: List[Task] = []  # Priority queue
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)  # Keep last 1000 completed
        self.resource_locks: Dict[str, str] = {}  # resource -> task_id
        self._running = False
        self._executor_task = None

    async def start(self):
        """Start the task executor"""
        self._running = True
        self._executor_task = asyncio.create_task(self._executor_loop())
        logger.info("Task queue manager started")

    async def stop(self):
        """Stop the task executor"""
        self._running = False
        if self._executor_task:
            await self._executor_task
        logger.info("Task queue manager stopped")

    async def submit_task(
        self,
        task_type: str,
        priority: str = "normal",
        dependencies: List[str] = None,
        resources: Dict[str, Any] = None,
        payload: Dict[str, Any] = None,
    ) -> str:
        """Submit a new task to the queue"""
        task_id = f"{task_type}-{uuid.uuid4().hex[:8]}"

        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=TaskPriority[priority.upper()],
            status=TaskStatus.PENDING,
            dependencies=dependencies or [],
            resources=resources or {},
            payload=payload or {},
        )

        self.tasks[task_id] = task
        heapq.heappush(self.pending_queue, task)

        logger.info(f"Task submitted: {task_id} (type: {task_type}, priority: {priority})")
        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now().timestamp()
            self.completed_tasks.append(task)
            # Remove from pending queue
            self.pending_queue = [t for t in self.pending_queue if t.task_id != task_id]
            heapq.heapify(self.pending_queue)
            logger.info(f"Task cancelled: {task_id}")
            return True

        elif task.status == TaskStatus.RUNNING:
            # TODO: Implement graceful cancellation of running tasks
            logger.warning(f"Cannot cancel running task: {task_id}")
            return False

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            "pending": len(self.pending_queue),
            "running": len(self.running_tasks),
            "completed": len(self.completed_tasks),
            "config": {
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "priority_levels": [p.name.lower() for p in TaskPriority],
                "resource_tracking": True,
            },
        }

    def get_tasks(self, status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get tasks with optional status filter"""
        tasks = []

        # Add tasks based on status filter
        if not status or status == "pending":
            tasks.extend([t for t in self.pending_queue])
        if not status or status == "running":
            tasks.extend(self.running_tasks.values())
        if not status or status in ["completed", "failed", "cancelled"]:
            tasks.extend(self.completed_tasks)

        # Filter by specific status if provided
        if status:
            status_enum = TaskStatus(status)
            tasks = [t for t in tasks if t.status == status_enum]

        # Sort by created_at descending and limit
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        tasks = tasks[:limit]

        # Convert to dict format
        return [self._task_to_dict(t) for t in tasks]

    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert task to dictionary format"""
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "priority": task.priority.name.lower(),
            "status": task.status.value,
            "dependencies": task.dependencies,
            "resources": task.resources,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error_message": task.error_message,
        }

    async def _executor_loop(self):
        """Main executor loop"""
        while self._running:
            try:
                # Check for tasks that can be executed
                await self._schedule_tasks()

                # Short sleep to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in executor loop: {e}")
                await asyncio.sleep(1)

    async def _schedule_tasks(self):
        """Schedule tasks that are ready to run"""
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return

        # Find tasks that can be executed
        ready_tasks = []
        temp_queue = []

        while self.pending_queue:
            task = heapq.heappop(self.pending_queue)

            if self._can_execute_task(task):
                ready_tasks.append(task)
                if len(self.running_tasks) + len(ready_tasks) >= self.max_concurrent_tasks:
                    break
            else:
                temp_queue.append(task)

        # Put back tasks that can't run yet
        for task in temp_queue:
            heapq.heappush(self.pending_queue, task)

        # Execute ready tasks
        for task in ready_tasks:
            asyncio.create_task(self._execute_task(task))

    def _can_execute_task(self, task: Task) -> bool:
        """Check if a task can be executed"""
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False

        # Check resource conflicts
        if task.resources.get("exclusive"):
            for device in task.resources.get("devices", []):
                if device in self.resource_locks:
                    return False

        return True

    async def _execute_task(self, task: Task):
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().timestamp()
        self.running_tasks[task.task_id] = task

        # Lock resources
        if task.resources.get("exclusive"):
            for device in task.resources.get("devices", []):
                self.resource_locks[device] = task.task_id

        logger.info(f"Task started: {task.task_id}")

        try:
            # Execute task based on type
            await self._execute_task_type(task)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().timestamp()
            logger.info(f"Task completed: {task.task_id}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now().timestamp()
            task.error_message = str(e)
            logger.error(f"Task failed: {task.task_id} - {e}")

        finally:
            # Release resources
            if task.resources.get("exclusive"):
                for device in task.resources.get("devices", []):
                    if self.resource_locks.get(device) == task.task_id:
                        del self.resource_locks[device]

            # Move to completed
            del self.running_tasks[task.task_id]
            self.completed_tasks.append(task)

    async def _execute_task_type(self, task: Task):
        """Execute specific task type"""
        # This is where actual task execution logic would go
        # For now, just simulate work

        if task.task_type == "download":
            await asyncio.sleep(5)  # Simulate download
        elif task.task_type == "move":
            await asyncio.sleep(2)  # Simulate move
        elif task.task_type == "serve":
            await asyncio.sleep(1)  # Simulate service start
        else:
            await asyncio.sleep(1)  # Default simulation

        logger.debug(f"Task {task.task_id} execution simulated")


# Global instance
queue_manager = TaskQueueManager()

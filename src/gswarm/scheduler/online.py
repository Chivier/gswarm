"""
Online Greedy Scheduler for AI Workflows
Implements real-time scheduling with focus on minimizing P99 latency and average waiting time.
"""

import heapq
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from gswarm.scheduler.base import (
    SchedulerBase,
    ModelInfo,
    Workflow,
    Request,
    ScheduledTask,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class OnlineTask:
    """Represents a task in the online scheduler."""
    workflow_id: str
    node_id: str
    model_type: str
    request_id: str
    arrival_time: float
    dependencies: Set[str] = field(default_factory=set)
    estimated_time: float = 0.0
    priority: float = 0.0  # Dynamic priority

    # Tracking fields
    ready_time: float = 0.0  # When all dependencies are satisfied
    scheduled_time: float = 0.0  # When scheduled to GPU
    start_time: float = 0.0  # When actually started
    completion_time: float = 0.0

    def __lt__(self, other):
        # Higher priority (lower value) comes first
        return self.priority < other.priority

    @property
    def task_id(self) -> str:
        return f"{self.request_id}_{self.node_id}"

    @property
    def waiting_time(self) -> float:
        """Time spent waiting since becoming ready."""
        if self.start_time > 0:
            return self.start_time - self.ready_time
        return 0.0

    @property
    def response_time(self) -> float:
        """Total time from arrival to completion."""
        if self.completion_time > 0:
            return self.completion_time - self.arrival_time
        return 0.0


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    arrival_time: float
    completion_time: float = 0.0
    total_waiting_time: float = 0.0
    node_count: int = 0
    completed_nodes: int = 0


class OnlineScheduler(SchedulerBase):
    """
    Online greedy scheduler with focus on P99 and average waiting time.
    
    Key features:
    - Dynamic priority adjustment based on waiting time
    - Shortest Job First (SJF) with aging to prevent starvation
    - GPU affinity consideration for model placement
    - Real-time request handling
    """

    def __init__(self, gpus: List[int], models: Dict[str, ModelInfo], simulate: bool = False):
        super().__init__(gpus, models, simulate)
        
        # GPU state tracking
        self.gpu_models: Dict[int, Optional[str]] = {gpu: None for gpu in gpus}
        self.gpu_available_at: Dict[int, float] = {gpu: 0.0 for gpu in gpus}
        self.gpu_reserved: Dict[int, bool] = {gpu: False for gpu in gpus}
        
        # Task queues
        self.ready_queue: List[OnlineTask] = []  # Priority queue of ready tasks
        self.pending_tasks: Dict[str, OnlineTask] = {}  # task_id -> task
        self.completed_task_ids: Set[str] = set()
        
        # Request tracking
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.request_tasks: Dict[str, List[str]] = defaultdict(list)  # request_id -> [task_ids]
        
        # Time and priority parameters
        self.current_time = 0.0
        self.aging_factor = 0.1  # How much to increase priority per time unit
        self.sjf_weight = 0.7  # Weight for shortest job first
        self.wait_weight = 0.3  # Weight for waiting time
        
        # Performance tracking
        self.latency_history: List[float] = []
        
    def add_request(self, request: Request) -> None:
        """Add a new request to the online scheduler."""
        workflow = self.get_workflow(request.workflow_id)
        if not workflow:
            logger.error(f"Workflow {request.workflow_id} not found")
            return
            
        # Initialize request metrics
        self.request_metrics[request.id] = RequestMetrics(
            request_id=request.id,
            arrival_time=request.arrival_time,
            node_count=len(workflow.nodes),
        )
        
        # Create tasks for all nodes
        dependencies = workflow.get_dependencies()
        
        for node in workflow.nodes:
            task = OnlineTask(
                workflow_id=workflow.id,
                node_id=node.id,
                model_type=node.model,
                request_id=request.id,
                arrival_time=request.arrival_time,
                dependencies=dependencies.get(node.id, set()),
                estimated_time=self._estimate_node_time(node.model),
            )
            
            task_id = task.task_id
            self.pending_tasks[task_id] = task
            self.request_tasks[request.id].append(task_id)
            
            # If task has no dependencies, mark as ready
            if not task.dependencies:
                task.ready_time = request.arrival_time
                self._add_to_ready_queue(task)
                
        logger.info(f"Added online request {request.id} with {len(workflow.nodes)} nodes")
        
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task to execute based on online scheduling policy."""
        if not self.ready_queue:
            return None
            
        # Update priorities based on current time
        self._update_task_priorities()
        
        # Re-heapify after priority updates
        heapq.heapify(self.ready_queue)
        
        # Get highest priority task
        task = heapq.heappop(self.ready_queue)
        
        # Select best GPU
        best_gpu = self._select_gpu_for_online_task(task)
        
        # Calculate switch time if needed
        switch_time = 0.0
        if self.gpu_models[best_gpu] != task.model_type:
            if self.gpu_models[best_gpu] is not None:
                model_info = self.models.get(task.model_type)
                if model_info:
                    switch_time = model_info.load_time_seconds
            self.gpu_models[best_gpu] = task.model_type
            
        # Schedule task
        scheduled_time = max(self.current_time, self.gpu_available_at[best_gpu]) + switch_time
        
        scheduled_task = ScheduledTask(
            request_id=task.request_id,
            workflow_id=task.workflow_id,
            node_id=task.node_id,
            model_name=task.model_type,
            gpu_id=best_gpu,
            scheduled_time=scheduled_time,
            estimated_duration=task.estimated_time,
            dependencies=task.dependencies,
        )
        
        # Update task and GPU state
        task.scheduled_time = scheduled_time
        task.start_time = scheduled_time  # Assuming immediate start after scheduling
        self.gpu_available_at[best_gpu] = scheduled_time + task.estimated_time
        
        # Update request metrics
        if task.request_id in self.request_metrics:
            waiting_time = task.start_time - task.ready_time
            self.request_metrics[task.request_id].total_waiting_time += waiting_time
            
        logger.debug(f"Scheduled online task {task.task_id} on GPU {best_gpu}")
        
        return scheduled_task
        
    def complete_task(self, task: ScheduledTask) -> None:
        """Mark a task as completed and update dependencies."""
        task_id = task.task_id
        
        if task_id not in self.pending_tasks:
            logger.warning(f"Attempting to complete unknown task {task_id}")
            return
            
        online_task = self.pending_tasks[task_id]
        online_task.completion_time = self.current_time
        self.completed_task_ids.add(task_id)
        
        # Update metrics
        self.update_metrics(task)
        
        # Update request metrics
        if task.request_id in self.request_metrics:
            self.request_metrics[task.request_id].completed_nodes += 1
            
        # Check and update dependent tasks
        workflow = self.get_workflow(task.workflow_id)
        if workflow:
            dependents = workflow.get_dependents(task.node_id)
            
            for dependent_node_id in dependents:
                dependent_task_id = f"{task.request_id}_{dependent_node_id}"
                if dependent_task_id in self.pending_tasks:
                    dependent_task = self.pending_tasks[dependent_task_id]
                    
                    # Check if all dependencies are satisfied
                    deps_satisfied = all(
                        f"{task.request_id}_{dep}" in self.completed_task_ids
                        for dep in dependent_task.dependencies
                    )
                    
                    if deps_satisfied and dependent_task.ready_time == 0:
                        dependent_task.ready_time = self.current_time
                        self._add_to_ready_queue(dependent_task)
                        
        # Check if request is complete
        request_complete = all(
            tid in self.completed_task_ids
            for tid in self.request_tasks[task.request_id]
        )
        
        if request_complete:
            self._complete_request(task.request_id)
            
    def _add_to_ready_queue(self, task: OnlineTask) -> None:
        """Add a task to the ready queue with initial priority."""
        # Calculate initial priority
        task.priority = self._calculate_task_priority(task)
        heapq.heappush(self.ready_queue, task)
        
    def _calculate_task_priority(self, task: OnlineTask) -> float:
        """
        Calculate dynamic priority for a task.
        Lower value = higher priority.
        """
        # Shortest Job First component
        sjf_priority = task.estimated_time * self.sjf_weight
        
        # Waiting time component (aging)
        wait_time = self.current_time - task.ready_time
        wait_priority = -wait_time * self.wait_weight * self.aging_factor
        
        # Request age component
        request_age = self.current_time - task.arrival_time
        age_priority = -request_age * 0.01  # Small weight for request age
        
        return sjf_priority + wait_priority + age_priority
        
    def _update_task_priorities(self) -> None:
        """Update priorities for all tasks in ready queue."""
        for task in self.ready_queue:
            task.priority = self._calculate_task_priority(task)
            
    def _select_gpu_for_online_task(self, task: OnlineTask) -> int:
        """Select optimal GPU for online task execution."""
        best_gpu = None
        best_score = float('inf')
        
        for gpu in self.gpus:
            if self.gpu_reserved.get(gpu, False):
                continue
                
            # Calculate score based on:
            # 1. Availability time
            # 2. Model switch cost
            availability = max(0, self.gpu_available_at[gpu] - self.current_time)
            
            switch_cost = 0.0
            if self.gpu_models[gpu] != task.model_type:
                if self.gpu_models[gpu] is not None:
                    model_info = self.models.get(task.model_type)
                    if model_info:
                        switch_cost = model_info.load_time_seconds
                        
            score = availability + switch_cost * 0.5  # Weight switch cost less for online
            
            if score < best_score:
                best_score = score
                best_gpu = gpu
                
        return best_gpu
        
    def _estimate_node_time(self, model_name: str) -> float:
        """Estimate execution time for a node."""
        model_info = self.models.get(model_name)
        if model_info:
            if model_info.inference_time_mean:
                return model_info.inference_time_mean
            elif model_info.tokens_per_second:
                # Assume average of 100 tokens for LLM
                return 100.0 / model_info.tokens_per_second
        return 1.0  # Default estimation
        
    def _complete_request(self, request_id: str) -> None:
        """Mark a request as completed and update metrics."""
        if request_id not in self.request_metrics:
            return
            
        metrics = self.request_metrics[request_id]
        metrics.completion_time = self.current_time
        
        # Calculate total latency
        latency = metrics.completion_time - metrics.arrival_time
        self.latency_history.append(latency)
        
        # Update scheduler metrics
        self.metrics.completed_requests += 1
        
        # Calculate percentiles if enough data
        if len(self.latency_history) >= 10:
            sorted_latencies = sorted(self.latency_history)
            n = len(sorted_latencies)
            self.metrics.p50_latency = sorted_latencies[int(n * 0.5)]
            self.metrics.p90_latency = sorted_latencies[int(n * 0.9)]
            self.metrics.p99_latency = sorted_latencies[int(n * 0.99)]
            self.metrics.average_request_latency = sum(sorted_latencies) / n
            
        logger.info(f"Completed online request {request_id} with latency {latency:.2f}s")
        
    def advance_time(self, new_time: float) -> None:
        """Advance the scheduler's current time."""
        self.current_time = new_time
        
    def schedule(self, requests: List[Request]) -> List[ScheduledTask]:
        """
        Schedule requests in online mode.
        Note: This is mainly for compatibility; online scheduler works best with
        add_request/get_next_task for real-time scheduling.
        """
        scheduled_tasks = []
        
        # Sort requests by arrival time
        sorted_requests = sorted(requests, key=lambda r: r.arrival_time)
        
        for request in sorted_requests:
            # Advance time to request arrival
            self.advance_time(request.arrival_time)
            self.add_request(request)
            
            # Try to schedule any ready tasks
            while True:
                task = self.get_next_task()
                if task is None:
                    break
                scheduled_tasks.append(task)
                
        return scheduled_tasks
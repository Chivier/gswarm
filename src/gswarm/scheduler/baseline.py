"""
"""
Baseline Scheduler for GSwarm Workflows
Uses Ray-like scheduling: runs models one by one, maintaining a queue of ready nodes.
"""

import heapq
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import requests
from gswarm.predictor.predicting import update_predictor


from gswarm.scheduler.base import (
    SchedulerBase,
    ModelInfo,
    Workflow,
    Request,
    ScheduledTask,
    ExecutionMetrics,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeState:
    """Track node execution state in baseline scheduler."""
    request_id: str
    workflow_id: str
    node_id: str
    model_name: str
    dependencies: Set[str]
    completed_dependencies: Set[str] = field(default_factory=set)
    status: str = "pending"  # pending, ready, running, completed
    estimated_time: float = 0.0
    priority: int = 0  # Lower value = higher priority
    
    @property
    def is_ready(self) -> bool:
        """Check if node is ready to execute."""
        return len(self.dependencies) == len(self.completed_dependencies)
    
    @property
    def task_id(self) -> str:
        """Unique task identifier."""
        return f"{self.request_id}_{self.node_id}"


@dataclass
class GPUState:
    """Track GPU state in baseline scheduler."""
    gpu_id: int
    current_model: Optional[str] = None
    available_at: float = 0.0
    total_busy_time: float = 0.0
    model_switch_count: int = 0


class BaselineScheduler(SchedulerBase):
    """
    Baseline scheduler implementing Ray-like scheduling strategy.
    - Executes nodes one by one as dependencies are satisfied
    - Maintains a ready queue of executable nodes
    - Minimizes model switching when possible
    """
    
    def __init__(
        self, 
        devices: Union[List[int], List[str]] = None,
        gpus: List[int] = None,  # Legacy parameter
        models: Dict[str, ModelInfo] = None,
        simulate: bool = False,
        default_client: str = "localhost",
        profiler_url: Optional[str] = None,
    ):
        # Handle legacy parameter
        if devices is None and gpus is not None:
            devices = gpus
        elif devices is None:
            raise ValueError("Either 'devices' or 'gpus' parameter must be provided")
            
        super().__init__(devices, models, simulate, default_client)
        
        self.profiler_url = profiler_url
        # GPU state tracking
        self.gpu_states = {gpu_id: GPUState(gpu_id) for gpu_id in self.gpus}
        
        # Node execution tracking
        self.node_states: Dict[str, NodeState] = {}  # task_id -> NodeState
        self.ready_queue: List[Tuple[float, str]] = []  # heap of (priority, task_id)
        self.completed_nodes: Set[str] = set()  # Set of completed task_ids
        
        # Request tracking
        self.request_nodes: Dict[str, List[str]] = defaultdict(list)  # request_id -> [task_ids]
        
        # Time tracking
        self.current_time = 0.0
        
    def schedule(self, requests: List[Request]) -> List[ScheduledTask]:
        """
        Schedule a batch of requests using baseline strategy.
        Note: This is mainly for compatibility; baseline scheduler works best with add_request/get_next_task.
        """
        scheduled_tasks = []
        
        # Add all requests
        for request in requests:
            self.add_request(request)
        
        # Schedule all ready tasks
        while True:
            task = self.get_next_task()
            if task is None:
                break
            scheduled_tasks.append(task)
            
        return scheduled_tasks
    
    def add_request(self, request: Request) -> None:
        """Add a new request to the scheduler."""
        workflow = self.get_workflow(request.workflow_id)
        if not workflow:
            logger.error(f"Workflow {request.workflow_id} not found")
            return
            
        self.pending_requests.append(request)
        self.active_requests[request.id] = request
        
        # Create node states for all nodes in the workflow
        dependencies = workflow.get_dependencies()
        
        for node in workflow.nodes:
            node_state = NodeState(
                request_id=request.id,
                workflow_id=workflow.id,
                node_id=node.id,
                model_name=node.model,
                dependencies=dependencies.get(node.id, set()),
                estimated_time=self._estimate_node_time(node.model),
                priority=request.priority,
            )
            
            task_id = node_state.task_id
            self.node_states[task_id] = node_state
            self.request_nodes[request.id].append(task_id)
            
            # If node has no dependencies, add to ready queue
            if node_state.is_ready:
                node_state.status = "ready"
                heapq.heappush(self.ready_queue, (node_state.priority, task_id))
                
        logger.info(f"Added request {request.id} with {len(workflow.nodes)} nodes")
        
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task to execute."""
        if not self.ready_queue:
            return None
            
        # Find best GPU for next task
        _, task_id = heapq.heappop(self.ready_queue)
        
        if task_id not in self.node_states:
            # Task was cancelled or completed
            return self.get_next_task()
            
        node_state = self.node_states[task_id]
        if node_state.status != "ready":
            # Task is no longer ready
            return self.get_next_task()
            
        # Select GPU with minimal switch cost
        best_gpu_id = self._select_best_gpu(node_state.model_name)
        gpu_state = self.gpu_states[best_gpu_id]
        
        # Find corresponding device string
        device_str = None
        for i, device_info in enumerate(self.device_infos):
            if device_info.device_type == "cuda" and device_info.device_id == best_gpu_id:
                device_str = self.devices[i]
                break
        
        if device_str is None:
            device_str = f"localhost:cuda:{best_gpu_id}"  # Fallback
        
        # Calculate switch time if needed
        switch_time = 0.0
        if gpu_state.current_model != node_state.model_name:
            if gpu_state.current_model is not None:
                # Need to switch models
                model_info = self.models.get(node_state.model_name)
                if model_info:
                    switch_time = model_info.load_time_seconds
                    gpu_state.model_switch_count += 1
                    self.metrics.model_switch_count += 1
                    self.metrics.total_model_load_time += switch_time
            gpu_state.current_model = node_state.model_name
            
        # Create scheduled task
        scheduled_time = max(self.current_time, gpu_state.available_at) + switch_time
        
        task = ScheduledTask(
            request_id=node_state.request_id,
            workflow_id=node_state.workflow_id,
            node_id=node_state.node_id,
            model_name=node_state.model_name,
            device=device_str,
            scheduled_time=scheduled_time,
            estimated_duration=node_state.estimated_time,
            dependencies=node_state.dependencies,
        )
        
        # Update states
        node_state.status = "running"
        gpu_state.available_at = scheduled_time + node_state.estimated_time
        gpu_state.total_busy_time += node_state.estimated_time + switch_time
        
        self.scheduled_tasks.append(task)
        self.metrics.total_tasks += 1
        
        logger.debug(f"Scheduled task {task_id} on device {device_str} at time {scheduled_time:.2f}")
        
        return task
        
    def complete_task(self, task: ScheduledTask) -> None:
        """Mark a task as completed and update dependencies."""
        task_id = task.task_id
        
        if task_id not in self.node_states:
            logger.warning(f"Attempting to complete unknown task {task_id}")
            return
            
        node_state = self.node_states[task_id]
        node_state.status = "completed"
        self.completed_nodes.add(task_id)
        
        # Update metrics
        self.update_metrics(task)
        
        # Update predictor
        if self.profiler_url and task.actual_duration is not None:
            try:
                response = requests.get(f"{self.profiler_url}/metrics/latest")
                response.raise_for_status()
                latest_metrics = response.json()

                task_hostname = task.device.split(':')[0]
                gpu_physical_idx = int(task.device.split(':')[-1])
                
                gpu_util = 0
                mem_util = 0

                for client_id, client_data in latest_metrics.get('client_data', {}).items():
                    if task_hostname in client_id:
                        for gpu_metric in client_data.get('gpus_metrics', []):
                            if gpu_metric.get('physical_idx') == gpu_physical_idx:
                                gpu_util = gpu_metric.get('gpu_util', 0)
                                mem_util = gpu_metric.get('mem_util', 0)
                                break
                        break
                
                request = self.active_requests.get(task.request_id)
                inputs = request.metadata if request else {}
                
                data_features = [{
                    "processing_time": task.actual_duration,
                    "gpu_utilization": gpu_util,
                    "gpu_memory_cost": mem_util,
                    **inputs 
                }]

                update_predictor(
                    model_type='auto',
                    model_name=task.model_name,
                    device=task.device,
                    data_features=data_features
                )
            except Exception as e:
                logger.warning(f"Failed to update predictor for task {task.task_id}: {e}")

        # Update dependent nodes
        workflow = self.get_workflow(task.workflow_id)
        if workflow:
            dependents = workflow.get_dependents(task.node_id)
            
            for dependent_node_id in dependents:
                dependent_task_id = f"{task.request_id}_{dependent_node_id}"
                if dependent_task_id in self.node_states:
                    dependent_state = self.node_states[dependent_task_id]
                    dependent_state.completed_dependencies.add(task.node_id)
                    
                    # Check if dependent is now ready
                    if dependent_state.is_ready and dependent_state.status == "pending":
                        dependent_state.status = "ready"
                        heapq.heappush(
                            self.ready_queue,
                            (dependent_state.priority, dependent_task_id)
                        )
                        logger.debug(f"Node {dependent_task_id} is now ready")
                        
        # Check if request is completed
        request_tasks = self.request_nodes.get(task.request_id, [])
        if all(tid in self.completed_nodes for tid in request_tasks):
            self._complete_request(task.request_id)
            
    def _select_best_gpu(self, model_name: str) -> int:
        """Select the best GPU for running a model."""
        # First, try to find a GPU already running this model
        for gpu_id, gpu_state in self.gpu_states.items():
            if gpu_state.current_model == model_name:
                return gpu_id
                
        # Otherwise, find the GPU that will be available soonest
        best_gpu = min(
            self.gpu_states.keys(),
            key=lambda gpu_id: self.gpu_states[gpu_id].available_at
        )
        
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
        """Mark a request as completed."""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            self.completed_requests.add(request_id)
            del self.active_requests[request_id]
            
            self.metrics.completed_requests += 1
            
            # Calculate request latency
            request_tasks = [
                task for task in self.scheduled_tasks
                if task.request_id == request_id
            ]
            if request_tasks:
                start_time = min(task.scheduled_time for task in request_tasks)
                end_time = max(
                    task.scheduled_time + task.estimated_duration
                    for task in request_tasks
                )
                latency = end_time - request.arrival_time
                
                # Update latency metrics (simplified)
                self.metrics.average_request_latency = (
                    (self.metrics.average_request_latency * (self.metrics.completed_requests - 1) + latency)
                    / self.metrics.completed_requests
                )
                
            logger.info(f"Completed request {request_id}")
            
    def get_gpu_utilization(self) -> Dict[int, float]:
        """Calculate GPU utilization percentages."""
        if self.current_time == 0:
            return {gpu_id: 0.0 for gpu_id in self.gpus}
            
        utilization = {}
        for gpu_id, gpu_state in self.gpu_states.items():
            utilization[gpu_id] = (gpu_state.total_busy_time / self.current_time) * 100
            
        return utilization
        
    def advance_time(self, new_time: float) -> None:
        """Advance the scheduler's current time."""
        self.current_time = new_time
"""
Offline Batch Processing Scheduler for AI Workflows
Implements optimized scheduling to minimize model switching overhead while maintaining dependencies.
"""

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

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
class Task:
    """Represents a single task to be executed in offline mode."""
    workflow_id: str
    node_id: str
    model_type: str
    dependencies: Set[str] = field(default_factory=set)
    ready_time: float = 0.0
    priority: int = 0  # Lower value = higher priority
    estimated_time: float = 0.0
    request_id: str = ""

    def __lt__(self, other):
        # For heap queue - prioritize by (priority, ready_time, workflow_id, node_id)
        return (self.priority, self.ready_time, self.workflow_id, self.node_id) < (
            other.priority,
            other.ready_time,
            other.workflow_id,
            other.node_id,
        )


@dataclass
class WorkflowDAG:
    """Represents a workflow as a directed acyclic graph."""
    workflow_id: str
    nodes: Dict[str, str]  # node_id -> model_name
    edges: List[Tuple[str, str]]  # (from_node, to_node)
    topological_order: List[str] = field(default_factory=list)

    def compute_topological_order(self):
        """Compute topological ordering of nodes."""
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)

        # Build adjacency list and in-degree map
        for from_node, to_node in self.edges:
            adj_list[from_node].append(to_node)
            in_degree[to_node] += 1

        # Initialize queue with nodes having no dependencies
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        self.topological_order = []

        while queue:
            node_id = queue.popleft()
            self.topological_order.append(node_id)

            # Reduce in-degree for dependent nodes
            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get all dependencies for a given node."""
        deps = set()
        for from_node, to_node in self.edges:
            if to_node == node_id:
                deps.add(from_node)
        return deps


class OfflineScheduler(SchedulerBase):
    """
    Optimized offline batch processing scheduler.
    
    Key optimizations:
    - Groups tasks by model type to minimize switching
    - Respects workflow dependencies
    - Balances load across GPUs
    - Optimizes for total makespan
    """

    def __init__(self, gpus: List[int], models: Dict[str, ModelInfo], simulate: bool = False):
        super().__init__(gpus, models, simulate)
        
        # GPU tracking
        self.gpu_models: Dict[int, Optional[str]] = {gpu: None for gpu in gpus}
        self.gpu_available_time: Dict[int, float] = {gpu: 0.0 for gpu in gpus}
        
        # Task grouping
        self.model_task_groups: Dict[str, List[Task]] = defaultdict(list)
        self.ready_tasks: List[Task] = []
        
        # Dependency tracking
        self.task_dependencies: Dict[str, Set[str]] = {}
        self.task_dependents: Dict[str, Set[str]] = defaultdict(set)
        self.completed_tasks: Set[str] = set()
        
    def schedule(self, requests: List[Request]) -> List[ScheduledTask]:
        """Schedule a batch of requests for offline processing."""
        logger.info(f"Offline scheduling {len(requests)} requests")
        
        # Reset state for new batch
        self._reset_batch_state()
        
        # Build workflow DAGs and create tasks
        all_tasks = []
        workflow_dags = []
        
        for request in requests:
            workflow = self.get_workflow(request.workflow_id)
            if not workflow:
                logger.error(f"Workflow {request.workflow_id} not found")
                continue
                
            # Create workflow DAG
            dag = self._create_workflow_dag(workflow)
            dag.compute_topological_order()
            workflow_dags.append(dag)
            
            # Create tasks for each node
            for node in workflow.nodes:
                task = Task(
                    workflow_id=workflow.id,
                    node_id=node.id,
                    model_type=node.model,
                    dependencies=dag.get_dependencies(node.id),
                    estimated_time=self._estimate_node_time(node.model),
                    request_id=request.id,
                    priority=request.priority,
                )
                
                task_id = f"{request.id}_{node.id}"
                all_tasks.append((task_id, task))
                
                # Track dependencies
                self.task_dependencies[task_id] = {
                    f"{request.id}_{dep}" for dep in task.dependencies
                }
                for dep in task.dependencies:
                    dep_task_id = f"{request.id}_{dep}"
                    self.task_dependents[dep_task_id].add(task_id)
                    
        # Group tasks by model
        for task_id, task in all_tasks:
            self.model_task_groups[task.model_type].append((task_id, task))
            
        # Schedule tasks
        scheduled_tasks = self._optimize_schedule(all_tasks)
        
        logger.info(f"Scheduled {len(scheduled_tasks)} tasks")
        return scheduled_tasks
        
    def _optimize_schedule(self, all_tasks: List[Tuple[str, Task]]) -> List[ScheduledTask]:
        """Optimize task scheduling to minimize makespan."""
        scheduled_tasks = []
        current_time = 0.0
        
        # Initialize ready tasks (no dependencies)
        ready_heap = []
        for task_id, task in all_tasks:
            if not self.task_dependencies.get(task_id):
                heapq.heappush(ready_heap, (task.priority, task_id, task))
                
        # Process tasks
        while ready_heap or any(
            task_id not in self.completed_tasks
            for task_id, _ in all_tasks
        ):
            # Update ready tasks based on completed dependencies
            newly_ready = []
            for task_id, task in all_tasks:
                if (task_id not in self.completed_tasks and
                    task_id not in [t[1] for t in ready_heap] and
                    self._are_dependencies_satisfied(task_id)):
                    newly_ready.append((task.priority, task_id, task))
                    
            for item in newly_ready:
                heapq.heappush(ready_heap, item)
                
            if not ready_heap:
                # No tasks ready, advance time to next completion
                if self.gpu_available_time:
                    current_time = min(self.gpu_available_time.values())
                continue
                
            # Get next task
            _, task_id, task = heapq.heappop(ready_heap)
            
            # Find best GPU considering model affinity
            best_gpu = self._select_gpu_for_task(task, current_time)
            
            # Calculate start time and switch overhead
            gpu_available = self.gpu_available_time[best_gpu]
            start_time = max(current_time, gpu_available)
            
            switch_time = 0.0
            if self.gpu_models[best_gpu] != task.model_type:
                if self.gpu_models[best_gpu] is not None:
                    model_info = self.models.get(task.model_type)
                    if model_info:
                        switch_time = model_info.load_time_seconds
                self.gpu_models[best_gpu] = task.model_type
                
            # Create scheduled task
            scheduled_task = ScheduledTask(
                request_id=task.request_id,
                workflow_id=task.workflow_id,
                node_id=task.node_id,
                model_name=task.model_type,
                gpu_id=best_gpu,
                scheduled_time=start_time + switch_time,
                estimated_duration=task.estimated_time,
                dependencies=self.task_dependencies.get(task_id, set()),
            )
            
            scheduled_tasks.append(scheduled_task)
            
            # Update GPU availability
            self.gpu_available_time[best_gpu] = (
                start_time + switch_time + task.estimated_time
            )
            
            # Mark task as completed for dependency tracking
            self.completed_tasks.add(task_id)
            
        return scheduled_tasks
        
    def _select_gpu_for_task(self, task: Task, current_time: float) -> int:
        """Select optimal GPU for task considering model affinity and load balance."""
        best_gpu = None
        best_score = float('inf')
        
        for gpu in self.gpus:
            # Calculate score based on:
            # 1. Model switch cost
            # 2. GPU availability time
            # 3. Load balance
            
            switch_cost = 0.0
            if self.gpu_models[gpu] != task.model_type:
                if self.gpu_models[gpu] is not None:
                    model_info = self.models.get(task.model_type)
                    if model_info:
                        switch_cost = model_info.load_time_seconds
                        
            availability_cost = max(0, self.gpu_available_time[gpu] - current_time)
            
            # Combined score (lower is better)
            score = switch_cost + availability_cost
            
            if score < best_score:
                best_score = score
                best_gpu = gpu
                
        return best_gpu
        
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies of a task are satisfied."""
        deps = self.task_dependencies.get(task_id, set())
        return all(dep in self.completed_tasks for dep in deps)
        
    def _create_workflow_dag(self, workflow: Workflow) -> WorkflowDAG:
        """Create a DAG representation of the workflow."""
        nodes = {node.id: node.model for node in workflow.nodes}
        edges = [(edge.from_node, edge.to_node) for edge in workflow.edges]
        
        return WorkflowDAG(
            workflow_id=workflow.id,
            nodes=nodes,
            edges=edges,
        )
        
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
        
    def _reset_batch_state(self):
        """Reset state for a new batch."""
        self.gpu_models = {gpu: None for gpu in self.gpus}
        self.gpu_available_time = {gpu: 0.0 for gpu in self.gpus}
        self.model_task_groups.clear()
        self.ready_tasks.clear()
        self.task_dependencies.clear()
        self.task_dependents.clear()
        self.completed_tasks.clear()
        
    def add_request(self, request: Request) -> None:
        """Not used in offline scheduler - use schedule() instead."""
        logger.warning("Offline scheduler uses batch scheduling. Use schedule() method.")
        
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Not used in offline scheduler - use schedule() instead."""
        logger.warning("Offline scheduler uses batch scheduling. Use schedule() method.")
        return None
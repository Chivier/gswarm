"""
Static Deployment Scheduler for GSwarm Workflows

This scheduler implements a static deployment strategy where:
- Each GPU permanently hosts specific models (no switching)
- Models are grouped to minimize cross-server communication
- Workflows are scheduled to complete within a single server when possible
"""

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any

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
class StaticGPUState:
    """Extended GPU state for static deployment."""
    gpu_id: int
    server_id: int = 0
    assigned_models: Set[str] = field(default_factory=set)
    current_load: float = 0.0
    available_at: float = 0.0
    total_busy_time: float = 0.0
    execution_count: int = 0


@dataclass
class ModelAffinity:
    """Track model co-occurrence patterns."""
    model1: str
    model2: str
    frequency: int = 0
    total_time: float = 0.0


@dataclass
class ServerInfo:
    """Server configuration and state."""
    server_id: int
    gpu_ids: List[int]
    models: Set[str] = field(default_factory=set)

    def can_complete_workflow(self, workflow_models: Set[str]) -> bool:
        """Check if server has all models for a workflow."""
        return workflow_models.issubset(self.models)


@dataclass
class StaticTask:
    """Task representation for static scheduling."""
    request_id: str
    workflow_id: str
    node_id: str
    model_name: str
    dependencies: Set[str]
    estimated_time: float
    priority: int = 0
    server_id: Optional[int] = None
    
    @property
    def task_id(self) -> str:
        return f"{self.request_id}_{self.node_id}"


class StaticScheduler(SchedulerBase):
    """
    Static deployment scheduler with fixed model assignments.
    
    Key features:
    - Models are pre-assigned to GPUs and don't move
    - Workflows are assigned to servers that have all required models
    - Cross-server communication is minimized
    - No model switching overhead
    """
    
    def __init__(
        self,
        gpus: List[int],
        models: Dict[str, ModelInfo],
        servers: Optional[Dict[int, List[int]]] = None,
        model_assignments: Optional[Dict[str, List[int]]] = None,
        simulate: bool = False
    ):
        super().__init__(gpus, models, simulate)
        
        # Server configuration
        if servers is None:
            # Default: all GPUs in one server
            servers = {0: gpus}
        self.servers = {
            server_id: ServerInfo(server_id, gpu_list)
            for server_id, gpu_list in servers.items()
        }
        
        # GPU states
        self.gpu_states: Dict[int, StaticGPUState] = {}
        for server_id, server_info in self.servers.items():
            for gpu_id in server_info.gpu_ids:
                self.gpu_states[gpu_id] = StaticGPUState(
                    gpu_id=gpu_id,
                    server_id=server_id
                )
                
        # Model assignments
        if model_assignments:
            self._apply_model_assignments(model_assignments)
        else:
            # Auto-assign models based on workflow patterns
            self._auto_assign_models()
            
        # Task queues per server
        self.server_ready_queues: Dict[int, List[StaticTask]] = defaultdict(list)
        self.pending_tasks: Dict[str, StaticTask] = {}
        self.completed_tasks: Set[str] = set()
        
        # Affinity tracking
        self.model_affinities: Dict[Tuple[str, str], ModelAffinity] = {}
        
        # Cross-server communication tracking
        self.cross_server_transfers = 0
        
    def _apply_model_assignments(self, assignments: Dict[str, List[int]]) -> None:
        """Apply manual model-to-GPU assignments."""
        for model_name, gpu_list in assignments.items():
            for gpu_id in gpu_list:
                if gpu_id in self.gpu_states:
                    self.gpu_states[gpu_id].assigned_models.add(model_name)
                    
        # Update server model sets
        for server in self.servers.values():
            server.models.clear()
            for gpu_id in server.gpu_ids:
                server.models.update(self.gpu_states[gpu_id].assigned_models)
                
    def _auto_assign_models(self) -> None:
        """Automatically assign models to GPUs based on capacity and affinity."""
        # Simple round-robin assignment for now
        # In practice, this would analyze workflow patterns
        
        model_list = list(self.models.keys())
        gpu_list = list(self.gpus)
        
        if not model_list or not gpu_list:
            return
            
        # Distribute models evenly across GPUs
        for i, model_name in enumerate(model_list):
            gpu_id = gpu_list[i % len(gpu_list)]
            self.gpu_states[gpu_id].assigned_models.add(model_name)
            
        # Update server model sets
        for server in self.servers.values():
            server.models.clear()
            for gpu_id in server.gpu_ids:
                server.models.update(self.gpu_states[gpu_id].assigned_models)
                
        logger.info("Auto-assigned models to GPUs")
        
    def schedule(self, requests: List[Request]) -> List[ScheduledTask]:
        """Schedule requests with static model deployment."""
        logger.info(f"Static scheduling {len(requests)} requests")
        
        scheduled_tasks = []
        
        for request in requests:
            workflow = self.get_workflow(request.workflow_id)
            if not workflow:
                logger.error(f"Workflow {request.workflow_id} not found")
                continue
                
            # Determine which server can handle this workflow
            workflow_models = workflow.get_model_requirements()
            assigned_server = self._assign_workflow_to_server(workflow_models)
            
            if assigned_server is None:
                logger.warning(
                    f"No server can handle workflow {request.workflow_id} "
                    f"with models {workflow_models}"
                )
                continue
                
            # Create tasks for this workflow
            dependencies = workflow.get_dependencies()
            
            for node in workflow.nodes:
                task = StaticTask(
                    request_id=request.id,
                    workflow_id=workflow.id,
                    node_id=node.id,
                    model_name=node.model,
                    dependencies=dependencies.get(node.id, set()),
                    estimated_time=self._estimate_node_time(node.model),
                    priority=request.priority,
                    server_id=assigned_server,
                )
                
                task_id = task.task_id
                self.pending_tasks[task_id] = task
                
                # If no dependencies, add to server's ready queue
                if not task.dependencies:
                    heapq.heappush(
                        self.server_ready_queues[assigned_server],
                        (task.priority, task_id, task)
                    )
                    
        # Schedule tasks from ready queues
        for server_id in self.servers:
            server_tasks = self._schedule_server_tasks(server_id)
            scheduled_tasks.extend(server_tasks)
            
        return scheduled_tasks
        
    def _assign_workflow_to_server(self, workflow_models: Set[str]) -> Optional[int]:
        """Assign a workflow to a server that has all required models."""
        # First, try to find a server with all models
        for server in self.servers.values():
            if server.can_complete_workflow(workflow_models):
                return server.server_id
                
        # If no single server has all models, find the best server
        # (one with most models to minimize cross-server communication)
        best_server = None
        best_coverage = 0
        
        for server in self.servers.values():
            coverage = len(workflow_models.intersection(server.models))
            if coverage > best_coverage:
                best_coverage = coverage
                best_server = server.server_id
                
        if best_coverage > 0:
            self.cross_server_transfers += len(workflow_models) - best_coverage
            
        return best_server
        
    def _schedule_server_tasks(self, server_id: int) -> List[ScheduledTask]:
        """Schedule tasks for a specific server."""
        scheduled_tasks = []
        server = self.servers[server_id]
        ready_queue = self.server_ready_queues[server_id]
        
        # Get GPUs for this server
        server_gpus = {
            gpu_id: self.gpu_states[gpu_id]
            for gpu_id in server.gpu_ids
        }
        
        # Process ready tasks
        while ready_queue:
            _, task_id, task = heapq.heappop(ready_queue)
            
            if task_id in self.completed_tasks:
                continue
                
            # Find GPU with the model
            gpu_id = self._find_gpu_with_model(task.model_name, server_gpus)
            if gpu_id is None:
                logger.error(
                    f"Model {task.model_name} not found on server {server_id}"
                )
                continue
                
            gpu_state = server_gpus[gpu_id]
            
            # Schedule task (no switch time in static deployment)
            scheduled_time = gpu_state.available_at
            
            scheduled_task = ScheduledTask(
                request_id=task.request_id,
                workflow_id=task.workflow_id,
                node_id=task.node_id,
                model_name=task.model_name,
                gpu_id=gpu_id,
                scheduled_time=scheduled_time,
                estimated_duration=task.estimated_time,
                dependencies=task.dependencies,
            )
            
            scheduled_tasks.append(scheduled_task)
            
            # Update GPU state
            gpu_state.available_at = scheduled_time + task.estimated_time
            gpu_state.total_busy_time += task.estimated_time
            gpu_state.execution_count += 1
            
            # Mark task as scheduled
            self.completed_tasks.add(task_id)
            
            # Check dependencies for other tasks
            self._update_dependencies(task)
            
        return scheduled_tasks
        
    def _find_gpu_with_model(
        self, model_name: str, server_gpus: Dict[int, StaticGPUState]
    ) -> Optional[int]:
        """Find GPU with the specified model on the server."""
        available_gpus = [
            (gpu_state.available_at, gpu_id)
            for gpu_id, gpu_state in server_gpus.items()
            if model_name in gpu_state.assigned_models
        ]
        
        if not available_gpus:
            return None
            
        # Return GPU that will be available soonest
        available_gpus.sort()
        return available_gpus[0][1]
        
    def _update_dependencies(self, completed_task: StaticTask) -> None:
        """Update task dependencies after task completion."""
        workflow = self.get_workflow(completed_task.workflow_id)
        if not workflow:
            return
            
        dependents = workflow.get_dependents(completed_task.node_id)
        
        for dependent_node_id in dependents:
            dependent_task_id = f"{completed_task.request_id}_{dependent_node_id}"
            
            if dependent_task_id in self.pending_tasks:
                dependent_task = self.pending_tasks[dependent_task_id]
                
                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    f"{completed_task.request_id}_{dep}" in self.completed_tasks
                    for dep in dependent_task.dependencies
                )
                
                if deps_satisfied and dependent_task.server_id is not None:
                    # Add to server's ready queue
                    heapq.heappush(
                        self.server_ready_queues[dependent_task.server_id],
                        (dependent_task.priority, dependent_task_id, dependent_task)
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
        
    def add_request(self, request: Request) -> None:
        """Add request in static mode (delegates to schedule)."""
        self.schedule([request])
        
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Not typically used in static scheduler."""
        logger.warning("Static scheduler typically uses batch scheduling.")
        return None
        
    def get_model_assignments(self) -> Dict[str, List[int]]:
        """Get current model-to-GPU assignments."""
        assignments = defaultdict(list)
        for gpu_id, gpu_state in self.gpu_states.items():
            for model in gpu_state.assigned_models:
                assignments[model].append(gpu_id)
        return dict(assignments)
        
    def get_server_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about server configurations."""
        info = {}
        for server_id, server in self.servers.items():
            info[server_id] = {
                "gpu_ids": server.gpu_ids,
                "models": list(server.models),
                "gpu_count": len(server.gpu_ids),
                "model_count": len(server.models),
            }
        return info
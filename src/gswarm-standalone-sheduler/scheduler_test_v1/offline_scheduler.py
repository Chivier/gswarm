#!/usr/bin/env python3
"""
Offline Scheduler for GSwarm Workflows
Optimizes for minimal makespan using better load balancing and smaller batches.
"""

import argparse
import json
import yaml
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import sys
import numpy as np
import heapq
from threading import Thread, Lock
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("offline_scheduler.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
SERVER_URL = "http://localhost:8000"
PCIE_BANDWIDTH_GB_S = 16.0  # PCIe 4.0 x16 bandwidth in GB/s
MAX_BATCH_SIZE = 10  # Maximum nodes per batch to improve parallelism


@dataclass
class ModelInfo:
    """Model configuration information"""

    name: str
    memory_gb: float
    gpus_required: int
    load_time_seconds: float
    tokens_per_second: Optional[float] = None
    token_mean: Optional[float] = None
    token_std: Optional[float] = None
    inference_time_mean: Optional[float] = None
    inference_time_std: Optional[float] = None


@dataclass
class WorkflowNode:
    """Workflow node definition"""

    id: str
    model: str
    inputs: List[str]
    outputs: List[str]
    config_options: Optional[List[str]] = None


@dataclass
class WorkflowEdge:
    """Workflow edge definition"""

    from_node: str
    to_node: str


@dataclass
class Workflow:
    """Workflow definition"""

    id: str
    name: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]

    def get_dependencies(self) -> Dict[str, Set[str]]:
        """Get dependency map: node -> set of nodes it depends on"""
        deps = defaultdict(set)
        for edge in self.edges:
            deps[edge.to_node].add(edge.from_node)
        # Add nodes with no dependencies
        for node in self.nodes:
            if node.id not in deps:
                deps[node.id] = set()
        return dict(deps)

    def get_dependents(self) -> Dict[str, Set[str]]:
        """Get dependent map: node -> set of nodes that depend on it"""
        dependents = defaultdict(set)
        for edge in self.edges:
            dependents[edge.from_node].add(edge.to_node)
        return dict(dependents)

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


@dataclass
class Request:
    """Workflow request"""

    request_id: str
    timestamp: datetime
    workflow_id: str
    input_data: Dict[str, Any]
    node_configs: Dict[str, Dict[str, Any]]
    node_execution_times: Dict[str, float]


@dataclass
class NodeExecution:
    """Execution state for a node"""

    request_id: str
    workflow_id: str
    node_id: str
    model_name: str
    status: str = "pending"  # pending, ready, running, completed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    gpu_id: Optional[int] = None
    instance_id: Optional[str] = None
    estimated_time: Optional[float] = None
    dependencies_completed: Set[str] = field(default_factory=set)
    total_dependencies: int = 0
    level: int = 0  # Topological level for scheduling

    @property
    def execution_time(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def is_ready(self) -> bool:
        """Check if all dependencies are completed"""
        return len(self.dependencies_completed) == self.total_dependencies


@dataclass
class GPUState:
    """GPU state tracking"""

    gpu_id: int
    current_model: Optional[str] = None
    current_instance: Optional[str] = None
    busy: bool = False
    last_switch_time: float = 0.0
    available_at: float = 0.0  # Timestamp when GPU will be available
    total_busy_time: float = 0.0  # Total time GPU has been busy


@dataclass
class ScheduledTask:
    """A scheduled task on a GPU"""
    node: NodeExecution
    gpu_id: int
    start_time: float
    end_time: float
    model_name: str
    switch_time: float = 0.0


class OfflineScheduler:
    """Offline scheduler with improved load balancing"""

    def __init__(self, gpus: List[int], simulate: bool = False):
        self.gpus = gpus
        self.simulate = simulate
        self.mode = "offline"
        self.server_url = SERVER_URL

        # GPU states
        self.gpu_states = {gpu_id: GPUState(gpu_id) for gpu_id in gpus}

        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, Workflow] = {}

        # Execution tracking
        self.executions: Dict[str, NodeExecution] = {}  # node_key -> execution
        self.completed_executions: List[NodeExecution] = []
        self.scheduled_tasks: List[ScheduledTask] = []

        # Metrics
        self.request_start_times: Dict[str, float] = {}
        self.request_end_times: Dict[str, float] = {}
        self.model_switch_count = 0
        self.total_switch_time = 0.0

        # Track workflow dependencies
        self.workflow_dependencies: Dict[str, Dict[str, Set[str]]] = {}
        self.workflow_dependents: Dict[str, Dict[str, Set[str]]] = {}

    def load_config(self, config_path: Path):
        """Load system configuration"""
        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix == ".yaml":
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Load models
        for model_id, model_data in config["models"].items():
            self.models[model_id] = ModelInfo(
                name=model_data["name"],
                memory_gb=model_data["memory_gb"],
                gpus_required=model_data["gpus_required"],
                load_time_seconds=model_data["load_time_seconds"],
                tokens_per_second=model_data.get("tokens_per_second"),
                token_mean=model_data.get("token_mean"),
                token_std=model_data.get("token_std"),
                inference_time_mean=model_data.get("inference_time_mean"),
                inference_time_std=model_data.get("inference_time_std"),
            )

        # Load workflows
        for workflow_id, workflow_data in config["workflows"].items():
            nodes = []
            for node_data in workflow_data["nodes"]:
                nodes.append(
                    WorkflowNode(
                        id=node_data["id"],
                        model=node_data["model"],
                        inputs=node_data["inputs"],
                        outputs=node_data["outputs"],
                        config_options=node_data.get("config_options"),
                    )
                )

            edges = []
            for edge_data in workflow_data.get("edges", []):
                edges.append(WorkflowEdge(from_node=edge_data["from"], to_node=edge_data["to"]))

            workflow = Workflow(id=workflow_id, name=workflow_data["name"], nodes=nodes, edges=edges)
            self.workflows[workflow_id] = workflow
            
            # Cache dependencies and dependents
            self.workflow_dependencies[workflow_id] = workflow.get_dependencies()
            self.workflow_dependents[workflow_id] = workflow.get_dependents()

    def load_requests(self, requests_path: Path) -> List[Request]:
        """Load workflow requests"""
        logger.info(f"Loading requests from {requests_path}")

        with open(requests_path, "r") as f:
            if requests_path.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        requests = []
        for req_data in data["requests"]:
            requests.append(
                Request(
                    request_id=req_data["request_id"],
                    timestamp=datetime.fromisoformat(req_data["timestamp"]),
                    workflow_id=req_data["workflow_id"],
                    input_data=req_data["input_data"],
                    node_configs=req_data.get("node_configs", {}),
                    node_execution_times=req_data["node_execution_times"],
                )
            )

        return requests

    def _get_model_switch_time(self, from_model: Optional[str], to_model: str) -> float:
        """Calculate model switch time based on model size and PCIe bandwidth"""
        if from_model == to_model:
            return 0.0

        # If no previous model, just load time
        if not from_model:
            return self.models[to_model].load_time_seconds

        # Calculate switch time: unload old + load new
        from_size_gb = self.models[from_model].memory_gb
        to_size_gb = self.models[to_model].memory_gb

        # Assume we need to transfer both models over PCIe
        transfer_time = (from_size_gb + to_size_gb) / PCIE_BANDWIDTH_GB_S

        # Add some overhead
        overhead = 2.0  # seconds

        return transfer_time + overhead

    def _create_node_executions(self, requests: List[Request]) -> Dict[str, List[NodeExecution]]:
        """Create all node executions and group by model"""
        model_groups = defaultdict(list)
        
        for request in requests:
            workflow = self.workflows[request.workflow_id]
            dependencies = self.workflow_dependencies[request.workflow_id]
            
            # Create node executions
            request_executions = {}
            for node in workflow.nodes:
                node_key = f"{request.request_id}_{node.id}"
                node_exec = NodeExecution(
                    request_id=request.request_id, 
                    workflow_id=request.workflow_id, 
                    node_id=node.id, 
                    model_name=node.model,
                    total_dependencies=len(dependencies.get(node.id, set())),
                    estimated_time=request.node_execution_times.get(node.id, 10.0)
                )
                request_executions[node.id] = node_exec
                self.executions[node_key] = node_exec
                model_groups[node.model].append(node_exec)
            
            # Compute topological levels within this workflow
            self._compute_topological_levels(request_executions, dependencies)
            
        return dict(model_groups)

    def _compute_topological_levels(self, executions: Dict[str, NodeExecution], dependencies: Dict[str, Set[str]]):
        """Compute topological levels for nodes in a workflow"""
        # Initialize levels
        for node_exec in executions.values():
            node_exec.level = 0
        
        # BFS to compute levels
        changed = True
        while changed:
            changed = False
            for node_id, node_exec in executions.items():
                deps = dependencies.get(node_id, set())
                if deps:
                    max_dep_level = max(executions[dep_id].level for dep_id in deps if dep_id in executions)
                    new_level = max_dep_level + 1
                    if new_level > node_exec.level:
                        node_exec.level = new_level
                        changed = True

    def _get_node_ready_time(self, node: NodeExecution, completion_times: Dict[str, float]) -> float:
        """Get the earliest time a node can start based on its dependencies"""
        deps = self.workflow_dependencies[node.workflow_id].get(node.node_id, set())
        if not deps:
            return 0.0
        
        ready_time = 0.0
        for dep_id in deps:
            dep_key = f"{node.request_id}_{dep_id}"
            if dep_key in completion_times:
                ready_time = max(ready_time, completion_times[dep_key])
        
        return ready_time

    def _find_best_gpu_for_node(self, node: NodeExecution, ready_time: float, 
                               gpu_states: Dict[int, GPUState]) -> Tuple[int, float, float]:
        """
        Find the best GPU for a node considering load balancing.
        Returns: (gpu_id, start_time, switch_time)
        """
        model_info = self.models[node.model_name]
        required_gpus = model_info.gpus_required
        
        if required_gpus > 1:
            # For multi-GPU models, find consecutive available GPUs
            gpu_ids = sorted(gpu_states.keys())
            
            # Check if we have enough GPUs
            if len(gpu_ids) < required_gpus:
                raise ValueError(
                    f"Model '{node.model_name}' requires {required_gpus} consecutive GPUs, "
                    f"but only {len(gpu_ids)} GPUs are available. "
                    f"Please run with at least --gpus {required_gpus}"
                )
            
            best_start = float('inf')
            best_gpus = None
            
            for i in range(len(gpu_ids) - required_gpus + 1):
                consecutive_gpus = gpu_ids[i:i + required_gpus]
                # Find when all GPUs are available
                all_available = max(gpu_states[g].available_at for g in consecutive_gpus)
                start_time = max(all_available, ready_time)
                
                # Add switch time for primary GPU
                primary_gpu = consecutive_gpus[0]
                switch_time = self._get_model_switch_time(
                    gpu_states[primary_gpu].current_model, node.model_name
                )
                
                total_start = start_time + switch_time
                
                if total_start < best_start:
                    best_start = total_start
                    best_gpus = consecutive_gpus
            
            if best_gpus:
                return best_gpus[0], best_start - switch_time, switch_time
            else:
                # This should not happen if we have enough GPUs
                raise ValueError(
                    f"Cannot find {required_gpus} consecutive GPUs for model '{node.model_name}'. "
                    f"This should not happen if validation passed."
                )
        
        else:
            # Single GPU model - find best GPU with load balancing
            best_gpu = None
            best_score = float('inf')
            best_switch_time = 0
            
            for gpu_id, gpu_state in gpu_states.items():
                # Calculate when this GPU can start this task
                gpu_available = gpu_state.available_at
                earliest_start = max(gpu_available, ready_time)
                
                # Calculate switch time
                switch_time = 0.0
                if gpu_state.current_model != node.model_name:
                    switch_time = self._get_model_switch_time(gpu_state.current_model, node.model_name)
                
                # Calculate a score that considers:
                # 1. When the task would complete (primary factor)
                # 2. GPU load balancing (secondary factor)
                task_completion = earliest_start + switch_time + node.estimated_time
                load_factor = gpu_state.total_busy_time / 1000.0  # Scale down for weighting
                
                # Score: earlier completion is better, less loaded GPU is better
                score = task_completion + load_factor * 0.1  # Small weight for load balancing
                
                if best_gpu is None or score < best_score:
                    best_gpu = gpu_id
                    best_score = score
                    best_switch_time = switch_time
            
            gpu_available = gpu_states[best_gpu].available_at
            start_time = max(gpu_available, ready_time)
            
            return best_gpu, start_time, best_switch_time

    def _schedule_with_load_balancing(self, model_groups: Dict[str, List[NodeExecution]]):
        """Schedule nodes using a priority queue for better load balancing"""
        # Track completion times for dependency resolution
        completion_times = {}
        
        # Create a priority queue of ready nodes
        ready_queue = []  # heap of (score, node)
        pending_nodes = []  # nodes waiting for dependencies - changed from set to list
        
        # Initialize with nodes that have no dependencies
        for model_name, nodes in model_groups.items():
            for node in nodes:
                if node.total_dependencies == 0:
                    # Score: level (lower is better) + small random factor for tie-breaking
                    score = node.level + np.random.random() * 0.01
                    heapq.heappush(ready_queue, (score, id(node), node))
                else:
                    pending_nodes.append(node)  # changed from add to append
        
        # Process nodes
        scheduled_count = 0
        total_nodes = sum(len(nodes) for nodes in model_groups.values())
        
        while ready_queue or pending_nodes:
            if not ready_queue and pending_nodes:
                # Check if any pending nodes are now ready
                newly_ready = []
                for node in pending_nodes:
                    deps = self.workflow_dependencies[node.workflow_id].get(node.node_id, set())
                    all_deps_complete = True
                    for dep_id in deps:
                        dep_key = f"{node.request_id}_{dep_id}"
                        if dep_key not in completion_times:
                            all_deps_complete = False
                            break
                    
                    if all_deps_complete:
                        newly_ready.append(node)
                
                # Move newly ready nodes to ready queue
                for node in newly_ready:
                    pending_nodes.remove(node)
                    score = node.level + np.random.random() * 0.01
                    heapq.heappush(ready_queue, (score, id(node), node))
                
                if not ready_queue:
                    logger.error("Deadlock detected: nodes waiting for dependencies that will never complete")
                    break
            
            if not ready_queue:
                break
            
            # Get next node to schedule
            _, _, node = heapq.heappop(ready_queue)
            
            # Find when this node can start (dependencies)
            ready_time = self._get_node_ready_time(node, completion_times)
            
            # Find best GPU
            gpu_id, start_time, switch_time = self._find_best_gpu_for_node(
                node, ready_time, self.gpu_states
            )
            
            # Schedule the node
            end_time = start_time + switch_time + node.estimated_time
            
            # Create scheduled task
            task = ScheduledTask(
                node=node,
                gpu_id=gpu_id,
                start_time=start_time,
                end_time=end_time,
                model_name=node.model_name,
                switch_time=switch_time
            )
            self.scheduled_tasks.append(task)
            
            # Update node execution info
            node.gpu_id = gpu_id
            node.start_time = start_time + switch_time
            node.end_time = end_time
            node.status = "scheduled"
            
            # Update GPU state
            gpu_state = self.gpu_states[gpu_id]
            gpu_state.available_at = end_time
            gpu_state.total_busy_time += (end_time - start_time)
            if gpu_state.current_model != node.model_name:
                gpu_state.current_model = node.model_name
                if switch_time > 0:
                    self.model_switch_count += 1
                    self.total_switch_time += switch_time
            
            # Update completion time
            node_key = f"{node.request_id}_{node.node_id}"
            completion_times[node_key] = end_time
            
            # Check for newly ready nodes
            if node.workflow_id in self.workflow_dependents:
                dependents = self.workflow_dependents[node.workflow_id].get(node.node_id, set())
                for dep_node_id in dependents:
                    dep_key = f"{node.request_id}_{dep_node_id}"
                    if dep_key in self.executions:
                        dep_node = self.executions[dep_key]
                        if dep_node in pending_nodes:
                            # Check if all dependencies are now complete
                            deps = self.workflow_dependencies[dep_node.workflow_id].get(dep_node_id, set())
                            all_complete = True
                            for d in deps:
                                d_key = f"{node.request_id}_{d}"
                                if d_key not in completion_times:
                                    all_complete = False
                                    break
                            
                            if all_complete:
                                pending_nodes.remove(dep_node)
                                score = dep_node.level + np.random.random() * 0.01
                                heapq.heappush(ready_queue, (score, id(dep_node), dep_node))
            
            scheduled_count += 1
            if scheduled_count % 100 == 0:
                logger.info(f"Scheduled {scheduled_count}/{total_nodes} nodes...")
        
        logger.info(f"Scheduled {scheduled_count} nodes total")

    def _execute_schedule(self):
        """Execute the computed schedule"""
        # Sort tasks by start time
        self.scheduled_tasks.sort(key=lambda t: t.start_time)
        
        # Track request start/end times
        request_first_start = {}
        request_last_end = {}
        
        # Process each scheduled task
        for task in self.scheduled_tasks:
            node = task.node
            
            # Track request timing
            req_id = node.request_id
            if req_id not in request_first_start:
                request_first_start[req_id] = node.start_time
                self.request_start_times[req_id] = node.start_time
            
            request_last_end[req_id] = node.end_time
            
            # Mark as completed
            node.status = "completed"
            self.completed_executions.append(node)
            
            # Log execution
            logger.debug(
                f"[{node.start_time:.2f}s] Node {node.node_id} of request {node.request_id} "
                f"on GPU {node.gpu_id} (end: {node.end_time:.2f}s)"
            )
        
        # Update request completion times
        for req_id, end_time in request_last_end.items():
            self.request_end_times[req_id] = end_time

    def run(self, requests: List[Request]):
        """Run the scheduler on a list of requests"""
        logger.info(f"Starting offline scheduler with {len(requests)} requests")
        logger.info(f"Mode: offline (load-balanced), Simulate: {self.simulate}")
        logger.info(f"Available GPUs: {self.gpus}")

        # Store all requests for reference
        self.all_requests = requests

        # Phase 1: Create node executions grouped by model
        logger.info("Phase 1: Creating node executions...")
        model_groups = self._create_node_executions(requests)
        total_nodes = sum(len(nodes) for nodes in model_groups.values())
        logger.info(f"Created {total_nodes} node executions across {len(model_groups)} models")
        
        # Validate GPU requirements
        max_gpus_required = 0
        models_requiring_too_many_gpus = []
        for model_name, nodes in model_groups.items():
            if model_name in self.models:
                gpus_required = self.models[model_name].gpus_required
                max_gpus_required = max(max_gpus_required, gpus_required)
                if gpus_required > len(self.gpus):
                    models_requiring_too_many_gpus.append((model_name, gpus_required))
        
        if models_requiring_too_many_gpus:
            error_msg = f"Insufficient GPUs available. Have {len(self.gpus)} GPUs, but the following models require more:\n"
            for model_name, required in models_requiring_too_many_gpus:
                error_msg += f"  - {model_name}: requires {required} GPUs\n"
            error_msg += f"Please run with at least --gpus {max_gpus_required}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log model distribution
        for model, nodes in model_groups.items():
            model_info = self.models.get(model)
            gpu_req = model_info.gpus_required if model_info else 1
            logger.info(f"  {model}: {len(nodes)} nodes (requires {gpu_req} GPU{'s' if gpu_req > 1 else ''})")

        # Phase 2: Schedule with load balancing
        logger.info("Phase 2: Scheduling with load balancing...")
        self._schedule_with_load_balancing(model_groups)

        # Phase 3: Execute schedule
        logger.info("Phase 3: Executing schedule...")
        self._execute_schedule()

        # Print metrics
        self._print_metrics()

    def _print_metrics(self):
        """Print execution metrics"""
        logger.info("\n" + "=" * 60)
        logger.info("EXECUTION METRICS")
        logger.info("=" * 60)

        # Total execution time (makespan)
        if self.completed_executions:
            makespan = max(e.end_time for e in self.completed_executions)
            logger.info(f"Total execution time (makespan): {makespan:.2f} seconds")

        # Model switching metrics
        logger.info(f"\nModel switching metrics:")
        logger.info(f"  Total model switches: {self.model_switch_count}")
        logger.info(f"  Total switch time: {self.total_switch_time:.2f} seconds")
        if self.model_switch_count > 0:
            logger.info(f"  Average switch time: {self.total_switch_time / self.model_switch_count:.2f} seconds")

        # Request metrics
        if self.request_end_times:
            request_times = []
            for req_id, end_time in self.request_end_times.items():
                start_time = self.request_start_times[req_id]
                request_times.append(end_time - start_time)

            logger.info(f"\nRequest completion times:")
            logger.info(f"  Average: {np.mean(request_times):.2f}s")
            logger.info(f"  Median: {np.median(request_times):.2f}s")
            logger.info(f"  P99: {np.percentile(request_times, 99):.2f}s")
            logger.info(f"  Min: {np.min(request_times):.2f}s")
            logger.info(f"  Max: {np.max(request_times):.2f}s")

        # GPU utilization
        logger.info(f"\nGPU utilization:")
        gpu_exec_counts = defaultdict(int)
        gpu_busy_times = defaultdict(float)
        
        for task in self.scheduled_tasks:
            gpu_exec_counts[task.gpu_id] += 1
            gpu_busy_times[task.gpu_id] += (task.end_time - task.start_time)
        
        for gpu_id in sorted(self.gpus):
            exec_count = gpu_exec_counts[gpu_id]
            busy_time = gpu_busy_times[gpu_id]
            utilization = busy_time / makespan * 100 if makespan > 0 else 0
            
            logger.info(
                f"  GPU {gpu_id}: {exec_count} executions, "
                f"{busy_time:.2f}s busy time ({utilization:.1f}% utilization)"
            )

        # Model statistics
        logger.info(f"\nModel execution counts:")
        model_counts = defaultdict(int)
        for exec in self.completed_executions:
            model_counts[exec.model_name] += 1
        for model, count in sorted(model_counts.items()):
            logger.info(f"  {model}: {count} executions")

        # Write detailed log
        self._write_detailed_log()

    def _write_detailed_log(self):
        """Write detailed execution log"""
        log_file = Path("offline_execution_log.json")

        log_data = {
            "summary": {
                "total_requests": len(self.request_end_times),
                "total_nodes_executed": len(self.completed_executions),
                "mode": self.mode,
                "simulate": self.simulate,
                "gpus": self.gpus,
                "makespan": max(e.end_time for e in self.completed_executions) if self.completed_executions else 0,
                "total_model_switches": self.model_switch_count,
                "total_switch_time": self.total_switch_time,
            },
            "executions": [],
        }

        for exec in sorted(self.completed_executions, key=lambda e: e.start_time):
            exec_data = {
                "request_id": exec.request_id,
                "workflow_id": exec.workflow_id,
                "node_id": exec.node_id,
                "model_name": exec.model_name,
                "gpu_id": exec.gpu_id,
                "start_time": exec.start_time,
                "end_time": exec.end_time,
                "execution_time": exec.execution_time,
                "estimated_time": exec.estimated_time,
                "topological_level": exec.level,
            }
            log_data["executions"].append(exec_data)

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"\nDetailed execution log written to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Offline scheduler for GSwarm workflows")
    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
        help="Number of available GPUs (e.g., 2 for GPUs 0 and 1, 3 for GPUs 0, 1, and 2)",
    )
    parser.add_argument(
        "--simulate",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use simulation mode with actual model calls (true/false)",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("system_config.yaml"), help="Path to system configuration file"
    )
    parser.add_argument(
        "--requests", type=Path, default=Path("workflow_requests.yaml"), help="Path to workflow requests file"
    )

    args = parser.parse_args()

    # Generate GPU list from number of GPUs
    gpu_list = list(range(args.gpus))

    # Create scheduler
    scheduler = OfflineScheduler(gpus=gpu_list, simulate=args.simulate)

    # Load configuration
    scheduler.load_config(args.config)

    # Load requests
    requests = scheduler.load_requests(args.requests)

    # Run scheduler
    scheduler.run(requests)


if __name__ == "__main__":
    main()
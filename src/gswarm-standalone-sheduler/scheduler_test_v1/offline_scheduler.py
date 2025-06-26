#!/usr/bin/env python3
"""
Offline Scheduler for GSwarm Workflows
Simple batch scheduling that respects topology and minimizes model switches
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
    estimated_time: float
    dependencies: Set[str] = field(default_factory=set)
    level: int = 0  # Topological level
    
    # Execution tracking
    status: str = "pending"  # pending, ready, scheduled, completed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    gpu_id: Optional[int] = None
    
    @property
    def node_key(self) -> str:
        return f"{self.request_id}_{self.node_id}"


@dataclass
class GPUState:
    """GPU state tracking"""

    gpu_id: int
    current_model: Optional[str] = None
    available_at: float = 0.0
    total_busy_time: float = 0.0
    execution_count: int = 0


@dataclass
class ScheduledTask:
    """A scheduled task"""
    
    node: NodeExecution
    gpu_id: int
    start_time: float
    end_time: float
    switch_time: float = 0.0


class OfflineScheduler:
    """Simple offline scheduler that batches by model and uses all GPUs together"""

    def __init__(self, gpus: List[int], simulate: bool = False):
        self.gpus = gpus
        self.simulate = simulate
        self.mode = "offline"
        
        # GPU states
        self.gpu_states = {gpu_id: GPUState(gpu_id) for gpu_id in gpus}
        
        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, Workflow] = {}
        
        # Queue of ready nodes grouped by model
        self.ready_queue: Dict[str, List[NodeExecution]] = defaultdict(list)
        self.pending_nodes: List[NodeExecution] = []
        self.completed_nodes: Set[str] = set()
        
        # Scheduled tasks
        self.scheduled_tasks: List[ScheduledTask] = []
        
        # Metrics
        self.model_switch_count = 0
        self.total_switch_time = 0.0
        
        # Current loaded model across all GPUs
        self.current_model: Optional[str] = None

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

            workflow = Workflow(
                id=workflow_id, 
                name=workflow_data["name"], 
                nodes=nodes, 
                edges=edges
            )
            self.workflows[workflow_id] = workflow

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
        """Calculate model switch time"""
        if from_model == to_model:
            return 0.0

        if not from_model:
            return self.models[to_model].load_time_seconds

        # Calculate switch time: unload old + load new
        from_size_gb = self.models[from_model].memory_gb
        to_size_gb = self.models[to_model].memory_gb

        # Transfer time over PCIe
        transfer_time = (from_size_gb + to_size_gb) / PCIE_BANDWIDTH_GB_S
        overhead = 2.0  # seconds

        return transfer_time + overhead

    def _create_all_nodes(self, requests: List[Request]) -> None:
        """Create all node executions from requests"""
        self.pending_nodes.clear()
        self.ready_queue.clear()
        
        for request in requests:
            workflow = self.workflows[request.workflow_id]
            dependencies = workflow.get_dependencies()
            
            # Create nodes for this request
            for node in workflow.nodes:
                node_exec = NodeExecution(
                    request_id=request.request_id,
                    workflow_id=request.workflow_id,
                    node_id=node.id,
                    model_name=node.model,
                    estimated_time=request.node_execution_times.get(node.id, 10.0),
                    dependencies={f"{request.request_id}_{dep}" for dep in dependencies.get(node.id, set())}
                )
                
                # If node has no dependencies, it's ready
                if not node_exec.dependencies:
                    self.ready_queue[node_exec.model_name].append(node_exec)
                else:
                    self.pending_nodes.append(node_exec)

    def _update_ready_queue(self):
        """Move nodes from pending to ready queue when dependencies are satisfied"""
        newly_ready = []
        remaining_pending = []
        
        for node in self.pending_nodes:
            if all(dep in self.completed_nodes for dep in node.dependencies):
                newly_ready.append(node)
            else:
                remaining_pending.append(node)
        
        self.pending_nodes = remaining_pending
        
        # Add newly ready nodes to ready queue
        for node in newly_ready:
            self.ready_queue[node.model_name].append(node)

    def _select_next_model(self) -> Optional[str]:
        """Select next model to process based on ready queue"""
        if not self.ready_queue:
            return None
        
        # Prefer current model if it has ready nodes
        if self.current_model and self.current_model in self.ready_queue and self.ready_queue[self.current_model]:
            return self.current_model
        
        # Otherwise, select model with most ready nodes
        best_model = max(self.ready_queue.keys(), key=lambda m: len(self.ready_queue[m]))
        return best_model if self.ready_queue[best_model] else None

    def _schedule_batch(self, model_name: str, current_time: float) -> float:
        """Schedule a batch of nodes with the same model across all GPUs"""
        nodes = self.ready_queue[model_name]
        if not nodes:
            return current_time
        
        model_info = self.models[model_name]
        
        # Handle model switch
        switch_time = self._get_model_switch_time(self.current_model, model_name)
        if switch_time > 0:
            self.model_switch_count += 1
            self.total_switch_time += switch_time
            current_time += switch_time
            self.current_model = model_name
            
            # Update all GPU models
            for gpu_id in self.gpus:
                self.gpu_states[gpu_id].current_model = model_name
        
        # Schedule nodes in batch
        batch_end_time = current_time
        
        if model_info.gpus_required > 1:
            # Multi-GPU model: process sequentially
            num_gpus = len(self.gpus)
            gpus_per_task = model_info.gpus_required
            concurrent_tasks = num_gpus // gpus_per_task
            
            task_idx = 0
            while task_idx < len(nodes) and nodes:
                batch_start = current_time
                
                # Schedule up to concurrent_tasks in parallel
                for i in range(concurrent_tasks):
                    if task_idx >= len(nodes):
                        break
                    
                    node = nodes[task_idx]
                    gpu_start_idx = i * gpus_per_task
                    
                    # Schedule node
                    node.status = "scheduled"
                    node.gpu_id = self.gpus[gpu_start_idx]  # Primary GPU
                    node.start_time = batch_start
                    node.end_time = batch_start + node.estimated_time
                    
                    # Update GPU states for all required GPUs
                    for j in range(gpus_per_task):
                        gpu_id = self.gpus[gpu_start_idx + j]
                        self.gpu_states[gpu_id].total_busy_time += node.estimated_time
                        self.gpu_states[gpu_id].execution_count += 1
                    
                    # Create scheduled task
                    task = ScheduledTask(
                        node=node,
                        gpu_id=node.gpu_id,
                        start_time=node.start_time,
                        end_time=node.end_time,
                        switch_time=switch_time if task_idx == 0 else 0.0
                    )
                    self.scheduled_tasks.append(task)
                    
                    task_idx += 1
                
                # Update current time to when this sub-batch finishes
                if task_idx > 0:
                    current_time = batch_start + nodes[task_idx-1].estimated_time
                    batch_end_time = current_time
        
        else:
            # Single GPU model: true parallel execution
            # All GPUs execute different nodes in parallel
            gpu_idx = 0
            batch_nodes = []
            
            while nodes and gpu_idx < len(self.gpus):
                node = nodes.pop(0)
                gpu_id = self.gpus[gpu_idx]
                
                # Schedule node
                node.status = "scheduled"
                node.gpu_id = gpu_id
                node.start_time = current_time
                node.end_time = current_time + node.estimated_time
                
                # Update GPU state
                self.gpu_states[gpu_id].total_busy_time += node.estimated_time
                self.gpu_states[gpu_id].execution_count += 1
                
                # Create scheduled task
                task = ScheduledTask(
                    node=node,
                    gpu_id=node.gpu_id,
                    start_time=node.start_time,
                    end_time=node.end_time,
                    switch_time=switch_time if gpu_idx == 0 else 0.0
                )
                self.scheduled_tasks.append(task)
                
                batch_nodes.append(node)
                gpu_idx += 1
                
                # Mark as completed
                node.status = "completed"
                self.completed_nodes.add(node.node_key)
            
            # If we have more nodes than GPUs, continue in batches
            if nodes:
                # Find max execution time in this batch
                if batch_nodes:
                    batch_end_time = max(n.end_time for n in batch_nodes)
                    return self._schedule_batch(model_name, batch_end_time)
            else:
                # All done with this model
                if batch_nodes:
                    batch_end_time = max(n.end_time for n in batch_nodes)
        
        # Mark all scheduled nodes as completed
        for task in self.scheduled_tasks:
            if task.node.status == "scheduled":
                task.node.status = "completed"
                self.completed_nodes.add(task.node.node_key)
        
        # Clear processed nodes from ready queue
        self.ready_queue[model_name] = nodes
        
        return batch_end_time

    def run(self, requests: List[Request]):
        """Run the scheduler on requests"""
        logger.info(f"Starting offline scheduler with {len(requests)} requests")
        logger.info(f"Available GPUs: {self.gpus}")
        
        # Step 1: Create all nodes
        self._create_all_nodes(requests)
        
        total_nodes = sum(len(nodes) for nodes in self.ready_queue.values()) + len(self.pending_nodes)
        logger.info(f"Created {total_nodes} total nodes to schedule")
        
        # Step 2: Process nodes in batches
        current_time = 0.0
        
        while self.ready_queue or self.pending_nodes:
            # Update ready queue with newly ready nodes
            self._update_ready_queue()
            
            # Select next model to process
            next_model = self._select_next_model()
            
            if not next_model:
                if self.pending_nodes:
                    logger.error("Deadlock detected: nodes pending but none ready")
                    break
                else:
                    break
            
            # Schedule batch of same model
            logger.info(f"Scheduling batch of model {next_model} with {len(self.ready_queue[next_model])} nodes")
            current_time = self._schedule_batch(next_model, current_time)
        
        # Print final metrics
        self._print_metrics()

    def _print_metrics(self):
        """Print execution metrics"""
        logger.info("\n" + "=" * 60)
        logger.info("SCHEDULING METRICS")
        logger.info("=" * 60)
        
        if not self.scheduled_tasks:
            logger.warning("No tasks scheduled")
            return
        
        # Calculate makespan
        makespan = max(task.end_time for task in self.scheduled_tasks)
        total_tasks = len(self.scheduled_tasks)
        
        logger.info(f"Total execution time: {makespan:.2f} seconds")
        logger.info(f"Total tasks: {total_tasks}")
        logger.info(f"Average throughput: {total_tasks / makespan:.2f} tasks/second")
        
        # Model switching
        logger.info(f"\nModel switching:")
        logger.info(f"  Total switches: {self.model_switch_count}")
        logger.info(f"  Total switch time: {self.total_switch_time:.2f} seconds")
        logger.info(f"  Switch overhead: {self.total_switch_time / makespan * 100:.1f}%")
        
        # GPU utilization and execution counts
        logger.info(f"\nGPU utilization:")
        for gpu_id, gpu_state in sorted(self.gpu_states.items()):
            utilization = gpu_state.total_busy_time / makespan * 100 if makespan > 0 else 0
            logger.info(f"  GPU {gpu_id}: {utilization:.1f}% utilization, {gpu_state.execution_count} executions")
        
        # Model statistics
        model_stats = defaultdict(lambda: {"count": 0, "time": 0})
        for task in self.scheduled_tasks:
            model_name = task.node.model_name
            model_stats[model_name]["count"] += 1
            model_stats[model_name]["time"] += task.node.estimated_time
        
        logger.info(f"\nModel statistics:")
        for model, stats in sorted(model_stats.items()):
            logger.info(f"  {model}: {stats['count']} tasks, {stats['time']:.2f}s total time")
        
        # Write detailed log
        self._write_detailed_log()

    def _write_detailed_log(self):
        """Write detailed execution log"""
        log_file = Path("offline_execution_log.json")
        
        # Calculate request-level metrics
        request_times = {}
        for task in self.scheduled_tasks:
            req_id = task.node.request_id
            if req_id not in request_times:
                request_times[req_id] = {"start": float('inf'), "end": 0}
            request_times[req_id]["start"] = min(request_times[req_id]["start"], task.start_time)
            request_times[req_id]["end"] = max(request_times[req_id]["end"], task.end_time)
        
        log_data = {
            "summary": {
                "total_requests": len(request_times),
                "total_nodes_executed": len(self.scheduled_tasks),
                "mode": self.mode,
                "simulate": self.simulate,
                "gpus": self.gpus,
                "makespan": max(t.end_time for t in self.scheduled_tasks) if self.scheduled_tasks else 0,
                "total_model_switches": self.model_switch_count,
                "total_switch_time": self.total_switch_time,
            },
            "executions": []
        }
        
        for task in sorted(self.scheduled_tasks, key=lambda t: t.start_time):
            log_data["executions"].append({
                "request_id": task.node.request_id,
                "workflow_id": task.node.workflow_id,
                "node_id": task.node.node_id,
                "model_name": task.node.model_name,
                "gpu_id": task.gpu_id,
                "start_time": task.start_time,
                "end_time": task.end_time,
                "execution_time": task.node.estimated_time,
                "estimated_time": task.node.estimated_time,
            })
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"\nDetailed log written to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Offline scheduler for GSwarm workflows")
    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
        help="Number of available GPUs",
    )
    parser.add_argument(
        "--simulate",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use simulation mode (true/false)",
    )
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("system_config.yaml"), 
        help="Path to system configuration file"
    )
    parser.add_argument(
        "--requests", 
        type=Path, 
        default=Path("workflow_requests.yaml"), 
        help="Path to workflow requests file"
    )

    args = parser.parse_args()

    # Generate GPU list
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
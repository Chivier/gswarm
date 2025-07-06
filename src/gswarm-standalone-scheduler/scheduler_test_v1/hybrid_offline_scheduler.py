#!/usr/bin/env python3
"""
Hybrid Offline Scheduler for AI Workflows
Combines batch processing (to minimize model switches) with dynamic GPU scheduling
"""

import json
import yaml
import time
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import heapq
import logging
import argparse
from pathlib import Path
import sys

from scheduler_component import ModelInfo, WorkflowNode, NodeExecution, GPUState, ScheduledTask


@dataclass
class Task:
    """Represents a single task to be executed"""

    workflow_id: str
    node_id: str
    model_type: str
    dependencies: Set[str] = field(default_factory=set)
    ready_time: float = 0.0
    priority: int = 0  # Lower value = higher priority
    estimated_time: float = 0.0
    request_id: str = ""

    def __lt__(self, other):
        return (self.priority, self.ready_time, self.workflow_id, self.node_id) < (
            other.priority,
            other.ready_time,
            other.workflow_id,
            other.node_id,
        )


@dataclass
class GPUAvailability:
    """Track when a GPU becomes available"""

    gpu_id: int
    available_at: float
    current_model: Optional[str] = None

    def __lt__(self, other):
        return self.available_at < other.available_at


@dataclass
class ModelBatch:
    """A batch of tasks using the same model"""

    model_type: str
    tasks: List[Task]
    total_execution_time: float = 0.0
    priority: float = 0.0  # Average priority


@dataclass
class WorkflowDAG:
    """Represents a workflow as a directed acyclic graph"""

    workflow_id: str
    nodes: Dict[str, WorkflowNode]
    edges: List[Tuple[str, str]]  # (from_node, to_node)
    topological_order: List[str] = field(default_factory=list)

    def compute_topological_order(self):
        """Compute topological ordering of nodes"""
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)

        for from_node, to_node in self.edges:
            adj_list[from_node].append(to_node)
            in_degree[to_node] += 1

        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        self.topological_order = []

        while queue:
            node_id = queue.popleft()
            self.topological_order.append(node_id)

            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get all dependencies for a given node"""
        deps = set()
        for from_node, to_node in self.edges:
            if to_node == node_id:
                deps.add(from_node)
        return deps


class HybridOfflineScheduler:
    """Hybrid scheduler combining batch optimization with dynamic GPU allocation"""

    def __init__(self, gpus: List[int], simulate: bool = False):
        self.gpus = gpus
        self.num_gpus = len(gpus)
        self.simulate = simulate
        self.mode = "hybrid_offline"
        self.logger = self._setup_logger()

        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, WorkflowDAG] = {}
        self.workflow_requests: List[Dict] = []

        # Scheduling state
        self.gpu_availability = [GPUAvailability(gpu_id=i, available_at=0.0) for i in gpus]
        self.scheduled_tasks: List[ScheduledTask] = []
        self.completed_tasks: Set[Tuple[str, str]] = set()

        # Model batches for optimization
        self.model_batches: Dict[str, ModelBatch] = {}
        self.ready_batches: List[ModelBatch] = []  # Priority queue

        # Metrics
        self.model_switch_count = 0
        self.total_switch_time = 0.0
        self.current_time = 0.0

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("HybridOfflineScheduler")
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler("hybrid_offline_scheduler.log", mode="w")
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def load_config(self, config_path: Path):
        """Load model and workflow configurations"""
        with open(config_path, "r") as f:
            if config_path.suffix == ".yaml":
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Load models
        for model_id, model_data in config["models"].items():
            model_info_data = {
                "name": model_data["name"],
                "memory_gb": model_data["memory_gb"],
                "gpus_required": model_data["gpus_required"],
                "load_time_seconds": model_data["load_time_seconds"],
            }
            # Add optional fields
            for field in ["tokens_per_second", "token_mean", "token_std", "inference_time_mean", "inference_time_std"]:
                if field in model_data:
                    model_info_data[field] = model_data[field]

            self.models[model_id] = ModelInfo(**model_info_data)

        # Load workflows
        for workflow_id, workflow_data in config["workflows"].items():
            dag = WorkflowDAG(
                workflow_id=workflow_id,
                nodes={node["id"]: WorkflowNode(**node) for node in workflow_data["nodes"]},
                edges=[(e["from"], e["to"]) for e in workflow_data.get("edges", [])],
            )
            dag.compute_topological_order()
            self.workflows[workflow_id] = dag

        self.logger.info(f"Loaded {len(self.models)} models and {len(self.workflows)} workflows")

    def load_requests(self, requests_path: Path) -> List[Dict]:
        """Load workflow requests"""
        with open(requests_path, "r") as f:
            if requests_path.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        self.workflow_requests = data["requests"]
        self.logger.info(f"Loaded {len(self.workflow_requests)} workflow requests")
        return self.workflow_requests

    def parse_workflows(self) -> List[Task]:
        """Parse all workflow requests into tasks"""
        all_tasks = []

        for request in self.workflow_requests:
            workflow_id = request["workflow_id"]
            request_id = request["request_id"]
            timestamp = datetime.fromisoformat(request["timestamp"]).timestamp()
            node_execution_times = request.get("node_execution_times", {})

            if workflow_id not in self.workflows:
                self.logger.warning(f"Unknown workflow: {workflow_id}")
                continue

            dag = self.workflows[workflow_id]

            for node_id in dag.topological_order:
                node = dag.nodes[node_id]

                # Get estimated execution time
                if node_id in node_execution_times:
                    estimated_time = node_execution_times[node_id]
                else:
                    model = self.models[node.model]
                    tokens = 1000  # Default
                    tokens_per_second = model.tokens_per_second if model.tokens_per_second else 100
                    estimated_time = tokens / tokens_per_second

                task = Task(
                    workflow_id=workflow_id,
                    node_id=node_id,
                    model_type=node.model,
                    dependencies={f"{request_id}_{dep}" for dep in dag.get_dependencies(node_id)},
                    ready_time=timestamp,
                    estimated_time=estimated_time,
                    request_id=request_id,
                )
                all_tasks.append(task)

        self.logger.info(f"Parsed {len(all_tasks)} tasks from workflows")
        return all_tasks

    def create_model_batches(self, tasks: List[Task]):
        """Group tasks by model type into batches"""
        self.model_batches.clear()

        for task in tasks:
            if task.model_type not in self.model_batches:
                self.model_batches[task.model_type] = ModelBatch(model_type=task.model_type, tasks=[])

            batch = self.model_batches[task.model_type]
            batch.tasks.append(task)
            batch.total_execution_time += task.estimated_time

        # Calculate batch priorities (lower is better)
        for batch in self.model_batches.values():
            # Priority based on: number of tasks, total execution time, and model size
            model = self.models[batch.model_type]
            batch.priority = -(len(batch.tasks) * 1000 + batch.total_execution_time - model.load_time_seconds * 10)

        self.logger.info(f"Created {len(self.model_batches)} model batches")

    def _get_model_switch_time(self, from_model: Optional[str], to_model: str) -> float:
        """Calculate model switch time"""
        if from_model == to_model:
            return 0.0

        if not from_model:
            return self.models[to_model].load_time_seconds

        # Unload old + load new
        return self.models[from_model].memory_gb / 16.0 + self.models[to_model].load_time_seconds

    def find_available_gpus(self, model_name: str, current_time: float) -> Optional[List[int]]:
        """Find available GPUs for a model at the given time"""
        model = self.models[model_name]
        required_gpus = model.gpus_required

        # Sort GPUs by availability time
        sorted_gpus = sorted(self.gpu_availability, key=lambda g: g.available_at)

        if required_gpus == 1:
            # For single GPU, find the earliest available
            return [sorted_gpus[0].gpu_id]
        else:
            # For multi-GPU, find the best consecutive set
            best_set = None
            best_available_time = float("inf")

            for i in range(self.num_gpus - required_gpus + 1):
                # Check consecutive GPUs
                gpu_set = []
                max_available = 0.0

                for j in range(required_gpus):
                    gpu = next((g for g in sorted_gpus if g.gpu_id == i + j), None)
                    if gpu:
                        gpu_set.append(gpu.gpu_id)
                        max_available = max(max_available, gpu.available_at)

                if len(gpu_set) == required_gpus and max_available < best_available_time:
                    best_set = gpu_set
                    best_available_time = max_available

            return best_set

    def schedule_task(self, task: Task, gpu_ids: List[int], start_time: float):
        """Schedule a task on the given GPUs"""
        model = self.models[task.model_type]
        primary_gpu = gpu_ids[0]

        # Check if we need to switch models
        gpu_info = next(g for g in self.gpu_availability if g.gpu_id == primary_gpu)
        switch_time = 0.0

        if gpu_info.current_model != task.model_type:
            switch_time = self._get_model_switch_time(gpu_info.current_model, task.model_type)
            self.model_switch_count += 1
            self.total_switch_time += switch_time

        # Create execution record
        actual_start = start_time + switch_time
        end_time = actual_start + task.estimated_time

        node_exec = NodeExecution(
            request_id=task.request_id,
            workflow_id=task.workflow_id,
            node_id=task.node_id,
            model_name=task.model_type,
            status="scheduled",
            gpu_id=primary_gpu,
            start_time=actual_start,
            end_time=end_time,
            estimated_time=task.estimated_time,
        )

        scheduled_task = ScheduledTask(
            node=node_exec, gpu_id=primary_gpu, start_time=actual_start, end_time=end_time, switch_time=switch_time
        )
        self.scheduled_tasks.append(scheduled_task)

        # Update GPU availability for all GPUs used
        for gpu_id in gpu_ids:
            gpu_info = next(g for g in self.gpu_availability if g.gpu_id == gpu_id)
            gpu_info.available_at = end_time
            gpu_info.current_model = task.model_type

        self.completed_tasks.add((task.request_id, task.node_id))

    def can_schedule_task(self, task: Task) -> bool:
        """Check if all dependencies are satisfied"""
        for dep in task.dependencies:
            if dep not in self.completed_tasks:
                return False
        return True

    def schedule_batch(self, batch: ModelBatch):
        """Schedule all ready tasks from a batch"""
        scheduled_count = 0
        remaining_tasks = []

        # Sort tasks by dependencies and priority
        sorted_tasks = sorted(batch.tasks, key=lambda t: (len(t.dependencies), t.priority))

        for task in sorted_tasks:
            if self.can_schedule_task(task):
                # Find available GPUs
                gpu_ids = self.find_available_gpus(task.model_type, self.current_time)

                if gpu_ids:
                    # Get the actual start time (when all required GPUs are available)
                    start_time = max(g.available_at for g in self.gpu_availability if g.gpu_id in gpu_ids)

                    # Schedule the task
                    self.schedule_task(task, gpu_ids, start_time)
                    scheduled_count += 1

                    # Update current time
                    self.current_time = max(self.current_time, start_time)
            else:
                remaining_tasks.append(task)

        # Update batch with remaining tasks
        batch.tasks = remaining_tasks

        return scheduled_count

    def run(self, requests: List[Dict]):
        """Execute the hybrid offline batch processing"""
        self.logger.info("Starting hybrid offline batch processing")
        self.logger.info(f"Available GPUs: {self.gpus}")

        # Step 1: Parse workflows into tasks
        all_tasks = self.parse_workflows()

        # Step 2: Create model batches
        self.create_model_batches(all_tasks)

        # Step 3: Process batches in priority order
        total_scheduled = 0
        iterations = 0
        max_iterations = len(all_tasks) * 2  # Safety limit

        while total_scheduled < len(all_tasks) and iterations < max_iterations:
            iterations += 1
            scheduled_in_round = 0

            # Sort batches by priority and readiness
            batch_list = []
            for model_type, batch in self.model_batches.items():
                if batch.tasks:
                    # Count ready tasks
                    ready_count = sum(1 for t in batch.tasks if self.can_schedule_task(t))
                    if ready_count > 0:
                        # Adjust priority based on ready tasks and GPU availability
                        min_gpu_available = min(g.available_at for g in self.gpu_availability)
                        adjusted_priority = batch.priority - ready_count * 1000 + min_gpu_available
                        batch_list.append((adjusted_priority, batch))

            # Sort by adjusted priority
            batch_list.sort(key=lambda x: x[0])

            # Try to schedule from each batch
            for _, batch in batch_list:
                scheduled = self.schedule_batch(batch)
                scheduled_in_round += scheduled
                total_scheduled += scheduled

            # If nothing was scheduled, advance time to next GPU availability
            if scheduled_in_round == 0 and total_scheduled < len(all_tasks):
                next_available = min(g.available_at for g in self.gpu_availability)
                self.current_time = next_available

            self.logger.info(
                f"Iteration {iterations}: Scheduled {scheduled_in_round} tasks, "
                f"total: {total_scheduled}/{len(all_tasks)}"
            )

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Print results
        self._print_metrics(metrics)

        # Save execution log
        self._save_execution_log(metrics)

        return metrics

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.scheduled_tasks:
            return {}

        makespan = max(task.end_time for task in self.scheduled_tasks)

        # Calculate GPU utilization
        gpu_busy_time = defaultdict(float)
        gpu_task_count = defaultdict(int)

        for task in self.scheduled_tasks:
            gpu_busy_time[task.gpu_id] += task.node.estimated_time
            gpu_task_count[task.gpu_id] += 1

        metrics = {
            "total_tasks": len(self.scheduled_tasks),
            "total_model_switches": self.model_switch_count,
            "total_switch_time": self.total_switch_time,
            "estimated_makespan": makespan,
            "average_throughput": len(self.scheduled_tasks) / makespan if makespan > 0 else 0,
            "switch_overhead": self.total_switch_time / makespan * 100 if makespan > 0 else 0,
            "gpu_utilization": {},
            "gpu_task_count": gpu_task_count,
        }

        for gpu_id in self.gpus:
            busy_time = gpu_busy_time.get(gpu_id, 0)
            metrics["gpu_utilization"][gpu_id] = (busy_time / makespan * 100) if makespan > 0 else 0

        return metrics

    def _print_metrics(self, metrics: Dict):
        """Print execution metrics"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SCHEDULING METRICS")
        self.logger.info("=" * 60)

        if not metrics:
            self.logger.warning("No tasks scheduled")
            return

        self.logger.info(f"Total execution time: {metrics['estimated_makespan']:.2f} seconds")
        self.logger.info(f"Total tasks: {metrics['total_tasks']}")
        self.logger.info(f"Average throughput: {metrics['average_throughput']:.2f} tasks/second")

        self.logger.info(f"\nModel switching:")
        self.logger.info(f"  Total switches: {metrics['total_model_switches']}")
        self.logger.info(f"  Total switch time: {metrics['total_switch_time']:.2f} seconds")
        self.logger.info(f"  Switch overhead: {metrics['switch_overhead']:.1f}%")

        self.logger.info(f"\nGPU utilization:")
        for gpu_id in self.gpus:
            utilization = metrics["gpu_utilization"].get(gpu_id, 0)
            task_count = metrics["gpu_task_count"].get(gpu_id, 0)
            self.logger.info(f"  GPU {gpu_id}: {utilization:.1f}% utilization, {task_count} executions")

    def _save_execution_log(self, metrics: Dict):
        """Save detailed execution log"""
        request_times = {}
        for task in self.scheduled_tasks:
            req_id = task.node.request_id
            if req_id not in request_times:
                request_times[req_id] = {"start": float("inf"), "end": 0}
            request_times[req_id]["start"] = min(request_times[req_id]["start"], task.start_time)
            request_times[req_id]["end"] = max(request_times[req_id]["end"], task.end_time)

        log_data = {
            "summary": {
                "total_requests": len(request_times),
                "total_nodes_executed": len(self.scheduled_tasks),
                "mode": self.mode,
                "simulate": self.simulate,
                "gpus": self.gpus,
                "makespan": metrics.get("estimated_makespan", 0),
                "total_model_switches": metrics.get("total_model_switches", 0),
                "total_switch_time": metrics.get("total_switch_time", 0),
            },
            "executions": [],
        }

        for task in sorted(self.scheduled_tasks, key=lambda t: t.start_time):
            log_data["executions"].append(
                {
                    "request_id": task.node.request_id,
                    "workflow_id": task.node.workflow_id,
                    "node_id": task.node.node_id,
                    "model_name": task.node.model_name,
                    "gpu_id": task.gpu_id,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                    "execution_time": task.node.estimated_time,
                    "estimated_time": task.node.estimated_time,
                }
            )

        with open("hybrid_offline_execution_log.json", "w") as f:
            json.dump(log_data, f, indent=2)

        self.logger.info(f"\nDetailed log written to hybrid_offline_execution_log.json")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Offline Batch Processing Scheduler")
    parser.add_argument("--gpus", type=int, required=True, help="Number of available GPUs")
    parser.add_argument("--simulate", type=lambda x: x.lower() == "true", default=False, help="Use simulation mode")
    parser.add_argument("--config", type=Path, default=Path("system_config.yaml"), help="Configuration file")
    parser.add_argument("--requests", type=Path, default=Path("workflow_requests.yaml"), help="Requests file")

    args = parser.parse_args()

    gpu_list = list(range(args.gpus))

    scheduler = HybridOfflineScheduler(gpus=gpu_list, simulate=args.simulate)
    scheduler.load_config(args.config)
    scheduler.load_requests(args.requests)
    scheduler.run(scheduler.workflow_requests)


if __name__ == "__main__":
    main()

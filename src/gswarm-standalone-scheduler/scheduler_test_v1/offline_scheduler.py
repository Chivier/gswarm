#!/usr/bin/env python3
"""
Offline Batch Processing Scheduler for AI Workflows
Implements optimized scheduling to minimize model switching overhead while maintaining dependencies
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
import numpy as np

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
        # For heap queue - prioritize by (priority, ready_time, workflow_id, node_id)
        return (self.priority, self.ready_time, self.workflow_id, self.node_id) < (
            other.priority,
            other.ready_time,
            other.workflow_id,
            other.node_id,
        )


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
        """Get all dependencies for a given node"""
        deps = set()
        for from_node, to_node in self.edges:
            if to_node == node_id:
                deps.add(from_node)
        return deps


class OptimizedOfflineScheduler:
    """Optimized offline batch processing scheduler"""

    def __init__(self, gpus: List[int], simulate: bool = False):
        self.gpus = gpus
        self.num_gpus = len(gpus)
        self.simulate = simulate
        self.mode = "offline"
        self.logger = self._setup_logger()

        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, WorkflowDAG] = {}
        self.workflow_requests: List[Dict] = []

        # Scheduling state
        self.gpu_states: List[GPUState] = [GPUState(gpu_id=i) for i in gpus]
        self.task_queue: List[Task] = []  # Priority queue
        self.completed_tasks: Set[Tuple[str, str]] = set()  # (request_id, node_id)
        self.scheduled_tasks: List[ScheduledTask] = []

        # Metrics
        self.model_switch_count = 0
        self.total_switch_time = 0.0
        self.gpu_utilization: Dict[int, float] = {i: 0.0 for i in gpus}
        self.task_waiting_times: List[float] = []
        self.task_response_times: List[float] = []
        self.request_response_times: Dict[str, float] = {}

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("OptimizedOfflineScheduler")
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler("offline_scheduler.log", mode="w")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
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
            # Extract only the fields that ModelInfo expects
            model_info_data = {
                "name": model_data["name"],
                "memory_gb": model_data["memory_gb"],
                "gpus_required": model_data["gpus_required"],
                "load_time_seconds": model_data["load_time_seconds"],
            }
            # Add optional fields if present
            if "tokens_per_second" in model_data:
                model_info_data["tokens_per_second"] = model_data["tokens_per_second"]
            if "token_mean" in model_data:
                model_info_data["token_mean"] = model_data["token_mean"]
            if "token_std" in model_data:
                model_info_data["token_std"] = model_data["token_std"]
            if "inference_time_mean" in model_data:
                model_info_data["inference_time_mean"] = model_data["inference_time_mean"]
            if "inference_time_std" in model_data:
                model_info_data["inference_time_std"] = model_data["inference_time_std"]

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

        # Normalize timestamps to start from 0
        first_timestamp = None
        if self.workflow_requests:
            first_timestamp = datetime.fromisoformat(self.workflow_requests[0]["timestamp"]).timestamp()

        for request in self.workflow_requests:
            workflow_id = request["workflow_id"]
            request_id = request["request_id"]
            timestamp = datetime.fromisoformat(request["timestamp"]).timestamp()
            # Normalize timestamp
            if first_timestamp:
                timestamp = timestamp - first_timestamp
            node_execution_times = request.get("node_execution_times", {})

            if workflow_id not in self.workflows:
                self.logger.warning(f"Unknown workflow: {workflow_id}")
                continue

            dag = self.workflows[workflow_id]

            # Create tasks for each node in the workflow
            for node_id in dag.topological_order:
                node = dag.nodes[node_id]

                # Get estimated execution time
                if node_id in node_execution_times:
                    estimated_time = node_execution_times[node_id]
                else:
                    # Estimate based on model performance
                    model = self.models[node.model]
                    tokens = 1000  # Default token count
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

    def topological_sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks topologically while respecting dependencies"""
        # Build dependency graph
        task_map = {f"{t.request_id}_{t.node_id}": t for t in tasks}
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)

        for task in tasks:
            task_key = f"{task.request_id}_{task.node_id}"
            for dep in task.dependencies:
                if dep in task_map:
                    adj_list[dep].append(task_key)
                    in_degree[task_key] += 1

        # Topological sort
        queue = deque([key for key in task_map if in_degree[key] == 0])
        sorted_tasks = []

        while queue:
            task_key = queue.popleft()
            sorted_tasks.append(task_map[task_key])

            for neighbor in adj_list[task_key]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return sorted_tasks

    def greedy_optimize_tasks(self, tasks: List[Task]) -> List[Task]:
        """Optimize task order to minimize model switches while respecting dependencies"""
        # Group tasks by model type
        model_groups = defaultdict(list)
        for task in tasks:
            model_groups[task.model_type].append(task)

        # Build dependency graph
        task_map = {f"{t.request_id}_{t.node_id}": t for t in tasks}
        dependencies = defaultdict(set)
        dependents = defaultdict(set)

        for task in tasks:
            task_key = f"{task.request_id}_{task.node_id}"
            for dep in task.dependencies:
                if dep in task_map:
                    dependencies[task_key].add(dep)
                    dependents[dep].add(task_key)

        # Greedy scheduling
        scheduled = []
        scheduled_set = set()
        ready_tasks = []

        # Initialize with tasks that have no dependencies
        for task in tasks:
            task_key = f"{task.request_id}_{task.node_id}"
            if not dependencies[task_key]:
                heapq.heappush(ready_tasks, (task.model_type, task.ready_time, task))

        current_model = None

        while ready_tasks:
            # Try to pick a task with the same model as current
            best_task = None
            best_idx = -1

            for idx, (model_type, _, task) in enumerate(ready_tasks):
                if model_type == current_model:
                    best_task = task
                    best_idx = idx
                    break

            # If no same model found, pick the earliest ready task
            if best_task is None:
                _, _, best_task = heapq.heappop(ready_tasks)
            else:
                # Remove the selected task from heap
                ready_tasks.pop(best_idx)
                heapq.heapify(ready_tasks)

            # Schedule the task
            scheduled.append(best_task)
            scheduled_set.add(f"{best_task.request_id}_{best_task.node_id}")

            if current_model != best_task.model_type:
                if current_model is not None:
                    self.model_switch_count += 1
                current_model = best_task.model_type

            # Update ready tasks
            task_key = f"{best_task.request_id}_{best_task.node_id}"
            for dependent in dependents[task_key]:
                # Check if all dependencies are satisfied
                if all(dep in scheduled_set for dep in dependencies[dependent]):
                    dep_task = task_map[dependent]
                    heapq.heappush(ready_tasks, (dep_task.model_type, dep_task.ready_time, dep_task))

        return scheduled

    def _get_model_switch_time(self, from_model: Optional[str], to_model: str) -> float:
        """Calculate model switch time"""
        if from_model == to_model:
            return 0.0

        if not from_model:
            return self.models[to_model].load_time_seconds

        # Unload old + load new
        return self.models[from_model].memory_gb / 16.0 + self.models[to_model].load_time_seconds

    def allocate_tasks_to_gpus(self, tasks: List[Task]) -> Dict[int, List[Task]]:
        """Allocate tasks to GPUs using load balancing"""
        gpu_tasks = defaultdict(list)
        gpu_load = {i: 0.0 for i in self.gpus}
        gpu_current_model = {i: None for i in self.gpus}

        for task in tasks:
            model = self.models[task.model_type]

            if model.gpus_required > 1:
                # Multi-GPU model - allocate to consecutive GPUs
                best_gpu_set = None
                min_load = float("inf")

                for start_gpu in range(self.num_gpus - model.gpus_required + 1):
                    gpu_set = self.gpus[start_gpu : start_gpu + model.gpus_required]
                    total_load = sum(gpu_load[g] for g in gpu_set)

                    # Add switch penalty if models don't match
                    switch_penalty = sum(
                        self._get_model_switch_time(gpu_current_model[g], task.model_type) for g in gpu_set
                    )

                    if total_load + switch_penalty < min_load:
                        min_load = total_load + switch_penalty
                        best_gpu_set = gpu_set

                # Allocate to primary GPU only (task will use multiple GPUs during execution)
                primary_gpu = best_gpu_set[0]
                gpu_tasks[primary_gpu].append(task)

                # Update load for all GPUs in the set
                for gpu in best_gpu_set:
                    gpu_load[gpu] += task.estimated_time
                    if gpu_current_model[gpu] != task.model_type:
                        gpu_load[gpu] += self._get_model_switch_time(gpu_current_model[gpu], task.model_type)
                        gpu_current_model[gpu] = task.model_type

            else:
                # Single-GPU model - find best GPU
                best_gpu = min(
                    self.gpus,
                    key=lambda g: gpu_load[g] + self._get_model_switch_time(gpu_current_model[g], task.model_type),
                )

                gpu_tasks[best_gpu].append(task)
                gpu_load[best_gpu] += task.estimated_time
                if gpu_current_model[best_gpu] != task.model_type:
                    gpu_load[best_gpu] += self._get_model_switch_time(gpu_current_model[best_gpu], task.model_type)
                    gpu_current_model[best_gpu] = task.model_type

        return dict(gpu_tasks)

    def execute_schedule(self, gpu_allocation: Dict[int, List[Task]]):
        """Execute the scheduled tasks and create ScheduledTask objects"""
        # Track request start/end times
        request_times = {}

        for gpu_id, tasks in gpu_allocation.items():
            current_time = 0.0
            current_model = None

            for task in tasks:
                # Handle model switch
                switch_time = 0.0
                if current_model != task.model_type:
                    switch_time = self._get_model_switch_time(current_model, task.model_type)
                    current_time += switch_time
                    self.total_switch_time += switch_time
                    current_model = task.model_type
                    self.gpu_states[gpu_id].current_model = current_model

                # Calculate waiting time (from ready time to start time)
                waiting_time = current_time - task.ready_time
                self.task_waiting_times.append(waiting_time)

                # Create NodeExecution
                node_exec = NodeExecution(
                    request_id=task.request_id,
                    workflow_id=task.workflow_id,
                    node_id=task.node_id,
                    model_name=task.model_type,
                    status="scheduled",
                    gpu_id=gpu_id,
                    start_time=current_time,
                    end_time=current_time + task.estimated_time,
                    estimated_time=task.estimated_time,
                )

                # Create ScheduledTask
                scheduled_task = ScheduledTask(
                    node=node_exec,
                    gpu_id=gpu_id,
                    start_time=current_time,
                    end_time=current_time + task.estimated_time,
                    switch_time=switch_time,
                )
                self.scheduled_tasks.append(scheduled_task)

                # Track request times
                if task.request_id not in request_times:
                    request_times[task.request_id] = {"start": float("inf"), "end": 0}
                request_times[task.request_id]["start"] = min(request_times[task.request_id]["start"], task.ready_time)
                request_times[task.request_id]["end"] = max(
                    request_times[task.request_id]["end"], current_time + task.estimated_time
                )

                # Calculate task response time
                response_time = (current_time + task.estimated_time) - task.ready_time
                self.task_response_times.append(response_time)

                # Update GPU state
                self.gpu_states[gpu_id].total_busy_time += task.estimated_time
                self.gpu_states[gpu_id].execution_count += 1

                # Update current time
                current_time += task.estimated_time

        # Calculate request response times
        for request_id, times in request_times.items():
            self.request_response_times[request_id] = times["end"] - times["start"]

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.scheduled_tasks:
            return {}

        makespan = max(task.end_time for task in self.scheduled_tasks)

        # Calculate P99 and average metrics
        avg_waiting_time = np.mean(self.task_waiting_times) if self.task_waiting_times else 0.0
        p99_waiting_time = np.percentile(self.task_waiting_times, 99) if self.task_waiting_times else 0.0
        avg_response_time = np.mean(self.task_response_times) if self.task_response_times else 0.0
        p99_response_time = np.percentile(self.task_response_times, 99) if self.task_response_times else 0.0

        request_response_list = list(self.request_response_times.values())
        avg_request_response_time = np.mean(request_response_list) if request_response_list else 0.0
        p99_request_response_time = np.percentile(request_response_list, 99) if request_response_list else 0.0

        metrics = {
            "total_tasks": len(self.scheduled_tasks),
            "total_requests": len(self.request_response_times),
            "completed_requests": len(self.request_response_times),
            "total_model_switches": self.model_switch_count,
            "total_switch_time": self.total_switch_time,
            "estimated_makespan": makespan,
            "average_throughput": len(self.scheduled_tasks) / makespan if makespan > 0 else 0,
            "switch_overhead": self.total_switch_time / makespan * 100 if makespan > 0 else 0,
            # Task-level metrics
            "avg_waiting_time": avg_waiting_time,
            "p99_waiting_time": p99_waiting_time,
            "avg_response_time": avg_response_time,
            "p99_response_time": p99_response_time,
            # Request-level metrics
            "avg_request_response_time": avg_request_response_time,
            "p99_request_response_time": p99_request_response_time,
            "gpu_utilization": {},
            "gpu_execution_count": {},
        }

        for gpu_id, gpu_state in enumerate(self.gpu_states):
            metrics["gpu_utilization"][gpu_id] = (gpu_state.total_busy_time / makespan * 100) if makespan > 0 else 0
            metrics["gpu_execution_count"][gpu_id] = gpu_state.execution_count

        return metrics

    def run(self, requests: List[Dict]):
        """Execute the offline batch processing"""
        self.logger.info("Starting optimized offline batch processing")
        self.logger.info(f"Available GPUs: {self.gpus}")

        # Step 1: Parse workflows into tasks
        all_tasks = self.parse_workflows()

        # Step 2: Topological sort
        sorted_tasks = self.topological_sort_tasks(all_tasks)
        self.logger.info(f"Topologically sorted {len(sorted_tasks)} tasks")

        # Step 3: Greedy optimization
        optimized_tasks = self.greedy_optimize_tasks(sorted_tasks)
        self.logger.info(f"Optimized task order to minimize model switches")

        # Step 4: Allocate to GPUs
        gpu_allocation = self.allocate_tasks_to_gpus(optimized_tasks)

        # Step 5: Execute schedule
        self.execute_schedule(gpu_allocation)

        # Step 6: Calculate metrics
        metrics = self.calculate_metrics()

        # Print results
        self._print_metrics(metrics)

        # Save execution log
        self._save_execution_log(metrics)

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
        self.logger.info(
            f"Total requests: {metrics.get('total_requests', 'N/A')} (completed: {metrics.get('completed_requests', 'N/A')})"
        )
        self.logger.info(f"Total tasks: {metrics['total_tasks']}")
        self.logger.info(f"Average throughput: {metrics['average_throughput']:.2f} tasks/second")

        self.logger.info(f"\nLatency metrics:")
        self.logger.info(f"  Task-level:")
        self.logger.info(f"    Average waiting time: {metrics['avg_waiting_time']:.2f} seconds")
        self.logger.info(f"    P99 waiting time: {metrics['p99_waiting_time']:.2f} seconds")
        self.logger.info(f"    Average response time: {metrics['avg_response_time']:.2f} seconds")
        self.logger.info(f"    P99 response time: {metrics['p99_response_time']:.2f} seconds")
        self.logger.info(f"  Request-level:")
        self.logger.info(f"    Average response time: {metrics['avg_request_response_time']:.2f} seconds")
        self.logger.info(f"    P99 response time: {metrics['p99_request_response_time']:.2f} seconds")

        self.logger.info(f"\nModel switching:")
        self.logger.info(f"  Total switches: {metrics['total_model_switches']}")
        self.logger.info(f"  Total switch time: {metrics['total_switch_time']:.2f} seconds")
        self.logger.info(f"  Switch overhead: {metrics['switch_overhead']:.1f}%")

        self.logger.info(f"\nGPU utilization:")
        for gpu_id in self.gpus:
            utilization = metrics["gpu_utilization"].get(gpu_id, 0)
            exec_count = metrics["gpu_execution_count"].get(gpu_id, 0)
            self.logger.info(f"  GPU {gpu_id}: {utilization:.1f}% utilization, {exec_count} executions")

    def _save_execution_log(self, metrics: Dict):
        """Save detailed execution log"""
        # Calculate request-level metrics
        request_times = {}
        for task in self.scheduled_tasks:
            req_id = task.node.request_id
            if req_id not in request_times:
                request_times[req_id] = {"start": float("inf"), "end": 0}
            request_times[req_id]["start"] = min(request_times[req_id]["start"], task.start_time)
            request_times[req_id]["end"] = max(request_times[req_id]["end"], task.end_time)

        log_data = {
            "summary": {
                "total_requests": metrics.get("total_requests", len(request_times)),
                "completed_requests": metrics.get("completed_requests", len(request_times)),
                "total_nodes_executed": len(self.scheduled_tasks),
                "mode": self.mode,
                "simulate": self.simulate,
                "gpus": self.gpus,
                "makespan": metrics.get("estimated_makespan", 0),
                "total_model_switches": metrics.get("total_model_switches", 0),
                "total_switch_time": metrics.get("total_switch_time", 0),
                "avg_waiting_time": metrics.get("avg_waiting_time", 0),
                "p99_waiting_time": metrics.get("p99_waiting_time", 0),
                "avg_response_time": metrics.get("avg_response_time", 0),
                "p99_response_time": metrics.get("p99_response_time", 0),
                "avg_request_response_time": metrics.get("avg_request_response_time", 0),
                "p99_request_response_time": metrics.get("p99_request_response_time", 0),
            },
            "executions": [],
        }

        # Create a map of task to waiting time
        task_to_waiting_time = {}
        for i, task in enumerate(self.scheduled_tasks):
            if i < len(self.task_waiting_times):
                task_to_waiting_time[f"{task.node.request_id}_{task.node.node_id}"] = self.task_waiting_times[i]

        for task in sorted(self.scheduled_tasks, key=lambda t: t.start_time):
            task_key = f"{task.node.request_id}_{task.node.node_id}"
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
                    "switch_time": task.switch_time,
                    "waiting_time": task_to_waiting_time.get(task_key, 0.0),
                }
            )

        with open("offline_execution_log.json", "w") as f:
            json.dump(log_data, f, indent=2)

        self.logger.info(f"\nDetailed log written to offline_execution_log.json")


def main():
    parser = argparse.ArgumentParser(description="Optimized Offline Batch Processing Scheduler")
    parser.add_argument("--gpus", type=int, required=True, help="Number of available GPUs")
    parser.add_argument(
        "--simulate", type=lambda x: x.lower() == "true", default=False, help="Use simulation mode (true/false)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("system_config.yaml"), help="Path to system configuration file"
    )
    parser.add_argument(
        "--requests", type=Path, default=Path("workflow_requests.yaml"), help="Path to workflow requests file"
    )

    args = parser.parse_args()

    # Generate GPU list
    gpu_list = list(range(args.gpus))

    # Create scheduler
    scheduler = OptimizedOfflineScheduler(gpus=gpu_list, simulate=args.simulate)

    # Load configuration
    scheduler.load_config(args.config)

    # Load requests
    requests = scheduler.load_requests(args.requests)

    # Run scheduler
    scheduler.run(requests)


if __name__ == "__main__":
    main()

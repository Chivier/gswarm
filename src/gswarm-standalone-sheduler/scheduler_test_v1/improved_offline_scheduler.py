#!/usr/bin/env python3
"""
Improved Offline Scheduler - Event-driven with batch optimization
Minimizes model switches while maintaining good GPU utilization
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

from scheduler_component import (
    ModelInfo, WorkflowNode, NodeExecution, GPUState, ScheduledTask
)


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
    
    @property
    def task_id(self) -> str:
        return f"{self.request_id}_{self.node_id}"
    
    def __lt__(self, other):
        return (self.priority, self.ready_time, self.workflow_id, self.node_id) < \
               (other.priority, other.ready_time, other.workflow_id, other.node_id)


@dataclass
class Event:
    """Event for discrete event simulation"""
    timestamp: float
    event_type: str  # "task_complete", "gpu_available"
    data: Dict
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


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
    
    def get_dependents(self, node_id: str) -> Set[str]:
        """Get all nodes that depend on this node"""
        deps = set()
        for from_node, to_node in self.edges:
            if from_node == node_id:
                deps.add(to_node)
        return deps


class ImprovedOfflineScheduler:
    """Event-driven scheduler with batch optimization"""
    
    def __init__(self, gpus: List[int], simulate: bool = False):
        self.gpus = gpus
        self.num_gpus = len(gpus)
        self.simulate = simulate
        self.mode = "improved_offline"
        self.logger = self._setup_logger()
        
        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, WorkflowDAG] = {}
        self.workflow_requests: List[Dict] = []
        
        # GPU states
        self.gpu_states = {gpu_id: GPUState(gpu_id) for gpu_id in gpus}
        self.gpu_available_at = {gpu_id: 0.0 for gpu_id in gpus}
        
        # Task management
        self.all_tasks: Dict[str, Task] = {}  # task_id -> Task
        self.ready_queue: Dict[str, List[Task]] = defaultdict(list)  # model -> [tasks]
        self.completed_tasks: Set[str] = set()
        self.scheduled_tasks: List[ScheduledTask] = []
        self.pending_dependencies: Dict[str, Set[str]] = {}  # task_id -> set of dependencies
        
        # Event queue
        self.event_queue: List[Event] = []
        self.current_time = 0.0
        
        # Metrics
        self.model_switch_count = 0
        self.total_switch_time = 0.0
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ImprovedOfflineScheduler')
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler('improved_offline_scheduler.log', mode='w')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_config(self, config_path: Path):
        """Load model and workflow configurations"""
        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Load models
        for model_id, model_data in config['models'].items():
            model_info_data = {
                'name': model_data['name'],
                'memory_gb': model_data['memory_gb'],
                'gpus_required': model_data['gpus_required'],
                'load_time_seconds': model_data['load_time_seconds']
            }
            # Add optional fields
            for field in ['tokens_per_second', 'token_mean', 'token_std', 
                         'inference_time_mean', 'inference_time_std']:
                if field in model_data:
                    model_info_data[field] = model_data[field]
            
            self.models[model_id] = ModelInfo(**model_info_data)
        
        # Load workflows
        for workflow_id, workflow_data in config['workflows'].items():
            dag = WorkflowDAG(
                workflow_id=workflow_id,
                nodes={node['id']: WorkflowNode(**node) for node in workflow_data['nodes']},
                edges=[(e['from'], e['to']) for e in workflow_data.get('edges', [])]
            )
            dag.compute_topological_order()
            self.workflows[workflow_id] = dag
        
        self.logger.info(f"Loaded {len(self.models)} models and {len(self.workflows)} workflows")
    
    def load_requests(self, requests_path: Path) -> List[Dict]:
        """Load workflow requests"""
        with open(requests_path, 'r') as f:
            if requests_path.suffix == '.yaml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
            
        self.workflow_requests = data['requests']
        self.logger.info(f"Loaded {len(self.workflow_requests)} workflow requests")
        return self.workflow_requests
    
    def parse_workflows(self):
        """Parse all workflow requests and initialize tasks"""
        for request in self.workflow_requests:
            workflow_id = request['workflow_id']
            request_id = request['request_id']
            timestamp = datetime.fromisoformat(request['timestamp']).timestamp()
            node_execution_times = request.get('node_execution_times', {})
            
            if workflow_id not in self.workflows:
                self.logger.warning(f"Unknown workflow: {workflow_id}")
                continue
            
            dag = self.workflows[workflow_id]
            
            # Create tasks for each node
            for node_id in dag.nodes:
                node = dag.nodes[node_id]
                
                # Get estimated execution time
                if node_id in node_execution_times:
                    estimated_time = node_execution_times[node_id]
                else:
                    model = self.models[node.model]
                    tokens = 1000  # Default
                    tokens_per_second = model.tokens_per_second if model.tokens_per_second else 100
                    estimated_time = tokens / tokens_per_second
                
                # Create task
                task = Task(
                    workflow_id=workflow_id,
                    node_id=node_id,
                    model_type=node.model,
                    dependencies={f"{request_id}_{dep}" for dep in dag.get_dependencies(node_id)},
                    ready_time=timestamp,
                    estimated_time=estimated_time,
                    request_id=request_id
                )
                
                self.all_tasks[task.task_id] = task
                self.pending_dependencies[task.task_id] = task.dependencies.copy()
                
                # If no dependencies, add to ready queue
                if not task.dependencies:
                    self.ready_queue[task.model_type].append(task)
        
        self.logger.info(f"Parsed {len(self.all_tasks)} tasks from workflows")
    
    def _get_model_switch_time(self, from_model: Optional[str], to_model: str) -> float:
        """Calculate model switch time"""
        if from_model == to_model:
            return 0.0
        
        if not from_model:
            return self.models[to_model].load_time_seconds
        
        # Unload old + load new
        return self.models[from_model].memory_gb / 16.0 + self.models[to_model].load_time_seconds
    
    def find_best_gpu_for_task(self, task: Task) -> Optional[Tuple[int, float]]:
        """Find the best GPU(s) for a task, returns (primary_gpu_id, available_time)"""
        model = self.models[task.model_type]
        required_gpus = model.gpus_required
        
        if required_gpus == 1:
            # Single GPU - find earliest available with preference for same model
            best_gpu = None
            best_time = float('inf')
            best_switch_time = float('inf')
            
            for gpu_id in self.gpus:
                available_time = self.gpu_available_at[gpu_id]
                current_model = self.gpu_states[gpu_id].current_model
                switch_time = self._get_model_switch_time(current_model, task.model_type)
                
                # Prefer GPU with same model already loaded
                if current_model == task.model_type and available_time < best_time:
                    best_gpu = gpu_id
                    best_time = available_time
                    best_switch_time = 0
                elif available_time + switch_time < best_time + best_switch_time:
                    best_gpu = gpu_id
                    best_time = available_time
                    best_switch_time = switch_time
            
            return (best_gpu, best_time) if best_gpu is not None else None
        
        else:
            # Multi-GPU - find consecutive GPUs
            best_gpu_set = None
            best_time = float('inf')
            
            for start_gpu in range(self.num_gpus - required_gpus + 1):
                gpu_set = list(range(start_gpu, start_gpu + required_gpus))
                max_available = max(self.gpu_available_at[g] for g in gpu_set)
                
                if max_available < best_time:
                    best_time = max_available
                    best_gpu_set = gpu_set
            
            return (best_gpu_set[0], best_time) if best_gpu_set else None
    
    def schedule_task(self, task: Task, gpu_id: int, start_time: float):
        """Schedule a task on GPU(s)"""
        model = self.models[task.model_type]
        
        # Handle model switch
        current_model = self.gpu_states[gpu_id].current_model
        switch_time = self._get_model_switch_time(current_model, task.model_type)
        
        if switch_time > 0:
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
            gpu_id=gpu_id,
            start_time=actual_start,
            end_time=end_time,
            estimated_time=task.estimated_time
        )
        
        scheduled_task = ScheduledTask(
            node=node_exec,
            gpu_id=gpu_id,
            start_time=actual_start,
            end_time=end_time,
            switch_time=switch_time
        )
        self.scheduled_tasks.append(scheduled_task)
        
        # Update GPU states
        if model.gpus_required == 1:
            self.gpu_states[gpu_id].current_model = task.model_type
            self.gpu_available_at[gpu_id] = end_time
        else:
            # Update all GPUs in the set
            for i in range(model.gpus_required):
                gpu = gpu_id + i
                self.gpu_states[gpu].current_model = task.model_type
                self.gpu_available_at[gpu] = end_time
        
        # Schedule completion event
        completion_event = Event(
            timestamp=end_time,
            event_type="task_complete",
            data={"task": task}
        )
        heapq.heappush(self.event_queue, completion_event)
        
        self.logger.debug(f"Scheduled {task.task_id} on GPU {gpu_id} "
                         f"(start: {actual_start:.2f}, end: {end_time:.2f})")
    
    def handle_task_completion(self, task: Task):
        """Handle task completion and update dependencies"""
        self.completed_tasks.add(task.task_id)
        
        # Find dependent tasks
        dag = self.workflows[task.workflow_id]
        dependents = dag.get_dependents(task.node_id)
        
        for dep_node_id in dependents:
            dep_task_id = f"{task.request_id}_{dep_node_id}"
            if dep_task_id in self.pending_dependencies:
                # Remove this dependency
                self.pending_dependencies[dep_task_id].discard(task.task_id)
                
                # If all dependencies satisfied, add to ready queue
                if not self.pending_dependencies[dep_task_id]:
                    dep_task = self.all_tasks[dep_task_id]
                    self.ready_queue[dep_task.model_type].append(dep_task)
                    del self.pending_dependencies[dep_task_id]
    
    def select_next_batch(self) -> Optional[Tuple[str, List[Task]]]:
        """Select the next batch of tasks to schedule"""
        if not self.ready_queue:
            return None
        
        # Score each model based on:
        # 1. Number of ready tasks
        # 2. Total execution time
        # 3. Current GPU state (how many GPUs have this model loaded)
        best_model = None
        best_score = -float('inf')
        
        for model_type, tasks in self.ready_queue.items():
            if not tasks:
                continue
            
            # Count GPUs with this model
            gpus_with_model = sum(1 for g in self.gpu_states.values() 
                                if g.current_model == model_type)
            
            # Calculate score
            num_tasks = len(tasks)
            total_time = sum(t.estimated_time for t in tasks)
            model_load_time = self.models[model_type].load_time_seconds
            
            # Heuristic: prioritize models with many tasks and/or already loaded
            score = num_tasks * 100 + total_time + gpus_with_model * model_load_time * 5
            
            if score > best_score:
                best_score = score
                best_model = model_type
        
        if best_model:
            # Return all ready tasks for this model
            tasks = self.ready_queue[best_model]
            self.ready_queue[best_model] = []
            return (best_model, tasks)
        
        return None
    
    def schedule_batch(self, model_type: str, tasks: List[Task]):
        """Schedule a batch of tasks of the same model type"""
        scheduled_count = 0
        
        for task in tasks:
            # Find best GPU
            result = self.find_best_gpu_for_task(task)
            if result:
                gpu_id, available_time = result
                # Only schedule if GPU is available soon (within reasonable time)
                if available_time <= self.current_time + 100:  # 100s threshold
                    self.schedule_task(task, gpu_id, max(self.current_time, available_time))
                    scheduled_count += 1
                else:
                    # Put back in ready queue
                    self.ready_queue[model_type].append(task)
            else:
                # Put back in ready queue
                self.ready_queue[model_type].append(task)
        
        return scheduled_count
    
    def run(self, requests: List[Dict]):
        """Execute the improved offline batch processing"""
        self.logger.info("Starting improved offline batch processing")
        self.logger.info(f"Available GPUs: {self.gpus}")
        
        # Initialize tasks
        self.parse_workflows()
        
        # Initialize event queue with GPU availability events
        for gpu_id in self.gpus:
            event = Event(
                timestamp=0.0,
                event_type="gpu_available",
                data={"gpu_id": gpu_id}
            )
            heapq.heappush(self.event_queue, event)
        
        # Main event loop
        scheduled_total = 0
        while self.event_queue and scheduled_total < len(self.all_tasks):
            # Get next event
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp
            
            if event.event_type == "task_complete":
                # Handle task completion
                task = event.data["task"]
                self.handle_task_completion(task)
                scheduled_total += 1
                
                if scheduled_total % 100 == 0:
                    self.logger.info(f"Progress: {scheduled_total}/{len(self.all_tasks)} tasks completed")
            
            # Try to schedule ready tasks
            batch = self.select_next_batch()
            if batch:
                model_type, tasks = batch
                scheduled = self.schedule_batch(model_type, tasks)
                self.logger.debug(f"Scheduled {scheduled} {model_type} tasks at time {self.current_time:.2f}")
        
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
            'total_tasks': len(self.scheduled_tasks),
            'total_model_switches': self.model_switch_count,
            'total_switch_time': self.total_switch_time,
            'estimated_makespan': makespan,
            'average_throughput': len(self.scheduled_tasks) / makespan if makespan > 0 else 0,
            'switch_overhead': self.total_switch_time / makespan * 100 if makespan > 0 else 0,
            'gpu_utilization': {},
            'gpu_task_count': gpu_task_count
        }
        
        for gpu_id in self.gpus:
            busy_time = gpu_busy_time.get(gpu_id, 0)
            metrics['gpu_utilization'][gpu_id] = (busy_time / makespan * 100) if makespan > 0 else 0
        
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
            utilization = metrics['gpu_utilization'].get(gpu_id, 0)
            task_count = metrics['gpu_task_count'].get(gpu_id, 0)
            self.logger.info(f"  GPU {gpu_id}: {utilization:.1f}% utilization, {task_count} executions")
    
    def _save_execution_log(self, metrics: Dict):
        """Save detailed execution log"""
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
                "makespan": metrics.get('estimated_makespan', 0),
                "total_model_switches": metrics.get('total_model_switches', 0),
                "total_switch_time": metrics.get('total_switch_time', 0),
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
        
        with open("improved_offline_execution_log.json", "w") as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"\nDetailed log written to improved_offline_execution_log.json")


def main():
    parser = argparse.ArgumentParser(description="Improved Offline Batch Processing Scheduler")
    parser.add_argument("--gpus", type=int, required=True, help="Number of available GPUs")
    parser.add_argument("--simulate", type=lambda x: x.lower() == "true", default=False, help="Use simulation mode")
    parser.add_argument("--config", type=Path, default=Path("system_config.yaml"), help="Configuration file")
    parser.add_argument("--requests", type=Path, default=Path("workflow_requests.yaml"), help="Requests file")
    
    args = parser.parse_args()
    
    gpu_list = list(range(args.gpus))
    
    scheduler = ImprovedOfflineScheduler(gpus=gpu_list, simulate=args.simulate)
    scheduler.load_config(args.config)
    scheduler.load_requests(args.requests)
    scheduler.run(scheduler.workflow_requests)


if __name__ == "__main__":
    main()
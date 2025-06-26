#!/usr/bin/env python3
"""
Online Greedy Scheduler for AI Workflows
Implements real-time scheduling with focus on minimizing P99 latency and average waiting time
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

from scheduler_component import (
    ModelInfo, WorkflowNode, NodeExecution, GPUState, ScheduledTask
)


@dataclass
class OnlineTask:
    """Represents a task in the online scheduler"""
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
    def waiting_time(self) -> float:
        """Time spent waiting since becoming ready"""
        if self.start_time > 0:
            return self.start_time - self.ready_time
        return 0.0
    
    @property
    def response_time(self) -> float:
        """Total time from arrival to completion"""
        if self.completion_time > 0:
            return self.completion_time - self.arrival_time
        return 0.0


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    arrival_time: float
    completion_time: float = 0.0
    total_waiting_time: float = 0.0
    node_count: int = 0
    completed_nodes: int = 0


class OnlineGreedyScheduler:
    """Online greedy scheduler with focus on P99 and average waiting time"""
    
    def __init__(self, gpus: List[int], simulate: bool = False):
        self.gpus = gpus
        self.num_gpus = len(gpus)
        self.simulate = simulate
        self.mode = "online"
        self.logger = self._setup_logger()
        
        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, Dict] = {}  # workflow_id -> workflow_data
        
        # Scheduling state
        self.gpu_states: List[GPUState] = [GPUState(gpu_id=i) for i in gpus]
        self.gpu_reserved: Dict[int, bool] = {i: False for i in gpus}  # Track GPU reservations
        self.gpu_reserved_until: Dict[int, float] = {i: 0.0 for i in gpus}  # Track reservation end times
        self.ready_queue: List[OnlineTask] = []  # Priority queue of ready tasks
        self.pending_tasks: Dict[str, OnlineTask] = {}  # task_id -> task
        self.completed_tasks: Set[str] = set()  # Set of completed task_ids
        self.scheduled_tasks: List[ScheduledTask] = []
        
        # Time tracking
        self.current_time = 0.0
        self.event_queue: List[Tuple[float, str, Dict]] = []  # (time, event_type, data)
        
        # Metrics
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.task_waiting_times: List[float] = []
        self.task_response_times: List[float] = []
        self.model_switch_count = 0
        self.total_switch_time = 0.0
        
        # Greedy parameters (tunable)
        self.alpha = 1.0  # Wait time weight
        self.beta = 1.5   # Switch cost weight
        self.gamma = 0.3  # Future impact weight
        self.delta = 0.5  # Load balancing weight
        self.lookahead_window = 3  # Number of tasks to consider for future impact
        self.enable_work_stealing = True  # Enable work stealing for idle GPUs
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('OnlineGreedyScheduler')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('online_scheduler.log', mode='w')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
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
        self.workflows = config['workflows']
        
        self.logger.info(f"Loaded {len(self.models)} models and {len(self.workflows)} workflows")
    
    def _get_model_switch_time(self, from_model: Optional[str], to_model: str) -> float:
        """Calculate model switch time"""
        if from_model == to_model:
            return 0.0
        
        if not from_model:
            return self.models[to_model].load_time_seconds
        
        # Unload old + load new
        return self.models[from_model].memory_gb / 16.0 + self.models[to_model].load_time_seconds
    
    def _calculate_wait_time_cost(self, task: OnlineTask, gpu: GPUState) -> float:
        """Calculate wait time cost for scheduling task on GPU"""
        # When will the GPU be available?
        gpu_available_time = max(gpu.available_at, self.gpu_reserved_until.get(gpu.gpu_id, 0))
        wait_time = max(0, gpu_available_time - self.current_time)
        
        # Add time already waited
        time_already_waited = self.current_time - task.ready_time
        
        return wait_time + time_already_waited
    
    def _calculate_switch_cost(self, task: OnlineTask, gpu: GPUState) -> float:
        """Calculate model switch cost"""
        return self._get_model_switch_time(gpu.current_model, task.model_type)
    
    def _calculate_future_impact(self, task: OnlineTask, gpu: GPUState) -> float:
        """Estimate impact on future tasks"""
        future_cost = 0.0
        
        # Look at next k tasks in ready queue
        lookahead_tasks = []
        temp_queue = list(self.ready_queue)
        
        for _ in range(min(self.lookahead_window, len(temp_queue))):
            if temp_queue:
                lookahead_tasks.append(heapq.heappop(temp_queue))
        
        # Estimate delay caused to future tasks
        gpu_available_time = max(gpu.available_at, self.gpu_reserved_until.get(gpu.gpu_id, 0))
        task_end_time = gpu_available_time + self._calculate_switch_cost(task, gpu) + task.estimated_time
        
        for future_task in lookahead_tasks:
            if future_task.model_type == task.model_type:
                # Same model - benefit from batching
                future_cost -= 0.1 * self._get_model_switch_time(None, task.model_type)
            else:
                # Different model - potential delay
                future_cost += 0.1 * (task_end_time - self.current_time)
        
        return future_cost
    
    def _calculate_load_balance_cost(self, gpu: GPUState) -> float:
        """Calculate load balancing cost - penalize overloaded GPUs"""
        # Calculate current load across all GPUs
        total_load = sum(g.execution_count for g in self.gpu_states)
        avg_load = total_load / self.num_gpus if self.num_gpus > 0 else 0
        
        # Penalize deviation from average
        load_deviation = gpu.execution_count - avg_load
        
        # Also consider current utilization
        total_busy_time = sum(g.total_busy_time for g in self.gpu_states)
        avg_busy_time = total_busy_time / self.num_gpus if self.num_gpus > 0 else 0
        busy_time_deviation = gpu.total_busy_time - avg_busy_time
        
        # Normalize by current time to get relative load
        if self.current_time > 0:
            relative_load = busy_time_deviation / self.current_time
        else:
            relative_load = 0
        
        return load_deviation * 10 + relative_load * 100
    
    def _calculate_greedy_cost(self, task: OnlineTask, gpu: GPUState) -> float:
        """Calculate total greedy cost for task-GPU assignment"""
        wait_cost = self._calculate_wait_time_cost(task, gpu)
        switch_cost = self._calculate_switch_cost(task, gpu)
        future_impact = self._calculate_future_impact(task, gpu)
        load_balance_cost = self._calculate_load_balance_cost(gpu)
        
        # Apply weights
        total_cost = (self.alpha * wait_cost + 
                     self.beta * switch_cost + 
                     self.gamma * future_impact +
                     self.delta * load_balance_cost)
        
        # Add affinity bonus if same model
        if gpu.current_model == task.model_type:
            total_cost -= 0.2 * switch_cost
        
        return total_cost
    
    def _find_best_gpu_assignment(self, task: OnlineTask) -> Tuple[int, float]:
        """Find best GPU for task using greedy cost function"""
        model = self.models[task.model_type]
        best_gpu = -1
        best_cost = float('inf')
        
        # Check if we have enough GPUs for this model
        if model.gpus_required > self.num_gpus:
            self.logger.warning(f"Task {task.request_id}_{task.node_id} requires {model.gpus_required} GPUs "
                              f"but only {self.num_gpus} are available. Task cannot be scheduled.")
            return -1, float('inf')
        
        if model.gpus_required > 1:
            # Multi-GPU model - prefer less loaded GPU sets
            gpu_set_costs = []
            
            for start_idx in range(self.num_gpus - model.gpus_required + 1):
                gpu_set = list(range(start_idx, start_idx + model.gpus_required))
                
                # Check if all GPUs in set are available
                can_use = True
                for gpu_id in gpu_set:
                    if self.gpu_reserved.get(gpu_id, False):
                        can_use = False
                        break
                
                if can_use:
                    # Calculate combined cost for GPU set
                    total_cost = 0.0
                    max_wait = 0.0
                    
                    for gpu_id in gpu_set:
                        cost = self._calculate_greedy_cost(task, self.gpu_states[gpu_id])
                        total_cost += cost
                        
                        # Track maximum wait time across the set
                        gpu_available_time = max(self.gpu_states[gpu_id].available_at, 
                                               self.gpu_reserved_until.get(gpu_id, 0))
                        wait = max(0, gpu_available_time - self.current_time)
                        max_wait = max(max_wait, wait)
                    
                    # Average the cost and add penalty for maximum wait
                    avg_cost = total_cost / model.gpus_required + max_wait * 50
                    gpu_set_costs.append((avg_cost, gpu_set[0]))
            
            # Choose the best GPU set
            if gpu_set_costs:
                gpu_set_costs.sort(key=lambda x: x[0])
                best_cost, best_gpu = gpu_set_costs[0]
        else:
            # Single-GPU model
            for gpu_id in self.gpus:
                if not self.gpu_reserved.get(gpu_id, False):
                    cost = self._calculate_greedy_cost(task, self.gpu_states[gpu_id])
                    if cost < best_cost:
                        best_cost = cost
                        best_gpu = gpu_id
        
        return best_gpu, best_cost
    
    def _schedule_task_on_gpu(self, task: OnlineTask, gpu_id: int):
        """Schedule task on specified GPU"""
        gpu = self.gpu_states[gpu_id]
        model = self.models[task.model_type]
        
        # Calculate timing
        switch_time = self._get_model_switch_time(gpu.current_model, task.model_type)
        gpu_available_time = max(gpu.available_at, self.gpu_reserved_until.get(gpu_id, 0))
        start_time = max(self.current_time, gpu_available_time) + switch_time
        end_time = start_time + task.estimated_time
        
        # Update task
        task.scheduled_time = self.current_time
        task.start_time = start_time
        task.completion_time = end_time
        
        # Create scheduled task
        node_exec = NodeExecution(
            request_id=task.request_id,
            workflow_id=task.workflow_id,
            node_id=task.node_id,
            model_name=task.model_type,
            status="scheduled",
            gpu_id=gpu_id,
            start_time=start_time,
            end_time=end_time,
            estimated_time=task.estimated_time
        )
        
        scheduled = ScheduledTask(
            node=node_exec,
            gpu_id=gpu_id,
            start_time=start_time,
            end_time=end_time,
            switch_time=switch_time
        )
        self.scheduled_tasks.append(scheduled)
        
        # Update GPU state
        gpu.current_model = task.model_type
        gpu.available_at = end_time
        gpu.total_busy_time += task.estimated_time
        gpu.execution_count += 1
        
        # Reserve additional GPUs if needed
        if model.gpus_required > 1:
            for i in range(1, model.gpus_required):
                aux_gpu_id = gpu_id + i
                if aux_gpu_id < self.num_gpus:
                    self.gpu_reserved[aux_gpu_id] = True
                    self.gpu_reserved_until[aux_gpu_id] = end_time
                    self.gpu_states[aux_gpu_id].available_at = end_time
        
        # Schedule completion event
        heapq.heappush(self.event_queue, (end_time, "task_complete", {
            "task_id": f"{task.request_id}_{task.node_id}",
            "gpu_id": gpu_id,
            "gpus_used": model.gpus_required
        }))
        
        # Update metrics
        if gpu.current_model != task.model_type and gpu.current_model is not None:
            self.model_switch_count += 1
            self.total_switch_time += switch_time
        
        self.logger.debug(f"Scheduled {task.request_id}_{task.node_id} on GPU {gpu_id} "
                         f"at time {start_time:.2f}-{end_time:.2f}")
    
    def _process_request_arrival(self, request: Dict):
        """Process a new workflow request arrival"""
        workflow_id = request['workflow_id']
        request_id = request['request_id']
        # Use normalized timestamp if available, otherwise parse from timestamp
        if '_normalized_timestamp' in request:
            timestamp = request['_normalized_timestamp']
        else:
            timestamp = datetime.fromisoformat(request['timestamp']).timestamp()
        node_execution_times = request.get('node_execution_times', {})
        
        if workflow_id not in self.workflows:
            self.logger.warning(f"Unknown workflow: {workflow_id}")
            return
        
        workflow = self.workflows[workflow_id]
        
        # Initialize request metrics
        self.request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            arrival_time=timestamp,
            node_count=len(workflow['nodes'])
        )
        
        # Create tasks for each node
        node_map = {node['id']: node for node in workflow['nodes']}
        edges = workflow.get('edges', [])
        
        # Build dependency map
        dependencies = defaultdict(set)
        for edge in edges:
            dependencies[edge['to']].add(edge['from'])
        
        # Create tasks
        for node in workflow['nodes']:
            node_id = node['id']
            task_id = f"{request_id}_{node_id}"
            
            # Get execution time
            if node_id in node_execution_times:
                exec_time = node_execution_times[node_id]
            else:
                model = self.models[node['model']]
                tokens = 1000  # Default
                tokens_per_second = model.tokens_per_second if model.tokens_per_second else 100
                exec_time = tokens / tokens_per_second
            
            task = OnlineTask(
                workflow_id=workflow_id,
                node_id=node_id,
                model_type=node['model'],
                request_id=request_id,
                arrival_time=timestamp,
                dependencies={f"{request_id}_{dep}" for dep in dependencies[node_id]},
                estimated_time=exec_time,
                ready_time=timestamp  # Will be updated when dependencies are met
            )
            
            self.pending_tasks[task_id] = task
            
            # If no dependencies, add to ready queue
            if not task.dependencies:
                task.ready_time = self.current_time
                heapq.heappush(self.ready_queue, task)
    
    def _process_task_completion(self, event_data: Dict):
        """Process task completion event"""
        task_id = event_data['task_id']
        gpu_id = event_data['gpu_id']
        gpus_used = event_data['gpus_used']
        
        # Mark task as completed
        self.completed_tasks.add(task_id)
        
        # Free up reserved GPUs
        if gpus_used > 1:
            for i in range(1, gpus_used):
                aux_gpu_id = gpu_id + i
                if aux_gpu_id < self.num_gpus:
                    self.gpu_reserved[aux_gpu_id] = False
                    # Check if reservation time has passed
                    if self.gpu_reserved_until.get(aux_gpu_id, 0) <= self.current_time:
                        self.gpu_reserved_until[aux_gpu_id] = 0
        
        # Get task
        if task_id in self.pending_tasks:
            task = self.pending_tasks[task_id]
            
            # Update metrics
            waiting_time = task.waiting_time
            response_time = task.response_time
            self.task_waiting_times.append(waiting_time)
            self.task_response_times.append(response_time)
            
            # Update request metrics
            req_metrics = self.request_metrics[task.request_id]
            req_metrics.completed_nodes += 1
            req_metrics.total_waiting_time += waiting_time
            
            if req_metrics.completed_nodes == req_metrics.node_count:
                req_metrics.completion_time = self.current_time
            
            # Check for newly ready tasks
            for pending_id, pending_task in self.pending_tasks.items():
                if (pending_id not in self.completed_tasks and 
                    task_id in pending_task.dependencies and
                    pending_task.dependencies.issubset(self.completed_tasks)):
                    # All dependencies satisfied
                    pending_task.ready_time = self.current_time
                    heapq.heappush(self.ready_queue, pending_task)
    
    def _update_task_priorities(self):
        """Update priorities of tasks in ready queue based on waiting time"""
        updated_tasks = []
        
        while self.ready_queue:
            task = heapq.heappop(self.ready_queue)
            
            # Update priority based on time waited
            wait_time = self.current_time - task.ready_time
            
            # Priority increases with wait time to prevent starvation
            task.priority = -wait_time  # Negative because lower value = higher priority
            
            updated_tasks.append(task)
        
        # Re-add all tasks with updated priorities
        for task in updated_tasks:
            heapq.heappush(self.ready_queue, task)
    
    def _get_idle_gpus(self) -> List[int]:
        """Get list of currently idle GPUs"""
        idle_gpus = []
        for gpu in self.gpu_states:
            gpu_available_time = max(gpu.available_at, self.gpu_reserved_until.get(gpu.gpu_id, 0))
            if gpu_available_time <= self.current_time and not self.gpu_reserved.get(gpu.gpu_id, False):
                idle_gpus.append(gpu.gpu_id)
        return idle_gpus
    
    def _try_work_stealing(self, idle_gpus: List[int]):
        """Try to steal work for idle GPUs"""
        if not idle_gpus or not self.ready_queue:
            return []
        
        scheduled = []
        # For each idle GPU, try to find suitable work
        for gpu_id in idle_gpus:
            if not self.ready_queue:
                break
                
            # Look for tasks that can run on this GPU
            best_task = None
            best_cost = float('inf')
            best_idx = -1
            
            # Check all tasks in ready queue
            for idx, task in enumerate(self.ready_queue):
                model = self.models[task.model_type]
                
                # Skip if this GPU can't handle the task
                if model.gpus_required > 1:
                    # Check if we have enough consecutive GPUs
                    if gpu_id + model.gpus_required > self.num_gpus:
                        continue
                    # Check if all required GPUs are available
                    can_use_set = True
                    for aux_gpu in range(gpu_id, gpu_id + model.gpus_required):
                        if self.gpu_reserved.get(aux_gpu, False):
                            can_use_set = False
                            break
                    if not can_use_set:
                        continue
                
                # Calculate cost for this assignment
                cost = self._calculate_greedy_cost(task, self.gpu_states[gpu_id])
                
                # Bonus for immediate execution on idle GPU
                cost -= 100  # Strong incentive to use idle GPU
                
                if cost < best_cost:
                    best_cost = cost
                    best_task = task
                    best_idx = idx
            
            if best_task and best_idx >= 0:
                # Remove task from queue and schedule it
                self.ready_queue.pop(best_idx)
                heapq.heapify(self.ready_queue)
                self._schedule_task_on_gpu(best_task, gpu_id)
                scheduled.append(f"{best_task.request_id}_{best_task.node_id}")
                
        return scheduled
    
    def _greedy_schedule_step(self):
        """Perform one step of greedy scheduling with work stealing"""
        if not self.ready_queue:
            return
        
        # Update task priorities
        self._update_task_priorities()
        
        # First, check for idle GPUs and try work stealing
        scheduled_this_step = []
        if self.enable_work_stealing:
            idle_gpus = self._get_idle_gpus()
            if idle_gpus:
                stolen_tasks = self._try_work_stealing(idle_gpus)
                scheduled_this_step.extend(stolen_tasks)
                if stolen_tasks:
                    self.logger.debug(f"Work stealing scheduled {len(stolen_tasks)} tasks on idle GPUs")
        
        # Then proceed with regular scheduling
        temp_ready = []
        tasks_to_schedule = []
        
        # Separate multi-GPU and single-GPU tasks
        multi_gpu_tasks = []
        single_gpu_tasks = []
        
        # Extract tasks from queue
        max_batch_size = min(len(self.ready_queue), self.num_gpus * 2)
        for _ in range(max_batch_size):
            if self.ready_queue:
                task = heapq.heappop(self.ready_queue)
                model = self.models[task.model_type]
                if model.gpus_required > 1:
                    multi_gpu_tasks.append(task)
                else:
                    single_gpu_tasks.append(task)
        
        # Schedule multi-GPU tasks first (they're harder to place)
        for task in multi_gpu_tasks:
            best_gpu, best_cost = self._find_best_gpu_assignment(task)
            
            if best_gpu >= 0 and best_cost < float('inf'):
                self._schedule_task_on_gpu(task, best_gpu)
                scheduled_this_step.append(f"{task.request_id}_{task.node_id}")
            else:
                temp_ready.append(task)
        
        # Then schedule single-GPU tasks
        single_gpu_tasks.sort(key=lambda t: self._estimate_queue_impact(t))
        
        for task in single_gpu_tasks:
            best_gpu, best_cost = self._find_best_gpu_assignment(task)
            
            if best_gpu >= 0 and best_cost < float('inf'):
                self._schedule_task_on_gpu(task, best_gpu)
                scheduled_this_step.append(f"{task.request_id}_{task.node_id}")
            else:
                temp_ready.append(task)
        
        # Re-add unscheduled tasks
        for task in temp_ready:
            heapq.heappush(self.ready_queue, task)
        
        if scheduled_this_step:
            self.logger.debug(f"Time {self.current_time:.2f}: Scheduled {len(scheduled_this_step)} tasks")
    
    def _estimate_queue_impact(self, task: OnlineTask) -> float:
        """Estimate the impact of scheduling this task on total queue wait time"""
        # Estimate how many other tasks in queue depend on this one
        dependent_count = 0
        for other_task in self.ready_queue:
            if f"{task.request_id}_{task.node_id}" in other_task.dependencies:
                dependent_count += 1
        
        # Higher priority for tasks that unblock others
        impact = -dependent_count * 100
        
        # Also consider how long this task has been waiting
        wait_time = self.current_time - task.ready_time
        impact += wait_time
        
        return impact
    
    def run(self, requests: List[Dict]):
        """Run online scheduler simulation"""
        self.logger.info("Starting online greedy scheduler")
        self.logger.info(f"Available GPUs: {self.gpus}")
        self.logger.info(f"Greedy parameters: α={self.alpha}, β={self.beta}, γ={self.gamma}, δ={self.delta}")
        self.logger.info(f"Work stealing: {'enabled' if self.enable_work_stealing else 'disabled'}")
        
        # Initialize event queue with request arrivals
        # Normalize timestamps to start from 0
        if requests:
            first_timestamp = datetime.fromisoformat(requests[0]['timestamp']).timestamp()
            for request in requests:
                timestamp = datetime.fromisoformat(request['timestamp']).timestamp()
                normalized_timestamp = timestamp - first_timestamp
                request['_normalized_timestamp'] = normalized_timestamp
                heapq.heappush(self.event_queue, (normalized_timestamp, "request_arrival", request))
        
        # Main simulation loop
        max_iterations = 10000  # Safety limit
        iteration = 0
        
        while (self.event_queue or self.ready_queue) and iteration < max_iterations:
            iteration += 1
            
            # Get next event
            if self.event_queue:
                next_time, event_type, event_data = heapq.heappop(self.event_queue)
                self.current_time = next_time
                
                # Process event
                if event_type == "request_arrival":
                    self._process_request_arrival(event_data)
                elif event_type == "task_complete":
                    self._process_task_completion(event_data)
            
            # Try to schedule ready tasks
            self._greedy_schedule_step()
            
            # If no events and tasks are ready, advance time
            if not self.event_queue and self.ready_queue:
                # Find earliest GPU available time
                min_gpu_time = float('inf')
                for gpu in self.gpu_states:
                    gpu_time = max(gpu.available_at, self.gpu_reserved_until.get(gpu.gpu_id, 0))
                    min_gpu_time = min(min_gpu_time, gpu_time)
                if min_gpu_time > self.current_time and min_gpu_time < float('inf'):
                    self.current_time = min_gpu_time
                else:
                    # If we can't advance time and have ready tasks, something is wrong
                    self.logger.warning(f"Cannot advance time. Current: {self.current_time}, Ready tasks: {len(self.ready_queue)}")
                    break
        
        if iteration >= max_iterations:
            self.logger.error(f"Simulation stopped after {max_iterations} iterations to prevent infinite loop")
        
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        # Print results
        self._print_metrics(metrics)
        
        # Save execution log
        self._save_execution_log(metrics)
        
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics with focus on P99 and avg waiting time"""
        if not self.scheduled_tasks:
            return {}
        
        makespan = max(task.end_time for task in self.scheduled_tasks)
        
        # Calculate P99 waiting time
        if self.task_waiting_times:
            p99_waiting_time = np.percentile(self.task_waiting_times, 99)
            avg_waiting_time = np.mean(self.task_waiting_times)
        else:
            p99_waiting_time = 0.0
            avg_waiting_time = 0.0
        
        # Calculate P99 response time
        if self.task_response_times:
            p99_response_time = np.percentile(self.task_response_times, 99)
            avg_response_time = np.mean(self.task_response_times)
        else:
            p99_response_time = 0.0
            avg_response_time = 0.0
        
        # Calculate request-level metrics
        request_response_times = []
        for req_id, metrics in self.request_metrics.items():
            if metrics.completion_time > 0:
                response_time = metrics.completion_time - metrics.arrival_time
                request_response_times.append(response_time)
        
        if request_response_times:
            p99_request_response_time = np.percentile(request_response_times, 99)
            avg_request_response_time = np.mean(request_response_times)
        else:
            p99_request_response_time = 0.0
            avg_request_response_time = 0.0
        
        metrics = {
            'total_tasks': len(self.scheduled_tasks),
            'total_requests': len(self.request_metrics),
            'completed_requests': len([r for r in self.request_metrics.values() if r.completion_time > 0]),
            'total_model_switches': self.model_switch_count,
            'total_switch_time': self.total_switch_time,
            'estimated_makespan': makespan,
            'average_throughput': len(self.scheduled_tasks) / makespan if makespan > 0 else 0,
            'switch_overhead': self.total_switch_time / makespan * 100 if makespan > 0 else 0,
            
            # Task-level metrics
            'avg_waiting_time': avg_waiting_time,
            'p99_waiting_time': p99_waiting_time,
            'avg_response_time': avg_response_time,
            'p99_response_time': p99_response_time,
            
            # Request-level metrics
            'avg_request_response_time': avg_request_response_time,
            'p99_request_response_time': p99_request_response_time,
            
            # GPU metrics
            'gpu_utilization': {},
            'gpu_execution_count': {}
        }
        
        for gpu_id, gpu_state in enumerate(self.gpu_states):
            metrics['gpu_utilization'][gpu_id] = (gpu_state.total_busy_time / makespan * 100) if makespan > 0 else 0
            metrics['gpu_execution_count'][gpu_id] = gpu_state.execution_count
        
        return metrics
    
    def _print_metrics(self, metrics: Dict):
        """Print execution metrics"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ONLINE SCHEDULING METRICS")
        self.logger.info("=" * 60)
        
        if not metrics:
            self.logger.warning("No tasks scheduled")
            return
        
        self.logger.info(f"Total execution time: {metrics['estimated_makespan']:.2f} seconds")
        self.logger.info(f"Total requests: {metrics['total_requests']} (completed: {metrics['completed_requests']})")
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
            utilization = metrics['gpu_utilization'].get(gpu_id, 0)
            exec_count = metrics['gpu_execution_count'].get(gpu_id, 0)
            self.logger.info(f"  GPU {gpu_id}: {utilization:.1f}% utilization, {exec_count} executions")
    
    def _save_execution_log(self, metrics: Dict):
        """Save detailed execution log"""
        log_data = {
            "summary": {
                "total_requests": metrics.get('total_requests', 0),
                "completed_requests": metrics.get('completed_requests', 0),
                "total_nodes_executed": len(self.scheduled_tasks),
                "mode": self.mode,
                "simulate": self.simulate,
                "gpus": self.gpus,
                "makespan": metrics.get('estimated_makespan', 0),
                "total_model_switches": metrics.get('total_model_switches', 0),
                "total_switch_time": metrics.get('total_switch_time', 0),
                "avg_waiting_time": metrics.get('avg_waiting_time', 0),
                "p99_waiting_time": metrics.get('p99_waiting_time', 0),
                "avg_response_time": metrics.get('avg_response_time', 0),
                "p99_response_time": metrics.get('p99_response_time', 0),
                "avg_request_response_time": metrics.get('avg_request_response_time', 0),
                "p99_request_response_time": metrics.get('p99_request_response_time', 0),
                "greedy_parameters": {
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "lookahead_window": self.lookahead_window
                }
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
                "switch_time": task.switch_time,
                "waiting_time": self.pending_tasks.get(
                    f"{task.node.request_id}_{task.node.node_id}", 
                    OnlineTask("", "", "", "", 0)
                ).waiting_time
            })
        
        with open("online_execution_log.json", "w") as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"\nDetailed log written to online_execution_log.json")


def main():
    parser = argparse.ArgumentParser(description="Online Greedy Scheduler for AI Workflows")
    parser.add_argument("--gpus", type=int, required=True, help="Number of available GPUs")
    parser.add_argument("--simulate", type=lambda x: x.lower() == "true", default=False, help="Use simulation mode (true/false)")
    parser.add_argument("--config", type=Path, default=Path("simple_config.json"), help="Path to system configuration file")
    parser.add_argument("--requests", type=Path, default=Path("simple_requests.yaml"), help="Path to workflow requests file")
    
    # Greedy parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Wait time weight (default: 1.0)")
    parser.add_argument("--beta", type=float, default=1.5, help="Switch cost weight (default: 1.5)")
    parser.add_argument("--gamma", type=float, default=0.3, help="Future impact weight (default: 0.3)")
    parser.add_argument("--delta", type=float, default=0.5, help="Load balancing weight (default: 0.5)")
    parser.add_argument("--lookahead", type=int, default=3, help="Lookahead window size (default: 3)")
    parser.add_argument("--no-work-stealing", action="store_true", help="Disable work stealing")
    
    args = parser.parse_args()
    
    # Generate GPU list
    gpu_list = list(range(args.gpus))
    
    # Create scheduler
    scheduler = OnlineGreedyScheduler(gpus=gpu_list, simulate=args.simulate)
    
    # Set greedy parameters
    scheduler.alpha = args.alpha
    scheduler.beta = args.beta
    scheduler.gamma = args.gamma
    scheduler.delta = args.delta
    scheduler.lookahead_window = args.lookahead
    scheduler.enable_work_stealing = not args.no_work_stealing
    
    # Load configuration
    scheduler.load_config(args.config)
    
    # Load requests
    with open(args.requests, 'r') as f:
        if args.requests.suffix == '.yaml':
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    requests = data['requests']
    
    # Run scheduler
    scheduler.run(requests)


if __name__ == "__main__":
    main()
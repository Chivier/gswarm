#!/usr/bin/env python3
"""
Baseline Scheduler for GSwarm Workflows
Uses Ray-like scheduling: runs models one by one, maintaining a queue of ready nodes.
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
    handlers=[logging.FileHandler("baseline_scheduler.log"), logging.StreamHandler(sys.stdout)],
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
    gpu_ids: Optional[List[int]] = None  # All GPUs for multi-GPU models
    instance_id: Optional[str] = None
    estimated_time: Optional[float] = None
    switch_time: float = 0.0  # Time spent switching models
    ready_time: float = 0.0  # When the task became ready

    @property
    def execution_time(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class GPUState:
    """GPU state tracking"""

    gpu_id: int
    current_model: Optional[str] = None
    current_instance: Optional[str] = None
    busy: bool = False
    last_switch_time: float = 0.0
    available_at: float = 0.0  # Timestamp when GPU will be available


@dataclass
class Event:
    """Event for discrete event simulation"""

    timestamp: float
    event_type: str  # "node_complete", "request_arrival"
    data: Any

    def __lt__(self, other):
        return self.timestamp < other.timestamp


class BaselineScheduler:
    """Baseline scheduler using Ray-like scheduling strategy"""

    def __init__(self, gpus: List[int], simulate: bool = False, mode: str = "offline"):
        self.gpus = gpus
        self.simulate = simulate
        self.mode = mode
        self.server_url = SERVER_URL

        # GPU states
        self.gpu_states = {gpu_id: GPUState(gpu_id) for gpu_id in gpus}

        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, Workflow] = {}

        # Execution tracking
        self.node_queue: deque[NodeExecution] = deque()
        self.executions: Dict[str, NodeExecution] = {}  # node_key -> execution
        self.completed_executions: List[NodeExecution] = []

        # Model instances for simulation mode
        self.model_instances: Dict[str, str] = {}  # model_name -> instance_id

        # Metrics
        self.request_start_times: Dict[str, float] = {}
        self.request_end_times: Dict[str, float] = {}
        self.model_switch_count = 0
        self.total_switch_time = 0.0

        # Event queue for discrete event simulation
        self.event_queue: List[Event] = []
        self.current_sim_time: float = 0.0

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

            self.workflows[workflow_id] = Workflow(id=workflow_id, name=workflow_data["name"], nodes=nodes, edges=edges)

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

    def _find_available_gpu(self, model_name: str, current_time: float) -> Optional[List[int]]:
        """Find available GPU(s) for the model at current time"""
        model_info = self.models[model_name]
        required_gpus = model_info.gpus_required

        # For single GPU models
        if required_gpus == 1:
            best_gpu = None
            best_available_time = float("inf")

            for gpu_id, gpu_state in self.gpu_states.items():
                # Check if GPU is available now or will be soon
                if gpu_state.available_at <= current_time:
                    # Prefer GPU with same model already loaded
                    if gpu_state.current_model == model_name:
                        return [gpu_id]
                    elif best_gpu is None or gpu_state.available_at < best_available_time:
                        best_gpu = gpu_id
                        best_available_time = gpu_state.available_at

            return [best_gpu] if best_gpu is not None else None

        # For multi-GPU models (simplified: use consecutive GPUs)
        else:
            gpu_ids = sorted(self.gpu_states.keys())
            for i in range(len(gpu_ids) - required_gpus + 1):
                consecutive_gpus = gpu_ids[i : i + required_gpus]
                # Check if all GPUs will be available
                max_available_time = max(self.gpu_states[gpu_id].available_at for gpu_id in consecutive_gpus)
                if max_available_time <= current_time:
                    return consecutive_gpus  # Return all GPUs in the group

        return None

    def _download_and_load_models(self):
        """Download and load all models if in simulate mode"""
        if not self.simulate:
            return

        logger.info("Downloading and loading models for simulation mode...")

        for model_id, model_info in self.models.items():
            # Map model IDs to actual model names
            model_name_map = {
                "llm7b": "gpt2",
                "llm30b": "gpt2-medium",
                "stable_diffusion": "CompVis/stable-diffusion-v1-4",
            }

            actual_model_name = model_name_map.get(model_id, "gpt2")

            try:
                # Download model
                response = requests.post(
                    f"{self.server_url}/standalone/download", json={"model_name": actual_model_name}
                )
                if response.status_code != 200:
                    logger.error(f"Failed to download {actual_model_name}: {response.text}")
                    continue

                # Load to DRAM
                response = requests.post(
                    f"{self.server_url}/standalone/load", json={"model_name": actual_model_name, "target": "dram"}
                )
                if response.status_code != 200:
                    logger.error(f"Failed to load {actual_model_name}: {response.text}")

                logger.info(f"Successfully downloaded and loaded {actual_model_name}")

            except Exception as e:
                logger.error(f"Error setting up model {model_id}: {e}")

    def _create_model_instance(self, model_name: str, gpu_id: int) -> Optional[str]:
        """Create a serving instance for a model on a specific GPU"""
        # Map model IDs to actual model names
        model_name_map = {"llm7b": "gpt2", "llm30b": "gpt2-medium", "stable_diffusion": "CompVis/stable-diffusion-v1-4"}

        actual_model_name = model_name_map.get(model_name, "gpt2")

        try:
            # Check if we already have an instance
            instance_key = f"{actual_model_name}_gpu{gpu_id}"
            if instance_key in self.model_instances:
                return self.model_instances[instance_key]

            # Create new serving instance
            model_info = self.models[model_name]
            if model_info.gpus_required == 1:
                device = f"cuda:{gpu_id}"
            else:
                # Multi-GPU: use consecutive GPUs
                devices = [f"cuda:{gpu_id + i}" for i in range(model_info.gpus_required)]
                device = ",".join(devices)

            response = requests.post(
                f"{self.server_url}/standalone/serve", json={"model_name": actual_model_name, "device": device}
            )

            if response.status_code == 200:
                data = response.json()["data"]
                instance_id = data["instance_id"]
                self.model_instances[instance_key] = instance_id
                logger.info(f"Created instance {instance_id} for {actual_model_name} on {device}")
                return instance_id
            else:
                logger.error(f"Failed to create instance: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error creating instance for {model_name}: {e}")
            return None

    def _estimate_execution_time(self, node_exec: NodeExecution, request: Request) -> float:
        """Estimate execution time for a node"""
        # In non-simulate mode, just use pre-computed times
        if not self.simulate:
            return request.node_execution_times.get(node_exec.node_id, 10.0)

        # In simulate mode, try to use the estimate API
        try:
            # Map model IDs to actual model names
            model_name_map = {
                "llm7b": "gpt2",
                "llm30b": "gpt2-medium",
                "stable_diffusion": "CompVis/stable-diffusion-v1-4",
            }

            actual_model_name = model_name_map.get(node_exec.model_name, "gpt2")

            # Determine device based on assigned GPU
            model_info = self.models[node_exec.model_name]
            if model_info.gpus_required == 1:
                device = f"cuda:{node_exec.gpu_id}"
            else:
                # Multi-GPU: use consecutive GPUs starting from assigned GPU
                devices = [f"cuda:{node_exec.gpu_id + i}" for i in range(model_info.gpus_required)]
                device = devices[0]  # Use first device for estimation

            # Prepare request data
            node_config = request.node_configs.get(node_exec.node_id, {})

            # Build data for estimation
            data = {"prompt": request.input_data.get("user_prompt", "Sample prompt"), **node_config}

            # Use direct estimation API (no instance required)
            response = requests.post(
                f"{self.server_url}/standalone/estimate",
                json={"model_name": actual_model_name, "device": device, "data": data},
            )

            if response.status_code == 200:
                result = response.json()
                estimated_time = result["data"]["estimated_execution_time"]
                logger.debug(f"Estimated execution time for {node_exec.node_id} on {device}: {estimated_time:.2f}s")
                return estimated_time
            else:
                logger.warning(f"Direct estimation failed, using pre-computed time: {response.text}")
                return request.node_execution_times.get(node_exec.node_id, 10.0)

        except Exception as e:
            logger.warning(f"Direct estimation error, using pre-computed time: {e}")
            return request.node_execution_times.get(node_exec.node_id, 10.0)

    def _call_model(self, node_exec: NodeExecution, request: Request) -> float:
        """Call model and return actual execution time"""
        try:
            # Get instance ID
            gpu_state = self.gpu_states[node_exec.gpu_id]
            instance_id = gpu_state.current_instance

            # Prepare request data
            node_config = request.node_configs.get(node_exec.node_id, {})

            # Build data for call
            data = {"prompt": request.input_data.get("user_prompt", "Sample prompt"), **node_config}

            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/standalone/call/{instance_id}", json={"instance_id": instance_id, "data": data}
            )
            end_time = time.time()

            if response.status_code == 200:
                return end_time - start_time
            else:
                logger.error(f"Model call failed: {response.text}")
                # Use pre-computed time as fallback
                return request.node_execution_times.get(node_exec.node_id, 10.0)

        except Exception as e:
            logger.error(f"Model call error: {e}")
            # Use pre-computed time as fallback
            return request.node_execution_times.get(node_exec.node_id, 10.0)

    def _schedule_node_execution(self, node_exec: NodeExecution, request: Request):
        """Schedule a node execution on its assigned GPU(s)"""
        model_name = node_exec.model_name
        model_info = self.models[model_name]
        
        # Get all GPUs involved
        gpu_ids = node_exec.gpu_ids if node_exec.gpu_ids else [node_exec.gpu_id]
        
        # Calculate when this node can actually start (when all required GPUs are available)
        start_time = max(self.gpu_states[gpu_id].available_at for gpu_id in gpu_ids)
        start_time = max(self.current_sim_time, start_time)

        # Handle model switching on primary GPU (for logging)
        primary_gpu_id = node_exec.gpu_id
        primary_gpu_state = self.gpu_states[primary_gpu_id]
        switch_time = 0.0
        
        if primary_gpu_state.current_model != model_name:
            switch_time = self._get_model_switch_time(primary_gpu_state.current_model, model_name)
            if primary_gpu_state.current_model is not None:  # Don't count initial load as switch
                self.model_switch_count += 1
                self.total_switch_time += switch_time
            logger.info(
                f"[{start_time:.2f}s] Scheduling model switch on GPU(s) {gpu_ids} "
                f"from {primary_gpu_state.current_model} to {model_name} "
                f"(switch time: {switch_time:.2f}s)"
            )

        # Calculate execution time
        execution_time = self._estimate_execution_time(node_exec, request)
        node_exec.estimated_time = execution_time
        node_exec.switch_time = switch_time

        # Set timestamps
        node_exec.start_time = start_time + switch_time
        node_exec.end_time = node_exec.start_time + execution_time

        # Update ALL GPU states involved
        for gpu_id in gpu_ids:
            gpu_state = self.gpu_states[gpu_id]
            gpu_state.available_at = node_exec.end_time
            gpu_state.busy = True
            gpu_state.current_model = model_name

        # Schedule completion event
        completion_event = Event(
            timestamp=node_exec.end_time, event_type="node_complete", data={"node_exec": node_exec, "request": request}
        )
        heapq.heappush(self.event_queue, completion_event)

        logger.info(
            f"[{self.current_sim_time:.2f}s] Scheduled node {node_exec.node_id} "
            f"of request {node_exec.request_id} on GPU(s) {gpu_ids} "
            f"(start: {node_exec.start_time:.2f}s, end: {node_exec.end_time:.2f}s)"
        )

    def _handle_node_completion(self, node_exec: NodeExecution, request: Request):
        """Handle node completion event"""
        # Mark node as completed
        node_exec.status = "completed"
        self.completed_executions.append(node_exec)

        # Update GPU state for ALL GPUs involved
        gpu_ids = node_exec.gpu_ids if node_exec.gpu_ids else [node_exec.gpu_id]
        for gpu_id in gpu_ids:
            gpu_state = self.gpu_states[gpu_id]
            if gpu_state.available_at <= self.current_sim_time:
                gpu_state.busy = False

        logger.info(
            f"[{self.current_sim_time:.2f}s] Completed node {node_exec.node_id} "
            f"of request {node_exec.request_id} on GPU(s) {gpu_ids}"
        )

        # Update ready nodes
        self._update_ready_nodes(node_exec)

        # Try to schedule more work
        self._try_schedule_ready_nodes()

    def _process_request(self, request: Request):
        """Process a single request by creating node executions"""
        workflow = self.workflows[request.workflow_id]
        dependencies = workflow.get_dependencies()

        # Track request start time
        self.request_start_times[request.request_id] = self.current_sim_time

        logger.info(f"[{self.current_sim_time:.2f}s] Processing request {request.request_id} (workflow: {workflow.id})")

        # Create node executions
        node_execs = {}
        for node in workflow.nodes:
            node_key = f"{request.request_id}_{node.id}"
            node_exec = NodeExecution(
                request_id=request.request_id, workflow_id=request.workflow_id, node_id=node.id, model_name=node.model
            )
            node_execs[node.id] = node_exec
            self.executions[node_key] = node_exec

        # Mark nodes with no dependencies as ready
        for node_id, deps in dependencies.items():
            if len(deps) == 0:
                node_execs[node_id].status = "ready"
                node_execs[node_id].ready_time = self.current_sim_time
                self.node_queue.append(node_execs[node_id])

        # Try to schedule ready nodes immediately
        self._try_schedule_ready_nodes()

    def _update_ready_nodes(self, completed_node: NodeExecution):
        """Update node statuses after a node completes"""
        request_id = completed_node.request_id
        workflow = self.workflows[completed_node.workflow_id]
        dependencies = workflow.get_dependencies()

        # Check each node to see if it's now ready
        for node in workflow.nodes:
            node_key = f"{request_id}_{node.id}"
            node_exec = self.executions.get(node_key)

            if node_exec and node_exec.status == "pending":
                # Check if all dependencies are completed
                deps = dependencies[node.id]
                all_deps_complete = True
                for dep_id in deps:
                    dep_key = f"{request_id}_{dep_id}"
                    dep_exec = self.executions.get(dep_key)
                    if not dep_exec or dep_exec.status != "completed":
                        all_deps_complete = False
                        break

                if all_deps_complete:
                    node_exec.status = "ready"
                    node_exec.ready_time = self.current_sim_time
                    self.node_queue.append(node_exec)

        # Check if request is complete
        request_complete = True
        for node in workflow.nodes:
            node_key = f"{request_id}_{node.id}"
            node_exec = self.executions.get(node_key)
            if not node_exec or node_exec.status != "completed":
                request_complete = False
                break

        if request_complete:
            self.request_end_times[request_id] = self.current_sim_time
            logger.info(f"[{self.current_sim_time:.2f}s] Request {request_id} completed")

    def _try_schedule_ready_nodes(self):
        """Try to schedule all ready nodes on available GPUs"""
        scheduled_any = True
        while scheduled_any and self.node_queue:
            scheduled_any = False

            # Try to schedule each ready node
            temp_queue = deque()
            while self.node_queue:
                node_exec = self.node_queue.popleft()

                # Find available GPU(s)
                gpu_ids = self._find_available_gpu(node_exec.model_name, self.current_sim_time)

                if gpu_ids is not None:
                    # Assign to GPU and schedule (use first GPU as primary)
                    node_exec.gpu_id = gpu_ids[0]
                    node_exec.gpu_ids = gpu_ids  # Store all GPUs for multi-GPU models
                    node_exec.status = "running"

                    # Find the corresponding request
                    request = None
                    for req_id in self.request_start_times:
                        if req_id == node_exec.request_id:
                            # This is inefficient but works for now
                            for r in self.all_requests:
                                if r.request_id == req_id:
                                    request = r
                                    break
                            break

                    if request:
                        self._schedule_node_execution(node_exec, request)
                        scheduled_any = True
                else:
                    # No GPU available, keep in queue
                    temp_queue.append(node_exec)

            # Restore queue
            self.node_queue = temp_queue

    def run(self, requests: List[Request]):
        """Run the scheduler on a list of requests"""
        logger.info(f"Starting baseline scheduler with {len(requests)} requests")
        logger.info(f"Mode: {self.mode}, Simulate: {self.simulate}")
        logger.info(f"Available GPUs: {self.gpus}")

        # Store all requests for reference
        self.all_requests = requests

        # Download and load models if in simulate mode
        if self.simulate:
            self._download_and_load_models()

        # Initialize event queue
        self.event_queue = []
        self.current_sim_time = 0.0

        # Schedule initial events based on mode
        if self.mode == "offline":
            # Process all requests at time 0
            for request in requests:
                event = Event(timestamp=0.0, event_type="request_arrival", data={"request": request})
                heapq.heappush(self.event_queue, event)
        else:
            # Online mode: schedule requests based on arrival time
            base_timestamp = requests[0].timestamp if requests else datetime.now()
            for request in requests:
                arrival_time = (request.timestamp - base_timestamp).total_seconds()
                event = Event(timestamp=arrival_time, event_type="request_arrival", data={"request": request})
                heapq.heappush(self.event_queue, event)

        # Main event processing loop
        logger.info("Starting discrete event simulation...")

        while self.event_queue:
            # Get next event
            event = heapq.heappop(self.event_queue)
            self.current_sim_time = event.timestamp

            # Process event
            if event.event_type == "request_arrival":
                self._process_request(event.data["request"])
            elif event.event_type == "node_complete":
                self._handle_node_completion(event.data["node_exec"], event.data["request"])

        # Final attempt to schedule any remaining nodes
        self._try_schedule_ready_nodes()

        logger.info(f"Simulation completed at time {self.current_sim_time:.2f}s")
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
        for gpu_id, gpu_state in self.gpu_states.items():
            # Count executions per GPU
            gpu_execs = [e for e in self.completed_executions if e.gpu_id == gpu_id]
            if gpu_execs:
                # Calculate total busy time (including switch times)
                busy_intervals = []
                for exec in sorted(gpu_execs, key=lambda e: e.start_time):
                    # Find if there was a switch before this execution
                    switch_start = exec.start_time
                    for other in gpu_execs:
                        if other.end_time <= exec.start_time and other.model_name != exec.model_name:
                            switch_time = self._get_model_switch_time(other.model_name, exec.model_name)
                            switch_start = exec.start_time - switch_time
                            break
                    busy_intervals.append((switch_start, exec.end_time))

                # Merge overlapping intervals
                merged_intervals = []
                for start, end in sorted(busy_intervals):
                    if merged_intervals and start <= merged_intervals[-1][1]:
                        merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
                    else:
                        merged_intervals.append((start, end))

                total_busy_time = sum(end - start for start, end in merged_intervals)
                utilization = total_busy_time / makespan * 100 if makespan > 0 else 0

                logger.info(
                    f"  GPU {gpu_id}: {len(gpu_execs)} executions, "
                    f"{total_busy_time:.2f}s busy time ({utilization:.1f}% utilization)"
                )

        # Model statistics
        logger.info(f"\nModel execution counts:")
        model_counts = defaultdict(int)
        for exec in self.completed_executions:
            model_counts[exec.model_name] += 1
        for model, count in sorted(model_counts.items()):
            logger.info(f"  {model}: {count} executions")
        
        # Model switching statistics
        logger.info(f"\nModel switching:")
        logger.info(f"  Total switches: {self.model_switch_count}")
        logger.info(f"  Total switch time: {self.total_switch_time:.2f} seconds")
        if makespan > 0:
            switch_overhead = (self.total_switch_time / makespan) * 100
            logger.info(f"  Switch overhead: {switch_overhead:.1f}%")

        # Write detailed log
        self._write_detailed_log()

    def _write_detailed_log(self):
        """Write detailed execution log"""
        log_file = Path("baseline_execution_log.json")

        # Calculate latency metrics
        makespan = max(e.end_time for e in self.completed_executions) if self.completed_executions else 0
        
        # Task-level metrics (waiting time and response time)
        task_waiting_times = []
        task_response_times = []
        for exec in self.completed_executions:
            # Waiting time: from when task was ready to when it started
            # For simplicity, we'll use 0 as ready time for all tasks in offline mode
            waiting_time = exec.start_time
            task_waiting_times.append(waiting_time)
            
            # Response time: from ready to completion
            response_time = exec.end_time
            task_response_times.append(response_time)
        
        # Request-level response times
        request_response_times = []
        for req_id, end_time in self.request_end_times.items():
            start_time = self.request_start_times.get(req_id, 0)
            request_response_times.append(end_time - start_time)
        
        # Calculate percentiles
        avg_waiting_time = np.mean(task_waiting_times) if task_waiting_times else 0.0
        p99_waiting_time = np.percentile(task_waiting_times, 99) if task_waiting_times else 0.0
        avg_response_time = np.mean(task_response_times) if task_response_times else 0.0
        p99_response_time = np.percentile(task_response_times, 99) if task_response_times else 0.0
        avg_request_response_time = np.mean(request_response_times) if request_response_times else 0.0
        p99_request_response_time = np.percentile(request_response_times, 99) if request_response_times else 0.0
        
        log_data = {
            "summary": {
                "total_requests": len(self.all_requests) if hasattr(self, 'all_requests') else len(self.request_end_times),
                "completed_requests": len(self.request_end_times),
                "total_nodes_executed": len(self.completed_executions),
                "mode": self.mode,
                "simulate": self.simulate,
                "gpus": self.gpus,
                "makespan": makespan,
                "total_model_switches": self.model_switch_count,
                "total_switch_time": self.total_switch_time,
                "avg_waiting_time": avg_waiting_time,
                "p99_waiting_time": p99_waiting_time,
                "avg_response_time": avg_response_time,
                "p99_response_time": p99_response_time,
                "avg_request_response_time": avg_request_response_time,
                "p99_request_response_time": p99_request_response_time,
            },
            "executions": [],
        }

        for exec in sorted(self.completed_executions, key=lambda e: e.start_time):
            # Calculate waiting time
            waiting_time = exec.start_time - exec.switch_time - exec.ready_time if exec.start_time else 0.0
            
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
                "switch_time": exec.switch_time,
                "waiting_time": waiting_time,
            }
            log_data["executions"].append(exec_data)

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"\nDetailed execution log written to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Baseline scheduler for GSwarm workflows")
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
        "--mode",
        choices=["offline", "online"],
        default="offline",
        help="Scheduling mode: offline (batch) or online (streaming)",
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
    scheduler = BaselineScheduler(gpus=gpu_list, simulate=args.simulate, mode=args.mode)

    # Load configuration
    scheduler.load_config(args.config)

    # Load requests
    requests = scheduler.load_requests(args.requests)

    # Run scheduler
    scheduler.run(requests)


if __name__ == "__main__":
    main()

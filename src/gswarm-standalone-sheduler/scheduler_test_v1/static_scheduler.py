#!/usr/bin/env python3
"""
Static Deployment Scheduler for GSwarm Workflows

This scheduler implements a static deployment strategy where:
- Each GPU permanently hosts specific models (no switching)
- Models are grouped to minimize cross-server communication
- Workflows are scheduled to complete within a single server when possible
"""

import json
import yaml
import logging
import argparse
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import heapq
import numpy as np
import sys

from scheduler_component import ModelInfo, WorkflowNode, GPUState, ScheduledTask


# Extended NodeExecution for static scheduler
@dataclass
class NodeExecution:
    """Extended execution state for a node"""

    request_id: str
    workflow_id: str
    node_id: str
    model_name: str
    estimated_time: float
    dependencies: Set[str] = field(default_factory=set)
    level: int = 0

    # Execution tracking
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    gpu_id: Optional[int] = None
    switch_time: float = 0.0  # Always 0 for static
    ready_time: float = 0.0

    @property
    def node_key(self) -> str:
        return f"{self.request_id}_{self.node_id}"

    @property
    def execution_time(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class StaticGPUState:
    """Extended GPU state for static deployment"""

    gpu_id: int
    server_id: int = 0
    assigned_models: Set[str] = field(default_factory=set)
    current_load: float = 0.0
    available_at: float = 0.0
    total_busy_time: float = 0.0
    execution_count: int = 0


@dataclass
class ModelAffinity:
    """Track model co-occurrence patterns"""

    model1: str
    model2: str
    frequency: int = 0
    total_time: float = 0.0


@dataclass
class ServerInfo:
    """Server configuration and state"""

    server_id: int
    gpu_ids: List[int]
    models: Set[str] = field(default_factory=set)

    def can_complete_workflow(self, workflow_models: Set[str]) -> bool:
        """Check if server has all models for a workflow"""
        return workflow_models.issubset(self.models)


@dataclass
class Event:
    """Event for discrete event simulation"""

    timestamp: float
    event_type: str  # "node_complete", "request_arrival"
    data: Any

    def __lt__(self, other):
        return self.timestamp < other.timestamp


class StaticScheduler:
    """Static deployment scheduler with persistent model loading"""

    def __init__(self, gpus: List[int], gpus_per_server: int = 4, simulate: bool = False, mode: str = "offline"):
        self.gpus = gpus
        self.gpus_per_server = gpus_per_server
        self.simulate = simulate
        self.mode = mode

        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, Any] = {}

        # Static deployment state
        self.gpu_states: Dict[int, StaticGPUState] = {}
        self.servers: Dict[int, ServerInfo] = {}
        self.model_to_gpus: Dict[str, List[int]] = defaultdict(list)

        # Execution tracking
        self.node_queue: deque = deque()
        self.completed_executions: List[NodeExecution] = []
        self.executions: Dict[str, NodeExecution] = {}

        # Metrics
        self.model_switch_count = 0  # Should always be 0 for static
        self.total_switch_time = 0.0  # Should always be 0 for static
        self.cross_server_comms = 0
        self.intra_server_workflows = 0
        self.inter_server_workflows = 0
        self.request_start_times: Dict[str, float] = {}
        self.request_end_times: Dict[str, float] = {}

        # Event queue for simulation
        self.event_queue: List[Event] = []
        self.current_sim_time: float = 0.0

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("StaticScheduler")
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler("static_scheduler.log", mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def load_config(self, config_path: Path):
        """Load system configuration"""
        self.logger.info(f"Loading configuration from {config_path}")

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
        self.workflows = config["workflows"]

        self.logger.info(f"Loaded {len(self.models)} models and {len(self.workflows)} workflows")

    def load_requests(self, requests_path: Path) -> List[Dict]:
        """Load workflow requests"""
        self.logger.info(f"Loading requests from {requests_path}")

        with open(requests_path, "r") as f:
            if requests_path.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        self.workflow_requests = data["requests"]
        self.logger.info(f"Loaded {len(self.workflow_requests)} workflow requests")
        return self.workflow_requests

    def analyze_workflow_patterns(self, requests: List[Dict]) -> Dict[str, Any]:
        """Analyze workflow patterns to determine model placement"""
        self.logger.info("Analyzing workflow patterns for model placement...")

        # Track model usage and co-occurrence
        model_usage = defaultdict(int)
        model_cooccurrence = defaultdict(int)
        workflow_frequency = defaultdict(int)

        for request in requests:
            workflow_id = request["workflow_id"]
            workflow_frequency[workflow_id] += 1

            if workflow_id not in self.workflows:
                continue

            workflow = self.workflows[workflow_id]
            nodes = workflow["nodes"]

            # Track model usage
            for node in nodes:
                model_usage[node["model"]] += 1

            # Track model co-occurrence in workflows
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pair = tuple(sorted([nodes[i]["model"], nodes[j]["model"]]))
                    model_cooccurrence[pair] += 1

        return {
            "model_usage": dict(model_usage),
            "model_cooccurrence": dict(model_cooccurrence),
            "workflow_frequency": dict(workflow_frequency),
        }

    def assign_models_to_gpus(self, analysis: Dict[str, Any]):
        """Assign models to GPUs based on analysis"""
        self.logger.info("Assigning models to GPUs...")

        # Initialize GPU states
        for i, gpu_id in enumerate(self.gpus):
            server_id = i // self.gpus_per_server
            self.gpu_states[gpu_id] = StaticGPUState(gpu_id=gpu_id, server_id=server_id)

        # Initialize servers
        num_servers = (len(self.gpus) + self.gpus_per_server - 1) // self.gpus_per_server
        for server_id in range(num_servers):
            start_gpu = server_id * self.gpus_per_server
            end_gpu = min(start_gpu + self.gpus_per_server, len(self.gpus))
            self.servers[server_id] = ServerInfo(server_id=server_id, gpu_ids=self.gpus[start_gpu:end_gpu])

        # Simple assignment strategy: distribute models based on usage
        # More sophisticated strategies can be implemented based on the slides
        model_usage = analysis["model_usage"]
        sorted_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)

        # Round-robin assignment with affinity consideration
        gpu_idx = 0
        for model_name, usage_count in sorted_models:
            model_info = self.models[model_name]

            # For multi-GPU models, assign to consecutive GPUs in same server
            if model_info.gpus_required > 1:
                # Find a server with enough consecutive GPUs
                assigned = False
                for server in self.servers.values():
                    available_gpus = []
                    for gpu_id in server.gpu_ids:
                        if len(self.gpu_states[gpu_id].assigned_models) == 0:
                            available_gpus.append(gpu_id)
                            if len(available_gpus) >= model_info.gpus_required:
                                # Assign to these GPUs
                                for i in range(model_info.gpus_required):
                                    self.gpu_states[available_gpus[i]].assigned_models.add(model_name)
                                    self.model_to_gpus[model_name].append(available_gpus[i])
                                server.models.add(model_name)
                                assigned = True
                                break
                    if assigned:
                        break

                if not assigned:
                    self.logger.warning(f"Could not assign multi-GPU model {model_name}")
            else:
                # Single GPU model - use round-robin
                gpu_id = self.gpus[gpu_idx % len(self.gpus)]
                self.gpu_states[gpu_id].assigned_models.add(model_name)
                self.model_to_gpus[model_name].append(gpu_id)

                # Update server models
                server_id = self.gpu_states[gpu_id].server_id
                self.servers[server_id].models.add(model_name)

                gpu_idx += 1

        # Log assignment results
        self.logger.info("Model-to-GPU assignment completed:")
        for gpu_id, gpu_state in self.gpu_states.items():
            self.logger.info(f"  GPU {gpu_id} (Server {gpu_state.server_id}): {gpu_state.assigned_models}")

    def _find_gpu_for_model(self, model_name: str, current_time: float) -> Optional[int]:
        """Find an available GPU that has the required model"""
        candidate_gpus = self.model_to_gpus.get(model_name, [])

        if not candidate_gpus:
            self.logger.error(f"No GPU has model {model_name} loaded!")
            return None

        # Find the GPU with earliest availability
        best_gpu = None
        earliest_time = float("inf")

        for gpu_id in candidate_gpus:
            if self.gpu_states[gpu_id].available_at <= earliest_time:
                earliest_time = self.gpu_states[gpu_id].available_at
                best_gpu = gpu_id

        return best_gpu

    def _estimate_execution_time(self, node: Dict, request: Dict) -> float:
        """Estimate execution time for a node"""
        node_id = node["id"]
        model_name = node["model"]

        # Use provided execution time if available
        if "node_execution_times" in request and node_id in request["node_execution_times"]:
            return request["node_execution_times"][node_id]

        # Otherwise estimate based on model info
        model_info = self.models[model_name]
        if model_info.inference_time_mean:
            return model_info.inference_time_mean

        # Default estimation
        return 5.0

    def _schedule_node_execution(self, node_exec: NodeExecution, request: Dict):
        """Schedule a node execution on appropriate GPU"""
        model_name = node_exec.model_name

        # Find GPU with this model
        gpu_id = self._find_gpu_for_model(model_name, self.current_sim_time)

        if gpu_id is None:
            self.logger.error(f"Cannot schedule node {node_exec.node_id} - no GPU has model {model_name}")
            return

        gpu_state = self.gpu_states[gpu_id]

        # No model switching in static deployment!
        switch_time = 0.0

        # Calculate when this node can start
        start_time = max(self.current_sim_time, gpu_state.available_at)

        # Get execution time
        workflow = self.workflows[node_exec.workflow_id]
        node_info = next(n for n in workflow["nodes"] if n["id"] == node_exec.node_id)
        execution_time = self._estimate_execution_time(node_info, request)

        # Set node execution details
        node_exec.gpu_id = gpu_id
        node_exec.start_time = start_time
        node_exec.end_time = start_time + execution_time
        node_exec.estimated_time = execution_time
        node_exec.switch_time = switch_time
        node_exec.status = "running"

        # Update GPU state
        gpu_state.available_at = node_exec.end_time
        gpu_state.total_busy_time += execution_time
        gpu_state.execution_count += 1

        # Schedule completion event
        completion_event = Event(
            timestamp=node_exec.end_time, event_type="node_complete", data={"node_exec": node_exec, "request": request}
        )
        heapq.heappush(self.event_queue, completion_event)

        self.logger.info(
            f"[{self.current_sim_time:.2f}s] Scheduled node {node_exec.node_id} "
            f"of request {node_exec.request_id} on GPU {gpu_id} "
            f"(Server {gpu_state.server_id}, start: {node_exec.start_time:.2f}s, end: {node_exec.end_time:.2f}s)"
        )

    def _handle_node_completion(self, node_exec: NodeExecution, request: Dict):
        """Handle node completion event"""
        node_exec.status = "completed"
        self.completed_executions.append(node_exec)

        gpu_state = self.gpu_states[node_exec.gpu_id]
        if gpu_state.available_at <= self.current_sim_time:
            gpu_state.current_load = 0.0

        self.logger.info(
            f"[{self.current_sim_time:.2f}s] Completed node {node_exec.node_id} "
            f"of request {node_exec.request_id} on GPU {node_exec.gpu_id}"
        )

        # Check for newly ready nodes
        self._update_ready_nodes(node_exec)

        # Try to schedule more work
        self._try_schedule_ready_nodes()

    def _update_ready_nodes(self, completed_node: NodeExecution):
        """Update ready nodes after completion"""
        request_id = completed_node.request_id
        workflow = self.workflows[completed_node.workflow_id]

        # Build dependency map
        dependencies = defaultdict(list)
        for edge in workflow.get("edges", []):
            dependencies[edge["to"]].append(edge["from"])

        # Check each node
        for node in workflow["nodes"]:
            node_key = f"{request_id}_{node['id']}"
            node_exec = self.executions.get(node_key)

            if node_exec and node_exec.status == "pending":
                # Check if dependencies are met
                deps_met = True
                for dep_id in dependencies.get(node["id"], []):
                    dep_key = f"{request_id}_{dep_id}"
                    dep_exec = self.executions.get(dep_key)
                    if not dep_exec or dep_exec.status != "completed":
                        deps_met = False
                        break

                if deps_met:
                    node_exec.status = "ready"
                    node_exec.ready_time = self.current_sim_time
                    self.node_queue.append(node_exec)

        # Check if request is complete
        request_complete = all(
            self.executions.get(f"{request_id}_{node['id']}", None)
            and self.executions[f"{request_id}_{node['id']}"].status == "completed"
            for node in workflow["nodes"]
        )

        if request_complete:
            self.request_end_times[request_id] = self.current_sim_time

            # Check if workflow stayed within one server
            server_ids = set()
            for node in workflow["nodes"]:
                node_key = f"{request_id}_{node['id']}"
                node_exec = self.executions[node_key]
                gpu_state = self.gpu_states[node_exec.gpu_id]
                server_ids.add(gpu_state.server_id)

            if len(server_ids) == 1:
                self.intra_server_workflows += 1
            else:
                self.inter_server_workflows += 1
                # Count cross-server communications
                for edge in workflow.get("edges", []):
                    src_key = f"{request_id}_{edge['from']}"
                    dst_key = f"{request_id}_{edge['to']}"
                    src_server = self.gpu_states[self.executions[src_key].gpu_id].server_id
                    dst_server = self.gpu_states[self.executions[dst_key].gpu_id].server_id
                    if src_server != dst_server:
                        self.cross_server_comms += 1

            self.logger.info(f"[{self.current_sim_time:.2f}s] Request {request_id} completed")

    def _try_schedule_ready_nodes(self):
        """Try to schedule ready nodes"""
        scheduled_any = True
        while scheduled_any and self.node_queue:
            scheduled_any = False
            temp_queue = deque()

            while self.node_queue:
                node_exec = self.node_queue.popleft()

                # Try to schedule
                if self._find_gpu_for_model(node_exec.model_name, self.current_sim_time) is not None:
                    # Find the matching request
                    request = next(r for r in self.all_requests if r["request_id"] == node_exec.request_id)
                    self._schedule_node_execution(node_exec, request)
                    scheduled_any = True
                else:
                    temp_queue.append(node_exec)

            self.node_queue = temp_queue

    def _process_request(self, request: Dict):
        """Process a workflow request"""
        workflow_id = request["workflow_id"]
        workflow = self.workflows[workflow_id]

        self.request_start_times[request["request_id"]] = self.current_sim_time

        self.logger.info(
            f"[{self.current_sim_time:.2f}s] Processing request {request['request_id']} (workflow: {workflow_id})"
        )

        # Create node executions
        dependencies = defaultdict(list)
        for edge in workflow.get("edges", []):
            dependencies[edge["to"]].append(edge["from"])

        node_execs = {}
        for node in workflow["nodes"]:
            node_key = f"{request['request_id']}_{node['id']}"
            # Get estimated time
            estimated_time = self._estimate_execution_time(node, request)

            node_exec = NodeExecution(
                request_id=request["request_id"],
                workflow_id=workflow_id,
                node_id=node["id"],
                model_name=node["model"],
                estimated_time=estimated_time,
                dependencies=set(dependencies.get(node["id"], [])),
            )
            node_execs[node["id"]] = node_exec
            self.executions[node_key] = node_exec

        # Mark ready nodes
        for node in workflow["nodes"]:
            if node["id"] not in dependencies or len(dependencies[node["id"]]) == 0:
                node_execs[node["id"]].status = "ready"
                node_execs[node["id"]].ready_time = self.current_sim_time
                self.node_queue.append(node_execs[node["id"]])

        # Try to schedule immediately
        self._try_schedule_ready_nodes()

    def run(self, requests: List[Dict]):
        """Run the static scheduler"""
        self.logger.info(f"Starting static scheduler with {len(requests)} requests")
        self.logger.info(f"Mode: {self.mode}, Simulate: {self.simulate}")
        self.logger.info(f"Available GPUs: {self.gpus}")
        self.logger.info(f"GPUs per server: {self.gpus_per_server}")

        # Phase 1: Analyze workflow patterns
        analysis = self.analyze_workflow_patterns(requests)

        # Phase 2: Assign models to GPUs
        self.assign_models_to_gpus(analysis)

        # Store requests for reference
        self.all_requests = requests

        # Initialize event queue
        self.event_queue = []
        self.current_sim_time = 0.0

        # Schedule initial events
        if self.mode == "offline":
            # All requests at time 0
            for request in requests:
                event = Event(timestamp=0.0, event_type="request_arrival", data={"request": request})
                heapq.heappush(self.event_queue, event)
        else:
            # Online mode: use timestamps
            base_time = datetime.fromisoformat(requests[0]["timestamp"])
            for request in requests:
                arrival_time = (datetime.fromisoformat(request["timestamp"]) - base_time).total_seconds()
                event = Event(timestamp=arrival_time, event_type="request_arrival", data={"request": request})
                heapq.heappush(self.event_queue, event)

        # Main simulation loop
        self.logger.info("Starting discrete event simulation...")

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_sim_time = event.timestamp

            if event.event_type == "request_arrival":
                self._process_request(event.data["request"])
            elif event.event_type == "node_complete":
                self._handle_node_completion(event.data["node_exec"], event.data["request"])

        # Final scheduling attempt
        self._try_schedule_ready_nodes()

        self.logger.info(f"Simulation completed at time {self.current_sim_time:.2f}s")
        self._print_metrics()

    def _print_metrics(self):
        """Print performance metrics"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXECUTION METRICS")
        self.logger.info("=" * 60)

        if not self.completed_executions:
            self.logger.warning("No executions completed")
            return

        makespan = max(e.end_time for e in self.completed_executions)
        self.logger.info(f"Total execution time (makespan): {makespan:.2f} seconds")

        # Static deployment specific metrics
        self.logger.info(f"\nStatic deployment metrics:")
        total_workflows = self.intra_server_workflows + self.inter_server_workflows
        if total_workflows > 0:
            server_efficiency = (self.intra_server_workflows / total_workflows) * 100
            self.logger.info(f"  Server efficiency: {server_efficiency:.1f}%")
            self.logger.info(f"  Intra-server workflows: {self.intra_server_workflows}")
            self.logger.info(f"  Inter-server workflows: {self.inter_server_workflows}")
            self.logger.info(f"  Cross-server communications: {self.cross_server_comms}")

        # Request completion times
        if self.request_end_times:
            request_times = []
            for req_id, end_time in self.request_end_times.items():
                start_time = self.request_start_times[req_id]
                request_times.append(end_time - start_time)

            self.logger.info(f"\nRequest completion times:")
            self.logger.info(f"  Average: {np.mean(request_times):.2f}s")
            self.logger.info(f"  Median: {np.median(request_times):.2f}s")
            self.logger.info(f"  P99: {np.percentile(request_times, 99):.2f}s")
            self.logger.info(f"  Min: {np.min(request_times):.2f}s")
            self.logger.info(f"  Max: {np.max(request_times):.2f}s")

        # GPU utilization
        self.logger.info(f"\nGPU utilization:")
        utilizations = []
        for gpu_id, gpu_state in self.gpu_states.items():
            utilization = (gpu_state.total_busy_time / makespan * 100) if makespan > 0 else 0
            utilizations.append(utilization)
            self.logger.info(
                f"  GPU {gpu_id} (Server {gpu_state.server_id}): "
                f"{gpu_state.execution_count} executions, "
                f"{gpu_state.total_busy_time:.2f}s busy time ({utilization:.1f}% utilization), "
                f"Models: {gpu_state.assigned_models}"
            )

        # Overall utilization stats
        self.logger.info(f"\nOverall GPU statistics:")
        self.logger.info(f"  Average utilization: {np.mean(utilizations):.1f}%")
        self.logger.info(f"  Utilization variance: {np.var(utilizations):.1f}")

        # Model execution counts
        self.logger.info(f"\nModel execution counts:")
        model_counts = defaultdict(int)
        for exec in self.completed_executions:
            model_counts[exec.model_name] += 1
        for model, count in sorted(model_counts.items()):
            self.logger.info(f"  {model}: {count} executions")

        # Model switching (should be zero)
        self.logger.info(f"\nModel switching:")
        self.logger.info(f"  Total switches: {self.model_switch_count}")
        self.logger.info(f"  Total switch time: {self.total_switch_time:.2f} seconds")
        self.logger.info(f"  Switch overhead: 0.0%")

        # Write detailed log
        self._write_detailed_log()

    def _write_detailed_log(self):
        """Write detailed execution log in the same format as other schedulers"""
        log_file = Path("static_execution_log.json")

        # Calculate metrics
        makespan = max(e.end_time for e in self.completed_executions) if self.completed_executions else 0

        # Task-level metrics
        task_waiting_times = []
        task_response_times = []
        for exec in self.completed_executions:
            waiting_time = exec.start_time - exec.ready_time if exec.start_time and exec.ready_time else 0.0
            task_waiting_times.append(waiting_time)
            response_time = exec.end_time - exec.ready_time if exec.end_time and exec.ready_time else 0.0
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

        # Static deployment specific metrics
        total_workflows = self.intra_server_workflows + self.inter_server_workflows
        server_efficiency = (self.intra_server_workflows / total_workflows * 100) if total_workflows > 0 else 0.0

        log_data = {
            "summary": {
                "total_requests": len(self.all_requests)
                if hasattr(self, "all_requests")
                else len(self.request_end_times),
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
                # Static deployment specific
                "server_efficiency": server_efficiency,
                "intra_server_workflows": self.intra_server_workflows,
                "inter_server_workflows": self.inter_server_workflows,
                "cross_server_communications": self.cross_server_comms,
                "gpus_per_server": self.gpus_per_server,
            },
            "executions": [],
            "gpu_assignments": {},
            "server_assignments": {},
        }

        # Add GPU assignments
        for gpu_id, gpu_state in self.gpu_states.items():
            log_data["gpu_assignments"][str(gpu_id)] = {
                "server_id": gpu_state.server_id,
                "assigned_models": list(gpu_state.assigned_models),
                "utilization": (gpu_state.total_busy_time / makespan * 100) if makespan > 0 else 0,
            }

        # Add server assignments
        for server_id, server_info in self.servers.items():
            log_data["server_assignments"][str(server_id)] = {
                "gpu_ids": server_info.gpu_ids,
                "models": list(server_info.models),
            }

        # Add executions
        for exec in sorted(self.completed_executions, key=lambda e: e.start_time):
            waiting_time = exec.start_time - exec.ready_time if exec.start_time and exec.ready_time else 0.0

            exec_data = {
                "request_id": exec.request_id,
                "workflow_id": exec.workflow_id,
                "node_id": exec.node_id,
                "model_name": exec.model_name,
                "gpu_id": exec.gpu_id,
                "server_id": self.gpu_states[exec.gpu_id].server_id,
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

        self.logger.info(f"\nDetailed execution log written to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Static deployment scheduler for GSwarm workflows")
    parser.add_argument("--gpus", type=int, required=True, help="Number of available GPUs")
    parser.add_argument("--gpus-per-server", type=int, default=4, help="Number of GPUs per server (default: 4)")
    parser.add_argument(
        "--simulate",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use simulation mode with actual model calls (true/false)",
    )
    parser.add_argument(
        "--mode", choices=["offline", "online"], default="offline", help="Scheduling mode (default: offline)"
    )
    parser.add_argument("--config", type=Path, default=Path("simple_config.json"), help="System configuration file")
    parser.add_argument("--requests", type=Path, default=Path("simple_requests.yaml"), help="Workflow requests file")

    args = parser.parse_args()

    # Create scheduler
    scheduler = StaticScheduler(
        gpus=list(range(args.gpus)), gpus_per_server=args.gpus_per_server, simulate=args.simulate, mode=args.mode
    )

    # Load configuration
    scheduler.load_config(args.config)

    # Load and run requests
    requests = scheduler.load_requests(args.requests)
    scheduler.run(requests)


if __name__ == "__main__":
    main()

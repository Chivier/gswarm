"""
Base classes and interfaces for GSwarm schedulers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
from gswarm.utils import DeviceInfo, parse_device, format_device


class SchedulingStrategy(Enum):
    """Available scheduling strategies."""
    BASELINE = "baseline"  # Ray-like: one-by-one execution
    OFFLINE = "offline"    # Batch processing with optimization
    ONLINE = "online"      # Real-time with P99 optimization
    STATIC = "static"      # Fixed model deployment


@dataclass
class ModelInfo:
    """Model configuration information."""
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
    """Workflow node definition."""
    id: str
    model: str
    inputs: List[str]
    outputs: List[str]
    config_options: Optional[List[str]] = None


@dataclass
class WorkflowEdge:
    """Workflow edge definition."""
    from_node: str
    to_node: str


@dataclass
class Workflow:
    """Workflow definition."""
    id: str
    name: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]

    def get_dependencies(self) -> Dict[str, Set[str]]:
        """Get dependency map: node -> set of nodes it depends on."""
        deps = defaultdict(set)
        for edge in self.edges:
            deps[edge.to_node].add(edge.from_node)
        # Add nodes with no dependencies
        for node in self.nodes:
            if node.id not in deps:
                deps[node.id] = set()
        return dict(deps)

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_dependents(self, node_id: str) -> Set[str]:
        """Get nodes that depend on the given node."""
        dependents = set()
        for edge in self.edges:
            if edge.from_node == node_id:
                dependents.add(edge.to_node)
        return dependents

    def get_model_requirements(self) -> Set[str]:
        """Get all unique models required by this workflow."""
        return {node.model for node in self.nodes}


@dataclass
class Request:
    """User request for workflow execution."""
    id: str
    workflow_id: str
    arrival_time: float
    priority: int = 0
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """A task scheduled for execution."""
    request_id: str
    workflow_id: str
    node_id: str
    model_name: str
    device: str  # Full device specification (e.g., "node1:cuda:0")
    scheduled_time: float
    estimated_duration: float
    actual_start_time: Optional[float] = None
    actual_end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    dependencies: Set[str] = field(default_factory=set)
    
    # Legacy compatibility
    @property
    def gpu_id(self) -> int:
        """Legacy property for backward compatibility."""
        device_info = parse_device(self.device)
        return device_info.device_id

    @property
    def task_id(self) -> str:
        """Unique task identifier."""
        return f"{self.request_id}_{self.node_id}"

    @property
    def actual_duration(self) -> Optional[float]:
        """Actual execution duration if completed."""
        if self.actual_start_time and self.actual_end_time:
            return self.actual_end_time - self.actual_start_time
        return None


@dataclass
class ExecutionMetrics:
    """Metrics for scheduler performance."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Timing metrics
    total_execution_time: float = 0.0
    average_request_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p99_latency: float = 0.0
    
    # Resource utilization
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    model_switch_count: int = 0
    total_model_load_time: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    tasks_per_second: float = 0.0


class SchedulerBase(ABC):
    """Abstract base class for all schedulers."""
    
    def __init__(
        self, 
        devices: Union[List[int], List[str]], 
        models: Dict[str, ModelInfo], 
        simulate: bool = False,
        default_client: str = "localhost"
    ):
        """
        Initialize the scheduler.
        
        Args:
            devices: List of devices available for scheduling.
                    Can be GPU IDs (legacy) or device strings (e.g., ["node1:cuda:0", "node2:cuda:1"])
            models: Dictionary of model configurations
            simulate: Whether to run in simulation mode
            default_client: Default client name for legacy GPU IDs
        """
        # Convert devices to normalized format
        self.devices: List[str] = []
        self.device_infos: List[DeviceInfo] = []
        
        for device in devices:
            if isinstance(device, int):
                # Legacy format: convert GPU ID to device string
                device_str = format_device(default_client, "cuda", device)
                self.devices.append(device_str)
                self.device_infos.append(DeviceInfo(default_client, "cuda", device))
            else:
                # New format: parse and normalize
                device_info = parse_device(device, default_client)
                self.devices.append(device_info.full_name)
                self.device_infos.append(device_info)
        
        self.num_devices = len(self.devices)
        self.models = models
        self.simulate = simulate
        self.metrics = ExecutionMetrics()
        
        # Legacy compatibility
        self.gpus = [d.device_id for d in self.device_infos if d.device_type == "cuda"]
        self.num_gpus = len(self.gpus)
        
        # Workflow definitions
        self.workflows: Dict[str, Workflow] = {}
        
        # Execution state
        self.pending_requests: List[Request] = []
        self.active_requests: Dict[str, Request] = {}
        self.completed_requests: Set[str] = set()
        self.scheduled_tasks: List[ScheduledTask] = []
        
    @abstractmethod
    def schedule(self, requests: List[Request]) -> List[ScheduledTask]:
        """
        Schedule a batch of requests.
        
        Args:
            requests: List of workflow execution requests
            
        Returns:
            List of scheduled tasks with GPU assignments and timing
        """
        pass
    
    @abstractmethod
    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler (for online scheduling).
        
        Args:
            request: New workflow execution request
        """
        pass
    
    @abstractmethod
    def get_next_task(self) -> Optional[ScheduledTask]:
        """
        Get the next task to execute (for online scheduling).
        
        Returns:
            Next task to execute, or None if no tasks ready
        """
        pass
    
    def add_workflow(self, workflow: Workflow) -> None:
        """Add a workflow definition."""
        self.workflows[workflow.id] = workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def update_metrics(self, task: ScheduledTask) -> None:
        """Update execution metrics based on completed task."""
        if task.status == "completed":
            self.metrics.completed_tasks += 1
            if task.actual_duration:
                self.metrics.total_execution_time += task.actual_duration
        elif task.status == "failed":
            self.metrics.failed_tasks += 1
            
    def get_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self.metrics = ExecutionMetrics()
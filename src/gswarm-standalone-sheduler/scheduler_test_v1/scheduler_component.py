"""
Scheduler Components - Basic data structures and class definitions
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict


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
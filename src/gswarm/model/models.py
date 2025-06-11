"""
Data models and structures for gswarm_model system.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Supported model types"""

    LLM = "llm"
    DIFFUSION = "diffusion"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class ModelStatus(str, Enum):
    """Model status states"""

    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    MOVING = "moving"
    SERVING = "serving"
    ERROR = "error"
    DELETED = "deleted"


class StorageType(str, Enum):
    """Storage device types"""

    WEB = "web"
    DISK = "disk"
    DRAM = "dram"
    GPU = "gpu"


class ActionType(str, Enum):
    """Job action types"""

    DOWNLOAD = "download"
    MOVE = "move"
    COPY = "copy"
    SERVE = "serve"
    STOP_SERVE = "stop_serve"
    DELETE = "delete"
    HEALTH_CHECK = "health_check"


class JobStatus(str, Enum):
    """Job execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelRequirements(BaseModel):
    """Model resource requirements"""

    min_memory: Optional[int] = Field(None, description="Minimum system memory in bytes")
    min_vram: Optional[int] = Field(None, description="Minimum GPU memory in bytes")
    gpu_arch: Optional[List[str]] = Field(None, description="Supported GPU architectures")


class ModelMetadata(BaseModel):
    """Model metadata"""

    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    requirements: Optional[ModelRequirements] = None


class HostModelInfo(BaseModel):
    """Model information stored on host node"""

    model_name: str
    model_type: ModelType
    model_size: Optional[int] = Field(None, description="Size in bytes")
    model_hash: Optional[str] = Field(None, description="Content hash for integrity")
    stored_locations: List[str] = Field(default_factory=list, description="List of device_name")
    available_services: Dict[str, str] = Field(default_factory=dict, description="device_name -> service_url")
    metadata: Optional[ModelMetadata] = None


class ClientModelInfo(BaseModel):
    """Model information stored on client node"""

    model_name: str
    stored_locations: List[str] = Field(default_factory=list, description="Local storage locations")
    status: ModelStatus = ModelStatus.AVAILABLE
    service_port: Optional[int] = Field(None, description="Port if serving")
    last_accessed: datetime = Field(default_factory=datetime.now)
    local_path: Optional[str] = Field(None, description="Filesystem path")
    size: Optional[int] = Field(None, description="Actual size on disk")
    integrity_hash: Optional[str] = Field(None, description="Verification hash")


class StorageInfo(BaseModel):
    """Storage device information"""

    total: int = Field(description="Total capacity in bytes")
    used: int = Field(description="Used capacity in bytes")
    available: int = Field(description="Available capacity in bytes")

    @property
    def utilization_percent(self) -> float:
        """Calculate utilization percentage"""
        if self.total == 0:
            return 0.0
        return (self.used / self.total) * 100.0


class NodeInfo(BaseModel):
    """Node information and capabilities"""

    node_id: str
    hostname: str
    ip_address: Optional[str] = None
    storage_devices: Dict[str, StorageInfo] = Field(default_factory=dict)
    gpu_info: List[Dict[str, Any]] = Field(default_factory=list)
    last_seen: datetime = Field(default_factory=datetime.now)
    is_online: bool = True


class JobAction(BaseModel):
    """Individual action in a job workflow"""

    action_id: str
    action_type: ActionType
    model_name: str
    devices: List[str] = Field(default_factory=list, description="Target devices for action")
    dependencies: List[str] = Field(default_factory=list, description="Action IDs this depends on")
    source_url: Optional[str] = Field(None, description="Source URL for download")
    port: Optional[int] = Field(None, description="Port for serving")
    target_url: Optional[str] = Field(None, description="Target URL for health check")
    keep_source: Optional[bool] = Field(None, description="Keep source when moving/copying")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional action configuration")
    status: JobStatus = JobStatus.PENDING
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class Job(BaseModel):
    """Model execution workflow job"""

    job_id: str
    name: str
    description: Optional[str] = None
    actions: List[JobAction]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# API Request/Response Models


class RegisterModelRequest(BaseModel):
    """Request to register a new model"""

    model_type: ModelType
    source_url: Optional[str] = None
    metadata: Optional[ModelMetadata] = None


class DownloadModelRequest(BaseModel):
    """Request to download a model"""

    source: str = Field(description="Source: 'web' or 'node_id:device'")
    target_device: str = Field(description="Target storage device")
    source_url: Optional[str] = Field(None, description="URL if source is 'web'")
    priority: str = Field(default="normal", description="Priority: high, normal, low")


class MoveModelRequest(BaseModel):
    """Request to move a model between devices"""

    from_device: str
    to_device: str
    keep_source: bool = False


class ServeModelRequest(BaseModel):
    """Request to serve a model"""

    port: int
    device: str
    config: Optional[Dict[str, Any]] = None


class CreateJobRequest(BaseModel):
    """Request to create a job"""

    name: str
    description: Optional[str] = None
    actions: List[JobAction]


class ModelSummary(BaseModel):
    """Summary of a model for listing"""

    model_name: str
    model_type: ModelType
    size: Optional[int] = None
    locations: List[str] = Field(default_factory=list)
    services: List[str] = Field(default_factory=list)
    status: str = "available"


class ListModelsResponse(BaseModel):
    """Response for listing models"""

    models: List[ModelSummary]
    total_count: int


class SystemStatusResponse(BaseModel):
    """System-wide status response"""

    total_nodes: int
    online_nodes: int
    total_models: int
    active_services: int
    storage_utilization: Dict[str, float] = Field(default_factory=dict)


class NodeStatusResponse(BaseModel):
    """Node status response"""

    nodes: List[NodeInfo]


# Utility functions for device naming


def parse_device_name(device_name: str) -> tuple[str, str, Optional[str]]:
    """
    Parse device name into components.

    Args:
        device_name: Device name like 'node1:disk', 'node1:gpu0', 'web'

    Returns:
        Tuple of (node_identifier, storage_type, index)
    """
    if device_name == "web":
        return "web", "web", None

    parts = device_name.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid device name format: {device_name}")

    node_id, storage_part = parts

    # Extract index from storage type (e.g., gpu0 -> gpu, 0)
    if storage_part.startswith("gpu"):
        storage_type = "gpu"
        index = storage_part[3:] if len(storage_part) > 3 else "0"
    else:
        storage_type = storage_part
        index = None

    return node_id, storage_type, index


def format_device_name(node_id: str, storage_type: str, index: Optional[str] = None) -> str:
    """
    Format device name from components.

    Args:
        node_id: Node identifier
        storage_type: Storage type (disk, dram, gpu)
        index: Device index (for GPU)

    Returns:
        Formatted device name
    """
    if node_id == "web":
        return "web"

    if storage_type == "gpu" and index is not None:
        return f"{node_id}:gpu{index}"
    else:
        return f"{node_id}:{storage_type}"

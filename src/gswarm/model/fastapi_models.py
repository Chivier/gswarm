"""
Updated data models for FastAPI-based gswarm_model system.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid


class ModelType(str, Enum):
    """Supported model types"""

    LLM = "llm"
    DIFFUSION = "diffusion"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class StorageType(str, Enum):
    """Storage device types - GPU removed as it's not for storage"""

    DISK = "disk"
    DRAM = "dram"


class CopyMethod(str, Enum):
    """Copy methods between devices"""

    DISK_TO_DRAM = "disk_to_dram"
    DRAM_TO_DISK = "dram_to_disk"
    GPU_TO_DRAM = "gpu_to_dram"
    DRAM_TO_GPU = "dram_to_gpu"
    DISK_TO_GPU = "disk_to_gpu"
    GPU_TO_DISK = "gpu_to_disk"
    GPU_TO_GPU = "gpu_to_gpu"


class ActionType(str, Enum):
    """Job action types"""

    DOWNLOAD = "download"
    COPY = "copy"
    SERVE = "serve"
    STOP_SERVE = "stop_serve"
    DELETE = "delete"
    HEALTH_CHECK = "health_check"


class ModelStatus(str, Enum):
    REGISTERED = "registered"  # Just registered, not downloaded yet
    DOWNLOADING = "downloading"  # Currently downloading
    DISK = "disk"  # Available on disk
    DRAM = "dram"  # Available in DRAM
    COPYING = "copying"  # Being copied between devices
    SERVING = "serving"  # Currently serving on GPU
    ERROR = "error"  # Error state


# Request/Response Models


class ModelInstance(BaseModel):
    """Instance of a model with unique ID"""

    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    type: ModelType
    size: Optional[int] = Field(None, description="Size in bytes")
    checkpoints: Dict[str, str] = Field(
        default_factory=dict, description="Storage locations: device -> path (only disk/dram)"
    )
    serving_instances: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Active services: instance_id -> {device, url, port, pid}"
    )
    metadata: Optional[Dict[str, Any]] = None
    status: ModelStatus = ModelStatus.REGISTERED
    created_at: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """Basic model information (backward compatibility)"""

    name: str
    type: ModelType
    size: Optional[int] = Field(None, description="Size in bytes")
    locations: List[str] = Field(default_factory=list, description="Storage locations (disk/dram only)")
    services: Dict[str, str] = Field(default_factory=dict, description="Active services: device -> url")
    metadata: Optional[Dict[str, Any]] = None
    status: str = "registered"
    download_progress: Optional[Dict[str, Any]] = None
    path_validated: Optional[bool] = Field(None, description="Whether the model paths have been validated")


class NodeInfo(BaseModel):
    """Node information"""

    node_id: str
    hostname: str
    ip_address: str
    storage_devices: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Device info")
    gpu_devices: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="GPU info: gpu0 -> {memory, compute_capability}"
    )
    gpu_count: int = 0
    last_seen: datetime = Field(default_factory=datetime.now)


class RegisterModelRequest(BaseModel):
    """Request to register a model"""

    name: str
    type: ModelType
    source_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DownloadRequest(BaseModel):
    """Request to download a model"""

    model_name: str
    source_url: str
    target_device: str  # Must be disk or dram


class CopyRequest(BaseModel):
    """Request to copy a model between devices"""

    model_name: str
    source_device: str
    target_device: str
    method: Optional[CopyMethod] = None  # Auto-detect if not specified
    keep_source: bool = True  # Keep source by default
    instance_id: Optional[str] = None  # For serving multiple instances
    use_pinned_memory: bool = True  # For disk->gpu transfers


class ServeRequest(BaseModel):
    """Request to serve a model on GPU"""

    model_name: str
    source_device: str  # Where to load from (disk/dram)
    gpu_device: str  # Target GPU (e.g., gpu0, gpu1)
    port: int
    instance_id: Optional[str] = Field(None, description="Specific instance ID, auto-generated if not provided")
    config: Optional[Dict[str, Any]] = None


class StopServeRequest(BaseModel):
    """Request to stop serving a specific instance"""

    model_name: str
    instance_id: str  # Specific instance to stop


class JobRequest(BaseModel):
    """Simple job request"""

    name: str
    description: Optional[str] = None
    actions: List[Dict[str, Any]]


class StandardResponse(BaseModel):
    """Standard API response"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ModelServingStatus(BaseModel):
    """Status of a serving model instance"""

    instance_id: str
    model_name: str
    gpu_device: str
    port: int
    url: str
    pid: int
    status: str  # running, loading, error
    started_at: datetime
    last_health_check: Optional[datetime] = None
    memory_usage: Optional[Dict[str, Any]] = None  # GPU memory stats

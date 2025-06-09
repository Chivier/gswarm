"""
Simplified data models for FastAPI-based gswarm_model system.
"""

from typing import Dict, List, Optional, Any
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


class StorageType(str, Enum):
    """Storage device types"""
    DISK = "disk"
    DRAM = "dram"
    GPU = "gpu"


class ActionType(str, Enum):
    """Job action types"""
    DOWNLOAD = "download"
    MOVE = "move"
    SERVE = "serve"
    STOP_SERVE = "stop_serve"
    DELETE = "delete"
    HEALTH_CHECK = "health_check"


class ModelStatus(str, Enum):
    REGISTERED = "registered"     # Just registered, not downloaded yet
    DOWNLOADING = "downloading"   # Currently downloading
    READY = "ready"              # Available for use
    MOVING = "moving"           # Being moved between devices
    SERVING = "serving"         # Currently serving
    ERROR = "error"             # Error state


# Request/Response Models

class ModelInfo(BaseModel):
    """Basic model information"""
    name: str
    type: ModelType
    size: Optional[int] = Field(None, description="Size in bytes")
    locations: List[str] = Field(default_factory=list, description="Storage locations")
    services: Dict[str, str] = Field(default_factory=dict, description="Active services: device -> url")
    metadata: Optional[Dict[str, Any]] = None
    status: str = "registered"  # ✅ Add status field
    download_progress: Optional[Dict[str, Any]] = None  # ✅ Add progress field


class NodeInfo(BaseModel):
    """Node information"""
    node_id: str
    hostname: str
    ip_address: str
    storage_devices: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Device info")
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
    target_device: str


class MoveRequest(BaseModel):
    """Request to move a model"""
    model_name: str
    source_device: str
    target_device: str
    keep_source: bool = False


class ServeRequest(BaseModel):
    """Request to serve a model"""
    model_name: str
    device: str
    port: int
    config: Optional[Dict[str, Any]] = None


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
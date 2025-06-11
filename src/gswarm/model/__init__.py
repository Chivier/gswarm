"""
GSwarm Model Manager - Updated FastAPI version
"""

from gswarm.model.fastapi_models import (
    ModelType,
    StorageType,
    CopyMethod,
    ActionType,
    ModelStatus,
    ModelInstance,
    ModelInfo,
    NodeInfo,
    RegisterModelRequest,
    DownloadRequest,
    CopyRequest,
    ServeRequest,
    StopServeRequest,
    JobRequest,
    StandardResponse,
    ModelServingStatus,
)

from gswarm.model.fastapi_client import ModelClient
from gswarm.model.fastapi_head import app as head_app

__all__ = [
    # Enums
    "ModelType",
    "StorageType",
    "CopyMethod",
    "ActionType",
    "ModelStatus",
    # Models
    "ModelInstance",
    "ModelInfo",
    "NodeInfo",
    "RegisterModelRequest",
    "DownloadRequest",
    "CopyRequest",
    "ServeRequest",
    "StopServeRequest",
    "JobRequest",
    "StandardResponse",
    "ModelServingStatus",
    # Client
    "ModelClient",
    # Head app
    "head_app",
]

__version__ = "0.4.0"

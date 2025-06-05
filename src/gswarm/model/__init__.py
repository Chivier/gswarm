"""
GSwarm Model Manager - Simplified FastAPI version
"""

from gswarm.model.fastapi_models import (
    ModelType, StorageType, ActionType,
    ModelInfo, NodeInfo, RegisterModelRequest,
    DownloadRequest, MoveRequest, ServeRequest,
    JobRequest, StandardResponse
)

from gswarm.model.fastapi_client import ModelClient
from gswarm.model.fastapi_head import app as head_app

__all__ = [
    # Models
    "ModelType", "StorageType", "ActionType",
    "ModelInfo", "NodeInfo", "RegisterModelRequest",
    "DownloadRequest", "MoveRequest", "ServeRequest",
    "JobRequest", "StandardResponse",
    
    # Client
    "ModelClient",
    
    # Head app
    "head_app"
]

__version__ = "0.3.0"

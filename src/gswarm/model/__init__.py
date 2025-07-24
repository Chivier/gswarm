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
from gswarm.model.cost_models import (
    CostModel,
    LLMCostModel,
    SDCostModel,
    get_estimation_cost,
    update_predictor,
)

# Custom service framework
from gswarm.model.base_service import ModelService, BatchModelService, ServiceConfig
from gswarm.model.service_manager import ServiceManager, get_service_manager

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
    # Cost models
    "CostModel",
    "LLMCostModel",
    "SDCostModel",
    "get_estimation_cost",
    "update_predictor",
    # Custom service framework
    "ModelService",
    "BatchModelService",
    "ServiceConfig",
    "ServiceManager",
    "get_service_manager",
]

__version__ = "0.4.0"

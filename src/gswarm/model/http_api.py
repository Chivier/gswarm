"""
HTTP API for gswarm_model head node.
Provides REST endpoints for model management operations.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
import yaml
import json
import uuid
from datetime import datetime
from loguru import logger

# Import the head module's state and models
from gswarm.model.fastapi_head import state
from gswarm.model.models import (
    RegisterModelRequest,
    DownloadModelRequest,
    MoveModelRequest,
    ServeModelRequest,
    CreateJobRequest,
    ModelSummary,
    ListModelsResponse,
    SystemStatusResponse,
    NodeStatusResponse,
    JobAction,
    ActionType,
    ModelType,
    Job,
)

app = FastAPI(
    title="GSwarm Model Manager HTTP API", description="HTTP API for GSwarm Model Management System", version="0.1.0"
)


# Response models
class StandardResponse(BaseModel):
    """Standard API response"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    """Job operation response"""

    success: bool
    message: str
    job_id: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model information response"""

    model_name: str
    model_type: str
    model_size: Optional[int] = None
    model_hash: Optional[str] = None
    stored_locations: List[str] = []
    available_services: Dict[str, str] = {}
    metadata: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "GSwarm Model Manager HTTP API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Model Registry Management


@app.get("/models", response_model=ListModelsResponse)
async def list_models():
    """List all registered models"""
    try:
        models = []
        async with state.registry_lock:
            for model_name, model_info in state.model_registry.items():
                model_summary = ModelSummary(
                    model_name=model_info.model_name,
                    model_type=model_info.model_type,
                    size=model_info.model_size,
                    locations=model_info.stored_locations,
                    services=list(model_info.available_services.values()),
                    status="available",
                )
                models.append(model_summary)

        return ListModelsResponse(models=models, total_count=len(models))

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str):
    """Get detailed model information"""
    try:
        async with state.registry_lock:
            model_info = state.model_registry.get(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

            return ModelInfoResponse(
                model_name=model_info.model_name,
                model_type=model_info.model_type.value,
                model_size=model_info.model_size,
                model_hash=model_info.model_hash,
                stored_locations=model_info.stored_locations,
                available_services=model_info.available_services,
                metadata=model_info.metadata.model_dump() if model_info.metadata else None,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/register", response_model=StandardResponse)
async def register_model(model_name: str, request: RegisterModelRequest):
    """Register a new model in the system"""
    try:
        from gswarm.model.models import HostModelInfo, ModelMetadata

        async with state.registry_lock:
            if model_name in state.model_registry:
                return StandardResponse(success=False, message=f"Model {model_name} is already registered")

            # Create model info
            model_info = HostModelInfo(model_name=model_name, model_type=request.model_type, metadata=request.metadata)

            state.model_registry[model_name] = model_info
            logger.info(f"Registered model {model_name} of type {request.model_type}")

        if state.enable_persistence:
            from gswarm.model.fastapi_head import save_registry

            await save_registry()

        return StandardResponse(success=True, message=f"Model {model_name} registered successfully")

    except Exception as e:
        logger.error(f"Error registering model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}", response_model=StandardResponse)
async def unregister_model(model_name: str):
    """Unregister model and cleanup all instances"""
    try:
        async with state.registry_lock:
            if model_name not in state.model_registry:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

            del state.model_registry[model_name]
            logger.info(f"Unregistered model {model_name}")

        if state.enable_persistence:
            from gswarm.model.head import save_registry

            await save_registry()

        return StandardResponse(success=True, message=f"Model {model_name} unregistered successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Location Management


@app.get("/models/{model_name}/locations")
async def get_model_locations(model_name: str):
    """Get all storage locations for a model"""
    try:
        async with state.registry_lock:
            model_info = state.model_registry.get(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

            return {
                "model_name": model_name,
                "locations": model_info.stored_locations,
                "total_locations": len(model_info.stored_locations),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting locations for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/locations/{device_name}", response_model=StandardResponse)
async def add_model_location(model_name: str, device_name: str):
    """Track a new storage location for a model"""
    try:
        async with state.registry_lock:
            model_info = state.model_registry.get(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

            if device_name not in model_info.stored_locations:
                model_info.stored_locations.append(device_name)
                logger.info(f"Added location {device_name} for model {model_name}")

        if state.enable_persistence:
            from gswarm.model.head import save_registry

            await save_registry()

        return StandardResponse(success=True, message=f"Location {device_name} added for model {model_name}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding location for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}/locations/{device_name}", response_model=StandardResponse)
async def remove_model_location(model_name: str, device_name: str):
    """Remove a storage location record"""
    try:
        async with state.registry_lock:
            model_info = state.model_registry.get(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

            if device_name in model_info.stored_locations:
                model_info.stored_locations.remove(device_name)
                logger.info(f"Removed location {device_name} for model {model_name}")

        if state.enable_persistence:
            from gswarm.model.head import save_registry

            await save_registry()

        return StandardResponse(success=True, message=f"Location {device_name} removed for model {model_name}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing location for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Service Management


@app.get("/models/{model_name}/services")
async def get_model_services(model_name: str):
    """Get all active service endpoints for a model"""
    try:
        async with state.registry_lock:
            model_info = state.model_registry.get(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

            return {
                "model_name": model_name,
                "services": model_info.available_services,
                "total_services": len(model_info.available_services),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting services for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/services", response_model=JobResponse)
async def request_model_serving(model_name: str, request: ServeModelRequest):
    """Request model serving on specified nodes"""
    try:
        # Create a simple serve job
        job_id = str(uuid.uuid4())

        serve_action = JobAction(
            action_id=f"serve_{model_name}",
            action_type=ActionType.SERVE,
            model_name=model_name,
            devices=[request.device],
            port=request.port,
            config=request.config,
        )

        job = Job(
            job_id=job_id,
            name=f"Serve {model_name}",
            description=f"Serve model {model_name} on {request.device}",
            actions=[serve_action],
        )

        async with state.job_lock:
            state.jobs[job_id] = job
            # Start job execution
            from gswarm.model.head import execute_job

            task = asyncio.create_task(execute_job(job))
            state.active_jobs[job_id] = task

        logger.info(f"Created serve job {job_id} for model {model_name}")

        return JobResponse(success=True, message=f"Serve job created for model {model_name}", job_id=job_id)

    except Exception as e:
        logger.error(f"Error creating serve job for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}/services/{device_name}", response_model=StandardResponse)
async def stop_model_service(model_name: str, device_name: str):
    """Stop model service on specified device"""
    try:
        async with state.registry_lock:
            model_info = state.model_registry.get(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

            if device_name in model_info.available_services:
                del model_info.available_services[device_name]
                logger.info(f"Stopped service for model {model_name} on {device_name}")

        if state.enable_persistence:
            from gswarm.model.head import save_registry

            await save_registry()

        return StandardResponse(success=True, message=f"Service stopped for model {model_name} on {device_name}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping service for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Job Management


@app.post("/jobs", response_model=JobResponse)
async def create_job(request: CreateJobRequest):
    """Create and execute a job workflow"""
    try:
        job_id = str(uuid.uuid4())

        job = Job(job_id=job_id, name=request.name, description=request.description, actions=request.actions)

        async with state.job_lock:
            state.jobs[job_id] = job
            # Start job execution
            from gswarm.model.head import execute_job

            task = asyncio.create_task(execute_job(job))
            state.active_jobs[job_id] = task

        logger.info(f"Created and started job {job_id}: {request.name}")

        return JobResponse(success=True, message=f"Job {job_id} created and started", job_id=job_id)

    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/from-yaml", response_model=JobResponse)
async def create_job_from_yaml(file: UploadFile = File(...)):
    """Create job from YAML file"""
    try:
        content = await file.read()
        job_def = yaml.safe_load(content)

        # Convert YAML to job format
        actions = []
        for action_data in job_def.get("actions", []):
            # Handle optional model_name for certain action types
            action_type = ActionType(action_data["action_type"])
            model_name = action_data.get("model_name")

            # For health_check actions, model_name is optional - use placeholder if not provided
            if action_type == ActionType.HEALTH_CHECK and model_name is None:
                model_name = "health_check_placeholder"

            action = JobAction(
                action_id=action_data["action_id"],
                action_type=action_type,
                model_name=model_name,
                devices=action_data.get("devices", []),
                dependencies=action_data.get("dependencies", []),
                source_url=action_data.get("source_url"),
                port=action_data.get("port"),
                target_url=action_data.get("target_url"),
                keep_source=action_data.get("keep_source", False),
                config=action_data.get("config"),
            )
            actions.append(action)

        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, name=job_def["name"], description=job_def.get("description"), actions=actions)

        async with state.job_lock:
            state.jobs[job_id] = job
            # Start job execution
            from gswarm.model.head import execute_job

            task = asyncio.create_task(execute_job(job))
            state.active_jobs[job_id] = task

        logger.info(f"Created job {job_id} from YAML: {job_def['name']}")

        return JobResponse(success=True, message=f"Job {job_id} created from YAML", job_id=job_id)

    except Exception as e:
        logger.error(f"Error creating job from YAML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get job execution status"""
    try:
        async with state.job_lock:
            job = state.jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            completed_actions = sum(1 for action in job.actions if action.status.value == "completed")

            return {
                "job_id": job_id,
                "name": job.name,
                "status": job.status.value,
                "error_message": job.error_message,
                "completed_actions": completed_actions,
                "total_actions": len(job.actions),
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "actions": [
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type.value,
                        "status": action.status.value,
                        "error_message": action.error_message,
                    }
                    for action in job.actions
                ],
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/{job_id}/cancel", response_model=StandardResponse)
async def cancel_job(job_id: str):
    """Cancel running job"""
    try:
        async with state.job_lock:
            job = state.jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            # Cancel the job task if it's running
            if job_id in state.active_jobs:
                task = state.active_jobs[job_id]
                task.cancel()
                del state.active_jobs[job_id]

            # Update job status
            from gswarm.model.models import JobStatus

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

        logger.info(f"Cancelled job {job_id}")

        return StandardResponse(success=True, message=f"Job {job_id} cancelled successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    try:
        jobs = []
        async with state.job_lock:
            for job_id, job in state.jobs.items():
                completed_actions = sum(1 for action in job.actions if action.status.value == "completed")
                jobs.append(
                    {
                        "job_id": job_id,
                        "name": job.name,
                        "status": job.status.value,
                        "completed_actions": completed_actions,
                        "total_actions": len(job.actions),
                        "created_at": job.created_at.isoformat(),
                    }
                )

        return {"jobs": jobs, "total_count": len(jobs)}

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System Status


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system-wide model management status"""
    try:
        async with state.registry_lock:
            total_nodes = len(state.node_registry)
            online_nodes = sum(1 for node in state.node_registry.values() if node.is_online)
            total_models = len(state.model_registry)

            # Count active services
            active_services = 0
            for model_info in state.model_registry.values():
                active_services += len(model_info.available_services)

            # Calculate storage utilization by node
            storage_utilization = {}
            for node_id, node_info in state.node_registry.items():
                for device_name, storage_info in node_info.storage_devices.items():
                    key = f"{node_id}:{device_name}"
                    storage_utilization[key] = storage_info.utilization_percent

        return SystemStatusResponse(
            total_nodes=total_nodes,
            online_nodes=online_nodes,
            total_models=total_models,
            active_services=active_services,
            storage_utilization=storage_utilization,
        )

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nodes", response_model=NodeStatusResponse)
async def get_nodes():
    """List all connected nodes and their capabilities"""
    try:
        nodes = []
        async with state.registry_lock:
            for node_id, node_info in state.node_registry.items():
                nodes.append(node_info)

        return NodeStatusResponse(nodes=nodes)

    except Exception as e:
        logger.error(f"Error getting nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_http_server(host: str, port: int):
    """Run the HTTP API server"""
    config = uvicorn.Config(app, host=host, port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)
    logger.info(f"Starting HTTP API server on {host}:{port}")
    await server.serve()

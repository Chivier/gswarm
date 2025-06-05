"""
Simplified FastAPI head node for gswarm_model system.
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
import asyncio
import uuid
from datetime import datetime
from loguru import logger

from .fastapi_models import (
    ModelInfo, NodeInfo, RegisterModelRequest, DownloadRequest,
    MoveRequest, ServeRequest, JobRequest, StandardResponse
)


app = FastAPI(
    title="GSwarm Model Manager",
    description="Simplified Model Management API",
    version="0.3.0"
)


class HeadState:
    """Simple state management"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.nodes: Dict[str, NodeInfo] = {}
        self.jobs: Dict[str, Dict] = {}
        

state = HeadState()


# Basic endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "GSwarm Model Manager API", "version": "0.3.0"}


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Model management

@app.get("/models")
async def list_models():
    """List all models"""
    return {"models": list(state.models.values()), "count": len(state.models)}


@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """Get model info"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return state.models[model_name]


@app.post("/models")
async def register_model(request: RegisterModelRequest):
    """Register a new model"""
    if request.name in state.models:
        return StandardResponse(
            success=False,
            message=f"Model {request.name} already exists"
        )
    
    model = ModelInfo(
        name=request.name,
        type=request.type,
        metadata=request.metadata
    )
    state.models[request.name] = model
    
    logger.info(f"Registered model: {request.name}")
    return StandardResponse(
        success=True,
        message=f"Model {request.name} registered successfully"
    )


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    del state.models[model_name]
    logger.info(f"Deleted model: {model_name}")
    
    return StandardResponse(
        success=True,
        message=f"Model {model_name} deleted successfully"
    )


# Model operations

@app.post("/download")
async def download_model(request: DownloadRequest):
    """Download a model to a device"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    model = state.models[request.model_name]
    
    # Simulate download (in real implementation, this would trigger actual download)
    logger.info(f"Downloading {request.model_name} from {request.source_url} to {request.target_device}")
    
    # Update model locations
    if request.target_device not in model.locations:
        model.locations.append(request.target_device)
    
    return StandardResponse(
        success=True,
        message=f"Download initiated for {request.model_name}"
    )


@app.post("/move")
async def move_model(request: MoveRequest):
    """Move a model between devices"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    model = state.models[request.model_name]
    
    if request.source_device not in model.locations:
        raise HTTPException(status_code=400, detail=f"Model not found on {request.source_device}")
    
    # Simulate move
    logger.info(f"Moving {request.model_name} from {request.source_device} to {request.target_device}")
    
    # Update locations
    if request.target_device not in model.locations:
        model.locations.append(request.target_device)
    
    if not request.keep_source and request.source_device in model.locations:
        model.locations.remove(request.source_device)
    
    return StandardResponse(
        success=True,
        message=f"Move initiated for {request.model_name}"
    )


@app.post("/serve")
async def serve_model(request: ServeRequest):
    """Start serving a model"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    model = state.models[request.model_name]
    
    if request.device not in model.locations:
        raise HTTPException(status_code=400, detail=f"Model not found on {request.device}")
    
    # Simulate serving
    logger.info(f"Serving {request.model_name} on {request.device}:{request.port}")
    
    # Update services
    service_url = f"http://{request.device}:{request.port}"
    model.services[request.device] = service_url
    
    return StandardResponse(
        success=True,
        message=f"Model {request.model_name} now serving on {service_url}"
    )


@app.post("/stop/{model_name}/{device}")
async def stop_serving(model_name: str, device: str):
    """Stop serving a model on a device"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = state.models[model_name]
    
    if device in model.services:
        del model.services[device]
        logger.info(f"Stopped serving {model_name} on {device}")
    
    return StandardResponse(
        success=True,
        message=f"Stopped serving {model_name} on {device}"
    )


# Node management

@app.post("/nodes")
async def register_node(node: NodeInfo):
    """Register a node"""
    state.nodes[node.node_id] = node
    logger.info(f"Registered node: {node.node_id}")
    
    return StandardResponse(
        success=True,
        message=f"Node {node.node_id} registered successfully"
    )


@app.get("/nodes")
async def list_nodes():
    """List all nodes"""
    return {"nodes": list(state.nodes.values()), "count": len(state.nodes)}


@app.post("/nodes/{node_id}/heartbeat")
async def node_heartbeat(node_id: str):
    """Update node heartbeat"""
    if node_id not in state.nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    state.nodes[node_id].last_seen = datetime.now()
    return {"status": "ok"}


# Job management (simplified)

@app.post("/jobs")
async def create_job(request: JobRequest):
    """Create a job"""
    job_id = str(uuid.uuid4())
    
    job = {
        "id": job_id,
        "name": request.name,
        "description": request.description,
        "actions": request.actions,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    state.jobs[job_id] = job
    logger.info(f"Created job: {job_id}")
    
    # In real implementation, this would trigger job execution
    asyncio.create_task(execute_job(job_id))
    
    return {
        "job_id": job_id,
        "message": f"Job {request.name} created successfully"
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return state.jobs[job_id]


async def execute_job(job_id: str):
    """Execute a job (simplified)"""
    job = state.jobs[job_id]
    job["status"] = "running"
    
    # Simulate job execution
    await asyncio.sleep(2)
    
    job["status"] = "completed"
    job["completed_at"] = datetime.now().isoformat()
    logger.info(f"Job {job_id} completed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100) 
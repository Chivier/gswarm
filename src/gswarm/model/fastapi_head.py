"""
Simplified FastAPI head node for gswarm model system.
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
import asyncio
import uuid
from datetime import datetime
from loguru import logger
import subprocess
from pathlib import Path
import signal
import psutil
import os

from gswarm.model.fastapi_models import (
    ModelInfo, NodeInfo, RegisterModelRequest, DownloadRequest,
    MoveRequest, ServeRequest, JobRequest, StandardResponse, ModelStatus
)


app = FastAPI(
    title="GSwarm Model Manager",
    description="Simplified Model Management API",
    version="0.3.0"
)


class VLLMServer:
    """Track vLLM server instances"""
    def __init__(self, model_name: str, device: str, port: int, process: subprocess.Popen, model_path: str):
        self.model_name = model_name
        self.device = device
        self.port = port
        self.process = process
        self.model_path = model_path
        self.started_at = datetime.now()
        self.pid = process.pid
    
    def is_running(self) -> bool:
        """Check if the vLLM server is still running"""
        try:
            return self.process.poll() is None
        except:
            return False
    
    def stop(self):
        """Stop the vLLM server"""
        try:
            if self.is_running():
                logger.info(f"Stopping vLLM server for {self.model_name} (PID: {self.pid})")
                # Try graceful shutdown first
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop gracefully
                    logger.warning(f"Force killing vLLM server {self.pid}")
                    self.process.kill()
                    self.process.wait()
                logger.info(f"vLLM server for {self.model_name} stopped")
        except Exception as e:
            logger.error(f"Error stopping vLLM server: {e}")


class HeadState:
    """Simple state management"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.nodes: Dict[str, NodeInfo] = {}
        self.jobs: Dict[str, Dict] = {}
        self.vllm_servers: Dict[str, VLLMServer] = {}  # key: f"{model_name}:{device}"
        

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
    """List all models with status"""
    models_with_status = []
    
    for model in state.models.values():
        models_with_status.append({
            "name": model.name,
            "type": model.type,
            "size": model.size,
            "locations": model.locations,
            "services": model.services,
            "metadata": model.metadata,
            "status": model.status,  # ✅ Include status in list
            "download_progress": model.download_progress  # ✅ Include progress
        })
    
    return {"models": models_with_status, "count": len(models_with_status)}


@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """Get model info with status and progress"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = state.models[model_name]
    
    # ✅ Return model with status and progress
    return {
        "name": model.name,
        "type": model.type,
        "size": model.size,
        "locations": model.locations,
        "services": model.services,
        "metadata": model.metadata,
        "status": model.status,  # ✅ Include status
        "download_progress": model.download_progress  # ✅ Include progress
    }


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
    
    # ✅ DON'T add to locations yet - just set status to downloading
    model.status = ModelStatus.DOWNLOADING
    model.download_progress = {
        "target_device": request.target_device,
        "source_url": request.source_url,
        "started_at": datetime.now().isoformat(),
        "progress_percent": 0
    }
    
    logger.info(f"Starting download: {request.model_name} from {request.source_url} to {request.target_device}")
    
    # ✅ Start actual download asynchronously
    asyncio.create_task(perform_actual_download(request.model_name, request.source_url, request.target_device))
    
    return StandardResponse(
        success=True,
        message=f"Download started for {request.model_name}"
    )


async def perform_actual_download(model_name: str, source_url: str, target_device: str):
    """Actually download the model (async)"""
    model = state.models[model_name]
    
    try:
        # Parse target device (e.g., "node1:disk" -> node1, disk)
        if ":" in target_device:
            node_id, device_type = target_device.split(":", 1)
        else:
            node_id = "local"
            device_type = target_device
        
        # ✅ Determine download path based on device type
        if device_type == "disk":
            # Use HuggingFace cache for disk storage
            from gswarm.utils.cache import get_model_cache_dir
            download_path = get_model_cache_dir()
        else:
            download_path = Path(f"/tmp/gswarm/models/{model_name}")
            download_path.mkdir(parents=True, exist_ok=True)
        
        # ✅ Extract model repo from HuggingFace URL
        if "huggingface.co" in source_url:
            # Extract org/model from https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
            repo_id = source_url.replace("https://huggingface.co/", "")
            
            # ✅ Use huggingface-hub to download
            await download_from_huggingface(model_name, repo_id, download_path)
        else:
            # Handle other sources (S3, HTTP, etc.)
            await download_from_url(model_name, source_url, download_path)
        
        # ✅ Download completed successfully
        model.status = ModelStatus.READY
        if target_device not in model.locations:
            model.locations.append(target_device)
        
        # Update with actual file info
        model.download_progress = {
            "completed_at": datetime.now().isoformat(),
            "progress_percent": 100,
            "local_path": str(download_path),
            "repo_id": repo_id if "huggingface.co" in source_url else None
        }
        
        logger.info(f"✅ Download completed: {model_name} -> {target_device}")
        
        # ✅ Notify client node if needed
        await notify_client_download_complete(model_name, target_device)
        
    except Exception as e:
        # ✅ Handle download failure
        model.status = ModelStatus.ERROR
        model.download_progress["error"] = str(e)
        model.download_progress["failed_at"] = datetime.now().isoformat()
        
        logger.error(f"❌ Download failed: {model_name} - {e}")


async def download_from_huggingface(model_name: str, repo_id: str, download_path: Path):
    """Download model from HuggingFace using huggingface-hub"""
    
    # ✅ Use subprocess to call huggingface-cli download
    cmd = [
        "huggingface-cli", "download",
        repo_id,
        "--cache-dir", str(download_path),
        "--resume-download"
    ]
    
    # ✅ Run download with progress tracking
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # ✅ Monitor progress (simplified)
    model = state.models[model_name]
    progress = 0
    
    while process.returncode is None:
        await asyncio.sleep(1)
        progress = min(progress + 5, 95)  # Simulate progress
        model.download_progress["progress_percent"] = progress
        logger.info(f"Download progress: {model_name} - {progress}%")
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise Exception(f"HuggingFace download failed: {stderr.decode()}")
    
    logger.info(f"HuggingFace download completed: {repo_id}")


async def download_from_url(model_name: str, source_url: str, download_path: Path):
    """Download from generic URL"""
    raise NotImplementedError("Generic URL download not implemented yet")


async def notify_client_download_complete(model_name: str, target_device: str):
    """Notify client node that download is complete"""
    if ":" in target_device:
        node_id, device = target_device.split(":", 1)
        
        # ✅ In a real implementation, you'd send a message to the client node
        # For now, just log
        logger.info(f"📡 Notifying {node_id}: Model {model_name} ready on {device}")
        
        # TODO: Implement actual client notification via gRPC/HTTP


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


def get_model_path(model_name: str, device: str) -> Optional[str]:
    """Get the actual model path for serving"""
    model = state.models.get(model_name)
    if not model:
        return None
    
    # Check if we have download progress with repo_id (preferred)
    download_progress = model.download_progress
    if download_progress and "repo_id" in download_progress:
        return download_progress["repo_id"]
    
    # Check if we have download progress with local path
    if download_progress and "local_path" in download_progress:
        return download_progress["local_path"]
    
    # Try to infer from metadata
    metadata = model.metadata
    if "url" in metadata and "huggingface.co" in metadata["url"]:
        # Extract repo_id from HuggingFace URL
        repo_id = metadata["url"].replace("https://huggingface.co/", "")
        return repo_id
    
    # Fall back to model name (assume it's a HuggingFace repo)
    return model_name


def extract_gpu_device_id(device: str) -> Optional[int]:
    """Extract GPU device ID from device string (e.g., 'node1:gpu4' -> 4, 'gpu0' -> 0)"""
    try:
        if ":" in device:
            device = device.split(":", 1)[1]
        
        if device.startswith("gpu"):
            return int(device[3:])  # Extract number from 'gpu4' -> 4
        return None
    except:
        return None


@app.post("/serve")
async def serve_model(request: ServeRequest):
    """Start serving a model using vLLM"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    model = state.models[request.model_name]
    
    if request.device not in model.locations:
        raise HTTPException(status_code=400, detail=f"Model not found on {request.device}")
    
    # Check if already serving on this device
    server_key = f"{request.model_name}:{request.device}"
    if server_key in state.vllm_servers:
        existing_server = state.vllm_servers[server_key]
        if existing_server.is_running():
            return StandardResponse(
                success=False,
                message=f"Model {request.model_name} already serving on {request.device}:{existing_server.port}"
            )
        else:
            # Clean up dead server
            del state.vllm_servers[server_key]
            if request.device in model.services:
                del model.services[request.device]
    
    # Get model path
    model_path = get_model_path(request.model_name, request.device)
    if not model_path:
        raise HTTPException(status_code=400, detail=f"Could not determine model path for {request.model_name}")
    
    try:
        # Prepare vLLM command
        vllm_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", str(request.port),
            "--trust-remote-code",  # For custom models
        ]
        
        # Add GPU device specification if applicable
        gpu_id = extract_gpu_device_id(request.device)
        env = {}
        if gpu_id is not None:
            vllm_cmd.extend(["--tensor-parallel-size", "1"])
            # Set CUDA_VISIBLE_DEVICES environment variable
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        logger.info(f"Starting vLLM server: {' '.join(vllm_cmd)}")
        logger.info(f"Environment: {env}")
        
        # Start vLLM server
        process = subprocess.Popen(
            vllm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **env} if env else None,
            preexec_fn=os.setsid  # Create new process group for easier cleanup
        )
        
        # Wait a moment to check if it started successfully
        await asyncio.sleep(2)
        
        if process.poll() is not None:
            # Process died immediately
            stdout, stderr = process.communicate()
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"vLLM server failed to start: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Failed to start vLLM server: {error_msg}")
        
        # Create server tracker
        vllm_server = VLLMServer(
            model_name=request.model_name,
            device=request.device,
            port=request.port,
            process=process,
            model_path=model_path
        )
        
        state.vllm_servers[server_key] = vllm_server
        
        # Update services
        service_url = f"http://0.0.0.0:{request.port}"
        model.services[request.device] = service_url
        
        logger.info(f"✅ vLLM server started for {request.model_name} on {request.device}:{request.port} (PID: {process.pid})")
        
        # Start monitoring task
        asyncio.create_task(monitor_vllm_server(server_key))
        
        return StandardResponse(
            success=True,
            message=f"Model {request.model_name} now serving on {service_url} (PID: {process.pid})"
        )
        
    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start vLLM server: {str(e)}")


async def monitor_vllm_server(server_key: str):
    """Monitor vLLM server and clean up if it dies"""
    while server_key in state.vllm_servers:
        server = state.vllm_servers[server_key]
        
        if not server.is_running():
            logger.warning(f"vLLM server for {server.model_name} on {server.device} has died")
            
            # Clean up
            del state.vllm_servers[server_key]
            
            # Remove from model services
            model = state.models.get(server.model_name)
            if model and server.device in model.services:
                del model.services[server.device]
            
            break
        
        await asyncio.sleep(5)  # Check every 5 seconds


@app.post("/stop/{model_name}/{device}")
async def stop_serving(model_name: str, device: str):
    """Stop serving a model on a device"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = state.models[model_name]
    server_key = f"{model_name}:{device}"
    
    # Stop vLLM server if running
    if server_key in state.vllm_servers:
        vllm_server = state.vllm_servers[server_key]
        vllm_server.stop()
        del state.vllm_servers[server_key]
    
    # Remove from services
    if device in model.services:
        del model.services[device]
        logger.info(f"Stopped serving {model_name} on {device}")
    
    return StandardResponse(
        success=True,
        message=f"Stopped serving {model_name} on {device}"
    )


@app.get("/servers")
async def list_servers():
    """List all running vLLM servers"""
    servers = []
    for server_key, server in state.vllm_servers.items():
        servers.append({
            "model_name": server.model_name,
            "device": server.device,
            "port": server.port,
            "pid": server.pid,
            "started_at": server.started_at.isoformat(),
            "is_running": server.is_running(),
            "model_path": server.model_path
        })
    
    return {"servers": servers, "count": len(servers)}


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


@app.get("/models/{model_name}/status")
async def get_download_status(model_name: str):
    """Get model download/status information"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = state.models[model_name]
    
    return {
        "model_name": model_name,
        "status": model.status,
        "locations": model.locations,
        "download_progress": model.download_progress,
        "services": model.services
    }


# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up vLLM servers on shutdown"""
    logger.info("Shutting down, stopping all vLLM servers...")
    for server in state.vllm_servers.values():
        server.stop()


def create_app() -> FastAPI:
    """Factory function to create and return the FastAPI app instance"""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100) 
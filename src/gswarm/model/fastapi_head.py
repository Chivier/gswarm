"""
Updated FastAPI head node for gswarm model system with enhanced configuration and model management.
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional, Tuple
import asyncio
import uuid
from datetime import datetime
from loguru import logger
import subprocess
from pathlib import Path
import signal
import psutil
import os
import aiohttp
import threading
import shutil
import torch

from gswarm.model.fastapi_models import (
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
    ModelStatus,
    StorageType,
    CopyMethod,
    ModelServingStatus,
)

# Import enhanced utilities
from gswarm.utils.config import load_config, get_model_cache_dir, get_dram_cache_dir, HostConfig, GSwarmConfig
from gswarm.utils.cache import model_storage, scan_all_models, save_model_to_disk, load_safetensors_to_dram


app = FastAPI(title="GSwarm Model Manager", description="Simplified Model Management API", version="0.4.0")


class VLLMServer:
    """Track vLLM server instances with unique IDs"""

    def __init__(
        self, instance_id: str, model_name: str, gpu_device: str, port: int, process: subprocess.Popen, model_path: str
    ):
        self.instance_id = instance_id
        self.model_name = model_name
        self.gpu_device = gpu_device
        self.port = port
        self.process = process
        self.model_path = model_path
        self.started_at = datetime.now()
        self.pid = process.pid
        self.is_ready = False
        self.last_health_check = None
        self.stdout_thread = None
        self.stderr_thread = None
        self._stop_monitoring = threading.Event()

    def is_running(self) -> bool:
        """Check if the vLLM server is still running"""
        try:
            return self.process.poll() is None
        except:
            return False

    def stop(self):
        """Stop the vLLM server"""
        try:
            self._stop_monitoring.set()  # Stop monitoring threads

            if self.is_running():
                logger.info(f"Stopping vLLM server instance {self.instance_id} for {self.model_name} (PID: {self.pid})")
                # Try graceful shutdown first
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop gracefully
                    logger.warning(f"Force killing vLLM server {self.pid}")
                    self.process.kill()
                    self.process.wait()
                logger.info(f"vLLM server instance {self.instance_id} stopped")
        except Exception as e:
            logger.error(f"Error stopping vLLM server: {e}")


class HeadState:
    """Enhanced state management with model variable storage"""

    def __init__(self):
        self.models: Dict[str, ModelInstance] = {}  # model_name -> ModelInstance
        self.nodes: Dict[str, NodeInfo] = {}
        self.jobs: Dict[str, Dict] = {}
        self.vllm_servers: Dict[str, VLLMServer] = {}  # instance_id -> VLLMServer
        self.model_checkpoints: Dict[str, Dict[str, str]] = {}  # model_name -> {device: path}
        self.discovery_completed = False

    async def discover_and_register_models(self):
        """Discover and register existing models on startup with path validation"""
        try:
            logger.info("Discovering existing models...")
            discovered_models = scan_all_models()

            for model_info in discovered_models:
                model_name = model_info["model_name"]

                # Skip if already registered
                if model_name in self.models:
                    continue

                # Create model instance with REGISTERED status initially
                model = ModelInstance(
                    model_name=model_name,
                    type=model_info["model_type"],
                    size=model_info["size"],
                    metadata={"source": model_info["source"], "discovered_at_startup": True},
                    status=ModelStatus.REGISTERED,
                )

                # Add checkpoint location and validate
                checkpoint_path = None
                if model_info["source"] == "gswarm_cache":
                    checkpoint_path = model_info["local_path"]
                elif model_info["source"] == "huggingface_cache":
                    # Copy HF model to gswarm cache if not already there
                    gswarm_path = get_model_cache_dir() / model_name.replace("/", "--")
                    if not gswarm_path.exists():
                        logger.info(f"Copying HF model {model_name} to gswarm cache...")
                        try:
                            shutil.copytree(model_info["local_path"], gswarm_path)
                            checkpoint_path = str(gswarm_path)
                        except Exception as e:
                            # Skip this model silently if copy fails
                            logger.debug(f"Skipping {model_name}: failed to copy from HF cache - {e}")
                            continue
                    else:
                        checkpoint_path = str(gswarm_path)

                # Validate the checkpoint path before adding
                if checkpoint_path and validate_model_path(checkpoint_path):
                    model.checkpoints["disk"] = checkpoint_path
                    model.status = ModelStatus.DISK
                    self.models[model_name] = model
                    logger.info(f"Auto-registered model: {model_name} - ✓ DISK")
                else:
                    # Skip models with invalid paths silently
                    logger.debug(f"Skipping {model_name}: no valid model files found at {checkpoint_path}")
                    continue

            logger.info(f"Discovered and registered {len(self.models)} valid models")

        except Exception as e:
            logger.error(f"Error during model discovery: {e}")


state = HeadState()


# Helper functions


def get_storage_path(model_name: str, device: str) -> Path:
    """Get storage path for model checkpoint"""
    # Extract device type from node:device format
    if ":" in device:
        _, device_type = device.split(":", 1)
    else:
        device_type = device

    if device_type == "disk":
        return get_model_cache_dir() / model_name.replace("/", "--")
    elif device_type == "dram":
        return get_dram_cache_dir() / model_name.replace("/", "--")
    else:
        raise ValueError(f"Invalid storage device: {device}")


def extract_gpu_device_id(device: str) -> Optional[int]:
    """Extract GPU device ID from device string (e.g., 'gpu0' -> 0, 'node1:gpu4' -> 4)"""
    try:
        if ":" in device:
            device = device.split(":", 1)[1]

        if device.startswith("gpu"):
            return int(device[3:])
        return None
    except:
        return None


def validate_storage_device(device: str) -> bool:
    """Validate that device is a valid storage location (disk/dram only)"""
    if ":" in device:
        _, device_type = device.split(":", 1)
    else:
        device_type = device

    return device_type in ["disk", "dram"]


def infer_copy_method(source: str, target: str) -> CopyMethod:
    """Infer the copy method based on source and target devices"""
    # Extract device types
    source_type = source.split(":")[-1] if ":" in source else source
    target_type = target.split(":")[-1] if ":" in target else target

    # Remove numeric suffix from gpu devices
    if source_type.startswith("gpu"):
        source_type = "gpu"
    if target_type.startswith("gpu"):
        target_type = "gpu"

    mapping = {
        ("disk", "dram"): CopyMethod.DISK_TO_DRAM,
        ("dram", "disk"): CopyMethod.DRAM_TO_DISK,
        ("gpu", "dram"): CopyMethod.GPU_TO_DRAM,
        ("dram", "gpu"): CopyMethod.DRAM_TO_GPU,
        ("disk", "gpu"): CopyMethod.DISK_TO_GPU,
        ("gpu", "disk"): CopyMethod.GPU_TO_DISK,
        ("gpu", "gpu"): CopyMethod.GPU_TO_GPU,
    }

    method = mapping.get((source_type, target_type))
    if not method:
        raise ValueError(f"Invalid copy operation from {source} to {target}")

    return method


def validate_model_path(path: str) -> bool:
    """Validate that a model path exists and contains valid model files"""
    try:
        model_path = Path(path)
        if not model_path.exists():
            return False

        # Check if it's a directory with model files or a single model file
        if model_path.is_dir():
            # Check for common model files
            has_model_files = any(
                (model_path / filename).exists()
                for filename in [
                    "config.json",
                    "pytorch_model.bin",
                    "model.safetensors",
                    "pytorch_model-00001-of-*.bin",
                    "model-00001-of-*.safetensors",
                ]
            )
            # Also check for any .bin or .safetensors files
            has_weight_files = any(model_path.glob("*.bin") or model_path.glob("*.safetensors"))
            return has_model_files or has_weight_files
        else:
            # Single file - check if it's a model file
            return model_path.suffix in [".bin", ".safetensors", ".pt", ".pth"]
    except Exception as e:
        logger.error(f"Error validating model path {path}: {e}")
        return False


def validate_model_checkpoints(model: ModelInstance) -> bool:
    """Validate all checkpoints for a model and remove invalid ones"""
    valid_checkpoints = {}
    has_valid_checkpoint = False

    for device, path in model.checkpoints.items():
        if validate_model_path(path):
            valid_checkpoints[device] = path
            has_valid_checkpoint = True
            logger.debug(f"✓ Valid checkpoint for {model.model_name} on {device}: {path}")
        else:
            logger.warning(f"✗ Invalid checkpoint for {model.model_name} on {device}: {path}")

    # Update model checkpoints to only include valid ones
    model.checkpoints = valid_checkpoints
    return has_valid_checkpoint


def update_model_status_based_on_checkpoints(model: ModelInstance) -> None:
    """Update model status based on available valid checkpoints using disk/dram priority"""
    if not model.checkpoints:
        # No valid checkpoints
        if model.status in [ModelStatus.DISK, ModelStatus.DRAM, ModelStatus.SERVING]:
            model.status = ModelStatus.REGISTERED
            logger.info(f"Model {model.model_name} status changed to REGISTERED (no valid checkpoints)")
    else:
        # Has valid checkpoints
        if len(model.serving_instances) > 0:
            model.status = ModelStatus.SERVING
        elif model.status in [ModelStatus.REGISTERED, ModelStatus.ERROR]:
            # Prioritize DRAM over DISK if both are available
            if "dram" in model.checkpoints:
                model.status = ModelStatus.DRAM
                logger.info(f"Model {model.model_name} status changed to DRAM (available in DRAM)")
            elif "disk" in model.checkpoints:
                model.status = ModelStatus.DISK
                logger.info(f"Model {model.model_name} status changed to DISK (available on disk)")


def discover_model_cache_path(model_name: str) -> Optional[str]:
    """Discover the actual cache path for a model by checking common locations"""
    # Check gswarm cache
    gswarm_path = get_model_cache_dir() / model_name.replace("/", "--")
    if validate_model_path(str(gswarm_path)):
        return str(gswarm_path)

    # Check DRAM cache
    dram_path = get_dram_cache_dir() / model_name.replace("/", "--")
    if validate_model_path(str(dram_path)):
        return str(dram_path)

    return None


# Basic endpoints


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "GSwarm Model Manager API", "version": "0.5.0", "config_loaded": True}


@app.get("/health")
async def health():
    """Health check with memory usage"""
    memory_usage = model_storage.get_memory_usage()
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "memory_usage": memory_usage}


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "model_cache_dir": str(get_model_cache_dir()),
        "dram_cache_dir": str(get_dram_cache_dir()),
        "model_manager_port": load_config().host.model_manager_port,
    }


# Model management


@app.get("/models")
async def list_models():
    """List all models with their checkpoints and serving instances"""
    models_info = []

    for model in state.models.values():
        # Count total serving instances
        serving_count = len(model.serving_instances)

        # Check if model is in DRAM
        dram_loaded = model.model_name in model_storage.list_dram_models()

        # Check if all paths are valid
        paths_validated = all(validate_model_path(path) for path in model.checkpoints.values())

        models_info.append(
            {
                "name": model.model_name,
                "type": model.type,
                "size": model.size,
                "checkpoints": list(model.checkpoints.keys()),  # Only show devices, not paths
                "serving_instances": serving_count,
                "status": model.status,
                "dram_loaded": dram_loaded,
                "metadata": model.metadata,
                "created_at": model.created_at.isoformat(),
                "paths_validated": paths_validated,
                "has_valid_cache": len(model.checkpoints) > 0 and paths_validated,
            }
        )

    return {"models": models_info, "count": len(models_info)}


@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """Get detailed model info including all instances"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = state.models[model_name]

    # Get serving instances details
    serving_details = []
    for instance_id, info in model.serving_instances.items():
        serving_details.append(
            {
                "instance_id": instance_id,
                "gpu_device": info["device"],
                "url": info["url"],
                "port": info["port"],
                "pid": info.get("pid"),
                "status": info.get("status", "unknown"),
            }
        )

    # Check if model is loaded in DRAM
    dram_model = model_storage.get_dram_model(model_name)

    return {
        "name": model.model_name,
        "type": model.type,
        "size": model.size,
        "checkpoints": model.checkpoints,
        "serving_instances": serving_details,
        "status": model.status,
        "dram_loaded": dram_model is not None,
        "metadata": model.metadata,
        "created_at": model.created_at.isoformat(),
    }


@app.post("/models")
async def register_model(request: RegisterModelRequest):
    """Register a new model with path validation"""
    if request.name in state.models:
        return StandardResponse(success=False, message=f"Model {request.name} already exists")

    # Create model with REGISTERED status initially
    model = ModelInstance(model_name=request.name, type=request.type, metadata=request.metadata)

    # Try to discover existing cache path for this model
    discovered_path = discover_model_cache_path(request.name)
    status_info = {"status": "registered"}

    if discovered_path:
        model.checkpoints["disk"] = discovered_path
        model.status = ModelStatus.DISK
        status_info.update(
            {
                "status": "disk",
                "discovered_path": discovered_path,
                "message": "Model found in cache and marked as available on disk",
            }
        )
        logger.info(f"Registered model {request.name} with discovered path: {discovered_path}")
    else:
        logger.info(f"Registered model {request.name} without valid cache path - use download to make it available")
        status_info["message"] = "Model registered but no valid cache found - download required"

    state.models[request.name] = model

    return StandardResponse(
        success=True,
        message=f"Model {request.name} registered successfully",
        data={"instance_id": model.instance_id, **status_info},
    )


# Download operations


@app.post("/download")
async def download_model(request: DownloadRequest):
    """Download a model to disk or dram"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")

    # Validate target device
    if not validate_storage_device(request.target_device):
        raise HTTPException(
            status_code=400, detail=f"Invalid storage device: {request.target_device}. Only 'disk' or 'dram' allowed."
        )

    model = state.models[request.model_name]
    model.status = ModelStatus.DOWNLOADING

    logger.info(f"Starting download: {request.model_name} from {request.source_url} to {request.target_device}")

    # Start actual download asynchronously
    asyncio.create_task(perform_download(request.model_name, request.source_url, request.target_device))

    return StandardResponse(
        success=True, message=f"Download started for {request.model_name} to {request.target_device}"
    )


async def perform_download(model_name: str, source_url: str, target_device: str):
    """Actually download the model with enhanced HuggingFace integration"""
    model = state.models[model_name]

    try:
        # Get storage path
        storage_path = get_storage_path(model_name, target_device)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Download based on source
        if "huggingface.co" in source_url or source_url.startswith("hf://"):
            if source_url.startswith("hf://"):
                repo_id = source_url.replace("hf://", "")
            else:
                repo_id = source_url.replace("https://huggingface.co/", "")
            await download_from_huggingface_enhanced(model_name, repo_id, storage_path, target_device)
        else:
            await download_from_url(model_name, source_url, storage_path)

        # Validate the downloaded path and set appropriate status
        if validate_model_path(str(storage_path)):
            model.checkpoints[target_device] = str(storage_path)

            # Set status based on target device
            if target_device == "dram":
                model.status = ModelStatus.DRAM
                # Load model into memory variables
                load_safetensors_to_dram(storage_path, model_name)
            else:  # target_device == "disk"
                model.status = ModelStatus.DISK

            logger.info(f"✅ Download completed and validated: {model_name} -> {target_device}")
        else:
            model.status = ModelStatus.ERROR
            logger.error(f"❌ Download completed but path validation failed: {model_name} -> {storage_path}")
            raise Exception(f"Downloaded model path validation failed: {storage_path}")

    except Exception as e:
        model.status = ModelStatus.ERROR
        logger.error(f"❌ Download failed: {model_name} - {e}")
        raise


async def download_from_huggingface_enhanced(model_name: str, repo_id: str, storage_path: Path, target_device: str):
    """Enhanced HuggingFace download with proper API integration"""
    try:
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading {repo_id} using HuggingFace Hub API...")

        # Use HuggingFace Hub for better download control
        downloaded_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=repo_id,
                cache_dir=str(storage_path.parent),
                local_dir=str(storage_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=None,  # Add token support if needed
            ),
        )

        logger.info(f"HuggingFace download completed: {repo_id} -> {storage_path}")
        logger.info(f"Downloaded path: {downloaded_path}")

    except ImportError:
        # Fallback to huggingface-cli if hub not available
        logger.warning("huggingface_hub not available, falling back to CLI")
        await download_from_huggingface_cli(model_name, repo_id, storage_path)
    except Exception as e:
        logger.error(f"HuggingFace Hub download failed: {e}")
        # Try CLI as fallback
        await download_from_huggingface_cli(model_name, repo_id, storage_path)


async def download_from_huggingface_cli(model_name: str, repo_id: str, storage_path: Path):
    """Fallback HuggingFace download using CLI"""
    cmd = ["huggingface-cli", "download", repo_id, "--local-dir", str(storage_path), "--resume-download"]

    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise Exception(f"HuggingFace CLI download failed: {stderr.decode()}")

    logger.info(f"HuggingFace CLI download completed: {repo_id}")


async def download_from_url(model_name: str, source_url: str, storage_path: Path):
    """Download from generic URL"""
    raise NotImplementedError("Generic URL download not implemented yet")


# Copy operations


@app.post("/copy")
async def copy_model(request: CopyRequest):
    """Copy model between devices (disk/dram/gpu)"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")

    model = state.models[request.model_name]

    # Infer copy method if not specified
    if not request.method:
        request.method = infer_copy_method(request.source_device, request.target_device)

    # Validate source for checkpoint copies
    if request.source_device in ["disk", "dram"] and request.source_device not in model.checkpoints:
        raise HTTPException(status_code=400, detail=f"Model checkpoint not found on {request.source_device}")

    # If target is GPU, this is a serve operation
    if "gpu" in request.target_device:
        # Auto-generate instance ID if not provided
        instance_id = request.instance_id or str(uuid.uuid4())

        # Extract GPU device
        gpu_device = request.target_device

        # Find available port
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        # Create serve request
        serve_req = ServeRequest(
            model_name=request.model_name,
            source_device=request.source_device,
            gpu_device=gpu_device,
            port=port,
            instance_id=instance_id,
        )

        # Start serving
        return await serve_model(serve_req)

    # For checkpoint copies (disk<->dram)
    model.status = ModelStatus.COPYING

    # Start copy operation
    asyncio.create_task(perform_copy(request))

    return StandardResponse(
        success=True,
        message=f"Copy started: {request.model_name} from {request.source_device} to {request.target_device}",
    )


async def perform_copy(request: CopyRequest):
    """Enhanced copy operation with model variable handling"""
    model = state.models[request.model_name]

    try:
        if request.method == CopyMethod.DISK_TO_DRAM:
            # Load model from disk to DRAM with variable storage
            logger.info(f"Loading {request.model_name} from disk to DRAM...")
            source_path = Path(model.checkpoints[request.source_device])
            target_path = get_storage_path(request.model_name, request.target_device)
            target_path.mkdir(parents=True, exist_ok=True)

            # Copy files first
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)

            # Load model into memory variables
            model_obj = load_safetensors_to_dram(target_path, request.model_name)
            if model_obj is None:
                raise Exception("Failed to load model variable to DRAM")

        elif request.method == CopyMethod.DRAM_TO_DISK:
            # Save model from DRAM back to disk
            logger.info(f"Saving {request.model_name} from DRAM to disk...")

            # Get model object from DRAM storage
            model_obj = model_storage.get_dram_model(request.model_name)
            if model_obj is None:
                # Fallback to file copy if no model variable stored
                source_path = Path(model.checkpoints[request.source_device])
                target_path = get_storage_path(request.model_name, request.target_device)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                # Save model variable to disk
                target_path = get_storage_path(request.model_name, request.target_device)
                success = save_model_to_disk(request.model_name, model_obj, target_path)
                if not success:
                    raise Exception("Failed to save model from DRAM to disk")

        else:
            # Standard file copy for other operations
            source_path = Path(model.checkpoints[request.source_device])
            target_path = get_storage_path(request.model_name, request.target_device)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Copying {request.model_name} from {request.source_device} to {request.target_device}...")
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)

        # Validate the target path before updating checkpoints
        if validate_model_path(str(target_path)):
            model.checkpoints[request.target_device] = str(target_path)

            # Remove source if requested
            if not request.keep_source:
                if request.source_device == "dram":
                    # Remove from DRAM variable storage
                    model_storage.remove_dram_model(request.model_name)

                # Remove filesystem path
                if request.source_device in model.checkpoints:
                    source_path = Path(model.checkpoints[request.source_device])
                    if source_path.exists():
                        shutil.rmtree(source_path)
                    del model.checkpoints[request.source_device]

            # Update status based on remaining valid checkpoints
            update_model_status_based_on_checkpoints(model)
            logger.info(f"✅ Copy completed and validated: {request.model_name} to {request.target_device}")
        else:
            model.status = ModelStatus.ERROR
            logger.error(f"❌ Copy completed but target path validation failed: {request.model_name} -> {target_path}")
            raise Exception(f"Copy target path validation failed: {target_path}")

    except Exception as e:
        model.status = ModelStatus.ERROR
        logger.error(f"❌ Copy failed: {request.model_name} - {e}")
        raise


# Serving operations


@app.post("/serve")
async def serve_model(request: ServeRequest):
    """Start serving a model on GPU"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")

    model = state.models[request.model_name]

    # Validate source
    if request.source_device not in model.checkpoints:
        raise HTTPException(status_code=400, detail=f"Model checkpoint not found on {request.source_device}")

    # Generate instance ID if not provided
    instance_id = request.instance_id or str(uuid.uuid4())

    # Check if instance already exists
    if instance_id in state.vllm_servers:
        server = state.vllm_servers[instance_id]
        if server.is_running():
            return StandardResponse(
                success=True,
                message=f"Instance {instance_id} already running",
                data={"instance_id": instance_id, "url": f"http://0.0.0.0:{server.port}", "port": server.port},
            )

    try:
        # Get model path from checkpoint
        model_path = model.checkpoints[request.source_device]

        # Extract GPU ID
        gpu_id = extract_gpu_device_id(request.gpu_device)
        if gpu_id is None:
            raise ValueError(f"Invalid GPU device: {request.gpu_device}")

        # Prepare vLLM command
        vllm_cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(request.port),
            "--trust-remote-code",
            "--download-dir",
            "/tmp/vllm_models",
        ]

        # Set GPU environment
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Add config options from request only
        gpu_memory_util = 0.9  # Default value
        if request.config and "gpu_memory_utilization" in request.config:
            gpu_memory_util = request.config["gpu_memory_utilization"]
        vllm_cmd.extend(["--gpu-memory-utilization", str(gpu_memory_util)])

        # Add max concurrent requests (default)
        vllm_cmd.extend(["--max-num-seqs", "256"])

        # Optimize for fast loading from DRAM
        if request.source_device == "dram":
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            if request.config and request.config.get("use_pinned_memory", True):
                env["CUDA_LAUNCH_BLOCKING"] = "0"

            # Check if model is already loaded in DRAM variables
            dram_model = model_storage.get_dram_model(request.model_name)
            if dram_model is not None:
                logger.info(f"Model {request.model_name} found in DRAM variables, optimizing GPU transfer...")
                env["GSWARM_DRAM_LOADED"] = "1"

        logger.info(f"Starting vLLM server instance {instance_id} for {request.model_name} on {request.gpu_device}")
        logger.info(f"Command: {' '.join(vllm_cmd)}")

        # Start vLLM server
        process = subprocess.Popen(
            vllm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=1,
            universal_newlines=False,
            preexec_fn=os.setsid,
        )

        # Create server tracker
        vllm_server = VLLMServer(
            instance_id=instance_id,
            model_name=request.model_name,
            gpu_device=request.gpu_device,
            port=request.port,
            process=process,
            model_path=model_path,
        )

        # Start monitoring threads
        vllm_server.stdout_thread = threading.Thread(
            target=monitor_process_output, args=(process, vllm_server), daemon=True
        )
        vllm_server.stderr_thread = threading.Thread(
            target=monitor_process_errors, args=(process, vllm_server), daemon=True
        )
        vllm_server.stdout_thread.start()
        vllm_server.stderr_thread.start()

        # Store server
        state.vllm_servers[instance_id] = vllm_server

        # Store GPU inference endpoint in model variable storage
        inference_endpoint = {
            "server": vllm_server,
            "url": f"http://0.0.0.0:{request.port}",
            "port": request.port,
            "gpu_device": request.gpu_device,
            "model_path": model_path,
        }
        model_storage.store_gpu_model(
            instance_id, inference_endpoint, {"model_name": request.model_name, "gpu_device": request.gpu_device}
        )

        # Update model serving instances
        service_url = f"http://0.0.0.0:{request.port}"
        model.serving_instances[instance_id] = {
            "device": request.gpu_device,
            "url": service_url,
            "port": request.port,
            "pid": process.pid,
            "status": "loading",
        }

        # Update model status if first instance
        if len(model.serving_instances) == 1:
            model.status = ModelStatus.SERVING

        logger.info(f"🔄 vLLM server starting: instance {instance_id} on {request.gpu_device}:{request.port}")

        # Monitor server readiness
        asyncio.create_task(monitor_server_ready(instance_id))

        return StandardResponse(
            success=True,
            message=f"Model {request.model_name} serving started",
            data={
                "instance_id": instance_id,
                "url": service_url,
                "port": request.port,
                "gpu_device": request.gpu_device,
                "pid": process.pid,
            },
        )

    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")


@app.post("/stop_serve")
async def stop_serving(request: StopServeRequest):
    """Stop a specific serving instance"""
    if request.model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")

    if request.instance_id not in state.vllm_servers:
        raise HTTPException(status_code=404, detail=f"Instance {request.instance_id} not found")

    model = state.models[request.model_name]
    server = state.vllm_servers[request.instance_id]

    # Stop the server
    server.stop()
    del state.vllm_servers[request.instance_id]

    # Remove from GPU model variable storage
    model_storage.remove_gpu_model(request.instance_id)

    # Remove from model serving instances
    if request.instance_id in model.serving_instances:
        del model.serving_instances[request.instance_id]

    # Update model status if no more instances
    if len(model.serving_instances) == 0:
        # Update status based on remaining checkpoints
        update_model_status_based_on_checkpoints(model)

    logger.info(f"Stopped serving instance {request.instance_id} for {request.model_name}")

    return StandardResponse(success=True, message=f"Stopped instance {request.instance_id}")


@app.get("/serving")
async def list_serving_instances():
    """List all serving instances across all models"""
    instances = []

    for instance_id, server in state.vllm_servers.items():
        model = state.models.get(server.model_name)
        if model and instance_id in model.serving_instances:
            info = model.serving_instances[instance_id]

            # Get GPU memory usage if available
            memory_usage = None
            try:
                gpu_id = extract_gpu_device_id(server.gpu_device)
                if gpu_id is not None and torch.cuda.is_available():
                    memory_usage = {
                        "allocated": torch.cuda.memory_allocated(gpu_id),
                        "reserved": torch.cuda.memory_reserved(gpu_id),
                        "total": torch.cuda.get_device_properties(gpu_id).total_memory,
                    }
            except:
                pass

            instances.append(
                ModelServingStatus(
                    instance_id=instance_id,
                    model_name=server.model_name,
                    gpu_device=server.gpu_device,
                    port=server.port,
                    url=info["url"],
                    pid=server.pid,
                    status="running" if server.is_running() else "stopped",
                    started_at=server.started_at,
                    last_health_check=server.last_health_check,
                    memory_usage=memory_usage,
                )
            )

    return {"instances": instances, "count": len(instances)}


@app.get("/serving/{model_name}")
async def get_model_serving_instances(model_name: str):
    """Get all serving instances for a specific model"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = state.models[model_name]
    instances = []

    for instance_id, info in model.serving_instances.items():
        if instance_id in state.vllm_servers:
            server = state.vllm_servers[instance_id]
            instances.append(
                {
                    "instance_id": instance_id,
                    "gpu_device": info["device"],
                    "url": info["url"],
                    "port": info["port"],
                    "pid": info["pid"],
                    "status": "running" if server.is_running() else "stopped",
                    "started_at": server.started_at.isoformat(),
                }
            )

    return {"model_name": model_name, "instances": instances, "count": len(instances)}


# Model variable storage endpoints


@app.get("/memory/models")
async def get_memory_models():
    """Get all models loaded in memory (DRAM and GPU)"""
    dram_models = model_storage.list_dram_models()
    gpu_models = model_storage.list_gpu_models()
    memory_usage = model_storage.get_memory_usage()

    return {
        "dram_models": dram_models,
        "gpu_models": gpu_models,
        "memory_usage": memory_usage,
        "total_dram_models": len(dram_models),
        "total_gpu_models": len(gpu_models),
    }


@app.get("/memory/dram")
async def get_dram_models():
    """Get all models loaded in DRAM"""
    dram_models = model_storage.list_dram_models()
    model_details = []

    for model_name in dram_models:
        model_obj = model_storage.get_dram_model(model_name)
        details = {
            "model_name": model_name,
            "loaded": model_obj is not None,
            "type": type(model_obj).__name__ if model_obj else None,
        }

        # Try to get model size information
        try:
            if hasattr(model_obj, "num_parameters"):
                details["parameters"] = model_obj.num_parameters()
            elif hasattr(model_obj, "get_memory_footprint"):
                details["memory_footprint"] = model_obj.get_memory_footprint()
        except:
            pass

        model_details.append(details)

    return {"dram_models": model_details, "count": len(dram_models)}


@app.get("/memory/gpu")
async def get_gpu_models():
    """Get all models/instances loaded on GPU"""
    gpu_instances = model_storage.list_gpu_models()
    instance_details = []

    for instance_id in gpu_instances:
        endpoint = model_storage.get_gpu_model(instance_id)
        if endpoint:
            details = {
                "instance_id": instance_id,
                "url": endpoint.get("url"),
                "port": endpoint.get("port"),
                "gpu_device": endpoint.get("gpu_device"),
                "model_path": endpoint.get("model_path"),
                "running": endpoint.get("server", {}).is_running()
                if hasattr(endpoint.get("server", {}), "is_running")
                else False,
            }
            instance_details.append(details)

    return {"gpu_instances": instance_details, "count": len(gpu_instances)}


@app.delete("/memory/dram/{model_name}")
async def unload_dram_model(model_name: str):
    """Unload a model from DRAM"""
    success = model_storage.remove_dram_model(model_name)

    if success:
        return StandardResponse(success=True, message=f"Model {model_name} unloaded from DRAM")
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found in DRAM")


@app.post("/discover")
async def discover_models():
    """Manually trigger model discovery and registration"""
    try:
        await state.discover_and_register_models()
        state.discovery_completed = True

        return StandardResponse(
            success=True,
            message="Model discovery completed",
            data={
                "total_models": len(state.models),
                "dram_models": len(model_storage.list_dram_models()),
                "gpu_models": len(model_storage.list_gpu_models()),
            },
        )
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model discovery failed: {str(e)}")


@app.post("/models/validate")
async def validate_all_models():
    """Validate all registered models and update their statuses based on cache paths"""
    validated_models = []

    for model_name, model in state.models.items():
        logger.info(f"Validating model: {model_name}")

        # Validate existing checkpoints
        had_valid_checkpoints = validate_model_checkpoints(model)

        # Try to discover additional paths if no valid checkpoints
        if not had_valid_checkpoints:
            discovered_path = discover_model_cache_path(model_name)
            if discovered_path:
                model.checkpoints["disk"] = discovered_path
                had_valid_checkpoints = True
                logger.info(f"Discovered path for {model_name}: {discovered_path}")

        # Update status based on validation
        old_status = model.status
        update_model_status_based_on_checkpoints(model)

        validated_models.append(
            {
                "model_name": model_name,
                "old_status": old_status,
                "new_status": model.status,
                "valid_checkpoints": list(model.checkpoints.keys()),
                "checkpoint_paths": model.checkpoints,
            }
        )

    disk_count = sum(1 for m in state.models.values() if m.status == ModelStatus.DISK)
    dram_count = sum(1 for m in state.models.values() if m.status == ModelStatus.DRAM)
    registered_count = sum(1 for m in state.models.values() if m.status == ModelStatus.REGISTERED)

    return StandardResponse(
        success=True,
        message="Model validation completed",
        data={
            "total_models": len(state.models),
            "disk_models": disk_count,
            "dram_models": dram_count,
            "registered_models": registered_count,
            "validated_models": validated_models,
        },
    )


@app.get("/models/{model_name}/validate")
async def validate_single_model(model_name: str):
    """Validate a specific model and update its status"""
    if model_name not in state.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = state.models[model_name]
    old_status = model.status
    old_checkpoints = dict(model.checkpoints)

    # Validate existing checkpoints
    had_valid_checkpoints = validate_model_checkpoints(model)

    # Try to discover additional paths if no valid checkpoints
    if not had_valid_checkpoints:
        discovered_path = discover_model_cache_path(model_name)
        if discovered_path:
            model.checkpoints["disk"] = discovered_path
            had_valid_checkpoints = True

    # Update status based on validation
    update_model_status_based_on_checkpoints(model)

    return {
        "model_name": model_name,
        "old_status": old_status,
        "new_status": model.status,
        "old_checkpoints": old_checkpoints,
        "new_checkpoints": model.checkpoints,
        "validation_successful": had_valid_checkpoints,
    }


# Helper functions for monitoring


def monitor_process_output(process: subprocess.Popen, server: VLLMServer):
    """Monitor process stdout in a separate thread"""
    try:
        for line in iter(process.stdout.readline, b""):
            if server._stop_monitoring.is_set():
                break
            line_str = line.decode("utf-8", errors="ignore").strip()
            if line_str:
                logger.info(f"[vLLM {server.instance_id}] {line_str}")

                # Check for readiness
                if "Uvicorn running on" in line_str or "Application startup complete" in line_str:
                    server.is_ready = True
                    logger.info(f"🚀 vLLM instance {server.instance_id} is ready!")
    except Exception as e:
        logger.error(f"Error monitoring stdout: {e}")


def monitor_process_errors(process: subprocess.Popen, server: VLLMServer):
    """Monitor process stderr in a separate thread"""
    try:
        for line in iter(process.stderr.readline, b""):
            if server._stop_monitoring.is_set():
                break
            line_str = line.decode("utf-8", errors="ignore").strip()
            if line_str:
                logger.error(f"[vLLM {server.instance_id} ERROR] {line_str}")
    except Exception as e:
        logger.error(f"Error monitoring stderr: {e}")


async def monitor_server_ready(instance_id: str):
    """Monitor server readiness and update status"""
    if instance_id not in state.vllm_servers:
        return

    server = state.vllm_servers[instance_id]
    model = state.models.get(server.model_name)

    # Wait for server to be ready
    timeout = 300  # 5 minutes
    start_time = asyncio.get_event_loop().time()
    health_url = f"http://localhost:{server.port}/health"

    async with aiohttp.ClientSession() as session:
        while asyncio.get_event_loop().time() - start_time < timeout:
            if not server.is_running():
                logger.error(f"vLLM instance {instance_id} died during startup")
                if model and instance_id in model.serving_instances:
                    model.serving_instances[instance_id]["status"] = "error"
                return

            try:
                async with session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        server.is_ready = True
                        server.last_health_check = datetime.now()
                        if model and instance_id in model.serving_instances:
                            model.serving_instances[instance_id]["status"] = "running"
                        logger.info(f"✅ vLLM instance {instance_id} is ready!")

                        # Start periodic health checks
                        asyncio.create_task(periodic_health_check(instance_id))
                        return
            except:
                pass

            await asyncio.sleep(2)

    logger.error(f"Timeout waiting for vLLM instance {instance_id}")
    if model and instance_id in model.serving_instances:
        model.serving_instances[instance_id]["status"] = "timeout"


async def periodic_health_check(instance_id: str):
    """Periodic health check for running instances"""
    while instance_id in state.vllm_servers:
        server = state.vllm_servers[instance_id]

        if not server.is_running():
            logger.warning(f"vLLM instance {instance_id} has stopped")
            model = state.models.get(server.model_name)
            if model and instance_id in model.serving_instances:
                model.serving_instances[instance_id]["status"] = "stopped"
            break

        # Health check every 30 seconds
        await asyncio.sleep(30)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{server.port}/health", timeout=5) as response:
                    if response.status == 200:
                        server.last_health_check = datetime.now()
        except:
            logger.warning(f"Health check failed for instance {instance_id}")


# Startup and cleanup events
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    logger.info("Starting GSwarm Model Manager...")
    logger.info(f"Model cache directory: {get_model_cache_dir()}")
    logger.info(f"DRAM cache directory: {get_dram_cache_dir()}")

    # Don't auto-discover models on the host
    # Model discovery should happen on clients
    logger.info("Model discovery disabled on host - models should be discovered on client nodes")
    state.discovery_completed = True


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all resources on shutdown"""
    logger.info("Shutting down GSwarm Model Manager...")

    # Stop all vLLM servers
    logger.info("Stopping all vLLM servers...")
    for server in state.vllm_servers.values():
        server.stop()

    logger.info("Shutdown complete")


def create_app(
    host: Optional[str] = None, port: Optional[int] = None, model_port: Optional[int] = None, **kwargs
) -> FastAPI:
    """Factory function to create and return the FastAPI app instance"""
    # Load config only when creating the app
    config = load_config()

    # Override config with CLI parameters if provided
    if host is not None or port is not None or model_port is not None:
        logger.info("Applying CLI parameter overrides to configuration...")

        # Create a new config instance with overrides
        host_config = HostConfig(
            huggingface_cache_dir=config.host.huggingface_cache_dir,
            model_cache_dir=config.host.model_cache_dir,
            model_manager_port=model_port if model_port is not None else config.host.model_manager_port,
            gswarm_grpc_port=port if port is not None else config.host.gswarm_grpc_port,
            gswarm_http_port=config.host.gswarm_http_port,
        )

        config = GSwarmConfig(host=host_config, client=config.client)
        logger.info(
            f"Configuration overridden - Port: {config.host.gswarm_grpc_port}, Model Port: {config.host.model_manager_port}"
        )

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)

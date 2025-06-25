"""
Standalone Model Server for GSwarm
Provides HTTP APIs for model management including LLM and Stable Diffusion models.
"""

import os
import sys
import json
import uuid
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import threading
import signal
import psutil
import shutil
import socket
import random
import string

from .cost import get_estimation_cost

# Disable flash attention and xformers to avoid compatibility issues
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DIFFUSERS_DISABLE_XFORMERS_WARNING"] = "1"

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch

# Robust transformers import with fallback
TRANSFORMERS_AVAILABLE = False
PIPELINE_AVAILABLE = False

try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    # Try to import pipeline separately as it's often the problematic one
    try:
        from transformers import pipeline
        PIPELINE_AVAILABLE = True
    except ImportError as e:
        print(f"WARNING: transformers pipeline not available: {e}")
        PIPELINE_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = True
    print("SUCCESS: transformers library loaded successfully")
except ImportError as e:
    print(f"ERROR: transformers not available: {e}")
    print("The server will start but model functionality will be limited")
    TRANSFORMERS_AVAILABLE = False
    PIPELINE_AVAILABLE = False
    # Create dummy classes to prevent AttributeError
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("transformers not available - please install compatible version")
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("transformers not available - please install compatible version")
    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("transformers not available - please install compatible version")

import requests
import uvicorn
from loguru import logger

# Robust diffusers import with fallback
DIFFUSERS_AVAILABLE = False
try:
    # Disable xformers to avoid flash attention issues with diffusers
    os.environ["XFORMERS_DISABLED"] = "1"
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
    print("SUCCESS: diffusers library loaded successfully")
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    print(f"WARNING: diffusers not available - diffusion model support will be limited: {e}")
    # Create a dummy class to prevent AttributeError
    class StableDiffusionPipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("diffusers not available - please install compatible version or fix flash-attn version")
except Exception as e:
    DIFFUSERS_AVAILABLE = False
    print(f"WARNING: diffusers failed to load - diffusion model support will be limited: {e}")
    # Create a dummy class to prevent AttributeError
    class StableDiffusionPipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("diffusers not available - please install compatible version or fix flash-attn version")

# Configure logger
logger.add("standalone_server.log", rotation="10 MB", level="INFO")

# Global state
app = FastAPI(
    title="GSwarm Standalone Model Server",
    description="Standalone server for model serving with download, load, serve, and inference capabilities",
    version="1.0.0"
)

# Data models
class ModelInstance:
    def __init__(self, model_name: str, device: str, instance_id: str = None, port: int = None):
        self.instance_id = instance_id or self._generate_short_id()
        self.model_name = model_name
        self.device = device
        self.port = port or self._allocate_port()
        self.created_at = datetime.now()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.process = None  # For external model servers like vLLM
        self.url = f"http://localhost:{self.port}"
        
    def _generate_short_id(self) -> str:
        """Generate a short 6-character instance ID"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    def _allocate_port(self) -> int:
        """Allocate a random available port in range 10000-20000"""
        return allocate_random_port()

class ServerState:
    def __init__(self):
        self.models_disk: Dict[str, str] = {}  # model_name -> path
        self.models_dram: Dict[str, Any] = {}  # model_name -> loaded model objects
        self.serving_instances: Dict[str, ModelInstance] = {}  # instance_id -> ModelInstance
        self.used_ports: set = set()  # Track allocated ports
        self.model_cache_dir = Path.home() / ".cache" / "gswarm" / "models"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

state = ServerState()

# Request/Response models
class ServeRequest(BaseModel):
    model_name: str
    device: Union[str, List[str]] = Field(
        default=None, 
        description="Device(s) to serve on. Examples: 'cuda:0', 'cuda:1,2', 'cuda:1,cuda:2', ['cuda:1', 'cuda:2'], 'auto'"
    )

class OffloadRequest(BaseModel):
    model_name: str = Field(description="Model name to offload")
    target: str = Field(description="Where to offload: dram or disk")

class LoadRequest(BaseModel):
    model_name: str
    target: str = "dram"
    device: Optional[str] = Field(default=None, description="Specific device to load to (e.g., 'cuda:0', 'cuda:1')")

class DownloadRequest(BaseModel):
    model_name: str
    source: str = "huggingface"

class CallRequest(BaseModel):
    instance_id: str
    data: Dict[str, Any]

class EstimateRequest(BaseModel):
    instance_id: str
    data: Dict[str, Any]

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Helper functions
def get_model_type(model_name: str) -> str:
    """Determine model type based on model name"""
    name_lower = model_name.lower()
    if any(x in name_lower for x in ["stable-diffusion", "sd-", "flux", "diffusion"]):
        return "diffusion"
    elif any(x in name_lower for x in ["llama", "gpt", "mistral", "gemma", "qwen", "chat", "instruct"]):
        return "llm"
    else:
        return "llm"  # Default to LLM

def get_available_device() -> str:
    """Get the best available device"""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"

def parse_device_specification(device_spec: Union[str, List[str]]) -> List[str]:
    """Parse device specification into a list of devices"""
    if device_spec is None:
        return [get_available_device()]
    
    if isinstance(device_spec, list):
        # Already a list of devices
        return device_spec
    
    if isinstance(device_spec, str):
        device_spec = device_spec.strip()
        
        if device_spec.lower() == "auto":
            # Use all available GPUs
            if torch.cuda.is_available():
                return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            else:
                return ["cpu"]
        
        if "," in device_spec:
            # Parse comma-separated format
            if device_spec.startswith("cuda:"):
                # Format: "cuda:1,2,3" or "cuda:1,cuda:2,cuda:3"
                parts = device_spec.split(",")
                devices = []
                for part in parts:
                    part = part.strip()
                    if part.startswith("cuda:"):
                        devices.append(part)
                    elif part.isdigit():
                        # Handle "cuda:1,2,3" format
                        devices.append(f"cuda:{part}")
                    else:
                        raise ValueError(f"Invalid device specification: {part}")
                return devices
            else:
                # Format: "cuda:1,cuda:2,cuda:3"
                return [part.strip() for part in device_spec.split(",")]
        else:
            # Single device
            return [device_spec]
    
    raise ValueError(f"Invalid device specification: {device_spec}")

def create_device_map_for_multi_gpu(devices: List[str]) -> Dict[str, str]:
    """Create device map for multi-GPU model loading"""
    if len(devices) == 1:
        return devices[0]
    
    # For multiple devices, use device_map="auto" or create custom mapping
    # This will let transformers handle the distribution
    return "auto"

def download_model_from_huggingface(model_name: str) -> str:
    """Download model from Hugging Face to local cache"""
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers library not available. Please install a compatible version.")
    
    try:
        model_path = state.model_cache_dir / model_name.replace("/", "--")
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists at {model_path}")
            return str(model_path)
        
        logger.info(f"Downloading model {model_name} from Hugging Face...")
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(model_path))
        model = AutoModel.from_pretrained(model_name, cache_dir=str(model_path))
        
        # Save to our cache directory structure
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))
        
        logger.info(f"Successfully downloaded {model_name} to {model_path}")
        return str(model_path)
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise

def load_model_to_dram(model_name: str, model_path: str, target_devices: Union[str, List[str]] = None):
    """Load model into DRAM for faster access with multi-GPU support"""
    try:
        # FIX: Default to GPU if no device specified
        if target_devices is None:
            target_devices = get_available_device()  # Will return cuda:0 if available
        
        model_type = get_model_type(model_name)
        
        if model_type == "llm":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("transformers library not available for LLM models")
                
            logger.info(f"Loading LLM model {model_name} to DRAM...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Handle multi-GPU device mapping
            if isinstance(target_devices, list) and len(target_devices) > 1:
                logger.info(f"Loading model across multiple devices: {target_devices}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",  # Let transformers distribute across GPUs
                    low_cpu_mem_usage=True
                )
                device_info = f"multi-gpu:{','.join(target_devices)}"
            elif target_devices and target_devices != "cpu":
                single_device = target_devices if isinstance(target_devices, str) else target_devices[0]
                logger.info(f"Loading model on single device: {single_device}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=single_device
                )
                device_info = single_device
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="cpu"
                )
                device_info = "cpu"
            
            state.models_dram[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "llm",
                "loaded_at": datetime.now(),
                "device": device_info,
                "devices": target_devices if isinstance(target_devices, list) else [target_devices or "cpu"]
            }
        elif model_type == "diffusion":
            logger.info(f"Loading diffusion model {model_name} to DRAM...")
            # FIX: For stable diffusion models, properly handle GPU device
            if DIFFUSERS_AVAILABLE:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                # FIX: Move pipeline to GPU if available
                if target_devices and target_devices != "cpu" and torch.cuda.is_available():
                    device = target_devices if isinstance(target_devices, str) else target_devices[0]
                    logger.info(f"Moving diffusion pipeline to {device}")
                    pipe = pipe.to(device)
                    device_info = device
                else:
                    device_info = "cpu"
                    
                state.models_dram[model_name] = {
                    "pipeline": pipe,
                    "type": "diffusion",
                    "loaded_at": datetime.now(),
                    "device": device_info
                }
            else:
                if not TRANSFORMERS_AVAILABLE:
                    raise RuntimeError("Neither diffusers nor transformers available for diffusion models")
                logger.warning("diffusers not available, loading as generic model")
                # Fallback to generic loading
                model = AutoModel.from_pretrained(model_path)
                # FIX: Move to GPU if available
                if target_devices and target_devices != "cpu" and torch.cuda.is_available():
                    device = target_devices if isinstance(target_devices, str) else target_devices[0]
                    model = model.to(device)
                    device_info = device
                else:
                    device_info = "cpu"
                    
                state.models_dram[model_name] = {
                    "model": model,
                    "type": "diffusion",
                    "loaded_at": datetime.now(),
                    "device": device_info
                }
        
        logger.info(f"Successfully loaded {model_name} to DRAM on {device_info}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name} to DRAM: {e}")
        raise

def is_port_available(port: int) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('localhost', port))
            return True
    except OSError:
        return False

def allocate_random_port(min_port: int = 10000, max_port: int = 20000) -> int:
    """Allocate a random available port in the specified range"""
    max_attempts = 100
    for _ in range(max_attempts):
        port = random.randint(min_port, max_port)
        if port not in state.used_ports and is_port_available(port):
            state.used_ports.add(port)
            return port
    
    # If random selection fails, try sequential search
    for port in range(min_port, max_port + 1):
        if port not in state.used_ports and is_port_available(port):
            state.used_ports.add(port)
            return port
    
    raise RuntimeError(f"No available ports in range {min_port}-{max_port}")

def release_port(port: int):
    """Release a port back to the available pool"""
    state.used_ports.discard(port)

def create_serving_instance(model_name: str, devices: Union[str, List[str]]) -> ModelInstance:
    """Create a serving instance for a model"""
    # Handle device specification properly
    if isinstance(devices, list):
        if len(devices) == 1:
            primary_device = devices[0]
            device_display = devices[0]
        else:
            primary_device = devices[0]  # Use first device as primary
            device_display = ",".join(devices)
    else:
        primary_device = devices
        device_display = devices
        devices = [devices]
    
    instance = ModelInstance(model_name, device_display)
    
    try:
        # Check if model is in DRAM first
        if model_name in state.models_dram:
            logger.info(f"Using model {model_name} from DRAM")
            dram_model = state.models_dram[model_name]
            dram_device = dram_model.get("device", "unknown")
            
            # IMPORTANT: Check device compatibility
            if len(devices) == 1 and primary_device != dram_device and not (
                hasattr(dram_model.get("model"), 'hf_device_map') and dram_model.get("model").hf_device_map
            ):
                logger.warning(f"Model {model_name} is loaded on {dram_device} but requested on {primary_device}")
                logger.info(f"Attempting to move model from {dram_device} to {primary_device}")
            
            # Create a separate copy/reference for this instance to allow multiple instances
            if dram_model.get("type") == "llm":
                instance.model = dram_model.get("model")
                instance.tokenizer = dram_model.get("tokenizer")
                
                # Handle multi-GPU models properly
                if instance.model:
                    try:
                        # Check if model has device_map (multi-GPU model)
                        if hasattr(instance.model, 'hf_device_map') and instance.model.hf_device_map:
                            logger.info(f"Model already distributed across devices: {instance.model.hf_device_map}")
                            # For multi-GPU models, don't try to move - they're already distributed
                        elif len(devices) > 1:
                            logger.warning(f"Multi-GPU serving requested but model loaded on single device {dram_device}")
                            logger.info(f"Consider loading model with multi-GPU support for: {devices}")
                        elif primary_device != "cpu" and primary_device != dram_device:
                            # Single GPU case - check if we need to move the model
                            try:
                                logger.info(f"Moving model from {dram_device} to {primary_device}")
                                instance.model = instance.model.to(primary_device)
                                # Update the stored device info
                                state.models_dram[model_name]["device"] = primary_device
                            except Exception as e:
                                logger.error(f"Failed to move model to {primary_device}: {e}")
                                raise HTTPException(status_code=500, detail=f"Cannot move model to {primary_device}: {e}")
                    except Exception as e:
                        logger.warning(f"Device handling error: {e}")
                        
            elif dram_model.get("type") == "diffusion":
                instance.pipeline = dram_model.get("pipeline")
                
                # Move pipeline to specified device if needed (only for single GPU)
                if instance.pipeline and len(devices) == 1 and primary_device != "cpu" and primary_device != dram_device:
                    try:
                        logger.info(f"Moving diffusion pipeline from {dram_device} to {primary_device}")
                        instance.pipeline = instance.pipeline.to(primary_device)
                        # Update the stored device info
                        state.models_dram[model_name]["device"] = primary_device
                    except Exception as e:
                        logger.error(f"Failed to move pipeline to {primary_device}: {e}")
                        raise HTTPException(status_code=500, detail=f"Cannot move pipeline to {primary_device}: {e}")
                
        # Otherwise load from disk with target device(s)
        elif model_name in state.models_disk:
            model_path = state.models_disk[model_name]
            # Load model with proper multi-GPU support
            load_model_to_dram(model_name, model_path, devices)
            # Recursively call to use DRAM version
            return create_serving_instance(model_name, devices)
        else:
            raise ValueError(f"Model {model_name} not found in DRAM or disk cache")
        
        state.serving_instances[instance.instance_id] = instance
        logger.info(f"Created serving instance {instance.instance_id} for {model_name} on {device_display}:{instance.port}")
        return instance
        
    except Exception as e:
        # Release the allocated port on failure
        if hasattr(instance, 'port') and instance.port:
            release_port(instance.port)
        logger.error(f"Failed to create serving instance for {model_name}: {e}")
        raise

# API endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GSwarm Standalone Model Server", 
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_in_dram": len(state.models_dram),
        "models_on_disk": len(state.models_disk),
        "serving_instances": len(state.serving_instances)
    }

@app.post("/standalone/download")
async def download_model(request: DownloadRequest):
    """Download a model from the specified source to disk"""
    try:
        model_name = request.model_name
        source = request.source
        
        if source.lower() == "huggingface":
            model_path = download_model_from_huggingface(model_name)
            state.models_disk[model_name] = model_path
            
            return StandardResponse(
                success=True,
                message=f"Successfully downloaded {model_name}",
                data={"model_path": model_path, "source": source}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source: {source}")
            
    except Exception as e:
        logger.error(f"Download failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standalone/load")
async def load_model(request: LoadRequest):
    """Load a model to DRAM for faster access on specific device"""
    try:
        model_name = request.model_name
        target = request.target
        device = request.device
        
        if target.lower() != "dram":
            raise HTTPException(status_code=400, detail="Only 'dram' target is supported for loading")
        
        if model_name not in state.models_disk:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found on disk. Download it first.")
        
        # Check if already loaded
        if model_name in state.models_dram:
            existing_device = state.models_dram[model_name].get("device", "unknown")
            if device and device != existing_device:
                logger.warning(f"Model {model_name} already loaded on {existing_device} but requested {device}")
            return StandardResponse(
                success=True,
                message=f"Model {model_name} already loaded in DRAM on {existing_device}",
                data={"already_loaded": True, "device": existing_device}
            )
        
        model_path = state.models_disk[model_name]
        
        # Use specified device or default
        target_device = device if device else get_available_device()
        
        # Validate device
        if target_device.startswith("cuda:"):
            gpu_id = int(target_device.split(":")[1])
            if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
                raise HTTPException(status_code=400, detail=f"GPU device {target_device} not available")
        
        load_model_to_dram(model_name, model_path, target_device)
        
        return StandardResponse(
            success=True,
            message=f"Successfully loaded {model_name} to DRAM on {target_device}",
            data={"target": target, "model_path": model_path, "device": target_device}
        )
        
    except Exception as e:
        logger.error(f"Load failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standalone/serve")
async def serve_model(request: ServeRequest):
    """Serve a model on the specified device(s), return instance endpoint"""
    try:
        model_name = request.model_name
        device_spec = request.device
        
        # Parse device specification
        devices = parse_device_specification(device_spec)
        
        # Validate devices
        for device in devices:
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
                if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
                    raise HTTPException(status_code=400, detail=f"GPU device {device} not available")
        
        # For display purposes, create a device string
        device_display = ",".join(devices) if len(devices) > 1 else devices[0]
        
        # Ensure model is available
        if model_name not in state.models_dram and model_name not in state.models_disk:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found. Download and load it first.")
        
        # Create serving instance
        instance = create_serving_instance(model_name, devices)
        
        return StandardResponse(
            success=True,
            message=f"Successfully started serving {model_name}",
            data={
                "instance_id": instance.instance_id,
                "model_name": model_name,
                "device": device_display,
                "devices": devices,
                "port": instance.port,
                "url": instance.url,
                "endpoint": f"/standalone/call/{instance.instance_id}",
                "created_at": instance.created_at.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Serve failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standalone/offload")
async def offload_model(request: OffloadRequest):
    """Stop model serving and move model to DRAM or disk"""
    try:
        model_name = request.model_name
        target = request.target
        
        # Stop any serving instances for this model
        instances_to_remove = []
        for instance_id, instance in state.serving_instances.items():
            if instance.model_name == model_name:
                instances_to_remove.append(instance_id)
        
        for instance_id in instances_to_remove:
            del state.serving_instances[instance_id]
            logger.info(f"Stopped serving instance {instance_id}")
        
        if target.lower() == "disk":
            # Remove from DRAM if present
            if model_name in state.models_dram:
                del state.models_dram[model_name]
                logger.info(f"Removed {model_name} from DRAM")
            
            # Ensure it's on disk
            if model_name not in state.models_disk:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found on disk")
                
        elif target.lower() == "dram":
            # Ensure model is loaded to DRAM
            if model_name not in state.models_dram:
                if model_name not in state.models_disk:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found on disk")
                model_path = state.models_disk[model_name]
                load_model_to_dram(model_name, model_path)
        else:
            raise HTTPException(status_code=400, detail="Target must be 'disk' or 'dram'")
        
        return StandardResponse(
            success=True,
            message=f"Successfully offloaded {model_name} to {target}",
            data={
                "target": target,
                "instances_stopped": len(instances_to_remove)
            }
        )
        
    except Exception as e:
        logger.error(f"Offload failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standalone/call/{instance_id}")
async def call_model(instance_id: str, request: CallRequest):
    """Send data to model instance and get response"""
    try:
        if instance_id not in state.serving_instances:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
        
        instance = state.serving_instances[instance_id]
        model_type = get_model_type(instance.model_name)
        
        # Extract input data
        input_data = request.data
        
        if model_type == "llm":
            # Handle LLM inference
            if "prompt" not in input_data:
                raise HTTPException(status_code=400, detail="LLM models require 'prompt' in data")
            
            prompt = input_data["prompt"]
            max_length = input_data.get("max_length", 100)
            temperature = input_data.get("temperature", 0.7)
            
            if instance.tokenizer and instance.model:
                # Direct model inference with better device handling
                inputs = instance.tokenizer(prompt, return_tensors="pt")
                
                # FIX: Better device detection and handling
                try:
                    model_device = next(instance.model.parameters()).device
                    logger.info(f"Model is on device: {model_device}")
                    
                    # Move inputs to the same device as the model
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                    logger.info(f"Moved inputs to device: {model_device}")
                    
                except Exception as e:
                    logger.warning(f"Could not determine model device, falling back to instance.device: {e}")
                    # FIX: Better fallback device handling
                    if instance.device and instance.device != "cpu" and "cuda" in instance.device:
                        # Extract device from instance.device (could be "cuda:0" or "multi-gpu:cuda:0,cuda:1")
                        if instance.device.startswith("multi-gpu:"):
                            # For multi-GPU, use the first device
                            devices_str = instance.device.split("multi-gpu:")[1]
                            primary_device = devices_str.split(",")[0]
                        else:
                            primary_device = instance.device
                        
                        try:
                            inputs = {k: v.to(primary_device) for k, v in inputs.items()}
                            logger.info(f"Moved inputs to fallback device: {primary_device}")
                        except Exception as e2:
                            logger.warning(f"Failed to move inputs to {primary_device}: {e2}")
                
                with torch.no_grad():
                    outputs = instance.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=instance.tokenizer.eos_token_id
                    )
                
                response_text = instance.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return StandardResponse(
                    success=True,
                    message="Inference completed",
                    data={
                        "response": response_text,
                        "model_type": "llm",
                        "instance_id": instance_id,
                        "port": instance.port,
                        "device_used": str(model_device) if 'model_device' in locals() else instance.device
                    }
                )
            else:
                raise HTTPException(status_code=500, detail="Model or tokenizer not properly loaded")
                
        elif model_type == "diffusion":
            # Handle diffusion model inference
            if "prompt" not in input_data:
                raise HTTPException(status_code=400, detail="Diffusion models require 'prompt' in data")
            
            prompt = input_data["prompt"]
            num_inference_steps = input_data.get("steps", 20)
            guidance_scale = input_data.get("guidance_scale", 7.5)
            
            if instance.pipeline:
                # Use diffusers pipeline
                image = instance.pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
                
                # Save image and return path
                output_path = state.model_cache_dir / f"output_{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                image.save(output_path)
                
                return StandardResponse(
                    success=True,
                    message="Image generation completed",
                    data={
                        "image_path": str(output_path),
                        "model_type": "diffusion",
                        "instance_id": instance_id,
                        "port": instance.port
                    }
                )
            else:
                raise HTTPException(status_code=500, detail="Diffusion pipeline not properly loaded")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Inference failed for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standalone/estimate/{instance_id}")
async def estimate_model(instance_id: str, request: EstimateRequest):
    """Get estimated execution time for model inference without actually running it"""
    try:
        if instance_id not in state.serving_instances:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
        
        instance = state.serving_instances[instance_id]
        model_type = get_model_type(instance.model_name)
        
        # Extract input data for feature analysis
        input_data = request.data
        
        # Extract data features for cost estimation
        data_features = []
        
        if model_type == "llm":
            if "prompt" not in input_data:
                raise HTTPException(status_code=400, detail="LLM models require 'prompt' in data")
            
            prompt = input_data["prompt"]
            max_length = input_data.get("max_length", 100)
            temperature = input_data.get("temperature", 0.7)
            
            # Add relevant features for cost estimation
            data_features.extend([
                f"prompt_length:{len(prompt)}",
                f"max_length:{max_length}",
                f"temperature:{temperature}"
            ])
            
        elif model_type == "diffusion":
            if "prompt" not in input_data:
                raise HTTPException(status_code=400, detail="Diffusion models require 'prompt' in data")
            
            prompt = input_data["prompt"]
            num_inference_steps = input_data.get("steps", 20)
            guidance_scale = input_data.get("guidance_scale", 7.5)
            
            # Add relevant features for cost estimation
            data_features.extend([
                f"prompt_length:{len(prompt)}",
                f"steps:{num_inference_steps}",
                f"guidance_scale:{guidance_scale}"
            ])
        
        # Get estimation from cost module
        estimated_time = get_estimation_cost(
            instance.model_name, 
            instance.device, 
            *data_features
        )
        
        return StandardResponse(
            success=True,
            message="Estimation completed",
            data={
                "estimated_execution_time": estimated_time,
                "estimated_time_unit": "seconds",
                "model_type": model_type,
                "instance_id": instance_id,
                "device": instance.device,
                "data_features": data_features,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Estimation failed for instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/standalone/status")
async def get_status():
    """Get server status and model information"""
    return {
        "models_on_disk": list(state.models_disk.keys()),
        "models_in_dram": list(state.models_dram.keys()),
        "serving_instances": [
            {
                "instance_id": instance.instance_id,
                "model_name": instance.model_name,
                "device": instance.device,
                "port": instance.port,
                "url": instance.url,
                "created_at": instance.created_at.isoformat()
            }
            for instance in state.serving_instances.values()
        ],
        "cache_directory": str(state.model_cache_dir),
        "used_ports": sorted(list(state.used_ports))
    }

@app.delete("/standalone/instance/{instance_id}")
async def stop_instance(instance_id: str):
    """Stop a specific serving instance"""
    try:
        if instance_id not in state.serving_instances:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
        
        instance = state.serving_instances[instance_id]
        model_name = instance.model_name
        port = instance.port
        
        # Release the port
        release_port(port)
        
        del state.serving_instances[instance_id]
        
        return StandardResponse(
            success=True,
            message=f"Successfully stopped instance {instance_id}",
            data={"instance_id": instance_id, "model_name": model_name, "port": port}
        )
        
    except Exception as e:
        logger.error(f"Failed to stop instance {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_server(host: str = "localhost", port: int = 8000):
    """Run the server"""
    logger.info(f"Starting GSwarm Standalone Model Server on {host}:{port}")
    logger.info(f"Model cache directory: {state.model_cache_dir}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GSwarm Standalone Model Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    run_server(args.host, args.port)

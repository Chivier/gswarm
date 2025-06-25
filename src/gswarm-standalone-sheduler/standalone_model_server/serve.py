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
    def __init__(self, model_name: str, device: str, instance_id: str = None):
        self.instance_id = instance_id or str(uuid.uuid4())
        self.model_name = model_name
        self.device = device
        self.created_at = datetime.now()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.process = None  # For external model servers like vLLM
        self.port = None
        self.url = None
        
class ServerState:
    def __init__(self):
        self.models_disk: Dict[str, str] = {}  # model_name -> path
        self.models_dram: Dict[str, Any] = {}  # model_name -> loaded model objects
        self.serving_instances: Dict[str, ModelInstance] = {}  # instance_id -> ModelInstance
        self.model_cache_dir = Path.home() / ".cache" / "gswarm" / "models"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

state = ServerState()

# Request/Response models
class ServeRequest(BaseModel):
    model_name: str
    device: str = Field(description="Device to serve on: cpu, cuda:0, cuda:1, etc.")

class OffloadRequest(BaseModel):
    model_name: str = Field(description="Model name to offload")
    target: str = Field(description="Where to offload: dram or disk")

class LoadRequest(BaseModel):
    model_name: str
    target: str = "dram"

class DownloadRequest(BaseModel):
    model_name: str
    source: str = "huggingface"

class CallRequest(BaseModel):
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

def load_model_to_dram(model_name: str, model_path: str):
    """Load model into DRAM for faster access"""
    try:
        model_type = get_model_type(model_name)
        
        if model_type == "llm":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("transformers library not available for LLM models")
                
            logger.info(f"Loading LLM model {model_name} to DRAM...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            state.models_dram[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "llm",
                "loaded_at": datetime.now()
            }
        elif model_type == "diffusion":
            logger.info(f"Loading diffusion model {model_name} to DRAM...")
            # For stable diffusion models, we'll use diffusers pipeline
            if DIFFUSERS_AVAILABLE:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                state.models_dram[model_name] = {
                    "pipeline": pipe,
                    "type": "diffusion",
                    "loaded_at": datetime.now()
                }
            else:
                if not TRANSFORMERS_AVAILABLE:
                    raise RuntimeError("Neither diffusers nor transformers available for diffusion models")
                logger.warning("diffusers not available, loading as generic model")
                # Fallback to generic loading
                model = AutoModel.from_pretrained(model_path)
                state.models_dram[model_name] = {
                    "model": model,
                    "type": "diffusion",
                    "loaded_at": datetime.now()
                }
        
        logger.info(f"Successfully loaded {model_name} to DRAM")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name} to DRAM: {e}")
        raise

def create_serving_instance(model_name: str, device: str) -> ModelInstance:
    """Create a serving instance for a model"""
    instance = ModelInstance(model_name, device)
    
    try:
        # Check if model is in DRAM first
        if model_name in state.models_dram:
            logger.info(f"Using model {model_name} from DRAM")
            dram_model = state.models_dram[model_name]
            instance.model = dram_model.get("model")
            instance.tokenizer = dram_model.get("tokenizer")
            instance.pipeline = dram_model.get("pipeline")
            
            # Move to specified device if needed
            if instance.model and device != "cpu":
                instance.model = instance.model.to(device)
                
        # Otherwise load from disk
        elif model_name in state.models_disk:
            model_path = state.models_disk[model_name]
            load_model_to_dram(model_name, model_path)
            # Recursively call to use DRAM version
            return create_serving_instance(model_name, device)
        else:
            raise ValueError(f"Model {model_name} not found in DRAM or disk cache")
        
        state.serving_instances[instance.instance_id] = instance
        logger.info(f"Created serving instance {instance.instance_id} for {model_name} on {device}")
        return instance
        
    except Exception as e:
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

@app.post("/standalone/download/{model_name}")
async def download_model(model_name: str, source: str = Query("huggingface")):
    """Download a model from the specified source to disk"""
    try:
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

@app.post("/standalone/load/{model_name}")
async def load_model(model_name: str, target: str = Query("dram")):
    """Load a model to DRAM for faster access"""
    try:
        if target.lower() != "dram":
            raise HTTPException(status_code=400, detail="Only 'dram' target is supported for loading")
        
        if model_name not in state.models_disk:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found on disk. Download it first.")
        
        if model_name in state.models_dram:
            return StandardResponse(
                success=True,
                message=f"Model {model_name} already loaded in DRAM",
                data={"already_loaded": True}
            )
        
        model_path = state.models_disk[model_name]
        load_model_to_dram(model_name, model_path)
        
        return StandardResponse(
            success=True,
            message=f"Successfully loaded {model_name} to DRAM",
            data={"target": target, "model_path": model_path}
        )
        
    except Exception as e:
        logger.error(f"Load failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standalone/serve/{model_name}")
async def serve_model(model_name: str, device: str = Query(None)):
    """Serve a model on the specified device, return instance endpoint"""
    try:
        if device is None:
            device = get_available_device()
        
        # Validate device
        if device.startswith("cuda:"):
            gpu_id = int(device.split(":")[1])
            if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
                raise HTTPException(status_code=400, detail=f"GPU device {device} not available")
        
        # Ensure model is available
        if model_name not in state.models_dram and model_name not in state.models_disk:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found. Download and load it first.")
        
        # Create serving instance
        instance = create_serving_instance(model_name, device)
        
        return StandardResponse(
            success=True,
            message=f"Successfully started serving {model_name}",
            data={
                "instance_id": instance.instance_id,
                "model_name": model_name,
                "device": device,
                "endpoint": f"/standalone/call/{instance.instance_id}",
                "created_at": instance.created_at.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Serve failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/standalone/offload/{model_name}")
async def offload_model(model_name: str, target: str = Query("disk")):
    """Stop model serving and move model to DRAM or disk"""
    try:
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
                # Direct model inference
                inputs = instance.tokenizer(prompt, return_tensors="pt")
                if instance.device != "cpu":
                    inputs = {k: v.to(instance.device) for k, v in inputs.items()}
                
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
                        "instance_id": instance_id
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
                        "instance_id": instance_id
                    }
                )
            else:
                raise HTTPException(status_code=500, detail="Diffusion pipeline not properly loaded")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Inference failed for instance {instance_id}: {e}")
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
                "created_at": instance.created_at.isoformat()
            }
            for instance in state.serving_instances.values()
        ],
        "cache_directory": str(state.model_cache_dir)
    }

@app.delete("/standalone/instance/{instance_id}")
async def stop_instance(instance_id: str):
    """Stop a specific serving instance"""
    try:
        if instance_id not in state.serving_instances:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
        
        instance = state.serving_instances[instance_id]
        model_name = instance.model_name
        del state.serving_instances[instance_id]
        
        return StandardResponse(
            success=True,
            message=f"Successfully stopped instance {instance_id}",
            data={"instance_id": instance_id, "model_name": model_name}
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

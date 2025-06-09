"""
Cache directory management utilities
"""

import os
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger


def get_cache_dir() -> Path:
    """Get the main gswarm cache directory"""
    cache_dir = Path.home() / ".cache" / "gswarm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_cache_dir(custom_path: Optional[str] = None) -> Path:
    """Get the model cache directory (defaults to HuggingFace cache)"""
    if custom_path:
        model_dir = Path(custom_path)
    else:
        model_dir = Path.home() / ".cache" / "huggingface"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def clean_history() -> bool:
    """Clean the gswarm cache directory"""
    try:
        cache_dir = get_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info(f"Cleaned cache directory: {cache_dir}")
            return True
        else:
            logger.info("Cache directory does not exist")
            return True
    except Exception as e:
        logger.error(f"Failed to clean cache directory: {e}")
        return False 


def scan_huggingface_models() -> List[Dict[str, Any]]:
    """Scan HuggingFace cache for already downloaded models"""
    discovered_models = []
    hf_cache_dir = get_model_cache_dir()
    
    try:
        # HuggingFace models are typically stored in:
        # ~/.cache/huggingface/hub/models--{org}--{model_name}/
        hub_dir = hf_cache_dir / "hub"
        
        if not hub_dir.exists():
            logger.info("No HuggingFace hub cache directory found")
            return discovered_models
        
        for model_dir in hub_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("models--"):
                try:
                    # Parse model name from directory (models--org--model_name)
                    model_parts = model_dir.name.replace("models--", "").split("--")
                    if len(model_parts) >= 2:
                        org = model_parts[0]
                        model_name = "--".join(model_parts[1:])  # Handle models with -- in name
                        full_model_name = f"{org}/{model_name}"
                        
                        # Check for model files to determine model type
                        model_type = detect_model_type(model_dir)
                        
                        # Get model size
                        model_size = get_directory_size(model_dir)
                        
                        discovered_models.append({
                            "model_name": full_model_name,
                            "model_type": model_type,
                            "local_path": str(model_dir),
                            "size": model_size,
                            "source": "huggingface_cache",
                            "stored_locations": ["disk"]  # HF cache is on disk
                        })
                        
                        logger.info(f"Discovered cached model: {full_model_name} ({model_size / 1e9:.2f} GB)")
                        
                except Exception as e:
                    logger.warning(f"Error processing model directory {model_dir}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error scanning HuggingFace cache: {e}")
    
    return discovered_models


def detect_model_type(model_dir: Path) -> str:
    """Detect model type based on files in the model directory"""
    # Look for config files to determine model type
    config_file = None
    
    # Find the actual model files (not just refs)
    for item in model_dir.rglob("*"):
        if item.is_file() and item.name == "config.json":
            config_file = item
            break
    
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Determine model type based on config
            architectures = config.get("architectures", [])
            model_type = config.get("model_type", "").lower()
            
            if any("llama" in arch.lower() for arch in architectures):
                return "llm"
            elif any("bert" in arch.lower() for arch in architectures):
                return "embedding"
            elif any("clip" in arch.lower() for arch in architectures):
                return "multimodal"
            elif any("diffusion" in arch.lower() or "unet" in arch.lower() for arch in architectures):
                return "diffusion"
            elif "text" in model_type or any("gpt" in arch.lower() for arch in architectures):
                return "llm"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Could not parse config.json: {e}")
    
    return "unknown"


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes"""
    total_size = 0
    try:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception as e:
        logger.warning(f"Could not calculate size for {path}: {e}")
    return total_size 
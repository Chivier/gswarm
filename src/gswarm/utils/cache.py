"""
Cache directory management utilities
"""

import os
import shutil
from pathlib import Path
from typing import Optional
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
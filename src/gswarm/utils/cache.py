"""
Cache directory management utilities with model variable storage for DRAM and GPU
"""

from gswarm.utils.config import get_huggingface_cache_dir

import os
import shutil
import json
import torch
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from loguru import logger
from transformers import AutoModel, AutoConfig
from safetensors import safe_open


def get_cache_dir() -> Path:
    """Get the main gswarm cache directory"""
    cache_dir = Path.home() / ".cache" / "gswarm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_cache_dir(custom_path: Optional[str] = None) -> Path:
    """Get the model cache directory (fixed to ~/.cache/gswarm/models)"""
    if custom_path:
        model_dir = Path(custom_path)
    else:
        # Always use the fixed path
        model_dir = Path.home() / ".cache" / "gswarm" / "models"

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


# Model variable storage for DRAM and GPU models
class ModelVariableStorage:
    """Storage for model variables in memory (DRAM) and GPU"""

    def __init__(self):
        self._dram_models: Dict[str, Any] = {}  # model_name -> model object
        self._gpu_models: Dict[str, Any] = {}  # instance_id -> model/inference object
        self._model_configs: Dict[str, Dict] = {}  # model_name -> config

    def store_dram_model(self, model_name: str, model_obj: Any, config: Optional[Dict] = None) -> bool:
        """Store a model object in DRAM"""
        try:
            self._dram_models[model_name] = model_obj
            if config:
                self._model_configs[model_name] = config
            logger.info(f"Stored model {model_name} in DRAM")
            return True
        except Exception as e:
            logger.error(f"Failed to store model {model_name} in DRAM: {e}")
            return False

    def get_dram_model(self, model_name: str) -> Optional[Any]:
        """Get a model object from DRAM"""
        return self._dram_models.get(model_name)

    def remove_dram_model(self, model_name: str) -> bool:
        """Remove a model from DRAM"""
        try:
            if model_name in self._dram_models:
                del self._dram_models[model_name]
                if model_name in self._model_configs:
                    del self._model_configs[model_name]
                logger.info(f"Removed model {model_name} from DRAM")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove model {model_name} from DRAM: {e}")
            return False

    def store_gpu_model(self, instance_id: str, model_obj: Any, config: Optional[Dict] = None) -> bool:
        """Store a GPU model/inference object"""
        try:
            self._gpu_models[instance_id] = model_obj
            if config:
                self._model_configs[instance_id] = config
            logger.info(f"Stored GPU model instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store GPU model instance {instance_id}: {e}")
            return False

    def get_gpu_model(self, instance_id: str) -> Optional[Any]:
        """Get a GPU model/inference object"""
        return self._gpu_models.get(instance_id)

    def remove_gpu_model(self, instance_id: str) -> bool:
        """Remove a GPU model instance"""
        try:
            if instance_id in self._gpu_models:
                del self._gpu_models[instance_id]
                if instance_id in self._model_configs:
                    del self._model_configs[instance_id]
                logger.info(f"Removed GPU model instance {instance_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove GPU model instance {instance_id}: {e}")
            return False

    def list_dram_models(self) -> List[str]:
        """List all models in DRAM"""
        return list(self._dram_models.keys())

    def list_gpu_models(self) -> List[str]:
        """List all GPU model instances"""
        return list(self._gpu_models.keys())

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        dram_count = len(self._dram_models)
        gpu_count = len(self._gpu_models)

        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i),
                    "reserved": torch.cuda.memory_reserved(i),
                    "total": torch.cuda.get_device_properties(i).total_memory,
                }

        return {"dram_models": dram_count, "gpu_models": gpu_count, "gpu_memory": gpu_memory}


# Global instance for model variable storage
model_storage = ModelVariableStorage()


def scan_gswarm_models() -> List[Dict[str, Any]]:
    """Scan gswarm model cache for already downloaded models"""
    discovered_models = []
    model_cache_dir = get_model_cache_dir()

    try:
        if not model_cache_dir.exists():
            logger.info("No gswarm model cache directory found")
            return discovered_models

        for model_dir in model_cache_dir.iterdir():
            if model_dir.is_dir():
                try:
                    model_name = model_dir.name

                    # Check for model files to determine model type
                    model_type = detect_model_type(model_dir)

                    # Get model size
                    model_size = get_directory_size(model_dir)

                    discovered_models.append(
                        {
                            "model_name": model_name,
                            "model_type": model_type,
                            "local_path": str(model_dir),
                            "size": model_size,
                            "source": "gswarm_cache",
                            "stored_locations": ["disk"],
                        }
                    )

                    logger.info(f"Discovered cached model: {model_name} ({model_size / 1e9:.2f} GB)")

                except Exception as e:
                    logger.warning(f"Error processing model directory {model_dir}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error scanning gswarm model cache: {e}")

    return discovered_models


def scan_huggingface_models() -> List[Dict[str, Any]]:
    """Scan HuggingFace cache for already downloaded models"""
    discovered_models = []
    hf_cache_dir = get_huggingface_cache_dir()

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

                        discovered_models.append(
                            {
                                "model_name": full_model_name,
                                "model_type": model_type,
                                "local_path": str(model_dir),
                                "size": model_size,
                                "source": "huggingface_cache",
                                "stored_locations": ["disk"],
                            }
                        )

                        logger.info(f"Discovered cached HF model: {full_model_name} ({model_size / 1e9:.2f} GB)")

                except Exception as e:
                    logger.warning(f"Error processing model directory {model_dir}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error scanning HuggingFace cache: {e}")

    return discovered_models


def scan_all_models() -> List[Dict[str, Any]]:
    """Scan both gswarm and HuggingFace caches for models"""
    all_models = []

    # Scan gswarm cache first
    gswarm_models = scan_gswarm_models()
    all_models.extend(gswarm_models)

    # Scan HuggingFace cache
    hf_models = scan_huggingface_models()
    all_models.extend(hf_models)

    logger.info(f"Found {len(all_models)} total cached models ({len(gswarm_models)} gswarm, {len(hf_models)} HF)")

    return all_models


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
            with open(config_file, "r") as f:
                config = json.load(f)

            # Determine model type based on config
            architectures = config.get("architectures", [])
            model_type = config.get("model_type", "").lower()

            if any("llama" in arch.lower() for arch in architectures):
                return "llm"
            elif any("bert" in arch.lower() for arch in architectures):
                return "llm"
            elif any("clip" in arch.lower() for arch in architectures):
                return "multimodal"
            elif any("diffusion" in arch.lower() or "unet" in arch.lower() for arch in architectures):
                return "diffusion"
            elif "text" in model_type or any("gpt" in arch.lower() for arch in architectures):
                return "llm"
            else:
                return "llm"

        except Exception as e:
            logger.warning(f"Could not parse config.json: {e}")

    return "llm"


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


def save_model_to_disk(model_name: str, model_obj: Any, target_path: Optional[Path] = None) -> bool:
    """Save a model object from DRAM back to disk"""
    try:
        if target_path is None:
            target_path = get_model_cache_dir() / model_name

        target_path.mkdir(parents=True, exist_ok=True)

        # Handle different model types
        if hasattr(model_obj, "save_pretrained"):
            # HuggingFace models
            model_obj.save_pretrained(target_path)
            logger.info(f"Saved HuggingFace model {model_name} to {target_path}")
        elif hasattr(model_obj, "state_dict"):
            # PyTorch models
            torch.save(model_obj.state_dict(), target_path / "model.pt")
            logger.info(f"Saved PyTorch model {model_name} to {target_path}")
        else:
            # Generic pickle save

            with open(target_path / "model.pkl", "wb") as f:
                pickle.dump(model_obj, f)
            logger.info(f"Saved model {model_name} to {target_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save model {model_name} to disk: {e}")
        return False


def load_safetensors_to_dram(model_path: Path, model_name: str) -> Optional[Any]:
    """Load safetensors model to DRAM and return the model variable"""
    try:
        # Try to load with transformers first
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="cpu",  # Load to CPU/DRAM first
            )

            # Store in our variable storage
            model_storage.store_dram_model(model_name, model, {"config": config})

            logger.info(f"Loaded model {model_name} to DRAM using transformers")
            return model

        except Exception as e:
            logger.warning(f"Failed to load with transformers: {e}")

            # Try safetensors directly

            safetensors_files = list(model_path.glob("*.safetensors"))
            if not safetensors_files:
                raise Exception("No safetensors files found")

            # Load the first safetensors file as example
            tensors = {}
            for sf_file in safetensors_files:
                with safe_open(sf_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)

            # Store tensors in our variable storage
            model_storage.store_dram_model(model_name, tensors)

            logger.info(f"Loaded safetensors {model_name} to DRAM")
            return tensors

    except Exception as e:
        logger.error(f"Failed to load safetensors {model_name} to DRAM: {e}")
        return None

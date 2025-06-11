"""
Updated REST client for gswarm_model system with new design.
"""

import requests
import platform
import psutil
import shutil
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from loguru import logger
import torch

from gswarm.model.fastapi_models import NodeInfo, StorageType, CopyMethod


class ModelClient:
    """Enhanced REST client for model management with GPU serving support"""

    def __init__(self, head_url: str, node_id: Optional[str] = None):
        self.head_url = head_url.rstrip("/")
        self.node_id = node_id or platform.node()
        self.session = requests.Session()

    def register_node(self) -> bool:
        """Register this node with the head"""
        try:
            node_info = self._get_node_info()
            response = self.session.post(f"{self.head_url}/nodes", json=node_info.model_dump(mode="json"))
            response.raise_for_status()
            result = response.json()
            logger.info(f"Node registered: {result['message']}")

            # Scan for local models after successful registration
            if result["success"]:
                self._discover_and_register_local_models()

            return result["success"]
        except Exception as e:
            logger.debug(f"Failed to register node: {e}")
            return False

    def _get_node_info(self) -> NodeInfo:
        """Get current node information with GPU details"""
        # Get storage info
        storage_devices = {}

        # Disk
        try:
            disk = shutil.disk_usage("/")
            storage_devices["disk"] = {"total": disk.total, "used": disk.used, "available": disk.free}
        except:
            pass

        # DRAM (Memory)
        try:
            mem = psutil.virtual_memory()
            storage_devices["dram"] = {"total": mem.total, "used": mem.used, "available": mem.available}
        except:
            pass

        # GPU info
        gpu_devices = {}
        gpu_count = 0

        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_devices[f"gpu{i}"] = {
                        "name": props.name,
                        "memory_total": props.total_memory,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    }
        except:
            pass

        return NodeInfo(
            node_id=self.node_id,
            hostname=platform.node(),
            ip_address=self._get_ip(),
            storage_devices=storage_devices,
            gpu_devices=gpu_devices,
            gpu_count=gpu_count,
        )

    def _get_ip(self) -> str:
        """Get local IP address"""
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"

    def heartbeat(self) -> bool:
        """Send heartbeat to head"""
        try:
            response = self.session.post(f"{self.head_url}/nodes/{self.node_id}/heartbeat")
            response.raise_for_status()
            return True
        except:
            return False

    # Model operations

    def list_models(self) -> List[Dict]:
        """List all models with checkpoints and serving info"""
        try:
            response = self.session.get(f"{self.head_url}/models")
            response.raise_for_status()
            return response.json()["models"]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_details(self, model_name: str) -> Optional[Dict]:
        """Get detailed model information"""
        try:
            response = self.session.get(f"{self.head_url}/models/{model_name}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model details: {e}")
            return None

    def register_model(
        self, name: str, model_type: str, source_url: Optional[str] = None, metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """Register a new model, returns (success, instance_id)"""
        try:
            response = self.session.post(
                f"{self.head_url}/models",
                json={"name": name, "type": model_type, "source_url": source_url, "metadata": metadata},
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Model registered: {result['message']}")
            instance_id = result.get("data", {}).get("instance_id")
            return result["success"], instance_id
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False, None

    def download_model(self, model_name: str, source_url: str, target_device: str = "disk") -> bool:
        """Download a model to disk or dram"""
        # Validate target device
        if target_device not in ["disk", "dram"]:
            logger.error(f"Invalid target device: {target_device}. Must be 'disk' or 'dram'")
            return False

        try:
            response = self.session.post(
                f"{self.head_url}/download",
                json={"model_name": model_name, "source_url": source_url, "target_device": target_device},
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Download initiated: {result['message']}")
            return result["success"]
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    def copy_model(
        self,
        model_name: str,
        source_device: str,
        target_device: str,
        keep_source: bool = True,
        use_pinned_memory: bool = True,
    ) -> bool:
        """Copy model between devices (disk/dram/gpu)"""
        try:
            response = self.session.post(
                f"{self.head_url}/copy",
                json={
                    "model_name": model_name,
                    "source_device": source_device,
                    "target_device": target_device,
                    "keep_source": keep_source,
                    "use_pinned_memory": use_pinned_memory,
                },
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Copy operation: {result['message']}")
            return result["success"]
        except Exception as e:
            logger.error(f"Failed to copy model: {e}")
            return False

    def serve_model(
        self,
        model_name: str,
        source_device: str,
        gpu_device: str,
        port: int,
        instance_id: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """Start serving a model on GPU, returns (success, server_info)"""
        try:
            response = self.session.post(
                f"{self.head_url}/serve",
                json={
                    "model_name": model_name,
                    "source_device": source_device,
                    "gpu_device": gpu_device,
                    "port": port,
                    "instance_id": instance_id,
                    "config": config,
                },
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Model serving: {result['message']}")
            return result["success"], result.get("data")
        except Exception as e:
            logger.error(f"Failed to serve model: {e}")
            return False, None

    def serve_multiple_instances(
        self,
        model_name: str,
        source_device: str,
        gpu_device: str,
        num_instances: int,
        base_port: int = 8000,
        config: Optional[Dict] = None,
    ) -> List[Dict]:
        """Start multiple serving instances of the same model on a GPU"""
        instances = []

        for i in range(num_instances):
            port = base_port + i
            success, server_info = self.serve_model(
                model_name=model_name, source_device=source_device, gpu_device=gpu_device, port=port, config=config
            )

            if success and server_info:
                instances.append(server_info)
                logger.info(f"Started instance {i + 1}/{num_instances}: {server_info['instance_id']}")
            else:
                logger.error(f"Failed to start instance {i + 1}/{num_instances}")

        return instances

    def stop_serving(self, model_name: str, instance_id: str) -> bool:
        """Stop a specific serving instance"""
        try:
            response = self.session.post(
                f"{self.head_url}/stop_serve", json={"model_name": model_name, "instance_id": instance_id}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Stop serving: {result['message']}")
            return result["success"]
        except Exception as e:
            logger.error(f"Failed to stop serving: {e}")
            return False

    def list_serving_instances(self) -> List[Dict]:
        """List all serving instances"""
        try:
            response = self.session.get(f"{self.head_url}/serving")
            response.raise_for_status()
            return response.json()["instances"]
        except Exception as e:
            logger.error(f"Failed to list serving instances: {e}")
            return []

    def get_model_serving_instances(self, model_name: str) -> List[Dict]:
        """Get all serving instances for a specific model"""
        try:
            response = self.session.get(f"{self.head_url}/serving/{model_name}")
            response.raise_for_status()
            return response.json()["instances"]
        except Exception as e:
            logger.error(f"Failed to get model serving instances: {e}")
            return []

    # Convenience methods

    def load_to_gpu(
        self, model_name: str, source_device: str = "disk", gpu_device: str = "gpu0", port: Optional[int] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """Load a model from disk/dram to GPU and start serving"""
        if port is None:
            # Find available port
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        return self.serve_model(model_name=model_name, source_device=source_device, gpu_device=gpu_device, port=port)

    def preload_to_dram(self, model_name: str) -> bool:
        """Preload model from disk to DRAM for faster GPU loading"""
        return self.copy_model(model_name=model_name, source_device="disk", target_device="dram", keep_source=True)

    def offload_from_gpu(self, model_name: str, instance_id: str, target_device: str = "dram") -> bool:
        """Stop serving and optionally save checkpoint"""
        # First stop serving
        if not self.stop_serving(model_name, instance_id):
            return False

        # GPU state is automatically cleared when serving stops
        logger.info(f"Model {model_name} instance {instance_id} offloaded from GPU")
        return True

    # Job operations

    def create_job(self, name: str, actions: List[Dict], description: Optional[str] = None) -> Optional[str]:
        """Create a job"""
        try:
            response = self.session.post(
                f"{self.head_url}/jobs", json={"name": name, "description": description, "actions": actions}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Job created: {result['message']}")
            return result["job_id"]
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return None

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        try:
            response = self.session.get(f"{self.head_url}/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    def get_download_status(self, model_name: str) -> Optional[Dict]:
        """Get model download/status information"""
        try:
            response = self.session.get(f"{self.head_url}/models/{model_name}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get download status: {e}")
            return None

    def validate_model(self, model_name: str) -> Optional[Dict]:
        """Validate a specific model and get updated status"""
        try:
            response = self.session.get(f"{self.head_url}/models/{model_name}/validate")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to validate model: {e}")
            return None

    def validate_all_models(self) -> Optional[Dict]:
        """Validate all models and get updated statuses"""
        try:
            response = self.session.post(f"{self.head_url}/models/validate")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to validate all models: {e}")
            return None

    def is_model_ready(self, model_name: str) -> bool:
        """Check if a model is ready with valid cache paths"""
        try:
            model_details = self.get_model_details(model_name)
            if not model_details:
                return False

            status = model_details.get("status", "")
            checkpoints = model_details.get("checkpoints", {})

            # Model is ready if status is 'ready' and has at least one valid checkpoint
            return status == "ready" and len(checkpoints) > 0
        except Exception as e:
            logger.error(f"Failed to check if model is ready: {e}")
            return False

    def _discover_and_register_local_models(self) -> None:
        """Discover and register locally cached models"""
        try:
            from gswarm.utils.cache import scan_huggingface_models

            discovered_models = scan_huggingface_models()

            if not discovered_models:
                logger.info("No cached HuggingFace models found")
                return

            logger.info(f"Found {len(discovered_models)} cached models, registering...")

            for model_info in discovered_models:
                try:
                    # Register each discovered model
                    success, instance_id = self.register_model(
                        name=model_info["model_name"],
                        model_type=model_info["model_type"],
                        source_url=f"https://huggingface.co/{model_info['model_name']}",
                        metadata={
                            "local_path": model_info["local_path"],
                            "size": model_info["size"],
                            "source": "discovered_cache",
                            "auto_discovered": True,
                        },
                    )

                    if success:
                        logger.info(f"✓ Registered cached model: {model_info['model_name']} (instance: {instance_id})")
                        # Validate that the model is properly registered with a valid path
                        model_details = self.get_model_details(model_info["model_name"])
                        if model_details and model_details.get("status") == "ready":
                            logger.info(f"✓ Model {model_info['model_name']} confirmed ready with valid path")
                        elif model_details and model_details.get("status") == "registered":
                            logger.warning(f"⚠ Model {model_info['model_name']} registered but no valid path found")
                    else:
                        logger.warning(f"✗ Failed to register: {model_info['model_name']}")

                except Exception as e:
                    logger.warning(f"Error registering model {model_info['model_name']}: {e}")

        except Exception as e:
            logger.error(f"Error during model discovery: {e}")

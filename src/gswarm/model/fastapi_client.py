"""
Simplified REST client for gswarm_model system.
"""

import requests
import platform
import psutil
import shutil
from typing import Dict, Optional, List, Any
from datetime import datetime
from loguru import logger

from gswarm.model.fastapi_models import NodeInfo


class ModelClient:
    """Simple REST client for model management"""
    
    def __init__(self, head_url: str, node_id: Optional[str] = None):
        self.head_url = head_url.rstrip('/')
        self.node_id = node_id or platform.node()
        self.session = requests.Session()
        
    def register_node(self) -> bool:
        """Register this node with the head"""
        try:
            node_info = self._get_node_info()
            response = self.session.post(
                f"{self.head_url}/nodes",
                json=node_info.model_dump(mode='json')
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Node registered: {result['message']}")
            return result['success']
        except Exception as e:
            logger.debug(f"Failed to register node: {e}")
            return False
    
    def _get_node_info(self) -> NodeInfo:
        """Get current node information"""
        # Get storage info
        storage_devices = {}
        
        # Disk
        try:
            disk = shutil.disk_usage("/")
            storage_devices["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "available": disk.free
            }
        except:
            pass
        
        # Memory
        try:
            mem = psutil.virtual_memory()
            storage_devices["dram"] = {
                "total": mem.total,
                "used": mem.used,
                "available": mem.available
            }
        except:
            pass
        
        # GPU count (simplified)
        gpu_count = self._get_gpu_count()
        
        return NodeInfo(
            node_id=self.node_id,
            hostname=platform.node(),
            ip_address=self._get_ip(),
            storage_devices=storage_devices,
            gpu_count=gpu_count
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
    
    def _get_gpu_count(self) -> int:
        """Get GPU count"""
        try:
            import nvitop
            return len(nvitop.Device.all())
        except:
            try:
                import pynvml
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return count
            except:
                return 0
    
    def heartbeat(self) -> bool:
        """Send heartbeat to head"""
        try:
            response = self.session.post(
                f"{self.head_url}/nodes/{self.node_id}/heartbeat"
            )
            response.raise_for_status()
            return True
        except:
            return False
    
    def list_models(self) -> List[Dict]:
        """List all models"""
        try:
            response = self.session.get(f"{self.head_url}/models")
            response.raise_for_status()
            return response.json()['models']
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def register_model(self, name: str, model_type: str, 
                      source_url: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> bool:
        """Register a new model"""
        try:
            response = self.session.post(
                f"{self.head_url}/models",
                json={
                    "name": name,
                    "type": model_type,
                    "source_url": source_url,
                    "metadata": metadata
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Model registered: {result['message']}")
            return result['success']
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def download_model(self, model_name: str, source_url: str, 
                      target_device: str) -> bool:
        """Download a model"""
        try:
            response = self.session.post(
                f"{self.head_url}/download",
                json={
                    "model_name": model_name,
                    "source_url": source_url,
                    "target_device": target_device
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Download initiated: {result['message']}")
            return result['success']
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def serve_model(self, model_name: str, device: str, port: int,
                   config: Optional[Dict] = None) -> bool:
        """Start serving a model"""
        try:
            response = self.session.post(
                f"{self.head_url}/serve",
                json={
                    "model_name": model_name,
                    "device": device,
                    "port": port,
                    "config": config
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Model serving: {result['message']}")
            return result['success']
        except Exception as e:
            logger.error(f"Failed to serve model: {e}")
            return False
    
    def create_job(self, name: str, actions: List[Dict],
                  description: Optional[str] = None) -> Optional[str]:
        """Create a job"""
        try:
            response = self.session.post(
                f"{self.head_url}/jobs",
                json={
                    "name": name,
                    "description": description,
                    "actions": actions
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Job created: {result['message']}")
            return result['job_id']
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
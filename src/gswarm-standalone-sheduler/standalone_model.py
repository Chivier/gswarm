import argparse
import asyncio
import grpc
import requests
from typing import Dict, List, Optional, Any
from loguru import logger

# Import necessary models and gRPC components
from gswarm.model.fastapi_client import ModelClient
from gswarm.profiler import profiler_pb2, profiler_pb2_grpc

class StandaloneStrategyAPI:
    """API helper for standalone strategy to read system status"""
    
    def __init__(self, model_head_url: str, profiler_head_address: str):
        """
        Initialize the API helper
        
        Args:
            model_head_url: URL of the model management head (e.g., "http://localhost:9010")
            profiler_head_address: Address of the profiler head (e.g., "localhost:9011")
        """
        self.model_head_url = model_head_url.rstrip("/")
        self.profiler_head_address = profiler_head_address
        self.model_client = ModelClient(model_head_url)
    
    def read_all_device_status(self) -> List[Dict]:
        """
        Read all device status and load them into a list[dict]
        
        Returns:
            List[Dict]: List of device status dictionaries containing GPU and storage info
        """
        try:
            # Use asyncio to run the async gRPC call
            return asyncio.run(self._read_device_status_async())
        except Exception as e:
            logger.error(f"Failed to read device status: {e}")
            return []
    
    async def _read_device_status_async(self) -> List[Dict]:
        """Async implementation of device status reading via gRPC"""
        devices_list = []
        
        try:
            async with grpc.aio.insecure_channel(self.profiler_head_address) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                
                # Read cluster status to get all nodes
                request = profiler_pb2.ReadClusterStatusRequest()
                response = await stub.ReadClusterStatus(request)
                
                if not response.success:
                    logger.error(f"Failed to read cluster status: {response.message}")
                    return devices_list
                
                # Convert gRPC response to list of dictionaries
                for node_status in response.nodes:
                    node_dict = {
                        "node_id": node_status.node_id,
                        "gpus": []
                    }
                    
                    # Add GPU information
                    for gpu in node_status.gpus:
                        gpu_dict = {
                            "gpu_id": gpu.gpu_id,
                            "utilization": gpu.utilization,
                            "memory_used_mb": gpu.memory_used,
                            "memory_total_mb": gpu.memory_total,
                            "dram_bandwidth_gbps": gpu.dram_bandwidth,
                            "nvlink_bandwidth_gbps": gpu.nvlink_bandwidth,
                            "temperature": gpu.temperature if gpu.temperature > 0 else None,
                            "power": gpu.power if gpu.power > 0 else None,
                        }
                        node_dict["gpus"].append(gpu_dict)
                    
                    devices_list.append(node_dict)
                
                logger.info(f"Successfully read status for {len(devices_list)} devices")
                return devices_list
                
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC error reading device status: {e}")
            return devices_list
        except Exception as e:
            logger.error(f"Error reading device status: {e}")
            return devices_list
    
    def read_all_model_status(self) -> Dict:
        """
        Read all model status and load them into a dict
        
        Returns:
            Dict: Dictionary containing model status information with model names as keys
        """
        try:
            # Get models from model management system
            models = self.model_client.list_models()
            
            # Convert to dictionary with model names as keys
            model_status_dict = {}
            
            for model in models:
                model_name = model.get("name")
                if model_name:
                    model_status_dict[model_name] = {
                        "name": model_name,
                        "type": model.get("type"),
                        "size": model.get("size"),
                        "status": model.get("status"),
                        "checkpoints": model.get("checkpoints", []),
                        "serving_instances": model.get("serving_instances", 0),
                        "dram_loaded": model.get("dram_loaded", False),
                        "metadata": model.get("metadata"),
                        "created_at": model.get("created_at"),
                        "paths_validated": model.get("paths_validated", False),
                        "has_valid_cache": model.get("has_valid_cache", False)
                    }
            
            logger.info(f"Successfully read status for {len(model_status_dict)} models")
            return model_status_dict
            
        except Exception as e:
            logger.error(f"Failed to read model status: {e}")
            return {}
    
    def read_node_storage_info(self) -> List[Dict]:
        """
        Read storage information from all nodes
        
        Returns:
            List[Dict]: List of node storage information
        """
        try:
            # Use requests to get node information from model management system
            response = requests.get(f"{self.model_head_url}/nodes")
            response.raise_for_status()
            
            nodes_data = response.json()
            storage_info = []
            
            for node in nodes_data.get("nodes", []):
                node_storage = {
                    "node_id": node.get("node_id"),
                    "hostname": node.get("hostname"),
                    "ip_address": node.get("ip_address"),
                    "storage_devices": node.get("storage_devices", {}),
                    "gpu_devices": node.get("gpu_devices", {}),
                    "gpu_count": node.get("gpu_count", 0),
                    "last_seen": node.get("last_seen"),
                    "is_online": True  # Assuming if it's in the list, it's online
                }
                storage_info.append(node_storage)
            
            logger.info(f"Successfully read storage info for {len(storage_info)} nodes")
            return storage_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to read node storage info: {e}")
            return []
        except Exception as e:
            logger.error(f"Error reading node storage info: {e}")
            return []
    
    def get_system_status_summary(self) -> Dict:
        """
        Get a comprehensive system status summary
        
        Returns:
            Dict: System status summary including devices, models, and storage
        """
        try:
            # Get system status from model management system
            response = requests.get(f"{self.model_head_url}/status")
            response.raise_for_status()
            
            system_status = response.json()
            
            # Combine with device and model status
            device_status = self.read_all_device_status()
            model_status = self.read_all_model_status()
            
            summary = {
                "system_status": system_status,
                "total_devices": len(device_status),
                "total_models": len(model_status),
                "device_details": device_status,
                "model_details": model_status,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get system status summary: {e}")
            return {}

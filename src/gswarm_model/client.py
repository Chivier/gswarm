"""
Client node implementation for gswarm_model system.
Worker nodes that store, serve, and manage models.
"""

import asyncio
import platform
import psutil
import shutil
import socket
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
import grpc
from loguru import logger

from .models import ClientModelInfo, StorageInfo, ModelStatus, parse_device_name

# Import generated protobuf classes
try:
    from . import model_pb2
    from . import model_pb2_grpc
except ImportError:
    logger.error("gRPC protobuf files not found. Please run 'python generate_grpc.py' first.")
    raise


class ClientNodeState:
    """State for the client node"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.hostname = platform.node()
        self.ip_address = self.get_local_ip()
        
        # Local model registry
        self.local_models: Dict[str, ClientModelInfo] = {}
        
        # Storage devices
        self.storage_devices: Dict[str, StorageInfo] = {}
        
        # Connection info
        self.head_address = None
        self.connected = False
        
        # Initialize storage info
        self.update_storage_info()
    
    def get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to get the local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def update_storage_info(self):
        """Update storage device information"""
        # Disk storage
        try:
            disk_usage = shutil.disk_usage("/")
            self.storage_devices["disk"] = StorageInfo(
                total=disk_usage.total,
                used=disk_usage.used,
                available=disk_usage.free
            )
        except Exception as e:
            logger.warning(f"Failed to get disk usage: {e}")
        
        # RAM storage
        try:
            memory = psutil.virtual_memory()
            self.storage_devices["dram"] = StorageInfo(
                total=memory.total,
                used=memory.used,
                available=memory.available
            )
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
        
        # GPU storage - improved memory detection based on gswarm_profiler
        try:
            import nvitop
            devices = nvitop.Device.all()
            for i, device in enumerate(devices):
                try:
                    # Try nvitop memory detection first (most reliable)
                    memory = device.memory()
                    gpu_total = memory.total
                    gpu_used = memory.used
                    gpu_available = gpu_total - gpu_used
                    
                    self.storage_devices[f"gpu{i}"] = StorageInfo(
                        total=gpu_total,
                        used=gpu_used,
                        available=gpu_available
                    )
                except (AttributeError, NotImplementedError) as e:
                    logger.debug(f"nvitop memory API not available for GPU {i}: {e}")
                    # Try alternative approaches for GPU memory detection
                    try:
                        # Fallback using pynvml directly
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # Fixed: use correct function
                        gpu_total = info.total
                        gpu_used = info.used
                        gpu_available = info.free
                        
                        self.storage_devices[f"gpu{i}"] = StorageInfo(
                            total=gpu_total,
                            used=gpu_used,
                            available=gpu_available
                        )
                        pynvml.nvmlShutdown()
                    except Exception as e2:
                        logger.warning(f"Fallback GPU memory detection also failed for GPU {i}: {e2}")
                        # Final fallback to minimal GPU info if we can't get memory details
                        self.storage_devices[f"gpu{i}"] = StorageInfo(
                            total=0,
                            used=0,
                            available=0
                        )
                except Exception as e:
                    logger.warning(f"Error collecting memory info for GPU {i}: {e}")
                    # Fallback: Create storage device entry even without memory info
                    self.storage_devices[f"gpu{i}"] = StorageInfo(
                        total=0,
                        used=0,
                        available=0
                    )
        except ImportError:
            logger.warning("nvitop not available, trying direct pynvml approach")
            # Try direct pynvml approach
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # Fixed: use correct function
                        gpu_total = info.total
                        gpu_used = info.used
                        gpu_available = info.free
                        
                        self.storage_devices[f"gpu{i}"] = StorageInfo(
                            total=gpu_total,
                            used=gpu_used,
                            available=gpu_available
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get memory info for GPU {i}: {e}")
                        self.storage_devices[f"gpu{i}"] = StorageInfo(
                            total=0,
                            used=0,
                            available=0
                        )
                pynvml.nvmlShutdown()
            except ImportError:
                logger.warning("pynvml not available, cannot get GPU memory information")
            except Exception as e:
                logger.warning(f"Failed to get GPU information: {e}")
        except Exception as e:
            logger.warning(f"Failed to get GPU information: {e}")


def create_node_registration(state: ClientNodeState) -> model_pb2.NodeRegistration:
    """Create node registration message"""
    
    # Convert storage devices to protobuf format
    storage_devices = []
    for device_name, storage_info in state.storage_devices.items():
        storage_device = model_pb2.StorageDevice(
            device_name=device_name,
            storage_type=device_name.split("_")[0] if "_" in device_name else device_name,
            total_capacity=storage_info.total,
            used_capacity=storage_info.used,
            available_capacity=storage_info.available
        )
        storage_devices.append(storage_device)
    
    # GPU info (simplified)
    gpu_info = get_gpu_information()
    
    return model_pb2.NodeRegistration(
        node_id=state.node_id,
        hostname=state.hostname,
        ip_address=state.ip_address,
        storage_devices=storage_devices,
        gpu_info=gpu_info
    )


async def connect_to_head(state: ClientNodeState, head_address: str) -> bool:
    """Connect to head node and register"""
    try:
        async with grpc.aio.insecure_channel(head_address) as channel:
            stub = model_pb2_grpc.ModelManagerStub(channel)
            
            # Register with head node
            registration = create_node_registration(state)
            response = await stub.Connect(registration)
            
            if response.success:
                logger.info(f"Connected to head node: {response.message}")
                state.head_address = head_address
                state.connected = True
                return True
            else:
                logger.error(f"Failed to connect: {response.message}")
                return False
                
    except Exception as e:
        logger.error(f"Error connecting to head node: {e}")
        return False


async def send_heartbeat(state: ClientNodeState):
    """Send periodic heartbeat to head node"""
    if not state.connected or not state.head_address:
        return
    
    try:
        async with grpc.aio.insecure_channel(state.head_address) as channel:
            stub = model_pb2_grpc.ModelManagerStub(channel)
            await stub.Heartbeat(model_pb2.Empty())
            logger.debug("Heartbeat sent")
            
    except Exception as e:
        logger.warning(f"Failed to send heartbeat: {e}")
        state.connected = False


async def heartbeat_loop(state: ClientNodeState):
    """Periodic heartbeat loop"""
    while True:
        if state.connected:
            await send_heartbeat(state)
        
        # Update storage info periodically
        state.update_storage_info()
        
        await asyncio.sleep(30)  # Send heartbeat every 30 seconds


async def run_client_node(node_id: str, head_address: str):
    """Run the client node"""
    logger.info(f"Starting client node {node_id}")
    
    # Initialize state
    state = ClientNodeState(node_id)
    logger.info(f"Node hostname: {state.hostname}")
    logger.info(f"Node IP: {state.ip_address}")
    logger.info(f"Storage devices: {list(state.storage_devices.keys())}")
    
    # Connect to head node
    if not await connect_to_head(state, head_address):
        logger.error("Failed to connect to head node. Exiting.")
        return
    
    # Start heartbeat loop
    heartbeat_task = asyncio.create_task(heartbeat_loop(state))
    
    try:
        # Keep the client running
        while True:
            await asyncio.sleep(1)
            
            # Reconnect if disconnected
            if not state.connected:
                logger.info("Attempting to reconnect to head node...")
                await connect_to_head(state, head_address)
                
    except KeyboardInterrupt:
        logger.info("Client shutdown requested")
    finally:
        heartbeat_task.cancel()
        logger.info("Client node shutting down")


def start_client_node(node_id: str, head_address: str):
    """Start the client node (synchronous wrapper)"""
    try:
        asyncio.run(run_client_node(node_id, head_address))
    except KeyboardInterrupt:
        logger.info("Client shutdown requested")
    finally:
        logger.info("Client node exiting")


def detect_gsprofile_service(host: str = "localhost", port: int = 8090) -> Optional[Dict]:
    """
    Detect if gswarm-profiler is running and get GPU information from it.
    
    Args:
        host: gswarm-profiler host
        port: gswarm-profiler gRPC port
        
    Returns:
        Dict with GPU information if profiler is detected, None otherwise
    """
    try:
        # Try HTTP API first (more stable)
        http_ports = [8091, 8080, 8081]  # Common HTTP ports for gsprofile
        for http_port in http_ports:
            try:
                url = f"http://{host}:{http_port}/clients"
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Detected gswarm-profiler at {host}:{http_port}")
                    return {"clients": data.get("clients", {}), "method": "http", "port": http_port}
            except:
                continue
        
        # Fallback to gRPC if HTTP doesn't work
        try:
            import asyncio
            # Try to import profiler modules - they might not be available
            try:
                import sys
                import os
                # Add the parent directory to path to access gswarm_profiler
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)
                
                from gswarm_profiler import profiler_pb2, profiler_pb2_grpc
                
                async def check_grpc():
                    try:
                        async with grpc.aio.insecure_channel(f"{host}:{port}") as channel:
                            stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                            response = await asyncio.wait_for(stub.GetStatus(profiler_pb2.Empty()), timeout=2)
                            return {"connected_clients": response.connected_clients, "method": "grpc", "port": port}
                    except:
                        return None
                
                return asyncio.run(check_grpc())
            except ImportError:
                logger.debug("gswarm_profiler modules not available for gRPC detection")
                return None
        except:
            pass
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to detect gswarm-profiler: {e}")
        return None


def get_gpu_info_from_profiler(profiler_info: Dict) -> List[Dict[str, Any]]:
    """Extract GPU information from profiler data as dictionaries."""
    gpus = []
    
    if profiler_info.get("method") == "http":
        clients = profiler_info.get("clients", [])
        # Check if clients is a list (new API format) or dict (legacy format)
        if isinstance(clients, list):
            # New gswarm-profiler API format: clients is a list
            for client_data in clients:
                if "gpus" in client_data:
                    for i, gpu in enumerate(client_data["gpus"]):
                        gpus.append({
                            "physical_idx": i,
                            "name": gpu.get("name", "Unknown GPU"),
                            "id": f"gpu{i}"
                        })
        elif isinstance(clients, dict):
            # Legacy format: clients is a dict
            for client_id, client_data in clients.items():
                if "gpus" in client_data:
                    for i, gpu in enumerate(client_data["gpus"]):
                        gpus.append({
                            "physical_idx": i,
                            "name": gpu.get("name", "Unknown GPU"),
                            "id": f"gpu{i}"
                        })
        else:
            logger.warning(f"Unexpected clients data format: {type(clients)}")
    elif profiler_info.get("method") == "grpc":
        # For gRPC, we get client hostnames but not detailed GPU info
        # This is a limitation - we'll fall back to nvml for actual GPU names
        return get_gpu_info_from_nvml()
    
    return gpus if gpus else get_gpu_info_from_nvml()


def get_gpu_info_from_nvml() -> List[Dict[str, Any]]:
    """Get GPU information using nvml/nvitop as fallback, returning dictionaries."""
    try:
        import nvitop
        devices = nvitop.Device.all()
        if devices:
            gpus = []
            for i, device in enumerate(devices):
                try:
                    gpu_info = {
                        "physical_idx": i,
                        "name": device.name(),
                        "id": f"gpu{i}"
                    }
                    # Try to get additional info
                    try:
                        memory = device.memory()
                        gpu_info["memory_total"] = memory.total
                        gpu_info["memory_used"] = memory.used
                        gpu_info["memory_free"] = memory.total - memory.used
                    except (AttributeError, NotImplementedError):
                        logger.debug(f"Memory info not available for GPU {i}")
                    
                    gpus.append(gpu_info)
                except Exception as e:
                    logger.warning(f"Error getting info for GPU {i}: {e}")
                    # Fallback basic info
                    gpus.append({
                        "physical_idx": i,
                        "name": f"GPU_{i}",
                        "id": f"gpu{i}"
                    })
            return gpus
        else:
            logger.warning("No NVIDIA GPUs detected via nvitop")
            return []
    except ImportError:
        logger.warning("nvitop not available, trying pynvml for GPU detection")
        # Fallback to pynvml for basic GPU info
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    # Handle both string and bytes return types
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    
                    gpu_info = {
                        "physical_idx": i,
                        "name": name,
                        "id": f"gpu{i}"
                    }
                    
                    # Try to get memory info
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_info["memory_total"] = mem_info.total
                        gpu_info["memory_used"] = mem_info.used
                        gpu_info["memory_free"] = mem_info.free
                    except Exception:
                        logger.debug(f"Memory info not available for GPU {i}")
                    
                    gpus.append(gpu_info)
                except Exception as e:
                    logger.warning(f"Error getting info for GPU {i}: {e}")
                    gpus.append({
                        "physical_idx": i,
                        "name": f"GPU_{i}",
                        "id": f"gpu{i}"
                    })
            pynvml.nvmlShutdown()
            return gpus
        except ImportError:
            logger.warning("pynvml not available, cannot detect GPUs")
            return []
        except Exception as e:
            logger.warning(f"Failed to detect GPUs via pynvml: {e}")
            return []
    except Exception as e:
        logger.warning(f"Failed to detect GPUs via nvitop: {e}")
        return []


def get_gpu_information() -> List[str]:
    """
    Get GPU information by detecting gswarm-profiler first, then falling back to nvml.
    
    Returns:
        List of GPU names (strings) for protobuf compatibility
    """
    # Try to detect gswarm-profiler
    profiler_info = detect_gsprofile_service()
    
    if profiler_info:
        logger.info("Using gswarm-profiler for GPU detection")
        gpu_dicts = get_gpu_info_from_profiler(profiler_info)
        # Convert to strings for protobuf compatibility
        return [gpu["name"] for gpu in gpu_dicts]
    else:
        logger.info("gswarm-profiler not detected, using nvml for GPU detection")
        gpu_dicts = get_gpu_info_from_nvml()
        # Convert to strings for protobuf compatibility
        return [gpu["name"] for gpu in gpu_dicts]


def get_gpu_information_detailed() -> List[Dict[str, Any]]:
    """
    Get detailed GPU information as dictionaries.
    
    Returns:
        List of GPU dictionaries for internal use
    """
    # Try to detect gswarm-profiler
    profiler_info = detect_gsprofile_service()
    
    if profiler_info:
        logger.info("Using gswarm-profiler for GPU detection")
        return get_gpu_info_from_profiler(profiler_info)
    else:
        logger.info("gswarm-profiler not detected, using nvml for GPU detection")
        return get_gpu_info_from_nvml() 
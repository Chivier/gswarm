"""
Client node implementation for gswarm_model system.
Worker nodes that store, serve, and manage models.
"""

import asyncio
import platform
import psutil
import shutil
import socket
from datetime import datetime
from typing import Dict, List, Optional
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
        
        # GPU storage (simplified - would need nvidia-ml-py for real implementation)
        # For now, just add mock GPU info
        self.storage_devices["gpu0"] = StorageInfo(
            total=24 * 1024**3,  # 24GB
            used=0,
            available=24 * 1024**3
        )


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
    gpu_info = ["NVIDIA GeForce RTX 4090"]  # Mock GPU info
    
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
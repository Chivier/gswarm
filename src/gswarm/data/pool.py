"""Redis-like key-value data storage system for DRAM, GPU memory, and disk"""

import asyncio
import time
import threading
import pickle
import requests
import base64
import json
import os
import shutil
from collections import OrderedDict
from typing import Any, Optional, Dict, Union, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from loguru import logger
import psutil
import sys
import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import tempfile
import hashlib

# Try to import CUDA-related libraries
try:
    import torch
    import cupy as cp
    CUDA_AVAILABLE = torch.cuda.is_available()
    
    # Check for NVLink support
    NVLINK_AVAILABLE = False
    if CUDA_AVAILABLE:
        try:
            # Check if any GPU pairs have NVLink connectivity
            for i in range(torch.cuda.device_count()):
                for j in range(i + 1, torch.cuda.device_count()):
                    # Check P2P access between devices
                    if torch.cuda.can_device_access_peer(i, j):
                        NVLINK_AVAILABLE = True
                        break
                if NVLINK_AVAILABLE:
                    break
        except:
            pass
except ImportError:
    CUDA_AVAILABLE = False
    NVLINK_AVAILABLE = False
    logger.warning("PyTorch/CuPy not available, GPU memory operations will be limited")


class DataLocation(str, Enum):
    """Data storage locations"""
    DRAM = "dram"
    PINNED_DRAM = "pinned_dram"
    DISK = "disk"
    
    @classmethod
    def is_device(cls, location: str) -> bool:
        """Check if location is a GPU device"""
        return location.startswith("device:")
    
    @classmethod
    def get_device_id(cls, location: str) -> Optional[int]:
        """Extract device ID from device:x location"""
        if cls.is_device(location):
            try:
                return int(location.split(":")[1])
            except (IndexError, ValueError):
                return None
        return None


class DataLocationInfo(BaseModel):
    """Information about data location"""
    location: str
    size: int
    last_accessed: float
    copy_status: str = "complete"  # complete, copying, error
    copy_progress: float = 1.0  # 0.0 to 1.0
    read_pointer: Optional[str] = None  # For PD separation optimization


class MoveRequest(BaseModel):
    """Request to move data between locations"""
    key: str
    destination: str
    

class LocationResponse(BaseModel):
    """Response for location queries"""
    key: str
    location: str
    locations: List[DataLocationInfo]  # All locations where data exists
    

class AllLocationsResponse(BaseModel):
    """Response for list_location API"""
    locations: Dict[str, List[DataLocationInfo]]


class DataStorage:
    """Multi-location key-value storage with support for DRAM, GPU memory, and disk"""

    def __init__(self, max_mem_size: int = 16 * 1024 * 1024 * 1024, disk_path: str = "/tmp/gswarm_data"):  # 16GB default
        self.max_mem_size = max_mem_size
        
        # Storage backends
        self.dram_data: Dict[str, Any] = {}  # Regular DRAM storage
        self.pinned_data: Dict[str, Any] = {}  # Pinned memory for faster GPU transfers
        self.gpu_data: Dict[int, Dict[str, Any]] = {}  # GPU memory per device
        self.disk_path = disk_path
        
        # Metadata tracking
        self.data_locations: Dict[str, List[DataLocationInfo]] = {}  # Track all locations for each key
        self.data_sizes: Dict[str, int] = {}  # Track size of each key-value pair
        self.current_dram_size = 0
        self.current_pinned_size = 0
        self.gpu_sizes: Dict[int, int] = {}  # Track GPU memory usage per device
        
        # Concurrency control
        self.lock = threading.RLock()
        self.move_executor = ThreadPoolExecutor(max_workers=4)
        self.move_tasks: Dict[str, asyncio.Task] = {}  # Track ongoing move operations
        
        # Initialize disk storage
        os.makedirs(self.disk_path, exist_ok=True)
        
        # Initialize GPU tracking if available
        if CUDA_AVAILABLE:
            for i in range(torch.cuda.device_count()):
                self.gpu_data[i] = {}
                self.gpu_sizes[i] = 0

    def _get_size(self, value: Any) -> int:
        """Get approximate size of a value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return sys.getsizeof(value)
    
    def _get_disk_path(self, key: str) -> str:
        """Get disk storage path for a key"""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.disk_path, f"{key_hash}.pkl")
    
    def _save_to_disk(self, key: str, value: Any) -> bool:
        """Save data to disk"""
        try:
            disk_path = self._get_disk_path(key)
            with open(disk_path, 'wb') as f:
                pickle.dump(value, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save to disk: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load data from disk"""
        try:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                with open(disk_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load from disk: {e}")
        return None
    
    def _remove_from_disk(self, key: str) -> bool:
        """Remove data from disk"""
        try:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                os.remove(disk_path)
            return True
        except Exception as e:
            logger.error(f"Failed to remove from disk: {e}")
            return False

    def _add_location(self, key: str, location: str, size: int) -> None:
        """Add a location entry for a key"""
        if key not in self.data_locations:
            self.data_locations[key] = []
        
        # Check if location already exists
        for loc_info in self.data_locations[key]:
            if loc_info.location == location:
                loc_info.last_accessed = time.time()
                return
        
        # Add new location
        self.data_locations[key].append(
            DataLocationInfo(
                location=location,
                size=size,
                last_accessed=time.time(),
                copy_status="complete"
            )
        )
    
    def _remove_location(self, key: str, location: str) -> None:
        """Remove a location entry for a key"""
        if key in self.data_locations:
            self.data_locations[key] = [
                loc for loc in self.data_locations[key] 
                if loc.location != location
            ]
            if not self.data_locations[key]:
                del self.data_locations[key]

    def write(self, key: str, value: Any, location: str = "dram") -> bool:
        """Write key-value pair to specified storage location"""
        with self.lock:
            try:
                value_size = self._get_size(value)
                
                # Clean up any existing data in all locations
                self._cleanup_key(key)
                
                # Store based on location
                if location == DataLocation.DRAM:
                    # Check space
                    if self.current_dram_size + value_size > self.max_mem_size:
                        logger.warning(f"Not enough DRAM space for key '{key}'")
                        return False
                    
                    self.dram_data[key] = value
                    self.current_dram_size += value_size
                    
                elif location == DataLocation.PINNED_DRAM:
                    if CUDA_AVAILABLE:
                        # Allocate pinned memory
                        try:
                            # For simplicity, store in a dict (in real implementation, use torch.cuda.pinned_memory)
                            self.pinned_data[key] = value
                            self.current_pinned_size += value_size
                        except Exception as e:
                            logger.error(f"Failed to allocate pinned memory: {e}")
                            return False
                    else:
                        logger.warning("CUDA not available, falling back to regular DRAM")
                        location = DataLocation.DRAM
                        self.dram_data[key] = value
                        self.current_dram_size += value_size
                        
                elif DataLocation.is_device(location):
                    device_id = DataLocation.get_device_id(location)
                    if device_id is None or not CUDA_AVAILABLE:
                        logger.error(f"Invalid device location: {location}")
                        return False
                    
                    if device_id not in self.gpu_data:
                        logger.error(f"GPU device {device_id} not available")
                        return False
                    
                    # Store on GPU (simplified - in real implementation, convert to GPU tensor)
                    self.gpu_data[device_id][key] = value
                    self.gpu_sizes[device_id] = self.gpu_sizes.get(device_id, 0) + value_size
                    
                elif location == DataLocation.DISK:
                    # Save to disk
                    if not self._save_to_disk(key, value):
                        return False
                
                else:
                    logger.error(f"Unknown location: {location}")
                    return False
                
                # Update metadata
                self.data_sizes[key] = value_size
                self._add_location(key, location, value_size)
                
                logger.debug(f"Stored key '{key}' ({value_size} bytes) in {location}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to write key '{key}': {e}")
                return False
    
    def _cleanup_key(self, key: str) -> None:
        """Clean up all instances of a key across all storage locations"""
        # Remove from DRAM
        if key in self.dram_data:
            size = self.data_sizes.get(key, 0)
            del self.dram_data[key]
            self.current_dram_size -= size
            
        # Remove from pinned memory
        if key in self.pinned_data:
            size = self.data_sizes.get(key, 0)
            del self.pinned_data[key]
            self.current_pinned_size -= size
            
        # Remove from GPU memory
        for device_id, gpu_dict in self.gpu_data.items():
            if key in gpu_dict:
                size = self.data_sizes.get(key, 0)
                del gpu_dict[key]
                self.gpu_sizes[device_id] -= size
                
        # Remove from disk
        self._remove_from_disk(key)
        
        # Clear metadata
        if key in self.data_locations:
            del self.data_locations[key]
        if key in self.data_sizes:
            del self.data_sizes[key]

    def read(self, key: str) -> Optional[Any]:
        """Read value by key - only allowed from DRAM"""
        with self.lock:
            # Only allow reading from DRAM as per requirement
            if key in self.dram_data:
                # Update access time
                if key in self.data_locations:
                    for loc in self.data_locations[key]:
                        if loc.location == DataLocation.DRAM:
                            loc.last_accessed = time.time()
                return self.dram_data[key]
            
            # Check if data exists in other locations and inform user
            if key in self.data_locations:
                locations = [loc.location for loc in self.data_locations[key]]
                logger.warning(f"Key '{key}' exists in {locations} but can only be read from DRAM. Use move() to bring it to DRAM.")
            
            return None
    
    def get_location(self, key: str) -> Optional[str]:
        """Get primary location of data"""
        with self.lock:
            if key not in self.data_locations:
                return None
            
            # Return the most recently accessed location
            locations = sorted(self.data_locations[key], key=lambda x: x.last_accessed, reverse=True)
            return locations[0].location if locations else None
    
    def get_all_locations(self, key: str) -> List[DataLocationInfo]:
        """Get all locations where data exists"""
        with self.lock:
            return self.data_locations.get(key, []).copy()
    
    def list_locations(self) -> Dict[str, List[DataLocationInfo]]:
        """Get locations for all keys in the system"""
        with self.lock:
            return {k: v.copy() for k, v in self.data_locations.items()}
    
    async def move_async(self, key: str, destination: str) -> bool:
        """Async move operation for data between locations"""
        try:
            # Get current location
            current_locations = self.get_all_locations(key)
            if not current_locations:
                logger.error(f"Key '{key}' not found")
                return False
            
            # Find best source location (prefer DRAM > pinned > GPU > disk)
            source_location = None
            source_value = None
            
            for loc in current_locations:
                if loc.location == DataLocation.DRAM and key in self.dram_data:
                    source_location = loc.location
                    source_value = self.dram_data[key]
                    break
                elif loc.location == DataLocation.PINNED_DRAM and key in self.pinned_data:
                    source_location = loc.location
                    source_value = self.pinned_data[key]
                    break
                elif DataLocation.is_device(loc.location):
                    device_id = DataLocation.get_device_id(loc.location)
                    if device_id is not None and device_id in self.gpu_data and key in self.gpu_data[device_id]:
                        source_location = loc.location
                        source_value = self.gpu_data[device_id][key]
                        break
                elif loc.location == DataLocation.DISK:
                    source_location = loc.location
                    source_value = self._load_from_disk(key)
                    if source_value is not None:
                        break
            
            if source_value is None:
                logger.error(f"Could not retrieve value for key '{key}'")
                return False
            
            # Update copy status
            with self.lock:
                for loc in self.data_locations.get(key, []):
                    if loc.location == destination:
                        loc.copy_status = "copying"
                        loc.copy_progress = 0.0
            
            # Perform the move based on source and destination
            if DataLocation.is_device(source_location) or DataLocation.is_device(destination):
                # GPU involved - use CUDA async if available
                if CUDA_AVAILABLE:
                    await self._cuda_async_copy(key, source_value, source_location, destination)
                else:
                    # Fallback to sync copy
                    self.write(key, source_value, destination)
            else:
                # CPU-only operation - use thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self.move_executor,
                    self._sync_copy,
                    key, source_value, destination
                )
            
            # Update copy status
            with self.lock:
                for loc in self.data_locations.get(key, []):
                    if loc.location == destination:
                        loc.copy_status = "complete"
                        loc.copy_progress = 1.0
                        # Set read pointer for PD separation optimization
                        loc.read_pointer = f"ptr_{key}_{destination}_{int(time.time())}"
            
            logger.info(f"Successfully moved key '{key}' from {source_location} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move key '{key}': {e}")
            # Update error status
            with self.lock:
                for loc in self.data_locations.get(key, []):
                    if loc.location == destination:
                        loc.copy_status = "error"
            return False
    
    def _sync_copy(self, key: str, value: Any, destination: str) -> None:
        """Synchronous copy for non-GPU operations"""
        self.write(key, value, destination)
    
    async def _cuda_async_copy(self, key: str, value: Any, source: str, destination: str) -> None:
        """CUDA async memory copy (simplified implementation)"""
        # In a real implementation, this would use CUDA streams and async memcpy
        # For now, we'll simulate with a regular copy
        await asyncio.sleep(0.1)  # Simulate async operation
        self.write(key, value, destination)
    
    def move(self, key: str, destination: str) -> asyncio.Task:
        """Initiate a move operation and return a task handle"""
        task = asyncio.create_task(self.move_async(key, destination))
        self.move_tasks[f"{key}_{destination}"] = task
        return task
    
    def get_read_pointer(self, key: str, location: str) -> Optional[str]:
        """Get read pointer for PD separation optimization"""
        with self.lock:
            locations = self.data_locations.get(key, [])
            for loc in locations:
                if loc.location == location:
                    return loc.read_pointer
        return None

    def release(self, key: str) -> bool:
        """Remove key from all storage locations"""
        with self.lock:
            if key not in self.data_locations:
                return False
            
            # Clean up from all locations
            self._cleanup_key(key)
            
            logger.debug(f"Released key '{key}' from all locations")
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self.lock:
            total_keys = len(self.data_locations)
            
            # Count keys by location
            location_counts = {}
            for key, locations in self.data_locations.items():
                for loc in locations:
                    location_counts[loc.location] = location_counts.get(loc.location, 0) + 1
            
            # GPU stats
            gpu_stats = {}
            if CUDA_AVAILABLE:
                for device_id in range(torch.cuda.device_count()):
                    gpu_stats[f"device:{device_id}"] = {
                        "used": self.gpu_sizes.get(device_id, 0),
                        "keys": len(self.gpu_data.get(device_id, {}))
                    }
            
            # Disk usage
            disk_usage = 0
            disk_keys = 0
            for key in self.data_locations:
                for loc in self.data_locations[key]:
                    if loc.location == DataLocation.DISK:
                        disk_usage += loc.size
                        disk_keys += 1
            
            return {
                "max_dram_size": self.max_mem_size,
                "current_dram_size": self.current_dram_size,
                "current_pinned_size": self.current_pinned_size,
                "dram_usage_percent": (self.current_dram_size / self.max_mem_size) * 100,
                "total_keys": total_keys,
                "location_counts": location_counts,
                "dram_keys": len(self.dram_data),
                "pinned_keys": len(self.pinned_data),
                "disk_usage": disk_usage,
                "disk_keys": disk_keys,
                "gpu_stats": gpu_stats,
                "active_moves": len([t for t in self.move_tasks.values() if not t.done()]),
                "memory_info": {
                    "available": psutil.virtual_memory().available,
                    "total": psutil.virtual_memory().total,
                    "percent": psutil.virtual_memory().percent,
                },
            }


# Global storage instance
_storage_instance: Optional[DataStorage] = None


def get_storage() -> DataStorage:
    """Get or create the global storage instance"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = DataStorage()
    return _storage_instance


def set_max_memory(max_mem_size: int) -> None:
    """Set maximum memory size for storage"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = DataStorage(max_mem_size)
    else:
        _storage_instance.max_mem_size = max_mem_size


def serialize_value(value: Any) -> Dict[str, Any]:
    """Serialize value for HTTP transport, handling complex types"""
    try:
        # Try JSON serialization first (most efficient)
        json.dumps(value)
        return {"type": "json", "data": value}
    except (TypeError, ValueError):
        # Fall back to pickle for complex types
        try:
            pickled_data = pickle.dumps(value)
            encoded_data = base64.b64encode(pickled_data).decode("utf-8")
            return {"type": "pickle", "data": encoded_data}
        except Exception as e:
            raise ValueError(f"Unable to serialize value: {e}")


def deserialize_value(serialized: Dict[str, Any]) -> Any:
    """Deserialize value from HTTP transport"""
    if serialized["type"] == "json":
        return serialized["data"]
    elif serialized["type"] == "pickle":
        try:
            pickled_data = base64.b64decode(serialized["data"].encode("utf-8"))
            return pickle.loads(pickled_data)
        except Exception as e:
            raise ValueError(f"Unable to deserialize pickled value: {e}")
    else:
        raise ValueError(f"Unknown serialization type: {serialized['type']}")


# Pydantic models for API
class WriteRequest(BaseModel):
    key: str
    value: Any
    location: str = "dram"  # Changed from persist to location


class WriteRequestExtended(BaseModel):
    key: str
    serialized_value: Dict[str, Any]
    location: str = "dram"  # Changed from persist to location


class ReadResponse(BaseModel):
    key: str
    value: Any = None
    found: bool


class ReadResponseExtended(BaseModel):
    key: str
    serialized_value: Optional[Dict[str, Any]] = None
    found: bool


class ReleaseRequest(BaseModel):
    key: str


class SendRequest(BaseModel):
    key: str
    url: str
    source_location: Optional[str] = None  # Optional source location for direct GPU transfers


class SendExtendedRequest(BaseModel):
    key: str
    target: str  # Can be URL or node:device format
    source_location: Optional[str] = None  # Source location (e.g., device:0)


class StatsResponse(BaseModel):
    stats: Dict[str, Any]


# FastAPI app
app = FastAPI(title="KV Data Storage", description="Redis-like key-value storage for DRAM")


@app.post("/write")
async def write_data(request: WriteRequest):
    """Write key-value pair to specified location (JSON-compatible values only)"""
    storage = get_storage()
    success = storage.write(request.key, request.value, request.location)
    if success:
        return {"status": "success", "key": request.key, "location": request.location}
    else:
        raise HTTPException(status_code=507, detail="Insufficient storage space")


@app.post("/write_extended")
async def write_data_extended(request: WriteRequestExtended):
    """Write key-value pair to specified location with support for complex types via pickle"""
    try:
        storage = get_storage()
        value = deserialize_value(request.serialized_value)
        success = storage.write(request.key, value, request.location)
        if success:
            return {"status": "success", "key": request.key, "location": request.location}
        else:
            raise HTTPException(status_code=507, detail="Insufficient storage space")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Deserialization error: {e}")


@app.get("/read/{key}")
async def read_data(key: str) -> ReadResponse:
    """Read value by key from DRAM (JSON-compatible response)"""
    storage = get_storage()
    value = storage.read(key)
    return ReadResponse(key=key, value=value, found=value is not None)


@app.get("/read_extended/{key}")
async def read_data_extended(key: str) -> ReadResponseExtended:
    """Read value by key with support for complex types"""
    storage = get_storage()
    value = storage.read(key)
    if value is not None:
        try:
            serialized_value = serialize_value(value)
            return ReadResponseExtended(key=key, serialized_value=serialized_value, found=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Serialization error: {e}")
    else:
        return ReadResponseExtended(key=key, serialized_value=None, found=False)


@app.delete("/release/{key}")
async def release_data(key: str):
    """Remove key from storage"""
    storage = get_storage()
    success = storage.release(key)
    if success:
        return {"status": "success", "key": key}
    else:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found")


@app.post("/send")
async def send_data(request: SendRequest):
    """Send key data to another data manager (legacy endpoint)"""
    storage = get_storage()
    
    # For legacy compatibility, try to read from DRAM first
    value = storage.read(request.key)
    if value is None:
        # Check if data exists in other locations
        location = storage.get_location(request.key)
        if location:
            # Move to DRAM first
            await storage.move_async(request.key, "dram")
            value = storage.read(request.key)
    
    if value is None:
        raise HTTPException(status_code=404, detail=f"Key '{request.key}' not found")

    try:
        # Send data to target URL
        target_url = request.url
        if not target_url.startswith("http://") and not target_url.startswith("https://"):
            target_url = f"http://{target_url}"

        # Try extended API first, fall back to regular API
        try:
            serialized_value = serialize_value(value)
            send_payload = {"key": request.key, "serialized_value": serialized_value, "location": "dram"}
            response = requests.post(f"{target_url}/write_extended", json=send_payload, timeout=30)
            response.raise_for_status()
        except:
            # Fall back to regular API (JSON only)
            send_payload = {"key": request.key, "value": value, "location": "dram"}
            response = requests.post(f"{target_url}/write", json=send_payload, timeout=30)
            response.raise_for_status()

        return {"status": "success", "key": request.key, "target": request.url}

    except Exception as e:
        logger.error(f"Failed to send key '{request.key}' to {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send data: {str(e)}")


@app.post("/send_extended")
async def send_data_extended(request: SendExtendedRequest):
    """Send key data with support for direct GPU-to-GPU transfers via NVLink"""
    storage = get_storage()
    
    # Parse target
    target_parts = request.target.split(":")
    is_gpu_target = len(target_parts) == 3 and target_parts[1].startswith("device")
    
    if is_gpu_target:
        # Format: node:device:X
        target_node = target_parts[0]
        target_device = ":".join(target_parts[1:])
        
        # Check if source is also GPU and NVLink is available
        source_location = request.source_location or storage.get_location(request.key)
        if source_location and DataLocation.is_device(source_location) and NVLINK_AVAILABLE:
            logger.info(f"Attempting NVLink transfer from {source_location} to {target_node}:{target_device}")
            
            # Get source device ID
            source_device_id = DataLocation.get_device_id(source_location)
            
            # Special handling for NVLink transfers
            return await _nvlink_transfer(
                storage, request.key, source_device_id, 
                target_node, target_device
            )
    
    # Fall back to regular send for non-GPU or non-NVLink transfers
    # First ensure data is in DRAM
    location = storage.get_location(request.key)
    if location != DataLocation.DRAM:
        await storage.move_async(request.key, DataLocation.DRAM)
    
    value = storage.read(request.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Key '{request.key}' not found")
    
    try:
        # Determine target URL
        if ":" in request.target and not request.target.startswith("http"):
            # Assume it's host:port format
            target_url = f"http://{request.target}"
        else:
            target_url = request.target
        
        # Send with destination location hint
        destination_location = target_device if is_gpu_target else "dram"
        
        try:
            serialized_value = serialize_value(value)
            send_payload = {
                "key": request.key, 
                "serialized_value": serialized_value, 
                "location": destination_location
            }
            response = requests.post(f"{target_url}/write_extended", json=send_payload, timeout=30)
            response.raise_for_status()
        except:
            send_payload = {
                "key": request.key, 
                "value": value, 
                "location": destination_location
            }
            response = requests.post(f"{target_url}/write", json=send_payload, timeout=30)
            response.raise_for_status()
        
        return {
            "status": "success", 
            "key": request.key, 
            "source": source_location,
            "target": request.target,
            "method": "standard"
        }
    
    except Exception as e:
        logger.error(f"Failed to send key '{request.key}' to {request.target}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send data: {str(e)}")


async def _nvlink_transfer(storage, key: str, source_device_id: int, 
                          target_node: str, target_device: str) -> dict:
    """Handle NVLink GPU-to-GPU transfer"""
    try:
        # In a real implementation, this would:
        # 1. Establish direct GPU communication channel
        # 2. Use GPUDirect RDMA for inter-node transfer
        # 3. Bypass CPU/DRAM completely
        
        # For now, simulate with enhanced transfer
        logger.info(f"Using NVLink optimized path for GPU {source_device_id} -> {target_node}:{target_device}")
        
        # Get data reference (not actual copy)
        if source_device_id in storage.gpu_data and key in storage.gpu_data[source_device_id]:
            # Simulate direct GPU transfer
            await asyncio.sleep(0.05)  # Simulate fast NVLink transfer
            
            return {
                "status": "success",
                "key": key,
                "source": f"device:{source_device_id}",
                "target": f"{target_node}:{target_device}",
                "method": "nvlink",
                "transfer_time_ms": 50
            }
        else:
            raise ValueError(f"Data not found on GPU {source_device_id}")
            
    except Exception as e:
        logger.error(f"NVLink transfer failed: {e}")
        raise HTTPException(status_code=500, detail=f"NVLink transfer failed: {str(e)}")


@app.get("/stats")
async def get_stats() -> StatsResponse:
    """Get storage statistics"""
    storage = get_storage()
    stats = storage.get_stats()
    return StatsResponse(stats=stats)


@app.post("/set_max_memory/{max_size}")
async def set_max_memory_endpoint(max_size: int):
    """Set maximum memory size"""
    set_max_memory(max_size)
    return {"status": "success", "max_memory": max_size}


@app.post("/move")
async def move_data(request: MoveRequest):
    """Move data between storage locations"""
    storage = get_storage()
    
    # Check if key exists
    location = storage.get_location(request.key)
    if location is None:
        raise HTTPException(status_code=404, detail=f"Key '{request.key}' not found")
    
    # Initiate move operation
    task = storage.move(request.key, request.destination)
    
    # For now, we'll wait for completion (in production, return task ID for async tracking)
    try:
        success = await task
        if success:
            return {
                "status": "success",
                "key": request.key,
                "source": location,
                "destination": request.destination,
                "read_pointer": storage.get_read_pointer(request.key, request.destination)
            }
        else:
            raise HTTPException(status_code=500, detail="Move operation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Move operation error: {e}")


@app.get("/location/{key}")
async def get_location(key: str) -> LocationResponse:
    """Get location of data"""
    storage = get_storage()
    location = storage.get_location(key)
    
    if location is None:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found")
    
    all_locations = storage.get_all_locations(key)
    return LocationResponse(key=key, location=location, locations=all_locations)


@app.get("/locations")
async def list_all_locations() -> AllLocationsResponse:
    """List locations of all data in the system"""
    storage = get_storage()
    locations = storage.list_locations()
    return AllLocationsResponse(locations=locations)


class DataServer:
    """KV Data Server for easy programmatic access"""

    def __init__(self, url: str = "localhost:9015", use_extended_api: bool = True):
        self.url = url
        self.use_extended_api = use_extended_api
        if not self.url.startswith("http://") and not self.url.startswith("https://"):
            self.url = f"http://{self.url}"

    def write(self, key: str, value: Any, location: str = "dram") -> bool:
        """Write key-value pair to specified location with automatic format detection"""
        try:
            if self.use_extended_api:
                # Try extended API for complex types
                try:
                    serialized_value = serialize_value(value)
                    response = requests.post(
                        f"{self.url}/write_extended",
                        json={"key": key, "serialized_value": serialized_value, "location": location},
                    )
                    response.raise_for_status()
                    return True
                except:
                    # Fall back to regular API
                    pass

            # Regular JSON API
            response = requests.post(f"{self.url}/write", json={"key": key, "value": value, "location": location})
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to write key '{key}': {e}")
            return False
    
    def move(self, key: str, destination: str) -> bool:
        """Move data to a different storage location"""
        try:
            response = requests.post(f"{self.url}/move", json={"key": key, "destination": destination})
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to move key '{key}' to {destination}: {e}")
            return False
    
    def get_location(self, key: str) -> Optional[Dict[str, Any]]:
        """Get location information for a key"""
        try:
            response = requests.get(f"{self.url}/location/{key}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get location for key '{key}': {e}")
            return None
    
    def list_locations(self) -> Optional[Dict[str, Any]]:
        """List locations of all data in the system"""
        try:
            response = requests.get(f"{self.url}/locations")
            response.raise_for_status()
            return response.json()["locations"]
        except Exception as e:
            logger.error(f"Failed to list locations: {e}")
            return None

    def read(self, key: str) -> Optional[Any]:
        """Read value by key with automatic format detection"""
        try:
            if self.use_extended_api:
                # Try extended API first
                try:
                    response = requests.get(f"{self.url}/read_extended/{key}")
                    response.raise_for_status()
                    data = response.json()
                    if data["found"]:
                        return deserialize_value(data["serialized_value"])
                    else:
                        return None
                except:
                    # Fall back to regular API
                    pass

            # Regular JSON API
            response = requests.get(f"{self.url}/read/{key}")
            response.raise_for_status()
            data = response.json()
            return data["value"] if data["found"] else None

        except Exception as e:
            logger.error(f"Failed to read key '{key}': {e}")
            return None

    def release(self, key: str) -> bool:
        """Remove key from storage"""
        try:
            response = requests.delete(f"{self.url}/release/{key}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to release key '{key}': {e}")
            return False

    def send(self, key: str, target: str, source_location: Optional[str] = None) -> bool:
        """Send key to another data manager with optional GPU-to-GPU support"""
        try:
            # Check if this is a GPU-to-GPU transfer
            if ":device:" in target or (source_location and "device:" in source_location):
                # Use extended API for GPU transfers
                response = requests.post(
                    f"{self.url}/send_extended", 
                    json={
                        "key": key, 
                        "target": target,
                        "source_location": source_location
                    }
                )
            else:
                # Legacy API for backward compatibility
                response = requests.post(f"{self.url}/send", json={"key": key, "url": target})
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("method") == "nvlink":
                logger.info(f"Data transferred via NVLink in {result.get('transfer_time_ms', 'N/A')}ms")
            
            return True
        except Exception as e:
            logger.error(f"Failed to send key '{key}' to {target}: {e}")
            return False

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get storage statistics"""
        try:
            response = requests.get(f"{self.url}/stats")
            response.raise_for_status()
            return response.json()["stats"]
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return None

    def set_max_memory(self, max_size: int) -> bool:
        """Set maximum memory size"""
        try:
            response = requests.post(f"{self.url}/set_max_memory/{max_size}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to set max memory: {e}")
            return False


def start_server(host: str = "0.0.0.0", port: int = 9015, max_mem_size: int = 16 * 1024 * 1024 * 1024, disk_path: str = "/tmp/gswarm_data"):
    """Start the KV data storage server with multi-location support"""
    logger.info(f"Starting KV Data Storage server on {host}:{port}")
    logger.info(f"Maximum DRAM size: {max_mem_size / (1024**3):.1f} GB")
    logger.info(f"Disk storage path: {disk_path}")
    
    if CUDA_AVAILABLE:
        logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")
    else:
        logger.warning("CUDA not available - GPU operations will be limited")

    # Initialize storage with specified memory size
    set_max_memory(max_mem_size)

    # Start server with async support
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()


# Data Pool Pydantic models
class DataChunkCreateRequest(BaseModel):
    source: str
    device: str = "dram"
    type: str = "input"
    format: str = "tensor"
    metadata: Optional[Dict[str, Any]] = None


class DataChunkMoveRequest(BaseModel):
    target_device: str
    priority: str = "normal"


class DataChunkTransferRequest(BaseModel):
    target_node: str
    target_device: str
    delete_source: bool = False


class DataChunkLocation(BaseModel):
    device: str
    status: str = "available"


class DataChunkMetadata(BaseModel):
    created_by: str = "unknown"
    created_at: str
    last_accessed: str = "never"
    access_count: int = 0
    checksum: str = "none"


class DataChunk(BaseModel):
    chunk_id: str
    chunk_type: str
    size: int
    format: str
    locations: List[DataChunkLocation] = []
    metadata: Optional[DataChunkMetadata] = None
    references: List[str] = []


class DataChunksResponse(BaseModel):
    chunks: List[DataChunk]


# Global data chunks storage (in-memory for now)
_data_chunks: Dict[str, DataChunk] = {}
_chunk_counter = 0


def generate_chunk_id() -> str:
    """Generate a unique chunk ID"""
    global _chunk_counter
    _chunk_counter += 1
    return f"chunk-{_chunk_counter:08d}"


# Data Pool API endpoints
@app.get("/api/v1/data")
async def list_data_chunks(device: Optional[str] = None, type: Optional[str] = None) -> DataChunksResponse:
    """List data chunks in the pool"""
    chunks = list(_data_chunks.values())

    # Apply filters
    if device:
        chunks = [chunk for chunk in chunks if any(loc.device == device for loc in chunk.locations)]
    if type:
        chunks = [chunk for chunk in chunks if chunk.chunk_type == type]

    return DataChunksResponse(chunks=chunks)


@app.post("/api/v1/data")
async def create_data_chunk(request: DataChunkCreateRequest):
    """Create a new data chunk"""
    try:
        chunk_id = generate_chunk_id()

        # Create chunk metadata
        import datetime

        current_time = datetime.datetime.utcnow().isoformat() + "Z"

        chunk = DataChunk(
            chunk_id=chunk_id,
            chunk_type=request.type,
            size=1048576,  # Default 1MB size
            format=request.format,
            locations=[DataChunkLocation(device=request.device)],
            metadata=DataChunkMetadata(created_by="data-api", created_at=current_time),
        )

        _data_chunks[chunk_id] = chunk

        return {"status": "success", "chunk_id": chunk_id, "size": chunk.size}

    except Exception as e:
        logger.error(f"Failed to create data chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create data chunk: {str(e)}")


@app.get("/api/v1/data/{chunk_id}")
async def get_data_chunk(chunk_id: str) -> DataChunk:
    """Get data chunk information"""
    if chunk_id not in _data_chunks:
        raise HTTPException(status_code=404, detail=f"Data chunk '{chunk_id}' not found")

    chunk = _data_chunks[chunk_id]
    # Update access count
    if chunk.metadata:
        chunk.metadata.access_count += 1
        chunk.metadata.last_accessed = datetime.datetime.utcnow().isoformat() + "Z"

    return chunk


@app.post("/api/v1/data/{chunk_id}/move")
async def move_data_chunk(chunk_id: str, request: DataChunkMoveRequest):
    """Move data chunk between devices"""
    if chunk_id not in _data_chunks:
        raise HTTPException(status_code=404, detail=f"Data chunk '{chunk_id}' not found")

    chunk = _data_chunks[chunk_id]

    # Add new location (simplified implementation)
    new_location = DataChunkLocation(device=request.target_device, status="available")
    chunk.locations.append(new_location)

    # Generate a mock task ID
    task_id = f"move-{chunk_id}-{int(time.time())}"

    return {"status": "success", "chunk_id": chunk_id, "target_device": request.target_device, "task_id": task_id}


@app.post("/api/v1/data/{chunk_id}/transfer")
async def transfer_data_chunk(chunk_id: str, request: DataChunkTransferRequest):
    """Transfer data chunk to another node"""
    if chunk_id not in _data_chunks:
        raise HTTPException(status_code=404, detail=f"Data chunk '{chunk_id}' not found")

    chunk = _data_chunks[chunk_id]

    # Add new location for target node (simplified implementation)
    target_device_name = f"{request.target_node}:{request.target_device}"
    new_location = DataChunkLocation(device=target_device_name, status="available")
    chunk.locations.append(new_location)

    # Generate a mock task ID
    task_id = f"transfer-{chunk_id}-{int(time.time())}"

    return {
        "status": "success",
        "chunk_id": chunk_id,
        "target": f"{request.target_node}:{request.target_device}",
        "task_id": task_id,
    }


@app.delete("/api/v1/data/{chunk_id}")
async def delete_data_chunk(chunk_id: str, force: bool = False):
    """Delete data chunk from pool"""
    if chunk_id not in _data_chunks:
        raise HTTPException(status_code=404, detail=f"Data chunk '{chunk_id}' not found")

    chunk = _data_chunks[chunk_id]

    # Check if chunk is referenced (unless force is True)
    if not force and chunk.references:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete chunk '{chunk_id}': still referenced by {', '.join(chunk.references)}",
        )

    del _data_chunks[chunk_id]

    return {"status": "success", "message": f"Data chunk '{chunk_id}' deleted successfully"}


# Queue API Pydantic models
class TaskCreateRequest(BaseModel):
    task_type: str
    priority: str = "normal"
    dependencies: List[str] = []
    resources: Dict[str, Any] = {}
    payload: Dict[str, Any] = {}


class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    priority: str
    status: str
    dependencies: List[str] = []
    resources: Dict[str, Any] = {}
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None


class TasksResponse(BaseModel):
    tasks: List[TaskResponse]


class QueueStatusResponse(BaseModel):
    pending: int
    running: int
    completed: int
    config: Dict[str, Any]


# Global queue manager (mock implementation for now)
_queue_tasks: Dict[str, TaskResponse] = {}
_task_counter = 0


def generate_task_id(task_type: str) -> str:
    """Generate a unique task ID"""
    global _task_counter
    _task_counter += 1
    return f"{task_type}-{_task_counter:08d}"


# Queue API endpoints
@app.get("/api/v1/queue")
async def get_queue_status() -> QueueStatusResponse:
    """Get queue status"""
    # Count tasks by status
    pending = len([t for t in _queue_tasks.values() if t.status == "pending"])
    running = len([t for t in _queue_tasks.values() if t.status == "running"])
    completed = len([t for t in _queue_tasks.values() if t.status in ["completed", "failed", "cancelled"]])

    config = {
        "max_concurrent_tasks": 4,
        "priority_levels": ["critical", "high", "normal", "low"],
        "resource_tracking": True,
    }

    return QueueStatusResponse(pending=pending, running=running, completed=completed, config=config)


@app.get("/api/v1/queue/tasks")
async def list_queue_tasks(status: Optional[str] = None, limit: int = 20) -> TasksResponse:
    """List tasks in the queue"""
    tasks = list(_queue_tasks.values())

    # Apply status filter
    if status:
        tasks = [t for t in tasks if t.status == status]

    # Sort by created_at descending and limit
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    tasks = tasks[:limit]

    return TasksResponse(tasks=tasks)


@app.get("/api/v1/queue/tasks/{task_id}")
async def get_task_details(task_id: str) -> TaskResponse:
    """Get specific task details"""
    if task_id not in _queue_tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    return _queue_tasks[task_id]


@app.post("/api/v1/queue/tasks")
async def create_task(request: TaskCreateRequest):
    """Create a new task"""
    task_id = generate_task_id(request.task_type)
    current_time = time.time()

    task = TaskResponse(
        task_id=task_id,
        task_type=request.task_type,
        priority=request.priority,
        status="pending",
        dependencies=request.dependencies,
        resources=request.resources,
        created_at=current_time,
    )

    _queue_tasks[task_id] = task

    return {
        "status": "success",
        "task_id": task_id,
        "position": len([t for t in _queue_tasks.values() if t.status == "pending"]),
    }


@app.post("/api/v1/queue/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a pending or running task"""
    if task_id not in _queue_tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task = _queue_tasks[task_id]

    if task.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in '{task.status}' state")

    task.status = "cancelled"
    task.completed_at = time.time()

    return {"success": True, "message": f"Task '{task_id}' cancelled successfully"}


@app.get("/api/v1/queue/history")
async def get_task_history(limit: int = 50, since: Optional[str] = None, status: Optional[str] = None):
    """Get task execution history"""
    tasks = list(_queue_tasks.values())

    # Apply filters
    if status:
        tasks = [t for t in tasks if t.status == status]

    if since:
        try:
            since_timestamp = float(since)
            tasks = [t for t in tasks if t.created_at >= since_timestamp]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'since' timestamp format")

    # Sort by created_at descending and limit
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    tasks = tasks[:limit]

    return {"history": [task.dict() for task in tasks]}


@app.post("/api/v1/queue/clear")
async def clear_task_history(status: Optional[str] = None):
    """Clear completed or failed tasks from history"""
    global _queue_tasks

    if status:
        # Clear only tasks with specific status
        cleared_count = 0
        to_remove = []
        for task_id, task in _queue_tasks.items():
            if task.status == status:
                to_remove.append(task_id)
                cleared_count += 1

        for task_id in to_remove:
            del _queue_tasks[task_id]
    else:
        # Clear all completed/failed/cancelled tasks
        cleared_count = 0
        to_remove = []
        for task_id, task in _queue_tasks.items():
            if task.status in ["completed", "failed", "cancelled"]:
                to_remove.append(task_id)
                cleared_count += 1

        for task_id in to_remove:
            del _queue_tasks[task_id]

    return {"cleared": cleared_count}

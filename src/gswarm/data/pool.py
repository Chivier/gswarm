"""Redis-like key-value data storage system for DRAM"""

import asyncio
import time
import threading
import pickle
import requests
import base64
import json
from collections import OrderedDict
from typing import Any, Optional, Dict, Union, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from loguru import logger
import psutil
import sys
import datetime


class DataStorage:
    """In-memory key-value storage with LRU eviction for non-persistent data"""
    
    def __init__(self, max_mem_size: int = 16 * 1024 * 1024 * 1024):  # 16GB default
        self.max_mem_size = max_mem_size
        self.persistent_data: Dict[str, Any] = {}  # Persistent data that won't be evicted
        self.volatile_data: OrderedDict[str, Any] = OrderedDict()  # LRU cache for volatile data
        self.data_sizes: Dict[str, int] = {}  # Track size of each key-value pair
        self.current_size = 0
        self.lock = threading.RLock()
        
    def _get_size(self, value: Any) -> int:
        """Get approximate size of a value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return sys.getsizeof(value)
    
    def _evict_lru(self, needed_space: int) -> None:
        """Evict least recently used volatile data to free space"""
        while (self.current_size + needed_space > self.max_mem_size and 
               len(self.volatile_data) > 0):
            # Remove oldest item from volatile data
            key, value = self.volatile_data.popitem(last=False)
            freed_size = self.data_sizes.pop(key, 0)
            self.current_size -= freed_size
            logger.debug(f"Evicted key '{key}' to free {freed_size} bytes")
    
    def write(self, key: str, value: Any, persist: bool = True) -> bool:
        """Write key-value pair to storage"""
        with self.lock:
            try:
                value_size = self._get_size(value)
                
                # Remove existing key if it exists
                if key in self.persistent_data:
                    old_size = self.data_sizes.get(key, 0)
                    self.current_size -= old_size
                    del self.persistent_data[key]
                elif key in self.volatile_data:
                    old_size = self.data_sizes.get(key, 0)
                    self.current_size -= old_size
                    del self.volatile_data[key]
                
                # Check if we need to evict data (only for volatile data)
                if not persist:
                    self._evict_lru(value_size)
                
                # Check if we still have enough space
                if self.current_size + value_size > self.max_mem_size:
                    if persist:
                        raise MemoryError(f"Not enough space for persistent data. Need {value_size} bytes, available {self.max_mem_size - self.current_size}")
                    else:
                        logger.warning(f"Cannot store volatile data: not enough space even after eviction")
                        return False
                
                # Store the data
                if persist:
                    self.persistent_data[key] = value
                else:
                    self.volatile_data[key] = value
                
                self.data_sizes[key] = value_size
                self.current_size += value_size
                
                logger.debug(f"Stored key '{key}' ({value_size} bytes, persist={persist})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to write key '{key}': {e}")
                return False
    
    def read(self, key: str) -> Optional[Any]:
        """Read value by key"""
        with self.lock:
            # Check persistent data first
            if key in self.persistent_data:
                return self.persistent_data[key]
            
            # Check volatile data and move to end (LRU update)
            if key in self.volatile_data:
                value = self.volatile_data[key]
                # Move to end (most recently used)
                self.volatile_data.move_to_end(key)
                return value
            
            return None
    
    def release(self, key: str) -> bool:
        """Remove key from storage"""
        with self.lock:
            freed_size = 0
            found = False
            
            if key in self.persistent_data:
                del self.persistent_data[key]
                found = True
            elif key in self.volatile_data:
                del self.volatile_data[key]
                found = True
            
            if found:
                freed_size = self.data_sizes.pop(key, 0)
                self.current_size -= freed_size
                logger.debug(f"Released key '{key}' ({freed_size} bytes)")
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self.lock:
            return {
                "max_size": self.max_mem_size,
                "current_size": self.current_size,
                "usage_percent": (self.current_size / self.max_mem_size) * 100,
                "persistent_keys": len(self.persistent_data),
                "volatile_keys": len(self.volatile_data),
                "total_keys": len(self.persistent_data) + len(self.volatile_data),
                "memory_info": {
                    "available": psutil.virtual_memory().available,
                    "total": psutil.virtual_memory().total,
                    "percent": psutil.virtual_memory().percent
                }
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
            encoded_data = base64.b64encode(pickled_data).decode('utf-8')
            return {"type": "pickle", "data": encoded_data}
        except Exception as e:
            raise ValueError(f"Unable to serialize value: {e}")


def deserialize_value(serialized: Dict[str, Any]) -> Any:
    """Deserialize value from HTTP transport"""
    if serialized["type"] == "json":
        return serialized["data"]
    elif serialized["type"] == "pickle":
        try:
            pickled_data = base64.b64decode(serialized["data"].encode('utf-8'))
            return pickle.loads(pickled_data)
        except Exception as e:
            raise ValueError(f"Unable to deserialize pickled value: {e}")
    else:
        raise ValueError(f"Unknown serialization type: {serialized['type']}")


# Pydantic models for API
class WriteRequest(BaseModel):
    key: str
    value: Any
    persist: bool = True


class WriteRequestExtended(BaseModel):
    key: str
    serialized_value: Dict[str, Any]
    persist: bool = True


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


class StatsResponse(BaseModel):
    stats: Dict[str, Any]


# FastAPI app
app = FastAPI(title="KV Data Storage", description="Redis-like key-value storage for DRAM")


@app.post("/write")
async def write_data(request: WriteRequest):
    """Write key-value pair (JSON-compatible values only)"""
    storage = get_storage()
    success = storage.write(request.key, request.value, request.persist)
    if success:
        return {"status": "success", "key": request.key}
    else:
        raise HTTPException(status_code=507, detail="Insufficient storage space")


@app.post("/write_extended")
async def write_data_extended(request: WriteRequestExtended):
    """Write key-value pair with support for complex types via pickle"""
    try:
        storage = get_storage()
        value = deserialize_value(request.serialized_value)
        success = storage.write(request.key, value, request.persist)
        if success:
            return {"status": "success", "key": request.key}
        else:
            raise HTTPException(status_code=507, detail="Insufficient storage space")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Deserialization error: {e}")


@app.get("/read/{key}")
async def read_data(key: str) -> ReadResponse:
    """Read value by key (JSON-compatible response)"""
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
    """Send key data to another data manager"""
    storage = get_storage()
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
            send_payload = {
                "key": request.key,
                "serialized_value": serialized_value,
                "persist": True
            }
            response = requests.post(f"{target_url}/write_extended", json=send_payload, timeout=30)
            response.raise_for_status()
        except:
            # Fall back to regular API (JSON only)
            send_payload = {
                "key": request.key,
                "value": value,
                "persist": True
            }
            response = requests.post(f"{target_url}/write", json=send_payload, timeout=30)
            response.raise_for_status()
        
        return {"status": "success", "key": request.key, "target": request.url}
        
    except Exception as e:
        logger.error(f"Failed to send key '{request.key}' to {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send data: {str(e)}")


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


class DataServer:
    """KV Data Server for easy programmatic access"""
    
    def __init__(self, url: str = "localhost:9015", use_extended_api: bool = True):
        self.url = url
        self.use_extended_api = use_extended_api
        if not self.url.startswith("http://") and not self.url.startswith("https://"):
            self.url = f"http://{self.url}"
    
    def write(self, key: str, value: Any, persist: bool = True) -> bool:
        """Write key-value pair with automatic format detection"""
        try:
            if self.use_extended_api:
                # Try extended API for complex types
                try:
                    serialized_value = serialize_value(value)
                    response = requests.post(
                        f"{self.url}/write_extended",
                        json={"key": key, "serialized_value": serialized_value, "persist": persist}
                    )
                    response.raise_for_status()
                    return True
                except:
                    # Fall back to regular API
                    pass
            
            # Regular JSON API
            response = requests.post(
                f"{self.url}/write",
                json={"key": key, "value": value, "persist": persist}
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to write key '{key}': {e}")
            return False
    
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
    
    def send(self, key: str, target_url: str) -> bool:
        """Send key to another data manager"""
        try:
            response = requests.post(
                f"{self.url}/send",
                json={"key": key, "url": target_url}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send key '{key}' to {target_url}: {e}")
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


def start_server(host: str = "0.0.0.0", port: int = 9015, max_mem_size: int = 16 * 1024 * 1024 * 1024):
    """Start the KV data storage server"""
    logger.info(f"Starting KV Data Storage server on {host}:{port}")
    logger.info(f"Maximum memory size: {max_mem_size / (1024**3):.1f} GB")
    
    # Initialize storage with specified memory size
    set_max_memory(max_mem_size)
    
    # Start server
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
            metadata=DataChunkMetadata(
                created_by="data-api",
                created_at=current_time
            )
        )
        
        _data_chunks[chunk_id] = chunk
        
        return {
            "status": "success",
            "chunk_id": chunk_id,
            "size": chunk.size
        }
        
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
    
    return {
        "status": "success",
        "chunk_id": chunk_id,
        "target_device": request.target_device,
        "task_id": task_id
    }


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
        "task_id": task_id
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
            detail=f"Cannot delete chunk '{chunk_id}': still referenced by {', '.join(chunk.references)}"
        )
    
    del _data_chunks[chunk_id]
    
    return {
        "status": "success",
        "message": f"Data chunk '{chunk_id}' deleted successfully"
    }


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
        "resource_tracking": True
    }
    
    return QueueStatusResponse(
        pending=pending,
        running=running,
        completed=completed,
        config=config
    )


@app.get("/api/v1/queue/tasks")
async def list_queue_tasks(
    status: Optional[str] = None,
    limit: int = 20
) -> TasksResponse:
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
        created_at=current_time
    )
    
    _queue_tasks[task_id] = task
    
    return {
        "status": "success",
        "task_id": task_id,
        "position": len([t for t in _queue_tasks.values() if t.status == "pending"])
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
    
    return {
        "success": True,
        "message": f"Task '{task_id}' cancelled successfully"
    }


@app.get("/api/v1/queue/history")
async def get_task_history(
    limit: int = 50,
    since: Optional[str] = None,
    status: Optional[str] = None
):
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

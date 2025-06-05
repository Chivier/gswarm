"""
Head node implementation for gswarm_model system.
Central coordinator for model storage and management across GPU cluster.
"""

import asyncio
import hashlib
import json
import socket
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import grpc
from concurrent import futures
from loguru import logger
import aiofiles
import os
import sys

from .models import (
    HostModelInfo, NodeInfo, Job, JobAction, JobStatus, ActionType,
    ModelType, ModelStatus, StorageInfo, parse_device_name, format_device_name
)

# Import generated protobuf classes
try:
    from . import model_pb2
    from . import model_pb2_grpc
except ImportError:
    logger.error("gRPC protobuf files not found. Please run 'python generate_grpc.py' first.")
    raise


class HeadNodeState:
    """Global state for the head node"""
    
    def __init__(self):
        # Model registry: model_name -> HostModelInfo
        self.model_registry: Dict[str, HostModelInfo] = {}
        
        # Node registry: node_id -> NodeInfo
        self.node_registry: Dict[str, NodeInfo] = {}
        
        # Job management
        self.jobs: Dict[str, Job] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Locks for thread safety
        self.registry_lock = asyncio.Lock()
        self.job_lock = asyncio.Lock()
        
        # Configuration
        self.data_directory = ".gswarm_model_data"
        self.enable_persistence = True
        
        # Ensure data directory exists
        os.makedirs(self.data_directory, exist_ok=True)


state = HeadNodeState()


class ModelManagerServicer(model_pb2_grpc.ModelManagerServicer):
    """gRPC service implementation for model management"""
    
    async def Connect(self, request: model_pb2.NodeRegistration, context):
        """Handle client node connection and registration"""
        client_address = context.peer()
        node_id = request.node_id or f"{request.hostname}_{client_address}"
        
        logger.info(f"Node {node_id} (hostname: {request.hostname}) connecting via gRPC.")
        
        async with state.registry_lock:
            # Convert storage devices from protobuf
            storage_devices = {}
            for device in request.storage_devices:
                storage_devices[device.device_name] = StorageInfo(
                    total=device.total_capacity,
                    used=device.used_capacity,
                    available=device.available_capacity
                )
            
            # Convert GPU info from strings to dictionaries for NodeInfo model
            gpu_info_dicts = []
            for i, gpu_name in enumerate(request.gpu_info):
                gpu_info_dicts.append({
                    "physical_idx": i,
                    "name": gpu_name,
                    "id": f"gpu{i}"
                })
            
            # Create node info
            node_info = NodeInfo(
                node_id=node_id,
                hostname=request.hostname,
                ip_address=request.ip_address,
                storage_devices=storage_devices,
                gpu_info=gpu_info_dicts,  # Now using properly formatted dictionaries
                last_seen=datetime.now(),
                is_online=True
            )
            
            state.node_registry[node_id] = node_info
            logger.info(f"Registered node {node_id} with {len(storage_devices)} storage devices and {len(gpu_info_dicts)} GPUs")
        
        if state.enable_persistence:
            await save_registry()
        
        return model_pb2.ConnectResponse(
            success=True,
            message=f"Node {node_id} connected successfully. Registered {len(storage_devices)} storage devices and {len(gpu_info_dicts)} GPUs."
        )
    
    async def Heartbeat(self, request: model_pb2.Empty, context):
        """Handle heartbeat from client nodes"""
        # Extract node_id from context (this is simplified)
        client_address = context.peer()
        
        async with state.registry_lock:
            # Update last_seen for all nodes (simplified - in production, use proper node identification)
            for node_id, node_info in state.node_registry.items():
                if client_address in node_id:
                    node_info.last_seen = datetime.now()
                    node_info.is_online = True
                    break
        
        return model_pb2.Empty()
    
    async def RegisterModel(self, request: model_pb2.ModelInfo, context):
        """Register a new model in the system"""
        try:
            async with state.registry_lock:
                if request.model_name in state.model_registry:
                    return model_pb2.ModelOperationResponse(
                        success=False,
                        message=f"Model {request.model_name} is already registered",
                        operation_id=""
                    )
                
                # Create model info
                model_info = HostModelInfo(
                    model_name=request.model_name,
                    model_type=ModelType(request.model_type),
                    model_size=request.model_size if request.model_size > 0 else None,
                    model_hash=request.model_hash if request.model_hash else None,
                    stored_locations=list(request.stored_locations),
                    available_services=dict(request.available_services)
                )
                
                state.model_registry[request.model_name] = model_info
                logger.info(f"Registered model {request.model_name} of type {request.model_type}")
            
            if state.enable_persistence:
                await save_registry()
            
            return model_pb2.ModelOperationResponse(
                success=True,
                message=f"Model {request.model_name} registered successfully",
                operation_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            logger.error(f"Error registering model {request.model_name}: {e}")
            return model_pb2.ModelOperationResponse(
                success=False,
                message=f"Failed to register model: {str(e)}",
                operation_id=""
            )
    
    async def ListModels(self, request: model_pb2.Empty, context):
        """List all registered models"""
        try:
            models = []
            async with state.registry_lock:
                for model_name, model_info in state.model_registry.items():
                    model_proto = model_pb2.ModelInfo(
                        model_name=model_info.model_name,
                        model_type=model_info.model_type.value,
                        model_size=model_info.model_size or 0,
                        model_hash=model_info.model_hash or "",
                        stored_locations=model_info.stored_locations,
                        available_services=model_info.available_services
                    )
                    models.append(model_proto)
            
            return model_pb2.ModelList(
                models=models,
                total_count=len(models)
            )
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return model_pb2.ModelList(models=[], total_count=0)
    
    async def GetModelInfo(self, request: model_pb2.ModelInfo, context):
        """Get detailed information about a specific model"""
        try:
            async with state.registry_lock:
                model_info = state.model_registry.get(request.model_name)
                if not model_info:
                    return model_pb2.ModelInfo()
                
                return model_pb2.ModelInfo(
                    model_name=model_info.model_name,
                    model_type=model_info.model_type.value,
                    model_size=model_info.model_size or 0,
                    model_hash=model_info.model_hash or "",
                    stored_locations=model_info.stored_locations,
                    available_services=model_info.available_services
                )
                
        except Exception as e:
            logger.error(f"Error getting model info for {request.model_name}: {e}")
            return model_pb2.ModelInfo()
    
    async def CreateJob(self, request: model_pb2.JobDefinition, context):
        """Create and execute a job workflow"""
        try:
            job_id = request.job_id or str(uuid.uuid4())
            
            # Convert protobuf actions to internal format
            actions = []
            for action_proto in request.actions:
                action = JobAction(
                    action_id=action_proto.action_id,
                    action_type=ActionType(action_proto.action_type),
                    model_name=action_proto.model_name,
                    devices=list(action_proto.devices),
                    dependencies=list(action_proto.dependencies),
                    source_url=action_proto.source_url if action_proto.source_url else None,
                    port=action_proto.port if action_proto.port > 0 else None,
                    target_url=action_proto.target_url if action_proto.target_url else None,
                    keep_source=action_proto.keep_source,
                    config=dict(action_proto.config) if action_proto.config else None
                )
                actions.append(action)
            
            job = Job(
                job_id=job_id,
                name=request.name,
                description=request.description if request.description else None,
                actions=actions
            )
            
            async with state.job_lock:
                state.jobs[job_id] = job
                # Start job execution
                task = asyncio.create_task(execute_job(job))
                state.active_jobs[job_id] = task
            
            logger.info(f"Created and started job {job_id}: {request.name}")
            
            return model_pb2.ModelOperationResponse(
                success=True,
                message=f"Job {job_id} created and started",
                operation_id=job_id
            )
            
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            return model_pb2.ModelOperationResponse(
                success=False,
                message=f"Failed to create job: {str(e)}",
                operation_id=""
            )
    
    async def GetJobStatus(self, request: model_pb2.JobDefinition, context):
        """Get status of a job"""
        try:
            job_id = request.job_id
            
            async with state.job_lock:
                job = state.jobs.get(job_id)
                if not job:
                    return model_pb2.JobStatusResponse(
                        job_id=job_id,
                        status="not_found",
                        error_message="Job not found",
                        completed_actions=0,
                        total_actions=0
                    )
                
                completed_actions = sum(1 for action in job.actions if action.status == JobStatus.COMPLETED)
                
                return model_pb2.JobStatusResponse(
                    job_id=job_id,
                    status=job.status.value,
                    error_message=job.error_message or "",
                    completed_actions=completed_actions,
                    total_actions=len(job.actions)
                )
                
        except Exception as e:
            logger.error(f"Error getting job status for {request.job_id}: {e}")
            return model_pb2.JobStatusResponse(
                job_id=request.job_id,
                status="error",
                error_message=str(e),
                completed_actions=0,
                total_actions=0
            )
    
    async def GetSystemStatus(self, request: model_pb2.Empty, context):
        """Get system-wide status"""
        try:
            async with state.registry_lock:
                total_nodes = len(state.node_registry)
                online_nodes = sum(1 for node in state.node_registry.values() if node.is_online)
                total_models = len(state.model_registry)
                
                # Count active services
                active_services = 0
                for model_info in state.model_registry.values():
                    active_services += len(model_info.available_services)
                
                # Calculate storage utilization by node
                storage_utilization = {}
                for node_id, node_info in state.node_registry.items():
                    for device_name, storage_info in node_info.storage_devices.items():
                        key = f"{node_id}:{device_name}"
                        storage_utilization[key] = storage_info.utilization_percent
            
            return model_pb2.SystemStatus(
                total_nodes=total_nodes,
                online_nodes=online_nodes,
                total_models=total_models,
                active_services=active_services,
                storage_utilization=storage_utilization
            )
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return model_pb2.SystemStatus(
                total_nodes=0,
                online_nodes=0,
                total_models=0,
                active_services=0,
                storage_utilization={}
            )


async def execute_job(job: Job) -> None:
    """Execute a job workflow"""
    try:
        logger.info(f"Starting job execution: {job.job_id}")
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        # Build dependency graph
        completed_actions = set()
        
        while True:
            # Find actions ready to execute (all dependencies completed)
            ready_actions = []
            for action in job.actions:
                if action.status == JobStatus.PENDING:
                    if all(dep in completed_actions for dep in action.dependencies):
                        ready_actions.append(action)
            
            if not ready_actions:
                # Check if all actions are completed
                if all(action.status in [JobStatus.COMPLETED, JobStatus.FAILED] for action in job.actions):
                    break
                
                # Wait a bit and check again
                await asyncio.sleep(1)
                continue
            
            # Execute ready actions
            for action in ready_actions:
                try:
                    logger.info(f"Executing action {action.action_id}: {action.action_type}")
                    action.status = JobStatus.RUNNING
                    action.started_at = datetime.now()
                    
                    # Execute the action based on type
                    success = await execute_action(action)
                    
                    if success:
                        action.status = JobStatus.COMPLETED
                        action.completed_at = datetime.now()
                        completed_actions.add(action.action_id)
                        logger.info(f"Action {action.action_id} completed successfully")
                    else:
                        action.status = JobStatus.FAILED
                        action.completed_at = datetime.now()
                        logger.error(f"Action {action.action_id} failed")
                        
                except Exception as e:
                    action.status = JobStatus.FAILED
                    action.error_message = str(e)
                    action.completed_at = datetime.now()
                    logger.error(f"Action {action.action_id} failed with error: {e}")
        
        # Determine final job status
        failed_actions = [action for action in job.actions if action.status == JobStatus.FAILED]
        if failed_actions:
            job.status = JobStatus.FAILED
            job.error_message = f"Job failed: {len(failed_actions)} actions failed"
        else:
            job.status = JobStatus.COMPLETED
        
        job.completed_at = datetime.now()
        logger.info(f"Job {job.job_id} completed with status: {job.status}")
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.now()
        logger.error(f"Job {job.job_id} failed with error: {e}")
    
    finally:
        # Clean up
        async with state.job_lock:
            if job.job_id in state.active_jobs:
                del state.active_jobs[job.job_id]


async def execute_action(action: JobAction) -> bool:
    """Execute a single action"""
    try:
        if action.action_type == ActionType.DOWNLOAD:
            return await execute_download_action(action)
        elif action.action_type == ActionType.MOVE:
            return await execute_move_action(action)
        elif action.action_type == ActionType.SERVE:
            return await execute_serve_action(action)
        elif action.action_type == ActionType.HEALTH_CHECK:
            return await execute_health_check_action(action)
        else:
            logger.warning(f"Unsupported action type: {action.action_type}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing action {action.action_id}: {e}")
        return False


async def execute_download_action(action: JobAction) -> bool:
    """Execute a download action"""
    # This is a simplified implementation
    # In a real system, this would coordinate with client nodes
    logger.info(f"Download action: {action.model_name} from {action.source_url}")
    
    # Simulate download time
    await asyncio.sleep(2)
    
    # Update model registry
    async with state.registry_lock:
        if action.model_name not in state.model_registry:
            # Create basic model info
            model_info = HostModelInfo(
                model_name=action.model_name,
                model_type=ModelType.LLM,  # Default type
                stored_locations=action.devices
            )
            state.model_registry[action.model_name] = model_info
        else:
            # Update existing model
            model_info = state.model_registry[action.model_name]
            for device in action.devices:
                if device not in model_info.stored_locations:
                    model_info.stored_locations.append(device)
    
    return True


async def execute_move_action(action: JobAction) -> bool:
    """Execute a move action"""
    logger.info(f"Move action: {action.model_name} from {action.devices[0]} to {action.devices[1]}")
    
    # Simulate move time
    await asyncio.sleep(1)
    
    # Update model registry
    async with state.registry_lock:
        if action.model_name in state.model_registry:
            model_info = state.model_registry[action.model_name]
            if len(action.devices) >= 2:
                source_device, target_device = action.devices[0], action.devices[1]
                
                # Add target device
                if target_device not in model_info.stored_locations:
                    model_info.stored_locations.append(target_device)
                
                # Remove source device if keep_source is False
                if not action.keep_source and source_device in model_info.stored_locations:
                    model_info.stored_locations.remove(source_device)
    
    return True


async def execute_serve_action(action: JobAction) -> bool:
    """Execute a serve action"""
    logger.info(f"Serve action: {action.model_name} on {action.devices[0]} port {action.port}")
    
    # Simulate service startup time
    await asyncio.sleep(1)
    
    # Update model registry with service endpoint
    async with state.registry_lock:
        if action.model_name in state.model_registry and action.devices:
            model_info = state.model_registry[action.model_name]
            device = action.devices[0]
            
            # Extract node_id from device name
            try:
                node_id, _, _ = parse_device_name(device)
                if node_id in state.node_registry:
                    node_info = state.node_registry[node_id]
                    service_url = f"http://{node_info.hostname}:{action.port}"
                    model_info.available_services[device] = service_url
            except ValueError:
                logger.warning(f"Failed to parse device name: {device}")
    
    return True


async def execute_health_check_action(action: JobAction) -> bool:
    """Execute a health check action"""
    logger.info(f"Health check action: {action.target_url}")
    
    # Simulate health check
    await asyncio.sleep(0.5)
    
    # In a real implementation, this would make an HTTP request
    return True


async def save_registry():
    """Save registry state to disk"""
    try:
        registry_data = {
            "models": {},
            "nodes": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Serialize model registry
        for model_name, model_info in state.model_registry.items():
            registry_data["models"][model_name] = model_info.model_dump()
        
        # Serialize node registry
        for node_id, node_info in state.node_registry.items():
            registry_data["nodes"][node_id] = node_info.model_dump()
        
        registry_file = os.path.join(state.data_directory, "registry.json")
        async with aiofiles.open(registry_file, "w") as f:
            await f.write(json.dumps(registry_data, indent=2, default=str))
        
        logger.debug("Registry saved to disk")
        
    except Exception as e:
        logger.error(f"Failed to save registry: {e}")


async def load_registry():
    """Load registry state from disk"""
    try:
        registry_file = os.path.join(state.data_directory, "registry.json")
        if not os.path.exists(registry_file):
            logger.info("No existing registry file found")
            return
        
        async with aiofiles.open(registry_file, "r") as f:
            data = await f.read()
            registry_data = json.loads(data)
        
        # Load model registry
        for model_name, model_data in registry_data.get("models", {}).items():
            model_info = HostModelInfo(**model_data)
            state.model_registry[model_name] = model_info
        
        # Load node registry  
        for node_id, node_data in registry_data.get("nodes", {}).items():
            node_info = NodeInfo(**node_data)
            state.node_registry[node_id] = node_info
        
        logger.info(f"Loaded registry: {len(state.model_registry)} models, {len(state.node_registry)} nodes")
        
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")


def check_port_availability(host: str, port: int) -> bool:
    """Check if a port is available for binding"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            sock.close()
            return True
        except OSError:
            return False


async def run_grpc_server(host: str, port: int):
    """Run the gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = ModelManagerServicer()
    model_pb2_grpc.add_ModelManagerServicer_to_server(servicer, server)
    
    listen_addr = f"{host}:{port}"
    
    try:
        server.add_insecure_port(listen_addr)
    except Exception as e:
        logger.error(f"Failed to bind gRPC server to {listen_addr}: {e}")
        raise
    
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(5)


async def run_both_servers(grpc_host: str, grpc_port: int, http_host: str, http_port: int):
    """Run both gRPC and HTTP servers concurrently"""
    from .http_api import run_http_server
    
    # Load persisted state
    if state.enable_persistence:
        await load_registry()
    
    # Create tasks for both servers
    grpc_task = asyncio.create_task(run_grpc_server(grpc_host, grpc_port))
    http_task = asyncio.create_task(run_http_server(http_host, http_port))
    
    try:
        # Wait for both servers to run
        await asyncio.gather(grpc_task, http_task)
    except KeyboardInterrupt:
        logger.info("Shutting down both servers...")
        grpc_task.cancel()
        http_task.cancel()
        try:
            await asyncio.gather(grpc_task, http_task, return_exceptions=True)
        except Exception:
            pass


def run_head_node(host: str, port: int, http_port: Optional[int] = None):
    """Run the head node with gRPC server and optionally HTTP server"""
    # Check port availability before starting
    if not check_port_availability(host, port):
        logger.error(f"Port {port} is already in use on {host}")
        logger.info("Please choose a different port or stop the process using this port.")
        logger.info(f"You can find the process using: lsof -i :{port} or netstat -tulpn | grep :{port}")
        sys.exit(1)
    
    if http_port and not check_port_availability(host, http_port):
        logger.error(f"HTTP port {http_port} is already in use on {host}")
        logger.info("Please choose a different HTTP port or stop the process using this port.")
        logger.info(f"You can find the process using: lsof -i :{http_port} or netstat -tulpn | grep :{http_port}")
        sys.exit(1)
    
    try:
        if http_port:
            # Run both gRPC and HTTP servers
            asyncio.run(run_both_servers(host, port, host, http_port))
        else:
            # Run only gRPC server
            async def run_grpc_only():
                if state.enable_persistence:
                    await load_registry()
                await run_grpc_server(host, port)
            
            asyncio.run(run_grpc_only())
    except KeyboardInterrupt:
        logger.info("Head node shutdown requested")
    finally:
        logger.info("Head node exiting") 
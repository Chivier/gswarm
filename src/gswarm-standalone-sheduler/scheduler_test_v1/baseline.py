#!/usr/bin/env python3
"""
Baseline Scheduler for GSwarm Workflows
Uses Ray-like scheduling: runs models one by one, maintaining a queue of ready nodes.
"""

import argparse
import json
import yaml
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
SERVER_URL = "http://localhost:8000"
PCIE_BANDWIDTH_GB_S = 16.0  # PCIe 4.0 x16 bandwidth in GB/s

@dataclass
class ModelInfo:
    """Model configuration information"""
    name: str
    memory_gb: float
    gpus_required: int
    load_time_seconds: float
    tokens_per_second: Optional[float] = None
    token_mean: Optional[float] = None
    token_std: Optional[float] = None
    inference_time_mean: Optional[float] = None
    inference_time_std: Optional[float] = None

@dataclass
class WorkflowNode:
    """Workflow node definition"""
    id: str
    model: str
    inputs: List[str]
    outputs: List[str]
    config_options: Optional[List[str]] = None

@dataclass
class WorkflowEdge:
    """Workflow edge definition"""
    from_node: str
    to_node: str

@dataclass
class Workflow:
    """Workflow definition"""
    id: str
    name: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    
    def get_dependencies(self) -> Dict[str, Set[str]]:
        """Get dependency map: node -> set of nodes it depends on"""
        deps = defaultdict(set)
        for edge in self.edges:
            deps[edge.to_node].add(edge.from_node)
        # Add nodes with no dependencies
        for node in self.nodes:
            if node.id not in deps:
                deps[node.id] = set()
        return dict(deps)
    
    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

@dataclass
class Request:
    """Workflow request"""
    request_id: str
    timestamp: datetime
    workflow_id: str
    input_data: Dict[str, Any]
    node_configs: Dict[str, Dict[str, Any]]
    node_execution_times: Dict[str, float]

@dataclass
class NodeExecution:
    """Execution state for a node"""
    request_id: str
    workflow_id: str
    node_id: str
    model_name: str
    status: str = "pending"  # pending, ready, running, completed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    gpu_id: Optional[int] = None
    instance_id: Optional[str] = None
    estimated_time: Optional[float] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

@dataclass
class GPUState:
    """GPU state tracking"""
    gpu_id: int
    current_model: Optional[str] = None
    current_instance: Optional[str] = None
    busy: bool = False
    last_switch_time: float = 0.0

class BaselineScheduler:
    """Baseline scheduler using Ray-like scheduling strategy"""
    
    def __init__(self, gpus: List[int], simulate: bool = False, mode: str = "offline"):
        self.gpus = gpus
        self.simulate = simulate
        self.mode = mode
        self.server_url = SERVER_URL
        
        # GPU states
        self.gpu_states = {gpu_id: GPUState(gpu_id) for gpu_id in gpus}
        
        # Model and workflow definitions
        self.models: Dict[str, ModelInfo] = {}
        self.workflows: Dict[str, Workflow] = {}
        
        # Execution tracking
        self.node_queue: deque[NodeExecution] = deque()
        self.executions: Dict[str, NodeExecution] = {}  # node_key -> execution
        self.completed_executions: List[NodeExecution] = []
        
        # Model instances for simulation mode
        self.model_instances: Dict[str, str] = {}  # model_name -> instance_id
        
        # Metrics
        self.request_start_times: Dict[str, float] = {}
        self.request_end_times: Dict[str, float] = {}
        
    def load_config(self, config_path: Path):
        """Load system configuration"""
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Load models
        for model_id, model_data in config['models'].items():
            self.models[model_id] = ModelInfo(
                name=model_data['name'],
                memory_gb=model_data['memory_gb'],
                gpus_required=model_data['gpus_required'],
                load_time_seconds=model_data['load_time_seconds'],
                tokens_per_second=model_data.get('tokens_per_second'),
                token_mean=model_data.get('token_mean'),
                token_std=model_data.get('token_std'),
                inference_time_mean=model_data.get('inference_time_mean'),
                inference_time_std=model_data.get('inference_time_std')
            )
        
        # Load workflows
        for workflow_id, workflow_data in config['workflows'].items():
            nodes = []
            for node_data in workflow_data['nodes']:
                nodes.append(WorkflowNode(
                    id=node_data['id'],
                    model=node_data['model'],
                    inputs=node_data['inputs'],
                    outputs=node_data['outputs'],
                    config_options=node_data.get('config_options')
                ))
            
            edges = []
            for edge_data in workflow_data.get('edges', []):
                edges.append(WorkflowEdge(
                    from_node=edge_data['from'],
                    to_node=edge_data['to']
                ))
            
            self.workflows[workflow_id] = Workflow(
                id=workflow_id,
                name=workflow_data['name'],
                nodes=nodes,
                edges=edges
            )
    
    def load_requests(self, requests_path: Path) -> List[Request]:
        """Load workflow requests"""
        logger.info(f"Loading requests from {requests_path}")
        
        with open(requests_path, 'r') as f:
            if requests_path.suffix == '.yaml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        requests = []
        for req_data in data['requests']:
            requests.append(Request(
                request_id=req_data['request_id'],
                timestamp=datetime.fromisoformat(req_data['timestamp']),
                workflow_id=req_data['workflow_id'],
                input_data=req_data['input_data'],
                node_configs=req_data.get('node_configs', {}),
                node_execution_times=req_data['node_execution_times']
            ))
        
        return requests
    
    def _get_model_switch_time(self, from_model: Optional[str], to_model: str) -> float:
        """Calculate model switch time based on model size and PCIe bandwidth"""
        if from_model == to_model:
            return 0.0
        
        # If no previous model, just load time
        if not from_model:
            return self.models[to_model].load_time_seconds
        
        # Calculate switch time: unload old + load new
        from_size_gb = self.models[from_model].memory_gb
        to_size_gb = self.models[to_model].memory_gb
        
        # Assume we need to transfer both models over PCIe
        transfer_time = (from_size_gb + to_size_gb) / PCIE_BANDWIDTH_GB_S
        
        # Add some overhead
        overhead = 2.0  # seconds
        
        return transfer_time + overhead
    
    def _find_available_gpu(self, model_name: str) -> Optional[int]:
        """Find an available GPU for the model"""
        model_info = self.models[model_name]
        required_gpus = model_info.gpus_required
        
        # For single GPU models
        if required_gpus == 1:
            # First, try to find GPU with same model already loaded
            for gpu_id, gpu_state in self.gpu_states.items():
                if not gpu_state.busy and gpu_state.current_model == model_name:
                    return gpu_id
            
            # Then, try to find any free GPU
            for gpu_id, gpu_state in self.gpu_states.items():
                if not gpu_state.busy:
                    return gpu_id
        
        # For multi-GPU models (simplified: use consecutive GPUs)
        else:
            # Check if we have enough consecutive free GPUs
            gpu_ids = sorted(self.gpu_states.keys())
            for i in range(len(gpu_ids) - required_gpus + 1):
                consecutive_gpus = gpu_ids[i:i + required_gpus]
                if all(not self.gpu_states[gpu_id].busy for gpu_id in consecutive_gpus):
                    return consecutive_gpus[0]  # Return first GPU of the group
        
        return None
    
    def _download_and_load_models(self):
        """Download and load all models if in simulate mode"""
        if not self.simulate:
            return
        
        logger.info("Downloading and loading models for simulation mode...")
        
        for model_id, model_info in self.models.items():
            # Map model IDs to actual model names
            model_name_map = {
                "llm7b": "gpt2",  # Use GPT-2 as a stand-in for 7B model
                "llm30b": "gpt2-medium",  # Use GPT-2 medium for 30B
                "stable_diffusion": "CompVis/stable-diffusion-v1-4"
            }
            
            actual_model_name = model_name_map.get(model_id, "gpt2")
            
            try:
                # Download model
                response = requests.post(
                    f"{self.server_url}/standalone/download",
                    json={"model_name": actual_model_name}
                )
                if response.status_code != 200:
                    logger.error(f"Failed to download {actual_model_name}: {response.text}")
                    continue
                
                # Load to DRAM
                response = requests.post(
                    f"{self.server_url}/standalone/load",
                    json={"model_name": actual_model_name, "target": "dram"}
                )
                if response.status_code != 200:
                    logger.error(f"Failed to load {actual_model_name}: {response.text}")
                
                logger.info(f"Successfully downloaded and loaded {actual_model_name}")
                
            except Exception as e:
                logger.error(f"Error setting up model {model_id}: {e}")
    
    def _create_model_instance(self, model_name: str, gpu_id: int) -> Optional[str]:
        """Create a serving instance for a model on a specific GPU"""
        # Map model IDs to actual model names
        model_name_map = {
            "llm7b": "gpt2",
            "llm30b": "gpt2-medium",
            "stable_diffusion": "CompVis/stable-diffusion-v1-4"
        }
        
        actual_model_name = model_name_map.get(model_name, "gpt2")
        
        try:
            # Check if we already have an instance
            instance_key = f"{actual_model_name}_gpu{gpu_id}"
            if instance_key in self.model_instances:
                return self.model_instances[instance_key]
            
            # Create new serving instance
            model_info = self.models[model_name]
            if model_info.gpus_required == 1:
                device = f"cuda:{gpu_id}"
            else:
                # Multi-GPU: use consecutive GPUs
                devices = [f"cuda:{gpu_id + i}" for i in range(model_info.gpus_required)]
                device = ",".join(devices)
            
            response = requests.post(
                f"{self.server_url}/standalone/serve",
                json={"model_name": actual_model_name, "device": device}
            )
            
            if response.status_code == 200:
                data = response.json()['data']
                instance_id = data['instance_id']
                self.model_instances[instance_key] = instance_id
                logger.info(f"Created instance {instance_id} for {actual_model_name} on {device}")
                return instance_id
            else:
                logger.error(f"Failed to create instance: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating instance for {model_name}: {e}")
            return None
    
    def _estimate_execution_time(self, node_exec: NodeExecution, request: Request) -> float:
        """Estimate execution time using the estimate API"""
        try:
            # Get instance ID
            gpu_state = self.gpu_states[node_exec.gpu_id]
            instance_id = gpu_state.current_instance
            
            # Safety check: if no instance exists, use fallback
            if instance_id is None:
                logger.warning(f"No instance available for estimation, using pre-computed time")
                return request.node_execution_times.get(node_exec.node_id, 10.0)
            
            # Prepare request data
            node_config = request.node_configs.get(node_exec.node_id, {})
            
            # Build data for estimation
            data = {
                "prompt": request.input_data.get("user_prompt", "Sample prompt"),
                **node_config
            }
            
            response = requests.post(
                f"{self.server_url}/standalone/estimate/{instance_id}",
                json={"instance_id": instance_id, "data": data}
            )
            
            if response.status_code == 200:
                result = response.json()
                estimated_time = result['data']['estimated_execution_time']
                return estimated_time
            else:
                logger.warning(f"Estimation failed, using pre-computed time: {response.text}")
                return request.node_execution_times.get(node_exec.node_id, 10.0)
                
        except Exception as e:
            logger.warning(f"Estimation error, using pre-computed time: {e}")
            return request.node_execution_times.get(node_exec.node_id, 10.0)
    
    def _call_model(self, node_exec: NodeExecution, request: Request) -> float:
        """Call model and return actual execution time"""
        try:
            # Get instance ID
            gpu_state = self.gpu_states[node_exec.gpu_id]
            instance_id = gpu_state.current_instance
            
            # Prepare request data
            node_config = request.node_configs.get(node_exec.node_id, {})
            
            # Build data for call
            data = {
                "prompt": request.input_data.get("user_prompt", "Sample prompt"),
                **node_config
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/standalone/call/{instance_id}",
                json={"instance_id": instance_id, "data": data}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                return end_time - start_time
            else:
                logger.error(f"Model call failed: {response.text}")
                # Use pre-computed time as fallback
                return request.node_execution_times.get(node_exec.node_id, 10.0)
                
        except Exception as e:
            logger.error(f"Model call error: {e}")
            # Use pre-computed time as fallback
            return request.node_execution_times.get(node_exec.node_id, 10.0)
    
    def _execute_node(self, node_exec: NodeExecution, request: Request):
        """Execute a node on its assigned GPU"""
        gpu_id = node_exec.gpu_id
        gpu_state = self.gpu_states[gpu_id]
        model_name = node_exec.model_name
        
        # Handle model switching
        switch_time = 0.0
        if gpu_state.current_model != model_name:
            switch_time = self._get_model_switch_time(gpu_state.current_model, model_name)
            logger.info(f"Switching model on GPU {gpu_id} from {gpu_state.current_model} to {model_name} "
                       f"(switch time: {switch_time:.2f}s)")
            
            # In simulate mode, create new instance
            if self.simulate:
                instance_id = self._create_model_instance(model_name, gpu_id)
                gpu_state.current_instance = instance_id
            
            gpu_state.current_model = model_name
            time.sleep(switch_time)  # Simulate switch time
        
        # Execute the node
        node_exec.start_time = time.time()
        
        if self.simulate:
            # Use actual model call
            execution_time = self._call_model(node_exec, request)
        else:
            # Use estimate API
            estimated_time = self._estimate_execution_time(node_exec, request)
            node_exec.estimated_time = estimated_time
            # Use pre-computed time for actual execution simulation
            execution_time = request.node_execution_times.get(node_exec.node_id, estimated_time)
            time.sleep(execution_time)  # Simulate execution
        
        node_exec.end_time = time.time()
        node_exec.status = "completed"
        
        # Mark GPU as free
        gpu_state.busy = False
        
        logger.info(f"Completed node {node_exec.node_id} of request {node_exec.request_id} "
                   f"on GPU {gpu_id} (execution time: {execution_time:.2f}s)")
    
    def _process_request(self, request: Request):
        """Process a single request by creating node executions"""
        workflow = self.workflows[request.workflow_id]
        dependencies = workflow.get_dependencies()
        
        # Track request start time
        self.request_start_times[request.request_id] = time.time()
        
        # Create node executions
        node_execs = {}
        for node in workflow.nodes:
            node_key = f"{request.request_id}_{node.id}"
            node_exec = NodeExecution(
                request_id=request.request_id,
                workflow_id=request.workflow_id,
                node_id=node.id,
                model_name=node.model
            )
            node_execs[node.id] = node_exec
            self.executions[node_key] = node_exec
        
        # Mark nodes with no dependencies as ready
        for node_id, deps in dependencies.items():
            if len(deps) == 0:
                node_execs[node_id].status = "ready"
                self.node_queue.append(node_execs[node_id])
    
    def _update_ready_nodes(self, completed_node: NodeExecution):
        """Update node statuses after a node completes"""
        request_id = completed_node.request_id
        workflow = self.workflows[completed_node.workflow_id]
        dependencies = workflow.get_dependencies()
        
        # Check each node to see if it's now ready
        for node in workflow.nodes:
            node_key = f"{request_id}_{node.id}"
            node_exec = self.executions.get(node_key)
            
            if node_exec and node_exec.status == "pending":
                # Check if all dependencies are completed
                deps = dependencies[node.id]
                all_deps_complete = True
                for dep_id in deps:
                    dep_key = f"{request_id}_{dep_id}"
                    dep_exec = self.executions.get(dep_key)
                    if not dep_exec or dep_exec.status != "completed":
                        all_deps_complete = False
                        break
                
                if all_deps_complete:
                    node_exec.status = "ready"
                    self.node_queue.append(node_exec)
        
        # Check if request is complete
        request_complete = True
        for node in workflow.nodes:
            node_key = f"{request_id}_{node.id}"
            node_exec = self.executions.get(node_key)
            if not node_exec or node_exec.status != "completed":
                request_complete = False
                break
        
        if request_complete:
            self.request_end_times[request_id] = time.time()
            logger.info(f"Request {request_id} completed")
    
    def run(self, requests: List[Request]):
        """Run the scheduler on a list of requests"""
        logger.info(f"Starting baseline scheduler with {len(requests)} requests")
        logger.info(f"Mode: {self.mode}, Simulate: {self.simulate}")
        logger.info(f"Available GPUs: {self.gpus}")
        
        # Download and load models if in simulate mode
        if self.simulate:
            self._download_and_load_models()
        
        # Process requests based on mode
        if self.mode == "offline":
            # Process all requests at once
            for request in requests:
                self._process_request(request)
        else:
            # Online mode: process requests based on arrival time
            # Sort requests by timestamp
            requests_sorted = sorted(requests, key=lambda r: r.timestamp)
            request_idx = 0
            start_time = time.time()
            
            # Convert first request timestamp to relative time
            if requests_sorted:
                base_timestamp = requests_sorted[0].timestamp
        
        # Main scheduling loop
        logger.info("Starting main scheduling loop...")
        
        while self.node_queue or (self.mode == "online" and request_idx < len(requests)):
            current_time = time.time()
            
            # In online mode, check for new requests
            if self.mode == "online" and request_idx < len(requests):
                request = requests[request_idx]
                # Calculate when request should arrive
                arrival_offset = (request.timestamp - base_timestamp).total_seconds()
                if current_time - start_time >= arrival_offset:
                    self._process_request(request)
                    request_idx += 1
            
            # Try to schedule ready nodes
            if self.node_queue:
                node_exec = self.node_queue.popleft()
                
                # Find available GPU
                gpu_id = self._find_available_gpu(node_exec.model_name)
                
                if gpu_id is not None:
                    # Assign to GPU and execute
                    node_exec.gpu_id = gpu_id
                    node_exec.status = "running"
                    self.gpu_states[gpu_id].busy = True
                    
                    logger.info(f"Scheduling node {node_exec.node_id} of request {node_exec.request_id} "
                               f"on GPU {gpu_id}")
                    
                    # Execute in a separate thread (simplified: sequential for now)
                    self._execute_node(node_exec, next(r for r in requests if r.request_id == node_exec.request_id))
                    
                    # Mark as completed and update ready nodes
                    self.completed_executions.append(node_exec)
                    self._update_ready_nodes(node_exec)
                else:
                    # No GPU available, put back in queue
                    self.node_queue.appendleft(node_exec)
                    time.sleep(0.1)  # Small delay to avoid busy waiting
            else:
                time.sleep(0.1)  # Small delay when queue is empty
        
        logger.info("Scheduling completed")
        self._print_metrics()
    
    def _print_metrics(self):
        """Print execution metrics"""
        logger.info("\n" + "="*60)
        logger.info("EXECUTION METRICS")
        logger.info("="*60)
        
        # Total execution time
        if self.completed_executions:
            start_time = min(e.start_time for e in self.completed_executions)
            end_time = max(e.end_time for e in self.completed_executions)
            total_time = end_time - start_time
            logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Request metrics
        if self.request_end_times:
            request_times = []
            for req_id, end_time in self.request_end_times.items():
                start_time = self.request_start_times[req_id]
                request_times.append(end_time - start_time)
            
            logger.info(f"\nRequest completion times:")
            logger.info(f"  Average: {np.mean(request_times):.2f}s")
            logger.info(f"  Median: {np.median(request_times):.2f}s")
            logger.info(f"  P99: {np.percentile(request_times, 99):.2f}s")
            logger.info(f"  Min: {np.min(request_times):.2f}s")
            logger.info(f"  Max: {np.max(request_times):.2f}s")
        
        # GPU utilization
        logger.info(f"\nGPU utilization:")
        for gpu_id, gpu_state in self.gpu_states.items():
            # Count executions per GPU
            gpu_execs = [e for e in self.completed_executions if e.gpu_id == gpu_id]
            if gpu_execs:
                gpu_busy_time = sum(e.execution_time for e in gpu_execs)
                logger.info(f"  GPU {gpu_id}: {len(gpu_execs)} executions, "
                           f"{gpu_busy_time:.2f}s busy time")
        
        # Model statistics
        logger.info(f"\nModel execution counts:")
        model_counts = defaultdict(int)
        for exec in self.completed_executions:
            model_counts[exec.model_name] += 1
        for model, count in sorted(model_counts.items()):
            logger.info(f"  {model}: {count} executions")
        
        # Write detailed log
        self._write_detailed_log()
    
    def _write_detailed_log(self):
        """Write detailed execution log"""
        log_file = Path("baseline_execution_log.json")
        
        log_data = {
            "summary": {
                "total_requests": len(self.request_end_times),
                "total_nodes_executed": len(self.completed_executions),
                "mode": self.mode,
                "simulate": self.simulate,
                "gpus": self.gpus
            },
            "executions": []
        }
        
        for exec in self.completed_executions:
            exec_data = {
                "request_id": exec.request_id,
                "workflow_id": exec.workflow_id,
                "node_id": exec.node_id,
                "model_name": exec.model_name,
                "gpu_id": exec.gpu_id,
                "start_time": exec.start_time,
                "end_time": exec.end_time,
                "execution_time": exec.execution_time,
                "estimated_time": exec.estimated_time
            }
            log_data["executions"].append(exec_data)
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"\nDetailed execution log written to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Baseline scheduler for GSwarm workflows")
    parser.add_argument(
        "--gpus", 
        type=str, 
        required=True,
        help="Comma-separated list of GPU IDs (e.g., '2,3,4,5,10')"
    )
    parser.add_argument(
        "--simulate", 
        type=lambda x: x.lower() == 'true',
        default=False,
        help="Use simulation mode with actual model calls (true/false)"
    )
    parser.add_argument(
        "--mode", 
        choices=["offline", "online"],
        default="offline",
        help="Scheduling mode: offline (batch) or online (streaming)"
    )
    parser.add_argument(
        "--config", 
        type=Path,
        default=Path("system_config.yaml"),
        help="Path to system configuration file"
    )
    parser.add_argument(
        "--requests", 
        type=Path,
        default=Path("workflow_requests.yaml"),
        help="Path to workflow requests file"
    )
    
    args = parser.parse_args()
    
    # Parse GPU list
    gpu_list = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    
    # Create scheduler
    scheduler = BaselineScheduler(
        gpus=gpu_list,
        simulate=args.simulate,
        mode=args.mode
    )
    
    # Load configuration
    scheduler.load_config(args.config)
    
    # Load requests
    requests = scheduler.load_requests(args.requests)
    
    # Run scheduler
    scheduler.run(requests)


if __name__ == "__main__":
    main()

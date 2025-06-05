"""
CLI interface for gswarm_model system.
Provides command-line tools for model management operations.
"""

import os
import sys
import json
import yaml
import typer
from typing import Optional, List
from typing_extensions import Annotated
from pathlib import Path
from loguru import logger
import asyncio
import grpc
import requests
from datetime import datetime

# Configure Loguru
logger.remove()
logger.add(sys.stderr, level="INFO")

app = typer.Typer(
    name="gswarm-model",
    help="Distributed model storage and management system for GPU clusters.",
    epilog="For more information, visit: https://github.com/your-repo/gswarm-model",
)


def ensure_grpc_files():
    """Check if gRPC files exist and generate them if needed"""
    current_dir = Path(__file__).parent
    pb2_file = current_dir / "model_pb2.py"
    pb2_grpc_file = current_dir / "model_pb2_grpc.py"
    
    if not pb2_file.exists() or not pb2_grpc_file.exists():
        logger.info("gRPC protobuf files not found, generating them...")
        try:
            from .generate_grpc import generate_grpc_files
            generate_grpc_files()
            logger.info("gRPC protobuf files generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate gRPC files: {e}")
            logger.error("Please run 'gsmodel generate-grpc' manually")
            raise


# Head Node Commands

@app.command(
    name="start",
    help="Start the head node (model management coordinator).",
    epilog="Example: gsmodel start --host 0.0.0.0 --port 9010 --http-port 9011",
)
def start_head_node(
    host: Annotated[str, typer.Option(help="Host address for the head node.")] = "localhost",
    port: Annotated[int, typer.Option(help="Port for the head node gRPC server.")] = 9010,
    http_port: Annotated[int, typer.Option(help="Port for HTTP API server.")] = 9011,
    background: Annotated[bool, typer.Option("--background", help="Run head node in background mode.")] = False,
    data_dir: Annotated[str, typer.Option(help="Directory for storing persistent data.")] = ".gswarm_model_data",
):
    """
    Start the head node server for model management.
    Example: gsmodel start --port 9010 --http-port 9011
    """
    
    # Ensure gRPC files exist before starting
    ensure_grpc_files()
    
    if background:
        import subprocess
        cmd = [
            "gsmodel", "start",
            "--host", host,
            "--port", str(port),
            "--http-port", str(http_port),
            "--data-dir", data_dir,
        ]
        subprocess.Popen(cmd)
        logger.info("Head node started in background")
        logger.info(f"gRPC server: {host}:{port}")
        logger.info(f"HTTP API: http://{host}:{http_port}")
        return
    
    from .head import run_head_node
    
    # Set data directory
    from .head import state
    state.data_directory = data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f"Starting head node on {host}:{port}")
    logger.info(f"HTTP API will be available at http://{host}:{http_port}")
    logger.info(f"Data directory: {data_dir}")
    
    run_head_node(host, port, http_port)


# Client Node Commands

@app.command(
    name="connect",
    help="Connect this node as a client to the head node.",
    epilog="Example: gsmodel connect localhost:9010 --node-id worker1",
)
def connect_client_node(
    head_address: Annotated[str, typer.Argument(help="Address of the head node (e.g., localhost:9010).")],
    node_id: Annotated[str, typer.Option("--node-id", help="Unique identifier for this client node")] = None,
):
    """
    Connect this node as a client to the head node.
    Example: gsmodel connect localhost:9010 --node-id worker1
    """
    
    # Ensure gRPC files exist before connecting
    ensure_grpc_files()
    
    if not node_id:
        import platform
        import socket
        hostname = platform.node()
        node_id = f"{hostname}_{socket.gethostname()}"
    
    logger.info(f"Connecting client node {node_id} to head node at {head_address}")
    
    from .client import start_client_node
    start_client_node(node_id, head_address)


# Model Management Commands

@app.command(
    name="register",
    help="Register a new model in the system.",
    epilog="Example: gsmodel register llama-7b --type llm --url https://huggingface.co/...",
)
def register_model(
    model_name: Annotated[str, typer.Argument(help="Name of the model to register.")],
    model_type: Annotated[str, typer.Option("--type", help="Model type (llm, diffusion, vision, etc.)")] = "llm",
    source_url: Annotated[str, typer.Option("--url", help="Source URL (e.g., HuggingFace repository)")] = None,
    description: Annotated[str, typer.Option("--desc", help="Model description")] = None,
    tags: Annotated[List[str], typer.Option("--tag", help="Model tags")] = None,
    head_address: Annotated[str, typer.Option("--head", help="Head node address")] = "localhost:9011",
):
    """
    Register a new model in the system.
    Example: gsmodel register llama-7b --type llm --url https://huggingface.co/meta-llama/Llama-2-7b-hf
    """
    
    try:
        url = f"http://{head_address}/models/{model_name}/register"
        
        payload = {
            "model_type": model_type,
            "source_url": source_url,
            "metadata": {
                "description": description,
                "tags": tags or []
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        if data["success"]:
            logger.info(f"✅ {data['message']}")
        else:
            logger.error(f"❌ {data['message']}")
            
    except Exception as e:
        logger.error(f"Failed to register model: {e}")


@app.command(
    name="list",
    help="List all registered models.",
    epilog="Example: gsmodel list --head localhost:9011",
)
def list_models(
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed information")] = False,
):
    """
    List all registered models.
    Example: gsmodel list
    """
    
    try:
        url = f"http://{head_address}/models"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        models = data["models"]
        
        if not models:
            logger.info("No models registered")
            return
        
        logger.info(f"Found {len(models)} registered models:")
        
        for model in models:
            logger.info(f"📦 {model['model_name']} ({model['model_type']})")
            if verbose:
                logger.info(f"   Locations: {', '.join(model['locations']) if model['locations'] else 'None'}")
                logger.info(f"   Services: {', '.join(model['services']) if model['services'] else 'None'}")
                if model.get('size'):
                    size_gb = model['size'] / (1024**3)
                    logger.info(f"   Size: {size_gb:.2f} GB")
                logger.info("")
            
    except Exception as e:
        logger.error(f"Failed to list models: {e}")


@app.command(
    name="info",
    help="Get detailed information about a model.",
    epilog="Example: gsmodel info llama-7b",
)
def get_model_info(
    model_name: Annotated[str, typer.Argument(help="Name of the model.")],
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
):
    """
    Get detailed information about a specific model.
    Example: gsmodel info llama-7b
    """
    
    try:
        url = f"http://{head_address}/models/{model_name}"
        response = requests.get(url)
        response.raise_for_status()
        
        model = response.json()
        
        logger.info(f"📦 Model: {model['model_name']}")
        logger.info(f"   Type: {model['model_type']}")
        
        if model.get('model_size'):
            size_gb = model['model_size'] / (1024**3)
            logger.info(f"   Size: {size_gb:.2f} GB")
        
        if model.get('model_hash'):
            logger.info(f"   Hash: {model['model_hash'][:16]}...")
        
        logger.info(f"   Locations: {', '.join(model['stored_locations']) if model['stored_locations'] else 'None'}")
        
        if model['available_services']:
            logger.info(f"   Active Services:")
            for device, service_url in model['available_services'].items():
                logger.info(f"     {device}: {service_url}")
        else:
            logger.info(f"   Active Services: None")
        
        if model.get('metadata'):
            metadata = model['metadata']
            if metadata.get('description'):
                logger.info(f"   Description: {metadata['description']}")
            if metadata.get('tags'):
                logger.info(f"   Tags: {', '.join(metadata['tags'])}")
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Model '{model_name}' not found")
        else:
            logger.error(f"Failed to get model info: {e}")
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")


@app.command(
    name="unregister",
    help="Unregister a model from the system.",
    epilog="Example: gsmodel unregister llama-7b",
)
def unregister_model(
    model_name: Annotated[str, typer.Argument(help="Name of the model to unregister.")],
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
    force: Annotated[bool, typer.Option("--force", help="Force removal without confirmation")] = False,
):
    """
    Unregister a model from the system.
    Example: gsmodel unregister llama-7b
    """
    
    if not force:
        confirm = typer.confirm(f"Are you sure you want to unregister model '{model_name}'?")
        if not confirm:
            logger.info("Operation cancelled")
            return
    
    try:
        url = f"http://{head_address}/models/{model_name}"
        response = requests.delete(url)
        response.raise_for_status()
        
        data = response.json()
        if data["success"]:
            logger.info(f"✅ {data['message']}")
        else:
            logger.error(f"❌ {data['message']}")
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Model '{model_name}' not found")
        else:
            logger.error(f"Failed to unregister model: {e}")
    except Exception as e:
        logger.error(f"Failed to unregister model: {e}")


# Job Management Commands

@app.command(
    name="job",
    help="Create and execute a job from YAML file.",
    epilog="Example: gsmodel job workflow.yaml",
)
def create_job(
    job_file: Annotated[str, typer.Argument(help="YAML file containing job definition.")],
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
    wait: Annotated[bool, typer.Option("--wait", help="Wait for job completion")] = False,
):
    """
    Create and execute a job from YAML file.
    Example: gsmodel job examples/llama-deployment.yaml --wait
    """
    
    try:
        job_file_path = Path(job_file)
        if not job_file_path.exists():
            logger.error(f"Job file not found: {job_file}")
            return
        
        with open(job_file_path, 'r') as f:
            url = f"http://{head_address}/jobs/from-yaml"
            files = {"file": f}
            response = requests.post(url, files=files)
            response.raise_for_status()
        
        data = response.json()
        if data["success"]:
            job_id = data["job_id"]
            logger.info(f"✅ {data['message']}")
            logger.info(f"Job ID: {job_id}")
            
            if wait:
                logger.info("Waiting for job completion...")
                wait_for_job_completion(job_id, head_address)
        else:
            logger.error(f"❌ {data['message']}")
            
    except Exception as e:
        logger.error(f"Failed to create job: {e}")


@app.command(
    name="jobs",
    help="List all jobs.",
    epilog="Example: gsmodel jobs",
)
def list_jobs(
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
    status: Annotated[str, typer.Option("--status", help="Filter by status")] = None,
):
    """
    List all jobs.
    Example: gsmodel jobs --status running
    """
    
    try:
        url = f"http://{head_address}/jobs"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        jobs = data["jobs"]
        
        if status:
            jobs = [job for job in jobs if job["status"] == status]
        
        if not jobs:
            logger.info("No jobs found")
            return
        
        logger.info(f"Found {len(jobs)} jobs:")
        
        for job in jobs:
            status_icon = "✅" if job["status"] == "completed" else "❌" if job["status"] == "failed" else "🔄"
            logger.info(f"{status_icon} {job['name']} ({job['job_id'][:8]})")
            logger.info(f"   Status: {job['status']}")
            logger.info(f"   Progress: {job['completed_actions']}/{job['total_actions']} actions")
            logger.info(f"   Created: {job['created_at']}")
            logger.info("")
            
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")


@app.command(
    name="job-status",
    help="Get status of a specific job.",
    epilog="Example: gsmodel job-status abc123def",
)
def get_job_status(
    job_id: Annotated[str, typer.Argument(help="Job ID.")],
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show action details")] = False,
):
    """
    Get detailed status of a specific job.
    Example: gsmodel job-status abc123def --verbose
    """
    
    try:
        url = f"http://{head_address}/jobs/{job_id}/status"
        response = requests.get(url)
        response.raise_for_status()
        
        job = response.json()
        
        status_icon = "✅" if job["status"] == "completed" else "❌" if job["status"] == "failed" else "🔄"
        logger.info(f"{status_icon} Job: {job['name']} ({job_id})")
        logger.info(f"   Status: {job['status']}")
        logger.info(f"   Progress: {job['completed_actions']}/{job['total_actions']} actions")
        
        if job.get('error_message'):
            logger.info(f"   Error: {job['error_message']}")
        
        logger.info(f"   Created: {job['created_at']}")
        if job.get('started_at'):
            logger.info(f"   Started: {job['started_at']}")
        if job.get('completed_at'):
            logger.info(f"   Completed: {job['completed_at']}")
        
        if verbose and job.get('actions'):
            logger.info(f"   Actions:")
            for action in job['actions']:
                action_icon = "✅" if action["status"] == "completed" else "❌" if action["status"] == "failed" else "🔄"
                logger.info(f"     {action_icon} {action['action_id']} ({action['action_type']})")
                if action.get('error_message'):
                    logger.info(f"        Error: {action['error_message']}")
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Job '{job_id}' not found")
        else:
            logger.error(f"Failed to get job status: {e}")
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")


def wait_for_job_completion(job_id: str, head_address: str):
    """Wait for job to complete and show progress"""
    import time
    
    while True:
        try:
            url = f"http://{head_address}/jobs/{job_id}/status"
            response = requests.get(url)
            response.raise_for_status()
            
            job = response.json()
            status = job["status"]
            
            if status in ["completed", "failed", "cancelled"]:
                status_icon = "✅" if status == "completed" else "❌"
                logger.info(f"{status_icon} Job {status}")
                if job.get('error_message'):
                    logger.error(f"Error: {job['error_message']}")
                break
            
            logger.info(f"🔄 Job running... ({job['completed_actions']}/{job['total_actions']} actions completed)")
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            break


# System Commands

@app.command(
    name="status",
    help="Get system status.",
    epilog="Example: gsmodel status",
)
def get_system_status(
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
):
    """
    Get system-wide status.
    Example: gsmodel status
    """
    
    try:
        url = f"http://{head_address}/status"
        response = requests.get(url)
        response.raise_for_status()
        
        status = response.json()
        
        logger.info("🖥️  System Status:")
        logger.info(f"   Nodes: {status['online_nodes']}/{status['total_nodes']} online")
        logger.info(f"   Models: {status['total_models']} registered")
        logger.info(f"   Active Services: {status['active_services']}")
        
        if status.get('storage_utilization'):
            logger.info(f"   Storage Utilization:")
            for device, utilization in status['storage_utilization'].items():
                logger.info(f"     {device}: {utilization:.1f}%")
            
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")


@app.command(
    name="nodes",
    help="List connected nodes.",
    epilog="Example: gsmodel nodes",
)
def list_nodes(
    head_address: Annotated[str, typer.Option("--head", help="Head node HTTP address")] = "localhost:9011",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed information")] = False,
):
    """
    List all connected nodes and their capabilities.
    Example: gsmodel nodes --verbose
    """
    
    try:
        url = f"http://{head_address}/nodes"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        nodes = data["nodes"]
        
        if not nodes:
            logger.info("No nodes connected")
            return
        
        logger.info(f"Connected Nodes ({len(nodes)}):")
        
        for node in nodes:
            status_icon = "🟢" if node["is_online"] else "🔴"
            logger.info(f"{status_icon} {node['hostname']} ({node['node_id']})")
            
            if verbose:
                if node.get('ip_address'):
                    logger.info(f"   IP: {node['ip_address']}")
                
                logger.info(f"   Last Seen: {node['last_seen']}")
                
                if node.get('storage_devices'):
                    logger.info(f"   Storage Devices:")
                    for device_name, storage in node['storage_devices'].items():
                        total_gb = storage['total'] / (1024**3)
                        available_gb = storage['available'] / (1024**3)
                        utilization = (storage['used'] / storage['total']) * 100 if storage['total'] > 0 else 0
                        logger.info(f"     {device_name}: {available_gb:.1f}GB available / {total_gb:.1f}GB total ({utilization:.1f}% used)")
                
                if node.get('gpu_info'):
                    logger.info(f"   GPUs: {', '.join(node['gpu_info'])}")
                
                logger.info("")
            
    except Exception as e:
        logger.error(f"Failed to list nodes: {e}")


# Utility Commands

@app.command(
    name="generate-grpc",
    help="Generate gRPC protobuf files.",
)
def generate_grpc():
    """Generate gRPC protobuf files from .proto definition."""
    from .generate_grpc import generate_grpc_files
    generate_grpc_files()


@app.command(
    name="example",
    help="Generate example job YAML files.",
)
def generate_examples(
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "examples",
):
    """Generate example job YAML files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Simple download and serve example
    simple_example = {
        "name": "llama-deployment-pipeline",
        "description": "Download and serve Llama model",
        "actions": [
            {
                "action_id": "download_llama",
                "action_type": "download",
                "model_name": "llama-7b-chat",
                "source_url": "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
                "devices": ["node1:disk"],
                "dependencies": []
            },
            {
                "action_id": "move_to_gpu",
                "action_type": "move",
                "model_name": "llama-7b-chat",
                "devices": ["node1:disk", "node1:gpu0"],
                "keep_source": True,
                "dependencies": ["download_llama"]
            },
            {
                "action_id": "serve_model",
                "action_type": "serve",
                "model_name": "llama-7b-chat",
                "port": 9080,
                "devices": ["node1:gpu0"],
                "dependencies": ["move_to_gpu"]
            },
            {
                "action_id": "health_check",
                "action_type": "health_check",
                "target_url": "http://node1:9080/health",
                "devices": [],
                "dependencies": ["serve_model"]
            }
        ]
    }
    
    simple_file = output_path / "simple-deployment.yaml"
    with open(simple_file, 'w') as f:
        yaml.dump(simple_example, f, default_flow_style=False, sort_keys=False)
    
    # Multi-model example
    multi_example = {
        "name": "multi-model-inference",
        "description": "Deploy multiple models for inference",
        "actions": [
            {
                "action_id": "download_llama",
                "action_type": "download",
                "model_name": "llama-7b",
                "source_url": "https://huggingface.co/meta-llama/Llama-2-7b-hf",
                "devices": ["node1:disk"],
                "dependencies": []
            },
            {
                "action_id": "download_diffusion",
                "action_type": "download",
                "model_name": "stable-diffusion-xl",
                "source_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
                "devices": ["node2:disk"],
                "dependencies": []
            },
            {
                "action_id": "load_llama_gpu",
                "action_type": "move",
                "model_name": "llama-7b",
                "devices": ["node1:disk", "node1:gpu0"],
                "dependencies": ["download_llama"]
            },
            {
                "action_id": "load_diffusion_gpu",
                "action_type": "move",
                "model_name": "stable-diffusion-xl",
                "devices": ["node2:disk", "node2:gpu0"],
                "dependencies": ["download_diffusion"]
            },
            {
                "action_id": "serve_llama",
                "action_type": "serve",
                "model_name": "llama-7b",
                "port": 9080,
                "devices": ["node1:gpu0"],
                "dependencies": ["load_llama_gpu"]
            },
            {
                "action_id": "serve_diffusion",
                "action_type": "serve",
                "model_name": "stable-diffusion-xl",
                "port": 9081,
                "devices": ["node2:gpu0"],
                "dependencies": ["load_diffusion_gpu"]
            }
        ]
    }
    
    multi_file = output_path / "multi-model-deployment.yaml"
    with open(multi_file, 'w') as f:
        yaml.dump(multi_example, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Generated example files in {output_dir}/:")
    logger.info(f"  - simple-deployment.yaml")
    logger.info(f"  - multi-model-deployment.yaml")


if __name__ == "__main__":
    app() 
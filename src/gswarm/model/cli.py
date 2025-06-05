"""Model management CLI commands"""

import typer
from typing import Optional, List
from loguru import logger
import requests
import json

app = typer.Typer(help="Model management operations")

def get_api_url(host: str = "localhost:9010") -> str:
    """Ensure host has http:// prefix"""
    if not host.startswith("http://") and not host.startswith("https://"):
        return f"http://{host}"
    return host

@app.command()
def list(
    location: Optional[str] = typer.Option(None, "--location", "-l", help="Filter by location"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by model type"),
    host: str = typer.Option("localhost:9010", "--host", help="Host API address"),
):
    """List all registered models"""
    try:
        url = f"{get_api_url(host)}/api/v1/models"
        params = {}
        if location:
            params["location"] = location
        if type:
            params["type"] = type
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        models = data.get("models", [])
        
        if models:
            logger.info(f"Found {data.get('total', len(models))} model(s):")
            for model in models:
                logger.info(f"\n  Model: {model['model_name']}")
                logger.info(f"    Type: {model['model_type']}")
                logger.info(f"    Size: {model['model_size'] / 1e9:.2f} GB")
                logger.info(f"    Status: {model['status']}")
                if model.get('locations'):
                    logger.info(f"    Locations: {', '.join(model['locations'])}")
                if model.get('services'):
                    logger.info("    Services:")
                    for node, url in model['services'].items():
                        logger.info(f"      - {node}: {url}")
        else:
            logger.info("No models found")
    except Exception as e:
        logger.error(f"Failed to list models: {e}")

@app.command()
def info(
    model_name: str = typer.Argument(..., help="Model name"),
    host: str = typer.Option("localhost:9010", "--host", help="Host API address"),
):
    """Get detailed model information"""
    try:
        url = f"{get_api_url(host)}/api/v1/models/{model_name}"
        response = requests.get(url)
        response.raise_for_status()
        
        model = response.json()
        logger.info(f"Model: {model['model_name']}")
        logger.info(f"  Type: {model['model_type']}")
        logger.info(f"  Size: {model['model_size'] / 1e9:.2f} GB")
        logger.info(f"  Status: {model['status']}")
        
        if model.get('locations'):
            logger.info("  Locations:")
            for loc in model['locations']:
                logger.info(f"    - {loc}")
        
        if model.get('services'):
            logger.info("  Services:")
            for node, url in model['services'].items():
                logger.info(f"    - {node}: {url}")
        
        if model.get('metadata'):
            logger.info("  Metadata:")
            for key, value in model['metadata'].items():
                logger.info(f"    {key}: {value}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Model '{model_name}' not found")
        else:
            logger.error(f"Failed to get model info: {e}")
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")

@app.command()
def register(
    model_name: str = typer.Argument(..., help="Model name"),
    type: str = typer.Option(..., "--type", "-t", help="Model type (llm, diffusion, vision)"),
    source: str = typer.Option(..., "--source", "-s", help="Source URL or path"),
    size: Optional[int] = typer.Option(None, "--size", help="Model size in bytes"),
    host: str = typer.Option("localhost:9010", "--host", help="Host API address"),
):
    """Register a new model in the system"""
    try:
        url = f"{get_api_url(host)}/api/v1/models"
        data = {
            "model_name": model_name,
            "model_type": type,
            "source_url": source,
        }
        if size:
            data["model_size"] = size
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Model '{model_name}' registered successfully")
        if result.get("message"):
            logger.info(f"  {result['message']}")
    except Exception as e:
        logger.error(f"Failed to register model: {e}")

@app.command()
def download(
    model_name: str = typer.Argument(..., help="Model name to download"),
    device: str = typer.Option("disk", "--device", "-d", help="Target device"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Priority (high/normal/low)"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Download a model from source"""
    try:
        # This should be sent to the client node, not the host
        url = f"{get_api_url(host)}/api/v1/models/{model_name}/download"
        data = {
            "target_device": device,
            "priority": priority,
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Download started for model '{model_name}'")
        if result.get("task_id"):
            logger.info(f"  Task ID: {result['task_id']}")
            logger.info(f"  Status: {result.get('status', 'queued')}")
            if result.get("estimated_time"):
                logger.info(f"  Estimated time: {result['estimated_time']}s")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")

@app.command()
def move(
    model_name: str = typer.Argument(..., help="Model name to move"),
    source: str = typer.Option(..., "--from", "-f", help="Source device"),
    target: str = typer.Option(..., "--to", "-t", help="Target device"),
    keep_source: bool = typer.Option(False, "--keep-source", "-k", help="Keep source copy"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Priority"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Move model between storage devices"""
    try:
        url = f"{get_api_url(host)}/api/v1/models/{model_name}/move"
        data = {
            "source_device": source,
            "target_device": target,
            "keep_source": keep_source,
            "priority": priority,
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Move operation started for model '{model_name}'")
        logger.info(f"  From: {source} → To: {target}")
        if result.get("task_id"):
            logger.info(f"  Task ID: {result['task_id']}")
            logger.info(f"  Status: {result.get('status', 'queued')}")
    except Exception as e:
        logger.error(f"Failed to move model: {e}")

@app.command()
def copy(
    model_name: str = typer.Argument(..., help="Model name to copy"),
    source: str = typer.Option(..., "--from", "-f", help="Source device (can include node:)"),
    target: str = typer.Option(..., "--to", "-t", help="Target device (can include node:)"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Priority"),
    bandwidth_limit: Optional[int] = typer.Option(None, "--bandwidth", "-b", help="Bandwidth limit (bytes/sec)"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Copy model to another location (including cross-node)"""
    try:
        url = f"{get_api_url(host)}/api/v1/models/{model_name}/copy"
        data = {
            "source_device": source,
            "target_device": target,
            "priority": priority,
        }
        if bandwidth_limit:
            data["bandwidth_limit"] = bandwidth_limit
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Copy operation started for model '{model_name}'")
        logger.info(f"  From: {source} → To: {target}")
        if result.get("task_id"):
            logger.info(f"  Task ID: {result['task_id']}")
    except Exception as e:
        logger.error(f"Failed to copy model: {e}")

@app.command()
def delete(
    model_name: str = typer.Argument(..., help="Model name to delete"),
    device: str = typer.Option(..., "--device", "-d", help="Device to delete from"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Delete model from specified device"""
    try:
        url = f"{get_api_url(host)}/api/v1/models/{model_name}"
        params = {"device": device}
        if force:
            params["force"] = "true"
        
        response = requests.delete(url, params=params)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Model '{model_name}' deleted from {device}")
        if result.get("message"):
            logger.info(f"  {result['message']}")
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")

@app.command()
def serve(
    model_name: str = typer.Argument(..., help="Model name to serve"),
    device: str = typer.Option("gpu0", "--device", "-d", help="Device to serve from"),
    port: int = typer.Option(8080, "--port", "-p", help="Service port"),
    framework: str = typer.Option("vllm", "--framework", "-f", help="Serving framework"),
    max_batch_size: int = typer.Option(32, "--batch-size", help="Maximum batch size"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Start serving a model"""
    try:
        url = f"{get_api_url(host)}/api/v1/services"
        data = {
            "model_name": model_name,
            "device": device,
            "port": port,
            "config": {
                "framework": framework,
                "max_batch_size": max_batch_size,
                "gpu_memory_fraction": 0.9,
            }
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Model '{model_name}' serving started")
        if result.get("service_id"):
            logger.info(f"  Service ID: {result['service_id']}")
            logger.info(f"  URL: {result.get('url', f'http://localhost:{port}')}")
    except Exception as e:
        logger.error(f"Failed to start serving: {e}")

@app.command()
def stop(
    model_name: str = typer.Argument(..., help="Model name to stop serving"),
    service_id: Optional[str] = typer.Option(None, "--service-id", "-s", help="Specific service ID"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Stop serving a model"""
    try:
        if service_id:
            url = f"{get_api_url(host)}/api/v1/services/{service_id}"
        else:
            # Find service ID by model name
            url = f"{get_api_url(host)}/api/v1/services"
            response = requests.get(url)
            response.raise_for_status()
            services = response.json().get("services", [])
            
            service_id = None
            for service in services:
                if service.get("model_name") == model_name:
                    service_id = service.get("service_id")
                    break
            
            if not service_id:
                logger.error(f"No running service found for model '{model_name}'")
                return
            
            url = f"{get_api_url(host)}/api/v1/services/{service_id}"
        
        response = requests.delete(url)
        response.raise_for_status()
        
        logger.info(f"Service stopped successfully")
    except Exception as e:
        logger.error(f"Failed to stop service: {e}")

@app.command()
def services(
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """List all running services"""
    try:
        url = f"{get_api_url(host)}/api/v1/services"
        response = requests.get(url)
        response.raise_for_status()
        
        services = response.json().get("services", [])
        
        if services:
            logger.info(f"Found {len(services)} running service(s):")
            for service in services:
                logger.info(f"\n  Service ID: {service['service_id']}")
                logger.info(f"    Model: {service['model_name']}")
                logger.info(f"    Device: {service['device']}")
                logger.info(f"    Port: {service['port']}")
                logger.info(f"    Status: {service.get('status', 'running')}")
                logger.info(f"    URL: {service.get('url', 'N/A')}")
        else:
            logger.info("No running services found")
    except Exception as e:
        logger.error(f"Failed to list services: {e}")

@app.command()
def status(
    model_name: str = typer.Argument(..., help="Model name to check status"),
    host: str = typer.Option("localhost:9010", "--host", help="Host API address"),
):
    """Get detailed model status including operations"""
    try:
        # Get model info from host
        url = f"{get_api_url(host)}/api/v1/models/{model_name}"
        response = requests.get(url)
        response.raise_for_status()
        
        model = response.json()
        logger.info(f"Model: {model['model_name']}")
        logger.info(f"  Status: {model['status']}")
        
        # TODO: Get operation status from queue
        logger.info("  Operations: (queue status not yet implemented)")
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}") 
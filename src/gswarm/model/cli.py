"""
Model management CLI for gswarm unified system.
Provides commands for model download, deployment, and serving.
"""

import typer
from typing import Optional, List
from pathlib import Path
import requests
import json
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

# Create the model subcommand app
app = typer.Typer(
    help="Model management operations",
    rich_markup_mode="rich"
)

# Configuration - these would normally come from a config file
DEFAULT_HOST_URL = "http://localhost:8091"  # HTTP API port
DEFAULT_MODEL_API_URL = "http://localhost:9010"  # Model API port


def get_api_url(node: Optional[str] = None) -> str:
    """Get the appropriate API URL based on whether we're on host or client"""
    # In a real implementation, this would check if we're on host or client
    # and potentially use the node parameter to route to specific nodes
    return DEFAULT_MODEL_API_URL


@app.command()
def list(
    location: Optional[str] = typer.Option(None, "--location", "-l", help="Filter by storage location"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Filter by node name"),
):
    """List all available models"""
    try:
        api_url = get_api_url(node)
        response = requests.get(f"{api_url}/models")
        response.raise_for_status()
        
        data = response.json()
        models = data.get("models", [])
        
        if not models:
            console.print("No models found", style="yellow")
            return
        
        # Create a table for display
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Locations", style="yellow")
        table.add_column("Services", style="magenta")
        
        for model in models:
            # Filter by location if specified
            if location and location not in model.get("locations", []):
                continue
                
            table.add_row(
                model["name"],
                model["type"],
                ", ".join(model.get("locations", [])),
                ", ".join(model.get("services", {}).keys())
            )
        
        console.print(table)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list models: {e}")
        raise typer.Exit(1)


@app.command()
def info(
    model_name: str = typer.Argument(..., help="Name of the model"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Get detailed information about a model"""
    try:
        api_url = get_api_url(node)
        response = requests.get(f"{api_url}/models/{model_name}")
        response.raise_for_status()
        
        model = response.json()
        
        console.print(f"\n[bold cyan]Model: {model['name']}[/bold cyan]")
        console.print(f"Type: {model['type']}")
        console.print(f"Locations: {', '.join(model.get('locations', []))}")
        
        if model.get("services"):
            console.print("\n[bold]Active Services:[/bold]")
            for device, url in model["services"].items():
                console.print(f"  - {device}: {url}")
        
        if model.get("metadata"):
            console.print("\n[bold]Metadata:[/bold]")
            console.print(json.dumps(model["metadata"], indent=2))
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get model info: {e}")
        raise typer.Exit(1)


@app.command()
def register(
    model_name: str = typer.Argument(..., help="Name of the model"),
    type: str = typer.Option(..., "--type", "-t", help="Model type (llm, embedding, etc.)"),
    source: str = typer.Option(..., "--source", "-s", help="Model source URL"),
    metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="Additional metadata as JSON"),
):
    """Register a new model in the system"""
    try:
        api_url = get_api_url()
        
        request_data = {
            "name": model_name,
            "type": type,
            "metadata": json.loads(metadata) if metadata else {}
        }
        
        response = requests.post(f"{api_url}/models", json=request_data)
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to register model: {e}")
        raise typer.Exit(1)


@app.command()
def download(
    model_name: str = typer.Argument(..., help="Name of the model"),
    source: str = typer.Option(..., "--source", "-s", help="Source type (huggingface, s3, etc.) or hf:// URL"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Source URL (optional if using hf:// format)"),
    type: str = typer.Option(..., "--type", "-t", help="Model type (llm, embedding, etc.)"),
    device: Optional[str] = typer.Option("disk", "--device", "-d", help="Target device (default: disk)"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir", "-c", help="Cache directory"),
):
    """Download a model to specified device"""
    try:
        # Handle hf:// format in source parameter
        if source.startswith("hf://"):
            if url is not None:
                console.print("[red]Error:[/red] Cannot specify both hf:// format in --source and --url", style="red")
                raise typer.Exit(1)
            # Extract the model path from hf:// format
            model_path = source.replace("hf://", "")
            url = f"https://huggingface.co/{model_path}"
            source = "huggingface"
        elif url is None:
            console.print("[red]Error:[/red] --url is required when not using hf:// format", style="red")
            raise typer.Exit(1)
        
        # First register the model if it doesn't exist
        api_url = get_api_url(node)
        
        # Check if model exists
        check_response = requests.get(f"{api_url}/models/{model_name}")
        if check_response.status_code == 404:
            # Register the model first
            console.print(f"Registering model {model_name}...")
            register_data = {
                "name": model_name,
                "type": type,
                "metadata": {"source": source, "url": url}
            }
            reg_response = requests.post(f"{api_url}/models", json=register_data)
            reg_response.raise_for_status()
        
        # Now download
        target_device = f"{node}:{device}" if node else device
        download_data = {
            "model_name": model_name,
            "source_url": url,
            "target_device": target_device
        }
        
        console.print(f"Downloading {model_name} from {source} to {target_device}...")
        response = requests.post(f"{api_url}/download", json=download_data)
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download model: {e}")
        raise typer.Exit(1)


@app.command()
def move(
    model_name: str = typer.Argument(..., help="Name of the model"),
    source: str = typer.Option(..., "--from", help="Source device"),
    dest: str = typer.Option(..., "--to", help="Destination device"),
    keep_source: bool = typer.Option(False, "--keep-source", help="Keep model at source"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Move model between devices (disk, dram, gpu0, etc.)"""
    try:
        api_url = get_api_url(node)
        
        # Add node prefix if specified
        if node:
            source = f"{node}:{source}" if ":" not in source else source
            dest = f"{node}:{dest}" if ":" not in dest else dest
        
        move_data = {
            "model_name": model_name,
            "source_device": source,
            "target_device": dest,
            "keep_source": keep_source
        }
        
        console.print(f"Moving {model_name} from {source} to {dest}...")
        response = requests.post(f"{api_url}/move", json=move_data)
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to move model: {e}")
        raise typer.Exit(1)


@app.command()
def copy(
    model_name: str = typer.Argument(..., help="Name of the model"),
    source: str = typer.Option(..., "--from", help="Source device"),
    dest: str = typer.Option(..., "--to", help="Destination device"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Copy model to another device (keeps source)"""
    # This is just move with keep_source=True
    move(model_name, source, dest, keep_source=True, node=node)


@app.command()
def delete(
    model_name: str = typer.Argument(..., help="Name of the model"),
    device: str = typer.Option(..., "--device", "-d", help="Device to delete from"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Delete model from specified device"""
    try:
        api_url = get_api_url(node)
        
        # Add node prefix if specified
        if node and ":" not in device:
            device = f"{node}:{device}"
        
        # This would be implemented as removing from locations
        console.print(f"Deleting {model_name} from {device}...")
        
        # For now, we'll use the model delete endpoint
        # In a full implementation, this would just remove from locations
        response = requests.delete(f"{api_url}/models/{model_name}")
        response.raise_for_status()
        
        console.print(f"[green]✓[/green] Model deleted from {device}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to delete model: {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    model_name: str = typer.Argument(..., help="Name of the model"),
    device: str = typer.Option(..., "--device", "-d", help="Device to serve from"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve on"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Start serving a model"""
    try:
        api_url = get_api_url(node)
        
        # Add node prefix if specified
        if node and ":" not in device:
            device = f"{node}:{device}"
        
        serve_data = {
            "model_name": model_name,
            "device": device,
            "port": port
        }
        
        console.print(f"Starting to serve {model_name} on {device}:{port}...")
        response = requests.post(f"{api_url}/serve", json=serve_data)
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to serve model: {e}")
        raise typer.Exit(1)


@app.command()
def stop(
    model_name: str = typer.Argument(..., help="Name of the model"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device to stop serving on"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Stop serving a model"""
    try:
        api_url = get_api_url(node)
        
        if device:
            # Add node prefix if specified
            if node and ":" not in device:
                device = f"{node}:{device}"
            
            console.print(f"Stopping {model_name} on {device}...")
            response = requests.post(f"{api_url}/stop/{model_name}/{device}")
        else:
            # Stop all instances
            console.print(f"Stopping all instances of {model_name}...")
            # This would need to be implemented in the API
            response = requests.post(f"{api_url}/stop/{model_name}")
        
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to stop model: {e}")
        raise typer.Exit(1)


@app.command()
def services():
    """List all running model services"""
    try:
        api_url = get_api_url()
        response = requests.get(f"{api_url}/models")
        response.raise_for_status()
        
        data = response.json()
        models = data.get("models", [])
        
        # Create a table for services
        table = Table(title="Running Model Services")
        table.add_column("Model", style="cyan")
        table.add_column("Device", style="green")
        table.add_column("URL", style="yellow")
        
        has_services = False
        for model in models:
            for device, url in model.get("services", {}).items():
                has_services = True
                table.add_row(model["name"], device, url)
        
        if has_services:
            console.print(table)
        else:
            console.print("No running services found", style="yellow")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list services: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    model_name: str = typer.Argument(..., help="Name of the model"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Get status of a model"""
    # This is an alias for info command
    info(model_name, node)


@app.command(name="service-health")
def service_health(
    model_name: str = typer.Argument(..., help="Name of the model"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Check health of model service"""
    try:
        api_url = get_api_url(node)
        response = requests.get(f"{api_url}/models/{model_name}")
        response.raise_for_status()
        
        model = response.json()
        services = model.get("services", {})
        
        if not services:
            console.print(f"No active services for {model_name}", style="yellow")
            return
        
        console.print(f"\n[bold cyan]Service Health for {model_name}:[/bold cyan]")
        
        for device, url in services.items():
            # Try to ping the service
            try:
                health_response = requests.get(f"{url}/health", timeout=5)
                if health_response.status_code == 200:
                    console.print(f"  [green]✓[/green] {device}: {url} - Healthy")
                else:
                    console.print(f"  [red]✗[/red] {device}: {url} - Unhealthy (status: {health_response.status_code})")
            except:
                console.print(f"  [red]✗[/red] {device}: {url} - Unreachable")
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to check service health: {e}")
        raise typer.Exit(1)


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main() 
"""
Model management CLI for gswarm unified system.
Provides commands for model download, deployment, and serving.
"""

from ..utils.connection_info import connection_manager

import typer
from typing import Optional, List
from pathlib import Path
import requests
import json
import os
from loguru import logger
from rich.console import Console
from rich.table import Table
import asyncio
import platform

console = Console()

# Create the model subcommand app
app = typer.Typer(help="Model management operations", rich_markup_mode="rich")


def handle_api_error(feature: str, response: requests.exceptions.RequestException):
    """Handle API errors and log them"""
    if hasattr(response, "response") and response.response is not None:
        try:
            error_message = response.response.json().get("detail", "Unknown error")
        except:
            error_message = response.response.text or "Unknown error"
        logger.error(f"{feature} failed: {error_message}")
    else:
        logger.error(f"{feature} API request failed: {response}")
    raise typer.Exit(1)


def detect_node_context() -> Optional[str]:
    """Detect current node context for client operations"""
    try:
        # Check if we're on a client by looking for connection info
        conn_info = connection_manager.load_connection()
        if conn_info and conn_info.node_id:
            return conn_info.node_id

        # Otherwise return hostname
        return platform.node()

    except Exception:
        # Fall back to hostname if detection fails
        return platform.node()


def get_api_url(node: Optional[str] = None) -> str:
    """Get the appropriate API URL based on connection info"""
    return connection_manager.get_model_api_url()


def find_model_location(api_url: str, model_name: str) -> Optional[str]:
    """Find where a model actually exists"""
    try:
        response = requests.get(f"{api_url}/models/{model_name}")
        response.raise_for_status()

        model = response.json()
        locations = model.get("locations", [])

        if locations:
            # Return the first available location
            return locations[0]
        return None

    except Exception:
        return None


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

        # ✅ Enhanced table with status column
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")  # ✅ Add status column
        table.add_column("Locations", style="blue", no_wrap=True)
        table.add_column("Services", style="magenta")

        for model in models:
            # Filter by location if specified
            if location and location not in model.get("locations", []):
                continue

            # ✅ Format status with progress for downloading models
            status = model.get("status", "unknown")
            if status == "downloading":
                progress = model.get("download_progress", {})
                progress_pct = progress.get("progress_percent", 0)
                status_display = f"downloading ({progress_pct}%)"
            elif status == "ready":
                status_display = "✅ ready"
            elif status == "error":
                status_display = "❌ error"
            else:
                status_display = status

            table.add_row(
                model["name"],
                model["type"],
                status_display,  # ✅ Show status with progress
                ", ".join(model.get("locations", [])),
                ", ".join(model.get("services", {}).keys()),
            )

        console.print(table)

    except requests.exceptions.RequestException as e:
        handle_api_error("List Models", e)
        raise typer.Exit(1)


@app.command()
def info(
    model_name: str = typer.Argument(..., help="Name of the model"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow download progress until completion"),
):
    """Get detailed information about a model"""
    try:
        api_url = get_api_url(node)

        if follow:
            # ✅ Follow progress until completion
            asyncio.run(monitor_download_progress(api_url, model_name))
        else:
            # ✅ Single status check with download progress
            response = requests.get(f"{api_url}/models/{model_name}")
            response.raise_for_status()

            model = response.json()

            console.print(f"\n[bold cyan]Model: {model['name']}[/bold cyan]")
            console.print(f"Type: {model['type']}")

            # ✅ Show status and download progress if downloading
            status = model.get("status", "unknown")
            console.print(f"Status: {status}")

            if status == "downloading":
                progress = model.get("download_progress", {})
                progress_pct = progress.get("progress_percent", 0)
                target_device = progress.get("target_device", "unknown")
                started_at = progress.get("started_at", "unknown")

                console.print(f"[yellow]📥 Downloading: {progress_pct}%[/yellow]")
                console.print(f"Target: {target_device}")
                console.print(f"Started: {started_at}")

                if progress_pct < 100:
                    console.print(f"[blue]ℹ[/blue] Use 'gswarm model status {model_name} --follow' to track progress")

            elif status == "ready":
                console.print(f"[green]✅ Ready[/green]")
                locations = model.get("locations", [])
                if locations:
                    console.print("[bold]Locations:[/bold]")
                    for loc in locations:
                        console.print(f"  - [blue]{loc}[/blue]")

            elif status == "error":
                progress = model.get("download_progress", {})
                error_msg = progress.get("error", "Unknown error")
                console.print(f"[red]❌ Error: {error_msg}[/red]")

            # Show services if any
            if model.get("services"):
                console.print("\n[bold]Active Services:[/bold]")
                for device, url in model["services"].items():
                    console.print(f"  - {device}: {url}")

            # Show metadata
            if model.get("metadata"):
                console.print("\n[bold]Metadata:[/bold]")
                console.print(json.dumps(model["metadata"], indent=2))

    except requests.exceptions.RequestException as e:
        handle_api_error("Get model information", e)
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

        request_data = {"name": model_name, "type": type, "metadata": json.loads(metadata) if metadata else {}}

        response = requests.post(f"{api_url}/models", json=request_data)
        response.raise_for_status()

        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)

    except requests.exceptions.RequestException as e:
        handle_api_error("Register a new model in the system", e)
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
            register_data = {"name": model_name, "type": type, "metadata": {"source": source, "url": url}}
            reg_response = requests.post(f"{api_url}/models", json=register_data)
            reg_response.raise_for_status()

        # Now download
        target_device = f"{node}:{device}" if node else device
        download_data = {"model_name": model_name, "source_url": url, "target_device": target_device}

        console.print(f"Starting download: {model_name} from {source} to {target_device}...")
        response = requests.post(f"{api_url}/download", json=download_data)
        response.raise_for_status()

        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
            console.print(f"[blue]ℹ[/blue] Use 'gswarm model status {model_name}' to check progress")
            console.print(f"[blue]ℹ[/blue] Use 'gswarm model status {model_name} --follow' to track progress")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)

    except requests.exceptions.RequestException as e:
        handle_api_error("Download model", e)
        raise typer.Exit(1)


async def monitor_download_progress(api_url: str, model_name: str):
    """Monitor and display download progress"""
    while True:
        try:
            response = requests.get(f"{api_url}/models/{model_name}/status")
            response.raise_for_status()
            status_data = response.json()

            status = status_data["status"]
            progress = status_data.get("download_progress", {})

            if status == "downloading":
                progress_pct = progress.get("progress_percent", 0)
                console.print(f"Downloading... {progress_pct}%")
                await asyncio.sleep(2)
            elif status == "ready":
                console.print("[green]✅ Download completed![/green]")
                break
            elif status == "error":
                error_msg = progress.get("error", "Unknown error")
                console.print(f"[red]❌ Download failed: {error_msg}[/red]")
                break
            else:
                await asyncio.sleep(1)

        except Exception as e:
            logger.warning(f"Error monitoring progress: {e}")
            await asyncio.sleep(2)


@app.command()
def move(
    model_name: str = typer.Argument(..., help="Name of the model"),
    source: str = typer.Option(..., "--from", help="Source device"),
    dest: str = typer.Option(..., "--to", help="Destination device"),
    keep_source: bool = typer.Option(False, "--keep-source", help="Keep model at source"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Move model between devices (disk, dram, gpu0, etc.)"""
    # if node is not specified, use the current node
    if node is None:
        node = detect_node_context()

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
            "keep_source": keep_source,
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
        handle_api_error("Move model", e)
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
        handle_api_error("Delete model", e)
        raise typer.Exit(1)


@app.command()
def serve(
    model_name: str = typer.Argument(..., help="Name of the model"),
    device: str = typer.Option(..., "--device", "-d", help="Device to serve from"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve on"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
    auto_move: bool = typer.Option(
        True, "--auto-move/--no-auto-move", help="Automatically move model to target device if not present"
    ),
):
    """Start serving a model"""
    try:
        api_url = get_api_url(node)

        # Auto-detect node context if not specified
        if not node:
            detected_node = detect_node_context()
            if detected_node:
                node = detected_node
                console.print(f"[blue]ℹ[/blue] Auto-detected node context: {node}")

        # Find where the model actually exists
        model_location = find_model_location(api_url, model_name)
        if not model_location:
            console.print(f"[red]✗[/red] Model {model_name} not found in any location")
            raise typer.Exit(1)

        # Construct target device with node prefix
        target_device = device
        if node and ":" not in device:
            target_device = f"{node}:{device}"

        # Check if model is already at target device
        if model_location == target_device:
            console.print(f"[green]✓[/green] Model {model_name} already at {target_device}")
        else:
            console.print(f"[yellow]⚠[/yellow] Model {model_name} found at {model_location}, target is {target_device}")

            if auto_move:
                # Try to copy/move model to target device
                console.print(f"[blue]ℹ[/blue] Copying model from {model_location} to {target_device}...")

                # Parse source location
                if ":" in model_location:
                    source_node, source_device = model_location.split(":", 1)
                else:
                    source_node, source_device = None, model_location

                logger.info(f"Source node: {source_node}, Source device: {source_device}")

                # Use the copy command logic
                try:
                    move_data = {
                        "model_name": model_name,
                        "source_device": model_location,
                        "target_device": target_device,
                        "keep_source": True,  # Copy, don't move
                    }

                    copy_response = requests.post(f"{api_url}/move", json=move_data)
                    copy_response.raise_for_status()

                    copy_result = copy_response.json()
                    if copy_result.get("success"):
                        console.print(f"[green]✓[/green] Model copied to {target_device}")
                    else:
                        console.print(f"[yellow]⚠[/yellow] Copy failed: {copy_result.get('message', 'Unknown error')}")
                        console.print(f"[blue]ℹ[/blue] Attempting to serve from original location: {model_location}")
                        target_device = model_location

                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Auto-copy failed: {e}")
                    console.print(f"[blue]ℹ[/blue] Attempting to serve from original location: {model_location}")
                    target_device = model_location
            else:
                console.print(f"[blue]ℹ[/blue] Auto-move disabled, serving from original location: {model_location}")
                target_device = model_location

        # Now try to serve
        serve_data = {"model_name": model_name, "device": target_device, "port": port}

        console.print(f"Starting to serve {model_name} on {target_device}:{port}...")
        response = requests.post(f"{api_url}/serve", json=serve_data)
        response.raise_for_status()

        result = response.json()
        if result.get("success"):
            console.print(f"[green]✓[/green] {result['message']}")
        else:
            console.print(f"[red]✗[/red] {result['message']}")
            raise typer.Exit(1)

    except requests.exceptions.RequestException as e:
        handle_api_error("Serve model", e)
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
        handle_api_error("Stop model service", e)
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
                table.add_row(f"[cyan]{model['name']}[/cyan]", f"[green]{device}[/green]", f"[yellow]{url}[/yellow]")

        if has_services:
            console.print(table)
        else:
            console.print("No running services found", style="yellow")

    except requests.exceptions.RequestException as e:
        handle_api_error("List running model services", e)
        raise typer.Exit(1)


@app.command()
def status(
    model_name: str = typer.Argument(..., help="Name of the model"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow download progress until completion"),
):
    """Get status of a model"""
    # ✅ Enhanced status is just an alias for enhanced info
    info(model_name, node, follow)


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
        handle_api_error("Check model service health", e)
        raise typer.Exit(1)


@app.command()
def scan(
    register: bool = typer.Option(True, "--register/--no-register", help="Register discovered models"),
    node: Optional[str] = typer.Option(None, "--node", "-n", help="Target node (for host commands)"),
):
    """Scan for locally cached HuggingFace models"""
    try:
        from gswarm.utils.cache import scan_huggingface_models

        console.print("Scanning HuggingFace cache for models...")
        discovered_models = scan_huggingface_models()

        if not discovered_models:
            console.print("No cached models found", style="yellow")
            return

        # Display discovered models
        table = Table(title="Discovered Cached Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Path", style="magenta")

        for model in discovered_models:
            size_gb = model["size"] / (1024**3)
            table.add_row(model["model_name"], model["model_type"], f"{size_gb:.2f} GB", model["local_path"])

        console.print(table)

        if register:
            console.print(f"\nRegistering {len(discovered_models)} models...")
            api_url = get_api_url(node)

            for model in discovered_models:
                try:
                    # Check if already registered
                    check_response = requests.get(f"{api_url}/models/{model['model_name']}")
                    if check_response.status_code == 200:
                        console.print(f"  [yellow]~[/yellow] {model['model_name']} - Already registered")
                        continue

                    # Register the model
                    register_data = {
                        "name": model["model_name"],
                        "type": model["model_type"],
                        "metadata": {
                            "local_path": model["local_path"],
                            "size": model["size"],
                            "source": "discovered_cache",
                            "auto_discovered": True,
                        },
                    }

                    response = requests.post(f"{api_url}/models", json=register_data)
                    response.raise_for_status()

                    console.print(f"  [green]✓[/green] {model['model_name']} - Registered")

                except Exception as e:
                    console.print(f"  [red]✗[/red] {model['model_name']} - Error: {e}")

    except Exception as e:
        console.print(f"[red]Error during scan: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def service_register(
    name: str = typer.Argument(..., help="Service name"),
    service_type: str = typer.Option(..., "--type", "-t", help="Service type (e.g., 'vllm', 'simple', or module path)"),
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Service port (auto-assigned if not specified)"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device to use (cpu/cuda/cuda:0/etc)"),
    max_batch_size: int = typer.Option(1, "--batch-size", help="Maximum batch size"),
    extra_args: Optional[str] = typer.Option(None, "--args", help="Extra arguments as JSON"),
):
    """Register a custom model service"""
    try:
        from .service_manager import get_service_manager, ServiceConfig
        
        # Parse extra args if provided
        extra_args_dict = {}
        if extra_args:
            import json
            extra_args_dict = json.loads(extra_args)
        
        # Create service config
        config = ServiceConfig(
            name=name,
            model_path=model_path,
            port=port,
            device=device,
            max_batch_size=max_batch_size,
            extra_args=extra_args_dict
        )
        
        # Register the service
        manager = get_service_manager()
        service_info = manager.register_service(name, service_type, config)
        
        console.print(f"[green]✓[/green] Service '{name}' registered on port {service_info.config.port}")
        
    except Exception as e:
        console.print(f"[red]Error registering service: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def service_start(
    name: str = typer.Argument(..., help="Service name to start"),
):
    """Start a registered service"""
    try:
        from .service_manager import get_service_manager
        
        manager = get_service_manager()
        
        # Start the service
        asyncio.run(manager.start_service(name))
        
        console.print(f"[green]✓[/green] Service '{name}' started")
        
    except Exception as e:
        console.print(f"[red]Error starting service: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def service_stop(
    name: str = typer.Argument(..., help="Service name to stop"),
    force: bool = typer.Option(False, "--force", "-f", help="Force stop the service"),
):
    """Stop a running service"""
    try:
        from .service_manager import get_service_manager
        
        manager = get_service_manager()
        
        # Stop the service
        stopped = asyncio.run(manager.stop_service(name, force=force))
        
        if stopped:
            console.print(f"[green]✓[/green] Service '{name}' stopped")
        else:
            console.print(f"[yellow]![/yellow] Service '{name}' was not running")
        
    except Exception as e:
        console.print(f"[red]Error stopping service: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def service_list():
    """List all registered services"""
    try:
        from .service_manager import get_service_manager
        
        manager = get_service_manager()
        services = manager.list_services()
        
        if not services:
            console.print("No services registered")
            return
        
        table = Table(title="Registered Services")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Port", style="blue")
        table.add_column("Device", style="magenta")
        table.add_column("PID", style="red")
        
        for service in services:
            table.add_row(
                service.name,
                service.service_type,
                service.status,
                str(service.config.port) if service.config.port else "N/A",
                service.config.device,
                str(service.pid) if service.pid else "N/A"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing services: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def service_status(
    name: str = typer.Argument(..., help="Service name"),
):
    """Get detailed status of a service"""
    try:
        from .service_manager import get_service_manager
        
        manager = get_service_manager()
        status = manager.get_service_status(name)
        
        console.print(f"\n[bold]Service: {name}[/bold]")
        console.print(f"Type: {status['service_type']}")
        console.print(f"Status: {status['status']}")
        
        if status['pid']:
            console.print(f"PID: {status['pid']}")
            if 'cpu_percent' in status:
                console.print(f"CPU: {status['cpu_percent']:.1f}%")
            if 'memory_info' in status:
                mem_mb = status['memory_info']['rss'] / (1024 * 1024)
                console.print(f"Memory: {mem_mb:.1f} MB")
        
        console.print(f"\nConfiguration:")
        config = status['config']
        console.print(f"  Model: {config['model_path']}")
        console.print(f"  Port: {config['port']}")
        console.print(f"  Device: {config['device']}")
        console.print(f"  Max Batch Size: {config['max_batch_size']}")
        
        if config.get('extra_args'):
            console.print(f"  Extra Args: {config['extra_args']}")
        
        if status.get('error_message'):
            console.print(f"\n[red]Error: {status['error_message']}[/red]")
        
    except Exception as e:
        console.print(f"[red]Error getting service status: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def service_unregister(
    name: str = typer.Argument(..., help="Service name to unregister"),
):
    """Unregister a service (must be stopped first)"""
    try:
        from .service_manager import get_service_manager
        
        manager = get_service_manager()
        manager.unregister_service(name)
        
        console.print(f"[green]✓[/green] Service '{name}' unregistered")
        
    except Exception as e:
        console.print(f"[red]Error unregistering service: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()

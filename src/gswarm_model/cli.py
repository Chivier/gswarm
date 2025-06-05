"""
CLI interface for gswarm_model system.
Now using simplified FastAPI-based implementation.
"""

import typer
from pathlib import Path
from typing import Optional
import subprocess
import sys
from loguru import logger

# Create the main app
app = typer.Typer(
    name="gsmodel",
    help="GSwarm Model Manager - Simplified FastAPI version",
    rich_markup_mode="rich"
)

@app.command()
def head(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8100, help="Port to bind to"),
):
    """Start the FastAPI head node"""
    logger.info(f"Starting FastAPI head node on {host}:{port}")
    try:
        # Run the FastAPI head using uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "gswarm_model.fastapi_head:app",
            "--host", host,
            "--port", str(port),
            "--reload"
        ])
    except KeyboardInterrupt:
        logger.info("Head node stopped")
    except Exception as e:
        logger.error(f"Failed to start head node: {e}")
        raise typer.Exit(1)

@app.command()
def client(
    action: str = typer.Argument(..., help="Action to perform: register, list-models, etc."),
    head_url: str = typer.Option("http://localhost:8100", "--head-url", "-h", help="Head node URL"),
    node_id: Optional[str] = typer.Option(None, "--node-id", "-n", help="Custom node ID (defaults to hostname)"),
):
    """Run client commands against the head node"""
    from .fastapi_client import ModelClient
    
    client = ModelClient(head_url, node_id=node_id)
    
    if action == "register":
        success = client.register_node()
        if success:
            logger.info("Node registered successfully")
        else:
            logger.error("Failed to register node")
            raise typer.Exit(1)
    
    elif action == "list-models":
        models = client.list_models()
        if models:
            for model in models:
                typer.echo(f"- {model['name']} ({model['type']})")
        else:
            typer.echo("No models found")
    
    else:
        logger.error(f"Unknown action: {action}")
        typer.echo(f"Available actions: register, list-models")
        raise typer.Exit(1)

# Keep the original main function for compatibility
def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main() 
"""Data pool CLI commands"""

import typer
from typing import Optional
from loguru import logger
import requests

app = typer.Typer(help="Data pool management operations")


def get_api_url(host: str = "localhost:9011") -> str:
    """Ensure host has http:// prefix"""
    if not host.startswith("http://") and not host.startswith("https://"):
        return f"http://{host}"
    return host


@app.command()
def list(
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Filter by device"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by data type"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """List data chunks in the pool"""
    try:
        url = f"{get_api_url(host)}/api/v1/data"
        params = {}
        if device:
            params["device"] = device
        if type:
            params["type"] = type

        response = requests.get(url, params=params)
        response.raise_for_status()

        chunks = response.json().get("chunks", [])

        if chunks:
            logger.info(f"Found {len(chunks)} data chunk(s):")
            for chunk in chunks:
                logger.info(f"\n  Chunk ID: {chunk['chunk_id']}")
                logger.info(f"    Type: {chunk['chunk_type']}")
                logger.info(f"    Size: {chunk['size'] / 1e6:.2f} MB")
                logger.info(f"    Format: {chunk.get('format', 'unknown')}")
                if chunk.get("locations"):
                    logger.info(f"    Locations: {', '.join([loc['device'] for loc in chunk['locations']])}")
                if chunk.get("metadata"):
                    logger.info(f"    Created: {chunk['metadata'].get('created_at', 'unknown')}")
                    logger.info(f"    Access count: {chunk['metadata'].get('access_count', 0)}")
        else:
            logger.info("No data chunks found")
    except Exception as e:
        logger.error(f"Failed to list data chunks: {e}")


@app.command()
def create(
    source: str = typer.Option(..., "--source", "-s", help="Data source (URL or path)"),
    device: str = typer.Option("dram", "--device", "-d", help="Target device"),
    type: str = typer.Option("input", "--type", "-t", help="Data type (input/output/intermediate)"),
    format: str = typer.Option("tensor", "--format", "-f", help="Data format"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Create a new data chunk"""
    try:
        url = f"{get_api_url(host)}/api/v1/data"
        data = {
            "source": source,
            "device": device,
            "type": type,
            "format": format,
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Data chunk created successfully")
        if result.get("chunk_id"):
            logger.info(f"  Chunk ID: {result['chunk_id']}")
            logger.info(f"  Device: {device}")
            logger.info(f"  Size: {result.get('size', 0) / 1e6:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to create data chunk: {e}")


@app.command()
def info(
    chunk_id: str = typer.Argument(..., help="Data chunk ID"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Get data chunk information"""
    try:
        url = f"{get_api_url(host)}/api/v1/data/{chunk_id}"
        response = requests.get(url)
        response.raise_for_status()

        chunk = response.json()
        logger.info(f"Chunk ID: {chunk['chunk_id']}")
        logger.info(f"  Type: {chunk['chunk_type']}")
        logger.info(f"  Size: {chunk['size'] / 1e6:.2f} MB")
        logger.info(f"  Format: {chunk.get('format', 'unknown')}")

        if chunk.get("locations"):
            logger.info("  Locations:")
            for loc in chunk["locations"]:
                logger.info(f"    - {loc['device']} ({loc['status']})")

        if chunk.get("metadata"):
            meta = chunk["metadata"]
            logger.info("  Metadata:")
            logger.info(f"    Created by: {meta.get('created_by', 'unknown')}")
            logger.info(f"    Created at: {meta.get('created_at', 'unknown')}")
            logger.info(f"    Last accessed: {meta.get('last_accessed', 'never')}")
            logger.info(f"    Access count: {meta.get('access_count', 0)}")
            logger.info(f"    Checksum: {meta.get('checksum', 'none')}")

        if chunk.get("references"):
            logger.info(f"  Referenced by: {', '.join(chunk['references'])}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Data chunk '{chunk_id}' not found")
        else:
            logger.error(f"Failed to get chunk info: {e}")
    except Exception as e:
        logger.error(f"Failed to get chunk info: {e}")


@app.command()
def move(
    chunk_id: str = typer.Argument(..., help="Data chunk ID"),
    target: str = typer.Option(..., "--to", "-t", help="Target device"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Priority"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Move data chunk between devices"""
    try:
        url = f"{get_api_url(host)}/api/v1/data/{chunk_id}/move"
        data = {
            "target_device": target,
            "priority": priority,
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Data move operation started")
        logger.info(f"  Chunk ID: {chunk_id}")
        logger.info(f"  Target: {target}")
        if result.get("task_id"):
            logger.info(f"  Task ID: {result['task_id']}")
    except Exception as e:
        logger.error(f"Failed to move data chunk: {e}")


@app.command()
def transfer(
    chunk_id: str = typer.Argument(..., help="Data chunk ID"),
    target: str = typer.Option(..., "--to", "-t", help="Target node:device"),
    delete_source: bool = typer.Option(False, "--delete-source", help="Delete source after transfer"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Transfer data chunk to another node"""
    try:
        # Parse target node and device
        if ":" not in target:
            logger.error("Target must be in format 'node:device'")
            return

        target_node, target_device = target.split(":", 1)

        url = f"{get_api_url(host)}/api/v1/data/{chunk_id}/transfer"
        data = {
            "target_node": target_node,
            "target_device": target_device,
            "delete_source": delete_source,
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Data transfer started")
        logger.info(f"  Chunk ID: {chunk_id}")
        logger.info(f"  Target: {target}")
        if result.get("task_id"):
            logger.info(f"  Task ID: {result['task_id']}")
    except Exception as e:
        logger.error(f"Failed to transfer data chunk: {e}")


@app.command()
def delete(
    chunk_id: str = typer.Argument(..., help="Data chunk ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion even if referenced"),
    host: str = typer.Option("localhost:9011", "--host", help="Client API address"),
):
    """Delete data chunk from pool"""
    try:
        url = f"{get_api_url(host)}/api/v1/data/{chunk_id}"
        params = {}
        if force:
            params["force"] = "true"

        response = requests.delete(url, params=params)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Data chunk '{chunk_id}' deleted successfully")
        if result.get("message"):
            logger.info(f"  {result['message']}")
    except Exception as e:
        logger.error(f"Failed to delete data chunk: {e}")

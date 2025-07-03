"""Data pool CLI commands"""

import typer
from typing import Optional
from loguru import logger
import requests

app = typer.Typer(help="Data pool and KV storage management operations")


def get_api_url(host: str = "localhost:9015") -> str:
    """Ensure host has http:// prefix"""
    if not host.startswith("http://") and not host.startswith("https://"):
        return f"http://{host}"
    return host


def get_kv_api_url(host: str = "localhost:9015") -> str:
    """Ensure KV host has http:// prefix"""
    if not host.startswith("http://") and not host.startswith("https://"):
        return f"http://{host}"
    return host


# KV Storage Commands - now directly under main app
@app.command()
def start(
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    port: int = typer.Option(9015, "--port", help="Server port"),
    max_memory: str = typer.Option("16GB", "--max-memory", help="Maximum memory size (e.g., 16GB, 1TB)"),
):
    """Start KV storage server"""
    from .pool import start_server

    # Parse memory size
    max_mem_bytes = parse_memory_size(max_memory)

    logger.info(f"Starting KV storage server on {host}:{port}")
    logger.info(f"Maximum memory: {max_memory}")

    start_server(host=host, port=port, max_mem_size=max_mem_bytes)


@app.command()
def write(
    key: str = typer.Argument(..., help="Key to write"),
    value: str = typer.Argument(..., help="Value to write"),
    location: str = typer.Option("dram", "--location", "-l", help="Storage location (dram/pinned_dram/device:X/disk)"),
    host: str = typer.Option("localhost:9015", "--host", help="KV server address"),
):
    """Write key-value pair to specified storage location"""
    try:
        url = f"{get_kv_api_url(host)}/write"
        data = {"key": key, "value": value, "location": location}

        response = requests.post(url, json=data)
        response.raise_for_status()

        logger.info(f"Successfully wrote key '{key}' to {location}")
    except Exception as e:
        logger.error(f"Failed to write key '{key}': {e}")


@app.command()
def read(
    key: str = typer.Argument(..., help="Key to read"),
    host: str = typer.Option("localhost:9015", "--host", help="KV server address"),
):
    """Read value by key"""
    try:
        url = f"{get_kv_api_url(host)}/read/{key}"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        if data["found"]:
            logger.info(f"Key '{key}': {data['value']}")
        else:
            logger.info(f"Key '{key}' not found")
    except Exception as e:
        logger.error(f"Failed to read key '{key}': {e}")


@app.command()
def release(
    key: str = typer.Argument(..., help="Key to release"),
    host: str = typer.Option("localhost:9015", "--host", help="KV server address"),
):
    """Remove key from storage"""
    try:
        url = f"{get_kv_api_url(host)}/release/{key}"
        response = requests.delete(url)
        response.raise_for_status()

        logger.info(f"Successfully released key '{key}'")
    except Exception as e:
        logger.error(f"Failed to release key '{key}': {e}")


@app.command()
def send(
    key: str = typer.Argument(..., help="Key to send"),
    target: str = typer.Argument(..., help="Target URL (e.g., localhost:9016)"),
    host: str = typer.Option("localhost:9015", "--host", help="Source KV server address"),
):
    """Send key to another KV storage server"""
    try:
        url = f"{get_kv_api_url(host)}/send"
        data = {"key": key, "url": target}

        response = requests.post(url, json=data)
        response.raise_for_status()

        logger.info(f"Successfully sent key '{key}' to {target}")
    except Exception as e:
        logger.error(f"Failed to send key '{key}' to {target}: {e}")


@app.command()
def stats(
    host: str = typer.Option("localhost:9015", "--host", help="KV server address"),
):
    """Show storage statistics"""
    try:
        url = f"{get_kv_api_url(host)}/stats"
        response = requests.get(url)
        response.raise_for_status()

        stats = response.json()["stats"]

        logger.info("KV Storage Statistics:")
        logger.info(
            f"  Memory Usage: {stats['current_size'] / (1024**3):.2f} GB / {stats['max_size'] / (1024**3):.2f} GB ({stats['usage_percent']:.1f}%)"
        )
        logger.info(
            f"  Keys: {stats['total_keys']} total ({stats['persistent_keys']} persistent, {stats['volatile_keys']} volatile)"
        )

        mem_info = stats.get("memory_info", {})
        if mem_info:
            logger.info(f"  System Memory: {mem_info.get('percent', 0):.1f}% used")
            logger.info(f"    Available: {mem_info.get('available', 0) / (1024**3):.2f} GB")
            logger.info(f"    Total: {mem_info.get('total', 0) / (1024**3):.2f} GB")

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")


def parse_memory_size(size_str: str) -> int:
    """Parse memory size string to bytes"""
    size_str = size_str.upper().strip()

    # Check suffixes in order from longest to shortest to avoid partial matches
    suffixes = [
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("B", 1),
    ]

    for suffix, multiplier in suffixes:
        if size_str.endswith(suffix):
            number_str = size_str[: -len(suffix)].strip()
            if number_str:  # Make sure we have a number
                try:
                    number = float(number_str)
                    return int(number * multiplier)
                except ValueError:
                    continue

    # If no suffix matched, assume bytes
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid memory size format: {size_str}")


# Data Pool Commands
@app.command()
def list(
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Filter by device"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by data type"),
    host: str = typer.Option("localhost:9015", "--host", help="Client API address"),
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
    host: str = typer.Option("localhost:9015", "--host", help="Client API address"),
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
    host: str = typer.Option("localhost:9015", "--host", help="Client API address"),
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
    host: str = typer.Option("localhost:9015", "--host", help="Client API address"),
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
    host: str = typer.Option("localhost:9015", "--host", help="Client API address"),
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
    host: str = typer.Option("localhost:9015", "--host", help="Client API address"),
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


@app.command()
def move_data(
    key: str = typer.Argument(..., help="Key to move"),
    destination: str = typer.Argument(..., help="Destination location (dram/pinned_dram/device:X/disk)"),
    host: str = typer.Option("localhost:9015", "--host", help="KV server address"),
):
    """Move data between storage locations"""
    try:
        url = f"{get_kv_api_url(host)}/move"
        data = {"key": key, "destination": destination}

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Successfully initiated move of key '{key}'")
        logger.info(f"  From: {result.get('source', 'unknown')}")
        logger.info(f"  To: {result.get('destination')}")
        if result.get('read_pointer'):
            logger.info(f"  Read pointer: {result['read_pointer']}")
    except Exception as e:
        logger.error(f"Failed to move key '{key}': {e}")


@app.command()
def get_location(
    key: str = typer.Argument(..., help="Key to query"),
    host: str = typer.Option("localhost:9015", "--host", help="KV server address"),
):
    """Get location of data"""
    try:
        url = f"{get_kv_api_url(host)}/location/{key}"
        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Key '{key}' location info:")
        logger.info(f"  Primary location: {result['location']}")
        
        if result.get('locations'):
            logger.info("  All locations:")
            for loc in result['locations']:
                status = f" ({loc['copy_status']})" if loc['copy_status'] != 'complete' else ""
                pointer = f" [ptr: {loc['read_pointer']}]" if loc.get('read_pointer') else ""
                logger.info(f"    - {loc['location']}: {loc['size']} bytes{status}{pointer}")
    except Exception as e:
        logger.error(f"Failed to get location for key '{key}': {e}")


@app.command()
def list_locations(
    host: str = typer.Option("localhost:9015", "--host", help="KV server address"),
):
    """List locations of all data in the system"""
    try:
        url = f"{get_kv_api_url(host)}/locations"
        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        locations = result.get('locations', {})
        
        if locations:
            logger.info("Data locations in the system:")
            for key, loc_list in locations.items():
                logger.info(f"\n  Key: {key}")
                for loc in loc_list:
                    status = f" ({loc['copy_status']})" if loc['copy_status'] != 'complete' else ""
                    logger.info(f"    - {loc['location']}: {loc['size']} bytes{status}")
        else:
            logger.info("No data found in the system")
    except Exception as e:
        logger.error(f"Failed to list locations: {e}")

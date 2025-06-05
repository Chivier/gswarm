"""Client node CLI commands"""

import typer
from typing import Optional
from loguru import logger

app = typer.Typer(help="Client node management commands")

@app.command()
def connect(
    host_address: str = typer.Argument(..., help="Host node address (e.g., master:8090)"),
    resilient: bool = typer.Option(False, "--resilient", "-r", help="Enable resilient mode with auto-reconnect"),
    enable_bandwidth: bool = typer.Option(None, "--enable-bandwidth", help="Enable bandwidth profiling (can be overridden by host)"),
    node_id: Optional[str] = typer.Option(None, "--node-id", "-n", help="Custom node ID"),
):
    """Connect this node as a client to the host"""
    logger.info(f"Connecting to host at {host_address}")
    logger.info(f"  Resilient mode: {'enabled' if resilient else 'disabled'}")
    if enable_bandwidth is not None:
        logger.info(f"  Bandwidth profiling: {'enabled' if enable_bandwidth else 'disabled'}")
    if node_id:
        logger.info(f"  Node ID: {node_id}")
    
    logger.info("  Sampling configuration will be read from host")
    
    # Parse host address
    if ':' in host_address:
        host, port = host_address.split(':')
        port = int(port)
    else:
        host = host_address
        port = 8090
    
    # Start both profiler client and model client
    from ..profiler.client import start_client_node_sync
    from ..profiler.client_resilient import start_resilient_client
    from ..model.fastapi_client import ModelClient
    
    # Start model client registration
    model_host_port = port + 920  # Default offset from profiler to model port
    model_client = ModelClient(f"http://{host}:{model_host_port}", node_id=node_id)
    if not model_client.register_node():
        logger.warning("Failed to register with model service")
    
    # Start profiler client
    if resilient:
        start_resilient_client(host_address, enable_bandwidth)
    else:
        start_client_node_sync(host_address, enable_bandwidth)

@app.command()
def disconnect():
    """Disconnect from the host"""
    logger.info("Disconnecting from host...")
    # TODO: Implement graceful disconnect
    logger.info("Disconnect not yet implemented")

@app.command()
def status():
    """Get client node status"""
    logger.info("Getting client status...")
    # TODO: Check connection status
    logger.info("Client status not yet implemented") 
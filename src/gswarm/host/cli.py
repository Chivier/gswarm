"""Host node CLI commands"""

import typer
from typing import Optional
from loguru import logger
import asyncio

app = typer.Typer(help="Host node management commands")

@app.command()
def start(
    port: int = typer.Option(8090, "--port", "-p", help="gRPC port for profiler"),
    http_port: int = typer.Option(8091, "--http-port", help="HTTP API port"),
    model_port: int = typer.Option(9010, "--model-port", help="Model management API port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address to bind to"),
    enable_bandwidth: bool = typer.Option(False, "--enable-bandwidth", help="Enable bandwidth profiling"),
    enable_nvlink: bool = typer.Option(False, "--enable-nvlink", help="Enable NVLink profiling"),
):
    """Start the host node with all services"""
    logger.info(f"Starting host node on {host}")
    logger.info(f"  Profiler gRPC port: {port}")
    logger.info(f"  HTTP API port: {http_port}")
    logger.info(f"  Model API port: {model_port}")
    logger.info(f"  Bandwidth profiling: {'enabled' if enable_bandwidth else 'disabled'}")
    logger.info(f"  NVLink profiling: {'enabled' if enable_nvlink else 'disabled'}")
    logger.info(f"  Using adaptive sampling strategy (similar to WandB)")
    
    # Start both profiler and model services
    from ..profiler.head import run_head_node as run_profiler_head
    from ..model.fastapi_head import create_app as create_model_app
    
    async def run_all_services():
        # Start profiler in background
        # Start model service
        import uvicorn
        model_app = create_model_app()

        async with asyncio.TaskGroup() as tg:
            profiler_task = tg.create_task(
                asyncio.to_thread(
                    run_profiler_head, 
                    host, port, enable_bandwidth, enable_nvlink, http_port
                )
            )
            model_server_task = tg.create_task(
                uvicorn.Server(
                    uvicorn.Config(model_app, host=host, port=model_port)
                ).serve()
            )
        
        
        
        
    
    try:
        asyncio.run(run_all_services())
    except KeyboardInterrupt:
        logger.info("Host node stopped")

@app.command()
def stop():
    """Stop the host node"""
    logger.info("Stopping host node...")
    # TODO: Implement graceful shutdown
    logger.info("Host stop not yet implemented")

@app.command()
def status():
    """Get host node status"""
    logger.info("Getting host status...")
    # TODO: Check status of all services
    logger.info("Host status not yet implemented") 
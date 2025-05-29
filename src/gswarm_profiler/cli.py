from datetime import datetime
import os
import typer
from typing_extensions import Annotated
from loguru import logger
import sys
import asyncio
import grpc
from pathlib import Path
import requests

# Configure Loguru
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add back with specific level, can be configured

app = typer.Typer(
    name="gswarm-profiler",
    help="Multi-node multi-GPU profiler using nvitop.",
    epilog="For more information, visit: https://github.com/your-repo/gswarm-profiler",
)

def ensure_grpc_files():
    """Check if gRPC files exist and generate them if needed"""
    current_dir = Path(__file__).parent
    pb2_file = current_dir / "profiler_pb2.py"
    pb2_grpc_file = current_dir / "profiler_pb2_grpc.py"
    
    if not pb2_file.exists() or not pb2_grpc_file.exists():
        logger.info("gRPC protobuf files not found, generating them...")
        try:
            from .generate_grpc import generate_grpc_files
            generate_grpc_files()
            logger.info("gRPC protobuf files generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate gRPC files: {e}")
            logger.error("Please run 'gsprof generate-grpc' manually")
            raise

# Placeholder for head module if we need to access its state (e.g. enable_bandwidth)
# This gets tricky with separate processes. The head node's enable_bandwidth setting is key.
# The client-side enable_bandwidth should ideally match the head's setting for efficiency.

# Global state for sharing 'enable_bandwidth' between head 'start' and client 'connect' if run in same context
# This is not robust for separate CLI calls. Client and Head must agree on this.
# The Head's `--enable_bandwidth` flag on its `start` command is the source of truth
# for what data the head expects and processes.
# The Client's `connect` command should also have an `--enable-bandwidth` if we want it to conditionally send data.
# For now, let's assume client always sends if head enables it.
# This is hard to coordinate without IPC or head telling client.
# Easiest: head's --enable_bandwidth flag determines processing AND client is TOLD to send it.
# The head.py sets a global `state.enable_bandwidth_profiling`.
# The client then queries that. This is also not clean...

# Best approach:
# Head's `start` command has `--enable-bandwidth`. This sets a state in the running head server.
# Client's `connect` command *also* has an optional `--enable-bandwidth`.
# If client starts with it, it sends bandwidth data. If head is not expecting it, head might ignore.
# If client does not start with it, it won't send. If head expects it, it gets nothing for bandwidth.
# For simplicity:
# 1. Head's `start --enable-bandwidth` means head will *process* bandwidth data.
# 2. Client's `connect --enable-bandwidth` means client will *collect and send* bandwidth data.
# They should ideally be consistent.


@app.command(
    name="start",
    help="Starts the head node (data collector server).",
    epilog="Example: gswarm-profiler start --host 0.0.0.0 --port 8090 --freq 500 --enable-bandwidth --http-port 8080",
)
def start_head_node(
    host: Annotated[str, typer.Option(help="Host address for the head node.")] = "localhost",
    port: Annotated[int, typer.Option(help="Port for the head node.")] = 8090,
    freq: Annotated[int, typer.Option(help="Sampling frequency in milliseconds.")] = 500,
    enable_bandwidth: Annotated[
        bool,
        typer.Option(
            "--enable-bandwidth/--disable-bandwidth",
            help="Enable GPU-DRAM and GPU-GPU bandwidth profiling.",
        ),
    ] = False,
    enable_nvlink: Annotated[
        bool,
        typer.Option(
            "--enable-nvlink/--disable-nvlink",
            help="Enable NVLink bandwidth profiling.",
        ),
    ] = False,
    background: Annotated[bool, typer.Option("--background", help="Run head node in background mode.")] = False,
    http_port: Annotated[int, typer.Option(help="Port for HTTP API server. If not specified, HTTP API is disabled.")] = None,
):
    """
    Starts the head node server.
    Example: gswarm-profiler start --port 8090 --freq 500 --enable-bandwidth --http-port 8080
    """
    
    # Ensure gRPC files exist before starting
    ensure_grpc_files()

    if background:
        import subprocess
        
        cmd = [
            "gsprof", "start",
            "--host", host,
            "--port", str(port),
            "--freq", str(freq),
        ]
        if enable_bandwidth:
            cmd.append("--enable-bandwidth")
        if enable_nvlink:
            cmd.append("--enable-nvlink")
        if http_port:
            cmd.extend(["--http-port", str(http_port)])
            
        subprocess.Popen(cmd)
        logger.info("Head node started in background")
        if http_port:
            logger.info(f"HTTP API will be available at http://{host}:{http_port}")
        return

    from .head import run_head_node

    logger.info(f"Head node enable_bandwidth set to: {enable_bandwidth}")
    logger.info(f"Sampling frequency set to: {freq}ms")
    if http_port:
        logger.info(f"HTTP API enabled on port {http_port}")
    run_head_node(host, port, enable_bandwidth, enable_nvlink, freq, http_port)


@app.command(
    name="connect",
    help="Connects a client node to the head node.",
    epilog="Example: gswarm-profiler connect localhost:8090",
)
def connect_client_node(
    head_address: Annotated[str, typer.Argument(help="Address of the head node (e.g., localhost:8090).")],
    enable_bandwidth: Annotated[
        bool,
        typer.Option(
            "--enable-bandwidth/--disable-bandwidth",
            help="Enable bandwidth data collection on client side.",
        ),
    ] = None,
):
    """
    Connects this node as a client to the head server.
    Example: gswarm-profiler connect localhost:8090 --enable-bandwidth
    """
    
    # Ensure gRPC files exist before connecting
    ensure_grpc_files()
    
    async def get_head_status():
        try:
            # Import gRPC modules
            from . import profiler_pb2
            from . import profiler_pb2_grpc
                
            async with grpc.aio.insecure_channel(head_address) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                status_response = await stub.GetStatus(profiler_pb2.Empty())
                return status_response.freq, status_response.enable_bandwidth_profiling
        except Exception as e:
            logger.error(f"Failed to get head node status: {e}")
            logger.info("Using default settings: freq=500ms, enable_bandwidth=False")
            return 500, False
    
    # Get head node configuration
    freq, head_enable_bandwidth = asyncio.run(get_head_status())
    
    # Use client's bandwidth setting if specified, otherwise match head node
    if enable_bandwidth is None:
        enable_bandwidth = head_enable_bandwidth
        
    logger.info(f"Client node enable_bandwidth set to: {enable_bandwidth}")
    logger.info(f"Frequency: {freq}ms")
    
    if enable_bandwidth:
        logger.info("Client will attempt to collect and send bandwidth metrics.")
        if not head_enable_bandwidth:
            logger.warning("Head node has bandwidth profiling disabled - bandwidth data may be ignored.")

    from .client import start_client_node_sync

    start_client_node_sync(head_address, freq, enable_bandwidth)


@app.command(
    name="profile",
    help="Start profiling session on head node.",
    epilog="Example: gswarm-profiler profile localhost:8090 --name my_experiment",
)
def start_profiling(
    head_address: Annotated[str, typer.Argument(help="Address of the head node (e.g., localhost:8090).")] = "localhost:8090",
    name: Annotated[str, typer.Option(help="Name for the profiling session output file.")] = "profiling_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
):
    """
    Start a profiling session on the head node.
    If name is not provided, it will be set to "profiling_" + current timestamp.
    If head_address is not provided, it will be set to "localhost:8090".
    Example: gswarm-profiler profile localhost:8090 --name my_experiment
    """
    
    # Ensure gRPC files exist
    ensure_grpc_files()
    
    async def start_profiling_async():
        try:
            # Import gRPC modules
            from . import profiler_pb2
            from . import profiler_pb2_grpc
                
            async with grpc.aio.insecure_channel(head_address) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                request = profiler_pb2.StartProfilingRequest(name=name)
                response = await stub.StartProfiling(request)
                
                if response.success:
                    logger.info(f"Profiling started: {response.message}")
                    logger.info(f"Output file: {response.output_file}")
                else:
                    logger.error(f"Failed to start profiling: {response.message}")
                    
        except Exception as e:
            logger.error(f"Failed to start profiling: {e}")
    
    asyncio.run(start_profiling_async())


@app.command(
    name="stop",
    help="Stop profiling session on head node.",
    epilog="Example: gswarm-profiler stop localhost:8090 --name session1",
)
def stop_profiling(
    head_address: Annotated[str, typer.Argument(help="Address of the head node (e.g., localhost:8090).")] = "localhost:8090",
    name: Annotated[str, typer.Option(help="Name of specific session to stop. If not provided, stops all sessions.")] = None,
):
    """
    Stop the profiling session on the head node.
    If head_address is not provided, it will be set to "localhost:8090".
    Example: gswarm-profiler stop localhost:8090 --name session1
    """
    
    # Ensure gRPC files exist
    ensure_grpc_files()
    
    async def stop_profiling_async():
        try:
            # Import gRPC modules
            from . import profiler_pb2
            from . import profiler_pb2_grpc
                
            async with grpc.aio.insecure_channel(head_address) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                request = profiler_pb2.StopProfilingRequest()
                if name:
                    request.name = name
                response = await stub.StopProfiling(request)
                
                if response.success:
                    logger.info(f"Profiling stopped: {response.message}")
                else:
                    logger.error(f"Failed to stop profiling: {response.message}")
                    
        except Exception as e:
            logger.error(f"Failed to stop profiling: {e}")
    
    asyncio.run(stop_profiling_async())


@app.command(
    name="status",
    help="Get status of the head node.",
    epilog="Example: gswarm-profiler status localhost:8090",
)
def get_status(
    head_address: Annotated[str, typer.Argument(help="Address of the head node (e.g., localhost:8090).")] = "localhost:8090",
):
    """
    Get the status of the head node.
    Example: gswarm-profiler status localhost:8090   
    """
    
    # Ensure gRPC files exist
    ensure_grpc_files()
    
    async def get_status_async():
        try:
            # Import gRPC modules
            from . import profiler_pb2
            from . import profiler_pb2_grpc
                
            async with grpc.aio.insecure_channel(head_address) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                response = await stub.GetStatus(profiler_pb2.Empty())
                
                logger.info("Head Node Status:")
                logger.info(f"  Frequency: {response.freq}ms")
                logger.info(f"  Bandwidth Profiling: {'Enabled' if response.enable_bandwidth_profiling else 'Disabled'}")
                logger.info(f"  NVLink Profiling: {'Enabled' if response.enable_nvlink_profiling else 'Disabled'}")
                logger.info(f"  Is Profiling: {'Yes' if response.is_profiling else 'No'}")
                logger.info(f"  Output Filename: {response.output_filename}")
                logger.info(f"  Frame Counter: {response.frame_id_counter}")
                logger.info(f"  Connected Clients: {len(response.connected_clients)}")
                
                if response.connected_clients:
                    logger.info("  Client List:")
                    for client in response.connected_clients:
                        logger.info(f"    - {client}")
                        
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
    
    asyncio.run(get_status_async())


@app.command(
    name="exit",
    help="Exits the head node.",
    epilog="Example: gswarm-profiler exit localhost:8090",
)
def exit_head_node(
    head_address: Annotated[str, typer.Argument(help="Address of the head node (e.g., localhost:8090).")],
):
    """
    Exits the head node.
    Example: gswarm-profiler exit localhost:8090
    """
    
    # Ensure gRPC files exist
    ensure_grpc_files()
    
    async def exit_head_async():
        try:
            # Import gRPC modules
            from . import profiler_pb2
            from . import profiler_pb2_grpc
                
            async with grpc.aio.insecure_channel(head_address) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                await stub.Exit(profiler_pb2.Empty())
                logger.info("Head node exit signal sent.")
                
        except Exception as e:
            logger.error(f"Failed to exit head node: {e}")
    
    asyncio.run(exit_head_async())


@app.command(name="stat", help="Show the stat of the profiler.")
def show_stat(
    data: Annotated[str, typer.Option(help="Path to the data directory.")] = "result.json",
    plot: Annotated[str, typer.Option(help="Path to the plot directory.")] = "",
):
    """Show the stat of the profiler."""
    from .stat import show_stat

    # if plot is not empty, use data filename as the plot filename
    if plot == "":
        plot = os.path.splitext(data)[0] + ".pdf"

    show_stat(data, plot)


@app.command(name="generate-grpc", help="Generate gRPC protobuf files.")
def generate_grpc():
    """Generate gRPC protobuf files from .proto definition."""
    from .generate_grpc import generate_grpc_files
    generate_grpc_files()


@app.command(name="help", help="Show detailed help information.")
def show_help(
    command: Annotated[str, typer.Argument(help="Show help for specific command")] = None,
):
    """Show help for the application or specific commands."""
    if command:
        if command == "start":
            typer.echo("Detailed help for 'start' command...")
        elif command == "connect":
            typer.echo("Detailed help for 'connect' command...")
        else:
            typer.echo(f"Unknown command: {command}")
    else:
        typer.echo("Available commands: start, connect, profile, stop, status, exit, stat")
        typer.echo("Use --help with any command for detailed information")


@app.command(
    name="http-profile",
    help="Start profiling session via HTTP API.",
    epilog="Example: gswarm-profiler http-profile localhost:8080 --name my_experiment",
)
def start_profiling_http(
    http_address: Annotated[str, typer.Argument(help="Address of the HTTP API (e.g., localhost:8080).")] = "localhost:8080",
    name: Annotated[str, typer.Option(help="Name for the profiling session output file.")] = None,
):
    """
    Start a profiling session via HTTP API.
    Example: gswarm-profiler http-profile localhost:8080 --name my_experiment
    """
    
    if not name:
        name = "profiling_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        url = f"http://{http_address}/profiling/start"
        response = requests.post(url, json={"name": name})
        response.raise_for_status()
        
        data = response.json()
        if data["success"]:
            logger.info(f"Profiling started: {data['message']}")
            logger.info(f"Output file: {data['output_file']}")
        else:
            logger.error(f"Failed to start profiling: {data['message']}")
            
    except Exception as e:
        logger.error(f"Failed to start profiling via HTTP: {e}")


@app.command(
    name="http-stop",
    help="Stop profiling session via HTTP API.",
    epilog="Example: gswarm-profiler http-stop localhost:8091 --name session1",
)
def stop_profiling_http(
    http_address: Annotated[str, typer.Argument(help="Address of the HTTP API (e.g., localhost:8091).")] = "localhost:8091",
    name: Annotated[str, typer.Option(help="Name of specific session to stop. If not provided, stops all sessions.")] = None,
):
    """
    Stop the profiling session via HTTP API.
    Example: gswarm-profiler http-stop localhost:8091 --name session1
    """
    try:
        url = f"http://{http_address}/profiling/stop"
        data = {}
        if name:
            data["name"] = name
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        data = response.json()
        if data["success"]:
            logger.info(f"Profiling stopped: {data['message']}")
        else:
            logger.error(f"Failed to stop profiling: {data['message']}")
            
    except Exception as e:
        logger.error(f"Failed to stop profiling via HTTP: {e}")


@app.command(
    name="http-status",
    help="Get status via HTTP API.",
    epilog="Example: gswarm-profiler http-status localhost:8080",
)
def get_status_http(
    http_address: Annotated[str, typer.Argument(help="Address of the HTTP API (e.g., localhost:8080).")] = "localhost:8080",
):
    """
    Get the status via HTTP API.
    Example: gswarm-profiler http-status localhost:8080
    """
    try:
        url = f"http://{http_address}/status"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        logger.info("Head Node Status (via HTTP):")
        logger.info(f"  Frequency: {data['freq']}ms")
        logger.info(f"  Bandwidth Profiling: {'Enabled' if data['enable_bandwidth_profiling'] else 'Disabled'}")
        logger.info(f"  NVLink Profiling: {'Enabled' if data['enable_nvlink_profiling'] else 'Disabled'}")
        logger.info(f"  Is Profiling: {'Yes' if data['is_profiling'] else 'No'}")
        logger.info(f"  Output Filename: {data['output_filename']}")
        logger.info(f"  Frame Counter: {data['frame_id_counter']}")
        logger.info(f"  Total GPUs: {data['total_gpus']}")
        logger.info(f"  Connected Clients: {len(data['connected_clients'])}")
        
        if data['connected_clients']:
            logger.info("  Client List:")
            for client in data['connected_clients']:
                logger.info(f"    - {client}")
                
    except Exception as e:
        logger.error(f"Failed to get status via HTTP: {e}")


if __name__ == "__main__":
    app()

import typer
from typing_extensions import Annotated
from loguru import logger
import sys

# Configure Loguru
logger.remove() # Remove default handler
logger.add(sys.stderr, level="INFO") # Add back with specific level, can be configured

app = typer.Typer(
    name="gswarm-profiler", 
    help="Multi-node multi-GPU profiler using nvitop.",
    epilog="For more information, visit: https://github.com/your-repo/gswarm-profiler"
)

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
    epilog="Example: gswarm-profiler start --host 0.0.0.0 --port 8090 --freq 500 --enable-bandwidth"
)
def start_head_node(
    host: Annotated[str, typer.Option(help="Host address for the head node.")] = "localhost",
    port: Annotated[int, typer.Option(help="Port for the head node.")] = 8090,
    freq: Annotated[int, typer.Option(help="Sampling frequency in milliseconds.")] = 500,
    enable_bandwidth: Annotated[bool, typer.Option("--enable-bandwidth/--disable-bandwidth", help="Enable GPU-DRAM and GPU-GPU bandwidth profiling.")] = False,
    enable_nvlink: Annotated[bool, typer.Option("--enable-nvlink/--disable-nvlink", help="Enable NVLink bandwidth profiling.")] = False,
):
    """
    Starts the head node server.
    Example: gswarm-profiler start --port 8090 --freq 500 --enable-bandwidth
    """
    from .head import run_head_node # Local import to avoid circular dependencies if any
    logger.info(f"Head node enable_bandwidth set to: {enable_bandwidth}")
    logger.info(f"Sampling frequency set to: {freq}ms")
    run_head_node(host, port, enable_bandwidth, enable_nvlink, freq)

@app.command(
    name="connect", 
    help="Connects a client node to the head node.",
    epilog="Example: gswarm-profiler connect localhost:8090"
)
def connect_client_node(
    head_address: Annotated[str, typer.Argument(help="Address of the head node (e.g., localhost:8090).")],
):
    """
    Connects this node as a client to the head server.
    Example: gswarm-profiler connect localhost:8090 --enable-bandwidth
    """
    # read the head's freq and enable_bandwidth from the head node's state by calling the state endpoint
    import requests
    response = requests.post(f"http://{head_address}/state")
    state = response.json()
    freq = state["freq"]
    enable_bandwidth = state["enable_bandwidth_profiling"]
    logger.info(f"Client node enable_bandwidth set to: {enable_bandwidth}")
    if enable_bandwidth:
        logger.info("Client will attempt to collect and send bandwidth metrics.")
        logger.warning("Ensure the head node was also started with --enable-bandwidth to process this data.")
    
    from .client import start_client_node_sync # Local import
    start_client_node_sync(head_address, freq, enable_bandwidth)

@app.command(name="help", help="Show detailed help information.")
def show_help(
    command: Annotated[str, typer.Argument(help="Show help for specific command")] = None
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
        typer.echo("Available commands: start, connect")
        typer.echo("Use --help with any command for detailed information")

if __name__ == "__main__":
    app()

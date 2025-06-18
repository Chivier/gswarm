"""Profiler CLI commands"""

import typer
from typing import Optional
from datetime import datetime
from loguru import logger
import asyncio
import grpc
import enum
import traceback

app = typer.Typer(help="GPU profiling operations")


class AvailableReportMetrics(enum.Enum):
    UTILIZATION = "gpu_utilization"
    VMEMORY = "gpu_memory"
    VBANDWIDTH = "gpu_dram_bandwidth"
    VBUBBLE = "gpu_bubble"


@app.command()
def start(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for the profiling session"),
    freq: Optional[int] = typer.Option(None, "--freq", "-f", help="Override sampling frequency"),
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
    report_metrics: Optional[list[AvailableReportMetrics]] = typer.Option(
        None,
        "--report-metrics",
        "-r",
        help="List of metrics to report (e.g., gpu_utilization, gpu_memory, gpu_dram_bandwidth, gpu_bubble). Default = gpu_utilization,gpu_memory,gpu_dram_bandwidth.",
        case_sensitive=True,
        show_choices=True,
    ),
):
    """Start a profiling session"""
    if not name:
        name = f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info(f"Starting profiling session: {name}")
    logger.info(f"Connecting to profiler at: {host}")
    logger.info(
        f"  Report metrics: {', '.join([m.value for m in report_metrics]) if report_metrics else 'Default metrics'}"
    )
    if freq:
        logger.info(f"  Frequency override: {freq}ms")

    report_metrics = report_metrics or []

    async def start_profiling_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                request = profiler_pb2.StartProfilingRequest(
                    name=name, report_metrics=[m.value for m in report_metrics] or []
                )
                response = await stub.StartProfiling(request)

                if response.success:
                    logger.info(f"Profiling started: {response.message}")
                    logger.info(f"Output file: {response.output_file}")
                else:
                    logger.error(f"Failed to start profiling: {response.message}")
        except Exception as e:
            logger.error(f"Failed to start profiling: {e}")
            traceback.print_exc()

    asyncio.run(start_profiling_async())


@app.command()
def stop(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name of session to stop"),
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
):
    """Stop profiling session(s)"""
    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info(f"Stopping profiling session{'s' if not name else f': {name}'}")
    logger.info(f"Connecting to profiler at: {host}")

    async def stop_profiling_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
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


@app.command()
def status(
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
):
    """Get profiling status"""
    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info("Getting profiler status...")
    logger.info(f"Connecting to profiler at: {host}")

    async def get_status_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                response = await stub.GetStatus(profiler_pb2.Empty())

                logger.info("Profiler Status:")
                logger.info(f"  Frequency: {response.freq}ms")
                logger.info(
                    f"  Bandwidth Profiling: {'Enabled' if response.enable_bandwidth_profiling else 'Disabled'}"
                )
                logger.info(f"  NVLink Profiling: {'Enabled' if response.enable_nvlink_profiling else 'Disabled'}")
                logger.info(f"  Is Profiling: {'Yes' if response.is_profiling else 'No'}")
                if response.output_filename:
                    logger.info(f"  Current Session: {response.output_filename}")
                logger.info(f"  Connected Clients: {len(response.connected_clients)}")

                if response.connected_clients:
                    logger.info("  Client List:")
                    for client in response.connected_clients:
                        logger.info(f"    - {client}")
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            # Show discovered services for debugging
            try:
                from ..utils.service_discovery import get_all_service_ports

                services = get_all_service_ports()
                if services:
                    logger.info("Available services:")
                    for service_name, port, process_name in services:
                        logger.info(f"  - {service_name} on port {port} ({process_name})")
                else:
                    logger.info("No known services found running")
            except Exception as discover_error:
                logger.error(f"Failed to discover services: {discover_error}")

    asyncio.run(get_status_async())


@app.command()
def sessions(
    host: str = typer.Option("localhost:8091", "--host", help="HTTP API address"),
):
    """List all profiling sessions"""
    import requests

    try:
        url = f"http://{host}/profiling/sessions"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        sessions = data.get("sessions", [])

        if sessions:
            logger.info(f"Found {len(sessions)} session(s):")
            for session in sessions:
                status = "Active" if session.get("active") else "Completed"
                logger.info(f"  - {session['name']} ({status})")
                if session.get("start_time"):
                    logger.info(f"    Started: {session['start_time']}")
                if session.get("frames"):
                    logger.info(f"    Frames: {session['frames']}")
        else:
            logger.info("No profiling sessions found")
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")


@app.command()
def analyze(
    data: str = typer.Argument(..., help="Path to profiling data JSON file"),
    plot: Optional[str] = typer.Option(None, "--plot", "-p", help="Output plot file path"),
):
    """Analyze profiling data and generate plots"""
    logger.info(f"Analyzing profiling data: {data}")

    if not plot:
        import os

        plot = os.path.splitext(data)[0] + ".pdf"

    from .stat import show_stat

    show_stat(data, plot)
    logger.info(f"Analysis complete. Plot saved to: {plot}")


@app.command()
def recover(
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List recoverable sessions"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Recover specific session by ID"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Recover session by name"),
    export: bool = typer.Option(False, "--export", "-e", help="Export recovered data"),
):
    """Recover crashed profiling sessions"""
    if list_sessions:
        logger.info("Listing recoverable sessions...")
        # TODO: Implement session recovery listing
        logger.info("Session recovery not yet implemented")
    elif session_id or name:
        logger.info(f"Recovering session: {session_id or name}")
        if export:
            logger.info("Exporting recovered data...")
        # TODO: Implement session recovery
        logger.info("Session recovery not yet implemented")
    else:
        logger.error("Please specify --list, --session-id, or --name")


@app.command()
def read(
    node: Optional[str] = typer.Option(None, "--node", help="Specific node to read status from"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output to JSON file (default: output.json)"),
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
):
    """Read cluster or node status with metrics (GPU util, DRAM, etc.)"""
    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info(f"Reading {'cluster' if not node else f'node {node}'} status...")
    logger.info(f"Connecting to profiler at: {host}")

    async def read_status_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc
            import json
            from rich.console import Console
            from rich.table import Table

            async with grpc.aio.insecure_channel(host) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)

                # Create request for cluster or specific node
                if node:
                    request = profiler_pb2.ReadNodeStatusRequest(node_id=node)
                    response = await stub.ReadNodeStatus(request)
                    nodes_data = [response.node_status] if response.success else []
                else:
                    request = profiler_pb2.ReadClusterStatusRequest()
                    response = await stub.ReadClusterStatus(request)
                    nodes_data = response.nodes if response.success else []

                if not response.success:
                    logger.error(f"Failed to read status: {response.message}")
                    return

                # Prepare data for display and JSON export
                status_data = {
                    "timestamp": datetime.now().isoformat(),
                    "cluster_id": getattr(response, "cluster_id", "unknown"),
                    "nodes": [],
                }

                # Create rich table for CLI display
                console = Console()
                table = Table(title=f"{'Cluster' if not node else 'Node'} Status")

                table.add_column("Node ID", style="cyan")
                table.add_column("Device Type", style="magenta")
                table.add_column("GPU ID", style="green")
                table.add_column("GPU Util %", justify="right")
                table.add_column("Memory Used", justify="right")
                table.add_column("Memory Total", justify="right")
                table.add_column("DRAM BW %", justify="right")
                table.add_column("NVLink BW", justify="right")

                # Process each node's data
                for node_status in nodes_data:
                    node_data = {"node_id": node_status.node_id, "gpus": []}

                    for gpu in node_status.gpus:
                        # Use a simple fallback for GPU name since we can't access head node state from CLI
                        gpu_name = f"GPU_{gpu.gpu_id}"

                        gpu_data = {
                            "gpu_id": gpu.gpu_id,
                            "device_type": gpu_name,
                            "utilization": gpu.utilization,
                            "memory_used": gpu.memory_used,
                            "memory_total": gpu.memory_total,
                            "dram_bandwidth": gpu.dram_bandwidth,
                            "nvlink_bandwidth": gpu.nvlink_bandwidth,
                        }
                        node_data["gpus"].append(gpu_data)

                        # Add row to table
                        table.add_row(
                            node_status.node_id,
                            gpu_name,  # Use the fallback name
                            str(gpu.gpu_id),
                            f"{gpu.utilization:.1f}%" if gpu.utilization >= 0 else "N/A",
                            f"{gpu.memory_used:.0f} MB" if gpu.memory_used >= 0 else "N/A",
                            f"{gpu.memory_total:.0f} MB" if gpu.memory_total >= 0 else "N/A",
                            f"{gpu.dram_bandwidth:.1f}%" if gpu.dram_bandwidth >= 0 else "N/A",
                            f"{gpu.nvlink_bandwidth / 1024:.1f} GB/s" if gpu.nvlink_bandwidth >= 0 else "N/A",
                        )

                    status_data["nodes"].append(node_data)

                # Display table
                console.print(table)

                # Export to JSON if requested
                if output or (output is None):  # Default to output.json if no filename specified
                    output_file = output or "output.json"
                    with open(output_file, "w") as f:
                        json.dump(status_data, f, indent=2)
                    logger.info(f"Status data exported to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to read status: {e}")

    asyncio.run(read_status_async())


@app.command("enable-bandwidth")
def enable_bandwidth(
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
):
    """Enable bandwidth profiling"""
    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info("Enabling bandwidth profiling...")
    logger.info(f"Connecting to profiler at: {host}")

    async def enable_bandwidth_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                request = profiler_pb2.SetBandwidthProfilingRequest(enable=True)
                response = await stub.SetBandwidthProfiling(request)

                if response.success:
                    logger.info(f"Bandwidth profiling enabled: {response.message}")
                else:
                    logger.error(f"Failed to enable bandwidth profiling: {response.message}")
        except Exception as e:
            logger.error(f"Failed to enable bandwidth profiling: {e}")

    asyncio.run(enable_bandwidth_async())


@app.command("disable-bandwidth")
def disable_bandwidth(
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
):
    """Disable bandwidth profiling"""
    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info("Disabling bandwidth profiling...")
    logger.info(f"Connecting to profiler at: {host}")

    async def disable_bandwidth_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                request = profiler_pb2.SetBandwidthProfilingRequest(enable=False)
                response = await stub.SetBandwidthProfiling(request)

                if response.success:
                    logger.info(f"Bandwidth profiling disabled: {response.message}")
                else:
                    logger.error(f"Failed to disable bandwidth profiling: {response.message}")
        except Exception as e:
            logger.error(f"Failed to disable bandwidth profiling: {e}")

    asyncio.run(disable_bandwidth_async())


@app.command("enable-nvlink")
def enable_nvlink(
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
):
    """Enable NVLink profiling"""
    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info("Enabling NVLink profiling...")
    logger.info(f"Connecting to profiler at: {host}")

    async def enable_nvlink_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                request = profiler_pb2.SetNVLinkProfilingRequest(enable=True)
                response = await stub.SetNVLinkProfiling(request)

                if response.success:
                    logger.info(f"NVLink profiling enabled: {response.message}")
                else:
                    logger.error(f"Failed to enable NVLink profiling: {response.message}")
        except Exception as e:
            logger.error(f"Failed to enable NVLink profiling: {e}")

    asyncio.run(enable_nvlink_async())


@app.command("disable-nvlink")
def disable_nvlink(
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
):
    """Disable NVLink profiling"""
    # Auto-discover profiler address if not specified
    if not host:
        from ..utils.service_discovery import discover_profiler_address

        host = discover_profiler_address()

    logger.info("Disabling NVLink profiling...")
    logger.info(f"Connecting to profiler at: {host}")

    async def disable_nvlink_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                request = profiler_pb2.SetNVLinkProfilingRequest(enable=False)
                response = await stub.SetNVLinkProfiling(request)

                if response.success:
                    logger.info(f"NVLink profiling disabled: {response.message}")
                else:
                    logger.error(f"Failed to disable NVLink profiling: {response.message}")
        except Exception as e:
            logger.error(f"Failed to disable NVLink profiling: {e}")

    asyncio.run(disable_nvlink_async())

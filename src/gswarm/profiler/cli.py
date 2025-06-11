"""Profiler CLI commands"""

import typer
from typing import Optional
from datetime import datetime
from loguru import logger
import asyncio
import grpc

app = typer.Typer(help="GPU profiling operations")


@app.command()
def start(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for the profiling session"),
    freq: Optional[int] = typer.Option(None, "--freq", "-f", help="Override sampling frequency"),
    host: str = typer.Option(None, "--host", help="Host address (auto-discovered if not specified)"),
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
    if freq:
        logger.info(f"  Frequency override: {freq}ms")

    async def start_profiling_async():
        try:
            from . import profiler_pb2, profiler_pb2_grpc

            async with grpc.aio.insecure_channel(host) as channel:
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

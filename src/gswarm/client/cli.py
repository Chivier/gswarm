"""Client node CLI commands"""

import typer
import threading
import signal
import sys
import os
import platform
from typing import Optional
from loguru import logger
from gswarm.utils.connection_info import (
    get_connection_file,
    save_connection,
    clear_connection_info,
    get_connection_info,
)
from gswarm.utils.daemonizer import daemonize, get_pid_file, check_pid_file_exists, get_log_filepath
from gswarm.profiler.client_common import create_client_app, start_client
from gswarm.profiler.client_common import parse_extra_metrics

import requests

app = typer.Typer(help="Client node management commands")


# Global client state management
class ClientState:
    def __init__(self):
        self.is_connected = False
        self.host_address = None
        self.node_id = None
        self.resilient_mode = False
        self.enable_bandwidth = None
        self.client_thread = None
        self.model_client = None
        self.shutdown_event = threading.Event()

    def reset(self):
        """Reset client state"""
        self.is_connected = False
        self.host_address = None
        self.node_id = None
        self.resilient_mode = False
        self.enable_bandwidth = None
        self.client_thread = None
        self.model_client = None
        self.shutdown_event.clear()


# Global state instance
client_state = ClientState()


def create_runner(host_address: str, resilient: bool, enable_bandwidth: bool, extra_metrics: list[str] = []):
    """Run client in a separate thread with proper signal handling"""

    def client_runner():
        try:
            if resilient:
                from gswarm.profiler.client_resilient import create_resilient_lifespan as lifespan_func
            else:
                from gswarm.profiler.client import create_client_lifespan as lifespan_func

            app = create_client_app(host_address, enable_bandwidth, lifespan_func, extra_metrics)
            start_client(app)
        except KeyboardInterrupt:
            logger.info("Client interrupted")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            # Mark as disconnected when client exits
            client_state.is_connected = False
            logger.info("Client thread exited")

    return client_runner


@app.command()
def connect(
    host_address: str = typer.Argument(..., help="Host node address (e.g., master:8090)"),
    resilient: bool = typer.Option(False, "--resilient", "-r", help="Enable resilient mode with auto-reconnect"),
    enable_bandwidth: bool = typer.Option(
        None, "--enable-bandwidth", help="Enable bandwidth profiling (can be overridden by host)"
    ),
    node_id: Optional[str] = typer.Option(None, "--node-id", "-n", help="Custom node ID"),
    block: bool = typer.Option(False, "--block", "-b", help="Block until client is started (default: False)"),
    extra_metrics: Optional[str] = typer.Option(
        None,
        "--extra-metrics",
        "-e",
        help="Comma-separated list of extra metrics to collect (e.g., 'gpu_memory,gpu_utilization')",
    ),
):
    """Connect this node as a client to the host"""

    # Check if already connected - check both in-memory state AND persisted connection info
    connection_info = get_connection_info("client")
    if client_state.is_connected or connection_info:
        if connection_info:
            logger.warning(
                f"Already connected to {connection_info.host_address}:{connection_info.profiler_grpc_port}. Use 'disconnect' first."
            )
        else:
            logger.warning(f"Already connected to {client_state.host_address}. Use 'disconnect' first.")
        return

    logger.info(f"Connecting to host at {host_address}")
    logger.info(f"  Resilient mode: {'enabled' if resilient else 'disabled'}")
    if enable_bandwidth is not None:
        logger.info(f"  Bandwidth profiling: {'enabled' if enable_bandwidth else 'disabled'}")
    if node_id:
        logger.info(f"  Node ID: {node_id}")

    logger.info("  Sampling configuration will be read from host")

    # Parse host address
    if ":" in host_address:
        host, port = host_address.split(":")
        port = int(port)
    else:
        host = host_address
        port = 8090

    # Update client state
    client_state.host_address = host_address
    client_state.node_id = node_id or platform.node()
    client_state.resilient_mode = resilient
    client_state.enable_bandwidth = enable_bandwidth

    if not block:
        # connection_file = get_connection_file()
        log_file_path = get_log_filepath(component="client")
        logger.info(
            f"Running in non-blocking mode, client will run in background, please check logs at {log_file_path}"
        )
        daemonize(log_file_path)

    # Start model client (optional)
    from ..model.fastapi_client import ModelClient

    try:
        model_host_port = port + 920  # Default offset from profiler to model port
        model_client = ModelClient(f"http://{host}:{model_host_port}", node_id=node_id)

        # Initialize with empty model dictionary, then discover and register
        if model_client.register_node():
            logger.info("Successfully registered with model service")
            logger.info("Model discovery and registration completed")
            client_state.model_client = model_client
        else:
            logger.debug("Model service registration failed, continuing without it")
    except Exception as e:
        logger.debug(f"Model service not available (this is optional): {e}")

    # Parse extra metrics if provided

    if extra_metrics:
        supported_metrics = parse_extra_metrics(extra_metrics.split(","))
    else:
        supported_metrics = []

    logger.info(f"Extra metrics to collect: {', '.join(supported_metrics) if supported_metrics else 'None'}")

    # Start profiler client in a separate thread
    client_runner = create_runner(host_address, resilient, enable_bandwidth, supported_metrics)
    client_state.client_thread = threading.Thread(target=client_runner, daemon=True)
    client_state.client_thread.start()
    client_state.is_connected = True

    # Save connection info after daemonizing to prevent cleanup function run before daemonization
    save_connection(
        host=host,
        profiler_grpc_port=port,
        profiler_http_port=port + 1,  # Assuming HTTP is on next port
        model_api_port=port + 920,  # Default offset
        node_id=node_id or platform.node(),
        is_host=False,  # This is a client connection
    )

    logger.info("Client started successfully. Use 'gswarm client status' to check connection.")
    logger.info("Use 'gswarm client disconnect' to stop the client.")

    client_state.client_thread.join()


@app.command()
def disconnect():
    """Disconnect from the host"""

    # Check for persisted connection info first (daemon mode)
    connection_info = get_connection_info("client")

    if connection_info:
        logger.info(f"Disconnecting from host at {connection_info.host_address}:{connection_info.profiler_grpc_port}")

        # If we have a PID, try to terminate the daemon process
        if connection_info.control_port:
            logger.info(f"Shutting down client on port: {connection_info.control_port}")
            try:
                response = requests.get(f"http://localhost:{connection_info.control_port}/disconnect")
                if response.status_code == 200:
                    logger.info("Client daemon disconnected successfully")
                else:
                    logger.error(f"Failed to disconnect client daemon: {response.text}")
            except requests.RequestException as e:
                logger.error(f"Error disconnecting client daemon: {e}")
        elif connection_info.pid:
            try:
                import psutil

                if psutil.pid_exists(connection_info.pid):
                    logger.info(f"Terminating client daemon process (PID {connection_info.pid})")
                    process = psutil.Process(connection_info.pid)
                    os.kill(process.pid, signal.SIGINT)  # Send SIGINT to allow graceful shutdown
                    # Wait a bit for graceful termination
                    try:
                        process.wait(timeout=5)
                        logger.info("Client daemon terminated gracefully")
                    except psutil.TimeoutExpired:
                        logger.warning("Client daemon did not terminate gracefully, forcing termination")
                        os.kill(process.pid, signal.SIGKILL)  # Force kill if it didn't stop
                        process.wait()
                        logger.info("Client daemon forcefully terminated")
                else:
                    logger.info("Client daemon process not found (may have already exited)")
            except ImportError:
                logger.warning("psutil not available, cannot terminate daemon process")
                logger.info("You may need to manually kill the client process")
            except Exception as e:
                logger.error(f"Error terminating daemon process: {e}")

        # Clear connection information
        clear_connection_info("client")
        logger.info("Successfully disconnected from host")
        return

    # Fallback: check in-memory state (blocking mode)
    elif client_state.is_connected:
        logger.info(f"Disconnecting from host at {client_state.host_address}")

        # Disconnect from model service if connected
        if client_state.model_client:
            try:
                logger.info("Disconnecting from model service...")
                client_state.model_client = None
            except Exception as e:
                logger.debug(f"Error disconnecting from model service: {e}")

        # Signal shutdown to any running threads
        client_state.shutdown_event.set()

        # Wait for client thread to finish gracefully
        if client_state.client_thread and client_state.client_thread.is_alive():
            logger.info("Waiting for client thread to stop...")
            try:
                # Give the thread some time to stop gracefully
                client_state.client_thread.join(timeout=5.0)
                if client_state.client_thread.is_alive():
                    logger.warning("Client thread did not stop gracefully within timeout")
                else:
                    logger.info("Client thread stopped gracefully")
            except Exception as e:
                logger.error(f"Error waiting for client thread: {e}")

        # Reset client state
        client_state.reset()

        # Clear connection information
        clear_connection_info("client")

        logger.info("Successfully disconnected from host")
    else:
        logger.info("Not connected to any host")


@app.command()
def status():
    """Get client node status"""
    logger.info("Client Node Status:")
    logger.info("=" * 50)

    # Check for persisted connection info first
    connection_info = get_connection_info("client")

    # If we have connection info, we're potentially connected
    if connection_info:
        logger.info("Status: CONNECTED")
        logger.info(f"Host Address: {connection_info.host_address}:{connection_info.profiler_grpc_port}")
        logger.info(f"Node ID: {connection_info.node_id}")
        logger.info(f"Connected At: {connection_info.connected_at}")

        if connection_info.pid:
            # Check if the client process is still running
            try:
                import psutil

                if psutil.pid_exists(connection_info.pid):
                    logger.info(f"Client Process: running (PID {connection_info.pid})")
                else:
                    logger.info(f"Client Process: stopped (PID {connection_info.pid} not found)")
            except ImportError:
                logger.info(f"Client Process: PID {connection_info.pid} (psutil not available for verification)")

        # Try to check actual connection to host
        try:
            import grpc
            from ..profiler import profiler_pb2_grpc, profiler_pb2

            logger.info("\nHost Connection Test:")
            logger.info("-" * 30)

            async def test_connection():
                try:
                    host_address = f"{connection_info.host_address}:{connection_info.profiler_grpc_port}"
                    async with grpc.aio.insecure_channel(host_address) as channel:
                        stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                        status_response = await stub.GetStatus(profiler_pb2.Empty())
                        logger.info(f"Host Status: reachable")
                        logger.info(f"Host Frequency: {status_response.freq}ms")
                        logger.info(
                            f"Host Bandwidth Profiling: {'enabled' if status_response.enable_bandwidth_profiling else 'disabled'}"
                        )
                        logger.info(f"Connected Clients: {len(status_response.connected_clients)}")

                        # Check if this client is in the connected clients list
                        if connection_info.node_id in status_response.connected_clients:
                            logger.info(f"This Client: registered with host")
                        else:
                            logger.info(f"This Client: not found in host's client list")

                        return True
                except Exception as e:
                    logger.warning(f"Host Status: unreachable ({e})")
                    return False

            import asyncio

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(test_connection())
            finally:
                loop.close()

        except Exception as e:
            logger.debug(f"Could not test host connection: {e}")

    elif client_state.is_connected:
        # Fallback to in-memory state (for blocking mode)
        logger.info("Status: CONNECTED")
        logger.info(f"Host Address: {client_state.host_address}")
        logger.info(f"Node ID: {client_state.node_id}")
        logger.info(f"Resilient Mode: {'enabled' if client_state.resilient_mode else 'disabled'}")

        if client_state.enable_bandwidth is not None:
            logger.info(f"Bandwidth Profiling: {'enabled' if client_state.enable_bandwidth else 'disabled'}")
        else:
            logger.info("Bandwidth Profiling: configured by host")

        # Check thread status
        if client_state.client_thread:
            thread_status = "alive" if client_state.client_thread.is_alive() else "stopped"
            logger.info(f"Client Thread: {thread_status}")

        # Check model service status
        if client_state.model_client:
            try:
                if client_state.model_client.heartbeat():
                    logger.info("Model Service: connected")
                else:
                    logger.info("Model Service: connection lost")
            except Exception:
                logger.info("Model Service: connection error")
        else:
            logger.info("Model Service: not registered")
    else:
        logger.info("Status: DISCONNECTED")
        logger.info("No active connection to host")

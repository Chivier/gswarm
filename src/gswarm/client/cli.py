"""Client node CLI commands"""

import typer
import threading
import signal
import sys
import platform
from typing import Optional
from loguru import logger
from ..utils.connection_info import save_host_connection, clear_connection_info

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

def run_client_in_thread(host_address: str, resilient: bool, enable_bandwidth: bool):
    """Run client in a separate thread with proper signal handling"""
    def client_runner():
        try:
            if resilient:
                from ..profiler.client_resilient import start_resilient_client
                start_resilient_client(host_address, enable_bandwidth)
            else:
                from ..profiler.client import start_client_node_sync
                start_client_node_sync(host_address, enable_bandwidth)
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
    enable_bandwidth: bool = typer.Option(None, "--enable-bandwidth", help="Enable bandwidth profiling (can be overridden by host)"),
    node_id: Optional[str] = typer.Option(None, "--node-id", "-n", help="Custom node ID"),
):
    """Connect this node as a client to the host"""
    
    # Check if already connected
    if client_state.is_connected:
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
    if ':' in host_address:
        host, port = host_address.split(':')
        port = int(port)
    else:
        host = host_address
        port = 8090

    # Save connection info
    save_host_connection(
        host=host,
        profiler_grpc_port=port,
        profiler_http_port=port + 1,  # Assuming HTTP is on next port
        model_api_port=port + 920,  # Default offset
        node_id=node_id or platform.node()
    )

    # Update client state
    client_state.host_address = host_address
    client_state.node_id = node_id or platform.node()
    client_state.resilient_mode = resilient
    client_state.enable_bandwidth = enable_bandwidth
    
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
    
    # Start profiler client in a separate thread
    client_runner = run_client_in_thread(host_address, resilient, enable_bandwidth)
    client_state.client_thread = threading.Thread(target=client_runner, daemon=True)
    client_state.client_thread.start()
    client_state.is_connected = True
    
    logger.info("Client started successfully. Use 'gswarm client status' to check connection.")
    logger.info("Use 'gswarm client disconnect' to stop the client.")

@app.command()
def disconnect():
    """Disconnect from the host"""
    if not client_state.is_connected:
        logger.info("No active connection to disconnect")
        return
    
    logger.info(f"Disconnecting from host at {client_state.host_address}...")
    
    # Clear connection info
    clear_connection_info()
    
    try:
        # Signal shutdown
        client_state.shutdown_event.set()
        
        # If we have a model client, try to unregister
        if client_state.model_client:
            try:
                # Note: The current ModelClient doesn't have an unregister method,
                # but we can at least clear our reference
                logger.info("Clearing model service registration")
                client_state.model_client = None
            except Exception as e:
                logger.debug(f"Error during model service cleanup: {e}")
        
        # The profiler clients handle KeyboardInterrupt for graceful shutdown
        # Send SIGINT to the current process if the client thread is running
        if client_state.client_thread and client_state.client_thread.is_alive():
            logger.info("Sending shutdown signal to client...")
            # The client implementations handle KeyboardInterrupt properly
            import os
            os.kill(os.getpid(), signal.SIGINT)
        
        # Wait a moment for graceful shutdown
        if client_state.client_thread:
            client_state.client_thread.join(timeout=5.0)
            if client_state.client_thread.is_alive():
                logger.warning("Client thread did not shutdown gracefully")
    
    except Exception as e:
        logger.error(f"Error during disconnect: {e}")
    finally:
        # Reset state
        client_state.reset()
        logger.info("Disconnected successfully")

@app.command()
def status():
    """Get client node status"""
    logger.info("Client Node Status:")
    logger.info("=" * 50)
    
    if not client_state.is_connected:
        logger.info("Status: DISCONNECTED")
        logger.info("No active connection to host")
        return
    
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
    
    # Try to check actual connection to host
    try:
        from ..utils.service_discovery import discover_profiler_address
        import grpc
        from ..profiler import profiler_pb2_grpc, profiler_pb2
        
        # Try to connect and get status from host
        logger.info("\nHost Connection Test:")
        logger.info("-" * 30)
        
        async def test_connection():
            try:
                async with grpc.aio.insecure_channel(client_state.host_address) as channel:
                    stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                    status_response = await stub.GetStatus(profiler_pb2.Empty())
                    logger.info(f"Host Status: reachable")
                    logger.info(f"Host Frequency: {status_response.freq}ms")
                    logger.info(f"Host Bandwidth Profiling: {'enabled' if status_response.enable_bandwidth_profiling else 'disabled'}")
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
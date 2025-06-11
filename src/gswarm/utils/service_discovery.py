"""Service discovery utilities for gswarm services"""

import psutil
import grpc
from typing import Optional, List, Tuple
from loguru import logger
from .connection_info import connection_manager


def get_process_using_port(port: int) -> Optional[psutil.Process]:
    """Find process using a specific port"""
    try:
        # Use connections() method instead of 'connections' attribute
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                # Get connections separately
                connections = proc.connections(kind="inet")
                for conn in connections:
                    if hasattr(conn, "laddr") and conn.laddr.port == port:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.debug(f"Error checking processes: {e}")
    return None


def find_profiler_grpc_port(default_ports: List[int] = None) -> Optional[int]:
    """Find the port where profiler gRPC service is running"""
    # First check connection info
    conn_info = connection_manager.load_connection()
    if conn_info:
        return conn_info.profiler_grpc_port

    if default_ports is None:
        default_ports = [8090, 8091, 8092, 8093, 8094, 8095]  # Common profiler ports

    for port in default_ports:
        process = get_process_using_port(port)
        if process:
            # Try to verify it's actually a profiler gRPC service
            try:
                import asyncio

                async def test_grpc_connection():
                    try:
                        from ..profiler import profiler_pb2_grpc, profiler_pb2

                        async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                            stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                            await asyncio.wait_for(stub.GetStatus(profiler_pb2.Empty()), timeout=1.0)
                            return True
                    except Exception:
                        return False

                # Run the async test
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    is_profiler = loop.run_until_complete(test_grpc_connection())
                    if is_profiler:
                        logger.debug(f"Found profiler gRPC service on port {port}")
                        return port
                finally:
                    loop.close()
            except Exception:
                # If gRPC test fails, still consider it might be the right port
                # if it's on a known profiler port
                if port == 8090:  # Default profiler port
                    logger.debug(f"Found process on default profiler port {port}")
                    return port

    return None


def discover_profiler_address(host: str = "localhost") -> str:
    """Discover the profiler gRPC address"""
    # First check connection info
    conn_info = connection_manager.load_connection()
    if conn_info:
        return f"{conn_info.host_address}:{conn_info.profiler_grpc_port}"

    # Try to discover
    port = find_profiler_grpc_port()
    if port:
        return f"{host}:{port}"
    else:
        # Fall back to default
        logger.warning("Could not discover profiler port, using default localhost:8090")
        return f"{host}:8090"


def get_all_service_ports() -> List[Tuple[str, int, Optional[str]]]:
    """Get all services running on known ports

    Returns:
        List of (service_name, port, process_name) tuples
    """
    # Check connection info first
    conn_info = connection_manager.load_connection()
    if conn_info:
        return [
            ("profiler_grpc", conn_info.profiler_grpc_port, "gswarm"),
            ("profiler_http", conn_info.profiler_http_port, "gswarm"),
            ("model_api", conn_info.model_api_port, "gswarm"),
        ]

    # Otherwise try to discover
    known_services = {8090: "profiler_grpc", 8091: "profiler_http", 9010: "model_api"}

    services = []
    for port, service_name in known_services.items():
        process = get_process_using_port(port)
        if process:
            services.append((service_name, port, process.name()))

    return services

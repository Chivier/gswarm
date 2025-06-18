"""Host node CLI commands"""

from ..utils.connection_info import save_connection
import typer
from typing import Optional
from loguru import logger
import asyncio
import socket
import psutil
import signal
import os
import requests
import grpc
from pathlib import Path

app = typer.Typer(help="Host node management commands")

# Global service tracking
SERVICE_PORTS = {"profiler_grpc": 8090, "profiler_http": 8091, "model_api": 9010}


def get_process_using_port(port: int) -> Optional[psutil.Process]:
    """Find process using a specific port"""
    try:
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                # Get connections separately using the connections() method
                connections = proc.connections(kind="inet")
                for conn in connections:
                    if hasattr(conn, "laddr") and conn.laddr.port == port:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.debug(f"Error checking processes: {e}")
    return None


def check_port_availability(host: str, port: int) -> bool:
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


async def check_service_health(service_name: str, host: str, port: int) -> dict:
    """Check health of a specific service"""
    result = {"service": service_name, "host": host, "port": port, "status": "unknown", "pid": None, "details": ""}

    # Check if port is in use
    if check_port_availability(host, port):
        result["status"] = "stopped"
        result["details"] = "Port is available (service not running)"
        return result

    # Find process using the port
    process = get_process_using_port(port)
    if process:
        result["pid"] = process.pid
        result["status"] = "running"
        result["details"] = f"Process {process.pid} ({process.name()})"

        # Additional health checks for specific services
        try:
            if service_name == "profiler_grpc":
                # Try gRPC connection
                try:
                    from ..profiler import profiler_pb2_grpc, profiler_pb2

                    async with grpc.aio.insecure_channel(f"{host}:{port}") as channel:
                        stub = profiler_pb2_grpc.ProfilerServiceStub(channel)
                        await asyncio.wait_for(stub.GetStatus(profiler_pb2.Empty()), timeout=2.0)
                        result["details"] += " (gRPC responding)"
                except Exception as e:
                    result["status"] = "unhealthy"
                    result["details"] += f" (gRPC error: {str(e)[:50]})"

            elif service_name in ["profiler_http", "model_api"]:
                # Try HTTP connection
                try:
                    response = requests.get(f"http://{host}:{port}/health", timeout=2.0)
                    if response.status_code == 200:
                        result["details"] += " (HTTP responding)"
                    else:
                        result["status"] = "unhealthy"
                        result["details"] += f" (HTTP {response.status_code})"
                except requests.exceptions.RequestException:
                    # Try root endpoint as fallback
                    try:
                        response = requests.get(f"http://{host}:{port}/", timeout=2.0)
                        if response.status_code == 200:
                            result["details"] += " (HTTP responding)"
                        else:
                            result["status"] = "unhealthy"
                            result["details"] += f" (HTTP {response.status_code})"
                    except requests.exceptions.RequestException as e:
                        result["status"] = "unhealthy"
                        result["details"] += f" (HTTP error: {str(e)[:50]})"

        except Exception as e:
            result["details"] += f" (Health check failed: {str(e)[:50]})"
    else:
        result["status"] = "error"
        result["details"] = "Port in use but process not found"

    return result


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

    # Check all ports before starting any service
    ports_to_check = {"Profiler gRPC": port, "HTTP API": http_port, "Model API": model_port}

    ports_in_use = []
    for service_name, service_port in ports_to_check.items():
        if not check_port_availability(host, service_port):
            process = get_process_using_port(service_port)
            if process:
                ports_in_use.append(
                    f"{service_name} port {service_port} (used by PID {process.pid} - {process.name()})"
                )
            else:
                ports_in_use.append(f"{service_name} port {service_port}")

    if ports_in_use:
        logger.error("Cannot start host node - the following ports are already in use:")
        for port_info in ports_in_use:
            logger.error(f"  - {port_info}")
        logger.info("\nOptions:")
        logger.info("1. Stop the processes using these ports")
        logger.info("2. Choose different ports with --port, --http-port, and --model-port options")
        logger.info(f"\nTo find processes: lsof -i :{port} or netstat -tulpn | grep :{port}")
        raise typer.Exit(1)

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
        try:
            # Start profiler in background
            import uvicorn

            model_app = create_model_app(host=host, port=port, model_port=model_port)

            # Create tasks for both services
            profiler_task = asyncio.create_task(
                asyncio.to_thread(run_profiler_head, host, port, enable_bandwidth, enable_nvlink, http_port)
            )
            model_server_task = asyncio.create_task(
                uvicorn.Server(uvicorn.Config(model_app, host=host, port=model_port, log_level="warning")).serve()
            )

            # Wait for both tasks to complete
            await asyncio.gather(profiler_task, model_server_task)

        except asyncio.CancelledError:
            logger.info("Services cancelled")
            raise
        except Exception as e:
            # Cancel remaining tasks if one fails
            profiler_task.cancel()
            model_server_task.cancel()
            logger.error(f"Error starting host node: {e}")
            raise

    try:
        # Save connection info after validating ports
        save_connection(
            host=host or "localhost",
            profiler_grpc_port=port,
            profiler_http_port=http_port,
            model_api_port=model_port,
            is_host=True,  # This is a host connection
        )

        asyncio.run(run_all_services())
    except KeyboardInterrupt:
        logger.info("Host node stopped")
    except SystemExit:
        # Don't propagate SystemExit from child processes
        logger.error("One of the services failed to start")
        raise typer.Exit(1)


@app.command()
def stop():
    """Stop the host node"""
    logger.info("Stopping host node services...")

    stopped_services = []
    failed_services = []

    # Get current service ports (could be customized, so check defaults and common ports)
    ports_to_check = [8090, 8091, 9010]  # Default ports

    for port in ports_to_check:
        process = get_process_using_port(port)
        if process:
            try:
                service_name = f"service_on_port_{port}"
                if port == 8090:
                    service_name = "profiler_grpc"
                elif port == 8091:
                    service_name = "profiler_http"
                elif port == 9010:
                    service_name = "model_api"

                logger.info(f"Stopping {service_name} (PID: {process.pid})")

                # Try graceful shutdown first
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    stopped_services.append(f"{service_name} (PID: {process.pid})")
                    logger.info(f"Successfully stopped {service_name}")
                except psutil.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning(f"Graceful shutdown timeout for {service_name}, forcing kill...")
                    process.kill()
                    process.wait(timeout=5)
                    stopped_services.append(f"{service_name} (PID: {process.pid}) - force killed")
                    logger.warning(f"Force killed {service_name}")

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                failed_services.append(f"port_{port}: {str(e)}")
                logger.error(f"Failed to stop service on port {port}: {e}")

    # Summary
    if stopped_services:
        logger.info(f"Stopped services: {', '.join(stopped_services)}")

    if failed_services:
        logger.error(f"Failed to stop: {', '.join(failed_services)}")

    if not stopped_services and not failed_services:
        logger.info("No host node services were running")


@app.command()
def status():
    """Get host node status"""
    logger.info("Checking host node status...")

    # Default configuration
    host = "0.0.0.0"
    services = {"profiler_grpc": 8090, "profiler_http": 8091, "model_api": 9010}

    async def check_all_services():
        results = []
        for service_name, port in services.items():
            result = await check_service_health(service_name, host, port)
            results.append(result)
        return results

    try:
        results = asyncio.run(check_all_services())

        # Print status summary
        running_count = 0
        stopped_count = 0
        unhealthy_count = 0

        logger.info("\n" + "=" * 60)
        logger.info("HOST NODE SERVICE STATUS")
        logger.info("=" * 60)

        for result in results:
            status_emoji = {"running": "✅", "stopped": "❌", "unhealthy": "⚠️", "error": "🔥"}.get(
                result["status"], "❓"
            )

            logger.info(f"{status_emoji} {result['service']:<15} {result['host']}:{result['port']}")
            logger.info(f"   Status: {result['status']}")
            logger.info(f"   Details: {result['details']}")

            if result["status"] == "running":
                running_count += 1
            elif result["status"] == "stopped":
                stopped_count += 1
            elif result["status"] == "unhealthy":
                unhealthy_count += 1

            logger.info("")

        logger.info("=" * 60)
        logger.info(f"SUMMARY: {running_count} running, {stopped_count} stopped, {unhealthy_count} unhealthy")
        logger.info("=" * 60)

        # Overall status
        if running_count == len(services):
            logger.info("🎉 All services are running and healthy!")
        elif running_count > 0:
            logger.warning(f"⚠️  Partial service availability ({running_count}/{len(services)} running)")
        else:
            logger.error("❌ No services are running")

    except Exception as e:
        logger.error(f"Failed to check service status: {e}")
        raise typer.Exit(1)

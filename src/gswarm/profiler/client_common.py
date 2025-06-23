import os
import signal
from fastapi import FastAPI
from loguru import logger
import asyncio
import importlib.util
import ast
import nvitop
from gswarm.utils.connection_info import update_connection_info


def create_client_app(
    head_address: str, enable_bandwidth: bool, lifespan_func, extra_metrics: list[str] = []
) -> FastAPI:
    """Create FastAPI app with resilient client"""
    app = FastAPI(lifespan=lifespan_func(head_address, enable_bandwidth, extra_metrics))

    async def delayed_shutdown(pid: int):
        """Delayed shutdown to ensure response is sent"""
        await asyncio.sleep(0.5)  # Small delay to ensure response is sent
        os.kill(pid, signal.SIGINT)

    @app.get("/disconnect", summary="Graceful disconnect")
    async def disconnect():
        """Endpoint to gracefully disconnect the client"""
        pid = os.getpid()
        logger.info(f"Disconnect requested for client with PID {pid}")

        # Return response first, then shutdown
        response = {"message": "Client disconnect initiated"}

        # Schedule shutdown after response is sent
        asyncio.create_task(delayed_shutdown(pid))

        return response

    return app


def start_client(app):
    """Launch the client"""

    from uvicorn import Config, Server
    import socket
    # Find an available port from 10000 to 20000

    port = None
    for candidate_port in range(10000, 20001):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("0.0.0.0", candidate_port))
                port = candidate_port
                update_connection_info("control_port", port)
                break
        except OSError:
            continue

    if port is None:
        raise RuntimeError("No available port found in range 10000-20000")
    config = Config(app, host="0.0.0.0", port=port, log_level="info")
    server = Server(config)

    logger.info(f"Starting client server on port {port}...")
    asyncio.run(server.serve())


def check_nvitop_support(deivce, name: str):
    if not hasattr(deivce, name):
        return False
    value = getattr(deivce, name)()
    if value is None or value == nvitop.NA:
        return False
    return True


def parse_extra_metrics(metrics_list: list[str]) -> list[str]:
    """
    Check support metrics in metric list and return supported metrics.
    """

    test_device = nvitop.Device(0)  # By default, we use device 0 to check support.
    supported_metrics = []
    for metric in metrics_list:
        if hasattr(test_device, metric):
            value = getattr(test_device, metric)()
            if value is not None and value != nvitop.NA:
                supported_metrics.append(metric)
            else:
                logger.warning(f"Metric '{metric}' is not supported by the device.")
        else:
            logger.warning(f"Metric '{metric}' does not exist in the device.")
    return supported_metrics


def get_extra_metrics_value(device, metrics: list[str]) -> dict[str, float]:
    """
    Get the value of extra metrics from the device.
    """
    metrics_value = {}
    for metric in metrics:
        value = getattr(device, metric)()
        metrics_value[metric] = value if value is not None and value != nvitop.NA else 0.0

    return metrics_value

import asyncio
import grpc
import nvitop
import platform
import psutil  # Add psutil for system metrics
import time
import os
import signal
from loguru import logger
from typing import List, Dict, Any

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from contextlib import asynccontextmanager
from fastapi import FastAPI

# Import generated protobuf classes
try:
    from gswarm.profiler import profiler_pb2
    from gswarm.profiler import profiler_pb2_grpc
    from gswarm.profiler.adaptive_sampler import AdaptiveSampler
    from google.protobuf.struct_pb2 import Struct
except ImportError:
    logger.error("gRPC protobuf files not found. Please run 'python generate_grpc.py' first.")
    raise

from gswarm.profiler.client_common import get_extra_metrics_value


def display_gpu_info(payload: Dict[str, Any]):
    """Display GPU metrics and system metrics in a formatted table"""
    # Create GPU metrics table
    gpu_table = Table(title="GPU Metrics")
    gpu_table.add_column("GPU ID", justify="center")
    gpu_table.add_column("GPU Name", justify="center")
    gpu_table.add_column("GPU Utilization", justify="center")
    gpu_table.add_column("GPU Memory Utilization", justify="center")  # Fixed column name
    gpu_table.add_column("DRAM Bandwidth RX", justify="center")
    gpu_table.add_column("DRAM Bandwidth TX", justify="center")
    gpu_table.add_column("NVLink Bandwidth RX", justify="center")
    gpu_table.add_column("NVLink Bandwidth TX", justify="center")

    metrics = payload.get("gpus_metrics", [])
    for gpu in metrics:
        gpu_id = gpu.get("physical_idx", "N/A")
        gpu_name = gpu.get("name", "N/A")
        gpu_util = gpu.get("gpu_util", 0.0) * 100
        mem_util = gpu.get("mem_util", 0.0) * 100
        dram_bw_rx = gpu.get("dram_bw_gbps_rx", 0.0)
        dram_bw_tx = gpu.get("dram_bw_gbps_tx", 0.0)
        nvlink_bw_rx = gpu.get("nvlink_bw_gbps_rx", 0.0)
        nvlink_bw_tx = gpu.get("nvlink_bw_gbps_tx", 0.0)

        gpu_table.add_row(
            str(gpu_id),
            str(gpu_name),
            f"{gpu_util:.2f}%",
            f"{mem_util:.2f}%",
            f"{dram_bw_rx:.2f} Kbps",
            f"{dram_bw_tx:.2f} Kbps",
            f"{nvlink_bw_rx:.2f} Kbps",
            f"{nvlink_bw_tx:.2f} Kbps",
        )

    # Create system metrics table
    system_table = Table(title="System Metrics")
    system_table.add_column("Metric", justify="left")
    system_table.add_column("Value", justify="center")

    # Add system metrics
    system_metrics = payload.get("system_metrics", {})
    system_table.add_row("DRAM Utilization", f"{system_metrics.get('dram_util', 0.0):.2f}%")
    system_table.add_row("Disk Utilization", f"{system_metrics.get('disk_util', 0.0):.2f}%")

    # Create layout with both tables
    layout = Layout()
    layout.split_column(Layout(gpu_table, name="gpu"), Layout(system_table, name="system", size=6))

    return layout


def collect_system_metrics() -> Dict[str, float]:
    """Collect system-level metrics (DRAM and disk utilization)"""
    system_metrics = {}

    try:
        # Get DRAM (memory) utilization
        memory = psutil.virtual_memory()
        system_metrics["dram_util"] = memory.percent
    except Exception as e:
        logger.debug(f"Failed to get DRAM utilization: {e}")
        system_metrics["dram_util"] = 0.0

    try:
        # Get disk utilization (for root partition)
        disk = psutil.disk_usage("/")
        system_metrics["disk_util"] = disk.percent
    except Exception as e:
        logger.debug(f"Failed to get disk utilization: {e}")
        system_metrics["disk_util"] = 0.0

    return system_metrics


async def collect_gpu_metrics(enable_bandwidth: bool, extra_metrics: list[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "gpus_metrics": [],
    }
    if enable_bandwidth:
        payload["p2p_links"] = []

    devices = nvitop.Device.all()

    for i, device in enumerate(devices):
        try:
            gpu_metric = {
                "physical_idx": i,
                "name": device.name(),
                "gpu_util": 0.0,
                "mem_util": 0.0,
                "mem_used_mb": 0,
                "mem_total_mb": 0,
                "dram_bw_gbps_rx": 0.0,
                "dram_bw_gbps_tx": 0.0,
                "nvlink_bw_gbps_rx": 0.0,
                "nvlink_bw_gbps_tx": 0.0,
                "extra_metrics": {},
            }

            try:
                gpu_metric["gpu_util"] = device.gpu_percent() / 100.0
            except (AttributeError, NotImplementedError):
                logger.debug(f"GPU utilization not available for device {i}")

            try:
                gpu_metric["mem_util"] = device.memory_percent() / 100.0
            except (AttributeError, NotImplementedError):
                logger.debug(f"Memory utilization not available for device {i}")

            # Get actual memory values from nvitop
            try:
                memory_info = device.memory()
                if memory_info is not None:
                    gpu_metric["mem_used_mb"] = memory_info.used // (1024 * 1024)  # Convert to MB
                    gpu_metric["mem_total_mb"] = memory_info.total // (1024 * 1024)  # Convert to MB
                else:
                    # Fallback calculation using percentage if direct memory access fails
                    if gpu_metric["mem_util"] > 0:
                        # Try to get from torch as fallback
                        try:
                            import torch

                            if torch.cuda.is_available() and i < torch.cuda.device_count():
                                props = torch.cuda.get_device_properties(i)
                                gpu_metric["mem_total_mb"] = props.total_memory // (1024 * 1024)
                                gpu_metric["mem_used_mb"] = int(gpu_metric["mem_total_mb"] * gpu_metric["mem_util"])
                        except ImportError:
                            pass
            except (AttributeError, NotImplementedError):
                # Final fallback to torch if available
                try:
                    import torch

                    if torch.cuda.is_available() and i < torch.cuda.device_count():
                        props = torch.cuda.get_device_properties(i)
                        gpu_metric["mem_total_mb"] = props.total_memory // (1024 * 1024)
                        # Try to get current memory usage from torch
                        try:
                            gpu_metric["mem_used_mb"] = torch.cuda.memory_allocated(i) // (1024 * 1024)
                        except:
                            gpu_metric["mem_used_mb"] = int(gpu_metric["mem_total_mb"] * gpu_metric["mem_util"])
                except ImportError:
                    logger.debug(f"Neither nvitop memory info nor torch available for device {i}")

            if enable_bandwidth:
                try:
                    bw_kbps = device.pcie_throughput()
                    if bw_kbps[0] is not None:
                        gpu_metric["dram_bw_gbps_rx"] = bw_kbps[0]
                    if bw_kbps[1] is not None:
                        gpu_metric["dram_bw_gbps_tx"] = bw_kbps[1]
                except (AttributeError, NotImplementedError):
                    logger.debug(f"PCIe bandwidth not available for device {i}")

                try:
                    link_throughput = device.nvlink_total_throughput()
                    if link_throughput[0] is not None:
                        gpu_metric["nvlink_bw_gbps_tx"] = link_throughput[0]
                    elif link_throughput[1] is not None:
                        gpu_metric["nvlink_bw_gbps_tx"] = link_throughput[1]
                except (AttributeError, NotImplementedError):
                    logger.debug(f"NVLink information not available for device {i}")

            payload["gpus_metrics"].append(gpu_metric)

        except Exception as e:
            logger.warning(f"Error collecting metrics for device {i}: {e}")
            payload["gpus_metrics"].append(
                {
                    "physical_idx": i,
                    "name": device.name() if hasattr(device, "name") else f"GPU_{i}",
                    "gpu_util": 0.0,
                    "mem_util": 0.0,
                    "mem_used_mb": 0,
                    "mem_total_mb": 0,
                    "dram_bw_gbps_rx": 0.0,
                    "dram_bw_gbps_tx": 0.0,
                    "nvlink_bw_gbps_rx": 0.0,
                    "nvlink_bw_gbps_tx": 0.0,
                    "extra_metrics": {},
                }
            )

    # collect extra metrics if available
    for i, device in enumerate(devices):
        payload["gpus_metrics"][i]["extra_metrics"] = get_extra_metrics_value(device, extra_metrics)
    # Add system metrics to payload

    payload["system_metrics"] = collect_system_metrics()

    return payload


def dict_to_grpc_metrics_update(hostname: str, payload: Dict[str, Any]) -> profiler_pb2.MetricsUpdate:
    """Convert dictionary payload to gRPC MetricsUpdate message"""
    gpu_metrics = []
    for gpu in payload.get("gpus_metrics", []):
        extra_data_struct = Struct()
        extra_data_struct.update(gpu.get("extra_metrics", {}))  # Convert extra metrics to Struct
        gpu_metrics.append(
            profiler_pb2.GPUMetric(
                physical_idx=gpu["physical_idx"],
                name=gpu["name"],
                gpu_util=gpu["gpu_util"],
                mem_util=gpu["mem_util"],
                dram_bw_gbps_rx=gpu.get("dram_bw_gbps_rx", 0.0),
                dram_bw_gbps_tx=gpu.get("dram_bw_gbps_tx", 0.0),
                nvlink_bw_gbps_rx=gpu.get("nvlink_bw_gbps_rx", 0.0),
                nvlink_bw_gbps_tx=gpu.get("nvlink_bw_gbps_tx", 0.0),
                extra_metrics=extra_data_struct,  # Use Struct for extra metrics
            )
        )

    p2p_links = []
    for link in payload.get("p2p_links", []):
        p2p_links.append(
            profiler_pb2.P2PLink(
                local_gpu_physical_id=link["local_gpu_physical_id"],
                local_gpu_name=link["local_gpu_name"],
                remote_gpu_physical_id=link["remote_gpu_physical_id"],
                remote_gpu_name=link["remote_gpu_name"],
                type=link["type"],
                aggregated_max_bandwidth_gbps=link["aggregated_max_bandwidth_gbps"],
            )
        )

    # Add system metrics to gRPC message
    system_metrics = payload.get("system_metrics", {})
    system_metrics_proto = profiler_pb2.SystemMetrics(
        dram_util=system_metrics.get("dram_util", 0.0), disk_util=system_metrics.get("disk_util", 0.0)
    )

    return profiler_pb2.MetricsUpdate(
        hostname=hostname, gpus_metrics=gpu_metrics, p2p_links=p2p_links, system_metrics=system_metrics_proto
    )


async def run_client_node(head_address: str, enable_bandwidth: bool, extra_metrics: list[str] = []):
    hostname = platform.node()

    # These will be set from host config
    freq_ms = 200  # Default
    use_adaptive = False
    adaptive_sampler = None

    # Check if nvitop can find GPUs
    try:
        devices = nvitop.Device.all()
        if not devices:
            logger.error("No NVIDIA GPUs found on this client node. Exiting.")
            return
        logger.info(f"Found {len(devices)} GPU(s) on this client: {[d.name() for d in devices]}")

        # Prepare initial GPU info for gRPC
        gpu_infos = []
        for i, dev in enumerate(devices):
            gpu_infos.append(profiler_pb2.GPUInfo(physical_idx=i, name=dev.name()))

        initial_info = profiler_pb2.InitialInfo(hostname=hostname, gpus=gpu_infos)

    except nvitop.NVMLError as e:
        logger.error(f"NVML Error: {e}. Ensure NVIDIA drivers are installed and nvitop has permissions.")
        return
    except Exception as e:
        logger.error(f"Error initializing nvitop or finding GPUs: {e}")
        return

    logger.info(f"Attempting to connect to head node at {head_address} via gRPC")
    logger.info(f"Client-side bandwidth data collection: {'Enabled' if enable_bandwidth else 'Disabled'}")

    retry_delay = 5
    while True:  # Connection retry loop
        try:
            # Create gRPC channel
            async with grpc.aio.insecure_channel(head_address) as channel:
                stub = profiler_pb2_grpc.ProfilerServiceStub(channel)

                # Connect and send initial info
                connect_response = await stub.Connect(initial_info)
                if not connect_response.success:
                    logger.error(f"Failed to connect: {connect_response.message}")
                    await asyncio.sleep(retry_delay)
                    continue

                logger.info(f"Connected to head node: {connect_response.message}")
                retry_delay = 5  # Reset retry delay on successful connection

                # Get configuration from host
                try:
                    status = await stub.GetStatus(profiler_pb2.Empty())
                    freq_ms = status.freq if status.freq > 0 else 0
                    use_adaptive = status.freq == 0
                    enable_bandwidth = status.enable_bandwidth_profiling

                    if use_adaptive:
                        adaptive_sampler = AdaptiveSampler()
                        logger.info("Using adaptive sampling strategy (configured by host)")
                    else:
                        logger.info(f"Using fixed frequency sampling: {freq_ms}ms (configured by host)")

                    logger.info(
                        f"Bandwidth profiling: {'enabled' if enable_bandwidth else 'disabled'} (configured by host)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get host config: {e}. Using defaults.")

                async def metrics_generator():
                    while True:
                        try:
                            metrics_payload = await collect_gpu_metrics(enable_bandwidth, extra_metrics)

                            if use_adaptive:
                                # Check if we should sample based on adaptive strategy
                                should_sample = False

                                # Check GPU utilization changes
                                for gpu in metrics_payload.get("gpus_metrics", []):
                                    if await adaptive_sampler.should_sample("gpu_util", gpu["gpu_util"]):
                                        should_sample = True
                                        adaptive_sampler.update_metric("gpu_util", gpu["gpu_util"])
                                    if await adaptive_sampler.should_sample("memory", gpu["mem_util"]):
                                        should_sample = True
                                        adaptive_sampler.update_metric("memory", gpu["mem_util"])

                                # Check bandwidth changes if enabled
                                if enable_bandwidth:
                                    for gpu in metrics_payload.get("gpus_metrics", []):
                                        total_bw = gpu.get("dram_bw_gbps_rx", 0) + gpu.get("dram_bw_gbps_tx", 0)
                                        if await adaptive_sampler.should_sample("bandwidth", total_bw):
                                            should_sample = True
                                            adaptive_sampler.update_metric("bandwidth", total_bw)

                                # Check system metrics
                                system_metrics = metrics_payload.get("system_metrics", {})
                                if await adaptive_sampler.should_sample("system", system_metrics.get("dram_util", 0)):
                                    should_sample = True
                                    adaptive_sampler.update_metric("system", system_metrics.get("dram_util", 0))

                                if should_sample:
                                    grpc_update = dict_to_grpc_metrics_update(hostname, metrics_payload)
                                    grpc_update.timestamp = time.time()
                                    yield grpc_update

                                await asyncio.sleep(0.2)  # Minimum 200ms interval
                            else:
                                # Fixed frequency sampling
                                grpc_update = dict_to_grpc_metrics_update(hostname, metrics_payload)
                                grpc_update.timestamp = time.time()
                                yield grpc_update
                                await asyncio.sleep(freq_ms / 1000.0)

                        except Exception as e:
                            logger.error(f"Error in metrics generator: {e}")
                            break

                # Start the metrics streaming and display with initial empty payload
                metrics_payload = {"gpus_metrics": [], "system_metrics": {}}

                # Create console for rich display
                console = Console()

                with Live(display_gpu_info(metrics_payload), refresh_per_second=4, console=console) as live:
                    # Create a task for updating the display
                    async def update_display():
                        while True:
                            try:
                                metrics_payload = await collect_gpu_metrics(enable_bandwidth, extra_metrics)
                                live.update(display_gpu_info(metrics_payload))
                                # Display updates at fixed rate regardless of sampling mode
                                await asyncio.sleep(0.25)  # 4Hz display update
                            except Exception as e:
                                live.console.print(f"Error updating display: {e}")
                                break

                    # Run both the display update and metrics streaming concurrently
                    display_task = asyncio.create_task(update_display())

                    try:
                        # Start the metrics streaming
                        await stub.StreamMetrics(metrics_generator())
                    finally:
                        display_task.cancel()
                        try:
                            await display_task
                        except asyncio.CancelledError:
                            pass

                    # The streaming call has completed
                    logger.info("Metrics streaming completed")

        except grpc.aio.AioRpcError as e:
            logger.warning(f"gRPC error: {e.code()} - {e.details()}. Retrying in {retry_delay}s...")
        except Exception as e:
            logger.error(f"Error in client: {e}. Retrying in {retry_delay}s...")

        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)  # Exponential backoff up to 60s


def create_client_lifespan(head_address: str, enable_bandwidth: bool, extra_metrics: list[str] = []) -> FastAPI:
    """Create FastAPI app with resilient client context"""

    @asynccontextmanager
    async def resilient_client_context(app: FastAPI):
        """Context manager for ResilientClient"""
        task = asyncio.create_task(run_client_node(head_address, enable_bandwidth, extra_metrics))
        yield

        task.cancel()

    return resilient_client_context

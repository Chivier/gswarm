import asyncio
import grpc
import nvitop
import platform
import time
from loguru import logger
from typing import List, Dict, Any

from rich.live import Live
from rich.table import Table

# Import generated protobuf classes
try:
    from . import profiler_pb2
    from . import profiler_pb2_grpc
except ImportError:
    logger.error("gRPC protobuf files not found. Please run 'python generate_grpc.py' first.")
    raise


def display_gpu_info(payload: Dict[str, Any]):
    table = Table()
    table.add_column("GPU ID", justify="center")
    table.add_column("GPU Name", justify="center")
    table.add_column("GPU Utilization", justify="center")
    table.add_column("Memory Utilization", justify="center")
    table.add_column("DRAM Bandwidth RX", justify="center")
    table.add_column("DRAM Bandwidth TX", justify="center")
    table.add_column("NVLink Bandwidth RX", justify="center")
    table.add_column("NVLink Bandwidth TX", justify="center")

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

        table.add_row(
            str(gpu_id),
            str(gpu_name),
            f"{gpu_util:.2f}%",
            f"{mem_util:.2f}%",
            f"{dram_bw_rx:.2f} Kbps",
            f"{dram_bw_tx:.2f} Kbps",
            f"{nvlink_bw_rx:.2f} Kbps",
            f"{nvlink_bw_tx:.2f} Kbps",
        )
    return table


async def collect_gpu_metrics(enable_bandwidth: bool) -> Dict[str, Any]:
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
                "dram_bw_gbps_rx": 0.0,
                "dram_bw_gbps_tx": 0.0,
                "nvlink_bw_gbps_rx": 0.0,
                "nvlink_bw_gbps_tx": 0.0,
            }

            try:
                gpu_metric["gpu_util"] = device.gpu_percent() / 100.0
            except (AttributeError, NotImplementedError):
                logger.debug(f"GPU utilization not available for device {i}")

            try:
                gpu_metric["mem_util"] = device.memory_percent() / 100.0
            except (AttributeError, NotImplementedError):
                logger.debug(f"Memory utilization not available for device {i}")

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
            payload["gpus_metrics"].append({
                "physical_idx": i,
                "name": device.name() if hasattr(device, 'name') else f"GPU_{i}",
                "gpu_util": 0.0,
                "mem_util": 0.0,
                "dram_bw_gbps_rx": 0.0,
                "dram_bw_gbps_tx": 0.0,
                "nvlink_bw_gbps_rx": 0.0,
                "nvlink_bw_gbps_tx": 0.0,
            })

    return payload


def dict_to_grpc_metrics_update(hostname: str, payload: Dict[str, Any]) -> profiler_pb2.MetricsUpdate:
    """Convert dictionary payload to gRPC MetricsUpdate message"""
    gpu_metrics = []
    for gpu in payload.get("gpus_metrics", []):
        gpu_metrics.append(profiler_pb2.GPUMetric(
            physical_idx=gpu["physical_idx"],
            name=gpu["name"],
            gpu_util=gpu["gpu_util"],
            mem_util=gpu["mem_util"],
            dram_bw_gbps_rx=gpu.get("dram_bw_gbps_rx", 0.0),
            dram_bw_gbps_tx=gpu.get("dram_bw_gbps_tx", 0.0),
            nvlink_bw_gbps_rx=gpu.get("nvlink_bw_gbps_rx", 0.0),
            nvlink_bw_gbps_tx=gpu.get("nvlink_bw_gbps_tx", 0.0),
        ))
    
    p2p_links = []
    for link in payload.get("p2p_links", []):
        p2p_links.append(profiler_pb2.P2PLink(
            local_gpu_physical_id=link["local_gpu_physical_id"],
            local_gpu_name=link["local_gpu_name"],
            remote_gpu_physical_id=link["remote_gpu_physical_id"],
            remote_gpu_name=link["remote_gpu_name"],
            type=link["type"],
            aggregated_max_bandwidth_gbps=link["aggregated_max_bandwidth_gbps"],
        ))
    
    return profiler_pb2.MetricsUpdate(
        hostname=hostname,
        gpus_metrics=gpu_metrics,
        p2p_links=p2p_links
    )


async def run_client_node(head_address: str, freq_ms: int, enable_bandwidth: bool):
    hostname = platform.node()
    
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
            gpu_infos.append(profiler_pb2.GPUInfo(
                physical_idx=i,
                name=dev.name()
            ))
        
        initial_info = profiler_pb2.InitialInfo(
            hostname=hostname,
            gpus=gpu_infos
        )

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
                
                # Start streaming metrics
                async def metrics_generator():
                    while True:
                        try:
                            metrics_payload = await collect_gpu_metrics(enable_bandwidth)
                            grpc_update = dict_to_grpc_metrics_update(hostname, metrics_payload)
                            yield grpc_update
                            await asyncio.sleep(freq_ms / 1000.0)
                        except Exception as e:
                            logger.error(f"Error in metrics generator: {e}")
                            break
                
                # Start the metrics streaming and display
                metrics_payload = {}
                with Live(display_gpu_info(metrics_payload), refresh_per_second=4) as live:
                    # Start the metrics streaming
                    stream_response = await stub.StreamMetrics(metrics_generator())
                    
                    # The streaming call has completed
                    logger.info("Metrics streaming completed")

        except grpc.aio.AioRpcError as e:
            logger.warning(f"gRPC error: {e.code()} - {e.details()}. Retrying in {retry_delay}s...")
        except Exception as e:
            logger.error(f"Error in client: {e}. Retrying in {retry_delay}s...")

        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)  # Exponential backoff up to 60s


def start_client_node_sync(head_address: str, freq_ms: int, enable_bandwidth_client: bool):
    try:
        asyncio.run(run_client_node(head_address, freq_ms, enable_bandwidth_client))
    except KeyboardInterrupt:
        logger.info("Client shutdown requested.")
    finally:
        logger.info("Client exiting.")

import asyncio
import websockets
import nvitop
import platform
import time
import json
from loguru import logger
from typing import List, Dict, Any

from rich.live import Live
from rich.table import Table


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

    # Create a mapping using device index
    device_info = {i: {"device": dev, "physical_idx": i, "name": dev.name()} for i, dev in enumerate(devices)}

    for i, device in enumerate(devices):
        try:
            # Get basic GPU metrics using more fundamental NVML calls
            gpu_metric = {
                "physical_idx": i,
                "name": device.name(),
                "gpu_util": 0.0,  # We'll try to get this if available
                "mem_util": 0.0,  # We'll try to get this if available
            }

            # Try to get GPU utilization if available
            try:
                gpu_metric["gpu_util"] = device.gpu_percent() / 100.0
                # logger.info(f"Device GPU utilization: {gpu_metric['gpu_util']}")
            except (AttributeError, NotImplementedError):
                logger.debug(f"GPU utilization not available for device {i}")

            # Try to get memory utilization if available
            try:
                gpu_metric["mem_util"] = device.memory_percent() / 100.0
                # logger.info(f"Device memory utilization: {gpu_metric['mem_util']}")
            except (AttributeError, NotImplementedError):
                logger.debug(f"Memory utilization not available for device {i}")

            if enable_bandwidth:
                # DRAM (PCIe) Bandwidth - only if available
                try:
                    bw_kbps = device.pcie_throughput()

                    if bw_kbps[0] is not None:
                        gpu_metric["dram_bw_gbps_rx"] = bw_kbps[0]
                    else:
                        gpu_metric["dram_bw_gbps_rx"] = 0.0
                    if bw_kbps[1] is not None:
                        gpu_metric["dram_bw_gbps_tx"] = bw_kbps[1]
                    else:
                        gpu_metric["dram_bw_gbps_tx"] = 0.0
                    # logger.info(f"Device PCIe bandwidth (Receive): RX {gpu_metric['dram_bw_gbps_rx']} Gbps")
                    # logger.info(f"Device PCIe bandwidth (Transmit): TX {gpu_metric['dram_bw_gbps_tx']} Gbps")

                except (AttributeError, NotImplementedError):
                    gpu_metric["dram_bw_gbps"] = 0.0
                    logger.debug(f"PCIe bandwidth not available for device {i}")

                # NVLink information - only if available
                try:
                    # processed_nvlink_pairs: Dict[frozenset, float] = {}

                    # for link_info in device.nvlink_information():
                    #     if link_info.is_active:
                    #         remote_idx = link_info.remote_device_index
                    #         if remote_idx is not None:
                    #             pair_key = frozenset((i, remote_idx))
                    #             current_pair_bw = processed_nvlink_pairs.get(pair_key, 0.0)

                    #             try:
                    #                 speed_value = float(link_info.speed.split()[0])
                    #                 current_pair_bw += speed_value
                    #             except (ValueError, AttributeError):
                    #                 logger.debug(f"Could not parse NVLink speed for device {i}")

                    #             processed_nvlink_pairs[pair_key] = current_pair_bw
                    link_throughput = device.nvlink_total_throughput()
                    if link_throughput[0] is not None:
                        gpu_metric["nvlink_bw_gbps_tx"] = link_throughput[0]
                    else:
                        gpu_metric["nvlink_bw_gbps_tx"] = link_throughput[1]

                except (AttributeError, NotImplementedError):
                    logger.debug(f"NVLink information not available for device {i}")

                # # Format links for payload
                # for pair_key, total_bw_gbps in processed_nvlink_pairs.items():
                #     gpu_idx1, gpu_idx2 = list(pair_key)
                #     if i == min(gpu_idx1, gpu_idx2):
                #         local_idx, remote_idx = (gpu_idx1, gpu_idx2) if gpu_idx1 == i else (gpu_idx2, gpu_idx1)

                #         payload["p2p_links"].append({
                #             "local_gpu_physical_id": local_idx,
                #             "local_gpu_name": device_info[local_idx]["name"],
                #             "remote_gpu_physical_id": remote_idx,
                #             "remote_gpu_name": device_info[remote_idx]["name"],
                #             "type": "NVLink",
                #             "aggregated_max_bandwidth_gbps": total_bw_gbps
                #         })

            payload["gpus_metrics"].append(gpu_metric)

        except Exception as e:
            logger.warning(f"Error collecting metrics for device {i}: {e}")
            # Add a basic metric entry even if we couldn't get all the data
            payload["gpus_metrics"].append(
                {
                    "physical_idx": i,
                    "name": device.name(),
                    "gpu_util": 0.0,
                    "mem_util": 0.0,
                }
            )

    return payload


async def run_client_node(head_address: str, freq_ms: int, enable_bandwidth: bool):
    hostname = platform.node()
    uri = f"ws://{head_address}/connect/{hostname}"

    # Check if nvitop can find GPUs
    try:
        devices = nvitop.Device.all()
        if not devices:
            logger.error("No NVIDIA GPUs found on this client node. Exiting.")
            return
        logger.info(f"Found {len(devices)} GPU(s) on this client: {[d.name() for d in devices]}")
        initial_gpu_info = {
            "type": "initial_info",
            "gpus": [{"physical_idx": i, "name": dev.name()} for i, dev in enumerate(devices)],
        }

    except nvitop.NVMLError as e:
        logger.error(f"NVML Error: {e}. Ensure NVIDIA drivers are installed and nvitop has permissions.")
        return
    except Exception as e:
        logger.error(f"Error initializing nvitop or finding GPUs: {e}")
        return

    logger.info(f"Attempting to connect to head node at {uri}")
    logger.info(f"Client-side bandwidth data collection: {'Enabled' if enable_bandwidth else 'Disabled'}")

    retry_delay = 5
    while True:  # Connection retry loop
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                logger.info(f"Connected to head node: {uri}")
                retry_delay = 5  # Reset retry delay on successful connection

                # Send initial GPU info
                await websocket.send(json.dumps(initial_gpu_info))
                logger.info("Sent initial GPU info to head node.")

                metrics_payload = {}
                with Live(display_gpu_info(metrics_payload), refresh_per_second=4) as live:
                    while True:
                        metrics_payload = await collect_gpu_metrics(enable_bandwidth)  # Pass enable_bandwidth
                        live.update(display_gpu_info(metrics_payload))
                        message_to_send = {
                            "type": "metrics_update",
                            "payload": metrics_payload,
                        }
                        await websocket.send(json.dumps(message_to_send))
                        # logger.debug(f"Sent metrics update: {len(metrics_payload['gpus_metrics'])} GPUs")
                        await asyncio.sleep(freq_ms / 1000.0)

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Connection closed to {uri}: {e}. Retrying in {retry_delay}s...")
        except ConnectionRefusedError:
            logger.warning(f"Connection refused by {uri}. Retrying in {retry_delay}s...")
        except Exception as e:
            logger.error(f"Error in client: {e}. Retrying in {retry_delay}s...")

        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)  # Exponential backoff up to 60s


# This function will be called by Typer CLI, with `enable_bandwidth_client` passed from there.
def start_client_node_sync(head_address: str, freq_ms: int, enable_bandwidth_client: bool):
    try:
        asyncio.run(run_client_node(head_address, freq_ms, enable_bandwidth_client))
    except KeyboardInterrupt:
        logger.info("Client shutdown requested.")
    finally:
        logger.info("Client exiting.")

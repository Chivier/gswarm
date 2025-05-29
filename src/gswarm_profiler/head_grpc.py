import asyncio
import datetime
import json
import socket
from typing import Dict, List, Any, Tuple
import grpc
from concurrent import futures
from loguru import logger
import psutil
import aiofiles
import os
import nvitop

# Import generated protobuf classes
from . import profiler_pb2
from . import profiler_pb2_grpc


# --- Global State for Head Node ---
class HeadNodeState:
    def __init__(self):
        self.connected_clients: Dict[str, str] = {}  # client_id -> hostname
        self.client_gpu_info: Dict[
            str, List[Dict[str, Any]]
        ] = {}  # client_id -> [{"id": "...", "name": "...", "physical_idx": ...}, ...]
        self.latest_client_data: Dict[str, Dict[str, Any]] = {}  # client_id -> latest_payload

        self.is_profiling: bool = False
        self.profiling_data_frames: List[Dict[str, Any]] = []
        self.output_filename: str = ""
        self.frame_id_counter: int = 0
        self.freq: int = 500
        self.data_lock = asyncio.Lock()
        self.profiling_task: asyncio.Task = None
        self.enable_bandwidth_profiling: bool = False
        self.enable_nvlink_profiling: bool = False

        # New state for accumulated stats per device
        self.dram_total_util: Dict[str, float] = {}
        self.disk_total_util: Dict[str, float] = {}

        self.gpu_total_util: Dict[str, float] = {}
        self.gpu_util_count: Dict[str, int] = {}
        self.gpu_total_memory: Dict[str, float] = {}
        self.gpu_memory_count: Dict[str, int] = {}

        # gRPC specific state
        self.client_streams: Dict[str, Any] = {}  # client_id -> stream context


state = HeadNodeState()


def get_global_gpu_id(hostname: str, device_idx: int, device_name: str) -> str:
    return f"{hostname}:{device_idx}:{device_name}"


class ProfilerServicer(profiler_pb2_grpc.ProfilerServiceServicer):
    async def Connect(self, request: profiler_pb2.InitialInfo, context):
        """Handle client connection and initial GPU info"""
        client_address = context.peer()
        client_id = f"{request.hostname}_{client_address}"

        logger.info(f"Client {client_id} (hostname: {request.hostname}) connecting via gRPC.")

        async with state.data_lock:
            state.connected_clients[client_id] = request.hostname
            state.client_gpu_info[client_id] = []

            for gpu in request.gpus:
                state.client_gpu_info[client_id].append(
                    {
                        "id": get_global_gpu_id(request.hostname, gpu.physical_idx, gpu.name),
                        "name": gpu.name,
                        "physical_idx": gpu.physical_idx,
                        "hostname": request.hostname,
                    }
                )

            logger.info(f"Received initial GPU info from {client_id}: {len(request.gpus)} GPUs")
            log_total_gpus()

        return profiler_pb2.ConnectResponse(
            success=True, message=f"Connected successfully. Registered {len(request.gpus)} GPUs."
        )

    async def StreamMetrics(self, request_iterator, context):
        """Handle streaming metrics from clients"""
        # client_address = context.peer()
        client_id = None

        try:
            async for metrics_update in request_iterator:
                if client_id is None:
                    # Find client_id based on hostname
                    for cid, hostname in state.connected_clients.items():
                        if hostname == metrics_update.hostname:
                            client_id = cid
                            break

                    if client_id is None:
                        logger.warning(f"Received metrics from unknown client: {metrics_update.hostname}")
                        continue

                # Convert gRPC message to dictionary format (similar to original WebSocket format)
                payload = {"gpus_metrics": [], "p2p_links": []}

                for gpu_metric in metrics_update.gpus_metrics:
                    payload["gpus_metrics"].append(
                        {
                            "physical_idx": gpu_metric.physical_idx,
                            "name": gpu_metric.name,
                            "gpu_util": gpu_metric.gpu_util,
                            "mem_util": gpu_metric.mem_util,
                            "dram_bw_gbps_rx": gpu_metric.dram_bw_gbps_rx,
                            "dram_bw_gbps_tx": gpu_metric.dram_bw_gbps_tx,
                            "nvlink_bw_gbps_rx": gpu_metric.nvlink_bw_gbps_rx,
                            "nvlink_bw_gbps_tx": gpu_metric.nvlink_bw_gbps_tx,
                        }
                    )

                for p2p_link in metrics_update.p2p_links:
                    payload["p2p_links"].append(
                        {
                            "local_gpu_physical_id": p2p_link.local_gpu_physical_id,
                            "local_gpu_name": p2p_link.local_gpu_name,
                            "remote_gpu_physical_id": p2p_link.remote_gpu_physical_id,
                            "remote_gpu_name": p2p_link.remote_gpu_name,
                            "type": p2p_link.type,
                            "aggregated_max_bandwidth_gbps": p2p_link.aggregated_max_bandwidth_gbps,
                        }
                    )

                async with state.data_lock:
                    state.latest_client_data[client_id] = payload

        except Exception as e:
            logger.error(f"Error in StreamMetrics for client {client_id}: {e}")
        finally:
            # Clean up when client disconnects
            if client_id:
                async with state.data_lock:
                    if client_id in state.connected_clients:
                        del state.connected_clients[client_id]
                    if client_id in state.client_gpu_info:
                        del state.client_gpu_info[client_id]
                    if client_id in state.latest_client_data:
                        del state.latest_client_data[client_id]
                logger.info(f"Client {client_id} disconnected from gRPC stream.")
                log_total_gpus()

        return profiler_pb2.Empty()

    async def GetStatus(self, request: profiler_pb2.Empty, context):
        """Get current profiler status"""
        return profiler_pb2.StatusResponse(
            freq=state.freq,
            enable_bandwidth_profiling=state.enable_bandwidth_profiling,
            enable_nvlink_profiling=state.enable_nvlink_profiling,
            is_profiling=state.is_profiling,
            output_filename=state.output_filename,
            frame_id_counter=state.frame_id_counter,
            connected_clients=list(state.connected_clients.keys()),
        )

    async def StartProfiling(self, request: profiler_pb2.StartProfilingRequest, context):
        """Start profiling session"""
        if state.is_profiling:
            return profiler_pb2.StartProfilingResponse(
                success=False, message="Profiling is already active.", output_file=""
            )

        async with state.data_lock:
            state.is_profiling = True
            state.profiling_data_frames = []
            state.frame_id_counter = 0
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if request.name:
                state.output_filename = f"{request.name}.json"
            else:
                state.output_filename = f"gswarm_profiler_{timestamp}.json"

            # Clear stale data from previous runs or disconnected clients
            current_connected_ids = list(state.connected_clients.keys())
            state.latest_client_data = {k: v for k, v in state.latest_client_data.items() if k in current_connected_ids}

            # Reset accumulators for overall statistics
            state.gpu_total_util = {}
            state.gpu_util_count = {}
            state.gpu_total_memory = {}
            state.gpu_memory_count = {}

        state.profiling_task = asyncio.create_task(collect_and_store_frame())
        logger.info(f"Profiling started. Output will be saved to {state.output_filename}")
        log_total_gpus()

        return profiler_pb2.StartProfilingResponse(
            success=True, message="Profiling started.", output_file=state.output_filename
        )

    async def StopProfiling(self, request: profiler_pb2.Empty, context):
        """Stop profiling session"""
        if not state.is_profiling:
            return profiler_pb2.StopProfilingResponse(success=False, message="Profiling is not active.")

        logger.info("Stopping profiling...")
        async with state.data_lock:
            state.is_profiling = False

        if state.profiling_task:
            try:
                await asyncio.wait_for(state.profiling_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Profiling task did not finish in time. Data might be incomplete for the last frame.")
                state.profiling_task.cancel()
            except Exception as e:
                logger.error(f"Error during profiling task shutdown: {e}")

        state.profiling_task = None

        return profiler_pb2.StopProfilingResponse(
            success=True,
            message=f"Profiling stopped. Data saved to {state.output_filename if state.output_filename else 'N/A'}",
        )

    async def Exit(self, request: profiler_pb2.Empty, context):
        """Exit head node"""
        logger.info("Exiting head node...")
        state.is_profiling = False
        if state.profiling_task:
            state.profiling_task.cancel()
            state.profiling_task = None
        return profiler_pb2.Empty()


def log_total_gpus():
    total_gpus = sum(len(gpus) for gpus in state.client_gpu_info.values())
    logger.info(f"Total GPUs connected: {total_gpus} across {len(state.client_gpu_info)} client(s).")


async def collect_and_store_frame():
    """Periodically collects data from clients and stores a frame if profiling is active."""
    while state.is_profiling:
        await asyncio.sleep(1)  # Head node frame aggregation interval

        async with state.data_lock:
            if not state.is_profiling:
                break

            state.frame_id_counter += 1
            current_frame: Dict[str, Any] = {
                "frame_id": state.frame_id_counter,
                "time": datetime.datetime.now().isoformat(),
                "gpu_id": [],
                "gpu_util": [],
                "gpu_memory": [],
            }
            if state.enable_bandwidth_profiling:
                current_frame["dram_bandwidth"] = []
                current_frame["dram_bandwidth_rx"] = []
                current_frame["dram_bandwidth_tx"] = []
                current_frame["gpu_bandwidth"] = []

            active_clients_data = {k: v for k, v in state.latest_client_data.items() if k in state.connected_clients}

            for client_id, client_payload in active_clients_data.items():
                if client_id not in state.client_gpu_info or not state.client_gpu_info[client_id]:
                    logger.warning(
                        f"Skipping data for client {client_id} due to missing GPU info during frame collection."
                    )
                    continue
                client_hostname = state.client_gpu_info[client_id][0]["hostname"]

                for gpu_metric in client_payload.get("gpus_metrics", []):
                    gpu_global_id = get_global_gpu_id(client_hostname, gpu_metric["physical_idx"], gpu_metric["name"])
                    current_frame["gpu_id"].append(gpu_global_id)
                    current_frame["gpu_util"].append(f"{gpu_metric['gpu_util']:.2f}")
                    current_frame["gpu_memory"].append(f"{gpu_metric['mem_util']:.2f}")

                    # Accumulate stats for overall average
                    util_value = float(gpu_metric["gpu_util"])
                    mem_value = float(gpu_metric["mem_util"])

                    state.gpu_total_util[gpu_global_id] = state.gpu_total_util.get(gpu_global_id, 0.0) + util_value
                    state.gpu_util_count[gpu_global_id] = state.gpu_util_count.get(gpu_global_id, 0) + 1
                    state.gpu_total_memory[gpu_global_id] = state.gpu_total_memory.get(gpu_global_id, 0.0) + mem_value
                    state.gpu_memory_count[gpu_global_id] = state.gpu_memory_count.get(gpu_global_id, 0) + 1

                    if state.enable_bandwidth_profiling:
                        current_frame["dram_bandwidth_rx"].append(f"{gpu_metric.get('dram_bw_gbps_rx', 0.0):.2f}")
                        current_frame["dram_bandwidth_tx"].append(f"{gpu_metric.get('dram_bw_gbps_tx', 0.0):.2f}")
                        current_frame["dram_bandwidth"].append(
                            str(
                                float(current_frame["dram_bandwidth_rx"][-1])
                                + float(current_frame["dram_bandwidth_tx"][-1])
                            )
                        )

                if state.enable_nvlink_profiling:
                    for p2p_link in client_payload.get("p2p_links", []):
                        source_gpu_global_id = get_global_gpu_id(
                            client_hostname,
                            p2p_link["local_gpu_physical_id"],
                            p2p_link["local_gpu_name"],
                        )
                        target_gpu_global_id = get_global_gpu_id(
                            client_hostname,
                            p2p_link["remote_gpu_physical_id"],
                            p2p_link["remote_gpu_name"],
                        )
                        id1, id2 = sorted([source_gpu_global_id, target_gpu_global_id])

                        link_info = {
                            "id1": id1,
                            "id2": id2,
                            "utilization": f"{p2p_link.get('aggregated_max_bandwidth_gbps', 0.0):.2f}",
                        }
                        if link_info not in current_frame["gpu_bandwidth"]:
                            current_frame["gpu_bandwidth"].append(link_info)

            state.profiling_data_frames.append(current_frame)

    logger.info("Profiling data collection loop finished.")
    if state.output_filename and (state.profiling_data_frames or state.gpu_total_util):
        logger.info(f"Attempting to save data to {state.output_filename}...")

        summary_by_device = {}
        for gpu_id, total_util in state.gpu_total_util.items():
            count = state.gpu_util_count.get(gpu_id, 0)
            if count > 0:
                avg_util = total_util / count
                summary_by_device.setdefault(gpu_id, {})["avg_gpu_util"] = f"{avg_util:.2f}"

        for gpu_id, total_mem in state.gpu_total_memory.items():
            count = state.gpu_memory_count.get(gpu_id, 0)
            if count > 0:
                avg_mem = total_mem / count
                summary_by_device.setdefault(gpu_id, {})["avg_gpu_memory"] = f"{avg_mem:.2f}"

        output_data = {
            "frames": state.profiling_data_frames,
            "summary_by_device": summary_by_device,
        }

        try:
            async with aiofiles.open(state.output_filename, mode="w") as f:
                await f.write(json.dumps(output_data, indent=2))
            logger.info(f"Profiling data successfully saved to {state.output_filename}")

        except Exception as e:
            logger.error(f"Failed to save profiling data: {e}")
    else:
        logger.info("No profiling data to save.")


async def run_grpc_server(host: str, port: int):
    """Run the gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = ProfilerServicer()
    profiler_pb2_grpc.add_ProfilerServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(5)


def run_head_node(host: str, port: int, enable_bandwidth: bool, enable_nvlink: bool, freq: int):
    """Run the head node with gRPC server"""
    logger.info(f"Starting GSwarm Profiler Head Node on {host}:{port} using gRPC")
    logger.info(f"Bandwidth profiling: {'Enabled' if enable_bandwidth else 'Disabled'}")
    state.enable_bandwidth_profiling = enable_bandwidth
    state.enable_nvlink_profiling = enable_nvlink
    state.freq = freq

    # Log own GPUs if any for information
    try:
        local_devices = nvitop.Device.all()
        if local_devices:
            logger.info(f"Head node has {len(local_devices)} local GPU(s): {[d.name() for d in local_devices]}")
    except Exception:
        logger.info("Head node has no local NVIDIA GPUs or nvitop cannot access them.")

    asyncio.run(run_grpc_server(host, port))

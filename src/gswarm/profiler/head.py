import asyncio
import datetime
import json
import socket
from typing import Dict, List, Any, Tuple
import grpc
from google.protobuf.json_format import MessageToDict
from concurrent import futures
from loguru import logger
import psutil
import aiofiles
import os
import nvitop
import time
import sys
from gswarm.profiler.session_manager import SessionManager, ProfilingSession
from gswarm.profiler.persistence import FileBasedStorage
from gswarm.profiler.adaptive_sampler import AdaptiveSampler

from gswarm.profiler.head_common import profiler_stop_cleanup

# Import generated protobuf classes (these will be generated)
try:
    from gswarm.profiler import profiler_pb2
    from gswarm.profiler import profiler_pb2_grpc
except ImportError:
    logger.error("gRPC protobuf files not found. Please run 'python generate_grpc.py' first.")
    raise

from gswarm.utils.draw_metrics import draw_metrics


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
        self.report_filename: str = ""
        self.frame_id_counter: int = 0
        self.data_lock = asyncio.Lock()
        self.profiling_task: asyncio.Task = None
        self.enable_bandwidth_profiling: bool = False
        self.enable_nvlink_profiling: bool = False

        # Sampling configuration
        self.freq: int = 200  # Default 200ms, 0 for adaptive

        # New state for accumulated stats per device
        self.dram_total_util: Dict[str, float] = {}
        self.dram_util_count: Dict[str, int] = {}
        self.disk_total_util: Dict[str, float] = {}
        self.disk_util_count: Dict[str, int] = {}

        self.gpu_total_util: Dict[str, float] = {}
        self.gpu_util_count: Dict[str, int] = {}
        self.gpu_total_memory: Dict[str, float] = {}
        self.gpu_memory_count: Dict[str, int] = {}
        self.gpu_extra_metrics: Dict[str, Dict[str, Any]] = {}

        # Add these new attributes:
        self.session_manager = SessionManager()
        self.active_sessions: Dict[str, ProfilingSession] = {}
        self.client_last_seen: Dict[str, float] = {}
        self.client_health_timeout = 30  # seconds
        self.adaptive_sampler = AdaptiveSampler()

        # Report generation related attributes
        self.report_metrics = None

        # Time consumption of each request tracking
        self.time_consumption_data: Dict[str, List[Dict[str, float]]] = {}


state = HeadNodeState()


def get_global_gpu_id(hostname: str, device_idx: int, device_name: str) -> str:
    return f"{hostname}:{device_idx}:{device_name}"


# --- gRPC Server ---
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
                payload = {"gpus_metrics": [], "p2p_links": [], "system_metrics": {}}

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
                            "extra_metrics": MessageToDict(gpu_metric.extra_metrics),  # Use Struct for extra metrics
                        }
                    )

                # Add system metrics if available
                if metrics_update.HasField("system_metrics"):
                    payload["system_metrics"] = {
                        "dram_util": metrics_update.system_metrics.dram_util,
                        "disk_util": metrics_update.system_metrics.disk_util,
                    }

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
                state.report_filename = f"{request.name}.png"
            else:
                state.output_filename = f"gswarm_profiler_{timestamp}.json"
                state.report_filename = f"gswarm_profiler_{timestamp}.png"

            # Clear stale data from previous runs or disconnected clients
            current_connected_ids = list(state.connected_clients.keys())
            state.latest_client_data = {k: v for k, v in state.latest_client_data.items() if k in current_connected_ids}

            # Reset accumulators for overall statistics
            state.gpu_total_util = {}
            state.gpu_util_count = {}
            state.gpu_total_memory = {}
            state.gpu_memory_count = {}
            state.dram_total_util = {}
            state.dram_util_count = {}
            state.disk_total_util = {}
            state.disk_util_count = {}
            state.gpu_extra_metrics = {}

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

        await profiler_stop_cleanup(state)

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

    async def ReadClusterStatus(self, request: profiler_pb2.ReadClusterStatusRequest, context):
        """Read status for all nodes in the cluster"""
        try:
            nodes = []

            # Iterate through all connected clients and build node status
            async with state.data_lock:
                for client_id, hostname in state.connected_clients.items():
                    if client_id in state.client_gpu_info and client_id in state.latest_client_data:
                        gpus = []
                        client_data = state.latest_client_data[client_id]

                        for gpu_metric in client_data.get("gpus_metrics", []):
                            gpu_status = profiler_pb2.GPUStatus(
                                gpu_id=gpu_metric["physical_idx"],
                                utilization=gpu_metric["gpu_util"],
                                memory_used=gpu_metric.get(
                                    "mem_used_mb",
                                    int(gpu_metric.get("mem_util", 0) * gpu_metric.get("mem_total_mb", 16384)),
                                ),
                                memory_total=gpu_metric.get(
                                    "mem_total_mb", 16384
                                ),  # Use actual value or fallback to 16GB
                                dram_bandwidth=gpu_metric.get("dram_bw_gbps_rx", 0)
                                + gpu_metric.get("dram_bw_gbps_tx", 0),
                                nvlink_bandwidth=gpu_metric.get("nvlink_bw_gbps_rx", 0)
                                + gpu_metric.get("nvlink_bw_gbps_tx", 0),
                                temperature=-1,  # Not available in current metrics
                                power=-1,  # Not available in current metrics
                            )
                            gpus.append(gpu_status)

                        node_status = profiler_pb2.NodeStatus(node_id=hostname, gpus=gpus)
                        nodes.append(node_status)

            return profiler_pb2.ClusterStatusResponse(
                success=True, message="Cluster status retrieved successfully", cluster_id="default", nodes=nodes
            )
        except Exception as e:
            logger.error(f"Error reading cluster status: {e}")
            return profiler_pb2.ClusterStatusResponse(
                success=False, message=f"Failed to read cluster status: {str(e)}", cluster_id="default", nodes=[]
            )

    async def ReadNodeStatus(self, request: profiler_pb2.ReadNodeStatusRequest, context):
        """Read status for a specific node"""
        try:
            target_hostname = request.node_id

            async with state.data_lock:
                # Find the client with matching hostname
                target_client_id = None
                for client_id, hostname in state.connected_clients.items():
                    if hostname == target_hostname:
                        target_client_id = client_id
                        break

                if not target_client_id or target_client_id not in state.latest_client_data:
                    return profiler_pb2.NodeStatusResponse(
                        success=False,
                        message=f"Node {target_hostname} not found or no data available",
                        node_status=profiler_pb2.NodeStatus(),
                    )

                client_data = state.latest_client_data[target_client_id]
                gpus = []

                for gpu_metric in client_data.get("gpus_metrics", []):
                    gpu_status = profiler_pb2.GPUStatus(
                        gpu_id=gpu_metric["physical_idx"],
                        utilization=gpu_metric["gpu_util"],
                        memory_used=gpu_metric.get(
                            "mem_used_mb", int(gpu_metric.get("mem_util", 0) * gpu_metric.get("mem_total_mb", 16384))
                        ),
                        memory_total=gpu_metric.get("mem_total_mb", 16384),  # Use actual value or fallback to 16GB
                        dram_bandwidth=gpu_metric.get("dram_bw_gbps_rx", 0) + gpu_metric.get("dram_bw_gbps_tx", 0),
                        nvlink_bandwidth=gpu_metric.get("nvlink_bw_gbps_rx", 0)
                        + gpu_metric.get("nvlink_bw_gbps_tx", 0),
                        temperature=-1,
                        power=-1,
                    )
                    gpus.append(gpu_status)

                node_status = profiler_pb2.NodeStatus(node_id=target_hostname, gpus=gpus)

            return profiler_pb2.NodeStatusResponse(
                success=True,
                message=f"Node status for {target_hostname} retrieved successfully",
                node_status=node_status,
            )
        except Exception as e:
            logger.error(f"Error reading node status: {e}")
            return profiler_pb2.NodeStatusResponse(
                success=False, message=f"Failed to read node status: {str(e)}", node_status=profiler_pb2.NodeStatus()
            )

    async def SetBandwidthProfiling(self, request: profiler_pb2.SetBandwidthProfilingRequest, context):
        """Enable or disable bandwidth profiling"""
        try:
            state.enable_bandwidth_profiling = request.enable
            status = "enabled" if request.enable else "disabled"
            logger.info(f"Bandwidth profiling {status}")

            return profiler_pb2.SetBandwidthProfilingResponse(
                success=True, message=f"Bandwidth profiling {status} successfully"
            )
        except Exception as e:
            logger.error(f"Error setting bandwidth profiling: {e}")
            return profiler_pb2.SetBandwidthProfilingResponse(
                success=False, message=f"Failed to set bandwidth profiling: {str(e)}"
            )

    async def SetNVLinkProfiling(self, request: profiler_pb2.SetNVLinkProfilingRequest, context):
        """Enable or disable NVLink profiling"""
        try:
            state.enable_nvlink_profiling = request.enable
            status = "enabled" if request.enable else "disabled"
            logger.info(f"NVLink profiling {status}")

            return profiler_pb2.SetNVLinkProfilingResponse(
                success=True, message=f"NVLink profiling {status} successfully"
            )
        except Exception as e:
            logger.error(f"Error setting NVLink profiling: {e}")
            return profiler_pb2.SetNVLinkProfilingResponse(
                success=False, message=f"Failed to set NVLink profiling: {str(e)}"
            )


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
                "dram_util": [],  # Add system DRAM utilization
                "disk_util": [],  # Add disk utilization
                "extra_metrics": [],
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
                    current_frame["extra_metrics"].append(gpu_metric.get("extra_metrics", {}))
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

                # Add system metrics for this client
                system_metrics = client_payload.get("system_metrics", {})
                if system_metrics:
                    client_key = f"{client_hostname}"
                    current_frame["dram_util"].append(f"{system_metrics.get('dram_util', 0.0):.2f}")
                    current_frame["disk_util"].append(f"{system_metrics.get('disk_util', 0.0):.2f}")

                    # Accumulate for averages
                    dram_util_value = float(system_metrics.get("dram_util", 0.0))
                    disk_util_value = float(system_metrics.get("disk_util", 0.0))

                    state.dram_total_util[client_key] = state.dram_total_util.get(client_key, 0.0) + dram_util_value
                    state.dram_util_count[client_key] = state.dram_util_count.get(client_key, 0) + 1
                    state.disk_total_util[client_key] = state.disk_total_util.get(client_key, 0.0) + disk_util_value
                    state.disk_util_count[client_key] = state.disk_util_count.get(client_key, 0) + 1

            state.profiling_data_frames.append(current_frame)

    logger.info("Profiling data collection loop finished.")
    if state.output_filename and (state.profiling_data_frames or state.gpu_total_util):
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

        # Add system metrics to summary
        summary_by_client = {}
        for client_key, total_dram in state.dram_total_util.items():
            count = state.dram_util_count.get(client_key, 0)
            if count > 0:
                avg_dram = total_dram / count
                summary_by_client.setdefault(client_key, {})["avg_dram_util"] = f"{avg_dram:.2f}"

        for client_key, total_disk in state.disk_total_util.items():
            count = state.disk_util_count.get(client_key, 0)
            if count > 0:
                avg_disk = total_disk / count
                summary_by_client.setdefault(client_key, {})["avg_disk_util"] = f"{avg_disk:.2f}"

        output_data = {
            "summary_by_device": summary_by_device,
            "summary_by_client": summary_by_client,  # Add client-level system metrics summary
        }
        logger.info("Summary of profiling data collected:")
        logger.info(json.dumps(output_data, indent=2))

        return output_data
    else:
        logger.info("No profiling data to save.")
        return {}


async def health_monitor_task():
    """Periodically check client health"""
    while True:
        try:
            await asyncio.sleep(10)
            disconnected = await state.check_client_health()
            if disconnected:
                logger.info(f"Health check found {len(disconnected)} disconnected clients")
        except Exception as e:
            logger.error(f"Error in health monitor: {e}")


def check_port_availability(host: str, port: int) -> bool:
    """Check if a port is available for binding"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            sock.close()
            return True
        except OSError:
            return False


async def run_grpc_server(host: str, port: int):
    """Run the gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = ProfilerServicer()
    profiler_pb2_grpc.add_ProfilerServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"

    try:
        server.add_insecure_port(listen_addr)
    except Exception as e:
        logger.error(f"Failed to bind gRPC server to {listen_addr}: {e}")
        raise

    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(5)


async def run_both_servers(grpc_host: str, grpc_port: int, http_host: str, http_port: int):
    """Run both gRPC and HTTP servers concurrently"""
    from gswarm.profiler.http_api import run_http_server

    # Create tasks for both servers
    grpc_task = asyncio.create_task(run_grpc_server(grpc_host, grpc_port))
    http_task = asyncio.create_task(run_http_server(http_host, http_port))

    try:
        # Wait for both servers to run
        await asyncio.gather(grpc_task, http_task)
    except KeyboardInterrupt:
        logger.info("Shutting down both servers...")
        grpc_task.cancel()
        http_task.cancel()
        try:
            await asyncio.gather(grpc_task, http_task, return_exceptions=True)
        except Exception:
            pass


def run_head_node(
    host: str, port: int, enable_bandwidth: bool, enable_nvlink: bool, http_port: int = None, freq: int = 200
):
    """Run the head node with gRPC server and optionally HTTP server"""
    # Check port availability before starting
    if not check_port_availability(host, port):
        logger.error(f"Port {port} is already in use on {host}")
        logger.info("Please choose a different port or stop the process using this port.")
        logger.info(f"You can find the process using: lsof -i :{port} or netstat -tulpn | grep :{port}")
        raise RuntimeError(f"Port {port} is already in use on {host}")

    if http_port and not check_port_availability(host, http_port):
        logger.error(f"HTTP port {http_port} is already in use on {host}")
        logger.info("Please choose a different HTTP port or stop the process using this port.")
        logger.info(f"You can find the process using: lsof -i :{http_port} or netstat -tulpn | grep :{http_port}")
        raise RuntimeError(f"HTTP port {http_port} is already in use on {host}")

    logger.info(f"Starting GSwarm Profiler Head Node on {host}:{port} using gRPC")
    if http_port:
        logger.info(f"HTTP API will be available on {host}:{http_port}")
    logger.info(f"Bandwidth profiling: {'Enabled' if enable_bandwidth else 'Disabled'}")

    # Set sampling configuration
    state.freq = freq
    if freq == 0:
        logger.info(f"Using adaptive sampling strategy (similar to WandB)")
    else:
        logger.info(f"Using fixed frequency sampling: {freq}ms")

    state.enable_bandwidth_profiling = enable_bandwidth
    state.enable_nvlink_profiling = enable_nvlink

    # Log own GPUs if any for information
    try:
        local_devices = nvitop.Device.all()
        if local_devices:
            logger.info(f"Head node has {len(local_devices)} local GPU(s): {[d.name() for d in local_devices]}")
    except Exception:
        logger.info("Head node has no local NVIDIA GPUs or nvitop cannot access them.")

    # Add this before starting the server:
    async def initialize_and_run():
        await state.session_manager.initialize()  # Fixed: call initialize on session_manager

        # Then run the servers as before
        if http_port:
            await run_both_servers(host, port, host, http_port)
        else:
            await run_grpc_server(host, port)

    try:
        asyncio.run(initialize_and_run())
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start head node: {e}")
        sys.exit(1)

import asyncio
import datetime
import json
from typing import Dict, List, Any, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from loguru import logger
import uvicorn
import aiofiles
import os
import nvitop # For head node to potentially list its own local GPUs if any

# --- Global State for Head Node ---
# (In a real app, you might use a class structure or a more robust state management)
class HeadNodeState:
    def __init__(self):
        self.connected_clients: Dict[str, WebSocket] = {} # client_id -> websocket
        self.client_gpu_info: Dict[str, List[Dict[str, Any]]] = {} # client_id -> [{"id": "...", "name": "...", "physical_idx": ...}, ...]
        self.latest_client_data: Dict[str, Dict[str, Any]] = {} # client_id -> latest_payload
        
        self.is_profiling: bool = False
        self.profiling_data_frames: List[Dict[str, Any]] = []
        self.output_filename: str = ""
        self.frame_id_counter: int = 0
        self.freq: int = 500
        self.data_lock = asyncio.Lock()
        self.profiling_task: asyncio.Task = None
        self.enable_bandwidth_profiling: bool = False

        # New state for accumulated stats per device
        self.gpu_total_util: Dict[str, float] = {}
        self.gpu_util_count: Dict[str, int] = {}
        self.gpu_total_memory: Dict[str, float] = {}
        self.gpu_memory_count: Dict[str, int] = {}

state = HeadNodeState()
head_app = FastAPI()

def get_global_gpu_id(hostname: str, device_idx: int, device_name: str) -> str:
    return f"{hostname}:{device_idx}:{device_name}"

@head_app.post("/state")
async def get_state():
    return {
        "freq": state.freq,
        "enable_bandwidth_profiling": state.enable_bandwidth_profiling,
        "is_profiling": state.is_profiling,
        "profiling_data_frames": state.profiling_data_frames,
        "output_filename": state.output_filename,
        "frame_id_counter": state.frame_id_counter,
        "connected_clients": list(state.connected_clients.keys()),
        "client_gpu_info": state.client_gpu_info,
        "latest_client_data": state.latest_client_data
    }

@head_app.websocket("/connect/{hostname}")
async def websocket_endpoint(websocket: WebSocket, hostname: str):
    client_id = f"{hostname}_{websocket.client.host}_{websocket.client.port}"
    await websocket.accept()
    logger.info(f"Client {client_id} (hostname: {hostname}) connected.")
    
    async with state.data_lock:
        state.connected_clients[client_id] = websocket
        # Client should send its initial GPU setup first
        try:
            initial_gpu_data = await websocket.receive_json()
            if initial_gpu_data.get("type") == "initial_info":
                state.client_gpu_info[client_id] = []
                for gpu in initial_gpu_data.get("gpus", []):
                    state.client_gpu_info[client_id].append({
                        "id": get_global_gpu_id(hostname, gpu["physical_idx"], gpu["name"]),
                        "name": gpu["name"],
                        "physical_idx": gpu["physical_idx"],
                        "hostname": hostname
                    })
                logger.info(f"Received initial GPU info from {client_id}: {len(initial_gpu_data.get('gpus', []))} GPUs")
                log_total_gpus()
            else:
                logger.warning(f"Client {client_id} did not send initial_info first. Disconnecting.")
                await websocket.close()
                if client_id in state.connected_clients: del state.connected_clients[client_id]
                if client_id in state.client_gpu_info: del state.client_gpu_info[client_id]
                return

        except WebSocketDisconnect:
            logger.warning(f"Client {client_id} disconnected before sending initial info.")
            async with state.data_lock:
                if client_id in state.connected_clients: del state.connected_clients[client_id]
                if client_id in state.client_gpu_info: del state.client_gpu_info[client_id]
                if client_id in state.latest_client_data: del state.latest_client_data[client_id]
            log_total_gpus()
            return
        except Exception as e:
            logger.error(f"Error processing initial info from {client_id}: {e}")
            await websocket.close()
            async with state.data_lock:
                if client_id in state.connected_clients: del state.connected_clients[client_id]
                if client_id in state.client_gpu_info: del state.client_gpu_info[client_id]
            log_total_gpus()
            return


    try:
        while True:
            data = await websocket.receive_json() # Expecting data from client
            if data.get("type") == "metrics_update":
                async with state.data_lock:
                    state.latest_client_data[client_id] = data["payload"]
                # logger.debug(f"Received data from {client_id}")
    except WebSocketDisconnect:
        logger.warning(f"Client {client_id} (hostname: {hostname}) disconnected.")
    except Exception as e:
        logger.error(f"Error with client {client_id}: {e}")
    finally:
        async with state.data_lock:
            if client_id in state.connected_clients:
                del state.connected_clients[client_id]
            if client_id in state.client_gpu_info:
                del state.client_gpu_info[client_id]
            if client_id in state.latest_client_data:
                del state.latest_client_data[client_id]
        log_total_gpus()

def log_total_gpus():
    total_gpus = sum(len(gpus) for gpus in state.client_gpu_info.values())
    logger.info(f"Total GPUs connected: {total_gpus} across {len(state.client_gpu_info)} client(s).")


async def collect_and_store_frame():
    """Periodically collects data from clients and stores a frame if profiling is active."""
    while state.is_profiling:
        await asyncio.sleep(1) # Head node frame aggregation interval (can be different from client's freq)
        
        async with state.data_lock:
            if not state.is_profiling: # Check again after acquiring lock
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
                # Ensure client_id is still valid and has corresponding GPU info
                if client_id not in state.client_gpu_info or not state.client_gpu_info[client_id]:
                    logger.warning(f"Skipping data for client {client_id} due to missing GPU info during frame collection.")
                    continue
                client_hostname = state.client_gpu_info[client_id][0]["hostname"] # Assuming all GPUs from one client have same hostname
                
                for gpu_metric in client_payload.get("gpus_metrics", []):
                    gpu_global_id = get_global_gpu_id(client_hostname, gpu_metric["physical_idx"], gpu_metric["name"])
                    current_frame["gpu_id"].append(gpu_global_id)
                    current_frame["gpu_util"].append(f"{gpu_metric['gpu_util']:.2f}")
                    current_frame["gpu_memory"].append(f"{gpu_metric['mem_util']:.2f}")
                    
                    # Accumulate stats for overall average
                    util_value = float(gpu_metric['gpu_util'])
                    mem_value = float(gpu_metric['mem_util'])

                    state.gpu_total_util[gpu_global_id] = state.gpu_total_util.get(gpu_global_id, 0.0) + util_value
                    state.gpu_util_count[gpu_global_id] = state.gpu_util_count.get(gpu_global_id, 0) + 1
                    state.gpu_total_memory[gpu_global_id] = state.gpu_total_memory.get(gpu_global_id, 0.0) + mem_value
                    state.gpu_memory_count[gpu_global_id] = state.gpu_memory_count.get(gpu_global_id, 0) + 1
                    
                    if state.enable_bandwidth_profiling:
                        current_frame["dram_bandwidth_rx"].append(f"{gpu_metric.get('dram_bw_gbps_rx', 0.0):.2f}")
                        current_frame["dram_bandwidth_tx"].append(f"{gpu_metric.get('dram_bw_gbps_tx', 0.0):.2f}") 
                        current_frame["dram_bandwidth"].append(str(float(current_frame["dram_bandwidth_rx"][-1]) + float(current_frame["dram_bandwidth_tx"][-1]))) 

                if state.enable_bandwidth_profiling:
                    for p2p_link in client_payload.get("p2p_links", []):
                        source_gpu_global_id = get_global_gpu_id(
                            client_hostname, 
                            p2p_link["local_gpu_physical_id"],
                            p2p_link["local_gpu_name"]
                        )
                        # NVLink is intra-node, so remote GPU is on the same host
                        target_gpu_global_id = get_global_gpu_id(
                            client_hostname,
                            p2p_link["remote_gpu_physical_id"],
                            p2p_link["remote_gpu_name"]
                        )
                        # Ensure "id1" < "id2" lexicographically to avoid duplicates like (A,B) and (B,A)
                        id1, id2 = sorted([source_gpu_global_id, target_gpu_global_id])

                        link_info = {
                            "id1": id1,
                            "id2": id2,
                            # As discussed, "utilization" here means max capacity for P2P
                            "utilization": f"{p2p_link.get('aggregated_max_bandwidth_gbps', 0.0):.2f}" 
                        }
                        # Avoid duplicate links in the same frame
                        if link_info not in current_frame["gpu_bandwidth"]:
                             current_frame["gpu_bandwidth"].append(link_info)


            state.profiling_data_frames.append(current_frame)
            # logger.debug(f"Stored frame {state.frame_id_counter}")

    logger.info("Profiling data collection loop finished.")
    if state.output_filename and (state.profiling_data_frames or state.gpu_total_util): # Save if there are frames or accumulated stats
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
            "summary_by_device": summary_by_device
        }
        
        try:
            async with aiofiles.open(state.output_filename, mode='w') as f:
                await f.write(json.dumps(output_data, indent=2))
            logger.info(f"Profiling data successfully saved to {state.output_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save profiling data: {e}")
    else:
        logger.info("No profiling data to save.")


@head_app.post("/start")
async def start_profiling():
    if state.is_profiling:
        raise HTTPException(status_code=400, detail="Profiling is already active.")

    async with state.data_lock:
        state.is_profiling = True
        state.profiling_data_frames = []
        state.frame_id_counter = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        state.output_filename = f"gswarm_profiler_{timestamp}.json"
        
        # Clear stale data from previous runs or disconnected clients
        current_connected_ids = list(state.connected_clients.keys())
        state.latest_client_data = {k:v for k,v in state.latest_client_data.items() if k in current_connected_ids}

        # Reset accumulators for overall statistics
        state.gpu_total_util = {}
        state.gpu_util_count = {}
        state.gpu_total_memory = {}
        state.gpu_memory_count = {}

    state.profiling_task = asyncio.create_task(collect_and_store_frame())
    logger.info(f"Profiling started. Output will be saved to {state.output_filename}")
    log_total_gpus()
    return {"message": "Profiling started.", "output_file": state.output_filename}

@head_app.post("/stop")
async def stop_profiling():
    if not state.is_profiling:
        raise HTTPException(status_code=400, detail="Profiling is not active.")

    logger.info("Stopping profiling...")
    async with state.data_lock: # Ensure is_profiling is set under lock
        state.is_profiling = False # Signal the collection loop to stop

    if state.profiling_task:
        try:
            await asyncio.wait_for(state.profiling_task, timeout=5.0) # Wait for orderly shutdown
        except asyncio.TimeoutError:
            logger.warning("Profiling task did not finish in time. Data might be incomplete for the last frame.")
            state.profiling_task.cancel()
        except Exception as e:
            logger.error(f"Error during profiling task shutdown: {e}")
    
    state.profiling_task = None
    
    # Final save is handled by collect_and_store_frame when state.is_profiling becomes False
    # but let's ensure it's fully done. The task should have completed its final write.
    # If direct saving is needed here, it must be made sure it doesn't conflict with the task.

    return {"message": f"Profiling stopped. Data saved to {state.output_filename if state.output_filename else 'N/A'}"}

# This function will be called by Typer CLI
def run_head_node(host: str, port: int, enable_bandwidth: bool, freq: int):
    logger.info(f"Starting GSwarm Profiler Head Node on {host}:{port}")
    logger.info(f"Bandwidth profiling: {'Enabled' if enable_bandwidth else 'Disabled'}")
    state.enable_bandwidth_profiling = enable_bandwidth
    state.freq = freq
    # Log own GPUs if any for information
    try:
        local_devices = nvitop.Device.all()
        if local_devices:
            logger.info(f"Head node has {len(local_devices)} local GPU(s): {[d.name() for d in local_devices]}")
    except Exception:
        logger.info("Head node has no local NVIDIA GPUs or nvitop cannot access them.")
    
    uvicorn.run(head_app, host=host, port=port, log_level="warning")


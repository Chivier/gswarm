from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
from loguru import logger
from datetime import datetime

# Import the head module's state
from gswarm.profiler.head import state, HeadNodeState

app = FastAPI(
    title="GSwarm Profiler HTTP API", description="HTTP API for GSwarm Profiler control panel", version="0.1.0"
)


# Pydantic models for request/response
class StartProfilingRequest(BaseModel):
    name: Optional[str] = None


class ProfilingResponse(BaseModel):
    success: bool
    message: str
    output_file: Optional[str] = None


class StatusResponse(BaseModel):
    freq: int
    enable_bandwidth_profiling: bool
    enable_nvlink_profiling: bool
    is_profiling: bool
    output_filename: str
    frame_id_counter: int
    connected_clients: List[str]
    total_gpus: int
    gpu_info: Dict[str, List[Dict[str, Any]]]


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "GSwarm Profiler HTTP API", "version": "0.1.0"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current profiler status"""
    try:
        total_gpus = sum(len(gpus) for gpus in state.client_gpu_info.values())

        return StatusResponse(
            freq=state.freq,
            enable_bandwidth_profiling=state.enable_bandwidth_profiling,
            enable_nvlink_profiling=state.enable_nvlink_profiling,
            is_profiling=state.is_profiling,
            output_filename=state.output_filename,
            frame_id_counter=state.frame_id_counter,
            connected_clients=list(state.connected_clients.keys()),
            total_gpus=total_gpus,
            gpu_info=state.client_gpu_info,
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profiling/start", response_model=ProfilingResponse)
async def start_profiling(request: StartProfilingRequest):
    """Start profiling session"""
    try:
        if state.is_profiling:
            return ProfilingResponse(success=False, message="Profiling is already active.", output_file="")

        async with state.data_lock:
            state.is_profiling = True
            state.profiling_data_frames = []
            state.frame_id_counter = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

        # Import the collect function from head module
        from gswarm.profiler.head import collect_and_store_frame

        state.profiling_task = asyncio.create_task(collect_and_store_frame())
        logger.info(f"Profiling started via HTTP API. Output will be saved to {state.output_filename}")

        return ProfilingResponse(success=True, message="Profiling started.", output_file=state.output_filename)

    except Exception as e:
        logger.error(f"Error starting profiling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profiling/stop", response_model=ProfilingResponse)
async def stop_profiling():
    """Stop profiling session"""
    try:
        if not state.is_profiling:
            return ProfilingResponse(success=False, message="Profiling is not active.")

        logger.info("Stopping profiling via HTTP API...")
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

        return ProfilingResponse(
            success=True,
            message=f"Profiling stopped. Data saved to {state.output_filename if state.output_filename else 'N/A'}",
        )

    except Exception as e:
        logger.error(f"Error stopping profiling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clients")
async def get_clients():
    """Get connected clients and their GPU info"""
    try:
        clients_info = []
        for client_id, hostname in state.connected_clients.items():
            gpu_info = state.client_gpu_info.get(client_id, [])
            clients_info.append(
                {"client_id": client_id, "hostname": hostname, "gpu_count": len(gpu_info), "gpus": gpu_info}
            )

        return {"total_clients": len(clients_info), "clients": clients_info}
    except Exception as e:
        logger.error(f"Error getting clients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/latest")
async def get_latest_metrics():
    """Get latest metrics from all connected clients"""
    try:
        return {"timestamp": datetime.now().isoformat(), "client_data": state.latest_client_data}
    except Exception as e:
        logger.error(f"Error getting latest metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_http_server(host: str, port: int):
    """Run the HTTP API server"""
    config = uvicorn.Config(app, host=host, port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)
    logger.info(f"Starting HTTP API server on {host}:{port}")
    await server.serve()

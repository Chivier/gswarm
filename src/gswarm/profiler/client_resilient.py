"""
Resilient client implementation with buffering and automatic reconnection
"""

import asyncio
import grpc
import nvitop
import platform
import time
from typing import Dict, List, Any, Optional
from collections import deque
from loguru import logger
from datetime import datetime

from gswarm.profiler import profiler_pb2
from gswarm.profiler import profiler_pb2_grpc
from gswarm.profiler.client import collect_gpu_metrics, dict_to_grpc_metrics_update
from gswarm.profiler.adaptive_sampler import AdaptiveSampler

from contextlib import asynccontextmanager

from fastapi import FastAPI

import os
import signal

from gswarm.profiler.client_common import parse_extra_metrics
import traceback


class ResilientClient:
    """Client with automatic reconnection and data buffering"""

    def __init__(self, head_address: str, enable_bandwidth: bool, extra_metrics: List[str] = []):
        self.head_address = head_address
        self.enable_bandwidth = enable_bandwidth
        self.hostname = platform.node()

        # Sampling configuration from host
        self.freq_ms = 200  # Default until we get config from host
        self.use_adaptive = False

        # Connection state
        self.connected = False
        self.channel = None
        self.stub = None
        self.stream_call = None

        # Buffering
        self.buffer = deque(maxlen=10000)  # Buffer up to 10k frames
        self.buffer_lock = asyncio.Lock()

        # Historical data storage for adaptive sampling
        self.metrics_history = deque(maxlen=1000)  # Store last 1000 metrics with timestamps
        self.history_lock = asyncio.Lock()

        # Reconnection parameters
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 60
        self.reconnect_task = None

        # Metrics collection task
        self.collection_task = None
        self.running = True

        # Initialize GPU info
        self.gpu_infos = []
        self._init_gpu_info()

        # Adaptive sampler (initialized when needed)
        self.adaptive_sampler = None
        self._printed_sampling_config = False  # Track if we've printed sampling config

        self.extra_metrics = extra_metrics

    def _init_gpu_info(self):
        """Initialize GPU information"""
        try:
            devices = nvitop.Device.all()
            if not devices:
                logger.error("No NVIDIA GPUs found on this client node.")
                raise RuntimeError("No GPUs found")

            logger.info(f"Found {len(devices)} GPU(s): {[d.name() for d in devices]}")

            for i, dev in enumerate(devices):
                self.gpu_infos.append(profiler_pb2.GPUInfo(physical_idx=i, name=dev.name()))

        except Exception as e:
            logger.error(f"Error initializing GPUs: {e}")
            raise

    async def _get_host_config(self) -> bool:
        """Get sampling configuration from host"""
        try:
            status = await self.stub.GetStatus(profiler_pb2.Empty())
            self.freq_ms = status.freq if status.freq > 0 else 0
            self.use_adaptive = status.freq == 0

            if self.use_adaptive and not self.adaptive_sampler:
                self.adaptive_sampler = AdaptiveSampler()
                if not self._printed_sampling_config:
                    logger.info("Using adaptive sampling strategy (configured by host)")
                    self._printed_sampling_config = True
            elif not self.use_adaptive and not self._printed_sampling_config:
                logger.info(f"Using fixed frequency sampling: {self.freq_ms}ms (configured by host)")
                self._printed_sampling_config = True

            # Also get bandwidth config from host
            self.enable_bandwidth = status.enable_bandwidth_profiling

            return True
        except Exception as e:
            logger.warning(f"Failed to get host config: {e}. Using defaults.")
            return False

    async def connect(self) -> bool:
        """Establish connection to head node"""
        try:
            logger.info(f"Attempting to connect to {self.head_address}...")

            # Create channel and stub
            self.channel = grpc.aio.insecure_channel(
                self.head_address,
                options=[
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                ],
            )
            self.stub = profiler_pb2_grpc.ProfilerServiceStub(self.channel)

            # Send initial connection info
            initial_info = profiler_pb2.InitialInfo(hostname=self.hostname, gpus=self.gpu_infos)

            connect_response = await self.stub.Connect(initial_info)
            if not connect_response.success:
                logger.error(f"Failed to connect: {connect_response.message}")
                return False

            logger.info(f"Connected successfully: {connect_response.message}")
            self.connected = True
            self.reconnect_delay = 1  # Reset delay on successful connection

            # Get configuration from host
            await self._get_host_config()

            # Start metrics streaming
            await self._start_streaming()

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    async def _start_streaming(self):
        """Start streaming metrics to head node"""
        try:
            # Create async generator for metrics
            async def metrics_generator():
                sent_count = 0
                last_config_check = time.time()

                while self.running and self.connected:
                    try:
                        # Periodically check for config updates from host
                        if time.time() - last_config_check > 30:  # Check every 30 seconds
                            await self._get_host_config()
                            last_config_check = time.time()

                        # First, try to send buffered data if any
                        async with self.buffer_lock:
                            while self.buffer and self.connected:
                                buffered_metric = self.buffer.popleft()
                                yield buffered_metric
                                sent_count += 1

                                # Yield control periodically
                                if sent_count % 100 == 0:
                                    await asyncio.sleep(0.001)

                        # For adaptive sampling, let the sampler decide
                        if self.use_adaptive:
                            # Check if we should sample based on adaptive strategy
                            should_sample = False
                            metrics_payload = await collect_gpu_metrics(self.enable_bandwidth, self.extra_metrics)

                            # Check GPU utilization changes
                            for gpu in metrics_payload.get("gpus_metrics", []):
                                if await self.adaptive_sampler.should_sample("gpu_util", gpu["gpu_util"]):
                                    should_sample = True
                                    self.adaptive_sampler.update_metric("gpu_util", gpu["gpu_util"])
                                if await self.adaptive_sampler.should_sample("memory", gpu["mem_util"]):
                                    should_sample = True
                                    self.adaptive_sampler.update_metric("memory", gpu["mem_util"])

                            # Check bandwidth changes if enabled
                            if self.enable_bandwidth:
                                for gpu in metrics_payload.get("gpus_metrics", []):
                                    total_bw = gpu.get("dram_bw_gbps_rx", 0) + gpu.get("dram_bw_gbps_tx", 0)
                                    if await self.adaptive_sampler.should_sample("bandwidth", total_bw):
                                        should_sample = True
                                        self.adaptive_sampler.update_metric("bandwidth", total_bw)

                            if should_sample:
                                grpc_update = dict_to_grpc_metrics_update(self.hostname, metrics_payload)
                                grpc_update.timestamp = time.time()
                                yield grpc_update

                                # Store in history
                                async with self.history_lock:
                                    self.metrics_history.append(
                                        {"timestamp": grpc_update.timestamp, "metrics": metrics_payload}
                                    )

                            # Adaptive sleep - minimum interval from sampler configs
                            await asyncio.sleep(0.2)  # 200ms minimum
                        else:
                            # Fixed frequency sampling
                            metrics_payload = await collect_gpu_metrics(self.enable_bandwidth, self.extra_metrics)
                            grpc_update = dict_to_grpc_metrics_update(self.hostname, metrics_payload)
                            grpc_update.timestamp = time.time()
                            yield grpc_update

                            # Store in history
                            async with self.history_lock:
                                self.metrics_history.append(
                                    {"timestamp": grpc_update.timestamp, "metrics": metrics_payload}
                                )

                            await asyncio.sleep(self.freq_ms / 1000.0)

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error in metrics generator: {e}")
                        traceback.print_exc()
                        self.connected = False
                        break

            # Start streaming
            self.stream_call = self.stub.StreamMetrics(metrics_generator())
            await self.stream_call

        except grpc.aio.AioRpcError as e:
            logger.warning(f"Streaming error: {e.code()} - {e.details()}")
            self.connected = False
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}")
            self.connected = False

    async def disconnect(self):
        """Disconnect from head node"""
        self.connected = False

        if self.stream_call:
            self.stream_call.cancel()
            self.stream_call = None

        if self.channel:
            await self.channel.close()
            self.channel = None

        logger.info("Disconnected from head node")

    async def _collect_and_buffer(self):
        """Continuously collect metrics and buffer if disconnected"""
        while self.running:
            try:
                if self.use_adaptive:
                    # For adaptive mode, check if we should collect
                    should_collect = False
                    metrics_payload = await collect_gpu_metrics(self.enable_bandwidth, self.extra_metrics)

                    # Check for significant changes
                    for gpu in metrics_payload.get("gpus_metrics", []):
                        if await self.adaptive_sampler.should_sample("gpu_util", gpu["gpu_util"]):
                            should_collect = True
                            break

                    if should_collect or not self.connected:
                        grpc_update = dict_to_grpc_metrics_update(self.hostname, metrics_payload)
                        grpc_update.timestamp = time.time()

                        # Store in history
                        async with self.history_lock:
                            self.metrics_history.append(
                                {"timestamp": grpc_update.timestamp, "metrics": metrics_payload}
                            )

                        # If not connected, buffer the data
                        if not self.connected:
                            async with self.buffer_lock:
                                self.buffer.append(grpc_update)
                                if len(self.buffer) == self.buffer.maxlen:
                                    logger.warning("Buffer full, dropping oldest metrics")

                    await asyncio.sleep(0.2)  # Check every 200ms
                else:
                    # Fixed frequency mode
                    metrics_payload = await collect_gpu_metrics(self.enable_bandwidth, self.extra_metrics)
                    grpc_update = dict_to_grpc_metrics_update(self.hostname, metrics_payload)
                    grpc_update.timestamp = time.time()

                    # Store in history
                    async with self.history_lock:
                        self.metrics_history.append({"timestamp": grpc_update.timestamp, "metrics": metrics_payload})

                    # If not connected, buffer the data
                    if not self.connected:
                        async with self.buffer_lock:
                            self.buffer.append(grpc_update)
                            if len(self.buffer) == self.buffer.maxlen:
                                logger.warning("Buffer full, dropping oldest metrics")

                    await asyncio.sleep(self.freq_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _reconnect_loop(self):
        """Handle automatic reconnection"""
        while self.running:
            if not self.connected:
                logger.info(f"Attempting reconnection in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

                if await self.connect():
                    logger.info("Reconnection successful")
                    if self.buffer:
                        logger.info(f"Sending {len(self.buffer)} buffered metrics")
                else:
                    # Exponential backoff
                    self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

            else:
                # Check connection health periodically
                await asyncio.sleep(5)

    async def run(self):
        """Run the resilient client"""
        try:
            # Initial connection
            await self.connect()

            # Start background tasks
            self.collection_task = asyncio.create_task(self._collect_and_buffer())
            self.reconnect_task = asyncio.create_task(self._reconnect_loop())

            # Wait for tasks
            await asyncio.gather(self.collection_task, self.reconnect_task, return_exceptions=True)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down client...")
        self.running = False

        # Cancel tasks
        if self.collection_task:
            self.collection_task.cancel()
        if self.reconnect_task:
            self.reconnect_task.cancel()

        # Disconnect
        await self.disconnect()

        # Save buffered data if any
        if self.buffer:
            logger.info(f"Saving {len(self.buffer)} buffered metrics to disk")
            try:
                import pickle
                from gswarm.utils.cache import get_cache_dir

                cache_dir = get_cache_dir() / "metrics"
                cache_dir.mkdir(exist_ok=True)

                metrics_file = cache_dir / f"buffered_metrics_{self.hostname}_{int(time.time())}.pkl"
                with open(metrics_file, "wb") as f:
                    pickle.dump(list(self.buffer), f)
                logger.info(f"Saved buffered metrics to {metrics_file}")
            except Exception as e:
                logger.error(f"Failed to save buffered metrics: {e}")

        # Save historical data
        if self.metrics_history:
            logger.info(f"Saving {len(self.metrics_history)} historical metrics to disk")
            try:
                import pickle
                from gswarm.utils.cache import get_cache_dir

                cache_dir = get_cache_dir() / "metrics"
                cache_dir.mkdir(exist_ok=True)

                history_file = cache_dir / f"metrics_history_{self.hostname}_{int(time.time())}.pkl"
                with open(history_file, "wb") as f:
                    pickle.dump(list(self.metrics_history), f)
                logger.info(f"Saved historical metrics to {history_file}")
            except Exception as e:
                logger.error(f"Failed to save historical metrics: {e}")

        logger.info("Client shutdown complete")


def create_resilient_lifespan(head_address: str, enable_bandwidth: bool, extra_metrics: list[str] = []) -> FastAPI:
    """Create FastAPI app with resilient client context"""

    @asynccontextmanager
    async def resilient_client_context(app: FastAPI):
        """Context manager for ResilientClient"""
        app.state.client = ResilientClient(head_address, enable_bandwidth, extra_metrics)
        app.state.client_run_task = asyncio.create_task(app.state.client.run())

        yield

        app.state.client_run_task.cancel()

    return resilient_client_context

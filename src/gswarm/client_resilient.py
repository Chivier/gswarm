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


class ResilientClient:
    """Client with automatic reconnection and data buffering"""

    def __init__(self, head_address: str, enable_bandwidth: bool):
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
                logger.info("Using adaptive sampling strategy (configured by host)")
            elif not self.use_adaptive:
                logger.info(f"Using fixed frequency sampling: {self.freq_ms}ms (configured by host)")

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

    # ... rest of the existing methods stay the same ...


def start_resilient_client(head_address: str, enable_bandwidth: bool):
    """Start the resilient client

    Args:
        head_address: Address of the head node
        enable_bandwidth: Whether to collect bandwidth metrics (can be overridden by host)
    """
    client = ResilientClient(head_address, enable_bandwidth)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass

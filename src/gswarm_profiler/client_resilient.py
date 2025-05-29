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

from . import profiler_pb2
from . import profiler_pb2_grpc
from .client import collect_gpu_metrics, dict_to_grpc_metrics_update


class ResilientClient:
    """Client with automatic reconnection and data buffering"""

    def __init__(self, head_address: str, freq_ms: int, enable_bandwidth: bool):
        self.head_address = head_address
        self.freq_ms = freq_ms
        self.enable_bandwidth = enable_bandwidth
        self.hostname = platform.node()

        # Connection state
        self.connected = False
        self.channel = None
        self.stub = None
        self.stream_call = None

        # Buffering
        self.buffer = deque(maxlen=10000)  # Buffer up to 10k frames
        self.buffer_lock = asyncio.Lock()

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
                while self.running and self.connected:
                    try:
                        # First, try to send buffered data if any
                        async with self.buffer_lock:
                            while self.buffer and self.connected:
                                buffered_metric = self.buffer.popleft()
                                yield buffered_metric
                                sent_count += 1

                                # Yield control periodically
                                if sent_count % 100 == 0:
                                    await asyncio.sleep(0.001)

                        # Then send latest metric
                        metrics_payload = await collect_gpu_metrics(self.enable_bandwidth)
                        grpc_update = dict_to_grpc_metrics_update(self.hostname, metrics_payload)
                        grpc_update.timestamp = time.time()  # Add timestamp
                        yield grpc_update

                        await asyncio.sleep(self.freq_ms / 1000.0)

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error in metrics generator: {e}")
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
                # Collect metrics
                metrics_payload = await collect_gpu_metrics(self.enable_bandwidth)
                grpc_update = dict_to_grpc_metrics_update(self.hostname, metrics_payload)
                grpc_update.timestamp = time.time()

                # If not connected, buffer the data
                if not self.connected:
                    async with self.buffer_lock:
                        self.buffer.append(grpc_update)
                        if len(self.buffer) == self.buffer.maxlen:
                            logger.warning("Buffer full, dropping oldest metrics")

                await asyncio.sleep(self.freq_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
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

                with open(f"buffered_metrics_{self.hostname}_{int(time.time())}.pkl", "wb") as f:
                    pickle.dump(list(self.buffer), f)
            except Exception as e:
                logger.error(f"Failed to save buffered metrics: {e}")

        logger.info("Client shutdown complete")


def start_resilient_client(head_address: str, freq_ms: int, enable_bandwidth: bool):
    """Start the resilient client"""
    client = ResilientClient(head_address, freq_ms, enable_bandwidth)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass

"""Adaptive sampling strategy similar to WandB for efficient metric collection"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class MetricConfig:
    """Configuration for a specific metric type"""

    interval: float  # Sampling interval in seconds
    threshold: float  # Minimum change threshold to trigger update (percentage)
    last_sample_time: float = 0.0
    last_value: Optional[float] = None


class AdaptiveSampler:
    """
    Implements WandB-like adaptive sampling strategy.
    Different metrics are sampled at different intervals based on their characteristics.
    """

    def __init__(self):
        # Define sampling configurations for different metric types
        self.metric_configs = {
            # System metrics - sample every 200 ms
            "system": MetricConfig(interval=0.2, threshold=1.0),
            # GPU utilization - sample every 400 ms
            "gpu_util": MetricConfig(interval=0.4, threshold=2.0),
            # Memory utilization - sample every 400 ms
            "memory": MetricConfig(interval=0.4, threshold=1.0),
            # Bandwidth metrics - sample every 200 ms when enabled
            "bandwidth": MetricConfig(interval=0.2, threshold=5.0),
            # NVLink metrics - sample every 200 ms when enabled
            "nvlink": MetricConfig(interval=0.2, threshold=5.0),
        }

        self._callbacks: Dict[str, Callable] = {}
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}

    def register_callback(self, metric_type: str, callback: Callable):
        """Register a callback function for a specific metric type"""
        self._callbacks[metric_type] = callback

    async def should_sample(self, metric_type: str, current_value: Optional[float] = None) -> bool:
        """
        Determine if a metric should be sampled based on:
        1. Time since last sample
        2. Change threshold (if current value provided)
        """
        config = self.metric_configs.get(metric_type)
        if not config:
            return True

        current_time = time.time()

        # Check time interval
        if current_time - config.last_sample_time < config.interval:
            # Even if time hasn't passed, check for significant changes
            if current_value is not None and config.last_value is not None:
                change_percent = abs(current_value - config.last_value) / (config.last_value + 1e-6) * 100
                if change_percent >= config.threshold:
                    logger.debug(f"Metric {metric_type} changed by {change_percent:.1f}%, triggering early sample")
                    return True
            return False

        return True

    def update_metric(self, metric_type: str, value: float):
        """Update metric tracking information"""
        config = self.metric_configs.get(metric_type)
        if config:
            config.last_sample_time = time.time()
            config.last_value = value

    async def _sample_loop(self, metric_type: str):
        """Sampling loop for a specific metric type"""
        while self._running:
            try:
                config = self.metric_configs[metric_type]
                callback = self._callbacks.get(metric_type)

                if callback and await self.should_sample(metric_type):
                    await callback()

                # Wait for the configured interval
                await asyncio.sleep(config.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sampling loop for {metric_type}: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def start(self, enabled_metrics: Dict[str, bool]):
        """Start adaptive sampling for enabled metrics"""
        self._running = True

        # Start sampling tasks for enabled metrics
        for metric_type, enabled in enabled_metrics.items():
            if enabled and metric_type in self._callbacks:
                self._tasks[metric_type] = asyncio.create_task(self._sample_loop(metric_type))
                logger.info(
                    f"Started adaptive sampling for {metric_type} "
                    f"(interval: {self.metric_configs[metric_type].interval}s)"
                )

    async def stop(self):
        """Stop all sampling tasks"""
        self._running = False

        # Cancel all tasks
        for task in self._tasks.values():
            task.cancel()

        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        self._tasks.clear()
        logger.info("Adaptive sampling stopped")

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get statistics about sampling intervals and last sample times"""
        stats = {}
        for metric_type, config in self.metric_configs.items():
            stats[metric_type] = {
                "interval": config.interval,
                "threshold": config.threshold,
                "last_sample_time": config.last_sample_time,
                "time_since_last_sample": time.time() - config.last_sample_time
                if config.last_sample_time > 0
                else None,
                "last_value": config.last_value,
            }
        return stats

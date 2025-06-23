import json
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from typing import List
from loguru import logger
import seaborn as sns


class GPUData(BaseModel):
    gpu_name: str
    gpu_util: list[float]
    gpu_memory: list[float]
    gpu_dram_bandwidth: list[float]


def parse_frame_data(data) -> List[GPUData] | None:
    frames = data["frames"]
    if not frames:
        return None
    else:
        num_gpus = len(frames[0]["gpu_id"])
        gpu_data_list = []

        for i in range(num_gpus):
            gpu_name = frames[0]["gpu_id"][i]
            gpu_util = [float(frame["gpu_util"][i]) for frame in frames]
            gpu_memory = [float(frame["gpu_memory"][i]) for frame in frames]
            try:
                gpu_dram_bandwidth = [float(frame["dram_bandwidth"][i]) for frame in frames]
            except KeyError:
                logger.warning(f"Key 'dram_bandwidth' not found in frame data for GPU {gpu_name}. Setting to zero.")
                gpu_dram_bandwidth = [0.0] * len(frames)

            gpu_data_list.append(
                GPUData(
                    gpu_name=gpu_name, gpu_util=gpu_util, gpu_memory=gpu_memory, gpu_dram_bandwidth=gpu_dram_bandwidth
                )
            )

        return gpu_data_list


def draw_gpu_utilization(gpu_datalist, frame_ids, ax):
    for gpu_data in gpu_datalist:
        ax.plot(frame_ids, gpu_data.gpu_util, marker="o", linestyle="-", label=gpu_data.gpu_name)
    ax.set_title("GPU Utilization Over Time")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("GPU Utilization (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def draw_gpu_memory(gpu_datalist, frame_ids, ax):
    for gpu_data in gpu_datalist:
        ax.plot(frame_ids, gpu_data.gpu_memory, marker="o", linestyle="-", label=gpu_data.gpu_name)
    ax.set_title("GPU Memory Usage Over Time")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("GPU Memory (MB)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def draw_gpu_dram_bandwidth(gpu_datalist, frame_ids, ax):
    for gpu_data in gpu_datalist:
        ax.plot(frame_ids, gpu_data.gpu_dram_bandwidth, marker="o", linestyle="-", label=gpu_data.gpu_name)
    ax.set_title("GPU Dram Bandwidth Over Time")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Bandwidth (KB/s)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def draw_gpu_util_bubble(gpu_datalist, frame_ids, ax):
    gpu_util_matrix = np.array([gpu_data.gpu_util for gpu_data in gpu_datalist])

    # Create a heatmap showing GPU utilization bubbles
    im = ax.imshow(gpu_util_matrix, cmap="RdYlBu_r", aspect="auto", interpolation="nearest")
    ax.set_title("GPU Utilization Heatmap")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("GPU Index")
    ax.set_yticks(range(len(gpu_datalist)))
    ax.set_yticklabels([gpu_data.gpu_name for gpu_data in gpu_datalist])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("GPU Utilization (%)")


# TODO: Write your own code to draw your own metrics here
# def draw_custom_metric(gpu_datalist, frame_ids, ax):
#     ...

# TODO: Remember to add your own metrics to the metrics_mapping dictionary
metrics_mapping = {
    "gpu_utilization": draw_gpu_utilization,
    "gpu_memory": draw_gpu_memory,
    "gpu_dram_bandwidth": draw_gpu_dram_bandwidth,
    "gpu_bubble": draw_gpu_util_bubble,
}

default_metrics = ["gpu_utilization", "gpu_memory", "gpu_dram_bandwidth"]


def draw_metrics(data, target_filename, enable_metrics=None):
    if enable_metrics is None or len(enable_metrics) == 0:
        enable_metrics = default_metrics

    logger.info(f"Drawing metrics: {enable_metrics}")
    num_metrics = len(enable_metrics)
    if num_metrics == 0:
        logger.error("No metrics to draw.")
        return

    gpu_data_list = parse_frame_data(data)
    if gpu_data_list is None:
        logger.error("No GPU data found in the frames.")
        return
    frame_ids = list(range(len(gpu_data_list[0].gpu_util)))

    fig, axs = plt.subplots(num_metrics, 1, figsize=(18, 5 * num_metrics))
    for i, metric in enumerate(enable_metrics):
        if metric in metrics_mapping:
            logger.info(f"Drawing metric: {metric}")
            metrics_mapping[metric](gpu_data_list, frame_ids, axs[i])
        else:
            logger.error(f"Metric '{metric}' is not supported.")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    logger.info(f"Saving plot to {target_filename}")
    plt.savefig(target_filename)

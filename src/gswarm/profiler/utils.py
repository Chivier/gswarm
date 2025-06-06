import json
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from typing import List

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
            gpu_dram_bandwidth = [float(frame["dram_bandwidth"][i]) for frame in frames]

            gpu_data_list.append(GPUData(
                gpu_name=gpu_name,
                gpu_util=gpu_util,
                gpu_memory=gpu_memory,
                gpu_dram_bandwidth=gpu_dram_bandwidth
            ))

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

def draw_basic_metrics(data, target_filename):
    gpu_data_list = parse_frame_data(data)
    if gpu_data_list is None:
        print("No GPU data found in the frames.")
        return
    frame_ids = list(range(len(gpu_data_list[0].gpu_util)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10))
    draw_gpu_utilization(gpu_data_list, frame_ids, ax1)
    draw_gpu_memory(gpu_data_list, frame_ids, ax2)
    draw_gpu_dram_bandwidth(gpu_data_list, frame_ids, ax3)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(target_filename)
    

def draw_gpu_metrics(data, target_filename):
    frames = data["frames"]

    if not frames:
        print("No frames found in the data.")
    else:
        num_gpus = len(frames[0]["gpu_id"])
        frame_ids = [frame["frame_id"] for frame in frames]

        gpu_names = frames[0]["gpu_id"]
        gpu_util_data = [[] for _ in range(num_gpus)]
        gpu_memory_data = [[] for _ in range(num_gpus)]
        gpu_dram_band_data = [[] for _ in range(num_gpus)]

        for frame in frames:
            for i in range(num_gpus):
                gpu_util_data[i].append(float(frame["gpu_util"][i]))
                gpu_memory_data[i].append(float(frame["gpu_memory"][i]))
                gpu_dram_band_data[i].append(float(frame["dram_bandwidth"][i]))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10))

        for i in range(num_gpus):
            ax1.plot(frame_ids, gpu_util_data[i], marker="o", linestyle="-", label=f"{gpu_names[i]}")
        ax1.set_title("GPU Utilization Over Time")
        ax1.set_xlabel("Frame ID")
        ax1.set_ylabel("GPU Utilization")
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax1.grid(True)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        for i in range(num_gpus):
            ax2.plot(frame_ids, gpu_memory_data[i], marker="o", linestyle="-", label=f"{gpu_names[i]}")
        ax2.set_title("GPU Memory Usage Over Time")
        ax2.set_xlabel("Frame ID")
        ax2.set_ylabel("GPU Memory")
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax2.grid(True)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        for i in range(num_gpus):
            ax3.plot(
                frame_ids,
                gpu_dram_band_data[i],
                marker="o",
                linestyle="-",
                label=f"{gpu_names[i]}",
            )
        ax3.set_title("GPU Dram Bandwidth Over Time")
        ax3.set_xlabel("Frame ID")
        ax3.set_ylabel("Bandwidth KB/s")
        ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax3.grid(True)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(target_filename)

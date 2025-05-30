import json
import matplotlib.pyplot as plt
import numpy as np


# TODO: add more metrics to the plot (dram util)
# TODO: fix frame_id to be the actual time
def show_stat(data_path, plot_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    num_gpus = len(frames[0]["gpu_id"])
    frame_ids = [frame["frame_id"] for frame in frames]

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
        ax1.plot(frame_ids, gpu_util_data[i], marker="o", linestyle="-", label=f"GPU {i}")
    ax1.set_title("GPU Utilization Over Time")
    ax1.set_xlabel("Frame ID")
    ax1.set_ylabel("GPU Utilization")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax1.grid(True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for i in range(num_gpus):
        ax2.plot(frame_ids, gpu_memory_data[i], marker="o", linestyle="-", label=f"GPU {i}")
    ax2.set_title("GPU Memory Usage Over Time")
    ax2.set_xlabel("Frame ID")
    ax2.set_ylabel("GPU Memory")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax2.grid(True)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for i in range(num_gpus):
        ax3.plot(frame_ids, gpu_dram_band_data[i], marker="o", linestyle="-", label=f"GPU {i}")
    ax3.set_title("GPU DRAM Bandwidth Over Time")
    ax3.set_xlabel("Frame ID")
    ax3.set_ylabel("GPU DRAM Bandwidth")
    ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax3.grid(True)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plot_path)

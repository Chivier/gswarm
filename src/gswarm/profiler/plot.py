import json
import matplotlib.pyplot as plt
import numpy as np

# 从JSON文件加载数据
with open("gswarm_profiler_20250523_094345.json", "r") as f:
    data = json.load(f)

# 提取帧数据
frames = data["frames"]

# 如果没有帧数据，则不进行绘图
if not frames:
    print("No frames found in the data.")
else:
    # 获取GPU数量和帧ID
    num_gpus = len(frames[0]["gpu_id"])
    frame_ids = [frame["frame_id"] for frame in frames]

    # 准备绘图数据
    gpu_util_data = [[] for _ in range(num_gpus)]
    gpu_memory_data = [[] for _ in range(num_gpus)]
    gpu_dram_band_data = [[] for _ in range(num_gpus)]

    for frame in frames:
        for i in range(num_gpus):
            gpu_util_data[i].append(float(frame["gpu_util"][i]))
            gpu_memory_data[i].append(float(frame["gpu_memory"][i]))
            gpu_dram_band_data[i].append(float(frame["dram_bandwidth"][i]))

    # 创建两个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10))

    # 绘制GPU利用率图
    for i in range(num_gpus):
        ax1.plot(frame_ids, gpu_util_data[i], marker="o", linestyle="-", label=f"GPU {i}")
    ax1.set_title("GPU Utilization Over Time")
    ax1.set_xlabel("Frame ID")
    ax1.set_ylabel("GPU Utilization")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax1.grid(True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 绘制GPU显存使用图
    for i in range(num_gpus):
        ax2.plot(frame_ids, gpu_memory_data[i], marker="o", linestyle="-", label=f"GPU {i}")
    ax2.set_title("GPU Memory Usage Over Time")
    ax2.set_xlabel("Frame ID")
    ax2.set_ylabel("GPU Memory")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax2.grid(True)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 绘制GPU Dram Bandwidth使用图
    for i in range(num_gpus):
        ax3.plot(
            frame_ids,
            gpu_dram_band_data[i],
            marker="o",
            linestyle="-",
            label=f"GPU {i}",
        )
    ax3.set_title("GPU Dram Bandwidth Over Time")
    ax3.set_xlabel("Frame ID")
    ax3.set_ylabel("Bandwidth KB/s")
    ax3.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax3.grid(True)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("gpu_metrics.png")

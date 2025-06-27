import json
import matplotlib.pyplot as plt
import numpy as np


def read_data_from_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data["summary"]
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return None


def plot_offline_gpu_comparison():
    """Plot 1: Compare offline experiments across different GPU numbers"""
    plt.figure(figsize=(12, 8))

    gpu_nums = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    metrics = ["Total Time"]

    data_by_metric = {metric: [] for metric in metrics}
    valid_gpu_nums = []

    for gpu_num in gpu_nums:
        file_path = f"quickrun_result/offline_execution_log_h20_{gpu_num}.json"
        data = read_data_from_json(file_path)
        if data:
            valid_gpu_nums.append(gpu_num)
            # data_by_metric["Average Waiting Time"].append(data.get("avg_waiting_time", 0))
            # data_by_metric["P99 Waiting Time"].append(data.get("p99_waiting_time", 0))
            data_by_metric["Total Time"].append(data["makespan"])

    x = np.arange(len(valid_gpu_nums))
    width = 0.25

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, data_by_metric[metric], width, label=metric, alpha=0.8)

    plt.title("Offline Experiments: Performance vs GPU Number")
    plt.xlabel("GPU Number")
    plt.ylabel("Time (seconds)")
    plt.xticks(x + width, valid_gpu_nums)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("offline_gpu_comparison.png")
    print("Plot saved as 'offline_gpu_comparison.png'")


def plot_online_gpu_comparison():
    """Plot 2: Compare online experiments across different GPU numbers"""
    plt.figure(figsize=(15, 10))

    gpu_nums = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    metrics = [
        "Total Time",
        "P99 Response Time",
        "Avg Response Time",
        "P99 Waiting Time",
        "Avg Waiting Time",
        "P99 Request Response Time",
        "Avg Request Response Time",
    ]

    data_by_metric = {metric: [] for metric in metrics}
    valid_gpu_nums = []

    for gpu_num in gpu_nums:
        file_path = f"quickrun_result/online_execution_log_h20_{gpu_num}.json"
        data = read_data_from_json(file_path)
        if data:
            valid_gpu_nums.append(gpu_num)
            data_by_metric["Total Time"].append(data["makespan"])
            data_by_metric["P99 Response Time"].append(data.get("p99_response_time", 0))
            data_by_metric["Avg Response Time"].append(data.get("avg_response_time", 0))
            data_by_metric["P99 Waiting Time"].append(data.get("p99_waiting_time", 0))
            data_by_metric["Avg Waiting Time"].append(data.get("avg_waiting_time", 0))
            data_by_metric["P99 Request Response Time"].append(data.get("p99_request_response_time", 0))
            data_by_metric["Avg Request Response Time"].append(data.get("avg_request_response_time", 0))

    x = np.arange(len(valid_gpu_nums))
    width = 0.1

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, data_by_metric[metric], width, label=metric, alpha=0.8)

    plt.title("Online Experiments: Performance vs GPU Number")
    plt.xlabel("GPU Number")
    plt.ylabel("Time (seconds)")
    plt.xticks(x + width * 3, valid_gpu_nums)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("online_gpu_comparison.png")
    print("Plot saved as 'online_gpu_comparison.png'")


def plot_baseline_offline_comparison():
    """Plot 3: Compare baseline_offline vs offline Total Time across GPU numbers"""
    plt.figure(figsize=(12, 8))

    gpu_nums = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    baseline_times = []
    offline_times = []
    valid_gpu_nums = []

    for gpu_num in gpu_nums:
        baseline_file = f"quickrun_result/baseline_execution_log_h20_offline_{gpu_num}.json"
        offline_file = f"quickrun_result/offline_execution_log_h20_{gpu_num}.json"

        baseline_data = read_data_from_json(baseline_file)
        offline_data = read_data_from_json(offline_file)

        if baseline_data and offline_data:
            valid_gpu_nums.append(gpu_num)
            baseline_times.append(baseline_data["makespan"])
            offline_times.append(offline_data["makespan"])

    x = np.arange(len(valid_gpu_nums))
    width = 0.35

    plt.bar(x - width / 2, baseline_times, width, label="Baseline Offline", alpha=0.8)
    plt.bar(x + width / 2, offline_times, width, label="Offline", alpha=0.8)

    plt.title("Baseline Offline vs Offline: Total Time Comparison")
    plt.xlabel("GPU Number")
    plt.ylabel("Total Time (seconds)")
    plt.xticks(x, valid_gpu_nums)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("baseline_offline_comparison.png")
    print("Plot saved as 'baseline_offline_comparison.png'")


def plot_baseline_online_comparison():
    """Plot 4: Compare baseline_online vs online across all metrics and GPU numbers"""
    plt.figure(figsize=(16, 12))

    gpu_nums = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    metrics = [
        "Total Time",
        "P99 Response Time",
        "Avg Response Time",
        "P99 Waiting Time",
        "Avg Waiting Time",
        "P99 Request Response Time",
        "Avg Request Response Time",
    ]

    baseline_data_by_metric = {metric: [] for metric in metrics}
    online_data_by_metric = {metric: [] for metric in metrics}
    valid_gpu_nums = []

    for gpu_num in gpu_nums:
        baseline_file = f"quickrun_result/baseline_execution_log_h20_online_{gpu_num}.json"
        online_file = f"quickrun_result/online_execution_log_h20_{gpu_num}.json"

        baseline_data = read_data_from_json(baseline_file)
        online_data = read_data_from_json(online_file)

        if baseline_data and online_data:
            valid_gpu_nums.append(gpu_num)

            baseline_data_by_metric["Total Time"].append(baseline_data["makespan"])
            baseline_data_by_metric["P99 Response Time"].append(baseline_data.get("p99_response_time", 0))
            baseline_data_by_metric["Avg Response Time"].append(baseline_data.get("avg_response_time", 0))
            baseline_data_by_metric["P99 Waiting Time"].append(baseline_data.get("p99_waiting_time", 0))
            baseline_data_by_metric["Avg Waiting Time"].append(baseline_data.get("avg_waiting_time", 0))
            baseline_data_by_metric["P99 Request Response Time"].append(
                baseline_data.get("p99_request_response_time", 0)
            )
            baseline_data_by_metric["Avg Request Response Time"].append(
                baseline_data.get("avg_request_response_time", 0)
            )

            online_data_by_metric["Total Time"].append(online_data["makespan"])
            online_data_by_metric["P99 Response Time"].append(online_data.get("p99_response_time", 0))
            online_data_by_metric["Avg Response Time"].append(online_data.get("avg_response_time", 0))
            online_data_by_metric["P99 Waiting Time"].append(online_data.get("p99_waiting_time", 0))
            online_data_by_metric["Avg Waiting Time"].append(online_data.get("avg_waiting_time", 0))
            online_data_by_metric["P99 Request Response Time"].append(online_data.get("p99_request_response_time", 0))
            online_data_by_metric["Avg Request Response Time"].append(online_data.get("avg_request_response_time", 0))

    # Create subplots for better visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    x = np.arange(len(valid_gpu_nums))
    width = 0.35

    for i, metric in enumerate(metrics):
        if i < len(axes):
            axes[i].bar(x - width / 2, baseline_data_by_metric[metric], width, label="Baseline Online", alpha=0.8)
            axes[i].bar(x + width / 2, online_data_by_metric[metric], width, label="Online", alpha=0.8)
            axes[i].set_title(metric)
            axes[i].set_xlabel("GPU Number")
            axes[i].set_ylabel("Time (seconds)")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(valid_gpu_nums)
            axes[i].legend()
            axes[i].grid(axis="y", alpha=0.3)

    # Hide the last subplot if not needed
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)

    plt.suptitle("Baseline Online vs Online: Detailed Metrics Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig("baseline_online_comparison.png")
    print("Plot saved as 'baseline_online_comparison.png'")


def plot_static_offline_comparison():
    """Plot 5: Compare static_offline vs baseline_offline Total Time across GPU numbers"""
    plt.figure(figsize=(12, 8))

    gpu_nums = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    static_times = []
    baseline_times = []
    valid_gpu_nums = []

    for gpu_num in gpu_nums:
        static_file = f"quickrun_result/static_execution_log_h20_offline_{gpu_num}.json"
        baseline_file = f"quickrun_result/baseline_execution_log_h20_offline_{gpu_num}.json"

        static_data = read_data_from_json(static_file)
        baseline_data = read_data_from_json(baseline_file)

        if static_data and baseline_data:
            valid_gpu_nums.append(gpu_num)
            static_times.append(static_data["makespan"])
            baseline_times.append(baseline_data["makespan"])

    x = np.arange(len(valid_gpu_nums))
    width = 0.35

    plt.bar(x - width / 2, static_times, width, label="Static Offline", alpha=0.8)
    plt.bar(x + width / 2, baseline_times, width, label="Baseline Offline", alpha=0.8)

    plt.title("Static Offline vs Baseline Offline: Total Time Comparison")
    plt.xlabel("GPU Number")
    plt.ylabel("Total Time (seconds)")
    plt.xticks(x, valid_gpu_nums)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("static_baseline_offline_comparison.png")
    print("Plot saved as 'static_baseline_offline_comparison.png'")


def plot_static_online_comparison():
    """Plot 6: Compare static_online vs baseline_online across all metrics and GPU numbers"""
    plt.figure(figsize=(16, 12))

    gpu_nums = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    metrics = [
        "Total Time",
        "P99 Response Time",
        "Avg Response Time",
        "P99 Waiting Time",
        "Avg Waiting Time",
        "P99 Request Response Time",
        "Avg Request Response Time",
    ]

    static_data_by_metric = {metric: [] for metric in metrics}
    baseline_data_by_metric = {metric: [] for metric in metrics}
    valid_gpu_nums = []

    for gpu_num in gpu_nums:
        static_file = f"quickrun_result/static_execution_log_h20_online_{gpu_num}.json"
        baseline_file = f"quickrun_result/baseline_execution_log_h20_online_{gpu_num}.json"

        static_data = read_data_from_json(static_file)
        baseline_data = read_data_from_json(baseline_file)

        if static_data and baseline_data:
            valid_gpu_nums.append(gpu_num)

            static_data_by_metric["Total Time"].append(static_data["makespan"])
            static_data_by_metric["P99 Response Time"].append(static_data.get("p99_response_time", 0))
            static_data_by_metric["Avg Response Time"].append(static_data.get("avg_response_time", 0))
            static_data_by_metric["P99 Waiting Time"].append(static_data.get("p99_waiting_time", 0))
            static_data_by_metric["Avg Waiting Time"].append(static_data.get("avg_waiting_time", 0))
            static_data_by_metric["P99 Request Response Time"].append(static_data.get("p99_request_response_time", 0))
            static_data_by_metric["Avg Request Response Time"].append(static_data.get("avg_request_response_time", 0))

            baseline_data_by_metric["Total Time"].append(baseline_data["makespan"])
            baseline_data_by_metric["P99 Response Time"].append(baseline_data.get("p99_response_time", 0))
            baseline_data_by_metric["Avg Response Time"].append(baseline_data.get("avg_response_time", 0))
            baseline_data_by_metric["P99 Waiting Time"].append(baseline_data.get("p99_waiting_time", 0))
            baseline_data_by_metric["Avg Waiting Time"].append(baseline_data.get("avg_waiting_time", 0))
            baseline_data_by_metric["P99 Request Response Time"].append(
                baseline_data.get("p99_request_response_time", 0)
            )
            baseline_data_by_metric["Avg Request Response Time"].append(
                baseline_data.get("avg_request_response_time", 0)
            )

    # Create subplots for better visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    x = np.arange(len(valid_gpu_nums))
    width = 0.35

    for i, metric in enumerate(metrics):
        if i < len(axes):
            axes[i].bar(x - width / 2, static_data_by_metric[metric], width, label="Static Online", alpha=0.8)
            axes[i].bar(x + width / 2, baseline_data_by_metric[metric], width, label="Baseline Online", alpha=0.8)
            axes[i].set_title(metric)
            axes[i].set_xlabel("GPU Number")
            axes[i].set_ylabel("Time (seconds)")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(valid_gpu_nums)
            axes[i].legend()
            axes[i].grid(axis="y", alpha=0.3)

    # Hide the last subplot if not needed
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)

    plt.suptitle("Static Online vs Baseline Online: Detailed Metrics Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig("static_baseline_online_comparison.png")
    print("Plot saved as 'static_baseline_online_comparison.png'")


if __name__ == "__main__":
    plot_offline_gpu_comparison()
    plot_online_gpu_comparison()
    plot_baseline_offline_comparison()
    plot_baseline_online_comparison()
    plot_static_offline_comparison()
    plot_static_online_comparison()

    print("All plots have been generated successfully!")

#!/usr/bin/env python3
"""
Compare all schedulers: baseline, offline, online, and static
"""

import subprocess
import json
import sys


def run_scheduler(scheduler_type, config_file, requests_file, gpus=4, extra_args=None):
    """Run a scheduler and return the metrics"""
    if scheduler_type == "baseline":
        cmd = [
            "python",
            "baseline.py",
            "--gpus",
            str(gpus),
            "--config",
            config_file,
            "--requests",
            requests_file,
            "--simulate",
            "true",
        ]
        log_file = "baseline_execution_log.json"
    elif scheduler_type == "offline":
        cmd = [
            "python",
            "offline_scheduler.py",
            "--gpus",
            str(gpus),
            "--config",
            config_file,
            "--requests",
            requests_file,
            "--simulate",
            "true",
        ]
        log_file = "offline_execution_log.json"
    elif scheduler_type == "online":
        cmd = [
            "python",
            "online_scheduler.py",
            "--gpus",
            str(gpus),
            "--config",
            config_file,
            "--requests",
            requests_file,
            "--simulate",
            "true",
        ]
        if extra_args:
            cmd.extend(extra_args)
        log_file = "online_execution_log.json"
    else:  # static
        cmd = [
            "python",
            "static_scheduler.py",
            "--gpus",
            str(gpus),
            "--config",
            config_file,
            "--requests",
            requests_file,
            "--simulate",
            "true",
        ]
        if extra_args:
            cmd.extend(extra_args)
        log_file = "static_execution_log.json"

    print(f"\nRunning {scheduler_type} scheduler...")
    if scheduler_type == "online" and extra_args:
        print(f"Extra args: {' '.join(extra_args)}")

    try:
        # Run scheduler
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {scheduler_type} scheduler")
            return None

        # Run check.py to get metrics
        check_result = subprocess.run(["python", "check.py", log_file], capture_output=True, text=True)
        # Note: check.py may return non-zero if conflicts detected, but we still want metrics

        # Parse metrics from check output
        metrics = {}
        lines = check_result.stdout.split("\n")

        for i, line in enumerate(lines):
            if "Average waiting time:" in line:
                try:
                    metrics["avg_waiting_time"] = float(line.split(":")[1].split()[0])
                except:
                    pass
            elif "P99 waiting time:" in line:
                try:
                    metrics["p99_waiting_time"] = float(line.split(":")[1].split()[0])
                except:
                    pass
            elif "Average response time:" in line and "Request-level" not in lines[i - 2]:
                try:
                    metrics["avg_response_time"] = float(line.split(":")[1].split()[0])
                except:
                    pass
            elif "P99 response time:" in line and "Request-level" not in lines[i - 2]:
                try:
                    metrics["p99_response_time"] = float(line.split(":")[1].split()[0])
                except:
                    pass
            elif "Request-level" in line and i + 2 < len(lines) and "Average response time:" in lines[i + 1]:
                try:
                    metrics["avg_request_response_time"] = float(lines[i + 1].split(":")[1].split()[0])
                    if i + 2 < len(lines) and "P99 response time:" in lines[i + 2]:
                        metrics["p99_request_response_time"] = float(lines[i + 2].split(":")[1].split()[0])
                except:
                    pass
            elif "Makespan:" in line:
                try:
                    metrics["makespan"] = float(line.split(":")[1].split()[0])
                except:
                    pass

        # Also get summary metrics from log file
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
                summary = data.get("summary", {})

                # Prefer summary metrics if available
                for key in [
                    "avg_waiting_time",
                    "p99_waiting_time",
                    "avg_response_time",
                    "p99_response_time",
                    "avg_request_response_time",
                    "p99_request_response_time",
                ]:
                    if key in summary and summary[key] is not None:
                        metrics[key] = summary[key]

                if "makespan" in summary:
                    metrics["makespan"] = summary["makespan"]

        except:
            pass

        return metrics

    except Exception as e:
        print(f"Exception running {scheduler_type} scheduler: {e}")
        return None


def compare_metrics(baseline_metrics, offline_metrics, online_metrics, static_metrics):
    """Compare metrics between schedulers"""
    print("\n" + "=" * 120)
    print("SCHEDULER COMPARISON")
    print("=" * 120)

    # Metrics to compare
    metrics_info = [
        ("Makespan (s)", "makespan"),
        ("Avg waiting time (s)", "avg_waiting_time"),
        ("P99 waiting time (s)", "p99_waiting_time"),
        ("Avg response time (s)", "avg_response_time"),
        ("P99 response time (s)", "p99_response_time"),
        ("Avg request response (s)", "avg_request_response_time"),
        ("P99 request response (s)", "p99_request_response_time"),
    ]

    print(f"{'Metric':<25} {'Baseline':>15} {'Offline':>15} {'Online':>15} {'Static':>15} {'Static vs BL':>20}")
    print("-" * 120)

    for display_name, key in metrics_info:
        baseline_val = baseline_metrics.get(key, "N/A")
        offline_val = offline_metrics.get(key, "N/A")
        online_val = online_metrics.get(key, "N/A")
        static_val = static_metrics.get(key, "N/A")

        # Format values and calculate improvement
        if isinstance(baseline_val, (int, float)) and isinstance(static_val, (int, float)):
            improvement = ((baseline_val - static_val) / baseline_val * 100) if baseline_val > 0 else 0

            if key in ["avg_waiting_time", "p99_waiting_time"]:
                # For waiting time, lower is better
                if improvement > 0:
                    improvement_str = f"{improvement:+.1f}% ✓"
                else:
                    improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = f"{improvement:+.1f}%"

            print(
                f"{display_name:<25} {baseline_val:>15.2f} "
                f"{offline_val if isinstance(offline_val, str) else f'{offline_val:>15.2f}'} "
                f"{online_val if isinstance(online_val, str) else f'{online_val:>15.2f}'} "
                f"{static_val:>15.2f} {improvement_str:>20}"
            )
        else:
            print(
                f"{display_name:<25} {str(baseline_val):>15} {str(offline_val):>15} "
                f"{str(online_val):>15} {str(static_val):>15} {'N/A':>20}"
            )

    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Check if online beats baseline
    improvements = []

    if isinstance(baseline_metrics.get("avg_waiting_time"), (int, float)) and isinstance(
        online_metrics.get("avg_waiting_time"), (int, float)
    ):
        baseline_avg_wait = baseline_metrics["avg_waiting_time"]
        online_avg_wait = online_metrics["avg_waiting_time"]

        if online_avg_wait < baseline_avg_wait:
            improvement = (baseline_avg_wait - online_avg_wait) / baseline_avg_wait * 100
            improvements.append(f"Average waiting time improved by {improvement:.1f}%")
        else:
            degradation = (online_avg_wait - baseline_avg_wait) / baseline_avg_wait * 100
            improvements.append(f"Average waiting time degraded by {degradation:.1f}%")

    if isinstance(baseline_metrics.get("p99_waiting_time"), (int, float)) and isinstance(
        online_metrics.get("p99_waiting_time"), (int, float)
    ):
        baseline_p99_wait = baseline_metrics["p99_waiting_time"]
        online_p99_wait = online_metrics["p99_waiting_time"]

        if online_p99_wait < baseline_p99_wait:
            improvement = (baseline_p99_wait - online_p99_wait) / baseline_p99_wait * 100
            improvements.append(f"P99 waiting time improved by {improvement:.1f}%")
        else:
            degradation = (online_p99_wait - baseline_p99_wait) / baseline_p99_wait * 100
            improvements.append(f"P99 waiting time degraded by {degradation:.1f}%")

    print("Online vs Baseline:")
    for imp in improvements:
        print(f"  • {imp}")

    # Overall verdict
    print("\nVerdict:")
    if all("improved" in imp for imp in improvements):
        print("✓ Online scheduler achieves better latency metrics than baseline!")
    else:
        print("✗ Online scheduler does not beat baseline on all latency metrics.")


def main():
    config_file = "balanced_config.json"
    requests_file = "simple_requests.yaml"
    gpus = 4

    print(f"Configuration: {config_file}")
    print(f"Requests: {requests_file}")
    print(f"GPUs: {gpus}")

    # Run baseline
    baseline_metrics = run_scheduler("baseline", config_file, requests_file, gpus)
    if not baseline_metrics:
        print("Failed to get baseline metrics")
        return

    # Run offline
    offline_metrics = run_scheduler("offline", config_file, requests_file, gpus)
    if not offline_metrics:
        print("Failed to get offline metrics")
        offline_metrics = {}

    # Run online with optimized parameters
    online_args = ["--alpha", "0.8", "--beta", "1.0", "--delta", "3.0"]
    online_metrics = run_scheduler("online", config_file, requests_file, gpus, online_args)
    if not online_metrics:
        print("Failed to get online metrics")
        online_metrics = {}

    # Run static
    static_metrics = run_scheduler("static", config_file, requests_file, gpus)
    if not static_metrics:
        print("Failed to get static metrics")
        static_metrics = {}

    # Compare results
    compare_metrics(baseline_metrics, offline_metrics, online_metrics, static_metrics)


if __name__ == "__main__":
    main()

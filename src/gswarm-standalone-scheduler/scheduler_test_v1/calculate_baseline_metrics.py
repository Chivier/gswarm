#!/usr/bin/env python3
"""
Calculate waiting time metrics for baseline scheduler from execution log
"""

import json
import numpy as np
from datetime import datetime
import yaml


def calculate_waiting_times(log_file, requests_file):
    """Calculate waiting times from execution log and request timestamps"""

    # Load execution log
    with open(log_file, "r") as f:
        log_data = json.load(f)

    # Load requests to get arrival times
    with open(requests_file, "r") as f:
        if requests_file.endswith(".yaml"):
            request_data = yaml.safe_load(f)
        else:
            request_data = json.load(f)

    requests = request_data["requests"]

    # Create arrival time map
    arrival_times = {}
    for req in requests:
        req_id = req["request_id"]
        timestamp = datetime.fromisoformat(req["timestamp"]).timestamp()
        arrival_times[req_id] = timestamp

    # Normalize arrival times to start from 0
    if arrival_times:
        min_time = min(arrival_times.values())
        for req_id in arrival_times:
            arrival_times[req_id] -= min_time

    # Calculate waiting times
    task_waiting_times = []
    task_response_times = []

    # Group executions by request
    request_executions = {}

    for exec_data in log_data["executions"]:
        req_id = exec_data["request_id"]
        if req_id not in request_executions:
            request_executions[req_id] = []
        request_executions[req_id].append(exec_data)

    # Calculate metrics
    for req_id, execs in request_executions.items():
        if req_id not in arrival_times:
            continue

        arrival_time = arrival_times[req_id]

        # Sort executions by start time
        execs.sort(key=lambda x: x["start_time"])

        # For first task in workflow
        first_exec = execs[0]
        waiting_time = first_exec["start_time"] - arrival_time
        task_waiting_times.append(waiting_time)

        # Response time for each task
        for exec_data in execs:
            response_time = exec_data["end_time"] - arrival_time
            task_response_times.append(response_time)

    # Calculate statistics
    if task_waiting_times:
        avg_waiting = np.mean(task_waiting_times)
        p99_waiting = np.percentile(task_waiting_times, 99)
    else:
        avg_waiting = 0
        p99_waiting = 0

    if task_response_times:
        avg_response = np.mean(task_response_times)
        p99_response = np.percentile(task_response_times, 99)
    else:
        avg_response = 0
        p99_response = 0

    return {
        "avg_waiting_time": avg_waiting,
        "p99_waiting_time": p99_waiting,
        "avg_response_time": avg_response,
        "p99_response_time": p99_response,
        "num_tasks": len(task_waiting_times),
    }


def main():
    metrics = calculate_waiting_times("baseline_execution_log.json", "simple_requests.yaml")

    print("Baseline Scheduler Latency Metrics:")
    print(f"  Average waiting time: {metrics['avg_waiting_time']:.2f} seconds")
    print(f"  P99 waiting time: {metrics['p99_waiting_time']:.2f} seconds")
    print(f"  Average response time: {metrics['avg_response_time']:.2f} seconds")
    print(f"  P99 response time: {metrics['p99_response_time']:.2f} seconds")
    print(f"  Total tasks analyzed: {metrics['num_tasks']}")


if __name__ == "__main__":
    main()

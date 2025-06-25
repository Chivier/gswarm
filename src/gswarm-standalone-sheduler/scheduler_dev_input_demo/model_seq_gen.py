import json
import numpy as np
import random
from datetime import datetime, timedelta
import yaml

# Model definitions with GPU requirements
MODELS = {
    "llm7b": {
        "name": "LLM-7B",
        "memory_gb": 14,
        "gpus_required": 1,
        "load_time_seconds": 10,  # Estimated based on PCIe speed
        "tokens_per_second": 50,  # Average tokens per second
        "token_mean": 512,  # Mean tokens for normal distribution
        "token_std": 128,  # Standard deviation for tokens
    },
    "llm30b": {
        "name": "LLM-30B",
        "memory_gb": 60,
        "gpus_required": 4,  # 30B model needs 4 GPUs
        "load_time_seconds": 40,
        "tokens_per_second": 20,  # Slower due to size
        "token_mean": 1024,
        "token_std": 256,
    },
    "stable_diffusion": {
        "name": "Stable-Diffusion",
        "memory_gb": 8,
        "gpus_required": 1,
        "load_time_seconds": 6,
        "inference_time_mean": 2.0,  # Fixed time in seconds
        "inference_time_std": 0.5,
        "config": {"width": [512, 768, 1024], "height": [512, 768, 1024]},
    },
}

# Workflow definitions
WORKFLOWS = {
    "workflow1": {
        "name": "LLM to Image Generation",
        "nodes": [
            {"id": "node1", "model": "llm7b", "inputs": ["user_prompt"], "outputs": ["image_prompt"]},
            {
                "id": "node2",
                "model": "stable_diffusion",
                "inputs": ["image_prompt"],
                "outputs": ["generated_image"],
                "config_options": ["width", "height"],
            },
        ],
        "edges": [{"from": "node1", "to": "node2"}],
    },
    "workflow2": {
        "name": "LLM Fork and Merge",
        "nodes": [
            {"id": "node1", "model": "llm7b", "inputs": ["user_prompt"], "outputs": ["initial_analysis"]},
            {"id": "node2", "model": "llm30b", "inputs": ["initial_analysis"], "outputs": ["deep_analysis_1"]},
            {"id": "node3", "model": "llm30b", "inputs": ["initial_analysis"], "outputs": ["deep_analysis_2"]},
            {
                "id": "node4",
                "model": "llm7b",
                "inputs": ["deep_analysis_1", "deep_analysis_2"],
                "outputs": ["final_summary"],
            },
        ],
        "edges": [
            {"from": "node1", "to": "node2"},
            {"from": "node1", "to": "node3"},
            {"from": "node2", "to": "node4"},
            {"from": "node3", "to": "node4"},
        ],
    },
}


def generate_poisson_arrivals(num_requests=500, duration_minutes=20):
    """Generate Poisson-distributed arrival times"""
    # Calculate arrival rate (requests per minute)
    arrival_rate = num_requests / duration_minutes

    # Generate inter-arrival times
    inter_arrivals = np.random.exponential(1 / arrival_rate, num_requests)

    # Calculate actual arrival times
    arrival_times = np.cumsum(inter_arrivals)

    # Scale to fit within duration if needed
    if arrival_times[-1] > duration_minutes:
        arrival_times = arrival_times * (duration_minutes / arrival_times[-1])

    # Convert to timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(minutes=t) for t in arrival_times]

    return timestamps


def generate_node_execution_time(model_name: str) -> float:
    """Generate execution time based on token count (normal distribution)"""
    model = MODELS[model_name]

    if model_name == "stable_diffusion":
        # Fixed time for image generation
        time = np.random.normal(model["inference_time_mean"], model["inference_time_std"])
        return max(0.1, time)  # Ensure positive time
    else:
        # LLM models: time based on token count
        tokens = np.random.normal(model["token_mean"], model["token_std"])
        tokens = max(1, int(tokens))  # Ensure positive token count
        time = tokens / model["tokens_per_second"]
        return time


def generate_workflow_requests(num_requests=500, duration_minutes=20):
    """Generate workflow requests with Poisson arrival times"""
    timestamps = generate_poisson_arrivals(num_requests, duration_minutes)
    requests = []

    for i, timestamp in enumerate(timestamps):
        # Randomly choose workflow
        workflow_id = random.choice(list(WORKFLOWS.keys()))
        workflow = WORKFLOWS[workflow_id]

        # Generate request
        request = {
            "request_id": f"req_{i:04d}",
            "timestamp": timestamp.isoformat(),
            "workflow_id": workflow_id,
            "input_data": {"user_prompt": f"Sample prompt for request {i}"},
            "node_configs": {},
            "node_execution_times": {},  # Pre-generate execution times
        }

        # Add node-specific configurations and execution times
        for node in workflow["nodes"]:
            # Generate execution time for this node
            exec_time = generate_node_execution_time(node["model"])
            request["node_execution_times"][node["id"]] = exec_time

            # Add configurations for nodes that need them
            if "config_options" in node:
                node_config = {}
                model = MODELS[node["model"]]
                if "config" in model:
                    for option in node["config_options"]:
                        if option in model["config"]:
                            node_config[option] = random.choice(model["config"][option])
                if node_config:
                    request["node_configs"][node["id"]] = node_config

        requests.append(request)

    return requests


def save_to_yaml(data, filename):
    """Save data to YAML file"""
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_to_json(data, filename):
    """Save data to JSON file"""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    # Generate complete system description
    system_config = {"models": MODELS, "workflows": WORKFLOWS}

    # Save system configuration
    save_to_yaml(system_config, "system_config.yaml")
    save_to_json(system_config, "system_config.json")

    # Generate workflow requests
    requests = generate_workflow_requests(num_requests=500, duration_minutes=20)

    # Save requests
    save_to_yaml({"requests": requests}, "workflow_requests.yaml")
    save_to_json({"requests": requests}, "workflow_requests.json")

    # Generate summary statistics
    workflow_counts = {}
    total_gpu_hours = {}

    for req in requests:
        wf_id = req["workflow_id"]
        workflow_counts[wf_id] = workflow_counts.get(wf_id, 0) + 1

        # Calculate GPU hours needed
        workflow = WORKFLOWS[wf_id]
        for node in workflow["nodes"]:
            model = node["model"]
            gpus_needed = MODELS[model]["gpus_required"]
            exec_time = req["node_execution_times"][node["id"]]

            if model not in total_gpu_hours:
                total_gpu_hours[model] = 0
            total_gpu_hours[model] += gpus_needed * exec_time / 3600  # Convert to hours

    print("Workflow Request Generation Complete!")
    print(f"Total requests generated: {len(requests)}")
    print(f"\nWorkflow distribution:")
    for wf_id, count in workflow_counts.items():
        print(f"  {wf_id}: {count} requests ({count / len(requests) * 100:.1f}%)")

    print(f"\nEstimated GPU hours by model:")
    for model, hours in total_gpu_hours.items():
        print(f"  {model}: {hours:.2f} GPU-hours")


if __name__ == "__main__":
    main()

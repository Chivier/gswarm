import json
import numpy as np
import random
import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yaml


class WorkflowRequestGenerator:
    """General workflow request generator with configurable models and workflows"""
    
    def __init__(self, config_path: str):
        """Initialize generator with configuration file"""
        self.config_path = config_path
        self.config = self.load_config()
        self.models = self.config["models"]
        self.workflows = self.config["workflows"]
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_base_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get base model information for a given model ID"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in configuration")
            
        model = self.models[model_id]
        base_model_id = model.get("base_model", model_id)
        
        # If this model references a base model, inherit properties
        if base_model_id != model_id and base_model_id in self.models:
            base_model = self.models[base_model_id].copy()
            # Override with specific model properties
            base_model.update(model)
            return base_model
        
        return model
    
    def generate_poisson_arrivals(self, num_requests: int, duration_minutes: int) -> List[datetime]:
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

    def generate_node_execution_time(self, model_id: str) -> float:
        """Generate execution time based on model type and token count"""
        model = self.get_base_model_info(model_id)
        model_type = model.get("type", "llm")

        if model_type == "image_generation":
            # Fixed time for image generation
            time = np.random.normal(
                model["inference_time_mean"], 
                model["inference_time_std"]
            )
            return max(0.1, time)  # Ensure positive time
        elif model_type == "llm":
            # LLM models: time based on token count
            tokens = np.random.normal(model["token_mean"], model["token_std"])
            tokens = max(1, int(tokens))  # Ensure positive token count
            time = tokens / model["tokens_per_second"]
            return time
        elif model_type == "embedding":
            # Embedding models: typically faster and more predictable
            time = np.random.normal(
                model.get("inference_time_mean", 0.5),
                model.get("inference_time_std", 0.1)
            )
            return max(0.01, time)
        else:
            # Default fallback
            time = np.random.normal(1.0, 0.2)
            return max(0.1, time)

    def generate_workflow_requests(
        self, 
        num_requests: int = 500, 
        duration_minutes: int = 20,
        request_id_prefix: str = "req"
    ) -> List[Dict[str, Any]]:
        """Generate workflow requests with Poisson arrival times"""
        timestamps = self.generate_poisson_arrivals(num_requests, duration_minutes)
        requests = []

        for i, timestamp in enumerate(timestamps):
            # Randomly choose workflow
            workflow_id = random.choice(list(self.workflows.keys()))
            workflow = self.workflows[workflow_id]

            # Generate request
            request = {
                "request_id": f"{request_id_prefix}_{i:04d}",
                "timestamp": timestamp.isoformat(),
                "workflow_id": workflow_id,
                "input_data": {"user_prompt": f"Sample prompt for request {i}"},
                "node_configs": {},
                "node_execution_times": {},  # Pre-generate execution times
                "metadata": {
                    "base_models_used": [],  # Track base models for resource planning
                    "total_estimated_time": 0.0
                }
            }

            total_time = 0.0
            base_models_used = set()

            # Add node-specific configurations and execution times
            for node in workflow["nodes"]:
                model_id = node["model"]
                model = self.get_base_model_info(model_id)
                
                # Track base model usage
                base_model_id = model.get("base_model", model_id)
                base_models_used.add(base_model_id)
                
                # Generate execution time for this node
                exec_time = self.generate_node_execution_time(model_id)
                request["node_execution_times"][node["id"]] = exec_time
                total_time += exec_time

                # Add configurations for nodes that need them
                if "config_options" in node:
                    node_config = {}
                    if "config" in model:
                        for option in node["config_options"]:
                            if option in model["config"]:
                                node_config[option] = random.choice(model["config"][option])
                    if node_config:
                        request["node_configs"][node["id"]] = node_config

            request["metadata"]["base_models_used"] = list(base_models_used)
            request["metadata"]["total_estimated_time"] = total_time
            requests.append(request)

        return requests

    def save_to_yaml(self, data: Any, filename: str) -> None:
        """Save data to YAML file"""
        with open(filename, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def save_to_json(self, data: Any, filename: str) -> None:
        """Save data to JSON file"""
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def generate_and_save(
        self,
        num_requests: int = 500,
        duration_minutes: int = 20,
        output_prefix: str = "generated",
        output_format: str = "both"  # "json", "yaml", or "both"
    ) -> Dict[str, Any]:
        """Generate requests and save to files"""
        
        # Generate workflow requests
        requests = self.generate_workflow_requests(num_requests, duration_minutes)

        # Prepare output data
        output_data = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "config_file": self.config_path,
                "num_requests": num_requests,
                "duration_minutes": duration_minutes,
                "models_available": list(self.models.keys()),
                "workflows_available": list(self.workflows.keys())
            },
            "requests": requests
        }

        # Save files
        if output_format in ["json", "both"]:
            self.save_to_json(output_data, f"{output_prefix}_requests.json")
            
        if output_format in ["yaml", "both"]:
            self.save_to_yaml(output_data, f"{output_prefix}_requests.yaml")

        # Generate and print summary statistics
        return self.generate_summary_stats(requests)

    def generate_summary_stats(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate and print summary statistics"""
        workflow_counts = {}
        total_gpu_hours = {}
        base_model_usage = {}

        for req in requests:
            wf_id = req["workflow_id"]
            workflow_counts[wf_id] = workflow_counts.get(wf_id, 0) + 1

            # Track base model usage
            for base_model in req["metadata"]["base_models_used"]:
                base_model_usage[base_model] = base_model_usage.get(base_model, 0) + 1

            # Calculate GPU hours needed
            workflow = self.workflows[wf_id]
            for node in workflow["nodes"]:
                model_id = node["model"]
                model = self.get_base_model_info(model_id)
                gpus_needed = model["gpus_required"]
                exec_time = req["node_execution_times"][node["id"]]

                if model_id not in total_gpu_hours:
                    total_gpu_hours[model_id] = 0
                total_gpu_hours[model_id] += gpus_needed * exec_time / 3600  # Convert to hours

        stats = {
            "total_requests": len(requests),
            "workflow_distribution": workflow_counts,
            "gpu_hours_by_model": total_gpu_hours,
            "base_model_usage": base_model_usage
        }

        # Print summary
        print("Workflow Request Generation Complete!")
        print(f"Total requests generated: {len(requests)}")
        print(f"\nWorkflow distribution:")
        for wf_id, count in workflow_counts.items():
            print(f"  {wf_id}: {count} requests ({count / len(requests) * 100:.1f}%)")

        print(f"\nEstimated GPU hours by model:")
        for model, hours in total_gpu_hours.items():
            print(f"  {model}: {hours:.2f} GPU-hours")
            
        print(f"\nBase model usage:")
        for base_model, count in base_model_usage.items():
            print(f"  {base_model}: {count} requests")

        return stats


def main():
    parser = argparse.ArgumentParser(description="Generate workflow requests for scheduler testing")
    parser.add_argument(
        "--config", 
        type=str, 
        default="test_baseline.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--num-requests", 
        type=int, 
        default=500,
        help="Number of requests to generate"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=20,
        help="Duration in minutes over which requests arrive"
    )
    parser.add_argument(
        "--output-prefix", 
        type=str, 
        default="generated",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["json", "yaml", "both"],
        default="both",
        help="Output format"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducible results"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Initialize generator
    generator = WorkflowRequestGenerator(args.config)
    
    # Generate and save requests
    stats = generator.generate_and_save(
        num_requests=args.num_requests,
        duration_minutes=args.duration,
        output_prefix=args.output_prefix,
        output_format=args.format
    )

    # Save stats
    generator.save_to_json(stats, f"{args.output_prefix}_stats.json")


if __name__ == "__main__":
    main()

"""
Standalone Strategy API Helper Functions
Provides functions to read device status and model status for strategy planning
"""

import argparse
from typing import Dict, List, Optional, Any
from loguru import logger

from standalone_model import StandaloneStrategyAPI

def get_device_status_list(profiler_port: int = 8095) -> List[Dict]:
    """
    Convenience function to get device status as a list
    
    Args:
        profiler_port: Port of the profiler head (gRPC port)
    
    Returns:
        List[Dict]: Device status list
    """
    profiler_address = f"localhost:{profiler_port}"
    api = StandaloneStrategyAPI("http://localhost:9010", profiler_address)
    return api.read_all_device_status()


def get_model_status_dict(model_port: int = 9010) -> Dict:
    """
    Convenience function to get model status as a dictionary
    
    Args:
        model_port: Port of the model management head
    
    Returns:
        Dict: Model status dictionary
    """
    model_head_url = f"http://localhost:{model_port}"
    api = StandaloneStrategyAPI(model_head_url, "localhost:8095")
    return api.read_all_model_status()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Standalone Strategy API Helper")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8095, 
        help="gRPC port for profiler service (default: 8095)"
    )
    parser.add_argument(
        "--http-port", 
        type=int, 
        default=8096, 
        help="HTTP port for API service (default: 8096)"
    )
    parser.add_argument(
        "--model-port", 
        type=int, 
        default=9010, 
        help="Model management service port (default: 9010)"
    )
    return parser.parse_args()


# Example usage function
def main():
    """Example usage of the API functions"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Build URLs from command line arguments
    model_head_url = f"http://localhost:{args.model_port}"
    profiler_head_address = f"localhost:{args.port}"
    
    logger.info(f"Using model head URL: {model_head_url}")
    logger.info(f"Using profiler head address: {profiler_head_address}")
    logger.info(f"HTTP port configured: {args.http_port}")
    
    # Initialize the API helper
    api = StandaloneStrategyAPI(
        model_head_url=model_head_url,
        profiler_head_address=profiler_head_address
    )
    
    print("=== Reading Device Status ===")
    device_status = api.read_all_device_status()
    print(f"Found {len(device_status)} devices")
    for device in device_status:
        print(f"Node: {device['node_id']}, GPUs: {len(device['gpus'])}")
        for gpu in device['gpus']:
            print(f"  GPU {gpu['gpu_id']}: {gpu['utilization']:.1f}% util, {gpu['memory_used_mb']:.0f}MB used")
    
    print("\n=== Reading Model Status ===")
    model_status = api.read_all_model_status()
    print(f"Found {len(model_status)} models")
    for model_name, model_info in model_status.items():
        print(f"Model: {model_name}")
        print(f"  Status: {model_info['status']}")
        print(f"  Type: {model_info['type']}")
        print(f"  Checkpoints: {model_info['checkpoints']}")
        print(f"  Serving instances: {model_info['serving_instances']}")

if __name__ == "__main__":
    main()

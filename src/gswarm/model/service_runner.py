"""
Service Runner

This module is responsible for running model services as separate processes.
"""

import argparse
import json
import importlib
import sys
from pathlib import Path
from loguru import logger

from .base_service import ServiceConfig


def load_service_class(service_type: str):
    """
    Load a service class dynamically.
    
    Service types can be:
    - Built-in: "vllm", "simple", etc.
    - Module path: "mymodule.MyService"
    - File path: "/path/to/service.py:MyService"
    """
    
    # Built-in services
    builtin_services = {
        "vllm": "gswarm.model.examples.vllm_service:VLLMService",
        "simple": "gswarm.model.examples.simple_service:SimpleMLService",
    }
    
    if service_type in builtin_services:
        service_type = builtin_services[service_type]
    
    # Handle file path format
    if ":" in service_type and "/" in service_type:
        file_path, class_name = service_type.rsplit(":", 1)
        
        # Add the directory to Python path
        file_path = Path(file_path)
        sys.path.insert(0, str(file_path.parent))
        
        # Import the module
        module_name = file_path.stem
        module = importlib.import_module(module_name)
        
        # Get the class
        return getattr(module, class_name)
    
    # Handle module path format
    elif ":" in service_type:
        module_path, class_name = service_type.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    # Try as a module.class format
    elif "." in service_type:
        parts = service_type.rsplit(".", 1)
        if len(parts) == 2:
            module_path, class_name = parts
            try:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except:
                pass
    
    raise ValueError(f"Could not load service type: {service_type}")


def main():
    """Main entry point for running services"""
    parser = argparse.ArgumentParser(description="Model Service Runner")
    parser.add_argument("--name", required=True, help="Service name")
    parser.add_argument("--service-type", required=True, help="Service type or class path")
    parser.add_argument("--config", required=True, help="Service configuration as JSON")
    
    args = parser.parse_args()
    
    try:
        # Parse configuration
        config_dict = json.loads(args.config)
        config = ServiceConfig(**config_dict)
        
        logger.info(f"Starting service {args.name} of type {args.service_type}")
        logger.info(f"Configuration: {config}")
        
        # Load service class
        service_class = load_service_class(args.service_type)
        
        # Create and run service
        service = service_class(config)
        service.run()
        
    except Exception as e:
        logger.error(f"Failed to run service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
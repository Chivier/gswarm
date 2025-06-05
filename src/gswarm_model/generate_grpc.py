#!/usr/bin/env python3
"""
Generate gRPC Python files from protobuf definition.
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger


def generate_grpc_files():
    """Generate gRPC protobuf files from .proto definition"""
    
    # Get the directory where this script is located
    current_dir = Path(__file__).parent
    proto_file = current_dir / "model.proto"
    
    if not proto_file.exists():
        logger.error(f"Protobuf file not found: {proto_file}")
        sys.exit(1)
    
    logger.info(f"Generating gRPC files from {proto_file}")
    
    try:
        # Run protoc to generate Python files
        cmd = [
            "python", "-m", "grpc_tools.protoc",
            f"--proto_path={current_dir}",
            f"--python_out={current_dir}",
            f"--grpc_python_out={current_dir}",
            str(proto_file)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Successfully generated gRPC protobuf files")
        
        # Fix imports in generated files
        fix_imports()
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate gRPC files: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def fix_imports():
    """Fix relative imports in generated gRPC files"""
    current_dir = Path(__file__).parent
    
    # Fix model_pb2_grpc.py imports
    grpc_file = current_dir / "model_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        
        # Replace absolute import with relative import
        content = content.replace(
            "import model_pb2 as model__pb2",
            "from . import model_pb2 as model__pb2"
        )
        
        grpc_file.write_text(content)
        logger.info("Fixed imports in model_pb2_grpc.py")


if __name__ == "__main__":
    generate_grpc_files() 
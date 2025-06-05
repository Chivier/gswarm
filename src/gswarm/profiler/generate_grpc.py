#!/usr/bin/env python3
"""
Script to generate gRPC Python files from gswarm.profiler.proto definition
"""

import subprocess
import sys
import re
from pathlib import Path


def fix_imports_in_file(file_path: Path):
    """Fix imports in a single file"""
    with open(file_path, "r") as f:
        content = f.read()

    # Fix the import in profiler_pb2_grpc.py that imports profiler_pb2
    content = re.sub(
        r"^import profiler_pb2 as profiler__pb2\b",
        "from gswarm.profiler import profiler_pb2 as profiler__pb2",
        content,
        flags=re.MULTILINE,
    )

    with open(file_path, "w") as f:
        f.write(content)


def generate_grpc_files():
    """Generate gRPC Python files from profiler.proto"""
    current_dir = Path(__file__).parent
    proto_file = current_dir / "profiler.proto"

    if not proto_file.exists():
        print(f"Error: {proto_file} not found!")
        sys.exit(1)

    # Generate Python gRPC files
    cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={current_dir}",
        f"--python_out={current_dir}",
        f"--grpc_python_out={current_dir}",
        str(proto_file),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error generating gRPC files:")
        print(result.stderr)
        sys.exit(1)

    print("Successfully generated gRPC Python files:")
    print(f"  - {current_dir}/profiler_pb2.py")
    print(f"  - {current_dir}/profiler_pb2_grpc.py")

    # Fix imports in generated files
    pb2_grpc_file = current_dir / "profiler_pb2_grpc.py"
    if pb2_grpc_file.exists():
        fix_imports_in_file(pb2_grpc_file)
        print(f"Fixed imports in {pb2_grpc_file}")


if __name__ == "__main__":
    generate_grpc_files()

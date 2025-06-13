#!/usr/bin/env python3
"""
Script to fix imports in all files after generating protobuf files
"""

import re
from pathlib import Path


def fix_imports_in_file(file_path: Path):
    """Fix imports in a single file"""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        original_content = content

        # Fix direct imports of profiler_pb2 and profiler_pb2_grpc to use relative imports
        content = re.sub(
            r"^import profiler_pb2\b", "from gswarm.profiler import profiler_pb2", content, flags=re.MULTILINE
        )
        content = re.sub(
            r"^import profiler_pb2_grpc\b", "from gswarm.profiler import profiler_pb2_grpc", content, flags=re.MULTILINE
        )

        # Fix the import in profiler_pb2_grpc.py that imports profiler_pb2
        content = re.sub(
            r"^import profiler_pb2 as profiler__pb2\b",
            "from gswarm.profiler import profiler_pb2 as profiler__pb2",
            content,
            flags=re.MULTILINE,
        )

        if content != original_content:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Fixed imports in {file_path}")
        else:
            print(f"No changes needed in {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """Fix imports in all relevant files"""
    current_dir = Path(__file__).parent

    # Files to fix
    files_to_fix = [
        "profiler_pb2_grpc.py",
        "head.py",
        "client.py",
        "cli.py",
        "client_grpc.py",
    ]

    for filename in files_to_fix:
        file_path = current_dir / filename
        if file_path.exists():
            fix_imports_in_file(file_path)
        else:
            print(f"Warning: {file_path} not found")


if __name__ == "__main__":
    main()

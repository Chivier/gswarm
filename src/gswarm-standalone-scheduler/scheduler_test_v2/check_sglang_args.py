#!/usr/bin/env python3
"""
Check available arguments for sglang.launch_server
"""

import subprocess
import sys

print("Checking SGLang launch_server arguments...")
print("=" * 70)

# Run with --help to see all arguments
result = subprocess.run(
    [sys.executable, "-m", "sglang.launch_server", "--help"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("Available arguments for sglang.launch_server:")
    print("-" * 70)
    print(result.stdout)
else:
    print("Error getting help:")
    print(result.stderr)

# Also try to get version info
print("\n" + "=" * 70)
print("SGLang version info:")
try:
    import sglang
    print(f"Version: {getattr(sglang, '__version__', 'Unknown')}")
except:
    print("Could not import sglang")
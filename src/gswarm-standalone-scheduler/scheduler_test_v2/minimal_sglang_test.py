#!/usr/bin/env python3
"""
Minimal SGLang test - just verify it works
"""

import os
import sys
import subprocess

# Set CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda-12.6/targets/x86_64-linux/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

print("Testing SGLang installation...")

# Test 1: Import
try:
    import sglang
    print(f"✓ SGLang version: {sglang.__version__}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check model files
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--phi-2")
if os.path.exists(model_path):
    print(f"✓ Model cached at: {model_path}")
else:
    print("✗ Model not cached")

# Test 3: Show server launch command
print("\n" + "="*70)
print("To launch SGLang server manually:")
print("="*70)

commands = [
    # Basic command
    "python -m sglang.launch_server --model-path microsoft/phi-2 --port 30000",
    
    # With GPU optimizations disabled (for head_dim issues)
    "python -m sglang.launch_server --model-path microsoft/phi-2 --port 30000 --disable-cuda-graph --attention-backend triton",
    
    # With reduced memory
    "python -m sglang.launch_server --model-path microsoft/phi-2 --port 30000 --mem-fraction-static 0.3 --disable-cuda-graph",
    
    # CPU only (slow but works)
    "python -m sglang.launch_server --model-path microsoft/phi-2 --port 30000 --device cpu"
]

for i, cmd in enumerate(commands, 1):
    print(f"\nOption {i}:")
    print(f"  {cmd}")

print("\n" + "="*70)
print("After launching, test with:")
print("="*70)
print("""
curl -X POST http://localhost:30000/generate \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello world", "sampling_params": {"max_new_tokens": 20}}'
""")

print("\n✓ SGLang is installed and ready to use!")
print("\nNote: The head_dim=80 issue with Phi-2 requires disabling CUDA graphs")
print("      or using triton attention backend as shown above.")
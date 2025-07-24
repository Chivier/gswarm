#!/usr/bin/env python3
"""Test nvitop GPU information retrieval."""

import nvitop

print("Testing nvitop GPU information retrieval...\n")

devices = nvitop.Device.all()
print(f"Found {len(devices)} GPU devices\n")

for i, device in enumerate(devices):
    print(f"GPU {i}:")
    print(f"  Name: {device.name()}")
    print(f"  Memory Total: {device.memory_total()} bytes")
    print(f"  Memory Total (MB): {device.memory_total() // (1024 * 1024)} MB")
    print(f"  Memory Used: {device.memory_used()} bytes")
    print(f"  Memory Used (MB): {device.memory_used() // (1024 * 1024)} MB")
    print(f"  GPU Utilization: {device.gpu_percent()}%")
    print(f"  Memory Utilization: {device.memory_percent()}%")
    print()
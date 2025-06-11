"""
Example of using the updated model management system.
"""

import asyncio
from gswarm.model.fastapi_client import ModelClient
import time


async def main():
    # Initialize client
    client = ModelClient("http://localhost:8100")

    # Register node
    if client.register_node():
        print("✓ Node registered successfully")

    # Register a model
    success, instance_id = client.register_model(
        name="microsoft/phi-2", model_type="llm", source_url="https://huggingface.co/microsoft/phi-2"
    )
    print(f"✓ Model registered: {instance_id}")

    # Download model to disk
    if client.download_model("microsoft/phi-2", "https://huggingface.co/microsoft/phi-2", "disk"):
        print("✓ Download started to disk")

    # Wait for download to complete
    print("Waiting for download...")
    while True:
        status = client.get_download_status("microsoft/phi-2")
        if status and status["status"] == "ready":
            print("✓ Download completed")
            break
        time.sleep(5)

    # Preload to DRAM for faster GPU loading
    if client.preload_to_dram("microsoft/phi-2"):
        print("✓ Model preloaded to DRAM")

    # Serve single instance on GPU
    success, server_info = client.load_to_gpu(
        model_name="microsoft/phi-2",
        source_device="dram",  # Load from DRAM (faster)
        gpu_device="gpu0",
        port=8001,
    )
    if success:
        print(f"✓ Model serving on GPU: {server_info['url']}")
        print(f"  Instance ID: {server_info['instance_id']}")

    # Serve multiple instances on same GPU
    print("\nStarting multiple instances on same GPU...")
    instances = client.serve_multiple_instances(
        model_name="microsoft/phi-2",
        source_device="dram",
        gpu_device="gpu0",
        num_instances=3,
        base_port=8010,
        config={"gpu_memory_utilization": 0.3},  # Use less memory per instance
    )
    print(f"✓ Started {len(instances)} instances")

    # List all serving instances
    all_instances = client.list_serving_instances()
    print(f"\nTotal serving instances: {len(all_instances)}")
    for inst in all_instances:
        print(f"  - {inst['model_name']} on {inst['gpu_device']} (port {inst['port']}, status: {inst['status']})")

    # Copy between devices
    print("\nTesting copy operations...")

    # Copy from DRAM to another GPU (will start serving)
    success = client.copy_model(
        model_name="microsoft/phi-2", source_device="dram", target_device="gpu1", keep_source=True
    )
    if success:
        print("✓ Copied to GPU1 (serving started)")

    # Get model details
    details = client.get_model_details("microsoft/phi-2")
    print(f"\nModel details:")
    print(f"  Checkpoints: {details['checkpoints']}")
    print(f"  Serving instances: {len(details['serving_instances'])}")

    # Stop specific instance
    if instances:
        instance_to_stop = instances[0]["instance_id"]
        if client.stop_serving("microsoft/phi-2", instance_to_stop):
            print(f"\n✓ Stopped instance: {instance_to_stop}")

    # Offload from GPU
    if server_info:
        if client.offload_from_gpu("microsoft/phi-2", server_info["instance_id"]):
            print(f"✓ Offloaded instance from GPU")


if __name__ == "__main__":
    asyncio.run(main())

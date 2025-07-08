#!/usr/bin/env python3
"""
Demo script showing how to use the ComfyUI workflow converter
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"🚀 Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS!")
        if result.stdout.strip():
            print("📄 Output:")
            print(result.stdout)
    else:
        print("❌ FAILED!")
        if result.stderr.strip():
            print("💥 Error:")
            print(result.stderr)
        if result.stdout.strip():
            print("📄 Output:")
            print(result.stdout)
    
    return result.returncode == 0


def main():
    """Run demonstration of ComfyUI converter."""
    print("🎬 ComfyUI Workflow Converter Demo")
    
    # Change to the correct directory
    demo_dir = "/Users/ray/Projects/gswarm/src/gswarm-standalone-scheduler/scheduler_dev_input_demo"
    workflow_dir = "/Users/ray/Projects/gswarm/examples/ComfyUI-Workflows-ZHO/workflows/zho"
    
    print(f"📁 Working directory: {demo_dir}")
    print(f"📁 Workflow directory: {workflow_dir}")
    
    # Test 1: Convert a single workflow
    single_workflow = f"{workflow_dir}/FLUX.1 SCHNELL 1.0.json"
    cmd1 = [
        "uv", "run", "comfyui_to_config.py",
        "--workflow", single_workflow,
        "--output", "flux_config.json"
    ]
    
    success1 = run_command(cmd1, "Converting single FLUX workflow")
    
    # Test 2: Convert all workflows
    cmd2 = [
        "uv", "run", "comfyui_to_config.py", 
        "--workflow-dir", workflow_dir,
        "--output", "all_comfyui_config.json"
    ]
    
    success2 = run_command(cmd2, "Converting all ComfyUI workflows")
    
    # Test 3: Generate request sequences from converted config
    if success2:
        cmd3 = [
            "uv", "run", "model_seq_gen.py",
            "--config", "all_comfyui_config.json",
            "--num-requests", "20",
            "--duration", "10",
            "--output-prefix", "comfyui_demo",
            "--format", "yaml",
            "--seed", "123"
        ]
        
        success3 = run_command(cmd3, "Generating request sequences from converted workflows")
    
        # Show generated files
        if success3:
            print(f"\n{'='*60}")
            print("📁 Generated Files:")
            print(f"{'='*60}")
            
            demo_path = Path(demo_dir)
            generated_files = [
                "flux_config.json",
                "all_comfyui_config.json", 
                "comfyui_demo_requests.yaml",
                "comfyui_demo_stats.json"
            ]
            
            for filename in generated_files:
                filepath = demo_path / filename
                if filepath.exists():
                    size = filepath.stat().st_size
                    print(f"  ✅ {filename} ({size:,} bytes)")
                else:
                    print(f"  ❌ {filename} (not found)")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Demo Summary")
    print(f"{'='*60}")
    print("This demo showed how to:")
    print("1. ✅ Convert individual ComfyUI workflows to scheduler config format")
    print("2. ✅ Batch convert entire directories of workflows") 
    print("3. ✅ Use converted configs to generate realistic AI workload sequences")
    print("4. ✅ Integration with existing scheduler development tools")
    
    print(f"\n📚 Next Steps:")
    print("• Customize model resource requirements in comfyui_to_config.py")
    print("• Add new ComfyUI node type mappings")
    print("• Use generated configs for scheduler algorithm testing")
    print("• Integrate with real ComfyUI model deployment")


if __name__ == "__main__":
    # Change to the correct directory
    demo_dir = "/Users/ray/Projects/gswarm/src/gswarm-standalone-scheduler/scheduler_dev_input_demo"
    import os
    os.chdir(demo_dir)
    
    main()

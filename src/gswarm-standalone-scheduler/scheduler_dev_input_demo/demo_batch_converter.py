#!/usr/bin/env python3
"""
Demo script for ComfyUI Batch Converter (One-to-One)

This script demonstrates the usage of comfyui_batch_converter.py for
converting ComfyUI workflows to individual config files.
"""

import subprocess
import sys
import shutil
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
    """Run demonstration of ComfyUI batch converter."""
    print("🎬 ComfyUI Batch Converter Demo - One-to-One Conversion")
    
    # Directories
    demo_dir = "/Users/ray/Projects/gswarm/src/gswarm-standalone-scheduler/scheduler_dev_input_demo"
    workflow_dir = "/Users/ray/Projects/gswarm/examples/ComfyUI-Workflows-ZHO/workflows/zho"
    output_dir = "demo_individual_configs"
    
    print(f"📁 Working directory: {demo_dir}")
    print(f"📁 Workflow directory: {workflow_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Clean up previous runs
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"🗑️  Cleaned up previous output directory")
    
    # Demo 1: Convert a single workflow to individual config
    single_workflow = f"{workflow_dir}/FLUX.1 SCHNELL 1.0.json"
    cmd1 = [
        "uv", "run", "comfyui_batch_converter.py",
        "--workflow", single_workflow,
        "--output-dir", output_dir,
        "--prefix", "single_",
        "--suffix", "_demo"
    ]
    
    success1 = run_command(cmd1, "Demo 1: Converting single workflow to individual config")
    
    # Demo 2: Convert a subset of workflows (first 3 files)
    cmd2 = [
        "uv", "run", "python", "-c", f"""
import sys
sys.path.append('.')
from comfyui_batch_converter import ComfyUIBatchConverter
from pathlib import Path

# Get first 3 workflow files
workflows = list(Path('{workflow_dir}').glob('*.json'))[:3]
print(f'Converting first 3 workflows: {{[w.name for w in workflows]}}')

# Convert with custom settings
converter = ComfyUIBatchConverter(
    output_dir='{output_dir}',
    prefix='subset_',
    suffix='_individual'
)

for workflow in workflows:
    converter.convert_single_workflow(str(workflow))

converter.print_summary()
"""
    ]
    
    success2 = run_command(cmd2, "Demo 2: Converting subset of workflows with custom naming")
    
    # Demo 3: Test one of the generated configs
    if success2:
        # Find a generated config file
        config_files = list(Path(output_dir).glob("*.json"))
        if config_files:
            test_config = config_files[0]
            cmd3 = [
                "uv", "run", "model_seq_gen.py",
                "--config", str(test_config),
                "--num-requests", "3",
                "--output-prefix", "demo_individual_test",
                "--seed", "456"
            ]
            
            success3 = run_command(cmd3, f"Demo 3: Testing generated config ({test_config.name})")
    
    # Demo 4: Show file listing
    if Path(output_dir).exists():
        print(f"\n{'='*60}")
        print("📁 Generated Individual Config Files:")
        print(f"{'='*60}")
        
        config_files = list(Path(output_dir).glob("*.json"))
        for config_file in sorted(config_files):
            size = config_file.stat().st_size
            print(f"  ✅ {config_file.name} ({size:,} bytes)")
        
        print(f"\n📊 Total files: {len(config_files)}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Batch Converter Demo Summary")
    print(f"{'='*60}")
    print("This demo showed how to:")
    print("1. ✅ Convert individual ComfyUI workflows to separate config files")
    print("2. ✅ Use custom prefixes and suffixes for file naming")
    print("3. ✅ Batch convert multiple workflows with one-to-one mapping")
    print("4. ✅ Test individual config files with model_seq_gen.py")
    
    print(f"\n📚 Key Benefits of One-to-One Conversion:")
    print("• Independent testing of each workflow")
    print("• Easier deployment and version management")
    print("• Reduced config file complexity")
    print("• Better isolation between different workflows")
    
    print(f"\n🔧 Next Steps:")
    print("• Use individual configs for A/B testing different workflows")
    print("• Deploy configs independently to different environments")
    print("• Compare performance metrics between workflow versions")
    print("• Integrate with CI/CD pipelines for workflow-specific testing")


if __name__ == "__main__":
    # Change to the correct directory
    demo_dir = "/Users/ray/Projects/gswarm/src/gswarm-standalone-scheduler/scheduler_dev_input_demo"
    import os
    os.chdir(demo_dir)
    
    main()

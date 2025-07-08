#!/usr/bin/env python3
"""
Test script to validate converted ComfyUI workflows work with the scheduler.
"""

import json
import subprocess
import sys
from pathlib import Path


def test_config_generation(config_file: str, num_requests: int = 10):
    """Test that a config file can generate request sequences."""
    print(f"🧪 Testing config file: {config_file}")
    
    # Check if config file exists
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Load and validate config structure
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if 'models' not in config or 'workflows' not in config:
            print("❌ Config missing required 'models' or 'workflows' sections")
            return False
        
        print(f"✅ Config loaded: {len(config['models'])} models, {len(config['workflows'])} workflows")
        
        # Test with model_seq_gen.py
        cmd = [
            'uv', 'run', 'model_seq_gen.py',
            '--config', config_file,
            '--num-requests', str(num_requests),
            '--output-prefix', 'test_output',
            '--seed', '42'
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully generated request sequences!")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to generate request sequences")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing config: {e}")
        return False


def main():
    """Main test function."""
    print("🔧 ComfyUI Config Converter Test Suite")
    print("=" * 50)
    
    # Test the generated config files
    test_files = [
        'test_config.json',
        'config.json'
    ]
    
    results = []
    for config_file in test_files:
        if Path(config_file).exists():
            success = test_config_generation(config_file, num_requests=5)
            results.append((config_file, success))
        else:
            print(f"⏭️  Skipping {config_file} (not found)")
    
    # Summary
    print("\n📊 Test Results:")
    print("-" * 30)
    passed = 0
    for config_file, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{config_file}: {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total and total > 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

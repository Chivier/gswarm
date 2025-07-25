#!/usr/bin/env python3
"""
Run all GSwarm tests with proper virtual environment handling
"""
import subprocess
import sys
import os
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def print_header():
    """Print test suite header"""
    print("=" * 41)
    print("Running GSwarm Test Suite")
    print("=" * 41)

def check_gswarm():
    """Check if gswarm is available"""
    try:
        result = subprocess.run(["which", "gswarm"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{GREEN}✓ GSwarm is available{NC}")
            return True
    except FileNotFoundError:
        pass
    
    print(f"{RED}Error: gswarm command not found{NC}")
    print("Please ensure GSwarm is installed and your virtual environment is activated.")
    print("\nTo activate virtual environment:")
    print("  source ../../.venv/bin/activate")
    print("\nOr install gswarm:")
    print("  pip install -e ../..")
    return False

def run_test(test_name, test_path):
    """Run a single test and return success status"""
    print(f"\n{YELLOW}Running {test_name}...{NC}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        
        if result.returncode == 0:
            print(f"{GREEN}✓ {test_name} PASSED{NC}")
            return True
        else:
            print(f"{RED}✗ {test_name} FAILED{NC}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{RED}✗ {test_name} TIMED OUT{NC}")
        return False
    except Exception as e:
        print(f"{RED}✗ {test_name} ERROR: {e}{NC}")
        return False

def main():
    """Run all tests"""
    print_header()
    
    # Check if gswarm is available
    if not check_gswarm():
        sys.exit(1)
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Define tests to run
    tests = [
        ("Basic Functionality Test", test_dir / "basic" / "test_basic_functionality.py"),
        ("Profiler Test", test_dir / "profiler" / "test_profiler.py"),
        ("Model Serve Test", test_dir / "model" / "test_model_serve.py"),
        ("Prediction Test", test_dir / "prediction" / "test_prediction.py"),
        ("Data Handling Test", test_dir / "data" / "test_data_handling.py"),
    ]
    
    # Run tests
    print("\nStarting test execution...")
    total_tests = len(tests)
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_path in tests:
        if test_path.exists():
            if run_test(test_name, test_path):
                passed_tests += 1
            else:
                failed_tests += 1
        else:
            print(f"{RED}✗ {test_name} NOT FOUND at {test_path}{NC}")
            failed_tests += 1
    
    # Print summary
    print("\n" + "=" * 41)
    print("Test Summary")
    print("=" * 41)
    print(f"Total Tests: {total_tests}")
    print(f"{GREEN}Passed: {passed_tests}{NC}")
    print(f"{RED}Failed: {failed_tests}{NC}")
    
    if failed_tests == 0:
        print(f"\n{GREEN}All tests passed!{NC}")
        return 0
    else:
        print(f"\n{RED}Some tests failed. Please check the output above.{NC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
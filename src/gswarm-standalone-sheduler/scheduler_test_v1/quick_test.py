#!/usr/bin/env python3
"""Quick test script to verify performance comparison setup"""

import subprocess
import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """Run a quick test with 2 and 4 GPUs"""
    results = []
    
    # Test configurations
    test_cases = [
        ("baseline", 2, "simple_config.json", "simple_requests.yaml"),
        ("offline", 2, "simple_config.json", "simple_requests.yaml"),
        ("baseline", 4, "simple_config.json", "simple_requests.yaml"),
        ("offline", 4, "simple_config.json", "simple_requests.yaml"),
    ]
    
    for scheduler, gpus, config, requests in test_cases:
        logger.info(f"Testing {scheduler} with {gpus} GPUs")
        
        script = "baseline.py" if scheduler == "baseline" else "offline_scheduler.py"
        cmd = ["python", script, "--gpus", str(gpus), "--config", config, "--requests", requests]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read results
            log_file = f"{scheduler}_execution_log.json"
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            results.append({
                'scheduler': scheduler,
                'gpus': gpus,
                'makespan': data['summary']['makespan'],
                'nodes': data['summary']['total_nodes_executed']
            })
            
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Display results
    df = pd.DataFrame(results)
    print("\nQuick Test Results:")
    print(df)
    
    # Simple plot
    plt.figure(figsize=(8, 6))
    for scheduler in ['baseline', 'offline']:
        data = df[df['scheduler'] == scheduler]
        plt.plot(data['gpus'], data['makespan'], 'o-', label=scheduler, linewidth=2, markersize=10)
    
    plt.xlabel('Number of GPUs')
    plt.ylabel('Makespan (seconds)')
    plt.title('Quick Performance Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('quick_test_result.png')
    plt.show()

if __name__ == "__main__":
    quick_test()
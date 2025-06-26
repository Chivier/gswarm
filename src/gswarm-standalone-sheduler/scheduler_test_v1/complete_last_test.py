#!/usr/bin/env python3
"""
Complete the last missing test - baseline with 30 GPUs on complex config
"""

import subprocess
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Run the last test
logger.info("Running baseline with 30 GPUs on complex_config.json...")

cmd = [
    "python", "baseline.py",
    "--gpus", "30",
    "--simulate", "false",
    "--config", "complex_config.json",
    "--requests", "complex_requests.yaml"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
    
    # Parse execution log
    with open("baseline_execution_log.json", 'r') as f:
        log_data = json.load(f)
    
    summary = log_data['summary']
    
    # Calculate GPU utilization
    gpu_utilization = []
    if 'executions' in log_data:
        gpu_tasks = {}
        for exec in log_data['executions']:
            gpu_id = exec['gpu_id']
            if gpu_id not in gpu_tasks:
                gpu_tasks[gpu_id] = 0
            gpu_tasks[gpu_id] += 1
        
        for i in range(30):
            gpu_utilization.append(gpu_tasks.get(i, 0))
    
    # Extract metrics
    metrics = {
        'scheduler': 'baseline',
        'gpu_count': 30,
        'config': 'complex_config',
        'makespan': summary['makespan'],
        'total_nodes': summary['total_nodes_executed'],
        'total_requests': summary['total_requests'],
        'model_switches': summary.get('total_model_switches', -1),
        'switch_time': summary.get('total_switch_time', -1),
        'throughput': summary['total_nodes_executed'] / summary['makespan'] if summary['makespan'] > 0 else 0,
        'gpu_utilization_std': np.std(gpu_utilization) if gpu_utilization else 0
    }
    
    logger.info(f"  Makespan: {metrics['makespan']:.2f}s, Throughput: {metrics['throughput']:.3f} tasks/s")
    
    # Append to existing results
    csv_file = "benchmark_results/benchmark_results_20250626_083257.csv"
    df_existing = pd.read_csv(csv_file)
    df_new = pd.DataFrame([metrics])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_file, index=False)
    
    logger.info(f"Updated results saved to {csv_file}")
    logger.info("All benchmark tests completed!")
    
except Exception as e:
    logger.error(f"Error running final test: {e}")

# Generate final plots
logger.info("Generating final comprehensive plots...")
subprocess.run(["python", "plot_benchmark_results.py"])
#!/usr/bin/env python3
"""
Continue benchmark from where it left off - complex config with 6+ GPUs
"""

import subprocess
import json
import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmark_continue.log")
    ]
)
logger = logging.getLogger(__name__)


def run_scheduler(scheduler_type: str, gpu_count: int, 
                 config_file: str, request_file: str) -> dict:
    """Run a single scheduler test and return metrics"""
    
    logger.info(f"Running {scheduler_type} with {gpu_count} GPUs on {config_file}")
    
    # Determine script name
    script_name = "baseline.py" if scheduler_type == "baseline" else "offline_scheduler.py"
    
    # Run scheduler
    cmd = [
        "python", script_name,
        "--gpus", str(gpu_count),
        "--simulate", "false",
        "--config", config_file,
        "--requests", request_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        
        # Parse execution log
        log_file = "baseline_execution_log.json" if scheduler_type == "baseline" else "offline_execution_log.json"
        
        with open(log_file, 'r') as f:
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
            
            for i in range(gpu_count):
                gpu_utilization.append(gpu_tasks.get(i, 0))
        
        # Extract metrics
        import numpy as np
        metrics = {
            'scheduler': scheduler_type,
            'gpu_count': gpu_count,
            'config': Path(config_file).stem,
            'makespan': summary['makespan'],
            'total_nodes': summary['total_nodes_executed'],
            'total_requests': summary['total_requests'],
            'model_switches': summary.get('total_model_switches', -1),
            'switch_time': summary.get('total_switch_time', -1),
            'throughput': summary['total_nodes_executed'] / summary['makespan'] if summary['makespan'] > 0 else 0,
            'gpu_utilization_std': np.std(gpu_utilization) if gpu_utilization else 0
        }
        
        logger.info(f"  Makespan: {metrics['makespan']:.2f}s, Throughput: {metrics['throughput']:.3f} tasks/s")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout running {scheduler_type} with {gpu_count} GPUs")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {scheduler_type}: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error processing results: {e}")
        return None


def main():
    """Continue benchmark for complex config with remaining GPU counts"""
    
    # Load existing results
    existing_csv = "benchmark_results/benchmark_results_20250626_083257.csv"
    df_existing = pd.read_csv(existing_csv)
    
    # Define remaining tests for complex config
    gpu_counts = [6, 8, 10, 12, 16, 20, 24, 30]
    config_file = "complex_config.json"
    request_file = "complex_requests.yaml"
    
    logger.info("="*60)
    logger.info("CONTINUING SCHEDULER PERFORMANCE BENCHMARKS")
    logger.info("="*60)
    logger.info(f"Testing complex config with GPU counts: {gpu_counts}")
    logger.info("="*60)
    
    new_results = []
    
    for gpu_count in gpu_counts:
        for scheduler in ["baseline", "offline"]:
            logger.info(f"\nTesting {scheduler} with {gpu_count} GPUs...")
            
            metrics = run_scheduler(scheduler, gpu_count, config_file, request_file)
            if metrics:
                new_results.append(metrics)
                
                # Append to existing results
                df_new = pd.DataFrame([metrics])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                
                # Save updated results
                df_combined.to_csv(existing_csv, index=False)
                logger.info(f"Updated results saved to {existing_csv}")
    
    if new_results:
        logger.info("="*60)
        logger.info(f"BENCHMARK CONTINUATION COMPLETED!")
        logger.info(f"Added {len(new_results)} new results")
        logger.info("="*60)
        
        # Generate final plots
        logger.info("Generating updated plots...")
        subprocess.run(["python", "plot_benchmark_results.py"])
    else:
        logger.error("No new results collected!")


if __name__ == "__main__":
    main()
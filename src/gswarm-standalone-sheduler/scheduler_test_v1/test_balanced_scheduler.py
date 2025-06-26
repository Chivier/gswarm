#!/usr/bin/env python3
"""
Test online scheduler with different load balancing strategies
"""

import subprocess
import json
import sys

def run_test(delta, work_stealing=True):
    """Run online scheduler with specific parameters"""
    cmd = [
        "python", "online_scheduler.py",
        "--gpus", "4",
        "--config", "balanced_config.json",
        "--requests", "simple_requests.yaml",
        "--simulate", "true",
        "--delta", str(delta)
    ]
    
    if not work_stealing:
        cmd.append("--no-work-stealing")
    
    print(f"\nRunning with δ={delta}, work_stealing={work_stealing}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        # Parse metrics from output
        output_lines = result.stdout.split('\n')
        metrics = {}
        
        for line in output_lines:
            if "- OnlineGreedyScheduler - INFO -   GPU" in line and "utilization" in line:
                # Parse GPU utilization line from logger output
                # Example: "2025-06-26 11:40:46,807 - OnlineGreedyScheduler - INFO -   GPU 0: 67.8% utilization, 1221 executions"
                try:
                    parts = line.split("GPU")[1].split(":")
                    gpu_id = int(parts[0].strip())
                    
                    info_part = parts[1]
                    util_match = info_part.split('%')[0].strip()
                    exec_match = info_part.split('executions')[0].split(',')[-1].strip()
                    
                    metrics[f"gpu_{gpu_id}_util"] = float(util_match)
                    metrics[f"gpu_{gpu_id}_exec"] = int(exec_match)
                except:
                    pass
            
            elif "Average waiting time:" in line:
                metrics["avg_wait"] = float(line.split(':')[1].split()[0])
            elif "P99 waiting time:" in line:
                metrics["p99_wait"] = float(line.split(':')[1].split()[0])
        
        return metrics
    
    except Exception as e:
        print(f"Exception: {e}")
        return None

def calculate_balance_score(metrics):
    """Calculate load balance score (lower is better)"""
    if not metrics:
        return float('inf')
    
    # Get execution counts
    exec_counts = []
    for i in range(4):
        if f"gpu_{i}_exec" in metrics:
            exec_counts.append(metrics[f"gpu_{i}_exec"])
    
    if not exec_counts:
        return float('inf')
    
    # Calculate standard deviation
    avg_exec = sum(exec_counts) / len(exec_counts)
    variance = sum((x - avg_exec) ** 2 for x in exec_counts) / len(exec_counts)
    std_dev = variance ** 0.5
    
    # Balance score = coefficient of variation
    if avg_exec > 0:
        cv = std_dev / avg_exec
    else:
        cv = float('inf')
    
    return cv

def main():
    print("Testing Online Scheduler Load Balancing")
    print("=" * 60)
    
    # Test different delta values
    test_configs = [
        (0.0, True),   # No load balancing
        (0.5, True),   # Default
        (1.0, True),   # Stronger load balancing
        (2.0, True),   # Very strong load balancing
        (5.0, True),   # Extreme load balancing
        (2.0, False),  # Strong load balancing without work stealing
    ]
    
    results = []
    
    for delta, work_stealing in test_configs:
        metrics = run_test(delta, work_stealing)
        
        if metrics:
            balance_score = calculate_balance_score(metrics)
            
            result = {
                "delta": delta,
                "work_stealing": work_stealing,
                "balance_score": balance_score,
                "avg_wait": metrics.get("avg_wait", "N/A"),
                "p99_wait": metrics.get("p99_wait", "N/A"),
            }
            
            # Add GPU stats
            for i in range(4):
                result[f"gpu_{i}"] = metrics.get(f"gpu_{i}_exec", 0)
            
            results.append(result)
    
    # Print results table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Delta':>6} {'Work Steal':>11} {'Balance Score':>14} {'Avg Wait':>10} {'P99 Wait':>10} "
          f"{'GPU0':>6} {'GPU1':>6} {'GPU2':>6} {'GPU3':>6}")
    print("-" * 100)
    
    for r in results:
        ws = "Yes" if r["work_stealing"] else "No"
        print(f"{r['delta']:>6.1f} {ws:>11} {r['balance_score']:>14.3f} "
              f"{r['avg_wait']:>10.0f} {r['p99_wait']:>10.0f} "
              f"{r['gpu_0']:>6} {r['gpu_1']:>6} {r['gpu_2']:>6} {r['gpu_3']:>6}")
    
    # Find best configuration
    best = min(results, key=lambda x: x["balance_score"])
    print(f"\nBest load balance: δ={best['delta']}, work_stealing={best['work_stealing']}")
    print(f"Balance score: {best['balance_score']:.3f}")
    
    # Also consider latency
    best_latency = min(results, key=lambda x: x["avg_wait"] if x["avg_wait"] != "N/A" else float('inf'))
    print(f"\nBest latency: δ={best_latency['delta']}, work_stealing={best_latency['work_stealing']}")
    print(f"Average wait: {best_latency['avg_wait']:.0f}s")

if __name__ == "__main__":
    main()
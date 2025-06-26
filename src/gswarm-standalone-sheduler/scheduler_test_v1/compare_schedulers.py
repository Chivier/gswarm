#!/usr/bin/env python3
"""
Compare offline and online scheduler performance
"""

import subprocess
import json
import sys

def run_scheduler(scheduler_type, config_file, requests_file, gpus=2):
    """Run a scheduler and return the metrics"""
    if scheduler_type == "offline":
        cmd = [
            "python", "offline_scheduler.py",
            "--gpus", str(gpus),
            "--config", config_file,
            "--requests", requests_file,
            "--simulate", "true"
        ]
        log_file = "offline_execution_log.json"
    else:  # online
        cmd = [
            "python", "online_scheduler.py",
            "--gpus", str(gpus),
            "--config", config_file,
            "--requests", requests_file,
            "--simulate", "true"
        ]
        log_file = "online_execution_log.json"
    
    print(f"\nRunning {scheduler_type} scheduler...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {scheduler_type} scheduler:")
            print(result.stderr)
            return None
        
        # Read the execution log
        with open(log_file, 'r') as f:
            data = json.load(f)
            return data.get("summary", {})
    
    except Exception as e:
        print(f"Exception running {scheduler_type} scheduler: {e}")
        return None

def compare_metrics(offline_metrics, online_metrics):
    """Compare metrics between offline and online schedulers"""
    print("\n" + "=" * 80)
    print("SCHEDULER COMPARISON")
    print("=" * 80)
    
    # Metrics to compare
    metrics = [
        ("Total tasks", "total_nodes_executed"),
        ("Makespan", "makespan"),
        ("Model switches", "total_model_switches"),
        ("Switch time", "total_switch_time"),
        ("Avg waiting time", "avg_waiting_time"),
        ("P99 waiting time", "p99_waiting_time"),
        ("Avg response time", "avg_response_time"),
        ("P99 response time", "p99_response_time"),
        ("Avg request response", "avg_request_response_time"),
        ("P99 request response", "p99_request_response_time")
    ]
    
    print(f"{'Metric':<25} {'Offline':>15} {'Online':>15} {'Difference':>15}")
    print("-" * 70)
    
    for display_name, key in metrics:
        offline_val = offline_metrics.get(key, "N/A")
        online_val = online_metrics.get(key, "N/A")
        
        # Format values
        if isinstance(offline_val, (int, float)) and isinstance(online_val, (int, float)):
            diff = online_val - offline_val
            pct_diff = (diff / offline_val * 100) if offline_val > 0 else 0
            
            if key in ["makespan", "total_switch_time", "avg_waiting_time", "p99_waiting_time", 
                      "avg_response_time", "p99_response_time", "avg_request_response_time", 
                      "p99_request_response_time"]:
                # Time metrics - show with 2 decimal places
                print(f"{display_name:<25} {offline_val:>15.2f} {online_val:>15.2f} "
                      f"{diff:>+14.2f} ({pct_diff:+.1f}%)")
            else:
                # Count metrics - show as integers
                print(f"{display_name:<25} {int(offline_val):>15d} {int(online_val):>15d} "
                      f"{int(diff):>+14d} ({pct_diff:+.1f}%)")
        else:
            print(f"{display_name:<25} {str(offline_val):>15} {str(online_val):>15} {'N/A':>15}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Analyze results
    if isinstance(online_metrics.get("avg_waiting_time"), (int, float)) and \
       isinstance(offline_metrics.get("avg_waiting_time"), (int, float)):
        
        wait_improvement = (offline_metrics["avg_waiting_time"] - online_metrics["avg_waiting_time"]) / \
                         offline_metrics["avg_waiting_time"] * 100
        
        if wait_improvement > 0:
            print(f"✓ Online scheduler reduced average waiting time by {wait_improvement:.1f}%")
        else:
            print(f"✗ Online scheduler increased average waiting time by {-wait_improvement:.1f}%")
    
    if isinstance(online_metrics.get("p99_waiting_time"), (int, float)) and \
       isinstance(offline_metrics.get("p99_waiting_time"), (int, float)):
        
        p99_improvement = (offline_metrics["p99_waiting_time"] - online_metrics["p99_waiting_time"]) / \
                        offline_metrics["p99_waiting_time"] * 100
        
        if p99_improvement > 0:
            print(f"✓ Online scheduler reduced P99 waiting time by {p99_improvement:.1f}%")
        else:
            print(f"✗ Online scheduler increased P99 waiting time by {-p99_improvement:.1f}%")
    
    if isinstance(online_metrics.get("makespan"), (int, float)) and \
       isinstance(offline_metrics.get("makespan"), (int, float)):
        
        makespan_increase = (online_metrics["makespan"] - offline_metrics["makespan"]) / \
                          offline_metrics["makespan"] * 100
        
        print(f"• Online scheduler makespan is {makespan_increase:+.1f}% compared to offline")
    
    print("\nNote: Online scheduler optimizes for latency (waiting time), not throughput.")
    print("      Some increase in makespan is expected as a trade-off for better responsiveness.")

def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        requests_file = sys.argv[2] if len(sys.argv) > 2 else "simple_requests.yaml"
        gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    else:
        config_file = "simple_config.json"
        requests_file = "simple_requests.yaml"
        gpus = 4
    
    print(f"Configuration: {config_file}")
    print(f"Requests: {requests_file}")
    print(f"GPUs: {gpus}")
    
    # Run offline scheduler
    offline_metrics = run_scheduler("offline", config_file, requests_file, gpus)
    if not offline_metrics:
        print("Failed to run offline scheduler")
        return
    
    # Check offline results
    print("\nChecking offline scheduler results...")
    subprocess.run(["python", "check.py", "offline_execution_log.json"])
    
    # Run online scheduler
    online_metrics = run_scheduler("online", config_file, requests_file, gpus)
    if not online_metrics:
        print("Failed to run online scheduler")
        return
    
    # Check online results
    print("\nChecking online scheduler results...")
    subprocess.run(["python", "check.py", "online_execution_log.json"])
    
    # Compare results
    compare_metrics(offline_metrics, online_metrics)

if __name__ == "__main__":
    main()
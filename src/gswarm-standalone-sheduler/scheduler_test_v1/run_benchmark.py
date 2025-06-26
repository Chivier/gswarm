#!/usr/bin/env python3
"""
Comprehensive performance benchmark for schedulers
Tests with varying GPU counts and different workload configurations
"""

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmark.log")
    ]
)
logger = logging.getLogger(__name__)


class SchedulerBenchmark:
    """Run benchmarks for different schedulers and configurations"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def check_gpu_requirements(self, config_file: str) -> int:
        """Check minimum GPU requirements for a config"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        max_gpus = 1
        for model_id, model_info in config['models'].items():
            gpus_required = model_info.get('gpus_required', 1)
            max_gpus = max(max_gpus, gpus_required)
        
        return max_gpus
    
    def run_scheduler(self, scheduler_type: str, gpu_count: int, 
                     config_file: str, request_file: str) -> dict:
        """Run a single scheduler test and return metrics"""
        
        # Check if we have enough GPUs
        min_gpus = self.check_gpu_requirements(config_file)
        if gpu_count < min_gpus:
            logger.warning(f"Skipping {config_file} with {gpu_count} GPUs (requires {min_gpus})")
            return None
        
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
    
    def run_benchmarks(self, gpu_counts: list, configs: list):
        """Run all benchmark combinations"""
        total_runs = len(gpu_counts) * len(configs) * 2  # 2 schedulers
        current_run = 0
        
        for config_file, request_file in configs:
            for gpu_count in gpu_counts:
                for scheduler in ["baseline", "offline"]:
                    current_run += 1
                    logger.info(f"Progress: {current_run}/{total_runs}")
                    
                    metrics = self.run_scheduler(scheduler, gpu_count, config_file, request_file)
                    if metrics:
                        self.results.append(metrics)
                        
                        # Save intermediate results
                        self.save_results()
    
    def save_results(self):
        """Save results to JSON and CSV files"""
        result_file = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.output_dir / f"benchmark_results_{self.timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Results saved to {result_file} and {csv_file}")
        
        return pd.DataFrame(self.results) if self.results else None
    
    def plot_results(self):
        """Generate comprehensive performance comparison plots"""
        if not self.results:
            logger.error("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Scheduler Performance Comparison', fontsize=20, fontweight='bold')
        
        # Plot 1: Makespan vs GPU Count
        ax1 = fig.add_subplot(gs[0, :2])
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            
            for scheduler in ['baseline', 'offline']:
                sched_data = config_data[config_data['scheduler'] == scheduler]
                if not sched_data.empty:
                    marker = 'o' if scheduler == 'baseline' else 's'
                    linestyle = '-' if scheduler == 'baseline' else '--'
                    ax1.plot(sched_data['gpu_count'], sched_data['makespan'], 
                            marker=marker, linestyle=linestyle,
                            label=f'{scheduler} ({config})', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Number of GPUs', fontsize=12)
        ax1.set_ylabel('Makespan (seconds)', fontsize=12)
        ax1.set_title('Makespan vs GPU Count', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Throughput vs GPU Count
        ax2 = fig.add_subplot(gs[0, 2])
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            
            for scheduler in ['baseline', 'offline']:
                sched_data = config_data[config_data['scheduler'] == scheduler]
                if not sched_data.empty:
                    marker = 'o' if scheduler == 'baseline' else 's'
                    linestyle = '-' if scheduler == 'baseline' else '--'
                    ax2.plot(sched_data['gpu_count'], sched_data['throughput'], 
                            marker=marker, linestyle=linestyle,
                            label=f'{scheduler} ({config})', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Number of GPUs', fontsize=12)
        ax2.set_ylabel('Throughput (tasks/second)', fontsize=12)
        ax2.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Switches
        ax3 = fig.add_subplot(gs[1, 0])
        offline_data = df[df['scheduler'] == 'offline']
        if not offline_data.empty:
            for config in offline_data['config'].unique():
                config_data = offline_data[offline_data['config'] == config]
                ax3.plot(config_data['gpu_count'], config_data['model_switches'], 
                        'o-', label=config, linewidth=2, markersize=8)
        
        ax3.set_xlabel('Number of GPUs', fontsize=12)
        ax3.set_ylabel('Model Switches', fontsize=12)
        ax3.set_title('Model Switches (Offline Scheduler)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Speedup Ratio
        ax4 = fig.add_subplot(gs[1, 1])
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            
            baseline_data = config_data[config_data['scheduler'] == 'baseline'].sort_values('gpu_count')
            offline_data = config_data[config_data['scheduler'] == 'offline'].sort_values('gpu_count')
            
            # Find common GPU counts
            common_gpus = set(baseline_data['gpu_count']) & set(offline_data['gpu_count'])
            if common_gpus:
                speedup_data = []
                gpu_counts = []
                
                for gpu in sorted(common_gpus):
                    baseline_time = baseline_data[baseline_data['gpu_count'] == gpu]['makespan'].values[0]
                    offline_time = offline_data[offline_data['gpu_count'] == gpu]['makespan'].values[0]
                    speedup = baseline_time / offline_time
                    speedup_data.append(speedup)
                    gpu_counts.append(gpu)
                
                ax4.plot(gpu_counts, speedup_data, 'o-', label=config, linewidth=2, markersize=8)
        
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Equal Performance')
        ax4.set_xlabel('Number of GPUs', fontsize=12)
        ax4.set_ylabel('Speedup Ratio (Baseline/Offline)', fontsize=12)
        ax4.set_title('Offline Scheduler Speedup', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Scaling Efficiency
        ax5 = fig.add_subplot(gs[1, 2])
        for config in df['config'].unique():
            for scheduler in df['scheduler'].unique():
                data = df[(df['config'] == config) & (df['scheduler'] == scheduler)].sort_values('gpu_count')
                
                if len(data) > 1:
                    # Use smallest GPU count as baseline
                    base_idx = data['gpu_count'].idxmin()
                    base_throughput = data.loc[base_idx, 'throughput']
                    base_gpus = data.loc[base_idx, 'gpu_count']
                    
                    scaling_efficiency = []
                    gpu_counts = []
                    
                    for _, row in data.iterrows():
                        expected_throughput = base_throughput * (row['gpu_count'] / base_gpus)
                        efficiency = (row['throughput'] / expected_throughput) * 100
                        scaling_efficiency.append(efficiency)
                        gpu_counts.append(row['gpu_count'])
                    
                    marker = 'o' if scheduler == 'baseline' else 's'
                    linestyle = '-' if scheduler == 'baseline' else '--'
                    ax5.plot(gpu_counts, scaling_efficiency, 
                            marker=marker, linestyle=linestyle,
                            label=f'{scheduler} ({config})', linewidth=2, markersize=6)
        
        ax5.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect Scaling')
        ax5.set_xlabel('Number of GPUs', fontsize=12)
        ax5.set_ylabel('Scaling Efficiency (%)', fontsize=12)
        ax5.set_title('GPU Scaling Efficiency', fontsize=14, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Makespan Heatmap
        ax6 = fig.add_subplot(gs[2, :])
        pivot_data = df.pivot_table(
            values='makespan', 
            index='gpu_count', 
            columns=['config', 'scheduler']
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd_r', 
                   cbar_kws={'label': 'Makespan (seconds)'}, ax=ax6)
        ax6.set_title('Makespan Heatmap: GPU Count vs Configuration', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Configuration / Scheduler', fontsize=12)
        ax6.set_ylabel('GPU Count', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"performance_comparison_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_file}")
        
        # Also save as PDF for publication
        pdf_file = self.output_dir / f"performance_comparison_{self.timestamp}.pdf"
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        
        plt.show()
        
        # Generate summary statistics
        self.generate_summary_report(df)
    
    def generate_summary_report(self, df):
        """Generate a summary report of the benchmark results"""
        report_file = self.output_dir / f"benchmark_summary_{self.timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("SCHEDULER PERFORMANCE BENCHMARK SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            f.write("Overall Performance:\n")
            f.write("-" * 40 + "\n")
            
            for scheduler in df['scheduler'].unique():
                sched_data = df[df['scheduler'] == scheduler]
                f.write(f"\n{scheduler.upper()} Scheduler:\n")
                f.write(f"  Average Makespan: {sched_data['makespan'].mean():.2f}s\n")
                f.write(f"  Average Throughput: {sched_data['throughput'].mean():.3f} tasks/s\n")
                
                if scheduler == 'offline':
                    f.write(f"  Average Model Switches: {sched_data['model_switches'].mean():.1f}\n")
            
            # Per configuration statistics
            f.write("\n\nPer Configuration Performance:\n")
            f.write("-" * 40 + "\n")
            
            for config in df['config'].unique():
                f.write(f"\n{config}:\n")
                config_data = df[df['config'] == config]
                
                # Find best GPU count for each scheduler
                for scheduler in ['baseline', 'offline']:
                    sched_data = config_data[config_data['scheduler'] == scheduler]
                    if not sched_data.empty:
                        best_row = sched_data.loc[sched_data['throughput'].idxmax()]
                        f.write(f"  {scheduler}: Best at {best_row['gpu_count']} GPUs "
                               f"({best_row['throughput']:.3f} tasks/s)\n")
            
            # Speedup analysis
            f.write("\n\nSpeedup Analysis:\n")
            f.write("-" * 40 + "\n")
            
            for config in df['config'].unique():
                config_data = df[df['config'] == config]
                baseline_avg = config_data[config_data['scheduler'] == 'baseline']['makespan'].mean()
                offline_avg = config_data[config_data['scheduler'] == 'offline']['makespan'].mean()
                
                if baseline_avg > 0 and offline_avg > 0:
                    speedup = baseline_avg / offline_avg
                    f.write(f"{config}: Offline is {speedup:.2f}x faster on average\n")
        
        logger.info(f"Summary report saved to {report_file}")


def main():
    """Main benchmark execution"""
    # Create benchmark instance
    benchmark = SchedulerBenchmark()
    
    # Define GPU counts to test
    gpu_counts = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30]
    
    # Define configurations to test
    configs = [
        ("single_gpu_config.json", "single_gpu_requests.yaml"),  # Works with any GPU count
        ("simple_config.json", "simple_requests.yaml"),          # Requires 4+ GPUs
        ("complex_config.json", "complex_requests.yaml")         # Requires 4+ GPUs
    ]
    
    logger.info("=" * 60)
    logger.info("STARTING SCHEDULER PERFORMANCE BENCHMARKS")
    logger.info("=" * 60)
    logger.info(f"GPU counts to test: {gpu_counts}")
    logger.info(f"Configurations: {[c[0] for c in configs]}")
    logger.info(f"Output directory: {benchmark.output_dir}")
    logger.info("=" * 60)
    
    # Run benchmarks
    benchmark.run_benchmarks(gpu_counts, configs)
    
    # Generate plots and reports
    if benchmark.results:
        benchmark.plot_results()
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETED SUCCESSFULLY!")
        logger.info(f"Total successful runs: {len(benchmark.results)}")
        logger.info("=" * 60)
    else:
        logger.error("No successful benchmark runs!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Performance comparison script for baseline vs offline schedulers
Tests with varying GPU counts from 2 to 30
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
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SchedulerBenchmark:
    """Run benchmarks for different schedulers and configurations"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_scheduler(self, scheduler_type: str, gpu_count: int, 
                     config_file: str, request_file: str) -> dict:
        """Run a single scheduler test and return metrics"""
        logger.info(f"Running {scheduler_type} with {gpu_count} GPUs on {config_file}")
        
        # Determine script name
        script_name = "baseline.py" if scheduler_type == "baseline" else "offline_scheduler.py"
        
        # Run scheduler
        cmd = [
            "python", script_name,
            "--gpus", str(gpu_count),
            "--simulate", "false",  # Using estimate mode
            "--config", config_file,
            "--requests", request_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse execution log
            log_file = "baseline_execution_log.json" if scheduler_type == "baseline" else "offline_execution_log.json"
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            summary = log_data['summary']
            
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
                'throughput': summary['total_nodes_executed'] / summary['makespan'] if summary['makespan'] > 0 else 0
            }
            
            logger.info(f"  Makespan: {metrics['makespan']:.2f}s, Throughput: {metrics['throughput']:.3f} tasks/s")
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {scheduler_type}: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            return None
    
    def run_benchmarks(self, gpu_counts: list, configs: list):
        """Run all benchmark combinations"""
        for config_file, request_file in configs:
            for gpu_count in gpu_counts:
                for scheduler in ["baseline", "offline"]:
                    metrics = self.run_scheduler(scheduler, gpu_count, config_file, request_file)
                    if metrics:
                        self.results.append(metrics)
                        
                        # Save intermediate results
                        self.save_results()
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV for easy analysis
        df = pd.DataFrame(self.results)
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {result_file} and {csv_file}")
        
        return df
    
    def plot_results(self):
        """Generate performance comparison plots"""
        if not self.results:
            logger.error("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scheduler Performance Comparison', fontsize=16)
        
        # Plot 1: Makespan vs GPU Count
        ax1 = axes[0, 0]
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            
            baseline_data = config_data[config_data['scheduler'] == 'baseline']
            offline_data = config_data[config_data['scheduler'] == 'offline']
            
            ax1.plot(baseline_data['gpu_count'], baseline_data['makespan'], 
                    'o-', label=f'Baseline ({config})', linewidth=2, markersize=8)
            ax1.plot(offline_data['gpu_count'], offline_data['makespan'], 
                    's--', label=f'Offline ({config})', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Makespan (seconds)')
        ax1.set_title('Makespan vs GPU Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Throughput vs GPU Count
        ax2 = axes[0, 1]
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            
            baseline_data = config_data[config_data['scheduler'] == 'baseline']
            offline_data = config_data[config_data['scheduler'] == 'offline']
            
            ax2.plot(baseline_data['gpu_count'], baseline_data['throughput'], 
                    'o-', label=f'Baseline ({config})', linewidth=2, markersize=8)
            ax2.plot(offline_data['gpu_count'], offline_data['throughput'], 
                    's--', label=f'Offline ({config})', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Number of GPUs')
        ax2.set_ylabel('Throughput (tasks/second)')
        ax2.set_title('Throughput vs GPU Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Switches Comparison
        ax3 = axes[1, 0]
        offline_data = df[df['scheduler'] == 'offline']
        
        bar_width = 0.35
        x = np.arange(len(offline_data['gpu_count'].unique()))
        
        for i, config in enumerate(offline_data['config'].unique()):
            config_data = offline_data[offline_data['config'] == config]
            ax3.bar(x + i * bar_width, config_data['model_switches'], 
                   bar_width, label=f'{config}', alpha=0.8)
        
        ax3.set_xlabel('Number of GPUs')
        ax3.set_ylabel('Model Switches')
        ax3.set_title('Model Switches (Offline Scheduler)')
        ax3.set_xticks(x + bar_width / 2)
        ax3.set_xticklabels(offline_data['gpu_count'].unique())
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Speedup Ratio
        ax4 = axes[1, 1]
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            
            baseline_data = config_data[config_data['scheduler'] == 'baseline'].sort_values('gpu_count')
            offline_data = config_data[config_data['scheduler'] == 'offline'].sort_values('gpu_count')
            
            # Calculate speedup ratio (baseline/offline)
            if len(baseline_data) == len(offline_data):
                speedup = baseline_data['makespan'].values / offline_data['makespan'].values
                ax4.plot(baseline_data['gpu_count'], speedup, 
                        'o-', label=f'{config}', linewidth=2, markersize=8)
        
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Equal Performance')
        ax4.set_xlabel('Number of GPUs')
        ax4.set_ylabel('Speedup Ratio (Baseline/Offline)')
        ax4.set_title('Offline Scheduler Speedup over Baseline')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"performance_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_file}")
        
        # Also save individual plots
        self.save_individual_plots(df, timestamp)
        
        plt.show()
    
    def save_individual_plots(self, df, timestamp):
        """Save individual plots for detailed analysis"""
        # GPU Utilization Heatmap
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values='makespan', 
            index='gpu_count', 
            columns=['config', 'scheduler']
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd_r', 
                   cbar_kws={'label': 'Makespan (seconds)'})
        plt.title('Makespan Heatmap: GPU Count vs Configuration')
        plt.tight_layout()
        
        heatmap_file = self.output_dir / f"makespan_heatmap_{timestamp}.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scaling Efficiency Plot
        plt.figure(figsize=(10, 6))
        
        for config in df['config'].unique():
            for scheduler in df['scheduler'].unique():
                data = df[(df['config'] == config) & (df['scheduler'] == scheduler)].sort_values('gpu_count')
                
                if len(data) > 0:
                    # Calculate scaling efficiency
                    base_throughput = data.iloc[0]['throughput']
                    base_gpus = data.iloc[0]['gpu_count']
                    
                    scaling_efficiency = []
                    for _, row in data.iterrows():
                        expected_throughput = base_throughput * (row['gpu_count'] / base_gpus)
                        efficiency = (row['throughput'] / expected_throughput) * 100
                        scaling_efficiency.append(efficiency)
                    
                    plt.plot(data['gpu_count'], scaling_efficiency, 
                            'o-', label=f'{scheduler} ({config})', linewidth=2, markersize=6)
        
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect Scaling')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Scaling Efficiency (%)')
        plt.title('GPU Scaling Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scaling_file = self.output_dir / f"scaling_efficiency_{timestamp}.png"
        plt.savefig(scaling_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Additional plots saved to {heatmap_file} and {scaling_file}")


def main():
    """Main benchmark execution"""
    # Create benchmark instance
    benchmark = SchedulerBenchmark()
    
    # Define GPU counts to test
    gpu_counts = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30]
    
    # Define configurations to test
    configs = [
        ("simple_config.json", "simple_requests.yaml"),
        ("complex_config.json", "complex_requests.yaml")
    ]
    
    logger.info("Starting scheduler performance benchmarks")
    logger.info(f"GPU counts: {gpu_counts}")
    logger.info(f"Configurations: {[c[0] for c in configs]}")
    
    # Run benchmarks
    benchmark.run_benchmarks(gpu_counts, configs)
    
    # Generate plots
    benchmark.plot_results()
    
    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main()
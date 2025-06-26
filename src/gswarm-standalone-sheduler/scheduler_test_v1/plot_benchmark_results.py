#!/usr/bin/env python3
"""
Generate plots from existing benchmark results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_results(csv_file):
    """Generate comprehensive performance comparison plots from CSV data"""
    
    # Load data
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} benchmark results")
    
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
    
    # Plot 6: Summary Statistics
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create summary text
    summary_text = "Performance Summary:\n\n"
    
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        summary_text += f"{config.upper()}:\n"
        
        for scheduler in ['baseline', 'offline']:
            sched_data = config_data[config_data['scheduler'] == scheduler]
            if not sched_data.empty:
                avg_makespan = sched_data['makespan'].mean()
                avg_throughput = sched_data['throughput'].mean()
                min_makespan = sched_data['makespan'].min()
                best_gpu = sched_data.loc[sched_data['makespan'].idxmin(), 'gpu_count']
                
                summary_text += f"  {scheduler}: Avg makespan={avg_makespan:.1f}s, "
                summary_text += f"Avg throughput={avg_throughput:.3f} tasks/s\n"
                summary_text += f"         Best performance at {best_gpu} GPUs ({min_makespan:.1f}s)\n"
        
        summary_text += "\n"
    
    # Add speedup analysis
    summary_text += "Speedup Analysis:\n"
    for config in df['config'].unique():
        config_data = df[df['config'] == config]
        baseline_avg = config_data[config_data['scheduler'] == 'baseline']['makespan'].mean()
        offline_avg = config_data[config_data['scheduler'] == 'offline']['makespan'].mean()
        
        if baseline_avg > 0 and offline_avg > 0:
            speedup = baseline_avg / offline_avg
            summary_text += f"  {config}: Offline is {speedup:.2f}x faster on average\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("benchmark_results")
    plot_file = output_dir / "performance_comparison_partial.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {plot_file}")
    
    # Also save as PDF
    pdf_file = output_dir / "performance_comparison_partial.pdf"
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE STATISTICS")
    print("="*60)
    
    for config in df['config'].unique():
        print(f"\n{config.upper()} Configuration:")
        print("-"*40)
        
        config_data = df[df['config'] == config]
        
        # Compare baseline vs offline
        baseline = config_data[config_data['scheduler'] == 'baseline']
        offline = config_data[config_data['scheduler'] == 'offline']
        
        if not baseline.empty and not offline.empty:
            print(f"GPU Count\tBaseline(s)\tOffline(s)\tSpeedup\tModel Switches")
            print("-"*60)
            
            for gpu in sorted(config_data['gpu_count'].unique()):
                b_data = baseline[baseline['gpu_count'] == gpu]
                o_data = offline[offline['gpu_count'] == gpu]
                
                if not b_data.empty and not o_data.empty:
                    b_time = b_data['makespan'].values[0]
                    o_time = o_data['makespan'].values[0]
                    speedup = b_time / o_time
                    switches = o_data['model_switches'].values[0]
                    
                    print(f"{gpu:9d}\t{b_time:11.1f}\t{o_time:10.1f}\t{speedup:7.2f}\t{switches:14d}")


if __name__ == "__main__":
    csv_file = "benchmark_results/benchmark_results_20250626_083257.csv"
    plot_results(csv_file)
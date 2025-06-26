#!/usr/bin/env python3
"""
Generate separate plots for simple and complex configurations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Load data
csv_file = "benchmark_results/benchmark_results_20250626_083257.csv"
df = pd.read_csv(csv_file)

# Filter out single_gpu_config
df = df[df['config'] != 'single_gpu_config']

# Set up the plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create separate figures for each configuration
for config_name in ['simple_config', 'complex_config']:
    config_data = df[df['config'] == config_name]
    
    if config_data.empty:
        continue
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{config_name.upper()} - Scheduler Performance Comparison', fontsize=20, fontweight='bold')
    
    # Subplot 1: Makespan vs GPU Count
    ax1 = plt.subplot(2, 3, 1)
    for scheduler in ['baseline', 'offline']:
        sched_data = config_data[config_data['scheduler'] == scheduler]
        if not sched_data.empty:
            marker = 'o' if scheduler == 'baseline' else 's'
            linestyle = '-' if scheduler == 'baseline' else '--'
            ax1.plot(sched_data['gpu_count'], sched_data['makespan'], 
                    marker=marker, linestyle=linestyle,
                    label=scheduler, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of GPUs', fontsize=12)
    ax1.set_ylabel('Makespan (seconds)', fontsize=12)
    ax1.set_title('Makespan vs GPU Count', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Subplot 2: Throughput vs GPU Count
    ax2 = plt.subplot(2, 3, 2)
    for scheduler in ['baseline', 'offline']:
        sched_data = config_data[config_data['scheduler'] == scheduler]
        if not sched_data.empty:
            marker = 'o' if scheduler == 'baseline' else 's'
            linestyle = '-' if scheduler == 'baseline' else '--'
            ax2.plot(sched_data['gpu_count'], sched_data['throughput'], 
                    marker=marker, linestyle=linestyle,
                    label=scheduler, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of GPUs', fontsize=12)
    ax2.set_ylabel('Throughput (tasks/second)', fontsize=12)
    ax2.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Speedup Ratio
    ax3 = plt.subplot(2, 3, 3)
    baseline_data = config_data[config_data['scheduler'] == 'baseline'].sort_values('gpu_count')
    offline_data = config_data[config_data['scheduler'] == 'offline'].sort_values('gpu_count')
    
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
        
        ax3.plot(gpu_counts, speedup_data, 'go-', linewidth=2, markersize=8)
        ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Equal Performance')
    
    ax3.set_xlabel('Number of GPUs', fontsize=12)
    ax3.set_ylabel('Speedup Ratio (Baseline/Offline)', fontsize=12)
    ax3.set_title('Offline Scheduler Speedup', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Model Switches (Offline only)
    ax4 = plt.subplot(2, 3, 4)
    offline_only = config_data[config_data['scheduler'] == 'offline']
    if not offline_only.empty:
        ax4.plot(offline_only['gpu_count'], offline_only['model_switches'], 
                'bo-', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Number of GPUs', fontsize=12)
    ax4.set_ylabel('Model Switches', fontsize=12)
    ax4.set_title('Model Switches (Offline Scheduler)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Scaling Efficiency
    ax5 = plt.subplot(2, 3, 5)
    for scheduler in ['baseline', 'offline']:
        data = config_data[config_data['scheduler'] == scheduler].sort_values('gpu_count')
        
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
            ax5.plot(gpu_counts, scaling_efficiency, 
                    marker=marker, label=scheduler, linewidth=2, markersize=8)
    
    ax5.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect Scaling')
    ax5.set_xlabel('Number of GPUs', fontsize=12)
    ax5.set_ylabel('Scaling Efficiency (%)', fontsize=12)
    ax5.set_title('GPU Scaling Efficiency', fontsize=14, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Performance Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary statistics
    summary_text = f"Performance Summary for {config_name}:\n\n"
    
    baseline_stats = config_data[config_data['scheduler'] == 'baseline']
    offline_stats = config_data[config_data['scheduler'] == 'offline']
    
    if not baseline_stats.empty and not offline_stats.empty:
        # Average performance
        baseline_avg_makespan = baseline_stats['makespan'].mean()
        offline_avg_makespan = offline_stats['makespan'].mean()
        avg_speedup = baseline_avg_makespan / offline_avg_makespan
        
        summary_text += f"Average Speedup: {avg_speedup:.2f}x\n\n"
        
        # Best performance
        baseline_best = baseline_stats.loc[baseline_stats['throughput'].idxmax()]
        offline_best = offline_stats.loc[offline_stats['throughput'].idxmax()]
        
        summary_text += f"Best Throughput:\n"
        summary_text += f"  Baseline: {baseline_best['throughput']:.3f} tasks/s @ {baseline_best['gpu_count']} GPUs\n"
        summary_text += f"  Offline:  {offline_best['throughput']:.3f} tasks/s @ {offline_best['gpu_count']} GPUs\n\n"
        
        # Model switches
        avg_switches = offline_stats['model_switches'].mean()
        summary_text += f"Average Model Switches: {avg_switches:.1f}\n"
        
        # Switch overhead
        avg_switch_time = offline_stats['switch_time'].mean()
        avg_makespan = offline_stats['makespan'].mean()
        switch_overhead = (avg_switch_time / avg_makespan) * 100
        summary_text += f"Switch Time Overhead: {switch_overhead:.1f}%\n\n"
        
        # GPU utilization variance
        avg_gpu_util_std = offline_stats['gpu_utilization_std'].mean()
        summary_text += f"Avg GPU Utilization Std Dev: {avg_gpu_util_std:.1f}"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("benchmark_results")
    plot_file = output_dir / f"{config_name}_performance_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    pdf_file = output_dir / f"{config_name}_performance_comparison.pdf"
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    
    print(f"Saved plots for {config_name} to {plot_file}")

# Create a combined summary figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Scheduler Performance Comparison Summary', fontsize=16, fontweight='bold')

# Plot 1: Makespan comparison for both configs
ax1.set_title('Makespan Comparison')
for config in ['simple_config', 'complex_config']:
    config_data = df[df['config'] == config]
    for scheduler in ['baseline', 'offline']:
        sched_data = config_data[config_data['scheduler'] == scheduler]
        if not sched_data.empty:
            label = f"{config.split('_')[0]} - {scheduler}"
            ax1.plot(sched_data['gpu_count'], sched_data['makespan'], 
                    marker='o', label=label, linewidth=1.5)

ax1.set_xlabel('Number of GPUs')
ax1.set_ylabel('Makespan (seconds)')
ax1.set_yscale('log')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup comparison
ax2.set_title('Speedup Ratio (Baseline/Offline)')
for config in ['simple_config', 'complex_config']:
    config_data = df[df['config'] == config]
    baseline_data = config_data[config_data['scheduler'] == 'baseline'].sort_values('gpu_count')
    offline_data = config_data[config_data['scheduler'] == 'offline'].sort_values('gpu_count')
    
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
        
        ax2.plot(gpu_counts, speedup_data, marker='o', label=config.split('_')[0], linewidth=2)

ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of GPUs')
ax2.set_ylabel('Speedup Ratio')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Throughput comparison
ax3.set_title('Throughput Scaling')
for config in ['simple_config', 'complex_config']:
    config_data = df[df['config'] == config]
    offline_data = config_data[config_data['scheduler'] == 'offline']
    if not offline_data.empty:
        ax3.plot(offline_data['gpu_count'], offline_data['throughput'], 
                marker='o', label=f"{config.split('_')[0]} - offline", linewidth=2)

ax3.set_xlabel('Number of GPUs')
ax3.set_ylabel('Throughput (tasks/second)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Summary statistics
ax4.axis('off')

summary_stats = []
for config in ['simple_config', 'complex_config']:
    config_data = df[df['config'] == config]
    baseline_avg = config_data[config_data['scheduler'] == 'baseline']['makespan'].mean()
    offline_avg = config_data[config_data['scheduler'] == 'offline']['makespan'].mean()
    speedup = baseline_avg / offline_avg
    
    offline_switches = config_data[config_data['scheduler'] == 'offline']['model_switches'].mean()
    
    summary_stats.append({
        'Config': config.split('_')[0].capitalize(),
        'Avg Speedup': f"{speedup:.2f}x",
        'Model Switches': f"{offline_switches:.0f}"
    })

summary_df = pd.DataFrame(summary_stats)
table = ax4.table(cellText=summary_df.values,
                  colLabels=summary_df.columns,
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.3, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

ax4.text(0.5, 0.8, 'Summary Statistics', transform=ax4.transAxes,
         ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_results/performance_summary_clean.png', dpi=300, bbox_inches='tight')
plt.savefig('benchmark_results/performance_summary_clean.pdf', format='pdf', bbox_inches='tight')

print("\nAll plots generated successfully!")
print("- simple_config_performance_comparison.png/pdf")
print("- complex_config_performance_comparison.png/pdf")
print("- performance_summary_clean.png/pdf")
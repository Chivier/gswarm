import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from datetime import datetime
import seaborn as sns
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class GPUState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    COMPUTING = "computing"

@dataclass
class GPUAllocation:
    gpu_ids: List[int]
    model_name: str
    start_time: float
    end_time: float

class GPU:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.current_model = None
        self.state = GPUState.IDLE
        self.busy_until = 0
        self.allocated_to = None  # Track which node/allocation this GPU belongs to
        
    def is_available(self, current_time: float) -> bool:
        return current_time >= self.busy_until and self.state == GPUState.IDLE

class WorkflowNode:
    def __init__(self, node_id: str, model: str, depends_on: List[str], execution_time: float):
        self.node_id = node_id
        self.model = model
        self.depends_on = depends_on
        self.execution_time = execution_time  # Pre-determined execution time
        self.completed = False
        self.start_time = None
        self.end_time = None
        self.allocated_gpus = []

class WorkflowInstance:
    def __init__(self, request_id: str, workflow_id: str, nodes: List[WorkflowNode], arrival_time: float):
        self.request_id = request_id
        self.workflow_id = workflow_id
        self.nodes = {node.node_id: node for node in nodes}
        self.arrival_time = arrival_time
        self.completion_time = None
        
    def is_node_ready(self, node_id: str) -> bool:
        """Check if a node's dependencies are satisfied"""
        node = self.nodes[node_id]
        return all(self.nodes[dep].completed for dep in node.depends_on)
    
    def get_ready_nodes(self) -> List[str]:
        """Get all nodes that are ready to execute"""
        return [nid for nid, node in self.nodes.items() 
                if not node.completed and self.is_node_ready(nid)]
    
    def is_completed(self) -> bool:
        """Check if all nodes in workflow are completed"""
        return all(node.completed for node in self.nodes.values())

class BaselineScheduler:
    def __init__(self, n_gpus: int, models_config: dict):
        self.gpus = [GPU(i) for i in range(n_gpus)]
        self.models_config = models_config
        self.total_gpus = n_gpus
        
    def allocate_gpus(self, model_name: str, current_time: float) -> Tuple[List[int], float]:
        """Allocate required GPUs for a model, waiting if necessary"""
        gpus_required = self.models_config[model_name]["gpus_required"]
        
        if gpus_required > self.total_gpus:
            raise ValueError(f"Model {model_name} requires {gpus_required} GPUs but only {self.total_gpus} available")
        
        # Find earliest time when enough GPUs are available
        while True:
            available_gpus = [g for g in self.gpus if g.is_available(current_time)]
            if len(available_gpus) >= gpus_required:
                # Allocate GPUs
                allocated = available_gpus[:gpus_required]
                allocated_ids = [g.gpu_id for g in allocated]
                return allocated_ids, current_time
            else:
                # Wait for next GPU to become available
                next_available_time = min(g.busy_until for g in self.gpus if not g.is_available(current_time))
                current_time = next_available_time
        
    def execute(self, workflows: List[WorkflowInstance]) -> Dict:
        current_time = 0
        total_load_time = 0
        total_inference_time = 0
        total_model_switches = 0
        pending_workflows = list(workflows)
        active_workflows = []
        completed_workflows = []
        
        while pending_workflows or active_workflows:
            # Add workflows that have arrived
            while pending_workflows and pending_workflows[0].arrival_time <= current_time:
                active_workflows.append(pending_workflows.pop(0))
            
            # Process each active workflow
            made_progress = False
            workflows_to_complete = []
            
            for workflow in active_workflows:
                ready_nodes = workflow.get_ready_nodes()
                if ready_nodes:
                    node_id = ready_nodes[0]
                    node = workflow.nodes[node_id]
                    
                    # Try to allocate GPUs
                    gpus_required = self.models_config[node.model]["gpus_required"]
                    available_gpus = [g for g in self.gpus if g.is_available(current_time)]
                    
                    if len(available_gpus) >= gpus_required:
                        # Can execute this node
                        allocated_gpus = available_gpus[:gpus_required]
                        allocated_gpu_ids = [g.gpu_id for g in allocated_gpus]
                        
                        # Load model (always reload in baseline)
                        load_time = self.models_config[node.model]["load_time_seconds"]
                        for gpu_id in allocated_gpu_ids:
                            gpu = self.gpus[gpu_id]
                            total_model_switches += 1  # Always count as switch in baseline
                            gpu.current_model = node.model
                            gpu.state = GPUState.LOADING
                            gpu.busy_until = current_time + load_time + node.execution_time
                        
                        total_load_time += load_time
                        
                        # Execute node
                        node.start_time = current_time
                        node.end_time = current_time + load_time + node.execution_time
                        node.completed = True
                        node.allocated_gpus = allocated_gpu_ids
                        
                        total_inference_time += node.execution_time
                        made_progress = True
                        
                        # Check if workflow is complete
                        if workflow.is_completed():
                            workflow.completion_time = max(n.end_time for n in workflow.nodes.values())
                            workflows_to_complete.append(workflow)
            
            # Remove completed workflows
            for workflow in workflows_to_complete:
                active_workflows.remove(workflow)
                completed_workflows.append(workflow)
            
            # Advance time if no progress was made
            if not made_progress:
                next_times = []
                # Next GPU available
                for gpu in self.gpus:
                    if gpu.busy_until > current_time:
                        next_times.append(gpu.busy_until)
                # Next workflow arrival
                if pending_workflows:
                    next_times.append(pending_workflows[0].arrival_time)
                
                if next_times:
                    current_time = min(next_times)
                else:
                    # No more events
                    break
        
        # Calculate final completion time
        final_time = max(w.completion_time for w in completed_workflows) if completed_workflows else 0
        
        return {
            "total_time": final_time,
            "total_load_time": total_load_time,
            "total_inference_time": total_inference_time,
            "total_model_switches": total_model_switches,
            "workflows": completed_workflows
        }

class LazyScheduler:
    def __init__(self, n_gpus: int, models_config: dict):
        self.gpus = [GPU(i) for i in range(n_gpus)]
        self.models_config = models_config
        self.total_gpus = n_gpus
        self.model_gpu_mapping = {}  # Track which GPUs have which models
        
    def find_gpus_with_model(self, model_name: str, required_count: int, current_time: float) -> List[int]:
        """Find GPUs that already have the model loaded and are available"""
        available_with_model = []
        for gpu in self.gpus:
            if (gpu.current_model == model_name and 
                gpu.is_available(current_time)):
                available_with_model.append(gpu.gpu_id)
                if len(available_with_model) >= required_count:
                    return available_with_model[:required_count]
        return available_with_model
    
    def allocate_gpus_lazy(self, model_name: str, current_time: float) -> Tuple[List[int], float, bool]:
        """Allocate GPUs with lazy loading strategy"""
        gpus_required = self.models_config[model_name]["gpus_required"]
        
        # First, try to find GPUs that already have the model
        gpus_with_model = self.find_gpus_with_model(model_name, gpus_required, current_time)
        
        if len(gpus_with_model) >= gpus_required:
            # No loading needed
            return gpus_with_model, current_time, False
        
        # Need to wait or load on new GPUs
        while True:
            available_gpus = [g for g in self.gpus if g.is_available(current_time)]
            if len(available_gpus) >= gpus_required:
                # Use GPUs with model + additional ones
                allocated_ids = list(gpus_with_model)
                for gpu in available_gpus:
                    if gpu.gpu_id not in allocated_ids:
                        allocated_ids.append(gpu.gpu_id)
                        if len(allocated_ids) >= gpus_required:
                            break
                return allocated_ids[:gpus_required], current_time, True
            else:
                # Wait for GPUs
                next_available_time = min(g.busy_until for g in self.gpus if not g.is_available(current_time))
                current_time = next_available_time
    
    def execute(self, workflows: List[WorkflowInstance]) -> Dict:
        current_time = 0
        total_load_time = 0
        total_inference_time = 0
        total_model_switches = 0
        pending_workflows = list(workflows)
        active_workflows = []
        
        while pending_workflows or active_workflows:
            # Add workflows that have arrived
            while pending_workflows and pending_workflows[0].arrival_time <= current_time:
                active_workflows.append(pending_workflows.pop(0))
            
            # Find all ready nodes across all active workflows
            ready_tasks = []
            for workflow in active_workflows:
                for node_id in workflow.get_ready_nodes():
                    ready_tasks.append((workflow, node_id))
            
            if not ready_tasks:
                if pending_workflows:
                    # Jump to next arrival
                    current_time = pending_workflows[0].arrival_time
                    continue
                else:
                    # No more work
                    break
            
            # Group tasks by model
            tasks_by_model = defaultdict(list)
            for workflow, node_id in ready_tasks:
                model = workflow.nodes[node_id].model
                tasks_by_model[model].append((workflow, node_id))
            
            # Try to schedule tasks
            scheduled_something = False
            
            # Prioritize models that are already loaded
            for model_name in tasks_by_model:
                if not tasks_by_model[model_name]:
                    continue
                    
                workflow, node_id = tasks_by_model[model_name][0]
                node = workflow.nodes[node_id]
                
                # Try to allocate GPUs
                allocated_gpu_ids, start_time, needs_loading = self.allocate_gpus_lazy(
                    model_name, current_time
                )
                
                if start_time == current_time:
                    # Can schedule now
                    tasks_by_model[model_name].pop(0)
                    
                    # Handle loading if needed
                    if needs_loading:
                        load_time = self.models_config[model_name]["load_time_seconds"]
                        for gpu_id in allocated_gpu_ids:
                            gpu = self.gpus[gpu_id]
                            if gpu.current_model != model_name:
                                total_model_switches += 1
                                gpu.current_model = model_name
                            gpu.state = GPUState.LOADING
                            gpu.busy_until = current_time + load_time
                        total_load_time += load_time
                        current_time += load_time
                    
                    # Execute computation
                    node.start_time = current_time
                    compute_time = node.execution_time
                    total_inference_time += compute_time
                    
                    for gpu_id in allocated_gpu_ids:
                        gpu = self.gpus[gpu_id]
                        gpu.state = GPUState.COMPUTING
                        gpu.busy_until = current_time + compute_time
                    
                    node.end_time = current_time + compute_time
                    node.completed = True
                    node.allocated_gpus = allocated_gpu_ids
                    scheduled_something = True
                    
                    # Don't mark as idle yet - keep model loaded
                    for gpu_id in allocated_gpu_ids:
                        self.gpus[gpu_id].state = GPUState.IDLE
                    
                    break
            
            # Move completed workflows
            completed = [w for w in active_workflows if w.is_completed()]
            for workflow in completed:
                workflow.completion_time = max(node.end_time for node in workflow.nodes.values())
                active_workflows.remove(workflow)
            
            # Advance time if nothing scheduled
            if not scheduled_something:
                next_times = []
                for gpu in self.gpus:
                    if gpu.busy_until > current_time:
                        next_times.append(gpu.busy_until)
                if pending_workflows:
                    next_times.append(pending_workflows[0].arrival_time)
                
                if next_times:
                    current_time = min(next_times)
        
        # Set completion time for last workflows
        for workflow in workflows:
            if workflow.completion_time is None:
                workflow.completion_time = max(node.end_time for node in workflow.nodes.values() if node.end_time)
        
        return {
            "total_time": max(w.completion_time for w in workflows if w.completion_time),
            "total_load_time": total_load_time,
            "total_inference_time": total_inference_time,
            "total_model_switches": total_model_switches,
            "workflows": workflows
        }

def load_config_and_requests():
    """Load system configuration and workflow requests"""
    with open("system_config.json", "r") as f:
        config = json.load(f)
    
    with open("workflow_requests.json", "r") as f:
        requests_data = json.load(f)
    
    return config, requests_data["requests"]

def create_workflow_instances(requests: List[Dict], workflows_config: Dict, start_time: datetime) -> List[WorkflowInstance]:
    """Create workflow instances from requests"""
    instances = []
    
    for req in requests:
        workflow_def = workflows_config[req["workflow_id"]]
        
        # Create nodes
        nodes = []
        edges = {(e["from"], e["to"]) for e in workflow_def["edges"]}
        
        # Build dependency map
        dependencies = defaultdict(list)
        for from_node, to_node in edges:
            dependencies[to_node].append(from_node)
        
        # Create node instances
        for node_def in workflow_def["nodes"]:
            node = WorkflowNode(
                node_id=node_def["id"],
                model=node_def["model"],
                depends_on=dependencies.get(node_def["id"], []),
                execution_time=req["node_execution_times"][node_def["id"]]
            )
            nodes.append(node)
        
        # Calculate arrival time in seconds from start
        arrival_time = (datetime.fromisoformat(req["timestamp"]) - start_time).total_seconds()
        
        instance = WorkflowInstance(
            request_id=req["request_id"],
            workflow_id=req["workflow_id"],
            nodes=nodes,
            arrival_time=arrival_time
        )
        instances.append(instance)
    
    return instances

def calculate_metrics(results: dict) -> dict:
    workflows = results["workflows"]
    latencies = [w.completion_time - w.arrival_time for w in workflows]
    total_time = results["total_time"]
    throughput = len(workflows) / total_time if total_time > 0 else 0
    total_compute_time = results["total_inference_time"]
    total_load_time = results["total_load_time"]
    # 关键补丁：空latencies返回nan，避免绘图时崩溃
    if not latencies:
        return {
            "avg_latency": float("nan"),
            "p50_latency": float("nan"),
            "p99_latency": float("nan"),
            "throughput": 0,
            "load_overhead": float("nan"),
            "model_switches": results["total_model_switches"]
        }
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.percentile(latencies, 50),
        "p99_latency": np.percentile(latencies, 99),
        "throughput": throughput,
        "load_overhead": total_load_time / (total_compute_time + total_load_time) if (total_compute_time + total_load_time) > 0 else 0,
        "model_switches": results["total_model_switches"]
    }

def run_experiments(gpu_counts=[4, 6, 8, 10, 12]):
    config, requests = load_config_and_requests()
    models_config = config["models"]
    workflows_config = config["workflows"]
    start_time = datetime.fromisoformat(requests[0]["timestamp"])
    results = {
        "baseline": {},
        "lazy": {},
        "metrics": {
            "baseline": {},
            "lazy": {}
        }
    }
    max_gpus_required = max(m["gpus_required"] for m in models_config.values())
    for n_gpus in gpu_counts:
        print(f"\nRunning experiment with {n_gpus} A5000 GPUs...")
        if n_gpus < max_gpus_required:
            print(f"  Skipping: Need at least {max_gpus_required} GPUs for 30B model")
            continue
        workflows_baseline = create_workflow_instances(requests, workflows_config, start_time)
        workflows_lazy = create_workflow_instances(requests, workflows_config, start_time)
        baseline_scheduler = BaselineScheduler(n_gpus, models_config)
        baseline_results = baseline_scheduler.execute(workflows_baseline)
        # 如果没有workflows，则跳过，防止空数组的问题
        if not baseline_results["workflows"]:
            print(f"  Skipping: No workflows completed for {n_gpus} GPUs")
            continue
        results["baseline"][n_gpus] = baseline_results
        results["metrics"]["baseline"][n_gpus] = calculate_metrics(baseline_results)
        lazy_scheduler = LazyScheduler(n_gpus, models_config)
        lazy_results = lazy_scheduler.execute(workflows_lazy)
        if not lazy_results["workflows"]:
            print(f"  Skipping: No workflows completed for {n_gpus} GPUs (lazy)")
            continue
        results["lazy"][n_gpus] = lazy_results
        results["metrics"]["lazy"][n_gpus] = calculate_metrics(lazy_results)
        print(f"  Baseline - Total time: {baseline_results['total_time']:.2f}s, "
              f"Load time: {baseline_results['total_load_time']:.2f}s, "
              f"Model switches: {baseline_results['total_model_switches']}")
        print(f"  Lazy - Total time: {lazy_results['total_time']:.2f}s, "
              f"Load time: {lazy_results['total_load_time']:.2f}s, "
              f"Model switches: {lazy_results['total_model_switches']}")
    return results

def plot_results(results: Dict):
    """Create comprehensive visualization of experimental results"""
    gpu_counts = sorted(list(results["baseline"].keys()))
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'baseline': '#FF6B6B', 'lazy': '#4ECDC4'}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Total Execution Time
    ax1 = fig.add_subplot(gs[0, 0])
    baseline_times = [results["baseline"][n]["total_time"] for n in gpu_counts]
    lazy_times = [results["lazy"][n]["total_time"] for n in gpu_counts]
    
    x = np.arange(len(gpu_counts))
    width = 0.35
    ax1.bar(x - width/2, baseline_times, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    ax1.bar(x + width/2, lazy_times, width, label='Lazy Mode', color=colors['lazy'], alpha=0.8)
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Total Execution Time (s)')
    ax1.set_title('Total Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gpu_counts)
    ax1.legend()
    
    # 2. Speedup
    ax2 = fig.add_subplot(gs[0, 1])
    speedups = [baseline_times[i] / lazy_times[i] for i in range(len(gpu_counts))]
    bars = ax2.bar(gpu_counts, speedups, color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of GPUs')
    ax2.set_ylabel('Speedup (Baseline / Lazy)')
    ax2.set_title('Lazy Mode Speedup')
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom')
    
    # 3. Model Load Time
    ax3 = fig.add_subplot(gs[0, 2])
    baseline_loads = [results["baseline"][n]["total_load_time"] for n in gpu_counts]
    lazy_loads = [results["lazy"][n]["total_load_time"] for n in gpu_counts]
    
    ax3.bar(x - width/2, baseline_loads, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    ax3.bar(x + width/2, lazy_loads, width, label='Lazy Mode', color=colors['lazy'], alpha=0.8)
    ax3.set_xlabel('Number of GPUs')
    ax3.set_ylabel('Total Model Load Time (s)')
    ax3.set_title('Model Load Time Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(gpu_counts)
    ax3.legend()
    
    # 4. Load Time Reduction
    ax4 = fig.add_subplot(gs[1, 0])
    load_reductions = [(baseline_loads[i] - lazy_loads[i]) / baseline_loads[i] * 100 
                      for i in range(len(gpu_counts))]
    bars = ax4.bar(gpu_counts, load_reductions, color='#FFA502', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Number of GPUs')
    ax4.set_ylabel('Load Time Reduction (%)')
    ax4.set_title('Model Load Time Reduction in Lazy Mode')
    
    # Add value labels
    for bar, reduction in zip(bars, load_reductions):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%', ha='center', va='bottom')
    
    # 5. Average Latency
    ax5 = fig.add_subplot(gs[1, 1])
    baseline_latencies = [results["metrics"]["baseline"][n]["avg_latency"] for n in gpu_counts]
    lazy_latencies = [results["metrics"]["lazy"][n]["avg_latency"] for n in gpu_counts]
    
    ax5.bar(x - width/2, baseline_latencies, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    ax5.bar(x + width/2, lazy_latencies, width, label='Lazy Mode', color=colors['lazy'], alpha=0.8)
    ax5.set_xlabel('Number of GPUs')
    ax5.set_ylabel('Average Workflow Latency (s)')
    ax5.set_title('Average Workflow Latency')
    ax5.set_xticks(x)
    ax5.set_xticklabels(gpu_counts)
    ax5.legend()
    
    # 6. P99 Latency
    ax6 = fig.add_subplot(gs[1, 2])
    baseline_p99 = [results["metrics"]["baseline"][n]["p99_latency"] for n in gpu_counts]
    lazy_p99 = [results["metrics"]["lazy"][n]["p99_latency"] for n in gpu_counts]
    
    ax6.bar(x - width/2, baseline_p99, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    ax6.bar(x + width/2, lazy_p99, width, label='Lazy Mode', color=colors['lazy'], alpha=0.8)
    ax6.set_xlabel('Number of GPUs')
    ax6.set_ylabel('P99 Latency (s)')
    ax6.set_title('P99 Workflow Latency')
    ax6.set_xticks(x)
    ax6.set_xticklabels(gpu_counts)
    ax6.legend()
    
    # 7. Throughput
    ax7 = fig.add_subplot(gs[2, 0])
    baseline_throughput = [results["metrics"]["baseline"][n]["throughput"] for n in gpu_counts]
    lazy_throughput = [results["metrics"]["lazy"][n]["throughput"] for n in gpu_counts]
    
    ax7.plot(gpu_counts, baseline_throughput, 'o-', label='Baseline', color=colors['baseline'], linewidth=2, markersize=8)
    ax7.plot(gpu_counts, lazy_throughput, 's-', label='Lazy Mode', color=colors['lazy'], linewidth=2, markersize=8)
    ax7.set_xlabel('Number of GPUs')
    ax7.set_ylabel('Throughput (workflows/second)')
    ax7.set_title('System Throughput')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Model Switches
    ax8 = fig.add_subplot(gs[2, 1])
    baseline_switches = [results["baseline"][n]["total_model_switches"] for n in gpu_counts]
    lazy_switches = [results["lazy"][n]["total_model_switches"] for n in gpu_counts]
    
    ax8.bar(x - width/2, baseline_switches, width, label='Baseline', color=colors['baseline'], alpha=0.8)
    ax8.bar(x + width/2, lazy_switches, width, label='Lazy Mode', color=colors['lazy'], alpha=0.8)
    ax8.set_xlabel('Number of GPUs')
    ax8.set_ylabel('Total Model Switches')
    ax8.set_title('Model Loading Operations')
    ax8.set_xticks(x)
    ax8.set_xticklabels(gpu_counts)
    ax8.legend()
    
    # 9. Improvement Summary
    ax9 = fig.add_subplot(gs[2, 2])
    metrics = ['Speedup', 'Load Reduction', 'Latency Reduction', 'Switch Reduction']
    improvements = []
    
    for i, n in enumerate(gpu_counts):
        speedup = speedups[i]
        load_red = load_reductions[i]
        latency_red = (baseline_latencies[i] - lazy_latencies[i]) / baseline_latencies[i] * 100
        switch_red = (baseline_switches[i] - lazy_switches[i]) / baseline_switches[i] * 100
        improvements.append([speedup, load_red, latency_red, switch_red])
    
    improvements = np.array(improvements).T
    
    # Create grouped bar chart
    x_metrics = np.arange(len(metrics))
    bar_width = 0.15
    
    for i, n in enumerate(gpu_counts):
        offset = (i - len(gpu_counts)/2) * bar_width
        values = [improvements[j][i] for j in range(len(metrics))]
        # Normalize speedup to percentage for consistent scale
        values[0] = (values[0] - 1) * 100
        ax9.bar(x_metrics + offset, values, bar_width, label=f'{n} GPUs', alpha=0.8)
    
    ax9.set_xlabel('Metric')
    ax9.set_ylabel('Improvement (%)')
    ax9.set_title('Lazy Mode Improvements Summary')
    ax9.set_xticks(x_metrics)
    ax9.set_xticklabels(metrics, rotation=15, ha='right')
    ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.suptitle('Offline Evaluation: Baseline vs Lazy Mode GPU Scheduling', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed analysis
    create_detailed_analysis(results, gpu_counts)

def create_detailed_analysis(results: Dict, gpu_counts: List[int]):
    """Create additional detailed analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. GPU Efficiency (Compute Time / Total Time)
    ax1.set_title('GPU Compute Efficiency')
    for mode, marker, color in [('baseline', 'o', '#FF6B6B'), ('lazy', 's', '#4ECDC4')]:
        efficiencies = []
        for n in gpu_counts:
            total_time = results[mode][n]["total_time"]
            compute_time = results[mode][n]["total_inference_time"]
            efficiency = compute_time / total_time * 100
            efficiencies.append(efficiency)
        ax1.plot(gpu_counts, efficiencies, marker=marker, label=mode.capitalize(), 
                color=color, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Efficiency (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cost Efficiency (Assuming linear GPU cost)
    ax2.set_title('Cost Efficiency (Workflows per GPU-hour)')
    for mode, marker, color in [('baseline', 'o', '#FF6B6B'), ('lazy', 's', '#4ECDC4')]:
        cost_efficiencies = []
        for n in gpu_counts:
            total_time_hours = results[mode][n]["total_time"] / 3600
            gpu_hours = n * total_time_hours
            workflows_per_gpu_hour = 500 / gpu_hours  # 500 workflows
            cost_efficiencies.append(workflows_per_gpu_hour)
        ax2.plot(gpu_counts, cost_efficiencies, marker=marker, label=mode.capitalize(),
                color=color, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of GPUs')
    ax2.set_ylabel('Workflows per GPU-hour')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Latency Distribution
    ax3.set_title('Latency Distribution Comparison (12 A5000 GPUs)')
    if 12 in results["baseline"]:
        workflows_baseline = results["baseline"][12]["workflows"]
        workflows_lazy = results["lazy"][12]["workflows"]
        
        latencies_baseline = [w.completion_time - w.arrival_time for w in workflows_baseline]
        latencies_lazy = [w.completion_time - w.arrival_time for w in workflows_lazy]
        
        bins = np.linspace(0, max(max(latencies_baseline), max(latencies_lazy)), 50)
        ax3.hist(latencies_baseline, bins=bins, alpha=0.5, label='Baseline', color='#FF6B6B', edgecolor='black')
        ax3.hist(latencies_lazy, bins=bins, alpha=0.5, label='Lazy Mode', color='#4ECDC4', edgecolor='black')
        ax3.set_xlabel('Latency (seconds)')
        ax3.set_ylabel('Number of Workflows')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No data for 12 GPUs', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_xlabel('Latency (seconds)')
        ax3.set_ylabel('Number of Workflows')
    
    # 4. Model Switch Pattern
    ax4.set_title('Model Switch Reduction by GPU Count')
    switch_reductions = []
    for n in gpu_counts:
        baseline_switches = results["baseline"][n]["total_model_switches"]
        lazy_switches = results["lazy"][n]["total_model_switches"]
        reduction = (baseline_switches - lazy_switches) / baseline_switches * 100
        switch_reductions.append(reduction)
    
    bars = ax4.bar(gpu_counts, switch_reductions, color='#6C5CE7', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Number of GPUs')
    ax4.set_ylabel('Model Switch Reduction (%)')
    
    # Add value labels
    for bar, reduction in zip(bars, switch_reductions):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def main():
    print("Starting offline evaluation experiments...")
    print("Testing with 4-12 NVIDIA A5000 GPUs...")
    results = run_experiments(gpu_counts=[4, 6, 8, 10, 12])
    print("\nGenerating visualizations...")
    plot_results(results)
    # Save summary statistics
    summary = {
        "experiment_config": {
            "total_workflows": 500,
            "duration_minutes": 20,
            "gpu_type": "NVIDIA A5000",
            "gpu_counts": list(results["baseline"].keys())
        },
        "results_summary": {}
    }
    for n in results["baseline"].keys():
        baseline_time = results["baseline"][n]["total_time"]
        lazy_time = results["lazy"][n]["total_time"]
        speedup = baseline_time / lazy_time
        summary["results_summary"][f"{n}_gpus"] = {
            "baseline_time_seconds": round(baseline_time, 2),
            "lazy_time_seconds": round(lazy_time, 2),
            "speedup": round(speedup, 2),
            "load_time_reduction_percent": round(
                (results["baseline"][n]["total_load_time"] - results["lazy"][n]["total_load_time"]) /
                results["baseline"][n]["total_load_time"] * 100, 1
            ),
            "model_switch_reduction_percent": round(
                (results["baseline"][n]["total_model_switches"] - results["lazy"][n]["total_model_switches"]) /
                results["baseline"][n]["total_model_switches"] * 100, 1
            ),
            "avg_latency_reduction_percent": round(
                (results["metrics"]["baseline"][n]["avg_latency"] - results["metrics"]["lazy"][n]["avg_latency"]) /
                results["metrics"]["baseline"][n]["avg_latency"] * 100, 1
            )
        }
    with open("experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nExperiment complete! Results saved to:")
    print("- experiment_results.png")
    print("- detailed_analysis.png")
    print("- experiment_summary.json")
    print("\nKey Findings with NVIDIA A5000 GPUs:")
    print("=" * 50)
    for n in sorted(results["baseline"].keys()):
        s = summary["results_summary"][f"{n}_gpus"]
        print(f"\n{n} A5000 GPUs:")
        print(f"  - Speedup: {s['speedup']}x")
        print(f"  - Load time reduction: {s['load_time_reduction_percent']}%")
        print(f"  - Model switch reduction: {s['model_switch_reduction_percent']}%")
        print(f"  - Average latency reduction: {s['avg_latency_reduction_percent']}%")
if __name__ == "__main__":
    main()
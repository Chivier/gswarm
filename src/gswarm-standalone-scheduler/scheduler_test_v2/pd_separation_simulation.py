#!/usr/bin/env python3
"""
PD Separation Simulation for Online and Offline Scenarios

This simulation models Prompt/Decode (PD) separation with:
- Online: Random request arrivals over n minutes (default 10)
- Offline: Batch processing of all requests
- Static: Fixed model deployment strategy

Metrics tracked:
- Online: P99 latency, average latency
- Offline: Throughput
- Static: GPU utilization, P99 latency, average latency
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import heapq
from collections import defaultdict
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class Request:
    """Represents a single inference request"""
    id: int
    arrival_time: float  # in seconds
    prompt_tokens: int
    decode_tokens: int
    model_name: str
    priority: int = 1
    
    # Timing information (filled during execution)
    start_time: Optional[float] = None
    prompt_end_time: Optional[float] = None
    decode_end_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    @property
    def latency(self) -> float:
        """Total latency from arrival to completion"""
        if self.completion_time and self.arrival_time is not None:
            return self.completion_time - self.arrival_time
        return float('inf')
    
    @property
    def waiting_time(self) -> float:
        """Time spent waiting before processing starts"""
        if self.start_time and self.arrival_time is not None:
            return self.start_time - self.arrival_time
        return 0.0

@dataclass
class GPU:
    """Represents a GPU resource"""
    id: int
    current_model: Optional[str] = None
    busy_until: float = 0.0
    total_busy_time: float = 0.0
    model_switch_count: int = 0
    processed_requests: int = 0
    
    def is_available(self, current_time: float) -> bool:
        return current_time >= self.busy_until
    
    def utilization(self, total_time: float) -> float:
        return self.total_busy_time / total_time if total_time > 0 else 0.0

class PDSeparationSimulator:
    """Simulates PD separation with different scheduling strategies"""
    
    def __init__(self, 
                 num_gpus: int = 8,
                 simulation_duration: float = 600.0,  # 10 minutes default
                 model_switch_time: float = 5.0,      # 5 seconds to switch models
                 prompt_processing_rate: float = 500.0,  # tokens/second
                 decode_processing_rate: float = 100.0): # tokens/second
        
        self.num_gpus = num_gpus
        self.simulation_duration = simulation_duration
        self.model_switch_time = model_switch_time
        self.prompt_processing_rate = prompt_processing_rate
        self.decode_processing_rate = decode_processing_rate
        
        # Available models
        self.models = ["llama-7b", "llama-13b", "mistral-7b", "gpt-j-6b", "falcon-7b"]
        
        # GPU pools for PD separation
        self.prompt_gpus = []
        self.decode_gpus = []
        
    def generate_requests(self, num_requests: int, mode: str = "online") -> List[Request]:
        """Generate synthetic requests based on mode"""
        requests = []
        
        for i in range(num_requests):
            # Generate request characteristics
            model = random.choice(self.models)
            prompt_tokens = np.random.lognormal(6.0, 1.0)  # Mean ~400 tokens
            prompt_tokens = int(np.clip(prompt_tokens, 10, 4096))
            
            decode_tokens = np.random.lognormal(5.0, 0.8)  # Mean ~150 tokens
            decode_tokens = int(np.clip(decode_tokens, 10, 2048))
            
            # Priority based on request size
            priority = 2 if prompt_tokens > 1000 else 1
            
            if mode == "online":
                # Random arrival times spread across simulation duration
                arrival_time = random.uniform(0, self.simulation_duration)
            else:  # offline
                # All requests arrive at time 0
                arrival_time = 0.0
            
            requests.append(Request(
                id=i,
                arrival_time=arrival_time,
                prompt_tokens=prompt_tokens,
                decode_tokens=decode_tokens,
                model_name=model,
                priority=priority
            ))
        
        # Sort by arrival time
        requests.sort(key=lambda r: r.arrival_time)
        return requests
    
    def simulate_online_pd_separation(self, requests: List[Request]) -> Dict:
        """
        Online scheduling with PD separation
        - Prompt phase handled by dedicated prompt GPUs
        - Decode phase handled by dedicated decode GPUs
        - Random request arrivals
        """
        # Split GPUs between prompt and decode
        num_prompt_gpus = self.num_gpus // 2
        num_decode_gpus = self.num_gpus - num_prompt_gpus
        
        prompt_gpus = [GPU(id=i) for i in range(num_prompt_gpus)]
        decode_gpus = [GPU(id=i) for i in range(num_prompt_gpus, self.num_gpus)]
        
        # Priority queues for each phase
        prompt_queue = []
        decode_queue = []
        completed_requests = []
        
        # Event simulation
        current_time = 0.0
        request_idx = 0
        
        while request_idx < len(requests) or prompt_queue or decode_queue:
            # Add new arrivals to prompt queue
            while request_idx < len(requests) and requests[request_idx].arrival_time <= current_time:
                req = requests[request_idx]
                heapq.heappush(prompt_queue, (-req.priority, req.arrival_time, req))
                request_idx += 1
            
            # Process prompt phase
            for gpu in prompt_gpus:
                if gpu.is_available(current_time) and prompt_queue:
                    _, _, req = heapq.heappop(prompt_queue)
                    
                    # Model switching time if needed
                    switch_time = 0.0
                    if gpu.current_model != req.model_name:
                        switch_time = self.model_switch_time
                        gpu.current_model = req.model_name
                        gpu.model_switch_count += 1
                    
                    # Process prompt
                    req.start_time = current_time + switch_time
                    prompt_time = req.prompt_tokens / self.prompt_processing_rate
                    req.prompt_end_time = req.start_time + prompt_time
                    
                    gpu.busy_until = req.prompt_end_time
                    gpu.total_busy_time += switch_time + prompt_time
                    
                    # Add to decode queue
                    heapq.heappush(decode_queue, (req.prompt_end_time, req))
            
            # Process decode phase
            for gpu in decode_gpus:
                if gpu.is_available(current_time) and decode_queue:
                    ready_time, req = decode_queue[0]
                    if ready_time <= current_time:
                        heapq.heappop(decode_queue)
                        
                        # Model switching time if needed
                        switch_time = 0.0
                        if gpu.current_model != req.model_name:
                            switch_time = self.model_switch_time
                            gpu.current_model = req.model_name
                            gpu.model_switch_count += 1
                        
                        # Process decode
                        decode_start = current_time + switch_time
                        decode_time = req.decode_tokens / self.decode_processing_rate
                        req.decode_end_time = decode_start + decode_time
                        req.completion_time = req.decode_end_time
                        
                        gpu.busy_until = req.decode_end_time
                        gpu.total_busy_time += switch_time + decode_time
                        gpu.processed_requests += 1
                        
                        completed_requests.append(req)
            
            # Advance time
            next_events = []
            
            # Next request arrival
            if request_idx < len(requests):
                next_events.append(requests[request_idx].arrival_time)
            
            # Next GPU available times
            for gpu in prompt_gpus + decode_gpus:
                if gpu.busy_until > current_time:
                    next_events.append(gpu.busy_until)
            
            # Next decode ready time
            if decode_queue:
                next_events.append(decode_queue[0][0])
            
            if next_events:
                current_time = min(next_events)
            else:
                break
        
        # Calculate metrics
        latencies = [req.latency for req in completed_requests]
        waiting_times = [req.waiting_time for req in completed_requests]
        
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        avg_latency = np.mean(latencies) if latencies else 0
        
        # GPU utilization
        prompt_utilization = np.mean([gpu.utilization(current_time) for gpu in prompt_gpus])
        decode_utilization = np.mean([gpu.utilization(current_time) for gpu in decode_gpus])
        
        return {
            "strategy": "online_pd_separation",
            "completed_requests": len(completed_requests),
            "p99_latency": p99_latency,
            "avg_latency": avg_latency,
            "avg_waiting_time": np.mean(waiting_times) if waiting_times else 0,
            "prompt_gpu_utilization": prompt_utilization,
            "decode_gpu_utilization": decode_utilization,
            "total_gpu_utilization": (prompt_utilization + decode_utilization) / 2,
            "model_switches": sum(gpu.model_switch_count for gpu in prompt_gpus + decode_gpus),
            "simulation_time": current_time
        }
    
    def simulate_offline_pd_separation(self, requests: List[Request]) -> Dict:
        """
        Offline batch scheduling with PD separation
        - All requests known upfront
        - Optimize for throughput
        - Minimize model switching
        """
        # Split GPUs between prompt and decode
        num_prompt_gpus = self.num_gpus // 2
        num_decode_gpus = self.num_gpus - num_prompt_gpus
        
        prompt_gpus = [GPU(id=i) for i in range(num_prompt_gpus)]
        decode_gpus = [GPU(id=i) for i in range(num_prompt_gpus, self.num_gpus)]
        
        # Group requests by model to minimize switching
        model_groups = defaultdict(list)
        for req in requests:
            model_groups[req.model_name].append(req)
        
        # Sort models by total workload
        model_workload = {}
        for model, reqs in model_groups.items():
            total_prompt = sum(r.prompt_tokens for r in reqs)
            total_decode = sum(r.decode_tokens for r in reqs)
            model_workload[model] = total_prompt / self.prompt_processing_rate + \
                                   total_decode / self.decode_processing_rate
        
        sorted_models = sorted(model_workload.keys(), key=lambda m: model_workload[m], reverse=True)
        
        # Process all prompts first (grouped by model)
        current_time = 0.0
        decode_ready_queue = []
        
        # Round-robin assignment to prompt GPUs
        gpu_idx = 0
        for model in sorted_models:
            model_requests = model_groups[model]
            
            for req in model_requests:
                gpu = prompt_gpus[gpu_idx % num_prompt_gpus]
                
                # Model switching time if needed
                switch_time = 0.0
                if gpu.current_model != model:
                    switch_time = self.model_switch_time
                    gpu.current_model = model
                    gpu.model_switch_count += 1
                
                # Schedule prompt processing
                req.start_time = gpu.busy_until + switch_time
                prompt_time = req.prompt_tokens / self.prompt_processing_rate
                req.prompt_end_time = req.start_time + prompt_time
                
                gpu.busy_until = req.prompt_end_time
                gpu.total_busy_time += switch_time + prompt_time
                
                decode_ready_queue.append((req.prompt_end_time, req))
                gpu_idx += 1
        
        # Sort decode queue by ready time
        decode_ready_queue.sort(key=lambda x: x[0])
        
        # Process decode phase (also grouped by model where possible)
        completed_requests = []
        decode_by_model = defaultdict(list)
        
        for ready_time, req in decode_ready_queue:
            decode_by_model[req.model_name].append((ready_time, req))
        
        # Assign decode work to GPUs
        gpu_idx = 0
        for model in sorted_models:
            if model in decode_by_model:
                for ready_time, req in decode_by_model[model]:
                    gpu = decode_gpus[gpu_idx % num_decode_gpus]
                    
                    # Model switching time if needed
                    switch_time = 0.0
                    if gpu.current_model != model:
                        switch_time = self.model_switch_time
                        gpu.current_model = model
                        gpu.model_switch_count += 1
                    
                    # Schedule decode processing
                    decode_start = max(gpu.busy_until + switch_time, ready_time)
                    decode_time = req.decode_tokens / self.decode_processing_rate
                    req.decode_end_time = decode_start + decode_time
                    req.completion_time = req.decode_end_time
                    
                    gpu.busy_until = req.decode_end_time
                    gpu.total_busy_time += switch_time + decode_time
                    gpu.processed_requests += 1
                    
                    completed_requests.append(req)
                    gpu_idx += 1
        
        # Calculate metrics
        makespan = max(gpu.busy_until for gpu in prompt_gpus + decode_gpus)
        throughput = len(completed_requests) / makespan if makespan > 0 else 0
        
        # GPU utilization
        prompt_utilization = np.mean([gpu.utilization(makespan) for gpu in prompt_gpus])
        decode_utilization = np.mean([gpu.utilization(makespan) for gpu in decode_gpus])
        
        return {
            "strategy": "offline_pd_separation",
            "completed_requests": len(completed_requests),
            "throughput": throughput,
            "makespan": makespan,
            "prompt_gpu_utilization": prompt_utilization,
            "decode_gpu_utilization": decode_utilization,
            "total_gpu_utilization": (prompt_utilization + decode_utilization) / 2,
            "model_switches": sum(gpu.model_switch_count for gpu in prompt_gpus + decode_gpus),
            "avg_model_switches_per_gpu": sum(gpu.model_switch_count for gpu in prompt_gpus + decode_gpus) / self.num_gpus
        }
    
    def simulate_static_deployment(self, requests: List[Request]) -> Dict:
        """
        Static deployment strategy
        - Models permanently assigned to GPUs
        - No model switching overhead
        - Focus on GPU utilization and latency
        """
        # Assign models to GPUs statically
        gpu_model_assignment = {}
        models_per_gpu = len(self.models) / self.num_gpus
        
        for i in range(self.num_gpus):
            model_idx = int(i / self.num_gpus * len(self.models))
            gpu_model_assignment[i] = self.models[model_idx]
        
        # Create GPU objects with pre-assigned models
        gpus = []
        for i in range(self.num_gpus):
            gpu = GPU(id=i)
            gpu.current_model = gpu_model_assignment[i]
            gpus.append(gpu)
        
        # Create model-to-GPU mapping
        model_to_gpus = defaultdict(list)
        for gpu_id, model in gpu_model_assignment.items():
            model_to_gpus[model].append(gpus[gpu_id])
        
        # Process requests
        completed_requests = []
        request_queue = list(requests)
        current_time = 0.0
        
        while request_queue or any(gpu.busy_until > current_time for gpu in gpus):
            # Process arrivals
            ready_requests = [r for r in request_queue if r.arrival_time <= current_time]
            
            for req in ready_requests:
                # Find available GPU with the required model
                available_gpus = [gpu for gpu in model_to_gpus[req.model_name] 
                                if gpu.is_available(current_time)]
                
                if available_gpus:
                    # Choose least loaded GPU
                    gpu = min(available_gpus, key=lambda g: g.total_busy_time)
                    
                    # No model switching needed in static deployment
                    req.start_time = current_time
                    
                    # Process both prompt and decode on same GPU
                    prompt_time = req.prompt_tokens / self.prompt_processing_rate
                    decode_time = req.decode_tokens / self.decode_processing_rate
                    total_time = prompt_time + decode_time
                    
                    req.prompt_end_time = req.start_time + prompt_time
                    req.decode_end_time = req.start_time + total_time
                    req.completion_time = req.decode_end_time
                    
                    gpu.busy_until = req.completion_time
                    gpu.total_busy_time += total_time
                    gpu.processed_requests += 1
                    
                    completed_requests.append(req)
                    request_queue.remove(req)
            
            # Advance time
            next_events = []
            
            # Next request arrival
            remaining_arrivals = [r.arrival_time for r in request_queue if r.arrival_time > current_time]
            if remaining_arrivals:
                next_events.append(min(remaining_arrivals))
            
            # Next GPU available
            busy_gpus = [gpu.busy_until for gpu in gpus if gpu.busy_until > current_time]
            if busy_gpus:
                next_events.append(min(busy_gpus))
            
            if next_events:
                current_time = min(next_events)
            else:
                break
        
        # Calculate metrics
        latencies = [req.latency for req in completed_requests]
        
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        avg_latency = np.mean(latencies) if latencies else 0
        
        # GPU utilization
        total_time = max(current_time, max(gpu.busy_until for gpu in gpus))
        gpu_utilizations = [gpu.utilization(total_time) for gpu in gpus]
        
        return {
            "strategy": "static_deployment",
            "completed_requests": len(completed_requests),
            "p99_latency": p99_latency,
            "avg_latency": avg_latency,
            "gpu_utilization": np.mean(gpu_utilizations),
            "gpu_utilization_std": np.std(gpu_utilizations),
            "model_switches": 0,  # No switches in static deployment
            "requests_dropped": len(request_queue),  # Requests that couldn't be processed
            "simulation_time": total_time
        }
    
    def calculate_revenue_benefits(self, baseline_metrics: Dict, strategy_metrics: Dict, 
                                 hourly_gpu_cost: float = 2.0,
                                 revenue_per_request: float = 0.01,
                                 sla_penalty_per_second: float = 0.1) -> Dict:
        """
        Calculate revenue benefits compared to baseline
        
        Args:
            baseline_metrics: Metrics from baseline strategy
            strategy_metrics: Metrics from improved strategy
            hourly_gpu_cost: Cost per GPU per hour
            revenue_per_request: Revenue per completed request
            sla_penalty_per_second: Penalty for SLA violations (high latency)
        """
        # Calculate operational hours
        baseline_hours = baseline_metrics.get('simulation_time', self.simulation_duration) / 3600
        strategy_hours = strategy_metrics.get('simulation_time', self.simulation_duration) / 3600
        
        # GPU costs
        baseline_gpu_cost = self.num_gpus * hourly_gpu_cost * baseline_hours
        strategy_gpu_cost = self.num_gpus * hourly_gpu_cost * strategy_hours
        
        # Revenue from completed requests
        baseline_revenue = baseline_metrics['completed_requests'] * revenue_per_request
        strategy_revenue = strategy_metrics['completed_requests'] * revenue_per_request
        
        # SLA penalties (based on P99 latency exceeding threshold)
        sla_threshold = 10.0  # 10 second SLA
        baseline_sla_penalty = max(0, baseline_metrics.get('p99_latency', 0) - sla_threshold) * \
                              sla_penalty_per_second * baseline_metrics['completed_requests'] * 0.01
        strategy_sla_penalty = max(0, strategy_metrics.get('p99_latency', 0) - sla_threshold) * \
                              sla_penalty_per_second * strategy_metrics['completed_requests'] * 0.01
        
        # Total profit
        baseline_profit = baseline_revenue - baseline_gpu_cost - baseline_sla_penalty
        strategy_profit = strategy_revenue - strategy_gpu_cost - strategy_sla_penalty
        
        # Calculate improvements
        revenue_improvement = strategy_profit - baseline_profit
        revenue_improvement_pct = (revenue_improvement / abs(baseline_profit)) * 100 if baseline_profit != 0 else 0
        
        # Efficiency metrics
        gpu_efficiency_gain = strategy_metrics.get('total_gpu_utilization', 
                                                  strategy_metrics.get('gpu_utilization', 0)) - \
                             baseline_metrics.get('total_gpu_utilization', 
                                                baseline_metrics.get('gpu_utilization', 0))
        
        latency_improvement = baseline_metrics.get('p99_latency', float('inf')) - \
                            strategy_metrics.get('p99_latency', float('inf'))
        
        throughput_gain = strategy_metrics.get('throughput', 0) - baseline_metrics.get('throughput', 0)
        
        return {
            "strategy": strategy_metrics['strategy'],
            "revenue_improvement": revenue_improvement,
            "revenue_improvement_pct": revenue_improvement_pct,
            "baseline_profit": baseline_profit,
            "strategy_profit": strategy_profit,
            "gpu_efficiency_gain": gpu_efficiency_gain,
            "latency_improvement": latency_improvement,
            "throughput_gain": throughput_gain,
            "cost_savings": baseline_gpu_cost - strategy_gpu_cost,
            "sla_penalty_reduction": baseline_sla_penalty - strategy_sla_penalty,
            "additional_requests_served": strategy_metrics['completed_requests'] - \
                                        baseline_metrics['completed_requests']
        }

def run_comprehensive_simulation(num_requests: int = 1000, 
                               num_gpus: int = 8,
                               simulation_duration: float = 600.0):
    """Run comprehensive simulation comparing all strategies"""
    
    print(f"Starting PD Separation Simulation")
    print(f"Requests: {num_requests}, GPUs: {num_gpus}, Duration: {simulation_duration}s")
    print("="*60)
    
    simulator = PDSeparationSimulator(
        num_gpus=num_gpus,
        simulation_duration=simulation_duration
    )
    
    # Generate requests for online and offline scenarios
    online_requests = simulator.generate_requests(num_requests, mode="online")
    offline_requests = simulator.generate_requests(num_requests, mode="offline")
    
    results = {}
    
    # Run simulations
    print("\n1. Running Online PD Separation...")
    results['online'] = simulator.simulate_online_pd_separation(online_requests)
    
    print("\n2. Running Offline PD Separation...")
    results['offline'] = simulator.simulate_offline_pd_separation(offline_requests)
    
    print("\n3. Running Static Deployment...")
    results['static'] = simulator.simulate_static_deployment(online_requests)
    
    # Baseline simulation (simple FIFO without PD separation)
    print("\n4. Running Baseline (for comparison)...")
    # For baseline, we'll use a simple approximation
    baseline_latency = np.mean([r.prompt_tokens / simulator.prompt_processing_rate + 
                               r.decode_tokens / simulator.decode_processing_rate 
                               for r in online_requests])
    baseline_p99 = baseline_latency * 3  # Rough approximation
    
    results['baseline'] = {
        'strategy': 'baseline_fifo',
        'completed_requests': num_requests * 0.95,  # Assume 95% completion
        'p99_latency': baseline_p99,
        'avg_latency': baseline_latency,
        'gpu_utilization': 0.5,  # Typical utilization without optimization
        'throughput': num_requests * 0.95 / simulation_duration,
        'simulation_time': simulation_duration
    }
    
    # Calculate revenue benefits
    print("\n5. Calculating Revenue Benefits...")
    revenue_results = {}
    
    for strategy in ['online', 'offline', 'static']:
        revenue_results[strategy] = simulator.calculate_revenue_benefits(
            results['baseline'], 
            results[strategy]
        )
    
    # Print results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    # Online metrics
    print("\nOnline PD Separation:")
    print(f"  - P99 Latency: {results['online']['p99_latency']:.2f}s")
    print(f"  - Avg Latency: {results['online']['avg_latency']:.2f}s")
    print(f"  - GPU Utilization: {results['online']['total_gpu_utilization']*100:.1f}%")
    print(f"  - Completed Requests: {results['online']['completed_requests']}")
    
    # Offline metrics
    print("\nOffline PD Separation:")
    print(f"  - Throughput: {results['offline']['throughput']:.2f} req/s")
    print(f"  - Makespan: {results['offline']['makespan']:.2f}s")
    print(f"  - GPU Utilization: {results['offline']['total_gpu_utilization']*100:.1f}%")
    print(f"  - Model Switches: {results['offline']['model_switches']}")
    
    # Static metrics
    print("\nStatic Deployment:")
    print(f"  - P99 Latency: {results['static']['p99_latency']:.2f}s")
    print(f"  - Avg Latency: {results['static']['avg_latency']:.2f}s")
    print(f"  - GPU Utilization: {results['static']['gpu_utilization']*100:.1f}%")
    print(f"  - Requests Dropped: {results['static']['requests_dropped']}")
    
    # Revenue benefits
    print("\n" + "="*60)
    print("REVENUE BENEFITS vs BASELINE")
    print("="*60)
    
    for strategy in ['online', 'offline', 'static']:
        rev = revenue_results[strategy]
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  - Revenue Improvement: ${rev['revenue_improvement']:.2f} ({rev['revenue_improvement_pct']:.1f}%)")
        print(f"  - GPU Efficiency Gain: {rev['gpu_efficiency_gain']*100:.1f}%")
        print(f"  - Latency Improvement: {rev['latency_improvement']:.2f}s")
        print(f"  - Cost Savings: ${rev['cost_savings']:.2f}")
        print(f"  - SLA Penalty Reduction: ${rev['sla_penalty_reduction']:.2f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"pd_separation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'simulation_params': {
                'num_requests': num_requests,
                'num_gpus': num_gpus,
                'simulation_duration': simulation_duration
            },
            'results': results,
            'revenue_analysis': revenue_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create visualization
    create_performance_visualization(results, revenue_results)
    
    return results, revenue_results

def create_performance_visualization(results: Dict, revenue_results: Dict):
    """Create visualization of performance metrics"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    strategies = ['baseline', 'online', 'offline', 'static']
    
    # 1. Latency comparison
    latencies = []
    p99_latencies = []
    for s in strategies:
        if 'avg_latency' in results[s]:
            latencies.append(results[s]['avg_latency'])
            p99_latencies.append(results[s].get('p99_latency', 0))
        else:
            latencies.append(0)
            p99_latencies.append(0)
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax1.bar(x - width/2, latencies, width, label='Avg Latency', alpha=0.8)
    ax1.bar(x + width/2, p99_latencies, width, label='P99 Latency', alpha=0.8)
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Latency (seconds)')
    ax1.set_title('Latency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. GPU Utilization
    utilizations = []
    for s in strategies:
        util = results[s].get('total_gpu_utilization', results[s].get('gpu_utilization', 0))
        utilizations.append(util * 100)
    
    ax2.bar(strategies, utilizations, alpha=0.8, color='green')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('GPU Utilization (%)')
    ax2.set_title('GPU Utilization Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Revenue improvement
    revenue_strategies = ['online', 'offline', 'static']
    revenue_improvements = [revenue_results[s]['revenue_improvement_pct'] 
                          for s in revenue_strategies]
    
    colors = ['red' if x < 0 else 'green' for x in revenue_improvements]
    ax3.bar(revenue_strategies, revenue_improvements, alpha=0.8, color=colors)
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Revenue Improvement (%)')
    ax3.set_title('Revenue Improvement vs Baseline')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency metrics
    metrics = ['GPU Efficiency', 'Latency Reduction', 'Cost Savings']
    online_metrics = [
        revenue_results['online']['gpu_efficiency_gain'] * 100,
        revenue_results['online']['latency_improvement'],
        revenue_results['online']['cost_savings']
    ]
    offline_metrics = [
        revenue_results['offline']['gpu_efficiency_gain'] * 100,
        revenue_results['offline']['latency_improvement'],
        revenue_results['offline']['cost_savings']
    ]
    static_metrics = [
        revenue_results['static']['gpu_efficiency_gain'] * 100,
        revenue_results['static']['latency_improvement'],
        revenue_results['static']['cost_savings']
    ]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax4.bar(x - width, online_metrics, width, label='Online', alpha=0.8)
    ax4.bar(x, offline_metrics, width, label='Offline', alpha=0.8)
    ax4.bar(x + width, static_metrics, width, label='Static', alpha=0.8)
    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Improvement')
    ax4.set_title('Efficiency Improvements')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'pd_separation_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: pd_separation_analysis_{timestamp}.png")
    
    plt.close()

if __name__ == "__main__":
    # Run simulation with default parameters
    run_comprehensive_simulation(
        num_requests=1000,
        num_gpus=8,
        simulation_duration=600.0  # 10 minutes
    )
    
    # Run additional simulations with different parameters
    print("\n\n" + "="*60)
    print("Running scalability test with more GPUs...")
    print("="*60)
    
    run_comprehensive_simulation(
        num_requests=2000,
        num_gpus=16,
        simulation_duration=600.0
    )
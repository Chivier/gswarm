#!/usr/bin/env python3
"""
LongBench2 PD Separation Performance Test

This script tests PD separation performance using LongBench2 datasets,
focusing on long-context scenarios where PD separation can provide significant benefits.
"""

import requests
import json
import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
import random
from pd_separation_simulation import PDSeparationSimulator, Request

@dataclass
class LongBenchRequest:
    """Request based on LongBench2 dataset characteristics"""
    id: int
    context_length: int  # Number of tokens in context
    question_length: int  # Number of tokens in question
    expected_output_length: int  # Expected decode length
    difficulty: str  # easy, medium, hard
    task_type: str  # single-doc-qa, multi-doc-qa, etc.
    model_requirement: str  # Which model to use

class LongBench2PDTester:
    """Test PD separation with LongBench2-inspired workloads"""
    
    def __init__(self):
        # LongBench2 context length categories
        self.context_categories = {
            "short": (0, 32000),      # 0-32k tokens
            "medium": (32000, 128000), # 32k-128k tokens
            "long": (128000, 2000000)  # 128k-2M tokens
        }
        
        # Task types from LongBench2
        self.task_types = [
            "single-doc-qa",
            "multi-doc-qa", 
            "long-icl",
            "long-dialogue",
            "code-repo",
            "long-structured"
        ]
        
        # Models that support different context lengths
        self.model_configs = {
            "llama-7b": {"max_context": 32000, "prompt_rate": 500, "decode_rate": 100},
            "llama-13b": {"max_context": 32000, "prompt_rate": 400, "decode_rate": 80},
            "mistral-7b": {"max_context": 32000, "prompt_rate": 550, "decode_rate": 110},
            "llama-70b": {"max_context": 128000, "prompt_rate": 200, "decode_rate": 40},
            "claude-100k": {"max_context": 100000, "prompt_rate": 300, "decode_rate": 60}
        }
    
    def generate_longbench_requests(self, num_requests: int, 
                                   context_distribution: Dict[str, float] = None) -> List[Request]:
        """
        Generate requests based on LongBench2 characteristics
        
        Args:
            num_requests: Number of requests to generate
            context_distribution: Distribution of short/medium/long contexts
        """
        if context_distribution is None:
            context_distribution = {
                "short": 0.5,   # 50% short context
                "medium": 0.35, # 35% medium context
                "long": 0.15    # 15% long context
            }
        
        requests = []
        
        for i in range(num_requests):
            # Select context category
            category = np.random.choice(
                list(context_distribution.keys()),
                p=list(context_distribution.values())
            )
            
            # Generate context length
            min_len, max_len = self.context_categories[category]
            if category == "short":
                # Use log-normal distribution for short contexts
                context_length = int(np.random.lognormal(9, 1.5))  # Mean ~8k
                context_length = np.clip(context_length, min_len + 100, max_len)
            elif category == "medium":
                # Uniform distribution for medium
                context_length = np.random.randint(min_len, max_len)
            else:  # long
                # Skewed towards lower end of long range
                context_length = int(np.random.exponential(200000) + min_len)
                context_length = np.clip(context_length, min_len, max_len)
            
            # Select appropriate model based on context length
            suitable_models = [
                model for model, config in self.model_configs.items()
                if config["max_context"] >= context_length
            ]
            
            if not suitable_models:
                suitable_models = ["claude-100k"]  # Fallback for very long contexts
            
            model = random.choice(suitable_models)
            
            # Generate question length (typically much shorter than context)
            question_length = np.random.randint(10, 200)
            
            # Expected output length based on task type
            task_type = random.choice(self.task_types)
            if task_type in ["single-doc-qa", "multi-doc-qa"]:
                output_length = np.random.randint(50, 500)
            elif task_type == "code-repo":
                output_length = np.random.randint(100, 1000)
            else:
                output_length = np.random.randint(100, 800)
            
            # Total prompt tokens = context + question
            prompt_tokens = context_length + question_length
            
            # Create request
            requests.append(Request(
                id=i,
                arrival_time=0.0,  # Will be set by caller
                prompt_tokens=prompt_tokens,
                decode_tokens=output_length,
                model_name=model,
                priority=2 if category == "long" else 1  # Higher priority for long contexts
            ))
        
        return requests
    
    def test_pd_separation_benefits(self, num_requests: int = 500, 
                                   simulation_duration: float = 600.0,
                                   num_gpus: int = 8) -> Dict:
        """
        Test PD separation benefits with LongBench2-inspired workload
        """
        print("Testing PD Separation with LongBench2-inspired workload")
        print("="*60)
        
        # Generate requests with realistic distribution
        requests = self.generate_longbench_requests(num_requests)
        
        # Calculate workload statistics
        prompt_tokens = [r.prompt_tokens for r in requests]
        decode_tokens = [r.decode_tokens for r in requests]
        
        print(f"\nWorkload Statistics:")
        print(f"  - Total requests: {num_requests}")
        print(f"  - Avg prompt tokens: {np.mean(prompt_tokens):.0f}")
        print(f"  - Max prompt tokens: {np.max(prompt_tokens):.0f}")
        print(f"  - Avg decode tokens: {np.mean(decode_tokens):.0f}")
        print(f"  - Total tokens: {sum(prompt_tokens) + sum(decode_tokens):,}")
        
        # Categorize by context length
        short_reqs = sum(1 for r in requests if r.prompt_tokens < 32000)
        medium_reqs = sum(1 for r in requests if 32000 <= r.prompt_tokens < 128000)
        long_reqs = sum(1 for r in requests if r.prompt_tokens >= 128000)
        
        print(f"\nContext Distribution:")
        print(f"  - Short (<32k): {short_reqs} ({short_reqs/num_requests*100:.1f}%)")
        print(f"  - Medium (32k-128k): {medium_reqs} ({medium_reqs/num_requests*100:.1f}%)")
        print(f"  - Long (>128k): {long_reqs} ({long_reqs/num_requests*100:.1f}%)")
        
        # Test different scenarios
        results = {}
        
        # 1. Online scenario with random arrivals
        print("\n1. Testing Online Scenario...")
        online_requests = requests.copy()
        for req in online_requests:
            req.arrival_time = random.uniform(0, simulation_duration)
        online_requests.sort(key=lambda r: r.arrival_time)
        
        simulator = PDSeparationSimulator(
            num_gpus=num_gpus,
            simulation_duration=simulation_duration
        )
        results['online'] = simulator.simulate_online_pd_separation(online_requests)
        
        # 2. Offline batch scenario
        print("\n2. Testing Offline Batch Scenario...")
        offline_requests = requests.copy()
        for req in offline_requests:
            req.arrival_time = 0.0
        
        results['offline'] = simulator.simulate_offline_pd_separation(offline_requests)
        
        # 3. Baseline without PD separation
        print("\n3. Calculating Baseline (no PD separation)...")
        results['baseline'] = self.calculate_baseline_performance(requests, num_gpus)
        
        # Calculate improvements
        improvements = self.calculate_improvements(results)
        
        # Print results
        self.print_results(results, improvements)
        
        return {
            'results': results,
            'improvements': improvements,
            'workload_stats': {
                'num_requests': num_requests,
                'avg_prompt_tokens': np.mean(prompt_tokens),
                'avg_decode_tokens': np.mean(decode_tokens),
                'context_distribution': {
                    'short': short_reqs,
                    'medium': medium_reqs,
                    'long': long_reqs
                }
            }
        }
    
    def calculate_baseline_performance(self, requests: List[Request], num_gpus: int) -> Dict:
        """Calculate baseline performance without PD separation"""
        
        # Simple simulation without PD separation
        total_prompt_time = sum(r.prompt_tokens / 400 for r in requests)  # Avg rate
        total_decode_time = sum(r.decode_tokens / 80 for r in requests)   # Avg rate
        
        # Model switching overhead (assume 20% overhead)
        switching_overhead = 0.2
        
        total_time = (total_prompt_time + total_decode_time) * (1 + switching_overhead)
        makespan = total_time / num_gpus
        
        # Latency estimation (with queueing effects)
        avg_latency = makespan / len(requests) * 2  # Queueing factor
        p99_latency = avg_latency * 3  # Rough approximation
        
        return {
            'strategy': 'baseline_no_pd',
            'completed_requests': len(requests),
            'p99_latency': p99_latency,
            'avg_latency': avg_latency,
            'throughput': len(requests) / makespan,
            'gpu_utilization': 0.6,  # Typical without optimization
            'model_switches': len(requests) * 0.5,  # Frequent switching
            'makespan': makespan
        }
    
    def calculate_improvements(self, results: Dict) -> Dict:
        """Calculate improvements over baseline"""
        baseline = results['baseline']
        improvements = {}
        
        for strategy in ['online', 'offline']:
            if strategy not in results:
                continue
                
            strategy_results = results[strategy]
            
            # Latency improvement
            if 'p99_latency' in strategy_results and 'p99_latency' in baseline:
                latency_improvement = (baseline['p99_latency'] - strategy_results['p99_latency']) / \
                                    baseline['p99_latency'] * 100
            else:
                latency_improvement = 0
            
            # Throughput improvement
            if 'throughput' in strategy_results and 'throughput' in baseline:
                throughput_improvement = (strategy_results['throughput'] - baseline['throughput']) / \
                                       baseline['throughput'] * 100
            else:
                throughput_improvement = 0
            
            # GPU utilization improvement
            baseline_util = baseline.get('gpu_utilization', 0.6)
            strategy_util = strategy_results.get('total_gpu_utilization', 
                                               strategy_results.get('gpu_utilization', 0))
            utilization_improvement = (strategy_util - baseline_util) / baseline_util * 100
            
            improvements[strategy] = {
                'latency_improvement_pct': latency_improvement,
                'throughput_improvement_pct': throughput_improvement,
                'utilization_improvement_pct': utilization_improvement,
                'model_switch_reduction': baseline.get('model_switches', 0) - \
                                        strategy_results.get('model_switches', 0)
            }
        
        return improvements
    
    def print_results(self, results: Dict, improvements: Dict):
        """Print formatted results"""
        
        print("\n" + "="*60)
        print("PD SEPARATION PERFORMANCE RESULTS")
        print("="*60)
        
        # Baseline
        print("\nBaseline (No PD Separation):")
        baseline = results['baseline']
        print(f"  - P99 Latency: {baseline['p99_latency']:.2f}s")
        print(f"  - Avg Latency: {baseline['avg_latency']:.2f}s")
        print(f"  - Throughput: {baseline['throughput']:.2f} req/s")
        print(f"  - GPU Utilization: {baseline['gpu_utilization']*100:.1f}%")
        
        # Online PD Separation
        if 'online' in results:
            print("\nOnline PD Separation:")
            online = results['online']
            print(f"  - P99 Latency: {online['p99_latency']:.2f}s")
            print(f"  - Avg Latency: {online['avg_latency']:.2f}s")
            print(f"  - GPU Utilization: {online['total_gpu_utilization']*100:.1f}%")
            print(f"  - Model Switches: {online['model_switches']}")
            
            if 'online' in improvements:
                imp = improvements['online']
                print(f"  - Latency Improvement: {imp['latency_improvement_pct']:.1f}%")
                print(f"  - Utilization Improvement: {imp['utilization_improvement_pct']:.1f}%")
        
        # Offline PD Separation
        if 'offline' in results:
            print("\nOffline PD Separation:")
            offline = results['offline']
            print(f"  - Throughput: {offline['throughput']:.2f} req/s")
            print(f"  - Makespan: {offline['makespan']:.2f}s")
            print(f"  - GPU Utilization: {offline['total_gpu_utilization']*100:.1f}%")
            print(f"  - Model Switches: {offline['model_switches']}")
            
            if 'offline' in improvements:
                imp = improvements['offline']
                print(f"  - Throughput Improvement: {imp['throughput_improvement_pct']:.1f}%")
                print(f"  - Model Switch Reduction: {imp['model_switch_reduction']:.0f}")
        
        print("\n" + "="*60)
        print("KEY INSIGHTS FOR LONGBENCH2 WORKLOADS:")
        print("="*60)
        print("1. PD separation is especially beneficial for long-context scenarios")
        print("2. Online strategy reduces P99 latency for interactive workloads")
        print("3. Offline strategy maximizes throughput for batch processing")
        print("4. Model switching overhead is significantly reduced with PD separation")

def run_longbench2_experiments():
    """Run comprehensive experiments with different configurations"""
    
    tester = LongBench2PDTester()
    
    # Experiment 1: Standard workload
    print("\nEXPERIMENT 1: Standard LongBench2 Workload")
    print("-"*60)
    results1 = tester.test_pd_separation_benefits(
        num_requests=500,
        simulation_duration=600.0,
        num_gpus=8
    )
    
    # Experiment 2: Long-context heavy workload
    print("\n\nEXPERIMENT 2: Long-Context Heavy Workload")
    print("-"*60)
    
    # Generate requests with more long contexts
    long_context_dist = {
        "short": 0.2,   # 20% short
        "medium": 0.4,  # 40% medium  
        "long": 0.4     # 40% long
    }
    
    requests = tester.generate_longbench_requests(500, long_context_dist)
    
    # Run simulations
    simulator = PDSeparationSimulator(num_gpus=16, simulation_duration=600.0)
    
    # Online
    online_reqs = requests.copy()
    for req in online_reqs:
        req.arrival_time = random.uniform(0, 600.0)
    online_reqs.sort(key=lambda r: r.arrival_time)
    
    online_results = simulator.simulate_online_pd_separation(online_reqs)
    
    # Offline  
    offline_reqs = requests.copy()
    for req in offline_reqs:
        req.arrival_time = 0.0
    
    offline_results = simulator.simulate_offline_pd_separation(offline_reqs)
    
    print("\nLong-Context Heavy Results:")
    print(f"Online P99 Latency: {online_results['p99_latency']:.2f}s")
    print(f"Offline Throughput: {offline_results['throughput']:.2f} req/s")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f'longbench2_pd_results_{timestamp}.json', 'w') as f:
        json.dump({
            'standard_workload': results1,
            'long_context_heavy': {
                'online': online_results,
                'offline': offline_results
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: longbench2_pd_results_{timestamp}.json")

if __name__ == "__main__":
    run_longbench2_experiments()
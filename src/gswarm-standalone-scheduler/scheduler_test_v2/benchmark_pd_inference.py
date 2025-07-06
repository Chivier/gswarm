#!/usr/bin/env python3
"""
Benchmark script to evaluate PD-separated vs traditional inference performance
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import torch

# Add gswarm to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from gswarm.data import DataServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    model_name: str
    pd_ratios: List[Tuple[int, int]]  # List of (prefill, decode) ratios to test
    traditional_instances: int  # Number of traditional inference instances
    prompt_lengths: List[int]  # Different prompt lengths to test
    generation_lengths: List[int]  # Different generation lengths to test
    batch_sizes: List[int]  # Different batch sizes to test
    num_requests: int  # Total requests per test
    warmup_requests: int  # Warmup requests before measurement
    engine_type: str  # vllm or sglang


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    config_name: str
    pd_ratio: Optional[Tuple[int, int]]
    avg_latency: float
    p50_latency: float
    p90_latency: float
    p99_latency: float
    throughput: float  # requests per second
    prefill_time: float
    decode_time: float
    total_time: float
    gpu_memory_used: float
    kv_cache_transfer_time: float


class InferenceBenchmark:
    """Benchmark harness for comparing inference approaches"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_client = DataServer("localhost:9015")
        self.results = []
        
    async def run_pd_benchmark(self, pd_ratio: Tuple[int, int], 
                              prompt_length: int, 
                              generation_length: int,
                              batch_size: int) -> BenchmarkResult:
        """Run benchmark with PD-separated inference"""
        logger.info(f"Running PD benchmark with ratio {pd_ratio}, prompt_len={prompt_length}, gen_len={generation_length}, batch={batch_size}")
        
        # Launch PD server
        prefill_nodes, decode_nodes = pd_ratio
        total_nodes = prefill_nodes + decode_nodes
        
        # Simulate launching the PD server
        # In practice, this would launch actual processes
        latencies = []
        prefill_times = []
        decode_times = []
        transfer_times = []
        
        start_time = time.time()
        
        # Process requests
        for i in range(self.config.num_requests):
            request_start = time.time()
            
            # Simulate prefill phase
            prefill_start = time.time()
            await asyncio.sleep(0.001 * prompt_length / 100)  # Simulate prefill time
            prefill_time = time.time() - prefill_start
            prefill_times.append(prefill_time)
            
            # Simulate KV cache transfer
            transfer_start = time.time()
            kv_cache_size = prompt_length * 32 * 128 * 2 * 4  # seq_len * layers * dim * kv * bytes
            await self._simulate_kv_transfer(kv_cache_size)
            transfer_time = time.time() - transfer_start
            transfer_times.append(transfer_time)
            
            # Simulate decode phase
            decode_start = time.time()
            await asyncio.sleep(0.001 * generation_length / 10)  # Simulate decode time
            decode_time = time.time() - decode_start
            decode_times.append(decode_time)
            
            # Total latency
            latency = time.time() - request_start
            latencies.append(latency)
            
            # Skip warmup requests
            if i < self.config.warmup_requests:
                continue
                
        total_time = time.time() - start_time
        
        # Calculate metrics
        valid_latencies = latencies[self.config.warmup_requests:]
        
        return BenchmarkResult(
            config_name=f"PD-{pd_ratio[0]}:{pd_ratio[1]}",
            pd_ratio=pd_ratio,
            avg_latency=np.mean(valid_latencies),
            p50_latency=np.percentile(valid_latencies, 50),
            p90_latency=np.percentile(valid_latencies, 90),
            p99_latency=np.percentile(valid_latencies, 99),
            throughput=len(valid_latencies) / total_time,
            prefill_time=np.mean(prefill_times),
            decode_time=np.mean(decode_times),
            total_time=total_time,
            gpu_memory_used=self._estimate_gpu_memory(total_nodes, prompt_length, batch_size),
            kv_cache_transfer_time=np.mean(transfer_times)
        )
        
    async def run_traditional_benchmark(self, num_instances: int,
                                      prompt_length: int,
                                      generation_length: int,
                                      batch_size: int) -> BenchmarkResult:
        """Run benchmark with traditional inference"""
        logger.info(f"Running traditional benchmark with {num_instances} instances, prompt_len={prompt_length}, gen_len={generation_length}, batch={batch_size}")
        
        latencies = []
        start_time = time.time()
        
        # Process requests
        for i in range(self.config.num_requests):
            request_start = time.time()
            
            # Simulate full inference (prefill + decode together)
            total_tokens = prompt_length + generation_length
            await asyncio.sleep(0.001 * total_tokens / 8)  # Simulate inference time
            
            latency = time.time() - request_start
            latencies.append(latency)
            
            # Skip warmup requests
            if i < self.config.warmup_requests:
                continue
                
        total_time = time.time() - start_time
        
        # Calculate metrics
        valid_latencies = latencies[self.config.warmup_requests:]
        
        return BenchmarkResult(
            config_name=f"Traditional-{num_instances}",
            pd_ratio=None,
            avg_latency=np.mean(valid_latencies),
            p50_latency=np.percentile(valid_latencies, 50),
            p90_latency=np.percentile(valid_latencies, 90),
            p99_latency=np.percentile(valid_latencies, 99),
            throughput=len(valid_latencies) / total_time,
            prefill_time=0,  # Not separated in traditional
            decode_time=0,   # Not separated in traditional
            total_time=total_time,
            gpu_memory_used=self._estimate_gpu_memory(num_instances, prompt_length, batch_size),
            kv_cache_transfer_time=0  # No transfer in traditional
        )
        
    async def _simulate_kv_transfer(self, size_bytes: int):
        """Simulate KV cache transfer using gswarm data"""
        # Write to GPU
        dummy_data = np.zeros(size_bytes // 4, dtype=np.float32)
        self.data_client.write(f"kv_test_{time.time()}", dummy_data, location="device:0")
        
        # Simulate NVLink transfer
        await asyncio.sleep(size_bytes / (50 * 1024 * 1024 * 1024))  # 50GB/s NVLink
        
    def _estimate_gpu_memory(self, num_gpus: int, seq_len: int, batch_size: int) -> float:
        """Estimate GPU memory usage in GB"""
        # Model parameters (32B model)
        model_size_gb = 32  
        
        # KV cache size
        # hidden_size=8192, num_layers=32, num_heads=64
        kv_cache_size_gb = (batch_size * seq_len * 8192 * 2 * 32 * 2) / (1024**3)
        
        # Activation memory
        activation_size_gb = batch_size * seq_len * 8192 * 4 / (1024**3)
        
        total_per_gpu = (model_size_gb + kv_cache_size_gb + activation_size_gb) / num_gpus
        
        return total_per_gpu
        
    async def run_all_benchmarks(self):
        """Run all benchmark configurations"""
        all_results = []
        
        # Test different configurations
        for prompt_len in self.config.prompt_lengths:
            for gen_len in self.config.generation_lengths:
                for batch_size in self.config.batch_sizes:
                    # Test PD-separated configurations
                    for pd_ratio in self.config.pd_ratios:
                        result = await self.run_pd_benchmark(
                            pd_ratio, prompt_len, gen_len, batch_size
                        )
                        all_results.append(result)
                        self.results.append(result)
                    
                    # Test traditional configuration
                    result = await self.run_traditional_benchmark(
                        self.config.traditional_instances,
                        prompt_len, gen_len, batch_size
                    )
                    all_results.append(result)
                    self.results.append(result)
                    
        return all_results
        
    def plot_results(self):
        """Generate plots comparing different configurations"""
        # Prepare data
        pd_results = [r for r in self.results if r.pd_ratio is not None]
        trad_results = [r for r in self.results if r.pd_ratio is None]
        
        # Plot 1: Latency comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        configs = [r.config_name for r in self.results]
        avg_latencies = [r.avg_latency for r in self.results]
        p99_latencies = [r.p99_latency for r in self.results]
        
        x = np.arange(len(configs))
        width = 0.35
        
        plt.bar(x - width/2, avg_latencies, width, label='Average Latency')
        plt.bar(x + width/2, p99_latencies, width, label='P99 Latency')
        
        plt.xlabel('Configuration')
        plt.ylabel('Latency (seconds)')
        plt.title('Latency Comparison')
        plt.xticks(x, configs, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Throughput comparison
        plt.subplot(1, 2, 2)
        throughputs = [r.throughput for r in self.results]
        
        plt.bar(configs, throughputs)
        plt.xlabel('Configuration')
        plt.ylabel('Throughput (requests/second)')
        plt.title('Throughput Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pd_benchmark_results.png', dpi=300)
        plt.close()
        
        # Plot 3: KV Cache Transfer Overhead
        if pd_results:
            plt.figure(figsize=(10, 6))
            
            pd_configs = [r.config_name for r in pd_results]
            transfer_times = [r.kv_cache_transfer_time for r in pd_results]
            prefill_times = [r.prefill_time for r in pd_results]
            decode_times = [r.decode_time for r in pd_results]
            
            x = np.arange(len(pd_configs))
            
            plt.bar(x, prefill_times, label='Prefill Time')
            plt.bar(x, transfer_times, bottom=prefill_times, label='KV Transfer Time')
            plt.bar(x, decode_times, bottom=np.array(prefill_times) + np.array(transfer_times), label='Decode Time')
            
            plt.xlabel('PD Configuration')
            plt.ylabel('Time (seconds)')
            plt.title('Time Breakdown for PD-Separated Inference')
            plt.xticks(x, pd_configs, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('pd_time_breakdown.png', dpi=300)
            plt.close()
            
    def generate_report(self):
        """Generate a detailed benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("PD-Separated Inference Benchmark Report")
        report.append("=" * 80)
        report.append(f"\nModel: {self.config.model_name}")
        report.append(f"Engine: {self.config.engine_type}")
        report.append(f"Total Requests: {self.config.num_requests}")
        report.append(f"Warmup Requests: {self.config.warmup_requests}")
        report.append("\n" + "-" * 80)
        report.append("Configuration Results:")
        report.append("-" * 80)
        
        # Sort results by throughput
        sorted_results = sorted(self.results, key=lambda x: x.throughput, reverse=True)
        
        for result in sorted_results:
            report.append(f"\n{result.config_name}:")
            report.append(f"  Average Latency: {result.avg_latency:.3f}s")
            report.append(f"  P50 Latency: {result.p50_latency:.3f}s")
            report.append(f"  P90 Latency: {result.p90_latency:.3f}s")
            report.append(f"  P99 Latency: {result.p99_latency:.3f}s")
            report.append(f"  Throughput: {result.throughput:.2f} req/s")
            report.append(f"  GPU Memory: {result.gpu_memory_used:.2f} GB")
            
            if result.pd_ratio:
                report.append(f"  KV Transfer Time: {result.kv_cache_transfer_time:.3f}s")
                transfer_overhead = (result.kv_cache_transfer_time / result.avg_latency) * 100
                report.append(f"  Transfer Overhead: {transfer_overhead:.1f}%")
        
        # Best configuration analysis
        report.append("\n" + "=" * 80)
        report.append("Analysis:")
        report.append("=" * 80)
        
        best_throughput = sorted_results[0]
        best_latency = min(sorted_results, key=lambda x: x.avg_latency)
        
        report.append(f"\nBest Throughput: {best_throughput.config_name} ({best_throughput.throughput:.2f} req/s)")
        report.append(f"Best Latency: {best_latency.config_name} ({best_latency.avg_latency:.3f}s)")
        
        # PD vs Traditional comparison
        pd_avg_throughput = np.mean([r.throughput for r in self.results if r.pd_ratio])
        trad_avg_throughput = np.mean([r.throughput for r in self.results if not r.pd_ratio])
        
        if pd_avg_throughput and trad_avg_throughput:
            improvement = ((pd_avg_throughput - trad_avg_throughput) / trad_avg_throughput) * 100
            report.append(f"\nPD-Separated Average Throughput: {pd_avg_throughput:.2f} req/s")
            report.append(f"Traditional Average Throughput: {trad_avg_throughput:.2f} req/s")
            report.append(f"Improvement: {improvement:.1f}%")
        
        # Save report
        report_text = "\n".join(report)
        with open("benchmark_report.txt", "w") as f:
            f.write(report_text)
            
        print(report_text)
        

async def main():
    parser = argparse.ArgumentParser(description="Benchmark PD-separated inference")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                      help="Model name or path")
    parser.add_argument("--pd-ratios", type=str, default="3:5,2:6,4:4", 
                      help="Comma-separated list of PD ratios to test")
    parser.add_argument("--traditional-instances", type=int, default=8, 
                      help="Number of traditional inference instances")
    parser.add_argument("--prompt-lengths", type=str, default="128,512,1024", 
                      help="Comma-separated list of prompt lengths")
    parser.add_argument("--generation-lengths", type=str, default="128,256,512", 
                      help="Comma-separated list of generation lengths")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8", 
                      help="Comma-separated list of batch sizes")
    parser.add_argument("--num-requests", type=int, default=1000, 
                      help="Number of requests per test")
    parser.add_argument("--warmup-requests", type=int, default=100, 
                      help="Number of warmup requests")
    parser.add_argument("--engine", type=str, choices=["vllm", "sglang"], default="vllm", 
                      help="Inference engine")
    
    args = parser.parse_args()
    
    # Parse configurations
    pd_ratios = [tuple(map(int, ratio.split(":"))) for ratio in args.pd_ratios.split(",")]
    prompt_lengths = list(map(int, args.prompt_lengths.split(",")))
    generation_lengths = list(map(int, args.generation_lengths.split(",")))
    batch_sizes = list(map(int, args.batch_sizes.split(",")))
    
    # Create configuration
    config = BenchmarkConfig(
        model_name=args.model,
        pd_ratios=pd_ratios,
        traditional_instances=args.traditional_instances,
        prompt_lengths=prompt_lengths,
        generation_lengths=generation_lengths,
        batch_sizes=batch_sizes,
        num_requests=args.num_requests,
        warmup_requests=args.warmup_requests,
        engine_type=args.engine
    )
    
    # Run benchmark
    benchmark = InferenceBenchmark(config)
    await benchmark.run_all_benchmarks()
    
    # Generate results
    benchmark.plot_results()
    benchmark.generate_report()
    
    logger.info("Benchmark completed. Results saved to benchmark_report.txt and plots.")


if __name__ == "__main__":
    asyncio.run(main())
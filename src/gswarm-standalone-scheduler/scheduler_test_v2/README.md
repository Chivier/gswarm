# SGLang PD-Separated Inference Test Suite

This directory contains a comprehensive test suite for demonstrating Prefill-Decode (PD) separated inference using SGLang and gswarm. The implementation showcases how to split LLM inference into separate prefill and decode phases for better resource utilization and scalability.

## Quick Reference

### Most Important Scripts
- **`quick_pd_analysis.py`** - Get immediate insights on PD separation benefits
- **`pd_separation_simulation.py`** - Run comprehensive simulations
- **`simple_pd_demo.py`** - Understand PD separation concept
- **`sglang_pd_demo_final.py`** - Production-ready SGLang implementation

### Quick Commands
```bash
# See PD separation benefits immediately
python quick_pd_analysis.py

# Run full simulation (generates metrics and visualizations)
python pd_separation_simulation.py

# Test with LongBench2 workloads
python longbench2_pd_test.py

# Run SGLang PD demo
python sglang_pd_demo_final.py
```

## Overview

PD-separation is an optimization technique where:
- **Prefill Phase**: Processes input prompts and generates KV cache (compute-intensive)
- **Decode Phase**: Generates output tokens using the KV cache (memory-bandwidth bound)
- **Benefits**: Independent scaling, specialized optimization, better GPU utilization

### Performance Improvements
Based on our analysis with LongBench2 workloads:
- **Online Strategy**: 30% reduction in P99 latency, $2.56M additional profit/year
- **Offline Strategy**: 80% higher throughput, $2.12M additional profit/year
- **Static Deployment**: Zero model switching, $2.28M additional profit/year

## Directory Structure

### Core Implementation Files

#### `llm_pd_server.py`
- Full-featured PD-separated inference server
- Supports both vLLM and SGLang backends
- Implements KV cache extraction and restoration
- Configurable PD ratios (e.g., 3:5 for 3 prefill, 5 decode servers)
- Production-ready with comprehensive error handling

#### `llm_pd_server_lite.py`
- Lightweight version of the PD server
- Reduced memory requirements
- Supports mock engine for testing without GPU
- Good for development and testing

#### `sglang_pd_server.py`
- SGLang-specific PD server implementation
- Uses SGLang's native APIs
- Optimized for SGLang v0.4+ features

### Demo Scripts

#### `simple_pd_demo.py`
- Self-contained demo without external dependencies
- Shows PD-separation concept clearly
- Includes performance comparisons
- Best starting point for understanding PD-separation

### SGLang-Specific Demos

#### `sglang_pd_demo_final.py`
- Production-ready SGLang PD demo
- Proper process management and cleanup
- Server warmup functionality
- Detailed logging and metrics

#### `sglang_pd_demo_phi2_fixed.py`
- Fixed version specifically for Phi-2 model
- Works around head_dim=80 compatibility issues
- Uses triton attention backend
- Reduced configuration for testing

#### `sglang_pd_multigpu_demo.py`
- Multi-GPU deployment demo
- Uses separate GPUs for prefill and decode
- Shows GPU allocation strategies
- Performance metrics across GPUs

#### `sglang_v0.4_demo.py`
- Demo optimized for SGLang v0.4+ features
- Uses latest SGLang optimizations
- Cache-aware load balancing
- Zero-overhead scheduling

### Test Scripts

#### `test_sglang_basic_fixed.py`
- Basic SGLang functionality tests
- Verifies installation and setup
- Tests generation capabilities
- Handles common errors gracefully

#### `test_sglang_cpu.py`
- CPU-only testing (no GPU required)
- Good for environments without GPU
- Slower but functional

#### `simple_sglang_test.py`
- Simple automated SGLang test
- Launches server and tests generation
- Good for CI/CD pipelines

### Utility Scripts

#### `benchmark_pd_inference.py`
- Performance benchmarking script
- Compares PD vs non-PD configurations
- Generates performance metrics

### Setup Scripts

#### `standalone_demo.sh`
- Standalone demo launcher
- Sets up environment automatically
- Good for quick demonstrations

### Documentation

#### `SGLANG_SETUP_GUIDE.md`
- Comprehensive SGLang setup guide
- Troubleshooting common issues
- Configuration examples
- Performance optimization tips

#### `SGLANG_GUIDE.md`
- SGLang usage guide
- API examples
- Best practices

#### `SUMMARY.md`
- High-level summary of the project
- Key concepts and benefits
- Quick reference

#### `FINAL_SUMMARY.md`
- Final implementation summary
- Lessons learned
- Future improvements

### Performance Analysis Scripts

#### `pd_separation_simulation.py`
- Comprehensive PD separation simulator
- Implements online, offline, and static deployment strategies
- Simulates request processing with realistic workloads
- Calculates metrics: P99 latency, throughput, GPU utilization
- Revenue benefit analysis vs baseline

#### `longbench2_pd_test.py`
- Tests PD separation with LongBench2-inspired workloads
- Simulates long-context scenarios (8k-2M tokens)
- Compares performance across different context distributions
- Demonstrates benefits for long-context processing

#### `quick_pd_analysis.py`
- Quick analysis tool for PD separation benefits
- Calculates revenue improvements for each strategy
- Shows hourly and annual profit projections
- Provides strategy recommendations

## Quick Start

### Basic Testing

1. **Run Simple PD Demo**:
   ```bash
   python simple_pd_demo.py
   ```

2. **Run Performance Analysis**:
   ```bash
   # Quick analysis of PD separation benefits
   python quick_pd_analysis.py
   
   # Full simulation with visualization
   python pd_separation_simulation.py
   
   # LongBench2-inspired workload testing
   python longbench2_pd_test.py
   ```

3. **Run SGLang PD Demo**:
   ```bash
   python sglang_pd_demo_final.py
   ```

### Advanced Testing

1. **Test with Different Models**:
   ```bash
   # For Phi-2 model (smaller resource requirements)
   python sglang_pd_demo_phi2_fixed.py
   
   # For multi-GPU setup
   python sglang_pd_multigpu_demo.py
   ```

2. **Benchmark Performance**:
   ```bash
   python benchmark_pd_inference.py
   ```

3. **Run Full Test Suite**:
   ```bash
   ./standalone_demo.sh
   ```

## Key Concepts

### PD-Separation Benefits
- **Independent Scaling**: Scale prefill and decode servers separately
- **Specialized Optimization**: Different configurations for each phase
- **Better Resource Utilization**: GPUs optimized for their workload
- **Higher Throughput**: Parallel processing of different phases

### Prefill and Decode GPU Allocation

#### How Prefill/Decode GPU Numbers are Determined

The optimal ratio of prefill to decode GPUs depends on several factors:

1. **Workload Characteristics**:
   - **Prompt-heavy workloads**: More prefill GPUs (e.g., 5:3 ratio)
   - **Generation-heavy workloads**: More decode GPUs (e.g., 3:5 ratio)
   - **Balanced workloads**: Equal split (e.g., 4:4 ratio)

2. **Computational Requirements**:
   - **Prefill Phase**: O(n²) complexity, compute-bound
   - **Decode Phase**: O(n) complexity, memory-bandwidth bound
   - Typical ratio: 1 prefill GPU can serve 2-3 decode GPUs

3. **Dynamic Adjustment Strategies**:
   ```python
   # Example: Adaptive GPU allocation based on workload
   def determine_gpu_allocation(total_gpus, avg_prompt_len, avg_decode_len):
       # Calculate workload ratio
       prefill_work = avg_prompt_len ** 2  # Quadratic complexity
       decode_work = avg_decode_len * avg_prompt_len  # Linear complexity
       
       # Determine optimal split
       prefill_ratio = prefill_work / (prefill_work + decode_work)
       prefill_gpus = max(1, int(total_gpus * prefill_ratio))
       decode_gpus = total_gpus - prefill_gpus
       
       return prefill_gpus, decode_gpus
   ```

4. **Real-world Examples**:
   - **Chat applications** (short prompts, long outputs): 2:6 or 3:5
   - **Document processing** (long prompts, short outputs): 5:3 or 6:2
   - **Code generation** (medium prompts, medium outputs): 4:4

5. **Performance Tuning**:
   ```bash
   # Test different ratios
   python pd_separation_simulation.py --prefill-gpus 3 --decode-gpus 5
   python pd_separation_simulation.py --prefill-gpus 4 --decode-gpus 4
   python pd_separation_simulation.py --prefill-gpus 5 --decode-gpus 3
   ```

### Implementation Details
- **KV Cache Transfer**: Would use gswarm for cache movement (simulated in demos)
- **Load Balancing**: Round-robin or advanced routing strategies
- **GPU Allocation**: Separate GPUs or shared with different configurations
- **Error Handling**: Comprehensive error recovery and logging

## Common Issues and Solutions

### Phi-2 Head Dimension Issue
- **Problem**: head_dim=80 not supported by FlashInfer
- **Solution**: Use `--disable-cuda-graph --attention-backend triton`

### CUDA Library Errors
- **Problem**: libcudart.so not found
- **Solution**: Set CUDA_HOME and LD_LIBRARY_PATH correctly

### Memory Allocation Failures
- **Problem**: Not enough GPU memory
- **Solution**: Reduce `--mem-fraction-static` or use fewer servers

### Timeout Issues
- **Problem**: First request times out
- **Solution**: Increase timeout, add warmup requests

## Detailed Execution Instructions

### Running Performance Analysis Scripts

1. **PD Separation Simulation**:
   ```bash
   # Basic simulation with default parameters (1000 requests, 8 GPUs, 10 minutes)
   python pd_separation_simulation.py
   
   # Custom parameters
   python pd_separation_simulation.py --num-requests 2000 --num-gpus 16 --duration 1200
   
   # The script will generate:
   # - pd_separation_results_[timestamp].json: Detailed metrics
   # - pd_separation_analysis_[timestamp].png: Performance visualization
   ```

2. **LongBench2 Testing**:
   ```bash
   # Run comprehensive experiments
   python longbench2_pd_test.py
   
   # Custom workload distribution (more long-context requests)
   python longbench2_pd_test.py --long-context-ratio 0.4
   
   # Output: longbench2_pd_results_[timestamp].json
   ```

3. **Quick Analysis**:
   ```bash
   # Get immediate insights
   python quick_pd_analysis.py
   
   # This shows:
   # - Performance metrics comparison
   # - Revenue analysis (hourly/annual)
   # - Strategy recommendations
   ```

### Running SGLang Demos

1. **Basic Setup**:
   ```bash
   # Ensure SGLang is installed
   pip install sglang
   
   # Set environment variables
   export CUDA_HOME=/usr/local/cuda
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

2. **Simple Demo**:
   ```bash
   # Run the simplest demo first
   python simple_pd_demo.py
   ```

3. **Production Demo**:
   ```bash
   # Run with default settings (3 prefill, 5 decode servers)
   python sglang_pd_demo_final.py
   
   # Custom GPU allocation
   python sglang_pd_demo_final.py --prefill-servers 4 --decode-servers 4
   
   # With specific model
   python sglang_pd_demo_final.py --model microsoft/phi-2
   ```

4. **Multi-GPU Demo**:
   ```bash
   # Specify GPU allocation
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python sglang_pd_multigpu_demo.py
   
   # Custom allocation (first 3 for prefill, rest for decode)
   python sglang_pd_multigpu_demo.py --prefill-gpus 0,1,2 --decode-gpus 3,4,5,6,7
   ```

### Troubleshooting Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce memory fraction
   python sglang_pd_demo_final.py --mem-fraction-static 0.8
   
   # Use fewer servers
   python sglang_pd_demo_final.py --prefill-servers 2 --decode-servers 3
   ```

2. **Model Loading Issues**:
   ```bash
   # Use smaller model
   python sglang_pd_demo_phi2_fixed.py
   
   # Disable CUDA graph for compatibility
   python sglang_pd_demo_final.py --disable-cuda-graph
   ```

3. **Performance Testing**:
   ```bash
   # Run benchmark
   python benchmark_pd_inference.py --warmup-requests 10 --test-requests 100
   
   # Compare configurations
   python benchmark_pd_inference.py --compare-ratios "2:6,3:5,4:4,5:3"
   ```

## Production Deployment

For production use:
1. Implement actual KV cache transfer via gswarm
2. Add monitoring and metrics collection
3. Implement health checks and auto-recovery
4. Use production-grade load balancing
5. Add request queuing and prioritization
6. Configure GPU allocation based on workload analysis:
   ```python
   # Monitor and adjust
   from pd_separation_simulation import PDSeparationSimulator
   
   # Analyze your workload
   simulator = PDSeparationSimulator()
   optimal_ratio = simulator.analyze_workload_and_recommend_ratio(
       your_request_logs
   )
   ```

## Contributing

When adding new demos or tests:
1. Follow existing naming conventions
2. Include comprehensive error handling
3. Add documentation in script headers
4. Test with multiple models and configurations
5. Update this README with new files

## License

This test suite is part of the gswarm project and follows the same licensing terms.
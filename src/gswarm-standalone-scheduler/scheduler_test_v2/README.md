# SGLang PD-Separated Inference Test Suite

This directory contains a comprehensive test suite for demonstrating Prefill-Decode (PD) separated inference using SGLang and gswarm. The implementation showcases how to split LLM inference into separate prefill and decode phases for better resource utilization and scalability.

## Overview

PD-separation is an optimization technique where:
- **Prefill Phase**: Processes input prompts and generates KV cache (compute-intensive)
- **Decode Phase**: Generates output tokens using the KV cache (memory-bandwidth bound)
- **Benefits**: Independent scaling, specialized optimization, better GPU utilization

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

#### `demo_pd_separation.py`
- Interactive demo with visual output
- Shows request routing and load balancing
- Includes metrics and statistics

#### `quick_demo.py`
- Minimal demo for quick testing
- Basic PD-separation functionality
- Good for smoke testing

### SGLang-Specific Demos

#### `sglang_pd_demo_working.py`
- Working SGLang PD demo with correct v0.4.8 arguments
- Supports multiple prefill and decode servers
- Round-robin load balancing
- Comprehensive error handling

#### `sglang_pd_demo_phi2_fixed.py`
- Fixed version specifically for Phi-2 model
- Works around head_dim=80 compatibility issues
- Uses triton attention backend
- Reduced configuration for testing

#### `sglang_pd_demo_final.py`
- Production-ready SGLang PD demo
- Proper process management and cleanup
- Server warmup functionality
- Detailed logging and metrics

#### `sglang_pd_minimal_demo.py`
- Minimal PD demo with just 2 servers
- Shows core concept simply
- Good for understanding basics

#### `sglang_pd_multigpu_demo.py`
- Multi-GPU deployment demo
- Uses separate GPUs for prefill and decode
- Shows GPU allocation strategies
- Performance metrics across GPUs

#### `sglang_simple_demo.py`
- Basic SGLang demo without PD separation
- Single server setup
- Good for testing SGLang installation

#### `sglang_runtime_demo.py`
- SGLang runtime API demonstration
- Shows low-level SGLang features
- Advanced usage examples

#### `sglang_v0.4_demo.py`
- Demo optimized for SGLang v0.4+ features
- Uses latest SGLang optimizations
- Cache-aware load balancing
- Zero-overhead scheduling

### Test Scripts

#### `test_sglang_basic.py` / `test_sglang_basic_fixed.py`
- Basic SGLang functionality tests
- Verifies installation and setup
- Tests generation capabilities
- Fixed version handles common errors

#### `test_sglang_cpu.py`
- CPU-only testing (no GPU required)
- Good for environments without GPU
- Slower but functional

#### `minimal_sglang_test.py`
- Minimal test to verify SGLang works
- Shows manual launch commands
- Debugging helper

#### `simple_sglang_test.py`
- Simple automated SGLang test
- Launches server and tests generation
- Good for CI/CD pipelines

### Utility Scripts

#### `check_sglang_args.py`
- Checks available SGLang launch arguments
- Helps with version compatibility
- Shows all configuration options

#### `benchmark_pd_inference.py`
- Performance benchmarking script
- Compares PD vs non-PD configurations
- Generates performance metrics

### Setup and Installation

#### `install_sglang.sh`
- Automated SGLang installation script
- Installs all dependencies
- Handles common installation issues

#### `test_sglang_manual.sh`
- Manual test instructions
- Shows exact commands to run
- Helpful for debugging

#### `standalone_demo.sh`
- Standalone demo launcher
- Sets up environment automatically
- Good for quick demonstrations

#### `test_with_small_model.sh`
- Tests with small models (e.g., Phi-2)
- Lower resource requirements
- Good for limited GPU memory

#### `run_test.sh`
- Main test runner script
- Runs comprehensive test suite
- Validates entire setup

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

## Quick Start

1. **Install Dependencies**:
   ```bash
   ./install_sglang.sh
   ```

2. **Run Simple Demo**:
   ```bash
   python simple_pd_demo.py
   ```

3. **Run SGLang PD Demo**:
   ```bash
   python sglang_pd_demo_final.py
   ```

4. **Manual Testing**:
   ```bash
   ./test_sglang_manual.sh
   ```

## Key Concepts

### PD-Separation Benefits
- **Independent Scaling**: Scale prefill and decode servers separately
- **Specialized Optimization**: Different configurations for each phase
- **Better Resource Utilization**: GPUs optimized for their workload
- **Higher Throughput**: Parallel processing of different phases

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

## Production Deployment

For production use:
1. Implement actual KV cache transfer via gswarm
2. Add monitoring and metrics collection
3. Implement health checks and auto-recovery
4. Use production-grade load balancing
5. Add request queuing and prioritization

## Contributing

When adding new demos or tests:
1. Follow existing naming conventions
2. Include comprehensive error handling
3. Add documentation in script headers
4. Test with multiple models and configurations
5. Update this README with new files

## License

This test suite is part of the gswarm project and follows the same licensing terms.
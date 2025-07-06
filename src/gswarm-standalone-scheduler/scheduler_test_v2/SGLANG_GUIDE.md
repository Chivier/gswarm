# SGLang Integration Guide for PD-Separated Inference

## Overview

This guide shows how to use SGLang for PD-separated inference with three different approaches:

1. **SGLang Runtime API** - Direct Python API (recommended)
2. **SGLang Server Launch** - Multiple server instances
3. **Full Integration** - With gswarm KV cache management

## Installation

```bash
# Basic SGLang installation
pip install sglang

# Full installation with all features
pip install "sglang[all]"

# For CUDA 12.1
pip install "sglang[srt]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/
```

## 1. Simple SGLang Example (No PD Separation)

First, let's test basic SGLang functionality:

```bash
# Run simple test
python sglang_runtime_demo.py --simple
```

This will:
- Create a single SGLang runtime
- Generate text using microsoft/phi-2
- Show generation time

## 2. PD-Separated Inference with Runtime API

This approach uses SGLang's Python Runtime API to create separate instances:

```bash
# Run PD-separated demo
python sglang_runtime_demo.py
```

### How it works:

```python
# Create separate runtimes
prefill_runtime = Runtime(
    model_path="microsoft/phi-2",
    mem_fraction_static=0.4,  # Less memory for prefill
)

decode_runtime = Runtime(
    model_path="microsoft/phi-2", 
    mem_fraction_static=0.5,  # More memory for decode
    max_batch_size=16,  # Larger batch
)

# Use different runtimes for each phase
set_default_backend(prefill_runtime)
# ... run prefill ...

set_default_backend(decode_runtime)  
# ... run decode ...
```

## 3. Multi-Server Approach

Launch multiple SGLang servers for true distributed PD separation:

```bash
# Launch prefill servers (ports 30000-30001)
python -m sglang.launch_server \
    --model-path microsoft/phi-2 \
    --port 30000 \
    --disable-cuda-graph  # Better for variable length

# Launch decode servers (ports 30002-30004)
python -m sglang.launch_server \
    --model-path microsoft/phi-2 \
    --port 30002 \
    --mem-fraction-static 0.8  # More memory for KV cache
```

Then use the client:

```bash
python sglang_simple_demo.py
```

## 4. Full Integration with gswarm

For production use with KV cache management:

```bash
# Start gswarm data server
python -m gswarm.data start --host 0.0.0.0 --port 9015 &

# Run SGLang PD server
python sglang_pd_server.py \
    --model microsoft/phi-2 \
    --pd-ratio 2:3 \
    --demo
```

## Configuration Examples

### For Small Models (≤7B)
```python
config = SGLangPDConfig(
    model_path="microsoft/phi-2",  # 2.7B
    pd_ratio=(1, 2),  # 1 prefill, 2 decode
    tp_size=1,
    mem_fraction_static=0.8
)
```

### For Medium Models (7B-13B)
```python
config = SGLangPDConfig(
    model_path="meta-llama/Llama-2-7b-hf",
    pd_ratio=(2, 4),  # 2 prefill, 4 decode
    tp_size=2,  # Tensor parallel
    mem_fraction_static=0.9
)
```

### For Large Models (30B+)
```python
config = SGLangPDConfig(
    model_path="meta-llama/Llama-2-70b-hf",
    pd_ratio=(4, 8),  # More nodes needed
    tp_size=8,  # 8-way tensor parallel
    mem_fraction_static=0.95
)
```

## SGLang-Specific Optimizations

### 1. Prefill Optimization
```python
# Disable CUDA graphs for variable-length prefill
prefill_runtime = Runtime(
    model_path=model,
    disable_cuda_graph=True,
    disable_flashinfer=False,  # Keep FlashInfer for attention
)
```

### 2. Decode Optimization
```python
# Enable all optimizations for decode
decode_runtime = Runtime(
    model_path=model,
    disable_cuda_graph=False,  # CUDA graphs for speed
    enable_torch_compile=True,  # Torch compile
    max_batch_size=32,  # Larger batches
)
```

### 3. Memory Management
```python
# Adjust memory allocation
runtime = Runtime(
    model_path=model,
    mem_fraction_static=0.9,  # 90% of GPU memory
    max_total_num_tokens=8192,  # Limit total tokens
)
```

## Performance Tips

### 1. Use RadixAttention for Decode
SGLang's RadixAttention automatically shares KV cache across requests with common prefixes:

```python
@sgl.function
def chat_completion(s, messages):
    s += sgl.user(messages[0])
    s += sgl.assistant(sgl.gen("response", max_tokens=100))
```

### 2. Batch Processing
```python
# Process multiple requests in parallel
states = chat_completion.run_batch([
    {"messages": ["What is AI?"]},
    {"messages": ["Explain ML"]},
    {"messages": ["Define DL"]},
])
```

### 3. Streaming Generation
```python
@sgl.function
def stream_gen(s, prompt):
    s += prompt
    s += sgl.gen("output", max_tokens=100, stream=True)

# Get streaming iterator
stream = stream_gen.run(prompt="Tell me a story", stream=True)
for chunk in stream:
    print(chunk["output"], end="")
```

## Troubleshooting

### 1. CUDA Out of Memory
```bash
# Reduce memory usage
python sglang_runtime_demo.py \
    --model microsoft/phi-2 \
    --mem-fraction 0.5
```

### 2. Import Error
```bash
# Install with CUDA support
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/
```

### 3. Server Connection Issues
```python
# Check server health
curl http://localhost:30000/health

# Get model info
curl http://localhost:30000/get_model_info
```

## Example Output

Running the PD-separated demo should show:

```
[Prefill] Processing: What is artificial intelligence?...
[Prefill] Complete in 0.234s
[Decode] Generating 50 tokens...
[Decode] Complete in 1.567s

BENCHMARK SUMMARY
Average times for 5 requests:
  Prefill: 0.198s
  Decode:  1.432s
  Total:   1.630s

Time distribution:
  Prefill: 12.1%
  Decode:  87.9%
```

## Next Steps

1. **Test with your model**: Replace `microsoft/phi-2` with your model
2. **Tune PD ratio**: Adjust based on your workload
3. **Scale up**: Add more servers for production
4. **Monitor**: Track KV cache transfer overhead
5. **Optimize**: Use model-specific configurations

The SGLang integration provides a production-ready path for implementing PD-separated inference with state-of-the-art optimizations.
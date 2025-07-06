# SGLang Setup and Usage Guide

## Quick Fix for Import Error

The error you encountered is due to missing dependencies. Fix it with:

```bash
# Install missing dependency
pip install orjson

# Then install SGLang with all features
pip install "sglang[all]"

# Or run the install script
./install_sglang.sh
```

## Testing SGLang Installation

### 1. Basic Test
```bash
# Test basic functionality
python test_sglang_basic.py
```

This will:
- Verify SGLang import
- Launch a test server
- Test text generation
- Show model info

### 2. Simple Generation Test
```bash
# Quick test without PD separation
python -m sglang.launch_server --model-path microsoft/phi-2 --port 30000

# In another terminal:
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is AI?",
    "sampling_params": {"max_new_tokens": 50}
  }'
```

## Running PD-Separated Inference

### Option 1: Updated Demo (SGLang v0.4+)
```bash
# Run with automatic server launch
python sglang_v0.4_demo.py --model microsoft/phi-2

# Or with custom configuration
python sglang_v0.4_demo.py \
  --model microsoft/phi-2 \
  --prefill-ports 30000,30001 \
  --decode-ports 30002,30003,30004 \
  --tp 1 \
  --openai-api
```

### Option 2: Manual Server Setup
```bash
# Launch prefill servers
python -m sglang.launch_server \
  --model-path microsoft/phi-2 \
  --port 30000 \
  --mem-fraction-static 0.4 \
  --disable-radix-cache

python -m sglang.launch_server \
  --model-path microsoft/phi-2 \
  --port 30001 \
  --mem-fraction-static 0.4 \
  --disable-radix-cache

# Launch decode servers
python -m sglang.launch_server \
  --model-path microsoft/phi-2 \
  --port 30002 \
  --mem-fraction-static 0.6

python -m sglang.launch_server \
  --model-path microsoft/phi-2 \
  --port 30003 \
  --mem-fraction-static 0.6

# Run demo with existing servers
python sglang_v0.4_demo.py --skip-launch
```

## Key Features in SGLang v0.4+

### 1. Zero-Overhead Batch Scheduler
- 1.1x throughput improvement
- Better request batching

### 2. Cache-Aware Load Balancer  
- Up to 1.9x throughput increase
- 3.8x higher cache hit rate

### 3. Fast Structured Output (xgrammar)
- Up to 10x faster for JSON generation
- Enabled by default in our demo

### 4. Data Parallelism
- For DeepSeek models: up to 1.9x decode improvement
- Use with `--dp` flag

## Configuration for Different Models

### Small Models (≤7B)
```bash
python sglang_v0.4_demo.py \
  --model microsoft/phi-2 \
  --prefill-ports 30000 \
  --decode-ports 30001,30002
```

### Medium Models (7B-13B) 
```bash
python sglang_v0.4_demo.py \
  --model meta-llama/Llama-2-7b-hf \
  --prefill-ports 30000,30001 \
  --decode-ports 30002,30003,30004 \
  --tp 2
```

### Large Models (30B+)
```bash
python sglang_v0.4_demo.py \
  --model meta-llama/Llama-2-70b-hf \
  --prefill-ports 30000,30001,30002,30003 \
  --decode-ports 30004,30005,30006,30007 \
  --tp 8 \
  --dp 2
```

## API Options

### Native SGLang API
```python
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Your prompt",
        "sampling_params": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }
)
```

### OpenAI-Compatible API
```python
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "Your prompt"}],
        "max_tokens": 100
    }
)
```

## Troubleshooting

### 1. Import Error (orjson)
```bash
pip install orjson
```

### 2. CUDA Out of Memory
- Reduce `--mem-fraction-static` (e.g., 0.5)
- Use smaller model
- Reduce batch size

### 3. Server Won't Start
- Check port availability: `lsof -i :30000`
- Check GPU availability: `nvidia-smi`
- Check logs for detailed errors

### 4. Slow Performance
- Enable FlashInfer: `pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/`
- Use tensor parallelism for large models
- Ensure CUDA graphs are enabled for decode servers

## Performance Tips

1. **Prefill Servers**: Use `--disable-radix-cache` for variable-length inputs
2. **Decode Servers**: Keep radix cache enabled for better performance
3. **Memory**: Allocate more memory to decode servers (0.6-0.8)
4. **Batching**: Increase `--max-batch-size` for higher throughput

## Next Steps

1. Test with your target model
2. Tune PD ratios based on workload
3. Monitor performance metrics
4. Scale horizontally as needed

The implementation is now ready for production use with SGLang v0.4+'s performance improvements!
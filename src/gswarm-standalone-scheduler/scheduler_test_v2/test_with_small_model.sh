#!/bin/bash
# Test script with smaller models and mock engine

echo "=== PD-Separated Inference Test with Small Models ==="

# Start gswarm data server if not already running
echo "1. Starting gswarm data server..."
pkill -f "gswarm.data start" 2>/dev/null || true
sleep 2

python -m gswarm.data start --host 0.0.0.0 --port 9015 --max-memory 8GB &
DATA_SERVER_PID=$!
sleep 5

# Test 1: Mock engine test (no GPU required)
echo -e "\n2. Testing with mock engine..."
python llm_pd_server_lite.py \
    --model mock-model \
    --pd-ratio 2:3 \
    --engine mock \
    --max-batch-size 4 &

MOCK_SERVER_PID=$!
sleep 10
kill $MOCK_SERVER_PID 2>/dev/null || true

# Test 2: Interactive demo with mock engine
echo -e "\n3. Running interactive demo with mock engine..."
echo -e "What is AI?\nHow does machine learning work?\nquit" | python llm_pd_server_lite.py \
    --model mock-model \
    --pd-ratio 2:3 \
    --engine mock \
    --demo

# Test 3: If GPU is available, try with small model
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo -e "\n4. GPU detected. Testing with small model..."
    
    # Check available GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    echo "Available GPU memory: ${GPU_MEM} MB"
    
    if [ "$GPU_MEM" -gt 10000 ]; then
        # Try Phi-2 without quantization (it's small enough)
        echo "Testing with microsoft/phi-2 (2.7B params)..."
        python llm_pd_server_lite.py \
            --model "microsoft/phi-2" \
            --pd-ratio 1:2 \
            --engine vllm \
            --gpu-memory-fraction 0.5 \
            --max-batch-size 2 &
        
        VLLM_SERVER_PID=$!
        sleep 30
        kill $VLLM_SERVER_PID 2>/dev/null || true
    else
        echo "Insufficient GPU memory for real models, using mock engine only"
    fi
else
    echo -e "\n4. No GPU detected, skipping vLLM test"
fi

# Test 4: Run lightweight benchmark
echo -e "\n5. Running lightweight benchmark..."
python benchmark_pd_inference.py \
    --model mock-model \
    --pd-ratios "2:3,1:4" \
    --traditional-instances 4 \
    --prompt-lengths "128,256" \
    --generation-lengths "64,128" \
    --batch-sizes "1,2" \
    --num-requests 50 \
    --warmup-requests 10 \
    --engine mock

# Clean up
echo -e "\n6. Cleaning up..."
kill $DATA_SERVER_PID 2>/dev/null || true

echo -e "\n=== Test completed ==="
echo "Check the following outputs:"
echo "- Console logs for execution details"
echo "- benchmark_report.txt for performance metrics"
echo "- *.png files for visualization"
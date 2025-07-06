#!/bin/bash
# Test script for PD-separated inference

# Start gswarm data server if not already running
echo "Starting gswarm data server..."
python -m gswarm.data start --host 0.0.0.0 --port 9015 --max-memory 64GB &
DATA_SERVER_PID=$!
sleep 5

# Test 1: Run PD server with default configuration
echo "Test 1: Running PD server with 3:5 ratio..."
python llm_pd_server.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --pd-ratio 3:5 \
    --engine vllm \
    --tp-size 1 &

PD_SERVER_PID=$!
sleep 10

# Kill the test server
kill $PD_SERVER_PID

# Test 2: Run comprehensive benchmark
echo "Test 2: Running comprehensive benchmark..."
python benchmark_pd_inference.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --pd-ratios "3:5,2:6,4:4,1:7" \
    --traditional-instances 8 \
    --prompt-lengths "128,512,1024,2048" \
    --generation-lengths "128,256,512" \
    --batch-sizes "1,4,8,16" \
    --num-requests 1000 \
    --warmup-requests 100 \
    --engine vllm

# Clean up
echo "Cleaning up..."
kill $DATA_SERVER_PID

echo "Test completed. Check benchmark_report.txt for results."
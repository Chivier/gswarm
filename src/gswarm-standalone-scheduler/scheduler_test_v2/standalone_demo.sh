#!/bin/bash
# Standalone demo script that handles everything

echo "=== PD-Separated Inference Demo (No GPU Required) ==="

# Clean up any existing processes
echo "1. Cleaning up existing processes..."
pkill -f "gswarm.data start" 2>/dev/null || true
sleep 2

# Start gswarm data server
echo "2. Starting gswarm data server..."
python -m gswarm.data start --host 0.0.0.0 --port 9015 --max-memory 2GB > /tmp/gswarm_server.log 2>&1 &
DATA_SERVER_PID=$!

# Wait for server to start
echo "   Waiting for server to start..."
for i in {1..10}; do
    if curl -s http://localhost:9015/stats > /dev/null 2>&1; then
        echo "   ✓ Server started successfully"
        break
    fi
    sleep 1
done

# Run the demo
echo -e "\n3. Running PD-separation demo with mock engine..."
python quick_demo.py

# Run benchmark comparison
echo -e "\n4. Running performance comparison..."
python demo_pd_separation.py << EOF
2
EOF

# Cleanup
echo -e "\n5. Cleaning up..."
kill $DATA_SERVER_PID 2>/dev/null || true

echo -e "\n=== Demo completed! ==="
echo "Key takeaways:"
echo "- PD-separation splits inference into specialized phases"
echo "- Prefill nodes handle prompt processing"
echo "- Decode nodes handle token generation"
echo "- KV cache is efficiently shared via gswarm"
echo "- Better resource utilization for generation-heavy workloads"
#! /bin/bash

# if ollama server is not running, start it with the following command
# ollama run llama3.1:70b (or other models)

# if gswarm profiler head node is not running, start it with the following command
# gsprof start --host 0.0.0.0 --port 8090 --freq 500 --enable-bandwidth --enable-nvlink

# if gswarm profiler client node is not running, start it with the following command
# gsprof connect localhost:8090 --enable-bandwidth

# Now we can send our tests

# Start profiling
gsprof profile localhost:8090 --name llama-test

# run llama3.1:70b with the following command
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:70b",
  "prompt":"Why is the sky blue?"
}'

# Stop profiling
gsprof stop localhost:8090 --name llama-test

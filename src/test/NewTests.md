# 1. gswarm start

I need a script to start a gswarm host and client.

in first test, i wanna only use 2 process on 1 server to test the basic functionality.

process 1: gswarm host
gswarm host start --port 8095 --http-port 8096 --model-port 9010
process 2: gswarm client
gswarm client connect localhost:8095 --resilient

remember pid as well, we will need it later.
read output of gswarm host and client carefully, check if they are running correctly.
then use kill command to kill the process.


# 2. gswarm profiler test

start gswarm host and client as in 1.

then use gswarm profiler to profile the gswarm host and client:
gswarm profiler read

read output of gswarm profiler carefully, check if it is running correctly.

then use kill command to kill the process.

# 3. gswarm model serve

start gswarm host and client as in 1.

download llama-7b model from huggingface:
gswarm model download llama-7b --source hf://meta-llama/Llama-3.1-8B-Instruct --type llm

then use gswarm model serve to serve a model:
gswarm model serve llama-7b --device cuda:0 --port 9010

read output of gswarm model serve carefully, check if it is running correctly.

then use kill command to kill the process.

# 4. gswarm prediction test

start gswarm host and client as in 1.

then use gswarm prediction to predict a model executino time

# 5. gswarm data test

start gswarm host and client as in 1.

then use gswarm data to test the data handling functionality.

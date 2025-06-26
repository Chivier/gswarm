#!/bin/bash
if [ ! -d "quickrun_result" ]; then
  mkdir quickrun_result
fi


for i in {4..8}
do
uv run offline_scheduler.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false
mv offline_execution_log.json quickrun_result/offline_execution_log_h20_$i.json

uv run online_scheduler.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false
mv online_execution_log.json quickrun_result/online_execution_log_h20_$i.json

uv run baseline.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode online
mv baseline_execution_log.json quickrun_result/baseline_execution_log_h20_online_$i.json

uv run baseline.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode offline
mv baseline_execution_log.json quickrun_result/baseline_execution_log_h20_offline_$i.json

done

uv run plot.py
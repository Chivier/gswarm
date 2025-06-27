#!/bin/bash
if [ ! -d "quickrun_result" ]; then
  mkdir quickrun_result
fi


for i in 8 16 24 32 40
do
uv run baseline.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode online
mv baseline_execution_log.json quickrun_result/baseline_execution_log_h20_online_$i.json

uv run baseline.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode offline
mv baseline_execution_log.json quickrun_result/baseline_execution_log_h20_offline_$i.json

uv run static_scheduler.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode offline
mv static_execution_log.json quickrun_result/static_execution_log_h20_offline_$i.json

uv run static_scheduler.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode online
mv static_execution_log.json quickrun_result/static_execution_log_h20_online_$i.json

done

uv run quickrun_and_plot/plot.py
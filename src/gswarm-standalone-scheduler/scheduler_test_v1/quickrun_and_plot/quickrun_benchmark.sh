#!/bin/bash
if [ ! -d "quickrun_result" ]; then
  mkdir quickrun_result
fi


for i in {4..8}
do
python offline_scheduler.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false
mv offline_execution_log.json quickrun_result/offline_execution_log_h20_$i.json

python online_scheduler.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false
mv online_execution_log.json quickrun_result/online_execution_log_h20_$i.json

python baseline.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode online
mv baseline_execution_log.json quickrun_result/baseline_execution_log_h20_online_$i.json

python baseline.py --config complex_config_h20.json --requests complex_requests.yaml --gpus $i --simulate false --mode offline
mv baseline_execution_log.json quickrun_result/baseline_execution_log_h20_offline_$i.json

done

python quickrun_and_plot/plot.py
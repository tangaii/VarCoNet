#!/usr/bin/env bash
set -euo pipefail

atlas="$1"
visible_device="$2"
root_dir="$3"
venv_python="$4"
parent_pid="$5"

active_count() {
  ps -eo ppid=,args= | awk -v atlas="$atlas" -v root_parent="$parent_pid" '
    ($1 == 1 || $1 == root_parent) && $0 ~ /ASD_classification_ABIDEI/ && $0 ~ ("--atlas " atlas "($| )") { count++ }
    END { print count + 0 }
  '
}

launch() {
  local group="$1"
  local repeats="$2"
  local out="$root_dir/run_$group"
  setsid env CUDA_VISIBLE_DEVICES="$visible_device" "$venv_python" -m ASD_classification_ABIDEI \
    --path_data /data/adodas/VarCoNet-V2-main/dataset/ABIDEI \
    --path_save "$out" --atlas "$atlas" --device cuda:0 \
    --epochs 50 --warm_up_epochs 10 --epochs_cls 150 --lr_cls 5e-5 \
    --min_length 80 --save_models --save_results --repeat_indices $repeats \
    > "$out/abide1.log" 2>&1 < /dev/null &
}

for item in "1:1" "4:4" "5:5" "6_8:6 8" "7_9:7 9"; do
  group="${item%%:*}"
  repeats="${item#*:}"
  result="$root_dir/run_$group/results_ABIDEI/$atlas/ABIDEI_VarCoNet_results.pkl"
  [ -f "$result" ] && continue
  while [ "$(active_count)" -ge 3 ]; do sleep 20; done
  launch "$group" "$repeats"
  sleep 25
done

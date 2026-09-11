#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="/data/adodas/VarCoNet-V2-main/xiaolunwen/experiment1"
cd "$EXP_DIR/../.."
source /root/venvs/vraconet/bin/activate

# One process owns one repeat.  Odd/even repeat assignment keeps every
# baseline/treatment pair on the same physical GPU while allowing five
# independent pairs per GPU.  Each process uses six CPU threads (60 total).
PIDS=()
for repeat in 0 1 2 3 4 5 6 7 8 9; do
  gpu=$((repeat % 2))
  log="/tmp/varconet_experiment1_repeat_${repeat}.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONWARNINGS=ignore \
    python "$EXP_DIR/run_experiment1.py" \
      --mode worker --device cuda:0 --worker-id "r${repeat}" --repeats "$repeat" \
      > "$log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
python "$EXP_DIR/run_experiment1.py" --mode aggregate

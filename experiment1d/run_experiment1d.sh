#!/usr/bin/env bash
set -u

source ~/venvs/vraconet/bin/activate
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

EXP="/data/adodas/VarCoNet-V2-main/xiaolunwen/experiment1d"
mkdir -p "$EXP/logs" "$EXP/checkpoints" "$EXP/results"

if [ ! -f "$EXP/results/.smoke_pass.json" ]; then
  python "$EXP/run_experiment1d.py" --mode smoke --device cuda:0 >>"$EXP/logs/smoke.log" 2>&1 || exit 1
fi

pids=()
for repeat in 0 1 2 3 4 5 6 7 8 9; do
  if [ "$repeat" = 0 ] || [ "$repeat" = 2 ] || [ "$repeat" = 4 ] || [ "$repeat" = 6 ] || [ "$repeat" = 8 ]; then
    gpu=0
  else
    gpu=1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python "$EXP/run_experiment1d.py" --mode worker --repeat "$repeat" --device cuda:0 >>"$EXP/logs/repeat_${repeat}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[ "$status" -eq 0 ] || exit "$status"

python "$EXP/run_experiment1d.py" --mode aggregate --device cuda:0 >>"$EXP/logs/aggregate.log" 2>&1

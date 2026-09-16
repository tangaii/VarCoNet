#!/usr/bin/env bash
# Locked Experiment 1V launcher: five independent workers per 96-GB GPU.
set -euo pipefail

cd /data/adodas/VarCoNet-V2-main
source /root/venvs/vraconet/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

runner=xiaolunwen/experiment1v/run_experiment1v.py
log_dir=xiaolunwen/experiment1v/logs
mkdir -p "$log_dir"

if [[ ! -f xiaolunwen/experiment1v/SMOKE.json ]]; then
  echo "SMOKE_REQUIRED: refusing to start full workers" >&2
  exit 2
fi
if [[ ! -f xiaolunwen/experiment1v/work/pearson_cache/abide1.npy || ! -f xiaolunwen/experiment1v/work/pearson_cache/abide2.npy ]]; then
  echo "PEARSON_CACHE_REQUIRED: refusing to start full workers" >&2
  exit 2
fi

pids=()
for repeat in 0 2 4 6 8; do
  python "$runner" --repeat "$repeat" --device cuda:0 >"$log_dir/repeat_${repeat}.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$log_dir/repeat_${repeat}.pid"
  pids+=("$pid")
  echo "LAUNCH repeat=$repeat device=cuda:0 pid=$pid"
done
for repeat in 1 3 5 7 9; do
  python "$runner" --repeat "$repeat" --device cuda:1 >"$log_dir/repeat_${repeat}.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$log_dir/repeat_${repeat}.pid"
  pids+=("$pid")
  echo "LAUNCH repeat=$repeat device=cuda:1 pid=$pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "WORKER_FAILURE: aggregate intentionally not run" >&2
  exit 1
fi

python "$runner" --aggregate >"$log_dir/aggregate.log" 2>&1
echo "EXPERIMENT1V_COMPLETE"

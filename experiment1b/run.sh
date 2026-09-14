#!/usr/bin/env bash
# Full Experiment 1B runner.  It performs one smoke test, then ten independent
# repeat workers and one deterministic aggregation step.  Logs are append-only.
set -euo pipefail

cd "$(dirname "$0")"
source ~/venvs/vraconet/bin/activate

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints results

if [[ ! -f results/summary.json ]]; then
  CUDA_VISIBLE_DEVICES=0 python -u run_experiment1b.py --mode smoke --device cuda:0 2>&1 | tee -a logs/smoke.log
else
  echo "FINAL_ARTIFACTS_EXIST: skipping a second real smoke test" | tee -a logs/smoke.log
fi

pids=()
launch_worker() {
  local repeat="$1"
  local gpu="$2"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python -u run_experiment1b.py --mode worker --repeat "$repeat" --device cuda:0
  ) 2>&1 | tee -a "logs/repeat_${repeat}.log" &
  pids+=("$!")
}

# Fixed placement: five independent repeat workers per physical GPU.
launch_worker 0 0
launch_worker 2 0
launch_worker 4 0
launch_worker 6 0
launch_worker 8 0
launch_worker 1 1
launch_worker 3 1
launch_worker 5 1
launch_worker 7 1
launch_worker 9 1

for pid in "${pids[@]}"; do
  wait "$pid"
done

python -u run_experiment1b.py --mode aggregate 2>&1 | tee -a logs/aggregate.log

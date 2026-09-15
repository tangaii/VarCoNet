#!/usr/bin/env bash
# Full-precision two-GPU launcher for Experiment 1B-R + 1C.
set -Eeuo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
source ~/venvs/vraconet/bin/activate
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/results" "${EXPERIMENT_DIR}/checkpoints"
cd "${REPO_DIR}"

# One real-data smoke test is mandatory before workers start.  CUDA_VISIBLE_DEVICES
# preserves the documented physical-GPU mapping while each worker uses cuda:0.
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python "${EXPERIMENT_DIR}/run_experiment1c.py" \
  --mode smoke --device cuda:0 >> "${EXPERIMENT_DIR}/logs/smoke.log" 2>&1

declare -a WORKER_PIDS=()
for repeat in $(seq 0 9); do
  if (( repeat % 2 == 0 )); then
    physical_gpu=0
  else
    physical_gpu=1
  fi
  log_path="${EXPERIMENT_DIR}/logs/repeat_${repeat}.log"
  {
    echo "WORKER_LAUNCH_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) repeat=${repeat} physical_gpu=${physical_gpu} cpu_threads=6 full_precision=Y"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" stdbuf -oL -eL python "${EXPERIMENT_DIR}/run_experiment1c.py" \
      --mode worker --repeat "${repeat}" --device cuda:0
  } >> "${log_path}" 2>&1 &
  WORKER_PIDS+=("$!")
done

failed=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one Experiment 1C worker failed; inspect logs/repeat_*.log. Checkpoints were retained for resume." >&2
  exit 1
fi

python "${EXPERIMENT_DIR}/run_experiment1c.py" --mode aggregate

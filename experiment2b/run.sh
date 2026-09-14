#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="/data/adodas/VarCoNet-V2-main/xiaolunwen/experiment2b"
cd "/data/adodas/VarCoNet-V2-main"
source /root/venvs/vraconet/bin/activate
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints" "$EXP_DIR/results"

if [[ ! -f "$EXP_DIR/results/.smoke_pass.json" ]]; then
  {
    echo "=================================================="
    echo "INVOCATION_START $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=smoke gpu=0"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
      PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 \
      python -u "$EXP_DIR/run_experiment2b.py" --mode smoke --device cuda:0
  } 2>&1 | tee -a "$EXP_DIR/logs/smoke.log"
fi

echo "Experiment 2B started. Logs: $EXP_DIR/logs/repeat_*.log"
PIDS=()
for repeat in $(seq 0 9); do
  gpu=$((repeat % 2))
  log="$EXP_DIR/logs/repeat_${repeat}.log"
  {
    echo "=================================================="
    echo "INVOCATION_START $(date -u +%Y-%m-%dT%H:%M:%SZ) repeat=$repeat gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
      PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 \
      python -u "$EXP_DIR/run_experiment2b.py" --mode worker --device cuda:0 --repeat "$repeat"
  } >> "$log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "One or more Experiment 2B workers failed; inspect $EXP_DIR/logs/repeat_*.log" >&2
  exit 1
fi

python -u "$EXP_DIR/run_experiment2b.py" --mode aggregate

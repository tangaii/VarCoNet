#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="/data/adodas/VarCoNet-V2-main/xiaolunwen/experiment2"
cd "$EXP_DIR/../.."
source /root/venvs/vraconet/bin/activate

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints" "$EXP_DIR/results"

if [[ -f "$EXP_DIR/results/summary.json" ]]; then
  if python - "$EXP_DIR/results/summary.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("completion", {}).get("metrics_rows") == 260 else 1)
PY
  then
    echo "Experiment 2 is already fully aggregated: $EXP_DIR/results/summary.json"
    exit 0
  fi
fi

# Exactly one necessary real-batch smoke test is retained only until aggregate.
if [[ ! -f "$EXP_DIR/results/.smoke_pass.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
    PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 \
    python -u "$EXP_DIR/run_experiment2.py" --mode smoke --device cuda:0 \
    |& tee "$EXP_DIR/logs/smoke.log"
fi

echo "Experiment 2 started. Live log: tail -f $EXP_DIR/logs/repeat_0.log"

PIDS=()
for repeat in 0 1 2 3 4 5 6 7 8 9; do
  gpu=$((repeat % 2))
  log="$EXP_DIR/logs/repeat_${repeat}.log"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
    PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 \
    python -u "$EXP_DIR/run_experiment2.py" --mode worker --device cuda:0 --repeat "$repeat" \
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
  echo "One or more Experiment 2 workers failed; inspect $EXP_DIR/logs/repeat_*.log" >&2
  exit 1
fi

python -u "$EXP_DIR/run_experiment2.py" --mode aggregate

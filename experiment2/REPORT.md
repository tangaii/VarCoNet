# Experiment 2
Stable-Dynamic Residual VarCoNet

## 1. Problem

VarCoNet 的 global learned-FC 强调跨时间稳定 trait，但可能在最终时间聚合时丢弃 ASD-related dynamic switching information。

## 2. Literature rationale

- ASD functional-connectivity-dynamics switching work motivates testing temporal state changes.
- TSD-GCN motivates consecutive differential dynamics; VA-SDNet motivates ROI/graph variability summaries.
- SDF motivates static-dynamic complementarity, while DSAM motivates using latent temporal brain features.
- 这些概念仅用于特征定义；本实验没有复制 TCN、GNN、cross-attention 或任何完整外部网络。

## 3. Exact change

原 encoder、Conv1d kernel=4/stride=2、augmentation、optimizer/scheduler 与原 InfoNCE 均不变。每个 task 只训练一条 50-epoch SSL encoder trajectory；每个 epoch 从同一 encoder 批量提取 stable learned-FC `S` 和动态描述符 `D`。Treatment 是 frozen-stable residual logit：`W_s S + b + W_d D_z`，其中 `W_s,b` 冻结且 `W_d` 零初始化。

## 4. Dynamic feature definition

- ROI switching: 384 dimensions.
- ROI stable deviation: 384 dimensions.
- Graph switching mean/std: 2 dimensions.
- Graph stable-deviation mean/std: 2 dimensions.
- Consecutive-state transition-distance mean/std: 2 dimensions.
- Total: 774 dimensions; fixed non-overlapping latent window=15 tokens, stride=15.

## 5. Leakage / integrity

- No label, metadata, site, FD, TR, age or sex enters the encoder.
- Dynamic normalization uses only each epoch's training `D` mean/std.
- Classifier/encoder selection and Youden threshold use validation only.
- Test, Caltech and ABIDE-II are evaluated only after all 50 epochs are frozen.
- Duplicate ABIDE-I occurrences remain train-only; Caltech remains external-only.
- There is no Experiment 1 runtime import; original root source/data are not modified by this experiment.

### ABIDE-II technical eligibility

One ABIDE-II subject (sub-29623) was excluded from BOTH paired baseline and treatment evaluation because its scan was too short to construct the pre-specified two complete latent dynamic windows. The exclusion was determined solely by sequence-length eligibility and was independent of diagnosis or model predictions.

The eligibility audit found exactly one excluded subject: original index 568 (`sub-29623`). ABIDE-II therefore has original n=727 and paired evaluable n=726 for both methods. This is a technical evaluability constraint of the pre-specified dynamic descriptor, not a method optimization.

## 6. Efficiency

- One shared encoder trajectory per task, rather than separately retraining baseline/treatment encoders.
- 10 workers, five per GPU, six CPU threads per worker, full precision, no DDP/AMP/compile.
- Worker wall-clock maximum: 0.02 h; summed worker time: 0.23 h.
- Mean worker peak GPU allocation/reservation: 0.98 / 1.40 GB.
- Rolling `latest`/`best` checkpoints are atomic; resume support: True.

## 7. Main results

| Scope | Baseline AUC | SDR AUC | ΔAUC | Baseline BCE | SDR BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
| Internal | 0.7227 | 0.7201 | -0.0026 | 0.6123 | 0.6147 | 0.0024 |
| Caltech | 0.6564 | 0.6307 | -0.0257 | 0.6572 | 0.6754 | 0.0183 |
| ABIDE-II | 0.7385 | 0.7276 | -0.0109 | 0.6014 | 0.6109 | 0.0095 |

10 internal repeat ΔAUC (SDR - baseline):
`[-0.010763, 0.001801, -0.001445, -0.004025, -0.002859, -0.000738, -0.007668, 0.001166, -0.002533, 0.001414]`

10 ABIDE-II repeat ΔAUC (SDR - baseline):
`[-0.011114, -0.015959, 0.000000, 0.002326, -0.005022, -0.006917, -0.020187, 0.000000, -0.037764, -0.014827]`

Baseline sanity: internal OOF AUC=0.7227 (reference 0.723); ABIDE-II AUC=0.7385 (reference 0.739); pipeline drift=False.

## 8. Dynamic diagnostics

- Dynamic feature finite: all final task rows are PASS; dim=774.
- Latent complete windows (task-average min/mean/max): 5.663 / 7.402 / 9.494.
- Mean node switch: 0.362674; mean node deviation: 0.250718.
- Selected residual-epoch distribution: `{"0": 28, "1": 1, "2": 1, "3": 1, "4": 4, "5": 1, "6": 5, "7": 1, "8": 4, "9": 1, "10": 1, "11": 2, "18": 1, "19": 1, "21": 2, "22": 2, "27": 1, "31": 1, "32": 1, "33": 1, "34": 2, "35": 1, "36": 2, "37": 1, "40": 1, "43": 2, "51": 1, "55": 2, "56": 1, "57": 1, "58": 2, "60": 1, "61": 1, "62": 1, "63": 1, "65": 1, "66": 1, "67": 1, "72": 1, "73": 1, "75": 1, "78": 1, "88": 2, "90": 1, "106": 1, "110": 1, "115": 3, "121": 1, "126": 1, "130": 1, "139": 2, "144": 1, "149": 9}`.
- `residual_epoch=0` fraction: 0.2545 across 110 task selections.

## 9. Decision

**FAIL**

## 10. Interpretation

从当前 VarCoNet latent space 提取这种 fixed-window dynamic residual，没有稳定转化为疾病预测收益；本实验按预注册标准封存，不进行调参救援。

## 11. Exact handoff

- Exact files: `model_scripts/VarCoNet.py`, copied baseline helper files, `dynamic_features.py`, `run_experiment2.py`, `run.sh`, `results/metrics.csv`, `results/summary.json`, `results/predictions.npz`, and `results/external_best_models.pt`.
- Exact formula: local latent cosine FC vs whole-scan latent cosine FC; 384 ROI switch + 384 ROI deviation + six graph/transition statistics.
- Fixed hyperparameters: AICHA 384 ROI, stable=73536, window/stride=15/15, 50 SSL epochs, classifier/residual lr=5e-5, 149 updates, 10 repeats × 10 folds.
- Selected encoder epoch means: baseline=29.468, SDR=27.782. Per-task selected epochs and all paired metrics are in `metrics.csv`.
- The final decision and per-repeat deltas are recorded above and in `summary.json`; completed tasks have no retained per-fold weights. External selected models are bundled once in `external_best_models.pt`.
- The initial external evaluation correctly stopped on the one ineligible short ABIDE-II scan; after the authorized label-blind exclusion, all ten workers resumed from external epoch-50 checkpoints and completed inference without retraining. No numerical/runtime integrity errors remained in the resumed run.

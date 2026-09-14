# Experiment 2B
Pre-Transformer Edge-wise Dynamic Residual VarCoNet

## 1. Problem

VarCoNet 的 global learned-FC 强调跨时间稳定 trait，但可能在最终时间聚合时丢弃 ASD-related dynamic switching information。

## 2. Literature rationale

- ASD functional-connectivity-dynamics switching work motivates testing temporal state changes.
- TSD-GCN motivates consecutive differential dynamics; VA-SDNet motivates ROI/graph variability summaries.
- SDF motivates static-dynamic complementarity, while DSAM motivates using latent temporal brain features.
- 这些概念仅用于特征定义；本实验没有复制 TCN、GNN、cross-attention 或任何完整外部网络。

## 3. Exact change

原 encoder、Conv1d kernel=4/stride=2、augmentation、optimizer/scheduler 与原 InfoNCE 均不变。每个 task 只训练一条 50-epoch SSL encoder trajectory；baseline validation BCE 选择唯一 encoder/classifier。Treatment 复用该冻结 baseline，并令 `delta=w^T D_z`，残差 logits 为 `[-0.5*delta,+0.5*delta]`；只有 `w`（73,536 维、零初始化）训练。

## 4. Dynamic feature definition

This is a new, pre-Transformer edge-wise treatment relative to Experiment 2;
the earlier 774-dimensional stable/dynamic summary is not used or recomputed.

- Local FC is computed on each pre-Transformer CNN window; adjacent windows use
  `abs(F[t+1,e]-F[t,e])`, averaged per edge.
- One scalar per upper-triangular AICHA edge: 384×383/2 = 73,536 dimensions.
- Fixed non-overlapping latent window=15 tokens, stride=15; no ROI/graph summaries.

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
- Worker wall-clock maximum: 1.29 h; summed worker time: 10.23 h.
- Mean worker peak GPU allocation/reservation: 1.40 / 2.01 GB.
- Rolling `latest`/`best` checkpoints are atomic; resume support: True.

## 7. Main results

| Scope | Baseline AUC | Edge residual AUC | ΔAUC | Baseline BCE | Edge residual BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
| Internal | 0.7233 | 0.7186 | -0.0047 | 0.6118 | 0.6168 | 0.0050 |
| Caltech | 0.6573 | 0.6681 | 0.0108 | 0.6573 | 0.6525 | -0.0048 |
| ABIDE-II | 0.7384 | 0.7357 | -0.0027 | 0.6015 | 0.6031 | 0.0016 |

10 internal repeat ΔAUC (edge residual - baseline):
`[0.000372, -0.005077, -0.005841, -0.003710, -0.000273, 0.000134, -0.008343, -0.002869, -0.015665, -0.005521]`

10 ABIDE-II repeat ΔAUC (edge residual - baseline):
`[-0.000046, -0.002911, -0.006331, 0.000339, -0.000424, 0.001725, -0.017268, 0.000000, -0.002187, 0.000000]`

Baseline sanity: internal OOF AUC=0.7233 (reference 0.723); ABIDE-II AUC=0.7384 (reference 0.739); pipeline drift=False.

## 8. Dynamic diagnostics

- Dynamic feature finite: all final task rows are PASS; dim=73536.
- Latent complete windows (task-average min/mean/max): 5.663 / 7.402 / 9.494.
- Mean edge switch: 0.290478; per-subject edge-switch std: 0.115359.
- Dynamic head L2 range across task rows: 0.000000–0.173700 (mean 0.026072).
- Selected residual-epoch distribution: `{"0": 47, "1": 21, "2": 7, "3": 7, "4": 7, "5": 3, "6": 2, "7": 1, "8": 2, "9": 2, "10": 4, "11": 1, "13": 3, "15": 2, "17": 1}`.
- `residual_epoch=0` fraction: 0.4273 across 110 task selections.

## 9. Decision

**FAIL**

## 10. Interpretation

从当前 VarCoNet latent space 提取这种 fixed-window dynamic residual，没有稳定转化为疾病预测收益；本实验按预注册标准封存，不进行调参救援。

## 11. Exact handoff

- Exact files: `model_scripts/VarCoNet.py`, copied baseline helper files, `pretransformer_dynamic.py`, `run_experiment2b.py`, `run.sh`, `results/metrics.csv`, `results/summary.json`, `results/predictions.npz`, and `results/external_best_models.pt`.
- Exact formula: pre-Transformer local-window cosine-FC edge vectors, then per-edge mean adjacent-window absolute difference; `dynamic_dim=73536`.
- Fixed hyperparameters: AICHA 384 ROI, stable=73536, window/stride=15/15, 50 SSL epochs, classifier/residual lr=5e-5, 149 updates, 10 repeats × 10 folds.
- Selected encoder epoch means: baseline=29.815, edge residual=29.815. Per-task selected epochs and all paired metrics are in `metrics.csv`.
- The final decision and per-repeat deltas are recorded above and in `summary.json`; completed tasks have no retained per-fold weights. External selected models are bundled once in `external_best_models.pt`.
- External evaluation used the pre-specified label-blind eligibility rule before any label or prediction was read; both paired methods therefore use the same 726 ABIDE-II subjects. No numerical/runtime integrity errors remained.

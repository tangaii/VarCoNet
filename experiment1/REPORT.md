# Experiment 1
Subtype Prototype Guided VarCoNet

## 1. Research question
在保留 VarCoNet 原始 subject-identity InfoNCE 的前提下，训练期 class-conditional online subtype prototype guidance 是否提高 AICHA ASD prediction，尤其是 ABIDE-II external generalization？

## 2. Exact change from baseline
Baseline 的 VarCoNet encoder、learned-FC upper-triangle、原始 InfoNCE、augmentation、optimizer、scheduler 和线性 classifier 均保持不变。Treatment 仅增加训练期 `OnlineClassSubtypeBank` 与 `SubtypePrototypeContrastiveLoss`，总损失为 `loss_identity + 1.0 * loss_prototype`。Prototype 不进入 optimizer，测试/validation/Caltech/ABIDE-II 不使用 prototype。

## 3. Files
本实验所有新增源码均位于 `xiaolunwen/experiment1`：复制的 `model_scripts/VarCoNet.py`、`classifier.py`、`evaluation.py`、`scheduler.py`、`utils.py`，`subtype_prototype.py`，`run_experiment1.py`，`best_params_VarCoNet_AICHA.pkl`，以及 `results/metrics.csv`、`results/summary.json`。

## 4. Implementation
- AICHA 384 ROI，learned-FC 维度为 `384*383/2 = 73536`；原始 Conv1d 为 kernel 4、stride 2。
- 每个 diagnosis class（0=ASD，1=HC）使用 K=3 prototypes。
- 首次出现的 class 使用当前 batch normalized embedding 的 deterministic farthest-point initialization；第一个点最接近 centroid，后续点最大化与已选点的 cosine 距离。
- assignment 使用 epsilon=0.05、3 次 Sinkhorn 归一化；prototype 使用 momentum=0.99 EMA 更新。
- 正样本是 `P[y_i, subtype_i]`；负样本是 opposite-diagnosis 的全部 prototypes；同诊断其它 prototypes 不进正/负集合。
- `zbar = normalize((normalize(z1)+normalize(z2))/2)` 且 assignment 前 detach；训练 labels 在 removeDuplicates 后重新索引取得。
- inference 完全不使用 prototype。

## 5. Integrity
- 原项目根目录及 `Experiment/`、`dataset/`、`reproduction_results/` 均未写入。
- prototype assignment/update 只使用 training batch；没有 validation/test/Caltech/ABIDE-II labels 进入 prototype learning。
- encoder 与 classifier 均按 validation BCE 选择；threshold 仅由 selected classifier 的 validation probabilities 的 Youden J 决定并冻结。
- duplicate occurrences 只追加至 training；internal 为 10 repeats × 10 folds，Caltech 37 例仅 external test。
- baseline/treatment 对每一个 paired task 使用相同 split、batch schedule、augmentation、optimizer、epoch 和 task seed；每个 pair 在同一 GPU 上连续运行并分别 reset seed。
- 每个 worker 仅保留一个滚动 checkpoint：每 10 个 encoder epoch 原子替换 `xiaolunwen/experiment1/checkpoints/worker_*_latest.pt`，旧权重立即删除；checkpoint 不参与指标计算。

## 6. Results

| Scope | Baseline AUC | Treatment AUC | ΔAUC | Baseline BCE | Treatment BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
| Internal | 0.7224 | 0.7063 | -0.0162 | 0.6127 | 0.6253 | 0.0127 |
| Caltech | 0.6474 | 0.6442 | -0.0032 | 0.6698 | 0.6705 | 0.0007 |
| ABIDE-II | 0.7369 | 0.7279 | -0.0090 | 0.6020 | 0.6128 | 0.0108 |

10 internal repeat ΔAUC (treatment - baseline):
`[-0.020861, -0.010191, -0.020221, -0.026331, -0.017249, -0.013735, -0.010707, -0.018240, -0.015283, -0.008813]`

10 ABIDE-II repeat ΔAUC (treatment - baseline):
`[0.000246, -0.017675, -0.003747, -0.015441, -0.010573, 0.005774, -0.017138, -0.004983, -0.012078, -0.013997]`

Baseline sanity: internal OOF AUC=0.7224 (reference 0.723), ABIDE-II AUC=0.7369 (reference 0.739); pipeline drift=False.

## 7. Prototype diagnostics
- Treatment prototype loss mean over reported repeat summaries: `0.959739`.
- Active prototypes (out of 6): mean `6.000`; mean minimum occupancy `8.221`; mean maximum occupancy `12.515`.
- Both diagnosis classes are required by every stratified training task; the run records finite losses/gradients and reports any inactive/collapsed bank through these columns.

## 8. Decision
**FAIL**

## 9. Interpretation
结果说明这种 online prototype disease guidance 没有稳定转化为 ASD classification / external generalization gain；本实验不进行调参救援。

## 10. Next-step handoff
- Exact changed files：仅 `xiaolunwen/experiment1` 下列出的复制源码、`subtype_prototype.py`、`run_experiment1.py`、结果文件和本报告。
- Exact method：K=3 class-conditional online prototypes，FPS initialization，Sinkhorn epsilon=0.05/3 iterations，EMA momentum=0.99，lambda=1.0；VarCoNet identity InfoNCE 完全保留。
- Key metrics：见上表、`results/metrics.csv` 和 `results/summary.json`。
- Decision：`FAIL`。
- Code/data notes：AICHA 384 ROI；duplicate occurrences train-only；validation-only threshold；原始实现 Conv1d kernel 4/stride 2；运行时未依赖 Experiment/08 或 Experiment/09。

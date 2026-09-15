# Experiment 1B-R + 1C

## Corrected control-referenced boundaries and selective normative-deviation experts

This self-contained experiment retains the audited AICHA/384 VarCoNet pipeline and makes no change outside `xiaolunwen/experiment1c`. It tests three paired branches for every task: the original `baseline`, the corrected two-boundary `crsm_corrected` branch, and `selective_normative_deviation_experts` (SNDE). One original SSL trajectory and one validation-selected `Linear(73536,2)+Softmax` classifier are trained per task; both post-hoc branches receive tensor-identical frozen encoder/classifier states and the same extracted stable FC tensors.

## Locked baseline and evaluation

The baseline uses the copied audited VarCoNet source: AICHA 384 ROIs, Conv1d kernel 4 / stride 2, stable upper-triangle learned FC dimension 73,536, repository augmentation and InfoNCE, Adam plus warmup/cosine schedule, 50 SSL epochs, and a classifier trained with Adam `5e-5` for updates 1–149. Encoder and classifier selection use validation BCE. Internal evaluation is 10 repeats × 10 stratified folds (`random_state=42+repeat`) with 15% validation (`random_state=42`); external fitting uses the locked 90/10 unique split (`random_state=42+repeat`). The 76 duplicate ABIDE-I occurrences are train-only. `reference_raw` contains unique outer-training subjects only. Caltech has `n=37`, and the complete paired ABIDE-II cohort has `n=727` for all three methods.

Test stable FC is extracted once per task/scope and reused by all methods. Test labels are accessed only after validation-selected predictions are made. Each method has its own validation-only Youden threshold. No DDP, DataParallel, AMP, `torch.compile`, metadata, site, motion, demographic covariates, test-time adaptation, or hyperparameter sweep is used.

## 1B-R correction

The previous Experiment 1B implementation labelled a **post-PCA 1% symmetry perturbation** as `epoch=0`; it was not the baseline-equivalent all-zero residual state. In 1B-R, the two CRSM residual tensors are instantiated at exact bitwise zero, zero parity against baseline margin/probabilities/BCE/CRSM loss is asserted to `<=1e-6`, and that exact-zero checkpoint is saved **before** PCA symmetry initialization. Only strictly lower validation BCE can replace it. Thus every corrected `selected_crsm_epoch=0` in this report denotes the true baseline fallback, and has exact baseline prediction parity.

CRSM remains exactly two control-referenced residual boundaries (`147,074` trainable parameters), `tau=1`, fixed 1% PCA symmetry initialization after the zero candidate, and Adam `5e-5` for updates 1–149. It uses no prototype, Sinkhorn, hard assignment, SVM, GAN, entropy/balance/diversity/orthogonality loss, or encoder change. Per-diagnosis responsibility mass/entropy is reported only on the unique training reference set.

## 1C SNDE method

For each stable upper-triangle FC vector, SNDE forms the signed ROI connectivity profile `G_r = mean_(j!=r) FC_(r,j)` (`384` entries). On reference HC subjects only it estimates `mu_r` and population standard deviation `sigma_r`, floored at `max(1e-6, 0.1*median(positive HC std))`. It calculates `Z=(G-mu)/sigma`, centers reference ASD `Z` around its ASD mean, and obtains exactly one `torch.pca_lowrank(q=1, center=False, niter=5)` axis under a forked fixed RNG seed. Axis sign is canonicalized at its maximum-absolute loading. HC scores define zero mean/unit population standard deviation.

At inference, `S` outside `+1` or `-1` activates one of two excess gates; `|S|<=1` is the exact baseline generalist path. The two trainable residuals are `ReLU(S-1)(w_pos^T Z+b_pos)` and `ReLU(-S-1)(w_neg^T Z+b_neg)` added to the frozen baseline ASD margin. There are exactly `384+1+384+1=770` trainable parameters, all initialized to exact zero. SNDE uses only binary BCE on `p_HC`, Adam `1e-3`, updates 1–149, strict validation-BCE selection, no weight decay, and no other objective or model component.

## Results

| Scope | Baseline AUC | CRSM AUC | SNDE AUC | CRSM ΔAUC | SNDE ΔAUC | Baseline BCE | CRSM BCE | SNDE BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Internal OOF | 0.7225 | 0.7227 | 0.7184 | 0.0002 | -0.0041 | 0.6124 | 0.6124 | 0.6335 |
| Caltech | 0.6579 | 0.6588 | 0.6892 | 0.0009 | 0.0313 | 0.6570 | 0.6573 | 0.6504 |
| ABIDE-II | 0.7372 | 0.7373 | 0.7317 | 0.0001 | -0.0055 | 0.6024 | 0.6022 | 0.6162 |

### SNDE minus corrected CRSM

| Scope | ΔAUC | ΔBCE |
|---|---:|---:|
| Internal OOF | -0.0043 | 0.0211 |
| Caltech | 0.0304 | -0.0068 |
| ABIDE-II | -0.0056 | 0.0139 |

Baseline sanity: internal OOF AUC=0.7225 (reference 0.723), ABIDE-II AUC=0.7372 (reference 0.739), pipeline drift=False.

## Required audit answers

1. **True baseline epoch-0 effect/result:** corrected CRSM epoch 0 is the actual bitwise-zero residual and exactly reproduces baseline predictions; it is no longer the old PCA-perturbed state.
2. **Old 82.7% comparison:** the previous report's `91/110` (`82.7%`) “epoch 0” selections referred to a post-PCA perturbation. In this corrected run, exact-zero CRSM was selected in `92/110` tasks (`0.8364`).
3. **SNDE ASD gate fractions:** positive=0.0794, negative=0.0609, generalist=0.8597 (means across 110 selected internal-fold/Caltech tasks).
4. **HC extremes versus ASD:** HC positive/negative/generalist fractions are 0.0567/0.0424/0.9009, versus ASD 0.0794/0.0609/0.8597. These are gate diagnostics, not subtype labels.
5. **Overfit/collapse control:** SNDE uses 770 parameters rather than edge-level residuals, reference-only HC norming, a single fixed PCA axis, no sweep, and validation-only selection. Its aggregate mechanism status is `ACTIVE`; positive gate activated in any task=True, negative gate activated in any task=True; mean selected parameter L2=0.371601 and mean absolute correction=0.124693.
6. **Actual ABIDE-II improvement:** SNDE − baseline mean ABIDE-II ΔAUC=-0.005507 and ΔBCE=0.013806. Corrected CRSM − baseline ABIDE-II ΔAUC=0.000056.

Corrected CRSM selection distribution: `{"0": 92, "1": 7, "2": 6, "3": 3, "5": 1, "7": 1}`. SNDE selection distribution: `{"0": 16, "1": 9, "10": 2, "114": 1, "117": 1, "12": 1, "13": 1, "149": 8, "15": 2, "16": 2, "17": 1, "2": 8, "20": 2, "21": 1, "22": 3, "23": 1, "24": 1, "25": 1, "26": 1, "3": 9, "30": 1, "31": 2, "32": 1, "38": 1, "39": 1, "4": 3, "41": 2, "43": 1, "49": 1, "5": 4, "51": 1, "56": 1, "59": 1, "6": 4, "60": 1, "7": 2, "72": 1, "8": 4, "85": 1, "89": 1, "9": 4, "96": 1}`. SNDE PC1 explained-variance fraction mean=0.203682; HC std floor mean=0.00129647.

## Decisions

- SNDE vs baseline: **FAIL** (internal ΔAUC=-0.004129, positive internal repeats=1/10, ABIDE-II ΔAUC=-0.005507, internal/ABIDE-II ΔBCE=0.021027/0.013806).
- Corrected CRSM vs baseline: **FAIL** (internal ΔAUC=0.000160, positive internal repeats=5/10, ABIDE-II ΔAUC=0.000056).

The negative result indicates that even after correcting the original CRSM baseline fallback, moving heterogeneity modeling from full edge-level boundaries to selective control-referenced regional deviation experts does not yield stable incremental ASD classification.

## Integrity, source, and runtime audit

The source hash guard was checked before smoke and after aggregation for root `model_scripts`, root utility/classification scripts, and Experiment 1/1B/2/2B/3 directories; post-run status is `True`. A real-data smoke test checked root/local encoder parity, stable shape, CRSM zero parity before PCA, nonzero PCA symmetry state, profile reconstruction, normative HC normalization, ASD centering, unit/canonical axis, SNDE zero parity, finite forward/backward, gate fractions, and exact parameter counts.

Workers run full precision with repeats 0/2/4/6/8 on GPU0 and 1/3/5/7/9 on GPU1, six CPU threads each. Wall clock=0.956 h; summed task compute=9.009 h; mean peak GPU allocated/reserved=1.447/2.246 GB.

## Artifacts

`results/metrics.csv` contains 390 rows (100 internal folds + 10 internal OOF + 10 Caltech + 10 ABIDE-II for each of three methods). `results/predictions.npz` contains paired selected-scope subject IDs, labels, probabilities, and thresholds. `results/summary.json` holds all aggregate, decision, diagnostic, and integrity fields. `results/external_best_models.pt` holds the ten selected external states. Rolling checkpoints are resume-only and are removed after successful aggregation; the selected external weight artifact remains.

This experiment does not identify, assign, or claim biological ASD subtypes; its regional gate diagnostics are technical properties of a pre-specified classification readout.

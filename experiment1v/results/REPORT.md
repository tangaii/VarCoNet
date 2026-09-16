# Experiment 1V — Direct Mechanism Validation

**Question 1 — Does a cross-subject, cross-site, externally transferable ASD connectivity pattern exist?**  **SUPPORTED**

**Question 2 — Does VarCoNet suppress this shared disease structure while enhancing subject identity?**  **STRONG_SUPPORTED**

This is a preregistered mechanism audit, not a supervised rescue or an innovation search. No metric, window, top-K, epoch, direction, or subject subset was changed after observing results.

## Executive result

Internal strict LOSO Pearson macro AUC: **0.6201** (site-bootstrap 95% CI 0.5683–0.6706).  ABIDE-II Pearson site-balanced-axis AUC: **0.5706** (subject-bootstrap 95% CI 0.5266–0.6115; label-permutation p=0.00040).

## Core paired comparison

| Metric | Pearson | VarCoNet | Δ VarCoNet-Pearson | 95% CI |
|---|---:|---:|---:|---:|
| ABIDE-II fingerprint TOP1 | 0.8582 | 0.9941 | 0.1358 | 0.0959–0.1426 |
| ABIDE-II identity gap | 0.3066 | 0.2850 | -0.0216 | -0.0275–-0.0161 |
| Internal cross-site disease geometry gap | 0.0046 | 0.0030 | -0.0016 | -0.0029–-0.0005 |
| Internal LOSO disease AUC (Pearson) | 0.6201 | — | — | 0.5683–0.6706 |
| ABIDE-II disease-axis AUC | 0.5706 | 0.6836 | 0.1130 | 0.0661–0.1594 |
| ABIDE-II top-edge sign replication | 0.9469 | 0.9176 | — | — |
| ABIDE-II top-edge effect correlation (Spearman) | 0.4096 | 0.7423 | — | — |

All uncertainty intervals above are subject- or site-bootstrap intervals, never pair-level standard errors.

## SSL trajectory (10 external repeats; mean ± SD)

| Epoch | Identity TOP1 | Identity gap | Disease AUC | Disease geometry | Split-half stability |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9378 ± 0.0448 | 0.2418 ± 0.0243 | 0.6594 ± 0.0037 | 0.0036 ± 0.0006 | 0.4141 ± 0.0133 |
| 1 | 0.9495 ± 0.0225 | 0.2486 ± 0.0173 | 0.6596 ± 0.0038 | 0.0037 ± 0.0004 | 0.4129 ± 0.0122 |
| 5 | 0.9766 ± 0.0066 | 0.2657 ± 0.0086 | 0.6670 ± 0.0052 | 0.0036 ± 0.0002 | 0.3881 ± 0.0067 |
| 10 | 0.9930 ± 0.0013 | 0.2770 ± 0.0068 | 0.6781 ± 0.0059 | 0.0031 ± 0.0001 | 0.3519 ± 0.0032 |
| 20 | 0.9953 ± 0.0013 | 0.2858 ± 0.0035 | 0.6846 ± 0.0038 | 0.0029 ± 0.0000 | 0.3350 ± 0.0056 |
| 30 | 0.9957 ± 0.0020 | 0.2889 ± 0.0026 | 0.6853 ± 0.0025 | 0.0028 ± 0.0000 | 0.3289 ± 0.0043 |
| 40 | 0.9956 ± 0.0013 | 0.2903 ± 0.0021 | 0.6855 ± 0.0025 | 0.0028 ± 0.0001 | 0.3259 ± 0.0057 |
| 50 (selected 1/10) | 0.9958 ± 0.0015 | 0.2906 ± 0.0020 | 0.6858 ± 0.0026 | 0.0028 ± 0.0001 | 0.3252 ± 0.0052 |
| selected (per-repeat validation state) | 0.9892 ± 0.0168 | 0.2850 ± 0.0125 | 0.6820 ± 0.0068 | 0.0030 ± 0.0004 | 0.3405 ± 0.0271 |

Trajectory trade-off-consistent repeats (identity TOP1↑ and at least one disease-sharing metric↓): **10/10**.
The fixed-state rank correlations for every repeat are retained in `statistics.json` (`trajectory_correlations`); selected states were evaluated once per repeat and are marked above rather than retrained.

## Locked protocol and integrity

- AICHA 384 ROIs (73,536 upper-triangle edges); original VarCoNet architecture and AICHA hyperparameters; 50 SSL epochs; validation-BCE selected encoder; 10 repeats × 10 internal folds.
- P0 `pearson_windowavg` uses the exact one complete padded scan that Experiment 1D actually passes to VarCoNet. The original `test_augment()` (80/200/320, 10 windows) was audited separately and reproduced bitwise; it is not silently substituted for Experiment 1D stable extraction.
- P1 `pearson_full` uses all real contiguous points. VarCoNet receives zero-padded inputs; Pearson receives the corresponding real points only.
- Fingerprint views are deterministic first/last 80 valid points and are evaluated only when T_valid ≥ 160. Disease metrics use all 882 internal unique subjects, 37 Caltech subjects, and all 727 ABIDE-II rows.
- Duplicate ABIDE-I scans are training-only; internal audit references/tests contain no duplicate names. ABIDE-II row IDs are the original 0–726 occurrence indices, preserving the preregistered 727-row external cohort.
- Site IDs are used only for axis balancing, cross-site pair selection, LOSO, and bootstrap strata; never as model input. Official ABIDE-I and ABIDE-II phenotype mappings are 100% complete.
- Root/prior source hashes were unchanged before/after aggregation; no worker errors; temporary trajectory/checkpoint shards were removed.

## Interpretation

The decision follows the locked H1/H2 rules above; it does not claim complete removal or preservation of disease information.

## Required references audited

- VarCoNet paper and repository: `2026 Human Brain Mapping VarCoNet` / `CharLamp10/VarCoNet-V2`.
- Finn et al. (2015), *Nature Neuroscience*, DOI [10.1038/nn.4135](https://doi.org/10.1038/nn.4135).
- *Biological Psychiatry* (2024), DOI [10.1016/j.biopsych.2023.09.012](https://doi.org/10.1016/j.biopsych.2023.09.012).
- *Nature Mental Health* (2026), DOI [10.1038/s44220-026-00656-y](https://doi.org/10.1038/s44220-026-00656-y).

## Output files

`metrics.csv`, `trajectory.csv`, `statistics.json`, `summary.json`, `scores.npz`, and `top_edges.csv` are the compact reproducible outputs. No 73,536-dimensional feature matrix or permanent trajectory checkpoint is retained in `results/`.

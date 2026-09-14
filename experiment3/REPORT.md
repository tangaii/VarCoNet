# Experiment 3
## Consensus High-Order Modular Residual VarCoNet (CHMR-VarCoNet)

## Motivation

Experiment 1 changed cross-subject geometry with subtype prototypes and failed. Experiment 2 and Experiment 2B added temporal/dynamic residual signals and failed. Therefore Experiment 3 does not modify the VarCoNet representation: it tests whether flattening the stable learned connectome into edge-independent linear weights overlooks meso-scale brain organization.

## Literature rationale

HyBRiD motivates high-order ROI relationships and maximally informative/minimally redundant groups. BrainGNN motivates ROI selection and group-level consistency. CRGNN illustrates that learned graph structure can be task-aligned, while BrainOOD emphasizes stable structure selection for multi-site generalization. The 2026 ASD hypergraph study motivates complementary pairwise and higher-order information, and MCDGLN motivates filtering irrelevant connections. These lessons motivate only the fixed consensus readout; no cited full model is copied.

## Exact method

The original VarCoNet 1D CNN + Transformer, Conv1d kernel 4/stride 2, temporal-crop augmentation, InfoNCE objective, optimizer and scheduler are unchanged. Its stable learned-FC upper triangle is S ∈ R^73536 and the baseline is exactly Linear(73536, 2) + Softmax.

For each task, after baseline validation-BCE selection and freezing, consensus affinity is A_ij = mean over unique outer-training subjects |S_ij|. SpectralClustering uses n_clusters=32, affinity=precomputed, assign_labels=cluster_qr, eigen_solver=arpack, random_state=task.seed, n_jobs=1. Every upper-triangle edge is pooled once into H_pq = mean{S_ij: i∈M_p, j∈M_q}, yielding 32×33/2 = 528 features. Training-only mean/std normalization is applied to H. A zero-initialized single residual head δ=wᵀz(H) is trained for epochs 1..149 with Adam 5e-5 and BCELoss; treatment probabilities use baseline log-probabilities plus [-δ/2,+δ/2].

## Integrity and leakage controls

- Original encoder, InfoNCE and baseline classifier are unchanged.
- Module discovery accepts only `stable_unique_train` (`module_raw`); no labels, metadata, site, validation, test, Caltech or ABIDE-II data enter it.
- Duplicates are train-only; meta normalization, model selection and Youden thresholds are validation/train-only.
- Baseline and CHMR use tensor-identical encoder and stable-classifier states and identical selected epochs.
- ABIDE-II uses the complete baseline-eligible cohort (n=727) for both paired methods; predictions assert exact ID equality.
- No per-epoch test access; internal results are 10 repeat-level OOF aggregates over 100 folds.

## Results

| Scope | Baseline AUC | CHMR AUC | ΔAUC | Baseline BCE | CHMR BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
| Internal OOF | 0.7226 | 0.7223 | -0.0003 | 0.6125 | 0.6130 | 0.0006 |
| Caltech | 0.6561 | 0.6570 | 0.0009 | 0.6569 | 0.6566 | -0.0003 |
| ABIDE-II | 0.7373 | 0.7372 | -0.0002 | 0.6023 | 0.6027 | 0.0003 |

Internal repeat ΔAUC (CHMR − baseline): `[-0.000206, 0.000903, -0.000418, 0.000299, 0.000955, -0.000867, 0.000351, -0.001677, -0.000900, -0.001331]`

ABIDE-II repeat ΔAUC (CHMR − baseline): `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.001605, 0.000000, 0.000000]`

Baseline sanity: internal OOF AUC=0.7226 (reference 0.723); ABIDE-II AUC=0.7373 (reference 0.739); pipeline drift=False.

## Module diagnostics

- Fixed module count=32; meta dimension=528.
- Task-average module size min/mean/max=5.036 / 12.000 / 23.300; singleton≥8 count=0.
- Empty meta bins (mean)=0.000; affinity mean/std/max=0.17072638 / 0.04463240 / 0.82812344.
- Residual epoch-0 fraction=0.6364; epoch distribution=`{"0": 70, "114": 1, "117": 1, "129": 1, "136": 1, "148": 1, "149": 11, "16": 1, "17": 1, "2": 2, "25": 3, "28": 1, "30": 2, "39": 1, "40": 1, "45": 1, "49": 1, "5": 1, "6": 1, "60": 1, "73": 1, "75": 1, "8": 2, "80": 2, "86": 1}`.
- Residual-head L2 min/mean/max=0.000000 / 0.030124 / 0.156706.

## Runtime

10 workers (repeats 0,2,4,6,8 on GPU0; 1,3,5,7,9 on GPU1), six CPU threads per worker, full precision, no DDP/AMP/compile. Wall clock=1.067 h; summed worker compute=9.891 h; mean peak GPU allocation/reservation=1.401 / 1.676 GB. Atomic rolling latest/best checkpoints and append-only shards were used and removed after aggregation.

## Decision

**FAIL**

## Interpretation

Coarse-graining the stable learned connectome into train-derived consensus functional modules does not provide stable incremental ASD classification value beyond the original pairwise linear readout.

## Reproducibility

Configuration is AICHA/384 ROI, stable pairwise dimension 73536, 32 consensus modules, 528 meta features, 50 SSL epochs, 149 classifier/residual updates, 10 repeats × 10 internal folds, Caltech n=37 and ABIDE-II n=727. Final artifacts are `results/metrics.csv`, `results/summary.json`, `results/predictions.npz`, and `results/external_best_models.pt`.

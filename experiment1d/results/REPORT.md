# Experiment 1D

## Purpose and locked protocol

Experiment 1D tests three pre-specified disease-sharing readouts on top of one
audited original VarCoNet trajectory: Disease-Shared Low-rank Subspace (DSLS),
Cross-Site Disease-Shared Adapter (CSDA), and Individual-Conditioned Disease
Mask (ICDM). The baseline, encoder trajectory, classifier selection, splits,
validation threshold, and one-time test extraction are shared across all four
methods. No test labels or predictions are used for eligibility, fitting, or
selection.

The AICHA stable learned-FC vector has 73,536 edges from 384 ROIs. The baseline
is the original frozen `Linear(73536,2)+Softmax` classifier (label 0 ASD, label
1 HC). SSL runs for 50 epochs; at each trajectory epoch the classifier runs
updates 1--149 at Adam `5e-5`, with validation BCE selecting the encoder and
classifier. Each treatment has 149 Adam `1e-3` updates and can replace the
exact epoch-0 baseline only with strictly lower validation BCE.

DSLS fits an uncentered, fixed three-component PCA basis to unique reference
ASD-vs-HC deviations, then trains a 3→8→1 GELU residual (41 parameters).
CSDA uses one reference-only signed 384-ROI profile, a 384→32 projection and
zero-initialized head (12,321 parameters), with fixed cross-site same-diagnosis
SupCon (`tau=0.1`) plus full-training BCE. ICDM uses a reference-normalized
384→16→384 bias-free ROI gate, bounded to an edge-factor range of 0.905--1.105,
with 12,288 trainable parameters and BCE only. The baseline classifier remains
frozen in all treatment branches.

## Dataset and split audit

ABIDE-I contains 995 raw occurrences, 919 singleton names, 882 internal unique
subjects, 76 duplicate train-only occurrences, and 37 Caltech subjects. The
official ABIDE-I phenotype CSV maps all 995 occurrences to 20 sites; duplicate
names have consistent sites and all Caltech subjects map to `CALTECH`. ABIDE-II
is evaluated as the complete original 727-subject cohort. Ten repeats use
`random_state=42+repeat`; internal folds are ten stratified folds with a fixed
85/15 outer-train/validation split; the external reference split is 90/10.

## Results

| Scope | Baseline AUC | DSLS AUC | CSDA AUC | ICDM AUC | DSLS ΔAUC | CSDA ΔAUC | ICDM ΔAUC | Baseline BCE | DSLS BCE | CSDA BCE | ICDM BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Internal OOF | 0.7227 | 0.7225 | 0.7211 | 0.7227 | -0.0002 | -0.0016 | -0.0001 | 0.6123 | 0.6129 | 0.6139 | 0.6125 |
| Caltech | 0.6576 | 0.6576 | 0.6500 | 0.6570 | 0.0000 | -0.0076 | -0.0006 | 0.6575 | 0.6578 | 0.6631 | 0.6585 |
| ABIDE-II | 0.7371 | 0.7365 | 0.7386 | 0.7371 | -0.0006 | 0.0015 | 0.0000 | 0.6025 | 0.6034 | 0.6023 | 0.6025 |

### Mean paired deltas versus baseline

| Method | Internal ΔAUC | Caltech ΔAUC | ABIDE-II ΔAUC | Internal ΔBCE | Caltech ΔBCE | ABIDE-II ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
| dsls | -0.0002 | 0.0000 | -0.0006 | 0.0006 | 0.0003 | 0.0009 |
| csda | -0.0016 | -0.0076 | 0.0015 | 0.0017 | 0.0056 | -0.0002 |
| icdm | -0.0001 | -0.0006 | 0.0000 | 0.0003 | 0.0010 | 0.0000 |

### Ten-repeat paired internal/ABIDE-II ΔAUC

| Method | Internal OOF repeats 0--9 | ABIDE-II repeats 0--9 |
|---|---|---|
| dsls | 0.00138 -0.00084 -0.00056 -0.00139 -0.00163 0.00182 -0.00086 0.00055 0.00072 -0.00152 | 0.00000 -0.00031 0.00000 0.00000 0.00000 -0.00095 -0.00303 0.00000 -0.00005 -0.00137 |
| csda | 0.00038 0.00308 0.00314 -0.00344 -0.00918 0.00035 -0.00023 -0.00513 -0.00286 -0.00232 | 0.00445 -0.00202 0.00065 0.00424 0.00197 -0.00025 0.00686 0.00000 0.00021 -0.00063 |
| icdm | -0.00008 0.00004 -0.00003 -0.00090 -0.00028 0.00019 0.00022 -0.00063 0.00112 -0.00019 | 0.00034 0.00002 0.00000 0.00002 -0.00026 0.00000 0.00038 -0.00012 0.00000 -0.00015 |

Baseline sanity: internal OOF AUC=0.722736 (reference 0.723),
ABIDE-II AUC=0.737078 (reference 0.739),
pipeline drift=False.

## Mechanism diagnostics

- **dsls**: dsls_singular1_mean=781.619668, dsls_singular2_mean=683.333298, dsls_singular3_mean=590.785044, dsls_shared_energy_fraction_mean=0.048144, dsls_asd_score_mean_mean=0.074104, dsls_asd_score_std_mean=1.342478, dsls_hc_score_mean_mean=0.000000, dsls_hc_score_std_mean=0.999983, dsls_std_floor_mean=0.019468, dsls_parameter_l2_mean=2.141870, dsls_mean_abs_delta_mean=0.049415
- **csda**: csda_num_sites_mean=17.909091, csda_cross_site_positive_pairs_mean=213881.327273, csda_cross_site_same_y_cos_mean=0.038967, csda_same_site_same_y_cos_mean=0.064874, csda_opposite_y_cos_mean=-0.018802, csda_site_retrieval_accuracy_mean=0.262463, csda_profile_std_floor_mean=0.001354, csda_projection_l2_mean=7.920532, csda_head_l2_mean=0.376715, csda_mean_abs_delta_mean=0.102700
- **icdm**: icdm_mean_abs_roi_prompt_mean=0.032784, icdm_mean_abs_edge_log_gate_mean=0.032179, icdm_mean_edge_factor_mean=1.033588, icdm_std_edge_factor_mean=0.007173, icdm_factor_min_mean=0.962137, icdm_factor_max_mean=1.046651, icdm_fraction_gt1_mean=0.483072, icdm_mean_abs_fc_change_mean=0.005834, icdm_down_l2_mean=6.638895, icdm_up_l2_mean=2.214097

The DSLS diagnostic reports singular values, uncentered ASD shared-energy
fraction, score moments, and residual size. CSDA reports cross-site and
same-site same-diagnosis cosine similarity, opposite-diagnosis cosine,
site-retrieval accuracy, projection/head norms, and mean absolute delta. ICDM
reports ROI-prompt and edge-log-gate magnitude, factor range, fraction above
one, functional-connectivity change, and down/up norms. These are technical
readout diagnostics, not subtype labels.

## Decision

The pre-registered route rule is: STRONG_POSITIVE requires internal mean
ΔAUC ≥0.005, at least 7/10 positive repeats, ABIDE-II mean ΔAUC ≥0.003, and
both mean ΔBCE values ≤0.005; PROMISING uses either the specified internal
or external alternative; otherwise FAIL. If one route is positive it is the
sole candidate for the next experiment; multiple candidates are ranked by
ABIDE-II ΔAUC, internal ΔAUC, positive-repeat count, then BCE. If all fail the
overall decision is `ALL_FAIL`.

- **dsls**: `FAIL`; internal OOF ΔAUC=-0.000232, positive repeats=4/10, ABIDE-II ΔAUC=-0.000570, internal/ABIDE-II ΔBCE=0.000600/0.000913.
- **csda**: `FAIL`; internal OOF ΔAUC=-0.001622, positive repeats=4/10, ABIDE-II ΔAUC=0.001548, internal/ABIDE-II ΔBCE=0.001673/-0.000231.
- **icdm**: `FAIL`; internal OOF ΔAUC=-0.000055, positive repeats=4/10, ABIDE-II ΔAUC=0.000022, internal/ABIDE-II ΔBCE=0.000256/0.000034.

Overall decision: **ALL_FAIL**; selection record:
`{"candidate_methods": [], "overall_decision": "ALL_FAIL", "selected_method": null}`.

The preceding experiment conclusions remain unchanged: Experiment 1A's
prototype route failed to improve classification; Experiment 1B-R's corrected
control-referenced branch was near-neutral and its original post-PCA epoch-0
state was not an exact baseline; Experiment 1C SNDE activated a PC1-based
regional gate but did not provide stable disease generalization. Experiment 1D
therefore evaluates three independent, locked readouts rather than rescuing a
failed route.

## Integrity, runtime, and artifacts

All 520 metric rows (100 internal folds + 10 internal OOF + 10 Caltech + 10
ABIDE-II for each of four methods) passed paired-ID and label checks. The
prediction archive contains one shared extraction per scope and exact paired
subject IDs for all methods. Root and prior experiment source hashes were
checked before smoke and after aggregation: `True`.
Workers used full precision, six CPU threads, and repeats 0/2/4/6/8 on GPU0
and 1/3/5/7/9 on GPU1; no DDP, DataParallel, AMP, or compile was used.
Wall clock=1.589 h; summed task
compute=14.338 h; mean peak allocated /
reserved GPU=2.496 /
3.293 GB.

Artifacts: `results/metrics.csv`, `results/predictions.npz`,
`results/summary.json`, `results/REPORT.md`, and
`results/external_best_models.pt`. Rolling worker checkpoints are resume-only
and are removed after successful aggregation.

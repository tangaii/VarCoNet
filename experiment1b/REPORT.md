# Experiment 1B

## Control-Referenced Soft Multi-Boundary VarCoNet (CRSM-VarCoNet)

## Purpose and scope

Experiment 1A modified encoder geometry through subtype prototypes and did not improve the paired classification results. Experiment 1B therefore preserves the original VarCoNet encoder and selected stable linear classifier exactly, and tests only whether one frozen disease direction can be replaced by two softly combined, control-referenced directions. This is a new self-contained experiment in `xiaolunwen/experiment1b`; no root source or prior experiment directory was modified.

## Literature and source audit

- [HYDRA](https://pubmed.ncbi.nlm.nih.gov/26923371/) provides the conceptual precedent for a common control reference and multiple disease-facing boundaries: its public implementation labels patients `+1`, controls `-1`, assigns patients by maximum face score, and keeps controls common across faces. Its MATLAB solver, weighted SVMs, hard assignments, consensus clustering and DPP initialization were not copied or used.
- The 2025 diagnosis-informed ASD HYDRA study reported two reliable hyper- and hypoconnectivity patterns in a large functional-connectivity cohort; the [article record](https://doi.org/10.1016/j.pnpbp.2025.111452) is rationale for examining a fixed two-boundary control-referenced readout, not a claim that this experiment discovers subtypes.
- The 2026 cross-species study also reported two dominant hypo/hyperconnectivity patterns in human autism data ([Nature Neuroscience](https://doi.org/10.1038/s41593-026-02287-z)), while the 2025 functional-deviation study found two ASD profiles ([Molecular Psychiatry](https://doi.org/10.1038/s41380-025-03086-x)). These motivate `K=2` as a conservative fixed test.
- The [eLife FC-subtype study](https://doi.org/10.7554/eLife.56257) found continuous subtype assignments more robust than discrete assignments. Accordingly, CRSM uses soft responsibilities only; it never trains with argmax assignments.
- Smile-GAN was read only for its control-to-patient heterogeneity framing; no GAN, mapping, discriminator, reconstruction or clustering code was reused. BrainSCL was read only to distinguish a prototype-guided contrastive route already tested in Experiment 1A; no prototype loss or representation change enters this experiment.

## Exact locked baseline

The local audited implementation is AICHA/384 ROI VarCoNet with Conv1d kernel 4 and stride 2, stable upper-triangle learned FC dimension 73,536, the repository's temporal crop augmentation and InfoNCE objective, Adam with the audited warmup/cosine scheduler, 50 SSL epochs, and a `Linear(73536,2) + Softmax` classifier. Class `0` is ASD and class `1` is HC. The classifier uses Adam `5e-5` for updates 1–149, selects validation BCE, and then selects one baseline-best encoder epoch. The treatment has no independent encoder or baseline classifier selection.

## Exact CRSM method

For each task, `reference_raw` contains only unique singleton outer-training subjects; its duplicate occurrences are excluded. `train_raw = reference_raw + 76` train-only duplicate occurrences. No validation, internal test, Caltech, ABIDE-II, diagnosis metadata other than training labels required by the supervised loss, site, motion, TR, age, sex or IQ enters the reference centroid or PCA initialization.

Let `x` be the stable learned FC, `mu_HC` the mean unique-reference HC vector, and `m0 = z_ASD - z_HC = (W_ASD-W_HC)x + (b_ASD-b_HC)` from the frozen selected classifier. The model has exactly 147,074 trainable parameters: `delta_weight ∈ R^(2×73536)` and `delta_bias ∈ R^2`. It uses

`m_k = m0 + delta_w_k^T (x-mu_HC) + delta_b_k`, `k=1,2`, `tau=1.0`, and
`m_mix = tau [logsumexp(m_1/tau,m_2/tau) - log(2)]`.

Thus `p_ASD=sigmoid(m_mix)` and `p_HC=sigmoid(-m_mix)`. The loss is the pre-specified ASD `softplus(-m_mix)` plus HC mean-over-experts `softplus(m_k)`, weighted by the observed ASD and HC counts. There is no prototype, Sinkhorn, entropy/balance/diversity/orthogonality term, hard assignment, spectral clustering, high-order module, meta-feature, GAN, external fitting, or test-time adaptation.

Before training, zero deltas are asserted to reproduce baseline margin, probabilities, BCE and CRSM loss to `<=1e-6`. The fixed symmetry break takes the first PCA direction of unique-reference ASD residuals `(x-mu_HC)` under `torch.random.fork_rng`, and initializes `delta_w_1=+0.01*sigma_margin/sigma_projection*v1`, `delta_w_2=-...`, zero biases. If either scale standard deviation is below `1e-8`, the experiment terminates as `MECHANISM_INIT_FAIL`; no random fallback is allowed. CRSM updates 1–149 use Adam `5e-5`; initialized epoch 0 is a legal validation candidate.

## Integrity and leakage controls

- Internal evaluation is 10 repeats × 10 stratified folds (`random_state=42+repeat`), with 15% inner validation (`random_state=42`). External fitting uses the pre-specified 90/10 unique split (`random_state=42+repeat`).
- Baseline and treatment use tensor-identical selected encoder and stable-classifier states. Thresholds are separate, but each comes from validation-only Youden selection.
- Test data are touched only once after all SSL/classifier/CRSM selections. ABIDE-II remains the complete original cohort (`n=727`) for both paired methods; every repeat asserts identical subject IDs.
- A real smoke test compared local and root encoder outputs on an ABIDE-I batch and checked stable shape, margin/pHC parity, zero-delta BCE/loss parity, finite HC centroid/PCA, exact 1% initialization, 147,074 parameters, and finite forward/backward.
- Rolling `latest`/`best` task checkpoints hold RNG state, one shared SSL trajectory, selected baseline state, centroid, PCA diagnostics, current CRSM state and best delta state. They are resume-only and are deleted after successful aggregation; only external selected models remain.
- Source integrity was asserted before smoke and after aggregation with pre-recorded hashes for root `model_scripts`, root utility/classification scripts, and Experiment 1/2/2B/3 directories. The post-run audit passed: `True`.

## Results

| Scope | Baseline AUC | CRSM AUC | ΔAUC | Baseline BCE | CRSM BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
| Internal OOF | 0.7229 | 0.7230 | 0.0001 | 0.6123 | 0.6123 | -0.0000 |
| Caltech | 0.6579 | 0.6585 | 0.0006 | 0.6568 | 0.6570 | 0.0002 |
| ABIDE-II | 0.7371 | 0.7372 | 0.0000 | 0.6025 | 0.6023 | -0.0002 |

Internal repeat ΔAUC (CRSM − baseline): `[0.000702, 0.000077, 0.000392, -0.000010, 0.000266, -0.000413, 0.000671, -0.000299, -0.000377, 0.000114]`

ABIDE-II repeat ΔAUC (CRSM − baseline): `[0.000983, 0.000008, 0.000000, 0.000077, 0.000000, -0.000138, -0.000004, -0.000038, 0.000031, -0.000461]`

Baseline sanity: internal OOF AUC=0.7229 (reference 0.723); ABIDE-II AUC=0.7371 (reference 0.739); pipeline drift=False.

## CRSM diagnostics

- Selected CRSM epoch-0 fraction=0.8273; epoch distribution=`{"0": 91, "1": 8, "2": 5, "3": 4, "5": 1, "7": 1}`.
- HC centroid norm mean=24.443620; PCA unit norm mean=1.00000000; symmetry scale mean=0.002074128.
- Expert mass mean: `(0.501307, 0.498693)`; hard fractions: `(0.522228, 0.477772)`; responsibility entropy=0.693038.
- Delta L2 means: `(0.006900, 0.006802)`; weight cosine=-0.662237.
- Internal repeat ΔAUC standard deviation=0.000403; pre-specified high-instability flag=False; epoch-0/no-gain flag=False.

## Runtime

Ten independent workers use full precision without DDP, DataParallel, AMP or `torch.compile`: repeats 0,2,4,6,8 on GPU0 and 1,3,5,7,9 on GPU1, with six CPU threads per worker. Wall clock=0.898 h; summed task compute=7.828 h; mean peak GPU allocation/reservation=1.447 / 2.242 GB.

## Final decision

**FAIL**

## Interpretation

Preserving VarCoNet individualized representations while replacing a single disease direction with two soft control-referenced boundaries does not yield stable ASD classification gains.

The result does not assign biological ASD subtypes and should not be interpreted as such. It evaluates only the pre-specified classification-readout hypothesis.

## Reproducibility artifacts

`results/metrics.csv` contains 260 rows (100 internal folds + 10 internal OOF + 10 Caltech + 10 ABIDE-II per method); `results/summary.json` records aggregate/decision/integrity fields; `results/predictions.npz` contains paired final-scope IDs, labels, probabilities and thresholds; `results/external_best_models.pt` stores the 10 paired external baseline/CRSM selected states. Logs are append-only in `logs/`.

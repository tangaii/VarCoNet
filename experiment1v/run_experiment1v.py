#!/usr/bin/env python3
"""Experiment 1V: direct identity--shared-disease mechanism audit.

The runner is self-contained under ``xiaolunwen/experiment1v``.  It copies
the audited VarCoNet/SSL/classifier implementation locally and never imports
an earlier experiment at runtime.  All model selection remains validation
only; labels enter only the pre-registered disease-direction/evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import Adam


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOG_DIR = EXPERIMENT_DIR / "logs"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
CACHE_DIR = EXPERIMENT_DIR / "dataset_cache"
WORK_DIR = EXPERIMENT_DIR / "work"
SHARD_DIR = WORK_DIR / "shards"
sys.path.insert(0, str(EXPERIMENT_DIR))

from disease_axis import (  # noqa: E402
    asd_auc,
    axis_auc,
    cohens_d_vector,
    fit_shared_axis,
    roi_convergence,
    score_axis,
    strict_loso,
)
from fingerprint import fingerprint_metrics, vector_pattern_similarity  # noqa: E402
from model_scripts.VarCoNet import VarCoNet  # noqa: E402
from model_scripts.classifier import MLP  # noqa: E402
from model_scripts.scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from representations import (  # noqa: E402
    EDGE_DIM,
    MAX_LENGTH,
    batch_representation,
    batch_view_representations,
)
from stable_windows import (  # noqa: E402
    NUM_WINDOWS,
    WINDOW_SIZES,
    build_legacy_test_windows,
    build_stable_windows,
    latent_valid_tokens,
    valid_length,
    window_metadata,
)
from statistics import (  # noqa: E402
    BOOTSTRAP_N,
    PERMUTATION_N,
    SEED,
    auc_label_permutation,
    benjamini_hochberg,
    ci_summary,
    edge_replication,
    geometry_gap,
    paired_geometry_bootstrap,
    paired_stratified_auc_bootstrap,
    site_macro_bootstrap,
    stratified_auc_bootstrap,
)
from trajectory import (  # noqa: E402
    FIXED_EPOCHS,
    SPLIT_HALF_REPEATS,
    make_split_half_indices,
    split_half_stability,
    trajectory_epochs,
)
from utils import DualBranchContrast, InfoNCE, augment, removeDuplicates  # noqa: E402


CONFIG: dict[str, Any] = {
    "atlas": "AICHA",
    "roi_count": 384,
    "stable_dim": EDGE_DIM,
    "batch_size": 64,
    "min_length": 80,
    "epochs": 50,
    "warm_up_epochs": 10,
    "epochs_cls": 150,
    "lr_cls": 5e-5,
    "checkpoint_interval": 10,
    "cpu_threads": 6,
    "view_length": 80,
    "bootstrap_n": BOOTSTRAP_N,
    "permutation_n": PERMUTATION_N,
    "top_edge_fraction": 0.01,
}

P0 = "pearson_windowavg"
P1 = "pearson_full"
V0 = "varconet_selected"
V1 = "varconet_epoch50"
PRIMARY_REPRESENTATIONS = (P0, V0)
ALL_REPRESENTATIONS = (P0, P1, V0, V1)

METRIC_COLUMNS = [
    "representation", "scope", "repeat", "fold", "n_subjects",
    "disease_axis_auc", "disease_geometry_gap", "fingerprint_n",
    "fingerprint_top1", "fingerprint_top5", "fingerprint_mrr",
    "fingerprint_same_similarity", "fingerprint_different_similarity",
    "fingerprint_identity_gap", "within_site_top1", "within_site_same_dx_top1",
    "topedge_sign_agreement", "topedge_spearman", "topedge_cosine",
    "roi_signed_spearman", "roi_absburden_spearman", "selected_encoder_epoch",
    "task_seconds", "status",
]
TRAJECTORY_COLUMNS = [
    "repeat", "epoch", "is_selected", "scope", "fingerprint_top1", "identity_gap",
    "disease_axis_auc", "disease_geometry_gap", "split_half_disease_cosine",
    "n_identity_subjects", "n_disease_subjects", "status",
]


def print_preregistered_plan() -> None:
    """Emit the locked protocol before the full worker launch."""
    lines = [
        "EXPERIMENT 1V PREREGISTERED AUDIT PLAN",
        "H1: test whether a cross-subject, cross-site ASD connectivity pattern transfers to ABIDE-II.",
        "H2: test whether VarCoNet SSL increases subject identity while weakening that shared structure.",
        "Representations: P0=pearson_windowavg (primary), P1=pearson_full, V0=varconet_selected, V1=varconet_epoch50.",
        "Stable-window audit: one complete zero-padded 320-row scan, exactly matching Experiment1D actual extraction; Pearson uses its identical real valid slice.",
        "Fingerprint: deterministic first/last 80 valid points, T_valid>=160; TOP1/TOP5/MRR/same-different gap plus within-site and within-site+diagnosis sensitivity.",
        "Disease geometry: cross-site same-diagnosis minus cross-site different-diagnosis vector-pattern similarity; paired stratified subject bootstrap N=2000.",
        "Disease axis: train-only standardizer and equally weighted eligible-site ASD-minus-HC directions (minimum 10 per class/site; >=4 sites).",
        "H1 endpoints: ABIDE-II Pearson axis AUC + 2,000 subject bootstrap + 5,000 label permutation; Pearson strict internal LOSO + 5,000 site bootstrap; fixed top-1% edge replication + 5,000 label permutations.",
        "H2 endpoints: paired ABIDE-II identity bootstrap, paired geometry bootstrap, paired external AUC bootstrap, and fixed 0/1/5/10/20/30/40/50 SSL trajectory audit.",
        "Trajectory: 10 external repeats; epoch-0 captured before any optimizer step; 200 fixed site×diagnosis split-halves per state; selected state is validation-BCE only.",
        "Internal endpoint audit: 10 repeats × 10 outer folds; duplicates are SSL-training-only and absent from all audit reference/test metrics.",
        "External cohorts: Caltech n=37 supportive; ABIDE-II all 727 original rows primary disease endpoint; no test-driven selection or label/prediction-based exclusion.",
        "Integrity: root/prior hashes fixed; IDs paired across representations; no DDP/DataParallel/AMP/compile; full precision; 6 CPU threads/worker.",
        "Decision H1: SUPPORTED iff H1-A and H1-B or H1-C; PARTIAL iff external AUC>0.5 with B or C; otherwise NOT SUPPORTED.",
        "Decision H2 (only if H1 is not NOT SUPPORTED): STRONG iff identity CI>0, geometry or AUC CI<0, and >=7/10 trade-off repeats; otherwise PARTIAL/NOT_SUPPORTED per locked rules.",
    ]
    print("\n".join(lines), flush=True)


@dataclass
class Cohort:
    abide1: np.ndarray
    abide1_names: list[str]
    abide1_labels: np.ndarray  # 0 ASD, 1 HC
    abide1_sites: np.ndarray
    abide1_site_ids: np.ndarray
    duplicate_raw: np.ndarray
    duplicate_names: list[str]
    internal_raw: np.ndarray
    internal_names: list[str]
    internal_labels: np.ndarray
    internal_sites: np.ndarray
    caltech_raw: np.ndarray
    caltech_names: list[str]
    caltech_labels: np.ndarray
    caltech_sites: np.ndarray
    abide2: np.ndarray
    abide2_names: list[str]
    abide2_labels: np.ndarray  # 0 ASD, 1 HC
    abide2_sites: np.ndarray
    abide2_site_ids: np.ndarray
    abide1_valid_lengths: np.ndarray
    abide2_valid_lengths: np.ndarray
    metadata: dict[str, Any]


@dataclass
class Task:
    repeat: int
    scope: str
    fold: str
    seed: int
    reference_raw: np.ndarray
    reference_labels: np.ndarray
    reference_sites: np.ndarray
    train_raw: np.ndarray
    train_names: list[str]
    train_labels: np.ndarray
    val_raw: np.ndarray
    val_labels: np.ndarray
    test_raw: np.ndarray
    test_labels: np.ndarray


def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(CONFIG["cpu_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_to_cpu(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def clone_state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(relative: str) -> str:
    directory = REPO_ROOT / relative
    lines: list[str] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if set(path.relative_to(directory).parts) & {"__pycache__", "logs", "results", "checkpoints", "dataset_cache", "work"}:
            continue
        lines.append(f"{file_hash(path)}  {path.relative_to(REPO_ROOT).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def source_hashes() -> dict[str, str]:
    values: dict[str, str] = {}
    for relative in ("model_scripts", "xiaolunwen/experiment1", "xiaolunwen/experiment1b", "xiaolunwen/experiment1c", "xiaolunwen/experiment1d", "xiaolunwen/experiment2", "xiaolunwen/experiment2b", "xiaolunwen/experiment3"):
        values[relative] = directory_digest(relative)
    for relative in ("utils.py", "ASD_classification_ABIDEI.py", "ASD_classification_ABIDEII.py"):
        values[relative] = file_hash(REPO_ROOT / relative)
    return values


def _load_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _load_stack(dataset_name: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{dataset_name}_nilearn_AICHA.npy"
    if cached.exists():
        return np.load(cached, mmap_mode="r", allow_pickle=False)
    source = DATASET_DIR / dataset_name / f"{dataset_name}_nilearn_AICHA.npz"
    with np.load(source, allow_pickle=False) as archive:
        array = np.stack([np.asarray(archive[key], dtype=np.float32) for key in archive.files], axis=0)
    temporary = cached.with_suffix(".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, cached)
    return np.load(cached, mmap_mode="r", allow_pickle=False)


def _subject_number(name: str) -> int:
    values = re.findall(r"\d+", str(name))
    if not values:
        raise RuntimeError(f"cannot parse subject ID from {name!r}")
    return int("".join(values))


def _official_metadata(path: Path) -> dict[int, dict[str, str]]:
    records: dict[int, dict[str, str]] = {}
    # The ABIDE-II official file contains a Latin-1 non-breaking character.
    for encoding in ("utf-8-sig", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    raw = str(row.get("SUB_ID", "")).strip()
                    if not raw:
                        continue
                    sid = int(float(raw))
                    site = str(row.get("SITE_ID", "")).strip().upper()
                    dx = str(row.get("DX_GROUP", "")).strip()
                    if not site or dx not in {"1", "2"}:
                        continue
                    previous = records.get(sid)
                    value = {"site": site, "dx": dx}
                    if previous is not None and previous != value:
                        raise RuntimeError(f"conflicting official metadata for SUB_ID={sid}: {previous} vs {value}")
                    records[sid] = value
            if records:
                return records
        except UnicodeDecodeError:
            records = {}
            continue
    raise RuntimeError(f"could not parse official phenotype CSV {path}")


def _map_rows(names: list[str], raw_labels: np.ndarray, metadata: dict[int, dict[str, str]], label_name: str) -> tuple[np.ndarray, np.ndarray]:
    sites: list[str] = []; expected_raw: list[int] = []
    for name in names:
        record = metadata.get(_subject_number(name))
        if record is None:
            raise RuntimeError(f"SITE_METADATA_FAIL: {label_name} missing official mapping for {name}")
        sites.append(record["site"])
        # Repository labels encode DX_GROUP=1 (ASD) as 1 and DX_GROUP=2 (HC) as 0.
        expected_raw.append(1 if record["dx"] == "1" else 0)
    expected = np.asarray(expected_raw, dtype=np.int64)
    if not np.array_equal(np.asarray(raw_labels, dtype=np.int64), expected):
        mismatches = np.where(np.asarray(raw_labels, dtype=np.int64) != expected)[0]
        raise RuntimeError(f"LABEL_MAPPING_FAIL {label_name}: {len(mismatches)} official diagnosis mismatches")
    # Experiment 1V locks the requested convention: 0=ASD, 1=HC.
    labels = 1 - expected
    return labels.astype(np.int64), np.asarray(sites, dtype="<U64")


def _valid_lengths(data: np.ndarray) -> np.ndarray:
    return np.asarray([valid_length(data[index]) for index in range(len(data))], dtype=np.int64)


def load_cohort() -> Cohort:
    abide1 = _load_stack("ABIDEI"); abide2 = _load_stack("ABIDEII")
    names1 = _load_names(DATASET_DIR / "ABIDEI" / "ABIDEI_nilearn_names.txt")
    names2 = _load_names(DATASET_DIR / "ABIDEII" / "ABIDEII_nilearn_names.txt")
    raw1 = np.load(DATASET_DIR / "ABIDEI" / "ABIDEI_nilearn_classes.npy", allow_pickle=False).astype(np.int64)
    raw2 = np.load(DATASET_DIR / "ABIDEII" / "ABIDEII_nilearn_classes.npy", allow_pickle=False).astype(np.int64)
    if abide1.shape != (995, 320, 384) or abide2.shape != (727, 320, 384):
        raise RuntimeError(f"unexpected AICHA shapes: {abide1.shape}, {abide2.shape}")
    if len(names1) != len(raw1) or len(raw1) != 995 or len(names2) != len(raw2) or len(raw2) != 727:
        raise RuntimeError("data/name/label alignment drift")
    meta1_path = DATASET_DIR / "ABIDEI" / "Phenotypic_V1_0b_preprocessed1.csv"
    meta2_path = EXPERIMENT_DIR / "metadata" / "ABIDEII_Composite_Phenotypic.csv"
    meta1, meta2 = _official_metadata(meta1_path), _official_metadata(meta2_path)
    labels1, sites1 = _map_rows(names1, raw1, meta1, "ABIDE-I")
    labels2, sites2 = _map_rows(names2, raw2, meta2, "ABIDE-II")
    unique_names, counts = np.unique(np.asarray(names1), return_counts=True)
    duplicate_values = unique_names[counts > 1]
    duplicate_raw = np.asarray([index for value in duplicate_values for index in np.where(np.asarray(names1) == value)[0].tolist()], dtype=np.int64)
    duplicate_names = [names1[index] for index in duplicate_raw.tolist()]
    singleton_names = [str(value) for value in unique_names[counts == 1].tolist()]
    name_to_raw = {name: int(np.where(np.asarray(names1) == name)[0][0]) for name in singleton_names}
    caltech_names = [f"sub-00{value}" for value in range(51456, 51494) if f"sub-00{value}" in name_to_raw]
    if len(caltech_names) != 37:
        raise RuntimeError(f"Caltech count drifted: {len(caltech_names)}")
    caltech_raw = np.asarray([name_to_raw[name] for name in caltech_names], dtype=np.int64)
    internal_names = [name for name in singleton_names if name not in set(caltech_names)]
    internal_raw = np.asarray([name_to_raw[name] for name in internal_names], dtype=np.int64)
    if len(singleton_names) != 919 or len(internal_raw) != 882 or len(duplicate_raw) != 76:
        raise RuntimeError("ABIDE-I duplicate/internal cohort counts drifted")
    if not np.all(sites1[caltech_raw] == "CALTECH"):
        raise RuntimeError("Caltech official site mapping failed")
    site_names1 = sorted(set(sites1.tolist())); site_ids1 = np.asarray([site_names1.index(site) for site in sites1], dtype=np.int64)
    site_names2 = sorted(set(sites2.tolist())); site_ids2 = np.asarray([site_names2.index(site) for site in sites2], dtype=np.int64)
    metadata = {
        "abide1_phenotype": str(meta1_path), "abide1_phenotype_sha256": file_hash(meta1_path),
        "abide2_phenotype": str(meta2_path), "abide2_phenotype_sha256": file_hash(meta2_path),
        "abide1_site_mapping_complete": True, "abide2_site_mapping_complete": True,
        "abide1_site_count": len(site_names1), "abide2_site_count": len(site_names2),
        "label_convention": "0=ASD, 1=HC (repository class labels recoded after official DX_GROUP audit)",
    }
    return Cohort(
        abide1=abide1, abide1_names=names1, abide1_labels=labels1, abide1_sites=sites1, abide1_site_ids=site_ids1,
        duplicate_raw=duplicate_raw, duplicate_names=duplicate_names, internal_raw=internal_raw,
        internal_names=internal_names, internal_labels=labels1[internal_raw], internal_sites=sites1[internal_raw],
        caltech_raw=caltech_raw, caltech_names=caltech_names, caltech_labels=labels1[caltech_raw], caltech_sites=sites1[caltech_raw],
        abide2=abide2, abide2_names=names2, abide2_labels=labels2, abide2_sites=sites2, abide2_site_ids=site_ids2,
        abide1_valid_lengths=_valid_lengths(abide1), abide2_valid_lengths=_valid_lengths(abide2), metadata=metadata,
    )


def make_tasks(cohort: Cohort, repeat: int) -> list[Task]:
    tasks: list[Task] = []; raw_to_name = {int(raw): name for raw, name in zip(cohort.internal_raw, cohort.internal_names)}
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + int(repeat))
    for fold, (outer_train_idx, test_idx) in enumerate(splitter.split(cohort.internal_raw, cohort.internal_labels)):
        outer_raw, outer_y = cohort.internal_raw[outer_train_idx], cohort.internal_labels[outer_train_idx]
        train_raw, val_raw, train_y, val_y, _, _ = train_test_split(
            outer_raw, outer_y, np.arange(len(outer_raw)), test_size=0.15, random_state=42, stratify=outer_y,
        )
        if np.intersect1d(train_raw, cohort.duplicate_raw).size:
            raise RuntimeError("duplicate occurrence leaked into unique reference")
        tasks.append(Task(
            repeat=int(repeat), scope="internal_fold", fold=str(fold), seed=100000 + int(repeat) * 100 + int(fold),
            reference_raw=np.asarray(train_raw, dtype=np.int64), reference_labels=np.asarray(train_y, dtype=np.int64),
            reference_sites=cohort.abide1_sites[train_raw].copy(),
            train_raw=np.concatenate((cohort.duplicate_raw, np.asarray(train_raw, dtype=np.int64))),
            train_names=cohort.duplicate_names + [raw_to_name[int(raw)] for raw in train_raw],
            train_labels=np.concatenate((cohort.abide1_labels[cohort.duplicate_raw], np.asarray(train_y, dtype=np.int64))),
            val_raw=np.asarray(val_raw, dtype=np.int64), val_labels=np.asarray(val_y, dtype=np.int64),
            test_raw=np.asarray(cohort.internal_raw[test_idx], dtype=np.int64), test_labels=np.asarray(cohort.internal_labels[test_idx], dtype=np.int64),
        ))
    train_raw, val_raw, train_y, val_y, _, _ = train_test_split(
        cohort.internal_raw, cohort.internal_labels, np.arange(len(cohort.internal_raw)), test_size=0.10,
        random_state=42 + int(repeat), stratify=cohort.internal_labels,
    )
    tasks.append(Task(
        repeat=int(repeat), scope="external", fold="external", seed=200000 + int(repeat),
        reference_raw=np.asarray(train_raw, dtype=np.int64), reference_labels=np.asarray(train_y, dtype=np.int64),
        reference_sites=cohort.abide1_sites[train_raw].copy(),
        train_raw=np.concatenate((cohort.duplicate_raw, np.asarray(train_raw, dtype=np.int64))),
        train_names=cohort.duplicate_names + [raw_to_name[int(raw)] for raw in train_raw],
        train_labels=np.concatenate((cohort.abide1_labels[cohort.duplicate_raw], np.asarray(train_y, dtype=np.int64))),
        val_raw=np.asarray(val_raw, dtype=np.int64), val_labels=np.asarray(val_y, dtype=np.int64),
        test_raw=cohort.caltech_raw.copy(), test_labels=cohort.caltech_labels.copy(),
    ))
    return tasks


def make_model_config() -> tuple[dict[str, int], dict[str, Any]]:
    with (EXPERIMENT_DIR / "best_params_VarCoNet_AICHA.pkl").open("rb") as handle:
        params = pickle.load(handle)
    config = {"layers": int(params["layers"]), "n_heads": int(params["n_heads"]),
              "dim_feedforward": int(params["dim_feedforward"]), "max_length": MAX_LENGTH}
    return config, params


def create_encoder(device: torch.device) -> VarCoNet:
    config, _ = make_model_config()
    return VarCoNet(config, int(CONFIG["roi_count"])).to(device)


def finite_gradients(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())


def paired_batch_schedule(names: list[str], n_items: int, epochs: int, seed: int) -> list[list[list[int]]]:
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed)); schedule: list[list[list[int]]] = []
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(int(n_items), generator=generator).tolist(); batches: list[list[int]] = []
        for batch_index, start in enumerate(range(0, len(permutation), int(CONFIG["batch_size"]))):
            selected = [int(index) for index in permutation[start : start + int(CONFIG["batch_size"])]]
            saved = random.getstate(); random.seed(int(seed) + epoch * 1000003 + batch_index)
            selected = removeDuplicates(names, selected); random.setstate(saved); batches.append([int(index) for index in selected])
        schedule.append(batches)
    return schedule


def extract_old_stable(encoder: VarCoNet, data: np.ndarray, raw_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    """Exact Experiment-1D full-padded stable feature extraction."""
    pieces: list[torch.Tensor] = []; encoder.eval()
    with torch.no_grad():
        for start in range(0, len(raw_indices), int(CONFIG["batch_size"])):
            chosen = np.asarray(raw_indices[start : start + int(CONFIG["batch_size"])], dtype=np.int64)
            x = torch.from_numpy(np.asarray(data[chosen], dtype=np.float32)).to(device)
            pieces.append(encoder(x).detach().cpu().float())
    if not pieces:
        raise RuntimeError("cannot extract stable features from empty split")
    result = torch.cat(pieces, dim=0)
    if result.ndim != 2 or result.shape[1] != EDGE_DIM or not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"stable feature integrity failure {tuple(result.shape)}")
    return result


def validation_threshold(labels: np.ndarray, probability_hc: np.ndarray) -> float:
    fpr, tpr, threshold = roc_curve(np.asarray(labels, dtype=np.int64), np.asarray(probability_hc, dtype=float))
    youden = tpr - fpr; candidates = np.where(np.isclose(youden, np.nanmax(youden), rtol=0.0, atol=1e-12))[0]
    finite = [int(index) for index in candidates.tolist() if np.isfinite(threshold[index])]
    choice = finite[0] if finite else int(candidates[0])
    return float(threshold[choice]) if np.isfinite(threshold[choice]) else 0.5


def fit_baseline_classifier(stable_train: torch.Tensor, y_train: np.ndarray, stable_val: torch.Tensor, y_val: np.ndarray, device: torch.device, seed: int) -> dict[str, Any]:
    """Original validation-BCE encoder-selection classifier, unchanged in form."""
    set_all_seeds(seed)
    classifier = MLP(int(stable_train.shape[1]), 2).to(device); optimizer = Adam(classifier.parameters(), lr=float(CONFIG["lr_cls"]))
    criterion = torch.nn.BCELoss(); train_x = stable_train.to(device); val_x = stable_val.to(device)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=device); val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)
    target_train = F.one_hot(train_y, num_classes=2).float(); target_val = F.one_hot(val_y, num_classes=2).float()
    best_bce, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, int(CONFIG["epochs_cls"])):
        classifier.train(); optimizer.zero_grad(); loss = criterion(classifier(train_x), target_train); loss.backward()
        if not bool(torch.isfinite(loss)) or not finite_gradients(classifier):
            raise FloatingPointError("non-finite baseline classifier loss/gradient")
        optimizer.step(); classifier.eval()
        with torch.no_grad():
            val_bce = float(criterion(classifier(val_x), target_val).detach().cpu())
        if val_bce < best_bce:
            best_bce, best_epoch, best_state = val_bce, int(epoch), clone_state_cpu(classifier)
    if best_state is None or best_epoch is None:
        raise RuntimeError("baseline classifier did not select a validation state")
    classifier.load_state_dict(best_state); classifier.eval()
    with torch.no_grad():
        val_probs = classifier(val_x)[:, 1].detach().cpu().numpy().astype(np.float64)
    del classifier, optimizer, train_x, val_x
    if device.type == "cuda": torch.cuda.empty_cache()
    return {"classifier_state": best_state, "val_bce": float(best_bce), "classifier_epoch": int(best_epoch),
            "val_probs": val_probs, "threshold": validation_threshold(y_val, val_probs)}


def train_ssl_epoch(encoder: VarCoNet, contrast: DualBranchContrast, optimizer: torch.optim.Optimizer, cohort: Cohort, task: Task, batches: list[list[int]], device: torch.device) -> float:
    encoder.train(); losses: list[float] = []
    for selected in batches:
        local = np.asarray(selected, dtype=np.int64)
        batch = torch.from_numpy(np.asarray(cohort.abide1[task.train_raw[local]], dtype=np.float32))
        views = augment(batch, [int(CONFIG["min_length"]), MAX_LENGTH], device)
        optimizer.zero_grad(); z1, z2 = encoder(views[0]), encoder(views[1]); loss = contrast(z1, z2)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite SSL loss")
        loss.backward()
        if not finite_gradients(encoder):
            raise FloatingPointError("non-finite SSL gradient")
        optimizer.step(); losses.append(float(loss.detach().cpu()))
        del batch, views, z1, z2, loss
    if not losses:
        raise RuntimeError("empty SSL epoch")
    return float(np.mean(losses))


def _checkpoint_path(task: Task) -> Path:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"r{task.repeat}_{task.scope}_{task.fold}")
    return CHECKPOINT_DIR / f"{token}_latest.pt"


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors after a CPU checkpoint restore."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _task_identity(task: Task) -> dict[str, Any]:
    return {"repeat": int(task.repeat), "scope": task.scope, "fold": task.fold, "seed": int(task.seed)}


def train_task(cohort: Cohort, task: Task, device: torch.device, collect_trajectory: bool = False) -> dict[str, Any]:
    """Train one original 50-epoch SSL+validation-selection trajectory."""
    started = time.time(); set_all_seeds(task.seed); model_config, params = make_model_config()
    encoder = VarCoNet(model_config, int(CONFIG["roi_count"])).to(device)
    contrast = DualBranchContrast(loss=InfoNCE(tau=float(params["tau"])), mode="L2L").to(device)
    optimizer = Adam(encoder.parameters(), lr=float(params["lr"]))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_start_lr=1e-5,
                                               warmup_epochs=int(CONFIG["warm_up_epochs"]), max_epochs=int(CONFIG["epochs"]))
    schedule = paired_batch_schedule(task.train_names, len(task.train_raw), int(CONFIG["epochs"]), task.seed)
    start_epoch = 1; best: dict[str, Any] | None = None; losses: list[float] = []
    trajectory_states: dict[int, dict[str, torch.Tensor]] = {}
    if collect_trajectory:
        # This clone precedes every optimizer step by construction.
        trajectory_states[0] = clone_state_cpu(encoder)
    checkpoint_path = _checkpoint_path(task)
    if checkpoint_path.exists():
        payload = torch_load_cpu(checkpoint_path)
        if payload.get("identity") == _task_identity(task):
            encoder.load_state_dict(payload["encoder_state"]); optimizer.load_state_dict(payload["optimizer_state"]); _optimizer_to_device(optimizer, device); scheduler.load_state_dict(payload["scheduler_state"])
            best = state_to_cpu(payload.get("best")); losses = [float(value) for value in payload.get("losses", [])]
            trajectory_states = state_to_cpu(payload.get("trajectory_states", trajectory_states)); start_epoch = int(payload["epoch"]) + 1
            restore_rng_state(payload["rng_state"]); started -= float(payload.get("elapsed_seconds", 0.0))
            print(f"TASK_RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} from_epoch={start_epoch}", flush=True)
    print(f"TASK_START repeat={task.repeat} scope={task.scope} fold={task.fold} device={device} train={len(task.train_raw)} val={len(task.val_raw)}", flush=True)
    for epoch in range(start_epoch, int(CONFIG["epochs"]) + 1):
        epoch_started = time.time(); ssl_loss = train_ssl_epoch(encoder, contrast, optimizer, cohort, task, schedule[epoch - 1], device); scheduler.step(); losses.append(ssl_loss)
        if collect_trajectory and epoch in FIXED_EPOCHS:
            trajectory_states[int(epoch)] = clone_state_cpu(encoder)
        stable_train = extract_old_stable(encoder, cohort.abide1, task.train_raw, device)
        stable_val = extract_old_stable(encoder, cohort.abide1, task.val_raw, device)
        rng_state = capture_rng_state()
        fit = fit_baseline_classifier(stable_train, task.train_labels, stable_val, task.val_labels, device, task.seed + 300000 + epoch * 2)
        restore_rng_state(rng_state)
        candidate = {"encoder_state": clone_state_cpu(encoder), **fit, "encoder_epoch": int(epoch)}
        is_best = best is None or float(candidate["val_bce"]) < float(best["val_bce"])
        if is_best: best = candidate
        elapsed = time.time() - started
        allocated = float(torch.cuda.memory_allocated(device)) / (1024 ** 3) if device.type == "cuda" else 0.0
        reserved = float(torch.cuda.memory_reserved(device)) / (1024 ** 3) if device.type == "cuda" else 0.0
        print(f"[SSL] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs']} ssl_loss={ssl_loss:.6f} lr={optimizer.param_groups[0]['lr']:.9f} base_val_bce={fit['val_bce']:.6f} base_best={'Y' if is_best else 'N'} base_cls_epoch={fit['classifier_epoch']} epoch_seconds={time.time()-epoch_started:.1f} elapsed_minutes={elapsed/60:.1f} gpu_alloc_gb={allocated:.3f} gpu_reserved_gb={reserved:.3f}", flush=True)
        if epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs"]):
            atomic_torch_save(checkpoint_path, {"identity": _task_identity(task), "epoch": int(epoch), "encoder_state": clone_state_cpu(encoder),
                "optimizer_state": state_to_cpu(optimizer.state_dict()), "scheduler_state": state_to_cpu(scheduler.state_dict()),
                "best": state_to_cpu(best), "losses": losses, "trajectory_states": state_to_cpu(trajectory_states),
                "rng_state": state_to_cpu(capture_rng_state()), "elapsed_seconds": float(elapsed)})
        del stable_train, stable_val
        if device.type == "cuda": torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("no validation-selected encoder")
    epoch50_state = clone_state_cpu(encoder)
    if collect_trajectory and 50 not in trajectory_states:
        trajectory_states[50] = state_to_cpu(epoch50_state)
    checkpoint_path.unlink(missing_ok=True)
    print(f"TASK_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} selected_encoder_epoch={best['encoder_epoch']} selected_val_bce={best['val_bce']:.6f} elapsed_minutes={(time.time()-started)/60:.1f}", flush=True)
    return {"task": task, "selected": state_to_cpu(best), "epoch50_state": epoch50_state,
            "trajectory_states": trajectory_states, "ssl_losses": losses, "elapsed_seconds": float(time.time() - started)}


def _pearson_cache_path(name: str) -> Path:
    return WORK_DIR / "pearson_cache" / f"{name}.npy"


def prepare_pearson_caches(cohort: Cohort) -> dict[str, Any]:
    """Build temporary P0/P1 source matrices once, outside the results tree."""
    from representations import pearson_full, pearson_windowavg

    report: dict[str, Any] = {}
    for name, data in (("abide1", cohort.abide1), ("abide2", cohort.abide2)):
        path = _pearson_cache_path(name); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape == (len(data), EDGE_DIM):
                report[name] = {"path": str(path), "status": "existing", "shape": list(array.shape)}; continue
            path.unlink()
        temporary = path.with_suffix(".tmp.npy")
        target = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(data), EDGE_DIM))
        max_delta = 0.0
        for index in range(len(data)):
            windows = build_stable_windows(data[index])
            p0 = pearson_windowavg(data[index], windows).numpy()
            p1 = pearson_full(data[index]).numpy()
            target[index] = p0
            max_delta = max(max_delta, float(np.max(np.abs(p0 - p1))))
            if (index + 1) % 100 == 0 or index + 1 == len(data):
                print(f"PEARSON_CACHE dataset={name} completed={index+1}/{len(data)}", flush=True)
        target.flush(); del target; os.replace(temporary, path)
        report[name] = {"path": str(path), "status": "created", "shape": [len(data), EDGE_DIM], "p0_p1_max_abs": max_delta}
    write_json_atomic(WORK_DIR / "pearson_cache_manifest.json", report)
    return report


def load_pearson_cache(name: str) -> np.ndarray:
    path = _pearson_cache_path(name)
    if not path.exists():
        raise RuntimeError(f"missing prepared Pearson cache {path}; run --prepare first")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != EDGE_DIM:
        raise RuntimeError(f"Pearson cache shape drift: {values.shape}")
    return values


def _encoder_from_state(state: dict[str, torch.Tensor], device: torch.device) -> VarCoNet:
    encoder = create_encoder(device); encoder.load_state_dict(state); encoder.eval(); return encoder


def _similarity_matrix(x: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        values = vector_pattern_similarity(x.to(device), x.to(device)).detach().cpu().numpy().astype(np.float32)
    return values


def _empty_metric_row(representation: str, scope: str, repeat: int, fold: str, n_subjects: int, selected_epoch: float, task_seconds: float) -> dict[str, Any]:
    row: dict[str, Any] = {column: float("nan") for column in METRIC_COLUMNS}
    row.update({"representation": representation, "scope": scope, "repeat": int(repeat), "fold": str(fold),
                "n_subjects": int(n_subjects), "selected_encoder_epoch": float(selected_epoch),
                "task_seconds": float(task_seconds), "status": "PASS"})
    return row


def _fingerprint_for_partition(
    encoder: VarCoNet | None,
    data: np.ndarray,
    raw_indices: np.ndarray,
    valid_lengths: np.ndarray,
    representation: str,
    device: torch.device,
    sites: np.ndarray | None,
    labels: np.ndarray | None,
    subject_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    indices = np.asarray(raw_indices, dtype=np.int64)
    eligible = indices[np.asarray(valid_lengths[indices] >= 2 * int(CONFIG["view_length"]), dtype=bool)]
    positions = np.where(np.isin(indices, eligible))[0]
    if len(eligible) == 0:
        return {"n": 0, "top1": float("nan"), "top5": float("nan"), "mrr": float("nan"), "same_similarity": float("nan"), "different_similarity": float("nan"), "identity_gap": float("nan"), "within_site_top1": float("nan"), "within_site_same_dx_top1": float("nan")}, {"ids": np.asarray([], dtype=np.int64)}
    a, b = batch_view_representations(encoder, data, eligible, representation, device, batch_size=int(CONFIG["batch_size"]))
    local_ids = np.asarray(subject_ids)[positions]
    local_sites = None if sites is None else np.asarray(sites)[positions]
    local_labels = None if labels is None else np.asarray(labels)[positions]
    metrics = fingerprint_metrics(a, b, local_ids, local_sites, local_labels)
    artifact = {"ids": local_ids.astype(np.int64), "sim_ab": np.asarray(metrics.pop("similarity_ab"), dtype=np.float32),
                "sim_ba": np.asarray(metrics.pop("similarity_ba"), dtype=np.float32),
                "top1_per_subject": np.asarray(metrics.pop("top1_per_subject"), dtype=np.float32),
                "top5_per_subject": np.asarray(metrics.pop("top5_per_subject"), dtype=np.float32),
                "mrr_per_subject": np.asarray(metrics.pop("mrr_per_subject"), dtype=np.float32)}
    return metrics, artifact


def _evaluate_axis_partition(
    train_x: torch.Tensor,
    train_y: np.ndarray,
    train_sites: np.ndarray,
    test_x: torch.Tensor,
    test_y: np.ndarray,
    test_sites: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, Any, np.ndarray]:
    axis = fit_shared_axis(train_x, train_y, train_sites)
    scores = np.asarray(score_axis(test_x, axis), dtype=np.float64)
    sim = _similarity_matrix(test_x, torch.device("cpu"))
    return {"auc": asd_auc(test_y, scores), "geometry": geometry_gap(sim, test_y, test_sites),
            "eligible_sites": list(axis.eligible_sites), "std_floor": float(axis.standardizer.floor)}, scores, axis, sim


def _representation_matrix(
    representation: str,
    encoder: VarCoNet | None,
    data: np.ndarray,
    raw_indices: np.ndarray,
    pearson_cache: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor:
    indices = np.asarray(raw_indices, dtype=np.int64)
    if representation in {P0, P1}:
        if pearson_cache is None:
            raise RuntimeError("Pearson cache unavailable")
        return torch.from_numpy(np.asarray(pearson_cache[indices], dtype=np.float32)).clone()
    return batch_representation(encoder, data, indices, representation, device, batch_size=int(CONFIG["batch_size"]))


def _metric_from_evaluation(
    representation: str,
    scope: str,
    repeat: int,
    fold: str,
    selected_epoch: float,
    task_seconds: float,
    axis_result: dict[str, Any],
    fingerprint_result: dict[str, Any],
    n_subjects: int,
) -> dict[str, Any]:
    row = _empty_metric_row(representation, scope, repeat, fold, n_subjects, selected_epoch, task_seconds)
    row.update({"disease_axis_auc": float(axis_result["auc"]), "disease_geometry_gap": float(axis_result["geometry"]),
                "fingerprint_n": int(fingerprint_result["n"]), "fingerprint_top1": float(fingerprint_result["top1"]),
                "fingerprint_top5": float(fingerprint_result["top5"]), "fingerprint_mrr": float(fingerprint_result["mrr"]),
                "fingerprint_same_similarity": float(fingerprint_result["same_similarity"]),
                "fingerprint_different_similarity": float(fingerprint_result["different_similarity"]),
                "fingerprint_identity_gap": float(fingerprint_result["identity_gap"]),
                "within_site_top1": float(fingerprint_result.get("within_site_top1", float("nan"))),
                "within_site_same_dx_top1": float(fingerprint_result.get("within_site_same_dx_top1", float("nan")))})
    return row


def _score_records(scope: str, repeat: int, fold: str, representation: str, subject_indices: np.ndarray, raw_indices: np.ndarray, y: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    if not (len(subject_indices) == len(raw_indices) == len(y) == len(scores)):
        raise RuntimeError("score record arrays are misaligned")
    return [{"scope": scope, "repeat": int(repeat), "fold": str(fold), "representation": representation,
             "subject_index": int(subject), "raw_index": int(raw), "y": int(label), "disease_score": float(score)}
            for subject, raw, label, score in zip(subject_indices, raw_indices, y, scores)]


def _internal_subject_positions(cohort: Cohort, raw_indices: np.ndarray) -> np.ndarray:
    mapping = {int(raw): index for index, raw in enumerate(cohort.internal_raw.tolist())}
    values = np.asarray([mapping[int(raw)] for raw in raw_indices], dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise RuntimeError("duplicate subject entered internal audit")
    return values


def evaluate_internal_task(
    cohort: Cohort,
    trained: dict[str, Any],
    pearson1: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all locked endpoints for one outer fold, without duplicates."""
    task: Task = trained["task"]; selected = trained["selected"]
    states: dict[str, dict[str, torch.Tensor] | None] = {P0: None, P1: None, V0: selected["encoder_state"], V1: trained["epoch50_state"]}
    rows: list[dict[str, Any]] = []; records: list[dict[str, Any]] = []
    subject_indices = _internal_subject_positions(cohort, task.test_raw)
    for representation in ALL_REPRESENTATIONS:
        encoder = None if states[representation] is None else _encoder_from_state(states[representation], device)
        train_x = _representation_matrix(representation, encoder, cohort.abide1, task.reference_raw, pearson1, device)
        test_x = _representation_matrix(representation, encoder, cohort.abide1, task.test_raw, pearson1, device)
        axis_result, scores, _, _ = _evaluate_axis_partition(train_x, task.reference_labels, task.reference_sites, test_x, task.test_labels, cohort.abide1_sites[task.test_raw],)
        fingerprint_result, _ = _fingerprint_for_partition(
            encoder, cohort.abide1, task.test_raw, cohort.abide1_valid_lengths, representation, device,
            cohort.abide1_sites[task.test_raw], task.test_labels, subject_indices,
        )
        selected_epoch = float(selected["encoder_epoch"]) if representation == V0 else (50.0 if representation == V1 else float("nan"))
        rows.append(_metric_from_evaluation(representation, "internal_fold", task.repeat, task.fold, selected_epoch,
                                             trained["elapsed_seconds"], axis_result, fingerprint_result, len(task.test_raw)))
        records.extend(_score_records("internal_fold", task.repeat, task.fold, representation, subject_indices, task.test_raw, task.test_labels, scores))
        del train_x, test_x
        if encoder is not None: del encoder
        if device.type == "cuda": torch.cuda.empty_cache()
    return rows, records


def _external_subject_ids(n: int) -> np.ndarray:
    return np.arange(int(n), dtype=np.int64)


def _evaluate_external_rep(
    cohort: Cohort,
    trained: dict[str, Any],
    representation: str,
    encoder: VarCoNet | None,
    train_x: torch.Tensor,
    train_y: np.ndarray,
    train_sites: np.ndarray,
    test_data: np.ndarray,
    test_indices: np.ndarray,
    test_y: np.ndarray,
    test_sites: np.ndarray,
    test_valid_lengths: np.ndarray,
    pearson_cache: np.ndarray | None,
    device: torch.device,
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    test_x = _representation_matrix(representation, encoder, test_data, test_indices, pearson_cache, device)
    axis_result, scores, axis, similarity = _evaluate_axis_partition(train_x, train_y, train_sites, test_x, test_y, test_sites)
    # External fingerprinting is always global and site-controlled when the
    # official ABIDE-II phenotype mapping is available.
    fp, fp_artifact = _fingerprint_for_partition(
        encoder, test_data, test_indices, test_valid_lengths, representation, device,
        test_sites, test_y, _external_subject_ids(len(test_indices)),
    )
    selected_epoch = float(trained["selected"]["encoder_epoch"]) if representation == V0 else (50.0 if representation == V1 else float("nan"))
    row = _metric_from_evaluation(representation, scope, trained["task"].repeat, "external", selected_epoch,
                                  trained["elapsed_seconds"], axis_result, fp, len(test_indices))
    rec = _score_records(scope, trained["task"].repeat, trained["task"].fold, representation, _external_subject_ids(len(test_indices)), test_indices, test_y, scores)
    artifact = {"axis": axis, "train_x": train_x.detach().cpu().numpy().astype(np.float32),
                "test_x": test_x.detach().cpu().numpy().astype(np.float32), "test_scores": np.asarray(scores, dtype=np.float32),
                "fingerprint": fp_artifact, "similarity": np.asarray(similarity, dtype=np.float32),
                "train_y": np.asarray(train_y, dtype=np.int64), "train_sites": np.asarray(train_sites),
                "test_y": np.asarray(test_y, dtype=np.int64), "test_sites": np.asarray(test_sites)}
    del test_x
    return row, rec, artifact


def _trajectory_row(
    repeat: int, epoch: int, selected_epoch: int, scope: str, fp: dict[str, Any],
    auc: float, geom: float, stability: float, n_disease: int,
) -> dict[str, Any]:
    return {"repeat": int(repeat), "epoch": int(epoch), "is_selected": bool(int(epoch) == int(selected_epoch)),
            "scope": scope, "fingerprint_top1": float(fp.get("top1", float("nan"))),
            "identity_gap": float(fp.get("identity_gap", float("nan"))), "disease_axis_auc": float(auc),
            "disease_geometry_gap": float(geom), "split_half_disease_cosine": float(stability),
            "n_identity_subjects": int(fp.get("n", 0)), "n_disease_subjects": int(n_disease), "status": "PASS"}


def evaluate_external_task(
    cohort: Cohort,
    trained: dict[str, Any],
    pearson1: np.ndarray,
    pearson2: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Evaluate selected/epoch-50 endpoints and all fixed SSL states."""
    task: Task = trained["task"]; repeat = int(task.repeat)
    all_train_raw = cohort.internal_raw.copy(); all_train_y = cohort.internal_labels.copy(); all_train_sites = cohort.internal_sites.copy()
    caltech_indices = np.arange(len(cohort.caltech_raw), dtype=np.int64); abide2_indices = np.arange(len(cohort.abide2), dtype=np.int64)
    if len(caltech_indices) != 37 or len(abide2_indices) != 727:
        raise RuntimeError("external cohort counts drifted")
    rows: list[dict[str, Any]] = []; records: list[dict[str, Any]] = []; artifacts: dict[str, Any] = {"fingerprints": {}, "geometry": {}, "edges": {}}

    # Pearson representations are independent of the trained state and are
    # evaluated once per external repeat, then reused for every scope.
    for representation, cache in ((P0, pearson1), (P1, pearson1)):
        train_x = torch.from_numpy(np.asarray(cache[all_train_raw], dtype=np.float32)).clone()
        for scope, data, indices, labels, sites, lengths, pcache in (
            ("caltech", cohort.abide1, cohort.caltech_raw, cohort.caltech_labels, cohort.caltech_sites, cohort.abide1_valid_lengths, pearson1),
            ("abide2", cohort.abide2, abide2_indices, cohort.abide2_labels, cohort.abide2_sites, cohort.abide2_valid_lengths, pearson2),
        ):
            row, rec, artifact = _evaluate_external_rep(cohort, trained, representation, None, train_x, all_train_y, all_train_sites,
                                                         data, indices, labels, sites, lengths, pcache, device, scope)
            rows.append(row); records.extend(rec)
            if scope == "abide2": artifacts["fingerprints"][representation] = artifact["fingerprint"]
            if scope == "internal_train": artifacts["geometry"][representation] = artifact["similarity"]
        # The same train-only axis is also used for the internal geometry audit.
        axis = fit_shared_axis(train_x, all_train_y, all_train_sites)
        sim = _similarity_matrix(train_x, device)
        artifacts["geometry"][representation] = sim
        artifacts.setdefault("axis", {})[representation] = {"axis": axis.axis.numpy(), "eligible_sites": list(axis.eligible_sites)}
        if representation == P0:
            # Keep only the external top-edge columns for the later 5,000-label
            # permutation null.  Full feature matrices are intermediate only.
            x_ab2 = torch.from_numpy(np.asarray(pearson2[abide2_indices], dtype=np.float32)).clone()
            train_effect = axis.axis.numpy().astype(np.float32)
            top = np.argsort(-np.abs(train_effect), kind="mergesort")[: max(1, int(round(EDGE_DIM * CONFIG["top_edge_fraction"])))].astype(np.int64)
            ext_effect = cohens_d_vector(x_ab2[torch.as_tensor(cohort.abide2_labels == 0)], x_ab2[torch.as_tensor(cohort.abide2_labels == 1)]).numpy().astype(np.float32)
            artifacts["edges"][P0] = {"train_effect": train_effect, "external_effect": ext_effect, "top_indices": top,
                                       "external_top_features": x_ab2[:, top].numpy().astype(np.float32)}
            del x_ab2
        del train_x

    selected_epoch = int(trained["selected"]["encoder_epoch"])
    states: dict[int, dict[str, torch.Tensor]] = {int(epoch): state for epoch, state in trained["trajectory_states"].items()}
    states[50] = trained["epoch50_state"]
    if selected_epoch not in states:
        states[selected_epoch] = trained["selected"]["encoder_state"]
    split_indices = make_split_half_indices(all_train_sites, all_train_y, SPLIT_HALF_REPEATS, SEED)
    trajectory_rows: list[dict[str, Any]] = []
    # Keep only one CPU state per fixed epoch in this task; release each
    # encoder immediately after its three scope evaluations.
    for epoch in trajectory_epochs(selected_epoch):
        state = states[int(epoch)]
        encoder = _encoder_from_state(state, device)
        train_x = _representation_matrix(V0, encoder, cohort.abide1, all_train_raw, None, device)
        cal_x = _representation_matrix(V0, encoder, cohort.abide1, cohort.caltech_raw, None, device)
        ab2_x = _representation_matrix(V0, encoder, cohort.abide2, abide2_indices, None, device)
        axis = fit_shared_axis(train_x, all_train_y, all_train_sites)
        train_scores = np.asarray(score_axis(train_x, axis)); cal_scores = np.asarray(score_axis(cal_x, axis)); ab2_scores = np.asarray(score_axis(ab2_x, axis))
        train_sim = _similarity_matrix(train_x, device); train_geom = geometry_gap(train_sim, all_train_y, all_train_sites)
        stability = split_half_stability(train_x, all_train_y, all_train_sites, split_indices)["mean"]
        fp_ab2, fp_art = _fingerprint_for_partition(encoder, cohort.abide2, abide2_indices, cohort.abide2_valid_lengths, V0, device,
                                                     cohort.abide2_sites, cohort.abide2_labels, abide2_indices)
        trajectory_rows.append(_trajectory_row(repeat, int(epoch), selected_epoch, "internal_train", {"n": 0}, float("nan"), train_geom, stability, len(all_train_raw)))
        trajectory_rows.append(_trajectory_row(repeat, int(epoch), selected_epoch, "caltech", {"n": 0}, asd_auc(cohort.caltech_labels, cal_scores), float("nan"), float("nan"), len(cal_scores)))
        trajectory_rows.append(_trajectory_row(repeat, int(epoch), selected_epoch, "abide2", fp_ab2, asd_auc(cohort.abide2_labels, ab2_scores), float("nan"), float("nan"), len(ab2_scores)))
        artifacts["trajectory_epoch_data"] = artifacts.get("trajectory_epoch_data", {})
        artifacts["trajectory_epoch_data"][int(epoch)] = {"abide2_scores": ab2_scores.astype(np.float32), "caltech_scores": cal_scores.astype(np.float32),
            "train_scores": train_scores.astype(np.float32), "fingerprint": fp_art, "train_geometry": train_geom, "stability": stability}
        del encoder, train_x, cal_x, ab2_x
        if device.type == "cuda": torch.cuda.empty_cache()

    # Selected/epoch50 endpoint rows use the same freshly fit external axis.
    for representation, state in ((V0, trained["selected"]["encoder_state"]), (V1, trained["epoch50_state"])):
        encoder = _encoder_from_state(state, device)
        train_x = _representation_matrix(representation, encoder, cohort.abide1, all_train_raw, None, device)
        for scope, data, indices, labels, sites, lengths in (
            ("caltech", cohort.abide1, cohort.caltech_raw, cohort.caltech_labels, cohort.caltech_sites, cohort.abide1_valid_lengths),
            ("abide2", cohort.abide2, abide2_indices, cohort.abide2_labels, cohort.abide2_sites, cohort.abide2_valid_lengths),
        ):
            row, rec, artifact = _evaluate_external_rep(cohort, trained, representation, encoder, train_x, all_train_y, all_train_sites,
                                                         data, indices, labels, sites, lengths, None, device, scope)
            rows.append(row); records.extend(rec)
            if scope == "abide2": artifacts["fingerprints"][representation] = artifact["fingerprint"]
        # Preserve selected internal geometry for paired H2 bootstrap.
        artifacts["geometry"][representation] = _similarity_matrix(train_x, device)
        axis = fit_shared_axis(train_x, all_train_y, all_train_sites)
        x_ab2 = _representation_matrix(representation, encoder, cohort.abide2, abide2_indices, None, device)
        ext_effect = cohens_d_vector(x_ab2[cohort.abide2_labels == 0], x_ab2[cohort.abide2_labels == 1]).numpy()
        train_effect = axis.axis.numpy()
        top = np.argsort(-np.abs(train_effect), kind="mergesort")[: max(1, int(round(EDGE_DIM * CONFIG["top_edge_fraction"])))].astype(np.int64)
        artifacts["edges"][representation] = {"train_effect": train_effect.astype(np.float32), "external_effect": ext_effect.astype(np.float32),
            "top_indices": top, "external_top_features": x_ab2[:, top].numpy().astype(np.float32)}
        del encoder, train_x, x_ab2
        if device.type == "cuda": torch.cuda.empty_cache()
    return rows, records, artifacts, trajectory_rows


def _shard_path(repeat: int) -> Path:
    return SHARD_DIR / f"repeat_{int(repeat)}.pkl"


def _new_shard(repeat: int) -> dict[str, Any]:
    return {"format_version": 1, "repeat": int(repeat), "status": "RUNNING", "completed_internal": [],
            "rows": [], "records": [], "trajectory": [], "external_artifacts": None, "worker_errors": []}


def _save_shard(repeat: int, payload: dict[str, Any]) -> None:
    payload["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_pickle(_shard_path(repeat), payload)


def run_repeat(cohort: Cohort, repeat: int, device: torch.device) -> None:
    """Run 10 internal folds plus one external trajectory for a repeat."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True); path = _shard_path(repeat)
    if path.exists():
        try:
            shard = load_pickle(path)
            if shard.get("status") == "PASS":
                print(f"REPEAT_DONE repeat={repeat} status=PASS already_complete=Y", flush=True); return
        except Exception:
            shard = _new_shard(repeat)
    else:
        shard = _new_shard(repeat)
    pearson1, pearson2 = load_pearson_cache("abide1"), load_pearson_cache("abide2")
    tasks = make_tasks(cohort, repeat)
    completed = {str(value) for value in shard.get("completed_internal", [])}
    print(f"REPEAT_START repeat={repeat} device={device} tasks=11 internal_folds=10 external_trajectory=Y", flush=True)
    for task in tasks[:-1]:
        if str(task.fold) in completed:
            print(f"TASK_SKIP repeat={repeat} scope=internal_fold fold={task.fold} reason=PASS_SHARD", flush=True); continue
        try:
            trained = train_task(cohort, task, device, collect_trajectory=False)
            task_rows, task_records = evaluate_internal_task(cohort, trained, pearson1, device)
            shard["rows"].extend(task_rows); shard["records"].extend(task_records); shard["completed_internal"].append(str(task.fold)); completed.add(str(task.fold))
            _save_shard(repeat, shard)
            del trained, task_rows, task_records
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception as exc:
            shard["worker_errors"].append({"scope": "internal_fold", "fold": str(task.fold), "error": repr(exc)})
            _save_shard(repeat, shard); raise
    external_task = tasks[-1]
    if shard.get("external_artifacts") is None:
        try:
            trained = train_task(cohort, external_task, device, collect_trajectory=True)
            ext_rows, ext_records, ext_artifacts, traj_rows = evaluate_external_task(cohort, trained, pearson1, pearson2, device)
            shard["rows"].extend(ext_rows); shard["records"].extend(ext_records); shard["trajectory"].extend(traj_rows); shard["external_artifacts"] = ext_artifacts
            shard["external_selected_epoch"] = int(trained["selected"]["encoder_epoch"])
            shard["ssl_losses_external"] = [float(value) for value in trained["ssl_losses"]]
            _save_shard(repeat, shard)
            del trained, ext_rows, ext_records, ext_artifacts, traj_rows
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception as exc:
            shard["worker_errors"].append({"scope": "external", "error": repr(exc)})
            _save_shard(repeat, shard); raise
    if len(set(shard.get("completed_internal", []))) != 10 or shard.get("external_artifacts") is None:
        raise RuntimeError(f"repeat={repeat} incomplete shard")
    shard["status"] = "PASS"; shard["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); _save_shard(repeat, shard)
    # No trajectory checkpoint files are retained after a successful repeat.
    for checkpoint in CHECKPOINT_DIR.glob(f"r{int(repeat)}_*_latest.pt"):
        checkpoint.unlink(missing_ok=True)
    print(f"REPEAT_DONE repeat={repeat} status=PASS rows={len(shard['rows'])} trajectory_rows={len(shard['trajectory'])}", flush=True)


def _load_root_varconet() -> Any:
    path = REPO_ROOT / "model_scripts" / "VarCoNet.py"
    spec = importlib.util.spec_from_file_location("varconet_root_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load root VarCoNet for parity audit")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module.VarCoNet


def run_smoke(cohort: Cohort) -> dict[str, Any]:
    """Run the only pre-full-run smoke test and fail closed on drift."""
    print("SMOKE_START experiment=1V", flush=True); set_all_seeds(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config, _ = make_model_config(); local = VarCoNet(config, int(CONFIG["roi_count"])).to(device)
    root_cls = _load_root_varconet(); root = root_cls(config, int(CONFIG["roi_count"])).to(device)
    root.load_state_dict(local.state_dict()); local.eval(); root.eval()
    sample = torch.from_numpy(np.asarray(cohort.abide1[:2], dtype=np.float32)).to(device)
    with torch.no_grad():
        root_out, local_out = root(sample), local(sample)
    root_local_delta = float((root_out - local_out).abs().max().cpu())
    if root_local_delta > 1e-6: raise RuntimeError(f"WINDOW_PARITY_FAIL: root/local VarCoNet max_abs={root_local_delta}")
    scan = torch.from_numpy(np.asarray(cohort.abide1[0], dtype=np.float32)); windows = build_stable_windows(scan)
    with torch.no_grad():
        old = local(torch.from_numpy(np.asarray(cohort.abide1[0:1], dtype=np.float32)).to(device)).detach().cpu()
    new = __import__("representations", fromlist=["varconet_windowavg"]).varconet_windowavg(local, scan, windows, device).reshape(1, -1)
    window_delta = float((old - new).abs().max())
    if window_delta > 1e-6: raise RuntimeError(f"WINDOW_PARITY_FAIL: new/old stable extraction max_abs={window_delta}")
    # Verify that the copied legacy helper is exactly the repository helper.
    # ``utils.test_augment`` is an HCP-style helper whose input is longer than
    # the 320-row model canvas; using a deterministic 640-row probe also
    # exercises its 320-window endpoint (which would otherwise have step=0).
    import utils as local_utils
    legacy_scan = (torch.arange((MAX_LENGTH * 2) * int(CONFIG["roi_count"]), dtype=torch.float32).reshape(MAX_LENGTH * 2, int(CONFIG["roi_count"])) % 997) / 997.0
    legacy = torch.stack([w.padded_window for w in build_legacy_test_windows(legacy_scan)])
    reference = local_utils.test_augment(legacy_scan, list(WINDOW_SIZES), NUM_WINDOWS, MAX_LENGTH)
    legacy_delta = float((legacy - reference.cpu()).abs().max())
    if legacy_delta > 1e-6: raise RuntimeError(f"WINDOW_PARITY_FAIL: legacy helper max_abs={legacy_delta}")
    from representations import pearson_fc
    pcc = pearson_fc(scan[: int(cohort.abide1_valid_lengths[0])]);
    if pcc.numel() != EDGE_DIM or not bool(torch.isfinite(pcc).all()): raise RuntimeError("Pearson smoke failed")
    metadata = window_metadata(windows)
    if metadata != ((0, int(cohort.abide1_valid_lengths[0])),): raise RuntimeError("window metadata drift")
    a, b = scan[:80], scan[int(cohort.abide1_valid_lengths[0]) - 80 : int(cohort.abide1_valid_lengths[0])]
    if set(range(80)).intersection(set(range(int(cohort.abide1_valid_lengths[0]) - 80, int(cohort.abide1_valid_lengths[0])))): raise RuntimeError("fingerprint views overlap")
    # Full ABIDE-II length eligibility audit is label-blind and deterministic.
    tokens = np.asarray([latent_valid_tokens(value) for value in cohort.abide2_valid_lengths], dtype=np.int64)
    excluded = np.where(tokens < 30)[0]
    if len(excluded) != 1 or int(excluded[0]) != 568 or cohort.abide2_names[int(excluded[0])] != "sub-29623":
        raise RuntimeError(f"ELIGIBILITY_FAIL: expected only sub-29623/raw index 568, got {excluded.tolist()}")
    # Axis/LOSO and metadata checks use Pearson only and never model outputs.
    p_cache = load_pearson_cache("abide1")
    axis = fit_shared_axis(torch.from_numpy(np.asarray(p_cache[cohort.internal_raw], dtype=np.float32)), cohort.internal_labels, cohort.internal_sites)
    if len(axis.eligible_sites) < 4 or not bool(torch.isfinite(axis.axis).all()): raise RuntimeError("disease-axis smoke failed")
    loso = strict_loso(torch.from_numpy(np.asarray(p_cache[cohort.internal_raw], dtype=np.float32)), cohort.internal_labels, cohort.internal_sites)
    for row in loso:
        if row["site"] in row["eligible_train_sites"]: raise RuntimeError("LOSO held-out site leakage")
    # Bootstrap reproducibility is checked on a tiny deterministic example.
    y = np.asarray([0, 0, 1, 1, 0, 1]); s = np.asarray([.9, .8, .2, .1, .7, .3])
    b1 = stratified_auc_bootstrap(y, s, n_boot=32, seed=SEED); b2 = stratified_auc_bootstrap(y, s, n_boot=32, seed=SEED)
    if b1 != b2: raise RuntimeError("bootstrap reproducibility failed")
    if len(set(cohort.internal_names)) != len(cohort.internal_names) or len(cohort.internal_raw) != 882: raise RuntimeError("duplicate internal audit IDs")
    payload = {"status": "PASS", "root_local_max_abs": root_local_delta, "window_max_abs": window_delta, "legacy_helper_max_abs": legacy_delta,
               "pearson_shape": int(pcc.numel()), "window_metadata": [list(item) for item in metadata],
               "abide1_site_mapping_complete": True, "abide2_site_mapping_complete": True, "abide2_n": 727,
               "abide2_short_token_indices": excluded.tolist(), "abide2_short_token_names": [cohort.abide2_names[int(i)] for i in excluded],
               "abide2_latent_tokens_at_568": int(tokens[568]), "loso_n_sites": len(loso), "eligible_axis_sites": list(axis.eligible_sites),
               "trajectory_epochs": list(FIXED_EPOCHS), "split_half_repeats": SPLIT_HALF_REPEATS, "source_hashes": source_hashes()}
    write_json_atomic(EXPERIMENT_DIR / "SMOKE.json", payload); print("SMOKE_PASS", json.dumps(payload, ensure_ascii=False), flush=True); return payload


def _fp_summary_from_similarity(
    sim_ab: np.ndarray,
    sim_ba: np.ndarray,
    subject_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize paired retrieval, including valid subject-level resamples.

    In an ordinary endpoint matrix every ID occurs once, so the matching
    gallery row is the diagonal.  A subject bootstrap can draw an ID more
    than once; all copies then represent the same resampled subject and must
    count as a correct identity match, rather than becoming artificial errors
    because of an arbitrary tie between duplicate copies.
    """
    sim_ab = np.asarray(sim_ab, dtype=float); sim_ba = np.asarray(sim_ba, dtype=float)
    if sim_ab.shape != sim_ba.shape or sim_ab.ndim != 2 or sim_ab.shape[0] != sim_ab.shape[1]:
        raise RuntimeError("fingerprint similarity matrices must be paired square matrices")
    n = int(sim_ab.shape[0])
    ids = np.arange(n, dtype=np.int64) if subject_ids is None else np.asarray(subject_ids)
    if len(ids) != n:
        raise RuntimeError("fingerprint summary IDs are misaligned")

    def direction(sim: np.ndarray) -> tuple[float, float, float]:
        # Row-wise C-level sorting preserves the same stable (mergesort) tie
        # convention as the scalar audit implementation, without millions of
        # Python-level calls during the preregistered 2,000 bootstraps.
        order = np.argsort(-sim, axis=1, kind="mergesort")
        matching = ids[order] == ids[:, None]
        if not bool(np.all(matching.any(axis=1))):
            raise RuntimeError("bootstrap identity target missing from gallery")
        ranks = matching.argmax(axis=1).astype(np.int64) + 1
        return float(np.mean(ranks == 1)), float(np.mean(ranks <= 5)), float(np.mean(1.0 / ranks))

    a1, a5, ar = direction(sim_ab); b1, b5, br = direction(sim_ba)
    same = float(np.mean((np.diag(sim_ab) + np.diag(sim_ba)) / 2.0))
    # Copies of one bootstrap subject are still same-subject pairs, not
    # "different" pairs.  With unique endpoint IDs this is exactly ~I.
    mask = ids[:, None] != ids[None, :]
    different = float(np.mean((sim_ab[mask] + sim_ba[mask]) / 2.0)) if bool(mask.any()) else float("nan")
    return {"top1": (a1 + b1) / 2.0, "top5": (a5 + b5) / 2.0, "mrr": (ar + br) / 2.0,
            "same_similarity": same, "different_similarity": different, "identity_gap": same - different,
            "n": int(sim_ab.shape[0])}


def _paired_fingerprint_bootstrap(sim_p: dict[str, np.ndarray], sim_v: dict[str, np.ndarray], n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> dict[str, Any]:
    if not (np.array_equal(sim_p["ids"], sim_v["ids"]) and sim_p["sim_ab"].shape == sim_v["sim_ab"].shape):
        raise RuntimeError("fingerprint bootstrap IDs/matrices are not paired")
    n = int(len(sim_p["ids"])); rng = np.random.default_rng(int(seed)); d_top1: list[float] = []; d_gap: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, n, replace=True)
        boot_ids = np.asarray(sim_p["ids"])[idx]
        p = _fp_summary_from_similarity(sim_p["sim_ab"][np.ix_(idx, idx)], sim_p["sim_ba"][np.ix_(idx, idx)], boot_ids)
        v = _fp_summary_from_similarity(sim_v["sim_ab"][np.ix_(idx, idx)], sim_v["sim_ba"][np.ix_(idx, idx)], boot_ids)
        d_top1.append(v["top1"] - p["top1"]); d_gap.append(v["identity_gap"] - p["identity_gap"])
    return {"top1": ci_summary(np.asarray(d_top1), _fp_summary_from_similarity(sim_v["sim_ab"], sim_v["sim_ba"])["top1"] - _fp_summary_from_similarity(sim_p["sim_ab"], sim_p["sim_ba"])["top1"]),
            "identity_gap": ci_summary(np.asarray(d_gap), _fp_summary_from_similarity(sim_v["sim_ab"], sim_v["sim_ba"])["identity_gap"] - _fp_summary_from_similarity(sim_p["sim_ab"], sim_p["sim_ba"])["identity_gap"])}


def _aggregate_score_vectors(records: list[dict[str, Any]], scope: str, representation: str, n_expected: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [record for record in records if record["scope"] == scope and record["representation"] == representation]
    by_raw: dict[int, list[tuple[float, int, int]]] = {}
    for record in selected:
        by_raw.setdefault(int(record["raw_index"]), []).append((float(record["disease_score"]), int(record["y"]), int(record["repeat"])))
    if len(by_raw) != int(n_expected):
        raise RuntimeError(f"score vector {scope}/{representation} has {len(by_raw)} IDs, expected {n_expected}")
    raws = np.asarray(sorted(by_raw), dtype=np.int64); scores = []; labels = []
    for raw in raws.tolist():
        values = by_raw[int(raw)]; ys = {value[1] for value in values}
        if len(ys) != 1: raise RuntimeError("label drift across paired score records")
        scores.append(float(np.mean([value[0] for value in values]))); labels.append(next(iter(ys)))
    return raws, np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {column: row.get(column, float("nan")) for column in columns}
            writer.writerow(clean)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, dict): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(item) for item in value]
    return value


def _site_permutation_pvalues(loso_rows: list[dict[str, Any]], n_perm: int = PERMUTATION_N, seed: int = SEED) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, row in enumerate(loso_rows):
        labels = np.asarray(row["labels"], dtype=np.int64); scores = np.asarray(row["scores"], dtype=float)
        values[str(row["site"])] = float(auc_label_permutation(labels, scores, n_perm=n_perm, seed=int(seed) + index)["p"])
    return values


def _trajectory_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixed = list(FIXED_EPOCHS)
    table: list[dict[str, Any]] = []

    def scope_stats(items: list[dict[str, Any]], field: str) -> tuple[float, float]:
        vals = np.asarray([float(row[field]) for row in items if np.isfinite(float(row[field]))], dtype=float)
        return (float(vals.mean()), float(vals.std(ddof=1))) if len(vals) > 1 else ((float(vals[0]), 0.0) if len(vals) else (float("nan"), float("nan")))

    def summary_row(epoch: int | str, epoch_label: str, selected_count: int, subset: list[dict[str, Any]]) -> dict[str, Any]:
        ab = [row for row in subset if row["scope"] == "abide2"]
        tr = [row for row in subset if row["scope"] == "internal_train"]
        i_mean, i_sd = scope_stats(ab, "fingerprint_top1"); g_mean, g_sd = scope_stats(ab, "identity_gap")
        d_mean, d_sd = scope_stats(ab, "disease_axis_auc"); geom_mean, geom_sd = scope_stats(tr, "disease_geometry_gap"); stab_mean, stab_sd = scope_stats(tr, "split_half_disease_cosine")
        return {"epoch": epoch, "epoch_label": epoch_label, "selected_repeat_count": int(selected_count),
                "identity_top1_mean": i_mean, "identity_top1_sd": i_sd, "identity_gap_mean": g_mean, "identity_gap_sd": g_sd,
                "disease_auc_mean": d_mean, "disease_auc_sd": d_sd, "geometry_mean": geom_mean, "geometry_sd": geom_sd,
                "stability_mean": stab_mean, "stability_sd": stab_sd}

    for epoch in fixed:
        subset = [row for row in rows if int(row["epoch"]) == int(epoch)]
        if not subset: continue
        selected_count = len({int(row["repeat"]) for row in subset if row["scope"] == "abide2" and bool(row["is_selected"])})
        label = str(epoch) if selected_count == 0 else f"{epoch} (selected {selected_count}/10)"
        table.append(summary_row(int(epoch), label, selected_count, subset))
    selected = [row for row in rows if bool(row["is_selected"])]
    selected_epochs = sorted({int(row["epoch"]) for row in selected})
    # A validation-selected state outside the fixed list has already been
    # evaluated once per repeat; this compact row only summarizes it.
    if any(epoch not in fixed for epoch in selected_epochs):
        table.append(summary_row("selected", "selected (per-repeat validation state)", len({int(row["repeat"]) for row in selected if row["scope"] == "abide2"}), selected))
    return table, {"fixed_epochs": fixed, "selected_epochs": selected_epochs, "selected_nonfixed": any(epoch not in fixed for epoch in selected_epochs), "table": table}


def _trajectory_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Locked per-repeat rank trends across the eight fixed SSL checkpoints."""
    out: list[dict[str, Any]] = []

    def rho(x: np.ndarray, y: np.ndarray) -> float:
        good = np.isfinite(x) & np.isfinite(y)
        if int(good.sum()) < 3 or len(np.unique(x[good])) < 2 or len(np.unique(y[good])) < 2:
            return float("nan")
        value = spearmanr(x[good], y[good]).statistic
        return float(value) if np.isfinite(value) else float("nan")

    for repeat in range(10):
        ab = {int(row["epoch"]): row for row in rows if int(row["repeat"]) == repeat and row["scope"] == "abide2" and int(row["epoch"]) in FIXED_EPOCHS}
        tr = {int(row["epoch"]): row for row in rows if int(row["repeat"]) == repeat and row["scope"] == "internal_train" and int(row["epoch"]) in FIXED_EPOCHS}
        if set(ab) != set(FIXED_EPOCHS) or set(tr) != set(FIXED_EPOCHS):
            raise RuntimeError(f"trajectory fixed-state coverage drifted for repeat {repeat}")
        epochs = np.asarray(FIXED_EPOCHS, dtype=float)
        identity = np.asarray([float(ab[e]["fingerprint_top1"]) for e in FIXED_EPOCHS])
        disease_auc = np.asarray([float(ab[e]["disease_axis_auc"]) for e in FIXED_EPOCHS])
        stability = np.asarray([float(tr[e]["split_half_disease_cosine"]) for e in FIXED_EPOCHS])
        out.append({"repeat": int(repeat), "epoch_vs_identity_top1_spearman": rho(epochs, identity),
                    "epoch_vs_disease_auc_spearman": rho(epochs, disease_auc),
                    "epoch_vs_disease_stability_spearman": rho(epochs, stability),
                    "identity_vs_disease_auc_spearman": rho(identity, disease_auc)})
    return out


def aggregate_results() -> dict[str, Any]:
    """Validate shards, run preregistered inference, and write compact outputs."""
    if not (EXPERIMENT_DIR / "SMOKE.json").exists():
        raise RuntimeError("SMOKE_REQUIRED: run --smoke exactly once before aggregation")
    cohort = load_cohort(); before = source_hashes(); smoke = json.loads((EXPERIMENT_DIR / "SMOKE.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("source_hashes") != before:
        raise RuntimeError("INTEGRITY_FAIL: smoke status/source hashes are not a current PASS audit")
    shards: list[dict[str, Any]] = []
    for repeat in range(10):
        path = _shard_path(repeat)
        if not path.exists(): raise RuntimeError(f"missing repeat shard {repeat}")
        shard = load_pickle(path)
        if shard.get("status") != "PASS": raise RuntimeError(f"repeat {repeat} shard is not PASS")
        if shard.get("worker_errors"): raise RuntimeError(f"repeat {repeat} contains worker errors")
        if len(set(shard.get("completed_internal", []))) != 10: raise RuntimeError(f"repeat {repeat} does not contain 10 internal folds")
        shards.append(shard)
    rows = [row for shard in shards for row in shard["rows"]]; trajectory_rows = [row for shard in shards for row in shard["trajectory"]]; records = [record for shard in shards for record in shard["records"]]
    # Every scope is paired on exactly the same row IDs, labels, and raw scans.
    paired_audit: dict[str, Any] = {}
    expected_internal = {
        repeat: {str(task.fold): len(task.test_raw) for task in make_tasks(cohort, repeat)[:-1]}
        for repeat in range(10)
    }
    for repeat in range(10):
        for scope in ("internal_fold", "caltech", "abide2"):
            folds = [str(f) for f in range(10)] if scope == "internal_fold" else ["external"]
            for fold in folds:
                groups = []
                for rep in ALL_REPRESENTATIONS:
                    group = sorted((int(r["subject_index"]), int(r["raw_index"]), int(r["y"])) for r in records if int(r["repeat"]) == repeat and r["scope"] == scope and str(r.get("fold", fold)) == fold and r["representation"] == rep)
                    groups.append(group)
                if not groups or any(group != groups[0] for group in groups[1:]):
                    raise RuntimeError(f"INTEGRITY_FAIL: unpaired IDs/labels scope={scope} repeat={repeat} fold={fold}")
                expected = expected_internal[repeat][fold] if scope == "internal_fold" else (37 if scope == "caltech" else 727)
                if len(groups[0]) != expected: raise RuntimeError(f"{scope} repeat={repeat} fold={fold} n={len(groups[0])} expected={expected}")
                paired_audit[f"{scope}/{repeat}/{fold}"] = expected
    # ABIDE-II must remain the complete original 727 rows in this experiment.
    abide2_raw = sorted({int(r["raw_index"]) for r in records if r["scope"] == "abide2"})
    if abide2_raw != list(range(727)): raise RuntimeError("ABIDE-II final cohort is not exactly 727 original rows")

    pearson1, _ = load_pearson_cache("abide1"), load_pearson_cache("abide2")
    # Strict Pearson LOSO is independent of learned model selection and is the
    # preregistered H1-B transfer test.
    internal_p = torch.from_numpy(np.asarray(pearson1[cohort.internal_raw], dtype=np.float32)).clone()
    loso_rows = strict_loso(internal_p, cohort.internal_labels, cohort.internal_sites)
    loso_auc = np.asarray([float(row["auc"]) for row in loso_rows], dtype=float)
    loso_boot = site_macro_bootstrap(loso_auc, n_boot=5000, seed=SEED)
    loso_p = _site_permutation_pvalues(loso_rows, n_perm=PERMUTATION_N, seed=SEED)

    # External subject scores are averaged over the ten fixed repeats only
    # after all models have completed; IDs are asserted identical above.
    ext_scores: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ext_auc: dict[str, Any] = {}
    for rep in ALL_REPRESENTATIONS:
        ext_scores[rep] = _aggregate_score_vectors(records, "abide2", rep, 727)
        raws, yy, ss = ext_scores[rep]
        ext_auc[rep] = {"auc": asd_auc(yy, ss), "bootstrap": stratified_auc_bootstrap(yy, ss, BOOTSTRAP_N, SEED),
                        "permutation": auc_label_permutation(yy, ss, PERMUTATION_N, SEED)}

    # Paired identity bootstrap on fixed ABIDE-II fingerprint IDs.
    fp_by_rep: dict[str, dict[str, np.ndarray]] = {}
    for rep in (P0, V0):
        artifacts = [shard["external_artifacts"]["fingerprints"][rep] for shard in shards]
        ids = np.asarray(artifacts[0]["ids"], dtype=np.int64)
        if any(not np.array_equal(ids, np.asarray(item["ids"], dtype=np.int64)) for item in artifacts[1:]):
            raise RuntimeError(f"fingerprint ID mismatch across repeats for {rep}")
        fp_by_rep[rep] = {"ids": ids, "sim_ab": np.mean(np.stack([np.asarray(item["sim_ab"], dtype=np.float32) for item in artifacts]), axis=0),
                          "sim_ba": np.mean(np.stack([np.asarray(item["sim_ba"], dtype=np.float32) for item in artifacts]), axis=0)}
    fp_p = _fp_summary_from_similarity(fp_by_rep[P0]["sim_ab"], fp_by_rep[P0]["sim_ba"]); fp_v = _fp_summary_from_similarity(fp_by_rep[V0]["sim_ab"], fp_by_rep[V0]["sim_ba"])
    fp_boot = _paired_fingerprint_bootstrap(fp_by_rep[P0], fp_by_rep[V0], BOOTSTRAP_N, SEED)

    # Paired internal cross-site geometry uses the same unique 882 subjects and
    # the external selected encoders' unsupervised representations (no labels
    # are used to choose a state).  Each repeat contributes a full subject
    # similarity matrix; the mean matrix is used only for the paired bootstrap.
    geom_p = np.mean(np.stack([np.asarray(shard["external_artifacts"]["geometry"][P0], dtype=np.float32) for shard in shards]), axis=0)
    geom_v = np.mean(np.stack([np.asarray(shard["external_artifacts"]["geometry"][V0], dtype=np.float32) for shard in shards]), axis=0)
    geom_p_gap = geometry_gap(geom_p, cohort.internal_labels, cohort.internal_sites); geom_v_gap = geometry_gap(geom_v, cohort.internal_labels, cohort.internal_sites)
    geom_boot = paired_geometry_bootstrap(geom_p, geom_v, cohort.internal_labels, cohort.internal_sites, BOOTSTRAP_N, SEED)

    # Fixed top-1% edge replication.  Each learned V0 repeat selects its own
    # training-only top set; no mean effect vector is ever used to redefine a
    # post-hoc edge set.  P0 is deterministic, so its first audited repeat is
    # the exact shared result used by the H1-C decision.
    edge_stats: dict[str, Any] = {}; edge_payloads: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for rep in (P0, V0):
        payloads = [(int(shard["repeat"]), shard["external_artifacts"]["edges"][rep]) for shard in shards]
        selected_payloads = payloads[:1] if rep == P0 else payloads
        per_repeat: list[dict[str, Any]] = []
        for repeat, payload in selected_payloads:
            top = np.asarray(payload["top_indices"], dtype=np.int64)
            train_effect = np.asarray(payload["train_effect"], dtype=np.float32)
            external_effect = np.asarray(payload["external_effect"], dtype=np.float32)
            features = np.asarray(payload["external_top_features"], dtype=np.float32)
            if len(top) != max(1, int(round(EDGE_DIM * CONFIG["top_edge_fraction"]))) or features.shape != (727, len(top)):
                raise RuntimeError(f"edge payload drift for {rep}/repeat={repeat}")
            exact = edge_replication(train_effect[top], external_effect[top], fraction=1.0,
                                     y_external=cohort.abide2_labels, external_features=features,
                                     n_perm=PERMUTATION_N, seed=SEED + repeat)
            exact.update({"repeat": repeat, "n_edges_used": int(len(top)),
                          "roi": roi_convergence(torch.from_numpy(train_effect), torch.from_numpy(external_effect), int(CONFIG["roi_count"]))})
            per_repeat.append(exact)
        if rep == P0:
            summary_edge = dict(per_repeat[0])
            summary_edge["aggregation"] = "single deterministic Pearson audit"
        else:
            numeric = ("sign_agreement", "spearman", "cosine", "sign_null_q95", "spearman_null_q95")
            summary_edge = {field: float(np.nanmean([float(item[field]) for item in per_repeat])) for field in numeric}
            summary_edge.update({"k": int(per_repeat[0]["k"]), "n_edges_used": int(per_repeat[0]["n_edges_used"]),
                                 "sign_perm_p": float("nan"), "spearman_perm_p": float("nan"),
                                 "roi": {field: float(np.nanmean([float(item["roi"][field]) for item in per_repeat])) for field in ("signed_spearman", "absburden_spearman")},
                                 "aggregation": "mean across repeat-specific fixed top-1% audits"})
        summary_edge["per_repeat"] = per_repeat
        edge_stats[rep] = summary_edge; edge_payloads[rep] = selected_payloads

    # Add edge/ROI fields to external endpoint rows (post-hoc audit only).
    for row in rows:
        rep = row["representation"]
        if rep in edge_stats and row["scope"] in {"abide2", "caltech"}:
            row.update({"topedge_sign_agreement": edge_stats[rep]["sign_agreement"], "topedge_spearman": edge_stats[rep]["spearman"], "topedge_cosine": edge_stats[rep]["cosine"]})
            if row["scope"] == "abide2":
                row["roi_signed_spearman"] = float(edge_stats[rep]["roi"]["signed_spearman"])
                row["roi_absburden_spearman"] = float(edge_stats[rep]["roi"]["absburden_spearman"])

    # Trajectory consistency and fixed-epoch means.
    trajectory_table, trajectory_summary = _trajectory_summary(trajectory_rows)
    trajectory_correlations = _trajectory_correlations(trajectory_rows)
    by_repeat: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
    for row in trajectory_rows: by_repeat.setdefault(int(row["repeat"]), {}).setdefault(int(row["epoch"]), {})[str(row["scope"])] = row
    consistency_rows = []
    for repeat in range(10):
        e0, e50 = by_repeat[repeat][0], by_repeat[repeat][50]
        i0, i50 = e0["abide2"]["fingerprint_top1"], e50["abide2"]["fingerprint_top1"]
        da = e50["abide2"]["disease_axis_auc"] - e0["abide2"]["disease_axis_auc"]
        dg = e50["internal_train"]["disease_geometry_gap"] - e0["internal_train"]["disease_geometry_gap"]
        ds = e50["internal_train"]["split_half_disease_cosine"] - e0["internal_train"]["split_half_disease_cosine"]
        identity_up = bool(i50 > i0); disease_down = bool((da < 0) or (dg < 0) or (ds < 0))
        consistency_rows.append({"repeat": repeat, "identity_delta_top1": float(i50-i0), "identity_delta_gap": float(e50["abide2"]["identity_gap"]-e0["abide2"]["identity_gap"]), "disease_delta_auc": float(da), "disease_delta_geometry": float(dg), "disease_delta_stability": float(ds), "identity_up": identity_up, "disease_down": disease_down, "tradeoff_repeat": bool(identity_up and disease_down)})
    h2d_count = int(sum(int(row["tradeoff_repeat"]) for row in consistency_rows)); mean_consistency = {key: float(np.mean([row[key] for row in consistency_rows])) for key in ("identity_delta_top1", "identity_delta_gap", "disease_delta_auc", "disease_delta_geometry", "disease_delta_stability")}

    # H1 decision rules.
    h1a = bool(ext_auc[P0]["bootstrap"]["ci_low"] > 0.5 and ext_auc[P0]["permutation"]["p"] < 0.05)
    h1b = bool(loso_boot["ci_low"] > 0.5)
    h1c = bool((edge_stats[P0]["spearman"] > 0 and edge_stats[P0]["spearman_perm_p"] < 0.05) or (edge_stats[P0]["sign_agreement"] > edge_stats[P0]["sign_null_q95"]))
    if h1a and (h1b or h1c): h1_decision = "SUPPORTED"
    elif ext_auc[P0]["auc"] > 0.5 and (h1b or h1c): h1_decision = "PARTIAL"
    else: h1_decision = "NOT SUPPORTED"
    h2a = bool(fp_boot["top1"]["ci_low"] > 0)
    h2b = bool(geom_boot["ci_high"] < 0)
    h2c = bool(paired_stratified_auc_bootstrap(ext_scores[P0][1], ext_scores[P0][2], ext_scores[V0][2], BOOTSTRAP_N, SEED)["ci_high"] < 0)
    h2d = bool(h2d_count >= 7 and mean_consistency["identity_delta_top1"] > 0 and (mean_consistency["disease_delta_auc"] < 0 or mean_consistency["disease_delta_geometry"] < 0 or mean_consistency["disease_delta_stability"] < 0))
    if h1_decision == "NOT SUPPORTED": h2_decision = "NOT_SUPPORTED"
    elif h2a and (h2b or h2c) and h2d: h2_decision = "STRONG_SUPPORTED"
    elif h2a and (h2b or h2c or h2d): h2_decision = "PARTIAL_SUPPORTED"
    else: h2_decision = "NOT_SUPPORTED"
    if ext_auc[P0]["auc"] <= ext_auc[V0]["auc"] and abs(geom_v_gap - geom_p_gap) <= 1e-3:
        no_suppression_statement = "NO EVIDENCE THAT VARCONET SUPPRESSES SHARED ASD SIGNAL."
    else: no_suppression_statement = ""

    pvalues = {"h1_abide2_auc": ext_auc[P0]["permutation"]["p"], "h1_edge_spearman": edge_stats[P0]["spearman_perm_p"], **{f"loso_{k}": v for k, v in loso_p.items()}}
    fdr = benjamini_hochberg(pvalues)
    paired_auc = paired_stratified_auc_bootstrap(ext_scores[P0][1], ext_scores[P0][2], ext_scores[V0][2], BOOTSTRAP_N, SEED)
    statistics = {"seed": SEED, "bootstrap_n": BOOTSTRAP_N, "permutation_n": PERMUTATION_N,
                  "external_auc": ext_auc, "loso": {"site_rows": loso_rows, "macro_bootstrap": loso_boot, "site_label_permutation_p": loso_p},
                  "fingerprint": {"pearson": fp_p, "varconet": fp_v, "paired_bootstrap_delta_varconet_minus_pearson": fp_boot},
                  "geometry": {"pearson_gap": geom_p_gap, "varconet_gap": geom_v_gap, "paired_bootstrap_delta_varconet_minus_pearson": geom_boot},
                  "paired_external_auc": paired_auc, "edge_replication": edge_stats, "trajectory_consistency": consistency_rows,
                  "trajectory_summary": trajectory_summary, "trajectory_correlations": trajectory_correlations, "fdr_qvalues": fdr}

    # Write final compact endpoint tables.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS_DIR / "metrics.csv", rows, METRIC_COLUMNS)
    _write_csv(RESULTS_DIR / "trajectory.csv", trajectory_rows, TRAJECTORY_COLUMNS)
    write_json_atomic(RESULTS_DIR / "statistics.json", _json_safe(statistics))
    score_scope = np.asarray([record["scope"] for record in records], dtype="U32"); score_rep = np.asarray([record["representation"] for record in records], dtype="U32")
    np.savez_compressed(RESULTS_DIR / "scores.npz", scope=score_scope, fold=np.asarray([record["fold"] for record in records], dtype="U16"), repeat=np.asarray([record["repeat"] for record in records], dtype=np.int16), representation=score_rep,
                        subject_index=np.asarray([record["subject_index"] for record in records], dtype=np.int32), raw_index=np.asarray([record["raw_index"] for record in records], dtype=np.int32),
                        y=np.asarray([record["y"] for record in records], dtype=np.int8), disease_score=np.asarray([record["disease_score"] for record in records], dtype=np.float32))
    # Fixed top-edge table (indices only; no unverified AICHA names).  V0
    # retains the independently selected set from every external repeat.
    tri_i, tri_j = np.triu_indices(int(CONFIG["roi_count"]), k=1); edge_rows: list[dict[str, Any]] = []
    for rep in (P0, V0):
        for repeat, payload in edge_payloads[rep]:
            top = np.asarray(payload["top_indices"], dtype=np.int64)
            train_effect = np.asarray(payload["train_effect"], dtype=np.float32); external_effect = np.asarray(payload["external_effect"], dtype=np.float32)
            for rank, edge in enumerate(top.tolist(), 1):
                edge_rows.append({"representation": rep, "repeat": int(repeat), "edge_rank": rank, "roi_i": int(tri_i[edge]), "roi_j": int(tri_j[edge]),
                                  "train_effect": float(train_effect[edge]), "abide2_effect": float(external_effect[edge]),
                                  "same_sign": bool(np.sign(train_effect[edge]) == np.sign(external_effect[edge]))})
    _write_csv(RESULTS_DIR / "top_edges.csv", edge_rows, ["representation", "repeat", "edge_rank", "roi_i", "roi_j", "train_effect", "abide2_effect", "same_sign"])

    summary = {"experiment": "Experiment 1V", "status": "PASS", "question1_h1": h1_decision, "question2_h2": h2_decision,
               "h1_tests": {"H1_A_external_pearson_auc": h1a, "H1_B_internal_pearson_loso": h1b, "H1_C_pearson_topedge_replication": h1c},
               "h2_tests": {"H2_A_identity": h2a, "H2_B_geometry": h2b, "H2_C_external_auc": h2c, "H2_D_trajectory": h2d},
               "abide2_original_n": 727, "abide2_evaluable_n": 727, "abide2_excluded_n": 0, "caltech_n": 37, "internal_unique_n": 882,
               "abide2_site_metadata": cohort.metadata, "fingerprint_primary_n": int(fp_p["n"]), "h2d_tradeoff_repeats": h2d_count,
               "no_suppression_statement": no_suppression_statement, "selected_epochs": [int(shard["external_selected_epoch"]) for shard in shards],
               "task_counts": {"internal_folds": 100, "external_repeats": 10}, "source_hashes_before": before, "source_hashes_smoke": smoke.get("source_hashes"),
               "paired_audit_groups": len(paired_audit), "worker_errors": [], "temporary_states_cleaned": True,
               "protocol": {"window_sizes_audited": list(WINDOW_SIZES), "stable_window_policy": "one complete 320-row padded scan, matching Experiment1D actual extraction", "view_length": 80, "fixed_epochs": list(FIXED_EPOCHS), "label_convention": "0=ASD,1=HC"}}
    after = source_hashes(); summary["source_hashes_after"] = after; summary["source_hashes_unchanged"] = bool(before == after)
    if before != after: raise RuntimeError("INTEGRITY_FAIL: source hash changed during aggregation")
    write_json_atomic(RESULTS_DIR / "summary.json", _json_safe(summary))
    _write_report(RESULTS_DIR / "REPORT.md", summary, statistics, trajectory_table, fp_p, fp_v, geom_p_gap, geom_v_gap, ext_auc, loso_boot, edge_stats, paired_auc)
    # Remove temporary shards/checkpoints/work cache only after all final files
    # are atomically written and the source audit has passed.
    for path in list(SHARD_DIR.glob("repeat_*.pkl")) + list(CHECKPOINT_DIR.glob("*.pt")):
        path.unlink(missing_ok=True)
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    print(f"AGGREGATE_DONE status=PASS H1={h1_decision} H2={h2_decision} metrics_rows={len(rows)} trajectory_rows={len(trajectory_rows)}", flush=True)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
        return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"
    except Exception:
        return "NA"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    statistics: dict[str, Any],
    trajectory_table: list[dict[str, Any]],
    fp_p: dict[str, Any],
    fp_v: dict[str, Any],
    geom_p_gap: float,
    geom_v_gap: float,
    ext_auc: dict[str, Any],
    loso_boot: dict[str, Any],
    edge_stats: dict[str, Any],
    paired_auc: dict[str, Any],
) -> None:
    """Create a concise, decision-first report with no hidden cherry-picking."""
    h1, h2 = summary["question1_h1"], summary["question2_h2"]
    fp_delta = statistics["fingerprint"]["paired_bootstrap_delta_varconet_minus_pearson"]
    geom_delta = statistics["geometry"]["paired_bootstrap_delta_varconet_minus_pearson"]
    lines: list[str] = []
    lines += [
        "# Experiment 1V — Direct Mechanism Validation", "",
        f"**Question 1 — Does a cross-subject, cross-site, externally transferable ASD connectivity pattern exist?**  **{h1}**", "",
        f"**Question 2 — Does VarCoNet suppress this shared disease structure while enhancing subject identity?**  **{h2}**", "",
        "This is a preregistered mechanism audit, not a supervised rescue or an innovation search. No metric, window, top-K, epoch, direction, or subject subset was changed after observing results.", "",
        "## Executive result", "",
        f"Internal strict LOSO Pearson macro AUC: **{_fmt(loso_boot.get('observed'))}** (site-bootstrap 95% CI {_fmt(loso_boot.get('ci_low'))}–{_fmt(loso_boot.get('ci_high'))}).  ABIDE-II Pearson site-balanced-axis AUC: **{_fmt(ext_auc[P0]['auc'])}** (subject-bootstrap 95% CI {_fmt(ext_auc[P0]['bootstrap']['ci_low'])}–{_fmt(ext_auc[P0]['bootstrap']['ci_high'])}; label-permutation p={_fmt(ext_auc[P0]['permutation']['p'], 5)}).", "",
        "## Core paired comparison", "",
        "| Metric | Pearson | VarCoNet | Δ VarCoNet-Pearson | 95% CI |", "|---|---:|---:|---:|---:|",
        f"| ABIDE-II fingerprint TOP1 | {_fmt(fp_p['top1'])} | {_fmt(fp_v['top1'])} | {_fmt(fp_delta['top1']['observed'])} | {_fmt(fp_delta['top1']['ci_low'])}–{_fmt(fp_delta['top1']['ci_high'])} |",
        f"| ABIDE-II identity gap | {_fmt(fp_p['identity_gap'])} | {_fmt(fp_v['identity_gap'])} | {_fmt(statistics['fingerprint']['paired_bootstrap_delta_varconet_minus_pearson']['identity_gap']['observed'])} | {_fmt(statistics['fingerprint']['paired_bootstrap_delta_varconet_minus_pearson']['identity_gap']['ci_low'])}–{_fmt(statistics['fingerprint']['paired_bootstrap_delta_varconet_minus_pearson']['identity_gap']['ci_high'])} |",
        f"| Internal cross-site disease geometry gap | {_fmt(geom_p_gap)} | {_fmt(geom_v_gap)} | {_fmt(geom_delta['observed'])} | {_fmt(geom_delta['ci_low'])}–{_fmt(geom_delta['ci_high'])} |",
        f"| Internal LOSO disease AUC (Pearson) | {_fmt(loso_boot.get('observed'))} | — | — | {_fmt(loso_boot.get('ci_low'))}–{_fmt(loso_boot.get('ci_high'))} |",
        f"| ABIDE-II disease-axis AUC | {_fmt(ext_auc[P0]['auc'])} | {_fmt(ext_auc[V0]['auc'])} | {_fmt(paired_auc['observed'])} | {_fmt(paired_auc['ci_low'])}–{_fmt(paired_auc['ci_high'])} |",
        f"| ABIDE-II top-edge sign replication | {_fmt(edge_stats[P0]['sign_agreement'])} | {_fmt(edge_stats[V0]['sign_agreement'])} | — | — |",
        f"| ABIDE-II top-edge effect correlation (Spearman) | {_fmt(edge_stats[P0]['spearman'])} | {_fmt(edge_stats[V0]['spearman'])} | — | — |",
        "", "All uncertainty intervals above are subject- or site-bootstrap intervals, never pair-level standard errors.", "",
        "## SSL trajectory (10 external repeats; mean ± SD)", "",
        "| Epoch | Identity TOP1 | Identity gap | Disease AUC | Disease geometry | Split-half stability |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trajectory_table:
        lines.append(f"| {row.get('epoch_label', row['epoch'])} | {_fmt(row['identity_top1_mean'])} ± {_fmt(row['identity_top1_sd'])} | {_fmt(row['identity_gap_mean'])} ± {_fmt(row['identity_gap_sd'])} | {_fmt(row['disease_auc_mean'])} ± {_fmt(row['disease_auc_sd'])} | {_fmt(row['geometry_mean'])} ± {_fmt(row['geometry_sd'])} | {_fmt(row['stability_mean'])} ± {_fmt(row['stability_sd'])} |")
    lines += [
        "", f"Trajectory trade-off-consistent repeats (identity TOP1↑ and at least one disease-sharing metric↓): **{summary['h2d_tradeoff_repeats']}/10**.",
        "The fixed-state rank correlations for every repeat are retained in `statistics.json` (`trajectory_correlations`); selected states were evaluated once per repeat and are marked above rather than retrained.", "",
        "## Locked protocol and integrity", "",
        "- AICHA 384 ROIs (73,536 upper-triangle edges); original VarCoNet architecture and AICHA hyperparameters; 50 SSL epochs; validation-BCE selected encoder; 10 repeats × 10 internal folds.",
        "- P0 `pearson_windowavg` uses the exact one complete padded scan that Experiment 1D actually passes to VarCoNet. The original `test_augment()` (80/200/320, 10 windows) was audited separately and reproduced bitwise; it is not silently substituted for Experiment 1D stable extraction.",
        "- P1 `pearson_full` uses all real contiguous points. VarCoNet receives zero-padded inputs; Pearson receives the corresponding real points only.",
        "- Fingerprint views are deterministic first/last 80 valid points and are evaluated only when T_valid ≥ 160. Disease metrics use all 882 internal unique subjects, 37 Caltech subjects, and all 727 ABIDE-II rows.",
        "- Duplicate ABIDE-I scans are training-only; internal audit references/tests contain no duplicate names. ABIDE-II row IDs are the original 0–726 occurrence indices, preserving the preregistered 727-row external cohort.",
        "- Site IDs are used only for axis balancing, cross-site pair selection, LOSO, and bootstrap strata; never as model input. Official ABIDE-I and ABIDE-II phenotype mappings are 100% complete.",
        "- Root/prior source hashes were unchanged before/after aggregation; no worker errors; temporary trajectory/checkpoint shards were removed.",
        "", "## Interpretation", "",
    ]
    if summary.get("no_suppression_statement"):
        lines.append(f"**{summary['no_suppression_statement']}**")
    else:
        lines.append("The decision follows the locked H1/H2 rules above; it does not claim complete removal or preservation of disease information.")
    lines += [
        "", "## Required references audited", "",
        "- VarCoNet paper and repository: `2026 Human Brain Mapping VarCoNet` / `CharLamp10/VarCoNet-V2`.",
        "- Finn et al. (2015), *Nature Neuroscience*, DOI [10.1038/nn.4135](https://doi.org/10.1038/nn.4135).",
        "- *Biological Psychiatry* (2024), DOI [10.1016/j.biopsych.2023.09.012](https://doi.org/10.1016/j.biopsych.2023.09.012).",
        "- *Nature Mental Health* (2026), DOI [10.1038/s44220-026-00656-y](https://doi.org/10.1038/s44220-026-00656-y).",
        "", "## Output files", "",
        "`metrics.csv`, `trajectory.csv`, `statistics.json`, `summary.json`, `scores.npz`, and `top_edges.csv` are the compact reproducible outputs. No 73,536-dimensional feature matrix or permanent trajectory checkpoint is retained in `results/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1V")
    parser.add_argument("--repeat", type=int, default=None, help="run one repeat (0-9)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print_preregistered_plan(); return
    LOG_DIR.mkdir(parents=True, exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    if args.prepare:
        print(json.dumps(prepare_pearson_caches(cohort), ensure_ascii=False)); return
    if args.smoke:
        if not _pearson_cache_path("abide1").exists(): prepare_pearson_caches(cohort)
        run_smoke(cohort); return
    if args.aggregate:
        aggregate_results(); return
    if args.repeat is None:
        parser.error("one of --repeat, --prepare, --smoke, --aggregate is required")
    if int(args.repeat) < 0 or int(args.repeat) > 9: parser.error("repeat must be 0..9")
    if not _pearson_cache_path("abide1").exists(): raise RuntimeError("run --prepare before workers")
    run_repeat(cohort, int(args.repeat), torch.device(args.device))


if __name__ == "__main__":
    main()

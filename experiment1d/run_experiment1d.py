#!/usr/bin/env python3
"""Experiment 1D: three independent disease-sharing readouts.

The runner is deliberately self-contained under ``xiaolunwen/experiment1d``.
One audited VarCoNet SSL trajectory and one validation-selected baseline
classifier are shared by the baseline, DSLS, CSDA and ICDM branches for every
task.  No code from Experiment 1C is imported at runtime.
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
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import Adam


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = EXPERIMENT_DIR / "results"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
LOG_DIR = EXPERIMENT_DIR / "logs"
sys.path.insert(0, str(EXPERIMENT_DIR))

from csda import (  # noqa: E402
    CrossSiteDiseaseAdapter,
    cross_site_supcon_loss,
    diagnostics_from_model as csda_diagnostics_from_model,
    is_exact_zero_state as csda_is_exact_zero_state,
    load_adapter_state,
    state_payload as csda_state_payload,
    trainable_parameter_count as csda_parameter_count,
)
from dsls import (  # noqa: E402
    DSLSResidual,
    diagnostics_from_model as dsls_diagnostics_from_model,
    fit_dsls_state,
    is_exact_zero_state as dsls_is_exact_zero_state,
    load_residual_state as load_dsls_state,
    residual_state_payload as dsls_state_payload,
    state_from_payload as dsls_state_from_payload,
    state_payload as dsls_fit_state_payload,
    trainable_parameter_count as dsls_parameter_count,
)
from icdm import (  # noqa: E402
    IndividualConditionedDiseaseMask,
    diagnostics_from_model as icdm_diagnostics_from_model,
    is_exact_zero_state as icdm_is_exact_zero_state,
    load_mask_state,
    state_payload as icdm_state_payload,
    trainable_parameter_count as icdm_parameter_count,
)
from model_scripts.VarCoNet import VarCoNet  # noqa: E402
from model_scripts.classifier import MLP  # noqa: E402
from model_scripts.scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from shared_features import (  # noqa: E402
    ROIConnectivityProfile,
    SiteMetadata,
    ProfileState,
    build_site_metadata,
    fit_profile_state,
    margin_from_baseline_state,
    profile_state_from_payload,
    profile_state_payload,
)
from utils import DualBranchContrast, InfoNCE, augment, removeDuplicates  # noqa: E402


CONFIG: dict[str, Any] = {
    "atlas": "AICHA",
    "roi_count": 384,
    "stable_dim": 73536,
    "batch_size": 64,
    "min_length": 80,
    "epochs": 50,
    "warm_up_epochs": 10,
    "epochs_cls": 150,
    "lr_cls": 5e-5,
    "dsls_lr": 1e-3,
    "csda_lr": 1e-3,
    "icdm_lr": 1e-3,
    "csda_temperature": 0.1,
    "csda_lambda": 0.1,
    "checkpoint_interval": 10,
    "cpu_threads": 6,
    "dsls_components": 3,
    "dsls_hidden": 8,
    "dsls_trainable_parameters": 41,
    "csda_profile_dim": 384,
    "csda_embedding_dim": 32,
    "csda_trainable_parameters": 12321,
    "icdm_bottleneck": 16,
    "icdm_max_log_gate": 0.1,
    "icdm_trainable_parameters": 12288,
}

BASELINE_METHOD = "baseline"
DSLS_METHOD = "dsls"
CSDA_METHOD = "csda"
ICDM_METHOD = "icdm"
METHODS = (BASELINE_METHOD, DSLS_METHOD, CSDA_METHOD, ICDM_METHOD)

DSLS_DIAGNOSTIC_COLUMNS = (
    "dsls_singular1", "dsls_singular2", "dsls_singular3",
    "dsls_shared_energy_fraction", "dsls_asd_score_mean", "dsls_asd_score_std",
    "dsls_hc_score_mean", "dsls_hc_score_std", "dsls_std_floor",
    "dsls_parameter_l2", "dsls_mean_abs_delta",
)
CSDA_DIAGNOSTIC_COLUMNS = (
    "csda_num_sites", "csda_cross_site_positive_pairs", "csda_cross_site_same_y_cos",
    "csda_same_site_same_y_cos", "csda_opposite_y_cos", "csda_site_retrieval_accuracy",
    "csda_profile_std_floor", "csda_projection_l2", "csda_head_l2", "csda_mean_abs_delta",
)
ICDM_DIAGNOSTIC_COLUMNS = (
    "icdm_mean_abs_roi_prompt", "icdm_mean_abs_edge_log_gate", "icdm_mean_edge_factor",
    "icdm_std_edge_factor", "icdm_factor_min", "icdm_factor_max", "icdm_fraction_gt1",
    "icdm_mean_abs_fc_change", "icdm_down_l2", "icdm_up_l2",
)
ALL_DIAGNOSTIC_COLUMNS = DSLS_DIAGNOSTIC_COLUMNS + CSDA_DIAGNOSTIC_COLUMNS + ICDM_DIAGNOSTIC_COLUMNS
METRIC_COLUMNS = [
    "method", "scope", "repeat", "fold", "auc", "bce", "f1", "balanced_accuracy",
    "selected_encoder_epoch", "selected_classifier_epoch", "selected_dsls_epoch",
    "selected_csda_epoch", "selected_icdm_epoch", "threshold_hc",
    *ALL_DIAGNOSTIC_COLUMNS, "task_seconds", "n_test", "status",
]
FLOAT_COLUMNS = {
    "auc", "bce", "f1", "balanced_accuracy", "selected_encoder_epoch", "selected_classifier_epoch",
    "selected_dsls_epoch", "selected_csda_epoch", "selected_icdm_epoch", "threshold_hc",
    *ALL_DIAGNOSTIC_COLUMNS, "task_seconds",
}
INT_COLUMNS = {"repeat", "n_test"}
PREDICTION_FIELDS = (
    "scope", "method", "repeat", "fold", "subject_index", "raw_index", "y", "p_hc", "threshold",
)

# These values were audited before Experiment 1D was created.  The experiment
# directory itself is not included in this guard, so adding this runner cannot
# silently alter any earlier result.
EXPECTED_SOURCE_DIGESTS = {
    "model_scripts": "fa3af48175c37f0d03eda9863fa9ef1e5735e474cf34fb5f6751bb44733b34b0",
    "xiaolunwen/experiment1": "db87b36cdc5af09afb23701ed2591b6b93b51b7af35872c05fab61703f9e9f9b",
    "xiaolunwen/experiment1b": "a8c0d0fc0a1f2e04a69748004d242449948b34f4553417317ba944bed76b1ee3",
    "xiaolunwen/experiment1c": "62a986869622e00964431085008eb85b09c27e149ce86fab02bc6a5182f37197",
    "xiaolunwen/experiment2": "d26afee55cd10762fff9e60a417585150f8a876c62f3328487bd076c1a87481f",
    "xiaolunwen/experiment2b": "c5fce570f23f5e984b410c7010b35a0bafd561ae3f2fa78ea1884397978d98b6",
    "xiaolunwen/experiment3": "c074fd9cf4d8816d51a86a1c56bb1006bb03aa1ddc4135539c39e59594e20342",
    "utils.py": "aa1cd33e6f362dac51f9b71b252769da9b411676543f00a8d339ec7623dc0bfb",
    "ASD_classification_ABIDEI.py": "046b83d6caa8732d3fb389505f76573d58361b6c921b21719c0ce97f5c062665",
    "ASD_classification_ABIDEII.py": "d0d01a33139bcb97131250952ce2f8c4c6b72ae3efb19ad2d3d39727a07b5c10",
}


@dataclass
class Cohort:
    abide1: np.ndarray
    abide1_names: list[str]
    abide1_labels: np.ndarray
    abide1_sites: np.ndarray
    abide1_site_ids: np.ndarray
    duplicate_raw: np.ndarray
    duplicate_names: list[str]
    duplicate_labels: np.ndarray
    duplicate_sites: np.ndarray
    duplicate_site_ids: np.ndarray
    internal_raw: np.ndarray
    internal_names: list[str]
    internal_labels: np.ndarray
    internal_sites: np.ndarray
    internal_site_ids: np.ndarray
    caltech_raw: np.ndarray
    caltech_names: list[str]
    caltech_labels: np.ndarray
    caltech_sites: np.ndarray
    abide2: np.ndarray
    abide2_names: list[str]
    abide2_labels: np.ndarray
    site_metadata_source: str
    site_names: tuple[str, ...]


@dataclass
class Task:
    repeat: int
    scope: str
    fold: str
    seed: int
    reference_raw: np.ndarray
    reference_labels: np.ndarray
    reference_site_ids: np.ndarray
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
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
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


def clone_state_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


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
        if set(path.relative_to(directory).parts) & {"__pycache__", "logs", "results", "checkpoints", "dataset_cache"}:
            continue
        lines.append(f"{file_hash(path)}  {path.relative_to(REPO_ROOT).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def assert_external_integrity() -> dict[str, str]:
    observed: dict[str, str] = {}
    for key, expected in EXPECTED_SOURCE_DIGESTS.items():
        value = directory_digest(key) if not key.endswith(".py") else file_hash(REPO_ROOT / key)
        observed[key] = value
        if value != expected:
            raise RuntimeError(
                f"INTEGRITY_FAIL: source outside experiment1d changed: {key} expected={expected} observed={value}"
            )
    return observed


def load_npz_stack(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        values = [np.asarray(archive[key], dtype=np.float32) for key in archive.files]
    if not values:
        raise RuntimeError(f"empty dataset archive: {path}")
    return np.stack(values, axis=0)


def load_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def load_cohort() -> Cohort:
    """Load the audited cohorts and strictly map all ABIDE-I sites."""

    abide1_dir = DATASET_DIR / "ABIDEI"
    abide2_dir = DATASET_DIR / "ABIDEII"
    abide1 = load_npz_stack(abide1_dir / "ABIDEI_nilearn_AICHA.npz")
    abide1_names = load_names(abide1_dir / "ABIDEI_nilearn_names.txt")
    abide1_labels = np.load(abide1_dir / "ABIDEI_nilearn_classes.npy", allow_pickle=False).astype(np.int64)
    if len(abide1) != len(abide1_names) or len(abide1) != len(abide1_labels) or abide1.shape[2] != int(CONFIG["roi_count"]):
        raise RuntimeError(f"ABIDE-I data/name/label/ROI alignment drifted: {abide1.shape}, {len(abide1_names)}, {len(abide1_labels)}")
    metadata: SiteMetadata = build_site_metadata(abide1_names, DATASET_DIR)
    if len(metadata.raw_sites) != 995 or len(metadata.site_ids) != 995:
        raise RuntimeError(f"SITE_METADATA_FAIL: expected mappings for all 995 raw occurrences, got {len(metadata.raw_sites)}")
    raw_sites = np.asarray(metadata.raw_sites, dtype="<U64")
    raw_site_ids = np.asarray(metadata.site_ids, dtype=np.int64)
    names_array = np.asarray(abide1_names)
    unique_names, counts = np.unique(names_array, return_counts=True)
    duplicate_values = unique_names[counts > 1]
    duplicate_raw_list: list[int] = []
    duplicate_names: list[str] = []
    for name in duplicate_values:
        positions = np.where(names_array == name)[0]
        duplicate_raw_list.extend(int(value) for value in positions.tolist())
        duplicate_names.extend([str(name)] * len(positions))
    singleton_names = [str(name) for name in unique_names[counts == 1].tolist()]
    name_to_raw = {name: int(np.where(names_array == name)[0][0]) for name in singleton_names}
    singleton_raw = np.asarray([name_to_raw[name] for name in singleton_names], dtype=np.int64)
    duplicate_raw = np.asarray(duplicate_raw_list, dtype=np.int64)
    caltech_names = [
        f"sub-00{numeric_id}" for numeric_id in range(51456, 51494)
        if f"sub-00{numeric_id}" in name_to_raw
    ]
    if len(caltech_names) != 37:
        raise RuntimeError(f"audited Caltech count drifted: {len(caltech_names)}")
    caltech_raw = np.asarray([name_to_raw[name] for name in caltech_names], dtype=np.int64)
    if not np.all(raw_sites[caltech_raw] == "CALTECH"):
        raise RuntimeError("SITE_METADATA_FAIL: a Caltech processed subject is not mapped to CALTECH")
    caltech_set = set(caltech_names)
    internal_names = [name for name in singleton_names if name not in caltech_set]
    internal_raw = np.asarray([name_to_raw[name] for name in internal_names], dtype=np.int64)
    if len(singleton_raw) != 919 or len(internal_raw) != 882 or len(duplicate_raw) != 76:
        raise RuntimeError(f"audited ABIDE-I counts drifted: singleton={len(singleton_raw)} internal={len(internal_raw)} duplicate={len(duplicate_raw)}")
    abide2 = load_npz_stack(abide2_dir / "ABIDEII_nilearn_AICHA.npz")
    abide2_names = load_names(abide2_dir / "ABIDEII_nilearn_names.txt")
    abide2_labels = np.load(abide2_dir / "ABIDEII_nilearn_classes.npy", allow_pickle=False).astype(np.int64)
    if len(abide2) != len(abide2_names) or len(abide2) != len(abide2_labels) or len(abide2) != 727:
        raise RuntimeError("ABIDE-II must remain the complete original 727-subject cohort")
    if abide1.shape[1:] != abide2.shape[1:]:
        raise RuntimeError(f"ABIDE atlas shapes differ: {abide1.shape} vs {abide2.shape}")
    duplicate_sites = raw_sites[duplicate_raw]
    for name in set(duplicate_names):
        values = set(duplicate_sites[np.asarray([i for i, item in enumerate(duplicate_names) if item == name], dtype=np.int64)].tolist())
        if len(values) != 1:
            raise RuntimeError(f"SITE_METADATA_FAIL: duplicate scan site mismatch for {name}: {values}")
    return Cohort(
        abide1=abide1, abide1_names=abide1_names, abide1_labels=abide1_labels,
        abide1_sites=raw_sites, abide1_site_ids=raw_site_ids,
        duplicate_raw=duplicate_raw, duplicate_names=duplicate_names,
        duplicate_labels=abide1_labels[duplicate_raw], duplicate_sites=raw_sites[duplicate_raw],
        duplicate_site_ids=raw_site_ids[duplicate_raw],
        internal_raw=internal_raw, internal_names=internal_names,
        internal_labels=abide1_labels[internal_raw], internal_sites=raw_sites[internal_raw],
        internal_site_ids=raw_site_ids[internal_raw], caltech_raw=caltech_raw,
        caltech_names=caltech_names, caltech_labels=abide1_labels[caltech_raw],
        caltech_sites=raw_sites[caltech_raw], abide2=abide2, abide2_names=abide2_names,
        abide2_labels=abide2_labels, site_metadata_source=metadata.source_path,
        site_names=metadata.site_names,
    )


def make_tasks(cohort: Cohort, repeat: int) -> list[Task]:
    repeat = int(repeat)
    tasks: list[Task] = []
    raw_to_name = {int(raw): name for raw, name in zip(cohort.internal_raw, cohort.internal_names)}
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + repeat)
    for fold, (outer_train_idx, test_idx) in enumerate(splitter.split(cohort.internal_raw, cohort.internal_labels)):
        outer_raw = cohort.internal_raw[outer_train_idx]
        outer_y = cohort.internal_labels[outer_train_idx]
        train_unique_raw, val_raw, train_unique_y, val_y, _, _ = train_test_split(
            outer_raw, outer_y, np.arange(len(outer_raw)), test_size=0.15,
            random_state=42, stratify=outer_y,
        )
        if np.intersect1d(train_unique_raw, cohort.duplicate_raw).size:
            raise RuntimeError("reference_raw contains a duplicate occurrence")
        tasks.append(Task(
            repeat=repeat, scope="internal_cv", fold=str(fold), seed=100000 + repeat * 100 + fold,
            reference_raw=np.asarray(train_unique_raw, dtype=np.int64),
            reference_labels=np.asarray(train_unique_y, dtype=np.int64),
            reference_site_ids=cohort.abide1_site_ids[train_unique_raw].copy(),
            train_raw=np.concatenate((cohort.duplicate_raw, np.asarray(train_unique_raw, dtype=np.int64))),
            train_names=cohort.duplicate_names + [raw_to_name[int(raw)] for raw in train_unique_raw],
            train_labels=np.concatenate((cohort.abide1_labels[cohort.duplicate_raw], np.asarray(train_unique_y, dtype=np.int64))),
            val_raw=np.asarray(val_raw, dtype=np.int64), val_labels=np.asarray(val_y, dtype=np.int64),
            test_raw=np.asarray(cohort.internal_raw[test_idx], dtype=np.int64),
            test_labels=np.asarray(cohort.internal_labels[test_idx], dtype=np.int64),
        ))
    ext_train_raw, ext_val_raw, ext_train_y, ext_val_y, _, _ = train_test_split(
        cohort.internal_raw, cohort.internal_labels, np.arange(len(cohort.internal_raw)), test_size=0.10,
        random_state=42 + repeat, stratify=cohort.internal_labels,
    )
    if np.intersect1d(ext_train_raw, cohort.duplicate_raw).size:
        raise RuntimeError("external reference_raw contains a duplicate occurrence")
    tasks.append(Task(
        repeat=repeat, scope="external", fold="external", seed=200000 + repeat,
        reference_raw=np.asarray(ext_train_raw, dtype=np.int64),
        reference_labels=np.asarray(ext_train_y, dtype=np.int64),
        reference_site_ids=cohort.abide1_site_ids[ext_train_raw].copy(),
        train_raw=np.concatenate((cohort.duplicate_raw, np.asarray(ext_train_raw, dtype=np.int64))),
        train_names=cohort.duplicate_names + [raw_to_name[int(raw)] for raw in ext_train_raw],
        train_labels=np.concatenate((cohort.abide1_labels[cohort.duplicate_raw], np.asarray(ext_train_y, dtype=np.int64))),
        val_raw=np.asarray(ext_val_raw, dtype=np.int64), val_labels=np.asarray(ext_val_y, dtype=np.int64),
        test_raw=np.asarray(cohort.caltech_raw, dtype=np.int64), test_labels=np.asarray(cohort.caltech_labels, dtype=np.int64),
    ))
    return tasks


def paired_batch_schedule(names: list[str], n_items: int, epochs: int, seed: int) -> list[list[list[int]]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    schedule: list[list[list[int]]] = []
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(int(n_items), generator=generator).tolist()
        batches: list[list[int]] = []
        for batch_index, start in enumerate(range(0, len(permutation), int(CONFIG["batch_size"]))):
            selected = [int(index) for index in permutation[start:start + int(CONFIG["batch_size"])]]
            saved_python = random.getstate()
            random.seed(int(seed) + epoch * 1000003 + batch_index)
            selected = removeDuplicates(names, selected)
            random.setstate(saved_python)
            batches.append([int(index) for index in selected])
        schedule.append(batches)
    return schedule


def finite_gradients(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())


def extract_stable_features(encoder: VarCoNet, data: np.ndarray, raw_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    encoder.eval()
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(raw_indices), int(CONFIG["batch_size"])):
            selected = np.asarray(raw_indices[start:start + int(CONFIG["batch_size"])], dtype=np.int64)
            x = torch.from_numpy(np.asarray(data[selected], dtype=np.float32)).to(device)
            pieces.append(encoder(x).detach())
            del x
    if not pieces:
        raise RuntimeError("cannot extract stable features from an empty partition")
    stable = torch.cat(pieces, dim=0)
    if stable.ndim != 2 or tuple(stable.shape[1:]) != (int(CONFIG["stable_dim"]),):
        raise RuntimeError(f"stable learned-FC shape drifted: {tuple(stable.shape)}")
    if not bool(torch.isfinite(stable).all()):
        raise FloatingPointError("non-finite stable learned FC")
    return stable


def validation_threshold(labels: np.ndarray, p_hc: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(np.asarray(labels, dtype=int), np.asarray(p_hc, dtype=float))
    youden = tpr - fpr
    candidates = np.where(np.isclose(youden, np.nanmax(youden), rtol=0.0, atol=1e-12))[0]
    finite = [int(index) for index in candidates.tolist() if np.isfinite(thresholds[index])]
    choice = finite[0] if finite else int(candidates[0])
    return float(thresholds[choice]) if np.isfinite(thresholds[choice]) else 0.5


def classification_metrics(labels: np.ndarray, p_hc: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    p_hc = np.asarray(p_hc, dtype=np.float64)
    pred_hc = (p_hc >= float(threshold)).astype(np.int64)
    asd = (labels == 0).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, p_hc)),
        "bce": float(log_loss(labels, np.clip(p_hc, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "f1": float(f1_score(asd, (pred_hc == 0).astype(np.int64), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred_hc)),
    }


def metrics_with_thresholds(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pred_hc = (probabilities >= thresholds).astype(np.int64)
    asd = (labels == 0).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "bce": float(log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "f1": float(f1_score(asd, (pred_hc == 0).astype(np.int64), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred_hc)),
    }


def make_model_config(cohort: Cohort) -> tuple[dict[str, int], dict[str, Any]]:
    with (EXPERIMENT_DIR / "best_params_VarCoNet_AICHA.pkl").open("rb") as handle:
        best_params = pickle.load(handle)
    config = {
        "layers": int(best_params["layers"]), "n_heads": int(best_params["n_heads"]),
        "dim_feedforward": int(best_params["dim_feedforward"]), "max_length": int(cohort.abide1.shape[1]),
    }
    return config, best_params


def create_encoder(cohort: Cohort, device: torch.device) -> VarCoNet:
    model_config, _ = make_model_config(cohort)
    return VarCoNet(model_config, int(cohort.abide1.shape[2])).to(device)


def predict_baseline_tensor(stable: torch.Tensor, state: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    classifier = MLP(int(stable.shape[1]), 2).to(device)
    classifier.load_state_dict(state)
    classifier.eval()
    return classifier(stable)


def predict_baseline(stable: torch.Tensor, state: dict[str, torch.Tensor], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return predict_baseline_tensor(stable, state, device)[:, 1].detach().cpu().numpy().astype(np.float64)


def fit_baseline_classifier(
    stable_train: torch.Tensor, train_labels: torch.Tensor, stable_val: torch.Tensor,
    val_labels: torch.Tensor, device: torch.device, seed: int,
) -> dict[str, Any]:
    set_all_seeds(seed)
    classifier = MLP(int(stable_train.shape[1]), 2).to(device)
    optimizer = Adam(classifier.parameters(), lr=float(CONFIG["lr_cls"]))
    criterion = torch.nn.BCELoss()
    target_train = F.one_hot(train_labels, num_classes=2).float()
    target_val = F.one_hot(val_labels, num_classes=2).float()
    best_bce, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, int(CONFIG["epochs_cls"])):
        classifier.train()
        optimizer.zero_grad()
        loss = criterion(classifier(stable_train), target_train)
        loss.backward()
        if not bool(torch.isfinite(loss)) or not finite_gradients(classifier):
            raise FloatingPointError("non-finite baseline classifier loss or gradient")
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            val_bce = float(criterion(classifier(stable_val), target_val).detach().cpu())
        if val_bce < best_bce:
            best_bce, best_epoch, best_state = float(val_bce), int(epoch), clone_state_cpu(classifier.state_dict())
    if best_state is None or best_epoch is None:
        raise RuntimeError("baseline classifier did not select a validation state")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        p_hc = classifier(stable_val)[:, 1].detach().cpu().numpy().astype(np.float64)
    return {
        "classifier_state": best_state, "val_bce": float(best_bce), "classifier_epoch": int(best_epoch),
        "val_probs": p_hc, "threshold": validation_threshold(val_labels.detach().cpu().numpy(), p_hc),
    }


def train_ssl_epoch(
    encoder: VarCoNet, contrast: DualBranchContrast, optimizer: torch.optim.Optimizer,
    cohort: Cohort, task: Task, batches: list[list[int]], max_length: int, device: torch.device,
) -> float:
    encoder.train()
    losses: list[float] = []
    for selected in batches:
        local_indices = np.asarray(selected, dtype=np.int64)
        batch = torch.from_numpy(np.asarray(cohort.abide1[task.train_raw[local_indices]], dtype=np.float32))
        views = augment(batch, [int(CONFIG["min_length"]), int(max_length)], device)
        optimizer.zero_grad()
        z1, z2 = encoder(views[0]), encoder(views[1])
        loss = contrast(z1, z2)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite SSL loss")
        loss.backward()
        if not finite_gradients(encoder):
            raise FloatingPointError("non-finite SSL gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        del batch, views, z1, z2, loss
    if not losses:
        raise RuntimeError("empty SSL epoch")
    return float(np.mean(losses))


def gpu_memory_gb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    return float(torch.cuda.memory_allocated()) / (1024 ** 3), float(torch.cuda.memory_reserved()) / (1024 ** 3)


def task_identity(task: Task) -> dict[str, Any]:
    return {"repeat": int(task.repeat), "scope": str(task.scope), "fold": str(task.fold), "task_seed": int(task.seed)}


def identity_matches(payload: dict[str, Any], task: Task) -> bool:
    return all(payload.get(key) == value for key, value in task_identity(task).items())


def checkpoint_paths(repeat: int) -> tuple[Path, Path]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"r{int(repeat)}")
    return CHECKPOINT_DIR / f"worker_{token}_latest.pt", CHECKPOINT_DIR / f"worker_{token}_best.pt"


def checkpoint_rank(payload: dict[str, Any]) -> tuple[int, int]:
    return {"ssl": 0, "dsls": 1, "csda": 2, "icdm": 3}.get(str(payload.get("phase")), -1), int(payload.get("epoch", -1))


def load_task_checkpoint(repeat: int, task: Task) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for path in checkpoint_paths(repeat):
        if not path.exists():
            continue
        try:
            payload = torch_load_cpu(path)
        except Exception:
            path.unlink(missing_ok=True)
            continue
        if identity_matches(payload, task):
            candidates.append(payload)
    return max(candidates, key=checkpoint_rank) if candidates else None


def save_checkpoint(path: Path, kind: str, payload: dict[str, Any]) -> None:
    atomic_torch_save(path, payload)
    print(
        f"CHECKPOINT_{kind.upper()} repeat={payload['repeat']} scope={payload['scope']} fold={payload['fold']} "
        f"phase={payload['phase']} epoch={payload['epoch']} path={path}", flush=True,
    )


def clean_task_checkpoints(repeat: int, task: Task) -> None:
    for path in checkpoint_paths(repeat):
        if not path.exists():
            continue
        try:
            payload = torch_load_cpu(path)
            if identity_matches(payload, task):
                path.unlink(missing_ok=True)
        except Exception:
            path.unlink(missing_ok=True)


def base_checkpoint_payload(task: Task, phase: str, epoch: int, encoder: VarCoNet, best_baseline: dict[str, Any] | None, elapsed_seconds: float, **extra: Any) -> dict[str, Any]:
    return {
        "format_version": 1, "phase": str(phase), "saved_unix": time.time(), **task_identity(task),
        "epoch": int(epoch), "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "best_baseline": state_to_cpu(best_baseline), "rng_state": state_to_cpu(capture_rng_state()),
        "elapsed_seconds": float(elapsed_seconds), **state_to_cpu(extra),
    }


def make_baseline_branch(encoder: VarCoNet, encoder_epoch: int, fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()), "classifier_state": state_to_cpu(fit["classifier_state"]),
        "val_bce": float(fit["val_bce"]), "encoder_epoch": int(encoder_epoch),
        "classifier_epoch": int(fit["classifier_epoch"]), "threshold": float(fit["threshold"]),
        "val_probs": np.asarray(fit["val_probs"], dtype=np.float64).copy(), "dsls_diagnostics": None,
        "csda_diagnostics": None, "icdm_diagnostics": None,
    }


def assert_tensor_state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], label: str) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"INTEGRITY_FAIL: {label} state key mismatch")
    for key, value in left.items():
        if not torch.equal(value, right[key]):
            raise RuntimeError(f"INTEGRITY_FAIL: {label} differs at {key}")


def assert_shared_selection(task_result: dict[str, Any]) -> None:
    baseline = task_result[BASELINE_METHOD]
    for method in METHODS[1:]:
        branch = task_result[method]
        if int(branch["encoder_epoch"]) != int(baseline["encoder_epoch"]) or int(branch["classifier_epoch"]) != int(baseline["classifier_epoch"]):
            raise RuntimeError(f"INTEGRITY_FAIL: baseline/{method} selected encoder/classifier differs")
        assert_tensor_state_equal(baseline["encoder_state"], branch["encoder_state"], f"baseline/{method} encoder")
        assert_tensor_state_equal(baseline["classifier_state"], branch["baseline_classifier_state"], f"baseline/{method} classifier")


def binary_bce(p_hc: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(p_hc, (labels == 1).float())


def zero_parity_diagnostics(
    method: str, details: dict[str, torch.Tensor], stable_val: torch.Tensor, val_labels: torch.Tensor,
    baseline_state: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        baseline_out = predict_baseline_tensor(stable_val, baseline_state, device)
        zero_out = torch.stack((1.0 - details["p_hc"], details["p_hc"]), dim=1)
        direct_margin = margin_from_baseline_state(stable_val, baseline_state)
        criterion = torch.nn.BCELoss()
        baseline_bce = criterion(baseline_out, F.one_hot(val_labels, num_classes=2).float())
        zero_bce = criterion(zero_out, F.one_hot(val_labels, num_classes=2).float())
        diagnostics = {
            "zero_probability_max_abs": float((baseline_out - zero_out).abs().max().cpu()),
            "zero_margin_max_abs": float((details["m0"] - direct_margin).abs().max().cpu()),
            "zero_bce_difference": float((zero_bce - baseline_bce).abs().cpu()),
            "zero_binary_bce_difference": float((binary_bce(details["p_hc"], val_labels) - baseline_bce).abs().cpu()),
            "baseline_val_bce": float(baseline_bce.cpu()),
            "method": method,
        }
    if max(value for key, value in diagnostics.items() if key not in {"baseline_val_bce", "method"}) > 1e-6:
        raise RuntimeError(f"INTEGRITY_FAIL: exact-zero {method} does not equal frozen baseline")
    return diagnostics


def profile_features(profile_module: ROIConnectivityProfile, stable: torch.Tensor) -> torch.Tensor:
    profile = profile_module(stable)
    if profile.ndim != 2 or tuple(profile.shape[1:]) != (int(CONFIG["roi_count"]),) or not bool(torch.isfinite(profile).all()):
        raise RuntimeError(f"ROI profile shape/finite check failed: {tuple(profile.shape)}")
    return profile


def _allclose_nested(left: Any, right: Any, atol: float = 1e-6) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and bool(torch.allclose(left.cpu(), right.cpu(), rtol=0.0, atol=atol))
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_allclose_nested(left[key], right[key], atol) for key in left)
    if isinstance(left, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def dsls_validation(model: DSLSResidual, stable_val: torch.Tensor, val_labels: torch.Tensor) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable_val)
        output = torch.stack((1.0 - details["p_hc"], details["p_hc"]), dim=1)
        bce = float(torch.nn.BCELoss()(output, F.one_hot(val_labels, num_classes=2).float()).cpu())
        return bce, details["p_hc"].detach().cpu().numpy().astype(np.float64)


def csda_validation(model: CrossSiteDiseaseAdapter, stable_val: torch.Tensor, profile_val: torch.Tensor, val_labels: torch.Tensor) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable_val, profile_val)
        output = torch.stack((1.0 - details["p_hc"], details["p_hc"]), dim=1)
        bce = float(torch.nn.BCELoss()(output, F.one_hot(val_labels, num_classes=2).float()).cpu())
        return bce, details["p_hc"].detach().cpu().numpy().astype(np.float64)


def icdm_validation(model: IndividualConditionedDiseaseMask, stable_val: torch.Tensor, profile_val: torch.Tensor, val_labels: torch.Tensor) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable_val, profile_val)
        output = torch.stack((1.0 - details["p_hc"], details["p_hc"]), dim=1)
        bce = float(torch.nn.BCELoss()(output, F.one_hot(val_labels, num_classes=2).float()).cpu())
        return bce, details["p_hc"].detach().cpu().numpy().astype(np.float64)


def dsls_diagnostics(model: DSLSResidual, dsls_state: Any, stable_reference: torch.Tensor, reference_labels: torch.Tensor) -> dict[str, float]:
    diagnostic = dsls_diagnostics_from_model(model, stable_reference, reference_labels)
    singular = dsls_state.singular_values.detach().cpu().numpy()
    asd_mean = dsls_state.asd_score_mean.detach().cpu().numpy()
    asd_std = dsls_state.asd_score_std.detach().cpu().numpy()
    hc_scores = model.shared_scores(stable_reference)[reference_labels == 1].detach().cpu()
    diagnostic.update({
        "dsls_singular1": float(singular[0]), "dsls_singular2": float(singular[1]), "dsls_singular3": float(singular[2]),
        "dsls_shared_energy_fraction": float(dsls_state.shared_energy_fraction),
        "dsls_asd_score_mean": float(np.mean(asd_mean)), "dsls_asd_score_std": float(np.mean(asd_std)),
        "dsls_hc_score_mean": float(hc_scores.mean().item()) if hc_scores.numel() else float("nan"),
        "dsls_hc_score_std": float(hc_scores.std(unbiased=False).item()) if hc_scores.numel() else float("nan"),
        "dsls_std_floor": float(dsls_state.std_floor),
    })
    return diagnostic


def prepare_dsls_branch(
    task: Task, encoder: VarCoNet, best_baseline: dict[str, Any], stable_reference: torch.Tensor,
    stable_train: torch.Tensor, stable_val: torch.Tensor, reference_labels: torch.Tensor,
    train_labels: torch.Tensor, val_labels: torch.Tensor, dsls_state: Any, source: dict[str, Any] | None,
    source_phase_rank: int, started: float, latest_path: Path, best_path: Path,
) -> dict[str, Any]:
    """Train or resume DSLS; no validation/test feature is used for fitting."""

    model = DSLSResidual(best_baseline["classifier_state"], dsls_state).to(stable_train.device)
    if dsls_parameter_count(model) != int(CONFIG["dsls_trainable_parameters"]):
        raise RuntimeError(f"INTEGRITY_FAIL: DSLS parameter count={dsls_parameter_count(model)} expected=41")
    zero_state = dsls_state_payload(model)
    if not dsls_is_exact_zero_state(zero_state):
        raise RuntimeError("INTEGRITY_FAIL: DSLS output head is not exact-zero initialized")
    zero_diag = zero_parity_diagnostics(DSLS_METHOD, model.forward_details(stable_val), stable_val, val_labels, best_baseline["classifier_state"], stable_train.device)
    print(f"[DSLS_FIT] repeat={task.repeat} scope={task.scope} fold={task.fold} n_asd={int((reference_labels==0).sum())} n_hc={int((reference_labels==1).sum())} singular=({dsls_state.singular_values[0]:.6f},{dsls_state.singular_values[1]:.6f},{dsls_state.singular_values[2]:.6f}) shared_energy_fraction={dsls_state.shared_energy_fraction:.6f} asd_score_mean={float(dsls_state.asd_score_mean.mean()):.6g} asd_score_std={float(dsls_state.asd_score_std.mean()):.6g} std_floor={dsls_state.std_floor:.8g}", flush=True)
    optimizer = Adam(model.parameters(), lr=float(CONFIG["dsls_lr"]))
    if source is not None and source_phase_rank == 1 and source.get("dsls_progress") is not None:
        progress = source["dsls_progress"]
        if not _allclose_nested(progress["zero_diagnostics"], zero_diag):
            raise RuntimeError("INTEGRITY_FAIL: resumed DSLS zero diagnostic changed")
        load_dsls_state(model, progress["current_state"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        optimizer_to_device(optimizer, stable_train.device)
        best_state = state_to_cpu(progress["best_state"])
        best_epoch = int(progress["best_epoch"])
        best_bce = float(progress["best_val_bce"])
        start_epoch = int(source["epoch"]) + 1
        print(f"DSLS_RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} from_epoch={start_epoch} best_epoch={best_epoch} best_val_bce={best_bce:.6f}", flush=True)
    else:
        best_state, best_epoch, best_bce, start_epoch = state_to_cpu(zero_state), 0, float(zero_diag["baseline_val_bce"]), 1
        progress = {
            "zero_state": state_to_cpu(zero_state), "zero_diagnostics": state_to_cpu(zero_diag),
            "dsls_state": dsls_fit_state_payload(dsls_state), "current_state": state_to_cpu(zero_state),
            "optimizer_state_dict": state_to_cpu(optimizer.state_dict()), "best_state": state_to_cpu(best_state),
            "best_epoch": best_epoch, "best_val_bce": best_bce,
        }
        payload = base_checkpoint_payload(task, "dsls", 0, encoder, best_baseline, time.time() - started, dsls_progress=progress)
        save_checkpoint(latest_path, "latest", payload)
        save_checkpoint(best_path, "best", payload)
    for epoch in range(start_epoch, int(CONFIG["epochs_cls"])):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad()
        details = model.forward_details(stable_train)
        loss = binary_bce(details["p_hc"], train_labels)
        loss.backward()
        if not bool(torch.isfinite(loss)) or not finite_gradients(model):
            raise FloatingPointError("non-finite DSLS BCE or gradient")
        optimizer.step()
        val_bce, _ = dsls_validation(model, stable_val, val_labels)
        is_best = val_bce < best_bce
        if is_best:
            best_bce, best_epoch, best_state = float(val_bce), int(epoch), dsls_state_payload(model)
        progress = {
            "zero_state": state_to_cpu(zero_state), "zero_diagnostics": state_to_cpu(zero_diag),
            "dsls_state": dsls_fit_state_payload(dsls_state), "current_state": dsls_state_payload(model),
            "optimizer_state_dict": state_to_cpu(optimizer.state_dict()), "best_state": state_to_cpu(best_state),
            "best_epoch": int(best_epoch), "best_val_bce": float(best_bce),
        }
        payload = base_checkpoint_payload(task, "dsls", epoch, encoder, best_baseline, time.time() - started, dsls_progress=progress)
        if is_best:
            save_checkpoint(best_path, "best", payload)
        if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs_cls"]) - 1:
            save_checkpoint(latest_path, "latest", payload)
        if epoch % 10 == 0 or is_best or epoch == int(CONFIG["epochs_cls"]) - 1:
            diag = dsls_diagnostics(model, dsls_state, stable_reference, reference_labels)
            print(f"[DSLS] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs_cls']-1} train_bce={float(loss.detach().cpu()):.6f} val_bce={val_bce:.6f} best={'Y' if is_best else 'N'} best_epoch={best_epoch} parameter_l2={diag['dsls_parameter_l2']:.6f} mean_abs_delta={diag['dsls_mean_abs_delta']:.6f} epoch_seconds={time.time()-epoch_started:.2f}", flush=True)
    load_dsls_state(model, best_state)
    final_bce, final_probs = dsls_validation(model, stable_val, val_labels)
    if abs(final_bce - best_bce) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: selected DSLS state does not reproduce validation BCE")
    if best_epoch == 0:
        if not dsls_is_exact_zero_state(best_state) or float(np.max(np.abs(final_probs - np.asarray(best_baseline["val_probs"], dtype=np.float64)))) > 1e-6:
            raise RuntimeError("INTEGRITY_FAIL: DSLS epoch-0 test/validation parity failed")
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()), "baseline_classifier_state": state_to_cpu(best_baseline["classifier_state"]),
        "dsls_state": dsls_fit_state_payload(dsls_state), "residual_state": state_to_cpu(best_state),
        "val_bce": float(best_bce), "encoder_epoch": int(best_baseline["encoder_epoch"]), "classifier_epoch": int(best_baseline["classifier_epoch"]),
        "dsls_epoch": int(best_epoch), "threshold": validation_threshold(task.val_labels, final_probs), "val_probs": final_probs,
        "dsls_diagnostics": dsls_diagnostics(model, dsls_state, stable_reference, reference_labels),
        "csda_diagnostics": None, "icdm_diagnostics": None,
    }


def prepare_csda_branch(
    task: Task, encoder: VarCoNet, best_baseline: dict[str, Any], stable_reference: torch.Tensor,
    stable_train: torch.Tensor, stable_val: torch.Tensor, profile_reference: torch.Tensor,
    profile_train: torch.Tensor, profile_val: torch.Tensor, reference_labels: torch.Tensor,
    reference_site_ids: torch.Tensor, train_labels: torch.Tensor, val_labels: torch.Tensor,
    profile_state: ProfileState, dsls_branch: dict[str, Any], source: dict[str, Any] | None,
    source_phase_rank: int, started: float, latest_path: Path, best_path: Path,
) -> dict[str, Any]:
    model = CrossSiteDiseaseAdapter(best_baseline["classifier_state"], profile_state).to(stable_train.device)
    if csda_parameter_count(model) != int(CONFIG["csda_trainable_parameters"]):
        raise RuntimeError(f"INTEGRITY_FAIL: CSDA parameter count={csda_parameter_count(model)} expected=12321")
    positive_mask = (reference_labels[:, None] == reference_labels[None, :]) & (reference_site_ids[:, None] != reference_site_ids[None, :])
    negative_mask = (reference_labels[:, None] != reference_labels[None, :])
    positive_pairs = int(positive_mask.sum().detach().cpu())
    negative_pairs = int(negative_mask.sum().detach().cpu())
    if positive_pairs == 0:
        raise RuntimeError("CSDA insufficient cross-site same-diagnosis positives")
    site_counts = {str(int(site)): int((reference_site_ids == site).sum().detach().cpu()) for site in torch.unique(reference_site_ids).tolist()}
    print(f"[CSDA_SITE] repeat={task.repeat} scope={task.scope} fold={task.fold} n_sites={len(site_counts)} site_counts={json.dumps(site_counts, sort_keys=True)} cross_site_positive_pairs={positive_pairs} negative_pairs={negative_pairs}", flush=True)
    zero_state = csda_state_payload(model)
    if not csda_is_exact_zero_state(zero_state):
        raise RuntimeError("INTEGRITY_FAIL: CSDA head is not exact-zero initialized")
    zero_diag = zero_parity_diagnostics(CSDA_METHOD, model.forward_details(stable_val, profile_val), stable_val, val_labels, best_baseline["classifier_state"], stable_train.device)
    optimizer = Adam(model.parameters(), lr=float(CONFIG["csda_lr"]))
    if source is not None and source_phase_rank == 2 and source.get("csda_progress") is not None:
        progress = source["csda_progress"]
        if not _allclose_nested(progress["zero_diagnostics"], zero_diag):
            raise RuntimeError("INTEGRITY_FAIL: resumed CSDA zero diagnostic changed")
        load_adapter_state(model, progress["current_state"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        optimizer_to_device(optimizer, stable_train.device)
        best_state, best_epoch, best_bce = state_to_cpu(progress["best_state"]), int(progress["best_epoch"]), float(progress["best_val_bce"])
        start_epoch = int(source["epoch"]) + 1
        print(f"CSDA_RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} from_epoch={start_epoch} best_epoch={best_epoch} best_val_bce={best_bce:.6f}", flush=True)
    else:
        best_state, best_epoch, best_bce, start_epoch = state_to_cpu(zero_state), 0, float(zero_diag["baseline_val_bce"]), 1
        progress = {
            "zero_state": state_to_cpu(zero_state), "zero_diagnostics": state_to_cpu(zero_diag), "profile_state": profile_state_payload(profile_state),
            "current_state": state_to_cpu(zero_state), "optimizer_state_dict": state_to_cpu(optimizer.state_dict()), "best_state": state_to_cpu(best_state),
            "best_epoch": best_epoch, "best_val_bce": best_bce,
        }
        payload = base_checkpoint_payload(task, "csda", 0, encoder, best_baseline, time.time() - started, dsls_branch=dsls_branch, csda_progress=progress)
        save_checkpoint(latest_path, "latest", payload)
        save_checkpoint(best_path, "best", payload)
    for epoch in range(start_epoch, int(CONFIG["epochs_cls"])):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad()
        train_details = model.forward_details(stable_train, profile_train)
        bce = binary_bce(train_details["p_hc"], train_labels)
        supcon = cross_site_supcon_loss(model.disease_embedding(profile_reference), reference_labels, reference_site_ids, tau=float(CONFIG["csda_temperature"]))
        total = bce + float(CONFIG["csda_lambda"]) * supcon
        total.backward()
        if not bool(torch.isfinite(total)) or not finite_gradients(model):
            raise FloatingPointError("non-finite CSDA total loss or gradient")
        optimizer.step()
        val_bce, _ = csda_validation(model, stable_val, profile_val, val_labels)
        is_best = val_bce < best_bce
        if is_best:
            best_bce, best_epoch, best_state = float(val_bce), int(epoch), csda_state_payload(model)
        progress = {
            "zero_state": state_to_cpu(zero_state), "zero_diagnostics": state_to_cpu(zero_diag), "profile_state": profile_state_payload(profile_state),
            "current_state": csda_state_payload(model), "optimizer_state_dict": state_to_cpu(optimizer.state_dict()), "best_state": state_to_cpu(best_state),
            "best_epoch": int(best_epoch), "best_val_bce": float(best_bce),
        }
        payload = base_checkpoint_payload(task, "csda", epoch, encoder, best_baseline, time.time() - started, dsls_branch=dsls_branch, csda_progress=progress)
        if is_best:
            save_checkpoint(best_path, "best", payload)
        if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs_cls"]) - 1:
            save_checkpoint(latest_path, "latest", payload)
        if epoch % 10 == 0 or is_best or epoch == int(CONFIG["epochs_cls"]) - 1:
            diag = csda_diagnostics_from_model(model, stable_reference, profile_reference, reference_labels, reference_site_ids)
            print(f"[CSDA] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs_cls']-1} BCE={float(bce.detach().cpu()):.6f} SupCon={float(supcon.detach().cpu()):.6f} total={float(total.detach().cpu()):.6f} val_bce={val_bce:.6f} best={'Y' if is_best else 'N'} best_epoch={best_epoch} cross_site_same_y_cos={diag['csda_cross_site_same_y_cos']:.6f} same_site_same_y_cos={diag['csda_same_site_same_y_cos']:.6f} opposite_y_cos={diag['csda_opposite_y_cos']:.6f} head_l2={diag['csda_head_l2']:.6f} proj_l2={diag['csda_projection_l2']:.6f} mean_abs_delta={diag['csda_mean_abs_delta']:.6f} epoch_seconds={time.time()-epoch_started:.2f}", flush=True)
    load_adapter_state(model, best_state)
    final_bce, final_probs = csda_validation(model, stable_val, profile_val, val_labels)
    if abs(final_bce - best_bce) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: selected CSDA state does not reproduce validation BCE")
    if best_epoch == 0:
        if not csda_is_exact_zero_state(best_state) or float(np.max(np.abs(final_probs - np.asarray(best_baseline["val_probs"], dtype=np.float64)))) > 1e-6:
            raise RuntimeError("INTEGRITY_FAIL: CSDA epoch-0 parity failed")
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()), "baseline_classifier_state": state_to_cpu(best_baseline["classifier_state"]),
        "profile_state": profile_state_payload(profile_state), "adapter_state": state_to_cpu(best_state),
        "val_bce": float(best_bce), "encoder_epoch": int(best_baseline["encoder_epoch"]), "classifier_epoch": int(best_baseline["classifier_epoch"]),
        "csda_epoch": int(best_epoch), "threshold": validation_threshold(task.val_labels, final_probs), "val_probs": final_probs,
        "csda_diagnostics": {
            "csda_profile_std_floor": float(profile_state.std_floor),
            **csda_diagnostics_from_model(model, stable_reference, profile_reference, reference_labels, reference_site_ids),
        },
        "dsls_diagnostics": None, "icdm_diagnostics": None,
    }


def prepare_icdm_branch(
    task: Task, encoder: VarCoNet, best_baseline: dict[str, Any], stable_reference: torch.Tensor,
    stable_train: torch.Tensor, stable_val: torch.Tensor, profile_reference: torch.Tensor,
    profile_train: torch.Tensor, profile_val: torch.Tensor, train_labels: torch.Tensor, val_labels: torch.Tensor,
    profile_state: ProfileState, dsls_branch: dict[str, Any], csda_branch: dict[str, Any], source: dict[str, Any] | None,
    source_phase_rank: int, started: float, latest_path: Path, best_path: Path,
) -> dict[str, Any]:
    model = IndividualConditionedDiseaseMask(best_baseline["classifier_state"], profile_state, roi_num=int(CONFIG["roi_count"]), bottleneck=int(CONFIG["icdm_bottleneck"]), max_log_gate=float(CONFIG["icdm_max_log_gate"])).to(stable_train.device)
    if icdm_parameter_count(model) != int(CONFIG["icdm_trainable_parameters"]):
        raise RuntimeError(f"INTEGRITY_FAIL: ICDM parameter count={icdm_parameter_count(model)} expected=12288")
    zero_state = icdm_state_payload(model)
    if not icdm_is_exact_zero_state(zero_state):
        raise RuntimeError("INTEGRITY_FAIL: ICDM up projection is not exact-zero initialized")
    zero_details = model.forward_details(stable_val, profile_val)
    zero_diag = zero_parity_diagnostics(ICDM_METHOD, zero_details, stable_val, val_labels, best_baseline["classifier_state"], stable_train.device)
    if float((zero_details["roi_prompt"]).abs().max().detach().cpu()) > 0.0 or float((zero_details["edge_factor"] - 1.0).abs().max().detach().cpu()) > 1e-6 or float((zero_details["stable_adapted"] - stable_val).abs().max().detach().cpu()) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: ICDM epoch-0 gate is not identity")
    optimizer = Adam(model.parameters(), lr=float(CONFIG["icdm_lr"]))
    if source is not None and source_phase_rank == 3 and source.get("icdm_progress") is not None:
        progress = source["icdm_progress"]
        if not _allclose_nested(progress["zero_diagnostics"], zero_diag):
            raise RuntimeError("INTEGRITY_FAIL: resumed ICDM zero diagnostic changed")
        load_mask_state(model, progress["current_state"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        optimizer_to_device(optimizer, stable_train.device)
        best_state, best_epoch, best_bce = state_to_cpu(progress["best_state"]), int(progress["best_epoch"]), float(progress["best_val_bce"])
        start_epoch = int(source["epoch"]) + 1
        print(f"ICDM_RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} from_epoch={start_epoch} best_epoch={best_epoch} best_val_bce={best_bce:.6f}", flush=True)
    else:
        best_state, best_epoch, best_bce, start_epoch = state_to_cpu(zero_state), 0, float(zero_diag["baseline_val_bce"]), 1
        progress = {
            "zero_state": state_to_cpu(zero_state), "zero_diagnostics": state_to_cpu(zero_diag), "profile_state": profile_state_payload(profile_state),
            "current_state": state_to_cpu(zero_state), "optimizer_state_dict": state_to_cpu(optimizer.state_dict()), "best_state": state_to_cpu(best_state),
            "best_epoch": best_epoch, "best_val_bce": best_bce,
        }
        payload = base_checkpoint_payload(task, "icdm", 0, encoder, best_baseline, time.time() - started, dsls_branch=dsls_branch, csda_branch=csda_branch, icdm_progress=progress)
        save_checkpoint(latest_path, "latest", payload)
        save_checkpoint(best_path, "best", payload)
    for epoch in range(start_epoch, int(CONFIG["epochs_cls"])):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad()
        details = model.forward_details(stable_train, profile_train)
        loss = binary_bce(details["p_hc"], train_labels)
        loss.backward()
        if not bool(torch.isfinite(loss)) or not finite_gradients(model):
            raise FloatingPointError("non-finite ICDM BCE or gradient")
        optimizer.step()
        val_bce, _ = icdm_validation(model, stable_val, profile_val, val_labels)
        is_best = val_bce < best_bce
        if is_best:
            best_bce, best_epoch, best_state = float(val_bce), int(epoch), icdm_state_payload(model)
        progress = {
            "zero_state": state_to_cpu(zero_state), "zero_diagnostics": state_to_cpu(zero_diag), "profile_state": profile_state_payload(profile_state),
            "current_state": icdm_state_payload(model), "optimizer_state_dict": state_to_cpu(optimizer.state_dict()), "best_state": state_to_cpu(best_state),
            "best_epoch": int(best_epoch), "best_val_bce": float(best_bce),
        }
        payload = base_checkpoint_payload(task, "icdm", epoch, encoder, best_baseline, time.time() - started, dsls_branch=dsls_branch, csda_branch=csda_branch, icdm_progress=progress)
        if is_best:
            save_checkpoint(best_path, "best", payload)
        if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs_cls"]) - 1:
            save_checkpoint(latest_path, "latest", payload)
        if epoch % 10 == 0 or is_best or epoch == int(CONFIG["epochs_cls"]) - 1:
            diag = icdm_diagnostics_from_model(model, stable_reference, profile_reference)
            print(f"[ICDM] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs_cls']-1} train_bce={float(loss.detach().cpu()):.6f} val_bce={val_bce:.6f} best={'Y' if is_best else 'N'} best_epoch={best_epoch} mean_abs_roi_prompt={diag['icdm_mean_abs_roi_prompt']:.6f} mean_edge_factor={diag['icdm_mean_edge_factor']:.6f} std_edge_factor={diag['icdm_std_edge_factor']:.6f} factor_min={diag['icdm_factor_min']:.6f} factor_max={diag['icdm_factor_max']:.6f} mean_abs_fc_change={diag['icdm_mean_abs_fc_change']:.6f} up_l2={diag['icdm_up_l2']:.6f} down_l2={diag['icdm_down_l2']:.6f} epoch_seconds={time.time()-epoch_started:.2f}", flush=True)
    load_mask_state(model, best_state)
    final_bce, final_probs = icdm_validation(model, stable_val, profile_val, val_labels)
    if abs(final_bce - best_bce) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: selected ICDM state does not reproduce validation BCE")
    if best_epoch == 0:
        if not icdm_is_exact_zero_state(best_state) or float(np.max(np.abs(final_probs - np.asarray(best_baseline["val_probs"], dtype=np.float64)))) > 1e-6:
            raise RuntimeError("INTEGRITY_FAIL: ICDM epoch-0 parity failed")
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()), "baseline_classifier_state": state_to_cpu(best_baseline["classifier_state"]),
        "profile_state": profile_state_payload(profile_state), "mask_state": state_to_cpu(best_state),
        "val_bce": float(best_bce), "encoder_epoch": int(best_baseline["encoder_epoch"]), "classifier_epoch": int(best_baseline["classifier_epoch"]),
        "icdm_epoch": int(best_epoch), "threshold": validation_threshold(task.val_labels, final_probs), "val_probs": final_probs,
        "icdm_diagnostics": icdm_diagnostics_from_model(model, stable_reference, profile_reference),
        "dsls_diagnostics": None, "csda_diagnostics": None,
    }


def source_phase_rank(phase: str) -> int:
    return {"ssl": 0, "dsls": 1, "csda": 2, "icdm": 3}.get(str(phase), -1)


def train_shared_task(cohort: Cohort, task: Task, device: torch.device) -> dict[str, Any]:
    """Train one audited encoder trajectory and the three post-hoc branches."""

    started = time.time()
    print(
        f"TASK_START repeat={task.repeat} scope={task.scope} fold={task.fold} seed={task.seed} device={device} "
        f"methods={','.join(METHODS)} shared_encoder=Y reference_unique_n={len(task.reference_raw)} "
        f"train_n={len(task.train_raw)} val_n={len(task.val_raw)}",
        flush=True,
    )
    set_all_seeds(task.seed)
    model_config, best_params = make_model_config(cohort)
    encoder = VarCoNet(model_config, int(cohort.abide1.shape[2])).to(device)
    contrast = DualBranchContrast(loss=InfoNCE(tau=float(best_params["tau"])), mode="L2L").to(device)
    optimizer_ssl = Adam(encoder.parameters(), lr=float(best_params["lr"]))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer_ssl,
        warmup_start_lr=1e-5,
        warmup_epochs=int(CONFIG["warm_up_epochs"]),
        max_epochs=int(CONFIG["epochs"]),
    )
    schedule = paired_batch_schedule(task.train_names, len(task.train_raw), int(CONFIG["epochs"]), task.seed)
    latest_path, best_path = checkpoint_paths(task.repeat)
    source = load_task_checkpoint(task.repeat, task)
    phase = "ssl"
    start_ssl_epoch = 1
    best_baseline: dict[str, Any] | None = None
    ssl_losses: list[float] = []
    if source is not None:
        phase = str(source.get("phase", "ssl"))
        if source_phase_rank(phase) < 0:
            raise RuntimeError(f"unknown checkpoint phase {phase!r}")
        elapsed = float(source.get("elapsed_seconds", 0.0))
        started -= elapsed
        best_baseline = state_to_cpu(source.get("best_baseline"))
        if source.get("rng_state") is not None:
            restore_rng_state(source["rng_state"])
        if phase == "ssl":
            encoder.load_state_dict(source["encoder_state_dict"])
            if source.get("optimizer_state_dict") is None or source.get("scheduler_state_dict") is None:
                raise RuntimeError("SSL resume checkpoint lacks optimizer/scheduler state")
            optimizer_ssl.load_state_dict(source["optimizer_state_dict"])
            optimizer_to_device(optimizer_ssl, device)
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_ssl_epoch = int(source["epoch"]) + 1
        else:
            if best_baseline is None:
                raise RuntimeError("post-SSL resume checkpoint lacks baseline state")
            encoder.load_state_dict(best_baseline["encoder_state"])
            start_ssl_epoch = int(CONFIG["epochs"]) + 1
        print(
            f"RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} phase={phase} "
            f"from_epoch={int(source.get('epoch', 0)) + 1} elapsed_seconds={elapsed:.1f}",
            flush=True,
        )

    train_labels = torch.as_tensor(task.train_labels, dtype=torch.long, device=device)
    val_labels = torch.as_tensor(task.val_labels, dtype=torch.long, device=device)
    for epoch in range(start_ssl_epoch, int(CONFIG["epochs"]) + 1):
        epoch_started = time.time()
        ssl_loss = train_ssl_epoch(
            encoder, contrast, optimizer_ssl, cohort, task, schedule[epoch - 1],
            int(model_config["max_length"]), device,
        )
        scheduler.step()
        ssl_losses.append(ssl_loss)
        stable_train_epoch = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
        stable_val_epoch = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
        trajectory_rng = capture_rng_state()
        baseline_fit = fit_baseline_classifier(
            stable_train_epoch, train_labels, stable_val_epoch, val_labels, device,
            task.seed + 300000 + epoch * 2,
        )
        restore_rng_state(trajectory_rng)
        candidate = make_baseline_branch(encoder, epoch, baseline_fit)
        is_best = best_baseline is None or float(candidate["val_bce"]) < float(best_baseline["val_bce"])
        if is_best:
            best_baseline = candidate
        checkpoint = base_checkpoint_payload(
            task, "ssl", epoch, encoder, best_baseline, time.time() - started,
            optimizer_state_dict=state_to_cpu(optimizer_ssl.state_dict()),
            scheduler_state_dict=state_to_cpu(scheduler.state_dict()),
        )
        if is_best:
            save_checkpoint(best_path, "best", checkpoint)
        if epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs"]):
            save_checkpoint(latest_path, "latest", checkpoint)
        allocated, reserved = gpu_memory_gb(device)
        print(
            f"[SSL] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs']} "
            f"ssl_loss={ssl_loss:.6f} lr={optimizer_ssl.param_groups[0]['lr']:.9f} "
            f"base_val_bce={baseline_fit['val_bce']:.6f} base_best={'Y' if is_best else 'N'} "
            f"base_cls_epoch={baseline_fit['classifier_epoch']} epoch_seconds={time.time()-epoch_started:.1f} "
            f"task_elapsed_minutes={(time.time()-started)/60.0:.1f} gpu_alloc_gb={allocated:.3f} gpu_reserved_gb={reserved:.3f}",
            flush=True,
        )
        del stable_train_epoch, stable_val_epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if best_baseline is None:
        raise RuntimeError("no validation-selected baseline branch")

    # From this point every branch sees exactly the same selected encoder,
    # stable features, and reference-only ROI-profile normalization.
    encoder.load_state_dict(best_baseline["encoder_state"])
    encoder.eval()
    stable_reference = extract_stable_features(encoder, cohort.abide1, task.reference_raw, device)
    stable_train = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
    stable_val = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
    reference_labels = torch.as_tensor(task.reference_labels, dtype=torch.long, device=device)
    expected_ref_labels = torch.as_tensor(cohort.abide1_labels[task.reference_raw], dtype=torch.long, device=device)
    if not torch.equal(reference_labels, expected_ref_labels):
        raise RuntimeError("reference labels are not aligned with unique outer-training raw indices")
    reference_site_ids = torch.as_tensor(task.reference_site_ids, dtype=torch.long, device=device)
    profile_module = ROIConnectivityProfile(int(CONFIG["roi_count"])).to(device)
    profile_reference = profile_features(profile_module, stable_reference)
    profile_train = profile_features(profile_module, stable_train)
    profile_val = profile_features(profile_module, stable_val)
    profile_state = fit_profile_state(profile_reference)

    fitted_dsls = fit_dsls_state(
        stable_reference, reference_labels, seed=task.seed + 700000,
        n_components=int(CONFIG["dsls_components"]),
    )
    rank = source_phase_rank(phase)
    if rank >= 2 and source is not None and source.get("dsls_branch") is not None:
        dsls_branch = state_to_cpu(source["dsls_branch"])
        if int(dsls_branch["encoder_epoch"]) != int(best_baseline["encoder_epoch"]):
            raise RuntimeError("INTEGRITY_FAIL: resumed DSLS encoder selection differs")
    else:
        dsls_branch = prepare_dsls_branch(
            task, encoder, best_baseline, stable_reference, stable_train, stable_val,
            reference_labels, train_labels, val_labels, fitted_dsls,
            source if rank == 1 else None, rank, started, latest_path, best_path,
        )

    if rank >= 3 and source is not None and source.get("csda_branch") is not None:
        csda_branch = state_to_cpu(source["csda_branch"])
        if int(csda_branch["encoder_epoch"]) != int(best_baseline["encoder_epoch"]):
            raise RuntimeError("INTEGRITY_FAIL: resumed CSDA encoder selection differs")
    else:
        csda_branch = prepare_csda_branch(
            task, encoder, best_baseline, stable_reference, stable_train, stable_val,
            profile_reference, profile_train, profile_val, reference_labels, reference_site_ids,
            train_labels, val_labels, profile_state, dsls_branch,
            source if rank == 2 else None, rank, started, latest_path, best_path,
        )

    if rank >= 4 and source is not None and source.get("icdm_branch") is not None:
        icdm_branch = state_to_cpu(source["icdm_branch"])
    else:
        icdm_branch = prepare_icdm_branch(
            task, encoder, best_baseline, stable_reference, stable_train, stable_val,
            profile_reference, profile_train, profile_val, train_labels, val_labels,
            profile_state, dsls_branch, csda_branch,
            source if rank == 3 else None, rank, started, latest_path, best_path,
        )
    result = {
        "task": task,
        "baseline": best_baseline,
        "dsls": dsls_branch,
        "csda": csda_branch,
        "icdm": icdm_branch,
        "mean_ssl_loss": float(np.mean(ssl_losses)) if ssl_losses else float("nan"),
        "elapsed_seconds": float(time.time() - started),
    }
    for branch in (best_baseline, dsls_branch, csda_branch, icdm_branch):
        branch["task_seconds"] = float(result["elapsed_seconds"])
    assert_shared_selection(result)
    print(
        f"TASK_TRAINING_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"base_encoder_epoch={best_baseline['encoder_epoch']} base_classifier_epoch={best_baseline['classifier_epoch']} "
        f"base_val_bce={best_baseline['val_bce']:.6f} dsls_epoch={dsls_branch['dsls_epoch']} "
        f"dsls_val_bce={dsls_branch['val_bce']:.6f} csda_epoch={csda_branch['csda_epoch']} "
        f"csda_val_bce={csda_branch['val_bce']:.6f} icdm_epoch={icdm_branch['icdm_epoch']} "
        f"icdm_val_bce={icdm_branch['val_bce']:.6f} elapsed_minutes={result['elapsed_seconds']/60.0:.1f}",
        flush=True,
    )
    del encoder, contrast, optimizer_ssl, scheduler, profile_module
    del stable_reference, stable_train, stable_val, profile_reference, profile_train, profile_val
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def dsls_model_from_branch(branch: dict[str, Any], device: torch.device) -> DSLSResidual:
    state = dsls_state_from_payload(branch["dsls_state"])
    model = DSLSResidual(branch["baseline_classifier_state"], state).to(device)
    load_dsls_state(model, branch["residual_state"])
    model.eval()
    return model


def csda_model_from_branch(branch: dict[str, Any], device: torch.device) -> CrossSiteDiseaseAdapter:
    state = profile_state_from_payload(branch["profile_state"])
    model = CrossSiteDiseaseAdapter(branch["baseline_classifier_state"], state).to(device)
    load_adapter_state(model, branch["adapter_state"])
    model.eval()
    return model


def icdm_model_from_branch(branch: dict[str, Any], device: torch.device) -> IndividualConditionedDiseaseMask:
    state = profile_state_from_payload(branch["profile_state"])
    model = IndividualConditionedDiseaseMask(
        branch["baseline_classifier_state"], state,
        roi_num=int(CONFIG["roi_count"]), bottleneck=int(CONFIG["icdm_bottleneck"]),
        max_log_gate=float(CONFIG["icdm_max_log_gate"]),
    ).to(device)
    load_mask_state(model, branch["mask_state"])
    model.eval()
    return model


def evaluate_scope_all_methods(
    cohort: Cohort, task_result: dict[str, Any], data: np.ndarray,
    raw_indices: np.ndarray, labels: np.ndarray, device: torch.device, scope: str,
) -> dict[str, tuple[dict[str, float], np.ndarray]]:
    """Extract one frozen test trajectory and apply all four methods."""

    assert_shared_selection(task_result)
    baseline = task_result[BASELINE_METHOD]
    encoder = create_encoder(cohort, device)
    encoder.load_state_dict(baseline["encoder_state"])
    stable = extract_stable_features(encoder, data, raw_indices, device)
    profile_module = ROIConnectivityProfile(int(CONFIG["roi_count"])).to(device)
    profile = profile_features(profile_module, stable)
    probabilities: dict[str, np.ndarray] = {
        BASELINE_METHOD: predict_baseline(stable, baseline["classifier_state"], device)
    }
    dsls = dsls_model_from_branch(task_result[DSLS_METHOD], device)
    csda = csda_model_from_branch(task_result[CSDA_METHOD], device)
    icdm = icdm_model_from_branch(task_result[ICDM_METHOD], device)
    with torch.no_grad():
        probabilities[DSLS_METHOD] = dsls(stable)[:, 1].detach().cpu().numpy().astype(np.float64)
        probabilities[CSDA_METHOD] = csda(stable, profile)[:, 1].detach().cpu().numpy().astype(np.float64)
        probabilities[ICDM_METHOD] = icdm(stable, profile)[:, 1].detach().cpu().numpy().astype(np.float64)
    for method, key in ((DSLS_METHOD, "dsls_epoch"), (CSDA_METHOD, "csda_epoch"), (ICDM_METHOD, "icdm_epoch")):
        if int(task_result[method][key]) == 0:
            difference = float(np.max(np.abs(probabilities[method] - probabilities[BASELINE_METHOD])))
            if difference > 1e-6:
                raise RuntimeError(f"INTEGRITY_FAIL: {method} exact-zero test parity failed on {scope}: {difference}")
    branches = {method: task_result[method] for method in METHODS}
    answer = {
        method: (
            classification_metrics(labels, probabilities[method], float(branches[method]["threshold"])),
            probabilities[method],
        )
        for method in METHODS
    }
    del encoder, stable, profile_module, profile, dsls, csda, icdm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return answer


def diagnostic_fields(method: str, branch: dict[str, Any]) -> dict[str, float]:
    dsls_diag = branch.get("dsls_diagnostics") if method == DSLS_METHOD else None
    csda_diag = branch.get("csda_diagnostics") if method == CSDA_METHOD else None
    icdm_diag = branch.get("icdm_diagnostics") if method == ICDM_METHOD else None
    for diagnostic, columns, name in (
        (dsls_diag, DSLS_DIAGNOSTIC_COLUMNS, "DSLS"),
        (csda_diag, CSDA_DIAGNOSTIC_COLUMNS, "CSDA"),
        (icdm_diag, ICDM_DIAGNOSTIC_COLUMNS, "ICDM"),
    ):
        if diagnostic is not None:
            missing = [key for key in columns if key not in diagnostic]
            if missing:
                raise RuntimeError(f"{name} diagnostics missing fields: {missing}")
    return {
        **{key: float(dsls_diag[key]) if dsls_diag is not None else float("nan") for key in DSLS_DIAGNOSTIC_COLUMNS},
        **{key: float(csda_diag[key]) if csda_diag is not None else float("nan") for key in CSDA_DIAGNOSTIC_COLUMNS},
        **{key: float(icdm_diag[key]) if icdm_diag is not None else float("nan") for key in ICDM_DIAGNOSTIC_COLUMNS},
    }


def metric_row(
    method: str, scope: str, repeat: int, fold: str, metrics: dict[str, float],
    branch: dict[str, Any], n_test: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": str(method), "scope": str(scope), "repeat": int(repeat), "fold": str(fold),
        "auc": float(metrics["auc"]), "bce": float(metrics["bce"]),
        "f1": float(metrics["f1"]), "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "selected_encoder_epoch": float(branch["encoder_epoch"]),
        "selected_classifier_epoch": float(branch.get("classifier_epoch", float("nan"))),
        "selected_dsls_epoch": float(branch.get("dsls_epoch", float("nan"))),
        "selected_csda_epoch": float(branch.get("csda_epoch", float("nan"))),
        "selected_icdm_epoch": float(branch.get("icdm_epoch", float("nan"))),
        "threshold_hc": float(branch["threshold"]), "task_seconds": float(branch.get("task_seconds", float("nan"))),
        "n_test": int(n_test), "status": "PASS",
    }
    row.update(diagnostic_fields(method, branch))
    return row


def worker_paths(repeat: int) -> dict[str, Path]:
    prefix = RESULTS_DIR / f".worker_r{int(repeat)}"
    return {
        "csv": prefix.with_suffix(".csv"),
        "predictions": prefix.with_name(prefix.name + "_predictions.npz"),
        "external": prefix.with_name(prefix.name + "_external.pt"),
        "done": prefix.with_name(prefix.name + ".done.json"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"),
        "error": prefix.with_name(prefix.name + ".error.json"),
    }


def cast_metric_row(row: dict[str, str]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key in METRIC_COLUMNS:
        value = row.get(key, "")
        if key in FLOAT_COLUMNS:
            converted[key] = float(value) if value not in {"", None} else float("nan")
        elif key in INT_COLUMNS:
            converted[key] = int(float(value)) if value not in {"", None} else 0
        else:
            converted[key] = str(value)
    return converted


def read_metric_shard(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != METRIC_COLUMNS:
            raise RuntimeError(f"unexpected worker metric schema: {path}")
        return [cast_metric_row(row) for row in reader]


def write_metric_shard_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_metric_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())


def task_row_complete(rows: list[dict[str, Any]], scope: str, repeat: int, fold: str) -> bool:
    return {
        row["method"] for row in rows
        if row["scope"] == scope and int(row["repeat"]) == int(repeat) and row["fold"] == str(fold)
    } == set(METHODS)


def clear_metric_rows(rows: list[dict[str, Any]], scope: str, repeat: int, fold: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if not (row["scope"] == scope and int(row["repeat"]) == int(repeat) and row["fold"] == str(fold))
    ]


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTION_FIELDS):
            raise RuntimeError(f"unexpected prediction shard schema: {path}")
        count = len(archive["scope"])
        if any(len(archive[field]) != count for field in PREDICTION_FIELDS):
            raise RuntimeError(f"inconsistent prediction shard lengths: {path}")
        return [{
            "scope": str(archive["scope"][i]), "method": str(archive["method"][i]),
            "repeat": int(archive["repeat"][i]), "fold": str(archive["fold"][i]),
            "subject_index": int(archive["subject_index"][i]), "raw_index": int(archive["raw_index"][i]),
            "y": int(archive["y"][i]), "p_hc": float(archive["p_hc"][i]),
            "threshold": float(archive["threshold"][i]),
        } for i in range(count)]


def write_prediction_records_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "scope": np.asarray([item["scope"] for item in records], dtype="<U24"),
        "method": np.asarray([item["method"] for item in records], dtype="<U48"),
        "repeat": np.asarray([item["repeat"] for item in records], dtype=np.int16),
        "fold": np.asarray([item["fold"] for item in records], dtype="<U16"),
        "subject_index": np.asarray([item["subject_index"] for item in records], dtype=np.int32),
        "raw_index": np.asarray([item["raw_index"] for item in records], dtype=np.int32),
        "y": np.asarray([item["y"] for item in records], dtype=np.int8),
        "p_hc": np.asarray([item["p_hc"] for item in records], dtype=np.float64),
        "threshold": np.asarray([item["threshold"] for item in records], dtype=np.float64),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clear_prediction_group(
    records: list[dict[str, Any]], scope: str, repeat: int, fold: str,
    method: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record for record in records
        if not (
            record["scope"] == scope and int(record["repeat"]) == int(repeat)
            and record["fold"] == str(fold)
            and (method is None or record["method"] == method)
        )
    ]


def add_prediction_group(
    records: list[dict[str, Any]], scope: str, method: str, repeat: int, fold: str,
    subject_indices: np.ndarray, raw_indices: np.ndarray, labels: np.ndarray,
    probabilities: np.ndarray, threshold: float | np.ndarray,
) -> list[dict[str, Any]]:
    records = clear_prediction_group(records, scope, repeat, fold, method)
    subject_indices, raw_indices, labels, probabilities = (np.asarray(value) for value in (subject_indices, raw_indices, labels, probabilities))
    thresholds = np.full(len(labels), float(threshold), dtype=np.float64) if np.isscalar(threshold) else np.asarray(threshold, dtype=np.float64)
    if not all(len(value) == len(labels) for value in (subject_indices, raw_indices, probabilities, thresholds)):
        raise RuntimeError("prediction group lengths disagree")
    records.extend({
        "scope": str(scope), "method": str(method), "repeat": int(repeat), "fold": str(fold),
        "subject_index": int(subject_indices[i]), "raw_index": int(raw_indices[i]), "y": int(labels[i]),
        "p_hc": float(probabilities[i]), "threshold": float(thresholds[i]),
    } for i in range(len(labels)))
    return records


def subject_indices_for_task(cohort: Cohort, task: Task) -> np.ndarray:
    if task.scope == "internal_cv":
        source = cohort.internal_raw
    elif task.scope == "external":
        source = cohort.caltech_raw
    else:
        raise RuntimeError(f"unknown task scope {task.scope}")
    lookup = {int(raw): index for index, raw in enumerate(source)}
    return np.asarray([lookup[int(raw)] for raw in task.test_raw], dtype=np.int64)


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def build_internal_oof_row(
    cohort: Cohort, records: list[dict[str, Any]], fold_rows: list[dict[str, Any]],
    method: str, repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = sorted(
        [record for record in records if record["scope"] == "internal_fold" and record["method"] == method and int(record["repeat"]) == int(repeat)],
        key=lambda record: int(record["subject_index"]),
    )
    if len(selected) != len(cohort.internal_raw):
        raise RuntimeError(f"repeat={repeat} method={method} internal OOF count={len(selected)} expected=882")
    if not np.array_equal(np.asarray([record["subject_index"] for record in selected]), np.arange(len(cohort.internal_raw))):
        raise RuntimeError("internal OOF IDs are not an exact partition")
    labels = np.asarray([record["y"] for record in selected], dtype=np.int64)
    probabilities = np.asarray([record["p_hc"] for record in selected], dtype=np.float64)
    thresholds = np.asarray([record["threshold"] for record in selected], dtype=np.float64)
    source_rows = [row for row in fold_rows if row["scope"] == "internal_fold" and row["method"] == method and int(row["repeat"]) == int(repeat)]
    if len(source_rows) != 10:
        raise RuntimeError(f"repeat={repeat} method={method} internal fold rows={len(source_rows)} expected=10")
    branch: dict[str, Any] = {
        "encoder_epoch": mean_or_nan([row["selected_encoder_epoch"] for row in source_rows]),
        "classifier_epoch": mean_or_nan([row["selected_classifier_epoch"] for row in source_rows]),
        "threshold": float("nan"), "task_seconds": mean_or_nan([row["task_seconds"] for row in source_rows]),
        "dsls_diagnostics": None, "csda_diagnostics": None, "icdm_diagnostics": None,
    }
    for treatment, epoch_key, diag_key, columns in (
        (DSLS_METHOD, "selected_dsls_epoch", "dsls_diagnostics", DSLS_DIAGNOSTIC_COLUMNS),
        (CSDA_METHOD, "selected_csda_epoch", "csda_diagnostics", CSDA_DIAGNOSTIC_COLUMNS),
        (ICDM_METHOD, "selected_icdm_epoch", "icdm_diagnostics", ICDM_DIAGNOSTIC_COLUMNS),
    ):
        if method == treatment:
            branch[epoch_key.removeprefix("selected_")] = mean_or_nan([row[epoch_key] for row in source_rows])
            branch[diag_key] = {key: mean_or_nan([row[key] for row in source_rows]) for key in columns}
    oof_metrics = metrics_with_thresholds(labels, probabilities, thresholds)
    return metric_row(method, "internal_oof", repeat, "all", oof_metrics, branch, len(labels)), [
        {**record, "scope": "internal_oof", "fold": "all"} for record in selected
    ]


def evaluate_internal_task(
    cohort: Cohort, task_result: dict[str, Any], device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task: Task = task_result["task"]
    evaluations = evaluate_scope_all_methods(
        cohort, task_result, cohort.abide1, task.test_raw, task.test_labels, device, "internal_fold"
    )
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    indices = subject_indices_for_task(cohort, task)
    for method in METHODS:
        metrics, probabilities = evaluations[method]
        branch = task_result[method]
        rows.append(metric_row(method, "internal_fold", task.repeat, task.fold, metrics, branch, len(task.test_raw)))
        records = add_prediction_group(
            records, "internal_fold", method, task.repeat, task.fold, indices,
            task.test_raw, task.test_labels, probabilities, float(branch["threshold"]),
        )
        print(
            f"TASK_DONE repeat={task.repeat} scope=internal_fold fold={task.fold} method={method} "
            f"selected_encoder_epoch={branch['encoder_epoch']} test_auc={metrics['auc']:.6f} "
            f"test_bce={metrics['bce']:.6f} elapsed_minutes={task_result['elapsed_seconds']/60.0:.1f}",
            flush=True,
        )
    return rows, records


def evaluate_external_task(
    cohort: Cohort, task_result: dict[str, Any], device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task: Task = task_result["task"]
    caltech_indices = subject_indices_for_task(cohort, task)
    abide2_indices = np.arange(len(cohort.abide2), dtype=np.int64)
    if len(caltech_indices) != 37:
        raise RuntimeError(f"Caltech external cohort must contain 37 subjects, got {len(caltech_indices)}")
    if len(abide2_indices) != 727 or not np.array_equal(abide2_indices, np.arange(727)):
        raise RuntimeError("ABIDE-II must remain the complete original 727-subject cohort")
    specs = (
        ("caltech", cohort.abide1, task.test_raw, task.test_labels, caltech_indices),
        ("abide2", cohort.abide2, abide2_indices, cohort.abide2_labels, abide2_indices.copy()),
    )
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for scope, data, raw_indices, labels, subject_indices in specs:
        evaluations = evaluate_scope_all_methods(cohort, task_result, data, raw_indices, labels, device, scope)
        for method in METHODS:
            metrics, probabilities = evaluations[method]
            branch = task_result[method]
            rows.append(metric_row(method, scope, task.repeat, "external", metrics, branch, len(raw_indices)))
            records = add_prediction_group(
                records, scope, method, task.repeat, "external", subject_indices,
                raw_indices, labels, probabilities, float(branch["threshold"]),
            )
            print(
                f"TASK_DONE repeat={task.repeat} scope={scope} fold=external method={method} "
                f"selected_encoder_epoch={branch['encoder_epoch']} test_auc={metrics['auc']:.6f} "
                f"test_bce={metrics['bce']:.6f} elapsed_minutes={task_result['elapsed_seconds']/60.0:.1f}",
                flush=True,
            )
        expected = 37 if scope == "caltech" else 727
        paired: dict[str, list[tuple[int, int, int]]] = {}
        for method in METHODS:
            values = sorted(
                (record["subject_index"], record["raw_index"], record["y"])
                for record in records if record["scope"] == scope and record["method"] == method
            )
            if len(values) != expected:
                raise RuntimeError(f"{scope}/{method} count={len(values)} expected={expected}")
            paired[method] = values
        if not (paired[BASELINE_METHOD] == paired[DSLS_METHOD] == paired[CSDA_METHOD] == paired[ICDM_METHOD]):
            raise RuntimeError(f"INTEGRITY_FAIL: {scope} method subject IDs/labels differ")
    expected_abide2 = [(index, index, int(cohort.abide2_labels[index])) for index in range(727)]
    for method in METHODS:
        actual = sorted(
            (record["subject_index"], record["raw_index"], record["y"])
            for record in records if record["scope"] == "abide2" and record["method"] == method
        )
        if actual != expected_abide2:
            raise RuntimeError(f"INTEGRITY_FAIL: ABIDE-II/{method} IDs are not exactly 0..726")
    external_models = {
        "format_version": 3, "repeat": int(task.repeat), "task_identity": task_identity(task),
        "model_config": make_model_config(cohort)[0], "baseline": state_to_cpu(task_result[BASELINE_METHOD]),
        DSLS_METHOD: state_to_cpu(task_result[DSLS_METHOD]),
        CSDA_METHOD: state_to_cpu(task_result[CSDA_METHOD]),
        ICDM_METHOD: state_to_cpu(task_result[ICDM_METHOD]),
    }
    return rows, records, external_models


def external_rows_complete(rows: list[dict[str, Any]], repeat: int) -> bool:
    return all(task_row_complete(rows, scope, repeat, "external") for scope in ("caltech", "abide2"))


def remove_external_rows(rows: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    for scope in ("caltech", "abide2"):
        rows = clear_metric_rows(rows, scope, repeat, "external")
    return rows


def remove_external_predictions(records: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    for scope in ("caltech", "abide2"):
        records = clear_prediction_group(records, scope, repeat, "external")
    return records


def expected_repeat_rows(rows: list[dict[str, Any]], repeat: int) -> bool:
    selected = [row for row in rows if int(row["repeat"]) == int(repeat)]
    if len(selected) != 52:
        return False
    expected: set[tuple[str, str, str]] = set()
    for fold in range(10):
        for method in METHODS:
            expected.add((method, "internal_fold", str(fold)))
    for method in METHODS:
        expected.update({
            (method, "internal_oof", "all"),
            (method, "caltech", "external"),
            (method, "abide2", "external"),
        })
    return {(row["method"], row["scope"], row["fold"]) for row in selected} == expected


def update_runtime_task(repeat: int, key: str, seconds: float, device: torch.device) -> None:
    path = worker_paths(repeat)["runtime"]
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.setdefault("repeat", int(repeat))
    payload.setdefault("device", str(device))
    payload.setdefault("completed_tasks", {})
    payload["completed_tasks"][str(key)] = float(seconds)
    payload["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(path, payload)


def run_repeat(cohort: Cohort, repeat: int, device: torch.device) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = worker_paths(repeat)
    rows = read_metric_shard(paths["csv"])
    records = read_prediction_records(paths["predictions"])
    if expected_repeat_rows(rows, repeat) and paths["external"].exists():
        print(f"REPEAT_DONE repeat={repeat} status=PASS already_complete=Y", flush=True)
        return
    tasks = make_tasks(cohort, repeat)
    print(f"REPEAT_START repeat={repeat} device={device} tasks=11 methods=4", flush=True)
    for task in tasks[:-1]:
        if task_row_complete(rows, "internal_fold", repeat, task.fold):
            print(f"TASK_SKIP repeat={repeat} scope=internal_fold fold={task.fold} reason=PASS_SHARD", flush=True)
            clean_task_checkpoints(repeat, task)
            continue
        rows = clear_metric_rows(rows, "internal_fold", repeat, task.fold)
        records = clear_prediction_group(records, "internal_fold", repeat, task.fold)
        write_metric_shard_atomic(paths["csv"], rows)
        write_prediction_records_atomic(paths["predictions"], records)
        task_result = train_shared_task(cohort, task, device)
        task_rows, task_records = evaluate_internal_task(cohort, task_result, device)
        for method in METHODS:
            group = [record for record in task_records if record["method"] == method]
            records = add_prediction_group(
                records, "internal_fold", method, repeat, task.fold,
                np.asarray([record["subject_index"] for record in group]),
                np.asarray([record["raw_index"] for record in group]),
                np.asarray([record["y"] for record in group]),
                np.asarray([record["p_hc"] for record in group]),
                np.asarray([record["threshold"] for record in group]),
            )
        write_prediction_records_atomic(paths["predictions"], records)
        append_metric_rows(paths["csv"], task_rows)
        rows.extend(task_rows)
        update_runtime_task(repeat, f"internal_{task.fold}", task_result["elapsed_seconds"], device)
        clean_task_checkpoints(repeat, task)
    for method in METHODS:
        if any(
            row["scope"] == "internal_oof" and row["method"] == method and int(row["repeat"]) == int(repeat)
            for row in rows
        ):
            continue
        oof_row, oof_records = build_internal_oof_row(cohort, records, rows, method, repeat)
        records = clear_prediction_group(records, "internal_oof", repeat, "all", method)
        records.extend(oof_records)
        write_prediction_records_atomic(paths["predictions"], records)
        append_metric_rows(paths["csv"], [oof_row])
        rows.append(oof_row)
    external_task = tasks[-1]
    if not (external_rows_complete(rows, repeat) and paths["external"].exists()):
        rows = remove_external_rows(rows, repeat)
        records = remove_external_predictions(records, repeat)
        write_metric_shard_atomic(paths["csv"], rows)
        write_prediction_records_atomic(paths["predictions"], records)
        task_result = train_shared_task(cohort, external_task, device)
        external_rows, external_records, external_models = evaluate_external_task(cohort, task_result, device)
        for scope in ("caltech", "abide2"):
            for method in METHODS:
                group = [record for record in external_records if record["scope"] == scope and record["method"] == method]
                records = add_prediction_group(
                    records, scope, method, repeat, "external",
                    np.asarray([record["subject_index"] for record in group]),
                    np.asarray([record["raw_index"] for record in group]),
                    np.asarray([record["y"] for record in group]),
                    np.asarray([record["p_hc"] for record in group]),
                    np.asarray([record["threshold"] for record in group]),
                )
        atomic_torch_save(paths["external"], external_models)
        write_prediction_records_atomic(paths["predictions"], records)
        append_metric_rows(paths["csv"], external_rows)
        rows.extend(external_rows)
        update_runtime_task(repeat, "external", task_result["elapsed_seconds"], device)
        clean_task_checkpoints(repeat, external_task)
    if not expected_repeat_rows(rows, repeat):
        raise RuntimeError(f"repeat={repeat} incomplete: expected 52 final metric rows")
    write_json_atomic(paths["done"], {
        "repeat": int(repeat), "status": "PASS", "metrics_rows": 52,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"REPEAT_DONE repeat={repeat} status=PASS rows=52", flush=True)


def write_final_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    records.sort(key=lambda item: (item["scope"], item["method"], int(item["repeat"]), item["fold"], int(item["subject_index"])))
    write_prediction_records_atomic(path, records)


def validate_prediction_records(records: list[dict[str, Any]], cohort: Cohort) -> None:
    expected_count = {"internal_oof": 882, "caltech": 37, "abide2": 727}
    expected_ids = {
        "internal_oof": [(index, int(cohort.internal_raw[index])) for index in range(882)],
        "caltech": [(index, int(cohort.caltech_raw[index])) for index in range(37)],
        "abide2": [(index, index) for index in range(727)],
    }
    for scope, count in expected_count.items():
        for repeat in range(10):
            paired: dict[str, list[tuple[int, int, int]]] = {}
            for method in METHODS:
                subset = [
                    record for record in records
                    if record["scope"] == scope and record["method"] == method and int(record["repeat"]) == repeat
                ]
                if len(subset) != count:
                    raise RuntimeError(f"{scope}/{method}/repeat={repeat} has {len(subset)} predictions, expected {count}")
                identifiers = sorted((record["subject_index"], record["raw_index"]) for record in subset)
                if identifiers != expected_ids[scope]:
                    raise RuntimeError(f"INTEGRITY_FAIL: {scope}/{method}/repeat={repeat} IDs are not the audited cohort")
                paired[method] = sorted((record["subject_index"], record["raw_index"], record["y"]) for record in subset)
            if not (paired[BASELINE_METHOD] == paired[DSLS_METHOD] == paired[CSDA_METHOD] == paired[ICDM_METHOD]):
                raise RuntimeError(f"INTEGRITY_FAIL: {scope}/repeat={repeat} paired IDs or labels differ across methods")
    if len(cohort.internal_raw) != 882 or len(cohort.caltech_raw) != 37 or len(cohort.abide2) != 727:
        raise RuntimeError("INTEGRITY_FAIL: source cohort counts changed")


def _scope_aggregate(rows: list[dict[str, Any]], method: str, scope: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method and row["scope"] == scope]
    if len(selected) != 10:
        raise RuntimeError(f"aggregate {method}/{scope} has {len(selected)} rows, expected 10")
    output: dict[str, Any] = {"n": len(selected)}
    for metric in ("auc", "bce", "f1", "balanced_accuracy"):
        values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return output


def paired_delta(rows: list[dict[str, Any]], treatment: str, baseline: str, scope: str) -> dict[str, list[float]]:
    left = {int(row["repeat"]): row for row in rows if row["method"] == treatment and row["scope"] == scope}
    right = {int(row["repeat"]): row for row in rows if row["method"] == baseline and row["scope"] == scope}
    if set(left) != set(range(10)) or set(right) != set(range(10)):
        raise RuntimeError(f"cannot form paired deltas for {treatment} vs {baseline} / {scope}")
    return {
        metric: [float(left[repeat][metric] - right[repeat][metric]) for repeat in range(10)]
        for metric in ("auc", "bce", "f1", "balanced_accuracy")
    }


def decision_from_deltas(deltas: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    internal_auc = deltas["internal_oof"]["auc"]
    internal_bce = deltas["internal_oof"]["bce"]
    abide_auc = deltas["abide2"]["auc"]
    abide_bce = deltas["abide2"]["bce"]
    internal_mean = float(np.mean(internal_auc))
    internal_bce_mean = float(np.mean(internal_bce))
    abide_mean = float(np.mean(abide_auc))
    abide_bce_mean = float(np.mean(abide_bce))
    positive = int(sum(delta > 0.0 for delta in internal_auc))
    if (
        internal_mean >= 0.005 and positive >= 7 and abide_mean >= 0.003
        and internal_bce_mean <= 0.005 and abide_bce_mean <= 0.005
    ):
        decision = "STRONG_POSITIVE"
    elif (
        (internal_mean >= 0.003 and positive >= 7 and abide_mean >= 0.0)
        or (abide_mean >= 0.005 and internal_mean >= 0.0)
    ):
        decision = "PROMISING"
    else:
        decision = "FAIL"
    reasons: list[str] = []
    if internal_mean <= 0.0:
        reasons.append("internal_mean_delta_auc_nonpositive")
    if abide_mean < 0.0:
        reasons.append("abide2_mean_delta_auc_negative")
    if internal_bce_mean > 0.005 or abide_bce_mean > 0.005:
        reasons.append("bce_regression_above_0.005")
    return {
        "decision": decision,
        "internal_mean_delta_auc": internal_mean,
        "internal_positive_repeats": positive,
        "internal_mean_delta_bce": internal_bce_mean,
        "abide2_mean_delta_auc": abide_mean,
        "abide2_mean_delta_bce": abide_bce_mean,
        "failure_indicators": reasons,
    }


def diagnostic_summary(rows: list[dict[str, Any]], method: str, columns: tuple[str, ...]) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method and row["scope"] in {"internal_fold", "caltech"}]
    if len(selected) != 110:
        raise RuntimeError(f"{method} diagnostic rows={len(selected)}, expected 110")
    output: dict[str, Any] = {"task_count": len(selected)}
    for key in columns:
        values = [float(row[key]) for row in selected if np.isfinite(float(row[key]))]
        output[f"{key}_mean"] = mean_or_nan(values)
        output[f"{key}_min"] = float(min(values)) if values else float("nan")
        output[f"{key}_max"] = float(max(values)) if values else float("nan")
    return output


def _route_selection(decisions: dict[str, dict[str, Any]], deltas: dict[str, dict[str, dict[str, list[float]]]]) -> dict[str, Any]:
    candidates = [method for method, decision in decisions.items() if decision["decision"] in {"STRONG_POSITIVE", "PROMISING"}]
    if not candidates:
        return {"overall_decision": "ALL_FAIL", "selected_method": None, "candidate_methods": []}
    if len(candidates) == 1:
        return {"overall_decision": "SELECT_NEXT", "selected_method": candidates[0], "candidate_methods": candidates}
    ranked = sorted(
        candidates,
        key=lambda method: (
            float(decisions[method]["abide2_mean_delta_auc"]),
            float(decisions[method]["internal_mean_delta_auc"]),
            int(decisions[method]["internal_positive_repeats"]),
            -float(decisions[method]["internal_mean_delta_bce"]),
        ),
        reverse=True,
    )
    return {"overall_decision": "SELECT_NEXT", "selected_method": ranked[0], "candidate_methods": candidates, "ranking": ranked}


def aggregate_rows(rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], source_hashes: dict[str, str]) -> dict[str, Any]:
    scopes = ("internal_oof", "caltech", "abide2")
    aggregates = {f"{method}::{scope}": _scope_aggregate(rows, method, scope) for method in METHODS for scope in scopes}
    deltas_vs_baseline = {
        method: {scope: paired_delta(rows, method, BASELINE_METHOD, scope) for scope in scopes}
        for method in METHODS[1:]
    }
    decisions = {method: decision_from_deltas(deltas_vs_baseline[method]) for method in METHODS[1:]}
    baseline_internal = float(aggregates[f"{BASELINE_METHOD}::internal_oof"]["auc_mean"])
    baseline_abide2 = float(aggregates[f"{BASELINE_METHOD}::abide2"]["auc_mean"])
    pipeline_drift = (
        not np.isfinite(baseline_internal) or not np.isfinite(baseline_abide2)
        or abs(baseline_internal - 0.723) > 0.015 or abs(baseline_abide2 - 0.739) > 0.015
    )
    if pipeline_drift:
        for decision in decisions.values():
            decision["decision"] = "PIPELINE_DRIFT"
    diagnostic_summaries = {
        DSLS_METHOD: diagnostic_summary(rows, DSLS_METHOD, DSLS_DIAGNOSTIC_COLUMNS),
        CSDA_METHOD: diagnostic_summary(rows, CSDA_METHOD, CSDA_DIAGNOSTIC_COLUMNS),
        ICDM_METHOD: diagnostic_summary(rows, ICDM_METHOD, ICDM_DIAGNOSTIC_COLUMNS),
    }
    for method, columns in (
        (DSLS_METHOD, ("selected_dsls_epoch",)),
        (CSDA_METHOD, ("selected_csda_epoch",)),
        (ICDM_METHOD, ("selected_icdm_epoch",)),
    ):
        task_rows = [row for row in rows if row["method"] == method and row["scope"] in {"internal_fold", "caltech"}]
        key = columns[0]
        epochs = [int(round(float(row[key]))) for row in task_rows]
        diagnostic_summaries[method][f"{key}_distribution"] = {str(epoch): epochs.count(epoch) for epoch in sorted(set(epochs))}
        diagnostic_summaries[method][f"{key}_zero_count"] = int(sum(epoch == 0 for epoch in epochs))
    task_seconds = [float(sum(float(value) for value in row.get("completed_tasks", {}).values())) for row in runtime_rows]
    worker_wall = max([float(row.get("worker_wall_seconds", 0.0)) for row in runtime_rows] or [0.0])
    selection = _route_selection(decisions, deltas_vs_baseline)
    overall_decision = "PIPELINE_DRIFT" if pipeline_drift else selection["overall_decision"]
    return {
        "status": "PASS", "experiment": "Experiment 1D", "config": CONFIG,
        "dataset": {
            "atlas": "AICHA", "abide1_raw": 995, "abide1_internal_unique_names": 882,
            "caltech_external": 37, "abide1_duplicate_train_occurrences": 76,
            "abide1_site_count": 20, "site_metadata_source": load_cohort().site_metadata_source,
            "site_names": list(load_cohort().site_names), "abide2_external": 727,
        },
        "abide2_original_n": 727, "abide2_evaluable_n": 727, "abide2_excluded_n": 0,
        "abide2_excluded_subject": None, "abide2_excluded_index": None, "exclusion_reason": None,
        "aggregates": aggregates, "deltas_vs_baseline": deltas_vs_baseline,
        "decisions": decisions, "selection": selection, "decision": overall_decision,
        "baseline_sanity": {
            "internal_auc_reference": 0.723, "abide2_auc_reference": 0.739,
            "internal_auc": baseline_internal, "abide2_auc": baseline_abide2,
            "pipeline_drift": bool(pipeline_drift),
        },
        "mechanism_diagnostics": diagnostic_summaries,
        "completion": {
            "expected_internal_folds_per_method": 100, "expected_internal_oof_per_method": 10,
            "expected_caltech_per_method": 10, "expected_abide2_per_method": 10,
            "metrics_rows": len(rows), "methods": list(METHODS),
        },
        "runtime": {
            "workers": len(runtime_rows), "wall_clock_seconds": worker_wall,
            "worker_seconds": float(sum(task_seconds)), "max_repeat_cumulative_seconds": float(max(task_seconds or [0.0])),
            "mean_peak_gpu_alloc_gb": mean_or_nan([row.get("peak_gpu_alloc_gb", float("nan")) for row in runtime_rows]),
            "mean_peak_gpu_reserved_gb": mean_or_nan([row.get("peak_gpu_reserved_gb", float("nan")) for row in runtime_rows]),
            "checkpoint_resume_supported": True, "checkpoint_interval_epochs": int(CONFIG["checkpoint_interval"]),
        },
        "integrity": {
            "root_and_prior_experiments_unchanged": True, "source_hashes": source_hashes,
            "experiment1_runtime_import": False, "original_infonce_only": True,
            "shared_encoder_trajectory": True, "frozen_baseline_classifier": True,
            "reference_unique_outer_train_only": True, "duplicate_train_only": True,
            "validation_only_model_selection": True, "validation_only_threshold": True,
            "test_access_per_epoch": False, "all_method_subject_ids_equal": True,
            "dsls_trainable_parameters": 41, "csda_trainable_parameters": 12321,
            "icdm_trainable_parameters": 12288, "no_ddp_dataparallel_amp_compile": True,
        },
        "literature_audit": {
            "dsls": "Low-rank shared disease-vs-control deviation readout inspired by common-basis and shared-variation analyses; no exact COBE solver is claimed.",
            "csda": "Cross-site same-diagnosis supervised contrastive profile adapter; site is training-only and never an inference input.",
            "icdm": "Bounded individual-conditioned ROI gate before the frozen classifier; no subtype labels or biological subtype claims.",
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    aggregates = summary["aggregates"]
    def value(method: str, scope: str, metric: str) -> float:
        return float(aggregates[f"{method}::{scope}"][metric])

    comparison_rows: list[str] = []
    for display, scope in (("Internal OOF", "internal_oof"), ("Caltech", "caltech"), ("ABIDE-II", "abide2")):
        values = {method: value(method, scope, "auc_mean") for method in METHODS}
        bce = {method: value(method, scope, "bce_mean") for method in METHODS}
        comparison_rows.append(
            f"| {display} | {fmt(values[BASELINE_METHOD])} | {fmt(values[DSLS_METHOD])} | {fmt(values[CSDA_METHOD])} | {fmt(values[ICDM_METHOD])} | "
            f"{fmt(values[DSLS_METHOD]-values[BASELINE_METHOD])} | {fmt(values[CSDA_METHOD]-values[BASELINE_METHOD])} | {fmt(values[ICDM_METHOD]-values[BASELINE_METHOD])} | "
            f"{fmt(bce[BASELINE_METHOD])} | {fmt(bce[DSLS_METHOD])} | {fmt(bce[CSDA_METHOD])} | {fmt(bce[ICDM_METHOD])} |"
        )
    delta_rows: list[str] = []
    for method in METHODS[1:]:
        d = summary["deltas_vs_baseline"][method]
        delta_rows.append(
            f"| {method} | {fmt(np.mean(d['internal_oof']['auc']))} | {fmt(np.mean(d['caltech']['auc']))} | {fmt(np.mean(d['abide2']['auc']))} | "
            f"{fmt(np.mean(d['internal_oof']['bce']))} | {fmt(np.mean(d['caltech']['bce']))} | {fmt(np.mean(d['abide2']['bce']))} |"
        )
    repeat_rows: list[str] = []
    for method in METHODS[1:]:
        d = summary["deltas_vs_baseline"][method]
        repeat_rows.append(
            f"| {method} | "
            + " ".join(fmt(x, 5) for x in d["internal_oof"]["auc"])
            + " | " + " ".join(fmt(x, 5) for x in d["abide2"]["auc"]) + " |"
        )
    decision_lines = []
    for method in METHODS[1:]:
        decision = summary["decisions"][method]
        decision_lines.append(
            f"- **{method}**: `{decision['decision']}`; internal OOF ΔAUC={fmt(decision['internal_mean_delta_auc'], 6)}, "
            f"positive repeats={decision['internal_positive_repeats']}/10, ABIDE-II ΔAUC={fmt(decision['abide2_mean_delta_auc'], 6)}, "
            f"internal/ABIDE-II ΔBCE={fmt(decision['internal_mean_delta_bce'], 6)}/{fmt(decision['abide2_mean_delta_bce'], 6)}."
        )
    diag_lines = []
    for method in METHODS[1:]:
        diagnostic = summary["mechanism_diagnostics"][method]
        diag_lines.append(f"- **{method}**: " + ", ".join(f"{key}={fmt(val, 6)}" for key, val in diagnostic.items() if key.endswith("_mean")))
    runtime = summary["runtime"]
    report = f"""# Experiment 1D

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
{chr(10).join(comparison_rows)}

### Mean paired deltas versus baseline

| Method | Internal ΔAUC | Caltech ΔAUC | ABIDE-II ΔAUC | Internal ΔBCE | Caltech ΔBCE | ABIDE-II ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

### Ten-repeat paired internal/ABIDE-II ΔAUC

| Method | Internal OOF repeats 0--9 | ABIDE-II repeats 0--9 |
|---|---|---|
{chr(10).join(repeat_rows)}

Baseline sanity: internal OOF AUC={fmt(summary['baseline_sanity']['internal_auc'], 6)} (reference 0.723),
ABIDE-II AUC={fmt(summary['baseline_sanity']['abide2_auc'], 6)} (reference 0.739),
pipeline drift={summary['baseline_sanity']['pipeline_drift']}.

## Mechanism diagnostics

{chr(10).join(diag_lines)}

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

{chr(10).join(decision_lines)}

Overall decision: **{summary['decision']}**; selection record:
`{json.dumps(summary['selection'], sort_keys=True)}`.

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
checked before smoke and after aggregation: `{summary['integrity']['root_and_prior_experiments_unchanged']}`.
Workers used full precision, six CPU threads, and repeats 0/2/4/6/8 on GPU0
and 1/3/5/7/9 on GPU1; no DDP, DataParallel, AMP, or compile was used.
Wall clock={fmt(runtime['wall_clock_seconds']/3600.0, 3)} h; summed task
compute={fmt(runtime['worker_seconds']/3600.0, 3)} h; mean peak allocated /
reserved GPU={fmt(runtime['mean_peak_gpu_alloc_gb'], 3)} /
{fmt(runtime['mean_peak_gpu_reserved_gb'], 3)} GB.

Artifacts: `results/metrics.csv`, `results/predictions.npz`,
`results/summary.json`, `results/REPORT.md`, and
`results/external_best_models.pt`. Rolling worker checkpoints are resume-only
and are removed after successful aggregation.
"""
    (EXPERIMENT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def smoke_marker_path() -> Path:
    return RESULTS_DIR / ".smoke_pass.json"


def smoke_test(device: torch.device) -> None:
    """Run the single mandated real-data smoke test."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    marker = smoke_marker_path()
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") == "PASS" and int(payload.get("format_version", 0)) == 1:
            print("SMOKE_ALREADY_PASS " + json.dumps(payload, ensure_ascii=False), flush=True)
            return
        raise RuntimeError(f"existing smoke marker is not a valid Experiment 1D PASS marker: {marker}")
    source_hashes = assert_external_integrity()
    cohort = load_cohort()
    set_all_seeds(777)
    model_config, _ = make_model_config(cohort)
    roi_count = int(cohort.abide1.shape[2])
    local = VarCoNet(model_config, roi_count).to(device)
    root_spec = importlib.util.spec_from_file_location("varconet_root_smoke", REPO_ROOT / "model_scripts" / "VarCoNet.py")
    if root_spec is None or root_spec.loader is None:
        raise RuntimeError("cannot import audited root VarCoNet for smoke parity")
    root_module = importlib.util.module_from_spec(root_spec)
    root_spec.loader.exec_module(root_module)
    root = root_module.VarCoNet(model_config, roi_count).to(device)
    root.load_state_dict(clone_state_cpu(local.state_dict()))
    task = make_tasks(cohort, 0)[0]
    asd_positions = np.where(task.reference_labels == 0)[0]
    hc_positions = np.where(task.reference_labels == 1)[0]
    if len(asd_positions) < 24 or len(hc_positions) < 24:
        raise RuntimeError("smoke reference selection lacks both classes")
    # Keep enough distinct sites for the cross-site positive-mask assertion.
    positions = np.concatenate((asd_positions[:24], hc_positions[:24])).astype(np.int64)
    raw = task.reference_raw[positions]
    labels = torch.as_tensor(task.reference_labels[positions], dtype=torch.long, device=device)
    sites = torch.as_tensor(task.reference_site_ids[positions], dtype=torch.long, device=device)
    x = torch.from_numpy(np.asarray(cohort.abide1[raw], dtype=np.float32)).to(device)
    local.eval(); root.eval()
    with torch.no_grad():
        local_stable, root_stable = local(x), root(x)
    root_parity = float((local_stable - root_stable).abs().max().cpu())
    if root_parity > 1e-6 or tuple(local_stable.shape) != (len(raw), int(CONFIG["stable_dim"])):
        raise RuntimeError(f"root/local stable parity or shape failed: parity={root_parity} shape={tuple(local_stable.shape)}")
    profile_module = ROIConnectivityProfile(roi_count).to(device)
    profile = profile_features(profile_module, local_stable)
    fitted_profile = fit_profile_state(profile)
    dsls_state = fit_dsls_state(local_stable, labels, seed=778, n_components=3)
    dsls_baseline = MLP(int(CONFIG["stable_dim"]), 2).to(device)
    baseline_state = clone_state_cpu(dsls_baseline.state_dict())
    dsls = DSLSResidual(baseline_state, dsls_state).to(device)
    if dsls_parameter_count(dsls) != 41 or not dsls_is_exact_zero_state(dsls_state_payload(dsls)):
        raise RuntimeError("smoke DSLS zero state or parameter count failed")
    dsls_zero = zero_parity_diagnostics(DSLS_METHOD, dsls.forward_details(local_stable), local_stable, labels, baseline_state, device)
    if abs(dsls_zero["zero_probability_max_abs"]) > 1e-6:
        raise RuntimeError("smoke DSLS exact-zero parity failed")
    dsls_loss = binary_bce(dsls.forward_details(local_stable)["p_hc"], labels)
    dsls_loss.backward()
    if not bool(torch.isfinite(dsls_loss)) or not finite_gradients(dsls):
        raise RuntimeError("smoke DSLS forward/backward failed")
    csda = CrossSiteDiseaseAdapter(baseline_state, fitted_profile).to(device)
    if csda_parameter_count(csda) != 12321 or not csda_is_exact_zero_state(csda_state_payload(csda)):
        raise RuntimeError("smoke CSDA zero state or parameter count failed")
    positive = ((labels[:, None] == labels[None, :]) & (sites[:, None] != sites[None, :]) & ~torch.eye(len(labels), dtype=torch.bool, device=device))
    if int(positive.sum()) <= 0:
        raise RuntimeError("smoke CSDA has no cross-site positive pair")
    csda_details = csda.forward_details(local_stable, profile)
    csda_zero = zero_parity_diagnostics(CSDA_METHOD, csda_details, local_stable, labels, baseline_state, device)
    supcon = cross_site_supcon_loss(csda.disease_embedding(profile), labels, sites, tau=float(CONFIG["csda_temperature"]))
    csda_loss = binary_bce(csda_details["p_hc"], labels) + float(CONFIG["csda_lambda"]) * supcon
    csda_loss.backward()
    if not bool(torch.isfinite(csda_loss)) or not finite_gradients(csda):
        raise RuntimeError("smoke CSDA forward/backward failed")
    icdm = IndividualConditionedDiseaseMask(
        baseline_state, fitted_profile, roi_num=roi_count,
        bottleneck=int(CONFIG["icdm_bottleneck"]), max_log_gate=float(CONFIG["icdm_max_log_gate"]),
    ).to(device)
    if icdm_parameter_count(icdm) != 12288 or not icdm_is_exact_zero_state(icdm_state_payload(icdm)):
        raise RuntimeError("smoke ICDM zero state or parameter count failed")
    icdm_details = icdm.forward_details(local_stable, profile)
    if float(icdm_details["roi_prompt"].abs().max()) != 0.0 or float((icdm_details["edge_factor"] - 1.0).abs().max()) > 1e-6:
        raise RuntimeError("smoke ICDM epoch-0 identity failed")
    icdm_zero = zero_parity_diagnostics(ICDM_METHOD, icdm_details, local_stable, labels, baseline_state, device)
    icdm_loss = binary_bce(icdm_details["p_hc"], labels)
    icdm_loss.backward()
    if not bool(torch.isfinite(icdm_loss)) or not finite_gradients(icdm):
        raise RuntimeError("smoke ICDM forward/backward failed")
    payload = {
        "format_version": 1, "status": "PASS", "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_baseline_forward_max_abs": root_parity, "stable_shape": list(local_stable.shape),
        "roi_profile_shape": list(profile.shape), "profile_std_floor": fitted_profile.std_floor,
        "dsls_zero_diagnostics": dsls_zero, "dsls_trainable_parameters": dsls_parameter_count(dsls),
        "dsls_basis_shape": list(dsls_state.basis.shape), "dsls_uncentered_asd_mean_abs": float(dsls_state.asd_score_mean.abs().mean()),
        "csda_zero_diagnostics": csda_zero, "csda_trainable_parameters": csda_parameter_count(csda),
        "csda_cross_site_positive_pairs": int(positive.sum()), "csda_supcon": float(supcon.detach().cpu()),
        "icdm_zero_diagnostics": icdm_zero, "icdm_trainable_parameters": icdm_parameter_count(icdm),
        "source_hashes": source_hashes,
    }
    write_json_atomic(marker, payload)
    print("SMOKE_PASS " + json.dumps(payload, ensure_ascii=False), flush=True)
    del local, root, dsls_baseline, dsls, csda, icdm, x, local_stable, root_stable, profile, profile_module
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate_main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = RESULTS_DIR / "summary.json"
    if existing.exists() and not list(RESULTS_DIR.glob(".worker_r*.csv")):
        try:
            payload = json.loads(existing.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload.get("completion", {}).get("metrics_rows") == 520:
            print(json.dumps({"status": "PASS", "rows": 520, "decision": payload.get("decision"), "already_aggregated": True}), flush=True)
            return
    source_hashes = assert_external_integrity()
    cohort = load_cohort()
    all_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    external_models: dict[str, Any] = {}
    for repeat in range(10):
        paths = worker_paths(repeat)
        rows = read_metric_shard(paths["csv"])
        if not expected_repeat_rows(rows, repeat):
            raise RuntimeError(f"cannot aggregate: repeat {repeat} is incomplete")
        all_rows.extend(rows)
        records = read_prediction_records(paths["predictions"])
        all_records.extend(record for record in records if record["scope"] in {"internal_oof", "caltech", "abide2"})
        if not paths["external"].exists():
            raise RuntimeError(f"cannot aggregate: external model shard missing for repeat {repeat}")
        external = torch_load_cpu(paths["external"])
        if int(external.get("repeat", -1)) != repeat:
            raise RuntimeError(f"external model repeat mismatch: {paths['external']}")
        external_models[str(repeat)] = external
        if paths["runtime"].exists():
            runtime_rows.append(json.loads(paths["runtime"].read_text(encoding="utf-8")))
    if len(all_rows) != 520:
        raise RuntimeError(f"expected 520 metrics rows, found {len(all_rows)}")
    if any(row["status"] != "PASS" for row in all_rows):
        raise RuntimeError("cannot aggregate non-PASS metric rows")
    validate_prediction_records(all_records, cohort)
    source_hashes = assert_external_integrity()
    write_metric_shard_atomic(RESULTS_DIR / "metrics.csv", all_rows)
    write_final_predictions(RESULTS_DIR / "predictions.npz", all_records)
    atomic_torch_save(
        RESULTS_DIR / "external_best_models.pt",
        {"format_version": 3, "atlas": "AICHA", "config": state_to_cpu(CONFIG), "repeats": external_models},
    )
    summary = aggregate_rows(all_rows, runtime_rows, source_hashes)
    summary["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(RESULTS_DIR / "summary.json", summary)
    write_report(summary, all_rows)
    # Remove only exact worker artifacts after all aggregate integrity checks.
    for repeat in range(10):
        for path in worker_paths(repeat).values():
            path.unlink(missing_ok=True)
    for path in CHECKPOINT_DIR.glob("worker_r*_latest.pt*"):
        path.unlink(missing_ok=True)
    for path in CHECKPOINT_DIR.glob("worker_r*_best.pt*"):
        path.unlink(missing_ok=True)
    smoke_marker_path().unlink(missing_ok=True)
    print(json.dumps({"status": "PASS", "rows": len(all_rows), "decision": summary["decision"]}), flush=True)


def worker_main(repeat: int, device_name: str) -> None:
    repeat = int(repeat)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["VARCONET_WORKER_ID"] = f"r{repeat}"
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    paths = worker_paths(repeat)
    try:
        cohort = load_cohort()
        run_repeat(cohort, repeat, device)
        peak_alloc = float(torch.cuda.max_memory_allocated()) / (1024 ** 3) if device.type == "cuda" else 0.0
        peak_reserved = float(torch.cuda.max_memory_reserved()) / (1024 ** 3) if device.type == "cuda" else 0.0
        runtime = json.loads(paths["runtime"].read_text(encoding="utf-8")) if paths["runtime"].exists() else {}
        runtime.update({
            "repeat": repeat, "status": "PASS", "device": str(device),
            "worker_wall_seconds": time.time() - started,
            "peak_gpu_alloc_gb": peak_alloc, "peak_gpu_reserved_gb": peak_reserved,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        paths["error"].unlink(missing_ok=True)
        write_json_atomic(paths["runtime"], runtime)
    except Exception as exc:
        write_json_atomic(paths["error"], {
            "repeat": repeat, "status": "FAIL", "error": repr(exc),
            "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"WORKER_FAIL repeat={repeat} error={exc!r}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "worker", "aggregate"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke_test(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    elif args.mode == "worker":
        if args.repeat < 0 or args.repeat > 9:
            raise ValueError("repeat must be in [0, 9]")
        worker_main(args.repeat, args.device)
    else:
        aggregate_main()


if __name__ == "__main__":
    main()

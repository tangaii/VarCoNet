#!/usr/bin/env python3
"""Experiment 1B-R + 1C: corrected CRSM and selective normative experts.

This runner is self-contained under ``xiaolunwen/experiment1c``.  For every
task it trains one audited original VarCoNet baseline trajectory, freezes the
selected encoder/classifier, then fits two post-hoc branches on exactly the
same extracted stable features:

* ``baseline``
* ``crsm_corrected`` (Experiment 1B-R, exact zero fallback before PCA)
* ``selective_normative_deviation_experts`` (Experiment 1C, SNDE)

No previous experiment or root source file is imported at runtime or modified.
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

from corrected_soft_multiboundary import (  # noqa: E402
    ControlReferencedSoftMultiBoundary,
    MechanismInitError,
    crsm_loss,
    diagnostics_from_model as crsm_diagnostics_from_model,
    initialize_symmetric_pca,
    is_exact_zero_state as crsm_is_exact_zero_state,
    load_delta_state,
    margin_from_baseline_state,
    state_payload as crsm_state_payload,
    trainable_parameter_count as crsm_parameter_count,
)
from model_scripts.VarCoNet import VarCoNet  # noqa: E402
from model_scripts.classifier import MLP  # noqa: E402
from model_scripts.scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from normative_deviation_experts import (  # noqa: E402
    ROIConnectivityProfile,
    SelectiveNormativeDeviationExperts,
    binary_bce_from_details,
    fit_normative_heterogeneity_axis,
    is_exact_zero_state as snde_is_exact_zero_state,
    load_residual_state,
    normative_state_from_payload,
    normative_state_payload,
    snde_diagnostics,
    state_payload as snde_state_payload,
    trainable_parameter_count as snde_parameter_count,
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
    "crsm_lr": 5e-5,
    "snde_lr": 1e-3,
    "checkpoint_interval": 10,
    "cpu_threads": 6,
    "n_experts": 2,
    "temperature": 1.0,
    "symmetry_ratio": 0.01,
    "snde_gate_threshold": 1.0,
    "snde_roi_count": 384,
    "snde_trainable_parameters": 770,
}

BASELINE_METHOD = "baseline"
CRSM_METHOD = "crsm_corrected"
SNDE_METHOD = "selective_normative_deviation_experts"
METHODS = (BASELINE_METHOD, CRSM_METHOD, SNDE_METHOD)

CRSM_DIAGNOSTIC_COLUMNS = (
    "crsm_n_experts", "crsm_temperature", "crsm_hc_centroid_norm",
    "crsm_pca_unit_norm", "crsm_pca_singular_value", "crsm_sigma_margin",
    "crsm_sigma_projection", "crsm_symmetry_ratio", "crsm_symmetry_scale",
    "crsm_initial_delta_l2", "crsm_asd_mass1", "crsm_asd_mass2",
    "crsm_asd_entropy", "crsm_hc_mass1", "crsm_hc_mass2", "crsm_hc_entropy",
    "crsm_delta1_l2", "crsm_delta2_l2", "crsm_delta_cosine",
    "crsm_delta_bias1", "crsm_delta_bias2",
)
SNDE_DIAGNOSTIC_COLUMNS = (
    "snde_asd_score_mean", "snde_asd_score_std", "snde_asd_positive_fraction",
    "snde_asd_negative_fraction", "snde_asd_generalist_fraction",
    "snde_asd_mean_abs_correction", "snde_hc_score_mean", "snde_hc_score_std",
    "snde_hc_positive_fraction", "snde_hc_negative_fraction",
    "snde_hc_generalist_fraction", "snde_hc_mean_abs_correction",
    "snde_w_pos_l2", "snde_w_neg_l2", "snde_b_pos", "snde_b_neg",
    "snde_positive_weight_l2", "snde_negative_weight_l2",
    "snde_parameter_l2", "snde_mean_abs_correction",
    "snde_mean_abs_positive_correction", "snde_mean_abs_negative_correction",
    "snde_pc1_explained_variance_fraction", "snde_pc1_explained_fraction", "snde_std_floor",
    "snde_hc_reference_score_mean", "snde_hc_reference_score_std",
)
METRIC_COLUMNS = [
    "method", "scope", "repeat", "fold", "auc", "bce", "f1", "balanced_accuracy",
    "selected_encoder_epoch", "selected_classifier_epoch", "selected_crsm_epoch",
    "selected_snde_epoch", "threshold_hc", *CRSM_DIAGNOSTIC_COLUMNS,
    *SNDE_DIAGNOSTIC_COLUMNS, "task_seconds", "n_test", "status",
]
FLOAT_COLUMNS = {
    "auc", "bce", "f1", "balanced_accuracy", "selected_encoder_epoch",
    "selected_classifier_epoch", "selected_crsm_epoch", "selected_snde_epoch",
    "threshold_hc", *CRSM_DIAGNOSTIC_COLUMNS, *SNDE_DIAGNOSTIC_COLUMNS,
    "task_seconds",
}
INT_COLUMNS = {"repeat", "n_test"}
PREDICTION_FIELDS = (
    "scope", "method", "repeat", "fold", "subject_index", "raw_index", "y", "p_hc", "threshold",
)

# Captured before Experiment 1C creation; generated artifacts are excluded.
EXPECTED_SOURCE_DIGESTS = {
    "model_scripts": "fa3af48175c37f0d03eda9863fa9ef1e5735e474cf34fb5f6751bb44733b34b0",
    "xiaolunwen/experiment1": "db87b36cdc5af09afb23701ed2591b6b93b51b7af35872c05fab61703f9e9f9b",
    "xiaolunwen/experiment1b": "a8c0d0fc0a1f2e04a69748004d242449948b34f4553417317ba944bed76b1ee3",
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
    duplicate_raw: np.ndarray
    duplicate_names: list[str]
    duplicate_labels: np.ndarray
    internal_raw: np.ndarray
    internal_names: list[str]
    internal_labels: np.ndarray
    caltech_raw: np.ndarray
    caltech_names: list[str]
    caltech_labels: np.ndarray
    abide2: np.ndarray
    abide2_names: list[str]
    abide2_labels: np.ndarray


@dataclass
class Task:
    repeat: int
    scope: str
    fold: str
    seed: int
    reference_raw: np.ndarray
    reference_labels: np.ndarray
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
                f"INTEGRITY_FAIL: source outside experiment1c changed: {key} expected={expected} observed={value}"
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
    """Reconstruct the audited ABIDE cohort and its train-only duplicates."""
    abide1_dir = DATASET_DIR / "ABIDEI"
    abide2_dir = DATASET_DIR / "ABIDEII"
    abide1 = load_npz_stack(abide1_dir / "ABIDEI_nilearn_AICHA.npz")
    abide1_names = load_names(abide1_dir / "ABIDEI_nilearn_names.txt")
    abide1_labels = np.load(abide1_dir / "ABIDEI_nilearn_classes.npy", allow_pickle=False).astype(np.int64)
    if len(abide1) != len(abide1_names) or len(abide1) != len(abide1_labels):
        raise RuntimeError("ABIDE-I data, names, and labels are not aligned")
    if abide1.shape[2] != int(CONFIG["roi_count"]):
        raise RuntimeError(f"AICHA ROI count drifted: {abide1.shape}")
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
    caltech_set = set(caltech_names)
    internal_names = [name for name in singleton_names if name not in caltech_set]
    internal_raw = np.asarray([name_to_raw[name] for name in internal_names], dtype=np.int64)
    caltech_raw = np.asarray([name_to_raw[name] for name in caltech_names], dtype=np.int64)
    abide2 = load_npz_stack(abide2_dir / "ABIDEII_nilearn_AICHA.npz")
    abide2_names = load_names(abide2_dir / "ABIDEII_nilearn_names.txt")
    abide2_labels = np.load(abide2_dir / "ABIDEII_nilearn_classes.npy", allow_pickle=False).astype(np.int64)
    if len(abide2) != len(abide2_names) or len(abide2) != len(abide2_labels):
        raise RuntimeError("ABIDE-II data, names, and labels are not aligned")
    if abide1.shape[1:] != abide2.shape[1:]:
        raise RuntimeError(f"ABIDE atlas shapes differ: {abide1.shape} vs {abide2.shape}")
    counts_expected = (len(internal_raw), len(caltech_raw), len(duplicate_raw), len(abide2))
    if counts_expected != (882, 37, 76, 727):
        raise RuntimeError(f"audited cohort counts drifted: {counts_expected}")
    if len(singleton_raw) != 919:
        raise RuntimeError("ABIDE-I singleton count drifted")
    return Cohort(
        abide1=abide1, abide1_names=abide1_names, abide1_labels=abide1_labels,
        duplicate_raw=duplicate_raw, duplicate_names=duplicate_names,
        duplicate_labels=abide1_labels[duplicate_raw], internal_raw=internal_raw,
        internal_names=internal_names, internal_labels=abide1_labels[internal_raw],
        caltech_raw=caltech_raw, caltech_names=caltech_names,
        caltech_labels=abide1_labels[caltech_raw], abide2=abide2,
        abide2_names=abide2_names, abide2_labels=abide2_labels,
    )


def make_tasks(cohort: Cohort, repeat: int) -> list[Task]:
    """Create ten paired internal tasks and one paired external task."""
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
            train_raw=np.concatenate((cohort.duplicate_raw, np.asarray(train_unique_raw, dtype=np.int64))),
            train_names=cohort.duplicate_names + [raw_to_name[int(raw)] for raw in train_unique_raw],
            train_labels=np.concatenate((cohort.duplicate_labels, np.asarray(train_unique_y, dtype=np.int64))),
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
        train_raw=np.concatenate((cohort.duplicate_raw, np.asarray(ext_train_raw, dtype=np.int64))),
        train_names=cohort.duplicate_names + [raw_to_name[int(raw)] for raw in ext_train_raw],
        train_labels=np.concatenate((cohort.duplicate_labels, np.asarray(ext_train_y, dtype=np.int64))),
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
    """Extract stable learned-FC exactly once for a requested partition."""
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
    pred_asd = (pred_hc == 0).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, p_hc)),
        "bce": float(log_loss(labels, np.clip(p_hc, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "f1": float(f1_score(asd, pred_asd, zero_division=0)),
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
    """The locked original `Linear(73536,2)+Softmax` baseline readout."""
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
            best_bce, best_epoch = float(val_bce), int(epoch)
            best_state = clone_state_cpu(classifier.state_dict())
    if best_state is None or best_epoch is None:
        raise RuntimeError("baseline classifier did not select a validation state")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        p_hc = classifier(stable_val)[:, 1].detach().cpu().numpy().astype(np.float64)
    return {
        "classifier_state": best_state, "val_bce": float(best_bce),
        "classifier_epoch": int(best_epoch), "val_probs": p_hc,
        "threshold": validation_threshold(val_labels.detach().cpu().numpy(), p_hc),
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
    return (float(torch.cuda.memory_allocated(device)) / (1024 ** 3), float(torch.cuda.memory_reserved(device)) / (1024 ** 3))


def task_identity(task: Task) -> dict[str, Any]:
    return {"repeat": int(task.repeat), "scope": str(task.scope), "fold": str(task.fold), "task_seed": int(task.seed)}


def identity_matches(payload: dict[str, Any], task: Task) -> bool:
    return all(payload.get(key) == value for key, value in task_identity(task).items())


def checkpoint_paths(repeat: int) -> tuple[Path, Path]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"r{int(repeat)}")
    return CHECKPOINT_DIR / f"worker_{token}_latest.pt", CHECKPOINT_DIR / f"worker_{token}_best.pt"


def checkpoint_rank(payload: dict[str, Any]) -> tuple[int, int]:
    ranks = {"ssl": 0, "crsm_corrected": 1, "snde": 2}
    return ranks.get(str(payload.get("phase")), -1), int(payload.get("epoch", -1))


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


def base_checkpoint_payload(
    task: Task, phase: str, epoch: int, encoder: VarCoNet, best_baseline: dict[str, Any] | None,
    elapsed_seconds: float, **extra: Any,
) -> dict[str, Any]:
    return {
        "format_version": 2, "phase": str(phase), "saved_unix": time.time(), **task_identity(task),
        "epoch": int(epoch), "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "best_baseline": state_to_cpu(best_baseline), "rng_state": state_to_cpu(capture_rng_state()),
        "elapsed_seconds": float(elapsed_seconds), **state_to_cpu(extra),
    }


def make_baseline_branch(encoder: VarCoNet, encoder_epoch: int, fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()), "classifier_state": state_to_cpu(fit["classifier_state"]),
        "val_bce": float(fit["val_bce"]), "encoder_epoch": int(encoder_epoch),
        "classifier_epoch": int(fit["classifier_epoch"]), "threshold": float(fit["threshold"]),
        "val_probs": np.asarray(fit["val_probs"], dtype=np.float64).copy(),
        "crsm_diagnostics": None, "snde_diagnostics": None,
    }


def assert_tensor_state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], label: str) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"INTEGRITY_FAIL: {label} state key mismatch")
    for key, value in left.items():
        if not torch.equal(value, right[key]):
            raise RuntimeError(f"INTEGRITY_FAIL: {label} differs at {key}")


def assert_shared_selection(task_result: dict[str, Any]) -> None:
    baseline = task_result["baseline"]
    for key in ("crsm", "snde"):
        branch = task_result[key]
        if int(branch["encoder_epoch"]) != int(baseline["encoder_epoch"]):
            raise RuntimeError(f"INTEGRITY_FAIL: baseline/{key} selected encoder epoch differs")
        if int(branch["classifier_epoch"]) != int(baseline["classifier_epoch"]):
            raise RuntimeError(f"INTEGRITY_FAIL: baseline/{key} selected classifier epoch differs")
        assert_tensor_state_equal(baseline["encoder_state"], branch["encoder_state"], f"baseline/{key} encoder")
        assert_tensor_state_equal(baseline["classifier_state"], branch["baseline_classifier_state"], f"baseline/{key} classifier")


def zero_crsm_diagnostics(
    model: ControlReferencedSoftMultiBoundary, stable_val: torch.Tensor, val_labels: torch.Tensor,
    baseline_state: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, float]:
    criterion = torch.nn.BCELoss()
    target = F.one_hot(val_labels, num_classes=2).float()
    with torch.no_grad():
        baseline_out = predict_baseline_tensor(stable_val, baseline_state, device)
        details = model.forward_details(stable_val)
        zero_out = torch.stack((details["p_asd"], details["p_hc"]), dim=1)
        direct_margin = margin_from_baseline_state(stable_val, baseline_state)
        baseline_bce = criterion(baseline_out, target)
        zero_bce = criterion(zero_out, target)
        loss_parity = crsm_loss(details, val_labels)
        diagnostics = {
            "zero_probability_max_abs": float((baseline_out - zero_out).abs().max().cpu()),
            "zero_margin_max_abs": float((details["m0"] - direct_margin).abs().max().cpu()),
            "zero_p_hc_max_abs": float((details["p_hc"] - torch.sigmoid(-direct_margin)).abs().max().cpu()),
            "zero_bce_difference": float((zero_bce - baseline_bce).abs().cpu()),
            "zero_crsm_bce_difference": float((loss_parity - baseline_bce).abs().cpu()),
            "baseline_val_bce": float(baseline_bce.cpu()),
        }
    if max(diagnostics[key] for key in diagnostics if key != "baseline_val_bce") > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: exact zero CRSM does not equal frozen baseline")
    return diagnostics


def crsm_validation(model: ControlReferencedSoftMultiBoundary, stable_val: torch.Tensor, val_labels: torch.Tensor) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output = model(stable_val)
        bce = float(torch.nn.BCELoss()(output, F.one_hot(val_labels, num_classes=2).float()).cpu())
        p_hc = output[:, 1].detach().cpu().numpy().astype(np.float64)
    return bce, p_hc


def zero_snde_diagnostics(
    model: SelectiveNormativeDeviationExperts, stable_val: torch.Tensor, val_labels: torch.Tensor,
    baseline_state: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, float]:
    criterion = torch.nn.BCELoss()
    target = F.one_hot(val_labels, num_classes=2).float()
    with torch.no_grad():
        baseline_out = predict_baseline_tensor(stable_val, baseline_state, device)
        details = model.forward_details(stable_val)
        zero_out = torch.stack((details["p_asd"], details["p_hc"]), dim=1)
        direct_binary = F.binary_cross_entropy(details["p_hc"], (val_labels == 1).float())
        baseline_bce = criterion(baseline_out, target)
        zero_bce = criterion(zero_out, target)
        diagnostics = {
            "zero_probability_max_abs": float((baseline_out - zero_out).abs().max().cpu()),
            "zero_margin_max_abs": float((details["m0"] - margin_from_baseline_state(stable_val, baseline_state)).abs().max().cpu()),
            "zero_bce_difference": float((zero_bce - baseline_bce).abs().cpu()),
            "zero_binary_bce_difference": float((direct_binary - baseline_bce).abs().cpu()),
            "baseline_val_bce": float(baseline_bce.cpu()),
        }
    if max(diagnostics[key] for key in diagnostics if key != "baseline_val_bce") > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: zero SNDE does not equal frozen baseline")
    return diagnostics


def snde_validation(model: SelectiveNormativeDeviationExperts, stable_val: torch.Tensor, val_labels: torch.Tensor) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable_val)
        bce = float(binary_bce_from_details(details, val_labels).cpu())
        p_hc = details["p_hc"].detach().cpu().numpy().astype(np.float64)
    return bce, p_hc


def _init_diag_fields(raw: dict[str, float], hc_centroid: torch.Tensor) -> dict[str, float]:
    return {
        "crsm_n_experts": float(raw["n_experts"]), "crsm_temperature": float(raw["temperature"]),
        "crsm_hc_centroid_norm": float(torch.linalg.vector_norm(hc_centroid).detach().cpu()),
        "crsm_pca_unit_norm": float(raw["pca_unit_norm"]), "crsm_pca_singular_value": float(raw["pca_singular_value"]),
        "crsm_sigma_margin": float(raw["sigma_margin"]), "crsm_sigma_projection": float(raw["sigma_projection"]),
        "crsm_symmetry_ratio": float(raw["symmetry_ratio"]), "crsm_symmetry_scale": float(raw["symmetry_scale"]),
        "crsm_initial_delta_l2": float(raw["initial_delta_l2"]),
    }


def _allclose_state(left: dict[str, Any], right: dict[str, Any], atol: float = 1e-6) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or not torch.allclose(a.cpu(), b.cpu(), rtol=0.0, atol=atol):
                return False
        elif isinstance(a, (float, int)):
            if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol):
                return False
        elif a != b:
            return False
    return True


def train_shared_task(cohort: Cohort, task: Task, device: torch.device) -> dict[str, Any]:
    """Train one baseline, then corrected CRSM and SNDE without test access."""
    started = time.time()
    print(
        f"TASK_START repeat={task.repeat} scope={task.scope} fold={task.fold} seed={task.seed} device={device} "
        f"methods={','.join(METHODS)} shared_encoder=Y reference_unique_n={len(task.reference_raw)} "
        f"train_n={len(task.train_raw)} val_n={len(task.val_raw)}", flush=True,
    )
    set_all_seeds(task.seed)
    model_config, best_params = make_model_config(cohort)
    encoder = VarCoNet(model_config, int(cohort.abide1.shape[2])).to(device)
    contrast = DualBranchContrast(loss=InfoNCE(tau=float(best_params["tau"])), mode="L2L").to(device)
    optimizer_ssl = Adam(encoder.parameters(), lr=float(best_params["lr"]))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer_ssl, warmup_start_lr=1e-5, warmup_epochs=int(CONFIG["warm_up_epochs"]), max_epochs=int(CONFIG["epochs"]),
    )
    schedule = paired_batch_schedule(task.train_names, len(task.train_raw), int(CONFIG["epochs"]), task.seed)
    latest_path, best_path = checkpoint_paths(task.repeat)
    source = load_task_checkpoint(task.repeat, task)
    phase, start_ssl_epoch, best_baseline = "ssl", 1, None
    ssl_losses: list[float] = []
    if source is not None:
        phase = str(source.get("phase", "ssl"))
        elapsed = float(source.get("elapsed_seconds", 0.0))
        started -= elapsed
        best_baseline = state_to_cpu(source.get("best_baseline"))
        if source.get("rng_state") is not None:
            restore_rng_state(source["rng_state"])
        if phase == "ssl":
            encoder.load_state_dict(source["encoder_state_dict"])
            optimizer_ssl.load_state_dict(source["optimizer_state_dict"])
            optimizer_to_device(optimizer_ssl, device)
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_ssl_epoch = int(source["epoch"]) + 1
        elif phase in {"crsm_corrected", "snde"}:
            if best_baseline is None:
                raise RuntimeError("resume checkpoint after SSL lacks baseline state")
            encoder.load_state_dict(best_baseline["encoder_state"])
            start_ssl_epoch = int(CONFIG["epochs"]) + 1
        else:
            raise RuntimeError(f"unknown checkpoint phase {phase!r}")
        print(
            f"RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} phase={phase} "
            f"from_epoch={int(source.get('epoch', 0)) + 1} elapsed_seconds={elapsed:.1f}", flush=True,
        )
    train_labels = torch.as_tensor(task.train_labels, dtype=torch.long, device=device)
    val_labels = torch.as_tensor(task.val_labels, dtype=torch.long, device=device)
    for epoch in range(start_ssl_epoch, int(CONFIG["epochs"]) + 1):
        epoch_started = time.time()
        ssl_loss = train_ssl_epoch(encoder, contrast, optimizer_ssl, cohort, task, schedule[epoch - 1], int(model_config["max_length"]), device)
        scheduler.step()
        ssl_losses.append(ssl_loss)
        stable_train = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
        stable_val = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
        trajectory_rng = capture_rng_state()
        baseline_fit = fit_baseline_classifier(stable_train, train_labels, stable_val, val_labels, device, task.seed + 300000 + epoch * 2)
        restore_rng_state(trajectory_rng)
        candidate = make_baseline_branch(encoder, epoch, baseline_fit)
        is_best = best_baseline is None or float(candidate["val_bce"]) < float(best_baseline["val_bce"])
        if is_best:
            best_baseline = candidate
        checkpoint = base_checkpoint_payload(
            task, "ssl", epoch, encoder, best_baseline, time.time() - started,
            optimizer_state_dict=state_to_cpu(optimizer_ssl.state_dict()), scheduler_state_dict=state_to_cpu(scheduler.state_dict()),
        )
        if is_best:
            save_checkpoint(best_path, "best", checkpoint)
        if epoch % int(CONFIG["checkpoint_interval"]) == 0:
            save_checkpoint(latest_path, "latest", checkpoint)
        allocated, reserved = gpu_memory_gb(device)
        print(
            f"[SSL] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs']} "
            f"ssl_loss={ssl_loss:.6f} lr={optimizer_ssl.param_groups[0]['lr']:.9f} "
            f"base_val_bce={baseline_fit['val_bce']:.6f} base_best={'Y' if is_best else 'N'} "
            f"base_cls_epoch={baseline_fit['classifier_epoch']} epoch_seconds={time.time()-epoch_started:.1f} "
            f"task_elapsed_minutes={(time.time()-started)/60.0:.1f} gpu_alloc_gb={allocated:.3f} gpu_reserved_gb={reserved:.3f}", flush=True,
        )
        del stable_train, stable_val
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if best_baseline is None:
        raise RuntimeError("no validation-selected baseline branch")
    encoder.load_state_dict(best_baseline["encoder_state"])
    encoder.eval()
    stable_reference = extract_stable_features(encoder, cohort.abide1, task.reference_raw, device)
    stable_train = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
    stable_val = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
    reference_labels = torch.as_tensor(task.reference_labels, dtype=torch.long, device=device)
    expected_ref_labels = torch.as_tensor(cohort.abide1_labels[task.reference_raw], dtype=torch.long, device=device)
    if not torch.equal(reference_labels, expected_ref_labels):
        raise RuntimeError("reference labels are not aligned with unique outer-training raw indices")
    hc_reference = stable_reference[reference_labels == 1]
    if len(hc_reference) == 0:
        raise MechanismInitError("MECHANISM_INIT_FAIL: no HC singleton reference subjects")
    hc_centroid = hc_reference.mean(dim=0)
    if not bool(torch.isfinite(hc_centroid).all()):
        raise MechanismInitError("MECHANISM_INIT_FAIL: non-finite HC reference centroid")

    def crsm_checkpoint(epoch: int, progress: dict[str, Any], is_best: bool = False) -> None:
        payload = base_checkpoint_payload(task, "crsm_corrected", epoch, encoder, best_baseline, time.time() - started, crsm_progress=progress)
        if is_best:
            save_checkpoint(best_path, "best", payload)
        if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs_cls"]) - 1:
            save_checkpoint(latest_path, "latest", payload)

    # Experiment 1B-R: exact-zero candidate BEFORE the PCA symmetry perturbation.
    if phase == "snde" and source is not None:
        crsm_branch = state_to_cpu(source["crsm_branch"])
        if not torch.allclose(crsm_branch["hc_centroid"], hc_centroid.detach().cpu(), rtol=0.0, atol=1e-6):
            raise RuntimeError("INTEGRITY_FAIL: resumed CRSM centroid differs from reference centroid")
        print(f"CRSM_RESTORE repeat={task.repeat} scope={task.scope} fold={task.fold} selected_epoch={crsm_branch['crsm_epoch']} from_snde_checkpoint=Y", flush=True)
    else:
        crsm = ControlReferencedSoftMultiBoundary(best_baseline["classifier_state"], hc_centroid, temperature=float(CONFIG["temperature"])).to(device)
        if crsm_parameter_count(crsm) != 147074:
            raise RuntimeError(f"INTEGRITY_FAIL: CRSM parameter count={crsm_parameter_count(crsm)} expected=147074")
        zero_state = crsm_state_payload(crsm)
        if not crsm_is_exact_zero_state(zero_state):
            raise RuntimeError("INTEGRITY_FAIL: CRSM constructor is not exact-zero initialized")
        zero_diag = zero_crsm_diagnostics(crsm, stable_val, val_labels, best_baseline["classifier_state"], device)
        if abs(float(best_baseline["val_bce"]) - zero_diag["baseline_val_bce"]) > 1e-6:
            raise RuntimeError("INTEGRITY_FAIL: CRSM exact-zero BCE differs from selected baseline BCE")
        print(f"[CRSM_ZERO_BASELINE] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch=0 exact_zero=Y val_bce={zero_diag['baseline_val_bce']:.6f} probability_max_abs={zero_diag['zero_probability_max_abs']:.3g}", flush=True)
        optimizer_crsm = Adam(crsm.parameters(), lr=float(CONFIG["crsm_lr"]))
        if phase == "crsm_corrected" and source is not None:
            progress = source["crsm_progress"]
            stored_centroid = progress["hc_centroid"].to(device)
            if not torch.allclose(stored_centroid, hc_centroid, rtol=0.0, atol=1e-6):
                raise RuntimeError("INTEGRITY_FAIL: resumed CRSM centroid differs from reference centroid")
            if not crsm_is_exact_zero_state(progress["zero_delta_state"]):
                raise RuntimeError("INTEGRITY_FAIL: stored CRSM zero checkpoint is not bitwise zero")
            for key, value in progress["zero_diagnostics"].items():
                if abs(float(value) - float(zero_diag[key])) > 1e-6:
                    raise RuntimeError(f"INTEGRITY_FAIL: resumed CRSM zero diagnostic changed: {key}")
            if progress.get("stage") == "exact_zero":
                # Resume safely if interrupted between the mandated exact-zero
                # save and the deterministic PCA symmetry state.
                initialization = initialize_symmetric_pca(crsm, stable_reference, reference_labels, seed=task.seed + 700000, ratio=float(CONFIG["symmetry_ratio"]))
                init_fields = _init_diag_fields(initialization.as_dict(), hc_centroid)
                if abs(init_fields["crsm_pca_unit_norm"] - 1.0) > 1e-5 or crsm_is_exact_zero_state(crsm_state_payload(crsm)):
                    raise MechanismInitError("MECHANISM_INIT_FAIL: resumed PCA symmetry state is invalid")
                best_delta, best_crsm_epoch, best_crsm_bce, crsm_start_epoch = state_to_cpu(zero_state), 0, float(zero_diag["baseline_val_bce"]), 1
                progress = {
                    "stage": "active", "hc_centroid": hc_centroid.detach().cpu().clone(), "zero_delta_state": state_to_cpu(zero_state),
                    "zero_diagnostics": state_to_cpu(zero_diag), "init_fields": state_to_cpu(init_fields),
                    "current_delta_state": crsm_state_payload(crsm), "optimizer_state_dict": state_to_cpu(optimizer_crsm.state_dict()),
                    "best_delta_state": state_to_cpu(best_delta), "best_crsm_epoch": best_crsm_epoch, "best_val_bce": best_crsm_bce,
                }
                crsm_checkpoint(0, progress, is_best=True)
                print(f"[CRSM_SYMMETRY_START] repeat={task.repeat} scope={task.scope} fold={task.fold} resumed_after_exact_zero=Y pca_unit_norm={init_fields['crsm_pca_unit_norm']:.8f} symmetry_scale={init_fields['crsm_symmetry_scale']:.9g}", flush=True)
            else:
                init_fields = {key: float(value) for key, value in progress["init_fields"].items()}
                load_delta_state(crsm, progress["current_delta_state"])
                optimizer_crsm.load_state_dict(progress["optimizer_state_dict"])
                optimizer_to_device(optimizer_crsm, device)
                best_delta = state_to_cpu(progress["best_delta_state"])
                best_crsm_epoch, best_crsm_bce = int(progress["best_crsm_epoch"]), float(progress["best_val_bce"])
                crsm_start_epoch = int(source["epoch"]) + 1
                print(f"CRSM_RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} from_epoch={crsm_start_epoch} best_epoch={best_crsm_epoch} best_val_bce={best_crsm_bce:.6f}", flush=True)
        else:
            zero_payload = base_checkpoint_payload(
                task, "crsm_corrected", 0, encoder, best_baseline, time.time() - started,
                crsm_progress={"stage": "exact_zero", "hc_centroid": hc_centroid.detach().cpu().clone(), "zero_delta_state": state_to_cpu(zero_state), "zero_diagnostics": state_to_cpu(zero_diag)},
            )
            save_checkpoint(latest_path, "latest", zero_payload)
            save_checkpoint(best_path, "best", zero_payload)
            initialization = initialize_symmetric_pca(crsm, stable_reference, reference_labels, seed=task.seed + 700000, ratio=float(CONFIG["symmetry_ratio"]))
            init_fields = _init_diag_fields(initialization.as_dict(), hc_centroid)
            if abs(init_fields["crsm_pca_unit_norm"] - 1.0) > 1e-5 or crsm_is_exact_zero_state(crsm_state_payload(crsm)):
                raise MechanismInitError("MECHANISM_INIT_FAIL: PCA symmetry state is invalid")
            best_delta, best_crsm_epoch, best_crsm_bce, crsm_start_epoch = state_to_cpu(zero_state), 0, float(zero_diag["baseline_val_bce"]), 1
            progress = {
                "stage": "active", "hc_centroid": hc_centroid.detach().cpu().clone(), "zero_delta_state": state_to_cpu(zero_state),
                "zero_diagnostics": state_to_cpu(zero_diag), "init_fields": state_to_cpu(init_fields),
                "current_delta_state": crsm_state_payload(crsm), "optimizer_state_dict": state_to_cpu(optimizer_crsm.state_dict()),
                "best_delta_state": state_to_cpu(best_delta), "best_crsm_epoch": best_crsm_epoch, "best_val_bce": best_crsm_bce,
            }
            crsm_checkpoint(0, progress, is_best=True)
            print(f"[CRSM_SYMMETRY_START] repeat={task.repeat} scope={task.scope} fold={task.fold} after_exact_zero=Y pca_unit_norm={init_fields['crsm_pca_unit_norm']:.8f} pca_singular={init_fields['crsm_pca_singular_value']:.6f} symmetry_scale={init_fields['crsm_symmetry_scale']:.9g}", flush=True)
        for epoch in range(crsm_start_epoch, int(CONFIG["epochs_cls"])):
            epoch_started = time.time()
            crsm.train()
            optimizer_crsm.zero_grad()
            details = crsm.forward_details(stable_train)
            train_loss = crsm_loss(details, train_labels)
            train_loss.backward()
            if not bool(torch.isfinite(train_loss)) or not finite_gradients(crsm):
                raise FloatingPointError("non-finite corrected CRSM loss or gradient")
            optimizer_crsm.step()
            val_bce, _ = crsm_validation(crsm, stable_val, val_labels)
            is_best = val_bce < best_crsm_bce
            if is_best:
                best_crsm_bce, best_crsm_epoch, best_delta = float(val_bce), int(epoch), crsm_state_payload(crsm)
            progress = {
                "stage": "active", "hc_centroid": hc_centroid.detach().cpu().clone(), "zero_delta_state": state_to_cpu(zero_state),
                "zero_diagnostics": state_to_cpu(zero_diag), "init_fields": state_to_cpu(init_fields),
                "current_delta_state": crsm_state_payload(crsm), "optimizer_state_dict": state_to_cpu(optimizer_crsm.state_dict()),
                "best_delta_state": state_to_cpu(best_delta), "best_crsm_epoch": int(best_crsm_epoch), "best_val_bce": float(best_crsm_bce),
            }
            if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs_cls"]) - 1:
                crsm_checkpoint(epoch, progress, is_best=is_best)
            if epoch % 10 == 0 or is_best or epoch == int(CONFIG["epochs_cls"]) - 1:
                diagnostic = crsm_diagnostics_from_model(crsm, stable_reference, reference_labels)
                print(
                    f"[CRSM_CORRECTED] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs_cls']-1} "
                    f"train_loss={float(train_loss.detach().cpu()):.6f} val_bce={val_bce:.6f} best={'Y' if is_best else 'N'} best_epoch={best_crsm_epoch} "
                    f"asd_mass=({diagnostic['crsm_asd_mass1']:.4f},{diagnostic['crsm_asd_mass2']:.4f}) hc_mass=({diagnostic['crsm_hc_mass1']:.4f},{diagnostic['crsm_hc_mass2']:.4f}) "
                    f"delta_l2=({diagnostic['crsm_delta1_l2']:.6f},{diagnostic['crsm_delta2_l2']:.6f}) epoch_seconds={time.time()-epoch_started:.2f}", flush=True,
                )
        load_delta_state(crsm, best_delta)
        final_crsm_bce, final_crsm_probs = crsm_validation(crsm, stable_val, val_labels)
        if abs(final_crsm_bce - best_crsm_bce) > 1e-6:
            raise RuntimeError("INTEGRITY_FAIL: selected CRSM state does not reproduce selected validation BCE")
        if best_crsm_epoch == 0:
            if not crsm_is_exact_zero_state(best_delta):
                raise RuntimeError("INTEGRITY_FAIL: selected CRSM epoch 0 is not bitwise exact zero")
            if float(np.max(np.abs(final_crsm_probs - np.asarray(best_baseline["val_probs"], dtype=np.float64)))) > 1e-6:
                raise RuntimeError("INTEGRITY_FAIL: selected CRSM epoch-0 probabilities differ from baseline")
        crsm_diag = {**init_fields, **crsm_diagnostics_from_model(crsm, stable_reference, reference_labels)}
        crsm_branch = {
            "encoder_state": clone_state_cpu(encoder.state_dict()), "baseline_classifier_state": state_to_cpu(best_baseline["classifier_state"]),
            "hc_centroid": hc_centroid.detach().cpu().clone(), "delta_state": state_to_cpu(best_delta), "zero_delta_state": state_to_cpu(zero_state),
            "val_bce": float(best_crsm_bce), "encoder_epoch": int(best_baseline["encoder_epoch"]), "classifier_epoch": int(best_baseline["classifier_epoch"]),
            "crsm_epoch": int(best_crsm_epoch), "threshold": validation_threshold(task.val_labels, final_crsm_probs), "val_probs": final_crsm_probs,
            "crsm_diagnostics": crsm_diag, "snde_diagnostics": None,
        }
        del crsm, optimizer_crsm

    # Experiment 1C: reference-only control normalization and a single ASD PC.
    fitted_normative = fit_normative_heterogeneity_axis(stable_reference, reference_labels, seed=task.seed + 800000, roi_count=int(CONFIG["snde_roi_count"]))
    if phase == "snde" and source is not None:
        stored_normative = normative_state_from_payload(source["normative_state"])
        if not _allclose_state(normative_state_payload(fitted_normative), normative_state_payload(stored_normative)):
            raise RuntimeError("INTEGRITY_FAIL: resumed SNDE normative state differs from reference-only fit")
        normative = stored_normative
    else:
        normative = fitted_normative
    snde = SelectiveNormativeDeviationExperts(best_baseline["classifier_state"], normative, roi_count=int(CONFIG["snde_roi_count"]), gate_threshold=float(CONFIG["snde_gate_threshold"])).to(device)
    if snde_parameter_count(snde) != int(CONFIG["snde_trainable_parameters"]):
        raise RuntimeError(f"INTEGRITY_FAIL: SNDE parameter count={snde_parameter_count(snde)} expected=770")
    normative_gate_diag = snde_diagnostics(snde, stable_reference, reference_labels)
    print(
        f"[SNDE_NORMATIVE] repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"n_unique_asd={int((reference_labels==0).sum().item())} n_unique_hc={int((reference_labels==1).sum().item())} "
        f"roi_dim=384 std_floor={normative.std_floor:.8g} "
        f"pc1_explained_fraction={normative.pc1_explained_fraction:.6f} "
        f"hc_score_mean_before_norm={normative.hc_score_mean:.6g} hc_score_std_before_norm={normative.hc_score_std:.6g} "
        f"asd_positive_fraction={normative_gate_diag['snde_asd_positive_fraction']:.4f} "
        f"asd_negative_fraction={normative_gate_diag['snde_asd_negative_fraction']:.4f} "
        f"asd_generalist_fraction={normative_gate_diag['snde_asd_generalist_fraction']:.4f} "
        f"hc_positive_fraction={normative_gate_diag['snde_hc_positive_fraction']:.4f} "
        f"hc_negative_fraction={normative_gate_diag['snde_hc_negative_fraction']:.4f} "
        f"hc_generalist_fraction={normative_gate_diag['snde_hc_generalist_fraction']:.4f}",
        flush=True,
    )
    zero_snde_state = snde_state_payload(snde)
    if not snde_is_exact_zero_state(zero_snde_state):
        raise RuntimeError("INTEGRITY_FAIL: SNDE constructor is not exact-zero initialized")
    snde_zero_diag = zero_snde_diagnostics(snde, stable_val, val_labels, best_baseline["classifier_state"], device)
    if abs(float(best_baseline["val_bce"]) - snde_zero_diag["baseline_val_bce"]) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: SNDE zero BCE differs from selected baseline BCE")
    print(f"[SNDE_ZERO_BASELINE] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch=0 exact_zero=Y val_bce={snde_zero_diag['baseline_val_bce']:.6f} probability_max_abs={snde_zero_diag['zero_probability_max_abs']:.3g}", flush=True)
    optimizer_snde = Adam(snde.parameters(), lr=float(CONFIG["snde_lr"]))

    def snde_checkpoint(epoch: int, progress: dict[str, Any], is_best: bool = False) -> None:
        payload = base_checkpoint_payload(
            task, "snde", epoch, encoder, best_baseline, time.time() - started,
            crsm_branch=state_to_cpu(crsm_branch), normative_state=normative_state_payload(normative), snde_progress=progress,
        )
        if is_best:
            save_checkpoint(best_path, "best", payload)
        if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs_cls"]) - 1:
            save_checkpoint(latest_path, "latest", payload)

    if phase == "snde" and source is not None:
        progress = source["snde_progress"]
        if not snde_is_exact_zero_state(progress["zero_residual_state"]):
            raise RuntimeError("INTEGRITY_FAIL: stored SNDE zero state is not bitwise zero")
        for key, value in progress["zero_diagnostics"].items():
            if abs(float(value) - float(snde_zero_diag[key])) > 1e-6:
                raise RuntimeError(f"INTEGRITY_FAIL: resumed SNDE zero diagnostic changed: {key}")
        load_residual_state(snde, progress["current_residual_state"])
        optimizer_snde.load_state_dict(progress["optimizer_state_dict"])
        optimizer_to_device(optimizer_snde, device)
        best_residual = state_to_cpu(progress["best_residual_state"])
        best_snde_epoch, best_snde_bce = int(progress["best_snde_epoch"]), float(progress["best_val_bce"])
        snde_start_epoch = int(source["epoch"]) + 1
        print(f"SNDE_RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} from_epoch={snde_start_epoch} best_epoch={best_snde_epoch} best_val_bce={best_snde_bce:.6f}", flush=True)
    else:
        best_residual, best_snde_epoch, best_snde_bce, snde_start_epoch = state_to_cpu(zero_snde_state), 0, float(snde_zero_diag["baseline_val_bce"]), 1
        progress = {
            "zero_residual_state": state_to_cpu(zero_snde_state), "zero_diagnostics": state_to_cpu(snde_zero_diag),
            "current_residual_state": snde_state_payload(snde), "optimizer_state_dict": state_to_cpu(optimizer_snde.state_dict()),
            "best_residual_state": state_to_cpu(best_residual), "best_snde_epoch": best_snde_epoch, "best_val_bce": best_snde_bce,
        }
        snde_checkpoint(0, progress, is_best=True)
    for epoch in range(snde_start_epoch, int(CONFIG["epochs_cls"])):
        epoch_started = time.time()
        snde.train()
        optimizer_snde.zero_grad()
        details = snde.forward_details(stable_train)
        train_loss = binary_bce_from_details(details, train_labels)
        train_loss.backward()
        if not bool(torch.isfinite(train_loss)) or not finite_gradients(snde):
            raise FloatingPointError("non-finite SNDE binary BCE or gradient")
        optimizer_snde.step()
        val_bce, _ = snde_validation(snde, stable_val, val_labels)
        is_best = val_bce < best_snde_bce
        if is_best:
            best_snde_bce, best_snde_epoch, best_residual = float(val_bce), int(epoch), snde_state_payload(snde)
        progress = {
            "zero_residual_state": state_to_cpu(zero_snde_state), "zero_diagnostics": state_to_cpu(snde_zero_diag),
            "current_residual_state": snde_state_payload(snde), "optimizer_state_dict": state_to_cpu(optimizer_snde.state_dict()),
            "best_residual_state": state_to_cpu(best_residual), "best_snde_epoch": int(best_snde_epoch), "best_val_bce": float(best_snde_bce),
        }
        if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0 or epoch == int(CONFIG["epochs_cls"]) - 1:
            snde_checkpoint(epoch, progress, is_best=is_best)
        if epoch % 10 == 0 or is_best or epoch == int(CONFIG["epochs_cls"]) - 1:
            diagnostic = snde_diagnostics(snde, stable_reference, reference_labels)
            print(
                f"[SNDE] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs_cls']-1} "
                f"train_bce={float(train_loss.detach().cpu()):.6f} val_bce={val_bce:.6f} best={'Y' if is_best else 'N'} best_epoch={best_snde_epoch} "
                f"asd_gate=(+{diagnostic['snde_asd_positive_fraction']:.3f},-{diagnostic['snde_asd_negative_fraction']:.3f},g{diagnostic['snde_asd_generalist_fraction']:.3f}) "
                f"hc_gate=(+{diagnostic['snde_hc_positive_fraction']:.3f},-{diagnostic['snde_hc_negative_fraction']:.3f},g{diagnostic['snde_hc_generalist_fraction']:.3f}) "
                f"pos_weight_l2={diagnostic['snde_positive_weight_l2']:.6f} neg_weight_l2={diagnostic['snde_negative_weight_l2']:.6f} "
                f"mean_abs_pos_correction={diagnostic['snde_mean_abs_positive_correction']:.6f} "
                f"mean_abs_neg_correction={diagnostic['snde_mean_abs_negative_correction']:.6f} "
                f"elapsed_seconds={time.time()-epoch_started:.2f}", flush=True,
            )
    load_residual_state(snde, best_residual)
    final_snde_bce, final_snde_probs = snde_validation(snde, stable_val, val_labels)
    if abs(final_snde_bce - best_snde_bce) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: selected SNDE state does not reproduce selected validation BCE")
    if best_snde_epoch == 0:
        if not snde_is_exact_zero_state(best_residual):
            raise RuntimeError("INTEGRITY_FAIL: selected SNDE epoch 0 is not bitwise exact zero")
        if float(np.max(np.abs(final_snde_probs - np.asarray(best_baseline["val_probs"], dtype=np.float64)))) > 1e-6:
            raise RuntimeError("INTEGRITY_FAIL: selected SNDE epoch-0 probabilities differ from baseline")
    final_snde_diag = snde_diagnostics(snde, stable_reference, reference_labels)
    snde_branch = {
        "encoder_state": clone_state_cpu(encoder.state_dict()), "baseline_classifier_state": state_to_cpu(best_baseline["classifier_state"]),
        "normative_state": normative_state_payload(normative), "residual_state": state_to_cpu(best_residual), "zero_residual_state": state_to_cpu(zero_snde_state),
        "val_bce": float(best_snde_bce), "encoder_epoch": int(best_baseline["encoder_epoch"]), "classifier_epoch": int(best_baseline["classifier_epoch"]),
        "snde_epoch": int(best_snde_epoch), "threshold": validation_threshold(task.val_labels, final_snde_probs), "val_probs": final_snde_probs,
        "crsm_diagnostics": None, "snde_diagnostics": final_snde_diag,
    }
    elapsed_seconds = float(time.time() - started)
    best_baseline["task_seconds"] = elapsed_seconds
    crsm_branch["task_seconds"] = elapsed_seconds
    snde_branch["task_seconds"] = elapsed_seconds
    result = {"task": task, "baseline": best_baseline, "crsm": crsm_branch, "snde": snde_branch, "mean_ssl_loss": float(np.mean(ssl_losses)) if ssl_losses else float("nan"), "elapsed_seconds": elapsed_seconds}
    assert_shared_selection(result)
    print(
        f"TASK_TRAINING_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} base_encoder_epoch={best_baseline['encoder_epoch']} "
        f"base_classifier_epoch={best_baseline['classifier_epoch']} base_val_bce={best_baseline['val_bce']:.6f} "
        f"crsm_epoch={crsm_branch['crsm_epoch']} crsm_val_bce={crsm_branch['val_bce']:.6f} "
        f"snde_epoch={snde_branch['snde_epoch']} snde_val_bce={snde_branch['val_bce']:.6f} elapsed_minutes={elapsed_seconds/60.0:.1f}", flush=True,
    )
    del encoder, contrast, optimizer_ssl, scheduler, stable_reference, stable_train, stable_val, snde, optimizer_snde
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def crsm_model_from_branch(branch: dict[str, Any], device: torch.device) -> ControlReferencedSoftMultiBoundary:
    model = ControlReferencedSoftMultiBoundary(branch["baseline_classifier_state"], branch["hc_centroid"], temperature=float(CONFIG["temperature"])).to(device)
    load_delta_state(model, branch["delta_state"])
    model.eval()
    return model


def snde_model_from_branch(branch: dict[str, Any], device: torch.device) -> SelectiveNormativeDeviationExperts:
    model = SelectiveNormativeDeviationExperts(
        branch["baseline_classifier_state"], normative_state_from_payload(branch["normative_state"]),
        roi_count=int(CONFIG["snde_roi_count"]), gate_threshold=float(CONFIG["snde_gate_threshold"]),
    ).to(device)
    load_residual_state(model, branch["residual_state"])
    model.eval()
    return model


def evaluate_scope_all_methods(
    cohort: Cohort, task_result: dict[str, Any], data: np.ndarray, raw_indices: np.ndarray,
    labels: np.ndarray, device: torch.device, scope: str,
) -> dict[str, tuple[dict[str, float], np.ndarray]]:
    """Extract one test stable tensor, then apply all three frozen branches."""
    assert_shared_selection(task_result)
    baseline = task_result["baseline"]
    encoder = create_encoder(cohort, device)
    encoder.load_state_dict(baseline["encoder_state"])
    stable = extract_stable_features(encoder, data, raw_indices, device)
    probabilities: dict[str, np.ndarray] = {BASELINE_METHOD: predict_baseline(stable, baseline["classifier_state"], device)}
    crsm = crsm_model_from_branch(task_result["crsm"], device)
    with torch.no_grad():
        probabilities[CRSM_METHOD] = crsm(stable)[:, 1].detach().cpu().numpy().astype(np.float64)
    snde = snde_model_from_branch(task_result["snde"], device)
    with torch.no_grad():
        probabilities[SNDE_METHOD] = snde(stable)[:, 1].detach().cpu().numpy().astype(np.float64)
    if int(task_result["crsm"]["crsm_epoch"]) == 0 and float(np.max(np.abs(probabilities[CRSM_METHOD] - probabilities[BASELINE_METHOD]))) > 1e-6:
        raise RuntimeError(f"INTEGRITY_FAIL: CRSM exact-zero test parity failed on {scope}")
    if int(task_result["snde"]["snde_epoch"]) == 0 and float(np.max(np.abs(probabilities[SNDE_METHOD] - probabilities[BASELINE_METHOD]))) > 1e-6:
        raise RuntimeError(f"INTEGRITY_FAIL: SNDE exact-zero test parity failed on {scope}")
    branches = {BASELINE_METHOD: baseline, CRSM_METHOD: task_result["crsm"], SNDE_METHOD: task_result["snde"]}
    answer = {method: (classification_metrics(labels, p_hc, float(branches[method]["threshold"])), p_hc) for method, p_hc in probabilities.items()}
    del encoder, stable, crsm, snde
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return answer


def diagnostic_fields(method: str, branch: dict[str, Any]) -> dict[str, float]:
    crsm_diag = branch.get("crsm_diagnostics") if method == CRSM_METHOD else None
    snde_diag = branch.get("snde_diagnostics") if method == SNDE_METHOD else None
    if crsm_diag is not None:
        missing = [key for key in CRSM_DIAGNOSTIC_COLUMNS if key not in crsm_diag]
        if missing:
            raise RuntimeError(f"CRSM diagnostics missing fields: {missing}")
    if snde_diag is not None:
        missing = [key for key in SNDE_DIAGNOSTIC_COLUMNS if key not in snde_diag]
        if missing:
            raise RuntimeError(f"SNDE diagnostics missing fields: {missing}")
    return {
        **{key: float(crsm_diag[key]) if crsm_diag is not None else float("nan") for key in CRSM_DIAGNOSTIC_COLUMNS},
        **{key: float(snde_diag[key]) if snde_diag is not None else float("nan") for key in SNDE_DIAGNOSTIC_COLUMNS},
    }


def metric_row(method: str, scope: str, repeat: int, fold: str, metrics: dict[str, float], branch: dict[str, Any], n_test: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": str(method), "scope": str(scope), "repeat": int(repeat), "fold": str(fold),
        "auc": float(metrics["auc"]), "bce": float(metrics["bce"]), "f1": float(metrics["f1"]), "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "selected_encoder_epoch": float(branch["encoder_epoch"]), "selected_classifier_epoch": float(branch.get("classifier_epoch", float("nan"))),
        "selected_crsm_epoch": float(branch.get("crsm_epoch", float("nan"))), "selected_snde_epoch": float(branch.get("snde_epoch", float("nan"))),
        "threshold_hc": float(branch["threshold"]), "task_seconds": float(branch.get("task_seconds", float("nan"))), "n_test": int(n_test), "status": "PASS",
    }
    row.update(diagnostic_fields(method, branch))
    return row


def worker_paths(repeat: int) -> dict[str, Path]:
    prefix = RESULTS_DIR / f".worker_r{int(repeat)}"
    return {
        "csv": prefix.with_suffix(".csv"), "predictions": prefix.with_name(prefix.name + "_predictions.npz"),
        "external": prefix.with_name(prefix.name + "_external.pt"), "done": prefix.with_name(prefix.name + ".done.json"),
        "runtime": prefix.with_name(prefix.name + "_runtime.json"), "error": prefix.with_name(prefix.name + ".error.json"),
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
    return {row["method"] for row in rows if row["scope"] == scope and int(row["repeat"]) == int(repeat) and row["fold"] == str(fold)} == set(METHODS)


def clear_metric_rows(rows: list[dict[str, Any]], scope: str, repeat: int, fold: str) -> list[dict[str, Any]]:
    return [row for row in rows if not (row["scope"] == scope and int(row["repeat"]) == int(repeat) and row["fold"] == str(fold))]


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
            "scope": str(archive["scope"][i]), "method": str(archive["method"][i]), "repeat": int(archive["repeat"][i]), "fold": str(archive["fold"][i]),
            "subject_index": int(archive["subject_index"][i]), "raw_index": int(archive["raw_index"][i]), "y": int(archive["y"][i]),
            "p_hc": float(archive["p_hc"][i]), "threshold": float(archive["threshold"][i]),
        } for i in range(count)]


def write_prediction_records_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "scope": np.asarray([item["scope"] for item in records], dtype="<U24"), "method": np.asarray([item["method"] for item in records], dtype="<U48"),
        "repeat": np.asarray([item["repeat"] for item in records], dtype=np.int16), "fold": np.asarray([item["fold"] for item in records], dtype="<U16"),
        "subject_index": np.asarray([item["subject_index"] for item in records], dtype=np.int32), "raw_index": np.asarray([item["raw_index"] for item in records], dtype=np.int32),
        "y": np.asarray([item["y"] for item in records], dtype=np.int8), "p_hc": np.asarray([item["p_hc"] for item in records], dtype=np.float64),
        "threshold": np.asarray([item["threshold"] for item in records], dtype=np.float64),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clear_prediction_group(records: list[dict[str, Any]], scope: str, repeat: int, fold: str, method: str | None = None) -> list[dict[str, Any]]:
    return [record for record in records if not (record["scope"] == scope and int(record["repeat"]) == int(repeat) and record["fold"] == str(fold) and (method is None or record["method"] == method))]


def add_prediction_group(
    records: list[dict[str, Any]], scope: str, method: str, repeat: int, fold: str,
    subject_indices: np.ndarray, raw_indices: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, threshold: float | np.ndarray,
) -> list[dict[str, Any]]:
    records = clear_prediction_group(records, scope, repeat, fold, method)
    subject_indices, raw_indices, labels, probabilities = (np.asarray(v) for v in (subject_indices, raw_indices, labels, probabilities))
    thresholds = np.full(len(labels), float(threshold), dtype=np.float64) if np.isscalar(threshold) else np.asarray(threshold, dtype=np.float64)
    if not all(len(item) == len(labels) for item in (subject_indices, raw_indices, probabilities, thresholds)):
        raise RuntimeError("prediction group lengths disagree")
    records.extend({
        "scope": str(scope), "method": str(method), "repeat": int(repeat), "fold": str(fold),
        "subject_index": int(subject_indices[i]), "raw_index": int(raw_indices[i]), "y": int(labels[i]), "p_hc": float(probabilities[i]), "threshold": float(thresholds[i]),
    } for i in range(len(labels)))
    return records


def subject_indices_for_task(cohort: Cohort, task: Task) -> np.ndarray:
    lookup = {int(raw): index for index, raw in enumerate(cohort.internal_raw if task.scope == "internal_cv" else cohort.caltech_raw)}
    return np.asarray([lookup[int(raw)] for raw in task.test_raw], dtype=np.int64)


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def build_internal_oof_row(cohort: Cohort, records: list[dict[str, Any]], fold_rows: list[dict[str, Any]], method: str, repeat: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = sorted([record for record in records if record["scope"] == "internal_fold" and record["method"] == method and int(record["repeat"]) == int(repeat)], key=lambda record: int(record["subject_index"]))
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
        "encoder_epoch": mean_or_nan([row["selected_encoder_epoch"] for row in source_rows]), "classifier_epoch": mean_or_nan([row["selected_classifier_epoch"] for row in source_rows]),
        "threshold": float("nan"), "task_seconds": mean_or_nan([row["task_seconds"] for row in source_rows]), "crsm_diagnostics": None, "snde_diagnostics": None,
    }
    if method == CRSM_METHOD:
        branch["crsm_epoch"] = mean_or_nan([row["selected_crsm_epoch"] for row in source_rows])
        branch["crsm_diagnostics"] = {key: mean_or_nan([row[key] for row in source_rows]) for key in CRSM_DIAGNOSTIC_COLUMNS}
    if method == SNDE_METHOD:
        branch["snde_epoch"] = mean_or_nan([row["selected_snde_epoch"] for row in source_rows])
        branch["snde_diagnostics"] = {key: mean_or_nan([row[key] for row in source_rows]) for key in SNDE_DIAGNOSTIC_COLUMNS}
    return metric_row(method, "internal_oof", repeat, "all", metrics_with_thresholds(labels, probabilities, thresholds), branch, len(labels)), [{**record, "scope": "internal_oof", "fold": "all"} for record in selected]


def evaluate_internal_task(cohort: Cohort, task_result: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task: Task = task_result["task"]
    evaluations = evaluate_scope_all_methods(cohort, task_result, cohort.abide1, task.test_raw, task.test_labels, device, "internal_fold")
    branches = {BASELINE_METHOD: task_result["baseline"], CRSM_METHOD: task_result["crsm"], SNDE_METHOD: task_result["snde"]}
    indices = subject_indices_for_task(cohort, task)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for method in METHODS:
        metrics, probabilities = evaluations[method]
        branch = branches[method]
        rows.append(metric_row(method, "internal_fold", task.repeat, task.fold, metrics, branch, len(task.test_raw)))
        records = add_prediction_group(records, "internal_fold", method, task.repeat, task.fold, indices, task.test_raw, task.test_labels, probabilities, float(branch["threshold"]))
        print(f"TASK_DONE repeat={task.repeat} scope=internal_fold fold={task.fold} method={method} selected_encoder_epoch={branch['encoder_epoch']} test_auc={metrics['auc']:.6f} test_bce={metrics['bce']:.6f} elapsed_minutes={task_result['elapsed_seconds']/60.0:.1f}", flush=True)
    return rows, records


def evaluate_external_task(cohort: Cohort, task_result: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task: Task = task_result["task"]
    abide2_indices = np.arange(len(cohort.abide2), dtype=np.int64)
    if len(abide2_indices) != 727 or not np.array_equal(abide2_indices, np.arange(727)):
        raise RuntimeError("ABIDE-II must remain the complete original 727-subject cohort")
    specs = (("caltech", cohort.abide1, task.test_raw, task.test_labels, subject_indices_for_task(cohort, task)), ("abide2", cohort.abide2, abide2_indices, cohort.abide2_labels, abide2_indices.copy()))
    branches = {BASELINE_METHOD: task_result["baseline"], CRSM_METHOD: task_result["crsm"], SNDE_METHOD: task_result["snde"]}
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for scope, data, raw_indices, labels, indices in specs:
        evaluations = evaluate_scope_all_methods(cohort, task_result, data, raw_indices, labels, device, scope)
        for method in METHODS:
            metrics, probabilities = evaluations[method]
            branch = branches[method]
            rows.append(metric_row(method, scope, task.repeat, "external", metrics, branch, len(raw_indices)))
            records = add_prediction_group(records, scope, method, task.repeat, "external", indices, raw_indices, labels, probabilities, float(branch["threshold"]))
            print(f"TASK_DONE repeat={task.repeat} scope={scope} fold=external method={method} selected_encoder_epoch={branch['encoder_epoch']} test_auc={metrics['auc']:.6f} test_bce={metrics['bce']:.6f} elapsed_minutes={task_result['elapsed_seconds']/60.0:.1f}", flush=True)
        expected = 37 if scope == "caltech" else 727
        ids: dict[str, list[tuple[int, int]]] = {}
        for method in METHODS:
            values = sorted((r["subject_index"], r["raw_index"]) for r in records if r["scope"] == scope and r["method"] == method)
            if len(values) != expected:
                raise RuntimeError(f"{scope}/{method} count={len(values)} expected={expected}")
            ids[method] = values
        if not (ids[BASELINE_METHOD] == ids[CRSM_METHOD] == ids[SNDE_METHOD]):
            raise RuntimeError(f"INTEGRITY_FAIL: {scope} method subject IDs differ")
    expected_abide2_ids = [(index, index) for index in range(727)]
    for method in METHODS:
        actual = sorted((r["subject_index"], r["raw_index"]) for r in records if r["scope"] == "abide2" and r["method"] == method)
        if actual != expected_abide2_ids:
            raise RuntimeError(f"INTEGRITY_FAIL: ABIDE-II/{method} IDs are not exactly 0..726")
    external_models = {"format_version": 2, "repeat": int(task.repeat), "task_identity": task_identity(task), "model_config": make_model_config(cohort)[0], "baseline": state_to_cpu(task_result["baseline"]), CRSM_METHOD: state_to_cpu(task_result["crsm"]), SNDE_METHOD: state_to_cpu(task_result["snde"])}
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
    if len(selected) != 39:
        return False
    expected: set[tuple[str, str, str]] = set()
    for fold in range(10):
        for method in METHODS:
            expected.add((method, "internal_fold", str(fold)))
    for method in METHODS:
        expected.update({(method, "internal_oof", "all"), (method, "caltech", "external"), (method, "abide2", "external")})
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
    print(f"REPEAT_START repeat={repeat} device={device} tasks=11 methods=3", flush=True)
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
            records = add_prediction_group(records, "internal_fold", method, repeat, task.fold, np.asarray([record["subject_index"] for record in group]), np.asarray([record["raw_index"] for record in group]), np.asarray([record["y"] for record in group]), np.asarray([record["p_hc"] for record in group]), np.asarray([record["threshold"] for record in group]))
        write_prediction_records_atomic(paths["predictions"], records)
        append_metric_rows(paths["csv"], task_rows)
        rows.extend(task_rows)
        update_runtime_task(repeat, f"internal_{task.fold}", task_result["elapsed_seconds"], device)
        clean_task_checkpoints(repeat, task)
    for method in METHODS:
        if any(row["scope"] == "internal_oof" and row["method"] == method and int(row["repeat"]) == int(repeat) for row in rows):
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
                records = add_prediction_group(records, scope, method, repeat, "external", np.asarray([record["subject_index"] for record in group]), np.asarray([record["raw_index"] for record in group]), np.asarray([record["y"] for record in group]), np.asarray([record["p_hc"] for record in group]), np.asarray([record["threshold"] for record in group]))
        atomic_torch_save(paths["external"], external_models)
        write_prediction_records_atomic(paths["predictions"], records)
        append_metric_rows(paths["csv"], external_rows)
        rows.extend(external_rows)
        update_runtime_task(repeat, "external", task_result["elapsed_seconds"], device)
        clean_task_checkpoints(repeat, external_task)
    if not expected_repeat_rows(rows, repeat):
        raise RuntimeError(f"repeat={repeat} incomplete: expected 39 final metric rows")
    write_json_atomic(paths["done"], {"repeat": int(repeat), "status": "PASS", "metrics_rows": 39, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"REPEAT_DONE repeat={repeat} status=PASS rows=39", flush=True)


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
            if not (paired[BASELINE_METHOD] == paired[CRSM_METHOD] == paired[SNDE_METHOD]):
                raise RuntimeError(f"INTEGRITY_FAIL: {scope}/repeat={repeat} paired IDs or labels differ across methods")
    if len(cohort.abide2) != 727:
        raise RuntimeError("INTEGRITY_FAIL: ABIDE-II source cohort changed")


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
    if abide_mean <= -0.005:
        reasons.append("abide2_mean_delta_auc_at_or_below_minus_0.005")
    return {
        "decision": decision, "internal_mean_delta_auc": internal_mean,
        "internal_positive_repeats": positive, "internal_mean_delta_bce": internal_bce_mean,
        "abide2_mean_delta_auc": abide_mean, "abide2_mean_delta_bce": abide_bce_mean,
        "failure_indicators": reasons,
    }


def diagnostic_summary(rows: list[dict[str, Any]], method: str, columns: tuple[str, ...]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["method"] == method and row["scope"] in {"internal_fold", "caltech"}
    ]
    if len(selected) != 110:
        raise RuntimeError(f"{method} diagnostic rows={len(selected)}, expected 110")
    output: dict[str, Any] = {"task_count": len(selected)}
    for key in columns:
        values = [float(row[key]) for row in selected if np.isfinite(float(row[key]))]
        output[f"{key}_mean"] = mean_or_nan(values)
        output[f"{key}_min"] = float(min(values)) if values else float("nan")
        output[f"{key}_max"] = float(max(values)) if values else float("nan")
    return output


def aggregate_rows(rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], source_hashes: dict[str, str]) -> dict[str, Any]:
    scopes = ("internal_oof", "caltech", "abide2")
    aggregates = {f"{method}::{scope}": _scope_aggregate(rows, method, scope) for method in METHODS for scope in scopes}
    deltas_vs_baseline = {
        method: {scope: paired_delta(rows, method, BASELINE_METHOD, scope) for scope in scopes}
        for method in (CRSM_METHOD, SNDE_METHOD)
    }
    snde_vs_crsm = {scope: paired_delta(rows, SNDE_METHOD, CRSM_METHOD, scope) for scope in scopes}
    crsm_decision = decision_from_deltas(deltas_vs_baseline[CRSM_METHOD])
    snde_decision = decision_from_deltas(deltas_vs_baseline[SNDE_METHOD])
    baseline_internal = float(aggregates[f"{BASELINE_METHOD}::internal_oof"]["auc_mean"])
    baseline_abide2 = float(aggregates[f"{BASELINE_METHOD}::abide2"]["auc_mean"])
    pipeline_drift = (
        not np.isfinite(baseline_internal) or not np.isfinite(baseline_abide2)
        or abs(baseline_internal - 0.723) > 0.015 or abs(baseline_abide2 - 0.739) > 0.015
    )
    if pipeline_drift:
        crsm_decision["decision"] = "PIPELINE_DRIFT"
        snde_decision["decision"] = "PIPELINE_DRIFT"
    crsm_summary = diagnostic_summary(rows, CRSM_METHOD, CRSM_DIAGNOSTIC_COLUMNS)
    crsm_task_rows = [row for row in rows if row["method"] == CRSM_METHOD and row["scope"] in {"internal_fold", "caltech"}]
    selected_epochs = [int(round(float(row["selected_crsm_epoch"]))) for row in crsm_task_rows]
    zero_count = int(sum(epoch == 0 for epoch in selected_epochs))
    crsm_summary.update({
        "selected_crsm_epoch_distribution": {str(epoch): selected_epochs.count(epoch) for epoch in sorted(set(selected_epochs))},
        "corrected_exact_zero_selection_count": zero_count,
        "corrected_exact_zero_selection_fraction": float(zero_count / len(selected_epochs)),
        "old_post_pca_epoch0_count": 91,
        "old_post_pca_epoch0_fraction": float(91 / 110),
        "old_epoch0_was_exact_zero": False,
    })
    snde_summary = diagnostic_summary(rows, SNDE_METHOD, SNDE_DIAGNOSTIC_COLUMNS)
    snde_task_rows = [row for row in rows if row["method"] == SNDE_METHOD and row["scope"] in {"internal_fold", "caltech"}]
    positive_activated = any(
        float(row["snde_asd_positive_fraction"]) > 0.0 or float(row["snde_hc_positive_fraction"]) > 0.0
        for row in snde_task_rows
    )
    negative_activated = any(
        float(row["snde_asd_negative_fraction"]) > 0.0 or float(row["snde_hc_negative_fraction"]) > 0.0
        for row in snde_task_rows
    )
    mechanism_collapse = not positive_activated and not negative_activated
    selected_snde_epochs = [int(round(float(row["selected_snde_epoch"]))) for row in snde_task_rows]
    snde_summary.update({
        "selected_snde_epoch_distribution": {str(epoch): selected_snde_epochs.count(epoch) for epoch in sorted(set(selected_snde_epochs))},
        "selected_snde_exact_zero_count": int(sum(epoch == 0 for epoch in selected_snde_epochs)),
        "mechanism_status": "MECHANISM_COLLAPSE" if mechanism_collapse else "ACTIVE",
        "positive_gate_activated_any_task": bool(positive_activated),
        "negative_gate_activated_any_task": bool(negative_activated),
    })
    task_seconds = [float(sum(float(value) for value in row.get("completed_tasks", {}).values())) for row in runtime_rows]
    worker_wall = max([float(row.get("worker_wall_seconds", 0.0)) for row in runtime_rows] or [0.0])
    return {
        "status": "PASS", "config": CONFIG,
        "dataset": {
            "atlas": "AICHA", "abide1_raw": 995, "abide1_internal_unique_names": 882,
            "caltech_external": 37, "abide1_duplicate_train_occurrences": 76,
            "abide2_external": 727, "abide2_original_n": 727, "abide2_evaluable_n": 727,
            "abide2_excluded_n": 0,
        },
        "abide2_original_n": 727, "abide2_evaluable_n": 727, "abide2_excluded_n": 0,
        "abide2_excluded_subject": None, "abide2_excluded_index": None, "exclusion_reason": None,
        "aggregates": aggregates, "deltas_vs_baseline": deltas_vs_baseline,
        "snde_minus_corrected_crsm": snde_vs_crsm,
        "baseline_sanity": {
            "internal_auc_reference": 0.723, "abide2_auc_reference": 0.739,
            "internal_auc": baseline_internal, "abide2_auc": baseline_abide2,
            "pipeline_drift": bool(pipeline_drift),
        },
        "decision": snde_decision["decision"], "snde_decision": snde_decision,
        "crsm_corrected_decision": crsm_decision,
        "crsm_corrected_diagnostics": crsm_summary, "snde_diagnostics": snde_summary,
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
            "test_access_per_epoch": False, "full_abide2_paired_subject_ids_equal": True,
            "crsm_trainable_parameters": 147074, "snde_trainable_parameters": 770,
            "no_ddp_dataparallel_amp_compile": True,
        },
        "literature_audit": {
            "hydra": "Common-control/multiple-face rationale only; no HYDRA SVM, hard assignment, consensus, DPP, or source code was copied.",
            "normative_deviation": "Control-reference regional deviation rationale only; SNDE is a locked 384-profile, one-axis, two-gate residual readout.",
            "continuous_assignment": "Continuous gating rationale only; no subtype labels or biological subtype claims are made.",
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
        base_auc, crsm_auc, snde_auc = (value(method, scope, "auc_mean") for method in METHODS)
        base_bce, crsm_bce, snde_bce = (value(method, scope, "bce_mean") for method in METHODS)
        comparison_rows.append(
            f"| {display} | {fmt(base_auc)} | {fmt(crsm_auc)} | {fmt(snde_auc)} | {fmt(crsm_auc-base_auc)} | {fmt(snde_auc-base_auc)} | {fmt(base_bce)} | {fmt(crsm_bce)} | {fmt(snde_bce)} |"
        )
    cross_rows: list[str] = []
    for display, scope in (("Internal OOF", "internal_oof"), ("Caltech", "caltech"), ("ABIDE-II", "abide2")):
        delta = summary["snde_minus_corrected_crsm"][scope]
        cross_rows.append(f"| {display} | {fmt(np.mean(delta['auc']))} | {fmt(np.mean(delta['bce']))} |")
    crsm = summary["crsm_corrected_diagnostics"]
    snde = summary["snde_diagnostics"]
    snde_decision = summary["snde_decision"]
    crsm_decision = summary["crsm_corrected_decision"]
    failure_interpretation = (
        "The negative result indicates that even after correcting the original CRSM baseline fallback, moving heterogeneity modeling from full edge-level boundaries to selective control-referenced regional deviation experts does not yield stable incremental ASD classification."
        if summary["decision"] == "FAIL" else
        "The pre-specified decision rule did not classify the SNDE result as a negative result. This remains a classification-readout result, not a biological subtype claim."
    )
    report = f"""# Experiment 1B-R + 1C

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
{chr(10).join(comparison_rows)}

### SNDE minus corrected CRSM

| Scope | ΔAUC | ΔBCE |
|---|---:|---:|
{chr(10).join(cross_rows)}

Baseline sanity: internal OOF AUC={fmt(summary['baseline_sanity']['internal_auc'])} (reference 0.723), ABIDE-II AUC={fmt(summary['baseline_sanity']['abide2_auc'])} (reference 0.739), pipeline drift={summary['baseline_sanity']['pipeline_drift']}.

## Required audit answers

1. **True baseline epoch-0 effect/result:** corrected CRSM epoch 0 is the actual bitwise-zero residual and exactly reproduces baseline predictions; it is no longer the old PCA-perturbed state.
2. **Old 82.7% comparison:** the previous report's `91/110` (`82.7%`) “epoch 0” selections referred to a post-PCA perturbation. In this corrected run, exact-zero CRSM was selected in `{crsm['corrected_exact_zero_selection_count']}/110` tasks (`{fmt(crsm['corrected_exact_zero_selection_fraction'])}`).
3. **SNDE ASD gate fractions:** positive={fmt(snde['snde_asd_positive_fraction_mean'])}, negative={fmt(snde['snde_asd_negative_fraction_mean'])}, generalist={fmt(snde['snde_asd_generalist_fraction_mean'])} (means across 110 selected internal-fold/Caltech tasks).
4. **HC extremes versus ASD:** HC positive/negative/generalist fractions are {fmt(snde['snde_hc_positive_fraction_mean'])}/{fmt(snde['snde_hc_negative_fraction_mean'])}/{fmt(snde['snde_hc_generalist_fraction_mean'])}, versus ASD {fmt(snde['snde_asd_positive_fraction_mean'])}/{fmt(snde['snde_asd_negative_fraction_mean'])}/{fmt(snde['snde_asd_generalist_fraction_mean'])}. These are gate diagnostics, not subtype labels.
5. **Overfit/collapse control:** SNDE uses 770 parameters rather than edge-level residuals, reference-only HC norming, a single fixed PCA axis, no sweep, and validation-only selection. Its aggregate mechanism status is `{snde['mechanism_status']}`; positive gate activated in any task={snde['positive_gate_activated_any_task']}, negative gate activated in any task={snde['negative_gate_activated_any_task']}; mean selected parameter L2={fmt(snde['snde_parameter_l2_mean'], 6)} and mean absolute correction={fmt(snde['snde_mean_abs_correction_mean'], 6)}.
6. **Actual ABIDE-II improvement:** SNDE − baseline mean ABIDE-II ΔAUC={fmt(snde_decision['abide2_mean_delta_auc'], 6)} and ΔBCE={fmt(snde_decision['abide2_mean_delta_bce'], 6)}. Corrected CRSM − baseline ABIDE-II ΔAUC={fmt(crsm_decision['abide2_mean_delta_auc'], 6)}.

Corrected CRSM selection distribution: `{json.dumps(crsm['selected_crsm_epoch_distribution'], sort_keys=True)}`. SNDE selection distribution: `{json.dumps(snde['selected_snde_epoch_distribution'], sort_keys=True)}`. SNDE PC1 explained-variance fraction mean={fmt(snde['snde_pc1_explained_variance_fraction_mean'], 6)}; HC std floor mean={fmt(snde['snde_std_floor_mean'], 8)}.

## Decisions

- SNDE vs baseline: **{summary['decision']}** (internal ΔAUC={fmt(snde_decision['internal_mean_delta_auc'], 6)}, positive internal repeats={snde_decision['internal_positive_repeats']}/10, ABIDE-II ΔAUC={fmt(snde_decision['abide2_mean_delta_auc'], 6)}, internal/ABIDE-II ΔBCE={fmt(snde_decision['internal_mean_delta_bce'], 6)}/{fmt(snde_decision['abide2_mean_delta_bce'], 6)}).
- Corrected CRSM vs baseline: **{crsm_decision['decision']}** (internal ΔAUC={fmt(crsm_decision['internal_mean_delta_auc'], 6)}, positive internal repeats={crsm_decision['internal_positive_repeats']}/10, ABIDE-II ΔAUC={fmt(crsm_decision['abide2_mean_delta_auc'], 6)}).

{failure_interpretation}

## Integrity, source, and runtime audit

The source hash guard was checked before smoke and after aggregation for root `model_scripts`, root utility/classification scripts, and Experiment 1/1B/2/2B/3 directories; post-run status is `{summary['integrity']['root_and_prior_experiments_unchanged']}`. A real-data smoke test checked root/local encoder parity, stable shape, CRSM zero parity before PCA, nonzero PCA symmetry state, profile reconstruction, normative HC normalization, ASD centering, unit/canonical axis, SNDE zero parity, finite forward/backward, gate fractions, and exact parameter counts.

Workers run full precision with repeats 0/2/4/6/8 on GPU0 and 1/3/5/7/9 on GPU1, six CPU threads each. Wall clock={fmt(summary['runtime']['wall_clock_seconds']/3600.0, 3)} h; summed task compute={fmt(summary['runtime']['worker_seconds']/3600.0, 3)} h; mean peak GPU allocated/reserved={fmt(summary['runtime']['mean_peak_gpu_alloc_gb'], 3)}/{fmt(summary['runtime']['mean_peak_gpu_reserved_gb'], 3)} GB.

## Artifacts

`results/metrics.csv` contains 390 rows (100 internal folds + 10 internal OOF + 10 Caltech + 10 ABIDE-II for each of three methods). `results/predictions.npz` contains paired selected-scope subject IDs, labels, probabilities, and thresholds. `results/summary.json` holds all aggregate, decision, diagnostic, and integrity fields. `results/external_best_models.pt` holds the ten selected external states. Rolling checkpoints are resume-only and are removed after successful aggregation; the selected external weight artifact remains.

This experiment does not identify, assign, or claim biological ASD subtypes; its regional gate diagnostics are technical properties of a pre-specified classification readout.
"""
    (EXPERIMENT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def smoke_marker_path() -> Path:
    return RESULTS_DIR / ".smoke_pass.json"


def smoke_test(device: torch.device) -> None:
    """Run one real-data smoke test before full workers start."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    marker = smoke_marker_path()
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") == "PASS" and int(payload.get("format_version", 0)) == 2:
            print("SMOKE_ALREADY_PASS " + json.dumps(payload, ensure_ascii=False), flush=True)
            return
        raise RuntimeError(f"existing smoke marker is not a valid Experiment 1C PASS marker: {marker}")
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
    asd_positions, hc_positions = np.where(task.reference_labels == 0)[0][:16], np.where(task.reference_labels == 1)[0][:16]
    if len(asd_positions) < 2 or len(hc_positions) < 2:
        raise RuntimeError("smoke reference selection lacks both classes")
    positions = np.concatenate((asd_positions, hc_positions)).astype(np.int64)
    raw = task.reference_raw[positions]
    labels = torch.as_tensor(task.reference_labels[positions], dtype=torch.long, device=device)
    x = torch.from_numpy(np.asarray(cohort.abide1[raw], dtype=np.float32)).to(device)
    local.eval()
    root.eval()
    with torch.no_grad():
        local_stable, root_stable = local(x), root(x)
    root_parity = float((local_stable - root_stable).abs().max().cpu())
    if root_parity > 1e-6 or tuple(local_stable.shape) != (len(raw), int(CONFIG["stable_dim"])):
        raise RuntimeError(f"root/local stable parity or shape failed: parity={root_parity} shape={tuple(local_stable.shape)}")
    set_all_seeds(778)
    baseline = MLP(int(CONFIG["stable_dim"]), 2).to(device)
    baseline_state = clone_state_cpu(baseline.state_dict())
    hc_centroid = local_stable[labels == 1].mean(dim=0)
    crsm = ControlReferencedSoftMultiBoundary(baseline_state, hc_centroid, temperature=1.0).to(device)
    zero_crsm_state = crsm_state_payload(crsm)
    if not crsm_is_exact_zero_state(zero_crsm_state):
        raise RuntimeError("smoke CRSM is not exact zero before PCA")
    zero_crsm = zero_crsm_diagnostics(crsm, local_stable, labels, baseline_state, device)
    if crsm_parameter_count(crsm) != 147074:
        raise RuntimeError("smoke CRSM parameter count drifted")
    initialization = initialize_symmetric_pca(crsm, local_stable, labels, seed=779, ratio=0.01)
    if abs(initialization.pca_unit_norm - 1.0) > 1e-5 or crsm_is_exact_zero_state(crsm_state_payload(crsm)):
        raise RuntimeError("smoke CRSM PCA state failed")
    crsm_loss(crsm.forward_details(local_stable), labels).backward()
    if not finite_gradients(crsm):
        raise RuntimeError("smoke CRSM backward failed")
    profile = ROIConnectivityProfile(roi_count).to(device)
    computed_profile = profile(local_stable)
    manual_matrix = local_stable.new_zeros((len(raw), roi_count, roi_count))
    manual_matrix[:, profile.edge_i, profile.edge_j] = local_stable
    manual_matrix[:, profile.edge_j, profile.edge_i] = local_stable
    profile_parity = float((computed_profile - manual_matrix.sum(dim=2) / float(roi_count - 1)).abs().max().detach().cpu())
    if profile_parity > 1e-6:
        raise RuntimeError(f"smoke ROI profile reconstruction failed: {profile_parity}")
    normative = fit_normative_heterogeneity_axis(local_stable, labels, seed=780, roi_count=roi_count)
    z = (computed_profile - normative.mu.to(device)) / normative.sigma.to(device)
    heterogeneity = z[labels == 0] - normative.asd_mean.to(device)
    asd_center_abs = float(heterogeneity.mean(dim=0).abs().max().detach().cpu())
    axis = normative.axis.to(device)
    hc_raw = z[labels == 1] @ axis
    hc_score = (hc_raw - normative.hc_score_mean) / normative.hc_score_std
    hc_mean_abs, hc_std_abs = float(hc_score.mean().abs().detach().cpu()), float((hc_score.std(unbiased=False) - 1.0).abs().detach().cpu())
    pivot = int(torch.argmax(axis.abs()).detach().cpu())
    if asd_center_abs > 1e-5 or abs(float(torch.linalg.vector_norm(axis).detach().cpu()) - 1.0) > 1e-5 or float(axis[pivot].detach().cpu()) < 0.0 or hc_mean_abs > 1e-5 or hc_std_abs > 1e-5:
        raise RuntimeError("smoke normative state centering/axis/HC normalization failed")
    snde = SelectiveNormativeDeviationExperts(baseline_state, normative, roi_count=roi_count, gate_threshold=1.0).to(device)
    zero_snde_state = snde_state_payload(snde)
    if not snde_is_exact_zero_state(zero_snde_state) or snde_parameter_count(snde) != 770:
        raise RuntimeError("smoke SNDE zero state or parameter count failed")
    zero_snde = zero_snde_diagnostics(snde, local_stable, labels, baseline_state, device)
    snde_loss = binary_bce_from_details(snde.forward_details(local_stable), labels)
    snde_loss.backward()
    if not bool(torch.isfinite(snde_loss)) or not finite_gradients(snde):
        raise RuntimeError("smoke SNDE forward/backward failed")
    gate_diagnostic = snde_diagnostics(snde, local_stable, labels)
    for prefix in ("asd", "hc"):
        total = sum(gate_diagnostic[f"snde_{prefix}_{name}_fraction"] for name in ("positive", "negative", "generalist"))
        if abs(total - 1.0) > 1e-6:
            raise RuntimeError("smoke SNDE gate fractions do not sum to one")
    payload = {
        "format_version": 2, "status": "PASS", "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_baseline_forward_max_abs": root_parity, "stable_shape": list(local_stable.shape),
        "crsm_zero_diagnostics": zero_crsm, "crsm_pca_diagnostics": initialization.as_dict(),
        "roi_profile_max_abs": profile_parity, "asd_center_max_abs": asd_center_abs,
        "hc_score_mean_abs": hc_mean_abs, "hc_score_std_minus_one_abs": hc_std_abs,
        "snde_zero_diagnostics": zero_snde, "crsm_trainable_parameters": 147074,
        "snde_trainable_parameters": 770, "source_hashes": source_hashes,
    }
    write_json_atomic(marker, payload)
    print("SMOKE_PASS " + json.dumps(payload, ensure_ascii=False), flush=True)
    del local, root, baseline, crsm, snde, x, local_stable, root_stable, manual_matrix
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate_main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if (RESULTS_DIR / "summary.json").exists() and not list(RESULTS_DIR.glob(".worker_r*.csv")):
        existing = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
        if existing.get("completion", {}).get("metrics_rows") == 390:
            print(json.dumps({"status": "PASS", "rows": 390, "decision": existing.get("decision"), "already_aggregated": True}), flush=True)
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
    if len(all_rows) != 390:
        raise RuntimeError(f"expected 390 metrics rows, found {len(all_rows)}")
    if any(row["status"] != "PASS" for row in all_rows):
        raise RuntimeError("cannot aggregate non-PASS metric rows")
    validate_prediction_records(all_records, cohort)
    source_hashes = assert_external_integrity()
    write_metric_shard_atomic(RESULTS_DIR / "metrics.csv", all_rows)
    write_final_predictions(RESULTS_DIR / "predictions.npz", all_records)
    atomic_torch_save(RESULTS_DIR / "external_best_models.pt", {"format_version": 2, "atlas": "AICHA", "config": state_to_cpu(CONFIG), "repeats": external_models})
    summary = aggregate_rows(all_rows, runtime_rows, source_hashes)
    summary["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(RESULTS_DIR / "summary.json", summary)
    write_report(summary, all_rows)
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
        # This PPU/CUDA build accepts the current-device form but rejects a
        # torch.device argument.  CUDA_VISIBLE_DEVICES pins each worker to its
        # documented physical GPU, so the no-argument form is unambiguous.
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
            "repeat": repeat, "status": "PASS", "device": str(device), "worker_wall_seconds": time.time() - started,
            "peak_gpu_alloc_gb": peak_alloc, "peak_gpu_reserved_gb": peak_reserved,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        paths["error"].unlink(missing_ok=True)
        write_json_atomic(paths["runtime"], runtime)
    except Exception as exc:
        write_json_atomic(paths["error"], {"repeat": repeat, "status": "FAIL", "error": repr(exc), "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
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

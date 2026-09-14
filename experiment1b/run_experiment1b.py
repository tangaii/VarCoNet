#!/usr/bin/env python3
"""Experiment 1B: Control-Referenced Soft Multi-Boundary VarCoNet.

This is intentionally self-contained.  It copies the audited VarCoNet source
into this directory and uses the established Experiment 3 execution pattern,
but it does not import any prior experiment at runtime.  The only treatment is
a frozen-baseline, two-boundary readout fitted after the baseline trajectory is
selected by validation BCE.
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

from model_scripts.VarCoNet import VarCoNet  # noqa: E402
from model_scripts.classifier import MLP  # noqa: E402
from model_scripts.scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from soft_multiboundary import (  # noqa: E402
    ControlReferencedSoftMultiBoundary,
    MechanismInitError,
    crsm_loss,
    diagnostics_from_model,
    initialize_symmetric_pca,
    load_delta_state,
    margin_from_baseline_state,
    state_payload,
    trainable_parameter_count,
)
from utils import DualBranchContrast, InfoNCE, augment, removeDuplicates  # noqa: E402


CONFIG: dict[str, Any] = {
    "atlas": "AICHA",
    "batch_size": 64,
    "min_length": 80,
    "epochs": 50,
    "warm_up_epochs": 10,
    "epochs_cls": 150,  # range(1, 150): exactly 149 updates
    "lr_cls": 5e-5,
    "num_classes": 2,
    "checkpoint_interval": 10,
    "cpu_threads": 6,
    "n_experts": 2,
    "temperature": 1.0,
    "symmetry_ratio": 0.01,
    "stable_dim": 73536,
}

BASELINE_METHOD = "baseline"
TREATMENT_METHOD = "control_referenced_soft_multiboundary"
METHODS = (BASELINE_METHOD, TREATMENT_METHOD)

CRSM_DIAGNOSTIC_COLUMNS = (
    "n_experts",
    "temperature",
    "hc_centroid_norm",
    "pca_unit_norm",
    "pca_singular_value",
    "sigma_margin",
    "sigma_projection",
    "symmetry_ratio",
    "symmetry_scale",
    "initial_delta_l2",
    "expert_1_mean_mass",
    "expert_2_mean_mass",
    "expert_1_hard_fraction",
    "expert_2_hard_fraction",
    "responsibility_entropy",
    "delta_weight_1_l2",
    "delta_weight_2_l2",
    "delta_weight_cosine",
    "delta_bias_1",
    "delta_bias_2",
)

METRIC_COLUMNS = [
    "method",
    "scope",
    "repeat",
    "fold",
    "auc",
    "bce",
    "f1",
    "balanced_accuracy",
    "selected_encoder_epoch",
    "selected_classifier_epoch",
    "selected_crsm_epoch",
    "threshold_hc",
    *CRSM_DIAGNOSTIC_COLUMNS,
    "task_seconds",
    "n_test",
    "status",
]

FLOAT_COLUMNS = {
    "auc", "bce", "f1", "balanced_accuracy", "selected_encoder_epoch",
    "selected_classifier_epoch", "selected_crsm_epoch", "threshold_hc",
    *CRSM_DIAGNOSTIC_COLUMNS, "task_seconds",
}
INT_COLUMNS = {"repeat", "n_test"}
PREDICTION_FIELDS = (
    "scope", "method", "repeat", "fold", "subject_index", "raw_index", "y", "p_hc", "threshold"
)

# These were recorded before Experiment 1B creation.  With no repository-level
# git metadata, they provide a concrete integrity guard for all previously
# completed experiments and the audited root source used by this runner.
EXPECTED_SOURCE_DIGESTS = {
    "model_scripts": "fa3af48175c37f0d03eda9863fa9ef1e5735e474cf34fb5f6751bb44733b34b0",
    "xiaolunwen/experiment1": "db87b36cdc5af09afb23701ed2591b6b93b51b7af35872c05fab61703f9e9f9b",
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
    # Unique singleton subjects of the outer train partition only.  This is
    # the sole source for reference centroid and PCA initialization.
    reference_raw: np.ndarray
    reference_labels: np.ndarray
    # SSL and classifier training deliberately retain duplicate occurrences.
    train_raw: np.ndarray
    train_names: list[str]
    train_labels: np.ndarray
    val_raw: np.ndarray
    val_labels: np.ndarray
    test_raw: np.ndarray
    test_labels: np.ndarray


def set_all_seeds(seed: int) -> None:
    """Set every stochastic source used by the audited implementation."""
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
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return state


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
        # Cache/checkpoint/log/result artifacts are not source.  Excluding
        # them also prevents benign interpreter cache regeneration from being
        # mistaken for a change to a completed prior experiment.
        if set(path.relative_to(directory).parts) & {"__pycache__", "logs", "results", "checkpoints", "dataset_cache"}:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines.append(f"{file_hash(path)}  {rel}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def assert_external_integrity() -> dict[str, str]:
    """Assert that only Experiment 1B is being written during this run."""
    observed: dict[str, str] = {}
    for key, expected in EXPECTED_SOURCE_DIGESTS.items():
        value = directory_digest(key) if not key.endswith(".py") else file_hash(REPO_ROOT / key)
        observed[key] = value
        if value != expected:
            raise RuntimeError(
                f"INTEGRITY_FAIL: source outside experiment1b changed: {key} "
                f"expected={expected} observed={value}"
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
    """Reproduce the audited ABIDE-I and ABIDE-II cohort construction."""
    abide1_dir = DATASET_DIR / "ABIDEI"
    abide2_dir = DATASET_DIR / "ABIDEII"
    abide1 = load_npz_stack(abide1_dir / "ABIDEI_nilearn_AICHA.npz")
    abide1_names = load_names(abide1_dir / "ABIDEI_nilearn_names.txt")
    abide1_labels = np.load(abide1_dir / "ABIDEI_nilearn_classes.npy", allow_pickle=False).astype(np.int64)
    if len(abide1) != len(abide1_names) or len(abide1) != len(abide1_labels):
        raise RuntimeError("ABIDE-I data, names, and labels are not aligned")
    if abide1.shape[2] != 384:
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
    if not (len(internal_raw) == 882 and len(caltech_raw) == 37 and len(duplicate_raw) == 76 and len(abide2) == 727):
        raise RuntimeError(
            "audited cohort counts drifted: "
            f"internal={len(internal_raw)} caltech={len(caltech_raw)} duplicates={len(duplicate_raw)} abide2={len(abide2)}"
        )
    return Cohort(
        abide1=abide1, abide1_names=abide1_names, abide1_labels=abide1_labels,
        duplicate_raw=duplicate_raw, duplicate_names=duplicate_names,
        duplicate_labels=abide1_labels[duplicate_raw],
        internal_raw=internal_raw, internal_names=internal_names,
        internal_labels=abide1_labels[internal_raw],
        caltech_raw=caltech_raw, caltech_names=caltech_names,
        caltech_labels=abide1_labels[caltech_raw],
        abide2=abide2, abide2_names=abide2_names, abide2_labels=abide2_labels,
    )


def make_tasks(cohort: Cohort, repeat: int) -> list[Task]:
    """Build 10 internal tasks plus one paired external task for one repeat."""
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
        train_raw = np.concatenate((cohort.duplicate_raw, np.asarray(train_unique_raw, dtype=np.int64)))
        train_labels = np.concatenate((cohort.duplicate_labels, np.asarray(train_unique_y, dtype=np.int64)))
        train_names = cohort.duplicate_names + [raw_to_name[int(raw)] for raw in train_unique_raw]
        tasks.append(Task(
            repeat=repeat, scope="internal_cv", fold=str(fold), seed=100000 + repeat * 100 + fold,
            reference_raw=np.asarray(train_unique_raw, dtype=np.int64), reference_labels=np.asarray(train_unique_y, dtype=np.int64),
            train_raw=train_raw, train_names=train_names, train_labels=train_labels,
            val_raw=np.asarray(val_raw, dtype=np.int64), val_labels=np.asarray(val_y, dtype=np.int64),
            test_raw=np.asarray(cohort.internal_raw[test_idx], dtype=np.int64),
            test_labels=np.asarray(cohort.internal_labels[test_idx], dtype=np.int64),
        ))

    ext_train_unique_raw, ext_val_raw, ext_train_unique_y, ext_val_y, _, _ = train_test_split(
        cohort.internal_raw, cohort.internal_labels, np.arange(len(cohort.internal_raw)), test_size=0.10,
        random_state=42 + repeat, stratify=cohort.internal_labels,
    )
    if np.intersect1d(ext_train_unique_raw, cohort.duplicate_raw).size:
        raise RuntimeError("external reference_raw contains a duplicate occurrence")
    tasks.append(Task(
        repeat=repeat, scope="external", fold="external", seed=200000 + repeat,
        reference_raw=np.asarray(ext_train_unique_raw, dtype=np.int64), reference_labels=np.asarray(ext_train_unique_y, dtype=np.int64),
        train_raw=np.concatenate((cohort.duplicate_raw, np.asarray(ext_train_unique_raw, dtype=np.int64))),
        train_names=cohort.duplicate_names + [raw_to_name[int(raw)] for raw in ext_train_unique_raw],
        train_labels=np.concatenate((cohort.duplicate_labels, np.asarray(ext_train_unique_y, dtype=np.int64))),
        val_raw=np.asarray(ext_val_raw, dtype=np.int64), val_labels=np.asarray(ext_val_y, dtype=np.int64),
        test_raw=np.asarray(cohort.caltech_raw, dtype=np.int64), test_labels=np.asarray(cohort.caltech_labels, dtype=np.int64),
    ))
    return tasks


def paired_batch_schedule(names: list[str], n_items: int, epochs: int, seed: int) -> list[list[list[int]]]:
    """Precompute the source-style shuffled batches and duplicate replacement."""
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
    """Extract stable learned-FC vectors once per batch, without labels."""
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
    if stable.ndim != 2 or stable.shape[1] != int(CONFIG["stable_dim"]) or not bool(torch.isfinite(stable).all()):
        raise RuntimeError(f"stable learned-FC shape/finite integrity failure: {tuple(stable.shape)}")
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


def make_model_config(cohort: Cohort) -> tuple[dict[str, int], dict[str, Any]]:
    with (EXPERIMENT_DIR / "best_params_VarCoNet_AICHA.pkl").open("rb") as handle:
        best_params = pickle.load(handle)
    config = {
        "layers": int(best_params["layers"]),
        "n_heads": int(best_params["n_heads"]),
        "dim_feedforward": int(best_params["dim_feedforward"]),
        "max_length": int(cohort.abide1.shape[1]),
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
    """Exact frozen-stable linear readout: 149 updates and validation BCE selection."""
    set_all_seeds(seed)
    classifier = MLP(int(stable_train.shape[1]), 2).to(device)
    optimizer = Adam(classifier.parameters(), lr=float(CONFIG["lr_cls"]))
    criterion = torch.nn.BCELoss()
    target_train = F.one_hot(train_labels, num_classes=2).float()
    target_val = F.one_hot(val_labels, num_classes=2).float()
    best_bce = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(CONFIG["epochs_cls"])):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(stable_train)
        loss = criterion(output, target_train)
        loss.backward()
        if not bool(torch.isfinite(loss)) or not finite_gradients(classifier):
            raise FloatingPointError("non-finite baseline classifier loss or gradient")
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            val_bce = float(criterion(classifier(stable_val), target_val).detach().cpu())
        if val_bce < best_bce:
            best_bce = val_bce
            best_epoch = epoch
            best_state = clone_state_cpu(classifier.state_dict())
    if best_state is None or best_epoch is None:
        raise RuntimeError("baseline classifier did not select a validation state")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        p_hc = classifier(stable_val)[:, 1].detach().cpu().numpy().astype(np.float64)
    return {
        "classifier_state": best_state,
        "val_bce": float(best_bce),
        "classifier_epoch": int(best_epoch),
        "val_probs": p_hc,
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
    return (
        float(torch.cuda.memory_allocated(device)) / (1024 ** 3),
        float(torch.cuda.memory_reserved(device)) / (1024 ** 3),
    )


def task_identity(task: Task) -> dict[str, Any]:
    return {"repeat": int(task.repeat), "scope": str(task.scope), "fold": str(task.fold), "task_seed": int(task.seed)}


def identity_matches(payload: dict[str, Any], task: Task) -> bool:
    return all(payload.get(key) == value for key, value in task_identity(task).items())


def checkpoint_paths(repeat: int) -> tuple[Path, Path]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    worker = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"r{int(repeat)}")
    return CHECKPOINT_DIR / f"worker_{worker}_latest.pt", CHECKPOINT_DIR / f"worker_{worker}_best.pt"


def checkpoint_rank(payload: dict[str, Any]) -> tuple[int, int]:
    phase_rank = 1 if payload.get("phase") == "crsm" else 0
    return phase_rank, int(payload.get("epoch", -1))


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
        f"CHECKPOINT_{kind.upper()} repeat={payload['repeat']} scope={payload['scope']} "
        f"fold={payload['fold']} phase={payload['phase']} epoch={payload['epoch']} path={path}",
        flush=True,
    )


def clean_task_checkpoints(repeat: int, task: Task) -> None:
    for path in checkpoint_paths(repeat):
        if not path.exists():
            continue
        try:
            if identity_matches(torch_load_cpu(path), task):
                path.unlink(missing_ok=True)
        except Exception:
            path.unlink(missing_ok=True)


def ssl_checkpoint_payload(
    task: Task, epoch: int, encoder: VarCoNet, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler, best_baseline: dict[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "format_version": 1, "phase": "ssl", "saved_unix": time.time(), **task_identity(task),
        "epoch": int(epoch), "epochs": int(CONFIG["epochs"]),
        "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "optimizer_state_dict": state_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": state_to_cpu(scheduler.state_dict()),
        "best_baseline": state_to_cpu(best_baseline),
        "rng_state": state_to_cpu(capture_rng_state()), "elapsed_seconds": float(elapsed_seconds),
    }


def crsm_checkpoint_payload(
    task: Task, epoch: int, encoder: VarCoNet, best_baseline: dict[str, Any],
    hc_centroid: torch.Tensor, init_diagnostics: dict[str, float], zero_diagnostics: dict[str, float],
    current_delta: dict[str, Any], optimizer: torch.optim.Optimizer, best_delta: dict[str, Any],
    best_epoch: int, best_val_bce: float, elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "format_version": 1, "phase": "crsm", "saved_unix": time.time(), **task_identity(task),
        "epoch": int(epoch), "epochs": int(CONFIG["epochs_cls"]),
        "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "best_baseline": state_to_cpu(best_baseline),
        "hc_centroid": hc_centroid.detach().cpu().clone(),
        "init_diagnostics": state_to_cpu(init_diagnostics),
        "zero_diagnostics": state_to_cpu(zero_diagnostics),
        "current_delta_state": state_to_cpu(current_delta),
        "optimizer_state_dict": state_to_cpu(optimizer.state_dict()),
        "best_delta_state": state_to_cpu(best_delta),
        "best_crsm_epoch": int(best_epoch), "best_val_bce": float(best_val_bce),
        "rng_state": state_to_cpu(capture_rng_state()), "elapsed_seconds": float(elapsed_seconds),
    }


def make_baseline_branch(encoder: VarCoNet, encoder_epoch: int, fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()),
        "classifier_state": state_to_cpu(fit["classifier_state"]),
        "val_bce": float(fit["val_bce"]),
        "encoder_epoch": int(encoder_epoch),
        "classifier_epoch": int(fit["classifier_epoch"]),
        "threshold": float(fit["threshold"]),
        "val_probs": np.asarray(fit["val_probs"], dtype=np.float64).copy(),
        "crsm_diagnostics": None,
    }


def make_treatment_branch(
    encoder: VarCoNet, baseline: dict[str, Any], hc_centroid: torch.Tensor,
    delta: dict[str, Any], selected_crsm_epoch: int, val_bce: float,
    val_probs: np.ndarray, threshold: float, diagnostics: dict[str, float],
) -> dict[str, Any]:
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()),
        "baseline_classifier_state": state_to_cpu(baseline["classifier_state"]),
        "hc_centroid": hc_centroid.detach().cpu().clone(),
        "delta_state": state_to_cpu(delta),
        "val_bce": float(val_bce),
        "encoder_epoch": int(baseline["encoder_epoch"]),
        "classifier_epoch": int(baseline["classifier_epoch"]),
        "crsm_epoch": int(selected_crsm_epoch),
        "threshold": float(threshold),
        "val_probs": np.asarray(val_probs, dtype=np.float64).copy(),
        "crsm_diagnostics": {key: float(value) for key, value in diagnostics.items()},
    }


def assert_shared_selection(task_result: dict[str, Any]) -> None:
    baseline = task_result["baseline"]
    treatment = task_result["treatment"]
    if int(baseline["encoder_epoch"]) != int(treatment["encoder_epoch"]):
        raise RuntimeError("INTEGRITY_FAIL: baseline/treatment selected encoder epochs differ")
    if int(baseline["classifier_epoch"]) != int(treatment["classifier_epoch"]):
        raise RuntimeError("INTEGRITY_FAIL: baseline/treatment selected classifier epochs differ")
    for key, left in baseline["encoder_state"].items():
        if key not in treatment["encoder_state"] or not torch.equal(left, treatment["encoder_state"][key]):
            raise RuntimeError("INTEGRITY_FAIL: baseline/treatment encoder states differ")
    for key, left in baseline["classifier_state"].items():
        right = treatment["baseline_classifier_state"].get(key)
        if right is None or not torch.equal(left, right):
            raise RuntimeError("INTEGRITY_FAIL: baseline/treatment classifier states differ")


def zero_delta_diagnostics(
    model: ControlReferencedSoftMultiBoundary, stable_val: torch.Tensor, val_labels: torch.Tensor,
    baseline_state: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, float]:
    """Assert the mandated zero-delta equality before symmetry breaking."""
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
        p_hc_margin = torch.sigmoid(-direct_margin)
        diagnostics = {
            "zero_probability_max_abs": float((baseline_out - zero_out).abs().max().cpu()),
            "zero_margin_max_abs": float((details["m0"] - direct_margin).abs().max().cpu()),
            "zero_p_hc_max_abs": float((details["p_hc"] - p_hc_margin).abs().max().cpu()),
            "zero_bce_difference": float((zero_bce - baseline_bce).abs().cpu()),
            "zero_crsm_bce_difference": float((loss_parity - baseline_bce).abs().cpu()),
            "baseline_val_bce": float(baseline_bce.cpu()),
        }
    if max(
        diagnostics["zero_probability_max_abs"], diagnostics["zero_margin_max_abs"],
        diagnostics["zero_p_hc_max_abs"], diagnostics["zero_bce_difference"],
        diagnostics["zero_crsm_bce_difference"],
    ) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: zero-delta CRSM does not equal frozen baseline")
    return diagnostics


def crsm_validation(
    model: ControlReferencedSoftMultiBoundary, stable_val: torch.Tensor, val_labels: torch.Tensor,
) -> tuple[float, np.ndarray]:
    criterion = torch.nn.BCELoss()
    model.eval()
    with torch.no_grad():
        output = model(stable_val)
        bce = float(criterion(output, F.one_hot(val_labels, num_classes=2).float()).cpu())
        p_hc = output[:, 1].detach().cpu().numpy().astype(np.float64)
    return bce, p_hc


def train_shared_task(cohort: Cohort, task: Task, device: torch.device) -> dict[str, Any]:
    """Run one original SSL trajectory, then one frozen-baseline CRSM readout."""
    started = time.time()
    print(
        f"TASK_START repeat={task.repeat} scope={task.scope} fold={task.fold} seed={task.seed} "
        f"device={device} shared_encoder=Y reference_unique_n={len(task.reference_raw)} "
        f"train_n={len(task.train_raw)} val_n={len(task.val_raw)}",
        flush=True,
    )
    set_all_seeds(task.seed)
    model_config, best_params = make_model_config(cohort)
    encoder = VarCoNet(model_config, int(cohort.abide1.shape[2])).to(device)
    contrast = DualBranchContrast(loss=InfoNCE(tau=float(best_params["tau"])), mode="L2L").to(device)
    optimizer_ssl = Adam(encoder.parameters(), lr=float(best_params["lr"]))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer_ssl, warmup_start_lr=1e-5,
        warmup_epochs=int(CONFIG["warm_up_epochs"]), max_epochs=int(CONFIG["epochs"]),
    )
    schedule = paired_batch_schedule(task.train_names, len(task.train_raw), int(CONFIG["epochs"]), task.seed)
    latest_path, best_path = checkpoint_paths(task.repeat)
    source = load_task_checkpoint(task.repeat, task)
    phase = "ssl"
    start_ssl_epoch = 1
    best_baseline: dict[str, Any] | None = None
    resumed_elapsed = 0.0
    if source is not None:
        phase = str(source.get("phase", "ssl"))
        resumed_elapsed = float(source.get("elapsed_seconds", 0.0))
        started -= resumed_elapsed
        best_baseline = state_to_cpu(source.get("best_baseline"))
        if source.get("rng_state") is not None:
            restore_rng_state(source["rng_state"])
        if phase == "ssl":
            encoder.load_state_dict(source["encoder_state_dict"])
            optimizer_ssl.load_state_dict(source["optimizer_state_dict"])
            optimizer_to_device(optimizer_ssl, device)
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_ssl_epoch = int(source["epoch"]) + 1
        elif phase == "crsm":
            encoder.load_state_dict(source["encoder_state_dict"])
            start_ssl_epoch = int(CONFIG["epochs"]) + 1
        else:
            raise RuntimeError(f"unknown checkpoint phase {phase!r}")
        print(
            f"RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} phase={phase} "
            f"from_epoch={int(source.get('epoch', 0)) + 1} elapsed_seconds={resumed_elapsed:.1f}",
            flush=True,
        )

    train_labels = torch.as_tensor(task.train_labels, dtype=torch.long, device=device)
    val_labels = torch.as_tensor(task.val_labels, dtype=torch.long, device=device)
    ssl_losses: list[float] = []
    for epoch in range(start_ssl_epoch, int(CONFIG["epochs"]) + 1):
        epoch_start = time.time()
        ssl_loss = train_ssl_epoch(
            encoder, contrast, optimizer_ssl, cohort, task, schedule[epoch - 1],
            int(model_config["max_length"]), device,
        )
        scheduler.step()
        ssl_losses.append(ssl_loss)

        stable_train = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
        stable_val = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
        trajectory_rng = capture_rng_state()
        baseline_fit = fit_baseline_classifier(
            stable_train, train_labels, stable_val, val_labels, device,
            task.seed + 300000 + epoch * 2,
        )
        # The classifier's seeded initialization/training must not influence
        # later random SSL crops or the original encoder trajectory.
        restore_rng_state(trajectory_rng)
        candidate = make_baseline_branch(encoder, epoch, baseline_fit)
        is_best = best_baseline is None or float(candidate["val_bce"]) < float(best_baseline["val_bce"])
        if is_best:
            best_baseline = candidate
        elapsed = time.time() - started
        payload = ssl_checkpoint_payload(task, epoch, encoder, optimizer_ssl, scheduler, best_baseline, elapsed)
        if is_best:
            save_checkpoint(best_path, "best", payload)
        if epoch % int(CONFIG["checkpoint_interval"]) == 0:
            save_checkpoint(latest_path, "latest", payload)
        allocated, reserved = gpu_memory_gb(device)
        print(
            f"[SSL] repeat={task.repeat} scope={task.scope} fold={task.fold} epoch={epoch}/{CONFIG['epochs']} "
            f"ssl_loss={ssl_loss:.6f} lr={optimizer_ssl.param_groups[0]['lr']:.9f} "
            f"base_val_bce={baseline_fit['val_bce']:.6f} base_best={'Y' if is_best else 'N'} "
            f"base_cls_epoch={baseline_fit['classifier_epoch']} epoch_seconds={time.time()-epoch_start:.1f} "
            f"task_elapsed_minutes={elapsed/60.0:.1f} gpu_alloc_gb={allocated:.3f} gpu_reserved_gb={reserved:.3f}",
            flush=True,
        )
        del stable_train, stable_val
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_baseline is None:
        raise RuntimeError("no validation-selected baseline branch")

    # From this point onward the original representation is frozen.  The
    # reference extraction explicitly excludes duplicated occurrences, val,
    # internal test, Caltech, ABIDE-II, metadata, and site information.
    encoder.load_state_dict(best_baseline["encoder_state"])
    encoder.eval()
    stable_reference = extract_stable_features(encoder, cohort.abide1, task.reference_raw, device)
    stable_train = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
    stable_val = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
    reference_labels = torch.as_tensor(task.reference_labels, dtype=torch.long, device=device)
    if not torch.equal(reference_labels, torch.as_tensor(cohort.abide1_labels[task.reference_raw], dtype=torch.long, device=device)):
        raise RuntimeError("reference labels are not aligned with the unique outer-train raw indices")
    hc_reference = stable_reference[reference_labels == 1]
    if len(hc_reference) == 0:
        raise MechanismInitError("MECHANISM_INIT_FAIL: no HC singleton reference subjects")
    recomputed_centroid = hc_reference.mean(dim=0)
    if not bool(torch.isfinite(recomputed_centroid).all()):
        raise MechanismInitError("MECHANISM_INIT_FAIL: non-finite HC centroid")

    if phase == "crsm" and source is not None:
        hc_centroid = source["hc_centroid"].to(device)
        if not torch.allclose(hc_centroid, recomputed_centroid, rtol=0.0, atol=1e-6):
            raise RuntimeError("INTEGRITY_FAIL: resumed HC centroid differs from frozen reference centroid")
    else:
        hc_centroid = recomputed_centroid.detach().clone()
    crsm = ControlReferencedSoftMultiBoundary(
        best_baseline["classifier_state"], hc_centroid, temperature=float(CONFIG["temperature"])
    ).to(device)
    expected_parameters = 2 * int(CONFIG["stable_dim"]) + 2
    if trainable_parameter_count(crsm) != expected_parameters:
        raise RuntimeError(
            f"INTEGRITY_FAIL: CRSM trainable count {trainable_parameter_count(crsm)} != {expected_parameters}"
        )
    zero_diag = zero_delta_diagnostics(crsm, stable_val, val_labels, best_baseline["classifier_state"], device)
    if abs(float(best_baseline["val_bce"]) - zero_diag["baseline_val_bce"]) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: zero-delta BCE differs from selected baseline validation BCE")

    optimizer_crsm = Adam(crsm.parameters(), lr=float(CONFIG["lr_cls"]))
    crsm_start_epoch = 1
    if phase == "crsm" and source is not None:
        init_diag = {key: float(value) for key, value in source["init_diagnostics"].items()}
        stored_zero = {key: float(value) for key, value in source["zero_diagnostics"].items()}
        for key, value in stored_zero.items():
            if abs(zero_diag[key] - value) > 1e-6:
                raise RuntimeError(f"INTEGRITY_FAIL: resumed zero-parity diagnostic changed: {key}")
        load_delta_state(crsm, source["current_delta_state"])
        optimizer_crsm.load_state_dict(source["optimizer_state_dict"])
        optimizer_to_device(optimizer_crsm, device)
        best_delta = state_to_cpu(source["best_delta_state"])
        best_crsm_epoch = int(source["best_crsm_epoch"])
        best_crsm_bce = float(source["best_val_bce"])
        crsm_start_epoch = int(source["epoch"]) + 1
        print(
            f"CRSM_RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} "
            f"from_epoch={crsm_start_epoch} best_epoch={best_crsm_epoch} best_val_bce={best_crsm_bce:.6f}",
            flush=True,
        )
    else:
        # PCA is confined by fork_rng; it cannot perturb later task RNG state.
        init = initialize_symmetric_pca(
            crsm, stable_reference, reference_labels, seed=task.seed + 700000,
            ratio=float(CONFIG["symmetry_ratio"]),
        )
        init_diag = init.as_dict()
        if abs(init_diag["pca_unit_norm"] - 1.0) > 1e-5:
            raise MechanismInitError("MECHANISM_INIT_FAIL: PCA direction is not unit norm")
        initial_bce, _ = crsm_validation(crsm, stable_val, val_labels)
        best_delta = state_payload(crsm)
        best_crsm_epoch = 0
        best_crsm_bce = float(initial_bce)
        print(
            f"CRSM_INIT repeat={task.repeat} scope={task.scope} fold={task.fold} "
            f"n_experts=2 tau=1.0 trainable_parameters={expected_parameters} "
            f"hc_centroid_norm={torch.linalg.vector_norm(hc_centroid).item():.6f} "
            f"pca_unit_norm={init_diag['pca_unit_norm']:.8f} pca_singular={init_diag['pca_singular_value']:.6f} "
            f"sigma_margin={init_diag['sigma_margin']:.6f} sigma_projection={init_diag['sigma_projection']:.6f} "
            f"ratio={init_diag['symmetry_ratio']:.4f} scale={init_diag['symmetry_scale']:.9g} "
            f"epoch0_val_bce={initial_bce:.6f}",
            flush=True,
        )
        epoch0_payload = crsm_checkpoint_payload(
            task, 0, encoder, best_baseline, hc_centroid, init_diag, zero_diag,
            state_payload(crsm), optimizer_crsm, best_delta, best_crsm_epoch, best_crsm_bce,
            time.time() - started,
        )
        # Both rolling slots preserve the epoch-0 state and best candidate.
        save_checkpoint(latest_path, "latest", epoch0_payload)
        save_checkpoint(best_path, "best", epoch0_payload)

    for epoch in range(crsm_start_epoch, int(CONFIG["epochs_cls"])):
        epoch_start = time.time()
        crsm.train()
        optimizer_crsm.zero_grad()
        details = crsm.forward_details(stable_train)
        train_loss = crsm_loss(details, train_labels)
        train_loss.backward()
        if not bool(torch.isfinite(train_loss)) or not finite_gradients(crsm):
            raise FloatingPointError("non-finite CRSM loss or gradient")
        optimizer_crsm.step()
        val_bce, _ = crsm_validation(crsm, stable_val, val_labels)
        is_best = val_bce < best_crsm_bce
        if is_best:
            best_crsm_bce = float(val_bce)
            best_crsm_epoch = int(epoch)
            best_delta = state_payload(crsm)
        elapsed = time.time() - started
        if is_best or epoch % int(CONFIG["checkpoint_interval"]) == 0:
            payload = crsm_checkpoint_payload(
                task, epoch, encoder, best_baseline, hc_centroid, init_diag, zero_diag,
                state_payload(crsm), optimizer_crsm, best_delta, best_crsm_epoch, best_crsm_bce, elapsed,
            )
            save_checkpoint(best_path if is_best else latest_path, "best" if is_best else "latest", payload)
        if epoch % 10 == 0 or is_best:
            current_diag = diagnostics_from_model(crsm, stable_reference)
            print(
                f"[CRSM] repeat={task.repeat} scope={task.scope} fold={task.fold} "
                f"epoch={epoch}/{CONFIG['epochs_cls']-1} train_loss={float(train_loss.detach().cpu()):.6f} "
                f"val_bce={val_bce:.6f} best={'Y' if is_best else 'N'} best_epoch={best_crsm_epoch} "
                f"mass=({current_diag['expert_1_mean_mass']:.4f},{current_diag['expert_2_mean_mass']:.4f}) "
                f"hard=({current_diag['expert_1_hard_fraction']:.4f},{current_diag['expert_2_hard_fraction']:.4f}) "
                f"entropy={current_diag['responsibility_entropy']:.6f} "
                f"delta_l2=({current_diag['delta_weight_1_l2']:.6f},{current_diag['delta_weight_2_l2']:.6f}) "
                f"delta_cos={current_diag['delta_weight_cosine']:.6f} epoch_seconds={time.time()-epoch_start:.2f}",
                flush=True,
            )

    load_delta_state(crsm, best_delta)
    final_val_bce, final_val_probs = crsm_validation(crsm, stable_val, val_labels)
    if abs(final_val_bce - best_crsm_bce) > 1e-6:
        raise RuntimeError("INTEGRITY_FAIL: selected CRSM state does not reproduce selected validation BCE")
    final_diag = {**init_diag, **diagnostics_from_model(crsm, stable_reference)}
    treatment_threshold = validation_threshold(task.val_labels, final_val_probs)
    treatment = make_treatment_branch(
        encoder, best_baseline, hc_centroid, best_delta, best_crsm_epoch,
        best_crsm_bce, final_val_probs, treatment_threshold, final_diag,
    )
    treatment["task_seconds"] = float(time.time() - started)
    best_baseline["task_seconds"] = float(time.time() - started)
    result = {
        "task": task, "baseline": best_baseline, "treatment": treatment,
        "mean_ssl_loss": float(np.mean(ssl_losses)) if ssl_losses else float("nan"),
        "elapsed_seconds": float(time.time() - started),
    }
    assert_shared_selection(result)
    print(
        f"TASK_TRAINING_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"base_encoder_epoch={best_baseline['encoder_epoch']} base_classifier_epoch={best_baseline['classifier_epoch']} "
        f"base_val_bce={best_baseline['val_bce']:.6f} selected_crsm_epoch={best_crsm_epoch} "
        f"crsm_val_bce={best_crsm_bce:.6f} treatment_threshold={treatment_threshold:.6f} "
        f"elapsed_minutes={result['elapsed_seconds']/60.0:.1f}",
        flush=True,
    )
    del encoder, contrast, optimizer_ssl, scheduler, stable_reference, stable_train, stable_val, crsm, optimizer_crsm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def treatment_model_from_branch(branch: dict[str, Any], device: torch.device) -> ControlReferencedSoftMultiBoundary:
    model = ControlReferencedSoftMultiBoundary(
        branch["baseline_classifier_state"], branch["hc_centroid"],
        temperature=float(CONFIG["temperature"]),
    ).to(device)
    load_delta_state(model, branch["delta_state"])
    model.eval()
    return model


def evaluate_selected_branch(
    cohort: Cohort, branch: dict[str, Any], method: str, data: np.ndarray,
    raw_indices: np.ndarray, labels: np.ndarray, device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate only a fully validation-selected branch; no per-epoch test access."""
    encoder = create_encoder(cohort, device)
    encoder.load_state_dict(branch["encoder_state"])
    stable = extract_stable_features(encoder, data, raw_indices, device)
    if method == BASELINE_METHOD:
        probabilities = predict_baseline(stable, branch["classifier_state"], device)
    elif method == TREATMENT_METHOD:
        model = treatment_model_from_branch(branch, device)
        with torch.no_grad():
            probabilities = model(stable)[:, 1].detach().cpu().numpy().astype(np.float64)
        del model
    else:
        raise ValueError(f"unknown method {method}")
    metrics = classification_metrics(labels, probabilities, float(branch["threshold"]))
    del stable, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, probabilities


def crsm_metric_fields(method: str, diagnostics: dict[str, Any] | None) -> dict[str, float]:
    if method == BASELINE_METHOD or diagnostics is None:
        return {key: float("nan") for key in CRSM_DIAGNOSTIC_COLUMNS}
    missing = [key for key in CRSM_DIAGNOSTIC_COLUMNS if key not in diagnostics]
    if missing:
        raise RuntimeError(f"CRSM diagnostics missing required fields: {missing}")
    return {key: float(diagnostics[key]) for key in CRSM_DIAGNOSTIC_COLUMNS}


def metric_row(
    method: str, scope: str, repeat: int, fold: str, metrics: dict[str, float],
    branch: dict[str, Any], n_test: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": str(method), "scope": str(scope), "repeat": int(repeat), "fold": str(fold),
        "auc": float(metrics["auc"]), "bce": float(metrics["bce"]), "f1": float(metrics["f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "selected_encoder_epoch": float(branch["encoder_epoch"]),
        "selected_classifier_epoch": float(branch.get("classifier_epoch", float("nan"))),
        "selected_crsm_epoch": float(branch.get("crsm_epoch", float("nan"))),
        "threshold_hc": float(branch["threshold"]),
        "task_seconds": float(branch.get("task_seconds", float("nan"))),
        "n_test": int(n_test), "status": "PASS",
    }
    row.update(crsm_metric_fields(method, branch.get("crsm_diagnostics")))
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
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        if is_new:
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
        return [
            {
                "scope": str(archive["scope"][i]), "method": str(archive["method"][i]),
                "repeat": int(archive["repeat"][i]), "fold": str(archive["fold"][i]),
                "subject_index": int(archive["subject_index"][i]), "raw_index": int(archive["raw_index"][i]),
                "y": int(archive["y"][i]), "p_hc": float(archive["p_hc"][i]),
                "threshold": float(archive["threshold"][i]),
            }
            for i in range(count)
        ]


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
    records: list[dict[str, Any]], scope: str, repeat: int, fold: str, method: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record for record in records
        if not (
            record["scope"] == scope and int(record["repeat"]) == int(repeat) and record["fold"] == str(fold)
            and (method is None or record["method"] == method)
        )
    ]


def add_prediction_group(
    records: list[dict[str, Any]], scope: str, method: str, repeat: int, fold: str,
    subject_indices: np.ndarray, raw_indices: np.ndarray, labels: np.ndarray,
    probabilities: np.ndarray, threshold: float | np.ndarray,
) -> list[dict[str, Any]]:
    records = clear_prediction_group(records, scope, repeat, fold, method)
    subject_indices = np.asarray(subject_indices, dtype=np.int64)
    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.full(len(labels), float(threshold), dtype=np.float64) if np.isscalar(threshold) else np.asarray(threshold, dtype=np.float64)
    if not all(len(values) == len(labels) for values in (subject_indices, raw_indices, probabilities, thresholds)):
        raise RuntimeError("prediction group lengths disagree")
    records.extend({
        "scope": str(scope), "method": str(method), "repeat": int(repeat), "fold": str(fold),
        "subject_index": int(subject_indices[i]), "raw_index": int(raw_indices[i]), "y": int(labels[i]),
        "p_hc": float(probabilities[i]), "threshold": float(thresholds[i]),
    } for i in range(len(labels)))
    return records


def subject_indices_for_task(cohort: Cohort, task: Task) -> np.ndarray:
    if task.scope == "internal_cv":
        lookup = {int(raw): index for index, raw in enumerate(cohort.internal_raw)}
    elif task.scope == "external":
        lookup = {int(raw): index for index, raw in enumerate(cohort.caltech_raw)}
    else:
        raise ValueError(task.scope)
    return np.asarray([lookup[int(raw)] for raw in task.test_raw], dtype=np.int64)


def metrics_with_thresholds(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pred_hc = (probabilities >= thresholds).astype(np.int64)
    asd = (labels == 0).astype(np.int64)
    pred_asd = (pred_hc == 0).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "bce": float(log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "f1": float(f1_score(asd, pred_asd, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred_hc)),
    }


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
        raise RuntimeError(f"repeat={repeat} method={method} internal OOF count={len(selected)} expected={len(cohort.internal_raw)}")
    indices = np.asarray([record["subject_index"] for record in selected], dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(cohort.internal_raw), dtype=np.int64)):
        raise RuntimeError(f"repeat={repeat} method={method} internal OOF IDs are not an exact partition")
    labels = np.asarray([record["y"] for record in selected], dtype=np.int64)
    probabilities = np.asarray([record["p_hc"] for record in selected], dtype=np.float64)
    thresholds = np.asarray([record["threshold"] for record in selected], dtype=np.float64)
    rows = [
        row for row in fold_rows
        if row["scope"] == "internal_fold" and row["method"] == method and int(row["repeat"]) == int(repeat)
    ]
    if len(rows) != 10:
        raise RuntimeError(f"repeat={repeat} method={method} internal fold rows={len(rows)} expected=10")
    branch: dict[str, Any] = {
        "encoder_epoch": mean_or_nan([row["selected_encoder_epoch"] for row in rows]),
        "classifier_epoch": mean_or_nan([row["selected_classifier_epoch"] for row in rows]),
        "threshold": float("nan"), "task_seconds": mean_or_nan([row["task_seconds"] for row in rows]),
        "crsm_diagnostics": None,
    }
    if method == TREATMENT_METHOD:
        branch["crsm_epoch"] = mean_or_nan([row["selected_crsm_epoch"] for row in rows])
        branch["crsm_diagnostics"] = {
            key: mean_or_nan([row[key] for row in rows]) for key in CRSM_DIAGNOSTIC_COLUMNS
        }
    return (
        metric_row(method, "internal_oof", repeat, "all", metrics_with_thresholds(labels, probabilities, thresholds), branch, len(labels)),
        [{**record, "scope": "internal_oof", "fold": "all"} for record in selected],
    )


def evaluate_internal_task(
    cohort: Cohort, task_result: dict[str, Any], device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert_shared_selection(task_result)
    task: Task = task_result["task"]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    subject_indices = subject_indices_for_task(cohort, task)
    for method, branch_key in ((BASELINE_METHOD, "baseline"), (TREATMENT_METHOD, "treatment")):
        branch = task_result[branch_key]
        metrics, probabilities = evaluate_selected_branch(
            cohort, branch, method, cohort.abide1, task.test_raw, task.test_labels, device
        )
        rows.append(metric_row(method, "internal_fold", task.repeat, task.fold, metrics, branch, len(task.test_raw)))
        records = add_prediction_group(
            records, "internal_fold", method, task.repeat, task.fold, subject_indices, task.test_raw,
            task.test_labels, probabilities, float(branch["threshold"]),
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
    assert_shared_selection(task_result)
    task: Task = task_result["task"]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    abide2_indices = np.arange(len(cohort.abide2), dtype=np.int64)
    if len(abide2_indices) != 727 or not np.array_equal(abide2_indices, np.arange(727, dtype=np.int64)):
        raise RuntimeError("ABIDE-II must remain the complete original 727-subject external cohort")
    evaluation_specs = (
        ("caltech", cohort.abide1, task.test_raw, task.test_labels, subject_indices_for_task(cohort, task)),
        ("abide2", cohort.abide2, abide2_indices, cohort.abide2_labels[abide2_indices], abide2_indices.copy()),
    )
    for method, branch_key in ((BASELINE_METHOD, "baseline"), (TREATMENT_METHOD, "treatment")):
        branch = task_result[branch_key]
        for scope, data, raw_indices, labels, subject_indices in evaluation_specs:
            metrics, probabilities = evaluate_selected_branch(cohort, branch, method, data, raw_indices, labels, device)
            rows.append(metric_row(method, scope, task.repeat, "external", metrics, branch, len(raw_indices)))
            records = add_prediction_group(
                records, scope, method, task.repeat, "external", subject_indices, raw_indices,
                labels, probabilities, float(branch["threshold"]),
            )
            print(
                f"TASK_DONE repeat={task.repeat} scope={scope} fold=external method={method} "
                f"selected_encoder_epoch={branch['encoder_epoch']} test_auc={metrics['auc']:.6f} "
                f"test_bce={metrics['bce']:.6f} elapsed_minutes={task_result['elapsed_seconds']/60.0:.1f}",
                flush=True,
            )
    # Pairing is asserted before any external result shard can be written.
    for scope, expected_count in (("caltech", 37), ("abide2", 727)):
        paired: dict[str, list[tuple[int, int]]] = {}
        for method in METHODS:
            ids = sorted((record["subject_index"], record["raw_index"]) for record in records if record["scope"] == scope and record["method"] == method)
            if len(ids) != expected_count:
                raise RuntimeError(f"{scope}/{method} count={len(ids)} expected={expected_count}")
            paired[method] = ids
        if paired[BASELINE_METHOD] != paired[TREATMENT_METHOD]:
            raise RuntimeError(f"INTEGRITY_FAIL: {scope} paired baseline/treatment IDs differ")
    expected_abide2_ids = [(index, index) for index in range(727)]
    for method in METHODS:
        actual = sorted((record["subject_index"], record["raw_index"]) for record in records if record["scope"] == "abide2" and record["method"] == method)
        if actual != expected_abide2_ids:
            raise RuntimeError(f"INTEGRITY_FAIL: ABIDE-II/{method} IDs are not exactly 0..726")
    external_models = {
        "format_version": 1, "repeat": int(task.repeat), "task_identity": task_identity(task),
        "model_config": make_model_config(cohort)[0], "baseline": state_to_cpu(task_result["baseline"]),
        TREATMENT_METHOD: state_to_cpu(task_result["treatment"]),
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
    if len(selected) != 26:
        return False
    expected: set[tuple[str, str, str]] = set()
    for fold in range(10):
        for method in METHODS:
            expected.add((method, "internal_fold", str(fold)))
    for method in METHODS:
        expected.update({
            (method, "internal_oof", "all"), (method, "caltech", "external"), (method, "abide2", "external"),
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
    """Complete or resume all 10 internal folds and one external task."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = worker_paths(repeat)
    rows = read_metric_shard(paths["csv"])
    records = read_prediction_records(paths["predictions"])
    if expected_repeat_rows(rows, repeat) and paths["external"].exists():
        print(f"REPEAT_DONE repeat={repeat} status=PASS already_complete=Y", flush=True)
        return
    tasks = make_tasks(cohort, repeat)
    print(f"REPEAT_START repeat={repeat} device={device} tasks=11", flush=True)
    for task in tasks[:-1]:
        if task_row_complete(rows, "internal_fold", repeat, task.fold):
            print(f"TASK_SKIP repeat={repeat} scope=internal_fold fold={task.fold} reason=PASS_SHARD", flush=True)
            clean_task_checkpoints(repeat, task)
            continue
        # Any incomplete task's artifacts are replaced atomically only after
        # its paired predictions are available, so no duplicated fold result
        # can survive an interruption.
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
        already = any(
            row["scope"] == "internal_oof" and row["method"] == method and int(row["repeat"]) == int(repeat) and row["fold"] == "all"
            for row in rows
        )
        if already:
            continue
        oof_row, oof_records = build_internal_oof_row(cohort, records, rows, method, repeat)
        records = clear_prediction_group(records, "internal_oof", repeat, "all", method)
        records.extend(oof_records)
        write_prediction_records_atomic(paths["predictions"], records)
        sibling = [
            row for row in rows if row["scope"] == "internal_oof" and int(row["repeat"]) == int(repeat)
            and row["fold"] == "all" and row["method"] != method
        ]
        rows = clear_metric_rows(rows, "internal_oof", repeat, "all")
        write_metric_shard_atomic(paths["csv"], [*rows, *sibling])
        rows.extend(sibling)
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
        raise RuntimeError(f"repeat={repeat} incomplete: expected 26 final metric rows")
    write_json_atomic(paths["done"], {
        "repeat": int(repeat), "status": "PASS",
        "metrics_rows": len([row for row in rows if int(row["repeat"]) == int(repeat)]),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"REPEAT_DONE repeat={repeat} status=PASS rows=26", flush=True)


def write_final_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    records.sort(key=lambda item: (item["scope"], item["method"], int(item["repeat"]), item["fold"], int(item["subject_index"])))
    write_prediction_records_atomic(path, records)


def validate_prediction_records(records: list[dict[str, Any]], cohort: Cohort) -> None:
    expected_count = {"internal_oof": 882, "caltech": 37, "abide2": 727}
    for scope, count in expected_count.items():
        for method in METHODS:
            for repeat in range(10):
                subset = [
                    record for record in records
                    if record["scope"] == scope and record["method"] == method and int(record["repeat"]) == repeat
                ]
                if len(subset) != count:
                    raise RuntimeError(f"{scope}/{method}/repeat={repeat} has {len(subset)} predictions, expected {count}")
        for repeat in range(10):
            baseline_ids = sorted(
                (record["subject_index"], record["raw_index"])
                for record in records if record["scope"] == scope and record["method"] == BASELINE_METHOD and int(record["repeat"]) == repeat
            )
            treatment_ids = sorted(
                (record["subject_index"], record["raw_index"])
                for record in records if record["scope"] == scope and record["method"] == TREATMENT_METHOD and int(record["repeat"]) == repeat
            )
            if baseline_ids != treatment_ids:
                raise RuntimeError(f"INTEGRITY_FAIL: {scope}/repeat={repeat} paired IDs differ")
    expected_abide2 = [(index, index) for index in range(727)]
    for method in METHODS:
        for repeat in range(10):
            actual = sorted(
                (record["subject_index"], record["raw_index"])
                for record in records if record["scope"] == "abide2" and record["method"] == method and int(record["repeat"]) == repeat
            )
            if actual != expected_abide2:
                raise RuntimeError(f"INTEGRITY_FAIL: ABIDE-II/{method}/repeat={repeat} is not the full paired 727-ID cohort")
    if len(cohort.abide2) != 727:
        raise RuntimeError("INTEGRITY_FAIL: ABIDE-II source cohort changed")


def aggregate_rows(rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], source_hashes: dict[str, str]) -> dict[str, Any]:
    def select(method: str, scope: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["method"] == method and row["scope"] == scope]

    scopes = ("internal_oof", "caltech", "abide2")
    aggregates: dict[str, Any] = {}
    for method in METHODS:
        for scope in scopes:
            selected = select(method, scope)
            aucs = [float(row["auc"]) for row in selected]
            bces = [float(row["bce"]) for row in selected]
            aggregates[f"{method}::{scope}"] = {
                "n": len(selected), "auc_mean": mean_or_nan(aucs),
                "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "bce_mean": mean_or_nan(bces),
                "bce_std": float(np.std(bces, ddof=1)) if len(bces) > 1 else 0.0,
                "f1_mean": mean_or_nan([row["f1"] for row in selected]),
                "balanced_accuracy_mean": mean_or_nan([row["balanced_accuracy"] for row in selected]),
            }

    deltas: dict[str, list[float]] = {}
    for scope in scopes:
        baseline = {int(row["repeat"]): row for row in select(BASELINE_METHOD, scope)}
        treatment = {int(row["repeat"]): row for row in select(TREATMENT_METHOD, scope)}
        if set(baseline) != set(range(10)) or set(treatment) != set(range(10)):
            raise RuntimeError(f"cannot aggregate incomplete paired repeats for {scope}")
        deltas[f"{scope}_auc"] = [float(treatment[repeat]["auc"] - baseline[repeat]["auc"]) for repeat in range(10)]
        deltas[f"{scope}_bce"] = [float(treatment[repeat]["bce"] - baseline[repeat]["bce"]) for repeat in range(10)]

    internal_auc = aggregates[f"{BASELINE_METHOD}::internal_oof"]["auc_mean"]
    abide2_auc = aggregates[f"{BASELINE_METHOD}::abide2"]["auc_mean"]
    pipeline_drift = (
        not np.isfinite(internal_auc) or not np.isfinite(abide2_auc)
        or abs(float(internal_auc) - 0.723) > 0.015 or abs(float(abide2_auc) - 0.739) > 0.015
    )
    internal_delta = mean_or_nan(deltas["internal_oof_auc"])
    internal_bce_delta = mean_or_nan(deltas["internal_oof_bce"])
    abide2_delta = mean_or_nan(deltas["abide2_auc"])
    abide2_bce_delta = mean_or_nan(deltas["abide2_bce"])
    internal_positive = int(sum(delta > 0.0 for delta in deltas["internal_oof_auc"]))
    treatment_task_rows = [
        row for row in rows if row["method"] == TREATMENT_METHOD and row["scope"] in {"internal_fold", "caltech"}
    ]
    selected_epochs = [
        int(round(float(row["selected_crsm_epoch"]))) for row in treatment_task_rows
        if np.isfinite(float(row["selected_crsm_epoch"]))
    ]
    epoch0_fraction = float(np.mean([epoch == 0 for epoch in selected_epochs])) if selected_epochs else float("nan")
    epoch_distribution = {str(epoch): selected_epochs.count(epoch) for epoch in sorted(set(selected_epochs))}
    internal_delta_std = float(np.std(deltas["internal_oof_auc"], ddof=1))
    # These conditions operationalize the prompt's explicitly named failure
    # modes; they are fixed before inspecting final outcomes.
    highly_unstable = bool(internal_delta_std >= 0.030 and internal_positive not in {0, 10})
    epoch0_no_gain = bool(epoch0_fraction >= 0.80 and internal_delta < 0.003 and abide2_delta <= 0.0)
    threshold_only_gain = bool(
        internal_delta <= 0.0
        and (aggregates[f"{TREATMENT_METHOD}::internal_oof"]["f1_mean"] > aggregates[f"{BASELINE_METHOD}::internal_oof"]["f1_mean"]
             or aggregates[f"{TREATMENT_METHOD}::internal_oof"]["balanced_accuracy_mean"] > aggregates[f"{BASELINE_METHOD}::internal_oof"]["balanced_accuracy_mean"])
    )
    failure_reasons: list[str] = []
    if internal_delta <= 0.0:
        failure_reasons.append("internal_mean_delta_auc_nonpositive")
    if abide2_delta <= -0.005:
        failure_reasons.append("abide2_mean_delta_auc_at_or_below_minus_0.005")
    if highly_unstable:
        failure_reasons.append("highly_unstable_internal_repeat_deltas")
    if epoch0_no_gain:
        failure_reasons.append("vast_majority_epoch0_with_no_mean_gain")
    if threshold_only_gain:
        failure_reasons.append("threshold_metric_gain_without_auc_gain")

    if pipeline_drift:
        decision = "PIPELINE_DRIFT"
    elif failure_reasons:
        decision = "FAIL"
    elif (
        internal_delta >= 0.005 and internal_positive >= 7 and abide2_delta >= 0.003
        and internal_bce_delta <= 0.005 and abide2_bce_delta <= 0.005
    ):
        decision = "STRONG_POSITIVE"
    elif (
        (internal_delta >= 0.003 and internal_positive >= 7 and abide2_delta >= 0.0)
        or (abide2_delta >= 0.005 and internal_delta >= 0.0)
    ):
        decision = "PROMISING"
    else:
        decision = "FAIL"

    diagnostic_summary: dict[str, Any] = {
        "crsm_task_count": len(treatment_task_rows), "selected_crsm_epoch_distribution": epoch_distribution,
        "selected_crsm_epoch0_fraction": epoch0_fraction,
        "internal_delta_auc_std": internal_delta_std,
        "highly_unstable": highly_unstable, "epoch0_no_gain": epoch0_no_gain,
    }
    for key in CRSM_DIAGNOSTIC_COLUMNS:
        values = [float(row[key]) for row in treatment_task_rows if np.isfinite(float(row[key]))]
        diagnostic_summary[f"{key}_mean"] = mean_or_nan(values)
        diagnostic_summary[f"{key}_min"] = float(min(values)) if values else float("nan")
        diagnostic_summary[f"{key}_max"] = float(max(values)) if values else float("nan")

    task_seconds = [float(sum(float(value) for value in row.get("completed_tasks", {}).values())) for row in runtime_rows]
    peak_alloc = mean_or_nan([row.get("peak_gpu_alloc_gb", float("nan")) for row in runtime_rows])
    peak_reserved = mean_or_nan([row.get("peak_gpu_reserved_gb", float("nan")) for row in runtime_rows])
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
        "aggregates": aggregates, "deltas": deltas,
        "baseline_sanity": {
            "internal_auc_reference": 0.723, "abide2_auc_reference": 0.739,
            "internal_auc": internal_auc, "abide2_auc": abide2_auc, "pipeline_drift": bool(pipeline_drift),
        },
        "decision": decision,
        "decision_inputs": {
            "internal_mean_delta_auc": internal_delta, "internal_positive_repeats": internal_positive,
            "internal_mean_delta_bce": internal_bce_delta, "abide2_mean_delta_auc": abide2_delta,
            "abide2_mean_delta_bce": abide2_bce_delta, "failure_reasons": failure_reasons,
            "threshold_only_gain": threshold_only_gain,
        },
        "crsm_diagnostics": diagnostic_summary,
        "completion": {
            "expected_internal_folds_per_method": 100, "expected_internal_oof_per_method": 10,
            "expected_caltech_per_method": 10, "expected_abide2_per_method": 10, "metrics_rows": len(rows),
        },
        "runtime": {
            "workers": len(runtime_rows), "wall_clock_seconds": worker_wall,
            "worker_seconds": float(sum(task_seconds)), "max_repeat_cumulative_seconds": float(max(task_seconds or [0.0])),
            "mean_peak_gpu_alloc_gb": peak_alloc, "mean_peak_gpu_reserved_gb": peak_reserved,
            "checkpoint_resume_supported": True, "checkpoint_interval_epochs": int(CONFIG["checkpoint_interval"]),
        },
        "integrity": {
            "root_and_prior_experiments_unchanged": True, "source_hashes": source_hashes,
            "experiment1_runtime_import": False, "original_infonce_only": True,
            "shared_encoder_trajectory": True, "frozen_baseline_classifier": True,
            "reference_unique_outer_train_only": True, "duplicate_train_only": True,
            "no_validation_or_test_in_reference_pca_centroid": True, "validation_only_model_selection": True,
            "validation_only_threshold": True, "test_access_per_epoch": False,
            "abide2_full_paired_subject_ids_equal": True, "abide2_full_baseline_eligible": True,
            "n_experts": 2, "temperature": 1.0, "trainable_crsm_parameters": 147074,
        },
        "literature_audit": {
            "hydra": "Control-referenced convex-polytope motivation only; no HYDRA solver/SVM/consensus/DPP code copied.",
            "smile_gan": "Control-to-patient heterogeneity motivation only; no GAN/code/objective copied.",
            "brainscl_experiment1a": "Prototype-guided representation modification was already tested and failed; Experiment 1B keeps the encoder unchanged.",
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the required handoff report after all 10 paired repeats finish."""
    aggregates = summary["aggregates"]
    deltas = summary["deltas"]

    def value(method: str, scope: str, metric: str) -> float:
        return float(aggregates[f"{method}::{scope}"][metric])

    table: list[str] = []
    for name, scope in (("Internal OOF", "internal_oof"), ("Caltech", "caltech"), ("ABIDE-II", "abide2")):
        b_auc, t_auc = value(BASELINE_METHOD, scope, "auc_mean"), value(TREATMENT_METHOD, scope, "auc_mean")
        b_bce, t_bce = value(BASELINE_METHOD, scope, "bce_mean"), value(TREATMENT_METHOD, scope, "bce_mean")
        table.append(
            f"| {name} | {fmt(b_auc)} | {fmt(t_auc)} | {fmt(t_auc-b_auc)} | {fmt(b_bce)} | {fmt(t_bce)} | {fmt(t_bce-b_bce)} |"
        )
    diagnostic = summary["crsm_diagnostics"]
    decision = summary["decision"]
    if decision == "FAIL":
        interpretation = "Preserving VarCoNet individualized representations while replacing a single disease direction with two soft control-referenced boundaries does not yield stable ASD classification gains."
    elif decision in {"STRONG_POSITIVE", "PROMISING"}:
        interpretation = (
            "The frozen VarCoNet representation contains a reproducible classification signal that is better captured by the pre-specified soft control-referenced two-boundary readout. "
            "This is a classification result, not evidence for biological ASD subtypes."
        )
    else:
        interpretation = "The pre-specified integrity or baseline-sanity rule prevents a scientific interpretation of treatment gain."
    report = f"""# Experiment 1B

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
- Source integrity was asserted before smoke and after aggregation with pre-recorded hashes for root `model_scripts`, root utility/classification scripts, and Experiment 1/2/2B/3 directories. The post-run audit passed: `{summary['integrity']['root_and_prior_experiments_unchanged']}`.

## Results

| Scope | Baseline AUC | CRSM AUC | ΔAUC | Baseline BCE | CRSM BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Internal repeat ΔAUC (CRSM − baseline): `[{', '.join(fmt(item, 6) for item in deltas['internal_oof_auc'])}]`

ABIDE-II repeat ΔAUC (CRSM − baseline): `[{', '.join(fmt(item, 6) for item in deltas['abide2_auc'])}]`

Baseline sanity: internal OOF AUC={fmt(summary['baseline_sanity']['internal_auc'])} (reference 0.723); ABIDE-II AUC={fmt(summary['baseline_sanity']['abide2_auc'])} (reference 0.739); pipeline drift={summary['baseline_sanity']['pipeline_drift']}.

## CRSM diagnostics

- Selected CRSM epoch-0 fraction={fmt(diagnostic['selected_crsm_epoch0_fraction'])}; epoch distribution=`{json.dumps(diagnostic['selected_crsm_epoch_distribution'], sort_keys=True)}`.
- HC centroid norm mean={fmt(diagnostic['hc_centroid_norm_mean'], 6)}; PCA unit norm mean={fmt(diagnostic['pca_unit_norm_mean'], 8)}; symmetry scale mean={fmt(diagnostic['symmetry_scale_mean'], 9)}.
- Expert mass mean: `({fmt(diagnostic['expert_1_mean_mass_mean'], 6)}, {fmt(diagnostic['expert_2_mean_mass_mean'], 6)})`; hard fractions: `({fmt(diagnostic['expert_1_hard_fraction_mean'], 6)}, {fmt(diagnostic['expert_2_hard_fraction_mean'], 6)})`; responsibility entropy={fmt(diagnostic['responsibility_entropy_mean'], 6)}.
- Delta L2 means: `({fmt(diagnostic['delta_weight_1_l2_mean'], 6)}, {fmt(diagnostic['delta_weight_2_l2_mean'], 6)})`; weight cosine={fmt(diagnostic['delta_weight_cosine_mean'], 6)}.
- Internal repeat ΔAUC standard deviation={fmt(diagnostic['internal_delta_auc_std'], 6)}; pre-specified high-instability flag={diagnostic['highly_unstable']}; epoch-0/no-gain flag={diagnostic['epoch0_no_gain']}.

## Runtime

Ten independent workers use full precision without DDP, DataParallel, AMP or `torch.compile`: repeats 0,2,4,6,8 on GPU0 and 1,3,5,7,9 on GPU1, with six CPU threads per worker. Wall clock={fmt(summary['runtime']['wall_clock_seconds']/3600.0, 3)} h; summed task compute={fmt(summary['runtime']['worker_seconds']/3600.0, 3)} h; mean peak GPU allocation/reservation={fmt(summary['runtime']['mean_peak_gpu_alloc_gb'], 3)} / {fmt(summary['runtime']['mean_peak_gpu_reserved_gb'], 3)} GB.

## Final decision

**{decision}**

## Interpretation

{interpretation}

The result does not assign biological ASD subtypes and should not be interpreted as such. It evaluates only the pre-specified classification-readout hypothesis.

## Reproducibility artifacts

`results/metrics.csv` contains 260 rows (100 internal folds + 10 internal OOF + 10 Caltech + 10 ABIDE-II per method); `results/summary.json` records aggregate/decision/integrity fields; `results/predictions.npz` contains paired final-scope IDs, labels, probabilities and thresholds; `results/external_best_models.pt` stores the 10 paired external baseline/CRSM selected states. Logs are append-only in `logs/`.
"""
    (EXPERIMENT_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def aggregate_main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if (RESULTS_DIR / "summary.json").exists() and not list(RESULTS_DIR.glob(".worker_r*.csv")):
        existing = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
        if existing.get("completion", {}).get("metrics_rows") == 260:
            print(json.dumps({"status": "PASS", "rows": 260, "decision": existing.get("decision"), "already_aggregated": True}), flush=True)
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
    if len(all_rows) != 260:
        raise RuntimeError(f"expected 260 metrics rows, found {len(all_rows)}")
    if any(row["status"] != "PASS" for row in all_rows):
        raise RuntimeError("cannot aggregate non-PASS metric rows")
    validate_prediction_records(all_records, cohort)
    # A second source audit makes the scope assertion cover the entire run.
    source_hashes = assert_external_integrity()
    write_metric_shard_atomic(RESULTS_DIR / "metrics.csv", all_rows)
    write_final_predictions(RESULTS_DIR / "predictions.npz", all_records)
    atomic_torch_save(RESULTS_DIR / "external_best_models.pt", {
        "format_version": 1, "atlas": "AICHA", "config": state_to_cpu(CONFIG), "repeats": external_models,
    })
    summary = aggregate_rows(all_rows, runtime_rows, source_hashes)
    summary["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(RESULTS_DIR / "summary.json", summary)
    write_report(summary, all_rows)
    # Worker shards and rolling checkpoints are resume material only.  The
    # selected external models are deliberately retained above.
    for repeat in range(10):
        for path in worker_paths(repeat).values():
            path.unlink(missing_ok=True)
    for path in CHECKPOINT_DIR.glob("worker_r*_latest.pt*"):
        path.unlink(missing_ok=True)
    for path in CHECKPOINT_DIR.glob("worker_r*_best.pt*"):
        path.unlink(missing_ok=True)
    smoke_marker_path().unlink(missing_ok=True)
    print(json.dumps({"status": "PASS", "rows": len(all_rows), "decision": summary["decision"]}), flush=True)


def smoke_marker_path() -> Path:
    return RESULTS_DIR / ".smoke_pass.json"


def smoke_test(device: torch.device) -> None:
    """Run exactly one real-data technical smoke test before full workers."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    marker = smoke_marker_path()
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") == "PASS":
            print("SMOKE_ALREADY_PASS " + json.dumps(payload, ensure_ascii=False), flush=True)
            return
        raise RuntimeError(f"existing smoke marker is not PASS: {marker}")
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
    # Use a real outer-training reference subset containing both labels.
    asd_positions = np.where(task.reference_labels == 0)[0][:16]
    hc_positions = np.where(task.reference_labels == 1)[0][:16]
    positions = np.concatenate((asd_positions, hc_positions)).astype(np.int64)
    if len(asd_positions) < 2 or len(hc_positions) < 2:
        raise RuntimeError("smoke reference selection lacks both classes")
    raw = task.reference_raw[positions]
    labels = torch.as_tensor(task.reference_labels[positions], dtype=torch.long, device=device)
    x = torch.from_numpy(np.asarray(cohort.abide1[raw], dtype=np.float32)).to(device)
    local.eval()
    root.eval()
    with torch.no_grad():
        local_stable = local(x)
        root_stable = root(x)
    root_parity = float((local_stable - root_stable).abs().max().cpu())
    if root_parity > 1e-6:
        raise RuntimeError(f"root/local VarCoNet forward parity failed: {root_parity}")
    if tuple(local_stable.shape) != (len(raw), int(CONFIG["stable_dim"])):
        raise RuntimeError(f"stable learned-FC shape drifted: {tuple(local_stable.shape)}")

    set_all_seeds(778)
    baseline = MLP(int(CONFIG["stable_dim"]), 2).to(device)
    baseline_state = clone_state_cpu(baseline.state_dict())
    hc_centroid = local_stable[labels == 1].mean(dim=0)
    if not bool(torch.isfinite(hc_centroid).all()):
        raise RuntimeError("smoke HC centroid is non-finite")
    crsm = ControlReferencedSoftMultiBoundary(baseline_state, hc_centroid, temperature=1.0).to(device)
    zero_diag = zero_delta_diagnostics(crsm, local_stable, labels, baseline_state, device)
    parameters = trainable_parameter_count(crsm)
    if parameters != 147074:
        raise RuntimeError(f"smoke trainable parameter count={parameters}, expected=147074")
    initialization = initialize_symmetric_pca(crsm, local_stable, labels, seed=779, ratio=0.01)
    if abs(initialization.pca_unit_norm - 1.0) > 1e-5 or initialization.initial_delta_l2 <= 0.0:
        raise RuntimeError("smoke PCA symmetry initialization failed")
    # A nonzero ±1% perturbation must change at least one expert margin while
    # retaining finite probabilities and a finite CRSM backward pass.
    details = crsm.forward_details(local_stable)
    perturbation = float((details["expert_margins"][:, 0] - details["m0"]).abs().max().detach().cpu())
    loss = crsm_loss(details, labels)
    loss.backward()
    if perturbation <= 0.0 or not bool(torch.isfinite(loss)) or not finite_gradients(crsm):
        raise RuntimeError("smoke CRSM perturbation/forward/backward finite check failed")
    payload = {
        "status": "PASS", "device": str(device), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_baseline_forward_max_abs": root_parity, "stable_shape": list(local_stable.shape),
        "zero_diagnostics": zero_diag, "pca_diagnostics": initialization.as_dict(),
        "symmetry_perturbation_max_abs": perturbation, "trainable_parameters": parameters,
        "forward_backward_finite": True, "source_hashes": source_hashes,
    }
    write_json_atomic(marker, payload)
    print("SMOKE_PASS " + json.dumps(payload, ensure_ascii=False), flush=True)
    del local, root, baseline, crsm, x, local_stable, root_stable
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
            "repeat": repeat, "status": "PASS", "device": str(device), "worker_wall_seconds": time.time() - started,
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

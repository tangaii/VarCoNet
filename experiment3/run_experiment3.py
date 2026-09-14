#!/usr/bin/env python3
"""Experiment 3: Consensus High-Order Modular Residual VarCoNet.

The runner is self-contained under ``xiaolunwen/experiment3``.  It follows the
audited VarCoNet baseline exactly, selects one frozen baseline trajectory per
task, and then fits only a 528-parameter train-only consensus-module residual.
No test labels or external subjects are consulted before the selected model is
frozen.
"""

from __future__ import annotations

import argparse
import csv
import copy
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

from high_order_readout import (  # noqa: E402
    FrozenStableModuleResidualClassifier,
    ModulePairPooler,
    discover_consensus_modules,
)
from model_scripts.VarCoNet import VarCoNet  # noqa: E402
from model_scripts.classifier import MLP  # noqa: E402
from model_scripts.scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from utils import DualBranchContrast, InfoNCE, augment, removeDuplicates  # noqa: E402


CONFIG: dict[str, Any] = {
    "atlas": "AICHA",
    "batch_size": 64,
    "min_length": 80,
    "epochs": 50,
    "warm_up_epochs": 10,
    "epochs_cls": 150,
    "lr_cls": 5e-5,
    "num_classes": 2,
    "n_modules": 32,
    "meta_dim": 528,
    "checkpoint_interval": 10,
    "cpu_threads": 6,
}

BASELINE_METHOD = "baseline"
TREATMENT_METHOD = "consensus_highorder_residual"
METHODS = (BASELINE_METHOD, TREATMENT_METHOD)

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
    "selected_residual_epoch",
    "threshold_hc",
    "n_modules",
    "meta_dim",
    "module_size_min",
    "module_size_mean",
    "module_size_max",
    "singleton_modules",
    "empty_meta_bins",
    "affinity_mean",
    "affinity_std",
    "affinity_max",
    "residual_head_l2",
    "task_seconds",
    "n_test",
    "status",
]


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
    abide2_evaluable_indices: np.ndarray
    abide2_excluded_indices: np.ndarray


@dataclass
class Task:
    repeat: int
    scope: str
    fold: str
    seed: int
    # Outer-train singleton subjects only; duplicates are intentionally absent.
    module_raw: np.ndarray
    train_raw: np.ndarray
    train_names: list[str]
    train_labels: np.ndarray
    val_raw: np.ndarray
    val_labels: np.ndarray
    test_raw: np.ndarray
    test_labels: np.ndarray


def set_all_seeds(seed: int) -> None:
    """Set every stochastic source used by the original implementation."""
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
        # This is process-global and may already have been set by a previous
        # task in the same worker.
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
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
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


def load_npz_stack(path: Path) -> np.ndarray:
    cache_dir = EXPERIMENT_DIR / "dataset_cache"
    cache_path = cache_dir / f"{path.stem}.npy"
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r", allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        values = [np.asarray(archive[key], dtype=np.float32) for key in archive.files]
    if not values:
        raise RuntimeError(f"empty dataset archive: {path}")
    stacked = np.stack(values, axis=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, stacked, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)


def load_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def load_cohort() -> Cohort:
    """Reproduce the audited ABIDE-I/II cohort preparation exactly."""
    abide1_dir = DATASET_DIR / "ABIDEI"
    abide2_dir = DATASET_DIR / "ABIDEII"
    abide1 = load_npz_stack(abide1_dir / "ABIDEI_nilearn_AICHA.npz")
    abide1_names_raw = load_names(abide1_dir / "ABIDEI_nilearn_names.txt")
    abide1_labels_raw = np.load(
        abide1_dir / "ABIDEI_nilearn_classes.npy", allow_pickle=False
    ).astype(np.int64)
    if len(abide1) != len(abide1_names_raw) or len(abide1) != len(abide1_labels_raw):
        raise RuntimeError("ABIDE-I data/name/label lengths are not aligned")

    names_array = np.asarray(abide1_names_raw)
    names_unique, counts = np.unique(names_array, return_counts=True)
    duplicate_name_values = names_unique[counts > 1]
    duplicate_raw_parts: list[int] = []
    duplicate_names: list[str] = []
    for name in duplicate_name_values:
        positions = np.where(names_array == name)[0]
        duplicate_raw_parts.extend(int(x) for x in positions.tolist())
        duplicate_names.extend([str(name)] * len(positions))

    singleton_names = names_unique[counts == 1].tolist()
    singleton_raw = np.asarray(
        [int(np.where(names_array == name)[0][0]) for name in singleton_names],
        dtype=np.int64,
    )
    duplicate_raw = np.asarray(duplicate_raw_parts, dtype=np.int64)
    duplicate_labels = abide1_labels_raw[duplicate_raw]

    caltech_names = [
        f"sub-00{numeric_id}"
        for numeric_id in range(51456, 51494)
        if f"sub-00{numeric_id}" in singleton_names
    ]
    singleton_name_to_raw = {
        name: int(raw) for name, raw in zip(singleton_names, singleton_raw)
    }
    caltech_raw = np.asarray(
        [singleton_name_to_raw[name] for name in caltech_names], dtype=np.int64
    )
    caltech_name_set = set(caltech_names)
    internal_names = [name for name in singleton_names if name not in caltech_name_set]
    internal_raw = np.asarray(
        [singleton_name_to_raw[name] for name in internal_names], dtype=np.int64
    )
    internal_labels = abide1_labels_raw[internal_raw]
    caltech_labels = abide1_labels_raw[caltech_raw]

    abide2 = load_npz_stack(abide2_dir / "ABIDEII_nilearn_AICHA.npz")
    abide2_names = load_names(abide2_dir / "ABIDEII_nilearn_names.txt")
    abide2_labels = np.load(
        abide2_dir / "ABIDEII_nilearn_classes.npy", allow_pickle=False
    ).astype(np.int64)
    if len(abide2) != len(abide2_labels) or len(abide2) != len(abide2_names):
        raise RuntimeError("ABIDE-II data/name/label lengths are not aligned")
    if abide1.shape[1:] != abide2.shape[1:]:
        raise RuntimeError(f"AICHA shapes differ: {abide1.shape} vs {abide2.shape}")
    if len(internal_names) != 882 or len(caltech_names) != 37 or len(duplicate_raw) != 76:
        raise RuntimeError(
            "Unexpected ABIDE-I cohort preparation: "
            f"internal={len(internal_names)}, caltech={len(caltech_names)}, "
            f"duplicates={len(duplicate_raw)}"
        )

    # Experiment 3 has no temporal-window eligibility rule.  Restore the
    # complete audited baseline-eligible ABIDE-II cohort and use precisely the
    # same 727 raw indices for both paired methods.
    if len(abide2) != 727:
        raise RuntimeError(
            f"ABIDE-II baseline-eligible cohort must contain 727 subjects, found {len(abide2)}"
        )
    evaluable_indices = np.arange(len(abide2), dtype=np.int64)
    excluded_indices = np.asarray([], dtype=np.int64)

    return Cohort(
        abide1=abide1,
        abide1_names=abide1_names_raw,
        abide1_labels=abide1_labels_raw,
        duplicate_raw=duplicate_raw,
        duplicate_names=duplicate_names,
        duplicate_labels=duplicate_labels,
        internal_raw=internal_raw,
        internal_names=internal_names,
        internal_labels=internal_labels,
        caltech_raw=caltech_raw,
        caltech_names=caltech_names,
        caltech_labels=caltech_labels,
        abide2=abide2,
        abide2_names=abide2_names,
        abide2_labels=abide2_labels,
        abide2_evaluable_indices=evaluable_indices,
        abide2_excluded_indices=excluded_indices,
    )


def make_tasks(cohort: Cohort, repeat: int) -> list[Task]:
    """Use the Experiment 1 split logic without importing Experiment 1."""
    repeat = int(repeat)
    tasks: list[Task] = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + repeat)
    raw_to_name = {
        int(raw): name for raw, name in zip(cohort.internal_raw, cohort.internal_names)
    }
    for fold, (train_index, test_index) in enumerate(
        skf.split(cohort.internal_raw, cohort.internal_labels)
    ):
        candidate_raw = cohort.internal_raw[train_index]
        candidate_y = cohort.internal_labels[train_index]
        train_raw, val_raw, train_y, val_y, _, _ = train_test_split(
            candidate_raw,
            candidate_y,
            np.arange(len(candidate_raw)),
            test_size=0.15,
            random_state=42,
            stratify=candidate_y,
        )
        train_names = cohort.duplicate_names + [raw_to_name[int(x)] for x in train_raw]
        full_train_raw = np.concatenate(
            [cohort.duplicate_raw, np.asarray(train_raw, dtype=np.int64)]
        )
        if np.intersect1d(np.asarray(train_raw, dtype=np.int64), cohort.duplicate_raw).size:
            raise RuntimeError("module_raw unexpectedly contains duplicate occurrences")
        full_train_y = np.concatenate(
            [cohort.duplicate_labels, np.asarray(train_y, dtype=np.int64)]
        )
        tasks.append(
            Task(
                repeat=repeat,
                scope="internal_cv",
                fold=str(fold),
                seed=100000 + repeat * 100 + int(fold),
                module_raw=np.asarray(train_raw, dtype=np.int64),
                train_raw=full_train_raw,
                train_names=train_names,
                train_labels=full_train_y,
                val_raw=np.asarray(val_raw, dtype=np.int64),
                val_labels=np.asarray(val_y, dtype=np.int64),
                test_raw=np.asarray(cohort.internal_raw[test_index], dtype=np.int64),
                test_labels=np.asarray(cohort.internal_labels[test_index], dtype=np.int64),
            )
        )

    train_raw, val_raw, train_y, val_y, _, _ = train_test_split(
        cohort.internal_raw,
        cohort.internal_labels,
        np.arange(len(cohort.internal_raw)),
        test_size=0.1,
        random_state=42 + repeat,
        stratify=cohort.internal_labels,
    )
    ext_train_names = cohort.duplicate_names + [raw_to_name[int(x)] for x in train_raw]
    ext_train_raw = np.concatenate(
        [cohort.duplicate_raw, np.asarray(train_raw, dtype=np.int64)]
    )
    if np.intersect1d(np.asarray(train_raw, dtype=np.int64), cohort.duplicate_raw).size:
        raise RuntimeError("module_raw unexpectedly contains duplicate occurrences")
    ext_train_y = np.concatenate(
        [cohort.duplicate_labels, np.asarray(train_y, dtype=np.int64)]
    )
    tasks.append(
        Task(
            repeat=repeat,
            scope="caltech_external",
            fold="external",
            seed=200000 + repeat,
            module_raw=np.asarray(train_raw, dtype=np.int64),
            train_raw=ext_train_raw,
            train_names=ext_train_names,
            train_labels=ext_train_y,
            val_raw=np.asarray(val_raw, dtype=np.int64),
            val_labels=np.asarray(val_y, dtype=np.int64),
            test_raw=np.asarray(cohort.caltech_raw, dtype=np.int64),
            test_labels=np.asarray(cohort.caltech_labels, dtype=np.int64),
        )
    )
    return tasks


def paired_batch_schedule(
    names: list[str], n_items: int, epochs: int, seed: int
) -> list[list[list[int]]]:
    """Precompute the source-style batch schedule deterministically."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    schedule: list[list[list[int]]] = []
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(int(n_items), generator=generator).tolist()
        epoch_batches: list[list[int]] = []
        for batch_idx, start in enumerate(
            range(0, len(permutation), int(CONFIG["batch_size"]))
        ):
            raw = [int(x) for x in permutation[start : start + int(CONFIG["batch_size"])]
            ]
            saved_state = random.getstate()
            random.seed(int(seed) + epoch * 1000003 + batch_idx)
            selected = removeDuplicates(names, raw)
            random.setstate(saved_state)
            epoch_batches.append([int(x) for x in selected])
        schedule.append(epoch_batches)
    return schedule


def finite_gradients(module: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


def extract_stable_features(
    encoder: VarCoNet,
    data: np.ndarray,
    raw_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Extract only stable learned-FC features, once per batch."""
    stable_parts: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(raw_indices), int(CONFIG["batch_size"])):
            selected = np.asarray(raw_indices[start : start + int(CONFIG["batch_size"])])
            array = np.asarray(data[selected], dtype=np.float32)
            x = torch.from_numpy(array).to(device)
            stable_parts.append(encoder(x).detach())
            del x
    if not stable_parts:
        raise RuntimeError("cannot extract features from an empty split")
    return torch.cat(stable_parts, dim=0)


def extract_stable_unique_features(
    encoder: VarCoNet,
    data: np.ndarray,
    raw_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Extract stable learned-FC vectors for module discovery only."""
    return extract_stable_features(encoder, data, raw_indices, device)


def validation_threshold(y_true: np.ndarray, p_hc: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), p_hc.astype(float))
    youden = tpr - fpr
    best = np.nanmax(youden)
    candidates = np.where(np.isclose(youden, best, rtol=0.0, atol=1e-12))[0]
    finite = [int(index) for index in candidates.tolist() if np.isfinite(thresholds[index])]
    index = finite[0] if finite else int(candidates[0])
    return float(thresholds[index]) if np.isfinite(thresholds[index]) else 0.5


def classification_metrics(
    y_true: np.ndarray, p_hc: np.ndarray, threshold: float
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    p_hc = np.asarray(p_hc, dtype=np.float64)
    pred_hc = (p_hc >= float(threshold)).astype(np.int64)
    y_asd = (y_true == 0).astype(np.int64)
    pred_asd = (pred_hc == 0).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y_true, p_hc)),
        "bce": float(log_loss(y_true, np.clip(p_hc, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "f1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred_hc)),
    }


def fit_baseline_classifier(
    stable_train: torch.Tensor,
    y_train: torch.Tensor,
    stable_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Fit the exact 149-update stable-only linear readout."""
    set_all_seeds(seed)
    classifier = MLP(int(stable_train.shape[1]), int(CONFIG["num_classes"])).to(device)
    optimizer = Adam(classifier.parameters(), lr=float(CONFIG["lr_cls"]))
    criterion = torch.nn.BCELoss()
    best_val_loss = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(CONFIG["epochs_cls"])):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(stable_train)
        loss = criterion(output, F.one_hot(y_train, num_classes=2).float())
        loss.backward()
        if not finite_gradients(classifier) or not torch.isfinite(loss):
            raise FloatingPointError("non-finite baseline classifier gradient/loss")
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            val_output = classifier(stable_val)
            val_loss = float(
                criterion(val_output, F.one_hot(y_val, num_classes=2).float()).detach().cpu()
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state = clone_state_cpu(classifier.state_dict())
    if best_state is None or best_epoch is None:
        raise RuntimeError("baseline classifier did not produce a checkpoint")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        val_probs = classifier(stable_val)[:, -1].detach().cpu().numpy()
    return {
        "classifier_state": best_state,
        "val_bce": float(best_val_loss),
        "classifier_epoch": int(best_epoch),
        "val_probs": val_probs,
        "threshold": validation_threshold(y_val.detach().cpu().numpy(), val_probs),
    }


def fit_module_residual_classifier(
    baseline_fit: dict[str, Any],
    stable_train: torch.Tensor,
    meta_train: torch.Tensor,
    y_train: torch.Tensor,
    stable_val: torch.Tensor,
    meta_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Fit only the prescribed zero-initialized 528-d residual head."""
    if int(meta_train.shape[1]) != int(CONFIG["meta_dim"]):
        raise RuntimeError(f"meta dimension drifted to {meta_train.shape[1]}")
    meta_mean = meta_train.mean(dim=0)
    meta_std = meta_train.std(dim=0, unbiased=False)
    meta_std = torch.where(meta_std < 1e-6, torch.ones_like(meta_std), meta_std)
    train_z = (meta_train - meta_mean) / meta_std
    val_z = (meta_val - meta_mean) / meta_std
    if not bool(torch.isfinite(train_z).all()) or not bool(torch.isfinite(val_z).all()):
        raise FloatingPointError("non-finite train-only meta z-score")

    set_all_seeds(seed)
    residual = FrozenStableModuleResidualClassifier(
        baseline_state=baseline_fit["classifier_state"], meta_dim=int(CONFIG["meta_dim"])
    ).to(device)
    criterion = torch.nn.BCELoss()
    target_val = F.one_hot(y_val, num_classes=2).float()
    with torch.no_grad():
        baseline_output = predict_baseline_tensor(stable_val, baseline_fit["classifier_state"], device)
        epoch0_output = residual(stable_val, val_z)
        epoch0_prob_diff = float(
            (baseline_output - epoch0_output).abs().max().detach().cpu()
        )
        epoch0_bce = float(criterion(epoch0_output, target_val).detach().cpu())
        baseline_bce = float(criterion(baseline_output, target_val).detach().cpu())
    if epoch0_prob_diff > 1e-6 or abs(epoch0_bce - baseline_bce) > 1e-6:
        raise RuntimeError(
            "residual epoch-0 parity failed: "
            f"max_prob_diff={epoch0_prob_diff:.10g}, "
            f"bce={epoch0_bce:.10f} baseline={baseline_bce:.10f}"
        )
    if abs(epoch0_bce - float(baseline_fit["val_bce"])) > 1e-6:
        raise RuntimeError(
            "residual epoch-0 BCE differs from selected baseline: "
            f"{epoch0_bce:.10f} vs {float(baseline_fit['val_bce']):.10f}"
        )

    best_bce = epoch0_bce
    best_epoch = 0
    best_head_state = state_to_cpu(residual.residual_head.state_dict())
    optimizer = Adam(residual.residual_head.parameters(), lr=float(CONFIG["lr_cls"]))
    target_train = F.one_hot(y_train, num_classes=2).float()
    for epoch in range(1, int(CONFIG["epochs_cls"])):
        residual.train()
        optimizer.zero_grad()
        output = residual(stable_train, train_z)
        loss = criterion(output, target_train)
        loss.backward()
        if not finite_gradients(residual.residual_head) or not torch.isfinite(loss):
            raise FloatingPointError("non-finite residual classifier gradient/loss")
        optimizer.step()
        residual.eval()
        with torch.no_grad():
            val_output = residual(stable_val, val_z)
            val_bce = float(criterion(val_output, target_val).detach().cpu())
        if val_bce < best_bce:
            best_bce = val_bce
            best_epoch = int(epoch)
            best_head_state = state_to_cpu(residual.residual_head.state_dict())

    residual.residual_head.load_state_dict(best_head_state)
    residual.eval()
    with torch.no_grad():
        val_probs = residual(stable_val, val_z)[:, -1].detach().cpu().numpy()
    return {
        "stable_classifier_state": state_to_cpu(baseline_fit["classifier_state"]),
        "residual_head_state": best_head_state,
        "meta_mean": meta_mean.detach().cpu().clone(),
        "meta_std": meta_std.detach().cpu().clone(),
        "val_bce": float(best_bce),
        "residual_epoch": int(best_epoch),
        "val_probs": val_probs,
        "threshold": validation_threshold(y_val.detach().cpu().numpy(), val_probs),
        "epoch0_bce": float(epoch0_bce),
        "epoch0_prob_diff": float(epoch0_prob_diff),
    }


def train_ssl_epoch(
    encoder: VarCoNet,
    contrast_model: DualBranchContrast,
    optimizer: torch.optim.Optimizer,
    cohort: Cohort,
    task: Task,
    epoch_batches: list[list[int]],
    max_length: int,
    device: torch.device,
) -> float:
    encoder.train()
    losses: list[float] = []
    for selected in epoch_batches:
        selected_np = np.asarray(selected, dtype=np.int64)
        batch_array = np.asarray(cohort.abide1[task.train_raw[selected_np]], dtype=np.float32)
        batch_data = torch.from_numpy(batch_array)
        views = augment(batch_data, [int(CONFIG["min_length"]), max_length], device)
        optimizer.zero_grad()
        z1 = encoder(views[0])
        z2 = encoder(views[1])
        loss = contrast_model(z1, z2)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite SSL loss")
        loss.backward()
        if not finite_gradients(encoder):
            raise FloatingPointError("non-finite SSL gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        del batch_data, views, z1, z2, loss
    if not losses:
        raise RuntimeError("empty SSL epoch")
    return float(np.mean(losses))


def checkpoint_paths(repeat: int) -> tuple[Path, Path]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    worker = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"r{int(repeat)}")
    return (
        CHECKPOINT_DIR / f"worker_{worker}_latest.pt",
        CHECKPOINT_DIR / f"worker_{worker}_best.pt",
    )


def task_identity(task: Task) -> dict[str, Any]:
    return {
        "repeat": int(task.repeat),
        "scope": str(task.scope),
        "fold": str(task.fold),
        "task_seed": int(task.seed),
    }


def identity_matches(payload: dict[str, Any], task: Task) -> bool:
    identity = task_identity(task)
    return all(payload.get(key) == value for key, value in identity.items())


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


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def branch_epoch(branch: dict[str, Any] | None) -> int:
    if branch is None:
        return -1
    return int(branch.get("encoder_epoch", -1))


def merge_best_branch(
    current: dict[str, Any] | None, candidate: dict[str, Any] | None
) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None or float(candidate["val_bce"]) < float(current["val_bce"]):
        return state_to_cpu(candidate)
    return current


def ssl_checkpoint_payload(
    task: Task,
    current_epoch: int,
    encoder: VarCoNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_baseline: dict[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "phase": "ssl",
        "saved_unix": time.time(),
        **task_identity(task),
        "epoch": int(current_epoch),
        "epochs": int(CONFIG["epochs"]),
        "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "optimizer_state_dict": state_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": state_to_cpu(scheduler.state_dict()),
        "best_baseline": state_to_cpu(best_baseline),
        "rng_state": state_to_cpu(capture_rng_state()),
        "elapsed_seconds": float(elapsed_seconds),
    }


def residual_checkpoint_payload(
    task: Task,
    current_epoch: int,
    encoder: VarCoNet,
    best_baseline: dict[str, Any],
    module_labels: np.ndarray,
    module_diagnostics: dict[str, float],
    meta_mean: torch.Tensor,
    meta_std: torch.Tensor,
    current_head: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    best_head: dict[str, torch.Tensor],
    best_residual_epoch: int,
    best_val_bce: float,
    epoch0_bce: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "format_version": 4,
        "phase": "high_order_residual",
        "saved_unix": time.time(),
        **task_identity(task),
        "epoch": int(current_epoch),
        "epochs": int(CONFIG["epochs_cls"]),
        "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "best_baseline": state_to_cpu(best_baseline),
        "module_labels": np.asarray(module_labels, dtype=np.int64).copy(),
        "module_diagnostics": state_to_cpu(module_diagnostics),
        "meta_mean": meta_mean.detach().cpu().clone(),
        "meta_std": meta_std.detach().cpu().clone(),
        "current_head_state": state_to_cpu(current_head),
        "optimizer_state_dict": state_to_cpu(optimizer.state_dict()),
        "best_head_state": state_to_cpu(best_head),
        "best_residual_epoch": int(best_residual_epoch),
        "best_val_bce": float(best_val_bce),
        "epoch0_bce": float(epoch0_bce),
        "rng_state": state_to_cpu(capture_rng_state()),
        "elapsed_seconds": float(elapsed_seconds),
    }


def save_checkpoint_payload(path: Path, kind: str, payload: dict[str, Any]) -> None:
    atomic_torch_save(path, payload)
    print(
        f"CHECKPOINT_{kind.upper()} repeat={payload['repeat']} scope={payload['scope']} "
        f"fold={payload['fold']} phase={payload['phase']} epoch={payload['epoch']} path={path}",
        flush=True,
    )


def clean_task_checkpoints(repeat: int, task: Task | None = None) -> None:
    latest, best = checkpoint_paths(repeat)
    for path in (
        latest,
        best,
        latest.with_suffix(latest.suffix + ".tmp"),
        best.with_suffix(best.suffix + ".tmp"),
    ):
        if not path.exists():
            continue
        if task is not None and path.suffix == ".pt":
            try:
                if not identity_matches(torch_load_cpu(path), task):
                    # A later task may already have written this worker's
                    # checkpoint when a completed earlier task is skipped.
                    continue
            except Exception:
                # Corrupt/stale checkpoint belonging to the completed task is
                # safe to discard; a different valid task checkpoint is kept.
                pass
        path.unlink(missing_ok=True)


def gpu_memory_gb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = float(torch.cuda.memory_allocated(device)) / (1024**3)
    reserved = float(torch.cuda.memory_reserved(device)) / (1024**3)
    return allocated, reserved


def make_model_config(cohort: Cohort) -> tuple[dict[str, int], dict[str, Any]]:
    with (EXPERIMENT_DIR / "best_params_VarCoNet_AICHA.pkl").open("rb") as handle:
        best_params = pickle.load(handle)
    model_config = {
        "layers": int(best_params["layers"]),
        "n_heads": int(best_params["n_heads"]),
        "dim_feedforward": int(best_params["dim_feedforward"]),
        "max_length": int(cohort.abide1.shape[1]),
    }
    return model_config, best_params


def create_encoder(cohort: Cohort, device: torch.device) -> VarCoNet:
    model_config, _ = make_model_config(cohort)
    return VarCoNet(model_config, int(cohort.abide1.shape[2])).to(device)


def predict_baseline(
    stable: torch.Tensor, classifier_state: dict[str, torch.Tensor], device: torch.device
) -> np.ndarray:
    classifier = MLP(int(stable.shape[1]), 2).to(device)
    classifier.load_state_dict(classifier_state)
    classifier.eval()
    with torch.no_grad():
        probabilities = classifier(stable)[:, -1].detach().cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)


def predict_baseline_tensor(
    stable: torch.Tensor, classifier_state: dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    classifier = MLP(int(stable.shape[1]), 2).to(device)
    classifier.load_state_dict(classifier_state)
    classifier.eval()
    return classifier(stable)


def predict_treatment(
    stable: torch.Tensor,
    meta: torch.Tensor,
    branch: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    meta_mean = branch["meta_mean"].to(device)
    meta_std = branch["meta_std"].to(device)
    meta_z = (meta - meta_mean) / meta_std
    classifier = FrozenStableModuleResidualClassifier(
        baseline_state=branch["stable_classifier_state"],
        meta_dim=int(CONFIG["meta_dim"]),
    ).to(device)
    classifier.residual_head.load_state_dict(branch["residual_head_state"])
    classifier.eval()
    with torch.no_grad():
        probabilities = classifier(stable, meta_z)[:, -1].detach().cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)


def evaluate_selected_branch(
    cohort: Cohort,
    branch: dict[str, Any],
    method: str,
    data: np.ndarray,
    raw_indices: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """Run a selected model once, only after all 50 epochs are complete."""
    encoder = create_encoder(cohort, device)
    encoder.load_state_dict(branch["encoder_state"])
    if method == "baseline":
        stable = extract_stable_features(encoder, data, raw_indices, device)
        probabilities = predict_baseline(stable, branch["classifier_state"], device)
    elif method == TREATMENT_METHOD:
        stable = extract_stable_features(encoder, data, raw_indices, device)
        pooler = ModulePairPooler(
            branch["module_labels"],
            roi_num=int(cohort.abide1.shape[2]),
            n_modules=int(CONFIG["n_modules"]),
        ).to(device)
        meta = pooler(stable)
        probabilities = predict_treatment(stable, meta, branch, device)
    else:
        raise ValueError(f"unknown method {method}")
    metrics = classification_metrics(labels, probabilities, float(branch["threshold"]))
    del stable, encoder
    if "meta" in locals():
        del meta
    if "pooler" in locals():
        del pooler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, probabilities


def module_metric_fields(
    method: str, diagnostics: dict[str, Any] | None
) -> dict[str, float]:
    names = (
        "n_modules",
        "meta_dim",
        "module_size_min",
        "module_size_mean",
        "module_size_max",
        "singleton_modules",
        "empty_meta_bins",
        "affinity_mean",
        "affinity_std",
        "affinity_max",
        "residual_head_l2",
    )
    if method == BASELINE_METHOD or diagnostics is None:
        return {name: float("nan") for name in names}
    return {
        "n_modules": float(CONFIG["n_modules"]),
        "meta_dim": float(CONFIG["meta_dim"]),
        "module_size_min": float(diagnostics["module_size_min"]),
        "module_size_mean": float(diagnostics["module_size_mean"]),
        "module_size_max": float(diagnostics["module_size_max"]),
        "singleton_modules": float(diagnostics["singleton_modules"]),
        "empty_meta_bins": float(diagnostics["empty_meta_bins"]),
        "affinity_mean": float(diagnostics["affinity_mean"]),
        "affinity_std": float(diagnostics["affinity_std"]),
        "affinity_max": float(diagnostics["affinity_max"]),
        "residual_head_l2": float(diagnostics.get("residual_head_l2", float("nan"))),
    }


def metric_row(
    method: str,
    scope: str,
    repeat: int,
    fold: str,
    metrics: dict[str, float],
    branch: dict[str, Any],
    n_test: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "scope": scope,
        "repeat": int(repeat),
        "fold": str(fold),
        "auc": float(metrics["auc"]),
        "bce": float(metrics["bce"]),
        "f1": float(metrics["f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "selected_encoder_epoch": float(branch["encoder_epoch"]),
        "selected_classifier_epoch": float(branch.get("classifier_epoch", float("nan"))),
        "selected_residual_epoch": float(branch.get("residual_epoch", float("nan"))),
        "threshold_hc": float(branch["threshold"]),
        "task_seconds": float(branch.get("task_seconds", float("nan"))),
        "n_test": int(n_test),
        "status": "PASS",
    }
    row.update(module_metric_fields(method, branch.get("module_diagnostics")))
    return row


def make_baseline_branch(
    encoder: VarCoNet,
    encoder_epoch: int,
    fit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()),
        "classifier_state": state_to_cpu(fit["classifier_state"]),
        "val_bce": float(fit["val_bce"]),
        "encoder_epoch": int(encoder_epoch),
        "classifier_epoch": int(fit["classifier_epoch"]),
        "threshold": float(fit["threshold"]),
        "val_probs": np.asarray(fit["val_probs"], dtype=np.float64).copy(),
        "module_diagnostics": None,
    }


def make_treatment_branch(
    encoder: VarCoNet,
    encoder_epoch: int,
    fit: dict[str, Any],
    module_labels: np.ndarray,
    diagnostics: dict[str, float],
    baseline_classifier_epoch: int,
) -> dict[str, Any]:
    residual_head_l2 = float(
        torch.sqrt(
            sum(value.float().pow(2).sum() for value in fit["residual_head_state"].values())
        ).detach().cpu()
    )
    module_diagnostics = {key: float(value) for key, value in diagnostics.items()}
    module_diagnostics["residual_head_l2"] = residual_head_l2
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()),
        "stable_classifier_state": state_to_cpu(fit["stable_classifier_state"]),
        "module_labels": np.asarray(module_labels, dtype=np.int64).copy(),
        "residual_head_state": state_to_cpu(fit["residual_head_state"]),
        "meta_mean": fit["meta_mean"].detach().cpu().clone(),
        "meta_std": fit["meta_std"].detach().cpu().clone(),
        "val_bce": float(fit["val_bce"]),
        "encoder_epoch": int(encoder_epoch),
        "classifier_epoch": int(baseline_classifier_epoch),
        "residual_epoch": int(fit["residual_epoch"]),
        "threshold": float(fit["threshold"]),
        "val_probs": np.asarray(fit["val_probs"], dtype=np.float64).copy(),
        "epoch0_bce": float(fit["epoch0_bce"]),
        "module_diagnostics": module_diagnostics,
        "task_seconds": float(fit.get("task_seconds", float("nan"))),
    }


def train_shared_task(
    cohort: Cohort,
    task: Task,
    device: torch.device,
) -> dict[str, Any]:
    """Train one shared SSL trajectory, then one frozen-stable residual."""
    task_start = time.time()
    print(
        f"TASK_START repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"seed={task.seed} device={device} shared_encoder=Y",
        flush=True,
    )
    set_all_seeds(task.seed)
    model_config, best_params = make_model_config(cohort)
    max_length = int(model_config["max_length"])
    roi_num = int(cohort.abide1.shape[2])
    encoder = VarCoNet(model_config, roi_num).to(device)
    contrast_model = DualBranchContrast(
        loss=InfoNCE(tau=float(best_params["tau"])), mode="L2L"
    ).to(device)
    optimizer = Adam(encoder.parameters(), lr=float(best_params["lr"]))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_start_lr=1e-5,
        warmup_epochs=int(CONFIG["warm_up_epochs"]),
        max_epochs=int(CONFIG["epochs"]),
    )
    schedule = paired_batch_schedule(
        task.train_names, len(task.train_raw), int(CONFIG["epochs"]), task.seed
    )
    latest_path, best_path = checkpoint_paths(task.repeat)

    # Load the newest checkpoint for this exact task.  A best checkpoint is
    # retained as a fallback, while latest always takes precedence.
    source: dict[str, Any] | None = None
    for path in (latest_path, best_path):
        if path.exists():
            payload = torch_load_cpu(path)
            if identity_matches(payload, task):
                source = payload
                break
            if path == latest_path:
                path.unlink(missing_ok=True)
    best_baseline: dict[str, Any] | None = None
    phase = "ssl"
    start_epoch = 1
    resumed_elapsed = 0.0
    if source is not None:
        phase = str(source.get("phase", "ssl"))
        resumed_elapsed = float(source.get("elapsed_seconds", 0.0))
        task_start -= resumed_elapsed
        best_baseline = state_to_cpu(source.get("best_baseline"))
        if source.get("rng_state") is not None:
            restore_rng_state(source["rng_state"])
        if phase == "ssl":
            encoder.load_state_dict(source["encoder_state_dict"])
            optimizer.load_state_dict(source["optimizer_state_dict"])
            optimizer_to_device(optimizer, device)
            scheduler.load_state_dict(source["scheduler_state_dict"])
            start_epoch = int(source["epoch"]) + 1
        elif phase == "high_order_residual":
            encoder.load_state_dict(source["encoder_state_dict"])
            start_epoch = int(CONFIG["epochs"]) + 1
            print(
                f"RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} "
                f"phase=residual from_epoch={int(source['epoch']) + 1}", flush=True
            )
        else:
            raise RuntimeError(f"unknown checkpoint phase {phase!r}")
        print(
            f"RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} "
            f"phase={phase} elapsed_seconds={resumed_elapsed:.1f}", flush=True
        )

    train_labels = torch.as_tensor(task.train_labels, dtype=torch.long, device=device)
    val_labels = torch.as_tensor(task.val_labels, dtype=torch.long, device=device)
    ssl_losses: list[float] = []
    for epoch in range(start_epoch, int(CONFIG["epochs"]) + 1):
        epoch_start = time.time()
        ssl_loss = train_ssl_epoch(
            encoder,
            contrast_model,
            optimizer,
            cohort,
            task,
            schedule[epoch - 1],
            max_length,
            device,
        )
        scheduler.step()
        ssl_losses.append(ssl_loss)

        # Baseline-only epoch selection.  Dynamic features are intentionally
        # not extracted until the single baseline-best encoder is frozen.
        stable_train = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
        stable_val = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
        trajectory_rng = capture_rng_state()
        baseline_fit = fit_baseline_classifier(
            stable_train,
            train_labels,
            stable_val,
            val_labels,
            device,
            task.seed + 300000 + epoch * 2,
        )
        restore_rng_state(trajectory_rng)

        baseline_candidate = make_baseline_branch(encoder, epoch, baseline_fit)
        base_is_best = (
            best_baseline is None
            or float(baseline_candidate["val_bce"]) < float(best_baseline["val_bce"])
        )
        if base_is_best:
            best_baseline = baseline_candidate

        elapsed = time.time() - task_start
        if base_is_best:
            save_checkpoint_payload(
                best_path,
                "best",
                ssl_checkpoint_payload(task, epoch, encoder, optimizer, scheduler, best_baseline, elapsed),
            )
        if epoch % int(CONFIG["checkpoint_interval"]) == 0:
            save_checkpoint_payload(
                latest_path,
                "latest",
                ssl_checkpoint_payload(task, epoch, encoder, optimizer, scheduler, best_baseline, elapsed),
            )

        allocated, reserved = gpu_memory_gb(device)
        print(
            f"[repeat={task.repeat} fold={task.fold}] [epoch={epoch}/{CONFIG['epochs']}] "
            f"ssl_loss={ssl_loss:.6f} lr={optimizer.param_groups[0]['lr']:.8f} "
            f"base_val_bce={baseline_fit['val_bce']:.6f} "
            f"base_best={'Y' if base_is_best else 'N'} "
            f"base_cls_epoch={baseline_fit['classifier_epoch']} "
            f"epoch_sec={time.time() - epoch_start:.1f} "
            f"task_elapsed_min={elapsed / 60.0:.1f} "
            f"gpu_alloc_gb={allocated:.3f} gpu_reserved_gb={reserved:.3f}",
            flush=True,
        )
        print(
            f"[ssl_epoch={epoch}/{CONFIG['epochs']}] ssl_loss={ssl_loss:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f} "
            f"base_val_bce={baseline_fit['val_bce']:.6f} "
            f"base_best={'Y' if base_is_best else 'N'} "
            f"base_cls_epoch={baseline_fit['classifier_epoch']} "
            f"epoch_sec={time.time() - epoch_start:.1f} "
            f"elapsed_min={elapsed / 60.0:.1f} "
            f"gpu_alloc_gb={allocated:.3f} gpu_reserved_gb={reserved:.3f}",
            flush=True,
        )
        del stable_train, stable_val
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_baseline is None:
        raise RuntimeError("no validation-selected baseline branch")

    # Freeze exactly the baseline-selected encoder/classifier.  Only now do we
    # discover train-only consensus modules and fit one 528-parameter residual.
    encoder.load_state_dict(best_baseline["encoder_state"])
    stable_module = extract_stable_unique_features(
        encoder, cohort.abide1, task.module_raw, device
    )
    if phase == "high_order_residual" and source is not None and source.get("module_labels") is not None:
        module_labels = np.asarray(source["module_labels"], dtype=np.int64)
        module_diagnostics = {
            key: float(value) for key, value in source["module_diagnostics"].items()
        }
    else:
        discovery_start = time.time()
        module_labels, module_diagnostics = discover_consensus_modules(
            stable_module, roi_num=roi_num, n_modules=int(CONFIG["n_modules"]), seed=int(task.seed)
        )
        print(
            "[MODULE_DISCOVERY] "
            f"repeat={task.repeat} fold={task.fold} n_unique_subjects={len(task.module_raw)} "
            f"n_modules={int(module_diagnostics['n_modules'])} "
            f"module_size_min={int(module_diagnostics['module_size_min'])} "
            f"module_size_mean={module_diagnostics['module_size_mean']:.3f} "
            f"module_size_max={int(module_diagnostics['module_size_max'])} "
            f"singleton_modules={int(module_diagnostics['singleton_modules'])} "
            f"meta_dim={int(CONFIG['meta_dim'])} empty_meta_bins=0 "
            f"affinity_mean={module_diagnostics['affinity_mean']:.8f} "
            f"affinity_std={module_diagnostics['affinity_std']:.8f} "
            f"affinity_max={module_diagnostics['affinity_max']:.8f} "
            f"elapsed_sec={time.time() - discovery_start:.2f}", flush=True
        )
    if module_labels.shape != (roi_num,) or len(np.unique(module_labels)) != int(CONFIG["n_modules"]):
        raise RuntimeError("module discovery did not return exactly 32 ROI labels")
    pooler = ModulePairPooler(
        module_labels, roi_num=roi_num, n_modules=int(CONFIG["n_modules"])
    ).to(device)
    module_diagnostics["empty_meta_bins"] = float(pooler.empty_meta_bins)
    stable_train = extract_stable_features(encoder, cohort.abide1, task.train_raw, device)
    stable_val = extract_stable_features(encoder, cohort.abide1, task.val_raw, device)
    meta_train = pooler(stable_train)
    meta_val = pooler(stable_val)
    if int(meta_train.shape[1]) != int(CONFIG["meta_dim"]):
        raise RuntimeError(f"meta dimension drifted to {meta_train.shape[1]}")
    module_diagnostics["meta_dim"] = float(meta_train.shape[1])
    if not bool(torch.isfinite(meta_train).all()) or not bool(torch.isfinite(meta_val).all()):
        raise FloatingPointError("non-finite meta-connectome")
    meta_mean = meta_train.mean(dim=0)
    meta_std = meta_train.std(dim=0, unbiased=False)
    meta_std = torch.where(meta_std < 1e-6, torch.ones_like(meta_std), meta_std)
    train_z = (meta_train - meta_mean) / meta_std
    val_z = (meta_val - meta_mean) / meta_std
    if not bool(torch.isfinite(train_z).all()) or not bool(torch.isfinite(val_z).all()):
        raise FloatingPointError("meta z-score is non-finite")

    residual = FrozenStableModuleResidualClassifier(
        baseline_state=best_baseline["classifier_state"], meta_dim=int(CONFIG["meta_dim"])
    ).to(device)
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        baseline_val_output = predict_baseline_tensor(
            stable_val, best_baseline["classifier_state"], device
        )
        epoch0_output = residual(stable_val, val_z)
        epoch0_prob_diff = float(
            (baseline_val_output - epoch0_output).abs().max().detach().cpu()
        )
        epoch0_bce = float(
            criterion(epoch0_output, F.one_hot(val_labels, num_classes=2).float()).cpu()
        )
        baseline_val_bce = float(
            criterion(baseline_val_output, F.one_hot(val_labels, num_classes=2).float()).cpu()
        )
    if epoch0_prob_diff > 1e-6 or abs(epoch0_bce - baseline_val_bce) > 1e-6:
        raise RuntimeError(
            f"residual epoch-0 parity failed: prob_diff={epoch0_prob_diff:.10g} "
            f"bce={epoch0_bce:.10f} baseline={baseline_val_bce:.10f}"
        )
    if abs(epoch0_bce - float(best_baseline["val_bce"])) > 1e-6:
        raise RuntimeError("residual epoch-0 BCE differs from selected baseline")

    optimizer_res = Adam(residual.residual_head.parameters(), lr=float(CONFIG["lr_cls"]))
    best_head = state_to_cpu(residual.residual_head.state_dict())
    best_residual_epoch = 0
    best_val_bce = epoch0_bce
    residual_start = 1
    if phase == "high_order_residual" and source is not None:
        module_labels = np.asarray(source["module_labels"], dtype=np.int64)
        module_diagnostics = {
            key: float(value) for key, value in source["module_diagnostics"].items()
        }
        meta_mean = source["meta_mean"].to(device)
        meta_std = source["meta_std"].to(device)
        train_z = (meta_train - meta_mean) / meta_std
        val_z = (meta_val - meta_mean) / meta_std
        residual.residual_head.load_state_dict(source["current_head_state"])
        optimizer_res.load_state_dict(source["optimizer_state_dict"])
        optimizer_to_device(optimizer_res, device)
        best_head = state_to_cpu(source["best_head_state"])
        best_residual_epoch = int(source["best_residual_epoch"])
        best_val_bce = float(source["best_val_bce"])
        epoch0_bce = float(source["epoch0_bce"])
        residual_start = int(source["epoch"]) + 1
        print(
            f"RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} "
            f"phase=high_order_residual from_epoch={residual_start}", flush=True
        )
    print(
        f"[HIGH_ORDER_RESIDUAL] repeat={task.repeat} fold={task.fold} epoch=0 "
        f"train_bce=NA val_bce={epoch0_bce:.6f} best=Y best_epoch=0 "
        f"head_l2=0.000000 elapsed_sec={time.time() - task_start:.2f}", flush=True
    )
    if not (phase == "high_order_residual" and source is not None):
        # Persist epoch 0 immediately so an interruption before residual epoch
        # 10 can resume without re-running module discovery or SSL.
        save_checkpoint_payload(
            best_path,
            "best",
            residual_checkpoint_payload(
                task, 0, encoder, best_baseline, module_labels,
                module_diagnostics, meta_mean, meta_std,
                state_to_cpu(residual.residual_head.state_dict()), optimizer_res,
                best_head, best_residual_epoch, best_val_bce, epoch0_bce,
                time.time() - task_start,
            ),
        )
    for residual_epoch in range(residual_start, int(CONFIG["epochs_cls"])):
        residual_start_time = time.time()
        residual.train()
        optimizer_res.zero_grad()
        train_output = residual(stable_train, train_z)
        residual_loss = criterion(train_output, F.one_hot(train_labels, num_classes=2).float())
        residual_loss.backward()
        if not finite_gradients(residual.residual_head) or not torch.isfinite(residual_loss):
            raise FloatingPointError("non-finite high-order residual gradient/loss")
        optimizer_res.step()
        residual.eval()
        with torch.no_grad():
            val_output = residual(stable_val, val_z)
            val_bce = float(criterion(val_output, F.one_hot(val_labels, num_classes=2).float()).cpu())
        is_best = val_bce < best_val_bce
        if is_best:
            best_val_bce = val_bce
            best_residual_epoch = int(residual_epoch)
            best_head = state_to_cpu(residual.residual_head.state_dict())
        head_l2 = float(torch.linalg.vector_norm(residual.residual_head.weight.detach()).cpu())
        elapsed = time.time() - task_start
        if is_best or residual_epoch % int(CONFIG["checkpoint_interval"]) == 0:
            payload = residual_checkpoint_payload(
                task, residual_epoch, encoder, best_baseline, module_labels,
                module_diagnostics, meta_mean, meta_std,
                state_to_cpu(residual.residual_head.state_dict()), optimizer_res,
                best_head, best_residual_epoch, best_val_bce, epoch0_bce, elapsed,
            )
            save_checkpoint_payload(
                best_path if is_best else latest_path,
                "best" if is_best else "latest",
                payload,
            )
        if residual_epoch % 10 == 0 or is_best:
            print(
                f"[HIGH_ORDER_RESIDUAL] repeat={task.repeat} fold={task.fold} "
                f"epoch={residual_epoch}/{CONFIG['epochs_cls']-1} "
                f"train_bce={float(residual_loss.detach().cpu()):.6f} "
                f"val_bce={val_bce:.6f} best={'Y' if is_best else 'N'} "
                f"best_epoch={best_residual_epoch} head_l2={head_l2:.6f} "
                f"elapsed_sec={time.time()-residual_start_time:.2f}",
                flush=True,
            )
    residual.residual_head.load_state_dict(best_head)
    residual.eval()
    with torch.no_grad():
        val_probs = residual(stable_val, val_z)[:, -1].detach().cpu().numpy()
    treatment_fit = {
        "stable_classifier_state": state_to_cpu(best_baseline["classifier_state"]),
        "residual_head_state": best_head,
        "meta_mean": meta_mean.detach().cpu().clone(),
        "meta_std": meta_std.detach().cpu().clone(),
        "val_bce": float(best_val_bce),
        "residual_epoch": int(best_residual_epoch),
        "val_probs": val_probs,
        "threshold": validation_threshold(task.val_labels, val_probs),
        "epoch0_bce": float(epoch0_bce),
        "task_seconds": float(time.time() - task_start),
    }
    treatment = make_treatment_branch(
        encoder, best_baseline["encoder_epoch"], treatment_fit, module_labels,
        module_diagnostics,
        int(best_baseline["classifier_epoch"])
    )
    treatment["encoder_state"] = clone_state_cpu(best_baseline["encoder_state"])
    treatment["task_seconds"] = float(time.time() - task_start)
    best_baseline["task_seconds"] = float(time.time() - task_start)
    print(
        f"RESIDUAL_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"selected_encoder_epoch={best_baseline['encoder_epoch']} "
        f"selected_classifier_epoch={best_baseline['classifier_epoch']} "
        f"selected_residual_epoch={treatment['residual_epoch']} val_bce={treatment['val_bce']:.6f}", flush=True
    )
    result = {
        "task": task,
        "baseline": best_baseline,
        "treatment": treatment,
        "mean_ssl_loss": float(np.mean(ssl_losses)) if ssl_losses else float("nan"),
        "elapsed_seconds": float(time.time() - task_start),
    }
    print(
        f"TASK_TRAINING_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"base_epoch={best_baseline['encoder_epoch']} dyn_epoch={treatment['encoder_epoch']} "
        f"base_val_bce={best_baseline['val_bce']:.6f} "
        f"high_order_val_bce={treatment['val_bce']:.6f}",
        flush=True,
    )
    del encoder, contrast_model, optimizer, scheduler, stable_module, pooler, residual
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


FLOAT_COLUMNS = {
    "auc",
    "bce",
    "f1",
    "balanced_accuracy",
    "selected_encoder_epoch",
    "selected_classifier_epoch",
    "selected_residual_epoch",
    "threshold_hc",
    "n_modules",
    "meta_dim",
    "module_size_min",
    "module_size_mean",
    "module_size_max",
    "singleton_modules",
    "empty_meta_bins",
    "affinity_mean",
    "affinity_std",
    "affinity_max",
    "residual_head_l2",
    "task_seconds",
}
INT_COLUMNS = {"repeat", "n_test"}
PREDICTION_FIELDS = (
    "scope",
    "method",
    "repeat",
    "fold",
    "subject_index",
    "raw_index",
    "y",
    "p_hc",
    "threshold",
)


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
            raise RuntimeError(f"unexpected worker metrics schema: {path}")
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


def task_row_complete(
    rows: list[dict[str, Any]], scope: str, repeat: int, fold: str
) -> bool:
    methods = {
        row["method"]
        for row in rows
        if row["scope"] == scope and int(row["repeat"]) == int(repeat) and row["fold"] == str(fold)
    }
    return methods == set(METHODS)


def clear_metric_rows(
    rows: list[dict[str, Any]], scope: str, repeat: int, fold: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not (
            row["scope"] == scope
            and int(row["repeat"]) == int(repeat)
            and row["fold"] == str(fold)
        )
    ]


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTION_FIELDS):
            raise RuntimeError(f"unexpected prediction shard schema: {path}")
        n = len(archive["scope"])
        if any(len(archive[key]) != n for key in PREDICTION_FIELDS):
            raise RuntimeError(f"inconsistent prediction shard lengths: {path}")
        records: list[dict[str, Any]] = []
        for index in range(n):
            records.append(
                {
                    "scope": str(archive["scope"][index]),
                    "method": str(archive["method"][index]),
                    "repeat": int(archive["repeat"][index]),
                    "fold": str(archive["fold"][index]),
                    "subject_index": int(archive["subject_index"][index]),
                    "raw_index": int(archive["raw_index"][index]),
                    "y": int(archive["y"][index]),
                    "p_hc": float(archive["p_hc"][index]),
                    "threshold": float(archive["threshold"][index]),
                }
            )
    return records


def write_prediction_records_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "scope": np.asarray([record["scope"] for record in records], dtype="<U24"),
        "method": np.asarray([record["method"] for record in records], dtype="<U32"),
        "repeat": np.asarray([record["repeat"] for record in records], dtype=np.int16),
        "fold": np.asarray([record["fold"] for record in records], dtype="<U16"),
        "subject_index": np.asarray([record["subject_index"] for record in records], dtype=np.int32),
        "raw_index": np.asarray([record["raw_index"] for record in records], dtype=np.int32),
        "y": np.asarray([record["y"] for record in records], dtype=np.int8),
        "p_hc": np.asarray([record["p_hc"] for record in records], dtype=np.float64),
        "threshold": np.asarray([record["threshold"] for record in records], dtype=np.float64),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clear_prediction_group(
    records: list[dict[str, Any]],
    scope: str,
    repeat: int,
    fold: str,
    method: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if not (
            record["scope"] == scope
            and int(record["repeat"]) == int(repeat)
            and record["fold"] == str(fold)
            and (method is None or record["method"] == method)
        )
    ]


def add_prediction_group(
    records: list[dict[str, Any]],
    scope: str,
    method: str,
    repeat: int,
    fold: str,
    subject_indices: np.ndarray,
    raw_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | np.ndarray,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in records
        if not (
            record["scope"] == scope
            and record["method"] == method
            and int(record["repeat"]) == int(repeat)
            and record["fold"] == str(fold)
        )
    ]
    subject_indices = np.asarray(subject_indices, dtype=np.int64)
    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.isscalar(threshold):
        thresholds = np.full(len(labels), float(threshold), dtype=np.float64)
    else:
        thresholds = np.asarray(threshold, dtype=np.float64)
    n = len(labels)
    if not all(len(values) == n for values in (subject_indices, raw_indices, probabilities, thresholds)):
        raise RuntimeError("prediction lengths do not agree")
    records.extend(
        {
            "scope": str(scope),
            "method": str(method),
            "repeat": int(repeat),
            "fold": str(fold),
            "subject_index": int(subject_indices[index]),
            "raw_index": int(raw_indices[index]),
            "y": int(labels[index]),
            "p_hc": float(probabilities[index]),
            "threshold": float(thresholds[index]),
        }
        for index in range(n)
    )
    return records


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def subject_indices_for_task(cohort: Cohort, task: Task) -> np.ndarray:
    if task.scope == "internal_cv":
        lookup = {int(raw): index for index, raw in enumerate(cohort.internal_raw)}
        return np.asarray([lookup[int(raw)] for raw in task.test_raw], dtype=np.int64)
    if task.scope == "caltech_external":
        lookup = {int(raw): index for index, raw in enumerate(cohort.caltech_raw)}
        return np.asarray([lookup[int(raw)] for raw in task.test_raw], dtype=np.int64)
    raise ValueError(task.scope)


def metrics_with_thresholds(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pred_hc = (probabilities >= thresholds).astype(np.int64)
    y_asd = (labels == 0).astype(np.int64)
    pred_asd = (pred_hc == 0).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "bce": float(log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "f1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred_hc)),
    }


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def assert_shared_selection(task_result: dict[str, Any]) -> None:
    baseline = task_result["baseline"]
    treatment = task_result["treatment"]
    if int(baseline["encoder_epoch"]) != int(treatment["encoder_epoch"]):
        raise RuntimeError("INTEGRITY_FAIL: baseline/treatment selected encoder epochs differ")
    if int(baseline["classifier_epoch"]) != int(treatment["classifier_epoch"]):
        raise RuntimeError("INTEGRITY_FAIL: baseline/treatment selected classifier epochs differ")
    for key, left in baseline["encoder_state"].items():
        right = treatment["encoder_state"].get(key)
        if right is None or not torch.equal(left, right):
            raise RuntimeError("INTEGRITY_FAIL: baseline/treatment encoder states differ")
    for key, left in baseline["classifier_state"].items():
        right = treatment["stable_classifier_state"].get(key)
        if right is None or not torch.equal(left, right):
            raise RuntimeError("INTEGRITY_FAIL: baseline/treatment classifier states differ")


def build_internal_oof_row(
    cohort: Cohort,
    records: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    method: str,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        record
        for record in records
        if record["scope"] == "internal_fold"
        and record["method"] == method
        and int(record["repeat"]) == int(repeat)
    ]
    selected.sort(key=lambda record: int(record["subject_index"]))
    if len(selected) != len(cohort.internal_raw):
        raise RuntimeError(
            f"repeat {repeat} {method} OOF has {len(selected)} predictions, "
            f"expected {len(cohort.internal_raw)}"
        )
    subject_index = np.asarray([record["subject_index"] for record in selected], dtype=np.int64)
    if not np.array_equal(subject_index, np.arange(len(cohort.internal_raw), dtype=np.int64)):
        raise RuntimeError(f"repeat {repeat} {method} OOF indices are not an exact partition")
    labels = np.asarray([record["y"] for record in selected], dtype=np.int64)
    probabilities = np.asarray([record["p_hc"] for record in selected], dtype=np.float64)
    thresholds = np.asarray([record["threshold"] for record in selected], dtype=np.float64)
    metrics = metrics_with_thresholds(labels, probabilities, thresholds)
    task_rows = [
        row
        for row in fold_rows
        if row["scope"] == "internal_fold"
        and row["method"] == method
        and int(row["repeat"]) == int(repeat)
    ]
    if len(task_rows) != 10:
        raise RuntimeError(f"repeat {repeat} {method} has {len(task_rows)} fold rows, expected 10")
    if method == "baseline":
        branch = {
            "encoder_epoch": mean_or_nan([row["selected_encoder_epoch"] for row in task_rows]),
            "classifier_epoch": mean_or_nan(
                [row["selected_classifier_epoch"] for row in task_rows]
            ),
            "threshold": float("nan"),
            "module_diagnostics": None,
        }
    else:
        branch = {
            "encoder_epoch": mean_or_nan([row["selected_encoder_epoch"] for row in task_rows]),
            "classifier_epoch": mean_or_nan(
                [row["selected_classifier_epoch"] for row in task_rows]
            ),
            "residual_epoch": mean_or_nan(
                [row["selected_residual_epoch"] for row in task_rows]
            ),
            "threshold": float("nan"),
            "module_diagnostics": {
                "n_modules": mean_or_nan([row["n_modules"] for row in task_rows]),
                "meta_dim": mean_or_nan([row["meta_dim"] for row in task_rows]),
                "module_size_min": mean_or_nan([row["module_size_min"] for row in task_rows]),
                "module_size_mean": mean_or_nan([row["module_size_mean"] for row in task_rows]),
                "module_size_max": mean_or_nan([row["module_size_max"] for row in task_rows]),
                "singleton_modules": mean_or_nan([row["singleton_modules"] for row in task_rows]),
                "empty_meta_bins": mean_or_nan([row["empty_meta_bins"] for row in task_rows]),
                "affinity_mean": mean_or_nan([row["affinity_mean"] for row in task_rows]),
                "affinity_std": mean_or_nan([row["affinity_std"] for row in task_rows]),
                "affinity_max": mean_or_nan([row["affinity_max"] for row in task_rows]),
                "residual_head_l2": mean_or_nan([row["residual_head_l2"] for row in task_rows]),
            },
        }
    row = metric_row(method, "internal_oof", repeat, "all", metrics, branch, len(labels))
    oof_records = [
        {
            **record,
            "scope": "internal_oof",
            "fold": "all",
        }
        for record in selected
    ]
    return row, oof_records


def evaluate_internal_task(
    cohort: Cohort, task_result: dict[str, Any], device: torch.device
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert_shared_selection(task_result)
    task: Task = task_result["task"]
    rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    subject_indices = subject_indices_for_task(cohort, task)
    for method, branch_key in ((BASELINE_METHOD, "baseline"), (TREATMENT_METHOD, "treatment")):
        metrics, probabilities = evaluate_selected_branch(
            cohort,
            task_result[branch_key],
            method,
            cohort.abide1,
            task.test_raw,
            task.test_labels,
            device,
        )
        row = metric_row(
            method,
            "internal_fold",
            task.repeat,
            task.fold,
            metrics,
            task_result[branch_key],
            len(task.test_raw),
        )
        rows.append(row)
        prediction_records = add_prediction_group(
            prediction_records,
            "internal_fold",
            method,
            task.repeat,
            task.fold,
            subject_indices,
            task.test_raw,
            task.test_labels,
            probabilities,
            float(task_result[branch_key]["threshold"]),
        )
        print(
            f"TASK_DONE repeat={task.repeat} scope=internal_fold fold={task.fold} "
            f"method={method} selected_encoder_epoch={task_result[branch_key]['encoder_epoch']} "
            f"test_auc={metrics['auc']:.6f} test_bce={metrics['bce']:.6f} "
            f"elapsed_min={task_result['elapsed_seconds'] / 60.0:.1f}",
            flush=True,
        )
    return rows, prediction_records


def evaluate_external_task(
    cohort: Cohort, task_result: dict[str, Any], device: torch.device
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assert_shared_selection(task_result)
    task: Task = task_result["task"]
    rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    # The complete baseline-eligible index is reused for both branches.
    abide2_indices = np.asarray(cohort.abide2_evaluable_indices, dtype=np.int64)
    assert len(abide2_indices) == 727
    assert np.array_equal(abide2_indices, np.arange(727, dtype=np.int64))
    evaluation_specs = (
        ("caltech", cohort.abide1, task.test_raw, task.test_labels, subject_indices_for_task(cohort, task)),
        (
            "abide2",
            cohort.abide2,
            abide2_indices,
            cohort.abide2_labels[abide2_indices],
            abide2_indices.copy(),
        ),
    )
    for method, branch_key in ((BASELINE_METHOD, "baseline"), (TREATMENT_METHOD, "treatment")):
        branch = task_result[branch_key]
        for scope, data, raw_indices, labels, subject_indices in evaluation_specs:
            metrics, probabilities = evaluate_selected_branch(
                cohort, branch, method, data, raw_indices, labels, device
            )
            rows.append(
                metric_row(
                    method,
                    scope,
                    task.repeat,
                    "external",
                    metrics,
                    branch,
                    len(raw_indices),
                )
            )
            prediction_records = add_prediction_group(
                prediction_records,
                scope,
                method,
                task.repeat,
                "external",
                subject_indices,
                raw_indices,
                labels,
                probabilities,
                float(branch["threshold"]),
            )
            print(
                f"TASK_DONE repeat={task.repeat} scope={scope} fold=external method={method} "
                f"selected_encoder_epoch={branch['encoder_epoch']} "
                f"test_auc={metrics['auc']:.6f} test_bce={metrics['bce']:.6f} "
                f"elapsed_min={task_result['elapsed_seconds'] / 60.0:.1f}",
                flush=True,
            )
    # Explicitly verify the paired external subject identities before the
    # result shard is written.  This guards against branch-specific filtering.
    for method in METHODS:
        group = [
            record
            for record in prediction_records
            if record["scope"] == "abide2" and record["method"] == method
        ]
        ids = sorted((record["subject_index"], record["raw_index"]) for record in group)
        expected_ids = sorted((int(index), int(index)) for index in abide2_indices.tolist())
        assert ids == expected_ids, f"ABIDE-II {method} IDs do not match eligibility index"
    baseline_ids = sorted(
        (record["subject_index"], record["raw_index"])
        for record in prediction_records
        if record["scope"] == "abide2" and record["method"] == "baseline"
    )
    treatment_ids = sorted(
        (record["subject_index"], record["raw_index"])
        for record in prediction_records
        if record["scope"] == "abide2" and record["method"] == TREATMENT_METHOD
    )
    assert baseline_ids == treatment_ids, "ABIDE-II baseline/treatment subject IDs differ"
    external_models = {
        "format_version": 1,
        "repeat": int(task.repeat),
        "task_identity": task_identity(task),
        "model_config": make_model_config(cohort)[0],
        "baseline": state_to_cpu(task_result["baseline"]),
        TREATMENT_METHOD: state_to_cpu(task_result["treatment"]),
    }
    return rows, prediction_records, external_models


def external_rows_complete(rows: list[dict[str, Any]], repeat: int) -> bool:
    return all(
        task_row_complete(rows, scope, repeat, "external") for scope in ("caltech", "abide2")
    )


def remove_external_rows(rows: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    for scope in ("caltech", "abide2"):
        rows = clear_metric_rows(rows, scope, repeat, "external")
    return rows


def remove_external_predictions(
    records: list[dict[str, Any]], repeat: int
) -> list[dict[str, Any]]:
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
        expected.add((method, "internal_oof", "all"))
        expected.add((method, "caltech", "external"))
        expected.add((method, "abide2", "external"))
    found = {(row["method"], row["scope"], row["fold"]) for row in selected}
    return found == expected


def update_runtime_task(repeat: int, task_key: str, seconds: float, device: torch.device) -> None:
    """Atomically persist cumulative compute time for every completed task."""
    path = worker_paths(repeat)["runtime"]
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("repeat", int(repeat))
    payload.setdefault("device", str(device))
    payload.setdefault("completed_tasks", {})
    payload["completed_tasks"][str(task_key)] = float(seconds)
    payload["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(path, payload)


def run_repeat(cohort: Cohort, repeat: int, device: torch.device) -> None:
    """Complete or resume one repeat, appending each completed task safely."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = worker_paths(repeat)
    rows = read_metric_shard(paths["csv"])
    records = read_prediction_records(paths["predictions"])
    if expected_repeat_rows(rows, repeat) and paths["external"].exists():
        print(f"REPEAT_DONE repeat={repeat} status=PASS already_complete=Y", flush=True)
        return

    print(f"REPEAT_START repeat={repeat} device={device} tasks=11", flush=True)
    tasks = make_tasks(cohort, repeat)
    for task in tasks[:-1]:
        if task_row_complete(rows, "internal_fold", repeat, task.fold):
            print(
                f"TASK_SKIP repeat={repeat} scope=internal_fold fold={task.fold} reason=PASS_SHARD",
                flush=True,
            )
            clean_task_checkpoints(repeat, task)
            continue
        # A half-written task is discarded before it can be retrained, avoiding
        # duplicate rows and restoring a single source of truth.
        rows = clear_metric_rows(rows, "internal_fold", repeat, task.fold)
        records = clear_prediction_group(records, "internal_fold", repeat, task.fold)
        write_metric_shard_atomic(paths["csv"], rows)
        write_prediction_records_atomic(paths["predictions"], records)
        task_result = train_shared_task(cohort, task, device)
        task_rows, task_predictions = evaluate_internal_task(cohort, task_result, device)
        for method in METHODS:
            method_predictions = [
                record for record in task_predictions if record["method"] == method
            ]
            records = add_prediction_group(
                records,
                "internal_fold",
                method,
                repeat,
                task.fold,
                np.asarray([record["subject_index"] for record in method_predictions]),
                np.asarray([record["raw_index"] for record in method_predictions]),
                np.asarray([record["y"] for record in method_predictions]),
                np.asarray([record["p_hc"] for record in method_predictions]),
                np.asarray([record["threshold"] for record in method_predictions]),
            )
        # Predictions are written before metrics: a PASS metric row never
        # exists without the prediction material needed for OOF aggregation.
        write_prediction_records_atomic(paths["predictions"], records)
        append_metric_rows(paths["csv"], task_rows)
        rows.extend(task_rows)
        update_runtime_task(repeat, f"internal_{task.fold}", task_result["elapsed_seconds"], device)
        clean_task_checkpoints(repeat, task)

    # Fold predictions are now sufficient to calculate one 882-person OOF
    # result per method.  This does not access a model or retrain anything.
    for method in METHODS:
        if any(
            row["scope"] == "internal_oof"
            and int(row["repeat"]) == int(repeat)
            and row["fold"] == "all"
            and row["method"] == method
            for row in rows
        ):
            continue
        oof_row, oof_records = build_internal_oof_row(cohort, records, rows, method, repeat)
        records = clear_prediction_group(records, "internal_oof", repeat, "all", method)
        records.extend(oof_records)
        write_prediction_records_atomic(paths["predictions"], records)
        sibling_rows = [
            row
            for row in rows
            if row["scope"] == "internal_oof"
            and int(row["repeat"]) == int(repeat)
            and row["fold"] == "all"
            and row["method"] != method
        ]
        rows = clear_metric_rows(rows, "internal_oof", repeat, "all")
        write_metric_shard_atomic(paths["csv"], [*rows, *sibling_rows])
        rows.extend(sibling_rows)
        append_metric_rows(paths["csv"], [oof_row])
        rows.append(oof_row)

    external_task = tasks[-1]
    if not (external_rows_complete(rows, repeat) and paths["external"].exists()):
        rows = remove_external_rows(rows, repeat)
        records = remove_external_predictions(records, repeat)
        write_metric_shard_atomic(paths["csv"], rows)
        write_prediction_records_atomic(paths["predictions"], records)
        task_result = train_shared_task(cohort, external_task, device)
        external_rows, external_predictions, external_models = evaluate_external_task(
            cohort, task_result, device
        )
        for scope in ("caltech", "abide2"):
            for method in METHODS:
                group = [
                    record
                    for record in external_predictions
                    if record["scope"] == scope and record["method"] == method
                ]
                records = add_prediction_group(
                    records,
                    scope,
                    method,
                    repeat,
                    "external",
                    np.asarray([record["subject_index"] for record in group]),
                    np.asarray([record["raw_index"] for record in group]),
                    np.asarray([record["y"] for record in group]),
                    np.asarray([record["p_hc"] for record in group]),
                    np.asarray([record["threshold"] for record in group]),
                )
        # The external models are saved first, then predictions, then PASS rows.
        atomic_torch_save(paths["external"], external_models)
        write_prediction_records_atomic(paths["predictions"], records)
        append_metric_rows(paths["csv"], external_rows)
        rows.extend(external_rows)
        update_runtime_task(repeat, "external", task_result["elapsed_seconds"], device)
        clean_task_checkpoints(repeat, external_task)

    if not expected_repeat_rows(rows, repeat):
        raise RuntimeError(f"repeat {repeat} does not contain all 26 final metric rows")
    write_json_atomic(
        paths["done"],
        {
            "repeat": int(repeat),
            "status": "PASS",
            "metrics_rows": len([row for row in rows if int(row["repeat"]) == int(repeat)]),
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(f"REPEAT_DONE repeat={repeat} status=PASS rows=26", flush=True)


def aggregate_rows(
    rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], cohort: Cohort
) -> dict[str, Any]:
    def select(method: str, scope: str) -> list[dict[str, Any]]:
        return [
            row for row in rows if row["method"] == method and row["scope"] == scope
        ]

    methods = METHODS
    scopes = ("internal_oof", "caltech", "abide2")
    aggregates: dict[str, Any] = {}
    for method in methods:
        for scope in scopes:
            selected = select(method, scope)
            aggregates[f"{method}::{scope}"] = {
                "n": len(selected),
                "auc_mean": mean_or_nan([row["auc"] for row in selected]),
                "auc_std": float(np.std([row["auc"] for row in selected], ddof=1))
                if len(selected) > 1
                else 0.0,
                "bce_mean": mean_or_nan([row["bce"] for row in selected]),
                "bce_std": float(np.std([row["bce"] for row in selected], ddof=1))
                if len(selected) > 1
                else 0.0,
                "f1_mean": mean_or_nan([row["f1"] for row in selected]),
                "balanced_accuracy_mean": mean_or_nan(
                    [row["balanced_accuracy"] for row in selected]
                ),
            }

    deltas: dict[str, list[float]] = {}
    for scope in scopes:
        baseline = {int(row["repeat"]): row for row in select("baseline", scope)}
        treatment = {
            int(row["repeat"]): row
            for row in select(TREATMENT_METHOD, scope)
        }
        repeats = sorted(set(baseline) & set(treatment))
        deltas[f"{scope}_auc"] = [
            float(treatment[repeat]["auc"] - baseline[repeat]["auc"])
            for repeat in repeats
        ]
        deltas[f"{scope}_bce"] = [
            float(treatment[repeat]["bce"] - baseline[repeat]["bce"])
            for repeat in repeats
        ]

    internal_auc = aggregates["baseline::internal_oof"]["auc_mean"]
    abide2_auc = aggregates["baseline::abide2"]["auc_mean"]
    pipeline_drift = (
        not np.isfinite(internal_auc)
        or not np.isfinite(abide2_auc)
        or abs(float(internal_auc) - 0.723) > 0.015
        or abs(float(abide2_auc) - 0.739) > 0.015
    )
    internal_auc_delta = mean_or_nan(deltas["internal_oof_auc"])
    internal_bce_delta = mean_or_nan(deltas["internal_oof_bce"])
    abide2_auc_delta = mean_or_nan(deltas["abide2_auc"])
    abide2_bce_delta = mean_or_nan(deltas["abide2_bce"])
    internal_positive = sum(delta > 0.0 for delta in deltas["internal_oof_auc"])
    residual_task_rows = [
        row
        for row in rows
        if row["method"] == TREATMENT_METHOD
        and row["scope"] in {"internal_fold", "caltech"}
    ]
    residual_epochs = [
        int(round(row["selected_residual_epoch"]))
        for row in residual_task_rows
        if np.isfinite(row["selected_residual_epoch"])
    ]
    residual_epoch0_fraction = (
        float(np.mean([epoch == 0 for epoch in residual_epochs]))
        if residual_epochs
        else float("nan")
    )
    residual_epoch_distribution: dict[str, int] = {}
    for epoch in sorted(set(residual_epochs)):
        residual_epoch_distribution[str(epoch)] = residual_epochs.count(epoch)
    singleton_values = [
        float(row["singleton_modules"])
        for row in residual_task_rows
        if np.isfinite(row["singleton_modules"])
    ]
    residual_head_l2_values = [
        float(row["residual_head_l2"])
        for row in residual_task_rows
        if np.isfinite(row["residual_head_l2"])
    ]
    mechanism_collapse = (
        len(singleton_values) >= 10
        and sum(value >= 8.0 for value in singleton_values) >= 5
    )

    if pipeline_drift:
        decision = "PIPELINE_DRIFT"
    elif mechanism_collapse:
        decision = "MECHANISM_COLLAPSE"
    elif (
        internal_auc_delta >= 0.005
        and internal_positive >= 7
        and abide2_auc_delta >= 0.003
        and internal_bce_delta <= 0.005
        and abide2_bce_delta <= 0.005
    ):
        decision = "STRONG_POSITIVE"
    elif (
        (internal_auc_delta >= 0.003 and internal_positive >= 7 and abide2_auc_delta >= 0.0)
        or (abide2_auc_delta >= 0.005 and internal_auc_delta >= 0.0)
    ):
        decision = "PROMISING"
    else:
        decision = "FAIL"

    peak_alloc = mean_or_nan([row.get("peak_gpu_alloc_gb", float("nan")) for row in runtime_rows])
    peak_reserved = mean_or_nan(
        [row.get("peak_gpu_reserved_gb", float("nan")) for row in runtime_rows]
    )
    repeat_task_seconds = [
        float(sum(float(value) for value in row.get("completed_tasks", {}).values()))
        for row in runtime_rows
    ]
    worker_seconds = float(sum(repeat_task_seconds))
    max_repeat_cumulative = float(max(repeat_task_seconds or [0.0]))
    final_wall = float(max([float(row.get("worker_wall_seconds", 0.0)) for row in runtime_rows] or [0.0]))
    return {
        "status": "PASS",
        "config": CONFIG,
        "abide2_original_n": 727,
        "abide2_evaluable_n": 727,
        "abide2_excluded_n": 0,
        "abide2_excluded_subject": None,
        "abide2_excluded_index": None,
        "exclusion_reason": None,
        "dataset": {
            "atlas": "AICHA",
            "abide1_raw": 995,
            "abide1_internal_unique_names": 882,
            "caltech_external": 37,
            "abide1_duplicate_train_occurrences": 76,
            "abide2_external": 727,
            "abide2_original_n": 727,
            "abide2_evaluable_n": 727,
            "abide2_excluded_n": 0,
        },
        "aggregates": aggregates,
        "deltas": deltas,
        "baseline_sanity": {
            "internal_auc_reference": 0.723,
            "abide2_auc_reference": 0.739,
            "internal_auc": internal_auc,
            "abide2_auc": abide2_auc,
            "pipeline_drift": bool(pipeline_drift),
        },
        "module_diagnostics": {
            "n_modules": int(CONFIG["n_modules"]),
            "meta_dim": int(CONFIG["meta_dim"]),
            "singleton_modules_ge8_count": int(sum(value >= 8.0 for value in singleton_values)),
            "module_size_min_mean": mean_or_nan([row["module_size_min"] for row in residual_task_rows]),
            "module_size_mean_mean": mean_or_nan([row["module_size_mean"] for row in residual_task_rows]),
            "module_size_max_mean": mean_or_nan([row["module_size_max"] for row in residual_task_rows]),
            "singleton_modules_mean": mean_or_nan(singleton_values),
            "empty_meta_bins_mean": mean_or_nan([row["empty_meta_bins"] for row in residual_task_rows]),
            "affinity_mean_mean": mean_or_nan([row["affinity_mean"] for row in residual_task_rows]),
            "affinity_std_mean": mean_or_nan([row["affinity_std"] for row in residual_task_rows]),
            "affinity_max_mean": mean_or_nan([row["affinity_max"] for row in residual_task_rows]),
            "residual_head_l2_mean": mean_or_nan(residual_head_l2_values),
            "residual_head_l2_min": float(min(residual_head_l2_values)) if residual_head_l2_values else float("nan"),
            "residual_head_l2_max": float(max(residual_head_l2_values)) if residual_head_l2_values else float("nan"),
            "residual_epoch_distribution": residual_epoch_distribution,
            "residual_epoch0_fraction": residual_epoch0_fraction,
            "residual_task_count": len(residual_epochs),
            "mechanism_collapse": bool(mechanism_collapse),
        },
        "decision": decision,
        "completion": {
            "expected_internal_folds_per_method": 100,
            "expected_internal_oof_per_method": 10,
            "expected_caltech_per_method": 10,
            "expected_abide2_per_method": 10,
            "metrics_rows": len(rows),
        },
        "runtime": {
            "workers": len(runtime_rows),
            "wall_clock_seconds": final_wall,
            "final_wall_seconds": final_wall,
            "total_task_compute_seconds": worker_seconds,
            "max_repeat_cumulative_seconds": max_repeat_cumulative,
            "worker_seconds": worker_seconds,
            "mean_peak_gpu_alloc_gb": peak_alloc,
            "mean_peak_gpu_reserved_gb": peak_reserved,
            "checkpoint_resume_supported": True,
            "checkpoint_interval_epochs": int(CONFIG["checkpoint_interval"]),
        },
        "integrity": {
            "shared_encoder_trajectory": True,
            "original_infonce_only": True,
            "train_only_meta_normalization": True,
            "validation_only_model_selection": True,
            "validation_only_threshold": True,
            "test_access_per_epoch": False,
            "duplicate_train_only": True,
            "experiment1_runtime_import": False,
            "abide2_paired_subject_ids_equal": True,
            "abide2_full_baseline_eligible": True,
            "module_discovery_label_free": True,
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the final Experiment 3 handoff report from aggregated artifacts."""
    aggregates = summary["aggregates"]
    deltas = summary["deltas"]

    def value(method: str, scope: str, metric: str) -> float:
        return float(aggregates[f"{method}::{scope}"][metric])

    table: list[str] = []
    for label, scope in (("Internal OOF", "internal_oof"), ("Caltech", "caltech"), ("ABIDE-II", "abide2")):
        b_auc = value(BASELINE_METHOD, scope, "auc_mean")
        t_auc = value(TREATMENT_METHOD, scope, "auc_mean")
        b_bce = value(BASELINE_METHOD, scope, "bce_mean")
        t_bce = value(TREATMENT_METHOD, scope, "bce_mean")
        table.append(
            f"| {label} | {fmt(b_auc)} | {fmt(t_auc)} | {fmt(t_auc-b_auc)} | "
            f"{fmt(b_bce)} | {fmt(t_bce)} | {fmt(t_bce-b_bce)} |"
        )
    runtime = summary["runtime"]
    diagnostics = summary["module_diagnostics"]
    internal_deltas = ", ".join(fmt(x, 6) for x in deltas["internal_oof_auc"])
    abide2_deltas = ", ".join(fmt(x, 6) for x in deltas["abide2_auc"])
    decision = summary["decision"]
    if decision in {"STRONG_POSITIVE", "PROMISING"}:
        interpretation = (
            "Meso-scale consensus organization provides complementary ASD-relevant "
            "information beyond the original fine-grained VarCoNet pairwise readout."
        )
    elif decision == "FAIL":
        interpretation = (
            "Coarse-graining the stable learned connectome into train-derived consensus "
            "functional modules does not provide stable incremental ASD classification "
            "value beyond the original pairwise linear readout."
        )
    else:
        interpretation = "The pre-registered integrity/sanity rule prevents a scientific interpretation of treatment gain."
    report = f"""# Experiment 3
## Consensus High-Order Modular Residual VarCoNet (CHMR-VarCoNet)

## Motivation

Experiment 1 changed cross-subject geometry with subtype prototypes and failed. Experiment 2 and Experiment 2B added temporal/dynamic residual signals and failed. Therefore Experiment 3 does not modify the VarCoNet representation: it tests whether flattening the stable learned connectome into edge-independent linear weights overlooks meso-scale brain organization.

## Literature rationale

HyBRiD motivates high-order ROI relationships and maximally informative/minimally redundant groups. BrainGNN motivates ROI selection and group-level consistency. CRGNN illustrates that learned graph structure can be task-aligned, while BrainOOD emphasizes stable structure selection for multi-site generalization. The 2026 ASD hypergraph study motivates complementary pairwise and higher-order information, and MCDGLN motivates filtering irrelevant connections. These lessons motivate only the fixed consensus readout; no cited full model is copied.

## Exact method

The original VarCoNet 1D CNN + Transformer, Conv1d kernel 4/stride 2, temporal-crop augmentation, InfoNCE objective, optimizer and scheduler are unchanged. Its stable learned-FC upper triangle is S ∈ R^73536 and the baseline is exactly Linear(73536, 2) + Softmax.

For each task, after baseline validation-BCE selection and freezing, consensus affinity is A_ij = mean over unique outer-training subjects |S_ij|. SpectralClustering uses n_clusters=32, affinity=precomputed, assign_labels=cluster_qr, eigen_solver=arpack, random_state=task.seed, n_jobs=1. Every upper-triangle edge is pooled once into H_pq = mean{{S_ij: i∈M_p, j∈M_q}}, yielding 32×33/2 = 528 features. Training-only mean/std normalization is applied to H. A zero-initialized single residual head δ=wᵀz(H) is trained for epochs 1..149 with Adam 5e-5 and BCELoss; treatment probabilities use baseline log-probabilities plus [-δ/2,+δ/2].

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
{chr(10).join(table)}

Internal repeat ΔAUC (CHMR − baseline): `[{internal_deltas}]`

ABIDE-II repeat ΔAUC (CHMR − baseline): `[{abide2_deltas}]`

Baseline sanity: internal OOF AUC={fmt(summary['baseline_sanity']['internal_auc'])} (reference 0.723); ABIDE-II AUC={fmt(summary['baseline_sanity']['abide2_auc'])} (reference 0.739); pipeline drift={summary['baseline_sanity']['pipeline_drift']}.

## Module diagnostics

- Fixed module count={diagnostics['n_modules']}; meta dimension={diagnostics['meta_dim']}.
- Task-average module size min/mean/max={fmt(diagnostics['module_size_min_mean'], 3)} / {fmt(diagnostics['module_size_mean_mean'], 3)} / {fmt(diagnostics['module_size_max_mean'], 3)}; singleton≥8 count={diagnostics['singleton_modules_ge8_count']}.
- Empty meta bins (mean)={fmt(diagnostics['empty_meta_bins_mean'], 3)}; affinity mean/std/max={fmt(diagnostics['affinity_mean_mean'], 8)} / {fmt(diagnostics['affinity_std_mean'], 8)} / {fmt(diagnostics['affinity_max_mean'], 8)}.
- Residual epoch-0 fraction={fmt(diagnostics['residual_epoch0_fraction'], 4)}; epoch distribution=`{json.dumps(diagnostics['residual_epoch_distribution'], sort_keys=True)}`.
- Residual-head L2 min/mean/max={fmt(diagnostics['residual_head_l2_min'], 6)} / {fmt(diagnostics['residual_head_l2_mean'], 6)} / {fmt(diagnostics['residual_head_l2_max'], 6)}.

## Runtime

10 workers (repeats 0,2,4,6,8 on GPU0; 1,3,5,7,9 on GPU1), six CPU threads per worker, full precision, no DDP/AMP/compile. Wall clock={fmt(runtime['wall_clock_seconds']/3600.0, 3)} h; summed worker compute={fmt(runtime['worker_seconds']/3600.0, 3)} h; mean peak GPU allocation/reservation={fmt(runtime['mean_peak_gpu_alloc_gb'], 3)} / {fmt(runtime['mean_peak_gpu_reserved_gb'], 3)} GB. Atomic rolling latest/best checkpoints and append-only shards were used and removed after aggregation.

## Decision

**{decision}**

## Interpretation

{interpretation}

## Reproducibility

Configuration is AICHA/384 ROI, stable pairwise dimension 73536, 32 consensus modules, 528 meta features, 50 SSL epochs, 149 classifier/residual updates, 10 repeats × 10 internal folds, Caltech n=37 and ABIDE-II n=727. Final artifacts are `results/metrics.csv`, `results/summary.json`, `results/predictions.npz`, and `results/external_best_models.pt`.
"""
    (EXPERIMENT_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def smoke_marker_path() -> Path:
    return RESULTS_DIR / ".smoke_pass.json"


def smoke_test(device: torch.device) -> None:
    """Run exactly one real-batch Experiment 3 smoke test before workers."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    set_all_seeds(777)
    model_config, _ = make_model_config(cohort)
    roi_num = int(cohort.abide1.shape[2])
    expected_edges = roi_num * (roi_num - 1) // 2

    # Compare the copied implementation against the audited root source with
    # identical weights and one actual ABIDE-I batch.
    import importlib.util

    root_spec = importlib.util.spec_from_file_location(
        "varconet_root_smoke", REPO_ROOT / "model_scripts" / "VarCoNet.py"
    )
    if root_spec is None or root_spec.loader is None:
        raise RuntimeError("cannot load root VarCoNet for parity smoke")
    root_module = importlib.util.module_from_spec(root_spec)
    root_spec.loader.exec_module(root_module)
    local_encoder = VarCoNet(model_config, roi_num).to(device)
    root_encoder = root_module.VarCoNet(model_config, roi_num).to(device)
    root_encoder.load_state_dict(clone_state_cpu(local_encoder.state_dict()))
    local_encoder.eval()
    root_encoder.eval()
    selected = np.arange(8, dtype=np.int64)
    x = torch.from_numpy(np.asarray(cohort.abide1[cohort.internal_raw[selected]], dtype=np.float32)).to(device)
    with torch.no_grad():
        local_stable = local_encoder(x)
        root_stable = root_encoder(x)
    parity = float((local_stable - root_stable).abs().max().cpu())
    if parity > 1e-6:
        raise RuntimeError(f"root baseline forward parity failed: {parity}")
    if tuple(local_stable.shape) != (len(selected), expected_edges):
        raise RuntimeError(f"stable shape drifted: {tuple(local_stable.shape)}")

    # Use a real outer-training subset for the label-free spectral smoke.
    module_raw = cohort.internal_raw[:64]
    stable_unique = extract_stable_features(local_encoder, cohort.abide1, module_raw, device)
    module_labels, module_diag = discover_consensus_modules(
        stable_unique, roi_num=roi_num, n_modules=int(CONFIG["n_modules"]), seed=123
    )
    if module_labels.shape != (roi_num,) or len(np.unique(module_labels)) != 32:
        raise RuntimeError("smoke did not produce exactly 32 modules")
    if not np.array_equal(np.sort(np.unique(module_labels)), np.arange(32)):
        raise RuntimeError("smoke module labels are not canonical 0..31")
    tri_i, tri_j = np.triu_indices(roi_num, k=1)
    edge_mean_abs = stable_unique.detach().abs().mean(dim=0).cpu().numpy()
    affinity = np.zeros((roi_num, roi_num), dtype=np.float64)
    affinity[tri_i, tri_j] = edge_mean_abs
    affinity[tri_j, tri_i] = edge_mean_abs
    np.fill_diagonal(affinity, 0.0)
    if not np.isfinite(affinity).all() or not np.allclose(affinity, affinity.T, atol=1e-12, rtol=0.0):
        raise RuntimeError("smoke consensus affinity is not finite/symmetric")
    if np.any(affinity < 0.0) or not np.all(np.diag(affinity) == 0.0):
        raise RuntimeError("smoke consensus affinity violates nonnegative/zero-diagonal rule")

    pooler = ModulePairPooler(module_labels, roi_num=roi_num, n_modules=32).to(device)
    if pooler.meta_dim != 528 or int(pooler.bin_counts.sum().cpu()) != expected_edges:
        raise RuntimeError("smoke module-pair assignment does not cover 73536 edges")
    meta = pooler(stable_unique)
    if tuple(meta.shape) != (len(module_raw), 528) or not bool(torch.isfinite(meta).all()):
        raise RuntimeError("smoke meta-connectome shape/finite check failed")
    meta_std = meta.std(dim=0, unbiased=False)
    if not bool(torch.any(meta_std > 0)):
        raise RuntimeError("smoke meta-connectome is constant")

    labels = torch.as_tensor(cohort.internal_labels[selected], dtype=torch.long, device=device)
    baseline = MLP(expected_edges, 2).to(device)
    baseline.eval()
    criterion = torch.nn.BCELoss()
    baseline_output = baseline(local_stable.detach())
    baseline_loss = criterion(baseline_output, F.one_hot(labels, num_classes=2).float())
    baseline_loss.backward()
    if not bool(torch.isfinite(baseline_loss)) or not finite_gradients(baseline):
        raise RuntimeError("smoke baseline backward is non-finite")
    baseline_state = clone_state_cpu(baseline.state_dict())
    residual = FrozenStableModuleResidualClassifier(baseline_state, 528).to(device)
    residual.eval()
    # Use the same real batch for the epoch-0 parity check.
    meta_batch = pooler(local_stable.detach())
    batch_mean = meta_batch.mean(dim=0)
    batch_std = torch.where(meta_batch.std(dim=0, unbiased=False) < 1e-6,
                            torch.ones(528, device=device),
                            meta_batch.std(dim=0, unbiased=False))
    residual_output = residual(local_stable.detach(), (meta_batch - batch_mean) / batch_std)
    residual_parity = float((baseline_output.detach() - residual_output.detach()).abs().max().cpu())
    residual_bce = criterion(residual_output, F.one_hot(labels, num_classes=2).float())
    if residual_parity > 1e-6 or abs(float(residual_bce.cpu()) - float(baseline_loss.detach().cpu())) > 1e-6:
        raise RuntimeError(
            f"smoke residual epoch-0 parity failed: prob={residual_parity} "
            f"bce={float(residual_bce.cpu())} baseline={float(baseline_loss.detach().cpu())}"
        )
    residual_bce.backward()
    if not bool(torch.isfinite(residual_bce)) or not finite_gradients(residual.residual_head):
        raise RuntimeError("smoke residual backward is non-finite")
    payload = {
        "status": "PASS",
        "device": str(device),
        "root_baseline_forward_max_abs": parity,
        "residual_epoch0_max_abs": residual_parity,
        "stable_shape": list(local_stable.shape),
        "module_labels_shape": list(module_labels.shape),
        "n_modules": int(len(np.unique(module_labels))),
        "meta_shape": list(meta.shape),
        "meta_dim": int(pooler.meta_dim),
        "edge_assignment_sum": int(pooler.bin_counts.sum().cpu()),
        "affinity_finite_symmetric_nonnegative_diag0": True,
        "meta_finite_nonconstant": True,
        "baseline_backward_finite": True,
        "residual_backward_finite": True,
        "module_diagnostics": module_diag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(smoke_marker_path(), payload)
    print("SMOKE_PASS " + json.dumps(payload, ensure_ascii=False), flush=True)
    del local_encoder, root_encoder, baseline, residual, pooler, x
    if device.type == "cuda":
        torch.cuda.empty_cache()


def write_final_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    write_metric_shard_atomic(path, rows)


def write_final_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    records.sort(
        key=lambda record: (
            record["scope"],
            record["method"],
            int(record["repeat"]),
            record["fold"],
            int(record["subject_index"]),
        )
    )
    arrays = {
        "method": np.asarray([record["method"] for record in records], dtype="<U32"),
        "scope": np.asarray([record["scope"] for record in records], dtype="<U24"),
        "repeat": np.asarray([record["repeat"] for record in records], dtype=np.int16),
        "fold": np.asarray([record["fold"] for record in records], dtype="<U16"),
        "subject_index": np.asarray([record["subject_index"] for record in records], dtype=np.int32),
        "raw_index": np.asarray([record["raw_index"] for record in records], dtype=np.int32),
        "y": np.asarray([record["y"] for record in records], dtype=np.int8),
        "p_hc": np.asarray([record["p_hc"] for record in records], dtype=np.float64),
        "threshold": np.asarray([record["threshold"] for record in records], dtype=np.float64),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_prediction_records(records: list[dict[str, Any]], cohort: Cohort) -> None:
    expected_per_scope = {"internal_oof": 882, "caltech": 37, "abide2": 727}
    for scope, count in expected_per_scope.items():
        for method in METHODS:
            for repeat in range(10):
                subset = [
                    record
                    for record in records
                    if record["scope"] == scope
                    and record["method"] == method
                    and int(record["repeat"]) == repeat
                ]
                if len(subset) != count:
                    raise RuntimeError(
                        f"{scope}/{method}/repeat={repeat} has {len(subset)} predictions, "
                        f"expected {count}"
                    )
        for repeat in range(10):
            baseline_ids = sorted(
                (record["subject_index"], record["raw_index"])
                for record in records
                if record["scope"] == scope
                and record["method"] == "baseline"
                and int(record["repeat"]) == repeat
            )
            treatment_ids = sorted(
                (record["subject_index"], record["raw_index"])
                for record in records
                if record["scope"] == scope
                and record["method"] == TREATMENT_METHOD
                and int(record["repeat"]) == repeat
            )
            if baseline_ids != treatment_ids:
                raise RuntimeError(
                    f"{scope}/repeat={repeat} baseline and treatment subject IDs differ"
                )

    expected_abide2_ids = sorted(
        (int(index), int(index))
        for index in cohort.abide2_evaluable_indices.tolist()
    )
    if len(expected_abide2_ids) != 727 or expected_abide2_ids != [(i, i) for i in range(727)]:
        raise RuntimeError("ABIDE-II index is not the complete 727-subject set")
    for method in METHODS:
        for repeat in range(10):
            actual = sorted(
                (record["subject_index"], record["raw_index"])
                for record in records
                if record["scope"] == "abide2"
                and record["method"] == method
                and int(record["repeat"]) == repeat
            )
            if actual != expected_abide2_ids:
                raise RuntimeError(
                    f"ABIDE-II/{method}/repeat={repeat} IDs do not equal the complete 727-subject set"
                )


def aggregate_main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if (RESULTS_DIR / "summary.json").exists() and not list(RESULTS_DIR.glob(".worker_r*.csv")):
        existing = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
        if existing.get("completion", {}).get("metrics_rows") == 260:
            print(
                json.dumps(
                    {"status": "PASS", "rows": 260, "decision": existing.get("decision"), "already_aggregated": True},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return

    # Re-run the label-blind eligibility audit at aggregate time as well; this
    # does not retrain or recompute any internal result.
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
        all_records.extend(
            record
            for record in records
            if record["scope"] in {"internal_oof", "caltech", "abide2"}
        )
        if not paths["external"].exists():
            raise RuntimeError(f"cannot aggregate: missing external model shard for repeat {repeat}")
        model_payload = torch_load_cpu(paths["external"])
        if int(model_payload.get("repeat", -1)) != repeat:
            raise RuntimeError(f"external model shard repeat mismatch: {paths['external']}")
        external_models[str(repeat)] = model_payload
        if paths["runtime"].exists():
            runtime_rows.append(json.loads(paths["runtime"].read_text(encoding="utf-8")))
    if len(all_rows) != 260:
        raise RuntimeError(f"expected 260 metrics rows, got {len(all_rows)}")
    if any(row["status"] != "PASS" for row in all_rows):
        raise RuntimeError("cannot aggregate non-PASS metric rows")
    validate_prediction_records(all_records, cohort)

    write_final_metrics(RESULTS_DIR / "metrics.csv", all_rows)
    write_final_predictions(RESULTS_DIR / "predictions.npz", all_records)
    atomic_torch_save(
        RESULTS_DIR / "external_best_models.pt",
        {
            "format_version": 1,
            "atlas": "AICHA",
            "config": state_to_cpu(CONFIG),
            "repeats": external_models,
        },
    )
    summary = aggregate_rows(all_rows, runtime_rows, cohort)
    summary["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(RESULTS_DIR / "summary.json", summary)
    write_report(summary, all_rows)

    # Worker shards and rolling checkpoints are resume material only.  Their
    # removal leaves the requested final result set and avoids 100 fold weights.
    for repeat in range(10):
        paths = worker_paths(repeat)
        for path in paths.values():
            path.unlink(missing_ok=True)
    for path in CHECKPOINT_DIR.glob("worker_r*_latest.pt*"):
        path.unlink(missing_ok=True)
    for path in CHECKPOINT_DIR.glob("worker_r*_best.pt*"):
        path.unlink(missing_ok=True)
    smoke_marker_path().unlink(missing_ok=True)
    print(
        json.dumps(
            {"status": "PASS", "rows": len(all_rows), "decision": summary["decision"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


def worker_main(repeat: int, device_name: str) -> None:
    repeat = int(repeat)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["VARCONET_WORKER_ID"] = f"r{repeat}"
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # This PyTorch build accepts the current CUDA device here but rejects
        # a ``torch.device`` object after CUDA_VISIBLE_DEVICES remapping.
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    paths = worker_paths(repeat)
    try:
        cohort = load_cohort()
        run_repeat(cohort, repeat, device)
        peak_alloc = (
            float(torch.cuda.max_memory_allocated()) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved = (
            float(torch.cuda.max_memory_reserved()) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        runtime_payload = {}
        if paths["runtime"].exists():
            runtime_payload = json.loads(paths["runtime"].read_text(encoding="utf-8"))
        runtime_payload.update({
            "repeat": repeat,
            "status": "PASS",
            "device": str(device),
            "worker_wall_seconds": time.time() - started,
            "peak_gpu_alloc_gb": peak_alloc,
            "peak_gpu_reserved_gb": peak_reserved,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        # A prior interrupted attempt may have left an error marker.  A
        # successful resumed worker must not retain that stale failure state.
        paths["error"].unlink(missing_ok=True)
        write_json_atomic(paths["runtime"], runtime_payload)
    except Exception as exc:
        write_json_atomic(
            paths["error"],
            {
                "repeat": repeat,
                "status": "FAIL",
                "error": repr(exc),
                "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        print(f"WORKER_FAIL repeat={repeat} error={exc!r}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
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

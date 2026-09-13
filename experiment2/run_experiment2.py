#!/usr/bin/env python3
"""Experiment 2: Stable-Dynamic Residual VarCoNet.

This runner is intentionally self-contained under ``xiaolunwen/experiment2``.
It trains one shared VarCoNet/InfoNCE encoder trajectory for each task, then
fits the stable baseline and the frozen-stable dynamic residual readouts from
the same frozen features.  Test labels are never consulted during epoch
selection.
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

from dynamic_features import (  # noqa: E402
    FrozenStableResidualClassifier,
    LatentDynamicResidualExtractor,
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
    "window_tokens": 15,
    "stride_tokens": 15,
    "dynamic_dim": 774,
    "checkpoint_interval": 10,
    "cpu_threads": 6,
}

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
    "dynamic_dim",
    "window_tokens",
    "min_valid_tokens",
    "mean_valid_tokens",
    "min_windows",
    "mean_windows",
    "max_windows",
    "mean_node_switch",
    "mean_node_deviation",
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
    with np.load(path, allow_pickle=False) as archive:
        values = [np.asarray(archive[key], dtype=np.float32) for key in archive.files]
    if not values:
        raise RuntimeError(f"empty dataset archive: {path}")
    return np.stack(values, axis=0)


def load_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def latent_valid_tokens_from_raw(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return raw valid lengths and Conv1d-supported latent token lengths.

    VarCoNet identifies temporal padding from the first ROI.  The dynamic
    descriptor uses the valid support after the unchanged kernel=4,
    stride=2 convolution, so eligibility is computed from sequence length
    alone before any labels, predictions, or model outputs are consulted.
    """
    array = np.asarray(data)
    if array.ndim != 3:
        raise RuntimeError(f"expected [N,T,R] data for eligibility, got {array.shape}")
    raw_mask = array[:, :, 0] != 0
    raw_counts = raw_mask.sum(axis=1).astype(np.int64)
    positions = np.arange(array.shape[1], dtype=np.int64)[None, :]
    if not np.array_equal(raw_mask, positions < raw_counts[:, None]):
        raise RuntimeError("non-contiguous raw temporal validity encountered")
    kernel = 4
    stride = 2
    latent_counts = np.maximum(0, (raw_counts - kernel) // stride + 1)
    max_latent = max(0, (int(array.shape[1]) - kernel) // stride + 1)
    latent_counts = np.minimum(latent_counts, max_latent).astype(np.int64)
    return raw_counts, latent_counts


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

    # Label-blind technical eligibility for the paired ABIDE-II external
    # evaluation.  This is deliberately derived only from the input sequence
    # length and the fixed VarCoNet Conv1d geometry.
    _, abide2_latent_tokens = latent_valid_tokens_from_raw(abide2)
    required_tokens = 2 * int(CONFIG["window_tokens"])
    excluded_indices = np.where(abide2_latent_tokens < required_tokens)[0].astype(np.int64)
    assert len(excluded_indices) == 1, (
        "ABIDE-II eligibility changed: expected exactly one excluded subject, "
        f"found {len(excluded_indices)} at indices {excluded_indices.tolist()}"
    )
    assert int(excluded_indices[0]) == 568, (
        "ABIDE-II eligibility changed: expected excluded original_index=568, "
        f"found {int(excluded_indices[0])}"
    )
    assert abide2_names[int(excluded_indices[0])] == "sub-29623", (
        "ABIDE-II eligibility changed: expected excluded name sub-29623, "
        f"found {abide2_names[int(excluded_indices[0])]!r}"
    )
    evaluable_indices = np.setdiff1d(
        np.arange(len(abide2), dtype=np.int64), excluded_indices, assume_unique=True
    )
    if len(evaluable_indices) != 726:
        raise RuntimeError(
            f"ABIDE-II eligibility expected 726 evaluable subjects, found {len(evaluable_indices)}"
        )

    # Caltech is part of the dynamic evaluation too; unlike ABIDE-II it must
    # remain complete at n=37 under the pre-specified two-window constraint.
    _, caltech_latent_tokens = latent_valid_tokens_from_raw(abide1[caltech_raw])
    bad_caltech = np.where(caltech_latent_tokens < required_tokens)[0]
    if len(bad_caltech):
        bad_names = [caltech_names[int(index)] for index in bad_caltech.tolist()]
        raise RuntimeError(
            "Caltech contains ineligible scans for the fixed dynamic descriptor: "
            f"{bad_names}"
        )

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
        full_train_y = np.concatenate(
            [cohort.duplicate_labels, np.asarray(train_y, dtype=np.int64)]
        )
        tasks.append(
            Task(
                repeat=repeat,
                scope="internal_cv",
                fold=str(fold),
                seed=100000 + repeat * 100 + int(fold),
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
    ext_train_y = np.concatenate(
        [cohort.duplicate_labels, np.asarray(train_y, dtype=np.int64)]
    )
    tasks.append(
        Task(
            repeat=repeat,
            scope="caltech_external",
            fold="external",
            seed=200000 + repeat,
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


def extract_stable_dynamic_features(
    encoder: VarCoNet,
    data: np.ndarray,
    raw_indices: np.ndarray,
    device: torch.device,
    dynamic_extractor: LatentDynamicResidualExtractor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Batch-extract stable and dynamic features in one encoder pass per batch."""
    stable_parts: list[torch.Tensor] = []
    dynamic_parts: list[torch.Tensor] = []
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    encoder.eval()
    dynamic_extractor.eval()
    with torch.no_grad():
        for start in range(0, len(raw_indices), int(CONFIG["batch_size"])):
            selected = np.asarray(raw_indices[start : start + int(CONFIG["batch_size"])])
            array = np.asarray(data[selected], dtype=np.float32)
            x = torch.from_numpy(array).to(device)
            output = encoder.forward_with_latent(x)
            dynamic, diagnostics = dynamic_extractor(
                output["latent"], output["valid_mask"], return_diagnostics=True
            )
            stable_parts.append(output["stable"].detach())
            dynamic_parts.append(dynamic.detach())
            batch_count = int(len(selected))
            diagnostic_count += batch_count
            for key, value in diagnostics.items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value) * batch_count
            del x, output, dynamic
    if not stable_parts:
        raise RuntimeError("cannot extract features from an empty split")
    diagnostics = {
        key: value / float(max(diagnostic_count, 1)) for key, value in diagnostic_sums.items()
    }
    return torch.cat(stable_parts, dim=0), torch.cat(dynamic_parts, dim=0), diagnostics


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


def fit_dynamic_residual_classifier(
    baseline_fit: dict[str, Any],
    stable_train: torch.Tensor,
    dynamic_train: torch.Tensor,
    y_train: torch.Tensor,
    stable_val: torch.Tensor,
    dynamic_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Fit only the zero-initialized dynamic residual, with stable frozen."""
    dyn_mean = dynamic_train.mean(dim=0)
    dyn_std = dynamic_train.std(dim=0, unbiased=False)
    dyn_std = torch.where(dyn_std < 1e-6, torch.ones_like(dyn_std), dyn_std)
    train_z = (dynamic_train - dyn_mean) / dyn_std
    val_z = (dynamic_val - dyn_mean) / dyn_std

    set_all_seeds(seed)
    residual = FrozenStableResidualClassifier(
        baseline_state=baseline_fit["classifier_state"],
        dynamic_dim=int(CONFIG["dynamic_dim"]),
    ).to(device)
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        epoch0_output = residual(stable_val, val_z)
        epoch0_bce = float(
            criterion(epoch0_output, F.one_hot(y_val, num_classes=2).float()).detach().cpu()
        )
    if abs(epoch0_bce - float(baseline_fit["val_bce"])) > 1e-6:
        raise RuntimeError(
            "residual epoch-0 validation BCE is not baseline-identical: "
            f"{epoch0_bce:.10f} vs {float(baseline_fit['val_bce']):.10f}"
        )

    best_bce = epoch0_bce
    best_epoch = 0
    best_head_state = state_to_cpu(residual.dynamic_head.state_dict())
    optimizer = Adam(residual.dynamic_head.parameters(), lr=float(CONFIG["lr_cls"]))
    for epoch in range(1, int(CONFIG["epochs_cls"])):
        residual.train()
        optimizer.zero_grad()
        output = residual(stable_train, train_z)
        loss = criterion(output, F.one_hot(y_train, num_classes=2).float())
        loss.backward()
        if not finite_gradients(residual) or not torch.isfinite(loss):
            raise FloatingPointError("non-finite residual classifier gradient/loss")
        optimizer.step()
        residual.eval()
        with torch.no_grad():
            val_output = residual(stable_val, val_z)
            val_bce = float(
                criterion(val_output, F.one_hot(y_val, num_classes=2).float()).detach().cpu()
            )
        if val_bce < best_bce:
            best_bce = val_bce
            best_epoch = int(epoch)
            best_head_state = state_to_cpu(residual.dynamic_head.state_dict())

    residual.dynamic_head.load_state_dict(best_head_state)
    residual.eval()
    with torch.no_grad():
        val_probs = residual(stable_val, val_z)[:, -1].detach().cpu().numpy()
    return {
        "stable_classifier_state": state_to_cpu(baseline_fit["classifier_state"]),
        "dynamic_head_state": best_head_state,
        "dyn_mean": dyn_mean.detach().cpu().clone(),
        "dyn_std": dyn_std.detach().cpu().clone(),
        "val_bce": float(best_bce),
        "residual_epoch": int(best_epoch),
        "val_probs": val_probs,
        "threshold": validation_threshold(y_val.detach().cpu().numpy(), val_probs),
        "epoch0_bce": float(epoch0_bce),
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


def checkpoint_payload(
    task: Task,
    current_epoch: int,
    encoder: VarCoNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_baseline: dict[str, Any] | None,
    best_treatment: dict[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "saved_unix": time.time(),
        **task_identity(task),
        "epoch": int(current_epoch),
        "epochs": int(CONFIG["epochs"]),
        "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "optimizer_state_dict": state_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": state_to_cpu(scheduler.state_dict()),
        "best_baseline": state_to_cpu(best_baseline),
        "best_treatment": state_to_cpu(best_treatment),
        "rng_state": state_to_cpu(capture_rng_state()),
        "elapsed_seconds": float(elapsed_seconds),
    }
    return payload


def save_checkpoint(
    path: Path,
    kind: str,
    task: Task,
    current_epoch: int,
    encoder: VarCoNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_baseline: dict[str, Any] | None,
    best_treatment: dict[str, Any] | None,
    elapsed_seconds: float,
) -> None:
    atomic_torch_save(
        path,
        checkpoint_payload(
            task,
            current_epoch,
            encoder,
            optimizer,
            scheduler,
            best_baseline,
            best_treatment,
            elapsed_seconds,
        ),
    )
    print(
        f"CHECKPOINT_{kind.upper()} repeat={task.repeat} scope={task.scope} "
        f"fold={task.fold} epoch={current_epoch:02d}/{CONFIG['epochs']} path={path}",
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


def predict_treatment(
    stable: torch.Tensor,
    dynamic: torch.Tensor,
    branch: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    dyn_mean = branch["dyn_mean"].to(device)
    dyn_std = branch["dyn_std"].to(device)
    dynamic_z = (dynamic - dyn_mean) / dyn_std
    classifier = FrozenStableResidualClassifier(
        baseline_state=branch["stable_classifier_state"],
        dynamic_dim=int(CONFIG["dynamic_dim"]),
    ).to(device)
    classifier.dynamic_head.load_state_dict(branch["dynamic_head_state"])
    classifier.eval()
    with torch.no_grad():
        probabilities = classifier(stable, dynamic_z)[:, -1].detach().cpu().numpy()
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
    dynamic_extractor = LatentDynamicResidualExtractor(
        roi_num=int(cohort.abide1.shape[2]),
        window_tokens=int(CONFIG["window_tokens"]),
    ).to(device)
    stable, dynamic, _ = extract_stable_dynamic_features(
        encoder, data, raw_indices, device, dynamic_extractor
    )
    if method == "baseline":
        probabilities = predict_baseline(stable, branch["classifier_state"], device)
    elif method == "stable_dynamic_residual":
        probabilities = predict_treatment(stable, dynamic, branch, device)
    else:
        raise ValueError(f"unknown method {method}")
    metrics = classification_metrics(labels, probabilities, float(branch["threshold"]))
    del stable, dynamic, encoder, dynamic_extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, probabilities


def dynamic_metric_fields(
    method: str, diagnostics: dict[str, Any] | None
) -> dict[str, float]:
    if method == "baseline" or diagnostics is None:
        return {
            "dynamic_dim": float("nan"),
            "window_tokens": float("nan"),
            "min_valid_tokens": float("nan"),
            "mean_valid_tokens": float("nan"),
            "min_windows": float("nan"),
            "mean_windows": float("nan"),
            "max_windows": float("nan"),
            "mean_node_switch": float("nan"),
            "mean_node_deviation": float("nan"),
        }
    return {
        "dynamic_dim": float(CONFIG["dynamic_dim"]),
        "window_tokens": float(CONFIG["window_tokens"]),
        "min_valid_tokens": float(diagnostics["valid_tokens_min"]),
        "mean_valid_tokens": float(diagnostics["valid_tokens_mean"]),
        "min_windows": float(diagnostics["windows_min"]),
        "mean_windows": float(diagnostics["windows_mean"]),
        "max_windows": float(diagnostics["windows_max"]),
        "mean_node_switch": float(diagnostics["node_switch_mean"]),
        "mean_node_deviation": float(diagnostics["node_deviation_mean"]),
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
        "n_test": int(n_test),
        "status": "PASS",
    }
    row.update(dynamic_metric_fields(method, branch.get("dynamic_diagnostics")))
    return row


def resume_task_state(
    task: Task,
    encoder: VarCoNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, float]:
    """Restore the rolling state for exactly this task, if one exists."""
    latest_path, best_path = checkpoint_paths(task.repeat)
    latest_payload: dict[str, Any] | None = None
    if latest_path.exists():
        payload = torch_load_cpu(latest_path)
        if identity_matches(payload, task):
            latest_payload = payload
        else:
            print(
                f"RESUME_STALE repeat={task.repeat} expected={task.scope}/{task.fold}; "
                f"removing mismatched {latest_path}",
                flush=True,
            )
            latest_path.unlink(missing_ok=True)

    best_payload: dict[str, Any] | None = None
    if best_path.exists():
        payload = torch_load_cpu(best_path)
        if identity_matches(payload, task):
            best_payload = payload
        elif latest_payload is None:
            print(
                f"RESUME_STALE repeat={task.repeat} expected={task.scope}/{task.fold}; "
                f"removing mismatched {best_path}",
                flush=True,
            )
            best_path.unlink(missing_ok=True)

    source = latest_payload if latest_payload is not None else best_payload
    if source is None:
        return 1, None, None, 0.0

    encoder.load_state_dict(source["encoder_state_dict"])
    optimizer.load_state_dict(source["optimizer_state_dict"])
    optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(source["scheduler_state_dict"])
    best_baseline = state_to_cpu(source.get("best_baseline"))
    best_treatment = state_to_cpu(source.get("best_treatment"))
    if best_payload is not None:
        best_baseline = merge_best_branch(best_baseline, best_payload.get("best_baseline"))
        best_treatment = merge_best_branch(best_treatment, best_payload.get("best_treatment"))
    if source.get("rng_state") is not None:
        restore_rng_state(source["rng_state"])
    start_epoch = int(source["epoch"]) + 1
    elapsed = float(source.get("elapsed_seconds", 0.0))
    print(
        f"RESUME repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"from_epoch={start_epoch} latest_epoch={source['epoch']}",
        flush=True,
    )
    return start_epoch, best_baseline, best_treatment, elapsed


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
        "dynamic_diagnostics": None,
    }


def make_treatment_branch(
    encoder: VarCoNet,
    encoder_epoch: int,
    fit: dict[str, Any],
    diagnostics: dict[str, float],
    baseline_classifier_epoch: int,
) -> dict[str, Any]:
    return {
        "encoder_state": clone_state_cpu(encoder.state_dict()),
        "stable_classifier_state": state_to_cpu(fit["stable_classifier_state"]),
        "dynamic_head_state": state_to_cpu(fit["dynamic_head_state"]),
        "dyn_mean": fit["dyn_mean"].detach().cpu().clone(),
        "dyn_std": fit["dyn_std"].detach().cpu().clone(),
        "val_bce": float(fit["val_bce"]),
        "encoder_epoch": int(encoder_epoch),
        "classifier_epoch": int(baseline_classifier_epoch),
        "residual_epoch": int(fit["residual_epoch"]),
        "threshold": float(fit["threshold"]),
        "val_probs": np.asarray(fit["val_probs"], dtype=np.float64).copy(),
        "epoch0_bce": float(fit["epoch0_bce"]),
        "dynamic_diagnostics": {key: float(value) for key, value in diagnostics.items()},
    }


def train_shared_task(
    cohort: Cohort,
    task: Task,
    device: torch.device,
) -> dict[str, Any]:
    """Train one shared SSL trajectory and select both downstream branches."""
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
    dynamic_extractor = LatentDynamicResidualExtractor(
        roi_num=roi_num,
        window_tokens=int(CONFIG["window_tokens"]),
    ).to(device)
    schedule = paired_batch_schedule(
        task.train_names, len(task.train_raw), int(CONFIG["epochs"]), task.seed
    )
    start_epoch, best_baseline, best_treatment, resumed_elapsed = resume_task_state(
        task, encoder, optimizer, scheduler, device
    )
    task_start -= resumed_elapsed
    latest_path, best_path = checkpoint_paths(task.repeat)

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

        # No test split enters this block: one batched encoder extraction for
        # train and one for validation feed both readouts.
        stable_train, dynamic_train, train_diag = extract_stable_dynamic_features(
            encoder, cohort.abide1, task.train_raw, device, dynamic_extractor
        )
        stable_val, dynamic_val, _ = extract_stable_dynamic_features(
            encoder, cohort.abide1, task.val_raw, device, dynamic_extractor
        )
        if dynamic_train.shape[1] != int(CONFIG["dynamic_dim"]):
            raise RuntimeError(
                f"dynamic dimension drifted to {dynamic_train.shape[1]}, "
                f"expected {CONFIG['dynamic_dim']}"
            )

        # Readout initialization must not perturb the next SSL augmentation
        # RNG state; preserve the trajectory's RNG around both fits.
        trajectory_rng = capture_rng_state()
        baseline_fit = fit_baseline_classifier(
            stable_train,
            train_labels,
            stable_val,
            val_labels,
            device,
            task.seed + 300000 + epoch * 2,
        )
        treatment_fit = fit_dynamic_residual_classifier(
            baseline_fit,
            stable_train,
            dynamic_train,
            train_labels,
            stable_val,
            dynamic_val,
            val_labels,
            device,
            task.seed + 300001 + epoch * 2,
        )
        restore_rng_state(trajectory_rng)

        baseline_candidate = make_baseline_branch(encoder, epoch, baseline_fit)
        treatment_candidate = make_treatment_branch(
            encoder,
            epoch,
            treatment_fit,
            train_diag,
            int(baseline_fit["classifier_epoch"]),
        )
        base_is_best = (
            best_baseline is None
            or float(baseline_candidate["val_bce"]) < float(best_baseline["val_bce"])
        )
        treatment_is_best = (
            best_treatment is None
            or float(treatment_candidate["val_bce"]) < float(best_treatment["val_bce"])
        )
        if base_is_best:
            best_baseline = baseline_candidate
        if treatment_is_best:
            best_treatment = treatment_candidate

        elapsed = time.time() - task_start
        if base_is_best or treatment_is_best:
            save_checkpoint(
                best_path,
                "best",
                task,
                epoch,
                encoder,
                optimizer,
                scheduler,
                best_baseline,
                best_treatment,
                elapsed,
            )
        if epoch % int(CONFIG["checkpoint_interval"]) == 0:
            save_checkpoint(
                latest_path,
                "latest",
                task,
                epoch,
                encoder,
                optimizer,
                scheduler,
                best_baseline,
                best_treatment,
                elapsed,
            )

        allocated, reserved = gpu_memory_gb(device)
        print(
            f"[repeat={task.repeat} fold={task.fold}] [epoch={epoch}/{CONFIG['epochs']}] "
            f"ssl_loss={ssl_loss:.6f} lr={optimizer.param_groups[0]['lr']:.8f} "
            f"base_val_bce={baseline_fit['val_bce']:.6f} "
            f"dyn_val_bce={treatment_fit['val_bce']:.6f} "
            f"base_best={'Y' if base_is_best else 'N'} "
            f"dyn_best={'Y' if treatment_is_best else 'N'} "
            f"base_cls_epoch={baseline_fit['classifier_epoch']} "
            f"dyn_res_epoch={treatment_fit['residual_epoch']} "
            f"windows_mean={train_diag['windows_mean']:.3f} "
            f"node_switch={train_diag['node_switch_mean']:.6f} "
            f"node_dev={train_diag['node_deviation_mean']:.6f} "
            f"epoch_sec={time.time() - epoch_start:.1f} "
            f"elapsed_min={elapsed / 60.0:.1f} "
            f"gpu_alloc_gb={allocated:.3f} gpu_reserved_gb={reserved:.3f}",
            flush=True,
        )
        del stable_train, dynamic_train, stable_val, dynamic_val
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_baseline is None or best_treatment is None:
        raise RuntimeError("no validation-selected baseline/treatment branch")
    result = {
        "task": task,
        "baseline": best_baseline,
        "treatment": best_treatment,
        "mean_ssl_loss": float(np.mean(ssl_losses)) if ssl_losses else float("nan"),
        "elapsed_seconds": float(time.time() - task_start),
    }
    print(
        f"TASK_TRAINING_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"base_epoch={best_baseline['encoder_epoch']} dyn_epoch={best_treatment['encoder_epoch']} "
        f"base_val_bce={best_baseline['val_bce']:.6f} "
        f"dyn_val_bce={best_treatment['val_bce']:.6f}",
        flush=True,
    )
    del encoder, contrast_model, optimizer, scheduler, dynamic_extractor
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
    "dynamic_dim",
    "window_tokens",
    "min_valid_tokens",
    "mean_valid_tokens",
    "min_windows",
    "mean_windows",
    "max_windows",
    "mean_node_switch",
    "mean_node_deviation",
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
    return methods == {"baseline", "stable_dynamic_residual"}


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
            "dynamic_diagnostics": None,
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
            "dynamic_diagnostics": {
                "valid_tokens_min": mean_or_nan(
                    [row["min_valid_tokens"] for row in task_rows]
                ),
                "valid_tokens_mean": mean_or_nan(
                    [row["mean_valid_tokens"] for row in task_rows]
                ),
                "windows_min": mean_or_nan([row["min_windows"] for row in task_rows]),
                "windows_mean": mean_or_nan([row["mean_windows"] for row in task_rows]),
                "windows_max": mean_or_nan([row["max_windows"] for row in task_rows]),
                "node_switch_mean": mean_or_nan(
                    [row["mean_node_switch"] for row in task_rows]
                ),
                "node_deviation_mean": mean_or_nan(
                    [row["mean_node_deviation"] for row in task_rows]
                ),
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
    task: Task = task_result["task"]
    rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    subject_indices = subject_indices_for_task(cohort, task)
    for method, branch_key in (
        ("baseline", "baseline"),
        ("stable_dynamic_residual", "treatment"),
    ):
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
    task: Task = task_result["task"]
    rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    # The same label-blind eligibility index is reused for both branches.
    # Keeping original indices makes the paired identity auditable in the
    # final compressed prediction artifact.
    abide2_indices = np.asarray(cohort.abide2_evaluable_indices, dtype=np.int64)
    assert len(abide2_indices) == 726
    assert 568 not in set(abide2_indices.tolist())
    assert len(set(abide2_indices.tolist())) == 726
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
    for method, branch_key in (
        ("baseline", "baseline"),
        ("stable_dynamic_residual", "treatment"),
    ):
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
    for method in ("baseline", "stable_dynamic_residual"):
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
        if record["scope"] == "abide2" and record["method"] == "stable_dynamic_residual"
    )
    assert baseline_ids == treatment_ids, "ABIDE-II baseline/treatment subject IDs differ"
    external_models = {
        "format_version": 1,
        "repeat": int(task.repeat),
        "task_identity": task_identity(task),
        "model_config": make_model_config(cohort)[0],
        "baseline": state_to_cpu(task_result["baseline"]),
        "stable_dynamic_residual": state_to_cpu(task_result["treatment"]),
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
        for method in ("baseline", "stable_dynamic_residual"):
            expected.add((method, "internal_fold", str(fold)))
    for method in ("baseline", "stable_dynamic_residual"):
        expected.add((method, "internal_oof", "all"))
        expected.add((method, "caltech", "external"))
        expected.add((method, "abide2", "external"))
    found = {(row["method"], row["scope"], row["fold"]) for row in selected}
    return found == expected


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
        for method in ("baseline", "stable_dynamic_residual"):
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
        clean_task_checkpoints(repeat, task)

    # Fold predictions are now sufficient to calculate one 882-person OOF
    # result per method.  This does not access a model or retrain anything.
    for method in ("baseline", "stable_dynamic_residual"):
        if task_row_complete(rows, "internal_oof", repeat, "all"):
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
        external_rows, external_predictions, external_models = evaluate_external_task(
            cohort, task_result, device
        )
        for scope in ("caltech", "abide2"):
            for method in ("baseline", "stable_dynamic_residual"):
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

    methods = ("baseline", "stable_dynamic_residual")
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
            for row in select("stable_dynamic_residual", scope)
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
        if row["method"] == "stable_dynamic_residual"
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

    if pipeline_drift:
        decision = "PIPELINE_DRIFT"
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
    wall_clock = max(
        [float(row.get("elapsed_seconds", 0.0)) for row in runtime_rows] or [0.0]
    )
    worker_seconds = float(
        sum(float(row.get("elapsed_seconds", 0.0)) for row in runtime_rows)
    )
    return {
        "status": "PASS",
        "config": CONFIG,
        "abide2_original_n": 727,
        "abide2_evaluable_n": 726,
        "abide2_excluded_n": 1,
        "abide2_excluded_subject": cohort.abide2_names[int(cohort.abide2_excluded_indices[0])],
        "abide2_excluded_index": int(cohort.abide2_excluded_indices[0]),
        "exclusion_reason": "insufficient latent tokens for two 15-token windows",
        "dataset": {
            "atlas": "AICHA",
            "abide1_raw": 995,
            "abide1_internal_unique_names": 882,
            "caltech_external": 37,
            "abide1_duplicate_train_occurrences": 76,
            "abide2_external": 726,
            "abide2_original_n": 727,
            "abide2_evaluable_n": 726,
            "abide2_excluded_n": 1,
            "abide2_excluded_subject": cohort.abide2_names[int(cohort.abide2_excluded_indices[0])],
            "abide2_excluded_index": int(cohort.abide2_excluded_indices[0]),
            "exclusion_reason": "insufficient latent tokens for two 15-token windows",
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
        "dynamic_diagnostics": {
            "dynamic_dim": int(CONFIG["dynamic_dim"]),
            "window_tokens": int(CONFIG["window_tokens"]),
            "residual_epoch_distribution": residual_epoch_distribution,
            "residual_epoch0_fraction": residual_epoch0_fraction,
            "residual_task_count": len(residual_epochs),
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
            "wall_clock_seconds": wall_clock,
            "worker_seconds": worker_seconds,
            "mean_peak_gpu_alloc_gb": peak_alloc,
            "mean_peak_gpu_reserved_gb": peak_reserved,
            "checkpoint_resume_supported": True,
            "checkpoint_interval_epochs": int(CONFIG["checkpoint_interval"]),
        },
        "integrity": {
            "shared_encoder_trajectory": True,
            "original_infonce_only": True,
            "train_only_dynamic_normalization": True,
            "validation_only_model_selection": True,
            "validation_only_threshold": True,
            "test_access_per_epoch": False,
            "duplicate_train_only": True,
            "experiment1_runtime_import": False,
            "abide2_paired_subject_ids_equal": True,
            "abide2_eligibility_label_blind": True,
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
    deltas = summary["deltas"]

    def aggregate(method: str, scope: str, metric: str) -> float:
        return float(aggregates[f"{method}::{scope}"][metric])

    table_rows: list[str] = []
    for label, scope in (("Internal", "internal_oof"), ("Caltech", "caltech"), ("ABIDE-II", "abide2")):
        baseline_auc = aggregate("baseline", scope, "auc_mean")
        treatment_auc = aggregate("stable_dynamic_residual", scope, "auc_mean")
        baseline_bce = aggregate("baseline", scope, "bce_mean")
        treatment_bce = aggregate("stable_dynamic_residual", scope, "bce_mean")
        table_rows.append(
            f"| {label} | {fmt(baseline_auc)} | {fmt(treatment_auc)} | "
            f"{fmt(treatment_auc - baseline_auc)} | {fmt(baseline_bce)} | "
            f"{fmt(treatment_bce)} | {fmt(treatment_bce - baseline_bce)} |"
        )

    treatment_task_rows = [
        row
        for row in rows
        if row["method"] == "stable_dynamic_residual"
        and row["scope"] in {"internal_fold", "caltech"}
    ]
    windows_min = mean_or_nan([row["min_windows"] for row in treatment_task_rows])
    windows_mean = mean_or_nan([row["mean_windows"] for row in treatment_task_rows])
    windows_max = mean_or_nan([row["max_windows"] for row in treatment_task_rows])
    node_switch = mean_or_nan([row["mean_node_switch"] for row in treatment_task_rows])
    node_deviation = mean_or_nan(
        [row["mean_node_deviation"] for row in treatment_task_rows]
    )
    selected_encoder_epochs = {
        "baseline": mean_or_nan(
            [row["selected_encoder_epoch"] for row in rows if row["method"] == "baseline"]
        ),
        "stable_dynamic_residual": mean_or_nan(
            [
                row["selected_encoder_epoch"]
                for row in rows
                if row["method"] == "stable_dynamic_residual"
            ]
        ),
    }
    runtime = summary["runtime"]
    diagnostics = summary["dynamic_diagnostics"]
    internal_deltas = ", ".join(fmt(value, 6) for value in deltas["internal_oof_auc"])
    abide2_deltas = ", ".join(fmt(value, 6) for value in deltas["abide2_auc"])
    decision = summary["decision"]
    interpretation = (
        "VarCoNet latent temporal variability contains complementary ASD-relevant information "
        "that is discarded by global stable connectome aggregation."
        if decision in {"STRONG_POSITIVE", "PROMISING"}
        else "从当前 VarCoNet latent space 提取这种 fixed-window dynamic residual，"
        "没有稳定转化为疾病预测收益；本实验按预注册标准封存，不进行调参救援。"
    )
    report = f"""# Experiment 2
Stable-Dynamic Residual VarCoNet

## 1. Problem

VarCoNet 的 global learned-FC 强调跨时间稳定 trait，但可能在最终时间聚合时丢弃 ASD-related dynamic switching information。

## 2. Literature rationale

- ASD functional-connectivity-dynamics switching work motivates testing temporal state changes.
- TSD-GCN motivates consecutive differential dynamics; VA-SDNet motivates ROI/graph variability summaries.
- SDF motivates static-dynamic complementarity, while DSAM motivates using latent temporal brain features.
- 这些概念仅用于特征定义；本实验没有复制 TCN、GNN、cross-attention 或任何完整外部网络。

## 3. Exact change

原 encoder、Conv1d kernel=4/stride=2、augmentation、optimizer/scheduler 与原 InfoNCE 均不变。每个 task 只训练一条 50-epoch SSL encoder trajectory；每个 epoch 从同一 encoder 批量提取 stable learned-FC `S` 和动态描述符 `D`。Treatment 是 frozen-stable residual logit：`W_s S + b + W_d D_z`，其中 `W_s,b` 冻结且 `W_d` 零初始化。

## 4. Dynamic feature definition

- ROI switching: 384 dimensions.
- ROI stable deviation: 384 dimensions.
- Graph switching mean/std: 2 dimensions.
- Graph stable-deviation mean/std: 2 dimensions.
- Consecutive-state transition-distance mean/std: 2 dimensions.
- Total: 774 dimensions; fixed non-overlapping latent window=15 tokens, stride=15.

## 5. Leakage / integrity

- No label, metadata, site, FD, TR, age or sex enters the encoder.
- Dynamic normalization uses only each epoch's training `D` mean/std.
- Classifier/encoder selection and Youden threshold use validation only.
- Test, Caltech and ABIDE-II are evaluated only after all 50 epochs are frozen.
- Duplicate ABIDE-I occurrences remain train-only; Caltech remains external-only.
- There is no Experiment 1 runtime import; original root source/data are not modified by this experiment.

### ABIDE-II technical eligibility

One ABIDE-II subject (sub-29623) was excluded from BOTH paired baseline and treatment evaluation because its scan was too short to construct the pre-specified two complete latent dynamic windows. The exclusion was determined solely by sequence-length eligibility and was independent of diagnosis or model predictions.

The eligibility audit found exactly one excluded subject: original index 568 (`sub-29623`). ABIDE-II therefore has original n=727 and paired evaluable n=726 for both methods. This is a technical evaluability constraint of the pre-specified dynamic descriptor, not a method optimization.

## 6. Efficiency

- One shared encoder trajectory per task, rather than separately retraining baseline/treatment encoders.
- 10 workers, five per GPU, six CPU threads per worker, full precision, no DDP/AMP/compile.
- Worker wall-clock maximum: {fmt(runtime['wall_clock_seconds'] / 3600.0, 2)} h; summed worker time: {fmt(runtime['worker_seconds'] / 3600.0, 2)} h.
- Mean worker peak GPU allocation/reservation: {fmt(runtime['mean_peak_gpu_alloc_gb'], 2)} / {fmt(runtime['mean_peak_gpu_reserved_gb'], 2)} GB.
- Rolling `latest`/`best` checkpoints are atomic; resume support: {runtime['checkpoint_resume_supported']}.

## 7. Main results

| Scope | Baseline AUC | SDR AUC | ΔAUC | Baseline BCE | SDR BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

10 internal repeat ΔAUC (SDR - baseline):
`[{internal_deltas}]`

10 ABIDE-II repeat ΔAUC (SDR - baseline):
`[{abide2_deltas}]`

Baseline sanity: internal OOF AUC={fmt(summary['baseline_sanity']['internal_auc'])} (reference 0.723); ABIDE-II AUC={fmt(summary['baseline_sanity']['abide2_auc'])} (reference 0.739); pipeline drift={summary['baseline_sanity']['pipeline_drift']}.

## 8. Dynamic diagnostics

- Dynamic feature finite: all final task rows are PASS; dim={diagnostics['dynamic_dim']}.
- Latent complete windows (task-average min/mean/max): {fmt(windows_min, 3)} / {fmt(windows_mean, 3)} / {fmt(windows_max, 3)}.
- Mean node switch: {fmt(node_switch, 6)}; mean node deviation: {fmt(node_deviation, 6)}.
- Selected residual-epoch distribution: `{json.dumps(diagnostics['residual_epoch_distribution'], ensure_ascii=False)}`.
- `residual_epoch=0` fraction: {fmt(diagnostics['residual_epoch0_fraction'], 4)} across {diagnostics['residual_task_count']} task selections.

## 9. Decision

**{decision}**

## 10. Interpretation

{interpretation}

## 11. Exact handoff

- Exact files: `model_scripts/VarCoNet.py`, copied baseline helper files, `dynamic_features.py`, `run_experiment2.py`, `run.sh`, `results/metrics.csv`, `results/summary.json`, `results/predictions.npz`, and `results/external_best_models.pt`.
- Exact formula: local latent cosine FC vs whole-scan latent cosine FC; 384 ROI switch + 384 ROI deviation + six graph/transition statistics.
- Fixed hyperparameters: AICHA 384 ROI, stable=73536, window/stride=15/15, 50 SSL epochs, classifier/residual lr=5e-5, 149 updates, 10 repeats × 10 folds.
- Selected encoder epoch means: baseline={fmt(selected_encoder_epochs['baseline'], 3)}, SDR={fmt(selected_encoder_epochs['stable_dynamic_residual'], 3)}. Per-task selected epochs and all paired metrics are in `metrics.csv`.
- The final decision and per-repeat deltas are recorded above and in `summary.json`; completed tasks have no retained per-fold weights. External selected models are bundled once in `external_best_models.pt`.
- The initial external evaluation correctly stopped on the one ineligible short ABIDE-II scan; after the authorized label-blind exclusion, all ten workers resumed from external epoch-50 checkpoints and completed inference without retraining. No numerical/runtime integrity errors remained in the resumed run.
"""
    (EXPERIMENT_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def smoke_marker_path() -> Path:
    return RESULTS_DIR / ".smoke_pass.json"


def smoke_test(device: torch.device) -> None:
    """The sole pre-flight smoke test, using one actual ABIDE-I batch."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    set_all_seeds(777)
    model_config, _ = make_model_config(cohort)
    encoder = VarCoNet(model_config, int(cohort.abide1.shape[2])).to(device)
    encoder.eval()
    selected = np.arange(8, dtype=np.int64)
    x = torch.from_numpy(
        np.asarray(cohort.abide1[cohort.internal_raw[selected]], dtype=np.float32)
    ).to(device)
    with torch.no_grad():
        old = encoder(x)
        output = encoder.forward_with_latent(x)
        stable = output["stable"]
        latent = output["latent"]
        valid_mask = output["valid_mask"]
        dynamic_extractor = LatentDynamicResidualExtractor(
            roi_num=int(cohort.abide1.shape[2]),
            window_tokens=int(CONFIG["window_tokens"]),
        ).to(device)
        dynamic, diagnostics = dynamic_extractor(
            latent, valid_mask, return_diagnostics=True
        )
    parity = float((old - stable).abs().max().detach().cpu())
    expected_stable = int(cohort.abide1.shape[2] * (cohort.abide1.shape[2] - 1) / 2)
    if parity > 1e-6:
        raise RuntimeError(f"baseline forward parity failed: {parity}")
    if tuple(stable.shape) != (len(selected), expected_stable):
        raise RuntimeError(f"unexpected stable shape {tuple(stable.shape)}")
    if latent.shape[0] != len(selected) or latent.shape[1] != 384:
        raise RuntimeError(f"unexpected latent shape {tuple(latent.shape)}")
    if tuple(dynamic.shape) != (len(selected), int(CONFIG["dynamic_dim"])):
        raise RuntimeError(f"unexpected dynamic shape {tuple(dynamic.shape)}")
    if not bool(torch.isfinite(dynamic).all()):
        raise RuntimeError("smoke dynamic feature is non-finite")
    if diagnostics["windows_min"] < 2.0:
        raise RuntimeError("smoke batch has fewer than two complete latent windows")
    if diagnostics["node_switch_mean"] == 0.0 or diagnostics["node_deviation_mean"] == 0.0:
        raise RuntimeError("smoke dynamic node statistics are all zero")

    labels = torch.as_tensor(cohort.internal_labels[selected], dtype=torch.long, device=device)
    stable_detached = stable.detach()
    dynamic_detached = dynamic.detach()
    baseline = MLP(expected_stable, 2).to(device)
    criterion = torch.nn.BCELoss()
    baseline_output = baseline(stable_detached)
    baseline_loss = criterion(baseline_output, F.one_hot(labels, num_classes=2).float())
    baseline_loss.backward()
    baseline_finite = bool(torch.isfinite(baseline_loss)) and finite_gradients(baseline)
    baseline_state = clone_state_cpu(baseline.state_dict())
    dyn_mean = dynamic_detached.mean(dim=0)
    dyn_std = dynamic_detached.std(dim=0, unbiased=False)
    dyn_std = torch.where(dyn_std < 1e-6, torch.ones_like(dyn_std), dyn_std)
    residual = FrozenStableResidualClassifier(
        baseline_state, int(CONFIG["dynamic_dim"])
    ).to(device)
    residual_output = residual(stable_detached, (dynamic_detached - dyn_mean) / dyn_std)
    residual_parity = float((baseline_output.detach() - residual_output.detach()).abs().max().cpu())
    if residual_parity > 1e-6:
        raise RuntimeError(f"residual epoch-0 prediction parity failed: {residual_parity}")
    residual_loss = criterion(residual_output, F.one_hot(labels, num_classes=2).float())
    residual_loss.backward()
    residual_finite = bool(torch.isfinite(residual_loss)) and finite_gradients(residual)
    if not baseline_finite or not residual_finite:
        raise RuntimeError("smoke classifier backward has a non-finite gradient/loss")
    payload = {
        "status": "PASS",
        "device": str(device),
        "baseline_forward_max_abs": parity,
        "residual_epoch0_max_abs": residual_parity,
        "stable_shape": list(stable.shape),
        "latent_shape": list(latent.shape),
        "dynamic_shape": list(dynamic.shape),
        "dynamic_finite": True,
        "diagnostics": diagnostics,
        "baseline_backward_finite": baseline_finite,
        "residual_backward_finite": residual_finite,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(smoke_marker_path(), payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    del encoder, dynamic_extractor, baseline, residual, x
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
        "p_HC": np.asarray([record["p_hc"] for record in records], dtype=np.float64),
        "threshold": np.asarray([record["threshold"] for record in records], dtype=np.float64),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_prediction_records(records: list[dict[str, Any]], cohort: Cohort) -> None:
    expected_per_scope = {"internal_oof": 882, "caltech": 37, "abide2": 726}
    for scope, count in expected_per_scope.items():
        for method in ("baseline", "stable_dynamic_residual"):
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
                and record["method"] == "stable_dynamic_residual"
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
    if len(expected_abide2_ids) != 726 or (568, 568) in expected_abide2_ids:
        raise RuntimeError("ABIDE-II eligibility index is not the authorized 726-subject set")
    for method in ("baseline", "stable_dynamic_residual"):
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
                    f"ABIDE-II/{method}/repeat={repeat} IDs do not equal the eligible 726-subject set"
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
        runtime_payload = {
            "repeat": repeat,
            "status": "PASS",
            "device": str(device),
            "elapsed_seconds": time.time() - started,
            "peak_gpu_alloc_gb": peak_alloc,
            "peak_gpu_reserved_gb": peak_reserved,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
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

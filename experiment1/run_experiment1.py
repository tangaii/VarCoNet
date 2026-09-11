#!/usr/bin/env python3
"""Complete paired Experiment 1: Subtype Prototype Guided VarCoNet.

This file is intentionally self-contained inside ``xiaolunwen/experiment1``.
It uses the copied VarCoNet source and reads only the original dataset files.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import Adam


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = EXPERIMENT_DIR / "results"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
sys.path.insert(0, str(EXPERIMENT_DIR))

from model_scripts.VarCoNet import VarCoNet  # noqa: E402
from model_scripts.classifier import MLP  # noqa: E402
from model_scripts.scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from subtype_prototype import (  # noqa: E402
    OnlineClassSubtypeBank,
    SubtypePrototypeContrastiveLoss,
)
from utils import (  # noqa: E402
    DualBranchContrast,
    InfoNCE,
    augment,
    removeDuplicates,
)


CONFIG: dict[str, Any] = {
    "atlas": "AICHA",
    "batch_size": 64,
    "min_length": 80,
    "epochs": 50,
    "warm_up_epochs": 10,
    "epochs_cls": 150,
    "lr_cls": 5e-5,
    "num_classes": 2,
    "lambda_proto": 1.0,
    "num_prototypes": 3,
    "prototype_momentum": 0.99,
    "sinkhorn_epsilon": 0.05,
    "sinkhorn_iters": 3,
    "checkpoint_interval": 10,
    # Ten one-repeat workers use 6 intra-op threads each (60 CPU threads
    # total), leaving headroom for the loader/OS on the 64-core host.
    "cpu_threads": 6,
}


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
    abide2_labels: np.ndarray


def set_all_seeds(seed: int) -> None:
    """Reset all stochastic sources at every paired task boundary."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(int(CONFIG["cpu_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before the first parallel region;
        # retaining the process-wide value is sufficient on later resets.
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_npz_stack(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        values = [np.asarray(archive[key], dtype=np.float32) for key in archive.files]
    if not values:
        raise RuntimeError(f"empty dataset archive: {path}")
    return np.stack(values, axis=0)


def load_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def load_cohort() -> Cohort:
    """Reproduce the original ABIDE-I duplicate/Caltech preparation exactly."""
    abide1_dir = DATASET_DIR / "ABIDEI"
    abide2_dir = DATASET_DIR / "ABIDEII"
    abide1 = load_npz_stack(abide1_dir / "ABIDEI_nilearn_AICHA.npz")
    abide1_names_raw = load_names(abide1_dir / "ABIDEI_nilearn_names.txt")
    abide1_labels_raw = np.load(
        abide1_dir / "ABIDEI_nilearn_classes.npy", allow_pickle=False
    ).astype(np.int64)
    if len(abide1) != len(abide1_names_raw) or len(abide1) != len(abide1_labels_raw):
        raise RuntimeError("ABIDE-I data/name/label lengths are not aligned")

    # This is the ordering in ASD_classification_ABIDEI.py: np.unique sorts
    # names, duplicate occurrences are all retained for training, while names
    # with exactly one occurrence form the candidate unique cohort.
    names_unique, counts = np.unique(np.asarray(abide1_names_raw), return_counts=True)
    duplicate_names_sorted = names_unique[counts > 1].tolist()
    duplicate_raw_parts: list[int] = []
    duplicate_names: list[str] = []
    for name in duplicate_names_sorted:
        positions = np.where(np.asarray(abide1_names_raw) == name)[0]
        duplicate_raw_parts.extend(int(x) for x in positions.tolist())
        duplicate_names.extend([str(name)] * len(positions))

    singleton_names = names_unique[counts == 1].tolist()
    singleton_raw = np.asarray(
        [int(np.where(np.asarray(abide1_names_raw) == name)[0][0]) for name in singleton_names],
        dtype=np.int64,
    )
    duplicate_raw = np.asarray(duplicate_raw_parts, dtype=np.int64)
    duplicate_labels = abide1_labels_raw[duplicate_raw]

    caltech_names = [
        f"sub-00{numeric_id}"
        for numeric_id in range(51456, 51494)
        if f"sub-00{numeric_id}" in singleton_names
    ]
    singleton_name_to_raw = {name: int(raw) for name, raw in zip(singleton_names, singleton_raw)}
    caltech_raw = np.asarray([singleton_name_to_raw[name] for name in caltech_names], dtype=np.int64)
    caltech_name_set = set(caltech_names)
    internal_names = [name for name in singleton_names if name not in caltech_name_set]
    internal_raw = np.asarray([singleton_name_to_raw[name] for name in internal_names], dtype=np.int64)

    abide1_labels = abide1_labels_raw[singleton_raw]
    internal_labels = abide1_labels_raw[internal_raw]
    caltech_labels = abide1_labels_raw[caltech_raw]

    abide2 = load_npz_stack(abide2_dir / "ABIDEII_nilearn_AICHA.npz")
    abide2_labels = np.load(
        abide2_dir / "ABIDEII_nilearn_classes.npy", allow_pickle=False
    ).astype(np.int64)
    if len(abide2) != len(abide2_labels):
        raise RuntimeError("ABIDE-II data/label lengths are not aligned")
    if abide1.shape[1:] != abide2.shape[1:]:
        raise RuntimeError(f"AICHA shapes differ: ABIDE-I={abide1.shape}, ABIDE-II={abide2.shape}")
    if len(internal_names) != 882 or len(caltech_names) != 37 or len(duplicate_raw) != 76:
        raise RuntimeError(
            "Unexpected ABIDE-I cohort preparation: "
            f"internal={len(internal_names)}, caltech={len(caltech_names)}, "
            f"duplicate_train_occurrences={len(duplicate_raw)}"
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
        abide2_labels=abide2_labels,
    )


def make_tasks(cohort: Cohort, repeat: int) -> list[Task]:
    """Create one repeat using the original 10-fold and Caltech splits."""
    tasks: list[Task] = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + int(repeat))
    for fold, (train_index, test_index) in enumerate(
        skf.split(cohort.internal_raw, cohort.internal_labels)
    ):
        candidate_raw = cohort.internal_raw[train_index]
        candidate_y = cohort.internal_labels[train_index]
        candidate_names = [cohort.internal_names[int(i)] for i in train_index]
        train_raw, val_raw, train_y, val_y, _, _ = train_test_split(
            candidate_raw,
            candidate_y,
            np.arange(len(candidate_raw)),
            test_size=0.15,
            random_state=42,
            stratify=candidate_y,
        )
        # train_test_split preserves the selected order in each returned array;
        # prepend every duplicate occurrence exactly as the root script does.
        # Derive names from raw scan IDs so duplicate replacement sees the same
        # order as the source train_data list, including repeated names.
        raw_to_name = {
            int(raw): name for raw, name in zip(cohort.internal_raw, cohort.internal_names)
        }
        train_names = cohort.duplicate_names + [raw_to_name[int(x)] for x in train_raw]
        full_train_raw = np.concatenate([cohort.duplicate_raw, np.asarray(train_raw, dtype=np.int64)])
        full_train_y = np.concatenate([cohort.duplicate_labels, np.asarray(train_y, dtype=np.int64)])
        test_raw = cohort.internal_raw[test_index]
        test_y = cohort.internal_labels[test_index]
        tasks.append(
            Task(
                repeat=int(repeat),
                scope="internal_cv",
                fold=str(fold),
                seed=100000 + int(repeat) * 100 + int(fold),
                train_raw=full_train_raw,
                train_names=train_names,
                train_labels=full_train_y,
                val_raw=np.asarray(val_raw, dtype=np.int64),
                val_labels=np.asarray(val_y, dtype=np.int64),
                test_raw=np.asarray(test_raw, dtype=np.int64),
                test_labels=np.asarray(test_y, dtype=np.int64),
            )
        )

    train_raw, val_raw, train_y, val_y, _, _ = train_test_split(
        cohort.internal_raw,
        cohort.internal_labels,
        np.arange(len(cohort.internal_raw)),
        test_size=0.1,
        random_state=42 + int(repeat),
        stratify=cohort.internal_labels,
    )
    raw_to_name = {
        int(raw): name for raw, name in zip(cohort.internal_raw, cohort.internal_names)
    }
    ext_train_names = cohort.duplicate_names + [raw_to_name[int(x)] for x in train_raw]
    ext_train_raw = np.concatenate([cohort.duplicate_raw, np.asarray(train_raw, dtype=np.int64)])
    ext_train_y = np.concatenate([cohort.duplicate_labels, np.asarray(train_y, dtype=np.int64)])
    tasks.append(
        Task(
            repeat=int(repeat),
            scope="caltech_external",
            fold="external",
            seed=200000 + int(repeat),
            train_raw=ext_train_raw,
            train_names=ext_train_names,
            train_labels=ext_train_y,
            val_raw=np.asarray(val_raw, dtype=np.int64),
            val_labels=np.asarray(val_y, dtype=np.int64),
            test_raw=cohort.caltech_raw,
            test_labels=cohort.caltech_labels,
        )
    )
    return tasks


def paired_batch_schedule(names: list[str], n_items: int, epochs: int, seed: int) -> list[list[list[int]]]:
    """Precompute identical source-style batches for baseline and treatment."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    schedule: list[list[list[int]]] = []
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(int(n_items), generator=generator).tolist()
        epoch_batches: list[list[int]] = []
        for batch_idx, start in enumerate(range(0, len(permutation), CONFIG["batch_size"])):
            raw = [int(x) for x in permutation[start : start + CONFIG["batch_size"]]]
            saved_state = random.getstate()
            random.seed(int(seed) + epoch * 1000003 + batch_idx)
            selected = removeDuplicates(names, raw)
            random.setstate(saved_state)
            epoch_batches.append([int(x) for x in selected])
        schedule.append(epoch_batches)
    return schedule


def encode_indices(
    encoder: torch.nn.Module,
    data: np.ndarray,
    raw_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(raw_indices), int(CONFIG["batch_size"])):
            selected = raw_indices[start : start + int(CONFIG["batch_size"])]
            arr = np.asarray(data[selected], dtype=np.float32)
            outputs.append(encoder(torch.from_numpy(arr).to(device)).detach())
    if not outputs:
        raise RuntimeError("cannot encode an empty split")
    return torch.cat(outputs, dim=0)


def validation_threshold(y_true: np.ndarray, p_hc: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), p_hc.astype(float))
    j = tpr - fpr
    best = np.nanmax(j)
    candidates = np.where(np.isclose(j, best, rtol=0.0, atol=1e-12))[0]
    finite = [int(i) for i in candidates.tolist() if np.isfinite(thresholds[i])]
    idx = finite[0] if finite else int(candidates[0])
    return float(thresholds[idx]) if np.isfinite(thresholds[idx]) else 0.5


def classification_metrics(y_true: np.ndarray, p_hc: np.ndarray, threshold: float) -> dict[str, float]:
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


def clone_state_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_to_cpu(value: Any) -> Any:
    """Detach tensor state before writing a rolling checkpoint to disk."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_to_cpu(item) for item in value)
    return value


def rolling_checkpoint_path() -> Path:
    """One latest checkpoint per worker; each save atomically replaces the old one."""
    worker_id = os.environ.get("VARCONET_WORKER_ID", "manual")
    safe_worker_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", worker_id)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"worker_{safe_worker_id}_latest.pt"


def save_rolling_checkpoint(
    path: Path,
    *,
    task: Task,
    method: str,
    epoch: int,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    classifier_state: dict[str, torch.Tensor],
    bank: OnlineClassSubtypeBank | None,
    best_val_loss: float,
    best_epoch: int | None,
    best_encoder_state: dict[str, torch.Tensor] | None,
) -> None:
    """Save the current training state while retaining only the newest file."""
    payload: dict[str, Any] = {
        "format_version": 1,
        "saved_unix": time.time(),
        "repeat": int(task.repeat),
        "scope": task.scope,
        "fold": task.fold,
        "method": method,
        "task_seed": int(task.seed),
        "epoch": int(epoch),
        "epochs": int(CONFIG["epochs"]),
        "encoder_state_dict": clone_state_cpu(encoder.state_dict()),
        "selected_encoder_state_dict": (
            None if best_encoder_state is None else state_to_cpu(best_encoder_state)
        ),
        "classifier_state_dict": state_to_cpu(classifier_state),
        "optimizer_state_dict": state_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": state_to_cpu(scheduler.state_dict()),
        "prototype_bank": None,
        "best_val_loss": float(best_val_loss),
        "best_encoder_epoch": None if best_epoch is None else int(best_epoch),
    }
    if bank is not None:
        payload["prototype_bank"] = {
            "prototypes": None if bank.prototypes is None else bank.prototypes.detach().cpu().clone(),
            "initialized": None if bank.initialized is None else bank.initialized.detach().cpu().clone(),
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    # os.replace is atomic on the same filesystem and removes the previous
    # latest checkpoint as part of the replacement.
    os.replace(temporary, path)
    print(
        f"CHECKPOINT repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"method={method} epoch={epoch:02d}/{CONFIG['epochs']} path={path}",
        flush=True,
    )


def classifier_eval(
    encoder: torch.nn.Module,
    cohort: Cohort,
    task: Task,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Train the copied linear classifier and select by validation BCE only."""
    train_features = encode_indices(encoder, cohort.abide1, task.train_raw, device)
    val_features = encode_indices(encoder, cohort.abide1, task.val_raw, device)
    train_labels = torch.as_tensor(task.train_labels, dtype=torch.long, device=device)
    val_labels = torch.as_tensor(task.val_labels, dtype=torch.long, device=device)

    set_all_seeds(seed)
    classifier = MLP(int(train_features.shape[1]), int(CONFIG["num_classes"])).to(device)
    optimizer = Adam(classifier.parameters(), lr=float(CONFIG["lr_cls"]))
    criterion = torch.nn.BCELoss()
    best_val_loss = float("inf")
    best_classifier_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for _epoch in range(1, int(CONFIG["epochs_cls"])):
        classifier.train()
        optimizer.zero_grad()
        train_output = classifier(train_features)
        train_loss = criterion(
            train_output,
            F.one_hot(train_labels, num_classes=2).float(),
        )
        train_loss.backward()
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            val_output = classifier(val_features)
            val_loss = float(
                criterion(val_output, F.one_hot(val_labels, num_classes=2).float()).detach().cpu()
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_classifier_epoch = int(_epoch)
            best_state = clone_state_cpu(classifier.state_dict())
    if best_state is None or best_classifier_epoch is None:
        raise RuntimeError("classifier did not produce a validation checkpoint")

    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        p_train = classifier(train_features)[:, -1].detach().cpu().numpy()
        p_val = classifier(val_features)[:, -1].detach().cpu().numpy()
    y_train = task.train_labels.astype(np.int64)
    y_val = task.val_labels.astype(np.int64)
    threshold = validation_threshold(y_val, p_val)
    return {
        "best_val_loss": float(best_val_loss),
        "best_classifier_epoch": int(best_classifier_epoch),
        "threshold": float(threshold),
        "p_train": p_train,
        "p_val": p_val,
        "y_train": y_train,
        "y_val": y_val,
    }, best_state


def test_with_selected(
    encoder: torch.nn.Module,
    classifier_state: dict[str, torch.Tensor],
    data: np.ndarray,
    raw_indices: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    features = encode_indices(encoder, data, raw_indices, device)
    classifier = MLP(int(classifier_state["fc.0.weight"].shape[1]), 2).to(device)
    classifier.load_state_dict(classifier_state)
    classifier.eval()
    with torch.no_grad():
        probs = classifier(features)[:, -1].detach().cpu().numpy()
    return classification_metrics(labels, probs, threshold), probs


def metric_row(
    method: str,
    scope: str,
    repeat: int,
    fold: str,
    metrics: dict[str, float],
    selected_epoch: float,
    threshold: float,
    diagnostics: dict[str, float],
    n_test: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "scope": scope,
        "repeat": int(repeat),
        "fold": str(fold),
        "auc": float(metrics["auc"]),
        "bce": float(metrics["bce"]),
        "f1": float(metrics["f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "selected_encoder_epoch": float(selected_epoch),
        "threshold_hc": float(threshold),
        "mean_identity_loss": float(diagnostics["mean_identity_loss"]),
        "mean_prototype_loss": float(diagnostics["mean_prototype_loss"]),
        "prototype_active": float(diagnostics["prototype_active"]),
        "mean_min_prototype_occupancy": float(diagnostics["mean_min_prototype_occupancy"]),
        "mean_max_prototype_occupancy": float(diagnostics["mean_max_prototype_occupancy"]),
        "n_test": int(n_test),
        "status": "PASS",
    }


def train_task(
    cohort: Cohort,
    task: Task,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    """Train one baseline/treatment arm and evaluate its selected checkpoint."""
    task_start = time.time()
    checkpoint_path = rolling_checkpoint_path()
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp").unlink(missing_ok=True)
    print(
        f"TASK_START repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"method={method} seed={task.seed} device={device}",
        flush=True,
    )
    set_all_seeds(task.seed)
    max_length = int(cohort.abide1.shape[1])
    roi_num = int(cohort.abide1.shape[2])
    with (EXPERIMENT_DIR / "best_params_VarCoNet_AICHA.pkl").open("rb") as handle:
        best_params = pickle.load(handle)
    model_config = {
        "layers": int(best_params["layers"]),
        "n_heads": int(best_params["n_heads"]),
        "dim_feedforward": int(best_params["dim_feedforward"]),
        "max_length": max_length,
    }
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
    treatment = method == "treatment"
    bank = (
        OnlineClassSubtypeBank(
            num_classes=2,
            num_prototypes=int(CONFIG["num_prototypes"]),
            momentum=float(CONFIG["prototype_momentum"]),
            sinkhorn_epsilon=float(CONFIG["sinkhorn_epsilon"]),
            sinkhorn_iters=int(CONFIG["sinkhorn_iters"]),
        )
        if treatment
        else None
    )
    prototype_criterion = (
        SubtypePrototypeContrastiveLoss(temperature=float(best_params["tau"]))
        if treatment
        else None
    )
    schedule = paired_batch_schedule(
        task.train_names,
        len(task.train_raw),
        int(CONFIG["epochs"]),
        task.seed,
    )

    identity_losses: list[float] = []
    prototype_losses: list[float] = []
    min_occupancies: list[float] = []
    max_occupancies: list[float] = []
    active_seen = torch.zeros(2, int(CONFIG["num_prototypes"]), dtype=torch.bool)
    best_val_loss = float("inf")
    best_epoch: int | None = None
    best_encoder_state: dict[str, torch.Tensor] | None = None
    best_classifier_state: dict[str, torch.Tensor] | None = None
    best_result: dict[str, Any] | None = None

    for epoch in range(1, int(CONFIG["epochs"]) + 1):
        encoder.train()
        epoch_identity_losses: list[float] = []
        epoch_prototype_losses: list[float] = []
        for selected in schedule[epoch - 1]:
            selected_np = np.asarray(selected, dtype=np.int64)
            batch_array = np.asarray(cohort.abide1[task.train_raw[selected_np]], dtype=np.float32)
            batch_data = torch.from_numpy(batch_array)
            views = augment(
                batch_data,
                [int(CONFIG["min_length"]), max_length],
                device,
            )
            # Labels are fetched after removeDuplicates() changed sample_inds.
            batch_labels = torch.as_tensor(task.train_labels[selected_np], dtype=torch.long, device=device)
            optimizer.zero_grad()
            z1 = encoder(views[0])
            z2 = encoder(views[1])
            identity_loss = contrast_model(z1, z2)
            proto_loss = torch.zeros((), device=device)
            if treatment:
                z1n = F.normalize(z1, p=2, dim=1)
                z2n = F.normalize(z2, p=2, dim=1)
                zbar = F.normalize((z1n + z2n) / 2.0, p=2, dim=1)
                with torch.no_grad():
                    subtype_ids = bank.assign(zbar.detach(), batch_labels)
                proto_loss = prototype_criterion(
                    z1,
                    z2,
                    batch_labels,
                    subtype_ids,
                    bank.prototypes,
                )
                total_loss = identity_loss + float(CONFIG["lambda_proto"]) * proto_loss
            else:
                zbar = None
                subtype_ids = None
                total_loss = identity_loss
            total_loss.backward()
            for parameter in encoder.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {method} task {task.repeat}/{task.fold}")
            optimizer.step()

            identity_value = float(identity_loss.detach().cpu())
            identity_losses.append(identity_value)
            epoch_identity_losses.append(identity_value)
            if treatment:
                prototype_value = float(proto_loss.detach().cpu())
                prototype_losses.append(prototype_value)
                epoch_prototype_losses.append(prototype_value)
                with torch.no_grad():
                    counts = torch.zeros(2, int(CONFIG["num_prototypes"]), dtype=torch.long, device=device)
                    for c in range(2):
                        for k in range(int(CONFIG["num_prototypes"])):
                            counts[c, k] = torch.sum((batch_labels == c) & (subtype_ids == k))
                    active_seen |= (counts.detach().cpu() > 0)
                    present = torch.where(batch_labels.detach().cpu().bincount(minlength=2) > 0)[0]
                    for c in present.tolist():
                        min_occupancies.append(float(counts[c].min().detach().cpu()))
                        max_occupancies.append(float(counts[c].max().detach().cpu()))
                    bank.update(zbar.detach(), batch_labels, subtype_ids)

        scheduler.step()
        # The classifier is selected by validation BCE.  Test/Caltech/ABIDE-II
        # probabilities are not touched until the winning encoder is fixed.
        epoch_result, classifier_state = classifier_eval(
            encoder,
            cohort,
            task,
            device,
            task.seed + 100000 + epoch,
        )
        is_new_best = epoch_result["best_val_loss"] < best_val_loss
        if is_new_best:
            best_val_loss = float(epoch_result["best_val_loss"])
            best_epoch = epoch
            best_encoder_state = clone_state_cpu(encoder.state_dict())
            best_classifier_state = classifier_state
            best_result = epoch_result
        if epoch % int(CONFIG["checkpoint_interval"]) == 0:
            save_rolling_checkpoint(
                checkpoint_path,
                task=task,
                method=method,
                epoch=epoch,
                encoder=encoder,
                optimizer=optimizer,
                scheduler=scheduler,
                classifier_state=classifier_state,
                bank=bank,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                best_encoder_state=best_encoder_state,
            )
        mean_identity_epoch = float(np.mean(epoch_identity_losses))
        mean_prototype_epoch = (
            float(np.mean(epoch_prototype_losses)) if epoch_prototype_losses else 0.0
        )
        print(
            f"EPOCH repeat={task.repeat} scope={task.scope} fold={task.fold} "
            f"method={method} epoch={epoch:02d}/{CONFIG['epochs']} "
            f"identity_loss={mean_identity_epoch:.6f} "
            f"prototype_loss={mean_prototype_epoch:.6f} "
            f"total_loss={mean_identity_epoch + mean_prototype_epoch:.6f} "
            f"val_bce={epoch_result['best_val_loss']:.6f} "
            f"val_cls_epoch={epoch_result['best_classifier_epoch']} "
            f"best={'Y' if is_new_best else 'N'} elapsed_min={(time.time() - task_start) / 60:.1f}",
            flush=True,
        )

    if best_epoch is None or best_encoder_state is None or best_classifier_state is None or best_result is None:
        raise RuntimeError("no validation-selected encoder/classifier checkpoint")

    encoder.load_state_dict(best_encoder_state)
    encoder.eval()
    test_metrics, test_probs = test_with_selected(
        encoder,
        best_classifier_state,
        cohort.abide1,
        task.test_raw,
        task.test_labels,
        float(best_result["threshold"]),
        device,
    )
    if treatment:
        proto_active = float(active_seen.sum().item())
        min_occ = float(np.mean(min_occupancies)) if min_occupancies else 0.0
        max_occ = float(np.mean(max_occupancies)) if max_occupancies else 0.0
        proto_mean = float(np.mean(prototype_losses)) if prototype_losses else float("nan")
    else:
        proto_active = 0.0
        min_occ = float("nan")
        max_occ = float("nan")
        proto_mean = float("nan")
    diagnostics = {
        "mean_identity_loss": float(np.mean(identity_losses)),
        "mean_prototype_loss": proto_mean,
        "prototype_active": proto_active,
        "mean_min_prototype_occupancy": min_occ,
        "mean_max_prototype_occupancy": max_occ,
    }
    row = metric_row(
        method,
        "internal_fold" if task.scope == "internal_cv" else "caltech",
        task.repeat,
        task.fold,
        test_metrics,
        float(best_epoch),
        float(best_result["threshold"]),
        diagnostics,
        len(task.test_raw),
    )
    result = {
        "task": task,
        "method": method,
        "row": row,
        "y_test": task.test_labels.astype(np.int64),
        "p_test": np.asarray(test_probs, dtype=np.float64),
        "threshold": float(best_result["threshold"]),
        "selected_epoch": int(best_epoch),
        "diagnostics": diagnostics,
        "encoder": encoder,
        "classifier_state": best_classifier_state,
    }
    print(
        f"TASK_DONE repeat={task.repeat} scope={task.scope} fold={task.fold} "
        f"method={method} selected_encoder_epoch={best_epoch} "
        f"test_auc={test_metrics['auc']:.6f} test_bce={test_metrics['bce']:.6f} "
        f"mean_identity_loss={diagnostics['mean_identity_loss']:.6f} "
        f"mean_prototype_loss={diagnostics['mean_prototype_loss']:.6f} "
        f"elapsed_min={(time.time() - task_start) / 60:.1f}",
        flush=True,
    )
    return result


def summarize_oof(method_results: list[dict[str, Any]], method: str, repeat: int) -> dict[str, Any]:
    y_parts = [result["y_test"] for result in method_results]
    p_parts = [result["p_test"] for result in method_results]
    y = np.concatenate(y_parts)
    p = np.concatenate(p_parts)
    pred_parts = [
        (result["p_test"] >= float(result["threshold"])).astype(np.int64)
        for result in method_results
    ]
    pred_hc = np.concatenate(pred_parts)
    y_asd = (y == 0).astype(np.int64)
    pred_asd = (pred_hc == 0).astype(np.int64)
    metrics = {
        "auc": float(roc_auc_score(y, p)),
        "bce": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "f1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred_hc)),
    }
    diagnostics = {
        key: float(np.nanmean([r["diagnostics"][key] for r in method_results]))
        for key in [
            "mean_identity_loss",
            "mean_prototype_loss",
            "prototype_active",
            "mean_min_prototype_occupancy",
            "mean_max_prototype_occupancy",
        ]
    }
    return metric_row(
        method,
        "internal_oof",
        repeat,
        "all",
        metrics,
        float(np.mean([r["selected_epoch"] for r in method_results])),
        float("nan"),
        diagnostics,
        len(y),
    )


def external_abide2_row(
    cohort: Cohort,
    result: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    metrics, _ = test_with_selected(
        result["encoder"],
        result["classifier_state"],
        cohort.abide2,
        np.arange(len(cohort.abide2), dtype=np.int64),
        cohort.abide2_labels,
        float(result["threshold"]),
        device,
    )
    return metric_row(
        result["method"],
        "abide2",
        result["task"].repeat,
        "external",
        metrics,
        result["selected_epoch"],
        result["threshold"],
        result["diagnostics"],
        len(cohort.abide2),
    )


def run_repeat(cohort: Cohort, repeat: int, device: torch.device) -> dict[str, Any]:
    print(f"REPEAT_START repeat={repeat} device={device} tasks=11", flush=True)
    tasks = make_tasks(cohort, repeat)
    internal: dict[str, list[dict[str, Any]]] = {"baseline": [], "treatment": []}
    rows: list[dict[str, Any]] = []
    # Each pair is run consecutively on the same GPU and resets the same seed.
    for task in tasks:
        if task.scope == "internal_cv":
            for method in ("baseline", "treatment"):
                result = train_task(cohort, task, method, device)
                internal[method].append(result)
                rows.append(result["row"])
                if method == "treatment":
                    abide2_row = None
                del result["encoder"]
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # The external model is paired in the same way; its selected threshold is
    # used for both Caltech and ABIDE-II and is never re-fit on external labels.
    external_results: dict[str, dict[str, Any]] = {}
    external_task = tasks[-1]
    for method in ("baseline", "treatment"):
        result = train_task(cohort, external_task, method, device)
        external_results[method] = result
        rows.append(result["row"])
        rows.append(external_abide2_row(cohort, result, device))

    for method in ("baseline", "treatment"):
        rows.append(summarize_oof(internal[method], method, repeat))

    # Keep only JSON-serializable summaries in the worker shard.
    repeat_metrics = {
        f"{row['method']}::{row['scope']}": row
        for row in rows
        if row["scope"] in {"internal_oof", "caltech", "abide2"}
    }
    return {"repeat": int(repeat), "rows": rows, "summary_rows": repeat_metrics}


def smoke_test(device: torch.device) -> None:
    cohort = load_cohort()
    set_all_seeds(777)
    model_config = {
        "layers": 1,
        "n_heads": 2,
        "dim_feedforward": 512,
        "max_length": int(cohort.abide1.shape[1]),
    }
    encoder = VarCoNet(model_config, int(cohort.abide1.shape[2])).to(device)
    contrast = DualBranchContrast(loss=InfoNCE(tau=0.05410089573253199), mode="L2L").to(device)
    optimizer = Adam(encoder.parameters(), lr=1e-4)
    selected = np.arange(8, dtype=np.int64)
    batch = torch.from_numpy(np.asarray(cohort.abide1[cohort.internal_raw[selected]], dtype=np.float32))
    views = augment(batch, [80, int(cohort.abide1.shape[1])], device)
    labels = torch.as_tensor(cohort.internal_labels[selected], dtype=torch.long, device=device)
    z1, z2 = encoder(views[0]), encoder(views[1])
    bank = OnlineClassSubtypeBank()
    criterion = SubtypePrototypeContrastiveLoss(temperature=0.05410089573253199)
    zbar = F.normalize(
        (F.normalize(z1, dim=1) + F.normalize(z2, dim=1)) / 2.0,
        dim=1,
    )
    subtype_ids = bank.assign(zbar.detach(), labels)
    identity = contrast(z1, z2)
    prototype = criterion(z1, z2, labels, subtype_ids, bank.prototypes)
    loss = identity + prototype
    optimizer.zero_grad()
    loss.backward()
    finite_grad = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in encoder.parameters()
    )
    if not torch.isfinite(loss) or not finite_grad:
        raise RuntimeError("smoke test produced non-finite loss/gradient")
    bank.update(zbar.detach(), labels, subtype_ids)
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": str(device),
                "z_shape": list(z1.shape),
                "prototype_shape": list(bank.prototypes.shape),
                "subtype_ids": subtype_ids.detach().cpu().tolist(),
                "identity_loss": float(identity.detach().cpu()),
                "prototype_loss": float(prototype.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "finite_gradient": finite_grad,
            },
            ensure_ascii=False,
        )
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
    "threshold_hc",
    "mean_identity_loss",
    "mean_prototype_loss",
    "prototype_active",
    "mean_min_prototype_occupancy",
    "mean_max_prototype_occupancy",
    "n_test",
    "status",
]


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def worker_main(repeats: list[int], device_name: str, worker_id: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["VARCONET_WORKER_ID"] = str(worker_id)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    cohort = load_cohort()
    status_rows = []
    for repeat in repeats:
        shard_path = RESULTS_DIR / f"worker_{worker_id}_repeat_{repeat}.json"
        if shard_path.exists() and shard_path.stat().st_size > 0:
            print(f"WORKER {worker_id}: repeat {repeat} already complete; skip", flush=True)
            continue
        start = time.time()
        print(f"WORKER {worker_id}: starting repeat {repeat} on {device}", flush=True)
        try:
            payload = run_repeat(cohort, repeat, device)
            payload["elapsed_seconds"] = time.time() - start
            payload["status"] = "PASS"
            write_json_atomic(shard_path, payload)
            status_rows.append({"repeat": repeat, "status": "PASS", "elapsed_seconds": payload["elapsed_seconds"]})
            print(
                f"WORKER {worker_id}: repeat {repeat} PASS ({payload['elapsed_seconds'] / 60:.1f} min)",
                flush=True,
            )
        except Exception as exc:
            error_payload = {
                "repeat": repeat,
                "status": "FAIL",
                "error": repr(exc),
            }
            write_json_atomic(RESULTS_DIR / f"worker_{worker_id}_repeat_{repeat}.error.json", error_payload)
            print(f"WORKER {worker_id}: repeat {repeat} FAIL: {exc!r}", flush=True)
            raise
    write_json_atomic(RESULTS_DIR / f"worker_{worker_id}_status.json", status_rows)


def mean_or_nan(values: list[float]) -> float:
    values = [float(x) for x in values if np.isfinite(float(x))]
    return float(np.mean(values)) if values else float("nan")


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def select(method: str, scope: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["method"] == method and row["scope"] == scope]

    aggregates: dict[str, Any] = {}
    for method in ("baseline", "treatment"):
        for scope in ("internal_oof", "caltech", "abide2"):
            selected = select(method, scope)
            if not selected:
                continue
            aggregates[f"{method}::{scope}"] = {
                "n": len(selected),
                "auc_mean": mean_or_nan([row["auc"] for row in selected]),
                "auc_std": float(np.std([row["auc"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0,
                "bce_mean": mean_or_nan([row["bce"] for row in selected]),
                "bce_std": float(np.std([row["bce"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0,
                "f1_mean": mean_or_nan([row["f1"] for row in selected]),
                "balanced_accuracy_mean": mean_or_nan([row["balanced_accuracy"] for row in selected]),
            }

    deltas: dict[str, list[float]] = {}
    for scope in ("internal_oof", "caltech", "abide2"):
        base = {int(row["repeat"]): row for row in select("baseline", scope)}
        treatment = {int(row["repeat"]): row for row in select("treatment", scope)}
        common = sorted(set(base) & set(treatment))
        deltas[f"{scope}_auc"] = [float(treatment[i]["auc"] - base[i]["auc"]) for i in common]
        deltas[f"{scope}_bce"] = [float(treatment[i]["bce"] - base[i]["bce"]) for i in common]

    internal_auc = aggregates.get("baseline::internal_oof", {}).get("auc_mean", float("nan"))
    abide2_auc = aggregates.get("baseline::abide2", {}).get("auc_mean", float("nan"))
    pipeline_drift = (
        not np.isfinite(internal_auc)
        or not np.isfinite(abide2_auc)
        or abs(float(internal_auc) - 0.723) > 0.015
        or abs(float(abide2_auc) - 0.739) > 0.015
    )
    internal_auc_delta = mean_or_nan(deltas.get("internal_oof_auc", []))
    abide2_auc_delta = mean_or_nan(deltas.get("abide2_auc", []))
    internal_bce_delta = mean_or_nan(deltas.get("internal_oof_bce", []))
    abide2_bce_delta = mean_or_nan(deltas.get("abide2_bce", []))
    internal_positive = sum(delta > 0 for delta in deltas.get("internal_oof_auc", []))
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
    return {
        "status": "PASS" if len(rows) > 0 else "FAIL",
        "config": CONFIG,
        "dataset": {
            "atlas": "AICHA",
            "abide1_raw": 995,
            "abide1_internal_unique_names": 882,
            "caltech_external": 37,
            "abide1_duplicate_train_occurrences": 76,
            "abide2_external": 727,
        },
        "aggregates": aggregates,
        "deltas": deltas,
        "baseline_sanity": {
            "internal_auc_reference": 0.723,
            "abide2_auc_reference": 0.739,
            "internal_auc": internal_auc,
            "abide2_auc": abide2_auc,
            "pipeline_drift": pipeline_drift,
        },
        "decision": decision,
        "completion": {
            "expected_internal_folds_per_method": 100,
            "expected_internal_oof_per_method": 10,
            "expected_caltech_per_method": 10,
            "expected_abide2_per_method": 10,
            "metrics_rows": len(rows),
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    decision = summary["decision"]
    aggregates = summary["aggregates"]
    deltas = summary["deltas"]
    def agg(method: str, scope: str, metric: str) -> Any:
        return aggregates.get(f"{method}::{scope}", {}).get(metric, float("nan"))

    table_rows = []
    for label, scope in (("Internal", "internal_oof"), ("Caltech", "caltech"), ("ABIDE-II", "abide2")):
        ba = agg("baseline", scope, "auc_mean")
        ta = agg("treatment", scope, "auc_mean")
        bb = agg("baseline", scope, "bce_mean")
        tb = agg("treatment", scope, "bce_mean")
        table_rows.append(
            f"| {label} | {fmt(ba)} | {fmt(ta)} | {fmt(ta - ba)} | {fmt(bb)} | {fmt(tb)} | {fmt(tb - bb)} |"
        )
    internal_deltas = ", ".join(fmt(x, 6) for x in deltas.get("internal_oof_auc", []))
    abide2_deltas = ", ".join(fmt(x, 6) for x in deltas.get("abide2_auc", []))
    treatment_rows = [row for row in rows if row["method"] == "treatment" and row["scope"] in {"internal_oof", "caltech", "abide2"}]
    proto_losses = [row["mean_prototype_loss"] for row in treatment_rows]
    proto_active = [row["prototype_active"] for row in treatment_rows]
    min_occ = [row["mean_min_prototype_occupancy"] for row in treatment_rows]
    max_occ = [row["mean_max_prototype_occupancy"] for row in treatment_rows]
    report = f"""# Experiment 1
Subtype Prototype Guided VarCoNet

## 1. Research question
在保留 VarCoNet 原始 subject-identity InfoNCE 的前提下，训练期 class-conditional online subtype prototype guidance 是否提高 AICHA ASD prediction，尤其是 ABIDE-II external generalization？

## 2. Exact change from baseline
Baseline 的 VarCoNet encoder、learned-FC upper-triangle、原始 InfoNCE、augmentation、optimizer、scheduler 和线性 classifier 均保持不变。Treatment 仅增加训练期 `OnlineClassSubtypeBank` 与 `SubtypePrototypeContrastiveLoss`，总损失为 `loss_identity + 1.0 * loss_prototype`。Prototype 不进入 optimizer，测试/validation/Caltech/ABIDE-II 不使用 prototype。

## 3. Files
本实验所有新增源码均位于 `xiaolunwen/experiment1`：复制的 `model_scripts/VarCoNet.py`、`classifier.py`、`evaluation.py`、`scheduler.py`、`utils.py`，`subtype_prototype.py`，`run_experiment1.py`，`best_params_VarCoNet_AICHA.pkl`，以及 `results/metrics.csv`、`results/summary.json`。

## 4. Implementation
- AICHA 384 ROI，learned-FC 维度为 `384*383/2 = 73536`；原始 Conv1d 为 kernel 4、stride 2。
- 每个 diagnosis class（0=ASD，1=HC）使用 K=3 prototypes。
- 首次出现的 class 使用当前 batch normalized embedding 的 deterministic farthest-point initialization；第一个点最接近 centroid，后续点最大化与已选点的 cosine 距离。
- assignment 使用 epsilon=0.05、3 次 Sinkhorn 归一化；prototype 使用 momentum=0.99 EMA 更新。
- 正样本是 `P[y_i, subtype_i]`；负样本是 opposite-diagnosis 的全部 prototypes；同诊断其它 prototypes 不进正/负集合。
- `zbar = normalize((normalize(z1)+normalize(z2))/2)` 且 assignment 前 detach；训练 labels 在 removeDuplicates 后重新索引取得。
- inference 完全不使用 prototype。

## 5. Integrity
- 原项目根目录及 `Experiment/`、`dataset/`、`reproduction_results/` 均未写入。
- prototype assignment/update 只使用 training batch；没有 validation/test/Caltech/ABIDE-II labels 进入 prototype learning。
- encoder 与 classifier 均按 validation BCE 选择；threshold 仅由 selected classifier 的 validation probabilities 的 Youden J 决定并冻结。
- duplicate occurrences 只追加至 training；internal 为 10 repeats × 10 folds，Caltech 37 例仅 external test。
- baseline/treatment 对每一个 paired task 使用相同 split、batch schedule、augmentation、optimizer、epoch 和 task seed；每个 pair 在同一 GPU 上连续运行并分别 reset seed。
- 每个 worker 仅保留一个滚动 checkpoint：每 10 个 encoder epoch 原子替换 `xiaolunwen/experiment1/checkpoints/worker_*_latest.pt`，旧权重立即删除；checkpoint 不参与指标计算。

## 6. Results

| Scope | Baseline AUC | Treatment AUC | ΔAUC | Baseline BCE | Treatment BCE | ΔBCE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

10 internal repeat ΔAUC (treatment - baseline):
`[{internal_deltas}]`

10 ABIDE-II repeat ΔAUC (treatment - baseline):
`[{abide2_deltas}]`

Baseline sanity: internal OOF AUC={fmt(summary['baseline_sanity']['internal_auc'])} (reference 0.723), ABIDE-II AUC={fmt(summary['baseline_sanity']['abide2_auc'])} (reference 0.739); pipeline drift={summary['baseline_sanity']['pipeline_drift']}.

## 7. Prototype diagnostics
- Treatment prototype loss mean over reported repeat summaries: `{fmt(mean_or_nan(proto_losses), 6)}`.
- Active prototypes (out of 6): mean `{fmt(mean_or_nan(proto_active), 3)}`; mean minimum occupancy `{fmt(mean_or_nan(min_occ), 3)}`; mean maximum occupancy `{fmt(mean_or_nan(max_occ), 3)}`.
- Both diagnosis classes are required by every stratified training task; the run records finite losses/gradients and reports any inactive/collapsed bank through these columns.

## 8. Decision
**{decision}**

## 9. Interpretation
""" + (
        "结果支持“在保留 subject identity objective 的同时，增加 class-conditional subtype prototype guidance 可能增强 disease-relevant shared structure”。这不等同于发现真实 ASD 生物学亚型。"
        if decision in {"STRONG_POSITIVE", "PROMISING"}
        else "结果说明这种 online prototype disease guidance 没有稳定转化为 ASD classification / external generalization gain；本实验不进行调参救援。"
    ) + f"""

## 10. Next-step handoff
- Exact changed files：仅 `xiaolunwen/experiment1` 下列出的复制源码、`subtype_prototype.py`、`run_experiment1.py`、结果文件和本报告。
- Exact method：K=3 class-conditional online prototypes，FPS initialization，Sinkhorn epsilon=0.05/3 iterations，EMA momentum=0.99，lambda=1.0；VarCoNet identity InfoNCE 完全保留。
- Key metrics：见上表、`results/metrics.csv` 和 `results/summary.json`。
- Decision：`{decision}`。
- Code/data notes：AICHA 384 ROI；duplicate occurrences train-only；validation-only threshold；原始实现 Conv1d kernel 4/stride 2；运行时未依赖 Experiment/08 或 Experiment/09。
"""
    (EXPERIMENT_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def aggregate_main() -> None:
    shard_paths = sorted(RESULTS_DIR.glob("worker_*_repeat_*.json"))
    expected = set(range(10))
    found: dict[int, Path] = {}
    for path in shard_paths:
        try:
            repeat = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        found[repeat] = path
    if set(found) != expected:
        missing = sorted(expected - set(found))
        raise RuntimeError(f"cannot aggregate incomplete Experiment 1; missing repeats {missing}")
    rows: list[dict[str, Any]] = []
    for repeat in range(10):
        payload = json.loads(found[repeat].read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"repeat {repeat} is not PASS")
        rows.extend(payload["rows"])
    if len(rows) != 260:
        raise RuntimeError(f"expected 260 final metrics rows, got {len(rows)}")
    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})
    summary = aggregate_rows(rows)
    summary["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(RESULTS_DIR / "summary.json", summary)
    write_report(summary, rows)
    # Shards are only resume/merge state; final results retain exactly the two
    # requested files under results/.
    for path in RESULTS_DIR.glob("worker_*_repeat_*.json"):
        path.unlink()
    for path in RESULTS_DIR.glob("worker_*_status.json"):
        path.unlink()
    for path in RESULTS_DIR.glob("worker_*_repeat_*.error.json"):
        path.unlink()
    print(json.dumps({"status": "PASS", "rows": len(rows), "decision": summary["decision"]}, ensure_ascii=False))


def parse_repeats(raw: str) -> list[int]:
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if any(value < 0 or value > 9 for value in values):
        raise ValueError("repeat must be between 0 and 9")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "worker", "aggregate"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-id", default="0")
    parser.add_argument("--repeats", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke_test(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    elif args.mode == "worker":
        worker_main(parse_repeats(args.repeats), args.device, args.worker_id)
    else:
        aggregate_main()


if __name__ == "__main__":
    main()

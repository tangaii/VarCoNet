"""Fixed-state trajectory bookkeeping and split-half stability."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from disease_axis import fit_standardizer, pooled_disease_axis


FIXED_EPOCHS = (0, 1, 5, 10, 20, 30, 40, 50)
SPLIT_HALF_REPEATS = 200


def trajectory_epochs(selected_epoch: int) -> tuple[int, ...]:
    values = list(FIXED_EPOCHS)
    if int(selected_epoch) not in values:
        values.append(int(selected_epoch))
    return tuple(values)


def make_split_half_indices(
    sites: np.ndarray,
    labels: np.ndarray,
    repeats: int = SPLIT_HALF_REPEATS,
    seed: int = 20260915,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate fixed SITE×DIAGNOSIS stratified half splits."""
    sites = np.asarray(sites).astype(str); labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed)); groups = [(s, y) for s in sorted(set(sites)) for y in (0, 1)]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(int(repeats)):
        a: list[int] = []; b: list[int] = []
        for site, y in groups:
            idx = np.where((sites == site) & (labels == y))[0]
            if len(idx) < 2:
                continue
            perm = rng.permutation(idx)
            half = len(perm) // 2
            if half == 0:
                continue
            a.extend(int(v) for v in perm[:half]); b.extend(int(v) for v in perm[half:])
        if a and b:
            out.append((np.asarray(sorted(a), dtype=np.int64), np.asarray(sorted(b), dtype=np.int64)))
    if len(out) != int(repeats):
        raise RuntimeError(f"could not construct {repeats} fixed split-half indices (got {len(out)})")
    return out


def split_half_stability(
    x: torch.Tensor,
    y: np.ndarray,
    sites: np.ndarray,
    split_indices: Iterable[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    values: list[float] = []
    for ia, ib in split_indices:
        if len(np.unique(y[ia])) < 2 or len(np.unique(y[ib])) < 2:
            continue
        da = pooled_disease_axis(x[torch.as_tensor(ia)], y[ia])
        db = pooled_disease_axis(x[torch.as_tensor(ib)], y[ib])
        values.append(float(torch.dot(da, db).item()))
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "ci_low": float(np.quantile(arr, .025)), "ci_high": float(np.quantile(arr, .975)), "n": int(arr.size)}


__all__ = ["FIXED_EPOCHS", "SPLIT_HALF_REPEATS", "trajectory_epochs", "make_split_half_indices", "split_half_stability"]

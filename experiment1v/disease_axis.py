"""Leakage-controlled, site-balanced shared ASD disease axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: torch.Tensor
    std: torch.Tensor
    floor: float


@dataclass(frozen=True)
class SharedDiseaseAxis:
    standardizer: FeatureStandardizer
    axis: torch.Tensor
    eligible_sites: tuple[str, ...]


def fit_standardizer(x: torch.Tensor) -> FeatureStandardizer:
    if x.ndim != 2 or x.shape[0] < 2:
        raise RuntimeError("standardizer requires [N,D] with N>=2")
    x = x.float()
    mean = x.mean(dim=0)
    raw = x.std(dim=0, unbiased=False)
    positive = raw[raw > 0]
    if positive.numel() == 0:
        raise RuntimeError("standardizer has no positive feature variance")
    floor = max(1e-6, 0.1 * float(positive.median().item()))
    std = raw.clamp_min(float(floor))
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise FloatingPointError("non-finite standardizer")
    return FeatureStandardizer(mean.detach().cpu(), std.detach().cpu(), float(floor))


def apply_standardizer(x: torch.Tensor, state: FeatureStandardizer) -> torch.Tensor:
    z = (x.float() - state.mean.to(x.device)) / state.std.to(x.device)
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("non-finite standardized features")
    return z


def _normalize(v: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(v)
    if not bool(torch.isfinite(n)) or float(n) <= 1e-12:
        raise RuntimeError("disease direction has zero/invalid norm")
    return v / n


def fit_shared_axis(
    x_train: torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    sites_train: np.ndarray,
    min_per_class: int = 10,
) -> SharedDiseaseAxis:
    """Fit a train-only site-balanced ASD-minus-HC direction."""

    y = np.asarray(y_train, dtype=np.int64)
    sites = np.asarray(sites_train)
    if x_train.shape[0] != len(y) or len(y) != len(sites):
        raise RuntimeError("axis inputs are misaligned")
    state = fit_standardizer(x_train)
    z = apply_standardizer(x_train.cpu(), state)
    site_vectors: list[torch.Tensor] = []
    eligible: list[str] = []
    for site in sorted({str(s) for s in sites.tolist()}):
        mask_site = sites.astype(str) == site
        asd = np.where(mask_site & (y == 0))[0]
        hc = np.where(mask_site & (y == 1))[0]
        if len(asd) >= int(min_per_class) and len(hc) >= int(min_per_class):
            d = z[torch.as_tensor(asd)].mean(dim=0) - z[torch.as_tensor(hc)].mean(dim=0)
            site_vectors.append(_normalize(d))
            eligible.append(site)
    if len(site_vectors) < 4:
        raise RuntimeError(f"axis requires >=4 eligible sites, got {len(site_vectors)}")
    shared = _normalize(torch.stack(site_vectors, dim=0).mean(dim=0)).cpu()
    if not bool(torch.isfinite(shared).all()):
        raise FloatingPointError("non-finite shared disease axis")
    return SharedDiseaseAxis(state, shared, tuple(eligible))


def score_axis(x: torch.Tensor, axis: SharedDiseaseAxis) -> torch.Tensor:
    return (apply_standardizer(x, axis.standardizer) @ axis.axis.to(x.device)).detach().cpu()


def asd_auc(y: np.ndarray | torch.Tensor, scores: np.ndarray | torch.Tensor) -> float:
    labels = np.asarray(y, dtype=np.int64)
    target = (labels == 0).astype(np.int64)
    values = np.asarray(scores, dtype=float)
    if len(np.unique(target)) != 2:
        return float("nan")
    return float(roc_auc_score(target, values))


def axis_auc(x: torch.Tensor, y: np.ndarray, axis: SharedDiseaseAxis) -> float:
    return asd_auc(y, score_axis(x, axis))


def pooled_disease_axis(x: torch.Tensor, y: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Pooled (non-site-balanced) standardized disease direction."""
    labels = np.asarray(y, dtype=np.int64)
    state = fit_standardizer(x)
    z = apply_standardizer(x.cpu(), state)
    d = z[torch.as_tensor(labels == 0)].mean(dim=0) - z[torch.as_tensor(labels == 1)].mean(dim=0)
    return _normalize(d).cpu()


def strict_loso(
    x: torch.Tensor,
    y: np.ndarray,
    sites: np.ndarray,
    min_per_class: int = 10,
) -> list[dict[str, Any]]:
    """Hold out each eligible site; no held-out row enters fitting."""
    y = np.asarray(y, dtype=np.int64); sites = np.asarray(sites).astype(str)
    rows: list[dict[str, Any]] = []
    unique_sites = sorted(set(sites.tolist()))
    for held in unique_sites:
        test = sites == held
        train = ~test
        # Determine eligibility from the training partition only.
        eligible = []
        for site in sorted(set(sites[train].tolist())):
            if int(np.sum(train & (sites == site) & (y == 0))) >= min_per_class and int(np.sum(train & (sites == site) & (y == 1))) >= min_per_class:
                eligible.append(site)
        if len(eligible) < 4 or not np.any(test & (y == 0)) or not np.any(test & (y == 1)):
            continue
        axis = fit_shared_axis(x[torch.as_tensor(train)], y[train], sites[train], min_per_class)
        # Explicit leakage assertion: held-out site cannot be in the fit set.
        if held in axis.eligible_sites:
            raise RuntimeError(f"LOSO leakage: held-out site {held} entered axis")
        scores = score_axis(x[torch.as_tensor(test)], axis)
        rows.append({"site": held, "n": int(test.sum()), "n_asd": int(np.sum(test & (y == 0))),
                     "n_hc": int(np.sum(test & (y == 1))), "auc": asd_auc(y[test], scores),
                     "eligible_train_sites": list(axis.eligible_sites),
                     "scores": np.asarray(scores, dtype=np.float32), "labels": np.asarray(y[test], dtype=np.int64)})
    if not rows:
        raise RuntimeError("strict LOSO produced no eligible held-out sites")
    return rows


def cohens_d_vector(x_asd: torch.Tensor, x_hc: torch.Tensor) -> torch.Tensor:
    n1, n0 = int(x_asd.shape[0]), int(x_hc.shape[0])
    if n1 < 2 or n0 < 2:
        raise RuntimeError("Cohen d requires at least two subjects per class")
    m1, m0 = x_asd.float().mean(0), x_hc.float().mean(0)
    v1 = x_asd.float().var(0, unbiased=True); v0 = x_hc.float().var(0, unbiased=True)
    pooled = torch.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / max(n1 + n0 - 2, 1)).clamp_min(1e-8)
    out = (m1 - m0) / pooled
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError("non-finite Cohen d")
    return out


def edge_to_roi_effect(edge_effect: torch.Tensor, roi_count: int = 384) -> tuple[torch.Tensor, torch.Tensor]:
    tri = torch.triu_indices(int(roi_count), int(roi_count), offset=1, device=edge_effect.device)
    signed = torch.zeros(int(roi_count), dtype=edge_effect.dtype, device=edge_effect.device)
    absolute = torch.zeros_like(signed)
    signed.index_add_(0, tri[0], edge_effect); signed.index_add_(0, tri[1], edge_effect)
    absolute.index_add_(0, tri[0], edge_effect.abs()); absolute.index_add_(0, tri[1], edge_effect.abs())
    denom = float(roi_count - 1)
    return signed / denom, absolute / denom


def roi_convergence(train_effect: torch.Tensor, external_effect: torch.Tensor, roi_count: int = 384) -> dict[str, float]:
    ts, ta = edge_to_roi_effect(train_effect, roi_count); es, ea = edge_to_roi_effect(external_effect, roi_count)
    s = spearmanr(ts.numpy(), es.numpy()).statistic; a = spearmanr(ta.numpy(), ea.numpy()).statistic
    return {"signed_spearman": float(s) if np.isfinite(s) else float("nan"),
            "absburden_spearman": float(a) if np.isfinite(a) else float("nan")}


__all__ = ["FeatureStandardizer", "SharedDiseaseAxis", "fit_standardizer", "apply_standardizer",
           "fit_shared_axis", "score_axis", "asd_auc", "axis_auc", "pooled_disease_axis",
           "strict_loso", "cohens_d_vector", "edge_to_roi_effect", "roi_convergence"]

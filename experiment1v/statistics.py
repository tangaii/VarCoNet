"""Pre-registered bootstrap/permutation helpers for Experiment 1V."""

from __future__ import annotations

from typing import Callable, Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


BOOTSTRAP_N = 2000
PERMUTATION_N = 5000
SEED = 20260915


def ci_summary(values: np.ndarray | list[float], observed: float | None = None) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    return {"mean": float(np.mean(arr)), "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)), "n": int(arr.size),
            **({"observed": float(observed)} if observed is not None else {})}


def _target(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=np.int64) == 0).astype(np.int64)


def stratified_auc_bootstrap(y: np.ndarray, score: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=float)
    a = np.where(y == 0)[0]; h = np.where(y == 1)[0]
    if len(a) == 0 or len(h) == 0:
        return ci_summary([])
    rng = np.random.default_rng(int(seed)); vals = np.empty(int(n_boot), dtype=float)
    for k in range(int(n_boot)):
        idx = np.concatenate((rng.choice(a, len(a), replace=True), rng.choice(h, len(h), replace=True)))
        vals[k] = roc_auc_score(_target(y[idx]), score[idx])
    return ci_summary(vals, roc_auc_score(_target(y), score))


def paired_stratified_auc_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, float]:
    """Paired subject bootstrap; ``delta = AUC_b - AUC_a``."""
    y = np.asarray(y, dtype=np.int64); a_score = np.asarray(score_a, dtype=float); b_score = np.asarray(score_b, dtype=float)
    if not (len(y) == len(a_score) == len(b_score)):
        raise RuntimeError("paired AUC inputs are misaligned")
    ia, ih = np.where(y == 0)[0], np.where(y == 1)[0]
    rng = np.random.default_rng(int(seed)); vals = []
    for _ in range(int(n_boot)):
        idx = np.concatenate((rng.choice(ia, len(ia), replace=True), rng.choice(ih, len(ih), replace=True)))
        yy = _target(y[idx])
        vals.append(roc_auc_score(yy, b_score[idx]) - roc_auc_score(yy, a_score[idx]))
    observed = roc_auc_score(_target(y), b_score) - roc_auc_score(_target(y), a_score)
    return ci_summary(np.asarray(vals), observed)


def geometry_gap(similarity: np.ndarray, y: np.ndarray, sites: np.ndarray) -> float:
    """Same-diagnosis/different-site minus different-diagnosis/different-site."""
    sim = np.asarray(similarity, dtype=float); y = np.asarray(y); sites = np.asarray(sites).astype(str)
    if sim.shape != (len(y), len(y)):
        raise RuntimeError("similarity matrix and metadata dimensions differ")
    ii, jj = np.triu_indices(len(y), k=1)
    cross = sites[ii] != sites[jj]
    same = cross & (y[ii] == y[jj]); diff = cross & (y[ii] != y[jj])
    if not np.any(same) or not np.any(diff):
        return float("nan")
    # Symmetric matrices are expected; use the upper triangle exactly once.
    return float(np.mean(sim[ii[same], jj[same]]) - np.mean(sim[ii[diff], jj[diff]]))


def paired_geometry_bootstrap(
    sim_a: np.ndarray,
    sim_b: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, float]:
    """Stratified subject bootstrap for paired geometry gaps."""
    sim_a = np.asarray(sim_a, dtype=float); sim_b = np.asarray(sim_b, dtype=float)
    y = np.asarray(y, dtype=np.int64); sites = np.asarray(sites)
    ia, ih = np.where(y == 0)[0], np.where(y == 1)[0]
    rng = np.random.default_rng(int(seed)); deltas: list[float] = []
    for _ in range(int(n_boot)):
        idx = np.concatenate((rng.choice(ia, len(ia), replace=True), rng.choice(ih, len(ih), replace=True)))
        # Repeated bootstrap draws are separate units; only within-position
        # diagonals are excluded by geometry_gap's k=1 convention.
        ga = geometry_gap(sim_a[np.ix_(idx, idx)], y[idx], sites[idx])
        gb = geometry_gap(sim_b[np.ix_(idx, idx)], y[idx], sites[idx])
        if np.isfinite(ga) and np.isfinite(gb):
            deltas.append(float(gb - ga))
    observed = geometry_gap(sim_b, y, sites) - geometry_gap(sim_a, y, sites)
    return ci_summary(np.asarray(deltas), observed)


def site_macro_bootstrap(site_aucs: list[float] | np.ndarray, n_boot: int = 5000, seed: int = SEED) -> dict[str, float]:
    vals = np.asarray(site_aucs, dtype=float); vals = vals[np.isfinite(vals)]
    if not vals.size:
        return ci_summary([])
    rng = np.random.default_rng(int(seed)); boot = np.asarray([np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(int(n_boot))])
    return ci_summary(boot, float(np.mean(vals)))


def auc_label_permutation(y: np.ndarray, score: np.ndarray, n_perm: int = PERMUTATION_N, seed: int = SEED) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=float); target = _target(y)
    observed = roc_auc_score(target, score); rng = np.random.default_rng(int(seed)); exceed = 0
    null = np.empty(int(n_perm), dtype=float)
    for k in range(int(n_perm)):
        perm = rng.permutation(target); null[k] = roc_auc_score(perm, score)
        if null[k] >= observed:
            exceed += 1
    return {"observed": float(observed), "p": float((exceed + 1) / (int(n_perm) + 1)),
            "null_q95": float(np.quantile(null, 0.95)), "n": int(n_perm)}


def edge_replication(
    train_effect: np.ndarray,
    external_effect: np.ndarray,
    fraction: float = 0.01,
    y_external: np.ndarray | None = None,
    external_features: np.ndarray | None = None,
    n_perm: int = PERMUTATION_N,
    seed: int = SEED,
) -> dict[str, float | int]:
    """Fixed train-selected top-edge replication and diagnosis permutation null."""
    train = np.asarray(train_effect, dtype=float); ext = np.asarray(external_effect, dtype=float)
    if train.shape != ext.shape:
        raise RuntimeError("edge effects are not aligned")
    k = max(1, int(round(len(train) * float(fraction))))
    top = np.argsort(-np.abs(train), kind="mergesort")[:k]
    signs = np.sign(train[top]) == np.sign(ext[top])
    sign_obs = float(np.mean(signs))
    rho = spearmanr(train[top], ext[top]).statistic
    rho = float(rho) if np.isfinite(rho) else float("nan")
    denom = np.linalg.norm(train[top]) * np.linalg.norm(ext[top])
    cosine = float(np.dot(train[top], ext[top]) / denom) if denom > 0 else float("nan")
    result: dict[str, float | int] = {"k": int(k), "sign_agreement": sign_obs, "spearman": rho, "cosine": cosine}
    if y_external is None or external_features is None:
        result.update({"sign_perm_p": float("nan"), "spearman_perm_p": float("nan"), "sign_null_q95": float("nan"), "spearman_null_q95": float("nan")})
        return result
    y_external = np.asarray(y_external, dtype=np.int64); features = np.asarray(external_features, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(ext):
        raise RuntimeError("external edge feature shape mismatch")
    ia, ih = np.where(y_external == 0)[0], np.where(y_external == 1)[0]
    if len(ia) < 2 or len(ih) < 2:
        return result
    # Only the preregistered top edges enter the null.  Keeping this matrix at
    # N×K (rather than recomputing all 73,536 effects for every permutation)
    # is mathematically identical for the reported sign/Spearman statistics.
    train_top = train[top]
    top_features = features[:, top]
    rng = np.random.default_rng(int(seed)); null_sign = np.empty(int(n_perm)); null_rho = np.empty(int(n_perm))
    for p in range(int(n_perm)):
        yp = rng.permutation(y_external)
        xa = top_features[yp == 0]; xh = top_features[yp == 1]
        n_a, n_h = len(xa), len(xh)
        va = xa.var(0, ddof=1); vh = xh.var(0, ddof=1)
        pooled = np.sqrt(((n_a - 1) * va + (n_h - 1) * vh) / max(n_a + n_h - 2, 1)).clip(min=1e-8)
        ep = (xa.mean(0) - xh.mean(0)) / pooled
        ss = np.mean(np.sign(train_top) == np.sign(ep)); rr = spearmanr(train_top, ep).statistic
        null_sign[p] = ss; null_rho[p] = rr if np.isfinite(rr) else 0.0
    result.update({"sign_perm_p": float((np.sum(null_sign >= sign_obs) + 1) / (int(n_perm) + 1)),
                   "spearman_perm_p": float((np.sum(null_rho >= rho) + 1) / (int(n_perm) + 1)) if np.isfinite(rho) else float("nan"),
                   "sign_null_q95": float(np.quantile(null_sign, 0.95)), "spearman_null_q95": float(np.quantile(null_rho, 0.95))})
    return result


def benjamini_hochberg(pvalues: dict[str, float]) -> dict[str, float]:
    items = [(k, float(v)) for k, v in pvalues.items() if np.isfinite(v)]
    if not items:
        return {}
    order = sorted(range(len(items)), key=lambda i: items[i][1]); m = len(items); out = {}
    running = 1.0
    for rank in range(m, 0, -1):
        key, p = items[order[rank - 1]]; running = min(running, p * m / rank); out[key] = float(min(running, 1.0))
    return out


def reproducibility_check(fn: Callable[[], Any]) -> bool:
    first = fn(); second = fn()
    if isinstance(first, np.ndarray):
        return bool(np.array_equal(first, second))
    if isinstance(first, dict):
        return all(reproducibility_check(lambda k=k: first[k]) if False else first[k] == second[k] for k in first)
    return bool(first == second)


__all__ = ["BOOTSTRAP_N", "PERMUTATION_N", "SEED", "ci_summary", "stratified_auc_bootstrap",
           "paired_stratified_auc_bootstrap", "geometry_gap", "paired_geometry_bootstrap",
           "site_macro_bootstrap", "auc_label_permutation", "edge_replication", "benjamini_hochberg"]

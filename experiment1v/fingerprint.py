"""Connectome fingerprinting metrics (Finn et al.-style retrieval)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def vector_pattern_similarity(query: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """Row-wise centered/L2-normalized Pearson-pattern similarity."""

    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise RuntimeError("fingerprint matrices must be [N,D] with equal D")
    q = F.normalize(query.float() - query.float().mean(dim=1, keepdim=True), p=2, dim=1)
    g = F.normalize(gallery.float() - gallery.float().mean(dim=1, keepdim=True), p=2, dim=1)
    out = q @ g.T
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError("non-finite fingerprint similarity")
    return out


def _retrieval_direction(sim: np.ndarray, candidate_masks: list[np.ndarray] | None = None) -> dict[str, Any]:
    n = int(sim.shape[0])
    top1: list[float] = []
    top5: list[float] = []
    mrr: list[float] = []
    ranks: list[int] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool) if candidate_masks is None else np.asarray(candidate_masks[i], dtype=bool).copy()
        if not bool(mask.any()):
            continue
        values = np.asarray(sim[i], dtype=float).copy()
        values[~mask] = -np.inf
        order = np.argsort(-values, kind="mergesort")
        order = order[np.isfinite(values[order])]
        if order.size == 0:
            continue
        where = np.where(order == i)[0]
        if where.size == 0:
            # A query not present in its gallery has no defined same-subject rank.
            continue
        rank = int(where[0]) + 1
        ranks.append(rank)
        top1.append(float(rank == 1))
        top5.append(float(rank <= 5))
        mrr.append(1.0 / float(rank))
    return {
        "n": len(top1),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "top5": float(np.mean(top5)) if top5 else float("nan"),
        "mrr": float(np.mean(mrr)) if mrr else float("nan"),
        "top1_per_subject": np.asarray(top1, dtype=float),
        "top5_per_subject": np.asarray(top5, dtype=float),
        "mrr_per_subject": np.asarray(mrr, dtype=float),
        "ranks": np.asarray(ranks, dtype=np.int64),
    }


def _same_different(sim: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    n = int(sim.shape[0])
    diag = np.diag(sim).astype(float)
    if n <= 1:
        return float(np.mean(diag)), float("nan"), float("nan"), diag, np.empty(0, dtype=float)
    off = sim[~np.eye(n, dtype=bool)].astype(float)
    same = float(np.mean(diag))
    different = float(np.mean(off))
    return same, different, same - different, diag, off


def _sensitivity_masks(ids: np.ndarray, sites: np.ndarray | None, labels: np.ndarray | None, same_dx: bool) -> list[np.ndarray] | None:
    if sites is None:
        return None
    sites = np.asarray(sites)
    labels = None if labels is None else np.asarray(labels)
    masks: list[np.ndarray] = []
    for i in range(len(sites)):
        mask = sites == sites[i]
        if same_dx and labels is not None:
            mask &= labels == labels[i]
        mask[i] = False
        # The query itself is not in a within-site gallery; retrieval rank is
        # therefore computed against row IDs explicitly below.
        masks.append(mask)
    return masks


def _retrieval_with_explicit_ids(sim: np.ndarray, masks: list[np.ndarray], ids: np.ndarray) -> dict[str, Any]:
    """Retrieval where query and gallery row IDs are aligned but self is masked."""
    n = len(ids)
    vals1: list[float] = []
    vals5: list[float] = []
    valsm: list[float] = []
    ranks: list[int] = []
    for i in range(n):
        mask = np.asarray(masks[i], dtype=bool)
        if int(mask.sum()) < 5:
            continue
        order = np.argsort(-np.where(mask, sim[i], -np.inf), kind="mergesort")
        order = order[mask[order]]
        # For two views of the same row, the row identity is i.  If the row is
        # excluded (as it is here), no same-subject candidate exists; instead
        # sensitivity is defined as whether the top candidate is the matching
        # subject ID in a second aligned view.  Callers pass a diagonal-preserving
        # matrix, so retain i when available.
        if i not in order:
            # Recompute with self included solely for the target rank, while
            # requiring at least five non-self candidates.
            full_mask = mask.copy(); full_mask[i] = True
            order = np.argsort(-np.where(full_mask, sim[i], -np.inf), kind="mergesort")
            order = order[full_mask[order]]
        if i not in order:
            continue
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank); vals1.append(float(rank == 1)); vals5.append(float(rank <= 5)); valsm.append(1.0 / rank)
    return {"n": len(vals1), "top1": float(np.mean(vals1)) if vals1 else float("nan"),
            "top5": float(np.mean(vals5)) if vals5 else float("nan"), "mrr": float(np.mean(valsm)) if valsm else float("nan"),
            "top1_per_subject": np.asarray(vals1), "top5_per_subject": np.asarray(vals5),
            "mrr_per_subject": np.asarray(valsm), "ranks": np.asarray(ranks, dtype=np.int64)}


def fingerprint_metrics(
    view_a: torch.Tensor | np.ndarray,
    view_b: torch.Tensor | np.ndarray,
    subject_ids: np.ndarray,
    sites: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    min_candidates: int = 5,
) -> dict[str, Any]:
    """Compute bidirectional global and site-controlled retrieval metrics."""

    a = torch.as_tensor(view_a, dtype=torch.float32)
    b = torch.as_tensor(view_b, dtype=torch.float32)
    ids = np.asarray(subject_ids)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or len(ids) != a.shape[0]:
        raise RuntimeError("paired fingerprint views/IDs are misaligned")
    sim_ab = vector_pattern_similarity(a, b).cpu().numpy()
    sim_ba = vector_pattern_similarity(b, a).cpu().numpy()
    global_ab = _retrieval_direction(sim_ab)
    global_ba = _retrieval_direction(sim_ba)
    same_ab, diff_ab, gap_ab, diag_ab, off_ab = _same_different(sim_ab)
    same_ba, diff_ba, gap_ba, diag_ba, off_ba = _same_different(sim_ba)
    out: dict[str, Any] = {
        "n": int(len(ids)),
        "top1": float(np.nanmean([global_ab["top1"], global_ba["top1"]])),
        "top5": float(np.nanmean([global_ab["top5"], global_ba["top5"]])),
        "mrr": float(np.nanmean([global_ab["mrr"], global_ba["mrr"]])),
        "same_similarity": float(np.nanmean([same_ab, same_ba])),
        "different_similarity": float(np.nanmean([diff_ab, diff_ba])),
        "identity_gap": float(np.nanmean([gap_ab, gap_ba])),
        "top1_per_subject": (global_ab["top1_per_subject"] + global_ba["top1_per_subject"]) / 2.0,
        "top5_per_subject": (global_ab["top5_per_subject"] + global_ba["top5_per_subject"]) / 2.0,
        "mrr_per_subject": (global_ab["mrr_per_subject"] + global_ba["mrr_per_subject"]) / 2.0,
        "similarity_ab": sim_ab,
        "similarity_ba": sim_ba,
    }
    if sites is None:
        out.update({"within_site_n": 0, "within_site_top1": float("nan"), "within_site_same_dx_top1": float("nan")})
    else:
        sites = np.asarray(sites)
        if len(sites) != len(ids):
            raise RuntimeError("site IDs are not aligned with fingerprint IDs")
        base_masks = _sensitivity_masks(ids, sites, labels, False)
        dx_masks = _sensitivity_masks(ids, sites, labels, True)
        # Use the diagonal-preserving matrix and explicit candidate count.  The
        # self row is temporarily included only to define the same-subject rank.
        def sens(sim: np.ndarray, masks: list[np.ndarray]) -> dict[str, Any]:
            nvals: list[float] = []
            for i, mask in enumerate(masks):
                if int(mask.sum()) < int(min_candidates):
                    continue
                full = mask.copy(); full[i] = True
                order = np.argsort(-np.where(full, sim[i], -np.inf), kind="mergesort")
                order = order[full[order]]
                rank = int(np.where(order == i)[0][0]) + 1
                nvals.append(float(rank == 1))
            return {"n": len(nvals), "top1": float(np.mean(nvals)) if nvals else float("nan"), "per": np.asarray(nvals)}
        s_ab, s_ba = sens(sim_ab, base_masks), sens(sim_ba, base_masks)
        d_ab, d_ba = sens(sim_ab, dx_masks), sens(sim_ba, dx_masks)
        out.update({"within_site_n": int(round(np.nanmean([s_ab["n"], s_ba["n"]]))),
                    "within_site_top1": float(np.nanmean([s_ab["top1"], s_ba["top1"]])),
                    "within_site_same_dx_n": int(round(np.nanmean([d_ab["n"], d_ba["n"]]))),
                    "within_site_same_dx_top1": float(np.nanmean([d_ab["top1"], d_ba["top1"]]))})
    return out


__all__ = ["vector_pattern_similarity", "fingerprint_metrics"]

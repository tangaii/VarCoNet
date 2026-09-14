"""Label-free consensus modular readout for Experiment 3.

The encoder and its pairwise classifier are deliberately not implemented in
this module.  This file contains only the post-freeze, train-only structural
readout used by CHMR-VarCoNet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sklearn.cluster import SpectralClustering


@dataclass
class ModuleDiagnostics:
    n_modules: int
    module_size_min: int
    module_size_mean: float
    module_size_max: int
    singleton_modules: int
    affinity_mean: float
    affinity_std: float
    affinity_max: float
    meta_dim: int
    empty_meta_bins: int

    def as_dict(self) -> dict[str, float]:
        return {
            "n_modules": float(self.n_modules),
            "module_size_min": float(self.module_size_min),
            "module_size_mean": float(self.module_size_mean),
            "module_size_max": float(self.module_size_max),
            "singleton_modules": float(self.singleton_modules),
            "affinity_mean": float(self.affinity_mean),
            "affinity_std": float(self.affinity_std),
            "affinity_max": float(self.affinity_max),
            "meta_dim": float(self.meta_dim),
            "empty_meta_bins": float(self.empty_meta_bins),
        }


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Give spectral labels a deterministic ID ordered by minimum ROI index."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise RuntimeError("module labels must be one-dimensional")
    modules: list[tuple[int, int]] = []
    for old in np.unique(labels):
        members = np.where(labels == old)[0]
        if len(members) == 0:
            raise RuntimeError("empty spectral module")
        modules.append((int(members.min()), int(old)))
    modules.sort()
    mapping = {old: new for new, (_, old) in enumerate(modules)}
    output = np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)
    return output


@torch.no_grad()
def discover_consensus_modules(
    stable_unique_train: torch.Tensor,
    roi_num: int = 384,
    n_modules: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Discover modules from *only* unique outer-training stable FC vectors.

    No labels or metadata are accepted by this API.  The affinity is exactly
    ``mean_subject(abs(S_subject,e))`` and is passed to the prescribed
    ``SpectralClustering(cluster_qr, arpack)`` implementation.
    """
    if stable_unique_train.ndim != 2:
        raise RuntimeError("stable_unique_train must be [N,E]")
    expected_edges = roi_num * (roi_num - 1) // 2
    if stable_unique_train.shape[1] != expected_edges:
        raise RuntimeError(
            f"edge dim mismatch: {stable_unique_train.shape[1]} vs {expected_edges}"
        )
    if stable_unique_train.shape[0] < n_modules:
        raise RuntimeError("not enough unique training subjects for 32 modules")

    mean_abs_edge = (
        stable_unique_train.detach().abs().mean(dim=0).cpu().numpy().astype(np.float64)
    )
    if not np.isfinite(mean_abs_edge).all():
        raise FloatingPointError("non-finite consensus edge")

    tri_i, tri_j = np.triu_indices(roi_num, k=1)
    affinity = np.zeros((roi_num, roi_num), dtype=np.float64)
    affinity[tri_i, tri_j] = mean_abs_edge
    affinity[tri_j, tri_i] = mean_abs_edge
    np.fill_diagonal(affinity, 0.0)
    if not np.isfinite(affinity).all():
        raise FloatingPointError("non-finite consensus affinity")
    if not np.allclose(affinity, affinity.T, atol=1e-12, rtol=0.0):
        raise RuntimeError("consensus affinity not symmetric")
    if np.any(affinity < 0.0):
        raise RuntimeError("negative consensus affinity")
    if not np.all(np.diag(affinity) == 0.0):
        raise RuntimeError("consensus affinity diagonal is not exactly zero")
    positive = affinity[affinity > 0.0]
    if len(positive) == 0:
        raise RuntimeError("zero consensus affinity")
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("invalid affinity scale")

    # Do not replace cluster_qr if unavailable: the prescribed algorithm is
    # part of the experiment definition and a silent fallback would invalidate
    # comparability.
    clusterer = SpectralClustering(
        n_clusters=int(n_modules),
        affinity="precomputed",
        assign_labels="cluster_qr",
        eigen_solver="arpack",
        random_state=int(seed),
        n_jobs=1,
    )
    labels = canonicalize_labels(clusterer.fit_predict(affinity / scale))
    unique = np.unique(labels)
    if len(unique) != n_modules or not np.array_equal(unique, np.arange(n_modules)):
        raise RuntimeError(f"expected {n_modules} canonical modules, got {unique.tolist()}")
    sizes = np.asarray([np.sum(labels == k) for k in range(n_modules)], dtype=np.int64)
    diagnostics = ModuleDiagnostics(
        n_modules=int(n_modules),
        module_size_min=int(sizes.min()),
        module_size_mean=float(sizes.mean()),
        module_size_max=int(sizes.max()),
        singleton_modules=int(np.sum(sizes == 1)),
        affinity_mean=float(mean_abs_edge.mean()),
        affinity_std=float(mean_abs_edge.std()),
        affinity_max=float(mean_abs_edge.max()),
        meta_dim=int(n_modules * (n_modules + 1) // 2),
        empty_meta_bins=0,
    )
    return labels, diagnostics.as_dict()


class ModulePairPooler(nn.Module):
    """Pool every upper-triangular edge into one unordered module pair."""

    def __init__(
        self,
        module_labels: np.ndarray,
        roi_num: int = 384,
        n_modules: int = 32,
    ) -> None:
        super().__init__()
        labels = np.asarray(module_labels, dtype=np.int64)
        if labels.shape != (roi_num,):
            raise RuntimeError(f"module label shape mismatch: {labels.shape}")
        unique = np.unique(labels)
        if len(unique) != n_modules or not np.array_equal(unique, np.arange(n_modules)):
            raise RuntimeError("module count/canonical labels mismatch")

        tri_i, tri_j = np.triu_indices(roi_num, k=1)
        pair_lookup = np.full((n_modules, n_modules), -1, dtype=np.int64)
        index = 0
        for a in range(n_modules):
            for b in range(a, n_modules):
                pair_lookup[a, b] = index
                pair_lookup[b, a] = index
                index += 1
        meta_dim = n_modules * (n_modules + 1) // 2
        if index != meta_dim:
            raise RuntimeError("meta dimension construction error")

        edge_bins = pair_lookup[labels[tri_i], labels[tri_j]]
        if np.any(edge_bins < 0):
            raise RuntimeError("unassigned original edge")
        counts = np.bincount(edge_bins, minlength=meta_dim).astype(np.float32)
        expected_edges = roi_num * (roi_num - 1) // 2
        if int(counts.sum()) != expected_edges or len(edge_bins) != expected_edges:
            raise RuntimeError("not every original edge was assigned exactly once")

        self.roi_num = int(roi_num)
        self.n_modules = int(n_modules)
        self.meta_dim = int(meta_dim)
        self.register_buffer("edge_bins", torch.as_tensor(edge_bins, dtype=torch.long))
        self.register_buffer("bin_counts", torch.as_tensor(counts, dtype=torch.float32))
        self.register_buffer("module_labels", torch.as_tensor(labels, dtype=torch.long))
        self.empty_meta_bins = int(np.sum(counts == 0.0))

    def forward(self, stable: torch.Tensor) -> torch.Tensor:
        expected_edges = self.roi_num * (self.roi_num - 1) // 2
        if stable.ndim != 2 or stable.shape[1] != expected_edges:
            raise RuntimeError("stable must be [B,73536]")
        batch = stable.shape[0]
        output = torch.zeros(
            batch, self.meta_dim, device=stable.device, dtype=stable.dtype
        )
        bins = self.edge_bins.to(stable.device)
        output.scatter_add_(
            1, bins.unsqueeze(0).expand(batch, -1), stable
        )
        counts = self.bin_counts.to(stable.device, dtype=stable.dtype)
        output = output / counts.clamp_min(1.0).unsqueeze(0)
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite meta-connectome")
        return output


class FrozenStableModuleResidualClassifier(nn.Module):
    """Frozen original classifier plus a single zero-initialized residual."""

    def __init__(self, baseline_state: dict[str, torch.Tensor], meta_dim: int) -> None:
        super().__init__()
        from model_scripts.classifier import MLP

        stable_dim = int(baseline_state["fc.0.weight"].shape[1])
        self.stable_classifier = MLP(stable_dim, 2)
        self.stable_classifier.load_state_dict(baseline_state)
        for parameter in self.stable_classifier.parameters():
            parameter.requires_grad_(False)
        self.residual_head = nn.Linear(int(meta_dim), 1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    def forward(self, stable: torch.Tensor, meta_z: torch.Tensor) -> torch.Tensor:
        stable_probs = self.stable_classifier(stable)
        stable_logits = torch.log(stable_probs.clamp_min(1e-7))
        delta = self.residual_head(meta_z)
        residual_logits = torch.cat([-0.5 * delta, +0.5 * delta], dim=1)
        return torch.softmax(stable_logits + residual_logits, dim=1)

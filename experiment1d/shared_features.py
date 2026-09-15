"""Shared, leakage-audited feature utilities for Experiment 1D.

The module intentionally contains no experiment-specific training logic.  It
provides the single ROI profile extraction used by DSLS, CSDA and ICDM, the
frozen baseline margin helper, and strict ABIDE-I site metadata discovery.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProfileState:
    """Reference-only profile normalization state."""

    mu: torch.Tensor
    sigma: torch.Tensor
    std_floor: float


@dataclass(frozen=True)
class SiteMetadata:
    """ABIDE-I raw-occurrence site mapping.

    ``raw_sites`` is aligned with the rows in the processed ABIDE-I archive;
    ``site_ids`` is an integer encoding used only by the CSDA training loss.
    """

    source_path: str
    raw_sites: tuple[str, ...]
    site_ids: np.ndarray
    site_names: tuple[str, ...]


class ROIConnectivityProfile(nn.Module):
    """Signed mean connectivity incident on each ROI.

    For an upper-triangular stable FC vector ``S``,
    ``G_r = mean_{j != r} S_(r,j)``.  Edge indices are constructed once and
    reused for every branch and every scope.
    """

    def __init__(self, roi_count: int = 384, roi_num: int | None = None):
        super().__init__()
        if roi_num is not None:
            roi_count = int(roi_num)
        self.roi_count = int(roi_count)
        if self.roi_count < 2:
            raise ValueError("ROIConnectivityProfile requires at least two ROIs")
        tri = torch.triu_indices(self.roi_count, self.roi_count, offset=1)
        self.register_buffer("edge_i", tri[0], persistent=False)
        self.register_buffer("edge_j", tri[1], persistent=False)

    @property
    def roi_num(self) -> int:
        return self.roi_count

    @property
    def edge_dim(self) -> int:
        return self.roi_count * (self.roi_count - 1) // 2

    def forward(self, stable: torch.Tensor) -> torch.Tensor:
        if stable.ndim != 2 or stable.shape[1] != self.edge_dim:
            raise RuntimeError(
                f"ROI profile expected [B,{self.edge_dim}], got {tuple(stable.shape)}"
            )
        profile = stable.new_zeros((stable.shape[0], self.roi_count))
        profile.index_add_(1, self.edge_i, stable)
        profile.index_add_(1, self.edge_j, stable)
        return profile / float(self.roi_count - 1)


def fit_profile_state(profile_reference: torch.Tensor) -> ProfileState:
    """Fit profile normalization on unique outer-training subjects only."""

    if profile_reference.ndim != 2 or profile_reference.shape[0] < 2:
        raise RuntimeError("profile reference must contain at least two subjects")
    mu = profile_reference.mean(dim=0)
    raw_std = profile_reference.std(dim=0, unbiased=False)
    positive = raw_std[raw_std > 0]
    if positive.numel() == 0:
        raise RuntimeError("profile reference has no positive standard deviation")
    floor = max(1e-6, 0.1 * float(positive.median().detach().cpu()))
    sigma = raw_std.clamp_min(float(floor))
    if not bool(torch.isfinite(mu).all() and torch.isfinite(sigma).all()):
        raise FloatingPointError("non-finite profile normalization")
    return ProfileState(mu.detach().cpu(), sigma.detach().cpu(), float(floor))


def profile_state_payload(state: ProfileState) -> dict[str, object]:
    return {
        "mu": state.mu.detach().cpu().clone(),
        "sigma": state.sigma.detach().cpu().clone(),
        "std_floor": float(state.std_floor),
    }


def profile_state_from_payload(payload: dict[str, object]) -> ProfileState:
    return ProfileState(
        torch.as_tensor(payload["mu"]).detach().cpu().clone(),
        torch.as_tensor(payload["sigma"]).detach().cpu().clone(),
        float(payload["std_floor"]),
    )


def margin_from_baseline_state(stable: torch.Tensor, baseline_state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the frozen baseline ASD-minus-HC logit margin."""

    logits = F.linear(
        stable,
        baseline_state["fc.0.weight"].to(stable.device),
        baseline_state["fc.0.bias"].to(stable.device),
    )
    return logits[:, 0] - logits[:, 1]


def normalized_profile(profile: torch.Tensor, state: ProfileState) -> torch.Tensor:
    return (profile - state.mu.to(profile.device)) / state.sigma.to(profile.device)


def _parse_subject_id(value: object) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return int(float(text))
    return None


def _subject_id_from_name(name: str) -> int | None:
    digits = "".join(re.findall(r"\d+", str(name)))
    return int(digits) if digits else None


def discover_abide_phenotype_csv(dataset_dir: Path) -> Path:
    """Find the preferred official ABIDE-I phenotype CSV without guessing."""

    preferred = sorted(dataset_dir.rglob("Phenotypic_V1_0b_preprocessed1.csv"))
    if preferred:
        return preferred[0]
    candidates = sorted(dataset_dir.rglob("*.csv"))
    usable: list[Path] = []
    for path in candidates:
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                fields = set(next(csv.reader(handle)))
        except (OSError, StopIteration, UnicodeError):
            continue
        if {"SUB_ID", "SITE_ID"}.issubset(fields):
            usable.append(path)
    if len(usable) == 1:
        return usable[0]
    if not usable:
        raise RuntimeError("SITE_METADATA_FAIL: no ABIDE phenotype CSV with SUB_ID and SITE_ID")
    raise RuntimeError(f"SITE_METADATA_FAIL: ambiguous phenotype CSV candidates: {[str(p) for p in usable]}")


def build_site_metadata(names: Iterable[str], dataset_dir: Path) -> SiteMetadata:
    """Map every processed ABIDE-I raw occurrence to its official SITE_ID."""

    name_list = [str(name).strip() for name in names]
    source = discover_abide_phenotype_csv(dataset_dir)
    records: dict[int, str] = {}
    conflicts: dict[int, set[str]] = {}
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"SUB_ID", "SITE_ID"}.issubset(fields):
            raise RuntimeError(f"SITE_METADATA_FAIL: missing SUB_ID/SITE_ID in {source}")
        for row in reader:
            sid = _parse_subject_id(row.get("SUB_ID", ""))
            site = str(row.get("SITE_ID", "")).strip().upper()
            if sid is None or not site:
                continue
            if sid in records and records[sid] != site:
                conflicts.setdefault(sid, {records[sid]}).add(site)
            records[sid] = site
    if conflicts:
        raise RuntimeError(f"SITE_METADATA_FAIL: conflicting sites for subject IDs {conflicts}")

    missing: list[tuple[str, int | None]] = []
    raw_sites: list[str] = []
    for name in name_list:
        sid = _subject_id_from_name(name)
        site = records.get(sid) if sid is not None else None
        if not site:
            missing.append((name, sid))
            raw_sites.append("")
        else:
            raw_sites.append(site)
    if missing:
        raise RuntimeError(f"SITE_METADATA_FAIL: missing site mapping for {missing[:30]}")

    by_name: dict[str, str] = {}
    for name, site in zip(name_list, raw_sites):
        old = by_name.setdefault(name, site)
        if old != site:
            raise RuntimeError(f"SITE_METADATA_FAIL: duplicate scan site mismatch for {name}: {old} vs {site}")
    site_names = tuple(sorted(set(raw_sites)))
    encoding = {site: index for index, site in enumerate(site_names)}
    site_ids = np.asarray([encoding[site] for site in raw_sites], dtype=np.int64)
    return SiteMetadata(str(source), tuple(raw_sites), site_ids, site_names)


__all__ = [
    "ProfileState",
    "SiteMetadata",
    "ROIConnectivityProfile",
    "fit_profile_state",
    "profile_state_payload",
    "profile_state_from_payload",
    "margin_from_baseline_state",
    "normalized_profile",
    "discover_abide_phenotype_csv",
    "build_site_metadata",
]

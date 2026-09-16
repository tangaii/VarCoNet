"""Pearson and VarCoNet representation extraction for Experiment 1V."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from stable_windows import StableWindow, build_stable_windows, valid_length


ROI_COUNT = 384
EDGE_DIM = ROI_COUNT * (ROI_COUNT - 1) // 2
MAX_LENGTH = 320
VIEW_LENGTH = 80


def pearson_fc(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the upper-triangular Pearson FC for real ``[T,R]`` data."""

    if x.ndim != 2:
        raise RuntimeError("Pearson input must be [T,R]")
    if x.shape[0] < 2:
        raise RuntimeError("Pearson requires at least two real time points")
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    norm = torch.sqrt(torch.sum(x * x, dim=0)).clamp_min(float(eps))
    corr = (x.T @ x) / (norm[:, None] * norm[None, :])
    corr = torch.clamp(corr, -1.0, 1.0)
    tri = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1, device=corr.device)
    fc = corr[tri[0], tri[1]]
    if fc.numel() != EDGE_DIM:
        raise RuntimeError(f"Pearson FC dimension drifted: {fc.numel()} != {EDGE_DIM}")
    if not bool(torch.isfinite(fc).all()):
        raise FloatingPointError("non-finite Pearson FC")
    return fc


def _raw_fc_array(raw: torch.Tensor) -> torch.Tensor:
    return pearson_fc(raw).detach().cpu().float()


def pearson_full(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Pearson FC over all contiguous valid points in one padded scan."""

    tensor = torch.as_tensor(data, dtype=torch.float32)
    length = valid_length(tensor)
    return _raw_fc_array(tensor[:length])


def pearson_windowavg(
    data: np.ndarray | torch.Tensor,
    windows: Iterable[StableWindow] | None = None,
) -> torch.Tensor:
    """Average Pearson FCs over the exact stable-window records."""

    records = list(windows) if windows is not None else build_stable_windows(data)
    if not records:
        raise RuntimeError("no stable windows")
    vectors = [_raw_fc_array(record.raw_valid_window) for record in records]
    out = torch.stack(vectors, dim=0).mean(dim=0)
    if out.numel() != EDGE_DIM or not bool(torch.isfinite(out).all()):
        raise FloatingPointError("invalid Pearson window-average FC")
    return out


def _encode_padded(
    encoder: torch.nn.Module,
    padded: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, int(padded.shape[0]), int(batch_size)):
            batch = padded[start : start + int(batch_size)].to(device)
            outputs.append(encoder(batch).detach().cpu().float())
    if not outputs:
        raise RuntimeError("empty VarCoNet batch")
    out = torch.cat(outputs, dim=0)
    if out.ndim != 2 or out.shape[1] != EDGE_DIM or not bool(torch.isfinite(out).all()):
        raise RuntimeError(f"invalid VarCoNet FC shape: {tuple(out.shape)}")
    return out


def varconet_windowavg(
    encoder: torch.nn.Module,
    data: np.ndarray | torch.Tensor,
    windows: Iterable[StableWindow] | None,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode padded stable windows and average learned FC vectors."""

    records = list(windows) if windows is not None else build_stable_windows(data)
    if not records:
        raise RuntimeError("no stable windows")
    padded = torch.stack([record.padded_window for record in records], dim=0)
    return _encode_padded(encoder, padded, device, batch_size).mean(dim=0)


def varconet_full(
    encoder: torch.nn.Module,
    data: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compatibility alias for the one-window deployed stable extraction."""

    return varconet_windowavg(encoder, data, build_stable_windows(data), device, batch_size)


def view_data(data: np.ndarray | torch.Tensor, view_length: int = VIEW_LENGTH) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-overlapping first/last real views, without random cropping."""

    tensor = torch.as_tensor(data, dtype=torch.float32)
    length = valid_length(tensor)
    if length < 2 * int(view_length):
        raise ValueError(f"scan has {length} valid points; two {view_length}-point views unavailable")
    a = tensor[: int(view_length)].clone()
    b = tensor[length - int(view_length) : length].clone()
    if int(view_length) > 0 and not (0 + int(view_length) <= length - int(view_length)):
        raise RuntimeError("fingerprint views overlap")
    return a, b


def representation_views(
    encoder: torch.nn.Module | None,
    data: np.ndarray | torch.Tensor,
    representation: str,
    device: torch.device | None = None,
    max_length: int = MAX_LENGTH,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired A/B vectors for fingerprinting."""

    a, b = view_data(data)
    if representation.startswith("pearson"):
        return pearson_fc(a).cpu(), pearson_fc(b).cpu()
    if encoder is None or device is None:
        raise ValueError("VarCoNet view extraction requires encoder and device")
    def padded(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((int(max_length), x.shape[1]), dtype=torch.float32)
        out[: x.shape[0]] = x
        return out
    values = _encode_padded(encoder, torch.stack([padded(a), padded(b)]), device, batch_size)
    return values[0], values[1]


def batch_view_representations(
    encoder: torch.nn.Module | None,
    data: np.ndarray,
    indices: np.ndarray,
    representation: str,
    device: torch.device,
    max_length: int = MAX_LENGTH,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract A/B views for many subjects in one or more batches."""
    raw_indices = np.asarray(indices, dtype=np.int64)
    if raw_indices.size == 0:
        raise RuntimeError("empty fingerprint cohort")
    if representation.startswith("pearson"):
        va, vb = [], []
        for raw in raw_indices.tolist():
            a, b = view_data(data[int(raw)])
            va.append(pearson_fc(a)); vb.append(pearson_fc(b))
        return torch.stack(va), torch.stack(vb)
    if encoder is None:
        raise ValueError("encoder required for VarCoNet views")
    pa, pb = [], []
    for start in range(0, len(raw_indices), int(batch_size)):
        chunk = raw_indices[start : start + int(batch_size)]
        aa = torch.zeros((len(chunk), int(max_length), data.shape[2]), dtype=torch.float32)
        bb = torch.zeros_like(aa)
        for j, raw in enumerate(chunk.tolist()):
            a, b = view_data(data[int(raw)])
            aa[j, : a.shape[0]] = a; bb[j, : b.shape[0]] = b
        enc = _encode_padded(encoder, torch.cat((aa, bb), dim=0), device, batch_size)
        pa.append(enc[: len(chunk)]); pb.append(enc[len(chunk) :])
    return torch.cat(pa), torch.cat(pb)


def batch_representation(
    encoder: torch.nn.Module | None,
    data: np.ndarray,
    indices: np.ndarray,
    representation: str,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract one vector per row, preserving ``indices`` order."""

    raw_indices = np.asarray(indices, dtype=np.int64)
    if raw_indices.size == 0:
        raise RuntimeError("empty representation partition")
    if representation in {"pearson_full", "pearson_windowavg"}:
        values: list[torch.Tensor] = []
        for raw in raw_indices.tolist():
            item = data[int(raw)]
            if representation == "pearson_full":
                values.append(pearson_full(item))
            else:
                values.append(pearson_windowavg(item, build_stable_windows(item)))
        return torch.stack(values, dim=0)
    if not representation.startswith("varconet") or encoder is None:
        raise ValueError(f"unknown representation or missing encoder: {representation}")
    # The audited stable path has one padded window per subject.  Stack a
    # chunk at a time so the encoder is invoked with the same batch semantics
    # as the original extraction while avoiding one GPU launch per subject.
    outputs: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(raw_indices), int(batch_size)):
            chunk = raw_indices[start : start + int(batch_size)]
            padded = torch.zeros((len(chunk), MAX_LENGTH, data.shape[2]), dtype=torch.float32)
            for j, raw in enumerate(chunk.tolist()):
                item = torch.as_tensor(data[int(raw)], dtype=torch.float32)
                n = valid_length(item)
                padded[j, :n] = item[:n]
            outputs.append(encoder(padded.to(device)).detach().cpu().float())
    out = torch.cat(outputs, dim=0)
    if out.ndim != 2 or out.shape[1] != EDGE_DIM or not bool(torch.isfinite(out).all()):
        raise RuntimeError(f"invalid VarCoNet representation shape: {tuple(out.shape)}")
    return out


__all__ = [
    "ROI_COUNT", "EDGE_DIM", "MAX_LENGTH", "VIEW_LENGTH", "pearson_fc",
    "pearson_full", "pearson_windowavg", "varconet_windowavg", "varconet_full",
    "view_data", "representation_views", "batch_view_representations", "batch_representation",
]

"""Deterministic, audit-friendly stable-window construction.

Experiment 1D's deployed ``extract_stable_features`` feeds the complete
320-row zero-padded scan to VarCoNet.  It does *not* call ``test_augment``;
the latter belongs to the original HCP fingerprinting script.  This module
keeps that distinction explicit: the stable extraction window is one complete
scan, while the legacy HCP helper is reproduced verbatim for the parity audit.

Every returned record contains both the padded model input and the real
time-series slice used by the Pearson comparator.  No labels or predictions
are consulted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


MAX_LENGTH = 320
WINDOW_SIZES = (80, 200, 320)
NUM_WINDOWS = 10


@dataclass(frozen=True)
class StableWindow:
    """One deterministic window and its provenance."""

    padded_window: torch.Tensor
    raw_valid_window: torch.Tensor
    start: int
    length: int


def valid_length(data: np.ndarray | torch.Tensor) -> int:
    """Return the contiguous non-zero time length of a padded scan."""

    # Keep memmap-backed caches read-only-safe and avoid PyTorch's warning
    # about non-writable NumPy buffers before tensor construction.
    if isinstance(data, np.ndarray):
        array = torch.from_numpy(np.array(data, copy=True))
    else:
        array = data.detach().cpu().clone()
    if array.ndim != 2:
        raise RuntimeError(f"scan must be [T,R], got {tuple(array.shape)}")
    rows = torch.any(array != 0, dim=1)
    zero = torch.where(~rows)[0]
    length = int(zero[0].item()) if zero.numel() else int(array.shape[0])
    # A padded archive must not contain non-zero values after its first zero.
    if zero.numel() and bool(rows[int(zero[0].item()) :].any()):
        raise RuntimeError("non-contiguous padding in scan")
    return length


def latent_valid_tokens(raw_length: int, kernel_size: int = 4, stride: int = 2) -> int:
    """Number of convolution tokens with a fully real receptive field.

    The 16-channel average pooling in VarCoNet pools the channel dimension,
    not time.  Therefore temporal token count is the Conv1d output length.
    """

    raw_length = int(raw_length)
    if raw_length < int(kernel_size):
        return 0
    return (raw_length - int(kernel_size)) // int(stride) + 1


def _as_float_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    # Dataset caches may be read-only memmaps; make an owned tensor for the
    # audit path instead of exposing a non-writable NumPy buffer to PyTorch.
    tensor = torch.as_tensor(data, dtype=torch.float32).detach().cpu().clone()
    if tensor.ndim != 2:
        raise RuntimeError(f"scan must be [T,R], got {tuple(tensor.shape)}")
    return tensor


def build_stable_windows(
    data: np.ndarray | torch.Tensor,
    wind_sizes: Iterable[int] | None = None,
    num_winds: int = NUM_WINDOWS,
    max_length: int = MAX_LENGTH,
) -> list[StableWindow]:
    """Build the exact Experiment-1D stable extraction window.

    ``wind_sizes`` and ``num_winds`` are accepted to make the audited API
    explicit.  They are intentionally not used for the deployed path: the
    audited Experiment-1D function encodes the complete padded scan once.
    Supplying any non-default value is rejected instead of silently changing
    the preregistered protocol.
    """

    if wind_sizes is not None and tuple(int(x) for x in wind_sizes) != WINDOW_SIZES:
        raise ValueError(f"stable extraction locks wind_sizes={WINDOW_SIZES}")
    if int(num_winds) != NUM_WINDOWS:
        raise ValueError(f"stable extraction locks num_winds={NUM_WINDOWS}")
    tensor = _as_float_tensor(data)
    max_length = int(max_length)
    if tensor.shape[0] > max_length:
        raise RuntimeError(f"scan length {tensor.shape[0]} exceeds max_length={max_length}")
    length = valid_length(tensor)
    padded = torch.zeros((max_length, tensor.shape[1]), dtype=torch.float32)
    padded[:length] = tensor[:length]
    raw = tensor[:length].clone()
    return [StableWindow(padded_window=padded, raw_valid_window=raw, start=0, length=length)]


def build_legacy_test_windows(
    data: np.ndarray | torch.Tensor,
    wind_sizes: Iterable[int] = WINDOW_SIZES,
    num_winds: int = NUM_WINDOWS,
    max_length: int = MAX_LENGTH,
) -> list[StableWindow]:
    """Reproduce ``utils.test_augment`` for a full-length padded scan.

    This helper is used only by smoke/audit code.  It deliberately preserves
    the original integer-step rule (including its endpoint convention), while
    exposing the corresponding real slice for Pearson calculations.
    """

    tensor = _as_float_tensor(data)
    if tensor.shape[0] < int(max_length):
        raise RuntimeError("legacy test_augment parity requires input at least max_length rows")
    sizes = tuple(int(x) for x in wind_sizes)
    if int(num_winds) < 2:
        raise ValueError("num_winds must be >=2")
    out: list[StableWindow] = []
    for size in sizes:
        if size <= 0 or size > int(max_length):
            raise ValueError(f"invalid legacy window size {size}")
        step = (int(tensor.shape[0]) - size) // (int(num_winds) - 1)
        if step <= 0:
            raise ValueError("legacy test_augment would have a zero step")
        for start in range(0, step * int(num_winds), step):
            padded = torch.zeros((int(max_length), tensor.shape[1]), dtype=torch.float32)
            padded[:size] = tensor[start : start + size]
            out.append(
                StableWindow(
                    padded_window=padded,
                    raw_valid_window=tensor[start : start + size].clone(),
                    start=int(start),
                    length=int(size),
                )
            )
    return out


def window_metadata(windows: Iterable[StableWindow]) -> tuple[tuple[int, int], ...]:
    return tuple((int(window.start), int(window.length)) for window in windows)


__all__ = [
    "MAX_LENGTH",
    "WINDOW_SIZES",
    "NUM_WINDOWS",
    "StableWindow",
    "valid_length",
    "latent_valid_tokens",
    "build_stable_windows",
    "build_legacy_test_windows",
    "window_metadata",
]

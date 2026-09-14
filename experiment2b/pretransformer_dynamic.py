"""The fixed pre-Transformer edge-wise dynamic descriptor for Experiment 2B."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class PreTransformerEdgeSwitchExtractor(nn.Module):
    """Compute adjacent-window edge switching in the CNN token space.

    The input is ``[B, 384, L]`` and only complete, non-overlapping windows
    of exactly 15 tokens are used.  For each edge, ``D[e]`` is the mean across
    adjacent windows of ``abs(F[t+1,e] - F[t,e])``.  No graph/ROI summaries or
    stable-deviation statistics are included.
    """

    def __init__(self, roi_num: int = 384, window_tokens: int = 15, eps: float = 1e-8):
        super().__init__()
        self.roi_num = int(roi_num)
        self.window_tokens = int(window_tokens)
        self.eps = float(eps)
        if self.roi_num < 2 or self.window_tokens < 1:
            raise ValueError("invalid ROI/window size")
        tri_i, tri_j = torch.triu_indices(self.roi_num, self.roi_num, offset=1)
        self.register_buffer("triu_i", tri_i, persistent=False)
        self.register_buffer("triu_j", tri_j, persistent=False)
        self.dynamic_dim = int(self.roi_num * (self.roi_num - 1) // 2)

    def _edge_cosine(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,R,W] -> [B,E].  Normalization is per ROI over the window.
        h = F.normalize(h, p=2, dim=-1, eps=self.eps)
        fc = torch.bmm(h, h.transpose(1, 2)).clamp(-1.0, 1.0)
        return fc[:, self.triu_i, self.triu_j]

    def forward(
        self,
        pretransformer: torch.Tensor,
        valid_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        if pretransformer.ndim != 3:
            raise RuntimeError(f"expected pretransformer [B,R,L], got {pretransformer.shape}")
        b, r, length = pretransformer.shape
        if r != self.roi_num:
            raise RuntimeError(f"ROI mismatch: {r} vs {self.roi_num}")
        valid_mask = valid_mask.bool()
        if tuple(valid_mask.shape) != (b, length):
            raise RuntimeError("pretransformer validity mask shape mismatch")
        counts = valid_mask.sum(dim=1).to(torch.long)
        positions = torch.arange(length, device=pretransformer.device).unsqueeze(0)
        expected = positions < counts.unsqueeze(1)
        if not torch.equal(valid_mask, expected):
            raise RuntimeError("pretransformer valid mask is not contiguous")
        n_windows = counts // self.window_tokens
        if bool(torch.any(n_windows < 2)):
            bad = torch.where(n_windows < 2)[0].detach().cpu().tolist()
            raise RuntimeError(f"need >=2 complete 15-token windows; bad batch indices={bad}")

        max_windows = int(n_windows.max().item())
        descriptor = torch.zeros(
            b, self.dynamic_dim, device=pretransformer.device, dtype=pretransformer.dtype
        )
        transition_count = torch.zeros(
            b, 1, device=pretransformer.device, dtype=pretransformer.dtype
        )
        previous: torch.Tensor | None = None
        previous_active: torch.Tensor | None = None
        for window in range(max_windows):
            start = window * self.window_tokens
            end = start + self.window_tokens
            active = n_windows > window
            if not bool(torch.any(active)):
                break
            current = self._edge_cosine(pretransformer[:, :, start:end])
            if previous is not None and previous_active is not None:
                pair_active = previous_active & active
                delta = (current - previous).abs()
                factor = pair_active.to(delta.dtype).unsqueeze(1)
                descriptor += delta * factor
                transition_count += factor
            previous = current
            previous_active = active
        descriptor = descriptor / transition_count.clamp_min(1.0)
        if not bool(torch.isfinite(descriptor).all()):
            raise FloatingPointError("non-finite pre-transformer edge descriptor")

        if not return_diagnostics:
            return descriptor
        per_subject_mean = descriptor.mean(dim=1)
        per_subject_std = descriptor.std(dim=1, unbiased=False)
        diagnostics: dict[str, float] = {
            "valid_tokens_min": float(counts.min().detach().cpu()),
            "valid_tokens_mean": float(counts.float().mean().detach().cpu()),
            "valid_tokens_max": float(counts.max().detach().cpu()),
            "windows_min": float(n_windows.min().detach().cpu()),
            "windows_mean": float(n_windows.float().mean().detach().cpu()),
            "windows_max": float(n_windows.max().detach().cpu()),
            "edge_switch_mean": float(per_subject_mean.mean().detach().cpu()),
            "edge_switch_std": float(per_subject_std.mean().detach().cpu()),
        }
        return descriptor, diagnostics


class FrozenStableEdgeResidualClassifier(nn.Module):
    """Frozen stable MLP plus one zero-initialized scalar edge residual."""

    def __init__(self, baseline_state: dict[str, torch.Tensor], dynamic_dim: int):
        super().__init__()
        from model_scripts.classifier import MLP

        stable_dim = int(baseline_state["fc.0.weight"].shape[1])
        self.stable_classifier = MLP(stable_dim, 2)
        self.stable_classifier.load_state_dict(baseline_state)
        for parameter in self.stable_classifier.parameters():
            parameter.requires_grad_(False)
        self.dynamic_head = nn.Linear(int(dynamic_dim), 1, bias=False)
        nn.init.zeros_(self.dynamic_head.weight)

    def forward(self, stable: torch.Tensor, dynamic_z: torch.Tensor) -> torch.Tensor:
        stable_probs = self.stable_classifier(stable)
        stable_logits = torch.log(stable_probs.clamp_min(1e-7))
        delta = self.dynamic_head(dynamic_z)
        residual_logits = torch.cat((-0.5 * delta, 0.5 * delta), dim=1)
        return torch.softmax(stable_logits + residual_logits, dim=1)

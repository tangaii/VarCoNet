"""Fixed-size latent temporal variability features for Experiment 2.

The extractor deliberately computes only the 774 statistics specified in the
experiment prompt.  It never exposes complete local connectome edge vectors to
the classifier, and all operations are differentiability-safe for batched
inference (the runner calls it under ``torch.no_grad``).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LatentDynamicResidualExtractor(nn.Module):
    """Extract ROI/graph switching and stable-deviation statistics.

    Parameters
    ----------
    roi_num:
        Number of latent ROI channels (384 for AICHA).
    window_tokens:
        Number of consecutive, non-overlapping latent tokens in one window.
    eps:
        Numerical floor used by cosine similarities and normalization.
    """

    def __init__(self, roi_num: int = 384, window_tokens: int = 15, eps: float = 1e-8):
        super().__init__()
        self.roi_num = int(roi_num)
        self.window_tokens = int(window_tokens)
        self.eps = float(eps)
        if self.roi_num < 2:
            raise ValueError("roi_num must be at least 2")
        if self.window_tokens < 1:
            raise ValueError("window_tokens must be positive")
        triu_i, triu_j = torch.triu_indices(self.roi_num, self.roi_num, offset=1)
        self.register_buffer("triu_i", triu_i, persistent=False)
        self.register_buffer("triu_j", triu_j, persistent=False)
        self.register_buffer(
            "offdiag",
            1.0 - torch.eye(self.roi_num, dtype=torch.float32),
            persistent=False,
        )

    def _cosine_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Pairwise ROI cosine matrices for ``x`` with shape ``[B,R,W]``."""
        if x.ndim != 3:
            raise RuntimeError(f"Expected [B,R,W], got {tuple(x.shape)}")
        dot = torch.bmm(x, x.transpose(1, 2))
        norm = torch.sqrt(x.square().sum(dim=-1).clamp_min(self.eps))
        denom = norm.unsqueeze(2) * norm.unsqueeze(1)
        return (dot / denom).clamp(-1.0, 1.0)

    def _scalar_stats(
        self,
        total: torch.Tensor,
        total_sq: torch.Tensor,
        count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count_safe = count.clamp_min(1.0)
        mean = total / count_safe
        variance = (total_sq / count_safe - mean.square()).clamp_min(0.0)
        return mean, torch.sqrt(variance)

    def forward(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """Compute ``D`` from latent tokens and a contiguous validity mask.

        ``h`` is ``[B,R,L]`` and ``valid_mask`` is ``[B,L]`` with true entries
        first (the padding convention used by VarCoNet).
        """
        if h.ndim != 3:
            raise RuntimeError(f"Expected h [B,R,L], got {h.shape}")
        b, r, l = h.shape
        if r != self.roi_num:
            raise RuntimeError(f"ROI mismatch: {r} vs {self.roi_num}")
        valid_mask = valid_mask.bool()
        if valid_mask.shape != (b, l):
            raise RuntimeError(
                f"Mask shape {tuple(valid_mask.shape)} does not match {(b, l)}"
            )

        valid_counts = valid_mask.sum(dim=1)
        pos = torch.arange(l, device=h.device).unsqueeze(0)
        expected_mask = pos < valid_counts.unsqueeze(1)
        if not torch.equal(valid_mask, expected_mask):
            raise RuntimeError("Latent valid mask is not contiguous.")

        n_windows = valid_counts // self.window_tokens
        if torch.any(n_windows < 2):
            bad = torch.where(n_windows < 2)[0].tolist()
            raise RuntimeError(
                "Need >=2 complete latent windows. " f"Bad batch indices: {bad}"
            )
        max_windows = int(n_windows.max().item())

        # Whole-scan stable matrix in exactly the same latent space as the
        # original VarCoNet learned-FC output.
        h_full = h * valid_mask.unsqueeze(1)
        stable_fc = self._cosine_matrix(h_full)
        stable_vec = stable_fc[:, self.triu_i, self.triu_j]

        node_switch_sum = torch.zeros(b, r, device=h.device, dtype=h.dtype)
        node_dev_sum = torch.zeros_like(node_switch_sum)
        switch_count = torch.zeros(b, 1, device=h.device, dtype=h.dtype)
        dev_count = torch.zeros_like(switch_count)

        graph_switch_sum = torch.zeros(b, device=h.device, dtype=h.dtype)
        graph_switch_sq = torch.zeros_like(graph_switch_sum)
        graph_switch_count = torch.zeros_like(graph_switch_sum)
        graph_dev_sum = torch.zeros_like(graph_switch_sum)
        graph_dev_sq = torch.zeros_like(graph_switch_sum)
        graph_dev_count = torch.zeros_like(graph_switch_sum)
        transition_sum = torch.zeros_like(graph_switch_sum)
        transition_sq = torch.zeros_like(graph_switch_sum)
        transition_count = torch.zeros_like(graph_switch_sum)

        prev_fc = None
        prev_vec = None
        prev_active = None
        offdiag = self.offdiag.to(device=h.device, dtype=h.dtype)

        for w in range(max_windows):
            start = w * self.window_tokens
            end = start + self.window_tokens
            active = n_windows > w
            if not torch.any(active):
                break
            hw = h[:, :, start:end]
            fc = self._cosine_matrix(hw)
            vec = fc[:, self.triu_i, self.triu_j]

            # Local-vs-stable deviation.
            deviation = (fc - stable_fc).abs() * offdiag.unsqueeze(0)
            node_dev = deviation.sum(dim=-1) / float(r - 1)
            af = active.to(h.dtype)
            node_dev_sum += node_dev * af.unsqueeze(1)
            dev_count += af.unsqueeze(1)
            graph_dev = (vec - stable_vec).abs().mean(dim=1)
            graph_dev_sum += graph_dev * af
            graph_dev_sq += graph_dev.square() * af
            graph_dev_count += af

            # Consecutive state switching.
            if prev_fc is not None:
                pair_active = prev_active & active
                pf = pair_active.to(h.dtype)
                delta_abs = (fc - prev_fc).abs() * offdiag.unsqueeze(0)
                node_switch = delta_abs.sum(dim=-1) / float(r - 1)
                node_switch_sum += node_switch * pf.unsqueeze(1)
                switch_count += pf.unsqueeze(1)
                graph_switch = (vec - prev_vec).abs().mean(dim=1)
                graph_switch_sum += graph_switch * pf
                graph_switch_sq += graph_switch.square() * pf
                graph_switch_count += pf
                transition_distance = 1.0 - F.cosine_similarity(
                    vec, prev_vec, dim=1, eps=self.eps
                )
                transition_sum += transition_distance * pf
                transition_sq += transition_distance.square() * pf
                transition_count += pf

            prev_fc = fc
            prev_vec = vec
            prev_active = active

        node_switch_mean = node_switch_sum / switch_count.clamp_min(1.0)
        node_dev_mean = node_dev_sum / dev_count.clamp_min(1.0)
        graph_switch_mean, graph_switch_std = self._scalar_stats(
            graph_switch_sum, graph_switch_sq, graph_switch_count
        )
        graph_dev_mean, graph_dev_std = self._scalar_stats(
            graph_dev_sum, graph_dev_sq, graph_dev_count
        )
        transition_mean, transition_std = self._scalar_stats(
            transition_sum, transition_sq, transition_count
        )

        dynamic = torch.cat(
            [
                node_switch_mean,
                node_dev_mean,
                graph_switch_mean[:, None],
                graph_switch_std[:, None],
                graph_dev_mean[:, None],
                graph_dev_std[:, None],
                transition_mean[:, None],
                transition_std[:, None],
            ],
            dim=1,
        )
        expected_dim = 2 * self.roi_num + 6
        if dynamic.shape[1] != expected_dim:
            raise RuntimeError(f"Dynamic dim {dynamic.shape[1]} != {expected_dim}")
        if not torch.isfinite(dynamic).all():
            raise FloatingPointError("Non-finite dynamic feature.")

        if return_diagnostics:
            diagnostics = {
                "valid_tokens_min": float(valid_counts.min().detach().cpu()),
                "valid_tokens_mean": float(valid_counts.float().mean().detach().cpu()),
                "windows_min": float(n_windows.min().detach().cpu()),
                "windows_mean": float(n_windows.float().mean().detach().cpu()),
                "windows_max": float(n_windows.max().detach().cpu()),
                "node_switch_mean": float(node_switch_mean.mean().detach().cpu()),
                "node_deviation_mean": float(node_dev_mean.mean().detach().cpu()),
            }
            return dynamic, diagnostics
        return dynamic


class FrozenStableResidualClassifier(nn.Module):
    """Baseline logits plus a zero-initialized, trainable dynamic residual."""

    def __init__(self, baseline_state: dict[str, torch.Tensor], dynamic_dim: int):
        super().__init__()
        stable_w = baseline_state["fc.0.weight"].detach().clone()
        stable_b = baseline_state["fc.0.bias"].detach().clone()
        self.register_buffer("stable_weight", stable_w)
        self.register_buffer("stable_bias", stable_b)
        self.dynamic_head = nn.Linear(int(dynamic_dim), 2, bias=False)
        nn.init.zeros_(self.dynamic_head.weight)

    def forward(self, stable: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        stable_logits = F.linear(stable, self.stable_weight, self.stable_bias)
        dynamic_logits = self.dynamic_head(dynamic)
        return torch.softmax(stable_logits + dynamic_logits, dim=1)


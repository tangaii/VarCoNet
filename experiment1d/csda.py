"""Cross-Site Disease-Shared Adapter (CSDA) treatment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_features import ProfileState, margin_from_baseline_state, normalized_profile, profile_state_from_payload, profile_state_payload


def cross_site_supcon_loss(z: torch.Tensor, labels: torch.Tensor, site_ids: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Fixed same-diagnosis/different-site supervised contrastive loss."""

    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / float(tau)
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same_y = labels[:, None] == labels[None, :]
    diff_y = ~same_y
    diff_site = site_ids[:, None] != site_ids[None, :]
    positive = same_y & diff_site & ~eye
    negative = diff_y & ~eye
    valid_pair = positive | negative
    valid_anchor = positive.sum(1) > 0
    if int(valid_anchor.sum()) < 2:
        raise RuntimeError("CSDA insufficient cross-site positives")
    masked_sim = sim.masked_fill(~valid_pair, float("-inf"))
    log_denom = torch.logsumexp(masked_sim, dim=1, keepdim=True)
    log_prob = sim - log_denom
    pos_count = positive.sum(1).clamp_min(1)
    loss_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(1) / pos_count)
    return loss_anchor[valid_anchor].mean()


class CrossSiteDiseaseAdapter(nn.Module):
    """384-D profile projection plus a zero-initialized disease residual head."""

    def __init__(self, baseline_state: dict[str, torch.Tensor], profile_state: ProfileState):
        super().__init__()
        self.register_buffer("baseline_weight", baseline_state["fc.0.weight"].detach().float().clone())
        self.register_buffer("baseline_bias", baseline_state["fc.0.bias"].detach().float().clone())
        self.register_buffer("profile_mu", profile_state.mu.float().clone())
        self.register_buffer("profile_sigma", profile_state.sigma.float().clone())
        self.proj = nn.Linear(384, 32, bias=False)
        self.head = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def disease_embedding(self, profile: torch.Tensor) -> torch.Tensor:
        return self.proj((profile - self.profile_mu) / self.profile_sigma)

    def forward_details(self, stable: torch.Tensor, profile: torch.Tensor) -> dict[str, torch.Tensor]:
        z_raw = self.disease_embedding(profile)
        z = F.normalize(z_raw, dim=1)
        delta = self.head(z).squeeze(1)
        m0 = margin_from_baseline_state(stable, {"fc.0.weight": self.baseline_weight, "fc.0.bias": self.baseline_bias})
        margin = m0 + delta
        return {"z_raw": z_raw, "z": z, "delta": delta, "m0": m0, "margin": margin, "p_hc": torch.sigmoid(-margin)}

    def forward(self, stable: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        details = self.forward_details(stable, profile)
        return torch.stack((1.0 - details["p_hc"], details["p_hc"]), dim=1)


def state_payload(model: CrossSiteDiseaseAdapter) -> dict[str, Any]:
    return {
        "proj": {key: value.detach().clone() for key, value in model.proj.state_dict().items()},
        "head": {key: value.detach().clone() for key, value in model.head.state_dict().items()},
    }


def load_adapter_state(model: CrossSiteDiseaseAdapter, payload: dict[str, Any]) -> None:
    model.proj.load_state_dict(payload["proj"])
    model.head.load_state_dict(payload["head"])


def is_exact_zero_state(payload: dict[str, Any]) -> bool:
    return all(bool(torch.equal(torch.as_tensor(value), torch.zeros_like(torch.as_tensor(value)))) for group in ("head",) for value in payload[group].values())


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _pair_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean().detach().cpu()) if selected.numel() else float("nan")


def diagnostics_from_model(
    model: CrossSiteDiseaseAdapter,
    stable_reference: torch.Tensor,
    profile_reference: torch.Tensor,
    reference_labels: torch.Tensor,
    reference_site_ids: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable_reference, profile_reference)
        z = details["z"]
        sim = z @ z.T
        eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        same_y = reference_labels[:, None] == reference_labels[None, :]
        diff_y = ~same_y
        same_site = reference_site_ids[:, None] == reference_site_ids[None, :]
        diff_site = ~same_site
        cross_mask = same_y & diff_site & ~eye
        same_mask = same_y & same_site & ~eye
        opposite_mask = diff_y & ~eye
        nearest = sim.masked_fill(eye, float("-inf")).argmax(dim=1)
        site_retrieval = float((reference_site_ids[nearest] == reference_site_ids).float().mean().detach().cpu())
        return {
            "csda_num_sites": float(torch.unique(reference_site_ids).numel()),
            "csda_cross_site_positive_pairs": float(cross_mask.sum().detach().cpu()),
            "csda_cross_site_same_y_cos": _pair_mean(sim, cross_mask),
            "csda_same_site_same_y_cos": _pair_mean(sim, same_mask),
            "csda_opposite_y_cos": _pair_mean(sim, opposite_mask),
            "csda_site_retrieval_accuracy": site_retrieval,
            "csda_projection_l2": float(torch.linalg.vector_norm(model.proj.weight.detach()).cpu()),
            "csda_head_l2": float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.head.parameters())).cpu()),
            "csda_mean_abs_delta": float(details["delta"].abs().mean().detach().cpu()),
        }


__all__ = [
    "cross_site_supcon_loss", "CrossSiteDiseaseAdapter", "state_payload",
    "load_adapter_state", "is_exact_zero_state", "trainable_parameter_count",
    "diagnostics_from_model",
]

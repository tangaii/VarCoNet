"""Individual-Conditioned Disease Mask (ICDM) treatment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_features import ProfileState, margin_from_baseline_state


class IndividualConditionedDiseaseMask(nn.Module):
    """Bounded subject-conditioned ROI gates before the frozen classifier."""

    def __init__(
        self,
        baseline_state: dict[str, torch.Tensor],
        profile_state: ProfileState,
        roi_num: int = 384,
        bottleneck: int = 16,
        max_log_gate: float = 0.1,
    ):
        super().__init__()
        if roi_num != 384 or bottleneck != 16:
            raise ValueError("ICDM configuration is fixed at 384->16->384")
        self.roi_num = int(roi_num)
        self.max_log_gate = float(max_log_gate)
        self.register_buffer("baseline_weight", baseline_state["fc.0.weight"].detach().float().clone())
        self.register_buffer("baseline_bias", baseline_state["fc.0.bias"].detach().float().clone())
        self.register_buffer("profile_mu", profile_state.mu.float().clone())
        self.register_buffer("profile_sigma", profile_state.sigma.float().clone())
        tri = torch.triu_indices(self.roi_num, self.roi_num, offset=1)
        self.register_buffer("edge_i", tri[0], persistent=False)
        self.register_buffer("edge_j", tri[1], persistent=False)
        self.down = nn.Linear(self.roi_num, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, self.roi_num, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward_details(self, stable: torch.Tensor, profile: torch.Tensor) -> dict[str, torch.Tensor]:
        context = (profile - self.profile_mu) / self.profile_sigma
        hidden = F.silu(self.down(context))
        roi_prompt = self.max_log_gate * torch.tanh(self.up(hidden))
        edge_log_gate = 0.5 * (roi_prompt[:, self.edge_i] + roi_prompt[:, self.edge_j])
        edge_factor = torch.exp(edge_log_gate)
        stable_adapted = stable * edge_factor
        logits = F.linear(stable_adapted, self.baseline_weight, self.baseline_bias)
        m0 = F.linear(stable, self.baseline_weight, self.baseline_bias)
        baseline_margin = m0[:, 0] - m0[:, 1]
        margin = logits[:, 0] - logits[:, 1]
        return {
            "roi_prompt": roi_prompt,
            "edge_log_gate": edge_log_gate,
            "edge_factor": edge_factor,
            "stable_adapted": stable_adapted,
            "m0": baseline_margin,
            "margin": margin,
            "p_hc": torch.sigmoid(-margin),
        }

    def forward(self, stable: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        details = self.forward_details(stable, profile)
        return torch.stack((1.0 - details["p_hc"], details["p_hc"]), dim=1)


def state_payload(model: IndividualConditionedDiseaseMask) -> dict[str, Any]:
    return {
        "down": {key: value.detach().clone() for key, value in model.down.state_dict().items()},
        "up": {key: value.detach().clone() for key, value in model.up.state_dict().items()},
    }


def load_mask_state(model: IndividualConditionedDiseaseMask, payload: dict[str, Any]) -> None:
    model.down.load_state_dict(payload["down"])
    model.up.load_state_dict(payload["up"])


def is_exact_zero_state(payload: dict[str, Any]) -> bool:
    return all(bool(torch.equal(torch.as_tensor(value), torch.zeros_like(torch.as_tensor(value)))) for value in payload["up"].values())


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def diagnostics_from_model(model: IndividualConditionedDiseaseMask, stable_reference: torch.Tensor, profile_reference: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable_reference, profile_reference)
        factor = details["edge_factor"]
        log_gate = details["edge_log_gate"]
        adapted_diff = (details["stable_adapted"] - stable_reference).abs()
        return {
            "icdm_mean_abs_roi_prompt": float(details["roi_prompt"].abs().mean().detach().cpu()),
            "icdm_mean_abs_edge_log_gate": float(log_gate.abs().mean().detach().cpu()),
            "icdm_mean_edge_factor": float(factor.mean().detach().cpu()),
            "icdm_std_edge_factor": float(factor.std(unbiased=False).detach().cpu()),
            "icdm_factor_min": float(factor.min().detach().cpu()),
            "icdm_factor_max": float(factor.max().detach().cpu()),
            "icdm_fraction_gt1": float((factor > 1.0).float().mean().detach().cpu()),
            "icdm_mean_abs_fc_change": float(adapted_diff.mean().detach().cpu()),
            "icdm_down_l2": float(torch.linalg.vector_norm(model.down.weight.detach()).cpu()),
            "icdm_up_l2": float(torch.linalg.vector_norm(model.up.weight.detach()).cpu()),
        }


__all__ = [
    "IndividualConditionedDiseaseMask", "state_payload", "load_mask_state",
    "is_exact_zero_state", "trainable_parameter_count", "diagnostics_from_model",
]

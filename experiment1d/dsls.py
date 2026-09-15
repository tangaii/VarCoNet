"""Disease-Shared Low-Rank Subspace (DSLS) treatment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_features import margin_from_baseline_state


@dataclass(frozen=True)
class DSLSState:
    mu_hc: torch.Tensor
    sigma_hc: torch.Tensor
    basis: torch.Tensor
    score_mu_hc: torch.Tensor
    score_sigma_hc: torch.Tensor
    std_floor: float
    singular_values: torch.Tensor
    asd_score_mean: torch.Tensor
    asd_score_std: torch.Tensor
    shared_energy_fraction: float


def _fork_devices(tensor: torch.Tensor) -> list[int]:
    if tensor.is_cuda:
        return [int(tensor.device.index or 0)]
    return []


@torch.no_grad()
def fit_dsls_state(
    stable_reference: torch.Tensor,
    reference_labels: torch.Tensor,
    seed: int,
    n_components: int = 3,
) -> DSLSState:
    """Fit an uncentered ASD-vs-HC low-rank basis on reference subjects only."""

    labels = reference_labels.long()
    hc = stable_reference[labels == 1]
    asd = stable_reference[labels == 0]
    if hc.shape[0] < 10:
        raise RuntimeError("DSLS too few HC reference subjects")
    if asd.shape[0] < 10:
        raise RuntimeError("DSLS too few ASD reference subjects")
    if asd.shape[0] < n_components:
        raise RuntimeError("DSLS ASD reference rank is below three components")
    mu = hc.mean(dim=0)
    raw_std = hc.std(dim=0, unbiased=False)
    positive = raw_std[raw_std > 0]
    if positive.numel() == 0:
        raise RuntimeError("DSLS HC standard deviation is identically zero")
    floor = max(1e-6, 0.1 * float(positive.median().detach().cpu()))
    sigma = raw_std.clamp_min(float(floor))
    z_asd = (asd - mu) / sigma
    if not bool(torch.isfinite(z_asd).all()):
        raise FloatingPointError("DSLS non-finite ASD deviation matrix")
    devices = _fork_devices(stable_reference)
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if stable_reference.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        _, singular_values, right_vectors = torch.pca_lowrank(
            z_asd, q=int(n_components), center=False, niter=5
        )
    basis = right_vectors[:, :n_components].clone()
    for component in range(n_components):
        pivot = int(torch.argmax(basis[:, component].abs()).detach().cpu())
        if float(basis[pivot, component].detach().cpu()) < 0:
            basis[:, component].mul_(-1.0)
    gram = basis.T @ basis
    eye = torch.eye(n_components, device=gram.device, dtype=gram.dtype)
    if not torch.allclose(gram, eye, atol=1e-4, rtol=1e-4):
        raise RuntimeError("DSLS basis is not orthonormal")
    z_hc = (hc - mu) / sigma
    hc_scores = z_hc @ basis
    score_mu = hc_scores.mean(dim=0)
    score_sigma = hc_scores.std(dim=0, unbiased=False)
    if bool(torch.any(score_sigma <= 1e-6)):
        raise RuntimeError("DSLS HC score collapsed")
    asd_scores = (z_asd @ basis - score_mu) / score_sigma
    total_energy = z_asd.square().sum(dim=1).clamp_min(1e-12)
    shared_energy = (z_asd @ basis).square().sum(dim=1)
    energy_fraction = float((shared_energy / total_energy).mean().detach().cpu())
    return DSLSState(
        mu.detach().cpu(), sigma.detach().cpu(), basis.detach().cpu(),
        score_mu.detach().cpu(), score_sigma.detach().cpu(), float(floor),
        singular_values[:n_components].detach().cpu(),
        asd_scores.mean(dim=0).detach().cpu(), asd_scores.std(dim=0, unbiased=False).detach().cpu(),
        energy_fraction,
    )


def state_payload(state: DSLSState) -> dict[str, Any]:
    return {
        "mu_hc": state.mu_hc.detach().cpu().clone(),
        "sigma_hc": state.sigma_hc.detach().cpu().clone(),
        "basis": state.basis.detach().cpu().clone(),
        "score_mu_hc": state.score_mu_hc.detach().cpu().clone(),
        "score_sigma_hc": state.score_sigma_hc.detach().cpu().clone(),
        "std_floor": float(state.std_floor),
        "singular_values": state.singular_values.detach().cpu().clone(),
        "asd_score_mean": state.asd_score_mean.detach().cpu().clone(),
        "asd_score_std": state.asd_score_std.detach().cpu().clone(),
        "shared_energy_fraction": float(state.shared_energy_fraction),
    }


def state_from_payload(payload: dict[str, Any]) -> DSLSState:
    tensor = lambda key: torch.as_tensor(payload[key]).detach().cpu().clone()
    return DSLSState(
        tensor("mu_hc"), tensor("sigma_hc"), tensor("basis"),
        tensor("score_mu_hc"), tensor("score_sigma_hc"), float(payload["std_floor"]),
        tensor("singular_values"), tensor("asd_score_mean"), tensor("asd_score_std"),
        float(payload["shared_energy_fraction"]),
    )


class DSLSResidual(nn.Module):
    """A 3->8->1 shared-score residual added to the frozen baseline margin."""

    def __init__(self, baseline_state: dict[str, torch.Tensor], dsls_state: DSLSState):
        super().__init__()
        self.register_buffer("baseline_weight", baseline_state["fc.0.weight"].detach().float().clone())
        self.register_buffer("baseline_bias", baseline_state["fc.0.bias"].detach().float().clone())
        self.register_buffer("mu_hc", dsls_state.mu_hc.float().clone())
        self.register_buffer("sigma_hc", dsls_state.sigma_hc.float().clone())
        self.register_buffer("basis", dsls_state.basis.float().clone())
        self.register_buffer("score_mu", dsls_state.score_mu_hc.float().clone())
        self.register_buffer("score_sigma", dsls_state.score_sigma_hc.float().clone())
        self.hidden = nn.Linear(3, 8)
        self.out = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def baseline_margin(self, stable: torch.Tensor) -> torch.Tensor:
        return F.linear(stable, self.baseline_weight, self.baseline_bias)[:, 0] - F.linear(stable, self.baseline_weight, self.baseline_bias)[:, 1]

    def shared_scores(self, stable: torch.Tensor) -> torch.Tensor:
        z = (stable - self.mu_hc) / self.sigma_hc
        return (z @ self.basis - self.score_mu) / self.score_sigma

    def forward_details(self, stable: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = self.shared_scores(stable)
        delta = self.out(F.gelu(self.hidden(scores))).squeeze(1)
        m0 = self.baseline_margin(stable)
        margin = m0 + delta
        return {"scores": scores, "delta": delta, "m0": m0, "margin": margin, "p_hc": torch.sigmoid(-margin)}

    def forward(self, stable: torch.Tensor) -> torch.Tensor:
        details = self.forward_details(stable)
        return torch.stack((1.0 - details["p_hc"], details["p_hc"]), dim=1)


def residual_state_payload(model: DSLSResidual) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if key in {"hidden.weight", "hidden.bias", "out.weight", "out.bias"}}


def load_residual_state(model: DSLSResidual, payload: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    for key, value in payload.items():
        if key in current:
            current[key].copy_(torch.as_tensor(value, device=current[key].device, dtype=current[key].dtype))


def is_exact_zero_state(payload: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.equal(torch.as_tensor(payload[key]), torch.zeros_like(torch.as_tensor(payload[key])))) for key in ("out.weight", "out.bias"))


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def diagnostics_from_model(model: DSLSResidual, stable_reference: torch.Tensor, reference_labels: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable_reference)
        scores = details["scores"]
        delta = details["delta"]
        asd = reference_labels == 0
        hc = reference_labels == 1
        mean = lambda value: float(value.mean().detach().cpu()) if value.numel() else float("nan")
        std = lambda value: float(value.std(unbiased=False).detach().cpu()) if value.numel() else float("nan")
        return {
            "dsls_singular1": float("nan"), "dsls_singular2": float("nan"), "dsls_singular3": float("nan"),
            "dsls_shared_energy_fraction": float("nan"),
            "dsls_asd_score_mean": mean(scores[asd]), "dsls_asd_score_std": std(scores[asd]),
            "dsls_hc_score_mean": mean(scores[hc]), "dsls_hc_score_std": std(scores[hc]),
            "dsls_parameter_l2": float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).detach().cpu()),
            "dsls_mean_abs_delta": float(delta.abs().mean().detach().cpu()),
        }


__all__ = [
    "DSLSState", "fit_dsls_state", "state_payload", "state_from_payload",
    "DSLSResidual", "residual_state_payload", "load_residual_state",
    "is_exact_zero_state", "trainable_parameter_count", "diagnostics_from_model",
]

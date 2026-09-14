"""Control-referenced soft multi-boundary readout for Experiment 1B.

This module deliberately operates *after* the audited VarCoNet encoder and
stable linear classifier have been selected.  It contains no encoder, no
prototype assignment, no clustering objective, and no test-time adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MechanismInitError(RuntimeError):
    """Raised when the prescribed deterministic symmetry break is undefined."""


@dataclass(frozen=True)
class SymmetryDiagnostics:
    n_experts: int
    temperature: float
    pca_unit_norm: float
    pca_singular_value: float
    sigma_margin: float
    sigma_projection: float
    symmetry_ratio: float
    symmetry_scale: float
    initial_delta_l2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n_experts": float(self.n_experts),
            "temperature": float(self.temperature),
            "pca_unit_norm": float(self.pca_unit_norm),
            "pca_singular_value": float(self.pca_singular_value),
            "sigma_margin": float(self.sigma_margin),
            "sigma_projection": float(self.sigma_projection),
            "symmetry_ratio": float(self.symmetry_ratio),
            "symmetry_scale": float(self.symmetry_scale),
            "initial_delta_l2": float(self.initial_delta_l2),
        }


def margin_from_baseline_state(
    stable: torch.Tensor, baseline_state: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return m0=z_ASD-z_HC from the frozen audited softmax classifier."""
    weight = baseline_state["fc.0.weight"].to(device=stable.device, dtype=stable.dtype)
    bias = baseline_state["fc.0.bias"].to(device=stable.device, dtype=stable.dtype)
    # Compute the two frozen logits separately before subtraction.  This is
    # algebraically (W_ASD-W_HC)x+(b_ASD-b_HC), while matching the audited
    # classifier's floating-point accumulation order for parity checks.
    logits = F.linear(stable, weight, bias)
    return logits[:, 0] - logits[:, 1]


def baseline_probabilities_from_margin(margin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return p_ASD, p_HC with the repository's class order ASD=0, HC=1."""
    return torch.sigmoid(margin), torch.sigmoid(-margin)


class ControlReferencedSoftMultiBoundary(nn.Module):
    """Frozen baseline margin plus exactly two trainable residual boundaries.

    Trainable parameters are only ``delta_weight`` with shape ``[2, D]`` and
    ``delta_bias`` with shape ``[2]``.  The centroid and baseline classifier
    quantities are buffers so the original baseline cannot be changed.
    """

    def __init__(
        self,
        baseline_state: dict[str, torch.Tensor],
        hc_centroid: torch.Tensor,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if float(temperature) != 1.0:
            raise ValueError("Experiment 1B fixes temperature at 1.0")
        if hc_centroid.ndim != 1:
            raise ValueError("HC centroid must be a one-dimensional stable-FC vector")
        weight = baseline_state["fc.0.weight"].detach().float().cpu()
        bias = baseline_state["fc.0.bias"].detach().float().cpu()
        if tuple(weight.shape) != (2, int(hc_centroid.numel())) or tuple(bias.shape) != (2,):
            raise ValueError("frozen baseline classifier dimensions do not match HC centroid")
        self.register_buffer("baseline_weight", weight.clone())
        self.register_buffer("baseline_bias", bias.clone())
        self.register_buffer("hc_centroid", hc_centroid.detach().float().cpu().clone())
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))
        self.delta_weight = nn.Parameter(torch.zeros((2, int(hc_centroid.numel())), dtype=torch.float32))
        self.delta_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    @property
    def n_experts(self) -> int:
        return 2

    @property
    def stable_dim(self) -> int:
        return int(self.hc_centroid.numel())

    def baseline_margin(self, stable: torch.Tensor) -> torch.Tensor:
        logits = F.linear(stable, self.baseline_weight, self.baseline_bias)
        return logits[:, 0] - logits[:, 1]

    def expert_margins(self, stable: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m0 = self.baseline_margin(stable)
        residual = stable - self.hc_centroid
        corrections = F.linear(residual, self.delta_weight, self.delta_bias)
        return m0, m0.unsqueeze(1) + corrections

    def mixed_margin(self, expert_margins: torch.Tensor) -> torch.Tensor:
        tau = self.temperature
        return tau * (
            torch.logsumexp(expert_margins / tau, dim=1)
            - torch.log(torch.as_tensor(float(self.n_experts), device=expert_margins.device, dtype=expert_margins.dtype))
        )

    def forward_details(self, stable: torch.Tensor) -> dict[str, torch.Tensor]:
        m0, experts = self.expert_margins(stable)
        mixed = self.mixed_margin(experts)
        p_asd, p_hc = baseline_probabilities_from_margin(mixed)
        responsibilities = torch.softmax(experts / self.temperature, dim=1)
        return {
            "m0": m0,
            "expert_margins": experts,
            "mixed_margin": mixed,
            "p_asd": p_asd,
            "p_hc": p_hc,
            "responsibilities": responsibilities,
        }

    def forward(self, stable: torch.Tensor) -> torch.Tensor:
        """Return paired probabilities in audited classifier order [ASD, HC]."""
        details = self.forward_details(stable)
        return torch.stack((details["p_asd"], details["p_hc"]), dim=1)


def crsm_loss(details: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Prescribed soft two-boundary loss.

    ASD examples are optimized through the log-mean-exp mixed margin. HC
    examples incur a loss for every boundary, enforcing the shared control
    reference.  The two class terms are weighted by their observed counts.
    """
    labels = labels.long()
    asd = labels == 0
    hc = labels == 1
    n_asd = int(asd.sum().item())
    n_hc = int(hc.sum().item())
    if n_asd <= 0 or n_hc <= 0:
        raise RuntimeError("CRSM loss requires ASD and HC examples in every training task")
    asd_loss = F.softplus(-details["mixed_margin"][asd]).mean()
    hc_loss = F.softplus(details["expert_margins"][hc]).mean()
    return (n_asd * asd_loss + n_hc * hc_loss) / float(n_asd + n_hc)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def initialize_symmetric_pca(
    model: ControlReferencedSoftMultiBoundary,
    reference_stable: torch.Tensor,
    reference_labels: torch.Tensor,
    seed: int,
    ratio: float = 0.01,
) -> SymmetryDiagnostics:
    """Apply the fixed ± PCA symmetry break without consuming caller RNG.

    The direction comes only from ASD residuals of the unique outer-training
    reference set.  A random fallback is intentionally forbidden.
    """
    if float(ratio) != 0.01:
        raise ValueError("Experiment 1B fixes the symmetry ratio at 0.01")
    if reference_stable.ndim != 2 or reference_stable.shape[1] != model.stable_dim:
        raise ValueError("reference stable features have an unexpected shape")
    labels = reference_labels.to(device=reference_stable.device, dtype=torch.long)
    asd = reference_stable[labels == 0]
    hc = reference_stable[labels == 1]
    if len(asd) < 2 or len(hc) < 1:
        raise MechanismInitError("MECHANISM_INIT_FAIL: insufficient unique reference class counts")
    centroid = hc.mean(dim=0)
    if not bool(torch.isfinite(centroid).all()):
        raise MechanismInitError("MECHANISM_INIT_FAIL: non-finite HC centroid")
    residual = asd - centroid
    if not bool(torch.isfinite(residual).all()):
        raise MechanismInitError("MECHANISM_INIT_FAIL: non-finite ASD residual matrix")
    devices: list[int] = []
    if reference_stable.device.type == "cuda":
        devices = [int(reference_stable.device.index or 0)]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if reference_stable.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        _, singular_values, vectors = torch.pca_lowrank(residual, q=1, center=False, niter=3)
    direction = vectors[:, 0]
    direction_norm = torch.linalg.vector_norm(direction)
    m0 = model.baseline_margin(reference_stable)
    projection = residual @ direction
    sigma_margin = m0.std(unbiased=False)
    sigma_projection = projection.std(unbiased=False)
    quantities = (direction_norm, singular_values[0], sigma_margin, sigma_projection)
    if not all(bool(torch.isfinite(value)) for value in quantities):
        raise MechanismInitError("MECHANISM_INIT_FAIL: non-finite PCA or scale diagnostic")
    if float(sigma_margin.detach().cpu()) < 1e-8 or float(sigma_projection.detach().cpu()) < 1e-8:
        raise MechanismInitError(
            "MECHANISM_INIT_FAIL: sigma_margin or sigma_projection is below 1e-8"
        )
    unit_direction = direction / direction_norm
    scale = float(ratio) * sigma_margin / sigma_projection
    with torch.no_grad():
        model.delta_weight[0].copy_(scale * unit_direction)
        model.delta_weight[1].copy_(-scale * unit_direction)
        model.delta_bias.zero_()
    if not bool(torch.isfinite(model.delta_weight).all()):
        raise MechanismInitError("MECHANISM_INIT_FAIL: non-finite initialized delta weights")
    return SymmetryDiagnostics(
        n_experts=2,
        temperature=float(model.temperature.detach().cpu()),
        pca_unit_norm=float(torch.linalg.vector_norm(unit_direction).detach().cpu()),
        pca_singular_value=float(singular_values[0].detach().cpu()),
        sigma_margin=float(sigma_margin.detach().cpu()),
        sigma_projection=float(sigma_projection.detach().cpu()),
        symmetry_ratio=float(ratio),
        symmetry_scale=float(scale.detach().cpu()),
        initial_delta_l2=float(torch.linalg.vector_norm(model.delta_weight.detach()).cpu()),
    )


def diagnostics_from_model(
    model: ControlReferencedSoftMultiBoundary,
    stable: torch.Tensor,
) -> dict[str, float]:
    """Summarize soft-boundary use on a label-free evaluation feature matrix."""
    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable)
        q = details["responsibilities"]
        hard = q.argmax(dim=1)
        entropy = -(q * torch.log(q.clamp_min(1e-12))).sum(dim=1)
        norms = torch.linalg.vector_norm(model.delta_weight, dim=1)
        cosine = F.cosine_similarity(model.delta_weight[0], model.delta_weight[1], dim=0)
    return {
        "expert_1_mean_mass": float(q[:, 0].mean().cpu()),
        "expert_2_mean_mass": float(q[:, 1].mean().cpu()),
        "expert_1_hard_fraction": float((hard == 0).float().mean().cpu()),
        "expert_2_hard_fraction": float((hard == 1).float().mean().cpu()),
        "responsibility_entropy": float(entropy.mean().cpu()),
        "delta_weight_1_l2": float(norms[0].cpu()),
        "delta_weight_2_l2": float(norms[1].cpu()),
        "delta_weight_cosine": float(cosine.cpu()),
        "delta_bias_1": float(model.delta_bias[0].detach().cpu()),
        "delta_bias_2": float(model.delta_bias[1].detach().cpu()),
        "hc_centroid_norm": float(torch.linalg.vector_norm(model.hc_centroid).cpu()),
    }


def state_payload(model: ControlReferencedSoftMultiBoundary) -> dict[str, Any]:
    """CPU-only persistent state for the two trainable delta tensors."""
    return {
        "delta_weight": model.delta_weight.detach().cpu().clone(),
        "delta_bias": model.delta_bias.detach().cpu().clone(),
    }


def load_delta_state(model: ControlReferencedSoftMultiBoundary, state: dict[str, Any]) -> None:
    with torch.no_grad():
        model.delta_weight.copy_(state["delta_weight"].to(model.delta_weight.device))
        model.delta_bias.copy_(state["delta_bias"].to(model.delta_bias.device))

"""Selective normative-deviation experts for Experiment 1C.

The module is deliberately a small post-hoc readout over a frozen selected
VarCoNet representation.  It has no encoder parameters, no clustering,
prototype, adversarial, meta-learning, or test-time fitting components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ROIConnectivityProfile(nn.Module):
    """Map a stable upper-triangle FC vector to 384 signed ROI mean profiles.

    For ROI ``r``, ``G_r`` is the signed mean of its 383 off-diagonal learned
    FC entries.  The edge order is exactly the ``torch.triu_indices`` order
    used by the audited VarCoNet cosine-connectivity output.
    """

    def __init__(self, roi_count: int = 384, *, roi_num: int | None = None) -> None:
        super().__init__()
        if roi_num is not None:
            if int(roi_count) != 384 and int(roi_count) != int(roi_num):
                raise ValueError("roi_count and roi_num disagree")
            roi_count = int(roi_num)
        if int(roi_count) < 2:
            raise ValueError("roi_count must be at least two")
        edge_i, edge_j = torch.triu_indices(int(roi_count), int(roi_count), offset=1)
        self.roi_count = int(roi_count)
        # ``roi_num``/``edge_dim`` are prompt-facing aliases; the runner uses
        # the unambiguous roi_count/stable_dim names internally.
        self.roi_num = self.roi_count
        self.register_buffer("edge_i", edge_i, persistent=False)
        self.register_buffer("edge_j", edge_j, persistent=False)

    @property
    def stable_dim(self) -> int:
        return int(self.edge_i.numel())

    @property
    def edge_dim(self) -> int:
        return self.stable_dim

    def forward(self, stable: torch.Tensor) -> torch.Tensor:
        if stable.ndim != 2 or int(stable.shape[1]) != self.stable_dim:
            raise ValueError(
                f"stable FC shape must be [B,{self.stable_dim}], got {tuple(stable.shape)}"
            )
        profile = stable.new_zeros((int(stable.shape[0]), self.roi_count))
        edge_i = self.edge_i.view(1, -1).expand(int(stable.shape[0]), -1)
        edge_j = self.edge_j.view(1, -1).expand(int(stable.shape[0]), -1)
        profile.scatter_add_(1, edge_i, stable)
        profile.scatter_add_(1, edge_j, stable)
        profile.div_(float(self.roi_count - 1))
        if not bool(torch.isfinite(profile).all()):
            raise FloatingPointError("non-finite ROI connectivity profile")
        return profile


@dataclass(frozen=True)
class NormativeAxisState:
    """Frozen control-referenced normative statistics and one ASD axis."""

    mu: torch.Tensor
    sigma: torch.Tensor
    std_floor: float
    asd_mean: torch.Tensor
    axis: torch.Tensor
    hc_score_mean: float
    hc_score_std: float
    pc1_explained_variance_fraction: float

    # Canonical prompt-facing aliases retained alongside compact internal
    # field names.  The serialized state includes both spellings so selected
    # external models are self-describing and backward-readable.
    @property
    def mu_hc(self) -> torch.Tensor:
        return self.mu

    @property
    def sigma_hc(self) -> torch.Tensor:
        return self.sigma

    @property
    def asd_mean_z(self) -> torch.Tensor:
        return self.asd_mean

    @property
    def pc1_explained_fraction(self) -> float:
        return self.pc1_explained_variance_fraction


def _as_cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().clone()


def fit_normative_heterogeneity_axis(
    stable_reference: torch.Tensor,
    reference_labels: torch.Tensor,
    seed: int,
    roi_count: int = 384,
) -> NormativeAxisState:
    """Fit the locked normative axis using *only* unique training reference.

    HC mean/scale are calculated from reference controls.  The ASD mean is
    used only to center the ASD heterogeneity matrix prior to the one-component
    PCA.  Randomness inside ``torch.pca_lowrank`` is isolated with
    ``fork_rng`` so it cannot alter the SSL trajectory or later fitting.
    """

    profile = ROIConnectivityProfile(int(roi_count)).to(stable_reference.device)
    if stable_reference.ndim != 2 or int(stable_reference.shape[1]) != profile.stable_dim:
        raise ValueError("unexpected stable reference shape")
    labels = reference_labels.to(device=stable_reference.device, dtype=torch.long)
    if int(labels.numel()) != int(stable_reference.shape[0]):
        raise ValueError("reference labels and stable reference size differ")
    hc_mask = labels == 1
    asd_mask = labels == 0
    if int(hc_mask.sum().item()) < 2 or int(asd_mask.sum().item()) < 10:
        raise RuntimeError("SNDE normative fit requires at least two HC and ten ASD reference subjects")

    profiles = profile(stable_reference)
    g_hc = profiles[hc_mask]
    g_asd = profiles[asd_mask]
    mu = g_hc.mean(dim=0)
    raw_std = g_hc.std(dim=0, unbiased=False)
    positive = raw_std[raw_std > 0]
    if int(positive.numel()) == 0:
        raise RuntimeError("SNDE normative fit has no positive HC regional standard deviation")
    std_floor = max(1e-6, 0.1 * float(torch.median(positive).detach().cpu()))
    sigma = torch.clamp(raw_std, min=float(std_floor))
    if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()):
        raise FloatingPointError("non-finite SNDE normative mean or scale")

    z_hc = (g_hc - mu) / sigma
    z_asd = (g_asd - mu) / sigma
    asd_mean = z_asd.mean(dim=0)
    heterogeneity = z_asd - asd_mean
    if not bool(torch.isfinite(heterogeneity).all()):
        raise FloatingPointError("non-finite SNDE ASD heterogeneity matrix")
    heterogeneity_energy = torch.sum(heterogeneity.square())
    if float(heterogeneity_energy.detach().cpu()) <= 0.0:
        raise RuntimeError("SNDE normative fit has zero ASD heterogeneity energy")

    devices: list[int] = []
    if stable_reference.device.type == "cuda":
        devices = [int(stable_reference.device.index or 0)]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if stable_reference.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        _, singular_values, vectors = torch.pca_lowrank(heterogeneity, q=1, center=False, niter=5)
    axis = vectors[:, 0]
    axis_norm = torch.linalg.vector_norm(axis)
    if not bool(torch.isfinite(axis_norm)) or float(axis_norm.detach().cpu()) <= 1e-8:
        raise RuntimeError("SNDE normative fit returned an undefined PCA axis")
    axis = axis / axis_norm
    pivot = int(torch.argmax(axis.abs()).detach().cpu())
    if float(axis[pivot].detach().cpu()) < 0.0:
        axis = -axis
    if not bool(torch.isfinite(axis).all()):
        raise FloatingPointError("non-finite normalized SNDE axis")

    hc_raw_scores = z_hc @ axis
    hc_score_mean = hc_raw_scores.mean()
    hc_score_std = hc_raw_scores.std(unbiased=False)
    if not bool(torch.isfinite(hc_score_mean)) or not bool(torch.isfinite(hc_score_std)):
        raise FloatingPointError("non-finite SNDE HC score normalization")
    if float(hc_score_std.detach().cpu()) <= 1e-6:
        raise RuntimeError("SNDE normative fit has HC score std <= 1e-6")
    fraction = singular_values[0].square() / heterogeneity_energy
    if not bool(torch.isfinite(fraction)):
        raise FloatingPointError("non-finite SNDE PCA variance fraction")
    return NormativeAxisState(
        mu=_as_cpu_float(mu),
        sigma=_as_cpu_float(sigma),
        std_floor=float(std_floor),
        asd_mean=_as_cpu_float(asd_mean),
        axis=_as_cpu_float(axis),
        hc_score_mean=float(hc_score_mean.detach().cpu()),
        hc_score_std=float(hc_score_std.detach().cpu()),
        pc1_explained_variance_fraction=float(fraction.detach().cpu()),
    )


def normative_state_payload(state: NormativeAxisState) -> dict[str, Any]:
    return {
        "mu": _as_cpu_float(state.mu),
        "sigma": _as_cpu_float(state.sigma),
        "std_floor": float(state.std_floor),
        "asd_mean": _as_cpu_float(state.asd_mean),
        "axis": _as_cpu_float(state.axis),
        "hc_score_mean": float(state.hc_score_mean),
        "hc_score_std": float(state.hc_score_std),
        "pc1_explained_variance_fraction": float(state.pc1_explained_variance_fraction),
        "mu_hc": _as_cpu_float(state.mu_hc),
        "sigma_hc": _as_cpu_float(state.sigma_hc),
        "asd_mean_z": _as_cpu_float(state.asd_mean_z),
        "pc1_explained_fraction": float(state.pc1_explained_fraction),
    }


def normative_state_from_payload(payload: dict[str, Any]) -> NormativeAxisState:
    required = {"std_floor", "axis", "hc_score_mean", "hc_score_std"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"normative state payload missing fields: {sorted(missing)}")
    def choose(primary: str, legacy: str) -> Any:
        if primary in payload:
            return payload[primary]
        if legacy in payload:
            return payload[legacy]
        raise ValueError(f"normative state payload missing {primary}/{legacy}")
    return NormativeAxisState(
        mu=_as_cpu_float(choose("mu_hc", "mu")), sigma=_as_cpu_float(choose("sigma_hc", "sigma")),
        std_floor=float(payload["std_floor"]), asd_mean=_as_cpu_float(choose("asd_mean_z", "asd_mean")),
        axis=_as_cpu_float(payload["axis"]), hc_score_mean=float(payload["hc_score_mean"]),
        hc_score_std=float(payload["hc_score_std"]),
        pc1_explained_variance_fraction=float(choose("pc1_explained_fraction", "pc1_explained_variance_fraction")),
    )


class SelectiveNormativeDeviationExperts(nn.Module):
    """Frozen baseline plus two selective regional-deviation residuals.

    ``w_pos``, ``b_pos``, ``w_neg`` and ``b_neg`` are the only trainable
    tensors: exactly 384 + 1 + 384 + 1 = 770 parameters.  At their all-zero
    initialization, the module is exactly the frozen audited baseline.
    """

    def __init__(
        self,
        baseline_state: dict[str, torch.Tensor],
        normative_state: NormativeAxisState,
        roi_count: int = 384,
        gate_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if float(gate_threshold) != 1.0:
            raise ValueError("Experiment 1C fixes the SNDE gate threshold at 1.0")
        self.profile = ROIConnectivityProfile(int(roi_count))
        stable_dim = self.profile.stable_dim
        weight = baseline_state["fc.0.weight"].detach().float().cpu()
        bias = baseline_state["fc.0.bias"].detach().float().cpu()
        if tuple(weight.shape) != (2, stable_dim) or tuple(bias.shape) != (2,):
            raise ValueError("frozen baseline classifier shape does not match stable FC")
        for name, value in (("mu", normative_state.mu), ("sigma", normative_state.sigma),
                            ("asd_mean", normative_state.asd_mean), ("axis", normative_state.axis)):
            if tuple(value.shape) != (int(roi_count),):
                raise ValueError(f"normative {name} must have {roi_count} entries")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"normative {name} is non-finite")
        if float(normative_state.hc_score_std) <= 1e-6:
            raise ValueError("normative HC score std must exceed 1e-6")
        self.register_buffer("baseline_weight", weight.clone())
        self.register_buffer("baseline_bias", bias.clone())
        self.register_buffer("mu", _as_cpu_float(normative_state.mu))
        self.register_buffer("sigma", _as_cpu_float(normative_state.sigma))
        self.register_buffer("asd_mean", _as_cpu_float(normative_state.asd_mean))
        self.register_buffer("axis", _as_cpu_float(normative_state.axis))
        self.register_buffer("hc_score_mean", torch.tensor(float(normative_state.hc_score_mean), dtype=torch.float32))
        self.register_buffer("hc_score_std", torch.tensor(float(normative_state.hc_score_std), dtype=torch.float32))
        self.register_buffer("std_floor", torch.tensor(float(normative_state.std_floor), dtype=torch.float32))
        self.register_buffer("pc1_explained_variance_fraction", torch.tensor(float(normative_state.pc1_explained_variance_fraction), dtype=torch.float32))
        self.register_buffer("gate_threshold", torch.tensor(float(gate_threshold), dtype=torch.float32))
        self.w_pos = nn.Parameter(torch.zeros(int(roi_count), dtype=torch.float32))
        self.b_pos = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.w_neg = nn.Parameter(torch.zeros(int(roi_count), dtype=torch.float32))
        self.b_neg = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def profile_extractor(self) -> ROIConnectivityProfile:
        return self.profile

    @property
    def weight_positive(self) -> nn.Parameter:
        return self.w_pos

    @property
    def weight_negative(self) -> nn.Parameter:
        return self.w_neg

    @property
    def bias_positive(self) -> nn.Parameter:
        return self.b_pos

    @property
    def bias_negative(self) -> nn.Parameter:
        return self.b_neg

    @property
    def mu_hc(self) -> torch.Tensor:
        return self.mu

    @property
    def sigma_hc(self) -> torch.Tensor:
        return self.sigma

    @property
    def pc1_explained_fraction(self) -> torch.Tensor:
        return self.pc1_explained_variance_fraction

    @property
    def stable_dim(self) -> int:
        return self.profile.stable_dim

    @property
    def roi_count(self) -> int:
        return self.profile.roi_count

    def baseline_margin(self, stable: torch.Tensor) -> torch.Tensor:
        logits = F.linear(stable, self.baseline_weight, self.baseline_bias)
        return logits[:, 0] - logits[:, 1]

    def forward_details(self, stable: torch.Tensor) -> dict[str, torch.Tensor]:
        profile = self.profile(stable)
        z = (profile - self.mu) / self.sigma
        raw_score = z @ self.axis
        score = (raw_score - self.hc_score_mean) / self.hc_score_std
        pos_excess = F.relu(score - self.gate_threshold)
        neg_excess = F.relu(-score - self.gate_threshold)
        pos_linear = z @ self.w_pos + self.b_pos
        neg_linear = z @ self.w_neg + self.b_neg
        pos_correction = pos_excess * pos_linear
        neg_correction = neg_excess * neg_linear
        correction = pos_correction + neg_correction
        m0 = self.baseline_margin(stable)
        margin = m0 + correction
        p_asd = torch.sigmoid(margin)
        p_hc = torch.sigmoid(-margin)
        return {
            "profile": profile, "z": z, "raw_score": raw_score, "score": score,
            "pos_excess": pos_excess, "neg_excess": neg_excess,
            "pos_linear": pos_linear, "neg_linear": neg_linear,
            "pos_correction": pos_correction, "neg_correction": neg_correction,
            "correction": correction, "m0": m0, "margin": margin,
            "p_asd": p_asd, "p_hc": p_hc,
        }

    def forward(self, stable: torch.Tensor) -> torch.Tensor:
        details = self.forward_details(stable)
        return torch.stack((details["p_asd"], details["p_hc"]), dim=1)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def state_payload(model: SelectiveNormativeDeviationExperts) -> dict[str, torch.Tensor]:
    return {
        "w_pos": model.w_pos.detach().cpu().clone(),
        "b_pos": model.b_pos.detach().cpu().clone(),
        "w_neg": model.w_neg.detach().cpu().clone(),
        "b_neg": model.b_neg.detach().cpu().clone(),
    }


def load_residual_state(model: SelectiveNormativeDeviationExperts, state: dict[str, Any]) -> None:
    required = ("w_pos", "b_pos", "w_neg", "b_neg")
    if any(name not in state for name in required):
        raise ValueError("SNDE residual state is incomplete")
    with torch.no_grad():
        model.w_pos.copy_(state["w_pos"].to(model.w_pos.device))
        model.b_pos.copy_(state["b_pos"].to(model.b_pos.device))
        model.w_neg.copy_(state["w_neg"].to(model.w_neg.device))
        model.b_neg.copy_(state["b_neg"].to(model.b_neg.device))


def is_exact_zero_state(state: dict[str, Any]) -> bool:
    return all(torch.equal(state[name], torch.zeros_like(state[name])) for name in ("w_pos", "b_pos", "w_neg", "b_neg"))


def binary_bce_from_details(details: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    target_hc = (labels.to(device=details["p_hc"].device, dtype=torch.long) == 1).float()
    return F.binary_cross_entropy(details["p_hc"], target_hc)


def snde_diagnostics(
    model: SelectiveNormativeDeviationExperts,
    stable: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Reference-only class-stratified diagnostics for the selected SNDE state."""

    model.eval()
    with torch.no_grad():
        details = model.forward_details(stable)
        labels = labels.to(device=stable.device, dtype=torch.long)
        output: dict[str, float] = {}
        for name, mask in (("asd", labels == 0), ("hc", labels == 1)):
            if int(mask.sum().item()) == 0:
                raise RuntimeError("SNDE diagnostics require both reference classes")
            scores = details["score"][mask]
            positive = scores > 1.0
            negative = scores < -1.0
            generalist = torch.abs(scores) <= 1.0
            fractions = (
                float(positive.float().mean().cpu()),
                float(negative.float().mean().cpu()),
                float(generalist.float().mean().cpu()),
            )
            if abs(sum(fractions) - 1.0) > 1e-6:
                raise RuntimeError(f"SNDE gate fractions do not sum to one for {name}")
            output.update({
                f"snde_{name}_score_mean": float(scores.mean().cpu()),
                f"snde_{name}_score_std": float(scores.std(unbiased=False).cpu()),
                f"snde_{name}_positive_fraction": fractions[0],
                f"snde_{name}_negative_fraction": fractions[1],
                f"snde_{name}_generalist_fraction": fractions[2],
                f"snde_{name}_mean_abs_correction": float(details["correction"][mask].abs().mean().cpu()),
            })
        parameter_l2 = torch.sqrt(
            model.w_pos.square().sum() + model.w_neg.square().sum() + model.b_pos.square() + model.b_neg.square()
        )
        output.update({
            "snde_w_pos_l2": float(torch.linalg.vector_norm(model.w_pos).cpu()),
            "snde_w_neg_l2": float(torch.linalg.vector_norm(model.w_neg).cpu()),
            "snde_positive_weight_l2": float(torch.linalg.vector_norm(model.weight_positive).cpu()),
            "snde_negative_weight_l2": float(torch.linalg.vector_norm(model.weight_negative).cpu()),
            "snde_b_pos": float(model.b_pos.detach().cpu()),
            "snde_b_neg": float(model.b_neg.detach().cpu()),
            "snde_parameter_l2": float(parameter_l2.cpu()),
            "snde_mean_abs_correction": float(details["correction"].abs().mean().cpu()),
            "snde_mean_abs_positive_correction": float(details["pos_correction"].abs().mean().cpu()),
            "snde_mean_abs_negative_correction": float(details["neg_correction"].abs().mean().cpu()),
            "snde_pc1_explained_variance_fraction": float(model.pc1_explained_variance_fraction.cpu()),
            "snde_pc1_explained_fraction": float(model.pc1_explained_fraction.cpu()),
            "snde_std_floor": float(model.std_floor.cpu()),
            "snde_hc_reference_score_mean": float(model.hc_score_mean.cpu()),
            "snde_hc_reference_score_std": float(model.hc_score_std.cpu()),
        })
    return output

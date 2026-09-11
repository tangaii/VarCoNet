"""Online class-conditional subtype prototypes for Experiment 1.

The bank is training-only state.  Prototypes are buffers maintained with an
EMA and are deliberately detached from the encoder optimizer.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class OnlineClassSubtypeBank:
    def __init__(
        self,
        num_classes=2,
        num_prototypes=3,
        momentum=0.99,
        sinkhorn_epsilon=0.05,
        sinkhorn_iters=3,
    ):
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.momentum = momentum
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters

        self.prototypes = None
        self.initialized = None

    @torch.no_grad()
    def _ensure_storage(self, dim, device, dtype):
        if self.prototypes is None:
            self.prototypes = torch.zeros(
                self.num_classes,
                self.num_prototypes,
                dim,
                device=device,
                dtype=dtype,
            )
            self.initialized = torch.zeros(
                self.num_classes,
                dtype=torch.bool,
                device=device,
            )

    @torch.no_grad()
    def _farthest_point_init(self, feats):
        """Deterministically initialize prototypes from current batch feats."""
        n, _ = feats.shape
        if n == 0:
            raise RuntimeError("Cannot initialize prototype from empty class.")

        centroid = F.normalize(feats.mean(dim=0, keepdim=True), p=2, dim=1)
        first = torch.argmax(feats @ centroid.T).item()
        selected = [first]

        while len(selected) < self.num_prototypes:
            if n <= len(selected):
                selected.append(selected[len(selected) % n])
                continue

            selected_feats = feats[selected]
            max_sim = (feats @ selected_feats.T).max(dim=1).values
            max_sim[selected] = float("inf")
            selected.append(torch.argmin(max_sim).item())

        return F.normalize(feats[selected], p=2, dim=1)

    @torch.no_grad()
    def _sinkhorn(self, logits):
        """SACSSL-style approximately balanced online assignment."""
        n, k = logits.shape
        if n < k:
            return torch.softmax(logits / self.sinkhorn_epsilon, dim=1)

        scaled = logits / self.sinkhorn_epsilon
        scaled = scaled - scaled.max()
        q = torch.exp(scaled).T
        q = q / q.sum().clamp_min(1e-12)

        for _ in range(self.sinkhorn_iters):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q / k
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / n

        return (q * n).T

    @torch.no_grad()
    def assign(self, zbar, labels):
        """Return one subtype ID per detached normalized subject representation."""
        zbar = F.normalize(zbar, p=2, dim=1)
        self._ensure_storage(zbar.shape[1], zbar.device, zbar.dtype)
        subtype_ids = torch.full(
            (zbar.shape[0],), -1, dtype=torch.long, device=zbar.device
        )

        for c in range(self.num_classes):
            idx = torch.where(labels == c)[0]
            if idx.numel() == 0:
                continue

            feats = zbar[idx]
            if not bool(self.initialized[c]):
                self.prototypes[c] = self._farthest_point_init(feats)
                self.initialized[c] = True

            proto = F.normalize(self.prototypes[c], p=2, dim=1)
            probs = self._sinkhorn(feats @ proto.T)
            subtype_ids[idx] = torch.argmax(probs, dim=1)

        if torch.any(subtype_ids < 0):
            raise RuntimeError("Some samples did not receive subtype IDs.")
        return subtype_ids

    @torch.no_grad()
    def update(self, zbar, labels, subtype_ids):
        """Momentum update using only the current training batch."""
        zbar = F.normalize(zbar, p=2, dim=1)
        for c in range(self.num_classes):
            for k in range(self.num_prototypes):
                idx = torch.where((labels == c) & (subtype_ids == k))[0]
                if idx.numel() == 0:
                    continue

                center = F.normalize(
                    zbar[idx].mean(dim=0, keepdim=True), p=2, dim=1
                )[0]
                old = self.prototypes[c, k]
                new = self.momentum * old + (1.0 - self.momentum) * center
                self.prototypes[c, k] = F.normalize(
                    new.unsqueeze(0), p=2, dim=1
                )[0]


class SubtypePrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def _one_view(self, z, labels, subtype_ids, prototypes):
        """Own subtype prototype is positive; opposite-class prototypes are negatives."""
        z = F.normalize(z, p=2, dim=1)
        p = F.normalize(prototypes.detach(), p=2, dim=-1)
        c, k, d = p.shape
        flat_p = p.reshape(c * k, d)
        logits = (z @ flat_p.T) / self.temperature
        proto_class = torch.arange(c, device=z.device).repeat_interleave(k)
        target = labels.long() * k + subtype_ids.long()

        valid = proto_class.unsqueeze(0) != labels.unsqueeze(1)
        valid[
            torch.arange(z.shape[0], device=z.device), target
        ] = True
        logits = logits.masked_fill(~valid, -1e9)
        return F.cross_entropy(logits, target)

    def forward(self, z1, z2, labels, subtype_ids, prototypes):
        return 0.5 * (
            self._one_view(z1, labels, subtype_ids, prototypes)
            + self._one_view(z2, labels, subtype_ids, prototypes)
        )

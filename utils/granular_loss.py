import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .granular import GBList


_EPS = 1e-12


def _sinkhorn_log(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.10,
    iters: int = 30,
) -> torch.Tensor:
    """Entropic OT with unequal source/target cardinalities in log space."""
    eps = max(float(epsilon), 1e-4)
    a = a.clamp_min(_EPS)
    b = b.clamp_min(_EPS)
    a = a / a.sum()
    b = b / b.sum()

    log_k = -cost / eps
    log_a = torch.log(a)
    log_b = torch.log(b)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max(1, int(iters))):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)

    log_p = log_k + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return torch.exp(log_p)


@torch.no_grad()
def _overlap_matrix(view1: GBList, view2: GBList, device, dtype):
    """Sample-overlap prior between two *different* dynamic partitions."""
    out = torch.zeros((len(view1), len(view2)), device=device, dtype=dtype)
    for i, ball_i in enumerate(view1.granular_balls):
        ids_i = ball_i.indices
        ni = max(1, len(ids_i))
        for j, ball_j in enumerate(view2.granular_balls):
            ids_j = ball_j.indices
            nj = max(1, len(ids_j))
            inter = np.intersect1d(ids_i, ids_j, assume_unique=False).size
            if inter:
                out[i, j] = float(inter) / math.sqrt(float(ni * nj))
    return out


def _normalized_entropy(prob: torch.Tensor, dim: int = -1):
    n = prob.shape[dim]
    if n <= 1:
        return torch.zeros(prob.shape[:dim] + prob.shape[dim + 1:], device=prob.device, dtype=prob.dtype)
    p = prob.clamp_min(_EPS)
    ent = -(p * p.log()).sum(dim=dim)
    return ent / math.log(float(n))


def _sparsify_positive_weights(
    row_prob: torch.Tensor,
    target_conf: torch.Tensor,
    topk: int,
    rel_threshold: float,
    positive_mode: str = "multi",
) -> torch.Tensor:
    if positive_mode == "single":
        best = row_prob.argmax(dim=1, keepdim=True)
        weights = torch.zeros_like(row_prob)
        weights.scatter_(1, best, 1.0)
        return weights

    if positive_mode != "multi":
        raise ValueError(f"Unknown positive_mode: {positive_mode}")

    n_target = row_prob.shape[1]
    k = min(max(1, int(topk)), n_target)

    _, top_indices = torch.topk(row_prob, k=k, dim=1)
    keep = torch.zeros_like(row_prob, dtype=torch.bool)
    keep.scatter_(1, top_indices, True)

    row_max = row_prob.max(dim=1, keepdim=True).values
    keep = keep | (row_prob >= float(rel_threshold) * row_max)

    weights = row_prob * keep.to(row_prob.dtype) * target_conf.unsqueeze(0)
    row_sum = weights.sum(dim=1, keepdim=True)

    fallback = torch.zeros_like(weights)
    best = row_prob.argmax(dim=1, keepdim=True)
    fallback.scatter_(1, best, 1.0)
    weights = torch.where(row_sum > _EPS, weights, fallback)

    return weights / weights.sum(dim=1, keepdim=True).clamp_min(_EPS)

def _weighted_multi_positive_direction(
    source_centers: torch.Tensor,
    target_centers: torch.Tensor,
    source_conf: torch.Tensor,
    target_conf: torch.Tensor,
    transport: torch.Tensor,
    temperature: float,
    topk: int,
    rel_threshold: float,
    positive_mode: str = "multi",
    use_uncertainty: bool = True,
):
    src = F.normalize(source_centers, dim=1)
    tgt = F.normalize(target_centers, dim=1)
    logits = src @ tgt.T / max(float(temperature), 1e-4)

    row_prob = transport / transport.sum(
        dim=1, keepdim=True
    ).clamp_min(_EPS)

    effective_target_conf = (
        target_conf if use_uncertainty else torch.ones_like(target_conf)
    )

    positive_weights = _sparsify_positive_weights(
        row_prob,
        effective_target_conf,
        topk=topk,
        rel_threshold=rel_threshold,
        positive_mode=positive_mode,
    )

    log_denom = torch.logsumexp(logits, dim=1)
    log_positive_weights = torch.log(positive_weights.clamp_min(_EPS))
    log_num = torch.logsumexp(logits + log_positive_weights, dim=1)
    per_anchor = -(log_num - log_denom)

    match_entropy = _normalized_entropy(row_prob, dim=1)

    if use_uncertainty:
        match_conf = (1.0 - match_entropy).clamp(0.05, 1.0)
        expected_target_conf = (
            row_prob * target_conf.unsqueeze(0)
        ).sum(dim=1).clamp(0.05, 1.0)
        anchor_weight = (
            source_conf * match_conf * expected_target_conf
        ).detach()
    else:
        anchor_weight = torch.ones_like(per_anchor)

    loss = (
        anchor_weight * per_anchor
    ).sum() / anchor_weight.sum().clamp_min(_EPS)

    return loss, match_entropy.mean().detach()

@torch.no_grad()
def _pair_hard_correspondence(view1: GBList, view2: GBList):
    c1 = view1.get_centers()
    overlap = _overlap_matrix(
        view1,
        view2,
        device=c1.device,
        dtype=c1.dtype,
    )
    best = overlap.argmax(dim=1, keepdim=True)
    transport = torch.zeros_like(overlap)
    transport.scatter_(1, best, 1.0)
    return transport

def _pair_soft_correspondence(view1: GBList, view2: GBList, args):
    c1 = view1.get_centers()
    c2 = view2.get_centers()
    device, dtype = c1.device, c1.dtype

    z1 = F.normalize(c1, dim=1)
    z2 = F.normalize(c2, dim=1)
    semantic_cost = 1.0 - z1 @ z2.T

    overlap = _overlap_matrix(view1, view2, device=device, dtype=dtype)
    overlap_cost = 1.0 - overlap

    r1 = view1.get_rs().to(device=device, dtype=dtype)
    r2 = view2.get_rs().to(device=device, dtype=dtype)
    radius_cost = torch.abs(
        torch.log((r1.unsqueeze(1) + 1e-6) / (r2.unsqueeze(0) + 1e-6))
    ).clamp_max(3.0) / 3.0

    semantic_w = float(getattr(args, "ot_semantic_weight", 0.55))
    overlap_w = float(getattr(args, "ot_overlap_weight", 0.40))
    radius_w = max(0.0, 1.0 - semantic_w - overlap_w)
    cost = semantic_w * semantic_cost + overlap_w * overlap_cost + radius_w * radius_cost

    masses1 = view1.get_sizes().to(dtype=dtype)
    masses2 = view2.get_sizes().to(dtype=dtype)
    a = masses1 / masses1.sum().clamp_min(_EPS)
    b = masses2 / masses2.sum().clamp_min(_EPS)

    # The discrete GB memberships are not differentiable.  We use OT as a stable
    # soft correspondence estimator and backpropagate through the contrastive logits.
    with torch.no_grad():
        transport = _sinkhorn_log(
            cost.detach(),
            a.detach(),
            b.detach(),
            epsilon=float(getattr(args, "ot_epsilon", 0.10)),
            iters=int(getattr(args, "ot_iters", 30)),
        )
    return transport


class GranularPrototypeMemory:
    """Small temporal prototype memory used to suppress granular-ball drift."""

    def __init__(self, num_views: int, capacity: int = 32, momentum: float = 0.95):
        self.num_views = int(num_views)
        self.capacity = max(1, int(capacity))
        self.momentum = float(momentum)
        self.centers = [None for _ in range(self.num_views)]
        self.masses = [None for _ in range(self.num_views)]

    def reset(self):
        self.centers = [None for _ in range(self.num_views)]
        self.masses = [None for _ in range(self.num_views)]

    def _select_seed(self, centers, confidence):
        n = centers.shape[0]
        k = min(self.capacity, n)
        ids = torch.topk(confidence, k=k).indices
        return F.normalize(centers.detach()[ids], dim=1), confidence.detach()[ids].clamp_min(0.05)

    def loss_and_update(
        self,
        view_id: int,
        gblist: GBList,
        epsilon: float,
        iters: int,
        apply_loss: bool,
    ):
        current = F.normalize(gblist.get_centers(), dim=1)
        confidence = gblist.get_confidences().to(current.device, current.dtype)

        memory = self.centers[view_id]
        memory_mass = self.masses[view_id]
        if memory is None:
            seed, seed_mass = self._select_seed(current, confidence)
            self.centers[view_id] = seed
            self.masses[view_id] = seed_mass
            return current.sum() * 0.0

        memory = memory.to(current.device, current.dtype)
        memory_mass = memory_mass.to(current.device, current.dtype)
        cost = 1.0 - current @ memory.T

        a = gblist.get_sizes().to(current.dtype)
        a = a / a.sum().clamp_min(_EPS)
        b = memory_mass.clamp_min(0.05)
        b = b / b.sum().clamp_min(_EPS)

        with torch.no_grad():
            transport = _sinkhorn_log(cost.detach(), a.detach(), b.detach(), epsilon=epsilon, iters=iters)

        row_cost = (transport * cost).sum(dim=1) / transport.sum(dim=1).clamp_min(_EPS)
        drift_loss = (confidence * row_cost).sum() / confidence.sum().clamp_min(_EPS)

        with torch.no_grad():
            col_mass = transport.sum(dim=0).clamp_min(_EPS)
            target = transport.T @ current.detach()
            target = F.normalize(target / col_mass.unsqueeze(1), dim=1)
            updated = F.normalize(
                self.momentum * memory + (1.0 - self.momentum) * target,
                dim=1,
            )
            updated_mass = self.momentum * memory_mass + (1.0 - self.momentum) * col_mass
            self.centers[view_id] = updated.detach()
            self.masses[view_id] = updated_mass.detach()

            # Fill unused capacity with the most novel current GBs.
            free = self.capacity - updated.shape[0]
            if free > 0 and current.shape[0] > 0:
                novelty = (1.0 - current.detach() @ updated.T).min(dim=1).values
                k = min(free, current.shape[0])
                ids = torch.topk(novelty, k=k).indices
                self.centers[view_id] = torch.cat(
                    [self.centers[view_id], current.detach()[ids]], dim=0
                )
                self.masses[view_id] = torch.cat(
                    [self.masses[view_id], confidence.detach()[ids].clamp_min(0.05)], dim=0
                )

        return drift_loss if apply_loss else drift_loss.detach() * 0.0


class MultiviewGCLoss(torch.nn.Module):
    """OT-matched, uncertainty-aware weighted multi-positive GB contrastive loss."""

    def __init__(self, args, temperature: Optional[float] = None):
        super().__init__()
        self.args = args
        self.t = float(temperature if temperature is not None else getattr(args, "temperature", 0.2))
        self.current_epoch = 0
        self._last_epoch = None
        self.memory = GranularPrototypeMemory(
            num_views=int(getattr(args, "view", 2)),
            capacity=int(getattr(args, "gb_memory_size", 32)),
            momentum=float(getattr(args, "gb_memory_momentum", 0.95)),
        )
        self.last_stats: Dict[str, float] = {}

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        warmup = int(getattr(self.args, "gb_drift_warmup", 5))
        # Reset once at warmup so random early representations are not frozen in memory.
        if self._last_epoch is not None and self._last_epoch < warmup <= epoch:
            self.memory.reset()
        self.current_epoch = epoch
        self._last_epoch = epoch

    def _pair_loss(self, view1: GBList, view2: GBList):
        matching = str(getattr(self.args, "matching", "ot"))

        if matching == "hard":
            transport12 = _pair_hard_correspondence(view1, view2)
            transport21 = _pair_hard_correspondence(view2, view1)
        elif matching == "ot":
            transport12 = _pair_soft_correspondence(
                view1, view2, self.args
            )
            transport21 = transport12.T
        else:
            raise ValueError(f"Unknown matching mode: {matching}")

        conf1 = view1.get_confidences().to(
            view1.data.device, view1.data.dtype
        )
        conf2 = view2.get_confidences().to(
            view2.data.device, view2.data.dtype
        )

        kwargs = dict(
            temperature=self.t,
            topk=int(getattr(self.args, "gb_positive_topk", 3)),
            rel_threshold=float(
                getattr(self.args, "gb_positive_rel_threshold", 0.25)
            ),
            positive_mode=str(
                getattr(self.args, "positive_mode", "multi")
            ),
            use_uncertainty=bool(
                int(getattr(self.args, "use_uncertainty", 1))
            ),
        )

        l12, h12 = _weighted_multi_positive_direction(
            view1.get_centers(),
            view2.get_centers(),
            conf1,
            conf2,
            transport12,
            **kwargs,
        )
        l21, h21 = _weighted_multi_positive_direction(
            view2.get_centers(),
            view1.get_centers(),
            conf2,
            conf1,
            transport21,
            **kwargs,
        )
        return 0.5 * (l12 + l21), 0.5 * (h12 + h21)

    def forward(self, views, num_views=None, k=None, batch_size=None, mode=0):
        del k, batch_size, mode  # retained only for compatibility with the original call signature
        num_views = len(views)
        device = views[0].data.device

        pair_losses = []
        entropies = []
        for i in range(num_views):
            for j in range(i + 1, num_views):
                pair_loss, entropy = self._pair_loss(views[i], views[j])
                pair_losses.append(pair_loss)
                entropies.append(entropy)

        contrastive = (
            torch.stack(pair_losses).mean()
            if pair_losses
            else torch.zeros((), device=device, requires_grad=True)
        )

        drift_weight = float(
            getattr(self.args, "gb_drift_weight", 0.05)
        )

        if drift_weight > 0.0:
            drift_losses = []
            warmup = int(getattr(self.args, "gb_drift_warmup", 5))
            apply_drift = self.current_epoch >= warmup

            for v in range(num_views):
                drift_losses.append(
                    self.memory.loss_and_update(
                        v,
                        views[v],
                        epsilon=float(getattr(self.args, "ot_epsilon", 0.10)),
                        iters=int(getattr(self.args, "ot_iters", 30)),
                        apply_loss=apply_drift,
                    )
                )

            drift = (
                torch.stack(drift_losses).mean()
                if drift_losses
                else contrastive * 0.0
            )
        else:
            # Strict w/o-Drift: do not update temporal memory.
            drift = contrastive * 0.0

        total = contrastive + drift_weight * drift
        self.last_stats = {
            "gb_count": float(sum(len(v) for v in views) / max(1, num_views)),
            "ot_entropy": float(torch.stack(entropies).mean().cpu()) if entropies else 0.0,
            "contrastive": float(contrastive.detach().cpu()),
            "drift": float(drift.detach().cpu()),
        }
        return total

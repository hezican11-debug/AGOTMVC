import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans


_EPS = 1e-12


class GranularBall:
    """A differentiable granular-ball container with non-differentiable adaptive partitioning.

    ``data`` keeps the current representation tensor so that ball centers retain gradients.
    ``partition_data`` is detached and is used only to decide the discrete partition.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor],
        indices: Sequence[int],
        partition_data: Optional[torch.Tensor] = None,
    ):
        if data.ndim != 2:
            raise ValueError(f"GranularBall expects [N, D] data, got {tuple(data.shape)}")
        self.data = data
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.partition_data = data.detach() if partition_data is None else partition_data.detach()

        self.num_smp, self.dim = data.shape
        if self.num_smp == 0:
            raise ValueError("A granular ball cannot be empty.")

        # Differentiable center used by the contrastive objective.
        self.center = self.data.mean(dim=0)

        # Detached geometric statistics used by generation/uncertainty.
        p_center = self.partition_data.mean(dim=0)
        distances = torch.norm(self.partition_data - p_center, p=2, dim=1)
        self.r = distances.mean()
        self.r_max = distances.max() if distances.numel() else torch.zeros((), device=data.device)
        self.dispersion = torch.mean(distances * distances)
        self.radius_std = distances.std(unbiased=False) if distances.numel() > 1 else torch.zeros_like(self.r)

    @torch.no_grad()
    def _binary_division(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """One deterministic division step (no iterative k-means).

        We use two far-apart representatives and assign points once.  If the split is
        degenerate, a median split on the highest-variance dimension is used.
        """
        n = self.num_smp
        if n < 2:
            return None

        x = self.partition_data
        center = x.mean(dim=0, keepdim=True)
        first = torch.argmax(torch.norm(x - center, dim=1))
        second = torch.argmax(torch.norm(x - x[first:first + 1], dim=1))

        if int(first) == int(second):
            return self._median_fallback()

        d_first = torch.norm(x - x[first:first + 1], dim=1)
        d_second = torch.norm(x - x[second:second + 1], dim=1)
        right = d_second < d_first
        left = ~right

        if int(left.sum()) == 0 or int(right.sum()) == 0:
            return self._median_fallback()
        return torch.where(left)[0], torch.where(right)[0]

    @torch.no_grad()
    def _median_fallback(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        x = self.partition_data
        if self.num_smp < 2:
            return None
        variances = torch.var(x, dim=0, unbiased=False)
        dim = int(torch.argmax(variances))
        order = torch.argsort(x[:, dim])
        cut = self.num_smp // 2
        if cut <= 0 or cut >= self.num_smp:
            return None
        return order[:cut], order[cut:]

    def _child(self, local_ids: torch.Tensor) -> "GranularBall":
        local_ids = local_ids.to(self.data.device)
        sub_data = self.data.index_select(0, local_ids)
        sub_partition = self.partition_data.index_select(0, local_ids)
        sub_labels = None
        if self.labels is not None:
            sub_labels = self.labels.index_select(0, local_ids.to(self.labels.device))
        sub_indices = self.indices[local_ids.detach().cpu().numpy()]
        return GranularBall(sub_data, sub_labels, sub_indices, sub_partition)

    def propose_split(
        self,
        min_samples: int,
        min_gain: float,
        min_separation: float,
        global_radius: float,
    ):
        """Return an accepted adaptive binary split and a score, or ``None``.

        Since GFSAF is unsupervised, ground-truth labels are *not* used as purity
        supervision.  The adaptive quality criterion is therefore label-free:
        compactness gain + child separation + a global-to-local scale correction.
        """
        if self.num_smp < max(2, 2 * int(min_samples)):
            return None

        split = self._binary_division()
        if split is None:
            return None
        left_ids, right_ids = split
        if left_ids.numel() < min_samples or right_ids.numel() < min_samples:
            # Farthest-point division can be imbalanced. Median fallback guarantees support.
            split = self._median_fallback()
            if split is None:
                return None
            left_ids, right_ids = split
            if left_ids.numel() < min_samples or right_ids.numel() < min_samples:
                return None

        left = self._child(left_ids)
        right = self._child(right_ids)

        parent_radius = float(self.r.detach().cpu())
        if parent_radius <= _EPS:
            return None

        child_radius = (
            left.num_smp * float(left.r.detach().cpu())
            + right.num_smp * float(right.r.detach().cpu())
        ) / float(self.num_smp)
        compactness_gain = max(0.0, (parent_radius - child_radius) / (parent_radius + _EPS))

        p_left = left.partition_data.mean(dim=0)
        p_right = right.partition_data.mean(dim=0)
        center_distance = float(torch.norm(p_left - p_right).detach().cpu())
        separation = center_distance / (
            float(left.r.detach().cpu()) + float(right.r.detach().cpu()) + _EPS
        )

        # Coarser balls get a slightly easier splitting threshold (global precedence).
        scale_ratio = parent_radius / (float(global_radius) + _EPS)
        adaptive_gain_threshold = float(min_gain) * max(0.55, min(1.0, scale_ratio))
        accepted = compactness_gain >= adaptive_gain_threshold and separation >= min_separation
        if not accepted:
            return None

        # Prioritize large, clearly separable, compactness-improving balls.
        score = compactness_gain * (1.0 + math.log1p(self.num_smp)) * (1.0 + min(separation, 3.0))
        return left, right, score


class GBList:
    """Dynamic granular-ball set generated without k-means."""

    def __init__(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor],
        p: int = 8,
        max_balls: int = 64,
        min_gain: float = 0.08,
        min_separation: float = 0.20,
    ):
        self.data = data
        self.labels = labels
        self.indices = np.arange(data.shape[0], dtype=np.int64)
        self.y_parts = None
        self.min_samples = max(1, int(p))
        self.max_balls = max(1, int(max_balls))
        self.min_gain = float(min_gain)
        self.min_separation = float(min_separation)

        root = GranularBall(data, labels, self.indices)
        self.granular_balls: List[GranularBall] = [root]
        self.split_granular_balls()

    def __len__(self):
        return len(self.granular_balls)

    def __getitem__(self, i):
        return self.granular_balls[i]

    def split_granular_balls(self):
        """Adaptive coarse-to-fine generation with a global-best split schedule."""
        global_radius = max(float(self.granular_balls[0].r.detach().cpu()), _EPS)

        while len(self.granular_balls) < self.max_balls:
            best = None
            best_idx = None
            for idx, ball in enumerate(self.granular_balls):
                proposal = ball.propose_split(
                    min_samples=self.min_samples,
                    min_gain=self.min_gain,
                    min_separation=self.min_separation,
                    global_radius=global_radius,
                )
                if proposal is None:
                    continue
                left, right, score = proposal
                if best is None or score > best[2]:
                    best = (left, right, score)
                    best_idx = idx

            if best is None:
                break
            left, right, _ = best
            self.granular_balls = (
                self.granular_balls[:best_idx]
                + [left, right]
                + self.granular_balls[best_idx + 1:]
            )

        self._refresh_assignments()

    def _refresh_assignments(self):
        device = self.data.device
        parts = torch.full((self.data.shape[0],), -1, dtype=torch.long, device=device)
        for ball_id, ball in enumerate(self.granular_balls):
            ids = torch.as_tensor(ball.indices, dtype=torch.long, device=device)
            parts[ids] = ball_id
        self.y_parts = parts

    def get_centers(self):
        return torch.vstack([ball.center for ball in self.granular_balls])

    def get_rs(self):
        return torch.stack([ball.r.to(self.data.device) for ball in self.granular_balls]).reshape(-1)

    def get_radius_std(self):
        return torch.stack([ball.radius_std.to(self.data.device) for ball in self.granular_balls]).reshape(-1)

    def get_sizes(self):
        return torch.tensor(
            [ball.num_smp for ball in self.granular_balls],
            dtype=self.data.dtype,
            device=self.data.device,
        )

    @torch.no_grad()
    def get_confidences(self):
        """Geometry/support confidence used by uncertainty-aware soft matching."""
        radii = self.get_rs().float()
        radius_std = self.get_radius_std().float()
        sizes = self.get_sizes().float()

        positive = radii[radii > _EPS]
        median_radius = positive.median() if positive.numel() else torch.tensor(1.0, device=radii.device)
        median_radius = median_radius.clamp_min(_EPS)

        compact = torch.exp(-radii / median_radius)
        stability = 1.0 / (1.0 + radius_std / (radii + median_radius * 0.1 + _EPS))
        support = torch.sqrt(sizes / sizes.max().clamp_min(1.0))
        confidence = compact * stability * support
        return confidence.clamp(0.05, 1.0).to(self.data.dtype)

    @torch.no_grad()
    def affinity(self, spread=3):
        centers = self.get_centers().detach()
        dist = torch.cdist(centers, centers)
        rs = self.get_rs()
        extra = rs.unsqueeze(0) + rs.unsqueeze(-1)
        return (dist <= extra).to(torch.float32)

    def get_data(self):
        list_data = [ball.data for ball in self.granular_balls]
        if self.labels is None:
            list_labels = None
        else:
            list_labels = [ball.labels for ball in self.granular_balls]
        list_indices = [torch.as_tensor(ball.indices, device=self.data.device) for ball in self.granular_balls]
        labels = None if list_labels is None else torch.concat(list_labels, dim=0)
        return torch.concat(list_data, dim=0), labels, torch.concat(list_indices, dim=0)

    def del_ball(self, min_smp=0):
        kept = [ball for ball in self.granular_balls if ball.num_smp >= min_smp]
        if kept:
            self.granular_balls = kept
            self._refresh_assignments()


class KMeansGBList(GBList):
    """KMeans partition used only for the w/o-AGB ablation."""

    def __init__(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor],
        n_clusters: int,
        random_state: int = 0,
    ):
        self.data = data
        self.labels = labels
        self.indices = np.arange(data.shape[0], dtype=np.int64)
        self.y_parts = None

        k = max(1, min(int(n_clusters), int(data.shape[0])))

        assignments = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=int(random_state),
        ).fit_predict(data.detach().cpu().numpy())

        self.granular_balls: List[GranularBall] = []

        for cluster_id in range(k):
            ids_np = np.where(assignments == cluster_id)[0]
            if ids_np.size == 0:
                continue

            ids_tensor = torch.as_tensor(
                ids_np,
                dtype=torch.long,
                device=data.device,
            )
            sub_data = data.index_select(0, ids_tensor)

            sub_labels = None
            if labels is not None:
                sub_labels = labels.index_select(
                    0,
                    ids_tensor.to(labels.device),
                )

            self.granular_balls.append(
                GranularBall(sub_data, sub_labels, ids_np)
            )

        if not self.granular_balls:
            self.granular_balls = [
                GranularBall(data, labels, self.indices)
            ]

        self._refresh_assignments()


class MVGBList:
    """View-specific dynamic granular-ball sets.

    Different views are intentionally allowed to produce different numbers and
    memberships of granular balls; OT in ``granular_loss.py`` resolves this mismatch.
    """

    def __init__(
        self,
        mv_data,
        labels,
        p=8,
        max_balls=64,
        min_gain=0.08,
        min_separation=0.20,
        generation="adaptive",
        kmeans_k=None,
    ):
        self.num_view = len(mv_data)
        self.gblists = []

        for i in range(self.num_view):
            if generation == "kmeans":
                if kmeans_k is None:
                    raise ValueError("kmeans_k is required for KMeans ablation")
                current = KMeansGBList(
                    mv_data[i],
                    labels,
                    n_clusters=kmeans_k,
                    random_state=0,
                )
            elif generation == "adaptive":
                current = GBList(
                    mv_data[i],
                    labels,
                    p=p,
                    max_balls=max_balls,
                    min_gain=min_gain,
                    min_separation=min_separation,
                )
            else:
                raise ValueError(f"Unknown GB generation mode: {generation}")

            self.gblists.append(current)

    def __len__(self):
        return self.num_view

    def __getitem__(self, i):
        return self.gblists[i]


def contain_same_sample(ball0: GranularBall, ball1: GranularBall):
    return np.intersect1d(ball0.indices, ball1.indices, assume_unique=False).size > 0


def transitive_neighbor_relations(a, k=3):
    while k > 0:
        a_ = torch.where(a @ a > 0, 1.0, 0.0)
        a_ = torch.where(torch.logical_or(a.bool(), a_.bool()), 1.0, 0.0)
        a = a_
        k -= 1
    return a

"""Trainable UPAR adapter layers for frozen Hyper3-CLIP embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import exp_map0, lorentz_distance


class Hyper3AttributeQueryAdapter(nn.Module):
    """Map a 40-bit UPAR query into Hyper3-CLIP's 513-D Lorentz space."""

    def __init__(self, attr_dim: int = 40, hidden_dim: int = 512, tangent_dim: int = 512, curvature: float = 1.0) -> None:
        super().__init__()
        self.tangent_dim = tangent_dim
        self.curvature = curvature
        self.mlp = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, tangent_dim),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        tangent = self.mlp(query.float())
        active = query.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        tangent = tangent * (active / 40.0).clamp(min=0.05, max=1.0)
        return exp_map0(tangent, c=self.curvature)


class Hyper3AttributeHead(nn.Module):
    """Predict UPAR attributes from frozen Hyper3 spatial coordinates."""

    def __init__(self, input_dim: int = 512, attr_dim: int = 40, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, attr_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding[..., 1:])


def graded_pairwise_ranking_loss(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_attributes: torch.Tensor,
    queries: torch.Tensor,
    margin_scale: float = 0.5,
    min_agreement_gap: float = 0.05,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Prefer gallery images with higher attribute agreement for each query.

    The loss uses only within-batch gallery examples. Exact semantic-ID ranking
    can be added separately; this term supplies the graded mADM-aligned signal.
    """

    distances = lorentz_distance(
    query_embeddings[:, None, :].expand(-1, gallery_embeddings.size(0), -1).reshape(-1, query_embeddings.size(-1)),
        gallery_embeddings[None, :, :].expand(query_embeddings.size(0), -1, -1).reshape(-1, gallery_embeddings.size(-1)),
        c=curvature,
    ).view(query_embeddings.size(0), gallery_embeddings.size(0))
    agreement = 1.0 - torch.abs(queries[:, None, :] - gallery_attributes[None, :, :]).mean(dim=-1)
    better = agreement[:, :, None] - agreement[:, None, :]
    distance_delta = distances[:, :, None] - distances[:, None, :]
    pair_margin = margin_scale * better.clamp(min=0.0)
    mask = better > min_agreement_gap
    if not mask.any():
        return distances.mean() * 0.0
    return F.relu(pair_margin + distance_delta)[mask].mean()
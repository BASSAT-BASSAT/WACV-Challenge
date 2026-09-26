"""Training losses for HypPAR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ranking_infonce(distances: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE where distances is (B,) for positive and we pass logits = -d/tau.

    Prefer batch_ranking_loss for in-batch negatives.
    """
    logits = -distances / temperature
    # single positive assumed at index 0 after cat — use batch_ranking_loss instead.
    return -F.log_softmax(logits, dim=0)[0]


def batch_ranking_loss(
    query_emb: torch.Tensor,
    image_emb: torch.Tensor,
    semantic_ids: torch.Tensor,
    distance_fn,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch contrastive ranking: for each sample, positives share semantic_id.

    distance_fn(a, b) -> pairwise distances with a:(B,D), b:(B,D) producing (B,B)
    or we compute manually.
    """
    b = query_emb.size(0)
    # Pairwise distances Q_i vs I_j
    d = _pairwise(query_emb, image_emb, distance_fn)  # (B, B)
    logits = -d / temperature
    # Multi-positive InfoNCE
    ids = semantic_ids.view(-1, 1)
    mask = ids.eq(ids.T).float()
    # Exclude self? queries and images are paired same index — still OK as positive
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # Average over positives
    pos_count = mask.sum(dim=1).clamp(min=1.0)
    loss = -(log_prob * mask).sum(dim=1) / pos_count
    return loss.mean()


def _pairwise(a: torch.Tensor, b: torch.Tensor, distance_fn) -> torch.Tensor:
    """Compute (N,M) distances. distance_fn should accept flat pairs or we expand."""
    n, m = a.size(0), b.size(0)
    aa = a[:, None, :].expand(n, m, a.size(-1)).reshape(n * m, -1)
    bb = b[None, :, :].expand(n, m, b.size(-1)).reshape(n * m, -1)
    return distance_fn(aa, bb).view(n, m)


def attribute_bce_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)

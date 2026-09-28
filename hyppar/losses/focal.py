"""Focal loss for multi-label PAR (better calibration than plain BCE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Multi-label focal loss with optional label smoothing."""
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = alpha * (1.0 - pt).pow(gamma) * bce
    return focal.mean()

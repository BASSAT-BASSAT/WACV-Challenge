"""Retrieval distances for calibrated attribute probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def attr_l1_distances(
    gallery_probs: np.ndarray,
    queries: np.ndarray,
    attr_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted L1 between binary queries and gallery probabilities.

    gallery_probs: (G, A), queries: (Q, A) -> distances (Q, G), smaller = better.
    """
    g = gallery_probs.astype(np.float32)
    q = queries.astype(np.float32)
    if attr_weights is None:
        base = g.sum(1)[None, :] + q @ (1.0 - 2.0 * g).T
        return base.astype(np.float32)
    w = attr_weights.astype(np.float32)[None, None, :]
    diff = np.abs(q[:, None, :] - g[None, :, :])
    return (diff * w).sum(axis=-1).astype(np.float32)


def compute_error_weights(
    probs: np.ndarray,
    labels: np.ndarray,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> np.ndarray:
    """Trust attributes with lower validation error (Specker thesis recipe)."""
    preds = (probs >= 0.5).astype(np.float32)
    err = np.abs(preds - labels).mean(axis=0)
    # High error -> low weight
    w = 1.0 - err
    w = w / max(w.mean(), 1e-6)
    return np.clip(w, min_weight, max_weight).astype(np.float32)


def compute_dbd_weights(gallery_probs: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Distribution-based down-weight for overly common attributes in gallery."""
    p = gallery_probs.mean(axis=0).clip(eps, 1.0 - eps)
    # Attributes near 0 or 1 everywhere are less discriminative.
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    w = entropy / max(entropy.mean(), 1e-6)
    return w.astype(np.float32)


def combine_weights(error_w: np.ndarray, dbd_w: np.ndarray) -> np.ndarray:
    return (error_w * dbd_w).astype(np.float32)


def weights_to_json(weights: np.ndarray) -> dict[str, Any]:
    return {"attr_weights": weights.tolist()}


def weights_from_json(data: dict[str, Any]) -> np.ndarray:
    return np.asarray(data["attr_weights"], dtype=np.float32)


def save_weights(path: Path | str, weights: np.ndarray) -> None:
    Path(path).write_text(json.dumps(weights_to_json(weights), indent=2), encoding="utf-8")


def load_weights(path: Path | str) -> np.ndarray:
    return weights_from_json(json.loads(Path(path).read_text(encoding="utf-8")))

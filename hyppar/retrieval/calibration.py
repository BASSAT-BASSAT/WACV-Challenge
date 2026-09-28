"""Probability calibration for cross-domain retrieval (isotonic per attribute)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _fit_isotonic_1d(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """PAV isotonic regression without sklearn dependency."""
    order = np.argsort(scores)
    s = scores[order]
    y = labels[order].astype(np.float64)
    n = len(s)
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    # Block merging (pool adjacent violators)
    vals = y.copy()
    weights = np.ones(n)
    i = 0
    while i < n - 1:
        if vals[i] <= vals[i + 1] + 1e-12:
            i += 1
            continue
        j = i + 1
        while j < n and vals[j - 1] > vals[j] + 1e-12:
            w = weights[i] + weights[j]
            vals[i] = (vals[i] * weights[i] + vals[j] * weights[j]) / w
            weights[i] = w
            vals = np.delete(vals, j)
            weights = np.delete(weights, j)
            s = np.delete(s, j)
            n -= 1
        i = max(i - 1, 0)
    # Unique knot points
    xs, ys = [], []
    for k in range(n):
        if k == 0 or s[k] != s[k - 1]:
            xs.append(float(s[k]))
            ys.append(float(vals[k]))
    if len(xs) == 1:
        xs = [0.0, 1.0]
        ys = [ys[0], ys[0]]
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def fit_isotonic_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-attribute isotonic maps. probs/labels: (N, 40)."""
    maps: list[tuple[np.ndarray, np.ndarray]] = []
    for a in range(probs.shape[1]):
        maps.append(_fit_isotonic_1d(probs[:, a], labels[:, a]))
    return maps


def apply_isotonic_calibration(probs: np.ndarray, maps: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = np.empty_like(probs, dtype=np.float32)
    for a, (xs, ys) in enumerate(maps):
        out[:, a] = np.interp(probs[:, a], xs, ys).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def calibration_to_json(maps: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    return {
        "method": "isotonic",
        "attributes": [
            {"xs": xs.tolist(), "ys": ys.tolist()} for xs, ys in maps
        ],
    }


def calibration_from_json(data: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.asarray(entry["xs"], dtype=np.float32), np.asarray(entry["ys"], dtype=np.float32))
        for entry in data["attributes"]
    ]


def save_calibration(path: Path | str, maps: list[tuple[np.ndarray, np.ndarray]]) -> None:
    Path(path).write_text(json.dumps(calibration_to_json(maps), indent=2), encoding="utf-8")


def load_calibration(path: Path | str) -> list[tuple[np.ndarray, np.ndarray]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return calibration_from_json(data)

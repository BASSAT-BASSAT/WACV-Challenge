"""Codabench Track 2: calibrated PAR + weighted L1 retrieval (winning path).

Container: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime (no network).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ASSET_DIR = HERE / "assets"
DEFAULT_CKPT = ASSET_DIR / "model.pt"
DEFAULT_CALIB = ASSET_DIR / "calibration.json"
DEFAULT_WEIGHTS = ASSET_DIR / "attr_weights.json"

_MODEL = None
_DEVICE = None
_TRANSFORM = None
_CALIB_MAPS = None
_ATTR_WEIGHTS = None
_ARGS = None


def _build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model() -> None:
    global _MODEL, _DEVICE, _TRANSFORM, _CALIB_MAPS, _ATTR_WEIGHTS, _ARGS
    from hyppar.models.par_classifier import build_par_model
    from hyppar.retrieval.calibration import apply_isotonic_calibration, load_calibration
    from hyppar.retrieval.scoring import attr_l1_distances, load_weights

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TRANSFORM = _build_transform()

    ckpt_path = Path(os.environ.get("HYPPAR_CKPT", str(DEFAULT_CKPT)))
    calib_path = Path(os.environ.get("HYPPAR_CALIB", str(DEFAULT_CALIB)))
    weights_path = Path(os.environ.get("HYPPAR_WEIGHTS", str(DEFAULT_WEIGHTS)))

    ckpt = _torch_load(ckpt_path, _DEVICE)
    _ARGS = ckpt.get("args", {})
    model = build_par_model(
        backbone=_ARGS.get("backbone", "convnext_base"),
        pretrained=False,
        dropout=_ARGS.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(_DEVICE).eval()
    _MODEL = model

    if calib_path.is_file():
        _CALIB_MAPS = load_calibration(calib_path)
    else:
        _CALIB_MAPS = None

    if weights_path.is_file():
        _ATTR_WEIGHTS = load_weights(weights_path)
    else:
        _ATTR_WEIGHTS = None


@torch.no_grad()
def _predict_gallery_probs(gallery: list[dict[str, Any]], bs: int = 32, tta_flip: bool = True) -> np.ndarray:
    from hyppar.retrieval.calibration import apply_isotonic_calibration

    assert _MODEL is not None and _TRANSFORM is not None and _DEVICE is not None
    chunks: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush():
        nonlocal batch
        if not batch:
            return
        x = torch.stack(batch, dim=0).to(_DEVICE, non_blocking=True)
        logits = _MODEL(x)
        probs = torch.sigmoid(logits)
        if tta_flip:
            logits_f = _MODEL(torch.flip(x, dims=[3]))
            probs = 0.5 * (probs + torch.sigmoid(logits_f))
        chunks.append(probs.float().cpu().numpy())
        batch = []

    for g in gallery:
        img = Image.open(g["image_path"]).convert("RGB")
        batch.append(_TRANSFORM(img))
        if len(batch) >= bs:
            flush()
    flush()

    probs = np.concatenate(chunks, axis=0).astype(np.float32)
    if _CALIB_MAPS is not None:
        probs = apply_isotonic_calibration(probs, _CALIB_MAPS)
    return probs


@torch.no_grad()
def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    if _MODEL is None:
        load_model()
    from hyppar.retrieval.scoring import attr_l1_distances

    probs = _predict_gallery_probs(sample["gallery"])
    queries = np.asarray(sample["queries"], dtype=np.float32)
    distances = attr_l1_distances(probs, queries, attr_weights=_ATTR_WEIGHTS)
    return {"distances": distances.astype(np.float32)}

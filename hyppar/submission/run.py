"""Codabench Track 2: HypPAR Lorentz entailment / prototype ranking.

Container: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime (no network).
"""

from __future__ import annotations

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

_MODEL = None
_DEVICE = None
_TRANSFORM = None
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
    global _MODEL, _DEVICE, _TRANSFORM, _ARGS
    from hyppar.models.hyppar import AttributePrototypeModel, ImageQueryModel

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TRANSFORM = _build_transform()
    ckpt_path = Path(os.environ.get("HYPPAR_CKPT", str(DEFAULT_CKPT)))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = _torch_load(ckpt_path, _DEVICE)
    args = ckpt["args"]
    _ARGS = args
    common = dict(
        geometry=args.get("geometry", "lorentz"),
        embed_dim=args.get("embed_dim", 128),
        curvature=args.get("curvature", 1.0),
        learnable_c=args.get("learnable_c", False),
        freeze_backbone=True,
        pretrained=False,
        backbone=args.get("backbone", "convnext_tiny"),
    )
    if args.get("model", "prototypes") == "image_query":
        model = ImageQueryModel(**common)
    else:
        model = AttributePrototypeModel(
            **common,
            composition=args.get("composition", "learned"),
            score_mode=args.get("score_mode", "entailment"),
            entail_K=args.get("entail_k", 0.1),
            entail_beta=args.get("entail_beta", 1.0),
            radius_compose=not args.get("no_radius_compose", False),
            hybrid_mix=args.get("hybrid_mix", 0.7),
        )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(_DEVICE).eval()
    _MODEL = model


@torch.no_grad()
def _embed_gallery(gallery: list[dict[str, Any]], bs: int = 32):
    assert _MODEL is not None and _TRANSFORM is not None and _DEVICE is not None
    embs: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    batch: list[torch.Tensor] = []
    use_attr = hasattr(_MODEL, "attribute_logits_from_features")

    def flush():
        nonlocal batch
        if not batch:
            return
        x = torch.stack(batch, dim=0).to(_DEVICE, non_blocking=True)
        if use_attr:
            feat = _MODEL.encode_features(x)
            probs.append(torch.sigmoid(_MODEL.attribute_logits_from_features(feat)).float().cpu())
            embs.append(_MODEL.encode_image(x).float().cpu())
        else:
            embs.append(_MODEL.encode_image(x).float().cpu())
        batch = []

    for g in gallery:
        img = Image.open(g["image_path"]).convert("RGB")
        batch.append(_TRANSFORM(img))
        if len(batch) >= bs:
            flush()
    flush()
    gallery_emb = torch.cat(embs, dim=0)
    gallery_probs = torch.cat(probs, dim=0) if probs else None
    return gallery_emb, gallery_probs


@torch.no_grad()
def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    if _MODEL is None:
        load_model()
    assert _MODEL is not None and _DEVICE is not None and _ARGS is not None

    gallery_emb, gallery_probs = _embed_gallery(sample["gallery"])
    gallery_emb = gallery_emb.to(_DEVICE)
    if gallery_probs is not None:
        gallery_probs = gallery_probs.to(_DEVICE)

    queries = np.asarray(sample["queries"], dtype=np.float32)
    q = torch.from_numpy(queries).float().to(_DEVICE)

    sims: list[torch.Tensor] = []
    q_bs = 64
    for i in range(0, q.size(0), q_bs):
        qb = q[i : i + q_bs]
        if _ARGS.get("model", "prototypes") == "image_query":
            q_emb = _MODEL.encode_query(qb)
            sims.append(_MODEL.score_matrix(q_emb, gallery_emb))
        else:
            sims.append(_MODEL.score_matrix(qb, gallery_emb, gallery_attr_probs=gallery_probs))
    sim = torch.cat(sims, dim=0)
    distances = (-sim).cpu().numpy().astype(np.float32)
    return {"distances": distances}

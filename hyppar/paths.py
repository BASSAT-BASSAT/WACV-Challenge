"""Shared transforms and path defaults (override with env vars on Linux)."""

from __future__ import annotations

import os
from pathlib import Path

import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent  # .../hyppar package
REPO_ROOT = ROOT.parent

# Prefer env so Linux box can point at UPAR data wherever it lives.
UPAR_ROOT = Path(
    os.environ.get(
        "HYPPAR_UPAR_ROOT",
        os.environ.get("UPAR_ROOT", str(REPO_ROOT.parent / "UPAR-Challenge-2027")),
    )
)
DATA_ROOT = Path(os.environ.get("HYPPAR_DATA_ROOT", str(UPAR_ROOT / "data")))
ANNO_ROOT = Path(os.environ.get("HYPPAR_ANNO_ROOT", str(DATA_ROOT / "annotations")))
CKPT_DIR = Path(os.environ.get("HYPPAR_CKPT_DIR", str(REPO_ROOT / "checkpoints")))
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

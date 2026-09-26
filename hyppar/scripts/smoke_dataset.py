"""Quick smoke: imports + optional dataset check."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from hyppar.models.hyppar import AttributePrototypeModel
from hyppar.models.lorentz import entailment_distance, exp_map0
from hyppar.paths import ANNO_ROOT, DATA_ROOT
import torch


def main() -> None:
    print("DATA_ROOT", DATA_ROOT)
    print("ANNO_ROOT", ANNO_ROOT)
    m = AttributePrototypeModel(
        geometry="lorentz", score_mode="entailment", pretrained=False, backbone="convnext_tiny"
    )
    x = torch.randn(2, 3, 224, 224)
    q = torch.zeros(2, 40)
    q[:, 0] = 1
    h = m.encode_image(x)
    s = m.score_matrix(q, h)
    print("forward ok", tuple(h.shape), tuple(s.shape))
    if (DATA_ROOT / "annotations").exists() or ANNO_ROOT.exists():
        from hyppar.data.dataset import UPARTask2Dataset
        from hyppar.paths import train_transform

        ds = UPARTask2Dataset(
            DATA_ROOT, ANNO_ROOT, "train", transform=train_transform(), require_files=False
        )
        print("train records", len(ds), "queries", ds.queries.shape)
    else:
        print("skip dataset (set HYPPAR_UPAR_ROOT)")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()

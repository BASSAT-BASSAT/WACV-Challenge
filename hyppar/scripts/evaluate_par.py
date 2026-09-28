#!/usr/bin/env python3
"""Evaluate PAR classifier + calibrated weighted L1 retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hyppar.data.dataset import UPARTask2Dataset, load_split_tables
from hyppar.eval.metrics import evaluate_ranking
from hyppar.models.par_classifier import build_par_model
from hyppar.paths import ANNO_ROOT, CKPT_DIR, DATA_ROOT, eval_transform
from hyppar.retrieval.calibration import (
    apply_isotonic_calibration,
    fit_isotonic_calibration,
    save_calibration,
)
from hyppar.retrieval.inference import predict_probs
from hyppar.retrieval.scoring import (
    attr_l1_distances,
    combine_weights,
    compute_dbd_weights,
    compute_error_weights,
    save_weights,
)


def load_par_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = build_par_model(
        backbone=args.get("backbone", "convnext_base"),
        pretrained=False,
        dropout=args.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument(
        "--calib-split",
        type=str,
        default="train",
        help="Split to fit calibration/weights; use train for leakage-free val evaluation",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tta-flip", action="store_true")
    p.add_argument("--no-tta-flip", action="store_true")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--no-error-weights", action="store_true")
    p.add_argument("--no-dbd", action="store_true")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--anno-root", type=str, default=str(ANNO_ROOT))
    p.add_argument("--out", type=str, default="")
    p.add_argument("--save-artifacts", action="store_true", help="Write calibration.json + weights.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    model, model_args = load_par_model(ckpt_path, device)

    # Fit calibration on calib split
    calib_ds = UPARTask2Dataset(
        args.data_root, args.anno_root, args.calib_split, transform=eval_transform()
    )
    calib_loader = DataLoader(
        calib_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    calib_probs = predict_probs(model, calib_loader, device, tta_flip=False).numpy()
    calib_labels = np.stack([r["attrs"] for r in calib_ds.records], axis=0)

    maps = fit_isotonic_calibration(calib_probs, calib_labels)
    err_w = compute_error_weights(calib_probs, calib_labels)
    dbd_w = compute_dbd_weights(calib_probs)
    attr_w = err_w if args.no_dbd else combine_weights(err_w, dbd_w)
    if args.no_error_weights:
        attr_w = None

    # Eval split
    ds = UPARTask2Dataset(
        args.data_root, args.anno_root, args.split, transform=eval_transform()
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    use_tta = args.tta_flip or not args.no_tta_flip
    probs = predict_probs(model, loader, device, tta_flip=use_tta).numpy()
    if not args.no_calibration:
        probs = apply_isotonic_calibration(probs, maps)

    gallery_attrs = np.stack([r["attrs"] for r in ds.records], axis=0)
    gallery_ids = np.array([r["semantic_id"] for r in ds.records], dtype=np.int64)
    domains = [r["domain"] for r in ds.records]
    _, _, queries = load_split_tables(args.anno_root, args.split)

    distances = attr_l1_distances(probs, queries, attr_weights=attr_w)
    results = evaluate_ranking(distances, queries, gallery_attrs, gallery_ids, domains)
    results["model_type"] = "par"
    results["backbone"] = model_args.get("backbone")
    results["calibration"] = not args.no_calibration
    results["error_weights"] = not args.no_error_weights
    results["dbd_weights"] = not args.no_dbd
    results["tta_flip"] = use_tta

    print(json.dumps(results["overall"], indent=2), flush=True)
    out_dir = ckpt_path.parent
    out = Path(args.out) if args.out else out_dir / f"eval_{args.split}_par.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)

    if args.save_artifacts:
        save_calibration(out_dir / "calibration.json", maps)
        if attr_w is not None:
            save_weights(out_dir / "attr_weights.json", attr_w)
        print(f"artifacts -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

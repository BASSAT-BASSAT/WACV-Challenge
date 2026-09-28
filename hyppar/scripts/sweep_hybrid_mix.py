#!/usr/bin/env python3
"""Sweep hybrid_mix on val: embed once, mix entailment + attr_l1 scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hyppar.data.dataset import UPARTask2Dataset, load_split_tables
from hyppar.eval.metrics import evaluate_ranking
from hyppar.evaluate import load_model
from hyppar.paths import ANNO_ROOT, DATA_ROOT, eval_transform
from hyppar.train import embed_split


def _z(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp(min=1e-6)
    return (x - mu) / sd


@torch.no_grad()
def compute_base_sims(
    model,
    queries: np.ndarray,
    gallery_emb: torch.Tensor,
    gallery_attr_probs: np.ndarray,
    device: torch.device,
    chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hyp_sim, l1_sim) both (Q, G), higher = better."""
    q = torch.from_numpy(queries).float().to(device)
    g = gallery_emb.to(device)
    probs = torch.from_numpy(gallery_attr_probs).float().to(device)

    # attr L1 (same formula as AttributePrototypeModel)
    l1 = -(probs.sum(1)[None, :] + q @ (1.0 - 2.0 * probs).T)

    # entailment hyperbolic similarity
    prev = model.score_mode
    model.score_mode = "entailment" if model.geometry == "lorentz" else "distance"
    hyp_chunks = []
    for i in tqdm(range(0, q.size(0), chunk), desc="hyp_score"):
        qb = q[i : i + chunk]
        qe = model.compose_query(qb)
        dist = model._pairwise_retrieval_distance(qe, g)
        hyp_chunks.append((-dist).cpu())
    model.score_mode = prev
    hyp = torch.cat(hyp_chunks, dim=0)
    return hyp, l1.cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--mixes",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated hybrid_mix values (weight on hyperbolic/entailment)",
    )
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    mixes = [float(x.strip()) for x in args.mixes.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.ckpt), device, score_mode="hybrid")

    ds = UPARTask2Dataset(
        DATA_ROOT, ANNO_ROOT, args.split, transform=eval_transform(), require_files=True
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    emb, attrs, sids, domains, _rels, attr_probs = embed_split(model, loader, device)
    assert attr_probs is not None
    _, _, queries = load_split_tables(ANNO_ROOT, args.split)

    print("computing base scores (once)...", flush=True)
    hyp_sim, l1_sim = compute_base_sims(model, queries, emb, attr_probs, device)
    zh, zl = _z(hyp_sim), _z(l1_sim)

    rows = []
    print(f"{'mix':>6}  {'mADM':>8}  {'mAP':>8}  {'R1':>8}", flush=True)
    for m in mixes:
        sim = m * zh + (1.0 - m) * zl
        distances = (-sim).numpy()
        results = evaluate_ranking(distances, queries, attrs, sids, domains)
        row = {
            "hybrid_mix": m,
            "mADM": results["overall"]["mADM"],
            "mAP": results["overall"]["mAP"],
            "R1": results["overall"]["R1"],
            "by_domain": {k: v["mADM"] for k, v in results["by_domain"].items()},
        }
        rows.append(row)
        print(
            f"{m:6.2f}  {100 * row['mADM']:7.2f}%  {100 * row['mAP']:7.2f}%  {100 * row['R1']:7.2f}%",
            flush=True,
        )

    best = max(rows, key=lambda r: r["mADM"])
    print(
        f"\nBEST mix={best['hybrid_mix']:.2f}  mADM={100 * best['mADM']:.2f}%",
        flush=True,
    )

    out = (
        Path(args.out)
        if args.out
        else Path(args.ckpt).parent / f"hybrid_mix_sweep_{args.split}.json"
    )
    payload = {"ckpt": str(args.ckpt), "split": args.split, "rows": rows, "best": best}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

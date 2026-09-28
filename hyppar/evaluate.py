"""Evaluate a checkpoint on UPAR Track 2 val split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from hyppar.data.dataset import UPARTask2Dataset, load_split_tables
from hyppar.eval.metrics import evaluate_ranking
from hyppar.models.hyppar import AttributePrototypeModel, ImageQueryModel
from hyppar.paths import ANNO_ROOT, DATA_ROOT, eval_transform
from hyppar.train import build_model, embed_split


def load_model(
    ckpt_path: Path,
    device: torch.device,
    score_mode: str = "",
    hybrid_mix: float | None = None,
) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ns = argparse.Namespace(**ckpt["args"])
    # Defaults for older checkpoints.
    if not hasattr(args_ns, "score_mode"):
        args_ns.score_mode = "distance"
    if not hasattr(args_ns, "entail_k"):
        args_ns.entail_k = 0.1
    if not hasattr(args_ns, "entail_beta"):
        args_ns.entail_beta = 1.0
    if not hasattr(args_ns, "no_radius_compose"):
        args_ns.no_radius_compose = False
    if not hasattr(args_ns, "unfreeze"):
        args_ns.unfreeze = False
    if not hasattr(args_ns, "unfreeze_stages"):
        args_ns.unfreeze_stages = 0
    if not hasattr(args_ns, "no_pretrained"):
        args_ns.no_pretrained = False
    if not hasattr(args_ns, "backbone"):
        args_ns.backbone = "convnext_tiny"
    if not hasattr(args_ns, "hybrid_mix"):
        args_ns.hybrid_mix = 0.7
    if not hasattr(args_ns, "cosine"):
        args_ns.cosine = False
    if score_mode:
        args_ns.score_mode = score_mode
    if hybrid_mix is not None:
        args_ns.hybrid_mix = hybrid_mix
    model = build_model(args_ns)
    model.load_state_dict(ckpt["model"], strict=False)
    if score_mode:
        model.score_mode = score_mode
    if hybrid_mix is not None:
        model.hybrid_mix = hybrid_mix
    return model.to(device).eval()


@torch.no_grad()
def score_distances(
    model,
    queries: np.ndarray,
    gallery_emb: torch.Tensor,
    device,
    gallery_attr_probs: np.ndarray | None = None,
    chunk: int = 16,
    gallery_chunk: int = 512,
):
    q = torch.from_numpy(queries).float().to(device)
    g = gallery_emb.to(device)
    probs = (
        torch.from_numpy(gallery_attr_probs).float().to(device)
        if gallery_attr_probs is not None
        else None
    )
    scores = []
    for i in tqdm(range(0, q.size(0), chunk), desc="score"):
        qb = q[i : i + chunk]
        query_scores = []
        for j in range(0, g.size(0), gallery_chunk):
            gb = g[j : j + gallery_chunk]
            pb = probs[j : j + gallery_chunk] if probs is not None else None
            if isinstance(model, ImageQueryModel):
                qe = model.encode_query(qb)
                s = model.score_matrix(qe, gb)
            else:
                s = model.score_matrix(qb, gb, gallery_attr_probs=pb)
            query_scores.append((-s).cpu())
        scores.append(torch.cat(query_scores, dim=1))
    return torch.cat(scores, dim=0).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--score-mode", type=str, default="", help="Override ckpt score_mode")
    p.add_argument(
        "--hybrid-mix",
        type=float,
        default=None,
        help="Override hyperbolic weight in hybrid mode (0=attr_l1, 1=entailment)",
    )
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--anno-root", type=str, default=str(ANNO_ROOT))
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        Path(args.ckpt), device, score_mode=args.score_mode, hybrid_mix=args.hybrid_mix
    )
    mode = getattr(model, "score_mode", "distance")
    mix = getattr(model, "hybrid_mix", None)
    print(f"scoring with mode={mode} hybrid_mix={mix}", flush=True)

    ds = UPARTask2Dataset(
        args.data_root, args.anno_root, args.split, transform=eval_transform(), require_files=True
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    emb, attrs, sids, domains, _rels, attr_probs = embed_split(model, loader, device)
    _, _, queries = load_split_tables(args.anno_root, args.split)

    distances = score_distances(model, queries, emb, device, gallery_attr_probs=attr_probs)
    results = evaluate_ranking(distances, queries, attrs, sids, domains)
    results["score_mode"] = mode
    if mix is not None:
        results["hybrid_mix"] = mix
    print(json.dumps(results, indent=2))
    out = Path(args.out) if args.out else Path(args.ckpt).parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

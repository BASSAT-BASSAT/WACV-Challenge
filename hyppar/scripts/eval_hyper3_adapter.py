"""Evaluate a trained frozen Hyper3-CLIP UPAR adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hyppar.data.dataset import UPARTask2Dataset, load_split_tables
from hyppar.eval.metrics import evaluate_ranking
from hyppar.models.hyper3_adapter import Hyper3AttributeHead, Hyper3AttributeQueryAdapter
from hyppar.models.lorentz import lorentz_distance
from hyppar.models.hyper3_runtime import Hyper3ClipRuntime
from hyppar.paths import ANNO_ROOT, CKPT_DIR, DATA_ROOT, eval_transform
from hyppar.scripts.eval_hyper3_zero_shot import cache_gallery


def score_hyperbolic(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_chunk: int,
    gallery_chunk: int,
    curvature: float,
) -> np.ndarray:
    rows: list[torch.Tensor] = []
    for start in range(0, query_embeddings.size(0), query_chunk):
        qb = query_embeddings[start : start + query_chunk]
        parts: list[torch.Tensor] = []
        for gstart in range(0, gallery_embeddings.size(0), gallery_chunk):
            gb = gallery_embeddings[gstart : gstart + gallery_chunk]
            aa = qb[:, None, :].expand(-1, gb.size(0), -1).reshape(-1, qb.size(-1))
            bb = gb[None, :, :].expand(qb.size(0), -1, -1).reshape(-1, gb.size(-1))
            parts.append(
                lorentz_distance(aa, bb, c=curvature).view(qb.size(0), gb.size(0)).cpu()
            )
        rows.append(torch.cat(parts, dim=1))
    return torch.cat(rows, dim=0).numpy()


def attribute_distance(
    queries: np.ndarray, probabilities: np.ndarray, query_chunk: int = 64
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, queries.shape[0], query_chunk):
        q = queries[start : start + query_chunk]
        rows.append(
            probabilities.sum(axis=1)[None, :]
            + q @ (1.0 - 2.0 * probabilities).T
        )
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-mode", choices=["hyperbolic", "attr_l1", "hybrid"], default="hybrid")
    parser.add_argument("--hybrid-mix", type=float, default=0.7)
    parser.add_argument("--model", default="hyper3labs/hyper3-clip-v1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-chunk", type=int, default=8)
    parser.add_argument("--gallery-chunk", type=int, default=512)
    parser.add_argument("--cache", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--anno-root", default=str(ANNO_ROOT))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    curvature = float(checkpoint.get("curvature", 1.0))
    query_adapter = Hyper3AttributeQueryAdapter(curvature=curvature).to(device)
    attribute_head = Hyper3AttributeHead().to(device)
    query_adapter.load_state_dict(checkpoint["query_adapter"])
    attribute_head.load_state_dict(checkpoint["attribute_head"])
    query_adapter.eval()
    attribute_head.eval()

    dataset = UPARTask2Dataset(
        args.data_root, args.anno_root, args.split, transform=eval_transform(), require_files=True
    )
    runtime = Hyper3ClipRuntime(
        model_name=args.model, device=args.device, local_files_only=args.local_files_only
    )
    cache = Path(args.cache) if args.cache else CKPT_DIR / "hyper3_cache" / f"gallery_{args.split}.npz"
    gallery_np = cache_gallery(dataset, runtime, cache, args.batch_size)
    gallery = torch.from_numpy(gallery_np).float().to(device)
    _, _, queries = load_split_tables(args.anno_root, args.split)
    gallery_attrs = np.asarray([record["attrs"] for record in dataset.records], dtype=np.float32)
    gallery_ids = np.asarray([record["semantic_id"] for record in dataset.records], dtype=np.int64)
    domains = [record["domain"] for record in dataset.records]

    with torch.no_grad():
        query_tensor = torch.from_numpy(queries).float().to(device)
        query_embeddings = query_adapter(query_tensor)
        probabilities = torch.sigmoid(attribute_head(gallery)).cpu().numpy()
    hyperbolic = score_hyperbolic(
        query_embeddings, gallery, args.query_chunk, args.gallery_chunk, curvature
    )
    attr = attribute_distance(queries, probabilities)
    if args.score_mode == "hyperbolic":
        distances = hyperbolic
    elif args.score_mode == "attr_l1":
        distances = attr
    else:
        hyp_sim = -hyperbolic
        attr_sim = -attr
        hyp_z = (hyp_sim - hyp_sim.mean(axis=1, keepdims=True)) / hyp_sim.std(axis=1, keepdims=True).clip(min=1e-6)
        attr_z = (attr_sim - attr_sim.mean(axis=1, keepdims=True)) / attr_sim.std(axis=1, keepdims=True).clip(min=1e-6)
        distances = -(args.hybrid_mix * hyp_z + (1.0 - args.hybrid_mix) * attr_z)

    results = evaluate_ranking(distances, queries, gallery_attrs, gallery_ids, domains)
    results["score_mode"] = args.score_mode
    results["hybrid_mix"] = args.hybrid_mix
    results["curvature"] = curvature
    output = Path(args.out) if args.out else Path(args.ckpt).parent / f"eval_{args.split}_{args.score_mode}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["overall"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

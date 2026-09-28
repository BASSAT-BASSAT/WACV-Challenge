"""Evaluate zero-shot Hyper3-CLIP attribute prompts on UPAR Track 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hyppar.data.dataset import ATTRIBUTE_NAMES, UPARTask2Dataset, load_split_tables
from hyppar.eval.metrics import evaluate_ranking
from hyppar.models.lorentz import lorentz_distance
from hyppar.models.hyper3_runtime import Hyper3ClipRuntime
from hyppar.paths import ANNO_ROOT, CKPT_DIR, DATA_ROOT, eval_transform


def cache_gallery(
    dataset: UPARTask2Dataset,
    runtime: Hyper3ClipRuntime,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    if cache_path.is_file():
        return np.load(cache_path)["embeddings"]
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(dataset.records), batch_size), desc="hyper3 gallery"):
        records = dataset.records[start : start + batch_size]
        images = [Image.open(rec["path"]).convert("RGB") for rec in records]
        try:
            chunks.append(runtime.encode_images(images))
        finally:
            for image in images:
                image.close()
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings)
    return embeddings


def compose_text_queries(text_embeddings: np.ndarray, queries: np.ndarray, curvature: float) -> np.ndarray:
    spatial = text_embeddings[:, 1:]
    query_spatial = []
    for query in queries:
        active = float(query.sum())
        if active < 1.0:
            query_spatial.append(np.zeros(spatial.shape[1], dtype=np.float32))
            continue
        mixed = (query[:, None] * spatial).sum(axis=0) / active
        density = np.clip(active / 40.0, 0.05, 1.0)
        query_spatial.append(mixed * density)
    spatial_tensor = torch.from_numpy(np.asarray(query_spatial, dtype=np.float32))
    time = torch.sqrt((1.0 / curvature) + (spatial_tensor * spatial_tensor).sum(dim=-1, keepdim=True))
    return torch.cat([time, spatial_tensor], dim=-1).numpy()


def score_queries(
    queries: np.ndarray,
    gallery: np.ndarray,
    query_chunk: int,
    gallery_chunk: int,
    curvature: float,
) -> np.ndarray:
    q = torch.from_numpy(queries).float()
    g = torch.from_numpy(gallery).float()
    rows: list[torch.Tensor] = []
    for start in tqdm(range(0, q.size(0), query_chunk), desc="zero-shot score"):
        qb = q[start : start + query_chunk]
        parts: list[torch.Tensor] = []
        for gstart in range(0, g.size(0), gallery_chunk):
            gb = g[gstart : gstart + gallery_chunk]
            aa = qb[:, None, :].expand(-1, gb.size(0), -1).reshape(-1, q.size(-1))
            bb = gb[None, :, :].expand(qb.size(0), -1, -1).reshape(-1, g.size(-1))
            parts.append(lorentz_distance(aa, bb, c=curvature).view(qb.size(0), gb.size(0)))
        rows.append(torch.cat(parts, dim=1))
    return torch.cat(rows, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--model", default="hyper3labs/hyper3-clip-v1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-chunk", type=int, default=8)
    parser.add_argument("--gallery-chunk", type=int, default=512)
    parser.add_argument("--cache", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--anno-root", type=str, default=str(ANNO_ROOT))
    args = parser.parse_args()

    dataset = UPARTask2Dataset(
        args.data_root, args.anno_root, args.split, transform=eval_transform(), require_files=True
    )
    runtime = Hyper3ClipRuntime(
        model_name=args.model, device=args.device, local_files_only=args.local_files_only
    )
    cache = Path(args.cache) if args.cache else CKPT_DIR / "hyper3_cache" / f"gallery_{args.split}.npz"
    gallery_embeddings = cache_gallery(dataset, runtime, cache, args.batch_size)
    _, gallery_ids_table, query_attributes = load_split_tables(args.anno_root, args.split)
    gallery_ids = np.asarray([rec["semantic_id"] for rec in dataset.records], dtype=np.int64)
    gallery_attributes = np.asarray([rec["attrs"] for rec in dataset.records], dtype=np.float32)
    domains = [rec["domain"] for rec in dataset.records]

    prompts = [f"a pedestrian with {name.replace('-', ' ')}" for name in ATTRIBUTE_NAMES]
    text_embeddings = runtime.encode_texts(prompts)
    query_embeddings = compose_text_queries(text_embeddings, query_attributes, runtime.curvature)
    distances = score_queries(
        query_embeddings, gallery_embeddings, args.query_chunk, args.gallery_chunk, runtime.curvature
    )
    results = evaluate_ranking(distances, query_attributes, gallery_attributes, gallery_ids, domains)
    results["model"] = args.model
    results["curvature"] = runtime.curvature
    results["mode"] = "zero_shot_attribute_prompt_composition"
    output = Path(args.out) if args.out else CKPT_DIR / "hyper3_zero_shot" / f"eval_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["overall"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

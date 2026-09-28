"""Evaluate OpenAI CLIP zero-shot attribute composition on UPAR Track 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hyppar.data.dataset import ATTRIBUTE_NAMES, UPARTask2Dataset, load_split_tables
from hyppar.eval.metrics import evaluate_ranking
from hyppar.paths import ANNO_ROOT, CKPT_DIR, DATA_ROOT


def cache_gallery(dataset, processor, model, device, cache_path: Path, batch_size: int) -> np.ndarray:
    if cache_path.is_file():
        cached = np.load(cache_path)["embeddings"]
        if cached.shape == (len(dataset.records), model.config.projection_dim):
            return cached
        print(f"ignoring stale CLIP cache {cache_path}: shape={cached.shape}", flush=True)
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(dataset.records), batch_size), desc="CLIP gallery"):
        records = dataset.records[start : start + batch_size]
        images = [Image.open(record["path"]).convert("RGB") for record in records]
        try:
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                embeddings = model.get_image_features(pixel_values=pixel_values)
                embeddings = F.normalize(embeddings.float(), dim=-1)
            chunks.append(embeddings.cpu().numpy().astype(np.float32))
        finally:
            for image in images:
                image.close()
    result = np.concatenate(chunks, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=result)
    return result


def attribute_phrase(name: str) -> str:
    parts = name.split("-")
    if len(parts) == 2:
        return parts[1].replace("&", " and ").lower()
    return f"{parts[-1].replace('&', ' and ').lower()} {parts[0].lower()}"


def score_queries(query_vectors: np.ndarray, gallery_vectors: np.ndarray, query_chunk: int, gallery_chunk: int) -> np.ndarray:
    q = torch.from_numpy(query_vectors).float()
    g = torch.from_numpy(gallery_vectors).float()
    rows: list[torch.Tensor] = []
    for start in tqdm(range(0, q.size(0), query_chunk), desc="CLIP score"):
        qb = q[start : start + query_chunk]
        parts = []
        for gstart in range(0, g.size(0), gallery_chunk):
            parts.append(1.0 - qb @ g[gstart : gstart + gallery_chunk].T)
        rows.append(torch.cat(parts, dim=1))
    return torch.cat(rows, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--gallery-chunk", type=int, default=1024)
    parser.add_argument("--cache", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--anno-root", default=str(ANNO_ROOT))
    args = parser.parse_args()

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError("Install Transformers with: pip install transformers") from error

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    dataset = UPARTask2Dataset(args.data_root, args.anno_root, args.split, transform=None, require_files=True)
    gt, _, queries = load_split_tables(args.anno_root, args.split)
    print(
        f"data_root={args.data_root} anno_root={args.anno_root} split={args.split} | "
        f"annotation_rows={len(gt)} records={len(dataset.records)} missing={dataset.missing}",
        flush=True,
    )

    cache = Path(args.cache) if args.cache else CKPT_DIR / "clip_cache" / f"gallery_{args.split}.npz"
    gallery = cache_gallery(dataset, processor, model, device, cache, args.batch_size)
    prompts = [f"a photo of a pedestrian with {attribute_phrase(name)}" for name in ATTRIBUTE_NAMES]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        text_features = F.normalize(model.get_text_features(**text_inputs).float(), dim=-1)
    text_features = text_features.cpu().numpy()

    query_vectors = []
    for query in queries:
        active = float(query.sum())
        vector = (query[:, None] * text_features).sum(axis=0) / max(active, 1.0)
        query_vectors.append(vector)
    query_vectors = F.normalize(torch.from_numpy(np.asarray(query_vectors, dtype=np.float32)), dim=-1).numpy()
    distances = score_queries(query_vectors, gallery, args.query_chunk, args.gallery_chunk)

    gallery_attrs = np.asarray([record["attrs"] for record in dataset.records], dtype=np.float32)
    gallery_ids = np.asarray([record["semantic_id"] for record in dataset.records], dtype=np.int64)
    domains = [record["domain"] for record in dataset.records]
    results = evaluate_ranking(distances, queries, gallery_attrs, gallery_ids, domains)
    results["model"] = args.model
    results["mode"] = "clip_euclidean_attribute_prompt_composition"
    output = Path(args.out) if args.out else CKPT_DIR / "clip_zero_shot" / f"eval_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["overall"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

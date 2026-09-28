"""Train a frozen Hyper3-CLIP UPAR query adapter and attribute head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hyppar.data.dataset import UPARTask2Dataset
from hyppar.losses import attribute_bce_from_logits, batch_ranking_loss
from hyppar.losses.focal import focal_bce_with_logits
from hyppar.models.hyper3_adapter import (
    Hyper3AttributeHead,
    Hyper3AttributeQueryAdapter,
    graded_pairwise_ranking_loss,
)
from hyppar.models.hyper3_runtime import Hyper3ClipRuntime
from hyppar.models.lorentz import lorentz_distance
from hyppar.paths import ANNO_ROOT, CKPT_DIR, DATA_ROOT, eval_transform
from hyppar.scripts.eval_hyper3_zero_shot import cache_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="hyper3_frozen_adapter")
    parser.add_argument("--model", default="hyper3-clip-v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-attr", type=float, default=1.0)
    parser.add_argument("--lambda-exact", type=float, default=1.0)
    parser.add_argument("--lambda-graded", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--cache", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--anno-root", default=str(ANNO_ROOT))
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UPARTask2Dataset(
        args.data_root, args.anno_root, "train", transform=eval_transform(), require_files=True
    )
    if args.max_train_samples > 0:
        dataset.records = dataset.records[: args.max_train_samples]

    runtime = Hyper3ClipRuntime(
        model_name=args.model, device=args.device, local_files_only=args.local_files_only
    )
    cache_path = Path(args.cache) if args.cache else CKPT_DIR / "hyper3_cache" / "gallery_train.npz"
    gallery = cache_gallery(dataset, runtime, cache_path, batch_size=16)
    attrs = np.asarray([record["attrs"] for record in dataset.records], dtype=np.float32)
    queries = np.asarray([record["query"] for record in dataset.records], dtype=np.float32)
    semantic_ids = np.asarray([record["semantic_id"] for record in dataset.records], dtype=np.int64)

    tensor_set = TensorDataset(
        torch.from_numpy(gallery),
        torch.from_numpy(attrs),
        torch.from_numpy(queries),
        torch.from_numpy(semantic_ids),
    )
    loader = DataLoader(
        tensor_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    query_adapter = Hyper3AttributeQueryAdapter().to(device)
    attribute_head = Hyper3AttributeHead().to(device)
    model = torch.nn.ModuleDict({"query_adapter": query_adapter, "attribute_head": attribute_head})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = CKPT_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    distance_fn = lambda left, right: lorentz_distance(left, right, c=1.0)

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        steps = 0
        for step, (gallery_batch, attr_batch, query_batch, sid_batch) in enumerate(
            tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False), start=1
        ):
            gallery_batch = gallery_batch.to(device, non_blocking=True)
            attr_batch = attr_batch.to(device, non_blocking=True)
            query_batch = query_batch.to(device, non_blocking=True)
            sid_batch = sid_batch.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                query_embeddings = query_adapter(query_batch)
                logits = attribute_head(gallery_batch)
                attr_loss = focal_bce_with_logits(
                    logits, attr_batch, gamma=args.focal_gamma
                )
                exact_loss = batch_ranking_loss(
                    query_embeddings,
                    gallery_batch,
                    sid_batch,
                    distance_fn,
                    temperature=0.07,
                )
                graded_loss = graded_pairwise_ranking_loss(
                    query_embeddings, gallery_batch, attr_batch, query_batch
                )
                loss = (
                    args.lambda_attr * attr_loss
                    + args.lambda_exact * exact_loss
                    + args.lambda_graded * graded_loss
                )
            scaled_loss = loss / args.grad_accum_steps
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = step % args.grad_accum_steps == 0 or step == len(loader)
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
            steps += 1
        epoch_loss = running / max(steps, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"epoch {epoch}/{args.epochs} loss={epoch_loss:.5f}", flush=True)

    checkpoint = {
        "query_adapter": query_adapter.state_dict(),
        "attribute_head": attribute_head.state_dict(),
        "args": vars(args),
        "epoch": args.epochs,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {output_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()

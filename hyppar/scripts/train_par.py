#!/usr/bin/env python3
"""Train a strong PAR classifier for attribute-based retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hyppar.data.dataset import UPARTask2Dataset, filter_records_by_domains
from hyppar.losses.focal import focal_bce_with_logits
from hyppar.models.par_classifier import build_par_model
from hyppar.paths import ANNO_ROOT, CKPT_DIR, DATA_ROOT, train_transform
from hyppar.utils.ema import ModelEMA


def parse_domains(s: str) -> list[str] | None:
    if not s or s.lower() in ("all", "none", ""):
        return None
    return [d.strip() for d in s.split(",") if d.strip()]


def train_one_epoch(model, loader, optimizer, device, args, scaler, ema: ModelEMA | None):
    model.train()
    total, n = 0.0, 0
    use_amp = (args.amp or not args.no_amp) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        attrs = batch["attrs"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = focal_bce_with_logits(
                logits,
                attrs,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
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
            if ema is not None:
                ema.update(model)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def val_loss(model, loader, device, args) -> float:
    model.eval()
    total, n = 0.0, 0
    use_amp = (args.amp or not args.no_amp) and device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        attrs = batch["attrs"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = focal_bce_with_logits(
                logits, attrs, gamma=args.focal_gamma, label_smoothing=0.0
            )
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", type=str, default="par_convnext_base")
    p.add_argument("--backbone", type=str, default="convnext_base")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--merge-val-into-train",
        action="store_true",
        help="Train on train+val images (for final Codabench model)",
    )
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--train-domains", type=str, default="all")
    p.add_argument("--val-domains", type=str, default="all")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0 if sys.platform.startswith("win") else 4,
        help="DataLoader workers (use 0 on Windows to avoid shared-memory OOM)",
    )
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--anno-root", type=str, default=str(ANNO_ROOT))
    args = p.parse_args()
    if args.grad_accum_steps < 1:
        p.error("--grad-accum-steps must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = CKPT_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = UPARTask2Dataset(
        args.data_root, args.anno_root, args.train_split, transform=train_transform()
    )
    train_domains = parse_domains(args.train_domains)
    train_ds.records = filter_records_by_domains(train_ds.records, train_domains)
    if args.merge_val_into_train:
        val_for_merge = UPARTask2Dataset(
            args.data_root, args.anno_root, "val", transform=train_transform()
        )
        train_ds.records = train_ds.records + filter_records_by_domains(
            val_for_merge.records, train_domains
        )

    val_ds = UPARTask2Dataset(
        args.data_root, args.anno_root, args.val_split, transform=train_transform()
    )
    val_domains = parse_domains(args.val_domains)
    val_ds.records = filter_records_by_domains(val_ds.records, val_domains)

    print(
        f"train={len(train_ds)} domains={train_domains or 'all'} | "
        f"val={len(val_ds)} domains={val_domains or 'all'}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_par_model(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    use_amp = (args.amp or not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, args, scaler, ema)
        eval_model = ema.shadow if ema is not None else model
        vl = val_loss(eval_model, val_loader, device, args)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}/{args.epochs} train={tr:.4f} val={vl:.4f} lr={lr:.2e}", flush=True)
        history.append({"epoch": epoch, "train": tr, "val": vl, "lr": lr})

        ckpt_body = {
            "model": eval_model.state_dict(),
            "args": {
                "model_type": "par",
                "backbone": args.backbone,
                "dropout": args.dropout,
                "train_domains": train_domains,
                "val_domains": val_domains,
            },
            "epoch": epoch,
        }
        torch.save(ckpt_body, out_dir / "last.pt")
        if vl < best_val:
            best_val = vl
            torch.save(ckpt_body, out_dir / "model.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'model.pt'} best_val={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()

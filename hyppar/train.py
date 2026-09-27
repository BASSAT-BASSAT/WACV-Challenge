"""Train HypPAR image-query or hyperbolic prototype / entailment models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from hyppar.data.dataset import UPARTask2Dataset
from hyppar.losses import attribute_bce_from_logits, batch_ranking_loss
from hyppar.models.hyppar import AttributePrototypeModel, ImageQueryModel
from hyppar.models.lorentz import euclidean_distance, lorentz_distance
from hyppar.paths import ANNO_ROOT, CKPT_DIR, DATA_ROOT, train_transform


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    freeze = not getattr(args, "unfreeze", False)
    backbone = getattr(args, "backbone", "convnext_tiny")
    common = dict(
        geometry=args.geometry,
        embed_dim=args.embed_dim,
        curvature=args.curvature,
        learnable_c=args.learnable_c,
        freeze_backbone=freeze,
        pretrained=not args.no_pretrained,
        backbone=backbone,
    )
    if args.model == "image_query":
        model = ImageQueryModel(**common)
    else:
        model = AttributePrototypeModel(
            **common,
            composition=args.composition,
            score_mode=args.score_mode,
            entail_K=args.entail_k,
            entail_beta=args.entail_beta,
            radius_compose=not args.no_radius_compose,
            hybrid_mix=getattr(args, "hybrid_mix", 0.7),
        )
    stages = getattr(args, "unfreeze_stages", 0)
    if stages > 0 and hasattr(model, "encoder"):
        model.encoder.freeze_all()
        model.encoder.unfreeze_last_stages(stages)
    return model


def distance_fn_for(model: torch.nn.Module):
    if isinstance(model, AttributePrototypeModel):
        return model.retrieval_distance_fn()
    if getattr(model, "geometry", "euclidean") == "euclidean":
        return euclidean_distance

    def _d(a, b):
        c = model.img_proj.c if hasattr(model, "img_proj") else 1.0
        return lorentz_distance(a, b, c=c)

    return _d


def backbone_trainable(model: torch.nn.Module) -> bool:
    enc = getattr(model, "encoder", None)
    if enc is None:
        return False
    return any(p.requires_grad for p in enc.parameters())


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_ids = set()
    if hasattr(model, "encoder"):
        backbone_ids = {id(p) for p in model.encoder.parameters()}
    bb, head = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (bb if id(p) in backbone_ids else head).append(p)
    groups = []
    if head:
        groups.append({"params": head, "lr": args.lr})
    if bb:
        groups.append({"params": bb, "lr": args.backbone_lr})
    if not groups:
        raise RuntimeError("No trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=5e-4)


def train_one_epoch(model, loader, optimizer, device, args, scaler):
    model.train()
    if hasattr(model, "encoder") and not args.unfreeze:
        model.encoder.eval()
        if args.unfreeze_stages > 0:
            stages = list(model.encoder.features.children())
            for stage in stages[-args.unfreeze_stages :]:
                stage.train()

    total = 0.0
    n = 0
    dist_fn = distance_fn_for(model)
    use_amp = bool(args.amp) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        attrs = batch["attrs"].to(device, non_blocking=True)
        queries = batch["query"].to(device, non_blocking=True)
        sids = batch["semantic_id"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = torch.zeros((), device=device)
            if args.model == "image_query":
                h_img = model.encode_image(images)
                h_q = model.encode_query(queries)
                loss = loss + batch_ranking_loss(h_q, h_img, sids, dist_fn, temperature=args.tau)
            else:
                feat = model.encode_features(images)
                # Primary: strong Euclidean PAR head (keeps method competitive).
                attr_logits = model.attribute_logits_from_features(feat)
                loss = loss + args.lambda_attr * attribute_bce_from_logits(attr_logits, attrs)
                # Hyperbolic path: project + compose + entailment / distance ranking.
                h_img = model.img_proj(feat)
                if model.geometry == "lorentz" and getattr(model, "radius_compose", False):
                    dens = torch.sigmoid(attr_logits).sum(dim=-1, keepdim=True) / 40.0
                    dens = dens.clamp(0.15, 1.0)
                    spatial = h_img[..., 1:] * dens
                    c = model.c
                    x0 = torch.sqrt((1.0 / c) + (spatial * spatial).sum(dim=-1, keepdim=True))
                    h_img = torch.cat([x0, spatial], dim=-1)
                h_q = model.compose_query(queries)
                if args.lambda_rank > 0:
                    loss = loss + args.lambda_rank * batch_ranking_loss(
                        h_q, h_img, sids, dist_fn, temperature=args.tau
                    )

        loss_for_backward = loss / args.grad_accum_steps
        if use_amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

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
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def embed_split(model, loader, device):
    model.eval()
    embs, attrs, sids, domains, rels, attr_probs = [], [], [], [], [], []
    for batch in tqdm(loader, desc="embed", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if isinstance(model, AttributePrototypeModel):
            feat = model.encode_features(images)
            logits = model.attribute_logits_from_features(feat)
            attr_probs.append(torch.sigmoid(logits).cpu())
            h = model.encode_image(images)
        else:
            h = model.encode_image(images)
        embs.append(h.cpu())
        attrs.append(batch["attrs"])
        sids.append(batch["semantic_id"])
        domains.extend(batch["domain"])
        rels.extend(batch["rel"])
    out_probs = torch.cat(attr_probs, dim=0).numpy() if attr_probs else None
    return (
        torch.cat(embs, dim=0),
        torch.cat(attrs, dim=0).numpy(),
        torch.cat(sids, dim=0).numpy(),
        domains,
        rels,
        out_probs,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["image_query", "prototypes"], default="prototypes")
    p.add_argument("--geometry", choices=["euclidean", "lorentz"], default="lorentz")
    p.add_argument("--composition", choices=["mean", "learned"], default="learned")
    p.add_argument(
        "--score-mode",
        choices=["entailment", "distance", "attr_l1", "hybrid"],
        default="entailment",
        help="Retrieval score (entailment = hyperbolic main contribution)",
    )
    p.add_argument("--entail-k", type=float, default=0.1, help="Cone aperture constant K")
    p.add_argument("--entail-beta", type=float, default=1.0, help="Weight on entailment energy")
    p.add_argument("--hybrid-mix", type=float, default=0.7, help="Hyperbolic weight in hybrid mode")
    p.add_argument("--no-radius-compose", action="store_true")
    p.add_argument(
        "--backbone",
        choices=["convnext_tiny", "convnext_small", "convnext_base"],
        default="convnext_tiny",
    )
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--learnable-c", action="store_true")
    p.add_argument("--unfreeze", action="store_true")
    p.add_argument("--unfreeze-stages", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cosine", action="store_true", help="Cosine anneal LR over epochs")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lambda-attr", type=float, default=2.0, help="Weight on PAR BCE (primary)")
    p.add_argument("--lambda-rank", type=float, default=1.0, help="Weight on hyperbolic ranking")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--name", type=str, default="")
    p.add_argument("--init-ckpt", type=str, default="")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--anno-root", type=str, default=str(ANNO_ROOT))
    args = p.parse_args()
    if args.grad_accum_steps < 1:
        p.error("--grad-accum-steps must be >= 1")

    # Euclid cannot use entailment cones — coerce to distance for fair twin.
    if args.geometry == "euclidean" and args.score_mode == "entailment":
        print("note: euclidean geometry -> score_mode=distance", flush=True)
        args.score_mode = "distance"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or (
        f"{args.model}_{args.geometry}_{args.composition}_{args.score_mode}_{args.backbone}"
    )
    out_dir = CKPT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = UPARTask2Dataset(
        args.data_root, args.anno_root, "train", transform=train_transform(), require_files=True
    )
    if args.max_train_samples > 0:
        train_ds.records = train_ds.records[: args.max_train_samples]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = build_model(args).to(device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"warm-start {args.init_ckpt} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable {n_train/1e6:.2f}M / {n_total/1e6:.2f}M | "
        f"backbone={args.backbone} geom={args.geometry} score={args.score_mode} amp={args.amp}",
        flush=True,
    )
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scheduler = None
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss = train_one_epoch(model, train_loader, optimizer, device, args, scaler)
        if scheduler is not None:
            scheduler.step()
        print(f"epoch {epoch}/{args.epochs} loss={loss:.4f}", flush=True)
        history.append({"epoch": epoch, "loss": loss})
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
            out_dir / "last.pt",
        )
        if device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            print(f"peak_vram_gb={alloc:.2f} reserved_gb={reserved:.2f}", flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()

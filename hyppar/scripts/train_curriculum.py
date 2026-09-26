#!/usr/bin/env python3
"""Clean from-scratch two-stage curriculum for HypPAR (paper-reproducible).

Stage A: frozen ImageNet backbone — train PAR head + prototypes / hyperbolic heads.
Stage B: unfreeze backbone (full or last N stages) — continue from Stage A only.

No warm-start from older ad-hoc experiment chains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()})}
    env["PYTHONPATH"] = str(REPO) + (
        (":" if not sys.platform.startswith("win") else ";") + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else ""
    )
    # Prefer replacing PYTHONPATH entirely with repo root first
    env["PYTHONPATH"] = str(REPO)
    subprocess.check_call(cmd, cwd=str(REPO), env=env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", type=str, required=True, help="Run name under checkpoints/")
    p.add_argument("--geometry", choices=["euclidean", "lorentz"], default="lorentz")
    p.add_argument("--score-mode", default="entailment")
    p.add_argument("--composition", default="learned")
    p.add_argument("--backbone", choices=["convnext_tiny", "convnext_small"], default="convnext_tiny")
    p.add_argument("--stage-a-epochs", type=int, default=5)
    p.add_argument("--stage-b-epochs", type=int, default=10)
    p.add_argument("--batch-size-a", type=int, default=64)
    p.add_argument("--batch-size-b", type=int, default=32)
    p.add_argument("--lr-a", type=float, default=1e-3)
    p.add_argument("--lr-b", type=float, default=5e-4)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--lambda-attr", type=float, default=3.0)
    p.add_argument("--lambda-rank", type=float, default=1.0)
    p.add_argument("--entail-k", type=float, default=0.1)
    p.add_argument("--entail-beta", type=float, default=1.5)
    p.add_argument("--hybrid-mix", type=float, default=0.7)
    p.add_argument(
        "--unfreeze-mode",
        choices=["full", "stages"],
        default="full",
        help="Stage B: full backbone or last N stages",
    )
    p.add_argument("--unfreeze-stages", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--skip-a", action="store_true", help="Reuse existing stage_a.pt")
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()

    ckpt_root = REPO / "checkpoints" / args.name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    stage_a = ckpt_root / "stage_a.pt"
    final = ckpt_root / "model.pt"

    meta = {
        "protocol": "clean_curriculum_v1",
        "stage_a": "frozen ImageNet backbone",
        "stage_b": f"unfreeze {args.unfreeze_mode}",
        "args": vars(args),
    }
    (ckpt_root / "protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    common = [
        PY,
        "-m",
        "hyppar.train",
        "--model",
        "prototypes",
        "--geometry",
        args.geometry,
        "--composition",
        args.composition,
        "--score-mode",
        args.score_mode,
        "--backbone",
        args.backbone,
        "--amp",
        "--cosine",
        "--lambda-attr",
        str(args.lambda_attr),
        "--lambda-rank",
        str(args.lambda_rank),
        "--entail-k",
        str(args.entail_k),
        "--entail-beta",
        str(args.entail_beta),
        "--hybrid-mix",
        str(args.hybrid_mix),
        "--num-workers",
        str(args.num_workers),
    ]

    if not args.skip_a:
        # Stage A: frozen (no --unfreeze)
        run(
            common
            + [
                "--name",
                f"{args.name}/_stage_a_run",
                "--epochs",
                str(args.stage_a_epochs),
                "--batch-size",
                str(args.batch_size_a),
                "--lr",
                str(args.lr_a),
                "--backbone-lr",
                str(args.backbone_lr),
            ]
        )
        src = REPO / "checkpoints" / f"{args.name}/_stage_a_run" / "model.pt"
        stage_a.write_bytes(src.read_bytes())
        print(f"stage A -> {stage_a}", flush=True)

    if not args.skip_b:
        b_flags = ["--unfreeze"] if args.unfreeze_mode == "full" else [
            "--unfreeze-stages",
            str(args.unfreeze_stages),
        ]
        run(
            common
            + b_flags
            + [
                "--name",
                f"{args.name}/_stage_b_run",
                "--epochs",
                str(args.stage_b_epochs),
                "--batch-size",
                str(args.batch_size_b),
                "--lr",
                str(args.lr_b),
                "--backbone-lr",
                str(args.backbone_lr),
                "--init-ckpt",
                str(stage_a),
            ]
        )
        src = REPO / "checkpoints" / f"{args.name}/_stage_b_run" / "model.pt"
        # Rewrite args.name to the clean experiment name for packaging/eval.
        import torch

        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        ckpt["args"]["name"] = args.name
        ckpt["protocol"] = meta
        torch.save(ckpt, final)
        print(f"stage B -> {final}", flush=True)

    print("CURRICULUM_DONE", args.name, flush=True)


if __name__ == "__main__":
    main()

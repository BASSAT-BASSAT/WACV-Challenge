#!/usr/bin/env python3
"""Leave-one-domain-out evaluation for PAR retrieval (honest cross-domain proxy)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(REPO)
    subprocess.check_call(cmd, cwd=str(REPO), env=env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", type=str, default="convnext_base")
    p.add_argument("--epochs", type=int, default=8, help="Short LODO runs for model selection")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--name-prefix", type=str, default="lodo_par")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    from hyppar.data.domains import DOMAINS, lodo_folds

    summary: dict = {"folds": {}, "mean_mADM": 0.0}
    madms = []

    for held, (d1, d2) in lodo_folds():
        fold_name = f"{args.name_prefix}_{held.lower()}"
        ckpt = REPO / "checkpoints" / fold_name / "model.pt"
        if not args.skip_train:
            run(
                [
                    PY,
                    "-m",
                    "hyppar.scripts.train_par",
                    "--name",
                    fold_name,
                    "--backbone",
                    args.backbone,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--train-domains",
                    f"{d1},{d2}",
                    "--val-domains",
                    held,
                    "--train-split",
                    "train",
                    "--val-split",
                    "val",
                ]
            )
        eval_out = REPO / "checkpoints" / fold_name / "eval_lodo.json"
        run(
            [
                PY,
                "-m",
                "hyppar.scripts.evaluate_par",
                "--ckpt",
                str(ckpt),
                "--split",
                "val",
                "--calib-split",
                "val",
                "--out",
                str(eval_out),
                "--save-artifacts",
            ]
        )
        fold_res = json.loads(eval_out.read_text(encoding="utf-8"))
        madm = fold_res["overall"]["mADM"]
        madms.append(madm)
        summary["folds"][held] = {
            "train_domains": [d1, d2],
            "held_out": held,
            "mADM": madm,
            "mAP": fold_res["overall"]["mAP"],
            "R1": fold_res["overall"]["R1"],
            "by_domain": fold_res.get("by_domain", {}),
        }
        print(f"LODO {held}: mADM={madm:.2f}", flush=True)

    summary["mean_mADM"] = float(sum(madms) / max(len(madms), 1))
    summary["domains"] = list(DOMAINS)
    out = REPO / "checkpoints" / f"{args.name_prefix}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"mean LODO mADM={summary['mean_mADM']:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

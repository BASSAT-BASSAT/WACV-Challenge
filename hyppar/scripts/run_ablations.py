"""Run the paper ablation matrix and collect validation metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WACV = ROOT.parent
PY = sys.executable
CKPT = WACV / "checkpoints"


EXPERIMENTS = [
    dict(name="tiny_euclid_mean", backbone="convnext_tiny", geometry="euclidean", composition="mean", score_mode="distance", unfreeze_mode="stages"),
    dict(name="tiny_lorentz_distance", backbone="convnext_tiny", geometry="lorentz", composition="learned", score_mode="distance", unfreeze_mode="stages"),
    dict(name="tiny_lorentz_entailment", backbone="convnext_tiny", geometry="lorentz", composition="learned", score_mode="entailment", unfreeze_mode="stages"),
    dict(name="tiny_lorentz_hybrid", backbone="convnext_tiny", geometry="lorentz", composition="learned", score_mode="hybrid", unfreeze_mode="stages"),
    dict(name="small_lorentz_hybrid", backbone="convnext_small", geometry="lorentz", composition="learned", score_mode="hybrid", unfreeze_mode="stages"),
    dict(name="base_lorentz_hybrid", backbone="convnext_base", geometry="lorentz", composition="learned", score_mode="hybrid", unfreeze_mode="stages"),
    dict(name="base_lorentz_hybrid_full", backbone="convnext_base", geometry="lorentz", composition="learned", score_mode="hybrid", unfreeze_mode="full"),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(WACV), env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(WACV)})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--epochs-a", type=int, default=3)
    p.add_argument("--epochs-b", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--package", action="store_true", help="Build one named submission zip per checkpoint")
    p.add_argument("--only", type=str, default="", help="comma-separated experiment names")
    args = p.parse_args()

    exps = EXPERIMENTS
    if args.only:
        allow = set(args.only.split(","))
        exps = [e for e in exps if e["name"] in allow]

    summary = {}
    for e in exps:
        ckpt = CKPT / e["name"] / "model.pt"
        train_cmd = [
            PY,
            "-m",
            "hyppar.scripts.train_curriculum",
            "--name",
            e["name"],
            "--geometry",
            e["geometry"],
            "--score-mode",
            e["score_mode"],
            "--composition",
            e["composition"],
            "--backbone",
            e["backbone"],
            "--stage-a-epochs",
            str(args.epochs_a),
            "--stage-b-epochs",
            str(args.epochs_b),
            "--batch-size-a",
            str(args.batch_size),
            "--batch-size-b",
            str(args.batch_size),
            "--grad-accum-a",
            str(args.grad_accum),
            "--grad-accum-b",
            str(args.grad_accum),
            "--lambda-attr",
            "4",
            "--lambda-rank",
            "1",
            "--unfreeze-mode",
            e["unfreeze_mode"],
            "--unfreeze-stages",
            "2",
            "--num-workers",
            str(args.num_workers),
        ]
        eval_cmd = [
            PY,
            "-m",
            "hyppar.evaluate",
            "--ckpt",
            str(ckpt),
            "--score-mode",
            e["score_mode"],
            "--out",
            str(eval_out := CKPT / e["name"] / f"eval_val_{e['score_mode']}.json"),
            "--batch-size",
            "64",
            "--num-workers",
            str(args.num_workers),
        ]
        if args.dry_run:
            print("TRAIN +", " ".join(train_cmd))
            print("EVAL  +", " ".join(eval_cmd))
            continue
        if not args.skip_train:
            run(train_cmd)
        if not ckpt.is_file():
            print(f"missing {ckpt}, skip eval")
            continue
        run(eval_cmd)
        summary[e["name"]] = json.loads(eval_out.read_text(encoding="utf-8"))
        if args.package:
            package_out = CKPT / "submissions" / f"{e['name']}.zip"
            run(
                [
                    PY,
                    "-m",
                    "hyppar.scripts.package_submission",
                    "--ckpt",
                    str(ckpt),
                    "--out",
                    str(package_out),
                ]
            )

    if args.dry_run:
        return

    out = CKPT / "ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown table for paper draft
    lines = [
        "| Model | mADM | mAP | R-1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, res in summary.items():
        o = res["overall"]
        lines.append(
            f"| {name} | {100*o['mADM']:.2f} | {100*o['mAP']:.2f} | {100*o['R1']:.2f} |"
        )
    md = CKPT / "ablation_summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md.read_text(encoding="utf-8"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

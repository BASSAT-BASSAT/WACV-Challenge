"""Run the full experiment matrix and collect eval JSONs into a summary table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WACV = ROOT.parent
PY = sys.executable
CKPT = ROOT / "checkpoints"


EXPERIMENTS = [
    dict(model="image_query", geometry="euclidean", composition="mean", name="iq_euclid"),
    dict(model="image_query", geometry="lorentz", composition="mean", name="iq_lorentz"),
    dict(
        model="prototypes",
        geometry="euclidean",
        composition="mean",
        name="proto_euclid_mean",
    ),
    dict(
        model="prototypes",
        geometry="lorentz",
        composition="mean",
        name="proto_lorentz_mean",
    ),
    dict(
        model="prototypes",
        geometry="lorentz",
        composition="learned",
        name="proto_lorentz_learned",
    ),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(WACV), env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(WACV)})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--only", type=str, default="", help="comma-separated experiment names")
    args = p.parse_args()

    exps = EXPERIMENTS
    if args.only:
        allow = set(args.only.split(","))
        exps = [e for e in exps if e["name"] in allow]

    summary = {}
    for e in exps:
        ckpt = CKPT / e["name"] / "model.pt"
        if not args.skip_train:
            cmd = [
                PY,
                "-m",
                "hyppar.train",
                "--model",
                e["model"],
                "--geometry",
                e["geometry"],
                "--composition",
                e["composition"],
                "--name",
                e["name"],
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                "2",
            ]
            run(cmd)
        if not ckpt.is_file():
            print(f"missing {ckpt}, skip eval")
            continue
        eval_out = CKPT / e["name"] / "eval_val.json"
        run(
            [
                PY,
                "-m",
                "hyppar.evaluate",
                "--ckpt",
                str(ckpt),
                "--out",
                str(eval_out),
                "--batch-size",
                "64",
                "--num-workers",
                "2",
            ]
        )
        summary[e["name"]] = json.loads(eval_out.read_text(encoding="utf-8"))

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

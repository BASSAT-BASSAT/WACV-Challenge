"""Smoke-test sample prior baseline + metrics on val (annotation-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from hyppar.data.dataset import load_split_tables
from hyppar.eval.metrics import evaluate_ranking
from hyppar.paths import ANNO_ROOT, CKPT_DIR, UPAR_ROOT


def main() -> None:
    gt, ids, queries = load_split_tables(ANNO_ROOT, "val")
    attr_cols = [c for c in gt.columns if c != "image"]
    gallery_attrs = gt[attr_cols].to_numpy(dtype=np.float32)
    gallery_ids = ids["semantic_id"].to_numpy(dtype=np.int64)
    domains = [str(p).replace("\\", "/").split("/")[0] for p in gt["image"].tolist()]

    # Official-style L1 to attribute prior
    prior_path = (
        UPAR_ROOT
        / "examples"
        / "task2"
        / "sample_code_submission"
        / "assets"
        / "attribute_prior.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    rates = np.asarray(prior["positive_rate"], dtype=np.float32)
    # distances = sum|q-p| via formula in sample run.py
    # probs tiled: each gallery row = rates
    probs = np.tile(rates[None, :], (gallery_attrs.shape[0], 1))
    distances = probs.sum(1)[None, :] + queries @ (1.0 - 2.0 * probs).T

    results = evaluate_ranking(distances, queries, gallery_attrs, gallery_ids, domains)
    out = CKPT_DIR / "baseline_prior_eval_val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["overall"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

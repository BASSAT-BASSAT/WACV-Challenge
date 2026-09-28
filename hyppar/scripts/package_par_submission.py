"""Package Codabench zip for calibrated PAR retrieval."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SUB = ROOT / "submission"

INCLUDE_PY = [
    ROOT / "__init__.py",
    ROOT / "models" / "__init__.py",
    ROOT / "models" / "par_classifier.py",
    ROOT / "retrieval" / "__init__.py",
    ROOT / "retrieval" / "calibration.py",
    ROOT / "retrieval" / "scoring.py",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--calibration", type=str, default="", help="calibration.json from evaluate_par --save-artifacts")
    p.add_argument("--weights", type=str, default="", help="attr_weights.json")
    p.add_argument(
        "--out",
        type=str,
        default=str(REPO / "checkpoints" / "par_task2_submission.zip"),
    )
    args = p.parse_args()

    assets = SUB / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.ckpt, assets / "model.pt")

    ckpt_dir = Path(args.ckpt).parent
    calib_src = Path(args.calibration) if args.calibration else ckpt_dir / "calibration.json"
    weights_src = Path(args.weights) if args.weights else ckpt_dir / "attr_weights.json"
    if calib_src.is_file():
        shutil.copy2(calib_src, assets / "calibration.json")
    if weights_src.is_file():
        shutil.copy2(weights_src, assets / "attr_weights.json")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUB / "run_par.py", arcname="run.py")
        zf.write(SUB / "metadata.yaml", arcname="metadata.yaml")
        zf.write(assets / "model.pt", arcname="assets/model.pt")
        if (assets / "calibration.json").is_file():
            zf.write(assets / "calibration.json", arcname="assets/calibration.json")
        if (assets / "attr_weights.json").is_file():
            zf.write(assets / "attr_weights.json", arcname="assets/attr_weights.json")
        for path in INCLUDE_PY:
            rel = path.relative_to(REPO)
            zf.write(path, arcname=str(rel).replace("\\", "/"))

    print(f"wrote {out}")
    with zipfile.ZipFile(out, "r") as zf:
        print("entries:", ", ".join(sorted(zf.namelist())))


if __name__ == "__main__":
    main()

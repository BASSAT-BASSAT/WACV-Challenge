"""Package Codabench zip from a trained checkpoint.

Zip *contents* so run.py is at the archive root (Codabench requirement).
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # hyppar package
REPO = ROOT.parent
SUB = ROOT / "submission"

INCLUDE_PY = [
    ROOT / "__init__.py",
    ROOT / "models" / "__init__.py",
    ROOT / "models" / "hyppar.py",
    ROOT / "models" / "lorentz.py",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--out",
        type=str,
        default=str(REPO / "checkpoints" / "hyppar_task2_submission.zip"),
    )
    args = p.parse_args()

    assets = SUB / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.ckpt, assets / "model.pt")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUB / "run.py", arcname="run.py")
        zf.write(SUB / "metadata.yaml", arcname="metadata.yaml")
        zf.write(assets / "model.pt", arcname="assets/model.pt")
        for path in INCLUDE_PY:
            rel = path.relative_to(REPO)
            zf.write(path, arcname=str(rel).replace("\\", "/"))
    print(f"wrote {out}")
    with zipfile.ZipFile(out, "r") as zf:
        print("entries:", ", ".join(sorted(zf.namelist())))


if __name__ == "__main__":
    main()

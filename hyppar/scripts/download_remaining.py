"""Continue UPAR dataset downloads (Market already present)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPAR = ROOT.parent / "UPAR-Challenge-2027"
spec = importlib.util.spec_from_file_location("download_datasets", UPAR / "download_datasets.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["download_datasets"] = mod
# Script expects peta_file_mapping.txt in CWD
import os

os.chdir(UPAR)
spec.loader.exec_module(mod)


def main() -> None:
    data = UPAR / "data"
    data.mkdir(parents=True, exist_ok=True)
    print("=== PA100K ===")
    mod.prepare_pa100k(data)
    print("=== PETA ===")
    mod.prepare_peta(data)
    print("done")


if __name__ == "__main__":
    main()

"""Clone UPAR-Challenge-2027 and download Market / PA100K / PETA images."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_UPAR = Path(os.environ.get("HYPPAR_UPAR_ROOT", REPO.parent / "UPAR-Challenge-2027"))
UPAR_GIT = os.environ.get("UPAR_GIT_URL", "https://github.com/speckean/UPAR-Challenge-2027.git")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upar-root", type=str, default=str(DEFAULT_UPAR))
    p.add_argument("--skip-clone", action="store_true")
    p.add_argument("--skip-download", action="store_true", help="Only clone (annotations)")
    args = p.parse_args()

    upar = Path(args.upar_root).resolve()
    if not args.skip_clone and not (upar / "download_datasets.py").is_file():
        upar.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {UPAR_GIT} -> {upar}")
        subprocess.check_call(["git", "clone", UPAR_GIT, str(upar)])
    elif not (upar / "download_datasets.py").is_file():
        raise SystemExit(f"Missing {upar}/download_datasets.py — clone UPAR challenge first")

    if args.skip_download:
        print("skip download; annotations should be under", upar / "data" / "annotations")
        print(f"export HYPPAR_UPAR_ROOT={upar}")
        return

    # gdown required
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "tqdm", "numpy"])

    data = upar / "data"
    data.mkdir(parents=True, exist_ok=True)
    os.chdir(upar)  # peta_file_mapping.txt is resolved relative to CWD

    spec = importlib.util.spec_from_file_location("download_datasets", upar / "download_datasets.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("=== downloading Market1501 + PA100K + PETA into", data, "===")
    mod.prepare_datasets(str(data))
    print("DATA_DOWNLOAD_DONE")
    print()
    print("Next:")
    print(f"  export HYPPAR_UPAR_ROOT={upar}")
    print(f"  export PYTHONPATH={REPO}")
    print("  make smoke && make train-main")


if __name__ == "__main__":
    main()

"""Windows-safe PETA extraction + remapping for UPAR Challenge."""

from __future__ import annotations

import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm import tqdm

UPAR = Path(__file__).resolve().parents[1].parent / "UPAR-Challenge-2027"
DATA = UPAR / "data"
PETA = DATA / "PETA"
ZIP = PETA / "peta.zip"
MAP = UPAR / "peta_file_mapping.txt"
URL = "https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?dl=1"


def download() -> None:
    PETA.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and ZIP.stat().st_size > 1_000_000:
        print(f"zip exists: {ZIP}")
        return
    print(f"Downloading {URL}")
    start = time.time()

    def hook(count, block, total):
        if count == 0:
            return
        done = count * block
        pct = min(100, int(done * 100 / max(total, 1)))
        sys.stdout.write(f"\r{pct}% {done/1e6:.1f}MB {time.time()-start:.0f}s")
        sys.stdout.flush()

    urllib.request.urlretrieve(URL, ZIP, hook)
    print()


def extract() -> None:
    print("Extracting PETA zip")
    with zipfile.ZipFile(ZIP, "r") as zf:
        for member in tqdm(zf.infolist(), desc="extract"):
            zf.extract(member, PETA)


def remap() -> None:
    print("Loading mapping")
    rows = np.genfromtxt(MAP, dtype=str, delimiter=",")
    mapping = {row[0].replace("\\", "/"): row[1].replace("\\", "/") for row in rows}

    img_out = PETA / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0
    for file in tqdm(list(PETA.rglob("*")), desc="remap"):
        if not file.is_file():
            continue
        if file.suffix.lower() in {".txt", ".zip", ".md"}:
            continue
        if "images" in file.parts and file.parent == img_out:
            continue
        try:
            rel = file.relative_to(DATA).as_posix()  # PETA/PETA dataset/...
        except ValueError:
            continue
        target_rel = mapping.get(rel)
        if target_rel is None:
            missing += 1
            continue
        dest = DATA / Path(target_rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        shutil.move(str(file), str(dest))
        moved += 1
    print(f"moved={moved} unmatched_files={missing}")
    print(f"images dir count={len(list(img_out.glob('*')))}")


def main() -> None:
    download()
    # Only extract if images not already populated
    if not (PETA / "images").exists() or len(list((PETA / "images").glob("*"))) < 1000:
        extract()
        remap()
    else:
        print("PETA images already present")


if __name__ == "__main__":
    main()

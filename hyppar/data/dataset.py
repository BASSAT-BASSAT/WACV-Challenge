"""UPAR Challenge Track 2 dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

ATTRIBUTE_NAMES: list[str] = [
    "Age-Young",
    "Age-Adult",
    "Age-Old",
    "Gender-Female",
    "Hair-Length-Short",
    "Hair-Length-Long",
    "Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black",
    "UpperBody-Color-Blue",
    "UpperBody-Color-Brown",
    "UpperBody-Color-Green",
    "UpperBody-Color-Grey",
    "UpperBody-Color-Orange",
    "UpperBody-Color-Pink",
    "UpperBody-Color-Purple",
    "UpperBody-Color-Red",
    "UpperBody-Color-White",
    "UpperBody-Color-Yellow",
    "UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black",
    "LowerBody-Color-Blue",
    "LowerBody-Color-Brown",
    "LowerBody-Color-Green",
    "LowerBody-Color-Grey",
    "LowerBody-Color-Orange",
    "LowerBody-Color-Pink",
    "LowerBody-Color-Purple",
    "LowerBody-Color-Red",
    "LowerBody-Color-White",
    "LowerBody-Color-Yellow",
    "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack",
    "Accessory-Bag",
    "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun",
    "Accessory-Hat",
]

# Soft hierarchy categories for optional entailment / analysis.
ATTRIBUTE_CATEGORIES: dict[str, list[int]] = {
    "age": [0, 1, 2],
    "gender": [3],
    "hair": [4, 5, 6],
    "ub_length": [7],
    "ub_color": list(range(8, 20)),
    "lb_length": [20],
    "lb_color": list(range(21, 33)),
    "lb_type": [33, 34],
    "accessory": [35, 36, 37, 38, 39],
}


def _read_csv(path: Path) -> pd.DataFrame:
    # Challenge CSVs start with "# image,..." or "# Age-..."
    return pd.read_csv(path, comment=None)


def load_split_tables(anno_root: Path | str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load gt.csv, ids.csv, queries.csv for train or val."""
    root = Path(anno_root) / "task2" / split
    gt = pd.read_csv(root / "gt.csv")
    # First column is "# image" or "image"; remaining are 40 attributes.
    gt = gt.rename(columns={gt.columns[0]: "image"})
    gt["image"] = gt["image"].astype(str).str.strip()

    # Clean attribute column names that may start with "# "
    rename_attrs = {c: c.lstrip("# ").strip() for c in gt.columns if c != "image"}
    gt = gt.rename(columns=rename_attrs)

    ids = pd.read_csv(root / "ids.csv", header=None, names=["image", "semantic_id"])
    ids["image"] = ids["image"].astype(str).str.strip()

    queries = pd.read_csv(root / "queries.csv")
    queries.columns = [c.lstrip("# ").strip() for c in queries.columns]
    query_matrix = queries.to_numpy(dtype=np.float32)
    if query_matrix.shape[1] != 40:
        raise ValueError(f"Expected 40 query attributes, got {query_matrix.shape[1]}")
    return gt, ids, query_matrix


def resolve_image_path(data_root: Path, rel_path: str) -> Path:
    """Map annotation-relative paths onto downloaded dataset folders."""
    p = Path(rel_path.replace("\\", "/"))
    candidates = [
        data_root / p,
        data_root / "Market1501" / p.name if "Market" in str(p) else None,
    ]
    # PA100k images often live under release_data/release_data/
    if str(p).startswith("PA100k/") or str(p).startswith("PA100K/"):
        name = p.name
        candidates.extend(
            [
                data_root / "PA100k" / "release_data" / "release_data" / name,
                data_root / "PA100k" / "release_data" / name,
                data_root / "PA100k" / name,
            ]
        )
    if str(p).startswith("PETA/") or "PETA" in str(p):
        candidates.append(data_root / "PETA" / "images" / p.name)
        candidates.append(data_root / p)

    for c in candidates:
        if c is not None and c.is_file():
            return c
    return data_root / p


class UPARTask2Dataset(Dataset):
    """Image-level dataset for training HypPAR."""

    def __init__(
        self,
        data_root: Path | str,
        anno_root: Path | str,
        split: str = "train",
        transform: Callable | None = None,
        require_files: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.anno_root = Path(anno_root)
        self.split = split
        self.transform = transform

        gt, ids, queries = load_split_tables(self.anno_root, split)
        id_map = dict(zip(ids["image"].astype(str), ids["semantic_id"].astype(int)))

        attr_cols = [c for c in gt.columns if c != "image"]
        if len(attr_cols) != 40:
            raise ValueError(f"Expected 40 attribute columns, found {len(attr_cols)}")

        records: list[dict] = []
        missing = 0
        for _, row in gt.iterrows():
            rel = str(row["image"]).strip()
            path = resolve_image_path(self.data_root, rel)
            if require_files and not path.is_file():
                missing += 1
                continue
            sid = int(id_map[rel])
            attrs = row[attr_cols].to_numpy(dtype=np.float32)
            domain = rel.split("/")[0] if "/" in rel else rel.split("\\")[0]
            records.append(
                {
                    "path": path,
                    "rel": rel,
                    "attrs": attrs,
                    "semantic_id": sid,
                    "domain": domain,
                    "query": queries[sid],
                }
            )
        if require_files and missing and not records:
            raise FileNotFoundError(
                f"No images found under {self.data_root}. "
                f"Missing {missing} paths. Run download_datasets.py first."
            )
        self.records = records
        self.queries = queries
        self.attr_cols = attr_cols
        self.missing = missing

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        img = Image.open(rec["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "attrs": torch.from_numpy(rec["attrs"]),
            "semantic_id": rec["semantic_id"],
            "query": torch.from_numpy(rec["query"].astype(np.float32)),
            "domain": rec["domain"],
            "rel": rec["rel"],
        }


def domain_of(rel_path: str) -> str:
    return rel_path.replace("\\", "/").split("/")[0]

"""Package Codabench zip for calibrated PAR retrieval.

Competition requirements (ingestion program):
  - ``run.py`` must sit at the archive ROOT (it imports it and calls
    ``rank_gallery(sample)``).
  - ``metadata.yaml`` at the archive root (marks it as a code submission).
  - weights + python sources INSIDE the zip (container has no network).

Typical flow for a 30-epoch ConvNeXt run::

    python -m hyppar.scripts.train_par --name par_convnext_base --epochs 30 ...
    python -m hyppar.scripts.evaluate_par --ckpt checkpoints/par_convnext_base/model.pt --save-artifacts
    python -m hyppar.scripts.package_par_submission --ckpt checkpoints/par_convnext_base/model.pt

Use ``--ckpt .../last.pt`` if you explicitly want the final-epoch weights
instead of the best-val ``model.pt``.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SUB = ROOT / "submission"

# NOTE: hyppar.py / lorentz.py are included defensively so that even an
# eager `hyppar.models.__init__` (older checkouts) still imports inside the
# container. `hyppar/models/__init__.py` itself is lazy (PEP 562), so the
# PAR path never needs them, but they cost a few KB.
INCLUDE_PY = [
    ROOT / "__init__.py",
    ROOT / "models" / "__init__.py",
    ROOT / "models" / "par_classifier.py",
    ROOT / "models" / "hyppar.py",
    ROOT / "models" / "lorentz.py",
    ROOT / "retrieval" / "__init__.py",
    ROOT / "retrieval" / "calibration.py",
    ROOT / "retrieval" / "scoring.py",
]

REQUIRED_IN_ZIP = [
    "run.py",
    "metadata.yaml",
    "assets/model.pt",
    "hyppar/__init__.py",
    "hyppar/models/__init__.py",
    "hyppar/models/par_classifier.py",
    "hyppar/retrieval/__init__.py",
    "hyppar/retrieval/calibration.py",
    "hyppar/retrieval/scoring.py",
]


def _read_ckpt_meta(ckpt_path: Path) -> dict:
    """Best-effort checkpoint inspection without requiring torch.

    Returns {"kind": "trainer"|"state_dict"|"unknown", "epoch": ..., "args": {...}}.
    """
    meta: dict = {"kind": "unknown", "epoch": None, "args": {}}
    try:
        import torch  # type: ignore

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:  # old torch without weights_only
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            meta["kind"] = "trainer"
            meta["epoch"] = ckpt.get("epoch")
            meta["args"] = dict(ckpt.get("args") or {})
        elif isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            if any(k.startswith("features.") or k.startswith("head.") for k in keys):
                meta["kind"] = "state_dict"
        return meta
    except ImportError:
        pass  # fall through to pickle sniffing
    except Exception as exc:  # corrupt / unreadable
        meta["error"] = str(exc)
        return meta
    # No torch: sniff the pickled payload for the trainer-dict markers.
    try:
        with zipfile.ZipFile(ckpt_path) as zf:
            pkl_name = next(n for n in zf.namelist() if n.endswith(".pkl"))
            blob = zf.read(pkl_name)
        meta["kind"] = (
            "trainer"
            if b"model_type" in blob and b"backbone" in blob
            else ("state_dict" if b"features.0" in blob else "unknown")
        )
        for marker in (b"convnext_base", b"convnext_small", b"convnext_tiny"):
            if marker in blob:
                meta["args"] = {"backbone": marker.decode()}
                break
    except Exception as exc:
        meta["error"] = str(exc)
    return meta


def _normalize_ckpt(src: Path, dst: Path, backbone: str, dropout: float) -> dict:
    """Copy ``src`` to ``dst``, wrapping a raw state_dict if needed.

    ``train_par.py`` saves {"model": state_dict, "args": {...}, "epoch": N},
    which is what ``run_par.py`` expects. If the user saved a bare
    ``model.state_dict()`` (a common manual mistake after a long run), wrap it
    so the container-side ``ckpt["model"]`` / ``ckpt["args"]`` lookups work.
    """
    meta = _read_ckpt_meta(src)
    if meta.get("kind") == "state_dict":
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                f"ERROR: {src} looks like a raw state_dict, but torch is not "
                f"available here to wrap it. Re-save it as "
                "{'model': state_dict, 'args': {'backbone': ..., 'dropout': ...}} "
                "on your training machine."
            ) from exc
        try:
            state = torch.load(str(src), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(str(src), map_location="cpu")
        ckpt = {
            "model": state,
            "args": {"model_type": "par", "backbone": backbone, "dropout": dropout},
            "epoch": None,
        }
        torch.save(ckpt, str(dst))
        print(f"wrapped raw state_dict with args backbone={backbone} dropout={dropout}")
        meta = {"kind": "trainer", "epoch": None, "args": ckpt["args"]}
    elif meta.get("kind") == "trainer":
        shutil.copy2(src, dst)
    else:
        raise SystemExit(
            f"ERROR: cannot recognise checkpoint format of {src} "
            f"(meta={meta}). Expected the trainer dict saved by train_par.py "
            "({'model', 'args', 'epoch'}) or a raw state_dict."
        )
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True,
                   help="Trainer checkpoint (model.pt = best val, last.pt = final epoch)")
    p.add_argument("--calibration", type=str, default="",
                   help="calibration.json from `evaluate_par --save-artifacts`")
    p.add_argument("--weights", type=str, default="", help="attr_weights.json")
    p.add_argument("--backbone", type=str, default="convnext_base",
                   help="only used to wrap a raw state_dict checkpoint")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="only used to wrap a raw state_dict checkpoint")
    p.add_argument(
        "--out", type=str,
        default=str(REPO / "checkpoints" / "par_task2_submission.zip"),
    )
    args = p.parse_args()

    ckpt_src = Path(args.ckpt)
    if not ckpt_src.is_file():
        raise SystemExit(f"ERROR: checkpoint not found: {ckpt_src}")
    ckpt_dir = ckpt_src.parent
    calib_src = Path(args.calibration) if args.calibration else ckpt_dir / "calibration.json"
    weights_src = Path(args.weights) if args.weights else ckpt_dir / "attr_weights.json"

    # Stage everything in a temp dir: never pollute hyppar/submission/assets
    # with 350MB copies, and never silently reuse stale calibration files.
    with tempfile.TemporaryDirectory(prefix="par_pkg_") as tmp:
        stage = Path(tmp)
        (stage / "assets").mkdir(parents=True)
        meta = _normalize_ckpt(ckpt_src, stage / "assets" / "model.pt",
                               backbone=args.backbone, dropout=args.dropout)
        print(f"checkpoint: kind={meta.get('kind')} epoch={meta.get('epoch')} "
              f"args={json.dumps(meta.get('args', {}), default=str)}")

        if calib_src.is_file():
            shutil.copy2(calib_src, stage / "assets" / "calibration.json")
            print(f"calibration: {calib_src}")
        else:
            print(
                f"WARNING: no calibration.json at {calib_src}. The zip will still run "
                "(uncalibrated fallback) but scores will be worse. Generate it with:\n"
                f"  python -m hyppar.scripts.evaluate_par --ckpt {ckpt_src} --save-artifacts",
            )
        if weights_src.is_file():
            shutil.copy2(weights_src, stage / "assets" / "attr_weights.json")
            print(f"attr weights: {weights_src}")
        else:
            print(
                f"WARNING: no attr_weights.json at {weights_src} "
                "(uniform attribute weights will be used). "
                "It is written by the same `evaluate_par --save-artifacts` call."
            )

        for path in INCLUDE_PY:
            if not path.is_file():
                raise SystemExit(f"ERROR: missing source file {path}")
        if not (SUB / "run_par.py").is_file() or not (SUB / "metadata.yaml").is_file():
            raise SystemExit(f"ERROR: submission template missing in {SUB}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            # run.py MUST be at the archive root (Codabench imports it).
            zf.write(SUB / "run_par.py", arcname="run.py")
            zf.write(SUB / "metadata.yaml", arcname="metadata.yaml")
            zf.write(stage / "assets" / "model.pt", arcname="assets/model.pt")
            if (stage / "assets" / "calibration.json").is_file():
                zf.write(stage / "assets" / "calibration.json",
                         arcname="assets/calibration.json")
            if (stage / "assets" / "attr_weights.json").is_file():
                zf.write(stage / "assets" / "attr_weights.json",
                         arcname="assets/attr_weights.json")
            for path in INCLUDE_PY:
                rel = path.relative_to(REPO)
                zf.write(path, arcname=str(rel).replace("\\", "/"))

        # ---- verify the archive the way Codabench sees it ----
        with zipfile.ZipFile(out, "r") as zf:
            names = sorted(zf.namelist())
            bad = [n for n in names if "\\" in n or n.startswith("/") or n.startswith("./")]
            assert not bad, f"bad arcnames: {bad}"
            top_dirs = {n.split("/")[0] for n in names if "/" in n}
            assert "run.py" in names, "run.py missing from zip root"
            assert not any(n.endswith("run_par.py") for n in names), \
                "zip must contain run.py (not run_par.py) at root"
            missing = [r for r in REQUIRED_IN_ZIP if r not in names]
            assert not missing, f"missing entries: {missing}"
            info = {n: zf.getinfo(n).file_size for n in names}
            # smoke: byte-compile every shipped .py (catches syntax errors w/o torch)
            with tempfile.TemporaryDirectory(prefix="par_verify_") as vtmp:
                zf.extractall(vtmp)
                sys.path.insert(0, vtmp)
                try:
                    for n in names:
                        if n.endswith(".py"):
                            py_compile.compile(str(Path(vtmp) / n), doraise=True)
                    # the import chain run_par.py relies on must resolve
                    import importlib.util

                    spec = importlib.util.find_spec("hyppar.models.par_classifier")
                    assert spec is not None, "hyppar.models.par_classifier not importable from zip layout"
                finally:
                    sys.path.remove(vtmp)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        for n in names:
            print(f"  {info[n] / 1e6:8.1f} MB  {n}")


if __name__ == "__main__":
    main()

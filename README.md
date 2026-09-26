# HypPAR — Hyperbolic Pedestrian Attribute Retrieval

UPAR Challenge 2027 Track 2 (WACV / RWS).  
**Paper claim:** Lorentz **entailment** retrieval for multi-label attribute queries (sparse queries near origin, specific images deeper), with a strong PAR head and Euclidean twins.

Repo: [BASSAT-BASSAT/HYPPAR-WACV](https://github.com/BASSAT-BASSAT/HYPPAR-WACV)

## Clean training protocol (use this for the paper)

All reported runs use the **same from-scratch curriculum** (no ad-hoc warm-start chains):

1. **Stage A** — ImageNet ConvNeXt frozen; train PAR head + prototypes / hyperbolic heads  
2. **Stage B** — Unfreeze backbone (full by default); continue from Stage A only  

Only change across ablations: `geometry` and/or `score_mode`.

| Run | geometry | score_mode | backbone |
| --- | --- | --- | --- |
| **Main** | lorentz | entailment | convnext_tiny (or small) |
| Euclid twin | euclidean | distance | same |
| Ablations | lorentz | distance / attr_l1 / hybrid | same ckpt, eval only |

## Linux quickstart

```bash
git clone https://github.com/BASSAT-BASSAT/HYPPAR-WACV.git
cd HYPPAR-WACV

pip install -r requirements.txt   # includes gdown for dataset download

# 1) Download UPAR annotations + Market / PA100K / PETA images
make data
# details: DATA_SETUP.md

export HYPPAR_UPAR_ROOT="$(dirname "$PWD")/UPAR-Challenge-2027"
export PYTHONPATH=$PWD

make smoke
make train-main          # clean Lorentz entailment (Tiny), Stage A+B
make eval-modes
make package             # -> checkpoints/hyppar_task2_submission.zip
```

### Stronger backbone (still ~8GB with AMP)

```bash
make train-small         # ConvNeXt-Small, smaller batch
NAME=lorentz_entail_small make eval-modes
NAME=lorentz_entail_small make package
```

### Paper matrix

```bash
make train-paper         # main Lorentz + Euclid twin
```

### Euclid twin only

```bash
make train-euclid
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `HYPPAR_UPAR_ROOT` | Root of `UPAR-Challenge-2027` (contains `data/`) |
| `HYPPAR_DATA_ROOT` | Override images root (default `$UPAR_ROOT/data`) |
| `HYPPAR_ANNO_ROOT` | Override annotations (default `$DATA/annotations`) |
| `HYPPAR_CKPT_DIR` | Checkpoint dir (default `./checkpoints`) |

## Codabench

Container: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (no network).

```bash
make package
# Upload checkpoints/hyppar_task2_submission.zip
# Archive root must contain run.py (Makefile packaging already does this)
```

`rank_gallery` encodes the gallery in chunks (avoids OOM) and scores with the checkpoint’s `score_mode` (default **entailment**).

## Manual commands (without Make)

```bash
export PYTHONPATH=$PWD
export HYPPAR_UPAR_ROOT=/path/to/UPAR-Challenge-2027

python -m hyppar.scripts.train_curriculum \
  --name lorentz_entail_tiny \
  --geometry lorentz --score-mode entailment \
  --backbone convnext_tiny \
  --stage-a-epochs 5 --stage-b-epochs 10 \
  --batch-size-a 64 --batch-size-b 32 \
  --unfreeze-mode full --lambda-attr 3.0 --entail-beta 1.5

python -m hyppar.evaluate \
  --ckpt checkpoints/lorentz_entail_tiny/model.pt \
  --score-mode entailment

python -m hyppar.scripts.package_submission \
  --ckpt checkpoints/lorentz_entail_tiny/model.pt
```

## Layout

```
HYPPAR-WACV/
  Makefile
  configs/           # env files for Make
  hyppar/            # Python package
    models/          # Lorentz + HypPAR
    train.py
    evaluate.py
    scripts/train_curriculum.py
    submission/      # Codabench run.py
  checkpoints/       # created locally (gitignored)
```

## Citation / notes

See `hyppar/PAPER_NOTES.md` for framing, related work, and prior exploratory val numbers.

# Dataset setup (UPAR Track 2)

Images are **not** in this repo (too large). You need:

1. **UPAR-Challenge-2027** — annotations + official `download_datasets.py`  
2. **Market1501, PA-100K, PETA** images — downloaded by that script  

## One-shot (recommended)

From `HYPPAR-WACV` root on Linux:

```bash
# deps for the downloader
pip install gdown tqdm numpy

# clones UPAR next to this repo (../UPAR-Challenge-2027) and downloads all images
make data
# equivalent:
#   python scripts/setup_upar_data.py
#   bash scripts/setup_upar_data.sh
```

Then:

```bash
export HYPPAR_UPAR_ROOT="$(dirname "$PWD")/UPAR-Challenge-2027"
# if you used a custom path:
#   export HYPPAR_UPAR_ROOT=/abs/path/to/UPAR-Challenge-2027

export PYTHONPATH=$PWD
make smoke
```

## Manual steps

### 1) Clone challenge repo (annotations)

```bash
cd ..   # parent of HYPPAR-WACV
git clone https://github.com/speckean/UPAR-Challenge-2027.git
cd UPAR-Challenge-2027
```

Annotations live under:

```text
data/annotations/task2/train/{gt.csv,ids.csv,queries.csv}
data/annotations/task2/val/{gt.csv,ids.csv,queries.csv}
```

### 2) Download images

```bash
cd /path/to/UPAR-Challenge-2027
pip install gdown tqdm numpy
python download_datasets.py --data-dir ./data
```

This pulls:

| Dataset | Source |
| --- | --- |
| Market1501 | Google Drive (via `gdown`) |
| PA-100K | Google Drive folder (via `gdown`) |
| PETA | Dropbox zip + `peta_file_mapping.txt` remap |

Google Drive may ask you to retry / use cookies if rate-limited; re-run the same command (it skips files that already exist when possible).

### 3) Point HypPAR at the data

```bash
export HYPPAR_UPAR_ROOT=/path/to/UPAR-Challenge-2027
export PYTHONPATH=/path/to/HYPPAR-WACV
```

Expected final layout:

```text
$HYPPAR_UPAR_ROOT/
  data/
    annotations/task2/{train,val}/
    Market1501/bounding_box_train/ ...
    PA100k/release_data/...
    PETA/images/
  download_datasets.py
  peta_file_mapping.txt
```

## Verify

```bash
export HYPPAR_UPAR_ROOT=...
export PYTHONPATH=/path/to/HYPPAR-WACV
python -m hyppar.scripts.smoke_dataset
# or: make smoke
```

You should see train record counts and `SMOKE_OK`.

## Train after data is ready

```bash
make train-main
make eval-modes
make package
```

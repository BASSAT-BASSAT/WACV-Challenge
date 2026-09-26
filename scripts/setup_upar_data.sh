#!/usr/bin/env bash
# Clone UPAR challenge repo (annotations) + download Market / PA100K / PETA images.
# Usage (from HYPPAR-WACV root):
#   bash scripts/setup_upar_data.sh
#   # or:
#   make data

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPAR_PARENT="${HYPPAR_UPAR_PARENT:-$(dirname "$ROOT")}"
UPAR_ROOT="${HYPPAR_UPAR_ROOT:-$UPAR_PARENT/UPAR-Challenge-2027}"
UPAR_GIT="${UPAR_GIT_URL:-https://github.com/speckean/UPAR-Challenge-2027.git}"

echo "UPAR_ROOT=$UPAR_ROOT"

if [[ ! -d "$UPAR_ROOT/.git" && ! -f "$UPAR_ROOT/download_datasets.py" ]]; then
  echo "Cloning UPAR challenge (annotations + download script)..."
  git clone "$UPAR_GIT" "$UPAR_ROOT"
else
  echo "UPAR challenge already present."
fi

mkdir -p "$UPAR_ROOT/data"
cd "$UPAR_ROOT"

# Needs: pip install gdown tqdm numpy
python3 - <<'PY'
import importlib.util
from pathlib import Path
p = Path("download_datasets.py")
spec = importlib.util.spec_from_file_location("dl", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("Downloading Market1501 + PA100K + PETA into", Path("data").resolve())
mod.prepare_datasets("data")
print("DATA_DOWNLOAD_DONE")
PY

echo ""
echo "Set this before training:"
echo "  export HYPPAR_UPAR_ROOT=$UPAR_ROOT"
echo "  export PYTHONPATH=$ROOT"
echo ""
echo "Expected layout:"
echo "  $UPAR_ROOT/data/annotations/task2/{train,val}/"
echo "  $UPAR_ROOT/data/Market1501/"
echo "  $UPAR_ROOT/data/PA100k/"
echo "  $UPAR_ROOT/data/PETA/images/"

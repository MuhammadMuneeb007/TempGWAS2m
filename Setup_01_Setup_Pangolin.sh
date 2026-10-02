#!/bin/bash
set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"
REPO="$ROOT/external_tools/Pangolin"
ENV="$ROOT/envs/pangolin"
LOCK="$ROOT/setup_manifests/Pangolin.commit.txt"

PM="$(command -v mamba || command -v micromamba || command -v conda || true)"
if [[ -z "$PM" ]]; then
    echo "ERROR: mamba/micromamba/conda not found."
    exit 1
fi

mkdir -p "$ROOT/external_tools" "$ROOT/setup_manifests"

echo
echo "=============================================================="
echo "PANGOLIN SETUP"
echo "=============================================================="

if [[ ! -d "$REPO/.git" ]]; then
    echo "COMMAND: git clone https://github.com/tkzeng/Pangolin.git $REPO"
    git clone https://github.com/tkzeng/Pangolin.git "$REPO"
fi

cd "$REPO"

if [[ -s "$LOCK" ]]; then
    COMMIT="$(tr -d '[:space:]' < "$LOCK")"
    echo "Pinned Pangolin commit: $COMMIT"
    git fetch --all --tags --prune
    git checkout --detach "$COMMIT"
else
    COMMIT="$(git rev-parse HEAD)"
    printf '%s\n' "$COMMIT" > "$LOCK"
    echo "Locked Pangolin commit: $COMMIT"
fi

if [[ ! -d "$ENV/conda-meta" ]]; then
    echo "Creating isolated Pangolin environment..."
    "$PM" create -y -p "$ENV" -c conda-forge         python=3.10 pip "numpy<2" pytorch gffutils biopython pandas pyfastx
else
    echo "[CACHE] Pangolin environment exists: $ENV"
fi

"$ENV/bin/python" -m pip install --upgrade PyVCF3
"$ENV/bin/python" -m pip install -e "$REPO" --no-deps

"$ENV/bin/python" - <<'PANGPY'
import pangolin
import gffutils
import Bio
import pandas
import pyfastx
print("[OK] Pangolin import:", pangolin.__file__)
PANGPY

"$ENV/bin/pangolin" --help >/dev/null

echo "[OK] Pangolin executable: $ENV/bin/pangolin"
echo "[OK] Pangolin commit: $(git rev-parse HEAD)"

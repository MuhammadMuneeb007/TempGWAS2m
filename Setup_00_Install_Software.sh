#!/bin/bash
set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"
ENV="$ROOT/envs/pipeline"
SPLICE="$ROOT/envs/spliceai"
ENVFILE="$ROOT/environment.yml"

PM="$(command -v mamba || command -v micromamba || command -v conda || true)"

if [[ -z "$PM" ]]; then
    echo "ERROR: mamba, micromamba or conda is required."
    exit 1
fi

export CONDA_CHANNEL_PRIORITY=strict

echo
echo "=============================================================="
echo "[1/4] MAIN PIPELINE ENVIRONMENT"
echo "=============================================================="
echo "Package manager : $PM"
echo "Environment     : $ENV"
echo "Specification   : $ENVFILE"
echo

if [[ -d "$ENV/conda-meta" ]]; then
    echo "[CACHE] Environment exists; updating/verifying it."
    echo "COMMAND: $PM env update -y -p $ENV -f $ENVFILE --prune"
    "$PM" env update -y -p "$ENV" -f "$ENVFILE" --prune
else
    echo "COMMAND: $PM env create -y -p $ENV -f $ENVFILE"
    "$PM" env create -y -p "$ENV" -f "$ENVFILE"
fi

echo
echo "=============================================================="
echo "[2/4] GWASLAB + CORE SOFTWARE CHECK"
echo "=============================================================="

echo "COMMAND: $ENV/bin/python -m pip install gwaslab==4.2.3"
"$ENV/bin/python" -m pip install --upgrade "gwaslab==4.2.3"

"$ENV/bin/python" - <<'COREPY'
mods = [
    "gwaslab", "pandas", "numpy", "scipy", "polars", "pyarrow",
    "duckdb", "pysam", "cyvcf2", "sklearn", "statsmodels",
    "matplotlib", "requests", "httpx", "tenacity", "tqdm",
]
for name in mods:
    __import__(name)
    print(f"[OK] Python import: {name}")
COREPY

for TOOL in plink2 bcftools samtools tabix bgzip bedtools aria2c pigz vep Rscript; do
    test -x "$ENV/bin/$TOOL"
    echo "[OK] $TOOL -> $ENV/bin/$TOOL"
done

"$ENV/bin/Rscript" -e '
for (p in c("susieR", "coloc", "data.table")) {
  if (!requireNamespace(p, quietly=TRUE)) stop(paste("Missing R package:", p))
  cat("[OK] R package:", p, "\n")
}
'

echo
echo "=============================================================="
echo "[3/4] SPLICEAI ISOLATED ENVIRONMENT"
echo "=============================================================="

if [[ ! -d "$SPLICE/conda-meta" ]]; then
    echo "COMMAND: $PM create -y -p $SPLICE -c conda-forge python=3.10 pip setuptools<81 numpy<2"
    "$PM" create -y -p "$SPLICE" -c conda-forge python=3.10 pip "setuptools<81" "numpy<2"
else
    echo "[CACHE] SpliceAI environment exists: $SPLICE"
fi

echo "COMMAND: install pinned SpliceAI runtime"
"$SPLICE/bin/python" -m pip install --upgrade     "setuptools<81"     "numpy<2"     "tensorflow-cpu==2.15.1"     "keras==2.15.0"     "spliceai==1.3.1"

"$SPLICE/bin/python" - <<'SPLICEPY'
import pkg_resources
import tensorflow as tf
import keras
import spliceai
from spliceai.utils import Annotator
print("[OK] pkg_resources")
print("[OK] TensorFlow", tf.__version__)
print("[OK] Keras", keras.__version__)
print("[OK] SpliceAI", spliceai.__file__)
print("[OK] SpliceAI Annotator import")
SPLICEPY

echo
echo "=============================================================="
echo "[4/4] SOFTWARE INSTALLATION COMPLETE"
echo "=============================================================="
"$ENV/bin/python" --version
"$ENV/bin/plink2" --version | head -1
"$ENV/bin/bcftools" --version | head -1
"$ENV/bin/samtools" --version | head -1

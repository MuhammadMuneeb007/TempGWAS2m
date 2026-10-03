#!/bin/bash
#SBATCH --job-name=GWAS_multiple_sclerosis_european
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step02_download/multiple_sclerosis/european/step02_%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step02_download/multiple_sclerosis/european/step02_%A_%a.err
#SBATCH --array=1-34%4
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MANIFEST="/data/ascher02/uqmmune1/Splice2/GWAS2m/01_gwas_catalog/multiple_sclerosis/european/GWAS_download_manifest.tsv"
OUTDIR="/data/ascher02/uqmmune1/Splice2/GWAS2m/02_summary_stats/multiple_sclerosis/european/raw"

ROW=$(awk -F '\t' -v id="$SLURM_ARRAY_TASK_ID" '
    NR > 1 && $1 == id {print; exit}
' "$MANIFEST")

if [[ -z "$ROW" ]]; then
    echo "Could not find array task $SLURM_ARRAY_TASK_ID in $MANIFEST"
    exit 1
fi

IFS=$'\t' read -r ARRAY_ID ACCESSION DOWNLOAD_URL FILE_NAME REST <<< "$ROW"

DEST="$OUTDIR/$ACCESSION"

mkdir -p "$DEST"

TARGET="$DEST/$FILE_NAME"
TEMP="$TARGET.part"

echo "============================================================"
echo "GWAS download"
echo "============================================================"
echo "Array task : $ARRAY_ID"
echo "Accession  : $ACCESSION"
echo "URL        : $DOWNLOAD_URL"
echo "Output     : $TARGET"
echo

if [[ -s "$TARGET" ]]; then
    echo "File already exists. Skipping."
    exit 0
fi

if command -v wget >/dev/null 2>&1; then

    wget \
        --continue \
        --tries=10 \
        --timeout=60 \
        -O "$TEMP" \
        "$DOWNLOAD_URL"

elif command -v curl >/dev/null 2>&1; then

    curl \
        --location \
        --fail \
        --retry 10 \
        --retry-delay 5 \
        --continue-at - \
        --output "$TEMP" \
        "$DOWNLOAD_URL"

else

    echo "ERROR: neither wget nor curl is available."
    exit 1
fi

mv "$TEMP" "$TARGET"

echo
echo "DOWNLOAD COMPLETE"
echo "$TARGET"

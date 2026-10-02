#!/bin/bash
#SBATCH --job-name=res_gtex
#SBATCH --partition=ascher
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-6%4
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/03_gtex.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/03_gtex.%A_%a.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"
RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"

MANIFEST="$ROOT/setup_manifests/gtex_v11.tsv"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-}}"


if [[ -z "$TASK_ID" ]]; then

    echo "Usage locally:"
    echo "  bash $0 TASK_ID"

    exit 2

fi



download_resource() {

    local URL="$1"
    local OUT="$2"

    mkdir -p "$(dirname "$OUT")"

    if command -v flock >/dev/null 2>&1; then
        exec 9>"${OUT}.lock"
        flock 9
    fi

    if [[ -s "$OUT" && -f "${OUT}.complete" ]]; then
        echo "[CACHE] $OUT"
        return 0
    fi

    echo
    echo "=============================================================="
    echo "DOWNLOAD"
    echo "=============================================================="
    echo "URL : $URL"
    echo "OUT : $OUT"
    echo

    set +e
    "$ARIA2" \
        --continue=true \
        --max-connection-per-server=4 \
        --split=4 \
        --min-split-size=16M \
        --file-allocation=none \
        --auto-file-renaming=false \
        --allow-overwrite=true \
        --max-tries=10 \
        --retry-wait=5 \
        --connect-timeout=30 \
        --timeout=60 \
        --console-log-level=warn \
        --summary-interval=5 \
        --dir "$(dirname "$OUT")" \
        --out "$(basename "$OUT")" \
        "$URL"
    RC=$?
    set -e

    if [[ $RC -ne 0 ]]; then
        echo
        echo "aria2c failed (exit $RC). Trying curl resume fallback..."
        curl \
            --fail \
            --location \
            --retry 8 \
            --retry-delay 5 \
            --retry-all-errors \
            --continue-at - \
            --output "$OUT" \
            "$URL"
    fi

    test -s "$OUT"
    touch "${OUT}.complete"
    echo "[OK] $OUT"
}



LINE="$(
    awk         -F '\t'         -v id="$TASK_ID"         'NR>1 && $1==id {print; exit}'         "$MANIFEST"
)"


IFS=$'\t' read     -r     ID     NAME     URL     SUBDIR     TYPE     <<< "$LINE"


OUTDIR="$RES/gtex/v11/$SUBDIR"

mkdir -p "$OUTDIR"

OUT="$OUTDIR/$(basename "$URL")"


download_resource     "$URL"     "$OUT"


if [[ "$TYPE" == "tar" ]]; then

    DEST="$OUTDIR/$NAME"

    mkdir -p "$DEST"

    if [[ ! -f "$DEST/.complete" ]]; then

        echo
        echo "Extracting $NAME..."

        tar             -xf "$OUT"             -C "$DEST"

        touch             "$DEST/.complete"

    fi

fi


echo
echo "GTEx task complete:"
echo "$NAME"

#!/bin/bash
#SBATCH --job-name=res_vep
#SBATCH --partition=ascher
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/08_vep.%j.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/08_vep.%j.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"

RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"



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



DIR="$RES/vep/116"

CACHE="$RES/vep/cache"


mkdir -p     "$DIR"     "$CACHE"


TAR="$DIR/homo_sapiens_vep_116_GRCh38.tar.gz"


URL="https://ftp.ensembl.org/pub/release-116/variation/indexed_vep_cache/homo_sapiens_vep_116_GRCh38.tar.gz"


download_resource     "$URL"     "$TAR"


MARKER="$CACHE/.vep116_GRCh38.complete"


if [[ ! -f "$MARKER" ]]; then

    echo
    echo "Extracting VEP cache..."

    tar         -xzf "$TAR"         -C "$CACHE"


    touch "$MARKER"

fi


echo
echo "VEP cache:"
echo "$CACHE"

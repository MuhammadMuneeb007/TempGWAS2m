#!/bin/bash
#SBATCH --job-name=res_1000g
#SBATCH --partition=ascher
#SBATCH --time=240:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-23%6
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/04_1000g.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/04_1000g.%A_%a.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"

RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"

RAW="$RES/1000G/raw"

META="$RES/1000G/metadata"


mkdir -p     "$RAW"     "$META"


TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-}}"


if [[ -z "$TASK_ID" ]]; then

    echo "Usage:"
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



BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL"


if [[ "$TASK_ID" -le 22 ]]; then

    CHR="$TASK_ID"

    FN="ALL.chr${CHR}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"


    download_resource         "$BASE/$FN"         "$RAW/$FN"


    download_resource         "$BASE/$FN.tbi"         "$RAW/$FN.tbi"


else

    download_resource         "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"         "$META/integrated_call_samples_v3.20130502.ALL.panel"


    download_resource         "$BASE/20190312_biallelic_SNV_and_INDEL_README.txt"         "$META/20190312_biallelic_SNV_and_INDEL_README.txt"


    download_resource         "$BASE/20190312_biallelic_SNV_and_INDEL_MANIFEST.txt"         "$META/20190312_biallelic_SNV_and_INDEL_MANIFEST.txt"

fi

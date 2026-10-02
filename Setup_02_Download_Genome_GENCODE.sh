#!/bin/bash
#SBATCH --job-name=res_genome
#SBATCH --partition=ascher
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/02_genome.%j.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/02_genome.%j.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"
RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"
PIGZ="$ROOT/envs/pipeline/bin/pigz"
SAMTOOLS="$ROOT/envs/pipeline/bin/samtools"



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



GENOME="$RES/genome/GRCh38"

GENCODE="$RES/gencode/release50"

mkdir -p "$GENOME" "$GENCODE"


FA_GZ="$GENOME/GRCh38.primary_assembly.genome.fa.gz"

FA="$GENOME/GRCh38.primary_assembly.genome.fa"

GTF="$GENCODE/gencode.v50.primary_assembly.annotation.gtf.gz"


download_resource "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/GRCh38.primary_assembly.genome.fa.gz" "$FA_GZ"


download_resource "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.primary_assembly.annotation.gtf.gz" "$GTF"


if [[ ! -s "$FA" ]]; then

    echo
    echo "Decompressing GRCh38..."

    "$PIGZ"         -dc         "$FA_GZ"         > "$FA.part"

    mv "$FA.part" "$FA"

fi


if [[ ! -s "$FA.fai" ]]; then

    "$SAMTOOLS"         faidx         "$FA"

fi


echo
echo "Genome setup complete."

#!/bin/bash
#SBATCH --job-name=res_1000g_samples
#SBATCH --partition=ascher
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/06_samples.%j.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/06_samples.%j.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"

RES="$ROOT/resources"

PANEL="$RES/1000G/metadata/integrated_call_samples_v3.20130502.ALL.panel"


test -s "$PANEL"


for ANC in EUR AFR EAS SAS AMR; do

    DIR="$RES/1000G/$ANC"

    mkdir -p "$DIR"


    OUT="$DIR/samples.txt"


    awk         -F '\t'         -v anc="$ANC"         'BEGIN {print "#IID"} NR>1 && $3==anc {print $1}'         "$PANEL"         > "$OUT.tmp"


    test "$(
        wc -l < "$OUT.tmp"
    )" -gt 1


    mv         "$OUT.tmp"         "$OUT"


    echo "$ANC samples: $(( $(wc -l < "$OUT") - 1 ))"

done


# ----------------------------------------------------------------------
# Compatibility links for current pipeline
# ----------------------------------------------------------------------

mkdir -p     "$ROOT/04_ld_reference"


ln -sfn     "$RES/1000G/raw"     "$ROOT/04_ld_reference/1000G_GRCh38_RAW"


for ANC in EUR AFR EAS SAS AMR; do

    ln -sfn         "$RES/1000G/$ANC"         "$ROOT/04_ld_reference/1000G_${ANC}_GRCh38"

done

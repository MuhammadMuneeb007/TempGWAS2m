#!/bin/bash
#SBATCH --job-name=res_1000g_all
#SBATCH --partition=ascher
#SBATCH --time=120:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-22%8
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/05_1000g_all.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/05_1000g_all.%A_%a.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"

RES="$ROOT/resources"

PLINK="$ROOT/envs/pipeline/bin/plink2"

RAW="$RES/1000G/raw"

ALL="$RES/1000G/ALL"


mkdir -p "$ALL"


CHR="${SLURM_ARRAY_TASK_ID:-${1:-}}"


if [[ -z "$CHR" ]]; then

    echo "Usage:"
    echo "  bash $0 CHROMOSOME"

    exit 2

fi


FN="ALL.chr${CHR}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"

VCF="$RAW/$FN"

PREFIX="$ALL/chr${CHR}"


if [[     -s "$PREFIX.pgen"     && -s "$PREFIX.pvar"     && -s "$PREFIX.psam"     && -f "$PREFIX.complete" ]]; then

    echo "[CACHE] chr${CHR} ALL PGEN"

    exit 0

fi


test -s "$VCF"


rm -f     "$PREFIX.pgen"     "$PREFIX.pvar"     "$PREFIX.psam"


"$PLINK"     --vcf "$VCF"     --set-all-var-ids '@:#:$r:$a'     --new-id-max-allele-len 1000     --rm-dup force-first     --max-alleles 2     --make-pgen     --threads "${SLURM_CPUS_PER_TASK:-4}"     --out "$PREFIX"


test -s "$PREFIX.pgen"
test -s "$PREFIX.pvar"
test -s "$PREFIX.psam"


touch "$PREFIX.complete"


echo
echo "ALL-sample PGEN complete:"
echo "chr${CHR}"

#!/bin/bash
#SBATCH --job-name=res_1000g_pop
#SBATCH --partition=ascher
#SBATCH --time=120:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-110%12
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/07_1000g_pop.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/07_1000g_pop.%A_%a.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"

RES="$ROOT/resources"

PLINK="$ROOT/envs/pipeline/bin/plink2"


TASK="${SLURM_ARRAY_TASK_ID:-${1:-}}"


if [[ -z "$TASK" ]]; then

    echo "Usage:"
    echo "  bash $0 TASK"

    exit 2

fi


INDEX=$(( (TASK - 1) / 22 ))

CHR=$(( (TASK - 1) % 22 + 1 ))


ANCES=(EUR AFR EAS SAS AMR)

ANC="${ANCES[$INDEX]}"


ALL="$RES/1000G/ALL/chr${CHR}"

OUTDIR="$RES/1000G/$ANC"

KEEP="$OUTDIR/samples.txt"

PREFIX="$OUTDIR/chr${CHR}_${ANC}_GRCh38"


mkdir -p "$OUTDIR"


if [[     -s "$PREFIX.pgen"     && -s "$PREFIX.pvar"     && -s "$PREFIX.psam"     && -f "$PREFIX.complete" ]]; then

    echo "[CACHE] $ANC chr${CHR}"

    exit 0

fi


test -s "$ALL.pgen"
test -s "$ALL.pvar"
test -s "$ALL.psam"
test -s "$KEEP"


"$PLINK"     --pfile "$ALL"     --keep "$KEEP"     --make-pgen     --threads "${SLURM_CPUS_PER_TASK:-4}"     --out "$PREFIX"


test -s "$PREFIX.pgen"
test -s "$PREFIX.pvar"
test -s "$PREFIX.psam"


touch "$PREFIX.complete"


echo
echo "Population reference complete:"
echo "Ancestry   : $ANC"
echo "Chromosome : $CHR"

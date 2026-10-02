#!/bin/bash
#SBATCH --job-name=res_pangolin_db
#SBATCH --partition=ascher
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/09_pangolin_db.%j.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/09_pangolin_db.%j.err

set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"

RES="$ROOT/resources"

REPO="$ROOT/external_tools/Pangolin"

PY="$ROOT/envs/pangolin/bin/python"


OUT="$RES/pangolin"

mkdir -p "$OUT"


SOURCE="$RES/gencode/release50/gencode.v50.primary_assembly.annotation.gtf.gz"

LINK="$OUT/gencode.v50.primary_assembly.annotation.gtf.gz"

DB="$OUT/gencode.v50.primary_assembly.annotation.db"


test -s "$SOURCE"

ln -sfn     "$SOURCE"     "$LINK"


if [[ -s "$DB" ]]; then

    echo "[CACHE] Pangolin annotation database"

    exit 0

fi


cd "$OUT"


"$PY"     "$REPO/scripts/create_db.py"     "$(basename "$LINK")"


test -s "$DB"


echo
echo "Pangolin database complete:"
echo "$DB"

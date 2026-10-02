#!/bin/bash
#SBATCH --job-name=Coloc_migraine_eur
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_coloc/migraine/european/coloc.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_coloc/migraine/european/coloc.%A_%a.err
#SBATCH --array=1-5%2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

cd /data/ascher02/uqmmune1/Splice2/GWAS2m

export GWAS_COLOC_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/11_coloc/migraine/european/coloc_manifest.tsv

/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Colocalize_GTEx_SuSiE.py

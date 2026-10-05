#!/bin/bash
#SBATCH --job-name=PColExt_parkinson_s_disease
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --array=1-246
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_provider_susie/parkinson_s_disease/european/extract.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_provider_susie/parkinson_s_disease/european/extract.%A_%a.err
set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
export GWAS_PROVIDER_EXTRACT_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/11_coloc_provider_susie/parkinson_s_disease/european/provider_extraction_manifest.tsv
/data/ascher01/uqmmune1/miniconda3/envs/spliceheart/bin/python3.12 /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Provider_SuSiE_Formal.py \
  --phenotype 'parkinson'"'"'s disease' --ancestry EUR \
  --extract-chunksize 250000 --extract-worker

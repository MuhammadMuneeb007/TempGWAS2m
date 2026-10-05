#!/bin/bash
#SBATCH --job-name=AggColoc_parkinson_s_disease
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_formal_coloc/parkinson_s_disease/european/aggregate.%j.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_formal_coloc/parkinson_s_disease/european/aggregate.%j.err
set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Colocalize_GTEx_SuSiE.py \
  --phenotype 'parkinson'"'"'s disease' --ancestry EUR \
  --aggregate-all --registry resources/coloc/custom_qtl_registry.tsv \
  --qtl-types eQTL,sQTL,pQTL,caQTL --suggestive-pp4 0.5

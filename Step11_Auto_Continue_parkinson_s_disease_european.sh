#!/bin/bash
#SBATCH --job-name=PColAuto_parkinson_s_disease
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_provider_susie/parkinson_s_disease/european/auto_continue.%j.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_provider_susie/parkinson_s_disease/european/auto_continue.%j.err
set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
/data/ascher01/uqmmune1/miniconda3/envs/spliceheart/bin/python3.12 /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Provider_SuSiE_Formal.py \
  --phenotype 'parkinson'"'"'s disease' --ancestry EUR \
  --qtl-types eQTL,sQTL,pQTL,isoQTL,exonQTL \
  --candidate-p 1e-05 \
  --max-candidates-per-context-locus 100 \
  --strong-pp4 0.8 --suggestive-pp4 0.5 \
  --partition general \
  --max-parallel 0 --max-extract-parallel 0 \
  --array-limit 1000 --extract-chunksize 250000 \
  --candidates-per-task 50 --auto-continue

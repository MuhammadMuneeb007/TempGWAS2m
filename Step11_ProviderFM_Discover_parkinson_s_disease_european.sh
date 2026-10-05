#!/bin/bash
#SBATCH --job-name=PColDisc_parkinson_s_disease
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --array=1-32
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_provider_susie/parkinson_s_disease/european/discover.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_provider_susie/parkinson_s_disease/european/discover.%A_%a.err
set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
export GWAS_PROVIDER_DISCOVERY_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/11_coloc_provider_susie/parkinson_s_disease/european/discovery_manifest.tsv
/data/ascher01/uqmmune1/miniconda3/envs/spliceheart/bin/python3.12 /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Provider_SuSiE_Formal.py \
  --phenotype 'parkinson'"'"'s disease' \
  --ancestry EUR \
  --qtl-types eQTL,sQTL,pQTL,isoQTL,exonQTL \
  --candidate-p 1e-05 \
  --max-candidates-per-context-locus 100 \
  --catalog-study-regex '' \
   \
  --discover-worker

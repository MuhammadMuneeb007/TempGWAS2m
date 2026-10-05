#!/bin/bash
#SBATCH --job-name=FColoc_parkinson_s_disease
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_formal_coloc/parkinson_s_disease/european/coloc.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step11_formal_coloc/parkinson_s_disease/european/coloc.%A_%a.err
set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
export GWAS_FORMAL_COLOC_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/11_coloc_formal/parkinson_s_disease/european/coloc_manifest.tsv
/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Colocalize_GTEx_SuSiE.py --phenotype 'parkinson'"'"'s disease' --ancestry EUR --registry resources/coloc/custom_qtl_registry.tsv --qtl-types eQTL,sQTL,pQTL,caQTL --candidate-p 1e-05 --min-common-snps 100 --max-common-snps 12000 --strong-pp4 0.8 --suggestive-pp4 0.5 --max-candidates-per-context-locus 100 --cpus 4 --max-parallel 0 --remote-delay 2.0 --ld-maf 0.01 --ld-mismatch-s 0.1 --time-budget-minutes 1380.0 --prefetch-dir resources/coloc/prefetch

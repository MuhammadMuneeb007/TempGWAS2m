#!/bin/bash
#SBATCH --job-name=FM_multiple_sclerosis_eur
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step06_finemapping/multiple_sclerosis/european/finemap.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step06_finemapping/multiple_sclerosis/european/finemap.%A_%a.err
#SBATCH --array=1-8%2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
export GWAS_FINEMAP_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/06_finemapping/multiple_sclerosis/european/finemapping_manifest.tsv

printf '%s
' "=============================================================="
printf '%s
' "STEP 06 - SUSIE-RSS FINEMAPPING"
printf '%s
' "Job ID        : $SLURM_JOB_ID"
printf '%s
' "Array task ID : $SLURM_ARRAY_TASK_ID"
printf '%s
' "Hostname      : $(hostname)"
printf '%s
' "=============================================================="

/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python /data/ascher02/uqmmune1/Splice2/GWAS2m/Step06_FineMap_GWAS_Loci.py --phenotype 'multiple sclerosis' --ancestry EUR

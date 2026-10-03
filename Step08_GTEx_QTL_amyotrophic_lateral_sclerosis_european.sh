#!/bin/bash
#SBATCH --job-name=GTEx_amyotrophic_lateral_sclerosis_eur
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step08_qtl/amyotrophic_lateral_sclerosis/european/qtl.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step08_qtl/amyotrophic_lateral_sclerosis/european/qtl.%A_%a.err
#SBATCH --array=1-1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

cd /data/ascher02/uqmmune1/Splice2/GWAS2m

export GWAS_QTL_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/08_qtl/amyotrophic_lateral_sclerosis/european/qtl_manifest.tsv

printf '%s
' '=============================================================='
printf '%s
' 'STEP 08 - GTEx v11 ALL-TISSUE eQTL + sQTL'
printf '%s
' '=============================================================='
printf 'Job ID        : %s
' "$SLURM_JOB_ID"
printf 'Array task ID : %s
' "$SLURM_ARRAY_TASK_ID"
printf 'Phenotype     : %s
' 'amyotrophic lateral sclerosis'
printf 'Ancestry      : %s (%s)
' European EUR
printf 'GTEx root     : %s
' /data/ascher02/uqmmune1/Splice2/GWAS2m/resources/gtex/v11
printf 'Mode          : ALL TISSUES
'
printf 'Hostname      : %s
' "$(hostname)"
printf '%s
' '=============================================================='

/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python /data/ascher02/uqmmune1/Splice2/GWAS2m/Step08_Integrate_GTEx_eQTL_sQTL.py

printf '%s
' '=============================================================='
printf 'STEP 08 TASK COMPLETE: %s
' "$SLURM_ARRAY_TASK_ID"
printf '%s
' '=============================================================='

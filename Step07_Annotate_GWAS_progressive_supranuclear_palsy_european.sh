#!/bin/bash
#SBATCH --job-name=VEP_progressive_supranuclear_palsy_eur
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step07_annotation/progressive_supranuclear_palsy/european/annotation.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step07_annotation/progressive_supranuclear_palsy/european/annotation.%A_%a.err
#SBATCH --array=1-1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
export GWAS_ANNOTATION_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/07_annotation/progressive_supranuclear_palsy/european/annotation_manifest.tsv

printf '%s
' '=============================================================='
printf '%s
' 'STEP 07 - VEP FUNCTIONAL ANNOTATION'
printf 'Job ID        : %s
' "$SLURM_JOB_ID"
printf 'Array task ID : %s
' "$SLURM_ARRAY_TASK_ID"
printf 'Hostname      : %s
' "$(hostname)"
printf '%s
' '=============================================================='

/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python /data/ascher02/uqmmune1/Splice2/GWAS2m/Step07_Annotate_Finemapped_Variants.py

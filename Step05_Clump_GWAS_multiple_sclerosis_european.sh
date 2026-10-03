#!/bin/bash
#SBATCH --job-name=CLUMP_multiple_sclerosis_eur
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step05_clumping/multiple_sclerosis/european/clump.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step05_clumping/multiple_sclerosis/european/clump.%A_%a.err
#SBATCH --array=1-8%4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

cd /data/ascher02/uqmmune1/Splice2/GWAS2m

export GWAS_CLUMP_MANIFEST=/data/ascher02/uqmmune1/Splice2/GWAS2m/05_ld_clumping/multiple_sclerosis/european/clumping_manifest.tsv

printf '%s
' "=============================================================="
printf '%s
' "STEP 05 - LD CLUMPING"
printf '%s
' "=============================================================="
printf 'Job ID        : %s
' "$SLURM_JOB_ID"
printf 'Array task ID : %s
' "$SLURM_ARRAY_TASK_ID"
printf 'Hostname      : %s
' "$(hostname)"
printf 'Python        : %s
' /data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python
printf 'Phenotype     : %s
' 'multiple sclerosis'
printf 'Ancestry      : %s (%s)
' European EUR
printf '%s
' "=============================================================="

/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python /data/ascher02/uqmmune1/Splice2/GWAS2m/Step05_Clump_GWAS_Loci.py

printf '%s
' "=============================================================="
printf 'STEP 05 TASK COMPLETE: %s
' "$SLURM_ARRAY_TASK_ID"
printf '%s
' "=============================================================="

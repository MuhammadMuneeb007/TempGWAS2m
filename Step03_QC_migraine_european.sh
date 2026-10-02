#!/bin/bash
#SBATCH --job-name=QC_migraine_european
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step03_qc/migraine/european/qc.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step03_qc/migraine/european/qc.%A_%a.err
#SBATCH --array=1-19
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

cd "/data/ascher02/uqmmune1/Splice2/GWAS2m"

export GWAS_QC_MANIFEST="/data/ascher02/uqmmune1/Splice2/GWAS2m/01_gwas_catalog/migraine/european/GWAS_QC_manifest.tsv"

echo "=============================================================="
echo "STEP 03 - GWASLAB QC"
echo "=============================================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array task ID : $SLURM_ARRAY_TASK_ID"
echo "Phenotype     : migraine"
echo "Ancestry      : European"
echo "Ancestry code : EUR"
echo "Partition     : general"
echo "Walltime      : 24:00:00"
echo "Hostname      : $(hostname)"
echo "Python        : /data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python3.11"
echo "Script        : /data/ascher02/uqmmune1/Splice2/GWAS2m/Step03_QC_One_GWAS.py"
echo "=============================================================="

"/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python3.11" "/data/ascher02/uqmmune1/Splice2/GWAS2m/Step03_QC_One_GWAS.py"

echo
echo "=============================================================="
echo "STEP 03 TASK COMPLETE"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "=============================================================="

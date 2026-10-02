#!/bin/bash
#SBATCH --job-name=1000G_EUR
#SBATCH --nodes=1
#SBATCH --partition=ascher
#SBATCH --time=240:00:00
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step04_ld_reference/EUR/chr.%A_%a.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/step04_ld_reference/EUR/chr.%A_%a.err
#SBATCH --array=1-22
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

cd "/data/ascher02/uqmmune1/Splice2/GWAS2m"

export LD_REFERENCE_MANIFEST="/data/ascher02/uqmmune1/Splice2/GWAS2m/04_ld_reference/1000G_EUR_GRCh38/EUR_chromosome_manifest.tsv"

echo "=============================================================="
echo "STEP 04 - 1000 GENOMES LD REFERENCE"
echo "=============================================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array task ID : $SLURM_ARRAY_TASK_ID"
echo "Chromosome    : $SLURM_ARRAY_TASK_ID"
echo "Hostname      : $(hostname)"
echo "=============================================================="

"/data/ascher01/uqmmune1/miniconda3/envs/spliceheart/bin/python3.12" "/data/ascher02/uqmmune1/Splice2/GWAS2m/Step04_Prepare_LD_Reference.py"

echo
echo "=============================================================="
echo "STEP 04 CHROMOSOME COMPLETE"
echo "Chromosome: $SLURM_ARRAY_TASK_ID"
echo "=============================================================="

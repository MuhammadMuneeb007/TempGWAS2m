#!/bin/bash
set -euo pipefail
cd /data/ascher02/uqmmune1/Splice2/GWAS2m
EXTRACT_JOB=$(sbatch --parsable Step11_ProviderFM_Extract_parkinson_s_disease_european.sh)
echo "Provider extraction job: $EXTRACT_JOB"
COLOC_JOBS=()
J=$(sbatch --parsable --dependency=afterok:$EXTRACT_JOB Step11_ProviderFM_Coloc_parkinson_s_disease_european_B001.sh)
COLOC_JOBS+=("$J")
echo "Coloc batch Step11_ProviderFM_Coloc_parkinson_s_disease_european_B001: $J"
DEP=$(IFS=:; echo "${COLOC_JOBS[*]}")
AGG=$(sbatch --parsable --dependency=afterok:$DEP Step11_ProviderFM_Aggregate_parkinson_s_disease_european.sh)
echo "Aggregation job: $AGG"

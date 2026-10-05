#!/bin/bash
set -euo pipefail
JOBIDS=()
TOTAL=32
CHUNK=900
for ((START=1; START<=TOTAL; START+=CHUNK)); do
  END=$((START+CHUNK-1)); if (( END>TOTAL )); then END=$TOTAL; fi
  JID=$(sbatch --parsable --array=${START}-${END} /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Loci_parkinson_s_disease_european.sh)
  JOBIDS+=("$JID")
  echo "Submitted locus array $START-$END as $JID"
done
DEP=$(IFS=:; echo "${JOBIDS[*]}")
sbatch --dependency=afterany:$DEP /data/ascher02/uqmmune1/Splice2/GWAS2m/Step11_Aggregate_parkinson_s_disease_european.sh

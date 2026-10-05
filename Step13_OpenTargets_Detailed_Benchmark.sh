#!/bin/bash
#SBATCH --job-name=Step13_OT
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Step13_OpenTargets_%j.out
#SBATCH --error=logs/Step13_OpenTargets_%j.err

set -euo pipefail

cd /data/ascher02/uqmmune1/Splice2/GWAS2m

mkdir -p logs

echo "============================================================"
echo "STEP 13 - OPEN TARGETS DETAILED BENCHMARK"
echo "============================================================"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo

echo "===== FREE DISK SPACE BEFORE DOWNLOAD ====="
df -h .
echo

echo "===== CURRENT OPEN TARGETS CACHE ====="
du -sh resources/opentargets/26.09 2>/dev/null || echo "Open Targets cache not created yet"
echo

python Step13_OpenTargets_Detailed_Benchmark.py \
    --phenotype "parkinson's disease" \
    --ancestry EUR

echo
echo "===== OPEN TARGETS CACHE AFTER RUN ====="
du -sh resources/opentargets/26.09 2>/dev/null || true

echo
echo "===== DATASET SIZES ====="
du -sh resources/opentargets/26.09/* 2>/dev/null || true

echo
echo "===== FREE DISK SPACE AFTER RUN ====="
df -h .

echo
echo "Finished: $(date)"

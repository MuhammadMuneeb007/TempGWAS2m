#!/bin/bash
#SBATCH --job-name=res_verify
#SBATCH --partition=ascher
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/10_verify.%j.out
#SBATCH --error=/data/ascher02/uqmmune1/Splice2/GWAS2m/logs/resource_setup/10_verify.%j.err

set -euo pipefail

cd "/data/ascher02/uqmmune1/Splice2/GWAS2m"

"/data/ascher02/uqmmune1/Splice2/GWAS2m/envs/pipeline/bin/python"     "/data/ascher02/uqmmune1/Splice2/GWAS2m/Verify_Setup.py"

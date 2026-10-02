#!/bin/bash
set -euo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"
STATE="$ROOT/setup_manifests/resource_jobs.env"
cd "$ROOT"

mkdir -p "$ROOT/setup_manifests" "$ROOT/logs/resource_setup"

if [[ -s "$STATE" ]]; then
    source "$STATE" || true
    if [[ -n "${JVERIFY:-}" ]] && squeue -h -j "$JVERIFY" 2>/dev/null | grep -q .; then
        echo "Resource setup is already active."
        echo "Final verification job: $JVERIFY"
        echo "Monitor with: squeue --me"
        exit 0
    fi
fi

JGENOME=$(sbatch --parsable "$ROOT/Setup_02_Download_Genome_GENCODE.sh")
JGTEX=$(sbatch --parsable "$ROOT/Setup_03_Download_GTEx_V11.sh")
J1000=$(sbatch --parsable "$ROOT/Setup_04_Download_1000G.sh")
JVEP=$(sbatch --parsable "$ROOT/Setup_08_Download_VEP_Cache.sh")
JALL=$(sbatch --parsable --dependency=afterok:$J1000 "$ROOT/Setup_05_Prepare_1000G_ALL.sh")
JSAMPLES=$(sbatch --parsable --dependency=afterok:$J1000 "$ROOT/Setup_06_Prepare_1000G_Sample_Lists.sh")
JPOP=$(sbatch --parsable --dependency=afterok:$JALL:$JSAMPLES "$ROOT/Setup_07_Prepare_1000G_Populations.sh")
JPANG=$(sbatch --parsable --dependency=afterok:$JGENOME "$ROOT/Setup_09_Build_Pangolin_DB.sh")
JVERIFY=$(sbatch --parsable --dependency=afterok:$JGENOME:$JGTEX:$J1000:$JVEP:$JALL:$JSAMPLES:$JPOP:$JPANG "$ROOT/Setup_10_Verify.sh")

cat > "$STATE" <<EOF
JGENOME=$JGENOME
JGTEX=$JGTEX
J1000=$J1000
JVEP=$JVEP
JALL=$JALL
JSAMPLES=$JSAMPLES
JPOP=$JPOP
JPANG=$JPANG
JVERIFY=$JVERIFY
EOF

echo
echo "=============================================================="
echo "RESOURCE SETUP SUBMITTED"
echo "=============================================================="
echo "Genome/GENCODE        : $JGENOME"
echo "GTEx V11              : $JGTEX"
echo "1000G raw             : $J1000"
echo "VEP cache             : $JVEP"
echo "1000G ALL PGEN        : $JALL"
echo "1000G sample lists    : $JSAMPLES"
echo "1000G populations     : $JPOP"
echo "Pangolin DB           : $JPANG"
echo "Final verification    : $JVERIFY"
echo
echo "Monitor with: squeue --me"
echo "Live status: $ROOT/envs/pipeline/bin/python $ROOT/Verify_Setup.py"

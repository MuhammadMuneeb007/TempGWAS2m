#!/usr/bin/env bash

set -uo pipefail

ROOT="/data/ascher02/uqmmune1/Splice2/GWAS2m"
cd "$ROOT"

PY="$(command -v python)"
ANCESTRY="EUR"
ANC_SLUG="european"

mkdir -p logs/brain_pipeline

# ============================================================
# PHENOTYPES THAT CURRENTLY PASSED STEP01
# ============================================================

PHENOTYPES=(
    "Parkinson's disease"
    "progressive supranuclear palsy"
    "amyotrophic lateral sclerosis"
    "multiple sclerosis"
    "epilepsy"
)

SLUGS=(
    "parkinson_s_disease"
    "progressive_supranuclear_palsy"
    "amyotrophic_lateral_sclerosis"
    "multiple_sclerosis"
    "epilepsy"
)

ACTIVE=(0 1 2 3 4)

echo "============================================================"
echo "GWAS2m BRAIN-DISEASE PIPELINE"
echo "Started: $(date)"
echo "============================================================"


# ============================================================
# CHECK WHETHER A SLURM JOB FAILED
# ============================================================

job_failed() {

    local jid="$1"

    # Give accounting a moment to update.
    sleep 5

    states="$(
        sacct -n -P -j "$jid" \
        --format=State 2>/dev/null \
        | cut -d'|' -f1
    )"

    if echo "$states" | grep -Eq \
       'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED'; then
        return 0
    fi

    return 1
}


# ============================================================
# WAIT FOR A COLLECTION OF SLURM JOBS
# ============================================================

wait_for_jobs() {

    local label="$1"
    shift

    local jobs=("$@")

    echo
    echo "============================================================"
    echo "WAITING: $label"
    echo "Jobs: ${jobs[*]}"
    echo "============================================================"

    while true; do

        running=0

        for jid in "${jobs[@]}"; do

            if squeue -h -j "$jid" 2>/dev/null | grep -q .; then
                running=1
            fi

        done

        if [[ "$running" -eq 0 ]]; then
            break
        fi

        echo "$(date): $label still running..."
        sleep 120

    done

    echo "$(date): $label finished."
}


# ============================================================
# GENERIC PLANNER -> SBATCH -> WAIT
# ============================================================

run_stage() {

    local label="$1"
    local planner="$2"
    local batch_prefix="$3"

    echo
    echo "################################################################"
    echo "$label"
    echo "################################################################"

    local JOBS=()
    local JOB_INDEXES=()

    # --------------------------------------------------------
    # Run planner for every currently active phenotype
    # --------------------------------------------------------

    for idx in "${ACTIVE[@]}"; do

        phenotype="${PHENOTYPES[$idx]}"
        pslug="${SLUGS[$idx]}"

        echo
        echo "------------------------------------------------------------"
        echo "$label"
        echo "Phenotype: $phenotype"
        echo "------------------------------------------------------------"

        if "$PY" "$planner" \
            --phenotype "$phenotype" \
            --ancestry "$ANCESTRY"
        then

            batch="${batch_prefix}_${pslug}_${ANC_SLUG}.sh"

            if [[ -s "$batch" ]]; then

                jid="$(
                    sbatch --parsable "$batch"
                )"

                jid="${jid%%;*}"

                echo
                echo "SUBMITTED:"
                echo "  $batch"
                echo "JOB ID:"
                echo "  $jid"

                JOBS+=("$jid")
                JOB_INDEXES+=("$idx")

            else

                echo
                echo "WARNING: generated batch file missing:"
                echo "  $batch"

            fi

        else

            echo
            echo "WARNING:"
            echo "$label planner failed for:"
            echo "  $phenotype"
            echo
            echo "This phenotype will not continue."

        fi

    done


    # --------------------------------------------------------
    # No jobs generated
    # --------------------------------------------------------

    if [[ ${#JOBS[@]} -eq 0 ]]; then

        echo
        echo "ERROR: No jobs were generated for $label"
        exit 1

    fi


    # --------------------------------------------------------
    # Wait until every phenotype finishes this stage
    # --------------------------------------------------------

    wait_for_jobs "$label" "${JOBS[@]}"


    # --------------------------------------------------------
    # Keep only successful phenotypes
    # --------------------------------------------------------

    NEW_ACTIVE=()

    for k in "${!JOBS[@]}"; do

        jid="${JOBS[$k]}"
        idx="${JOB_INDEXES[$k]}"
        phenotype="${PHENOTYPES[$idx]}"

        if job_failed "$jid"; then

            echo
            echo "FAILED:"
            echo "  $phenotype"
            echo "  $label"
            echo "  Job $jid"
            echo
            echo "This phenotype will be removed from later stages."

        else

            echo
            echo "SUCCESS:"
            echo "  $phenotype"
            echo "  $label"

            NEW_ACTIVE+=("$idx")

        fi

    done

    ACTIVE=("${NEW_ACTIVE[@]}")

    if [[ ${#ACTIVE[@]} -eq 0 ]]; then

        echo
        echo "No phenotypes remain after $label."
        exit 1

    fi

    echo
    echo "Phenotypes continuing:"
    for idx in "${ACTIVE[@]}"; do
        echo "  ${PHENOTYPES[$idx]}"
    done
}


# ============================================================
# STEP 01
#
# Runs Step01 planner.
# Step01 generates Step02_Download_GWAS_*.sh
# Controller submits those automatically.
# ============================================================

run_stage \
    "STEP01/02 - GWAS DISCOVERY + DOWNLOAD" \
    "Step01_Plan_GWAS.py" \
    "Step02_Download_GWAS"


# ============================================================
# STEP 03
#
# This will ONLY execute now that all Step02 jobs have left
# the SLURM queue.
# ============================================================

run_stage \
    "STEP03 - GWAS QC" \
    "Step03_QC_One_GWAS.py" \
    "Step03_QC"


# ============================================================
# STEP 04
#
# NOT phenotype-specific.
# Your EUR 1000G reference already exists and is shared.
# ============================================================

echo
echo "============================================================"
echo "STEP04 SKIPPED"
echo "Using existing shared EUR 1000G LD reference."
echo "============================================================"


# ============================================================
# STEP 05
# ============================================================

run_stage \
    "STEP05 - LD CLUMPING" \
    "Step05_Clump_GWAS_Loci.py" \
    "Step05_Clump_GWAS"


# ============================================================
# STEP 06
# ============================================================

run_stage \
    "STEP06 - SUSIE-RSS FINE-MAPPING" \
    "Step06_FineMap_GWAS_Loci.py" \
    "Step06_FineMap_GWAS"


# ============================================================
# STEP 07
# ============================================================

run_stage \
    "STEP07 - VEP ANNOTATION" \
    "Step07_Annotate_Finemapped_Variants.py" \
    "Step07_Annotate_GWAS"


# ============================================================
# STEP 08
# ============================================================

run_stage \
    "STEP08 - GTEx eQTL/sQTL" \
    "Step08_Integrate_GTEx_eQTL_sQTL.py" \
    "Step08_GTEx_QTL"


# ============================================================
# STEP 09
# ============================================================

run_stage \
    "STEP09 - SPLICEAI" \
    "Step09_Run_SpliceAI.py" \
    "Step09_SpliceAI"


# ============================================================
# STEP 10
# ============================================================

run_stage \
    "STEP10 - PANGOLIN" \
    "Step10_Run_Pangolin.py" \
    "Step10_Pangolin"


# ============================================================
# STEP 11
# ============================================================

run_stage \
    "STEP11 - GTEx SUSIE COLOCALISATION" \
    "Step11_Colocalize_GTEx_SuSiE.py" \
    "Step11_Coloc"


# ============================================================
# STEP 12
#
# Step12 does not generate a SLURM array, so execute phenotype
# by phenotype after Step11 has completed.
# ============================================================

echo
echo "################################################################"
echo "STEP12 - OPEN TARGETS COMPARISON"
echo "################################################################"

STEP12_ACTIVE=()

for idx in "${ACTIVE[@]}"; do

    phenotype="${PHENOTYPES[$idx]}"

    echo
    echo "STEP12: $phenotype"

    if "$PY" Step12_Compare_OpenTargets.py \
        --phenotype "$phenotype" \
        --ancestry "$ANCESTRY"
    then

        STEP12_ACTIVE+=("$idx")

    else

        echo "STEP12 FAILED: $phenotype"

    fi

done

ACTIVE=("${STEP12_ACTIVE[@]}")


# ============================================================
# STEP 13
#
# Wait if your current migraine Step13 download is still active.
# ============================================================

echo
echo "################################################################"
echo "STEP13 - DETAILED OPEN TARGETS"
echo "################################################################"

while pgrep -u "$USER" \
    -f '[S]tep13_OpenTargets_Detailed_Benchmark.py' \
    >/dev/null 2>&1
do

    echo "$(date): another Step13 process is still running."
    echo "Waiting before using the Open Targets cache..."
    sleep 300

done


STEP13_ACTIVE=()
FIRST_OT=1

for idx in "${ACTIVE[@]}"; do

    phenotype="${PHENOTYPES[$idx]}"

    echo
    echo "STEP13: $phenotype"

    if [[ "$FIRST_OT" -eq 1 ]]; then

        # First run verifies/resumes the shared Open Targets cache.
        if "$PY" Step13_OpenTargets_Detailed_Benchmark.py \
            --phenotype "$phenotype" \
            --ancestry "$ANCESTRY"
        then

            STEP13_ACTIVE+=("$idx")
            FIRST_OT=0

        else

            echo "STEP13 FAILED: $phenotype"

        fi

    else

        # Remaining phenotypes reuse exactly the same local OT cache.
        if "$PY" Step13_OpenTargets_Detailed_Benchmark.py \
            --phenotype "$phenotype" \
            --ancestry "$ANCESTRY" \
            --no-download-ot
        then

            STEP13_ACTIVE+=("$idx")

        else

            echo "STEP13 FAILED: $phenotype"

        fi

    fi

done

ACTIVE=("${STEP13_ACTIVE[@]}")


# ============================================================
# STEP 14
# ============================================================

echo
echo "################################################################"
echo "STEP14 - LOCUS-TO-MECHANISM"
echo "################################################################"

for idx in "${ACTIVE[@]}"; do

    phenotype="${PHENOTYPES[$idx]}"

    echo
    echo "STEP14: $phenotype"

    "$PY" Step14_Interpret_Variant_Mechanisms.py \
        --phenotype "$phenotype" \
        --ancestry "$ANCESTRY" \
    || echo "STEP14 FAILED: $phenotype"

done


echo
echo "============================================================"
echo "GWAS2m BRAIN PIPELINE FINISHED"
echo "Finished: $(date)"
echo "============================================================"


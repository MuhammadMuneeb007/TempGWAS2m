#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic GWAS Pipeline
STEP 03 - GWASLab QC

This single Python file has three execution modes.

======================================================================
MODE 1 - PLANNER
======================================================================

When run normally:

    envs/pipeline/bin/python Step03_QC_One_GWAS.py \
        --phenotype migraine \
        --ancestry EUR

the script automatically:

1. Selects the EXACT Step 02 manifest for the requested phenotype/ancestry.
2. Never guesses from the most recently modified project.
3. Finds all GWAS files that are CURRENTLY fully downloaded.
4. Reports GWAS files that are still missing/downloading.
5. Continues with the available GWAS files.
6. Creates a Step 03 QC manifest with ancestry code + label.
7. Generates a SLURM array bash file on the general partition.
8. Forces workers to use envs/pipeline/bin/python.
9. Rejects requested walltimes above 24 hours.

Example generated file:

    Step03_QC_migraine_european.sh

Then run:

    sbatch Step03_QC_migraine_european.sh


======================================================================
MODE 2 - DIRECT SINGLE-INDEX
======================================================================

To run exactly one QC task directly from the existing Step 03 manifest:

    envs/pipeline/bin/python Step03_QC_One_GWAS.py \
        --phenotype migraine \
        --ancestry EUR \
        --index 1

This selects QC_TASK_ID=1 from the phenotype/ancestry-specific
GWAS_QC_manifest.tsv and executes the same worker code used by SLURM.

IMPORTANT: direct --index mode runs on the machine/node where the Python
command is executed. On HPC, use srun or sbatch --wrap if you want this
individual task to run on a compute node.


======================================================================
MODE 3 - SLURM WORKER
======================================================================

When the same Python file is executed from a SLURM array,
SLURM automatically defines:

    SLURM_ARRAY_TASK_ID

The Python script detects that variable automatically and switches
to worker mode.

Each SLURM task processes exactly ONE GWAS.

No command-line arguments are required.


======================================================================
OUTPUTS
======================================================================

02_summary_stats/
    migraine/
        european/
            qc/
                GCSTxxxx/
                    GCSTxxxx_GRCh38_QC.tsv.gz
                    GCSTxxxx_GRCh38_significant.tsv.gz
                    GCSTxxxx_QC_summary.tsv
                    GCSTxxxx_QC_summary.json


The GWAS studies are NEVER merged in this step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PARTITION = "general"

TIME_LIMIT = "24:00:00"

MEMORY = "100G"

CPUS_PER_TASK = 4

P_THRESHOLD = 5e-8


# =============================================================================
# DISPLAY
# =============================================================================

def banner(text):

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalize_text(value):

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def slugify(value):

    return normalize_text(
        value
    ).replace(
        " ",
        "_",
    )


# =============================================================================
# ANCESTRY NORMALIZATION
# =============================================================================

ANCESTRY_ALIASES = {
    "eur": ("EUR", "European"),
    "european": ("EUR", "European"),
    "afr": ("AFR", "African"),
    "african": ("AFR", "African"),
    "eas": ("EAS", "East Asian"),
    "east asian": ("EAS", "East Asian"),
    "sas": ("SAS", "South Asian"),
    "south asian": ("SAS", "South Asian"),
    "amr": ("AMR", "Hispanic or Latin American"),
    "hispanic": ("AMR", "Hispanic or Latin American"),
    "latino": ("AMR", "Hispanic or Latin American"),
    "hispanic or latin american": ("AMR", "Hispanic or Latin American"),
    "mid": ("MID", "Greater Middle Eastern"),
    "middle eastern": ("MID", "Greater Middle Eastern"),
    "greater middle eastern": ("MID", "Greater Middle Eastern"),
}


def canonical_ancestry(value):

    key = normalize_text(value)

    if key not in ANCESTRY_ALIASES:

        supported = (
            "EUR, AFR, EAS, SAS, AMR, MID"
        )

        raise ValueError(
            f"Unsupported ancestry: {value!r}. "
            f"Supported values: {supported}"
        )

    return ANCESTRY_ALIASES[key]


# =============================================================================
# SLURM ARGUMENTS
# =============================================================================

def validate_slurm_time(value):

    text = str(value).strip()

    try:

        if "-" in text:

            day_text, clock = text.split("-", 1)

            days = int(day_text)

        else:

            days = 0
            clock = text

        parts = clock.split(":")

        if len(parts) != 3:

            raise ValueError

        hours, minutes, seconds = (
            int(part)
            for part in parts
        )

        if (
            days < 0
            or hours < 0
            or minutes < 0
            or minutes > 59
            or seconds < 0
            or seconds > 59
        ):

            raise ValueError

        total_seconds = (
            days * 86400
            + hours * 3600
            + minutes * 60
            + seconds
        )

    except ValueError as error:

        raise argparse.ArgumentTypeError(
            "SLURM time must be HH:MM:SS or D-HH:MM:SS."
        ) from error

    if total_seconds <= 0:

        raise argparse.ArgumentTypeError(
            "SLURM time must be greater than zero."
        )

    if total_seconds > 86400:

        raise argparse.ArgumentTypeError(
            "Maximum allowed walltime is 24 hours."
        )

    return text


def arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Plan phenotype/ancestry-specific GWASLab QC jobs."
        )
    )

    parser.add_argument(
        "--phenotype",
        required=True,
        help="Phenotype used in Step 01, e.g. migraine or CAD.",
    )

    parser.add_argument(
        "--ancestry",
        required=True,
        help=(
            "Ancestry code or label, e.g. "
            "EUR, AFR, EAS, SAS, AMR, European."
        ),
    )

    parser.add_argument(
        "--partition",
        default=PARTITION,
        help=(
            f"SLURM partition. Default: {PARTITION}"
        ),
    )

    parser.add_argument(
        "--time",
        default=TIME_LIMIT,
        type=validate_slurm_time,
        help=(
            "Per-task SLURM walltime. Maximum 24 hours. "
            f"Default: {TIME_LIMIT}"
        ),
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Run exactly one existing QC manifest task directly, e.g. "
            "--index 1. Without --index, Step 03 runs in planner mode."
        ),
    )

    return parser.parse_args()


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def find_column(
    dataframe,
    candidates,
):

    lookup = {

        str(column).lower():
            column

        for column in dataframe.columns
    }

    for candidate in candidates:

        candidate_lower = (
            candidate.lower()
        )

        if candidate_lower in lookup:

            return lookup[
                candidate_lower
            ]

    return None


# =============================================================================
# FIND STEP 02 DOWNLOAD MANIFEST
# =============================================================================

def find_download_manifest(
    phenotype_slug,
    ancestry_slug,
):

    manifest = (
        Path("01_gwas_catalog")
        / phenotype_slug
        / ancestry_slug
        / "GWAS_download_manifest.tsv"
    )

    if not manifest.exists():

        raise FileNotFoundError(
            "\nCannot find the phenotype/ancestry-specific "
            "Step 02 manifest:\n"
            f"{manifest.resolve()}\n\n"
            "Run Step01_Plan_GWAS.py for this exact "
            "phenotype and ancestry, then submit the "
            "generated Step02 downloader."
        )

    return manifest.resolve()


# =============================================================================
# DETECT PHENOTYPE AND ANCESTRY
# =============================================================================

def determine_project(
    manifest_file,
    manifest,
):

    phenotype = None
    ancestry = None

    # -------------------------------------------------------------------------
    # Try manifest metadata first
    # -------------------------------------------------------------------------

    if (
        "PHENOTYPE"
        in manifest.columns
    ):

        values = (
            manifest[
                "PHENOTYPE"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        if len(values) > 0:

            phenotype = (
                values.iloc[0]
            )

    if (
        "ANCESTRY"
        in manifest.columns
    ):

        values = (
            manifest[
                "ANCESTRY"
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
        ]

        if len(values) > 0:

            ancestry = (
                values.iloc[0]
            )

    # -------------------------------------------------------------------------
    # Directory fallback
    #
    # Expected structure:
    #
    # 01_gwas_catalog/
    #     migraine/
    #         european/
    #             GWAS_download_manifest.tsv
    # -------------------------------------------------------------------------

    if phenotype is None:

        phenotype = (
            manifest_file
            .parent
            .parent
            .name
        )

    if ancestry is None:

        ancestry = (
            manifest_file
            .parent
            .name
        )

    return (
        phenotype,
        ancestry,
    )


# =============================================================================
# FIND DOWNLOADED GWAS FILE
# =============================================================================

def find_downloaded_file(
    raw_root,
    accession,
    expected_filename,
):

    # -------------------------------------------------------------------------
    # Expected Step 02 path
    # -------------------------------------------------------------------------

    expected = (
        raw_root
        / accession
        / expected_filename
    )

    if (
        expected.exists()
        and expected.is_file()
        and expected.stat().st_size > 0
    ):

        return (
            expected.resolve()
        )

    # -------------------------------------------------------------------------
    # Fallback:
    # Search inside accession directory.
    # -------------------------------------------------------------------------

    accession_directory = (
        raw_root
        / accession
    )

    if not accession_directory.exists():

        return None

    candidates = []

    valid_extensions = (
        ".tsv.gz",
        ".txt.gz",
        ".vcf.gz",
        ".tsv",
        ".txt",
        ".vcf",
    )

    for path in (
        accession_directory.iterdir()
    ):

        if not path.is_file():

            continue

        if (
            path.stat().st_size
            <= 0
        ):

            continue

        name = (
            path.name.lower()
        )

        # Ignore unfinished downloads.
        if name.endswith(
            ".part"
        ):

            continue

        if name.endswith(
            valid_extensions
        ):

            candidates.append(
                path
            )

    # If exactly one plausible GWAS file exists,
    # use it.
    if len(candidates) == 1:

        return (
            candidates[0]
            .resolve()
        )

    return None


# =============================================================================
# COLLECT EXISTING QC SUMMARIES
# =============================================================================

def collect_existing_qc_summaries(
    qc_root,
    destination,
):

    summary_files = list(
        qc_root.glob(
            "*/"
            "*_QC_summary.tsv"
        )
    )

    if not summary_files:

        return

    frames = []

    for path in summary_files:

        try:

            frame = pd.read_csv(
                path,
                sep="\t",
                low_memory=False,
            )

            frames.append(
                frame
            )

        except Exception:

            continue

    if not frames:

        return

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if (
        "STUDY_ACCESSION"
        in combined.columns
    ):

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "STUDY_ACCESSION"
                ],
                keep="last",
            )
        )

    combined.to_csv(
        destination,
        sep="\t",
        index=False,
    )


# =============================================================================
# PLANNER MODE
# =============================================================================

def planner_mode(args):

    banner(
        "STEP 03 - GWASLAB QC PLANNER"
    )

    # -------------------------------------------------------------------------
    # Absolute paths
    # -------------------------------------------------------------------------

    working_directory = (
        Path.cwd()
        .resolve()
    )

    script_path = (
        Path(__file__)
        .resolve()
    )

    python_path = (
        working_directory
        / "envs"
        / "pipeline"
        / "bin"
        / "python"
    ).resolve()

    if not python_path.exists():

        raise FileNotFoundError(
            "\nPipeline Python is missing:\n"
            f"{python_path}\n\n"
            "Run the software setup before Step 03."
        )

    print(
        "Working directory:"
    )

    print(
        f"  {working_directory}"
    )

    print()

    print(
        "Python:"
    )

    print(
        f"  {python_path}"
    )

    print()

    print(
        "Step 03 script:"
    )

    print(
        f"  {script_path}"
    )

    # -------------------------------------------------------------------------
    # Requested project
    # -------------------------------------------------------------------------

    phenotype = args.phenotype.strip()

    if not phenotype:

        raise ValueError(
            "--phenotype cannot be empty."
        )

    ancestry_code, ancestry_label = (
        canonical_ancestry(
            args.ancestry
        )
    )

    phenotype_slug = (
        slugify(
            phenotype
        )
    )

    ancestry_slug = (
        slugify(
            ancestry_label
        )
    )

    # -------------------------------------------------------------------------
    # Locate the EXACT Step 02 manifest
    # -------------------------------------------------------------------------

    download_manifest_file = (
        find_download_manifest(
            phenotype_slug=phenotype_slug,
            ancestry_slug=ancestry_slug,
        )
    )

    print()

    print(
        "Step 02 manifest:"
    )

    print(
        f"  {download_manifest_file}"
    )

    download_manifest = pd.read_csv(
        download_manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    required_columns = [
        "STUDY_ACCESSION",
        "FILE_NAME",
    ]

    for column in required_columns:

        if (
            column
            not in download_manifest.columns
        ):

            raise RuntimeError(
                "\nStep 02 manifest is missing "
                f"required column:\n{column}"
            )

    # -------------------------------------------------------------------------
    # Validate Step 01 project metadata
    # -------------------------------------------------------------------------

    if (
        "PHENOTYPE"
        in download_manifest.columns
    ):

        manifest_phenotypes = {
            normalize_text(value)
            for value in (
                download_manifest[
                    "PHENOTYPE"
                ]
                .dropna()
                .astype(str)
            )
            if str(value).strip()
        }

        if (
            manifest_phenotypes
            and manifest_phenotypes
            != {
                normalize_text(
                    phenotype
                )
            }
        ):

            raise RuntimeError(
                "\nManifest phenotype does not match "
                "the requested phenotype.\n"
                f"Requested : {phenotype}\n"
                f"Manifest  : "
                f"{sorted(manifest_phenotypes)}"
            )

    manifest_ancestry_codes = set()

    if (
        "ANCESTRY_CODE"
        in download_manifest.columns
    ):

        manifest_ancestry_codes = {
            str(value).strip().upper()
            for value in (
                download_manifest[
                    "ANCESTRY_CODE"
                ]
                .dropna()
            )
            if str(value).strip()
        }

    elif (
        "ANCESTRY"
        in download_manifest.columns
    ):

        for value in (
            download_manifest[
                "ANCESTRY"
            ]
            .dropna()
            .astype(str)
        ):

            if not value.strip():

                continue

            try:

                code, _ = (
                    canonical_ancestry(
                        value
                    )
                )

                manifest_ancestry_codes.add(
                    code
                )

            except ValueError:

                pass

    if (
        manifest_ancestry_codes
        and manifest_ancestry_codes
        != {
            ancestry_code
        }
    ):

        raise RuntimeError(
            "\nManifest ancestry does not match "
            "the requested ancestry.\n"
            f"Requested : {ancestry_code} "
            f"({ancestry_label})\n"
            f"Manifest  : "
            f"{sorted(manifest_ancestry_codes)}"
        )

    ancestry = ancestry_label

    banner(
        "PROJECT"
    )

    print(
        f"Phenotype : {phenotype}"
    )

    print(
        f"Ancestry  : {ancestry_label}"
    )

    print(
        f"Code      : {ancestry_code}"
    )

    print(
        f"Partition : {args.partition}"
    )

    print(
        f"Walltime  : {args.time}"
    )

    print(
        f"Studies   : "
        f"{len(download_manifest)}"
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------

    metadata_directory = (
        download_manifest_file
        .parent
    )

    raw_root = (
        working_directory
        / "02_summary_stats"
        / phenotype_slug
        / ancestry_slug
        / "raw"
    )

    qc_root = (
        working_directory
        / "02_summary_stats"
        / phenotype_slug
        / ancestry_slug
        / "qc"
    )

    log_root = (
        working_directory
        / "logs"
        / "step03_qc"
        / phenotype_slug
        / ancestry_slug
    )

    qc_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Collect summaries from previous QC runs
    # -------------------------------------------------------------------------

    combined_summary = (
        metadata_directory
        / "GWAS_QC_summary.tsv"
    )

    collect_existing_qc_summaries(
        qc_root=qc_root,
        destination=combined_summary,
    )

    # -------------------------------------------------------------------------
    # Find currently downloaded GWAS
    # -------------------------------------------------------------------------

    banner(
        "CHECKING STEP 02 DOWNLOADS"
    )

    qc_rows = []

    missing_rows = []

    for _, row in (
        download_manifest.iterrows()
    ):

        accession = str(
            row[
                "STUDY_ACCESSION"
            ]
        ).strip()

        expected_filename = str(
            row[
                "FILE_NAME"
            ]
        ).strip()

        input_file = (
            find_downloaded_file(
                raw_root=raw_root,
                accession=accession,
                expected_filename=expected_filename,
            )
        )

        if input_file is None:

            print(
                f"MISSING  {accession}"
            )

            missing_rows.append(
                {
                    "STUDY_ACCESSION":
                        accession,

                    "EXPECTED_FILE":
                        str(
                            raw_root
                            / accession
                            / expected_filename
                        ),
                }
            )

            continue

        print(
            f"READY    {accession}"
        )

        output_directory = (
            qc_root
            / accession
        )

        qc_rows.append(
            {
                "QC_TASK_ID":
                    len(qc_rows) + 1,

                "STUDY_ACCESSION":
                    accession,

                "INPUT_FILE":
                    str(input_file),

                "OUTPUT_DIR":
                    str(
                        output_directory.resolve()
                    ),

                "PHENOTYPE":
                    phenotype,

                "ANCESTRY":
                    ancestry_label,

                "ANCESTRY_CODE":
                    ancestry_code,

                "ANCESTRY_LABEL":
                    ancestry_label,

                "INPUT_SIZE_BYTES":
                    input_file.stat().st_size,
            }
        )

    # -------------------------------------------------------------------------
    # Missing files report
    #
    # IMPORTANT:
    # Missing files DO NOT stop Step 03 anymore.
    # -------------------------------------------------------------------------

    if missing_rows:

        missing_report = (
            metadata_directory
            / "GWAS_QC_missing_downloads.tsv"
        )

        pd.DataFrame(
            missing_rows
        ).to_csv(
            missing_report,
            sep="\t",
            index=False,
        )

        banner(
            "SOME DOWNLOADS ARE STILL INCOMPLETE"
        )

        print(
            f"Total expected : "
            f"{len(download_manifest)}"
        )

        print(
            f"Ready for QC   : "
            f"{len(qc_rows)}"
        )

        print(
            f"Still missing  : "
            f"{len(missing_rows)}"
        )

        print()

        print(
            "Missing report:"
        )

        print(
            f"  {missing_report}"
        )

        print()

        print(
            "Continuing with all currently "
            "available GWAS files."
        )

        print()

        print(
            "You can run this planner again later "
            "when more downloads finish."
        )

    # -------------------------------------------------------------------------
    # Must have at least one file
    # -------------------------------------------------------------------------

    if not qc_rows:

        raise RuntimeError(
            "\nNo completed GWAS files are currently "
            "available for QC."
        )

    # -------------------------------------------------------------------------
    # Create Step 03 manifest
    # -------------------------------------------------------------------------

    qc_manifest = pd.DataFrame(
        qc_rows
    )

    qc_manifest_file = (
        metadata_directory
        / "GWAS_QC_manifest.tsv"
    ).resolve()

    qc_manifest.to_csv(
        qc_manifest_file,
        sep="\t",
        index=False,
    )

    number_of_jobs = len(
        qc_manifest
    )

    # -------------------------------------------------------------------------
    # Generate SLURM bash
    # -------------------------------------------------------------------------

    bash_file = (
        working_directory
        / (
            f"Step03_QC_"
            f"{phenotype_slug}_"
            f"{ancestry_slug}.sh"
        )
    )

    job_name = (
        f"QC_"
        f"{phenotype_slug}_"
        f"{ancestry_slug}"
    )

    # There is intentionally NO concurrency cap.
    #
    # Example:
    #
    # #SBATCH --array=1-12
    #
    # NOT:
    #
    # #SBATCH --array=1-12%4

    bash_text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_root}/qc.%A_%a.out
#SBATCH --error={log_root}/qc.%A_%a.err
#SBATCH --array=1-{number_of_jobs}
#SBATCH --mem={MEMORY}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --ntasks=1

set -euo pipefail

cd "{working_directory}"

export GWAS_QC_MANIFEST="{qc_manifest_file}"

echo "=============================================================="
echo "STEP 03 - GWASLAB QC"
echo "=============================================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array task ID : $SLURM_ARRAY_TASK_ID"
echo "Phenotype     : {phenotype}"
echo "Ancestry      : {ancestry_label}"
echo "Ancestry code : {ancestry_code}"
echo "Partition     : {args.partition}"
echo "Walltime      : {args.time}"
echo "Hostname      : $(hostname)"
echo "Python        : {python_path}"
echo "Script        : {script_path}"
echo "=============================================================="

"{python_path}" "{script_path}"

echo
echo "=============================================================="
echo "STEP 03 TASK COMPLETE"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "=============================================================="
"""

    bash_file.write_text(
        bash_text,
        encoding="utf-8",
    )

    bash_file.chmod(
        0o755
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    banner(
        "STEP 03 BASH GENERATED"
    )

    print(
        f"Phenotype         : {phenotype}"
    )

    print(
        f"Ancestry          : {ancestry_label}"
    )

    print(
        f"Ancestry code     : {ancestry_code}"
    )

    print(
        f"Partition         : {args.partition}"
    )

    print(
        f"Walltime          : {args.time}"
    )

    print(
        f"Pipeline Python   : {python_path}"
    )

    print(
        f"Total Step 02 GWAS: "
        f"{len(download_manifest)}"
    )

    print(
        f"Available now     : "
        f"{number_of_jobs}"
    )

    print(
        f"Still missing     : "
        f"{len(missing_rows)}"
    )

    print(
        f"SLURM array       : "
        f"1-{number_of_jobs}"
    )

    print(
        "Concurrency cap   : NONE"
    )

    print(
        f"Memory per task   : {MEMORY}"
    )

    print(
        f"CPUs per task     : "
        f"{CPUS_PER_TASK}"
    )

    print()

    print(
        "QC manifest:"
    )

    print(
        f"  {qc_manifest_file}"
    )

    print()

    print(
        "Generated bash:"
    )

    print(
        f"  {bash_file}"
    )

    print()

    print(
        "RUN:"
    )

    print()

    print(
        f"  sbatch {bash_file.name}"
    )

    print()

    banner(
        "QC ARRAY TASKS"
    )

    print(
        qc_manifest[
            [
                "QC_TASK_ID",
                "STUDY_ACCESSION",
                "INPUT_FILE",
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================================
# WORKER MODE
# =============================================================================

def worker_mode():

    # -------------------------------------------------------------------------
    # SLURM automatically gives us the task ID.
    # -------------------------------------------------------------------------

    task_value = (
        os.environ.get(
            "SLURM_ARRAY_TASK_ID"
        )
    )

    if task_value is None:

        raise RuntimeError(
            "\nSLURM_ARRAY_TASK_ID is not defined."
        )

    task_id = int(
        task_value
    )

    # -------------------------------------------------------------------------
    # Generated bash automatically provides manifest.
    # -------------------------------------------------------------------------

    manifest_value = (
        os.environ.get(
            "GWAS_QC_MANIFEST"
        )
    )

    if not manifest_value:

        raise RuntimeError(
            "\nGWAS_QC_MANIFEST is not defined."
        )

    manifest_file = (
        Path(
            manifest_value
        )
        .resolve()
    )

    if not manifest_file.exists():

        raise FileNotFoundError(
            "\nQC manifest not found:\n"
            f"{manifest_file}"
        )

    # -------------------------------------------------------------------------
    # Load manifest
    # -------------------------------------------------------------------------

    manifest = pd.read_csv(
        manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    required_columns = [
        "QC_TASK_ID",
        "STUDY_ACCESSION",
        "INPUT_FILE",
        "OUTPUT_DIR",
        "PHENOTYPE",
        "ANCESTRY",
        "ANCESTRY_CODE",
        "ANCESTRY_LABEL",
    ]

    for column in required_columns:

        if (
            column
            not in manifest.columns
        ):

            raise RuntimeError(
                "\nQC manifest missing column:\n"
                f"{column}"
            )

    task_numbers = pd.to_numeric(
        manifest[
            "QC_TASK_ID"
        ],
        errors="coerce",
    )

    selected = manifest[
        task_numbers
        == task_id
    ]

    if len(selected) != 1:

        raise RuntimeError(
            "\nExpected exactly one manifest row "
            f"for array task {task_id}. "
            f"Found {len(selected)}."
        )

    row = (
        selected.iloc[0]
    )

    # -------------------------------------------------------------------------
    # Current GWAS
    # -------------------------------------------------------------------------

    accession = str(
        row[
            "STUDY_ACCESSION"
        ]
    ).strip()

    phenotype = str(
        row[
            "PHENOTYPE"
        ]
    ).strip()

    ancestry = str(
        row[
            "ANCESTRY"
        ]
    ).strip()

    ancestry_code = str(
        row[
            "ANCESTRY_CODE"
        ]
    ).strip().upper()

    ancestry_label = str(
        row[
            "ANCESTRY_LABEL"
        ]
    ).strip()

    input_file = (
        Path(
            row[
                "INPUT_FILE"
            ]
        )
        .resolve()
    )

    output_directory = (
        Path(
            row[
                "OUTPUT_DIR"
            ]
        )
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    full_output = (
        output_directory
        / (
            f"{accession}"
            "_GRCh38_QC.tsv.gz"
        )
    )

    significant_output = (
        output_directory
        / (
            f"{accession}"
            "_GRCh38_significant.tsv.gz"
        )
    )

    summary_tsv = (
        output_directory
        / (
            f"{accession}"
            "_QC_summary.tsv"
        )
    )

    summary_json = (
        output_directory
        / (
            f"{accession}"
            "_QC_summary.json"
        )
    )

    failed_output = (
        output_directory
        / (
            f"{accession}"
            "_FAILED.txt"
        )
    )

    # -------------------------------------------------------------------------
    # Resume support
    #
    # If this GWAS was already processed successfully,
    # do nothing.
    # -------------------------------------------------------------------------

    if (
        full_output.exists()
        and full_output.stat().st_size > 0
        and significant_output.exists()
        and significant_output.stat().st_size > 0
        and summary_tsv.exists()
        and summary_tsv.stat().st_size > 0
    ):

        banner(
            "GWAS ALREADY COMPLETE"
        )

        print(
            f"Accession: {accession}"
        )

        print(
            "Skipping QC."
        )

        return

    # -------------------------------------------------------------------------
    # Verify raw GWAS
    # -------------------------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            "\nGWAS input file not found:\n"
            f"{input_file}"
        )

    if (
        input_file.stat().st_size
        <= 0
    ):

        raise RuntimeError(
            "\nGWAS input file is empty:\n"
            f"{input_file}"
        )

    # -------------------------------------------------------------------------
    # Load GWASLab only on compute worker
    # -------------------------------------------------------------------------

    import gwaslab as gl

    threads = int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            CPUS_PER_TASK,
        )
    )

    banner(
        "STEP 03 - GWASLAB QC WORKER"
    )

    print(
        f"Task       : {task_id}"
    )

    print(
        f"Accession  : {accession}"
    )

    print(
        f"Phenotype  : {phenotype}"
    )

    print(
        f"Ancestry   : {ancestry_label}"
    )

    print(
        f"Code       : {ancestry_code}"
    )

    print(
        f"Input      : {input_file}"
    )

    print(
        f"Output     : {output_directory}"
    )

    print(
        f"Threads    : {threads}"
    )

    try:

        # =====================================================================
        # LOAD SUMMARY STATISTICS
        # =====================================================================

        banner(
            "LOADING GWAS"
        )

        ss = gl.Sumstats(
            str(input_file),
            fmt="auto",
            build="38",
            verbose=True,
        )

        n_before = len(
            ss.data
        )

        print()

        print(
            f"Variants before QC: "
            f"{n_before:,}"
        )

        # =====================================================================
        # GWASLAB STANDARDIZATION / QC
        # =====================================================================

        banner(
            "GWASLAB BASIC CHECK"
        )

        ss.basic_check(
            remove=True,
            remove_dup=True,
            normalize=True,
            threads=threads,
            verbose=True,
        )

        data = (
            ss.data.copy()
        )

        n_after_basic_check = len(
            data
        )

        # =====================================================================
        # DETECT STANDARDIZED COLUMNS
        # =====================================================================

        chr_col = find_column(
            data,
            [
                "CHR",
            ],
        )

        pos_col = find_column(
            data,
            [
                "POS",
            ],
        )

        p_col = find_column(
            data,
            [
                "P",
            ],
        )

        ea_col = find_column(
            data,
            [
                "EA",
            ],
        )

        nea_col = find_column(
            data,
            [
                "NEA",
            ],
        )

        beta_col = find_column(
            data,
            [
                "BETA",
            ],
        )

        se_col = find_column(
            data,
            [
                "SE",
            ],
        )

        z_col = find_column(
            data,
            [
                "Z",
            ],
        )

        eaf_col = find_column(
            data,
            [
                "EAF",
            ],
        )

        n_col = find_column(
            data,
            [
                "N",
            ],
        )

        rsid_col = find_column(
            data,
            [
                "rsID",
                "RSID",
            ],
        )

        snpid_col = find_column(
            data,
            [
                "SNPID",
            ],
        )

        # =====================================================================
        # ESSENTIAL FIELDS
        # =====================================================================

        missing_essential = []

        if chr_col is None:

            missing_essential.append(
                "CHR"
            )

        if pos_col is None:

            missing_essential.append(
                "POS"
            )

        if p_col is None:

            missing_essential.append(
                "P"
            )

        if missing_essential:

            raise RuntimeError(
                "\nMissing essential fields after "
                "GWASLab standardization:\n"
                + ", ".join(
                    missing_essential
                )
            )

        # =====================================================================
        # CLEAN P VALUES
        # =====================================================================

        data[
            p_col
        ] = pd.to_numeric(
            data[
                p_col
            ],
            errors="coerce",
        )

        n_missing_p = int(
            data[
                p_col
            ].isna().sum()
        )

        data = data[
            data[
                p_col
            ].notna()
        ].copy()

        valid_p = (
            (
                data[
                    p_col
                ]
                >= 0
            )
            &
            (
                data[
                    p_col
                ]
                <= 1
            )
        )

        n_invalid_p = int(
            (~valid_p).sum()
        )

        data = data[
            valid_p
        ].copy()

        # =====================================================================
        # ADD PIPELINE METADATA
        # =====================================================================

        data[
            "GWAS_ACCESSION"
        ] = accession

        data[
            "ANALYSIS_PHENOTYPE"
        ] = phenotype

        data[
            "ANALYSIS_ANCESTRY"
        ] = ancestry_label

        data[
            "ANALYSIS_ANCESTRY_CODE"
        ] = ancestry_code

        data[
            "PIPELINE_BUILD"
        ] = "GRCh38"

        # =====================================================================
        # SORT
        # =====================================================================

        data = data.sort_values(
            [
                chr_col,
                pos_col,
            ],
            kind="stable",
        )

        n_after = len(
            data
        )

        # =====================================================================
        # GENOME-WIDE SIGNIFICANT VARIANTS
        # =====================================================================

        significant = data[
            data[
                p_col
            ]
            < P_THRESHOLD
        ].copy()

        significant = (
            significant
            .sort_values(
                p_col,
                ascending=True,
                kind="stable",
            )
        )

        n_significant = len(
            significant
        )

        # =====================================================================
        # DOWNSTREAM READINESS
        # =====================================================================

        has_chr = (
            chr_col is not None
        )

        has_pos = (
            pos_col is not None
        )

        has_p = (
            p_col is not None
        )

        has_ea = (
            ea_col is not None
        )

        has_nea = (
            nea_col is not None
        )

        has_beta = (
            beta_col is not None
        )

        has_se = (
            se_col is not None
        )

        has_z = (
            z_col is not None
        )

        has_eaf = (
            eaf_col is not None
        )

        has_n = (
            n_col is not None
        )

        has_rsid = (
            rsid_col is not None
        )

        has_snpid = (
            snpid_col is not None
        )

        # ---------------------------------------------------------------------
        # Clumping readiness
        # ---------------------------------------------------------------------

        clumping_ready = (
            has_chr
            and has_pos
            and has_p
            and (
                has_rsid
                or has_snpid
                or (
                    has_ea
                    and has_nea
                )
            )
        )

        # ---------------------------------------------------------------------
        # SuSiE readiness
        #
        # We need effect information:
        #
        # BETA + SE
        #
        # OR
        #
        # Z
        # ---------------------------------------------------------------------

        susie_ready = (
            has_chr
            and has_pos
            and has_p
            and has_ea
            and has_nea
            and (
                (
                    has_beta
                    and has_se
                )
                or has_z
            )
        )

        if susie_ready:

            qc_status = "PASS"

        elif clumping_ready:

            qc_status = "WARNING"

        else:

            qc_status = "FAIL"

        # =====================================================================
        # SAVE FULL CLEAN GWAS
        # =====================================================================

        banner(
            "SAVING CLEAN GWAS"
        )

        data.to_csv(
            full_output,
            sep="\t",
            index=False,
            compression="gzip",
        )

        # =====================================================================
        # SAVE SIGNIFICANT VARIANTS
        # =====================================================================

        banner(
            "SAVING GENOME-WIDE SIGNIFICANT VARIANTS"
        )

        significant.to_csv(
            significant_output,
            sep="\t",
            index=False,
            compression="gzip",
        )

        # =====================================================================
        # SUMMARY
        # =====================================================================

        minimum_p = None

        if n_after > 0:

            minimum_p = float(
                data[
                    p_col
                ].min()
            )

        summary = {

            "QC_TASK_ID":
                task_id,

            "STUDY_ACCESSION":
                accession,

            "PHENOTYPE":
                phenotype,

            "ANCESTRY":
                ancestry_label,

            "ANCESTRY_CODE":
                ancestry_code,

            "ANCESTRY_LABEL":
                ancestry_label,

            "BUILD":
                "GRCh38",

            "INPUT_FILE":
                str(input_file),

            "FULL_QC_FILE":
                str(full_output),

            "SIGNIFICANT_FILE":
                str(
                    significant_output
                ),

            "N_BEFORE_QC":
                n_before,

            "N_AFTER_GWASLAB_BASIC_CHECK":
                n_after_basic_check,

            "N_AFTER_QC":
                n_after,

            "N_REMOVED":
                (
                    n_before
                    - n_after
                ),

            "N_MISSING_P_REMOVED":
                n_missing_p,

            "N_INVALID_P_REMOVED":
                n_invalid_p,

            "N_GENOME_WIDE_SIGNIFICANT":
                n_significant,

            "P_THRESHOLD":
                P_THRESHOLD,

            "MIN_P":
                minimum_p,

            "HAS_CHR":
                has_chr,

            "HAS_POS":
                has_pos,

            "HAS_P":
                has_p,

            "HAS_EA":
                has_ea,

            "HAS_NEA":
                has_nea,

            "HAS_BETA":
                has_beta,

            "HAS_SE":
                has_se,

            "HAS_Z":
                has_z,

            "HAS_EAF":
                has_eaf,

            "HAS_N":
                has_n,

            "HAS_RSID":
                has_rsid,

            "HAS_SNPID":
                has_snpid,

            "CLUMPING_READY":
                clumping_ready,

            "SUSIE_READY":
                susie_ready,

            "QC_STATUS":
                qc_status,

            "THREADS":
                threads,

            "QC_COMPLETED_UTC":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        # =====================================================================
        # SAVE SUMMARY TSV
        # =====================================================================

        pd.DataFrame(
            [summary]
        ).to_csv(
            summary_tsv,
            sep="\t",
            index=False,
        )

        # =====================================================================
        # SAVE SUMMARY JSON
        # =====================================================================

        with open(
            summary_json,
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                summary,
                handle,
                indent=2,
                default=str,
            )

        # =====================================================================
        # Remove stale failure file
        # =====================================================================

        if failed_output.exists():

            failed_output.unlink()

        # =====================================================================
        # FINAL REPORT
        # =====================================================================

        banner(
            "GWAS QC COMPLETE"
        )

        print(
            f"Accession               : "
            f"{accession}"
        )

        print(
            f"Variants before QC      : "
            f"{n_before:,}"
        )

        print(
            f"Variants after QC       : "
            f"{n_after:,}"
        )

        print(
            f"Variants removed        : "
            f"{n_before - n_after:,}"
        )

        print(
            f"Genome-wide significant : "
            f"{n_significant:,}"
        )

        print(
            f"Minimum P               : "
            f"{minimum_p}"
        )

        print(
            f"Clumping ready          : "
            f"{clumping_ready}"
        )

        print(
            f"SuSiE ready             : "
            f"{susie_ready}"
        )

        print(
            f"QC status               : "
            f"{qc_status}"
        )

        print()

        print(
            "Full cleaned GWAS:"
        )

        print(
            f"  {full_output}"
        )

        print()

        print(
            "Significant variants:"
        )

        print(
            f"  {significant_output}"
        )

        print()

        print(
            "QC summary:"
        )

        print(
            f"  {summary_tsv}"
        )

    # =========================================================================
    # FAILURE HANDLING
    # =========================================================================

    except Exception as error:

        failure_text = (
            "STEP 03 GWAS QC FAILED\n\n"
            f"Task ID: {task_id}\n"
            f"Accession: {accession}\n"
            f"Phenotype: {phenotype}\n"
            f"Ancestry: {ancestry_label} ({ancestry_code})\n"
            f"Input file: {input_file}\n\n"
            f"Error:\n"
            f"{type(error).__name__}: "
            f"{error}\n\n"
            f"{traceback.format_exc()}"
        )

        failed_output.write_text(
            failure_text,
            encoding="utf-8",
        )

        print(
            failure_text,
            file=sys.stderr,
        )

        raise


# =============================================================================
# DIRECT SINGLE-INDEX MODE
# =============================================================================

def direct_index_mode(args):
    """Run one QC_TASK_ID from an existing phenotype/ancestry QC manifest."""

    if args.index is None:
        raise ValueError("--index is required for direct single-index mode.")

    if args.index < 1:
        raise ValueError("--index must be >= 1.")

    phenotype = args.phenotype.strip()
    if not phenotype:
        raise ValueError("--phenotype cannot be empty.")

    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    phenotype_slug = slugify(phenotype)
    ancestry_slug = slugify(ancestry_label)

    working_directory = Path.cwd().resolve()
    manifest_file = (
        working_directory
        / "01_gwas_catalog"
        / phenotype_slug
        / ancestry_slug
        / "GWAS_QC_manifest.tsv"
    ).resolve()

    if not manifest_file.exists() or manifest_file.stat().st_size <= 0:
        raise FileNotFoundError(
            "\nStep 03 QC manifest not found:\n"
            f"{manifest_file}\n\n"
            "Run Step03_QC_One_GWAS.py without --index first to generate "
            "the QC manifest and SLURM script."
        )

    manifest = pd.read_csv(
        manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    required = ["QC_TASK_ID", "STUDY_ACCESSION"]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise RuntimeError(
            f"QC manifest is missing required columns: {missing}"
        )

    task_numbers = pd.to_numeric(
        manifest["QC_TASK_ID"],
        errors="coerce",
    )
    selected = manifest.loc[task_numbers == args.index]

    if len(selected) != 1:
        available = sorted(
            int(value)
            for value in task_numbers.dropna().astype(int).unique().tolist()
        )
        if available:
            available_text = (
                f"{available[0]}-{available[-1]}"
                if available == list(range(available[0], available[-1] + 1))
                else ", ".join(map(str, available))
            )
        else:
            available_text = "none"

        raise RuntimeError(
            f"Expected exactly one QC manifest row for --index {args.index}; "
            f"found {len(selected)}. Available indices: {available_text}"
        )

    row = selected.iloc[0]
    accession = str(row["STUDY_ACCESSION"]).strip()

    banner("STEP 03 - DIRECT SINGLE-INDEX MODE")
    print(f"Manifest : {manifest_file}")
    print(f"Index    : {args.index}")
    print(f"Study    : {accession}")
    print(f"Phenotype: {phenotype}")
    print(f"Ancestry : {ancestry_label} ({ancestry_code})")
    print(f"CPUs     : {os.environ.get('SLURM_CPUS_PER_TASK', CPUS_PER_TASK)}")
    print()
    print(
        "NOTE: direct --index mode runs on the machine/node where this "
        "Python command is executed. On an HPC login node, use srun or "
        "sbatch --wrap to place the same command on a compute node."
    )

    old_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    old_manifest = os.environ.get("GWAS_QC_MANIFEST")

    os.environ["SLURM_ARRAY_TASK_ID"] = str(args.index)
    os.environ["GWAS_QC_MANIFEST"] = str(manifest_file)

    try:
        worker_mode()
    finally:
        if old_task is None:
            os.environ.pop("SLURM_ARRAY_TASK_ID", None)
        else:
            os.environ["SLURM_ARRAY_TASK_ID"] = old_task

        if old_manifest is None:
            os.environ.pop("GWAS_QC_MANIFEST", None)
        else:
            os.environ["GWAS_QC_MANIFEST"] = old_manifest


# =============================================================================
# MAIN
# =============================================================================

def main():

    # Generated SLURM-array worker mode.
    if (
        os.environ.get("SLURM_ARRAY_TASK_ID")
        and os.environ.get("GWAS_QC_MANIFEST")
    ):
        worker_mode()
        return

    # Interactive/planner CLI modes.
    args = arguments()

    if args.index is not None:
        direct_index_mode(args)
    else:
        planner_mode(args)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
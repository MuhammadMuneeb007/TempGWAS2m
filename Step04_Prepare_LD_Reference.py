#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic GWAS Pipeline
STEP 04 - Prepare ancestry-specific 1000 Genomes GRCh38 LD reference

NORMAL MODE
===========

Run:

    python Step04_Prepare_LD_Reference.py

No arguments are required.

The script automatically:

1. Finds the most recent GWAS project manifest.
2. Detects phenotype and ancestry.
3. Determines the correct 1000 Genomes ancestry populations.
4. Uses the 1000 Genomes sample panel prepared by Step00 (never downloads).
5. Creates the ancestry-specific sample list.
6. Creates a chromosome manifest for chr1-22.
7. Generates a SLURM array bash script.

Example:

    Step04_Prepare_1000G_EUR_GRCh38.sh

Then run:

    sbatch Step04_Prepare_1000G_EUR_GRCh38.sh


SLURM WORKER MODE
=================

Inside the SLURM array, this SAME Python file is executed again.

The script automatically detects:

    SLURM_ARRAY_TASK_ID

and processes the corresponding chromosome.

Each chromosome:

1. Uses the 1000 Genomes GRCh38 VCF prepared by Step00
   (resources/1000G/raw; Step04 never downloads shared data).
2. Uses the matching VCF index.
3. Subsets samples to the requested ancestry.
4. Converts the subset to PLINK2:
       .pgen
       .pvar
       .psam
5. Preserves existing rsIDs.
6. Assigns CHR:POS:REF:ALT to variants without IDs.
7. Saves a per-chromosome summary.
8. Removes the temporary ancestry-specific VCF.

The full raw 1000 Genomes chromosome is retained and can be reused.

No GWAS data are merged or processed in this step.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwas2m_config  # noqa: E402  (SLURM settings: config/slurm.yaml)

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

# SLURM partition/time/memory/CPUs come from config/slurm.yaml (stage "ld");
# CPUS_PER_TASK is only the thread fallback outside SLURM.
CPUS_PER_TASK = 4


# =============================================================================
# 1000 GENOMES REFERENCE
# =============================================================================

BASE_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/"
    "vol1/ftp/data_collections/"
    "1000_genomes_project/release/"
    "20190312_biallelic_SNV_and_INDEL"
)

PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/"
    "vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)

README_URL = (
    BASE_URL
    + "/20190312_biallelic_SNV_and_INDEL_README.txt"
)

RELEASE_MANIFEST_URL = (
    BASE_URL
    + "/20190312_biallelic_SNV_and_INDEL_MANIFEST.txt"
)


# =============================================================================
# ANCESTRY DEFINITIONS
# =============================================================================

ANCESTRY_CONFIG = {

    "european": {
        "code": "EUR",
        "populations": [
            "CEU",
            "FIN",
            "GBR",
            "IBS",
            "TSI",
        ],
    },

    "eur": {
        "code": "EUR",
        "populations": [
            "CEU",
            "FIN",
            "GBR",
            "IBS",
            "TSI",
        ],
    },

    "african": {
        "code": "AFR",
        "populations": [
            "ACB",
            "ASW",
            "ESN",
            "GWD",
            "LWK",
            "MSL",
            "YRI",
        ],
    },

    "afr": {
        "code": "AFR",
        "populations": [
            "ACB",
            "ASW",
            "ESN",
            "GWD",
            "LWK",
            "MSL",
            "YRI",
        ],
    },

    "east asian": {
        "code": "EAS",
        "populations": [
            "CDX",
            "CHB",
            "CHS",
            "JPT",
            "KHV",
        ],
    },

    "eas": {
        "code": "EAS",
        "populations": [
            "CDX",
            "CHB",
            "CHS",
            "JPT",
            "KHV",
        ],
    },

    "south asian": {
        "code": "SAS",
        "populations": [
            "BEB",
            "GIH",
            "ITU",
            "PJL",
            "STU",
        ],
    },

    "sas": {
        "code": "SAS",
        "populations": [
            "BEB",
            "GIH",
            "ITU",
            "PJL",
            "STU",
        ],
    },

    "hispanic or latin american": {
        "code": "AMR",
        "populations": [
            "CLM",
            "MXL",
            "PEL",
            "PUR",
        ],
    },

    "american": {
        "code": "AMR",
        "populations": [
            "CLM",
            "MXL",
            "PEL",
            "PUR",
        ],
    },

    "amr": {
        "code": "AMR",
        "populations": [
            "CLM",
            "MXL",
            "PEL",
            "PUR",
        ],
    },
}


# =============================================================================
# DISPLAY
# =============================================================================

def banner(text):

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


# =============================================================================
# TEXT
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
# COMMAND
# =============================================================================

def run_command(command):

    print()
    print(
        " ".join(
            str(x)
            for x in command
        ),
        flush=True,
    )

    subprocess.run(
        command,
        check=True,
    )


# =============================================================================
# SOFTWARE
# =============================================================================

def require_program(program):

    path = shutil.which(
        program
    )

    if path is None:

        raise RuntimeError(
            "\nRequired program is not available:\n"
            f"  {program}\n\n"
            "For example, install with:\n\n"
            "  conda install -c bioconda "
            "bcftools plink2\n"
        )

    return path


# =============================================================================
# PROJECT MANIFEST
# =============================================================================

def find_project_manifest():

    root = Path(
        "01_gwas_catalog"
    )

    if not root.exists():

        raise FileNotFoundError(
            "\nCannot find:\n"
            "01_gwas_catalog/\n"
        )

    # Prefer Step 03 QC manifest.
    manifests = list(
        root.rglob(
            "GWAS_QC_manifest.tsv"
        )
    )

    # Fall back to Step 02 manifest.
    if not manifests:

        manifests = list(
            root.rglob(
                "GWAS_download_manifest.tsv"
            )
        )

    if not manifests:

        raise FileNotFoundError(
            "\nCould not find GWAS_QC_manifest.tsv "
            "or GWAS_download_manifest.tsv.\n"
        )

    manifests = sorted(
        manifests,
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )

    return (
        manifests[0]
        .resolve()
    )


# =============================================================================
# DETECT PROJECT
# =============================================================================

def determine_project(
    manifest_file,
    manifest,
):

    phenotype = None
    ancestry = None

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
# ANCESTRY CONFIG
# =============================================================================

def get_ancestry_config(
    ancestry,
):

    normalized = (
        normalize_text(
            ancestry
        )
    )

    if (
        normalized
        not in ANCESTRY_CONFIG
    ):

        raise RuntimeError(
            "\nThe requested ancestry is not currently "
            "represented by this 1000 Genomes configuration:\n"
            f"  {ancestry}\n\n"
            "Supported ancestry groups:\n"
            "  EUR / European\n"
            "  AFR / African\n"
            "  EAS / East Asian\n"
            "  SAS / South Asian\n"
            "  AMR / Hispanic or Latin American\n"
        )

    return (
        ANCESTRY_CONFIG[
            normalized
        ]
    )


# =============================================================================
# CHROMOSOME FILE
# =============================================================================

def chromosome_filename(
    chromosome,
):

    return (
        f"ALL.chr{chromosome}."
        "shapeit2_integrated_snvindels_"
        "v2a_27022019.GRCh38.phased.vcf.gz"
    )


# =============================================================================
# SHARED 1000 GENOMES FILES (prepared by Step00; never downloaded here)
# =============================================================================

def require_shared_file(
    url,
    destination,
):
    """Use the Step00-prepared copy of a shared 1000 Genomes file.

    Step04 no longer downloads anything. The file must already exist at
    `destination` or in the central Step00 location (resources/1000G/raw or
    resources/1000G/metadata); in the latter case `destination` is linked to it.
    `url` is kept only to state the upstream source in the error message.
    """

    destination = Path(
        destination
    )

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):

        return

    root = Path.cwd().resolve()

    for central in (
        root / "resources" / "1000G" / "raw" / destination.name,
        root / "resources" / "1000G" / "metadata" / destination.name,
    ):

        if (
            central.exists()
            and central.stat().st_size > 0
        ):

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if destination.is_symlink():
                destination.unlink()

            destination.symlink_to(
                central
            )

            print(
                f"Using Step00 shared file:\n  {central}"
            )

            return

    raise FileNotFoundError(
        "\nRequired shared resource is unavailable:\n"
        f"  {destination.name}\n"
        f"  upstream source: {url}\n\n"
        "Step04 does not download shared reference data.\n"
        "Run:\n"
        "  python Step00_Check_Resources.py --inspect\n"
        "then:\n"
        "  python Step00_Check_Resources.py --resources-only"
    )


# =============================================================================
# SAMPLE PANEL
# =============================================================================

def create_sample_list(
    panel_file,
    populations,
    code,
    output_directory,
):

    output_file = (
        output_directory
        / f"{code}.samples.txt"
    )

    panel = pd.read_csv(
        panel_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    required = [
        "sample",
        "pop",
    ]

    for column in required:

        if column not in panel.columns:

            raise RuntimeError(
                "\n1000 Genomes panel missing column:\n"
                f"  {column}"
            )

    selected = panel[
        panel[
            "pop"
        ].isin(
            populations
        )
    ].copy()

    if selected.empty:

        raise RuntimeError(
            "\nNo samples were found for populations:\n"
            + ", ".join(
                populations
            )
        )

    selected[
        [
            "sample"
        ]
    ].to_csv(
        output_file,
        index=False,
        header=False,
    )

    return (
        output_file.resolve(),
        selected,
    )


# =============================================================================
# COUNT NON-HEADER LINES
# =============================================================================

def count_nonheader_lines(
    path,
):

    count = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            if not line.strip():

                continue

            if line.startswith(
                "#"
            ):

                continue

            count += 1

    return count


# =============================================================================
# COLLECT REFERENCE SUMMARIES
# =============================================================================

def collect_reference_summaries(
    reference_directory,
    code,
):

    files = sorted(
        reference_directory.glob(
            f"chr*_{code}_summary.tsv"
        )
    )

    if not files:

        return

    frames = []

    for file in files:

        try:

            frame = pd.read_csv(
                file,
                sep="\t",
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

    if "CHR" in combined.columns:

        combined[
            "CHR"
        ] = pd.to_numeric(
            combined[
                "CHR"
            ],
            errors="coerce",
        )

        combined = (
            combined
            .sort_values(
                "CHR"
            )
        )

    output = (
        reference_directory
        / f"{code}_reference_manifest.tsv"
    )

    combined.to_csv(
        output,
        sep="\t",
        index=False,
    )


# =============================================================================
# PLANNER
# =============================================================================

def planner_mode():

    banner(
        "STEP 04 - LD REFERENCE PLANNER"
    )

    working_directory = (
        Path.cwd()
        .resolve()
    )

    script_path = (
        Path(__file__)
        .resolve()
    )

    python_path = (
        Path(sys.executable)
        .resolve()
    )

    # -------------------------------------------------------------------------
    # Previous pipeline manifest
    # -------------------------------------------------------------------------

    project_manifest_file = (
        find_project_manifest()
    )

    project_manifest = pd.read_csv(
        project_manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    phenotype, ancestry = (
        determine_project(
            project_manifest_file,
            project_manifest,
        )
    )

    config = (
        get_ancestry_config(
            ancestry
        )
    )

    code = (
        config[
            "code"
        ]
    )

    populations = (
        config[
            "populations"
        ]
    )

    banner(
        "PROJECT"
    )

    print(
        f"Phenotype  : {phenotype}"
    )

    print(
        f"Ancestry   : {ancestry}"
    )

    print(
        f"1000G code : {code}"
    )

    print(
        "Populations: "
        + ", ".join(
            populations
        )
    )

    # -------------------------------------------------------------------------
    # Software check
    # -------------------------------------------------------------------------

    banner(
        "SOFTWARE"
    )

    bcftools = (
        require_program(
            "bcftools"
        )
    )

    plink2 = (
        require_program(
            "plink2"
        )
    )

    print(
        f"bcftools : {bcftools}"
    )

    print(
        f"plink2   : {plink2}"
    )

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------

    reference_root = (
        working_directory
        / "04_ld_reference"
    )

    raw_directory = (
        reference_root
        / "1000G_GRCh38_RAW"
    )

    reference_directory = (
        reference_root
        / (
            f"1000G_"
            f"{code}_"
            f"GRCh38"
        )
    )

    temporary_directory = (
        reference_root
        / "tmp"
        / code
    )

    metadata_directory = (
        working_directory
        / "reference_metadata"
        / "1000G"
    )

    log_directory = (
        working_directory
        / "logs"
        / "step04_ld_reference"
        / code
    )

    for directory in [
        raw_directory,
        reference_directory,
        temporary_directory,
        metadata_directory,
        log_directory,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------------------
    # Small reference metadata
    # -------------------------------------------------------------------------

    panel_file = (
        metadata_directory
        / "integrated_call_samples_v3.20130502.ALL.panel"
    )

    readme_file = (
        metadata_directory
        / "20190312_biallelic_SNV_and_INDEL_README.txt"
    )

    release_manifest_file = (
        metadata_directory
        / "20190312_biallelic_SNV_and_INDEL_MANIFEST.txt"
    )

    banner(
        "REFERENCE METADATA"
    )

    require_shared_file(
        PANEL_URL,
        panel_file,
    )

    require_shared_file(
        README_URL,
        readme_file,
    )

    require_shared_file(
        RELEASE_MANIFEST_URL,
        release_manifest_file,
    )

    # -------------------------------------------------------------------------
    # Ancestry samples
    # -------------------------------------------------------------------------

    sample_file, selected_panel = (
        create_sample_list(
            panel_file=panel_file,
            populations=populations,
            code=code,
            output_directory=reference_directory,
        )
    )

    banner(
        "ANCESTRY SAMPLE PANEL"
    )

    print(
        f"Samples: "
        f"{len(selected_panel):,}"
    )

    print()

    if (
        "pop"
        in selected_panel.columns
    ):

        counts = (
            selected_panel[
                "pop"
            ]
            .value_counts()
            .sort_index()
        )

        print(
            counts.to_string()
        )

    print()

    print(
        f"Sample list:\n"
        f"  {sample_file}"
    )

    # -------------------------------------------------------------------------
    # Chromosome manifest
    # -------------------------------------------------------------------------

    rows = []

    for chromosome in range(
        1,
        23,
    ):

        filename = (
            chromosome_filename(
                chromosome
            )
        )

        raw_vcf = (
            raw_directory
            / filename
        )

        raw_tbi = (
            raw_directory
            / (
                filename
                + ".tbi"
            )
        )

        output_prefix = (
            reference_directory
            / (
                f"chr{chromosome}_"
                f"{code}_"
                f"GRCh38"
            )
        )

        rows.append(
            {
                "TASK_ID":
                    chromosome,

                "CHR":
                    chromosome,

                "ANCESTRY_CODE":
                    code,

                "PHENOTYPE":
                    phenotype,

                "ORIGINAL_ANCESTRY":
                    ancestry,

                "POPULATIONS":
                    ",".join(
                        populations
                    ),

                "SAMPLE_FILE":
                    str(
                        sample_file
                    ),

                "VCF_URL":
                    (
                        f"{BASE_URL}/"
                        f"{filename}"
                    ),

                "TBI_URL":
                    (
                        f"{BASE_URL}/"
                        f"{filename}.tbi"
                    ),

                "RAW_VCF":
                    str(
                        raw_vcf.resolve()
                    ),

                "RAW_TBI":
                    str(
                        raw_tbi.resolve()
                    ),

                "OUTPUT_PREFIX":
                    str(
                        output_prefix.resolve()
                    ),

                "TMP_DIR":
                    str(
                        temporary_directory.resolve()
                    ),
            }
        )

    manifest = pd.DataFrame(
        rows
    )

    ld_manifest_file = (
        reference_directory
        / (
            f"{code}_"
            "chromosome_manifest.tsv"
        )
    ).resolve()

    manifest.to_csv(
        ld_manifest_file,
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Collect already completed results
    # -------------------------------------------------------------------------

    collect_reference_summaries(
        reference_directory=reference_directory,
        code=code,
    )

    # -------------------------------------------------------------------------
    # Generate SLURM
    # -------------------------------------------------------------------------

    bash_file = (
        working_directory
        / (
            f"Step04_Prepare_1000G_"
            f"{code}_"
            f"GRCh38.sh"
        )
    )

    job_name = (
        f"1000G_{code}"
    )
    sbatch_header = gwas2m_config.sbatch_header_for_stage(
        "ld",
        job_name=job_name,
        output=f"{log_directory}/chr.%A_%a.out",
        error=f"{log_directory}/chr.%A_%a.err",
        array=gwas2m_config.array_spec(22, gwas2m_config.get_stage_resources("ld")["max_parallel"]),
    )


    bash_text = f"""#!/bin/bash
{sbatch_header}

set -euo pipefail

cd "{working_directory}"

export LD_REFERENCE_MANIFEST="{ld_manifest_file}"

echo "=============================================================="
echo "STEP 04 - 1000 GENOMES LD REFERENCE"
echo "=============================================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array task ID : $SLURM_ARRAY_TASK_ID"
echo "Chromosome    : $SLURM_ARRAY_TASK_ID"
echo "Hostname      : $(hostname)"
echo "=============================================================="

"{python_path}" "{script_path}"

echo
echo "=============================================================="
echo "STEP 04 CHROMOSOME COMPLETE"
echo "Chromosome: $SLURM_ARRAY_TASK_ID"
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
    # Summary
    # -------------------------------------------------------------------------

    completed = 0

    for chromosome in range(
        1,
        23,
    ):

        prefix = (
            reference_directory
            / (
                f"chr{chromosome}_"
                f"{code}_"
                "GRCh38"
            )
        )

        required = [
            Path(
                str(prefix)
                + ".pgen"
            ),
            Path(
                str(prefix)
                + ".pvar"
            ),
            Path(
                str(prefix)
                + ".psam"
            ),
        ]

        if all(
            x.exists()
            and x.stat().st_size > 0
            for x in required
        ):

            completed += 1

    banner(
        "STEP 04 BASH GENERATED"
    )

    print(
        f"Phenotype             : {phenotype}"
    )

    print(
        f"Ancestry              : {ancestry}"
    )

    print(
        f"1000 Genomes ancestry : {code}"
    )

    print(
        f"Chromosomes complete  : {completed}/22"
    )

    print(
        "SLURM array           : 1-22"
    )

    print(
        "Concurrency cap       : NONE"
    )

    print()

    print(
        "Reference directory:"
    )

    print(
        f"  {reference_directory}"
    )

    print()

    print(
        "Chromosome manifest:"
    )

    print(
        f"  {ld_manifest_file}"
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


# =============================================================================
# WORKER
# =============================================================================

def worker_mode():

    task_value = (
        os.environ.get(
            "SLURM_ARRAY_TASK_ID"
        )
    )

    if task_value is None:

        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is not defined."
        )

    task_id = int(
        task_value
    )

    manifest_value = (
        os.environ.get(
            "LD_REFERENCE_MANIFEST"
        )
    )

    if not manifest_value:

        raise RuntimeError(
            "LD_REFERENCE_MANIFEST is not defined."
        )

    manifest_file = (
        Path(
            manifest_value
        )
        .resolve()
    )

    manifest = pd.read_csv(
        manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    task_numbers = pd.to_numeric(
        manifest[
            "TASK_ID"
        ],
        errors="coerce",
    )

    selected = manifest[
        task_numbers
        == task_id
    ]

    if len(selected) != 1:

        raise RuntimeError(
            f"Expected one manifest row for task {task_id}; "
            f"found {len(selected)}."
        )

    row = (
        selected.iloc[0]
    )

    chromosome = int(
        row[
            "CHR"
        ]
    )

    code = str(
        row[
            "ANCESTRY_CODE"
        ]
    )

    sample_file = Path(
        row[
            "SAMPLE_FILE"
        ]
    )

    vcf_url = str(
        row[
            "VCF_URL"
        ]
    )

    tbi_url = str(
        row[
            "TBI_URL"
        ]
    )

    raw_vcf = Path(
        row[
            "RAW_VCF"
        ]
    )

    raw_tbi = Path(
        row[
            "RAW_TBI"
        ]
    )

    output_prefix = Path(
        row[
            "OUTPUT_PREFIX"
        ]
    )

    temporary_directory = Path(
        row[
            "TMP_DIR"
        ]
    )

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threads = int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            CPUS_PER_TASK,
        )
    )

    # -------------------------------------------------------------------------
    # Final PLINK files
    # -------------------------------------------------------------------------

    pgen = Path(
        str(output_prefix)
        + ".pgen"
    )

    pvar = Path(
        str(output_prefix)
        + ".pvar"
    )

    psam = Path(
        str(output_prefix)
        + ".psam"
    )

    summary_file = (
        output_prefix.parent
        / (
            f"chr{chromosome}_"
            f"{code}_summary.tsv"
        )
    )

    # -------------------------------------------------------------------------
    # Resume support
    # -------------------------------------------------------------------------

    if (
        pgen.exists()
        and pgen.stat().st_size > 0
        and pvar.exists()
        and pvar.stat().st_size > 0
        and psam.exists()
        and psam.stat().st_size > 0
    ):

        banner(
            "REFERENCE ALREADY COMPLETE"
        )

        print(
            f"chr{chromosome} {code}"
        )

        if not summary_file.exists():

            sample_count = (
                count_nonheader_lines(
                    psam
                )
            )

            variant_count = (
                count_nonheader_lines(
                    pvar
                )
            )

            pd.DataFrame(
                [
                    {
                        "CHR":
                            chromosome,

                        "ANCESTRY":
                            code,

                        "SAMPLES":
                            sample_count,

                        "VARIANTS":
                            variant_count,

                        "PGEN":
                            str(pgen),

                        "PVAR":
                            str(pvar),

                        "PSAM":
                            str(psam),
                    }
                ]
            ).to_csv(
                summary_file,
                sep="\t",
                index=False,
            )

        return

    # -------------------------------------------------------------------------
    # Software
    # -------------------------------------------------------------------------

    bcftools = (
        require_program(
            "bcftools"
        )
    )

    plink2 = (
        require_program(
            "plink2"
        )
    )

    banner(
        f"STEP 04 WORKER - chr{chromosome}"
    )

    print(
        f"Ancestry : {code}"
    )

    print(
        f"Threads  : {threads}"
    )

    print(
        f"Samples  : {sample_file}"
    )

    # =========================================================================
    # DOWNLOAD RAW CHROMOSOME
    # =========================================================================

    require_shared_file(
        url=vcf_url,
        destination=raw_vcf,
    )

    require_shared_file(
        url=tbi_url,
        destination=raw_tbi,
    )

    # =========================================================================
    # SUBSET ANCESTRY
    # =========================================================================

    ancestry_vcf = (
        temporary_directory
        / (
            f"chr{chromosome}."
            f"{code}."
            "GRCh38.vcf.gz"
        )
    )

    if not (
        ancestry_vcf.exists()
        and ancestry_vcf.stat().st_size > 0
    ):

        banner(
            "SUBSETTING ANCESTRY"
        )

        temporary_vcf = Path(
            str(ancestry_vcf)
            + ".part"
        )

        temporary_vcf.unlink(
            missing_ok=True
        )

        run_command(
            [
                bcftools,
                "view",

                "--threads",
                str(
                    threads
                ),

                "-S",
                str(
                    sample_file
                ),

                "--force-samples",

                "-Oz",

                "-o",
                str(
                    temporary_vcf
                ),

                str(
                    raw_vcf
                ),
            ]
        )

        if (
            not temporary_vcf.exists()
            or temporary_vcf.stat().st_size == 0
        ):

            raise RuntimeError(
                f"Failed to create ancestry VCF "
                f"for chr{chromosome}."
            )

        temporary_vcf.replace(
            ancestry_vcf
        )

    # =========================================================================
    # CONVERT TO PLINK2
    # =========================================================================

    banner(
        "CONVERTING TO PLINK2"
    )

    # Remove incomplete previous output.
    for path in [
        pgen,
        pvar,
        psam,
    ]:

        if path.exists():

            path.unlink()

    run_command(
        [
            plink2,

            "--vcf",
            str(
                ancestry_vcf
            ),

            "--set-missing-var-ids",
            "@:#:$r:$a",

            "--new-id-max-allele-len",
            "1000",

            "--rm-dup",
            "force-first",

            "--make-pgen",

            "--threads",
            str(
                threads
            ),

            "--out",
            str(
                output_prefix
            ),
        ]
    )

    # =========================================================================
    # VERIFY
    # =========================================================================

    for path in [
        pgen,
        pvar,
        psam,
    ]:

        if (
            not path.exists()
            or path.stat().st_size == 0
        ):

            raise RuntimeError(
                "PLINK2 output missing:\n"
                f"{path}"
            )

    # =========================================================================
    # COUNTS
    # =========================================================================

    sample_count = (
        count_nonheader_lines(
            psam
        )
    )

    variant_count = (
        count_nonheader_lines(
            pvar
        )
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = pd.DataFrame(
        [
            {
                "CHR":
                    chromosome,

                "ANCESTRY":
                    code,

                "SAMPLES":
                    sample_count,

                "VARIANTS":
                    variant_count,

                "RAW_VCF":
                    str(raw_vcf),

                "PGEN":
                    str(pgen),

                "PVAR":
                    str(pvar),

                "PSAM":
                    str(psam),
            }
        ]
    )

    summary.to_csv(
        summary_file,
        sep="\t",
        index=False,
    )

    # =========================================================================
    # REMOVE TEMP ANCESTRY VCF
    # =========================================================================

    ancestry_vcf.unlink(
        missing_ok=True
    )

    Path(
        str(ancestry_vcf)
        + ".tbi"
    ).unlink(
        missing_ok=True
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    banner(
        "CHROMOSOME COMPLETE"
    )

    print(
        f"Chromosome : {chromosome}"
    )

    print(
        f"Ancestry   : {code}"
    )

    print(
        f"Samples    : {sample_count:,}"
    )

    print(
        f"Variants   : {variant_count:,}"
    )

    print()

    print(
        f"PGEN:\n  {pgen}"
    )

    print(
        f"PVAR:\n  {pvar}"
    )

    print(
        f"PSAM:\n  {psam}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    # Normal shell:
    #
    #     no SLURM_ARRAY_TASK_ID
    #
    # therefore planner mode.
    #
    # Inside generated SLURM array:
    #
    #     SLURM_ARRAY_TASK_ID exists
    #
    # therefore worker mode.

    if os.environ.get(
        "SLURM_ARRAY_TASK_ID"
    ):

        worker_mode()

    else:

        planner_mode()


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":

    main()
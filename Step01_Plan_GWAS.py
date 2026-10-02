#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 01 — Generic GWAS discovery + download planner

Example:

    python Step01_Plan_GWAS.py \
        --phenotype migraine \
        --ancestry EUR

    python Step01_Plan_GWAS.py \
        --phenotype CAD \
        --ancestry EUR

    python Step01_Plan_GWAS.py \
        --phenotype asthma \
        --ancestry EUR

What this does
--------------
1. Downloads/caches GWAS Catalog study metadata.
2. Downloads/caches GWAS Catalog ancestry metadata.
3. Searches for ANY phenotype.
4. Filters studies by requested ancestry.
5. Requires full summary statistics.
6. Checks that harmonised summary statistics actually exist.
7. Determines the exact downloadable file.
8. Ranks eligible studies by sample size.
9. Creates a download manifest.
10. Generates Step02_Download_GWAS_<phenotype>_<ancestry>.sh

The generated SLURM script downloads each GWAS independently using
a SLURM array.

This script DOES NOT download the large GWAS files itself.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# GWAS CATALOG
# =============================================================================

STUDIES_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
    "gwas-catalog-download-studies-v1.0.3.1.txt"
)

ANCESTRY_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
    "gwas-catalog-download-ancestries-v1.0.3.1.txt"
)

SUMMARY_ROOT = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
)


# =============================================================================
# SLURM DEFAULTS
# =============================================================================

DEFAULT_PARTITION = "general"
DEFAULT_TIME_LIMIT = "24:00:00"
MAX_TIME_HOURS = 24
DEFAULT_MEMORY = "8G"
DEFAULT_CPUS_PER_TASK = 1


# =============================================================================
# COMMON PHENOTYPE ALIASES
# =============================================================================

PHENOTYPE_ALIASES = {

    "cad": [
        "coronary artery disease",
        "coronary heart disease",
    ],

    "chd": [
        "coronary heart disease",
        "coronary artery disease",
    ],

    "t2d": [
        "type 2 diabetes",
        "type 2 diabetes mellitus",
    ],

    "mdd": [
        "major depressive disorder",
        "major depression",
        "depression",
    ],
}


# =============================================================================
# ANCESTRY ALIASES
# =============================================================================

ANCESTRY_ALIASES = {

    "eur": "European",
    "european": "European",

    "afr": "African",
    "african": "African",

    "eas": "East Asian",
    "east asian": "East Asian",

    "sas": "South Asian",
    "south asian": "South Asian",

    "amr": "Hispanic or Latin American",
    "hispanic": "Hispanic or Latin American",
    "latino": "Hispanic or Latin American",
    "hispanic or latin american": "Hispanic or Latin American",

    "mid": "Greater Middle Eastern",
    "middle eastern": "Greater Middle Eastern",
    "greater middle eastern": "Greater Middle Eastern",
}

ANCESTRY_CODES = {
    "European": "EUR",
    "African": "AFR",
    "East Asian": "EAS",
    "South Asian": "SAS",
    "Hispanic or Latin American": "AMR",
    "Greater Middle Eastern": "MID",
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def normalize(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def slugify(value):

    value = normalize(value)

    return value.replace(
        " ",
        "_"
    )


def canonical_ancestry(value):

    key = normalize(value)

    return ANCESTRY_ALIASES.get(
        key,
        value
    )


def ancestry_code(value):

    label = canonical_ancestry(value)

    return ANCESTRY_CODES.get(
        label,
        slugify(label).upper()
    )


def validate_slurm_time(value):

    value = str(value).strip()

    match = re.fullmatch(
        r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})",
        value
    )

    if not match:

        raise argparse.ArgumentTypeError(
            "SLURM time must be HH:MM:SS or D-HH:MM:SS. "
            "Example: 24:00:00"
        )

    days = int(match.group(1) or 0)
    hours = int(match.group(2))
    minutes = int(match.group(3))
    seconds = int(match.group(4))

    if minutes >= 60 or seconds >= 60:

        raise argparse.ArgumentTypeError(
            "Invalid SLURM time. Minutes and seconds must be < 60."
        )

    total_seconds = (
        days * 86400
        + hours * 3600
        + minutes * 60
        + seconds
    )

    if total_seconds <= 0:

        raise argparse.ArgumentTypeError(
            "SLURM time must be greater than zero."
        )

    if total_seconds > MAX_TIME_HOURS * 3600:

        raise argparse.ArgumentTypeError(
            f"Maximum allowed walltime is {MAX_TIME_HOURS} hours."
        )

    return value


def build_terms(
    phenotype,
    user_aliases
):

    terms = [
        phenotype
    ]

    key = normalize(
        phenotype
    )

    if key in PHENOTYPE_ALIASES:

        terms.extend(
            PHENOTYPE_ALIASES[key]
        )

    terms.extend(
        user_aliases
    )

    final = []
    seen = set()

    for term in terms:

        n = normalize(
            term
        )

        if n and n not in seen:

            seen.add(n)

            final.append(
                term
            )

    return final


def make_session():

    session = requests.Session()

    retries = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.5,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ]
    )

    adapter = HTTPAdapter(
        max_retries=retries
    )

    session.mount(
        "https://",
        adapter
    )

    session.headers.update(
        {
            "User-Agent":
                "Generic-GWAS-Pipeline/1.0"
        }
    )

    return session


# =============================================================================
# DOWNLOAD SMALL METADATA FILES
# =============================================================================

def download_metadata(
    session,
    url,
    path,
    refresh=False
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        path.exists()
        and path.stat().st_size > 0
        and not refresh
    ):

        print(
            f"Using cached metadata: {path}"
        )

        return

    print(
        f"Downloading:\n  {url}"
    )

    response = session.get(
        url,
        timeout=(30, 300)
    )

    response.raise_for_status()

    temporary = Path(
        str(path) + ".part"
    )

    with open(
        temporary,
        "wb"
    ) as handle:

        handle.write(
            response.content
        )

    os.replace(
        temporary,
        path
    )


# =============================================================================
# SAMPLE SIZE
# =============================================================================

def estimate_sample_size(text):

    if text is None:
        return 0

    numbers = re.findall(
        r"\d[\d,]*",
        str(text)
    )

    values = []

    for number in numbers:

        try:

            x = int(
                number.replace(
                    ",",
                    ""
                )
            )

            if x >= 100:

                values.append(
                    x
                )

        except ValueError:

            continue

    return sum(values)


def parse_number(value):

    if value is None:
        return 0

    numbers = re.findall(
        r"\d[\d,]*",
        str(value)
    )

    if not numbers:
        return 0

    try:

        return int(
            numbers[0].replace(
                ",",
                ""
            )
        )

    except ValueError:

        return 0


# =============================================================================
# SUMMARY STATISTICS AVAILABILITY
# =============================================================================

def has_full_summary_stats(value):

    if value is None:
        return False

    value = str(
        value
    ).strip()

    if not value:
        return False

    bad = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "no",
        "false",
        "0",
        "not available",
        "nr",
    }

    return (
        value.lower()
        not in bad
    )


# =============================================================================
# PHENOTYPE MATCHING
# =============================================================================

def contains_term(
    series,
    term
):

    term = normalize(
        term
    )

    if not term:

        return pd.Series(
            False,
            index=series.index
        )

    if (
        " " not in term
        and len(term) <= 5
    ):

        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(term)}"
            rf"(?![a-z0-9])"
        )

    else:

        pattern = re.escape(
            term
        )

    return series.str.contains(
        pattern,
        regex=True,
        na=False
    )


def find_phenotype_studies(
    studies,
    terms
):

    disease = (
        studies["DISEASE/TRAIT"]
        .fillna("")
        .map(normalize)
    )

    mapped = (
        studies["MAPPED_TRAIT"]
        .fillna("")
        .map(normalize)
        if "MAPPED_TRAIT" in studies.columns
        else pd.Series(
            "",
            index=studies.index
        )
    )

    title = (
        studies["STUDY"]
        .fillna("")
        .map(normalize)
        if "STUDY" in studies.columns
        else pd.Series(
            "",
            index=studies.index
        )
    )

    scores = pd.Series(
        0,
        index=studies.index,
        dtype=int
    )

    for term in terms:

        t = normalize(
            term
        )

        # Exact reported trait
        exact_disease = (
            disease == t
        )

        scores.loc[
            exact_disease
        ] = np.maximum(
            scores.loc[
                exact_disease
            ],
            100
        )

        # Exact ontology mapped trait
        exact_mapped = (
            mapped == t
        )

        scores.loc[
            exact_mapped
        ] = np.maximum(
            scores.loc[
                exact_mapped
            ],
            95
        )

        # Contains reported trait
        disease_hit = contains_term(
            disease,
            term
        )

        scores.loc[
            disease_hit
        ] = np.maximum(
            scores.loc[
                disease_hit
            ],
            80
        )

        # Contains mapped trait
        mapped_hit = contains_term(
            mapped,
            term
        )

        scores.loc[
            mapped_hit
        ] = np.maximum(
            scores.loc[
                mapped_hit
            ],
            70
        )

        # Study title
        title_hit = contains_term(
            title,
            term
        )

        scores.loc[
            title_hit
        ] = np.maximum(
            scores.loc[
                title_hit
            ],
            40
        )

    result = studies[
        scores > 0
    ].copy()

    result[
        "MATCH_SCORE"
    ] = scores.loc[
        result.index
    ]

    result = result.drop_duplicates(
        subset=[
            "STUDY ACCESSION"
        ]
    )

    return result


# =============================================================================
# ANCESTRY FILTER
# =============================================================================

def get_ancestry_information(
    ancestry,
    target
):

    target_norm = normalize(
        target
    )

    # Use initial/discovery stage where available.
    if "STAGE" in ancestry.columns:

        stage = (
            ancestry["STAGE"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        initial = ancestry[
            stage.str.contains(
                "initial",
                regex=False
            )
        ].copy()

        if not initial.empty:

            ancestry = initial

    records = {}

    for accession, group in ancestry.groupby(
        "STUDY ACCESSION"
    ):

        categories = {

            normalize(x)

            for x in group[
                "BROAD ANCESTRAL CATEGORY"
            ].dropna()

            if normalize(x)
        }

        # STRICT ancestry:
        # every discovery ancestry category must equal requested ancestry.
        strict_match = (
            categories
            == {
                target_norm
            }
        )

        total_n = 0

        if (
            "NUMBER OF INDIVIDUALS"
            in group.columns
        ):

            total_n = sum(
                parse_number(x)
                for x in group[
                    "NUMBER OF INDIVIDUALS"
                ]
            )

        records[
            str(accession)
        ] = {

            "STRICT_MATCH":
                strict_match,

            "ANCESTRY_CATEGORIES":
                " | ".join(
                    sorted(categories)
                ),

            "ANCESTRY_N":
                total_n,
        }

    return records


# =============================================================================
# GWAS CATALOG FTP PATH
# =============================================================================

def gwas_bucket(
    accession
):

    numeric = accession.replace(
        "GCST",
        ""
    )

    number = int(
        numeric
    )

    width = max(
        len(numeric),
        8
    )

    start = (
        ((number - 1) // 1000)
        * 1000
        + 1
    )

    end = (
        start
        + 999
    )

    return (
        f"GCST{start:0{width}d}"
        "-"
        f"GCST{end:0{width}d}"
    )


# =============================================================================
# FIND HARMONISED FILE
# =============================================================================

def find_harmonised_file(
    session,
    accession
):

    bucket = gwas_bucket(
        accession
    )

    directory = (
        f"{SUMMARY_ROOT}/"
        f"{bucket}/"
        f"{accession}/"
        f"harmonised/"
    )

    try:

        response = session.get(
            directory,
            timeout=(20, 60)
        )

    except requests.RequestException:

        return None

    if response.status_code != 200:

        return None

    hrefs = re.findall(
        r'href=["\']([^"\']+)["\']',
        response.text,
        flags=re.I
    )

    files = []

    for href in hrefs:

        href = html.unescape(
            href
        )

        name = href.split("/")[-1]

        lower = name.lower()

        if not name:
            continue

        if lower.endswith(
            (
                ".tbi",
                ".log",
                ".yaml",
                ".yml",
                ".md5",
                ".json",
            )
        ):
            continue

        # Strong preference for harmonised GWAS files.
        if (
            ".h.tsv" in lower
            or lower.endswith(".tsv.gz")
            or lower.endswith(".tsv")
        ):

            files.append(
                name
            )

    if not files:

        return None

    def priority(name):

        lower = name.lower()

        if lower.endswith(
            ".h.tsv.gz"
        ):
            return 0

        if lower.endswith(
            ".h.tsv"
        ):
            return 1

        if lower.endswith(
            ".tsv.gz"
        ):
            return 2

        if lower.endswith(
            ".tsv"
        ):
            return 3

        return 10

    files.sort(
        key=priority
    )

    filename = files[0]

    return {

        "URL":
            urljoin(
                directory,
                filename
            ),

        "FILENAME":
            filename,

        "DIRECTORY":
            directory,
    }


# =============================================================================
# GENERATE SLURM DOWNLOAD SCRIPT
# =============================================================================

def generate_slurm_script(
    manifest,
    script_path,
    output_dir,
    phenotype_slug,
    ancestry_slug,
    n_jobs,
    max_parallel,
    partition,
    time_limit,
):

    manifest = Path(manifest).resolve()
    output_dir = Path(output_dir).resolve()
    script_path = Path(script_path).resolve()

    logs = (
        script_path.parent
        / "logs"
        / "step02_download"
        / phenotype_slug
        / ancestry_slug
    ).resolve()

    logs.mkdir(
        parents=True,
        exist_ok=True
    )

    array_spec = (
        f"1-{n_jobs}%{max_parallel}"
    )

    job_name = (
        f"GWAS_{phenotype_slug}_{ancestry_slug}"
    )[:100]

    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --output={logs}/step02_%A_%a.out
#SBATCH --error={logs}/step02_%A_%a.err
#SBATCH --array={array_spec}
#SBATCH --mem={DEFAULT_MEMORY}
#SBATCH --cpus-per-task={DEFAULT_CPUS_PER_TASK}
#SBATCH --ntasks=1

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MANIFEST="{manifest}"
OUTDIR="{output_dir}"

ROW=$(awk -F '\\t' -v id="$SLURM_ARRAY_TASK_ID" '
    NR > 1 && $1 == id {{print; exit}}
' "$MANIFEST")

if [[ -z "$ROW" ]]; then
    echo "Could not find array task $SLURM_ARRAY_TASK_ID in $MANIFEST"
    exit 1
fi

IFS=$'\\t' read -r ARRAY_ID ACCESSION DOWNLOAD_URL FILE_NAME REST <<< "$ROW"

DEST="$OUTDIR/$ACCESSION"

mkdir -p "$DEST"

TARGET="$DEST/$FILE_NAME"
TEMP="$TARGET.part"

echo "============================================================"
echo "GWAS download"
echo "============================================================"
echo "Array task : $ARRAY_ID"
echo "Accession  : $ACCESSION"
echo "URL        : $DOWNLOAD_URL"
echo "Output     : $TARGET"
echo

if [[ -s "$TARGET" ]]; then
    echo "File already exists. Skipping."
    exit 0
fi

if command -v wget >/dev/null 2>&1; then

    wget \\
        --continue \\
        --tries=10 \\
        --timeout=60 \\
        -O "$TEMP" \\
        "$DOWNLOAD_URL"

elif command -v curl >/dev/null 2>&1; then

    curl \\
        --location \\
        --fail \\
        --retry 10 \\
        --retry-delay 5 \\
        --continue-at - \\
        --output "$TEMP" \\
        "$DOWNLOAD_URL"

else

    echo "ERROR: neither wget nor curl is available."
    exit 1
fi

mv "$TEMP" "$TARGET"

echo
echo "DOWNLOAD COMPLETE"
echo "$TARGET"
"""

    script_path.write_text(
        text
    )

    script_path.chmod(
        0o755
    )


# =============================================================================
# ARGUMENTS
# =============================================================================

def arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phenotype",
        required=True
    )

    parser.add_argument(
        "--ancestry",
        required=True,
        help=(
            "EUR, AFR, EAS, SAS, "
            "European, East Asian, etc."
        )
    )

    parser.add_argument(
        "--alias",
        action="append",
        default=[]
    )

    parser.add_argument(
        "--max-studies",
        type=int,
        default=0,
        help=(
            "Maximum eligible GWAS to place in download manifest. "
            "0 = all eligible studies."
        )
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help=(
            "Maximum simultaneous SLURM downloads."
        )
    )

    parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
        help=(
            f"SLURM partition for generated Step 02 jobs. "
            f"Default: {DEFAULT_PARTITION}"
        )
    )

    parser.add_argument(
        "--time",
        type=validate_slurm_time,
        default=DEFAULT_TIME_LIMIT,
        help=(
            "SLURM walltime for generated Step 02 jobs. "
            "Maximum 24 hours. Default: 24:00:00"
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true"
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = arguments()

    phenotype = args.phenotype.strip()

    if not phenotype:

        raise RuntimeError(
            "Phenotype cannot be empty."
        )

    if args.max_studies < 0:

        raise RuntimeError(
            "--max-studies must be >= 0."
        )

    if args.max_parallel < 1:

        raise RuntimeError(
            "--max-parallel must be >= 1."
        )

    ancestry_name = canonical_ancestry(
        args.ancestry
    )

    ancestry_code_value = ancestry_code(
        ancestry_name
    )

    phenotype_slug = slugify(
        phenotype
    )

    ancestry_slug = slugify(
        ancestry_name
    )

    terms = build_terms(
        phenotype,
        args.alias
    )

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------

    cache = Path(
        "reference_metadata/gwas_catalog"
    )

    analysis_dir = (
        Path("01_gwas_catalog")
        / phenotype_slug
        / ancestry_slug
    )

    download_dir = (
        Path("02_summary_stats")
        / phenotype_slug
        / ancestry_slug
        / "raw"
    )

    cache.mkdir(
        parents=True,
        exist_ok=True
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    download_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    studies_file = (
        cache
        / "gwas_catalog_studies.tsv"
    )

    ancestry_file = (
        cache
        / "gwas_catalog_ancestry.tsv"
    )

    session = make_session()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    banner(
        "GWAS DOWNLOAD PLANNER"
    )

    print(
        f"Phenotype : {phenotype}"
    )

    print(
        f"Ancestry  : {ancestry_name}"
    )

    print(
        f"Code      : {ancestry_code_value}"
    )

    print(
        f"Partition : {args.partition}"
    )

    print(
        f"Walltime  : {args.time}"
    )

    print(
        "\nSearch terms:"
    )

    for term in terms:

        print(
            f"  - {term}"
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    banner(
        "GWAS CATALOG METADATA"
    )

    download_metadata(
        session,
        STUDIES_URL,
        studies_file,
        args.refresh
    )

    download_metadata(
        session,
        ANCESTRY_URL,
        ancestry_file,
        args.refresh
    )

    studies = pd.read_csv(
        studies_file,
        sep="\t",
        dtype=str,
        low_memory=False
    )

    ancestry = pd.read_csv(
        ancestry_file,
        sep="\t",
        dtype=str,
        low_memory=False
    )

    # -------------------------------------------------------------------------
    # Phenotype
    # -------------------------------------------------------------------------

    banner(
        "PHENOTYPE FILTER"
    )

    candidates = find_phenotype_studies(
        studies,
        terms
    )

    print(
        f"Phenotype matches: {len(candidates):,}"
    )

    if candidates.empty:

        raise RuntimeError(
            f"No GWAS studies found for phenotype: {phenotype}"
        )

    # -------------------------------------------------------------------------
    # Full summary statistics
    # -------------------------------------------------------------------------

    if (
        "FULL SUMMARY STATISTICS"
        not in candidates.columns
    ):

        raise RuntimeError(
            "GWAS Catalog studies file does not contain "
            "'FULL SUMMARY STATISTICS'."
        )

    candidates = candidates[
        candidates[
            "FULL SUMMARY STATISTICS"
        ].map(
            has_full_summary_stats
        )
    ].copy()

    print(
        f"With full summary statistics: {len(candidates):,}"
    )

    if candidates.empty:

        raise RuntimeError(
            "No phenotype-matching studies have full summary statistics."
        )

    # -------------------------------------------------------------------------
    # Ancestry
    # -------------------------------------------------------------------------

    banner(
        "ANCESTRY FILTER"
    )

    ancestry_info = get_ancestry_information(
        ancestry,
        ancestry_name
    )

    strict_accessions = {

        accession

        for accession, info
        in ancestry_info.items()

        if info[
            "STRICT_MATCH"
        ]
    }

    candidates = candidates[
        candidates[
            "STUDY ACCESSION"
        ].astype(str).isin(
            strict_accessions
        )
    ].copy()

    print(
        f"Strict {ancestry_name} studies: "
        f"{len(candidates):,}"
    )

    if candidates.empty:

        raise RuntimeError(
            f"No strict {ancestry_name} GWAS studies found."
        )

    # -------------------------------------------------------------------------
    # Sample size ranking
    # -------------------------------------------------------------------------

    ancestry_n = []

    categories = []

    for accession in candidates[
        "STUDY ACCESSION"
    ].astype(str):

        info = ancestry_info.get(
            accession,
            {}
        )

        ancestry_n.append(
            int(
                info.get(
                    "ANCESTRY_N",
                    0
                )
            )
        )

        categories.append(
            info.get(
                "ANCESTRY_CATEGORIES",
                ""
            )
        )

    candidates[
        "ANCESTRY_N"
    ] = ancestry_n

    candidates[
        "ANCESTRY_CATEGORIES"
    ] = categories

    if (
        "INITIAL SAMPLE SIZE"
        in candidates.columns
    ):

        candidates[
            "ESTIMATED_INITIAL_N"
        ] = candidates[
            "INITIAL SAMPLE SIZE"
        ].map(
            estimate_sample_size
        )

    else:

        candidates[
            "ESTIMATED_INITIAL_N"
        ] = 0

    candidates[
        "RANKING_N"
    ] = candidates[
        [
            "ANCESTRY_N",
            "ESTIMATED_INITIAL_N"
        ]
    ].max(
        axis=1
    )

    candidates = candidates.sort_values(
        by=[
            "MATCH_SCORE",
            "RANKING_N"
        ],
        ascending=[
            False,
            False
        ]
    )

    candidate_file = (
        analysis_dir
        / "candidate_studies.tsv"
    )

    candidates.to_csv(
        candidate_file,
        sep="\t",
        index=False
    )

    # -------------------------------------------------------------------------
    # Check harmonised data
    # -------------------------------------------------------------------------

    banner(
        "CHECKING HARMONISED GWAS FILES"
    )

    eligible = []

    for _, row in candidates.iterrows():

        accession = str(
            row[
                "STUDY ACCESSION"
            ]
        )

        print(
            f"{accession}: ",
            end="",
            flush=True
        )

        result = find_harmonised_file(
            session,
            accession
        )

        if result is None:

            print(
                "no harmonised file"
            )

            continue

        print(
            result[
                "FILENAME"
            ]
        )

        eligible.append(
            {
                "ROW":
                    row,

                "DOWNLOAD":
                    result,
            }
        )

        if (
            args.max_studies > 0
            and len(eligible)
            >= args.max_studies
        ):

            break

    if not eligible:

        raise RuntimeError(
            "No matching harmonised GWAS files were found."
        )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    manifest_rows = []

    for i, item in enumerate(
        eligible,
        start=1
    ):

        row = item[
            "ROW"
        ]

        download = item[
            "DOWNLOAD"
        ]

        manifest_rows.append(
            {

                "ARRAY_ID":
                    i,

                "STUDY_ACCESSION":
                    str(
                        row[
                            "STUDY ACCESSION"
                        ]
                    ),

                "DOWNLOAD_URL":
                    download[
                        "URL"
                    ],

                "FILE_NAME":
                    download[
                        "FILENAME"
                    ],

                "PHENOTYPE":
                    phenotype,

                # Backward-compatible human-readable ancestry label.
                "ANCESTRY":
                    ancestry_name,

                # Explicit machine-readable ancestry code for downstream
                # resource selection (EUR/AFR/EAS/SAS/AMR/MID).
                "ANCESTRY_CODE":
                    ancestry_code_value,

                "ANCESTRY_LABEL":
                    ancestry_name,

                "RANKING_N":
                    int(
                        row[
                            "RANKING_N"
                        ]
                    ),

                "MATCH_SCORE":
                    int(
                        row[
                            "MATCH_SCORE"
                        ]
                    ),

                "DISEASE_TRAIT":
                    row.get(
                        "DISEASE/TRAIT",
                        ""
                    ),

                "MAPPED_TRAIT":
                    row.get(
                        "MAPPED_TRAIT",
                        ""
                    ),

                "FIRST_AUTHOR":
                    row.get(
                        "FIRST AUTHOR",
                        ""
                    ),

                "PUBMEDID":
                    row.get(
                        "PUBMEDID",
                        ""
                    ),

                "INITIAL_SAMPLE_SIZE":
                    row.get(
                        "INITIAL SAMPLE SIZE",
                        ""
                    ),

            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_file = (
        analysis_dir
        / "GWAS_download_manifest.tsv"
    )

    manifest.to_csv(
        manifest_file,
        sep="\t",
        index=False
    )

    # -------------------------------------------------------------------------
    # Generate Step02
    # -------------------------------------------------------------------------

    script_file = Path(
        f"Step02_Download_GWAS_"
        f"{phenotype_slug}_"
        f"{ancestry_slug}.sh"
    )

    generate_slurm_script(

        manifest=manifest_file,

        script_path=script_file,

        output_dir=download_dir,

        phenotype_slug=phenotype_slug,

        ancestry_slug=ancestry_slug,

        n_jobs=len(manifest),

        max_parallel=max(
            1,
            args.max_parallel
        ),

        partition=args.partition,

        time_limit=args.time,
    )

    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    banner(
        "PLANNING COMPLETE"
    )

    print(
        f"Phenotype           : {phenotype}"
    )

    print(
        f"Ancestry            : {ancestry_name}"
    )

    print(
        f"Ancestry code       : {ancestry_code_value}"
    )

    print(
        f"SLURM partition     : {args.partition}"
    )

    print(
        f"SLURM walltime      : {args.time}"
    )

    print(
        f"Eligible GWAS files : {len(manifest):,}"
    )

    print(
        f"\nCandidate table:\n  {candidate_file}"
    )

    print(
        f"\nDownload manifest:\n  {manifest_file}"
    )

    print(
        f"\nGenerated SLURM script:\n  {script_file}"
    )

    print()

    show = [
        column
        for column in [
            "ARRAY_ID",
            "STUDY_ACCESSION",
            "RANKING_N",
            "DISEASE_TRAIT",
            "FILE_NAME",
        ]
        if column in manifest.columns
    ]

    print(
        manifest[
            show
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "NEXT COMMAND:"
    )

    print()

    print(
        f"  sbatch {script_file}"
    )

    print()


if __name__ == "__main__":

    main()
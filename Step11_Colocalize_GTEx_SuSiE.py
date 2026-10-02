 
"""
===============================================================================
GWAS2m - STEP 11
GTEx v11 SuSiE eQTL/sQTL colocalization screen
===============================================================================

PURPOSE
-------
Generic, phenotype-agnostic fine-mapping colocalization screen.

For each GWAS study:
    Step06 GWAS SuSiE PIP
        ×
    GTEx v11 SuSiE QTL PIP
        =
    vCLPP = GWAS_PIP * GTEX_QTL_PIP

For each locus / tissue / gene / QTL phenotype:
    LCLPP = 1 - product(1 - vCLPP)

IMPORTANT
---------
This is a CLPP-style fine-mapping colocalization SCREEN.
It is NOT formal coloc.susie on full dense regional summary statistics.

The screen:
    * uses ALL Step06 fine-mapped variants
    * scans ALL GTEx v11 SuSiE tissues
    * keeps eQTL and sQTL separate
    * keeps every GCST study separate
    * optionally merges Step10 SpliceAI/Pangolin/GTEx/VEP evidence afterward

Planner:
    python Step11_Colocalize_GTEx_SuSiE.py \
        --phenotype migraine \
        --ancestry EUR

Direct single-study run:
    python Step11_Colocalize_GTEx_SuSiE.py \
        --phenotype migraine \
        --ancestry EUR \
        --index 1

Generated SLURM:
    sbatch Step11_Coloc_migraine_european.sh
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STEP11_VERSION = "1.0.0"

DEFAULT_PARTITION = "general"
DEFAULT_TIME = "24:00:00"
DEFAULT_MEMORY = "64G"
DEFAULT_CPUS = 4
DEFAULT_MAX_PARALLEL = 2

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
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def normalize_text(value) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def slugify(value) -> str:
    return normalize_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = normalize_text(value)

    if key not in ANCESTRY_ALIASES:
        raise ValueError(
            f"Unsupported ancestry {value!r}. "
            "Use EUR, AFR, EAS, SAS, or AMR."
        )

    return ANCESTRY_ALIASES[key]


def safe_read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def validate_walltime(value: str) -> None:
    text = value.strip()
    days = 0

    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)

    parts = text.split(":")

    if len(parts) != 3:
        raise ValueError(
            "--time must use HH:MM:SS or D-HH:MM:SS"
        )

    h, m, s = map(int, parts)

    seconds = (
        days * 86400
        + h * 3600
        + m * 60
        + s
    )

    if seconds <= 0 or seconds > 86400:
        raise ValueError(
            "Requested walltime must be >0 and <=24 hours."
        )


def first_present(
    columns,
    aliases,
):
    lookup = {
        str(c).lower(): c
        for c in columns
    }

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]

    return None


def clean_chr(value):
    text = str(value).strip()
    text = re.sub(
        r"^chr",
        "",
        text,
        flags=re.I,
    )

    try:
        x = int(float(text))
    except Exception:
        return None

    if 1 <= x <= 22:
        return x

    return None


def normalize_allele(value):
    if pd.isna(value):
        return None

    x = str(value).strip().upper()

    if not x or x in {"NA", "NAN", "."}:
        return None

    return x


def make_variant_key(
    chrom,
    pos,
    ref,
    alt,
):
    chrom = clean_chr(chrom)

    try:
        pos = int(float(pos))
    except Exception:
        return None

    ref = normalize_allele(ref)
    alt = normalize_allele(alt)

    if (
        chrom is None
        or pos <= 0
        or ref is None
        or alt is None
    ):
        return None

    return f"{chrom}:{pos}:{ref}:{alt}"


def parse_gtex_variant_id(value):
    """
    Parse common GTEx forms such as:
        chr1_12345_A_G_b38
        1_12345_A_G_b38
        chr1_12345_A_G
    """

    if pd.isna(value):
        return None

    x = str(value).strip()

    m = re.match(
        r"^(?:chr)?(\d+)_(\d+)_([^_]+)_([^_]+)(?:_b\d+)?$",
        x,
        flags=re.I,
    )

    if not m:
        return None

    chrom, pos, ref, alt = m.groups()

    return make_variant_key(
        chrom,
        pos,
        ref,
        alt,
    )


def infer_tissue_from_path(path: Path) -> str:
    name = path.name

    for suffix in [
        ".parquet",
        ".pq",
        ".tsv.gz",
        ".tsv",
        ".txt.gz",
        ".txt",
        ".csv.gz",
        ".csv",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Remove common GTEx/SuSiE decorations.
    name = re.sub(
        r"^GTEx_Analysis_v11_",
        "",
        name,
    )

    name = re.sub(
        r"(_SuSiE|\.SuSiE|_susie|\.susie).*$",
        "",
        name,
        flags=re.I,
    )

    return name


def iter_data_files(root: Path):
    patterns = [
        "*.parquet",
        "*.pq",
        "*.tsv.gz",
        "*.tsv",
        "*.txt.gz",
        "*.txt",
        "*.csv.gz",
        "*.csv",
    ]

    seen = set()

    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                key = str(path.resolve())

                if key not in seen:
                    seen.add(key)
                    yield path.resolve()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.name.lower()

    if suffix.endswith(".parquet") or suffix.endswith(".pq"):
        return pd.read_parquet(path)

    if (
        suffix.endswith(".tsv")
        or suffix.endswith(".tsv.gz")
        or suffix.endswith(".txt")
        or suffix.endswith(".txt.gz")
    ):
        return pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )

    if (
        suffix.endswith(".csv")
        or suffix.endswith(".csv.gz")
    ):
        return pd.read_csv(
            path,
            low_memory=False,
        )

    raise RuntimeError(
        f"Unsupported GTEx file format: {path}"
    )


def atomic_write_tsv_gz(
    df: pd.DataFrame,
    path: Path,
) -> None:

    tmp = Path(
        str(path)
        + ".tmp"
    )

    df.to_csv(
        tmp,
        sep="\t",
        index=False,
        compression="gzip",
    )

    tmp.replace(path)


# =============================================================================
# CLI
# =============================================================================

def arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generic GWAS ↔ GTEx v11 SuSiE "
            "fine-mapping colocalization screen."
        )
    )

    parser.add_argument(
        "--phenotype",
        required=True,
    )

    parser.add_argument(
        "--ancestry",
        required=True,
        help=(
            "EUR, AFR, EAS, SAS, AMR "
            "or corresponding full label."
        ),
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Run exactly one COLOC_TASK_ID "
            "from the existing manifest."
        ),
    )

    parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
    )

    parser.add_argument(
        "--time",
        default=DEFAULT_TIME,
    )

    parser.add_argument(
        "--memory",
        default=DEFAULT_MEMORY,
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=DEFAULT_CPUS,
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help=(
            "Force rebuilding the shared GTEx SuSiE "
            "target-variant cache."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run this study even if Step11 "
            "already completed."
        ),
    )

    return parser.parse_args()


def validate_arguments(args) -> None:
    if not args.phenotype.strip():
        raise ValueError(
            "--phenotype cannot be empty."
        )

    if args.index is not None and args.index < 1:
        raise ValueError(
            "--index must be >=1."
        )

    if args.cpus < 1:
        raise ValueError(
            "--cpus must be >=1."
        )

    if args.max_parallel < 1:
        raise ValueError(
            "--max-parallel must be >=1."
        )

    validate_walltime(
        args.time
    )


# =============================================================================
# GTEx RESOURCE DISCOVERY
# =============================================================================

def resolve_gtex_susie_dirs(
    root: Path,
) -> tuple[Path, Path]:

    base = (
        root
        / "resources"
        / "gtex"
        / "v11"
        / "susie"
    )

    eqtl_candidates = [
        base / "eQTL_SuSiE",
        base / "eQTL",
        base / "eqtl_SuSiE",
    ]

    sqtl_candidates = [
        base / "sQTL_SuSiE",
        base / "sQTL",
        base / "sqtl_SuSiE",
    ]

    eqtl = next(
        (
            p.resolve()
            for p in eqtl_candidates
            if p.exists()
            and p.is_dir()
        ),
        None,
    )

    sqtl = next(
        (
            p.resolve()
            for p in sqtl_candidates
            if p.exists()
            and p.is_dir()
        ),
        None,
    )

    if eqtl is None:
        raise FileNotFoundError(
            "Could not find extracted GTEx v11 SuSiE eQTL directory.\n"
            f"Expected near:\n  {base}"
        )

    if sqtl is None:
        raise FileNotFoundError(
            "Could not find extracted GTEx v11 SuSiE sQTL directory.\n"
            f"Expected near:\n  {base}"
        )

    return (
        eqtl,
        sqtl,
    )


# =============================================================================
# STEP06 INPUT
# =============================================================================

def load_step06_variants(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
    )

    required = [
        "LOCUS_ID",
        "CHR",
        "REFERENCE_POS",
        "REFERENCE_REF",
        "REFERENCE_ALT",
        "PIP",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Step06 file is missing required columns: {missing}\n"
            f"{path}"
        )

    df = df.copy()

    df[
        "PIP"
    ] = pd.to_numeric(
        df[
            "PIP"
        ],
        errors="coerce",
    )

    df[
        "VARIANT_KEY"
    ] = [
        make_variant_key(
            chrom,
            pos,
            ref,
            alt,
        )
        for chrom, pos, ref, alt
        in zip(
            df[
                "CHR"
            ],
            df[
                "REFERENCE_POS"
            ],
            df[
                "REFERENCE_REF"
            ],
            df[
                "REFERENCE_ALT"
            ],
        )
    ]

    df = df.dropna(
        subset=[
            "VARIANT_KEY",
            "PIP",
        ]
    ).copy()

    df = (
        df
        .sort_values(
            [
                "LOCUS_ID",
                "PIP",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "LOCUS_ID",
                "VARIANT_KEY",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# =============================================================================
# PLANNER
# =============================================================================

def planner_mode(args) -> None:
    validate_arguments(args)

    root = Path.cwd().resolve()

    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(
        phenotype
    )

    ancestry_code, ancestry_label = (
        canonical_ancestry(
            args.ancestry
        )
    )

    ancestry_slug = slugify(
        ancestry_label
    )

    eqtl_dir, sqtl_dir = (
        resolve_gtex_susie_dirs(
            root
        )
    )

    step06_root = (
        root
        / "06_finemapping"
        / phenotype_slug
        / ancestry_slug
    )

    if not step06_root.exists():
        raise FileNotFoundError(
            f"Step06 directory missing:\n  {step06_root}"
        )

    step10_root = (
        root
        / "10_pangolin"
        / phenotype_slug
        / ancestry_slug
    )

    out_root = (
        root
        / "11_coloc"
        / phenotype_slug
        / ancestry_slug
    )

    log_root = (
        root
        / "logs"
        / "step11_coloc"
        / phenotype_slug
        / ancestry_slug
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    excluded = []

    accessions = set()

    step06_manifest = (
        step06_root
        / "finemapping_manifest.tsv"
    )

    if step06_manifest.exists():

        mf = pd.read_csv(
            step06_manifest,
            sep="\t",
            dtype=str,
            low_memory=False,
        )

        if "STUDY_ACCESSION" in mf.columns:

            accessions.update(
                x.strip()
                for x in mf[
                    "STUDY_ACCESSION"
                ].dropna().astype(str)
                if x.strip()
            )

    for study_dir in step06_root.glob(
        "GCST*"
    ):

        if study_dir.is_dir():
            accessions.add(
                study_dir.name
            )

    for accession in sorted(
        accessions
    ):

        step06_dir = (
            step06_root
            / accession
        )

        fine_file = (
            step06_dir
            / f"{accession}_finemapped_variants.tsv.gz"
        )

        summary_file = (
            step06_dir
            / "finemapping_summary.json"
        )

        summary = safe_read_json(
            summary_file
        )

        status = str(
            summary.get(
                "STATUS",
                "",
            )
        ).upper()

        n_success = int(
            summary.get(
                "N_LOCI_SUCCESS",
                0,
            )
            or 0
        )

        reason = ""

        if not summary:

            reason = (
                "Step06 summary missing/unreadable"
            )

        elif status == "NO_SIGNIFICANT_LOCI":

            reason = (
                "No significant loci"
            )

        elif n_success <= 0:

            reason = (
                "No successfully fine-mapped loci"
            )

        elif (
            not fine_file.exists()
            or fine_file.stat().st_size == 0
        ):

            reason = (
                "Step06 fine-mapped variant file missing/empty"
            )

        if reason:

            excluded.append(
                {
                    "STUDY_ACCESSION":
                        accession,
                    "STEP06_STATUS":
                        status,
                    "REASON":
                        reason,
                }
            )

            continue

        step10_file = (
            step10_root
            / accession
            / (
                f"{accession}_"
                "SpliceAI_Pangolin_integrated.tsv"
            )
        )

        rows.append(
            {
                "COLOC_TASK_ID":
                    len(rows) + 1,
                "STUDY_ACCESSION":
                    accession,
                "PHENOTYPE":
                    phenotype,
                "ANCESTRY_CODE":
                    ancestry_code,
                "ANCESTRY_LABEL":
                    ancestry_label,
                "STEP06_FINEMAPPED_FILE":
                    str(
                        fine_file.resolve()
                    ),
                "STEP06_SUMMARY_FILE":
                    str(
                        summary_file.resolve()
                    ),
                "STEP10_INTEGRATED_FILE":
                    (
                        str(
                            step10_file.resolve()
                        )
                        if step10_file.exists()
                        else ""
                    ),
                "OUTPUT_DIR":
                    str(
                        (
                            out_root
                            / accession
                        ).resolve()
                    ),
                "GTEX_EQTL_SUSIE_DIR":
                    str(
                        eqtl_dir
                    ),
                "GTEX_SQTL_SUSIE_DIR":
                    str(
                        sqtl_dir
                    ),
                "SHARED_DIR":
                    str(
                        (
                            out_root
                            / "shared"
                        ).resolve()
                    ),
                "REBUILD_CACHE":
                    bool(
                        args.rebuild_cache
                    ),
                "FORCE":
                    bool(
                        args.force
                    ),
                "STEP11_VERSION":
                    STEP11_VERSION,
            }
        )

    excluded_file = (
        out_root
        / "coloc_excluded_studies.tsv"
    )

    pd.DataFrame(
        excluded,
        columns=[
            "STUDY_ACCESSION",
            "STEP06_STATUS",
            "REASON",
        ],
    ).to_csv(
        excluded_file,
        sep="\t",
        index=False,
    )

    if not rows:

        raise RuntimeError(
            "No Step06 studies are eligible for Step11.\n"
            f"Inspect:\n  {excluded_file}"
        )

    manifest = pd.DataFrame(
        rows
    )

    manifest_file = (
        out_root
        / "coloc_manifest.tsv"
    ).resolve()

    manifest.to_csv(
        manifest_file,
        sep="\t",
        index=False,
    )

    pipeline_python = (
        root
        / "envs"
        / "pipeline"
        / "bin"
        / "python"
    )

    if not pipeline_python.exists():

        pipeline_python = Path(
            sys.executable
        ).resolve()

    script_path = Path(
        __file__
    ).resolve()

    bash_file = (
        root
        / (
            f"Step11_Coloc_"
            f"{phenotype_slug}_"
            f"{ancestry_slug}.sh"
        )
    )

    array_spec = (
        f"1-{len(manifest)}"
        f"%{args.max_parallel}"
    )

    job_name = (
        f"Coloc_"
        f"{phenotype_slug}_"
        f"{ancestry_code.lower()}"
    )[:100]

    bash_text = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_root}/coloc.%A_%a.out
#SBATCH --error={log_root}/coloc.%A_%a.err
#SBATCH --array={array_spec}
#SBATCH --mem={args.memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1

set -euo pipefail

cd {shlex.quote(str(root))}

export GWAS_COLOC_MANIFEST={shlex.quote(str(manifest_file))}

{shlex.quote(str(pipeline_python))} {shlex.quote(str(script_path))}
'''

    bash_file.write_text(
        bash_text,
        encoding="utf-8",
    )

    bash_file.chmod(
        0o755
    )

    banner(
        "STEP 11 - GTEx v11 SuSiE COLOCALIZATION PLANNER"
    )

    print(
        f"Phenotype        : {phenotype}"
    )

    print(
        f"Ancestry         : "
        f"{ancestry_label} ({ancestry_code})"
    )

    print(
        f"Eligible studies : {len(manifest)}"
    )

    print(
        f"Excluded studies : {len(excluded)}"
    )

    print(
        f"eQTL SuSiE       : {eqtl_dir}"
    )

    print(
        f"sQTL SuSiE       : {sqtl_dir}"
    )

    print(
        "Tissues          : ALL available GTEx v11 tissues"
    )

    print(
        f"Array            : {array_spec}"
    )

    print()

    print(
        manifest[
            [
                "COLOC_TASK_ID",
                "STUDY_ACCESSION",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Manifest:\n  {manifest_file}"
    )

    print(
        f"Excluded:\n  {excluded_file}"
    )

    print(
        f"Generated SLURM:\n  {bash_file}"
    )

    print()
    print(
        "NEXT COMMAND:"
    )

    print()
    print(
        f"  sbatch {bash_file.name}"
    )

    print()


# =============================================================================
# GTEx SuSiE PARSING
# =============================================================================

def standardize_gtex_susie(
    df: pd.DataFrame,
    *,
    tissue: str,
    qtl_type: str,
    source_file: Path,
) -> pd.DataFrame:

    columns = list(
        df.columns
    )

    variant_col = first_present(
        columns,
        [
            "variant_id",
            "variant",
            "variantid",
            "snp",
            "snpid",
        ],
    )

    chr_col = first_present(
        columns,
        [
            "chr",
            "chrom",
            "chromosome",
        ],
    )

    pos_col = first_present(
        columns,
        [
            "pos",
            "position",
            "bp",
        ],
    )

    ref_col = first_present(
        columns,
        [
            "ref",
            "reference_allele",
        ],
    )

    alt_col = first_present(
        columns,
        [
            "alt",
            "alternate_allele",
        ],
    )

    pip_col = first_present(
        columns,
        [
            "pip",
            "susie_pip",
            "posterior_inclusion_probability",
            "posterior_probability",
        ],
    )

    if pip_col is None:

        # Some GTEx SuSiE exports use combined field names.
        for c in columns:

            low = str(c).lower()

            if (
                "pip" in low
                and "min" not in low
                and "max" not in low
            ):

                pip_col = c
                break

    gene_col = first_present(
        columns,
        [
            "gene_id",
            "gene",
            "geneid",
            "molecular_trait_id",
        ],
    )

    phenotype_col = first_present(
        columns,
        [
            "phenotype_id",
            "phenotype",
            "intron_id",
            "junction_id",
            "molecular_trait_id",
        ],
    )

    cs_col = first_present(
        columns,
        [
            "cs_id",
            "credible_set",
            "cs",
            "component",
        ],
    )

    if pip_col is None:

        return pd.DataFrame()

    out = pd.DataFrame(
        index=df.index
    )

    if variant_col is not None:

        out[
            "VARIANT_KEY"
        ] = df[
            variant_col
        ].map(
            parse_gtex_variant_id
        )

    elif all(
        x is not None
        for x in [
            chr_col,
            pos_col,
            ref_col,
            alt_col,
        ]
    ):

        out[
            "VARIANT_KEY"
        ] = [
            make_variant_key(
                chrom,
                pos,
                ref,
                alt,
            )
            for chrom, pos, ref, alt
            in zip(
                df[
                    chr_col
                ],
                df[
                    pos_col
                ],
                df[
                    ref_col
                ],
                df[
                    alt_col
                ],
            )
        ]

    else:

        return pd.DataFrame()

    out[
        "GTEX_QTL_PIP"
    ] = pd.to_numeric(
        df[
            pip_col
        ],
        errors="coerce",
    )

    out[
        "QTL_GENE"
    ] = (
        df[
            gene_col
        ].astype(str)
        if gene_col is not None
        else ""
    )

    out[
        "QTL_PHENOTYPE"
    ] = (
        df[
            phenotype_col
        ].astype(str)
        if phenotype_col is not None
        else out[
            "QTL_GENE"
        ].astype(str)
    )

    out[
        "GTEX_CS_ID"
    ] = (
        df[
            cs_col
        ].astype(str)
        if cs_col is not None
        else ""
    )

    out[
        "TISSUE"
    ] = tissue

    out[
        "QTL_TYPE"
    ] = qtl_type

    out[
        "GTEX_SOURCE_FILE"
    ] = str(
        source_file
    )

    out = out.dropna(
        subset=[
            "VARIANT_KEY",
            "GTEX_QTL_PIP",
        ]
    ).copy()

    out = out[
        (
            out[
                "GTEX_QTL_PIP"
            ]
            >= 0
        )
        &
        (
            out[
                "GTEX_QTL_PIP"
            ]
            <= 1
        )
    ].copy()

    return out.reset_index(
        drop=True
    )


# =============================================================================
# SHARED CACHE
# =============================================================================

def build_target_keys_from_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    pieces = []

    for _, row in manifest.iterrows():

        path = Path(
            row[
                "STEP06_FINEMAPPED_FILE"
            ]
        )

        accession = str(
            row[
                "STUDY_ACCESSION"
            ]
        )

        if (
            not path.exists()
            or path.stat().st_size == 0
        ):
            continue

        x = load_step06_variants(
            path
        )

        if x.empty:
            continue

        y = x[
            [
                "VARIANT_KEY",
            ]
        ].drop_duplicates().copy()

        y[
            "STUDY_ACCESSION"
        ] = accession

        pieces.append(
            y
        )

    if not pieces:

        raise RuntimeError(
            "Could not obtain any Step06 target variants "
            "from the Step11 manifest."
        )

    all_targets = pd.concat(
        pieces,
        ignore_index=True,
    )

    return all_targets


def scan_gtex_directory(
    *,
    qtl_root: Path,
    qtl_type: str,
    target_set: set[str],
) -> tuple[pd.DataFrame, list[dict]]:

    matches = []
    logs = []

    files = list(
        iter_data_files(
            qtl_root
        )
    )

    if not files:

        raise RuntimeError(
            f"No readable GTEx {qtl_type} SuSiE files were found under:\n"
            f"  {qtl_root}"
        )

    banner(
        f"SCANNING ALL GTEx v11 {qtl_type} SuSiE TISSUES"
    )

    print(
        f"Files discovered : {len(files)}"
    )

    print(
        f"Target variants  : {len(target_set):,}"
    )

    for i, path in enumerate(
        files,
        start=1,
    ):

        tissue = infer_tissue_from_path(
            path
        )

        print(
            f"[{i}/{len(files)}] "
            f"{qtl_type} | {tissue} | {path.name}",
            flush=True,
        )

        status = "OK"
        error = ""
        n_rows = 0
        n_std = 0
        n_match = 0

        try:

            df = read_table(
                path
            )

            n_rows = len(
                df
            )

            std = standardize_gtex_susie(
                df,
                tissue=tissue,
                qtl_type=qtl_type,
                source_file=path,
            )

            n_std = len(
                std
            )

            if not std.empty:

                hit = std[
                    std[
                        "VARIANT_KEY"
                    ].isin(
                        target_set
                    )
                ].copy()

                n_match = len(
                    hit
                )

                if not hit.empty:
                    matches.append(
                        hit
                    )

        except Exception as exc:

            status = "ERROR"
            error = (
                f"{type(exc).__name__}: {exc}"
            )

        logs.append(
            {
                "QTL_TYPE":
                    qtl_type,
                "TISSUE":
                    tissue,
                "FILE":
                    str(path),
                "STATUS":
                    status,
                "N_SOURCE_ROWS":
                    n_rows,
                "N_STANDARDIZED_ROWS":
                    n_std,
                "N_TARGET_MATCH_ROWS":
                    n_match,
                "ERROR":
                    error,
            }
        )

    if matches:

        output = pd.concat(
            matches,
            ignore_index=True,
        )

        output = (
            output
            .sort_values(
                [
                    "QTL_TYPE",
                    "TISSUE",
                    "QTL_GENE",
                    "QTL_PHENOTYPE",
                    "GTEX_QTL_PIP",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
                kind="stable",
            )
            .drop_duplicates(
                subset=[
                    "VARIANT_KEY",
                    "QTL_TYPE",
                    "TISSUE",
                    "QTL_GENE",
                    "QTL_PHENOTYPE",
                    "GTEX_QTL_PIP",
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

    else:

        output = pd.DataFrame(
            columns=[
                "VARIANT_KEY",
                "GTEX_QTL_PIP",
                "QTL_GENE",
                "QTL_PHENOTYPE",
                "GTEX_CS_ID",
                "TISSUE",
                "QTL_TYPE",
                "GTEX_SOURCE_FILE",
            ]
        )

    return (
        output,
        logs,
    )


def ensure_shared_cache(
    manifest_file: Path,
    shared_dir: Path,
    eqtl_dir: Path,
    sqtl_dir: Path,
    rebuild: bool,
) -> dict[str, Path]:

    shared_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_file = (
        shared_dir
        / "GWAS_target_variant_keys.tsv.gz"
    )

    eqtl_file = (
        shared_dir
        / "GTEx_v11_eQTL_SuSiE_target_matches.tsv.gz"
    )

    sqtl_file = (
        shared_dir
        / "GTEx_v11_sQTL_SuSiE_target_matches.tsv.gz"
    )

    scan_file = (
        shared_dir
        / "GTEx_v11_SuSiE_files_scanned.tsv"
    )

    summary_file = (
        shared_dir
        / "GTEx_v11_SuSiE_cache_summary.json"
    )

    complete_file = (
        shared_dir
        / "GTEx_v11_SuSiE_cache.complete"
    )

    required = [
        target_file,
        eqtl_file,
        sqtl_file,
        scan_file,
        summary_file,
        complete_file,
    ]

    if (
        not rebuild
        and all(
            p.exists()
            and p.stat().st_size > 0
            for p in required
        )
    ):

        print(
            f"Reusing shared GTEx SuSiE cache:\n"
            f"  {shared_dir}"
        )

        return {
            "targets":
                target_file,
            "eqtl":
                eqtl_file,
            "sqtl":
                sqtl_file,
            "scan":
                scan_file,
            "summary":
                summary_file,
        }

    # Simple mkdir lock. One SLURM worker builds the cache;
    # others wait.
    lock_dir = (
        shared_dir
        / ".build_lock"
    )

    got_lock = False

    try:

        lock_dir.mkdir()
        got_lock = True

    except FileExistsError:

        got_lock = False

    if not got_lock:

        import time

        print(
            "Another Step11 worker is building the shared GTEx cache."
        )

        for _ in range(
            1440
        ):
            # Up to ~24h.
            if (
                complete_file.exists()
                and all(
                    p.exists()
                    and p.stat().st_size > 0
                    for p in [
                        target_file,
                        eqtl_file,
                        sqtl_file,
                        scan_file,
                        summary_file,
                    ]
                )
            ):

                return {
                    "targets":
                        target_file,
                    "eqtl":
                        eqtl_file,
                    "sqtl":
                        sqtl_file,
                    "scan":
                        scan_file,
                    "summary":
                        summary_file,
                }

            time.sleep(
                60
            )

        raise RuntimeError(
            "Timed out waiting for the shared GTEx SuSiE cache."
        )

    try:

        complete_file.unlink(
            missing_ok=True
        )

        manifest = pd.read_csv(
            manifest_file,
            sep="\t",
            dtype=str,
            low_memory=False,
        )

        targets = (
            build_target_keys_from_manifest(
                manifest
            )
        )

        atomic_write_tsv_gz(
            targets,
            target_file,
        )

        target_set = set(
            targets[
                "VARIANT_KEY"
            ].dropna().astype(str)
        )

        eqtl, log_eqtl = (
            scan_gtex_directory(
                qtl_root=eqtl_dir,
                qtl_type="eQTL",
                target_set=target_set,
            )
        )

        sqtl, log_sqtl = (
            scan_gtex_directory(
                qtl_root=sqtl_dir,
                qtl_type="sQTL",
                target_set=target_set,
            )
        )

        atomic_write_tsv_gz(
            eqtl,
            eqtl_file,
        )

        atomic_write_tsv_gz(
            sqtl,
            sqtl_file,
        )

        logs = pd.DataFrame(
            log_eqtl
            + log_sqtl
        )

        logs.to_csv(
            scan_file,
            sep="\t",
            index=False,
        )

        payload = {
            "STEP11_VERSION":
                STEP11_VERSION,
            "N_TARGET_ROWS":
                int(
                    len(targets)
                ),
            "N_UNIQUE_TARGET_VARIANTS":
                int(
                    targets[
                        "VARIANT_KEY"
                    ].nunique()
                ),
            "N_EQTL_MATCH_ROWS":
                int(
                    len(eqtl)
                ),
            "N_SQTL_MATCH_ROWS":
                int(
                    len(sqtl)
                ),
            "N_EQTL_TISSUES_WITH_MATCH":
                int(
                    eqtl[
                        "TISSUE"
                    ].nunique()
                    if not eqtl.empty
                    else 0
                ),
            "N_SQTL_TISSUES_WITH_MATCH":
                int(
                    sqtl[
                        "TISSUE"
                    ].nunique()
                    if not sqtl.empty
                    else 0
                ),
            "GTEX_EQTL_SUSIE_DIR":
                str(
                    eqtl_dir
                ),
            "GTEX_SQTL_SUSIE_DIR":
                str(
                    sqtl_dir
                ),
            "COMPLETED_UTC":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        write_json(
            summary_file,
            payload,
        )

        complete_file.write_text(
            "complete\n",
            encoding="utf-8",
        )

    finally:

        try:
            lock_dir.rmdir()
        except Exception:
            pass

    return {
        "targets":
            target_file,
        "eqtl":
            eqtl_file,
        "sqtl":
            sqtl_file,
        "scan":
            scan_file,
        "summary":
            summary_file,
    }


# =============================================================================
# COLOCALIZATION
# =============================================================================

def clpp_label(value):
    if pd.isna(value):
        return "NO_VALUE"

    value = float(
        value
    )

    if value >= 0.01:
        return "STRONG_SCREEN_GE_0.01"

    if value >= 0.001:
        return "MODERATE_SCREEN_GE_0.001"

    if value >= 0.0001:
        return "WEAK_SCREEN_GE_0.0001"

    return "BELOW_0.0001"


def multicausal_lclpp(values) -> float:
    values = pd.to_numeric(
        pd.Series(
            values
        ),
        errors="coerce",
    )

    values = values[
        values.notna()
    ]

    values = values.clip(
        lower=0,
        upper=1,
    )

    if values.empty:
        return np.nan

    # numerically stable:
    # 1 - product(1-v)
    log_prod = np.log1p(
        -values.clip(
            upper=
                1 - 1e-15
        )
    ).sum()

    return float(
        1 - np.exp(
            log_prod
        )
    )


def merge_step10_evidence(
    overlap: pd.DataFrame,
    step10_file: Path | None,
) -> pd.DataFrame:

    if (
        step10_file is None
        or not step10_file.exists()
        or step10_file.stat().st_size == 0
    ):

        return overlap

    step10 = pd.read_csv(
        step10_file,
        sep="\t",
        low_memory=False,
    )

    required = [
        "CHR",
        "REFERENCE_POS",
        "REFERENCE_REF",
        "REFERENCE_ALT",
    ]

    if not all(
        c in step10.columns
        for c in required
    ):

        return overlap

    step10 = step10.copy()

    step10[
        "VARIANT_KEY"
    ] = [
        make_variant_key(
            chrom,
            pos,
            ref,
            alt,
        )
        for chrom, pos, ref, alt
        in zip(
            step10[
                "CHR"
            ],
            step10[
                "REFERENCE_POS"
            ],
            step10[
                "REFERENCE_REF"
            ],
            step10[
                "REFERENCE_ALT"
            ],
        )
    ]

    useful = [
        c
        for c in step10.columns
        if c not in {
            "CHR",
            "REFERENCE_POS",
            "REFERENCE_REF",
            "REFERENCE_ALT",
        }
    ]

    step10 = (
        step10[
            useful
        ]
        .drop_duplicates(
            "VARIANT_KEY",
            keep="first",
        )
    )

    rename = {}

    for c in step10.columns:

        if (
            c != "VARIANT_KEY"
            and c in overlap.columns
        ):
            rename[
                c
            ] = (
                "STEP10_"
                + c
            )

    step10 = step10.rename(
        columns=rename
    )

    return overlap.merge(
        step10,
        on="VARIANT_KEY",
        how="left",
        validate="many_to_one",
    )


def summarize_pairs(
    overlap: pd.DataFrame,
) -> pd.DataFrame:

    group_cols = [
        "LOCUS_ID",
        "QTL_TYPE",
        "TISSUE",
        "QTL_GENE",
        "QTL_PHENOTYPE",
    ]

    rows = []

    for keys, group in overlap.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):

        (
            locus,
            qtl_type,
            tissue,
            gene,
            phenotype,
        ) = keys

        g = group.sort_values(
            "VCLPP",
            ascending=False,
            kind="stable",
        )

        top = g.iloc[
            0
        ]

        rows.append(
            {
                "LOCUS_ID":
                    locus,
                "QTL_TYPE":
                    qtl_type,
                "TISSUE":
                    tissue,
                "QTL_GENE":
                    gene,
                "QTL_PHENOTYPE":
                    phenotype,
                "N_SHARED_VARIANTS":
                    int(
                        len(g)
                    ),
                "MAX_GWAS_PIP":
                    float(
                        pd.to_numeric(
                            g[
                                "GWAS_PIP"
                            ],
                            errors="coerce",
                        ).max()
                    ),
                "MAX_QTL_PIP":
                    float(
                        pd.to_numeric(
                            g[
                                "GTEX_QTL_PIP"
                            ],
                            errors="coerce",
                        ).max()
                    ),
                "MAX_VCLPP":
                    float(
                        pd.to_numeric(
                            g[
                                "VCLPP"
                            ],
                            errors="coerce",
                        ).max()
                    ),
                "SUM_VCLPP":
                    float(
                        pd.to_numeric(
                            g[
                                "VCLPP"
                            ],
                            errors="coerce",
                        ).sum()
                    ),
                "LCLPP_MULTICAUSAL":
                    multicausal_lclpp(
                        g[
                            "VCLPP"
                        ]
                    ),
                "TOP_SHARED_VARIANT":
                    top[
                        "VARIANT_KEY"
                    ],
                "TOP_SHARED_GWAS_PIP":
                    top[
                        "GWAS_PIP"
                    ],
                "TOP_SHARED_QTL_PIP":
                    top[
                        "GTEX_QTL_PIP"
                    ],
                "TOP_SHARED_VCLPP":
                    top[
                        "VCLPP"
                    ],
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                *group_cols,
                "N_SHARED_VARIANTS",
                "MAX_GWAS_PIP",
                "MAX_QTL_PIP",
                "MAX_VCLPP",
                "SUM_VCLPP",
                "LCLPP_MULTICAUSAL",
                "TOP_SHARED_VARIANT",
                "TOP_SHARED_GWAS_PIP",
                "TOP_SHARED_QTL_PIP",
                "TOP_SHARED_VCLPP",
                "RANK_WITHIN_LOCUS_QTL_TYPE",
                "COLOC_SCREEN_CLASS",
            ]
        )

    pair = pd.DataFrame(
        rows
    )

    pair[
        "COLOC_SCREEN_CLASS"
    ] = pair[
        "LCLPP_MULTICAUSAL"
    ].apply(
        clpp_label
    )

    pair[
        "RANK_WITHIN_LOCUS_QTL_TYPE"
    ] = (
        pair
        .groupby(
            [
                "LOCUS_ID",
                "QTL_TYPE",
            ],
            dropna=False,
        )[
            "LCLPP_MULTICAUSAL"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return pair.sort_values(
        [
            "LOCUS_ID",
            "QTL_TYPE",
            "LCLPP_MULTICAUSAL",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        kind="stable",
    )


def summarize_genes(
    pair: pd.DataFrame,
) -> pd.DataFrame:

    if pair.empty:

        return pd.DataFrame(
            columns=[
                "QTL_TYPE",
                "QTL_GENE",
                "BEST_LCLPP",
                "N_LOCI",
                "N_TISSUES",
                "N_GENE_TISSUE_PAIRS",
            ]
        )

    return (
        pair
        .groupby(
            [
                "QTL_TYPE",
                "QTL_GENE",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            BEST_LCLPP=(
                "LCLPP_MULTICAUSAL",
                "max",
            ),
            N_LOCI=(
                "LOCUS_ID",
                "nunique",
            ),
            N_TISSUES=(
                "TISSUE",
                "nunique",
            ),
            N_GENE_TISSUE_PAIRS=(
                "TISSUE",
                "size",
            ),
        )
        .sort_values(
            [
                "BEST_LCLPP",
                "N_LOCI",
                "N_TISSUES",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="stable",
        )
    )


def summarize_loci(
    pair: pd.DataFrame,
) -> pd.DataFrame:

    if pair.empty:

        return pd.DataFrame(
            columns=[
                "LOCUS_ID",
                "QTL_TYPE",
                "BEST_LCLPP",
                "N_GENE_TISSUE_PHENOTYPE_PAIRS",
                "N_GENES",
                "N_TISSUES",
            ]
        )

    return (
        pair
        .groupby(
            [
                "LOCUS_ID",
                "QTL_TYPE",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            BEST_LCLPP=(
                "LCLPP_MULTICAUSAL",
                "max",
            ),
            N_GENE_TISSUE_PHENOTYPE_PAIRS=(
                "QTL_GENE",
                "size",
            ),
            N_GENES=(
                "QTL_GENE",
                "nunique",
            ),
            N_TISSUES=(
                "TISSUE",
                "nunique",
            ),
        )
        .sort_values(
            [
                "LOCUS_ID",
                "QTL_TYPE",
            ]
        )
    )


# =============================================================================
# WORKER
# =============================================================================

def worker_mode() -> None:

    task_value = os.environ.get(
        "SLURM_ARRAY_TASK_ID"
    )

    manifest_value = os.environ.get(
        "GWAS_COLOC_MANIFEST"
    )

    if not task_value:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is not defined."
        )

    if not manifest_value:
        raise RuntimeError(
            "GWAS_COLOC_MANIFEST is not defined."
        )

    task_id = int(
        task_value
    )

    manifest_file = Path(
        manifest_value
    ).resolve()

    manifest = pd.read_csv(
        manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    required = [
        "COLOC_TASK_ID",
        "STUDY_ACCESSION",
        "PHENOTYPE",
        "ANCESTRY_CODE",
        "ANCESTRY_LABEL",
        "STEP06_FINEMAPPED_FILE",
        "OUTPUT_DIR",
        "GTEX_EQTL_SUSIE_DIR",
        "GTEX_SQTL_SUSIE_DIR",
        "SHARED_DIR",
    ]

    missing = [
        c
        for c in required
        if c not in manifest.columns
    ]

    if missing:
        raise RuntimeError(
            f"Step11 manifest is missing required columns: {missing}"
        )

    task_numbers = pd.to_numeric(
        manifest[
            "COLOC_TASK_ID"
        ],
        errors="coerce",
    )

    selected = manifest.loc[
        task_numbers == task_id
    ]

    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one Step11 manifest row for task {task_id}; "
            f"found {len(selected)}."
        )

    row = selected.iloc[
        0
    ]

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

    ancestry_code = str(
        row[
            "ANCESTRY_CODE"
        ]
    ).strip()

    ancestry_label = str(
        row[
            "ANCESTRY_LABEL"
        ]
    ).strip()

    fine_file = Path(
        row[
            "STEP06_FINEMAPPED_FILE"
        ]
    ).resolve()

    output_dir = Path(
        row[
            "OUTPUT_DIR"
        ]
    ).resolve()

    eqtl_dir = Path(
        row[
            "GTEX_EQTL_SUSIE_DIR"
        ]
    ).resolve()

    sqtl_dir = Path(
        row[
            "GTEX_SQTL_SUSIE_DIR"
        ]
    ).resolve()

    shared_dir = Path(
        row[
            "SHARED_DIR"
        ]
    ).resolve()

    step10_value = str(
        row.get(
            "STEP10_INTEGRATED_FILE",
            "",
        )
    ).strip()

    step10_file = (
        Path(
            step10_value
        ).resolve()
        if step10_value
        else None
    )

    rebuild_cache = (
        str(
            row.get(
                "REBUILD_CACHE",
                "False",
            )
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )

    force = (
        str(
            row.get(
                "FORCE",
                "False",
            )
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlap_file = (
        output_dir
        / (
            f"{accession}_"
            "GTEx_SuSiE_variant_colocalization.tsv.gz"
        )
    )

    pair_file = (
        output_dir
        / (
            f"{accession}_"
            "GTEx_SuSiE_gene_tissue_colocalization.tsv"
        )
    )

    gene_file = (
        output_dir
        / (
            f"{accession}_"
            "GTEx_SuSiE_gene_summary.tsv"
        )
    )

    locus_file = (
        output_dir
        / (
            f"{accession}_"
            "GTEx_SuSiE_locus_summary.tsv"
        )
    )

    top_file = (
        output_dir
        / (
            f"{accession}_"
            "GTEx_SuSiE_top_candidates.tsv"
        )
    )

    summary_file = (
        output_dir
        / "coloc_summary.json"
    )

    failed_file = (
        output_dir
        / "COLOC_FAILED.txt"
    )

    if failed_file.exists():

        force = True

        print(
            "Previous Step11 failure marker found; forcing rerun."
        )

    if (
        not force
        and summary_file.exists()
        and str(
            safe_read_json(
                summary_file
            ).get(
                "STATUS",
                "",
            )
        ).upper()
        in {
            "COMPLETE",
            "NO_GTEX_SUSIE_OVERLAP",
        }
        and not failed_file.exists()
    ):

        banner(
            "STEP 11 ALREADY COMPLETE"
        )

        print(
            f"Study : {accession}"
        )

        print(
            f"Output: {output_dir}"
        )

        return

    banner(
        "STEP 11 - GTEx v11 SuSiE COLOCALIZATION WORKER"
    )

    print(
        f"Task       : {task_id}"
    )

    print(
        f"Study      : {accession}"
    )

    print(
        f"Phenotype  : {phenotype}"
    )

    print(
        f"Ancestry   : {ancestry_label} ({ancestry_code})"
    )

    print(
        f"Step06     : {fine_file}"
    )

    print(
        f"eQTL SuSiE : {eqtl_dir}"
    )

    print(
        f"sQTL SuSiE : {sqtl_dir}"
    )

    print(
        "Tissues    : ALL GTEx v11 tissues"
    )

    try:

        gwas = load_step06_variants(
            fine_file
        )

        if gwas.empty:

            raise RuntimeError(
                "No usable Step06 fine-mapped variants."
            )

        print()
        print(
            f"GWAS fine-mapped variants : {len(gwas):,}"
        )

        print(
            f"GWAS loci                 : {gwas['LOCUS_ID'].nunique():,}"
        )

        cache = ensure_shared_cache(
            manifest_file=manifest_file,
            shared_dir=shared_dir,
            eqtl_dir=eqtl_dir,
            sqtl_dir=sqtl_dir,
            rebuild=rebuild_cache,
        )

        eqtl = pd.read_csv(
            cache[
                "eqtl"
            ],
            sep="\t",
            compression="gzip",
            low_memory=False,
        )

        sqtl = pd.read_csv(
            cache[
                "sqtl"
            ],
            sep="\t",
            compression="gzip",
            low_memory=False,
        )

        qtl = pd.concat(
            [
                eqtl,
                sqtl,
            ],
            ignore_index=True,
        )

        if qtl.empty:

            overlap = pd.DataFrame()

        else:

            gwas_for_merge = (
                gwas
                .rename(
                    columns={
                        "PIP":
                            "GWAS_PIP"
                    }
                )
            )

            overlap = gwas_for_merge.merge(
                qtl,
                on="VARIANT_KEY",
                how="inner",
                validate="many_to_many",
            )

            if not overlap.empty:

                overlap[
                    "VCLPP"
                ] = (
                    pd.to_numeric(
                        overlap[
                            "GWAS_PIP"
                        ],
                        errors="coerce",
                    )
                    *
                    pd.to_numeric(
                        overlap[
                            "GTEX_QTL_PIP"
                        ],
                        errors="coerce",
                    )
                )

                overlap = merge_step10_evidence(
                    overlap,
                    step10_file,
                )

                overlap = overlap.sort_values(
                    [
                        "LOCUS_ID",
                        "QTL_TYPE",
                        "VCLPP",
                    ],
                    ascending=[
                        True,
                        True,
                        False,
                    ],
                    kind="stable",
                )

        if overlap.empty:

            pd.DataFrame(
                columns=[
                    "LOCUS_ID",
                    "VARIANT_KEY",
                    "GWAS_PIP",
                    "QTL_TYPE",
                    "TISSUE",
                    "QTL_GENE",
                    "QTL_PHENOTYPE",
                    "GTEX_QTL_PIP",
                    "VCLPP",
                ]
            ).to_csv(
                overlap_file,
                sep="\t",
                index=False,
                compression="gzip",
            )

            empty_pair = summarize_pairs(
                pd.DataFrame(
                    columns=[
                        "LOCUS_ID",
                        "QTL_TYPE",
                        "TISSUE",
                        "QTL_GENE",
                        "QTL_PHENOTYPE",
                        "VARIANT_KEY",
                        "GWAS_PIP",
                        "GTEX_QTL_PIP",
                        "VCLPP",
                    ]
                )
            )

            empty_pair.to_csv(
                pair_file,
                sep="\t",
                index=False,
            )

            summarize_genes(
                empty_pair
            ).to_csv(
                gene_file,
                sep="\t",
                index=False,
            )

            summarize_loci(
                empty_pair
            ).to_csv(
                locus_file,
                sep="\t",
                index=False,
            )

            empty_pair.to_csv(
                top_file,
                sep="\t",
                index=False,
            )

            payload = {
                "STEP11_VERSION":
                    STEP11_VERSION,
                "COLOC_TASK_ID":
                    task_id,
                "STUDY_ACCESSION":
                    accession,
                "PHENOTYPE":
                    phenotype,
                "ANCESTRY_CODE":
                    ancestry_code,
                "ANCESTRY_LABEL":
                    ancestry_label,
                "N_GWAS_FINE_MAPPED_VARIANTS":
                    int(
                        len(gwas)
                    ),
                "N_GWAS_LOCI":
                    int(
                        gwas[
                            "LOCUS_ID"
                        ].nunique()
                    ),
                "N_VARIANT_OVERLAP_ROWS":
                    0,
                "N_GENE_TISSUE_PAIRS":
                    0,
                "STATUS":
                    "NO_GTEX_SUSIE_OVERLAP",
                "COMPLETED_UTC":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
            }

            write_json(
                summary_file,
                payload,
            )

            failed_file.unlink(
                missing_ok=True
            )

            banner(
                "STEP 11 COMPLETE - NO GTEx SuSiE OVERLAP"
            )

            print(
                "No matched fine-mapped GTEx SuSiE variants "
                "were found for this GWAS."
            )

            return

        banner(
            "SUMMARIZING COLOCALIZATION EVIDENCE"
        )

        overlap.to_csv(
            overlap_file,
            sep="\t",
            index=False,
            compression="gzip",
        )

        pair = summarize_pairs(
            overlap
        )

        genes = summarize_genes(
            pair
        )

        loci = summarize_loci(
            pair
        )

        pair.to_csv(
            pair_file,
            sep="\t",
            index=False,
        )

        genes.to_csv(
            gene_file,
            sep="\t",
            index=False,
        )

        loci.to_csv(
            locus_file,
            sep="\t",
            index=False,
        )

        top = pair[
            (
                pair[
                    "RANK_WITHIN_LOCUS_QTL_TYPE"
                ]
                <= 3
            )
            |
            (
                pair[
                    "LCLPP_MULTICAUSAL"
                ]
                >= 0.001
            )
        ].copy()

        top.to_csv(
            top_file,
            sep="\t",
            index=False,
        )

        payload = {
            "STEP11_VERSION":
                STEP11_VERSION,
            "COLOC_TASK_ID":
                task_id,
            "STUDY_ACCESSION":
                accession,
            "PHENOTYPE":
                phenotype,
            "ANCESTRY_CODE":
                ancestry_code,
            "ANCESTRY_LABEL":
                ancestry_label,
            "N_GWAS_FINE_MAPPED_VARIANTS":
                int(
                    len(gwas)
                ),
            "N_GWAS_LOCI":
                int(
                    gwas[
                        "LOCUS_ID"
                    ].nunique()
                ),
            "N_VARIANT_OVERLAP_ROWS":
                int(
                    len(overlap)
                ),
            "N_UNIQUE_SHARED_VARIANTS":
                int(
                    overlap[
                        "VARIANT_KEY"
                    ].nunique()
                ),
            "N_COLOC_LOCI":
                int(
                    overlap[
                        "LOCUS_ID"
                    ].nunique()
                ),
            "N_GENE_TISSUE_PAIRS":
                int(
                    len(pair)
                ),
            "N_GENES":
                int(
                    pair[
                        "QTL_GENE"
                    ].nunique()
                ),
            "N_TISSUES":
                int(
                    pair[
                        "TISSUE"
                    ].nunique()
                ),
            "N_PAIR_LCLPP_GE_0_01":
                int(
                    (
                        pair[
                            "LCLPP_MULTICAUSAL"
                        ]
                        >= 0.01
                    ).sum()
                ),
            "N_PAIR_LCLPP_GE_0_001":
                int(
                    (
                        pair[
                            "LCLPP_MULTICAUSAL"
                        ]
                        >= 0.001
                    ).sum()
                ),
            "MAX_LCLPP":
                float(
                    pair[
                        "LCLPP_MULTICAUSAL"
                    ].max()
                ),
            "VARIANT_COLOC_FILE":
                str(
                    overlap_file
                ),
            "GENE_TISSUE_COLOC_FILE":
                str(
                    pair_file
                ),
            "GENE_SUMMARY_FILE":
                str(
                    gene_file
                ),
            "LOCUS_SUMMARY_FILE":
                str(
                    locus_file
                ),
            "TOP_CANDIDATES_FILE":
                str(
                    top_file
                ),
            "STATUS":
                "COMPLETE",
            "COMPLETED_UTC":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        write_json(
            summary_file,
            payload,
        )

        failed_file.unlink(
            missing_ok=True
        )

        banner(
            "STEP 11 COMPLETE"
        )

        print(
            f"Variant overlap rows      : {len(overlap):,}"
        )

        print(
            f"Unique shared variants    : "
            f"{overlap['VARIANT_KEY'].nunique():,}"
        )

        print(
            f"GWAS loci with overlap    : "
            f"{overlap['LOCUS_ID'].nunique():,}"
        )

        print(
            f"Gene/tissue/QTL pairs     : "
            f"{len(pair):,}"
        )

        print(
            f"Genes                     : "
            f"{pair['QTL_GENE'].nunique():,}"
        )

        print(
            f"Tissues                   : "
            f"{pair['TISSUE'].nunique():,}"
        )

        print(
            f"LCLPP >= 0.01             : "
            f"{(pair['LCLPP_MULTICAUSAL'] >= 0.01).sum():,}"
        )

        print(
            f"LCLPP >= 0.001            : "
            f"{(pair['LCLPP_MULTICAUSAL'] >= 0.001).sum():,}"
        )

        print()
        print(
            "MAIN downstream table:"
        )

        print(
            f"  {pair_file}"
        )

        print()
        print(
            "NOTE: CLPP/LCLPP here are screening scores, "
            "not formal coloc.susie posterior probabilities."
        )

    except Exception as error:

        failure_text = (
            "STEP 11 GTEx SuSiE COLOCALIZATION FAILED\n\n"
            f"Task ID: {task_id}\n"
            f"Accession: {accession}\n"
            f"Phenotype: {phenotype}\n"
            f"Ancestry: {ancestry_label} ({ancestry_code})\n"
            f"Step06: {fine_file}\n\n"
            f"Error:\n"
            f"{type(error).__name__}: {error}\n\n"
            f"{traceback.format_exc()}"
        )

        failed_file.write_text(
            failure_text,
            encoding="utf-8",
        )

        print(
            failure_text,
            file=sys.stderr,
        )

        raise


# =============================================================================
# DIRECT INDEX MODE
# =============================================================================

def direct_index_mode(
    args,
) -> None:

    validate_arguments(
        args
    )

    root = Path.cwd().resolve()

    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(
        phenotype
    )

    ancestry_code, ancestry_label = (
        canonical_ancestry(
            args.ancestry
        )
    )

    ancestry_slug = slugify(
        ancestry_label
    )

    manifest_file = (
        root
        / "11_coloc"
        / phenotype_slug
        / ancestry_slug
        / "coloc_manifest.tsv"
    ).resolve()

    if not manifest_file.exists():

        raise FileNotFoundError(
            "Step11 manifest does not exist:\n"
            f"  {manifest_file}\n\n"
            "Run the planner first without --index:\n\n"
            f"  python {Path(__file__).name} "
            f"--phenotype {shlex.quote(phenotype)} "
            f"--ancestry {shlex.quote(ancestry_code)}"
        )

    manifest = pd.read_csv(
        manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    task_numbers = pd.to_numeric(
        manifest[
            "COLOC_TASK_ID"
        ],
        errors="coerce",
    )

    selected = manifest.loc[
        task_numbers == args.index
    ]

    if len(selected) != 1:

        available = [
            int(x)
            for x in task_numbers.dropna()
        ]

        raise RuntimeError(
            f"Index {args.index} is not present exactly once.\n"
            f"Available indices: {available}"
        )

    row = selected.iloc[
        0
    ]

    banner(
        "STEP 11 - DIRECT SINGLE-INDEX MODE"
    )

    print(
        f"Manifest : {manifest_file}"
    )

    print(
        f"Index    : {args.index}"
    )

    print(
        f"Study    : {row['STUDY_ACCESSION']}"
    )

    print()
    print(
        "NOTE: direct mode runs on the current node. "
        "Use srun on HPC for the first cache-building run."
    )

    old_task = os.environ.get(
        "SLURM_ARRAY_TASK_ID"
    )

    old_manifest = os.environ.get(
        "GWAS_COLOC_MANIFEST"
    )

    old_cpus = os.environ.get(
        "SLURM_CPUS_PER_TASK"
    )

    try:

        os.environ[
            "SLURM_ARRAY_TASK_ID"
        ] = str(
            args.index
        )

        os.environ[
            "GWAS_COLOC_MANIFEST"
        ] = str(
            manifest_file
        )

        if old_cpus is None:

            os.environ[
                "SLURM_CPUS_PER_TASK"
            ] = str(
                args.cpus
            )

        worker_mode()

    finally:

        if old_task is None:

            os.environ.pop(
                "SLURM_ARRAY_TASK_ID",
                None,
            )

        else:

            os.environ[
                "SLURM_ARRAY_TASK_ID"
            ] = old_task

        if old_manifest is None:

            os.environ.pop(
                "GWAS_COLOC_MANIFEST",
                None,
            )

        else:

            os.environ[
                "GWAS_COLOC_MANIFEST"
            ] = old_manifest

        if old_cpus is None:

            os.environ.pop(
                "SLURM_CPUS_PER_TASK",
                None,
            )

        else:

            os.environ[
                "SLURM_CPUS_PER_TASK"
            ] = old_cpus


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    if (
        os.environ.get(
            "SLURM_ARRAY_TASK_ID"
        )
        and
        os.environ.get(
            "GWAS_COLOC_MANIFEST"
        )
    ):

        worker_mode()
        return

    args = arguments()

    if args.index is not None:

        direct_index_mode(
            args
        )

    else:

        planner_mode(
            args
        )


if __name__ == "__main__":
    main()
 

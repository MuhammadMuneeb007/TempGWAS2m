 
# -*- coding: utf-8 -*-

"""
===============================================================================
GWAS2m - STEP 10
Pangolin splicing prediction + integration with Step 09 SpliceAI evidence
===============================================================================

PURPOSE
-------
Generic, phenotype-agnostic Pangolin stage.

Input:
    09_splicing/<phenotype>/<ancestry>/<GCST>/
        <GCST>_SpliceAI_variant_summary.tsv
        input/<GCST>_SpliceAI_input.normalized.vcf
        spliceai_summary.json

Output:
    10_pangolin/<phenotype>/<ancestry>/<GCST>/

Important design rules:
    * studies are NEVER merged
    * no phenotype-specific logic
    * CSV mode is used for Pangolin to avoid PyVCF compatibility problems
    * all variants remain in the final integrated table
    * Pangolin-unscored != score zero
    * Pangolin percentiles are dataset-relative ranking features only
      and are NOT validated biological thresholds

Planner:
    python Step10_Run_Pangolin.py \
        --phenotype migraine \
        --ancestry EUR

Direct one-study test:
    python Step10_Run_Pangolin.py \
        --phenotype migraine \
        --ancestry EUR \
        --index 1

Generated SLURM array:
    sbatch Step10_Pangolin_migraine_european.sh
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STEP10_VERSION = "1.0.0"

DEFAULT_PARTITION = "general"
DEFAULT_TIME = "24:00:00"
DEFAULT_MEMORY = "64G"
DEFAULT_CPUS = 4
DEFAULT_MAX_PARALLEL = 2

DEFAULT_DISTANCE = 500
DEFAULT_MASK = "False"

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
# HELPERS
# =============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 104)
    print(text)
    print("=" * 104)


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
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def read_resource_env(root: Path) -> dict[str, str]:
    path = root / "resource_paths.env"
    values: dict[str, str] = {}

    if not path.exists():
        return values

    for line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():

        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')

        if key.startswith("export "):
            key = key[7:].strip()

        values[key] = value

    return values


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def to_float(value):
    if value is None:
        return np.nan

    value = str(value).strip()

    if value in {"", ".", "NA", "nan", "None"}:
        return np.nan

    try:
        return float(value)
    except Exception:
        return np.nan


def parse_pos_score(value):
    if not value or ":" not in str(value):
        return np.nan, np.nan

    pos, score = str(value).split(":", 1)

    try:
        pos = int(float(pos))
    except Exception:
        pos = np.nan

    return pos, to_float(score)


def validate_walltime(value: str) -> None:
    text = value.strip()
    days = 0

    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)

    parts = text.split(":")

    if len(parts) != 3:
        raise ValueError("--time must use HH:MM:SS or D-HH:MM:SS")

    h, m, s = map(int, parts)

    seconds = days * 86400 + h * 3600 + m * 60 + s

    if seconds <= 0 or seconds > 86400:
        raise ValueError("Requested walltime must be >0 and <=24 hours.")


def run_command(
    command,
    log_file: Path | None = None,
    env: dict | None = None,
) -> str:

    command = [str(x) for x in command]

    print()
    print("$ " + " ".join(shlex.quote(x) for x in command), flush=True)

    process_env = os.environ.copy()

    if env:
        process_env.update(
            {
                str(k): str(v)
                for k, v in env.items()
            }
        )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=process_env,
    )

    captured = []
    handle = None

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_file, "w", encoding="utf-8")

    try:
        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            captured.append(line)

            if handle:
                handle.write(line)
                handle.flush()

    finally:
        if handle:
            handle.close()

    rc = process.wait()
    output = "".join(captured)

    if rc != 0:
        raise RuntimeError(
            f"Command failed with exit code {rc}:\n"
            f"{' '.join(command)}\n\n"
            f"{output[-5000:]}"
        )

    return output


# =============================================================================
# CLI
# =============================================================================

def arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run Pangolin for completed Step09 studies and integrate "
            "Pangolin + SpliceAI + GTEx + VEP + fine-mapping evidence."
        )
    )

    parser.add_argument("--phenotype", required=True)

    parser.add_argument(
        "--ancestry",
        required=True,
        help="EUR, AFR, EAS, SAS, AMR or corresponding full label.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run exactly one PANGOLIN_TASK_ID from the existing manifest.",
    )

    parser.add_argument(
        "--distance",
        type=int,
        default=DEFAULT_DISTANCE,
        help=(
            "Pangolin -d distance around each variant. "
            f"Default: {DEFAULT_DISTANCE}."
        ),
    )

    parser.add_argument(
        "--mask",
        choices=["False", "True"],
        default=DEFAULT_MASK,
        help=(
            "Pangolin masking mode. Default False so alternative splice "
            "effects are retained."
        ),
    )

    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--time", default=DEFAULT_TIME)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
    )

    parser.add_argument(
        "--pangolin",
        default=None,
        help="Explicit Pangolin executable path.",
    )

    parser.add_argument(
        "--pangolin-db",
        default=None,
        help="Explicit Pangolin gffutils annotation DB.",
    )

    parser.add_argument(
        "--fasta",
        default=None,
        help="Explicit GRCh38 FASTA.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when a completed Step10 output exists.",
    )

    return parser.parse_args()


def validate_arguments(args) -> None:
    if not args.phenotype.strip():
        raise ValueError("--phenotype cannot be empty.")

    if args.index is not None and args.index < 1:
        raise ValueError("--index must be >=1.")

    if args.distance < 1:
        raise ValueError("--distance must be >=1.")

    if args.cpus < 1:
        raise ValueError("--cpus must be >=1.")

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >=1.")

    validate_walltime(args.time)


# =============================================================================
# RESOURCE RESOLUTION
# =============================================================================

def resolve_existing_file(
    candidates,
    label: str,
    executable: bool = False,
) -> Path:

    seen = set()

    for value in candidates:
        if not value:
            continue

        p = Path(value).expanduser().resolve()

        if str(p) in seen:
            continue

        seen.add(str(p))

        if (
            p.exists()
            and p.is_file()
            and p.stat().st_size > 0
            and (
                not executable
                or os.access(p, os.X_OK)
            )
        ):
            return p

    raise FileNotFoundError(
        f"Could not find {label}.\n"
        "Checked:\n  "
        + "\n  ".join(sorted(seen))
    )


def resolve_resources(root: Path, args) -> dict[str, Path]:
    env = read_resource_env(root)

    pangolin_candidates = [
        args.pangolin,
        os.environ.get("PANGOLIN"),
        env.get("PANGOLIN"),
        root / "envs" / "pangolin" / "bin" / "pangolin",
    ]

    path_pangolin = shutil.which("pangolin")

    if path_pangolin:
        pangolin_candidates.append(path_pangolin)

    db_candidates = [
        args.pangolin_db,
        os.environ.get("PANGOLIN_DB"),
        env.get("PANGOLIN_DB"),
        root / "resources" / "pangolin"
        / "gencode.v50.primary_assembly.annotation.db",
    ]

    fasta_candidates = [
        args.fasta,
        os.environ.get("GRCH38_FASTA"),
        env.get("GRCH38_FASTA"),
        root / "resources" / "genome" / "GRCh38"
        / "GRCh38.primary_assembly.genome.fa",
    ]

    return {
        "PANGOLIN": resolve_existing_file(
            pangolin_candidates,
            "Pangolin executable",
            executable=True,
        ),
        "PANGOLIN_DB": resolve_existing_file(
            db_candidates,
            "Pangolin annotation database",
        ),
        "FASTA": resolve_existing_file(
            fasta_candidates,
            "GRCh38 FASTA",
        ),
    }


# =============================================================================
# PLANNER
# =============================================================================

def planner_mode(args) -> None:
    validate_arguments(args)

    root = Path.cwd().resolve()

    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)

    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)

    resources = resolve_resources(root, args)

    step09_root = (
        root
        / "09_splicing"
        / phenotype_slug
        / ancestry_slug
    )

    if not step09_root.exists():
        raise FileNotFoundError(
            f"Step09 output directory not found:\n  {step09_root}"
        )

    out_root = (
        root
        / "10_pangolin"
        / phenotype_slug
        / ancestry_slug
    )

    log_root = (
        root
        / "logs"
        / "step10_pangolin"
        / phenotype_slug
        / ancestry_slug
    )

    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    rows = []
    excluded = []

    for study_dir in sorted(step09_root.glob("GCST*")):

        if not study_dir.is_dir():
            continue

        accession = study_dir.name

        step09_summary_file = (
            study_dir
            / "spliceai_summary.json"
        )

        step09_summary = safe_read_json(
            step09_summary_file
        )

        variant_file = (
            study_dir
            / f"{accession}_SpliceAI_variant_summary.tsv"
        )

        normalized_vcf = (
            study_dir
            / "input"
            / f"{accession}_SpliceAI_input.normalized.vcf"
        )

        reason = ""

        if not step09_summary:
            reason = "Step09 spliceai_summary.json missing/unreadable"

        elif (
            str(
                step09_summary.get(
                    "STATUS",
                    "",
                )
            ).upper()
            !=
            "COMPLETE"
        ):
            reason = (
                "Step09 status is "
                f"{step09_summary.get('STATUS', 'UNKNOWN')}"
            )

        elif (
            not variant_file.exists()
            or variant_file.stat().st_size == 0
        ):
            reason = "Step09 variant summary missing/empty"

        elif (
            not normalized_vcf.exists()
            or normalized_vcf.stat().st_size == 0
        ):
            reason = "Step09 normalized VCF missing/empty"

        if reason:
            excluded.append(
                {
                    "STUDY_ACCESSION": accession,
                    "STEP09_STATUS": step09_summary.get(
                        "STATUS",
                        "",
                    ),
                    "REASON": reason,
                }
            )
            continue

        rows.append(
            {
                "PANGOLIN_TASK_ID": len(rows) + 1,
                "STUDY_ACCESSION": accession,
                "PHENOTYPE": phenotype,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
                "STEP09_VARIANT_FILE": str(
                    variant_file.resolve()
                ),
                "STEP09_NORMALIZED_VCF": str(
                    normalized_vcf.resolve()
                ),
                "STEP09_SUMMARY_FILE": str(
                    step09_summary_file.resolve()
                ),
                "OUTPUT_DIR": str(
                    (out_root / accession).resolve()
                ),
                "PANGOLIN": str(resources["PANGOLIN"]),
                "PANGOLIN_DB": str(resources["PANGOLIN_DB"]),
                "GRCH38_FASTA": str(resources["FASTA"]),
                "DISTANCE": args.distance,
                "MASK": args.mask,
                "FORCE": bool(args.force),
                "STEP10_VERSION": STEP10_VERSION,
            }
        )

    excluded_file = (
        out_root
        / "pangolin_excluded_studies.tsv"
    )

    pd.DataFrame(
        excluded,
        columns=[
            "STUDY_ACCESSION",
            "STEP09_STATUS",
            "REASON",
        ],
    ).to_csv(
        excluded_file,
        sep="\t",
        index=False,
    )

    if not rows:
        raise RuntimeError(
            "No completed Step09 studies are currently eligible for Step10.\n"
            f"Inspect:\n  {excluded_file}"
        )

    manifest = pd.DataFrame(rows)

    manifest_file = (
        out_root
        / "pangolin_manifest.tsv"
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
        pipeline_python = Path(sys.executable).resolve()

    script_path = Path(__file__).resolve()

    bash_file = (
        root
        / (
            f"Step10_Pangolin_"
            f"{phenotype_slug}_"
            f"{ancestry_slug}.sh"
        )
    )

    array_spec = f"1-{len(manifest)}"

    job_name = (
        f"Pangolin_"
        f"{phenotype_slug}_"
        f"{ancestry_code.lower()}"
    )[:100]

    bash_text = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_root}/pangolin.%A_%a.out
#SBATCH --error={log_root}/pangolin.%A_%a.err
#SBATCH --array={array_spec}
#SBATCH --mem={args.memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1

set -euo pipefail

cd {shlex.quote(str(root))}

export GWAS_PANGOLIN_MANIFEST={shlex.quote(str(manifest_file))}

{shlex.quote(str(pipeline_python))} {shlex.quote(str(script_path))}
'''

    bash_file.write_text(
        bash_text,
        encoding="utf-8",
    )

    bash_file.chmod(0o755)

    banner(
        "STEP 10 - PANGOLIN PLANNER"
    )

    print(f"Phenotype        : {phenotype}")
    print(
        f"Ancestry         : "
        f"{ancestry_label} ({ancestry_code})"
    )
    print(f"Eligible studies : {len(manifest)}")
    print(f"Excluded studies : {len(excluded)}")
    print(f"Pangolin         : {resources['PANGOLIN']}")
    print(f"Pangolin DB      : {resources['PANGOLIN_DB']}")
    print(f"GRCh38 FASTA     : {resources['FASTA']}")
    print(f"Distance (-d)    : {args.distance}")
    print(f"Mask (-m)        : {args.mask}")
    print(f"Array            : {array_spec}")

    print()
    print(
        manifest[
            [
                "PANGOLIN_TASK_ID",
                "STUDY_ACCESSION",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Manifest:\n  {manifest_file}")
    print(f"Excluded:\n  {excluded_file}")
    print(f"Generated SLURM:\n  {bash_file}")

    print()
    print("NEXT COMMAND:")
    print()
    print(f"  sbatch {bash_file.name}")
    print()


# =============================================================================
# INPUT PREPARATION
# =============================================================================

def read_normalized_vcf(
    path: Path,
) -> pd.DataFrame:

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            if line.startswith("#"):
                continue

            fields = (
                line
                .rstrip("\n")
                .split("\t")
            )

            if len(fields) < 5:
                continue

            chrom, pos, vid, ref, alt = fields[:5]

            if not vid or vid == ".":
                continue

            for allele in alt.split(","):

                rows.append(
                    {
                        "CHROM": chrom,
                        "POS": int(pos),
                        "REF": ref.upper(),
                        "ALT": allele.upper(),
                        "VEP_ID": vid,
                    }
                )

    if not rows:
        raise RuntimeError(
            f"No variants could be read from normalized VCF:\n{path}"
        )

    df = pd.DataFrame(rows)

    duplicates = df["VEP_ID"].duplicated(
        keep=False
    )

    if duplicates.any():

        repeated = (
            df.loc[
                duplicates,
                "VEP_ID",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Step09 normalized VCF contains duplicated VEP_ID values. "
            "Pangolin integration requires one normalized allele per VEP_ID.\n"
            f"Examples: {repeated[:10]}"
        )

    return df.reset_index(drop=True)


def prepare_pangolin_csv(
    normalized_vcf: Path,
    output_csv: Path,
) -> pd.DataFrame:

    df = read_normalized_vcf(
        normalized_vcf
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df[
        [
            "CHROM",
            "POS",
            "REF",
            "ALT",
            "VEP_ID",
        ]
    ].to_csv(
        output_csv,
        index=False,
    )

    return df


# =============================================================================
# PANGOLIN
# =============================================================================

def run_pangolin(
    executable: Path,
    input_csv: Path,
    fasta: Path,
    annotation_db: Path,
    prefix: Path,
    log_file: Path,
    distance: int,
    mask: str,
    force: bool,
) -> Path:

    output_csv = Path(
        str(prefix)
        + ".csv"
    )

    done_file = Path(
        str(prefix)
        + ".done"
    )

    if (
        not force
        and done_file.exists()
        and output_csv.exists()
        and output_csv.stat().st_size > 100
    ):
        print(
            f"Reusing completed Pangolin output:\n"
            f"  {output_csv}"
        )
        return output_csv

    for stale in [
        output_csv,
        done_file,
    ]:
        if stale.exists():
            stale.unlink()

    command = [
        executable,
        "-m",
        mask,
        "-d",
        str(distance),
        "-c",
        "CHROM,POS,REF,ALT",
        input_csv,
        fasta,
        annotation_db,
        prefix,
    ]

    run_command(
        command,
        log_file=log_file,
    )

    if (
        not output_csv.exists()
        or output_csv.stat().st_size <= 100
    ):
        raise RuntimeError(
            "Pangolin did not create a usable CSV output.\n"
            f"Expected:\n  {output_csv}"
        )

    done_file.write_text(
        (
            "complete\n"
            f"step10_version={STEP10_VERSION}\n"
            f"distance={distance}\n"
            f"mask={mask}\n"
            "mode=csv\n"
        ),
        encoding="utf-8",
    )

    return output_csv


# =============================================================================
# PANGOLIN OUTPUT PARSER
# =============================================================================

def parse_pangolin_csv(
    path: Path,
) -> tuple[pd.DataFrame, int, int]:

    rows = []
    n_records = 0
    n_with_prediction = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        header = (
            handle
            .readline()
            .rstrip("\n")
        )

        header_fields = header.split(",")

        expected_prefix = [
            "CHROM",
            "POS",
            "REF",
            "ALT",
            "VEP_ID",
        ]

        if (
            len(header_fields) < 6
            or header_fields[:5] != expected_prefix
            or header_fields[-1] != "Pangolin"
        ):
            raise RuntimeError(
                "Unexpected Pangolin CSV header.\n"
                f"Observed:\n  {header}\n"
                "Expected the first columns:\n"
                "  CHROM,POS,REF,ALT,VEP_ID,...,Pangolin"
            )

        for line in handle:

            line = line.rstrip("\n")

            if not line:
                continue

            n_records += 1

            # Pangolin does not quote the Pangolin field when multiple
            # gene predictions are comma-separated, so split only the
            # first five commas.
            parts = line.split(",", 5)

            if len(parts) < 6:
                continue

            chrom, pos, ref, alt, vep_id, value = parts

            value = value.strip()

            if not value:
                continue

            n_with_prediction += 1

            for prediction in value.split(","):

                prediction = prediction.strip()

                if not prediction:
                    continue

                fields = prediction.split("|")

                if len(fields) < 3:
                    continue

                gene = fields[0].strip()

                gain_pos, gain_score = (
                    parse_pos_score(
                        fields[1]
                    )
                )

                loss_pos, loss_score = (
                    parse_pos_score(
                        fields[2]
                    )
                )

                warning = ""

                if len(fields) > 3:
                    warning = "|".join(
                        fields[3:]
                    )

                    warning = re.sub(
                        r"^Warnings?:",
                        "",
                        warning,
                        flags=re.I,
                    ).strip()

                gain_abs = (
                    abs(gain_score)
                    if pd.notna(gain_score)
                    else np.nan
                )

                loss_abs = (
                    abs(loss_score)
                    if pd.notna(loss_score)
                    else np.nan
                )

                finite = [
                    x
                    for x in [
                        gain_abs,
                        loss_abs,
                    ]
                    if pd.notna(x)
                ]

                max_abs = (
                    max(finite)
                    if finite
                    else np.nan
                )

                if pd.isna(max_abs):

                    max_event = np.nan
                    max_position = np.nan
                    max_signed = np.nan

                elif (
                    pd.notna(gain_abs)
                    and (
                        pd.isna(loss_abs)
                        or gain_abs >= loss_abs
                    )
                ):

                    max_event = "SPLICE_GAIN"
                    max_position = gain_pos
                    max_signed = gain_score

                else:

                    max_event = "SPLICE_LOSS"
                    max_position = loss_pos
                    max_signed = loss_score

                rows.append(
                    {
                        "VEP_ID": vep_id,
                        "PANGOLIN_CHROM": chrom,
                        "PANGOLIN_POS": int(pos),
                        "PANGOLIN_REF": ref,
                        "PANGOLIN_ALT": alt,
                        "PANGOLIN_GENE": gene,
                        "PANGOLIN_GAIN_SCORE": gain_score,
                        "PANGOLIN_GAIN_POS": gain_pos,
                        "PANGOLIN_LOSS_SCORE": loss_score,
                        "PANGOLIN_LOSS_POS": loss_pos,
                        "PANGOLIN_MAX_ABS": max_abs,
                        "PANGOLIN_MAX_SIGNED": max_signed,
                        "PANGOLIN_MAX_EVENT": max_event,
                        "PANGOLIN_MAX_POSITION": max_position,
                        "PANGOLIN_WARNINGS": warning,
                    }
                )

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        # A valid run may in principle score zero variants. Do not claim
        # those variants have score zero; preserve them as unscored.
        predictions = pd.DataFrame(
            columns=[
                "VEP_ID",
                "PANGOLIN_CHROM",
                "PANGOLIN_POS",
                "PANGOLIN_REF",
                "PANGOLIN_ALT",
                "PANGOLIN_GENE",
                "PANGOLIN_GAIN_SCORE",
                "PANGOLIN_GAIN_POS",
                "PANGOLIN_LOSS_SCORE",
                "PANGOLIN_LOSS_POS",
                "PANGOLIN_MAX_ABS",
                "PANGOLIN_MAX_SIGNED",
                "PANGOLIN_MAX_EVENT",
                "PANGOLIN_MAX_POSITION",
                "PANGOLIN_WARNINGS",
            ]
        )

    return (
        predictions,
        n_records,
        n_with_prediction,
    )


# =============================================================================
# INTEGRATION
# =============================================================================

def join_unique(
    series: pd.Series,
) -> str:

    values = []

    for value in (
        series
        .dropna()
        .astype(str)
    ):

        value = value.strip()

        if (
            value
            and value not in values
        ):
            values.append(value)

    return ";".join(values)


def make_integrated_summary(
    base: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    base = base.copy()

    if "VEP_ID" not in base.columns:
        raise RuntimeError(
            "Step09 variant summary does not contain VEP_ID."
        )

    base = (
        base
        .drop_duplicates(
            "VEP_ID",
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    if predictions.empty:

        strongest = pd.DataFrame(
            columns=[
                "VEP_ID",
                "PANGOLIN_GENE",
                "PANGOLIN_GAIN_SCORE",
                "PANGOLIN_GAIN_POS",
                "PANGOLIN_LOSS_SCORE",
                "PANGOLIN_LOSS_POS",
                "PANGOLIN_MAX_ABS",
                "PANGOLIN_MAX_SIGNED",
                "PANGOLIN_MAX_EVENT",
                "PANGOLIN_MAX_POSITION",
                "PANGOLIN_WARNINGS",
            ]
        )

        gene_summary = pd.DataFrame(
            columns=[
                "VEP_ID",
                "PANGOLIN_ALL_GENES",
            ]
        )

    else:

        predictions = predictions.copy()

        predictions[
            "PANGOLIN_MAX_ABS"
        ] = pd.to_numeric(
            predictions[
                "PANGOLIN_MAX_ABS"
            ],
            errors="coerce",
        )

        strongest = (
            predictions
            .sort_values(
                [
                    "VEP_ID",
                    "PANGOLIN_MAX_ABS",
                ],
                ascending=[
                    True,
                    False,
                ],
                na_position="last",
                kind="stable",
            )
            .drop_duplicates(
                "VEP_ID",
                keep="first",
            )
        )

        gene_summary = (
            predictions
            .groupby(
                "VEP_ID",
                dropna=False,
            )[
                "PANGOLIN_GENE"
            ]
            .agg(join_unique)
            .reset_index()
            .rename(
                columns={
                    "PANGOLIN_GENE":
                        "PANGOLIN_ALL_GENES"
                }
            )
        )

    pang_cols = [
        "VEP_ID",
        "PANGOLIN_GENE",
        "PANGOLIN_GAIN_SCORE",
        "PANGOLIN_GAIN_POS",
        "PANGOLIN_LOSS_SCORE",
        "PANGOLIN_LOSS_POS",
        "PANGOLIN_MAX_ABS",
        "PANGOLIN_MAX_SIGNED",
        "PANGOLIN_MAX_EVENT",
        "PANGOLIN_MAX_POSITION",
        "PANGOLIN_WARNINGS",
    ]

    integrated = base.merge(
        strongest[
            [
                c
                for c in pang_cols
                if c in strongest.columns
            ]
        ],
        on="VEP_ID",
        how="left",
        validate="one_to_one",
    )

    integrated = integrated.merge(
        gene_summary,
        on="VEP_ID",
        how="left",
        validate="one_to_one",
    )

    integrated[
        "PANGOLIN_SCORED"
    ] = integrated[
        "PANGOLIN_MAX_ABS"
    ].notna()

    # Relative ranking only. This is deliberately NOT described as a
    # validated Pangolin pathogenicity threshold.
    integrated[
        "PANGOLIN_PERCENTILE"
    ] = np.nan

    scored = integrated[
        "PANGOLIN_SCORED"
    ]

    if scored.any():

        integrated.loc[
            scored,
            "PANGOLIN_PERCENTILE",
        ] = (
            integrated.loc[
                scored,
                "PANGOLIN_MAX_ABS",
            ]
            .rank(
                method="average",
                pct=True,
            )
            .mul(100)
        )

    integrated[
        "PANGOLIN_TOP_10PCT"
    ] = (
        integrated[
            "PANGOLIN_PERCENTILE"
        ]
        >= 90
    )

    integrated[
        "PANGOLIN_TOP_5PCT"
    ] = (
        integrated[
            "PANGOLIN_PERCENTILE"
        ]
        >= 95
    )

    integrated[
        "PANGOLIN_TOP_1PCT"
    ] = (
        integrated[
            "PANGOLIN_PERCENTILE"
        ]
        >= 99
    )

    spliceai = pd.to_numeric(
        integrated.get(
            "SPLICEAI_MAX_DS",
            pd.Series(
                np.nan,
                index=integrated.index,
            ),
        ),
        errors="coerce",
    )

    integrated[
        "HAS_SPLICEAI_GE_0_20"
    ] = spliceai >= 0.20

    integrated[
        "HAS_SPLICEAI_GE_0_50"
    ] = spliceai >= 0.50

    if (
        "HAS_GTEX_SQTL"
        in integrated.columns
    ):

        sqtl = as_bool(
            integrated[
                "HAS_GTEX_SQTL"
            ]
        )

    elif (
        "HAS_GTEX_SQTL_BOOL"
        in integrated.columns
    ):

        sqtl = as_bool(
            integrated[
                "HAS_GTEX_SQTL_BOOL"
            ]
        )

    else:

        sqtl = pd.Series(
            False,
            index=integrated.index,
        )

    integrated[
        "HAS_GTEX_SQTL_BOOL"
    ] = sqtl

    if (
        "VEP_SPLICE_CATEGORY"
        in integrated.columns
    ):

        vep_splice = as_bool(
            integrated[
                "VEP_SPLICE_CATEGORY"
            ]
        )

    elif (
        "FUNCTIONAL_CATEGORY"
        in integrated.columns
    ):

        vep_splice = (
            integrated[
                "FUNCTIONAL_CATEGORY"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            ==
            "SPLICING"
        )

    else:

        vep_splice = pd.Series(
            False,
            index=integrated.index,
        )

    integrated[
        "HAS_VEP_SPLICE"
    ] = vep_splice

    evidence_count = (
        integrated[
            "HAS_SPLICEAI_GE_0_20"
        ].astype(int)
        +
        integrated[
            "PANGOLIN_TOP_5PCT"
        ].astype(int)
        +
        integrated[
            "HAS_GTEX_SQTL_BOOL"
        ].astype(int)
        +
        integrated[
            "HAS_VEP_SPLICE"
        ].astype(int)
    )

    integrated[
        "N_SPLICE_EVIDENCE_SOURCES"
    ] = evidence_count

    combinations = []

    for row in integrated.itertuples(
        index=False
    ):

        labels = []

        if bool(
            getattr(
                row,
                "HAS_SPLICEAI_GE_0_20",
            )
        ):
            labels.append(
                "SPLICEAI_GE_0.20"
            )

        if bool(
            getattr(
                row,
                "PANGOLIN_TOP_5PCT",
            )
        ):
            labels.append(
                "PANGOLIN_TOP5PCT"
            )

        if bool(
            getattr(
                row,
                "HAS_GTEX_SQTL_BOOL",
            )
        ):
            labels.append(
                "GTEX_SQTL"
            )

        if bool(
            getattr(
                row,
                "HAS_VEP_SPLICE",
            )
        ):
            labels.append(
                "VEP_SPLICE"
            )

        combinations.append(
            "+".join(labels)
            if labels
            else
            "NO_SELECTED_SPLICE_EVIDENCE"
        )

    integrated[
        "SPLICE_EVIDENCE_COMBINATION"
    ] = combinations

    integrated[
        "SPLICE_PRIORITY_INSPECTION"
    ] = (
        integrated[
            "HAS_SPLICEAI_GE_0_20"
        ]
        |
        integrated[
            "PANGOLIN_TOP_5PCT"
        ]
        |
        integrated[
            "HAS_GTEX_SQTL_BOOL"
        ]
        |
        integrated[
            "HAS_VEP_SPLICE"
        ]
    )

    return (
        integrated,
        strongest,
    )


def make_locus_summary(
    integrated: pd.DataFrame,
) -> pd.DataFrame:

    if (
        "LOCUS_ID"
        not in integrated.columns
    ):
        return pd.DataFrame()

    rows = []

    for locus, group in integrated.groupby(
        "LOCUS_ID",
        dropna=False,
    ):

        pip = pd.to_numeric(
            group.get(
                "PIP",
                pd.Series(
                    np.nan,
                    index=group.index,
                ),
            ),
            errors="coerce",
        )

        rows.append(
            {
                "LOCUS_ID": locus,
                "N_VARIANTS": len(group),
                "N_PANGOLIN_SCORED": int(
                    group[
                        "PANGOLIN_SCORED"
                    ].sum()
                ),
                "N_PANGOLIN_TOP_5PCT": int(
                    group[
                        "PANGOLIN_TOP_5PCT"
                    ].sum()
                ),
                "N_SPLICEAI_GE_0_20": int(
                    group[
                        "HAS_SPLICEAI_GE_0_20"
                    ].sum()
                ),
                "N_GTEX_SQTL": int(
                    group[
                        "HAS_GTEX_SQTL_BOOL"
                    ].sum()
                ),
                "N_VEP_SPLICE": int(
                    group[
                        "HAS_VEP_SPLICE"
                    ].sum()
                ),
                "N_MULTI_SOURCE_GE_2": int(
                    (
                        group[
                            "N_SPLICE_EVIDENCE_SOURCES"
                        ]
                        >= 2
                    ).sum()
                ),
                "MAX_PANGOLIN_ABS": pd.to_numeric(
                    group[
                        "PANGOLIN_MAX_ABS"
                    ],
                    errors="coerce",
                ).max(),
                "MAX_SPLICEAI_DS": pd.to_numeric(
                    group.get(
                        "SPLICEAI_MAX_DS",
                        pd.Series(
                            np.nan,
                            index=group.index,
                        ),
                    ),
                    errors="coerce",
                ).max(),
                "MAX_PIP": pip.max(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "N_MULTI_SOURCE_GE_2",
            "MAX_SPLICEAI_DS",
            "MAX_PANGOLIN_ABS",
            "MAX_PIP",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        kind="stable",
    )


# =============================================================================
# WORKER
# =============================================================================

def worker_mode() -> None:

    task_value = os.environ.get(
        "SLURM_ARRAY_TASK_ID"
    )

    manifest_value = os.environ.get(
        "GWAS_PANGOLIN_MANIFEST"
    )

    if not task_value:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is not defined."
        )

    if not manifest_value:
        raise RuntimeError(
            "GWAS_PANGOLIN_MANIFEST is not defined."
        )

    task_id = int(task_value)

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
        "PANGOLIN_TASK_ID",
        "STUDY_ACCESSION",
        "PHENOTYPE",
        "ANCESTRY_CODE",
        "ANCESTRY_LABEL",
        "STEP09_VARIANT_FILE",
        "STEP09_NORMALIZED_VCF",
        "OUTPUT_DIR",
        "PANGOLIN",
        "PANGOLIN_DB",
        "GRCH38_FASTA",
        "DISTANCE",
        "MASK",
    ]

    missing = [
        c
        for c in required
        if c not in manifest.columns
    ]

    if missing:
        raise RuntimeError(
            f"Pangolin manifest is missing required columns: {missing}"
        )

    ids = pd.to_numeric(
        manifest[
            "PANGOLIN_TASK_ID"
        ],
        errors="coerce",
    )

    selected = manifest.loc[
        ids == task_id
    ]

    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one manifest row for task {task_id}; "
            f"found {len(selected)}."
        )

    row = selected.iloc[0]

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
    ).strip().upper()

    ancestry_label = str(
        row[
            "ANCESTRY_LABEL"
        ]
    ).strip()

    step09_variant_file = Path(
        row[
            "STEP09_VARIANT_FILE"
        ]
    ).resolve()

    normalized_vcf = Path(
        row[
            "STEP09_NORMALIZED_VCF"
        ]
    ).resolve()

    output_dir = Path(
        row[
            "OUTPUT_DIR"
        ]
    ).resolve()

    pangolin = Path(
        row[
            "PANGOLIN"
        ]
    ).resolve()

    pangolin_db = Path(
        row[
            "PANGOLIN_DB"
        ]
    ).resolve()

    fasta = Path(
        row[
            "GRCH38_FASTA"
        ]
    ).resolve()

    distance = int(
        row[
            "DISTANCE"
        ]
    )

    mask = str(
        row[
            "MASK"
        ]
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

    force = (
        force
        or os.environ.get(
            "STEP10_FORCE",
            "0",
        )
        ==
        "1"
    )

    threads = int(
        os.environ.get(
            "SLURM_CPUS_PER_TASK",
            DEFAULT_CPUS,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pangolin_dir = (
        output_dir
        / "pangolin"
    )

    pangolin_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_csv = (
        pangolin_dir
        / f"{accession}_Pangolin_input.csv"
    )

    prefix = (
        pangolin_dir
        / (
            f"{accession}_Pangolin_"
            f"D{distance}"
        )
    )

    raw_output_csv = Path(
        str(prefix)
        + ".csv"
    )

    log_file = Path(
        str(prefix)
        + ".log"
    )

    gene_output = (
        output_dir
        / f"{accession}_Pangolin_all_gene_predictions.tsv.gz"
    )

    variant_output = (
        output_dir
        / f"{accession}_Pangolin_variant_summary.tsv"
    )

    integrated_output = (
        output_dir
        / f"{accession}_SpliceAI_Pangolin_integrated.tsv"
    )

    prioritized_output = (
        output_dir
        / f"{accession}_splicing_prioritized.tsv"
    )

    unscored_output = (
        output_dir
        / f"{accession}_Pangolin_unscored_variants.tsv"
    )

    locus_output = (
        output_dir
        / f"{accession}_Pangolin_locus_summary.tsv"
    )

    summary_output = (
        output_dir
        / "pangolin_summary.json"
    )

    failed_output = (
        output_dir
        / "PANGOLIN_FAILED.txt"
    )

    if failed_output.exists():
        print(
            "Previous Step10 failure marker found; "
            f"forcing rebuild: {failed_output}"
        )
        force = True

    if (
        not force
        and integrated_output.exists()
        and integrated_output.stat().st_size > 0
        and summary_output.exists()
        and str(
            safe_read_json(
                summary_output
            ).get(
                "STATUS",
                "",
            )
        ).upper()
        ==
        "COMPLETE"
        and not failed_output.exists()
    ):
        banner(
            "STEP 10 ALREADY COMPLETE"
        )
        print(f"Study : {accession}")
        print(f"Output: {output_dir}")
        return

    banner(
        "STEP 10 - PANGOLIN WORKER"
    )

    print(f"Task       : {task_id}")
    print(f"Study      : {accession}")
    print(f"Phenotype  : {phenotype}")
    print(
        f"Ancestry   : "
        f"{ancestry_label} ({ancestry_code})"
    )
    print(f"Step09     : {step09_variant_file}")
    print(f"VCF        : {normalized_vcf}")
    print(f"Pangolin   : {pangolin}")
    print(f"DB         : {pangolin_db}")
    print(f"FASTA      : {fasta}")
    print(f"Distance   : {distance}")
    print(f"Mask       : {mask}")
    print(f"Threads    : {threads}")

    try:

        for path, label in [
            (
                step09_variant_file,
                "Step09 variant summary",
            ),
            (
                normalized_vcf,
                "Step09 normalized VCF",
            ),
            (
                pangolin,
                "Pangolin executable",
            ),
            (
                pangolin_db,
                "Pangolin annotation DB",
            ),
            (
                fasta,
                "GRCh38 FASTA",
            ),
        ]:

            if (
                not path.exists()
                or path.stat().st_size == 0
            ):
                raise FileNotFoundError(
                    f"{label} missing/empty:\n{path}"
                )

        if not os.access(
            pangolin,
            os.X_OK,
        ):
            raise RuntimeError(
                f"Pangolin is not executable:\n{pangolin}"
            )

        base = pd.read_csv(
            step09_variant_file,
            sep="\t",
            low_memory=False,
        )

        if "VEP_ID" not in base.columns:
            raise RuntimeError(
                "Step09 variant table is missing VEP_ID."
            )

        banner(
            "PREPARING PANGOLIN CSV INPUT"
        )

        input_variants = (
            prepare_pangolin_csv(
                normalized_vcf,
                input_csv,
            )
        )

        print(
            f"Input variants : {len(input_variants):,}"
        )

        print(
            f"CSV            : {input_csv}"
        )

        banner(
            "RUNNING PANGOLIN"
        )

        # Limit common CPU libraries. Pangolin itself may use PyTorch.
        run_env = {
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
        }

        # run_pangolin currently calls run_command internally; apply CPU
        # settings for this process through the parent environment.
        old_values = {
            key: os.environ.get(key)
            for key in run_env
        }

        try:
            os.environ.update(run_env)

            raw_output_csv = run_pangolin(
                executable=pangolin,
                input_csv=input_csv,
                fasta=fasta,
                annotation_db=pangolin_db,
                prefix=prefix,
                log_file=log_file,
                distance=distance,
                mask=mask,
                force=force,
            )

        finally:
            for key, previous in old_values.items():
                if previous is None:
                    os.environ.pop(
                        key,
                        None,
                    )
                else:
                    os.environ[key] = previous

        banner(
            "PARSING PANGOLIN OUTPUT"
        )

        (
            predictions,
            n_records,
            n_with_prediction,
        ) = parse_pangolin_csv(
            raw_output_csv
        )

        print(
            f"Pangolin CSV records      : "
            f"{n_records:,}"
        )

        print(
            f"Records with prediction   : "
            f"{n_with_prediction:,}"
        )

        print(
            f"Gene-level predictions    : "
            f"{len(predictions):,}"
        )

        predictions.to_csv(
            gene_output,
            sep="\t",
            index=False,
            compression="gzip",
        )

        banner(
            "INTEGRATING SPLICE EVIDENCE"
        )

        (
            integrated,
            strongest,
        ) = make_integrated_summary(
            base,
            predictions,
        )

        # Pangolin-only view: input identity + Pangolin fields.
        pangolin_columns = [
            c
            for c in [
                "VEP_ID",
                "LOCUS_ID",
                "CHR",
                "REFERENCE_POS",
                "REFERENCE_REF",
                "REFERENCE_ALT",
                "REFERENCE_ID",
                "PIP",
                "PANGOLIN_GENE",
                "PANGOLIN_ALL_GENES",
                "PANGOLIN_GAIN_SCORE",
                "PANGOLIN_GAIN_POS",
                "PANGOLIN_LOSS_SCORE",
                "PANGOLIN_LOSS_POS",
                "PANGOLIN_MAX_ABS",
                "PANGOLIN_MAX_SIGNED",
                "PANGOLIN_MAX_EVENT",
                "PANGOLIN_MAX_POSITION",
                "PANGOLIN_WARNINGS",
                "PANGOLIN_SCORED",
                "PANGOLIN_PERCENTILE",
                "PANGOLIN_TOP_10PCT",
                "PANGOLIN_TOP_5PCT",
                "PANGOLIN_TOP_1PCT",
            ]
            if c in integrated.columns
        ]

        integrated[
            pangolin_columns
        ].to_csv(
            variant_output,
            sep="\t",
            index=False,
        )

        integrated.to_csv(
            integrated_output,
            sep="\t",
            index=False,
        )

        integrated.loc[
            integrated[
                "SPLICE_PRIORITY_INSPECTION"
            ]
        ].to_csv(
            prioritized_output,
            sep="\t",
            index=False,
        )

        unscored = integrated.loc[
            ~integrated[
                "PANGOLIN_SCORED"
            ]
        ].copy()

        unscored[
            "PANGOLIN_NOTE"
        ] = (
            "Pangolin did not return a prediction. "
            "This does not mean the Pangolin score is zero."
        )

        unscored.to_csv(
            unscored_output,
            sep="\t",
            index=False,
        )

        locus_summary = (
            make_locus_summary(
                integrated
            )
        )

        locus_summary.to_csv(
            locus_output,
            sep="\t",
            index=False,
        )

        n_input = len(integrated)

        n_scored = int(
            integrated[
                "PANGOLIN_SCORED"
            ].sum()
        )

        n_top10 = int(
            integrated[
                "PANGOLIN_TOP_10PCT"
            ].sum()
        )

        n_top5 = int(
            integrated[
                "PANGOLIN_TOP_5PCT"
            ].sum()
        )

        n_top1 = int(
            integrated[
                "PANGOLIN_TOP_1PCT"
            ].sum()
        )

        n_sai20 = int(
            integrated[
                "HAS_SPLICEAI_GE_0_20"
            ].sum()
        )

        n_sqtl = int(
            integrated[
                "HAS_GTEX_SQTL_BOOL"
            ].sum()
        )

        n_vep = int(
            integrated[
                "HAS_VEP_SPLICE"
            ].sum()
        )

        n_multi2 = int(
            (
                integrated[
                    "N_SPLICE_EVIDENCE_SOURCES"
                ]
                >= 2
            ).sum()
        )

        payload = {
            "STEP10_VERSION": STEP10_VERSION,
            "PANGOLIN_TASK_ID": task_id,
            "STUDY_ACCESSION": accession,
            "PHENOTYPE": phenotype,
            "ANCESTRY_CODE": ancestry_code,
            "ANCESTRY_LABEL": ancestry_label,
            "DISTANCE": distance,
            "MASK": mask,
            "N_INPUT_VARIANTS": int(n_input),
            "N_PANGOLIN_CSV_RECORDS": int(n_records),
            "N_RECORDS_WITH_PANGOLIN": int(n_with_prediction),
            "N_GENE_LEVEL_PREDICTIONS": int(len(predictions)),
            "N_PANGOLIN_SCORED": n_scored,
            "N_PANGOLIN_UNSCORED": int(n_input - n_scored),
            "PANGOLIN_COVERAGE_PERCENT": (
                100.0 * n_scored / n_input
                if n_input
                else 0.0
            ),
            "N_PANGOLIN_TOP_10PCT": n_top10,
            "N_PANGOLIN_TOP_5PCT": n_top5,
            "N_PANGOLIN_TOP_1PCT": n_top1,
            "N_SPLICEAI_GE_0_20": n_sai20,
            "N_GTEX_SQTL": n_sqtl,
            "N_VEP_SPLICE": n_vep,
            "N_MULTI_SOURCE_GE_2": n_multi2,
            "N_LOCI": (
                int(
                    integrated[
                        "LOCUS_ID"
                    ].nunique()
                )
                if "LOCUS_ID" in integrated.columns
                else None
            ),
            "PANGOLIN_EXECUTABLE": str(pangolin),
            "PANGOLIN_DB": str(pangolin_db),
            "GRCH38_FASTA": str(fasta),
            "STEP09_VARIANT_FILE": str(step09_variant_file),
            "STEP09_NORMALIZED_VCF": str(normalized_vcf),
            "PANGOLIN_INPUT_CSV": str(input_csv),
            "PANGOLIN_RAW_CSV": str(raw_output_csv),
            "ALL_GENE_PREDICTIONS": str(gene_output),
            "PANGOLIN_VARIANT_SUMMARY": str(variant_output),
            "INTEGRATED_VARIANT_SUMMARY": str(integrated_output),
            "PRIORITIZED_INSPECTION_FILE": str(prioritized_output),
            "UNSCORED_FILE": str(unscored_output),
            "LOCUS_SUMMARY_FILE": str(locus_output),
            "STATUS": "COMPLETE",
            "COMPLETED_UTC": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        write_json(
            summary_output,
            payload,
        )

        if failed_output.exists():
            failed_output.unlink()

        banner(
            "STEP 10 COMPLETE"
        )

        print(
            f"Study                    : {accession}"
        )

        print(
            f"Input variants           : {n_input:,}"
        )

        print(
            f"Pangolin-scored          : {n_scored:,}"
        )

        print(
            f"Pangolin-unscored        : "
            f"{n_input - n_scored:,}"
        )

        if n_input:
            print(
                f"Pangolin coverage         : "
                f"{100*n_scored/n_input:.2f}%"
            )

        print(
            f"Pangolin top 10%          : {n_top10:,}"
        )

        print(
            f"Pangolin top 5%           : {n_top5:,}"
        )

        print(
            f"Pangolin top 1%           : {n_top1:,}"
        )

        print(
            f"SpliceAI >=0.20           : {n_sai20:,}"
        )

        print(
            f"GTEx sQTL variants        : {n_sqtl:,}"
        )

        print(
            f"VEP splice variants       : {n_vep:,}"
        )

        print(
            f">=2 splice evidence types : {n_multi2:,}"
        )

        print()
        print(
            "MAIN downstream table:"
        )
        print(
            f"  {integrated_output}"
        )

        print()
        print(
            "IMPORTANT: Pangolin percentile flags are relative "
            "rankings within this study, not validated biological cutoffs."
        )

    except Exception as error:

        failure_text = (
            "STEP 10 PANGOLIN FAILED\n\n"
            f"Task ID: {task_id}\n"
            f"Accession: {accession}\n"
            f"Phenotype: {phenotype}\n"
            f"Ancestry: {ancestry_label} ({ancestry_code})\n"
            f"Input: {step09_variant_file}\n\n"
            f"Error:\n"
            f"{type(error).__name__}: {error}\n\n"
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
# DIRECT INDEX MODE
# =============================================================================

def direct_index_mode(
    args,
) -> None:

    validate_arguments(args)

    root = Path.cwd().resolve()

    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)

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
        / "10_pangolin"
        / phenotype_slug
        / ancestry_slug
        / "pangolin_manifest.tsv"
    ).resolve()

    if not manifest_file.exists():

        raise FileNotFoundError(
            "Step10 manifest does not exist:\n"
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

    if (
        "PANGOLIN_TASK_ID"
        not in manifest.columns
    ):
        raise RuntimeError(
            "Manifest is missing PANGOLIN_TASK_ID."
        )

    ids = pd.to_numeric(
        manifest[
            "PANGOLIN_TASK_ID"
        ],
        errors="coerce",
    )

    selected = manifest.loc[
        ids == args.index
    ]

    if len(selected) != 1:

        available = [
            int(x)
            for x in ids.dropna()
        ]

        raise RuntimeError(
            f"Index {args.index} is not present exactly once.\n"
            f"Available indices: {available}"
        )

    row = selected.iloc[0]

    banner(
        "STEP 10 - DIRECT SINGLE-INDEX MODE"
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
        "Use srun on HPC for substantial work."
    )

    old_task = os.environ.get(
        "SLURM_ARRAY_TASK_ID"
    )

    old_manifest = os.environ.get(
        "GWAS_PANGOLIN_MANIFEST"
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
            "GWAS_PANGOLIN_MANIFEST"
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
                "GWAS_PANGOLIN_MANIFEST",
                None,
            )
        else:
            os.environ[
                "GWAS_PANGOLIN_MANIFEST"
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
            "GWAS_PANGOLIN_MANIFEST"
        )
    ):
        worker_mode()
        return

    args = arguments()

    if args.index is not None:
        direct_index_mode(args)
    else:
        planner_mode(args)


if __name__ == "__main__":
    main()
 

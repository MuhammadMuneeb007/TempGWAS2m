#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic GWAS Pipeline
STEP 05 - Ancestry-matched LD clumping / independent locus definition

PLANNER MODE
------------
Run from the GWAS2m repository root, for example:

    envs/pipeline/bin/python Step05_Clump_GWAS_Loci.py \
        --phenotype migraine \
        --ancestry EUR

The planner:
  1. Resolves phenotype + ancestry deterministically.
  2. Finds completed Step 03 QC GWAS files only for that project.
  3. Verifies the matching 1000 Genomes ancestry reference (chr1-22).
  4. Creates a clumping manifest.
  5. Generates a phenotype/ancestry-specific SLURM array script.

Then submit the generated script, e.g.:

    sbatch Step05_Clump_GWAS_migraine_european.sh

WORKER MODE
-----------
The generated SLURM array sets SLURM_ARRAY_TASK_ID and
GWAS_CLUMP_MANIFEST. Each task processes exactly one QC'd GWAS.

For each GWAS, the worker:
  1. Streams the cleaned GRCh38 GWAS once and retains variants with P <= p2.
  2. Splits retained variants by chromosome.
  3. Maps GWAS variants to the exact variant IDs in the ancestry-matched
     1000G PGEN reference using position + allele matching when possible.
  4. Runs PLINK2 --clump separately on chr1-22.
  5. Combines chromosome-level clumps into one lead-variant table.
  6. Writes mapping/QC/clumping summaries for reproducibility.

OUTPUT LAYOUT
-------------
05_ld_clumping/
    <phenotype>/
        <ancestry>/
            clumping_manifest.tsv
            <GCST accession>/
                mapped_candidates.tsv.gz
                lead_variants.tsv
                clumping_summary.tsv
                clumping_summary.json
                chr01/
                    clump_input.tsv
                    plink.clumps
                    plink.log
                ...
                chr22/

NOTES
-----
- This step does NOT rebuild LD references. Shared ancestry-specific 1000G
  PGEN references must already exist under resources/1000G/<ANC>/.
- GWAS studies are never merged.
- Default clumping parameters are explicit and configurable:
      p1 = 5e-8
      p2 = 1e-2
      r2 = 0.1
      kb = 1000
- SLURM settings come from config/slurm.yaml (stage clumping); CLI flags override.
- Walltime is limited only by slurm.max_walltime when configured.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwas2m_config  # noqa: E402  (SLURM settings: config/slurm.yaml)

import pandas as pd


# =============================================================================
# DEFAULTS
# =============================================================================

# SLURM partition/time/memory/CPUs/throttle come from config/slurm.yaml
# (stage "clumping"); command-line flags still override them.
DEFAULT_CPUS = 4
DEFAULT_P1 = 5e-8
DEFAULT_P2 = 1e-2
DEFAULT_R2 = 0.1
DEFAULT_KB = 1000
DEFAULT_CHUNK_SIZE = 500_000

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
# DISPLAY / TEXT HELPERS
# =============================================================================


def banner(text: str) -> None:
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


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
        valid = "EUR, AFR, EAS, SAS, AMR"
        raise ValueError(
            f"Unsupported ancestry: {value!r}. "
            f"Use one of {valid} (or the corresponding full label)."
        )
    return ANCESTRY_ALIASES[key]


def parse_walltime(value: str) -> int:
    """Return walltime in seconds. Supports HH:MM:SS or D-HH:MM:SS."""
    text = str(value).strip()
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)

    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Walltime must be HH:MM:SS or D-HH:MM:SS")

    hours, minutes, seconds = map(int, parts)
    if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        raise ValueError("Invalid walltime minutes/seconds")

    total_hours = days * 24 + hours
    return total_hours * 3600 + minutes * 60 + seconds


def validate_walltime(value: str) -> str:
    """Central check (gwas2m_config): only a configured max_walltime applies."""
    return gwas2m_config.validate_walltime(value)


def find_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def clean_chr_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(
        r"^chr", "", regex=True, case=False
    )
    numeric = pd.to_numeric(text, errors="coerce")
    return numeric.astype("Int64")


def clean_allele(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    x = str(value).strip().upper()
    if not x or x in {"NA", "NAN", "<NA>", "NONE"}:
        return None
    return x


# =============================================================================
# CLI
# =============================================================================


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and run ancestry-matched LD clumping for QC'd GWAS files."
    )

    parser.add_argument("--phenotype", required=True)
    parser.add_argument(
        "--ancestry",
        required=True,
        help="EUR, AFR, EAS, SAS, AMR or corresponding full ancestry label.",
    )
    parser.add_argument("--partition", default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--memory", default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Run exactly one existing clumping-manifest task directly instead "
            "of generating/submitting the SLURM array. Example: --index 1"
        ),
    )

    parser.add_argument("--p1", type=float, default=DEFAULT_P1)
    parser.add_argument("--p2", type=float, default=DEFAULT_P2)
    parser.add_argument("--r2", type=float, default=DEFAULT_R2)
    parser.add_argument("--kb", type=int, default=DEFAULT_KB)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow clumping to proceed when some expected Step 03 QC files are still missing.",
    )

    args = parser.parse_args(argv)
    gwas2m_config.apply_stage_defaults(args, "clumping")
    return args


def validate_arguments(args) -> None:
    if not args.phenotype.strip():
        raise ValueError("--phenotype cannot be empty.")

    validate_walltime(args.time)

    if args.cpus < 1:
        raise ValueError("--cpus must be >= 1")
    if args.max_parallel < 0:
        raise ValueError("--max-parallel must be >= 0 (0 = no throttle)")
    if args.index is not None and args.index < 1:
        raise ValueError("--index must be >= 1")
    if args.chunk_size < 10_000:
        raise ValueError("--chunk-size must be >= 10000")

    if not (0 < args.p1 <= 1):
        raise ValueError("--p1 must be in (0, 1]")
    if not (0 < args.p2 <= 1):
        raise ValueError("--p2 must be in (0, 1]")
    if args.p1 > args.p2:
        raise ValueError("--p1 must be <= --p2")
    if not (0 < args.r2 <= 1):
        raise ValueError("--r2 must be in (0, 1]")
    if args.kb < 1:
        raise ValueError("--kb must be >= 1")


# =============================================================================
# REFERENCE VALIDATION
# =============================================================================


def reference_prefix(root: Path, ancestry_code: str, chrom: int) -> Path:
    return (
        root
        / "resources"
        / "1000G"
        / ancestry_code
        / f"chr{chrom}_{ancestry_code}_GRCh38"
    )


def validate_reference(root: Path, ancestry_code: str) -> None:
    missing = []
    for chrom in range(1, 23):
        prefix = reference_prefix(root, ancestry_code, chrom)
        for suffix in (".pgen", ".pvar", ".psam"):
            path = Path(str(prefix) + suffix)
            if not path.exists() or path.stat().st_size <= 0:
                missing.append(str(path))

    if missing:
        preview = "\n".join(f"  {x}" for x in missing[:20])
        if len(missing) > 20:
            preview += f"\n  ... and {len(missing) - 20} more"
        raise RuntimeError(
            f"1000G {ancestry_code} reference is incomplete. Missing:\n{preview}"
        )


# =============================================================================
# ONE CLUMPING MANIFEST ROW (shared by planner_mode and Step01_10_Run.py)
# =============================================================================


def build_clump_row(
    args,
    task_id: int,
    accession: str,
    qc_file: Path,
    output_root: Path,
    phenotype: str,
    ancestry_code: str,
    ancestry_label: str,
) -> tuple[dict | None, dict | None]:
    """Return (manifest_row, None) or (None, exclusion_record)."""
    summary_file = qc_file.parent / f"{accession}_QC_summary.tsv"

    clumping_ready = True
    qc_status = "UNKNOWN"
    reason = ""

    if summary_file.exists() and summary_file.stat().st_size > 0:
        try:
            summary_df = pd.read_csv(summary_file, sep="\t", dtype=str)
            if not summary_df.empty:
                summary_row = summary_df.iloc[0]
                qc_status = str(summary_row.get("QC_STATUS", "UNKNOWN"))
                ready_text = str(summary_row.get("CLUMPING_READY", "True")).strip().lower()
                clumping_ready = ready_text in {"true", "1", "yes", "y"}
        except Exception as exc:
            reason = f"Could not read QC summary: {exc}"

    if not clumping_ready:
        return None, {
            "STUDY_ACCESSION": accession,
            "QC_FILE": str(qc_file.resolve()),
            "QC_STATUS": qc_status,
            "REASON": reason or "Step 03 marked CLUMPING_READY=False",
        }

    study_output = output_root / accession
    return {
        "CLUMP_TASK_ID": task_id,
        "STUDY_ACCESSION": accession,
        "QC_FILE": str(qc_file.resolve()),
        "QC_SUMMARY_FILE": str(summary_file.resolve()) if summary_file.exists() else "",
        "QC_STATUS": qc_status,
        "OUTPUT_DIR": str(study_output.resolve()),
        "PHENOTYPE": phenotype,
        "ANCESTRY": ancestry_label,
        "ANCESTRY_CODE": ancestry_code,
        "ANCESTRY_LABEL": ancestry_label,
        "P1": args.p1,
        "P2": args.p2,
        "R2": args.r2,
        "KB": args.kb,
        "CHUNK_SIZE": args.chunk_size,
    }, None


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

    pipeline_python = root / "envs" / "pipeline" / "bin" / "python"
    plink2 = root / "envs" / "pipeline" / "bin" / "plink2"
    script_path = Path(__file__).resolve()

    if not pipeline_python.exists():
        raise FileNotFoundError(f"Pipeline Python not found: {pipeline_python}")
    if not plink2.exists():
        raise FileNotFoundError(f"PLINK2 not found: {plink2}")

    validate_reference(root, ancestry_code)

    qc_root = (
        root
        / "02_summary_stats"
        / phenotype_slug
        / ancestry_slug
        / "qc"
    )

    if not qc_root.exists():
        raise FileNotFoundError(
            "Step 03 QC directory not found:\n"
            f"  {qc_root}\n\n"
            "Run Step 03 for this phenotype/ancestry first."
        )

    # Resolve expected studies from the Step 03 QC manifest when available.
    qc_manifest_file = (
        root
        / "01_gwas_catalog"
        / phenotype_slug
        / ancestry_slug
        / "GWAS_QC_manifest.tsv"
    )

    expected_accessions = []
    if qc_manifest_file.exists() and qc_manifest_file.stat().st_size > 0:
        qc_manifest = pd.read_csv(
            qc_manifest_file, sep="\t", dtype=str, low_memory=False
        )
        if "STUDY_ACCESSION" in qc_manifest.columns:
            expected_accessions = [
                str(x).strip()
                for x in qc_manifest["STUDY_ACCESSION"].dropna().tolist()
                if str(x).strip()
            ]

    qc_files = sorted(qc_root.glob("*/*_GRCh38_QC.tsv.gz"))
    by_accession = {path.parent.name: path for path in qc_files}

    if not qc_files:
        raise RuntimeError(
            "No completed Step 03 QC files were found under:\n"
            f"  {qc_root}"
        )

    missing_qc_accessions = [
        accession for accession in expected_accessions if accession not in by_accession
    ]
    if missing_qc_accessions and not args.allow_partial:
        preview = ", ".join(missing_qc_accessions[:20])
        if len(missing_qc_accessions) > 20:
            preview += f", ... (+{len(missing_qc_accessions)-20} more)"
        raise RuntimeError(
            "Step 03 QC is not complete for all expected studies.\n"
            f"Missing QC outputs for {len(missing_qc_accessions)} accession(s): {preview}\n"
            "Wait for Step 03 to finish, or use --allow-partial intentionally."
        )

    output_root = (
        root
        / "05_ld_clumping"
        / phenotype_slug
        / ancestry_slug
    )
    log_root = (
        root
        / "logs"
        / "step05_clumping"
        / phenotype_slug
        / ancestry_slug
    )

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    rows = []
    exclusions = []

    for qc_file in qc_files:
        accession = qc_file.parent.name

        row, exclusion = build_clump_row(
            args,
            task_id=len(rows) + 1,
            accession=accession,
            qc_file=qc_file,
            output_root=output_root,
            phenotype=phenotype,
            ancestry_code=ancestry_code,
            ancestry_label=ancestry_label,
        )

        if exclusion is not None:
            exclusions.append(exclusion)
            continue

        rows.append(row)

    if exclusions:
        pd.DataFrame(exclusions).to_csv(
            output_root / "clumping_excluded_studies.tsv", sep="\t", index=False
        )

    if not rows:
        raise RuntimeError(
            "No Step 03 studies are marked ready for clumping. "
            "Inspect the QC summaries and clumping_excluded_studies.tsv."
        )

    manifest = pd.DataFrame(rows)
    manifest_file = (output_root / "clumping_manifest.tsv").resolve()
    manifest.to_csv(manifest_file, sep="\t", index=False)

    bash_file = (
        root
        / f"Step05_Clump_GWAS_{phenotype_slug}_{ancestry_slug}.sh"
    )

    job_name = f"CLUMP_{phenotype_slug}_{ancestry_code.lower()}"[:100]
    array_spec = gwas2m_config.array_spec(len(manifest), args.max_parallel)

    sbatch_header = gwas2m_config.sbatch_header_from_args(
        args,
        job_name=job_name,
        output=f"{log_root}/clump.%A_%a.out",
        error=f"{log_root}/clump.%A_%a.err",
        array=array_spec,
    )


    bash_text = f'''#!/bin/bash
{sbatch_header}

set -euo pipefail

cd {shlex.quote(str(root))}

export GWAS_CLUMP_MANIFEST={shlex.quote(str(manifest_file))}

printf '%s\n' "=============================================================="
printf '%s\n' "STEP 05 - LD CLUMPING"
printf '%s\n' "=============================================================="
printf 'Job ID        : %s\n' "$SLURM_JOB_ID"
printf 'Array task ID : %s\n' "$SLURM_ARRAY_TASK_ID"
printf 'Hostname      : %s\n' "$(hostname)"
printf 'Python        : %s\n' {shlex.quote(str(pipeline_python))}
printf 'Phenotype     : %s\n' {shlex.quote(phenotype)}
printf 'Ancestry      : %s (%s)\n' {shlex.quote(ancestry_label)} {shlex.quote(ancestry_code)}
printf '%s\n' "=============================================================="

{shlex.quote(str(pipeline_python))} {shlex.quote(str(script_path))}

printf '%s\n' "=============================================================="
printf 'STEP 05 TASK COMPLETE: %s\n' "$SLURM_ARRAY_TASK_ID"
printf '%s\n' "=============================================================="
'''

    bash_file.write_text(bash_text, encoding="utf-8")
    bash_file.chmod(0o755)

    banner("STEP 05 - LD CLUMPING PLANNER")
    print(f"Phenotype          : {phenotype}")
    print(f"Ancestry           : {ancestry_label}")
    print(f"Ancestry code      : {ancestry_code}")
    print(f"QC GWAS files      : {len(qc_files)}")
    print(f"Clumping-ready     : {len(manifest)}")
    print(f"Excluded           : {len(exclusions)}")
    print(f"Missing QC outputs : {len(missing_qc_accessions)}")
    print(f"Reference          : resources/1000G/{ancestry_code}/")
    print(f"Partition          : {args.partition or 'scheduler default'}")
    print(f"Walltime           : {args.time}")
    print(f"Memory/task        : {args.memory}")
    print(f"CPUs/task          : {args.cpus}")
    print(f"Array              : {array_spec}")
    print()
    print("Clumping parameters:")
    print(f"  p1               : {args.p1}")
    print(f"  p2               : {args.p2}")
    print(f"  r2               : {args.r2}")
    print(f"  kb               : {args.kb}")
    print()
    print(f"Manifest:\n  {manifest_file}")
    print(f"Output root:\n  {output_root}")
    print(f"Generated SLURM script:\n  {bash_file}")
    print()
    print("NEXT COMMAND:")
    print()
    print(f"  sbatch {bash_file.name}")
    print()


# =============================================================================
# GWAS STREAMING
# =============================================================================


def inspect_gwas_columns(path: Path):
    header = pd.read_csv(path, sep="\t", compression="infer", nrows=0)
    columns = list(header.columns)

    mapping = {
        "CHR": find_column(columns, ["CHR"]),
        "POS": find_column(columns, ["POS"]),
        "P": find_column(columns, ["P"]),
        "EA": find_column(columns, ["EA"]),
        "NEA": find_column(columns, ["NEA"]),
        "SNPID": find_column(columns, ["SNPID", "rsID", "RSID"]),
    }

    required = ["CHR", "POS", "P"]
    missing = [x for x in required if mapping[x] is None]
    if missing:
        raise RuntimeError(
            f"GWAS file is missing required columns {missing}: {path}"
        )

    return mapping


def stream_candidate_variants(
    qc_file: Path,
    p2: float,
    chunk_size: int,
):
    mapping = inspect_gwas_columns(qc_file)

    usecols = [mapping["CHR"], mapping["POS"], mapping["P"]]
    for optional in ("EA", "NEA", "SNPID"):
        if mapping[optional] is not None:
            usecols.append(mapping[optional])
    usecols = list(dict.fromkeys(usecols))

    per_chrom = defaultdict(list)
    n_total = 0
    n_valid_p = 0
    n_p2 = 0

    reader = pd.read_csv(
        qc_file,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        n_total += len(chunk)

        renamed = pd.DataFrame(index=chunk.index)
        renamed["CHR"] = clean_chr_series(chunk[mapping["CHR"]])
        renamed["POS"] = pd.to_numeric(chunk[mapping["POS"]], errors="coerce").astype("Int64")
        renamed["P"] = pd.to_numeric(chunk[mapping["P"]], errors="coerce")

        if mapping["EA"] is not None:
            renamed["EA"] = chunk[mapping["EA"]].map(clean_allele)
        else:
            renamed["EA"] = None

        if mapping["NEA"] is not None:
            renamed["NEA"] = chunk[mapping["NEA"]].map(clean_allele)
        else:
            renamed["NEA"] = None

        if mapping["SNPID"] is not None:
            renamed["SOURCE_ID"] = chunk[mapping["SNPID"]].astype(str)
        else:
            renamed["SOURCE_ID"] = ""

        valid = (
            renamed["CHR"].between(1, 22, inclusive="both")
            & renamed["POS"].notna()
            & renamed["P"].notna()
            & renamed["P"].between(0, 1, inclusive="both")
        )
        n_valid_p += int(valid.sum())

        renamed = renamed.loc[valid]
        selected = renamed.loc[renamed["P"] <= p2].copy()
        n_p2 += len(selected)

        if selected.empty:
            continue

        for chrom, frame in selected.groupby("CHR", sort=False):
            per_chrom[int(chrom)].append(frame)

    combined = {}
    for chrom, frames in per_chrom.items():
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("P", kind="stable")
        # Exact duplicate GWAS records are unnecessary for clumping.
        subset = ["POS", "P", "EA", "NEA"]
        df = df.drop_duplicates(subset=subset, keep="first")
        combined[chrom] = df

    stats = {
        "N_GWAS_ROWS": n_total,
        "N_VALID_ROWS": n_valid_p,
        "N_P_LE_P2": n_p2,
    }
    return combined, stats


# =============================================================================
# PVAR MAPPING
# =============================================================================


def load_reference_matches(pvar_path: Path, target_positions: set[int]):
    """
    Stream a PLINK2 .pvar and retain only target positions.

    Returns mapping:
      position -> list of dicts with ID, REF, ALT
    """
    result = defaultdict(list)

    with open(pvar_path, "rt", encoding="utf-8", errors="replace") as handle:
        header = None
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.rstrip("\n").lstrip("#").split("\t")
                break

        if header is None:
            raise RuntimeError(f"Could not find PVAR header in {pvar_path}")

        index = {name: i for i, name in enumerate(header)}
        required = ["POS", "ID", "REF", "ALT"]
        missing = [x for x in required if x not in index]
        if missing:
            raise RuntimeError(f"PVAR missing columns {missing}: {pvar_path}")

        for line in handle:
            fields = line.rstrip("\n").split("\t")
            try:
                pos = int(fields[index["POS"]])
            except (ValueError, IndexError):
                continue

            if pos not in target_positions:
                continue

            try:
                variant_id = fields[index["ID"]]
                ref = fields[index["REF"]].upper()
                alt = fields[index["ALT"]].upper()
            except IndexError:
                continue

            result[pos].append(
                {
                    "ID": variant_id,
                    "REF": ref,
                    "ALT": alt,
                }
            )

    return result


def map_gwas_to_reference(gwas: pd.DataFrame, pvar_path: Path) -> tuple[pd.DataFrame, dict]:
    if gwas.empty:
        return pd.DataFrame(columns=["ID", "P", "CHR", "POS", "EA", "NEA", "SOURCE_ID"]), {
            "N_INPUT": 0,
            "N_MAPPED": 0,
            "N_UNMAPPED": 0,
            "N_AMBIGUOUS": 0,
        }

    positions = set(int(x) for x in gwas["POS"].dropna().astype(int).unique())
    reference = load_reference_matches(pvar_path, positions)

    mapped_rows = []
    n_unmapped = 0
    n_ambiguous = 0

    for row in gwas.itertuples(index=False):
        pos = int(row.POS)
        options = reference.get(pos, [])

        if not options:
            n_unmapped += 1
            continue

        ea = clean_allele(row.EA)
        nea = clean_allele(row.NEA)

        chosen = None

        # Best case: allele-aware matching independent of effect direction.
        if ea and nea:
            allele_matches = [
                opt
                for opt in options
                if {opt["REF"], opt["ALT"]} == {ea, nea}
            ]
            if len(allele_matches) == 1:
                chosen = allele_matches[0]
            elif len(allele_matches) > 1:
                n_ambiguous += 1
                continue

        # Fallback: position is unique in the biallelic reference.
        if chosen is None:
            if len(options) == 1:
                chosen = options[0]
            else:
                n_ambiguous += 1
                continue

        mapped_rows.append(
            {
                "ID": chosen["ID"],
                "P": float(row.P),
                "CHR": int(row.CHR),
                "POS": pos,
                "REF": chosen["REF"],
                "ALT": chosen["ALT"],
                "EA": ea,
                "NEA": nea,
                "SOURCE_ID": getattr(row, "SOURCE_ID", ""),
            }
        )

    mapped = pd.DataFrame(mapped_rows)
    if not mapped.empty:
        mapped = (
            mapped.sort_values("P", kind="stable")
            .drop_duplicates(subset=["ID"], keep="first")
            .reset_index(drop=True)
        )

    stats = {
        "N_INPUT": int(len(gwas)),
        "N_MAPPED": int(len(mapped)),
        "N_UNMAPPED": int(n_unmapped),
        "N_AMBIGUOUS": int(n_ambiguous),
    }
    return mapped, stats


# =============================================================================
# PLINK CLUMPING
# =============================================================================


def run_plink_clump(
    plink2: Path,
    pfile_prefix: Path,
    clump_input: Path,
    out_prefix: Path,
    p1: float,
    p2: float,
    r2: float,
    kb: int,
    threads: int,
):
    cmd = [
        str(plink2),
        "--pfile", str(pfile_prefix),
        "--clump", str(clump_input),
        "--clump-id-field", "ID",
        "--clump-p-field", "P",
        "--clump-p1", str(p1),
        "--clump-p2", str(p2),
        "--clump-r2", str(r2),
        "--clump-kb", str(kb),
        "--threads", str(threads),
        "--out", str(out_prefix),
    ]

    print("$ " + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


# =============================================================================
# WORKER
# =============================================================================


def worker_mode() -> None:
    task_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    manifest_value = os.environ.get("GWAS_CLUMP_MANIFEST")

    if not task_value:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not defined.")
    if not manifest_value:
        raise RuntimeError("GWAS_CLUMP_MANIFEST is not defined.")

    task_id = int(task_value)
    manifest_file = Path(manifest_value).resolve()
    root = Path.cwd().resolve()
    plink2 = root / "envs" / "pipeline" / "bin" / "plink2"

    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str, low_memory=False)
    required = [
        "CLUMP_TASK_ID",
        "STUDY_ACCESSION",
        "QC_FILE",
        "OUTPUT_DIR",
        "PHENOTYPE",
        "ANCESTRY_CODE",
        "ANCESTRY_LABEL",
        "P1",
        "P2",
        "R2",
        "KB",
        "CHUNK_SIZE",
    ]
    missing = [x for x in required if x not in manifest.columns]
    if missing:
        raise RuntimeError(f"Clumping manifest missing columns: {missing}")

    task_numbers = pd.to_numeric(manifest["CLUMP_TASK_ID"], errors="coerce")
    selected = manifest.loc[task_numbers == task_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one manifest row for task {task_id}; found {len(selected)}"
        )

    row = selected.iloc[0]
    accession = str(row["STUDY_ACCESSION"]).strip()
    qc_file = Path(row["QC_FILE"]).resolve()
    output_dir = Path(row["OUTPUT_DIR"]).resolve()
    phenotype = str(row["PHENOTYPE"]).strip()
    ancestry_code = str(row["ANCESTRY_CODE"]).strip().upper()
    ancestry_label = str(row["ANCESTRY_LABEL"]).strip()
    p1 = float(row["P1"])
    p2 = float(row["P2"])
    r2 = float(row["R2"])
    kb = int(float(row["KB"]))
    chunk_size = int(float(row["CHUNK_SIZE"]))
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_CPUS))
    # Optional column (Step01_10_Run.py sets it when Step 03 was re-run);
    # planner manifests do not contain it, so default behaviour is unchanged.
    force = str(row.get("FORCE", "False")).strip().lower() in {"true", "1", "yes", "y"}

    output_dir.mkdir(parents=True, exist_ok=True)

    lead_output = output_dir / "lead_variants.tsv"
    mapped_output = output_dir / "mapped_candidates.tsv.gz"
    summary_tsv = output_dir / "clumping_summary.tsv"
    summary_json = output_dir / "clumping_summary.json"
    failed_output = output_dir / "CLUMPING_FAILED.txt"

    # Resume only when all key deliverables are present.
    if (
        not force
        and lead_output.exists()
        and summary_tsv.exists()
        and summary_json.exists()
        and mapped_output.exists()
    ):
        banner("CLUMPING ALREADY COMPLETE")
        print(f"Study: {accession}")
        print(f"Output: {output_dir}")
        return

    if not qc_file.exists() or qc_file.stat().st_size <= 0:
        raise FileNotFoundError(f"QC file not found or empty: {qc_file}")

    banner("STEP 05 - LD CLUMPING WORKER")
    print(f"Task          : {task_id}")
    print(f"Study         : {accession}")
    print(f"Phenotype     : {phenotype}")
    print(f"Ancestry      : {ancestry_label} ({ancestry_code})")
    print(f"QC file       : {qc_file}")
    print(f"Output        : {output_dir}")
    print(f"p1 / p2       : {p1} / {p2}")
    print(f"r2 / kb       : {r2} / {kb}")
    print(f"Threads       : {threads}")

    try:
        banner("STREAMING QC GWAS")
        per_chrom, stream_stats = stream_candidate_variants(
            qc_file=qc_file,
            p2=p2,
            chunk_size=chunk_size,
        )

        print(f"GWAS rows      : {stream_stats['N_GWAS_ROWS']:,}")
        print(f"Valid rows     : {stream_stats['N_VALID_ROWS']:,}")
        print(f"P <= p2        : {stream_stats['N_P_LE_P2']:,}")

        all_mapped = []
        all_leads = []
        chromosome_summaries = []

        for chrom in range(1, 23):
            chrom_dir = output_dir / f"chr{chrom:02d}"
            chrom_dir.mkdir(parents=True, exist_ok=True)

            gwas_chr = per_chrom.get(
                chrom,
                pd.DataFrame(columns=["CHR", "POS", "P", "EA", "NEA", "SOURCE_ID"]),
            )

            prefix = reference_prefix(root, ancestry_code, chrom)
            pvar = Path(str(prefix) + ".pvar")

            mapped, map_stats = map_gwas_to_reference(gwas_chr, pvar)

            if not mapped.empty:
                all_mapped.append(mapped)

            clump_input = chrom_dir / "clump_input.tsv"
            clump_report_prefix = chrom_dir / "plink"

            # Remove stale chromosome-level reports from an interrupted prior run.
            for suffix in (
                ".clumps",
                ".clumps.zst",
                ".clumps.missing_id",
                ".clumps.missing_id.zst",
                ".clumps.missing_allele",
                ".clumps.missing_allele.zst",
            ):
                stale = Path(str(clump_report_prefix) + suffix)
                if stale.exists():
                    stale.unlink()

            # PLINK only needs ID and P for this biallelic reference.
            mapped[["ID", "P"]].to_csv(
                clump_input,
                sep="\t",
                index=False,
            )

            n_p1 = int((mapped["P"] <= p1).sum()) if not mapped.empty else 0
            clumps_file = Path(str(clump_report_prefix) + ".clumps")

            if not mapped.empty and n_p1 > 0:
                banner(f"CHR {chrom} CLUMPING")
                run_plink_clump(
                    plink2=plink2,
                    pfile_prefix=prefix,
                    clump_input=clump_input,
                    out_prefix=clump_report_prefix,
                    p1=p1,
                    p2=p2,
                    r2=r2,
                    kb=kb,
                    threads=threads,
                )

            n_leads = 0
            if clumps_file.exists() and clumps_file.stat().st_size > 0:
                try:
                    clumps = pd.read_csv(
                        clumps_file,
                        sep=r"\s+",
                        engine="python",
                    )
                except pd.errors.EmptyDataError:
                    clumps = pd.DataFrame()

                if not clumps.empty:
                    clumps.insert(0, "STUDY_ACCESSION", accession)
                    clumps.insert(1, "PHENOTYPE", phenotype)
                    clumps.insert(2, "ANCESTRY_CODE", ancestry_code)
                    clumps.insert(3, "ANCESTRY_LABEL", ancestry_label)
                    clumps.insert(4, "SOURCE_CHROMOSOME", chrom)
                    all_leads.append(clumps)
                    n_leads = len(clumps)

            chromosome_summaries.append(
                {
                    "CHR": chrom,
                    "N_GWAS_P_LE_P2": map_stats["N_INPUT"],
                    "N_MAPPED": map_stats["N_MAPPED"],
                    "N_UNMAPPED": map_stats["N_UNMAPPED"],
                    "N_AMBIGUOUS": map_stats["N_AMBIGUOUS"],
                    "N_P_LE_P1_MAPPED": n_p1,
                    "N_LEAD_VARIANTS": n_leads,
                }
            )

        if all_mapped:
            mapped_all = pd.concat(all_mapped, ignore_index=True)
            mapped_all = mapped_all.sort_values(["CHR", "POS", "P"], kind="stable")
        else:
            mapped_all = pd.DataFrame(
                columns=["ID", "P", "CHR", "POS", "REF", "ALT", "EA", "NEA", "SOURCE_ID"]
            )

        mapped_all.to_csv(
            mapped_output,
            sep="\t",
            index=False,
            compression="gzip",
        )

        if all_leads:
            lead_all = pd.concat(all_leads, ignore_index=True)
            p_column = "P" if "P" in lead_all.columns else None
            if p_column:
                lead_all = lead_all.sort_values(p_column, kind="stable")
        else:
            lead_all = pd.DataFrame(
                columns=[
                    "STUDY_ACCESSION",
                    "PHENOTYPE",
                    "ANCESTRY_CODE",
                    "ANCESTRY_LABEL",
                    "SOURCE_CHROMOSOME",
                    "CHROM",
                    "POS",
                    "ID",
                    "P",
                ]
            )

        lead_all.to_csv(lead_output, sep="\t", index=False)

        chromosome_summary_file = output_dir / "chromosome_clumping_summary.tsv"
        chromosome_df = pd.DataFrame(chromosome_summaries)
        chromosome_df.to_csv(chromosome_summary_file, sep="\t", index=False)

        total_mapped = int(chromosome_df["N_MAPPED"].sum())
        total_input_for_mapping = int(chromosome_df["N_GWAS_P_LE_P2"].sum())
        mapping_rate = (
            total_mapped / total_input_for_mapping
            if total_input_for_mapping > 0
            else None
        )

        summary = {
            "CLUMP_TASK_ID": task_id,
            "STUDY_ACCESSION": accession,
            "PHENOTYPE": phenotype,
            "ANCESTRY": ancestry_label,
            "ANCESTRY_CODE": ancestry_code,
            "ANCESTRY_LABEL": ancestry_label,
            "BUILD": "GRCh38",
            "QC_FILE": str(qc_file),
            "OUTPUT_DIR": str(output_dir),
            "REFERENCE_ROOT": str(root / "resources" / "1000G" / ancestry_code),
            "P1": p1,
            "P2": p2,
            "R2": r2,
            "KB": kb,
            "N_GWAS_ROWS": stream_stats["N_GWAS_ROWS"],
            "N_VALID_ROWS": stream_stats["N_VALID_ROWS"],
            "N_P_LE_P2": stream_stats["N_P_LE_P2"],
            "N_REFERENCE_MAPPED": total_mapped,
            "MAPPING_RATE": mapping_rate,
            "N_INDEX_CANDIDATES_P_LE_P1": int(chromosome_df["N_P_LE_P1_MAPPED"].sum()),
            "N_LEAD_VARIANTS": int(len(lead_all)),
            "LEAD_VARIANTS_FILE": str(lead_output),
            "MAPPED_CANDIDATES_FILE": str(mapped_output),
            "CHROMOSOME_SUMMARY_FILE": str(chromosome_summary_file),
            "THREADS": threads,
            "COMPLETED_UTC": datetime.now(timezone.utc).isoformat(),
        }

        pd.DataFrame([summary]).to_csv(summary_tsv, sep="\t", index=False)
        summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        if failed_output.exists():
            failed_output.unlink()

        banner("LD CLUMPING COMPLETE")
        print(f"Study                  : {accession}")
        print(f"P <= p2 candidates     : {stream_stats['N_P_LE_P2']:,}")
        print(f"Reference mapped       : {total_mapped:,}")
        print(f"Mapping rate           : {mapping_rate}")
        print(f"Index candidates p<=p1 : {summary['N_INDEX_CANDIDATES_P_LE_P1']:,}")
        print(f"Lead variants          : {len(lead_all):,}")
        print()
        print(f"Lead variants:\n  {lead_output}")
        print(f"Mapped candidates:\n  {mapped_output}")
        print(f"Summary:\n  {summary_tsv}")

    except Exception as error:
        failure_text = (
            "STEP 05 LD CLUMPING FAILED\n\n"
            f"Task ID: {task_id}\n"
            f"Accession: {accession}\n"
            f"QC file: {qc_file}\n\n"
            f"Error:\n{type(error).__name__}: {error}\n\n"
            f"{traceback.format_exc()}"
        )
        failed_output.write_text(failure_text, encoding="utf-8")
        print(failure_text, file=sys.stderr)
        raise


# =============================================================================
# DIRECT SINGLE-INDEX MODE
# =============================================================================


def direct_index_mode(args) -> None:
    """Run one existing Step 05 manifest row using the normal worker logic.

    This is intentionally the same scientific worker used by the SLURM array.
    The only difference is that the requested task ID and manifest path are
    supplied locally through the same environment variables the generated
    SLURM script would normally set.
    """
    validate_arguments(args)

    root = Path.cwd().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)

    manifest_file = (
        root
        / "05_ld_clumping"
        / phenotype_slug
        / ancestry_slug
        / "clumping_manifest.tsv"
    ).resolve()

    if not manifest_file.exists() or manifest_file.stat().st_size <= 0:
        raise FileNotFoundError(
            "Clumping manifest not found or empty:\n"
            f"  {manifest_file}\n\n"
            "Run Step 05 once without --index first to generate the manifest."
        )

    manifest = pd.read_csv(
        manifest_file,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    if "CLUMP_TASK_ID" not in manifest.columns:
        raise RuntimeError(
            f"Clumping manifest is missing CLUMP_TASK_ID: {manifest_file}"
        )

    task_numbers = pd.to_numeric(
        manifest["CLUMP_TASK_ID"],
        errors="coerce",
    )
    selected = manifest.loc[task_numbers == int(args.index)]

    if len(selected) != 1:
        available = sorted(
            int(x)
            for x in task_numbers.dropna().astype(int).unique().tolist()
        )
        if available:
            available_text = (
                f"{available[0]}-{available[-1]}"
                if available == list(range(available[0], available[-1] + 1))
                else ", ".join(str(x) for x in available)
            )
        else:
            available_text = "none"
        raise RuntimeError(
            f"Expected exactly one manifest row for --index {args.index}; "
            f"found {len(selected)}. Available indices: {available_text}"
        )

    row = selected.iloc[0]
    accession = str(row.get("STUDY_ACCESSION", "")).strip()

    banner("STEP 05 - DIRECT SINGLE-INDEX MODE")
    print(f"Manifest : {manifest_file}")
    print(f"Index    : {args.index}")
    print(f"Study    : {accession}")
    print(f"Phenotype: {phenotype}")
    print(f"Ancestry : {ancestry_label} ({ancestry_code})")
    print(f"CPUs     : {args.cpus}")
    print()
    print(
        "NOTE: direct --index mode runs on the machine/node where this "
        "Python command is executed. On an HPC login node, use srun or "
        "sbatch --wrap to place this same command on a compute node."
    )

    old_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    old_manifest = os.environ.get("GWAS_CLUMP_MANIFEST")
    old_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    os.environ["SLURM_ARRAY_TASK_ID"] = str(args.index)
    os.environ["GWAS_CLUMP_MANIFEST"] = str(manifest_file)
    os.environ["SLURM_CPUS_PER_TASK"] = str(args.cpus)

    try:
        worker_mode()
    finally:
        if old_task is None:
            os.environ.pop("SLURM_ARRAY_TASK_ID", None)
        else:
            os.environ["SLURM_ARRAY_TASK_ID"] = old_task

        if old_manifest is None:
            os.environ.pop("GWAS_CLUMP_MANIFEST", None)
        else:
            os.environ["GWAS_CLUMP_MANIFEST"] = old_manifest

        if old_cpus is None:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
        else:
            os.environ["SLURM_CPUS_PER_TASK"] = old_cpus


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    # Generated SLURM-array worker mode takes priority and requires no CLI args.
    if os.environ.get("SLURM_ARRAY_TASK_ID") and os.environ.get("GWAS_CLUMP_MANIFEST"):
        worker_mode()
        return

    args = arguments()

    # Direct single-study execution using the same manifest/worker implementation.
    if args.index is not None:
        direct_index_mode(args)
        return

    # Standard planner mode: create/refresh manifest + SLURM array script.
    planner_mode(args)


if __name__ == "__main__":
    main()

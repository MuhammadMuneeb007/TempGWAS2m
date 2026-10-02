#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic GWAS Pipeline
STEP 09 - SpliceAI prediction and splice-evidence integration

Phenotype-agnostic and ancestry-agnostic. Each GWAS study is kept separate.

PLANNER:
    python Step09_Run_SpliceAI.py --phenotype migraine --ancestry EUR

DIRECT SINGLE TASK:
    python Step09_Run_SpliceAI.py --phenotype migraine --ancestry EUR --index 1

SLURM:
    sbatch Step09_SpliceAI_migraine_european.sh

INPUT per study:
    08_qtl/<phenotype>/<ancestry>/<GCST>/<GCST>_GTEx_QTL_variant_summary.tsv

WORKFLOW:
    Step08 QTL-integrated prioritized variants
        -> validate simple SNV/indel allele eligibility
        -> create GRCh38 VCF
        -> bcftools norm against GRCh38 FASTA
        -> SpliceAI 1.3.1
        -> parse DS_AG / DS_AL / DS_DG / DS_DL and DP values
        -> strongest prediction per variant
        -> merge with PIP + VEP + GTEx eQTL/sQTL
        -> splice-evidence summaries

DEFAULT SpliceAI settings:
    annotation = grch38
    distance   = 50
    mask       = 0  (raw/unmasked; appropriate for alternative-splicing analysis)

OUTPUT:
    09_splicing/<phenotype>/<ancestry>/
        spliceai_manifest.tsv
        spliceai_excluded_studies.tsv
        <GCST>/
            input/
                <GCST>_SpliceAI_input.raw.vcf
                <GCST>_SpliceAI_input.normalized.vcf
            spliceai/
                <GCST>_SpliceAI_output.vcf
                spliceai.log
            <GCST>_SpliceAI_all_gene_predictions.tsv.gz
            <GCST>_SpliceAI_variant_summary.tsv
            <GCST>_SpliceAI_prioritized.tsv
            <GCST>_SpliceAI_unscored_variants.tsv
            <GCST>_SpliceAI_locus_summary.tsv
            <GCST>_SpliceAI_category_summary.tsv
            spliceai_summary.json
            SPLICEAI_FAILED.txt  # hard failure only
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

STEP09_VERSION = "1.0.1"
DEFAULT_PARTITION = "general"
DEFAULT_TIME = "24:00:00"
DEFAULT_MEMORY = "64G"
DEFAULT_CPUS = 4
DEFAULT_MAX_PARALLEL = 2
DEFAULT_DISTANCE = 50
DEFAULT_MASK = 0
THRESHOLD_HIGH_RECALL = 0.20
THRESHOLD_RECOMMENDED = 0.50
THRESHOLD_HIGH_PRECISION = 0.80
ANNOTATION = "grch38"

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


def banner(text: str) -> None:
    print()
    print("=" * 104)
    print(text)
    print("=" * 104)


def normalize_text(value) -> str:
    x = str(value).strip().lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def slugify(value) -> str:
    return normalize_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = normalize_text(value)
    if key not in ANCESTRY_ALIASES:
        raise ValueError(
            f"Unsupported ancestry {value!r}. Use EUR, AFR, EAS, SAS, AMR "
            "or the corresponding full label."
        )
    return ANCESTRY_ALIASES[key]


def parse_walltime(value: str) -> int:
    text = str(value).strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Walltime must be HH:MM:SS or D-HH:MM:SS")
    h, m, s = map(int, parts)
    if days < 0 or h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError("Invalid walltime")
    return days * 86400 + h * 3600 + m * 60 + s


def validate_walltime(value: str) -> str:
    seconds = parse_walltime(value)
    if seconds <= 0:
        raise ValueError("Walltime must be greater than zero")
    if seconds > 86400:
        raise ValueError("Maximum allowed walltime is 24 hours")
    return value


def safe_read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_resource_env(root: Path) -> dict[str, str]:
    path = root / "resource_paths.env"
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        result[key] = value.strip().strip('"').strip("'")
    return result


def run_command(command, *, log_file: Path | None = None, env: dict | None = None) -> str:
    command = [str(x) for x in command]
    print()
    print("$ " + " ".join(shlex.quote(x) for x in command), flush=True)
    process_env = os.environ.copy()
    if env:
        process_env.update({str(k): str(v) for k, v in env.items()})
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=process_env,
    )
    captured: list[str] = []
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
            + " ".join(command)
            + "\n\n"
            + output[-8000:]
        )
    return output


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return np.nan


def arguments():
    parser = argparse.ArgumentParser(
        description="Run phenotype-agnostic SpliceAI on Step08 QTL-integrated variants."
    )
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--ancestry", required=True)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run exactly one SPLICEAI_TASK_ID from the existing manifest.",
    )
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--time", default=DEFAULT_TIME)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--distance", type=int, default=DEFAULT_DISTANCE)
    parser.add_argument(
        "--mask",
        type=int,
        choices=[0, 1],
        default=DEFAULT_MASK,
        help="SpliceAI -M value. Default 0 (raw/unmasked).",
    )
    parser.add_argument("--spliceai", default=None, help="Explicit SpliceAI executable path.")
    parser.add_argument("--fasta", default=None, help="Explicit GRCh38 FASTA path.")
    parser.add_argument("--bcftools", default=None, help="Explicit bcftools executable path.")
    parser.add_argument("--samtools", default=None, help="Explicit samtools executable path.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_arguments(args) -> None:
    if not args.phenotype.strip():
        raise ValueError("--phenotype cannot be empty")
    validate_walltime(args.time)
    if args.index is not None and args.index < 1:
        raise ValueError("--index must be >= 1")
    if args.cpus < 1:
        raise ValueError("--cpus must be >= 1")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    if args.distance < 1:
        raise ValueError("--distance must be >= 1")


def resolve_executable(explicit: str | None, candidates: list[Path], name: str) -> Path:
    ordered: list[Path] = []
    if explicit:
        ordered.append(Path(explicit).expanduser())
    ordered.extend(candidates)
    found = shutil.which(name)
    if found:
        ordered.append(Path(found))
    for path in ordered:
        try:
            p = path.resolve()
        except Exception:
            p = path
        if p.exists() and p.is_file() and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(f"Could not find executable: {name}")


def resolve_resources(root: Path, args) -> dict[str, Path]:
    env = read_resource_env(root)
    spliceai = resolve_executable(
        args.spliceai,
        [root / "envs" / "spliceai" / "bin" / "spliceai"],
        "spliceai",
    )
    bcftools = resolve_executable(
        args.bcftools,
        [root / "envs" / "pipeline" / "bin" / "bcftools"],
        "bcftools",
    )
    samtools = resolve_executable(
        args.samtools,
        [root / "envs" / "pipeline" / "bin" / "samtools"],
        "samtools",
    )
    fasta_candidates: list[Path] = []
    if args.fasta:
        fasta_candidates.append(Path(args.fasta).expanduser())
    if env.get("GRCH38_FASTA"):
        fasta_candidates.append(Path(env["GRCH38_FASTA"]).expanduser())
    fasta_candidates.append(
        root / "resources" / "genome" / "GRCh38" / "GRCh38.primary_assembly.genome.fa"
    )
    fasta = None
    for candidate in fasta_candidates:
        p = candidate.resolve()
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            fasta = p
            break
    if fasta is None:
        raise FileNotFoundError(
            "Could not find GRCh38 FASTA. Expected resource_paths.env GRCH38_FASTA "
            "or resources/genome/GRCh38/GRCh38.primary_assembly.genome.fa"
        )
    fai = Path(str(fasta) + ".fai")
    if not fai.exists() or fai.stat().st_size == 0:
        run_command([samtools, "faidx", fasta])
    return {
        "SPLICEAI": spliceai,
        "BCFTOOLS": bcftools,
        "SAMTOOLS": samtools,
        "FASTA": fasta,
    }


def planner_mode(args) -> None:
    validate_arguments(args)
    root = Path.cwd().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)
    resources = resolve_resources(root, args)

    qtl_root = root / "08_qtl" / phenotype_slug / ancestry_slug
    if not qtl_root.exists():
        raise FileNotFoundError(f"Step08 output directory not found:\n  {qtl_root}")

    out_root = root / "09_splicing" / phenotype_slug / ancestry_slug
    log_root = root / "logs" / "step09_spliceai" / phenotype_slug / ancestry_slug
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    rows = []
    excluded = []
    for study_dir in sorted(qtl_root.glob("GCST*")):
        if not study_dir.is_dir():
            continue
        accession = study_dir.name
        summary_file = study_dir / "qtl_summary.json"
        summary = safe_read_json(summary_file)
        input_file = study_dir / f"{accession}_GTEx_QTL_variant_summary.tsv"
        reason = ""
        if not summary:
            reason = "Step08 qtl_summary.json missing/unreadable"
        elif str(summary.get("STATUS", "")).upper() != "COMPLETE":
            reason = f"Step08 status is {summary.get('STATUS', 'UNKNOWN')}"
        elif not input_file.exists() or input_file.stat().st_size == 0:
            reason = "Step08 variant summary missing/empty"
        if reason:
            excluded.append(
                {
                    "STUDY_ACCESSION": accession,
                    "STEP08_STATUS": summary.get("STATUS", ""),
                    "REASON": reason,
                }
            )
            continue

        rows.append(
            {
                "SPLICEAI_TASK_ID": len(rows) + 1,
                "STUDY_ACCESSION": accession,
                "PHENOTYPE": phenotype,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
                "STEP08_VARIANT_FILE": str(input_file.resolve()),
                "STEP08_SUMMARY_FILE": str(summary_file.resolve()),
                "OUTPUT_DIR": str((out_root / accession).resolve()),
                "SPLICEAI": str(resources["SPLICEAI"]),
                "BCFTOOLS": str(resources["BCFTOOLS"]),
                "SAMTOOLS": str(resources["SAMTOOLS"]),
                "GRCH38_FASTA": str(resources["FASTA"]),
                "ANNOTATION": ANNOTATION,
                "DISTANCE": args.distance,
                "MASK": args.mask,
                "FORCE": bool(args.force),
                "STEP09_VERSION": STEP09_VERSION,
            }
        )

    excluded_file = out_root / "spliceai_excluded_studies.tsv"
    pd.DataFrame(
        excluded,
        columns=["STUDY_ACCESSION", "STEP08_STATUS", "REASON"],
    ).to_csv(excluded_file, sep="\t", index=False)

    if not rows:
        raise RuntimeError(
            "No completed Step08 studies are currently eligible for Step09.\n"
            f"Inspect:\n  {excluded_file}"
        )

    manifest = pd.DataFrame(rows)
    manifest_file = (out_root / "spliceai_manifest.tsv").resolve()
    manifest.to_csv(manifest_file, sep="\t", index=False)

    pipeline_python = root / "envs" / "pipeline" / "bin" / "python"
    if not pipeline_python.exists():
        pipeline_python = Path(sys.executable).resolve()

    script_path = Path(__file__).resolve()
    bash_file = root / f"Step09_SpliceAI_{phenotype_slug}_{ancestry_slug}.sh"
    array_spec = f"1-{len(manifest)}%{args.max_parallel}"
    job_name = f"SpAI_{phenotype_slug}_{ancestry_code.lower()}"[:100]

    bash_text = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_root}/spliceai.%A_%a.out
#SBATCH --error={log_root}/spliceai.%A_%a.err
#SBATCH --array={array_spec}
#SBATCH --mem={args.memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1

set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_SPLICEAI_MANIFEST={shlex.quote(str(manifest_file))}
{shlex.quote(str(pipeline_python))} {shlex.quote(str(script_path))}
'''
    bash_file.write_text(bash_text, encoding="utf-8")
    bash_file.chmod(0o755)

    banner("STEP 09 - SPLICEAI PLANNER")
    print(f"Phenotype          : {phenotype}")
    print(f"Ancestry           : {ancestry_label} ({ancestry_code})")
    print(f"Eligible studies   : {len(manifest)}")
    print(f"Excluded studies   : {len(excluded)}")
    print(f"SpliceAI           : {resources['SPLICEAI']}")
    print(f"bcftools           : {resources['BCFTOOLS']}")
    print(f"GRCh38 FASTA       : {resources['FASTA']}")
    print(f"Annotation         : {ANNOTATION}")
    print(f"Distance (-D)      : {args.distance}")
    print(f"Mask (-M)          : {args.mask}")
    print(f"Array              : {array_spec}")
    print()
    print(
        manifest[
            ["SPLICEAI_TASK_ID", "STUDY_ACCESSION"]
        ].to_string(index=False)
    )
    print(f"\nManifest:\n  {manifest_file}")
    print(f"Excluded:\n  {excluded_file}")
    print(f"Generated SLURM:\n  {bash_file}")
    print("\nNEXT COMMAND:\n")
    print(f"  sbatch {bash_file.name}")


def load_variants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    required = [
        "VEP_ID",
        "LOCUS_ID",
        "CHR",
        "REFERENCE_POS",
        "REFERENCE_REF",
        "REFERENCE_ALT",
        "PIP",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Step08 input is missing required columns: {missing}")

    for c in ["CHR", "REFERENCE_POS", "PIP"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).copy()
    df = df[df["CHR"].between(1, 22, inclusive="both")].copy()
    df["CHR"] = df["CHR"].astype(int)
    df["REFERENCE_POS"] = df["REFERENCE_POS"].astype(int)
    df["REFERENCE_REF"] = df["REFERENCE_REF"].astype(str).str.upper().str.strip()
    df["REFERENCE_ALT"] = df["REFERENCE_ALT"].astype(str).str.upper().str.strip()
    df = (
        df.sort_values(["LOCUS_ID", "PIP"], ascending=[True, False])
        .drop_duplicates(subset=["VEP_ID"], keep="first")
        .reset_index(drop=True)
    )
    dna = re.compile(r"^[ACGT]+$")
    valid = [
        bool(dna.fullmatch(ref)) and bool(dna.fullmatch(alt))
        for ref, alt in zip(df["REFERENCE_REF"], df["REFERENCE_ALT"])
    ]
    df["VALID_DNA_ALLELES"] = valid
    df["SPLICEAI_ALLELE_ELIGIBLE"] = [
        bool(ok and (len(ref) == 1 or len(alt) == 1))
        for ref, alt, ok in zip(
            df["REFERENCE_REF"],
            df["REFERENCE_ALT"],
            df["VALID_DNA_ALLELES"],
        )
    ]
    if df.empty:
        raise RuntimeError("No usable Step08 variants remain after validation")
    return df


def read_fasta_autosome_map(fasta: Path) -> tuple[dict[int, str], dict[str, int]]:
    """Return autosome -> FASTA contig name and contig lengths from <FASTA>.fai.

    Supports the two common GRCh38 naming schemes used by this pipeline:
      - chr1 ... chr22
      - 1 ... 22

    The exact names present in the FASTA are used in the VCF, preventing
    bcftools/faidx failures caused by chr-prefix mismatches.
    """
    fai = Path(str(fasta) + ".fai")
    if not fai.exists() or fai.stat().st_size == 0:
        raise FileNotFoundError(f"FASTA index missing: {fai}")

    lengths: dict[str, int] = {}
    with open(fai, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            try:
                lengths[fields[0]] = int(fields[1])
            except ValueError:
                continue

    mapping: dict[int, str] = {}
    missing = []
    for chrom in range(1, 23):
        candidates = [f"chr{chrom}", str(chrom)]
        hit = next((name for name in candidates if name in lengths), None)
        if hit is None:
            missing.append(chrom)
        else:
            mapping[chrom] = hit

    if missing:
        preview = ", ".join(list(lengths)[:12])
        raise RuntimeError(
            "Could not map GRCh38 autosomes 1-22 to FASTA contigs. "
            f"Missing chromosomes: {missing}. First FASTA contigs: {preview}"
        )

    return mapping, lengths


def create_raw_vcf(
    variants: pd.DataFrame,
    output: Path,
    fasta: Path,
) -> tuple[pd.DataFrame, dict[int, str]]:
    eligible = variants[variants["SPLICEAI_ALLELE_ELIGIBLE"]].copy()
    eligible = eligible.sort_values(["CHR", "REFERENCE_POS"])
    if eligible.empty:
        raise RuntimeError("No simple SNV/indel variants are eligible for SpliceAI")

    contig_map, contig_lengths = read_fasta_autosome_map(fasta)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##reference=GRCh38\n")
        handle.write("##source=GWAS2m_Step09_SpliceAI\n")

        for chrom in range(1, 23):
            name = contig_map[chrom]
            length = contig_lengths[name]
            handle.write(f"##contig=<ID={name},length={length}>\n")

        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        for row in eligible.itertuples(index=False):
            chrom_name = contig_map[int(row.CHR)]
            handle.write(
                f"{chrom_name}\t{int(row.REFERENCE_POS)}\t{row.VEP_ID}\t"
                f"{row.REFERENCE_REF}\t{row.REFERENCE_ALT}\t.\tPASS\t.\n"
            )

    return eligible, contig_map


def normalize_vcf(
    bcftools: Path,
    fasta: Path,
    raw_vcf: Path,
    normalized_vcf: Path,
    force: bool,
) -> None:
    if not force and normalized_vcf.exists() and normalized_vcf.stat().st_size > 100:
        print(f"Reusing normalized VCF: {normalized_vcf}")
        return
    run_command(
        [
            bcftools,
            "norm",
            "-f",
            fasta,
            "-c",
            "e",
            "-m",
            "-any",
            "-Ov",
            "-o",
            normalized_vcf,
            raw_vcf,
        ]
    )
    if not normalized_vcf.exists() or normalized_vcf.stat().st_size < 100:
        raise RuntimeError("Normalized VCF was not generated")


def run_spliceai(
    executable: Path,
    fasta: Path,
    normalized_vcf: Path,
    output_vcf: Path,
    log_file: Path,
    distance: int,
    mask: int,
    threads: int,
    force: bool,
) -> None:
    if not force and output_vcf.exists() and output_vcf.stat().st_size > 100:
        print(f"Reusing SpliceAI output: {output_vcf}")
        return
    env = {
        "OMP_NUM_THREADS": str(threads),
        "TF_NUM_INTRAOP_THREADS": str(threads),
        "TF_NUM_INTEROP_THREADS": "2",
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    }
    run_command(
        [
            executable,
            "-I",
            normalized_vcf,
            "-O",
            output_vcf,
            "-R",
            fasta,
            "-A",
            ANNOTATION,
            "-D",
            str(distance),
            "-M",
            str(mask),
        ],
        log_file=log_file,
        env=env,
    )
    if not output_vcf.exists() or output_vcf.stat().st_size < 100:
        raise RuntimeError("SpliceAI output VCF is missing/empty")


def parse_info(info: str) -> dict:
    result = {}
    if not info or info == ".":
        return result
    for item in info.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
        else:
            result[item] = True
    return result


def parse_spliceai_vcf(path: Path) -> tuple[pd.DataFrame, set[str], int]:
    records = []
    scored_ids: set[str] = set()
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, variant_id, ref, alt, _, _, info = fields[:8]
            total += 1
            value = parse_info(info).get("SpliceAI")
            if not value:
                continue
            for annotation in str(value).split(","):
                parts = annotation.split("|")
                if len(parts) != 10:
                    continue
                allele, symbol, ds_ag, ds_al, ds_dg, ds_dl, dp_ag, dp_al, dp_dg, dp_dl = parts
                scores = {
                    "DS_AG": safe_float(ds_ag),
                    "DS_AL": safe_float(ds_al),
                    "DS_DG": safe_float(ds_dg),
                    "DS_DL": safe_float(ds_dl),
                }
                finite = {k: v for k, v in scores.items() if np.isfinite(v)}
                if finite:
                    max_event = max(finite, key=finite.get)
                    max_ds = finite[max_event]
                else:
                    max_event = np.nan
                    max_ds = np.nan
                dps = {
                    "DS_AG": safe_int(dp_ag),
                    "DS_AL": safe_int(dp_al),
                    "DS_DG": safe_int(dp_dg),
                    "DS_DL": safe_int(dp_dl),
                }
                max_dp = dps.get(max_event, np.nan) if isinstance(max_event, str) else np.nan
                records.append(
                    {
                        "VEP_ID": variant_id,
                        "SPLICEAI_CHR": chrom,
                        "SPLICEAI_POS": safe_int(pos),
                        "SPLICEAI_REF": ref,
                        "SPLICEAI_ALT": alt,
                        "SPLICEAI_ALLELE": allele,
                        "SPLICEAI_GENE": symbol,
                        "DS_AG": scores["DS_AG"],
                        "DS_AL": scores["DS_AL"],
                        "DS_DG": scores["DS_DG"],
                        "DS_DL": scores["DS_DL"],
                        "DP_AG": safe_int(dp_ag),
                        "DP_AL": safe_int(dp_al),
                        "DP_DG": safe_int(dp_dg),
                        "DP_DL": safe_int(dp_dl),
                        "SPLICEAI_MAX_DS": max_ds,
                        "SPLICEAI_MAX_EVENT": max_event,
                        "SPLICEAI_MAX_DP": max_dp,
                    }
                )
                scored_ids.add(variant_id)
    return pd.DataFrame(records), scored_ids, total


def spliceai_class(score) -> str:
    if pd.isna(score):
        return "NO_PREDICTION"
    value = float(score)
    if value >= THRESHOLD_HIGH_PRECISION:
        return "HIGH_PRECISION_GE_0.80"
    if value >= THRESHOLD_RECOMMENDED:
        return "RECOMMENDED_GE_0.50"
    if value >= THRESHOLD_HIGH_RECALL:
        return "HIGH_RECALL_GE_0.20"
    return "LOW_LT_0.20"


def create_variant_summary(variants: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        top = pd.DataFrame(columns=["VEP_ID"])
        genes = pd.DataFrame(columns=["VEP_ID", "SPLICEAI_ALL_GENES"])
    else:
        predictions = predictions.copy()
        predictions["SPLICEAI_MAX_DS"] = pd.to_numeric(
            predictions["SPLICEAI_MAX_DS"], errors="coerce"
        )
        predictions = predictions.sort_values(
            ["VEP_ID", "SPLICEAI_MAX_DS"], ascending=[True, False]
        )
        top = predictions.drop_duplicates("VEP_ID", keep="first").copy()
        genes = (
            predictions.groupby("VEP_ID")["SPLICEAI_GENE"]
            .agg(
                lambda values: ";".join(
                    sorted({str(x) for x in values if pd.notna(x) and str(x).strip()})
                )
            )
            .reset_index()
            .rename(columns={"SPLICEAI_GENE": "SPLICEAI_ALL_GENES"})
        )

    summary = variants.merge(top, on="VEP_ID", how="left", validate="one_to_one")
    summary = summary.merge(genes, on="VEP_ID", how="left", validate="one_to_one")
    summary["SPLICEAI_STATUS"] = np.where(
        ~summary["SPLICEAI_ALLELE_ELIGIBLE"],
        "NOT_SIMPLE_SNV_OR_INDEL",
        np.where(summary["SPLICEAI_MAX_DS"].notna(), "SCORED", "NO_GENE_PREDICTION"),
    )
    summary["SPLICEAI_CLASS"] = summary["SPLICEAI_MAX_DS"].apply(spliceai_class)
    summary["SPLICEAI_GE_0_20"] = summary["SPLICEAI_MAX_DS"] >= THRESHOLD_HIGH_RECALL
    summary["SPLICEAI_GE_0_50"] = summary["SPLICEAI_MAX_DS"] >= THRESHOLD_RECOMMENDED
    summary["SPLICEAI_GE_0_80"] = summary["SPLICEAI_MAX_DS"] >= THRESHOLD_HIGH_PRECISION

    has_sqtl = (
        bool_series(summary["HAS_GTEX_SQTL"])
        if "HAS_GTEX_SQTL" in summary.columns
        else pd.Series(False, index=summary.index)
    )
    summary["HAS_GTEX_SQTL_BOOL"] = has_sqtl

    vep_splice = (
        summary["FUNCTIONAL_CATEGORY"].fillna("").astype(str).str.upper().eq("SPLICING")
        if "FUNCTIONAL_CATEGORY" in summary.columns
        else pd.Series(False, index=summary.index)
    )
    summary["VEP_SPLICE_CATEGORY"] = vep_splice

    evidence = []
    for _, row in summary.iterrows():
        score = row.get("SPLICEAI_MAX_DS", np.nan)
        sqtl = bool(row["HAS_GTEX_SQTL_BOOL"])
        vep = bool(row["VEP_SPLICE_CATEGORY"])
        score = -1 if pd.isna(score) else float(score)
        if score >= 0.50 and sqtl:
            label = "VERY_STRONG_MODEL_PLUS_SQTL"
        elif score >= 0.20 and sqtl:
            label = "STRONG_MODEL_PLUS_SQTL"
        elif score >= 0.50:
            label = "STRONG_SPLICEAI"
        elif score >= 0.20:
            label = "SPLICEAI_SUPPORTED"
        elif sqtl and vep:
            label = "SQTL_PLUS_VEP_SPLICE"
        elif sqtl:
            label = "SQTL_SUPPORTED"
        elif vep:
            label = "VEP_SPLICE_ONLY"
        else:
            label = "NO_STRONG_SPLICE_EVIDENCE"
        evidence.append(label)

    summary["SPLICE_EVIDENCE_CLASS"] = evidence
    rank = {
        "VERY_STRONG_MODEL_PLUS_SQTL": 1,
        "STRONG_MODEL_PLUS_SQTL": 2,
        "STRONG_SPLICEAI": 3,
        "SPLICEAI_SUPPORTED": 4,
        "SQTL_PLUS_VEP_SPLICE": 5,
        "SQTL_SUPPORTED": 6,
        "VEP_SPLICE_ONLY": 7,
        "NO_STRONG_SPLICE_EVIDENCE": 8,
    }
    summary["SPLICE_EVIDENCE_RANK"] = summary["SPLICE_EVIDENCE_CLASS"].map(rank).fillna(99)
    summary = summary.sort_values(
        ["SPLICE_EVIDENCE_RANK", "PIP", "SPLICEAI_MAX_DS"],
        ascending=[True, False, False],
    )
    return summary


def create_locus_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for locus, group in summary.groupby("LOCUS_ID", dropna=False):
        scored = group[group["SPLICEAI_STATUS"] == "SCORED"]
        if not scored.empty and scored["SPLICEAI_MAX_DS"].notna().any():
            top = scored.loc[scored["SPLICEAI_MAX_DS"].idxmax()]
            top_variant = top.get("REFERENCE_ID", top.get("VEP_ID", ""))
            top_gene = top.get("SPLICEAI_GENE", "")
            max_ds = top.get("SPLICEAI_MAX_DS", np.nan)
            top_event = top.get("SPLICEAI_MAX_EVENT", "")
        else:
            top_variant = ""
            top_gene = ""
            max_ds = np.nan
            top_event = ""
        rows.append(
            {
                "LOCUS_ID": locus,
                "N_INPUT_VARIANTS": int(len(group)),
                "N_SPLICEAI_SCORED": int((group["SPLICEAI_STATUS"] == "SCORED").sum()),
                "N_SPLICEAI_GE_0_20": int(group["SPLICEAI_GE_0_20"].sum()),
                "N_SPLICEAI_GE_0_50": int(group["SPLICEAI_GE_0_50"].sum()),
                "N_SPLICEAI_GE_0_80": int(group["SPLICEAI_GE_0_80"].sum()),
                "N_GTEX_SQTL": int(group["HAS_GTEX_SQTL_BOOL"].sum()),
                "N_SPLICEAI_GE_0_20_AND_SQTL": int(
                    (group["SPLICEAI_GE_0_20"] & group["HAS_GTEX_SQTL_BOOL"]).sum()
                ),
                "N_SPLICEAI_GE_0_50_AND_SQTL": int(
                    (group["SPLICEAI_GE_0_50"] & group["HAS_GTEX_SQTL_BOOL"]).sum()
                ),
                "MAX_SPLICEAI_DS": max_ds,
                "TOP_SPLICEAI_VARIANT": top_variant,
                "TOP_SPLICEAI_GENE": top_gene,
                "TOP_SPLICEAI_EVENT": top_event,
                "MAX_PIP": pd.to_numeric(group["PIP"], errors="coerce").max(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["N_SPLICEAI_GE_0_50_AND_SQTL", "MAX_SPLICEAI_DS", "MAX_PIP"],
        ascending=[False, False, False],
    )


def create_category_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if "FUNCTIONAL_CATEGORY" not in summary.columns:
        return pd.DataFrame(
            columns=[
                "FUNCTIONAL_CATEGORY",
                "N_VARIANTS",
                "N_SPLICEAI_SCORED",
                "N_SPLICEAI_GE_0_20",
                "N_SPLICEAI_GE_0_50",
                "N_SPLICEAI_GE_0_80",
                "N_SQTL",
                "N_GE_0_20_AND_SQTL",
                "MAX_SPLICEAI_DS",
                "MEDIAN_SPLICEAI_DS",
                "MAX_PIP",
            ]
        )
    rows = []
    for category, group in summary.groupby("FUNCTIONAL_CATEGORY", dropna=False):
        rows.append(
            {
                "FUNCTIONAL_CATEGORY": category,
                "N_VARIANTS": int(len(group)),
                "N_SPLICEAI_SCORED": int((group["SPLICEAI_STATUS"] == "SCORED").sum()),
                "N_SPLICEAI_GE_0_20": int(group["SPLICEAI_GE_0_20"].sum()),
                "N_SPLICEAI_GE_0_50": int(group["SPLICEAI_GE_0_50"].sum()),
                "N_SPLICEAI_GE_0_80": int(group["SPLICEAI_GE_0_80"].sum()),
                "N_SQTL": int(group["HAS_GTEX_SQTL_BOOL"].sum()),
                "N_GE_0_20_AND_SQTL": int(
                    (group["SPLICEAI_GE_0_20"] & group["HAS_GTEX_SQTL_BOOL"]).sum()
                ),
                "MAX_SPLICEAI_DS": pd.to_numeric(group["SPLICEAI_MAX_DS"], errors="coerce").max(),
                "MEDIAN_SPLICEAI_DS": pd.to_numeric(group["SPLICEAI_MAX_DS"], errors="coerce").median(),
                "MAX_PIP": pd.to_numeric(group["PIP"], errors="coerce").max(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["N_SPLICEAI_GE_0_20", "N_SQTL"], ascending=[False, False]
    )


def worker_mode() -> None:
    task_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    manifest_value = os.environ.get("GWAS_SPLICEAI_MANIFEST")
    if not task_value:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not defined")
    if not manifest_value:
        raise RuntimeError("GWAS_SPLICEAI_MANIFEST is not defined")

    task_id = int(task_value)
    manifest_file = Path(manifest_value).resolve()
    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str, low_memory=False)
    required = [
        "SPLICEAI_TASK_ID",
        "STUDY_ACCESSION",
        "PHENOTYPE",
        "ANCESTRY_CODE",
        "ANCESTRY_LABEL",
        "STEP08_VARIANT_FILE",
        "OUTPUT_DIR",
        "SPLICEAI",
        "BCFTOOLS",
        "GRCH38_FASTA",
        "DISTANCE",
        "MASK",
    ]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise RuntimeError(f"SpliceAI manifest is missing required columns: {missing}")
    ids = pd.to_numeric(manifest["SPLICEAI_TASK_ID"], errors="coerce")
    selected = manifest.loc[ids == task_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest row for task {task_id}; found {len(selected)}"
        )
    row = selected.iloc[0]
    accession = str(row["STUDY_ACCESSION"]).strip()
    phenotype = str(row["PHENOTYPE"]).strip()
    ancestry_code = str(row["ANCESTRY_CODE"]).strip().upper()
    ancestry_label = str(row["ANCESTRY_LABEL"]).strip()
    input_file = Path(row["STEP08_VARIANT_FILE"]).resolve()
    output_dir = Path(row["OUTPUT_DIR"]).resolve()
    spliceai = Path(row["SPLICEAI"]).resolve()
    bcftools = Path(row["BCFTOOLS"]).resolve()
    fasta = Path(row["GRCH38_FASTA"]).resolve()
    distance = int(row["DISTANCE"])
    mask = int(row["MASK"])
    force = str(row.get("FORCE", "False")).strip().lower() in {"true", "1", "yes", "y"}
    force = force or os.environ.get("STEP09_FORCE", "0") == "1"
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_CPUS))

    input_dir = output_dir / "input"
    splice_dir = output_dir / "spliceai"
    input_dir.mkdir(parents=True, exist_ok=True)
    splice_dir.mkdir(parents=True, exist_ok=True)

    raw_vcf = input_dir / f"{accession}_SpliceAI_input.raw.vcf"
    norm_vcf = input_dir / f"{accession}_SpliceAI_input.normalized.vcf"
    splice_vcf = splice_dir / f"{accession}_SpliceAI_output.vcf"
    splice_log = splice_dir / "spliceai.log"
    gene_output = output_dir / f"{accession}_SpliceAI_all_gene_predictions.tsv.gz"
    variant_output = output_dir / f"{accession}_SpliceAI_variant_summary.tsv"
    prioritized_output = output_dir / f"{accession}_SpliceAI_prioritized.tsv"
    unscored_output = output_dir / f"{accession}_SpliceAI_unscored_variants.tsv"
    locus_output = output_dir / f"{accession}_SpliceAI_locus_summary.tsv"
    category_output = output_dir / f"{accession}_SpliceAI_category_summary.tsv"
    summary_output = output_dir / "spliceai_summary.json"
    failed_output = output_dir / "SPLICEAI_FAILED.txt"

    # A previous failed run may have left partial/header-only intermediate files.
    # Never resume those blindly; rebuild the intermediate VCF/SpliceAI outputs.
    if failed_output.exists():
        print(f"Previous failure marker found; rebuilding intermediates: {failed_output}")
        force = True

    if (
        not force
        and variant_output.exists()
        and variant_output.stat().st_size > 0
        and summary_output.exists()
        and str(safe_read_json(summary_output).get("STATUS", "")).upper() == "COMPLETE"
        and not failed_output.exists()
    ):
        banner("STEP 09 ALREADY COMPLETE")
        print(f"Study : {accession}")
        print(f"Output: {output_dir}")
        return

    banner("STEP 09 - SPLICEAI WORKER")
    print(f"Task       : {task_id}")
    print(f"Study      : {accession}")
    print(f"Phenotype  : {phenotype}")
    print(f"Ancestry   : {ancestry_label} ({ancestry_code})")
    print(f"Input      : {input_file}")
    print(f"SpliceAI   : {spliceai}")
    print(f"FASTA      : {fasta}")
    print(f"Distance   : {distance}")
    print(f"Mask       : {mask}")
    print(f"Threads    : {threads}")

    try:
        for p, label in [(spliceai, "SpliceAI"), (bcftools, "bcftools"), (fasta, "GRCh38 FASTA")]:
            if not p.exists():
                raise FileNotFoundError(f"{label} missing: {p}")
        variants = load_variants(input_file)

        banner("CREATING SPLICEAI VCF")
        eligible, contig_map = create_raw_vcf(variants, raw_vcf, fasta)
        example_contig = contig_map.get(1, "?")
        contig_style = "chr-prefixed" if str(example_contig).lower().startswith("chr") else "unprefixed"
        print(f"FASTA contig style      : {contig_style} (chr1 -> {example_contig})")
        print(f"Input variants          : {len(variants):,}")
        print(f"SpliceAI allele-eligible: {len(eligible):,}")
        print(f"Allele-ineligible       : {len(variants) - len(eligible):,}")
        print(f"Loci                    : {variants['LOCUS_ID'].nunique():,}")

        banner("NORMALIZING VCF")
        normalize_vcf(bcftools, fasta, raw_vcf, norm_vcf, force)

        banner("RUNNING SPLICEAI")
        run_spliceai(
            spliceai,
            fasta,
            norm_vcf,
            splice_vcf,
            splice_log,
            distance,
            mask,
            threads,
            force,
        )

        banner("PARSING SPLICEAI")
        predictions, scored_ids, n_vcf = parse_spliceai_vcf(splice_vcf)
        if predictions.empty:
            print("WARNING: SpliceAI returned zero gene-level predictions.")
            predictions = pd.DataFrame(
                columns=[
                    "VEP_ID",
                    "SPLICEAI_CHR",
                    "SPLICEAI_POS",
                    "SPLICEAI_REF",
                    "SPLICEAI_ALT",
                    "SPLICEAI_ALLELE",
                    "SPLICEAI_GENE",
                    "DS_AG",
                    "DS_AL",
                    "DS_DG",
                    "DS_DL",
                    "DP_AG",
                    "DP_AL",
                    "DP_DG",
                    "DP_DL",
                    "SPLICEAI_MAX_DS",
                    "SPLICEAI_MAX_EVENT",
                    "SPLICEAI_MAX_DP",
                ]
            )
        predictions.to_csv(gene_output, sep="\t", index=False, compression="gzip")

        summary = create_variant_summary(variants, predictions)
        summary.to_csv(variant_output, sep="\t", index=False)

        prioritized = summary[
            summary["SPLICEAI_GE_0_20"]
            | summary["HAS_GTEX_SQTL_BOOL"]
            | summary["VEP_SPLICE_CATEGORY"]
        ].copy()
        prioritized = prioritized.sort_values(
            ["SPLICE_EVIDENCE_RANK", "PIP", "SPLICEAI_MAX_DS"],
            ascending=[True, False, False],
        )
        prioritized.to_csv(prioritized_output, sep="\t", index=False)

        unscored = summary[summary["SPLICEAI_STATUS"] != "SCORED"].copy()
        unscored.to_csv(unscored_output, sep="\t", index=False)

        locus_summary = create_locus_summary(summary)
        locus_summary.to_csv(locus_output, sep="\t", index=False)
        category_summary = create_category_summary(summary)
        category_summary.to_csv(category_output, sep="\t", index=False)

        n_scored = int((summary["SPLICEAI_STATUS"] == "SCORED").sum())
        n20 = int(summary["SPLICEAI_GE_0_20"].sum())
        n50 = int(summary["SPLICEAI_GE_0_50"].sum())
        n80 = int(summary["SPLICEAI_GE_0_80"].sum())
        n_sqtl = int(summary["HAS_GTEX_SQTL_BOOL"].sum())
        n20_sqtl = int((summary["SPLICEAI_GE_0_20"] & summary["HAS_GTEX_SQTL_BOOL"]).sum())
        n50_sqtl = int((summary["SPLICEAI_GE_0_50"] & summary["HAS_GTEX_SQTL_BOOL"]).sum())

        payload = {
            "STEP09_VERSION": STEP09_VERSION,
            "SPLICEAI_VERSION_EXPECTED": "1.3.1",
            "SPLICEAI_ANNOTATION": ANNOTATION,
            "STUDY_ACCESSION": accession,
            "PHENOTYPE": phenotype,
            "ANCESTRY_CODE": ancestry_code,
            "ANCESTRY_LABEL": ancestry_label,
            "DISTANCE": distance,
            "MASK": mask,
            "FASTA": str(fasta),
            "FASTA_CONTIG_STYLE": contig_style,
            "FASTA_CHR1_CONTIG": str(example_contig),
            "N_INPUT_VARIANTS": int(len(summary)),
            "N_ALLELE_ELIGIBLE": int(len(eligible)),
            "N_SPLICEAI_VCF_VARIANTS": int(n_vcf),
            "N_SPLICEAI_SCORED_VARIANTS": n_scored,
            "N_GENE_LEVEL_PREDICTIONS": int(len(predictions)),
            "N_SPLICEAI_GE_0_20": n20,
            "N_SPLICEAI_GE_0_50": n50,
            "N_SPLICEAI_GE_0_80": n80,
            "N_GTEX_SQTL": n_sqtl,
            "N_GE_0_20_AND_SQTL": n20_sqtl,
            "N_GE_0_50_AND_SQTL": n50_sqtl,
            "N_PRIORITIZED_SPLICE_VARIANTS": int(len(prioritized)),
            "N_LOCI": int(summary["LOCUS_ID"].nunique()),
            "INPUT_VARIANT_FILE": str(input_file),
            "RAW_VCF": str(raw_vcf),
            "NORMALIZED_VCF": str(norm_vcf),
            "SPLICEAI_VCF": str(splice_vcf),
            "ALL_GENE_PREDICTIONS": str(gene_output),
            "VARIANT_SUMMARY_FILE": str(variant_output),
            "PRIORITIZED_FILE": str(prioritized_output),
            "UNSCORED_FILE": str(unscored_output),
            "LOCUS_SUMMARY_FILE": str(locus_output),
            "CATEGORY_SUMMARY_FILE": str(category_output),
            "STATUS": "COMPLETE",
            "COMPLETED_UTC": datetime.now(timezone.utc).isoformat(),
        }
        write_json(summary_output, payload)
        if failed_output.exists():
            failed_output.unlink()

        banner("STEP 09 COMPLETE")
        print(f"Study                     : {accession}")
        print(f"Input variants            : {len(summary):,}")
        print(f"SpliceAI-scored variants  : {n_scored:,}")
        print(f"SpliceAI >= 0.20          : {n20:,}")
        print(f"SpliceAI >= 0.50          : {n50:,}")
        print(f"SpliceAI >= 0.80          : {n80:,}")
        print(f"GTEx sQTL variants        : {n_sqtl:,}")
        print(f">=0.20 + sQTL             : {n20_sqtl:,}")
        print(f">=0.50 + sQTL             : {n50_sqtl:,}")
        print(f"Prioritized splice variants: {len(prioritized):,}")
        print(f"Loci                      : {summary['LOCUS_ID'].nunique():,}")
        print(f"Main table                : {variant_output}")

    except Exception as error:
        failure_text = (
            "STEP 09 SPLICEAI FAILED\n\n"
            f"Task ID: {task_id}\n"
            f"Accession: {accession}\n"
            f"Phenotype: {phenotype}\n"
            f"Ancestry: {ancestry_label} ({ancestry_code})\n"
            f"Input: {input_file}\n\n"
            f"Error:\n{type(error).__name__}: {error}\n\n"
            f"{traceback.format_exc()}"
        )
        failed_output.write_text(failure_text, encoding="utf-8")
        print(failure_text, file=sys.stderr)
        raise


def direct_index_mode(args) -> None:
    validate_arguments(args)
    root = Path.cwd().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)
    manifest_file = (
        root
        / "09_splicing"
        / phenotype_slug
        / ancestry_slug
        / "spliceai_manifest.tsv"
    ).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Step09 manifest does not exist:\n  {manifest_file}\n\n"
            "Run the planner once without --index first."
        )
    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str, low_memory=False)
    if "SPLICEAI_TASK_ID" not in manifest.columns:
        raise RuntimeError("Manifest is missing SPLICEAI_TASK_ID")
    ids = pd.to_numeric(manifest["SPLICEAI_TASK_ID"], errors="coerce")
    selected = manifest.loc[ids == args.index]
    if len(selected) != 1:
        available = [int(x) for x in ids.dropna()]
        raise RuntimeError(f"Index {args.index} not found exactly once. Available: {available}")
    row = selected.iloc[0]
    banner("STEP 09 - DIRECT SINGLE-INDEX MODE")
    print(f"Manifest : {manifest_file}")
    print(f"Index    : {args.index}")
    print(f"Study    : {row['STUDY_ACCESSION']}")
    print("\nNOTE: direct mode runs on the current node. Use srun on HPC for substantial work.")

    keys = ["SLURM_ARRAY_TASK_ID", "GWAS_SPLICEAI_MANIFEST", "SLURM_CPUS_PER_TASK", "STEP09_FORCE"]
    old = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["SLURM_ARRAY_TASK_ID"] = str(args.index)
        os.environ["GWAS_SPLICEAI_MANIFEST"] = str(manifest_file)
        os.environ["SLURM_CPUS_PER_TASK"] = str(args.cpus)
        if args.force:
            os.environ["STEP09_FORCE"] = "1"
        worker_mode()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> None:
    if os.environ.get("SLURM_ARRAY_TASK_ID") and os.environ.get("GWAS_SPLICEAI_MANIFEST"):
        worker_mode()
        return
    args = arguments()
    if args.index is not None:
        direct_index_mode(args)
    else:
        planner_mode(args)


if __name__ == "__main__":
    main()

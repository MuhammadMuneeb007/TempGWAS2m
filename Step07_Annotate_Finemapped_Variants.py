#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic GWAS Pipeline
STEP 07 - Functional annotation of fine-mapped variants with Ensembl VEP

Planner mode:
    python Step07_Annotate_Finemapped_Variants.py \
        --phenotype migraine --ancestry EUR

Direct one-study mode:
    python Step07_Annotate_Finemapped_Variants.py \
        --phenotype migraine --ancestry EUR --index 1

Generated SLURM mode:
    sbatch Step07_Annotate_GWAS_migraine_european.sh

Selection rule (preserved from the prior CAD pipeline):
    - every 95% credible-set variant
    - every variant with PIP >= --min-pip (default 0.01)
    - the highest-PIP variant from every locus

Studies remain separate. Only Step 06 studies with successful loci are included
by default. Partial Step 06 studies can be included with --allow-partial.
"""

from __future__ import annotations

import argparse
import io
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

DEFAULT_PARTITION = "general"
DEFAULT_TIME = "24:00:00"
DEFAULT_MEMORY = "32G"
DEFAULT_CPUS = 4
DEFAULT_MAX_PARALLEL = 2
DEFAULT_MIN_PIP = 0.01
ASSEMBLY = "GRCh38"
SPECIES = "homo_sapiens"
STEP07_VERSION = "1.0.0"

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
    print("=" * 96)
    print(text)
    print("=" * 96)


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
            f"Unsupported ancestry {value!r}. Use EUR, AFR, EAS, SAS, or AMR."
        )
    return ANCESTRY_ALIASES[key]


def parse_walltime(value: str) -> int:
    text = str(value).strip()
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Walltime must be HH:MM:SS or D-HH:MM:SS")
    hours, minutes, seconds = map(int, parts)
    if days < 0 or hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("Invalid walltime")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def validate_walltime(value: str) -> str:
    seconds = parse_walltime(value)
    if seconds <= 0:
        raise ValueError("Walltime must be > 0")
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


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    x = series.fillna("").astype(str).str.strip().str.lower()
    return x.isin({"1", "true", "yes", "y", "t"})


def run_command(command, log_file: Path | None = None) -> str:
    command = [str(x) for x in command]
    print("$ " + " ".join(shlex.quote(x) for x in command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
            f"{' '.join(command)}\n\n{output[-5000:]}"
        )
    return output


def arguments():
    parser = argparse.ArgumentParser(
        description="Annotate Step 06 fine-mapped variants with Ensembl VEP."
    )
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--ancestry", required=True)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run one ANNOTATION_TASK_ID directly from the existing manifest.",
    )
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--time", default=DEFAULT_TIME)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--min-pip", type=float, default=DEFAULT_MIN_PIP)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--vep", default=None)
    parser.add_argument("--vep-cache", default=None)
    parser.add_argument(
        "--fasta",
        default=None,
        help=(
            "GRCh38 FASTA required for offline VEP HGVS generation. "
            "If omitted, GRCH38_FASTA/VEP_FASTA from resource_paths.env "
            "or resources/genome/GRCh38/GRCh38.primary_assembly.genome.fa is used."
        ),
    )
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
    if not 0 <= args.min_pip <= 1:
        raise ValueError("--min-pip must be in [0,1]")


def read_resource_env(root: Path) -> dict[str, str]:
    path = root / "resource_paths.env"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        value = value.strip().strip("'").strip('"')
        if key:
            out[key] = value
    return out


def resolve_vep_executable(root: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("VEP"):
        candidates.append(Path(os.environ["VEP"]).expanduser())
    resource_env = read_resource_env(root)
    for key in ("VEP", "VEP_BIN", "VEP_EXECUTABLE"):
        if resource_env.get(key):
            candidates.append(Path(resource_env[key]).expanduser())
    candidates.extend(
        [
            root / "external_tools" / "ensembl-vep" / "vep",
            root / "external_tools" / "vep" / "vep",
            root / "envs" / "pipeline" / "bin" / "vep",
        ]
    )
    which = shutil.which("vep")
    if which:
        candidates.append(Path(which))
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "Could not find Ensembl VEP. Pass --vep /path/to/ensembl-vep/vep"
    )


def cache_is_valid(path: Path) -> bool:
    human = path / "homo_sapiens"
    return path.is_dir() and human.is_dir() and bool(list(human.glob("*_GRCh38")))


def resolve_vep_cache(root: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for key in ("VEP_CACHE", "VEP_CACHE_DIR", "VEP_DIR_CACHE"):
        if os.environ.get(key):
            candidates.append(Path(os.environ[key]).expanduser())
    resource_env = read_resource_env(root)
    for key in ("VEP_CACHE", "VEP_CACHE_DIR", "VEP_DIR_CACHE"):
        if resource_env.get(key):
            candidates.append(Path(resource_env[key]).expanduser())
    candidates.extend(
        [
            root / "resources" / "vep_cache",
            root / "resources" / "VEP" / "cache",
            root / "resources" / "vep" / "cache",
            root / "external_tools" / "vep_cache",
            Path.home() / ".vep",
        ]
    )
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if cache_is_valid(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a human GRCh38 VEP cache. Expected "
        "<cache>/homo_sapiens/116_GRCh38/. Pass --vep-cache /path/to/cache"
    )


def resolve_grch38_fasta(root: Path, explicit: str | None) -> Path:
    """Resolve an indexed GRCh38 FASTA for offline VEP/HGVS."""
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    for key in ("GRCH38_FASTA", "VEP_FASTA", "REFERENCE_FASTA"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value).expanduser())

    resource_env = read_resource_env(root)
    for key in ("GRCH38_FASTA", "VEP_FASTA", "REFERENCE_FASTA"):
        value = resource_env.get(key)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.append(
        root / "resources" / "genome" / "GRCh38" / "GRCh38.primary_assembly.genome.fa"
    )

    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))

        if not candidate.exists() or not candidate.is_file() or candidate.stat().st_size <= 0:
            continue

        fai = Path(str(candidate) + ".fai")
        if fai.exists() and fai.stat().st_size > 0:
            return candidate

    raise FileNotFoundError(
        "Could not find an indexed GRCh38 FASTA for offline VEP/HGVS.\n"
        "Your setup should provide GRCH38_FASTA in resource_paths.env, or pass:\n"
        "  --fasta /path/to/GRCh38.fa"
    )


def detect_cache_release(cache_root: Path) -> str:
    releases = []
    for path in (cache_root / "homo_sapiens").glob("*_GRCh38"):
        try:
            releases.append(int(path.name.split("_", 1)[0]))
        except ValueError:
            pass
    return str(max(releases)) if releases else "UNKNOWN"


def planner_mode(args) -> None:
    validate_arguments(args)
    root = Path.cwd().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)

    pipeline_python = root / "envs" / "pipeline" / "bin" / "python"
    if not pipeline_python.exists():
        raise FileNotFoundError(f"Pipeline Python missing: {pipeline_python}")

    vep = resolve_vep_executable(root, args.vep)
    vep_cache = resolve_vep_cache(root, args.vep_cache)
    vep_fasta = resolve_grch38_fasta(root, args.fasta)
    cache_release = detect_cache_release(vep_cache)

    fine_root = root / "06_finemapping" / phenotype_slug / ancestry_slug
    if not fine_root.exists():
        raise FileNotFoundError(f"Step 06 directory not found: {fine_root}")

    out_root = root / "07_annotation" / phenotype_slug / ancestry_slug
    log_root = root / "logs" / "step07_annotation" / phenotype_slug / ancestry_slug
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    accessions = set()
    fm = fine_root / "finemapping_manifest.tsv"
    if fm.exists() and fm.stat().st_size > 0:
        x = pd.read_csv(fm, sep="\t", dtype=str, low_memory=False)
        if "STUDY_ACCESSION" in x.columns:
            accessions.update(v.strip() for v in x["STUDY_ACCESSION"].dropna().astype(str) if v.strip())
    accessions.update(p.name for p in fine_root.glob("GCST*") if p.is_dir())

    rows = []
    excluded = []

    for accession in sorted(accessions):
        study = fine_root / accession
        summary_file = study / "finemapping_summary.json"
        fine_file = study / f"{accession}_finemapped_variants.tsv.gz"
        cs_file = study / f"{accession}_95pct_credible_sets.tsv"
        summary = safe_read_json(summary_file)

        status = str(summary.get("STATUS", "")).strip().upper()
        n_loci = int(summary.get("N_LOCI", 0) or 0)
        n_success = int(summary.get("N_LOCI_SUCCESS", 0) or 0)
        n_failed = int(summary.get("N_LOCI_FAILED", 0) or 0)

        reason = ""
        if not summary:
            reason = "Step 06 summary missing/unreadable"
        elif status == "NO_SIGNIFICANT_LOCI" or n_loci == 0:
            reason = "No significant loci to annotate"
        elif n_success == 0:
            reason = "No successfully fine-mapped loci"
        elif n_failed > 0 and not args.allow_partial:
            reason = f"Partial Step 06 result: {n_success} successful / {n_failed} failed loci"
        elif not fine_file.exists() or fine_file.stat().st_size == 0:
            reason = "Fine-mapped variant file missing/empty"

        if reason:
            excluded.append(
                {
                    "STUDY_ACCESSION": accession,
                    "STEP06_STATUS": status,
                    "N_LOCI": n_loci,
                    "N_LOCI_SUCCESS": n_success,
                    "N_LOCI_FAILED": n_failed,
                    "REASON": reason,
                }
            )
            continue

        rows.append(
            {
                "ANNOTATION_TASK_ID": len(rows) + 1,
                "STUDY_ACCESSION": accession,
                "PHENOTYPE": phenotype,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
                "FINEMAPPED_VARIANTS_FILE": str(fine_file.resolve()),
                "CREDIBLE_SETS_FILE": str(cs_file.resolve()) if cs_file.exists() else "",
                "STEP06_SUMMARY_FILE": str(summary_file.resolve()),
                "N_LOCI": n_loci,
                "N_LOCI_SUCCESS": n_success,
                "N_LOCI_FAILED": n_failed,
                "MIN_PIP": args.min_pip,
                "OUTPUT_DIR": str((out_root / accession).resolve()),
                "VEP": str(vep),
                "VEP_CACHE": str(vep_cache),
                "VEP_FASTA": str(vep_fasta),
                "VEP_CACHE_RELEASE": cache_release,
                "FORCE": bool(args.force),
            }
        )

    excluded_file = out_root / "annotation_excluded_studies.tsv"
    if excluded:
        pd.DataFrame(excluded).to_csv(excluded_file, sep="\t", index=False)
    else:
        pd.DataFrame(columns=[
            "STUDY_ACCESSION", "STEP06_STATUS", "N_LOCI",
            "N_LOCI_SUCCESS", "N_LOCI_FAILED", "REASON",
        ]).to_csv(excluded_file, sep="\t", index=False)

    if not rows:
        raise RuntimeError(f"No Step 06 studies eligible for annotation. Inspect {excluded_file}")

    manifest = pd.DataFrame(rows)
    manifest_file = (out_root / "annotation_manifest.tsv").resolve()
    manifest.to_csv(manifest_file, sep="\t", index=False)

    script_path = Path(__file__).resolve()
    bash_file = root / f"Step07_Annotate_GWAS_{phenotype_slug}_{ancestry_slug}.sh"
    array_spec = f"1-{len(manifest)}%{args.max_parallel}"
    job_name = f"VEP_{phenotype_slug}_{ancestry_code.lower()}"[:100]

    bash_text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_root}/annotation.%A_%a.out
#SBATCH --error={log_root}/annotation.%A_%a.err
#SBATCH --array={array_spec}
#SBATCH --mem={args.memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1

set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_ANNOTATION_MANIFEST={shlex.quote(str(manifest_file))}

printf '%s\n' '=============================================================='
printf '%s\n' 'STEP 07 - VEP FUNCTIONAL ANNOTATION'
printf 'Job ID        : %s\n' \"$SLURM_JOB_ID\"
printf 'Array task ID : %s\n' \"$SLURM_ARRAY_TASK_ID\"
printf 'Hostname      : %s\n' \"$(hostname)\"
printf '%s\n' '=============================================================='

{shlex.quote(str(pipeline_python))} {shlex.quote(str(script_path))}
"""
    bash_file.write_text(bash_text, encoding="utf-8")
    bash_file.chmod(0o755)

    banner("STEP 07 - FUNCTIONAL ANNOTATION PLANNER")
    print(f"Phenotype         : {phenotype}")
    print(f"Ancestry          : {ancestry_label} ({ancestry_code})")
    print(f"Eligible studies  : {len(manifest)}")
    print(f"Excluded studies  : {len(excluded)}")
    print(f"Min PIP           : {args.min_pip}")
    print(f"VEP               : {vep}")
    print(f"VEP cache         : {vep_cache}")
    print(f"GRCh38 FASTA      : {vep_fasta}")
    print(f"Cache release     : {cache_release}")
    print(f"Array             : {array_spec}")
    print()
    print(manifest[["ANNOTATION_TASK_ID", "STUDY_ACCESSION", "N_LOCI_SUCCESS", "N_LOCI_FAILED"]].to_string(index=False))
    print()
    print(f"Manifest:\n  {manifest_file}")
    print(f"Excluded:\n  {excluded_file}")
    print(f"Generated SLURM:\n  {bash_file}")
    print()
    print(f"NEXT COMMAND:\n\n  sbatch {bash_file.name}\n")


def load_finemapped_variants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    required = ["LOCUS_ID", "CHR", "REFERENCE_POS", "REFERENCE_REF", "REFERENCE_ALT", "PIP"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise RuntimeError(f"Fine-mapped file missing columns {missing}: {path}")

    df["CHR"] = df["CHR"].astype(str).str.replace(r"^chr", "", regex=True, case=False)
    df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce")
    df["REFERENCE_POS"] = pd.to_numeric(df["REFERENCE_POS"], errors="coerce")
    df["PIP"] = pd.to_numeric(df["PIP"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df = df[df["CHR"].between(1, 22, inclusive="both")].copy()
    df["CHR"] = df["CHR"].astype(int)
    df["REFERENCE_POS"] = df["REFERENCE_POS"].astype(int)
    df["REFERENCE_REF"] = df["REFERENCE_REF"].astype(str).str.upper().str.strip()
    df["REFERENCE_ALT"] = df["REFERENCE_ALT"].astype(str).str.upper().str.strip()
    df = df[(df["REFERENCE_REF"] != "") & (df["REFERENCE_ALT"] != "")].copy()
    if df.empty:
        raise RuntimeError(f"No usable fine-mapped variants in {path}")
    return df


def select_variants(df: pd.DataFrame, min_pip: float) -> pd.DataFrame:
    if "IN_95_CREDIBLE_SET" in df.columns:
        credible = as_bool_series(df["IN_95_CREDIBLE_SET"])
    elif "CREDIBLE_SET" in df.columns:
        text = df["CREDIBLE_SET"].fillna("").astype(str).str.strip().str.lower()
        credible = ~text.isin({"", "na", "nan", "none", "."})
    else:
        credible = pd.Series(False, index=df.index)

    high_pip = df["PIP"] >= min_pip
    top_indices = df.dropna(subset=["PIP"]).groupby("LOCUS_ID")["PIP"].idxmax()
    top_variant = pd.Series(False, index=df.index)
    if len(top_indices):
        top_variant.loc[top_indices] = True

    selected = df.loc[credible | high_pip | top_variant].copy()
    selected["SELECTED_CREDIBLE_SET"] = credible.loc[selected.index].to_numpy()
    selected["SELECTED_PIP_THRESHOLD"] = high_pip.loc[selected.index].to_numpy()
    selected["SELECTED_TOP_LOCUS"] = top_variant.loc[selected.index].to_numpy()
    selected = selected.sort_values(
        ["CHR", "REFERENCE_POS", "PIP"],
        ascending=[True, True, False],
        kind="stable",
    )
    selected = selected.drop_duplicates(
        ["CHR", "REFERENCE_POS", "REFERENCE_REF", "REFERENCE_ALT"], keep="first"
    ).reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No variants selected for annotation")
    selected["VEP_ID"] = [f"GWASVEP_{i:08d}" for i in range(1, len(selected) + 1)]
    return selected


def create_vep_vcf(selected: pd.DataFrame, vcf_file: Path, mapping_file: Path) -> None:
    cols = [
        c for c in [
            "VEP_ID", "STUDY_ACCESSION", "PHENOTYPE", "ANCESTRY_CODE",
            "LOCUS_ID", "REFERENCE_ID", "CHR", "REFERENCE_POS",
            "REFERENCE_REF", "REFERENCE_ALT", "PIP", "IN_95_CREDIBLE_SET",
            "CREDIBLE_SET", "SELECTED_CREDIBLE_SET", "SELECTED_PIP_THRESHOLD",
            "SELECTED_TOP_LOCUS", "rsID", "SNPID", "P", "BETA", "SE", "Z", "Z_REF",
        ] if c in selected.columns
    ]
    selected[cols].to_csv(mapping_file, sep="\t", index=False)
    with open(vcf_file, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##reference=GRCh38\n")
        handle.write("##source=GWAS2m_Step07_VEP\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in selected.itertuples(index=False):
            handle.write(
                f"{int(row.CHR)}\t{int(row.REFERENCE_POS)}\t{row.VEP_ID}\t"
                f"{row.REFERENCE_REF}\t{row.REFERENCE_ALT}\t.\t.\t.\n"
            )


VEP_FIELDS = [
    "Uploaded_variation", "Location", "Allele", "Gene", "Feature", "Feature_type",
    "Consequence", "IMPACT", "SYMBOL", "SYMBOL_SOURCE", "HGNC_ID", "BIOTYPE",
    "EXON", "INTRON", "HGVSc", "HGVSp", "cDNA_position", "CDS_position",
    "Protein_position", "Amino_acids", "Codons", "Existing_variation", "DISTANCE",
    "STRAND", "FLAGS", "VARIANT_CLASS", "CANONICAL", "MANE_SELECT",
    "MANE_PLUS_CLINICAL", "TSL", "APPRIS", "DOMAINS", "SIFT", "PolyPhen", "PICK",
]


def run_vep(
    vep: Path,
    cache: Path,
    fasta: Path,
    input_vcf: Path,
    output_tsv: Path,
    log_file: Path,
    threads: int,
    force: bool,
) -> None:
    if not force and output_tsv.exists() and output_tsv.stat().st_size > 100:
        print(f"Reusing VEP output: {output_tsv}")
        return
    cmd = [
        vep,
        "--input_file", input_vcf,
        "--output_file", output_tsv,
        "--format", "vcf",
        "--tab",
        "--fields", ",".join(VEP_FIELDS),
        "--species", SPECIES,
        "--assembly", ASSEMBLY,
        "--cache",
        "--offline",
        "--dir_cache", cache,
        "--fasta", fasta,
        "--fork", threads,
        "--force_overwrite",
        "--symbol",
        "--numbers",
        "--hgvs",
        "--canonical",
        "--mane",
        "--appris",
        "--tsl",
        "--biotype",
        "--variant_class",
        "--domains",
        "--regulatory",
        "--sift", "b",
        "--polyphen", "b",
        "--flag_pick",
        "--no_stats",
    ]
    run_command(cmd, log_file)
    if not output_tsv.exists() or output_tsv.stat().st_size == 0:
        raise RuntimeError("VEP did not generate output")


def read_vep_output(path: Path) -> pd.DataFrame:
    header = None
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#Uploaded_variation"):
                header = line.rstrip("\n").lstrip("#").split("\t")
                continue
            if line.startswith("#"):
                continue
            if line.strip():
                lines.append(line)
    if header is None:
        raise RuntimeError("Could not find VEP #Uploaded_variation header")
    if not lines:
        raise RuntimeError("VEP output contains no annotation rows")
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", names=header, dtype=str, low_memory=False)
    if "Uploaded_variation" not in df.columns:
        raise RuntimeError("Uploaded_variation missing from VEP output")
    return df.rename(columns={"Uploaded_variation": "VEP_ID"})


def functional_category(row) -> str:
    consequence = str(row.get("Consequence", "")).lower()
    biotype = str(row.get("BIOTYPE", "")).lower()

    if any(x in consequence for x in [
        "splice_acceptor_variant", "splice_donor_variant", "splice_region_variant",
        "splice_donor_5th_base_variant", "splice_donor_region_variant",
        "splice_polypyrimidine_tract_variant",
    ]):
        return "SPLICING"

    if any(x in consequence for x in [
        "transcript_ablation", "stop_gained", "frameshift_variant", "stop_lost",
        "start_lost", "transcript_amplification", "inframe_insertion", "inframe_deletion",
        "missense_variant", "protein_altering_variant", "coding_sequence_variant",
        "synonymous_variant", "start_retained_variant", "stop_retained_variant",
        "incomplete_terminal_codon_variant",
    ]):
        return "CODING"

    if "5_prime_utr_variant" in consequence or "3_prime_utr_variant" in consequence:
        return "UTR"
    if "regulatory_region" in consequence or "tf_binding_site" in consequence:
        return "REGULATORY"
    if "non_coding_transcript" in consequence or any(x in biotype for x in [
        "lncrna", "lincrna", "mirna", "snrna", "snorna", "rrna", "ncrna",
        "antisense", "processed_transcript",
    ]):
        return "NONCODING_RNA"
    if "intron_variant" in consequence:
        return "INTRONIC"
    if "upstream_gene_variant" in consequence:
        return "UPSTREAM"
    if "downstream_gene_variant" in consequence:
        return "DOWNSTREAM"
    if "intergenic_variant" in consequence:
        return "INTERGENIC"
    return "OTHER"


CATEGORY_RANK = {
    "SPLICING": 0, "CODING": 1, "UTR": 2, "REGULATORY": 3,
    "NONCODING_RNA": 4, "INTRONIC": 5, "UPSTREAM": 6,
    "DOWNSTREAM": 7, "INTERGENIC": 8, "OTHER": 9,
}
IMPACT_RANK = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "MODIFIER": 3}


def merge_annotations(selected: pd.DataFrame, vep: pd.DataFrame, output: Path) -> pd.DataFrame:
    merged = selected.merge(vep, on="VEP_ID", how="left", validate="one_to_many")
    merged["FUNCTIONAL_CATEGORY"] = merged.apply(functional_category, axis=1)
    merged.to_csv(output, sep="\t", index=False, compression="gzip")
    return merged


def select_best_consequence(merged: pd.DataFrame) -> pd.DataFrame:
    x = merged.copy()
    impact = x["IMPACT"] if "IMPACT" in x.columns else pd.Series("", index=x.index)
    category = x["FUNCTIONAL_CATEGORY"]
    pick = x["PICK"] if "PICK" in x.columns else pd.Series("", index=x.index)
    mane = x["MANE_SELECT"] if "MANE_SELECT" in x.columns else pd.Series("", index=x.index)
    canonical = x["CANONICAL"] if "CANONICAL" in x.columns else pd.Series("", index=x.index)

    x["_IMPACT_RANK"] = impact.fillna("").astype(str).str.upper().map(IMPACT_RANK).fillna(9)
    x["_CATEGORY_RANK"] = category.map(CATEGORY_RANK).fillna(99)
    x["_PICK_RANK"] = np.where(pick.fillna("").astype(str).str.upper().isin({"1", "YES", "Y"}), 0, 1)
    x["_MANE_RANK"] = np.where(mane.fillna("").astype(str).str.strip() != "", 0, 1)
    x["_CANONICAL_RANK"] = np.where(canonical.fillna("").astype(str).str.upper().isin({"YES", "1"}), 0, 1)

    x = x.sort_values(
        ["VEP_ID", "_PICK_RANK", "_MANE_RANK", "_CANONICAL_RANK", "_IMPACT_RANK", "_CATEGORY_RANK"],
        kind="stable",
    )
    best = x.drop_duplicates("VEP_ID", keep="first").copy()
    return best.drop(columns=["_PICK_RANK", "_MANE_RANK", "_CANONICAL_RANK", "_IMPACT_RANK", "_CATEGORY_RANK"], errors="ignore")


def join_unique(series: pd.Series) -> str:
    seen = []
    for value in series.dropna().astype(str):
        for token in value.split(","):
            token = token.strip()
            if token and token not in seen:
                seen.append(token)
    return ";".join(seen)


def make_variant_summary(merged: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {"N_TRANSCRIPT_CONSEQUENCES": ("VEP_ID", "size")}
    for out, source in [
        ("ALL_CONSEQUENCES", "Consequence"),
        ("ALL_GENES", "Gene"),
        ("ALL_SYMBOLS", "SYMBOL"),
        ("ALL_BIOTYPES", "BIOTYPE"),
        ("ALL_FUNCTIONAL_CATEGORIES", "FUNCTIONAL_CATEGORY"),
    ]:
        if source in merged.columns:
            agg_spec[out] = (source, join_unique)
    grouped = merged.groupby("VEP_ID", sort=False, dropna=False).agg(**agg_spec).reset_index()
    return best.merge(grouped, on="VEP_ID", how="left", validate="one_to_one")


def create_summary_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    category = (
        summary.groupby("FUNCTIONAL_CATEGORY", dropna=False)
        .agg(N_VARIANTS=("VEP_ID", "nunique"), N_LOCI=("LOCUS_ID", "nunique"), MAX_PIP=("PIP", "max"))
        .reset_index()
        .sort_values(["N_VARIANTS", "MAX_PIP"], ascending=[False, False])
    )
    locus = (
        summary.groupby("LOCUS_ID", dropna=False)
        .agg(
            N_ANNOTATED_VARIANTS=("VEP_ID", "nunique"),
            MAX_PIP=("PIP", "max"),
            N_CREDIBLE_SELECTED=("SELECTED_CREDIBLE_SET", "sum"),
            N_SPLICING=("FUNCTIONAL_CATEGORY", lambda x: int((x == "SPLICING").sum())),
            N_CODING=("FUNCTIONAL_CATEGORY", lambda x: int((x == "CODING").sum())),
            N_REGULATORY=("FUNCTIONAL_CATEGORY", lambda x: int((x == "REGULATORY").sum())),
            N_NONCODING_RNA=("FUNCTIONAL_CATEGORY", lambda x: int((x == "NONCODING_RNA").sum())),
        )
        .reset_index()
    )
    return category, locus


def worker_mode() -> None:
    task_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    manifest_value = os.environ.get("GWAS_ANNOTATION_MANIFEST")
    if not task_value:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not defined")
    if not manifest_value:
        raise RuntimeError("GWAS_ANNOTATION_MANIFEST is not defined")

    task_id = int(task_value)
    manifest_file = Path(manifest_value).resolve()
    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str, low_memory=False)
    task_numbers = pd.to_numeric(manifest["ANNOTATION_TASK_ID"], errors="coerce")
    selected_row = manifest.loc[task_numbers == task_id]
    if len(selected_row) != 1:
        raise RuntimeError(f"Expected one manifest row for task {task_id}; found {len(selected_row)}")

    row = selected_row.iloc[0]
    accession = str(row["STUDY_ACCESSION"]).strip()
    phenotype = str(row["PHENOTYPE"]).strip()
    ancestry_code = str(row["ANCESTRY_CODE"]).strip().upper()
    ancestry_label = str(row["ANCESTRY_LABEL"]).strip()
    fine_file = Path(row["FINEMAPPED_VARIANTS_FILE"]).resolve()
    min_pip = float(row["MIN_PIP"])
    output_dir = Path(row["OUTPUT_DIR"]).resolve()
    vep = Path(row["VEP"]).resolve()
    cache = Path(row["VEP_CACHE"]).resolve()
    if "VEP_FASTA" not in manifest.columns or not str(row.get("VEP_FASTA", "")).strip():
        raise RuntimeError(
            "Annotation manifest is missing VEP_FASTA. Regenerate Step 07 by running "
            "the updated planner once without --index."
        )
    fasta = Path(row["VEP_FASTA"]).resolve()
    force = str(row.get("FORCE", "False")).strip().lower() in {"true", "1", "yes", "y"}
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_CPUS))
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_file = output_dir / f"{accession}_variants_selected_for_annotation.tsv"
    mapping_file = output_dir / f"{accession}_VEP_ID_mapping.tsv"
    input_vcf = output_dir / f"{accession}_variants_for_VEP.vcf"
    vep_output = output_dir / f"{accession}_VEP_raw.tsv"
    vep_log = output_dir / f"{accession}_VEP.log"
    all_output = output_dir / f"{accession}_VEP_all_consequences.tsv.gz"
    best_output = output_dir / f"{accession}_VEP_best_consequence.tsv"
    variant_output = output_dir / f"{accession}_VEP_variant_summary.tsv"
    category_output = output_dir / f"{accession}_VEP_category_summary.tsv"
    locus_output = output_dir / f"{accession}_VEP_locus_summary.tsv"
    summary_json = output_dir / "annotation_summary.json"
    failed_file = output_dir / "ANNOTATION_FAILED.txt"

    if not force and variant_output.exists() and variant_output.stat().st_size > 0 and summary_json.exists() and not failed_file.exists():
        banner("ANNOTATION ALREADY COMPLETE")
        print(accession)
        return

    banner("STEP 07 - VEP FUNCTIONAL ANNOTATION WORKER")
    print(f"Task       : {task_id}")
    print(f"Study      : {accession}")
    print(f"Phenotype  : {phenotype}")
    print(f"Ancestry   : {ancestry_label} ({ancestry_code})")
    print(f"Fine-map   : {fine_file}")
    print(f"Min PIP    : {min_pip}")
    print(f"VEP        : {vep}")
    print(f"VEP cache  : {cache}")
    print(f"FASTA      : {fasta}")
    print(f"Threads    : {threads}")

    try:
        if not vep.exists() or not os.access(vep, os.X_OK):
            raise FileNotFoundError(f"VEP executable unavailable: {vep}")
        if not cache_is_valid(cache):
            raise FileNotFoundError(f"Human GRCh38 VEP cache unavailable: {cache}")
        if not fasta.exists() or fasta.stat().st_size == 0:
            raise FileNotFoundError(f"GRCh38 FASTA unavailable: {fasta}")
        fai = Path(str(fasta) + ".fai")
        if not fai.exists() or fai.stat().st_size == 0:
            raise FileNotFoundError(
                f"GRCh38 FASTA index unavailable: {fai}. "
                "Run samtools faidx or rerun Setup_02_Download_Genome_GENCODE.sh."
            )
        if not fine_file.exists() or fine_file.stat().st_size == 0:
            raise FileNotFoundError(f"Fine-mapped file missing/empty: {fine_file}")

        fine = load_finemapped_variants(fine_file)
        fine["STUDY_ACCESSION"] = accession
        fine["PHENOTYPE"] = phenotype
        fine["ANCESTRY_CODE"] = ancestry_code

        selected = select_variants(fine, min_pip)
        selected.to_csv(selected_file, sep="\t", index=False)
        create_vep_vcf(selected, input_vcf, mapping_file)

        banner("SELECTED VARIANTS")
        print(f"Fine-mapped rows       : {len(fine):,}")
        print(f"Selected variants      : {len(selected):,}")
        print(f"Credible-set selected  : {int(selected['SELECTED_CREDIBLE_SET'].sum()):,}")
        print(f"PIP-threshold selected : {int(selected['SELECTED_PIP_THRESHOLD'].sum()):,}")
        print(f"Top-locus selected     : {int(selected['SELECTED_TOP_LOCUS'].sum()):,}")
        print(f"Loci represented       : {selected['LOCUS_ID'].nunique():,}")

        banner("RUNNING VEP")
        run_vep(vep, cache, fasta, input_vcf, vep_output, vep_log, threads, force)
        vep_df = read_vep_output(vep_output)
        merged = merge_annotations(selected, vep_df, all_output)
        best = select_best_consequence(merged)
        best.to_csv(best_output, sep="\t", index=False)
        variant_summary = make_variant_summary(merged, best)
        variant_summary.to_csv(variant_output, sep="\t", index=False)
        category_summary, locus_summary = create_summary_tables(variant_summary)
        category_summary.to_csv(category_output, sep="\t", index=False)
        locus_summary.to_csv(locus_output, sep="\t", index=False)

        n_annotated = int(vep_df["VEP_ID"].nunique())
        n_splicing = int((variant_summary["FUNCTIONAL_CATEGORY"] == "SPLICING").sum())
        n_coding = int((variant_summary["FUNCTIONAL_CATEGORY"] == "CODING").sum())
        n_regulatory = int((variant_summary["FUNCTIONAL_CATEGORY"] == "REGULATORY").sum())
        n_ncrna = int((variant_summary["FUNCTIONAL_CATEGORY"] == "NONCODING_RNA").sum())

        result = {
            "ANNOTATION_TASK_ID": task_id,
            "STUDY_ACCESSION": accession,
            "PHENOTYPE": phenotype,
            "ANCESTRY_CODE": ancestry_code,
            "ANCESTRY_LABEL": ancestry_label,
            "BUILD": ASSEMBLY,
            "MIN_PIP": min_pip,
            "N_FINEMAPPED_ROWS_AVAILABLE": int(len(fine)),
            "N_FINEMAPPED_LOCI": int(fine["LOCUS_ID"].nunique()),
            "N_SELECTED_VARIANTS": int(len(selected)),
            "N_SELECTED_CREDIBLE_SET": int(selected["SELECTED_CREDIBLE_SET"].sum()),
            "N_SELECTED_PIP_THRESHOLD": int(selected["SELECTED_PIP_THRESHOLD"].sum()),
            "N_SELECTED_TOP_LOCUS": int(selected["SELECTED_TOP_LOCUS"].sum()),
            "N_VEP_CONSEQUENCE_ROWS": int(len(vep_df)),
            "N_ANNOTATED_VARIANTS": n_annotated,
            "N_UNANNOTATED_VARIANTS": int(len(selected) - n_annotated),
            "N_SPLICING": n_splicing,
            "N_CODING": n_coding,
            "N_REGULATORY": n_regulatory,
            "N_NONCODING_RNA": n_ncrna,
            "STEP07_VERSION": STEP07_VERSION,
            "VEP_EXECUTABLE": str(vep),
            "VEP_CACHE": str(cache),
            "VEP_FASTA": str(fasta),
            "SELECTED_VARIANTS_FILE": str(selected_file),
            "VEP_ID_MAPPING_FILE": str(mapping_file),
            "VEP_INPUT_VCF": str(input_vcf),
            "VEP_RAW_OUTPUT": str(vep_output),
            "VEP_ALL_CONSEQUENCES": str(all_output),
            "VEP_BEST_CONSEQUENCE": str(best_output),
            "VEP_VARIANT_SUMMARY": str(variant_output),
            "VEP_CATEGORY_SUMMARY": str(category_output),
            "VEP_LOCUS_SUMMARY": str(locus_output),
            "STATUS": "COMPLETE",
            "COMPLETED_UTC": datetime.now(timezone.utc).isoformat(),
        }
        summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if failed_file.exists():
            failed_file.unlink()

        banner("STEP 07 ANNOTATION COMPLETE")
        print(f"Study                : {accession}")
        print(f"Selected variants    : {len(selected):,}")
        print(f"Annotated variants   : {n_annotated:,}")
        print(f"Splicing             : {n_splicing:,}")
        print(f"Coding               : {n_coding:,}")
        print(f"Regulatory           : {n_regulatory:,}")
        print(f"Noncoding RNA        : {n_ncrna:,}")
        print(f"Main table           : {variant_output}")

    except Exception as error:
        text = (
            "STEP 07 FUNCTIONAL ANNOTATION FAILED\n\n"
            f"Task ID: {task_id}\n"
            f"Accession: {accession}\n"
            f"Phenotype: {phenotype}\n"
            f"Ancestry: {ancestry_label} ({ancestry_code})\n\n"
            f"Error:\n{type(error).__name__}: {error}\n\n"
            f"{traceback.format_exc()}"
        )
        failed_file.write_text(text, encoding="utf-8")
        print(text, file=sys.stderr)
        raise


def direct_index_mode(args) -> None:
    validate_arguments(args)
    root = Path.cwd().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)
    manifest_file = (
        root / "07_annotation" / phenotype_slug / ancestry_slug / "annotation_manifest.tsv"
    ).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Annotation manifest not found:\n  {manifest_file}\n\n"
            "Run Step 07 once without --index first."
        )
    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str, low_memory=False)
    ids = pd.to_numeric(manifest["ANNOTATION_TASK_ID"], errors="coerce")
    selected = manifest.loc[ids == args.index]
    if len(selected) != 1:
        available = [int(x) for x in ids.dropna()]
        raise RuntimeError(f"Index {args.index} not found exactly once. Available: {available}")

    row = selected.iloc[0]
    banner("STEP 07 - DIRECT SINGLE-INDEX MODE")
    print(f"Manifest : {manifest_file}")
    print(f"Index    : {args.index}")
    print(f"Study    : {row['STUDY_ACCESSION']}")
    print()
    print("NOTE: direct --index mode runs on the current machine/node. Use srun on HPC for substantial work.")

    old_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    old_manifest = os.environ.get("GWAS_ANNOTATION_MANIFEST")
    old_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        os.environ["SLURM_ARRAY_TASK_ID"] = str(args.index)
        os.environ["GWAS_ANNOTATION_MANIFEST"] = str(manifest_file)
        os.environ["SLURM_CPUS_PER_TASK"] = str(args.cpus)
        worker_mode()
    finally:
        if old_task is None:
            os.environ.pop("SLURM_ARRAY_TASK_ID", None)
        else:
            os.environ["SLURM_ARRAY_TASK_ID"] = old_task
        if old_manifest is None:
            os.environ.pop("GWAS_ANNOTATION_MANIFEST", None)
        else:
            os.environ["GWAS_ANNOTATION_MANIFEST"] = old_manifest
        if old_cpus is None:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
        else:
            os.environ["SLURM_CPUS_PER_TASK"] = old_cpus


def main() -> None:
    if os.environ.get("SLURM_ARRAY_TASK_ID") and os.environ.get("GWAS_ANNOTATION_MANIFEST"):
        worker_mode()
        return
    args = arguments()
    if args.index is not None:
        direct_index_mode(args)
    else:
        planner_mode(args)


if __name__ == "__main__":
    main()

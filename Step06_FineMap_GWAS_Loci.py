#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic GWAS Pipeline
STEP 06 - Regional extraction + ancestry-matched LD + SuSiE-RSS fine-mapping

PLANNER MODE
------------
Run from the GWAS2m repository root:

    envs/pipeline/bin/python Step06_FineMap_GWAS_Loci.py \
        --phenotype migraine \
        --ancestry EUR

The planner:
  1. Resolves the phenotype/ancestry project deterministically.
  2. Verifies Step 05 clumping outputs for every expected GWAS.
  3. Fetches and caches metadata for every GCST accession from:
       - GWAS Catalog REST API v2 study endpoint
       - GWAS Catalog REST API v2 ancestry endpoint
       - official summary-statistics *-meta.yaml sidecar when available
  4. Resolves sample size for SuSiE-RSS. For case/control studies it prefers
     effective sample size: 4 / (1/Ncase + 1/Ncontrol).
  5. Counts the ancestry-specific 1000G LD-reference samples from .psam.
  6. Creates a fine-mapping manifest.
  7. Generates a phenotype/ancestry-specific SLURM array script.

Then submit the generated script, e.g.:

    sbatch Step06_FineMap_GWAS_migraine_european.sh

WORKER MODE
-----------
The generated SLURM array sets SLURM_ARRAY_TASK_ID and GWAS_FINEMAP_MANIFEST.
Each task processes exactly ONE GWAS study, never merging studies.

For each GWAS:
  1. Reads Step 05 independent lead variants.
  2. Expands each lead by --locus-window-kb (default 1000 kb) and merges
     overlapping windows into physical loci.
  3. Streams the full Step 03 cleaned GWAS once and extracts all regional
     variants for those loci.
  4. Matches each regional GWAS to the ancestry-matched 1000G GRCh38 panel.
  5. Harmonises effect direction to the PLINK reference allele.
  6. Builds a signed LD matrix with PLINK2 for each locus.
  7. Runs susieR::susie_rss using the resolved GWAS sample size.
  8. Produces PIP values, 95% credible sets, diagnostics, summaries, and
     failure reports.

OUTPUT LAYOUT
-------------
06_finemapping/
    <phenotype>/
        <ancestry>/
            finemapping_manifest.tsv
            gwas_metadata_summary.tsv
            metadata/
                <GCST>/
                    study_v2.json
                    ancestries_v2.json
                    summary_statistics_metadata.yaml
                    resolved_metadata.json
            <GCST>/
                loci_definition.tsv
                regional_gwas/
                    L001.tsv.gz
                    ...
                loci/
                    L001/
                        reference_region.pvar
                        matched_variants.tsv
                        matched_reference_ids.txt
                        <ANC>_LD.unphased.vcor1.bin
                        <ANC>_LD.unphased.vcor1.bin.vars
                        susie/
                            susie_variants.tsv
                            credible_sets.tsv
                            susie_summary.tsv
                            susie_diagnostics.tsv
                            susie_fit.rds
                            susie_rss.log
                <GCST>_finemapped_variants.tsv.gz
                <GCST>_95pct_credible_sets.tsv
                <GCST>_finemapping_summary.tsv
                <GCST>_failed_loci.tsv
                finemapping_summary.json

NOTES
-----
- Shared references under resources/1000G/<ANC>/ are reused; nothing is
  re-downloaded here.
- Default SLURM partition: general.
- Walltime is capped at 24 hours.
- Planner is strict by default: Step 05 must be complete for all expected
  studies. Use --allow-incomplete only for exploratory/testing runs.
- Effect statistics are accepted as Z, BETA+SE, OR+SE, or OR+95% CI.
  For OR-based studies, BETA is reconstructed as log(OR); when SE is absent
  but OR_95U/OR_95L are present, SE is reconstructed from the 95% CI.
- LD size protection is applied AFTER 1000G reference MAF/GENO filtering.
  If a locus is still larger than --max-ld-variants, a deterministic balanced
  subset nearest the independent lead signals is used, with sidecar audit files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_PARTITION = "general"
DEFAULT_TIME = "24:00:00"
DEFAULT_MEMORY = "64G"
DEFAULT_CPUS = 8
DEFAULT_MAX_PARALLEL = 2
DEFAULT_LOCUS_WINDOW_KB = 1000
DEFAULT_REFERENCE_MAF = 0.01
DEFAULT_REFERENCE_GENO = 0.05
DEFAULT_MIN_VARIANTS = 10
DEFAULT_MAX_LD_VARIANTS = 15000
DEFAULT_COVERAGE = 0.95
DEFAULT_SUSIE_L = 10
DEFAULT_MAX_SUSIE_L = 20
DEFAULT_LD_RIDGE = 1e-4
DEFAULT_CHUNK_SIZE = 500_000
DEFAULT_ESTIMATE_S_MAX = 5000

GWAS_API_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"

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
# GENERIC HELPERS
# =============================================================================


def banner(text: str) -> None:
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def normalize_text(value: Any) -> str:
    value = "" if value is None else str(value)
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def resolve_ancestry(value: str) -> tuple[str, str]:
    key = normalize_text(value)
    if key not in ANCESTRY_ALIASES:
        allowed = ", ".join(sorted({x[0] for x in ANCESTRY_ALIASES.values()}))
        raise ValueError(f"Unsupported ancestry '{value}'. Supported codes: {allowed}")
    return ANCESTRY_ALIASES[key]


def parse_time_seconds(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Time must be HH:MM:SS")
    try:
        hours, minutes, seconds = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Time must be HH:MM:SS") from exc
    if hours < 0 or minutes not in range(60) or seconds not in range(60):
        raise argparse.ArgumentTypeError("Invalid walltime")
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise argparse.ArgumentTypeError("Walltime must be positive")
    if total > 24 * 3600:
        raise argparse.ArgumentTypeError("Maximum allowed walltime is 24 hours")
    return total


def clean_chr(value: Any) -> int | None:
    text = str(value).strip().lower().replace("chr", "")
    try:
        chrom = int(float(text))
    except Exception:
        return None
    return chrom if 1 <= chrom <= 22 else None


def clean_allele(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    if not value or value in {"NAN", "NA", ".", "<NA>"}:
        return None
    return value


def clean_rsid(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if re.fullmatch(r"rs\d+", value, flags=re.I) else None


def allele_key(a: Any, b: Any) -> tuple[str, str] | None:
    a = clean_allele(a)
    b = clean_allele(b)
    if a is None or b is None:
        return None
    return tuple(sorted((a, b)))


def find_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) > 0 else None
    if isinstance(value, float):
        return int(round(value)) if np.isfinite(value) and value > 0 else None
    text = str(value).strip().replace(",", "")
    match = re.search(r"\d+", text)
    if not match:
        return None
    out = int(match.group())
    return out if out > 0 else None


def parse_numbers(text: Any) -> list[int]:
    if text is None or pd.isna(text):
        return []
    values = []
    for token in re.findall(r"\d[\d,]*", str(text)):
        try:
            number = int(token.replace(",", ""))
            if number >= 100:
                values.append(number)
        except ValueError:
            pass
    return values


def effective_n(n_cases: int | None, n_controls: int | None) -> int | None:
    if not n_cases or not n_controls or n_cases <= 0 or n_controls <= 0:
        return None
    return int(round(4.0 / ((1.0 / n_cases) + (1.0 / n_controls))))


def run_command(command, log_file: Path | None = None) -> str:
    print()
    print(" ".join(shlex.quote(str(x)) for x in command), flush=True)
    process = subprocess.Popen(
        [str(x) for x in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured = []
    handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = log_file.open("w", encoding="utf-8")
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
    code = process.wait()
    output = "".join(captured)
    if code != 0:
        raise RuntimeError(
            f"Command failed with exit code {code}:\n"
            + " ".join(str(x) for x in command)
            + "\n\n"
            + output[-5000:]
        )
    return output


# =============================================================================
# ARGUMENTS
# =============================================================================


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--ancestry", required=True)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--time", default=DEFAULT_TIME)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--locus-window-kb", type=int, default=DEFAULT_LOCUS_WINDOW_KB)
    parser.add_argument("--reference-maf", type=float, default=DEFAULT_REFERENCE_MAF)
    parser.add_argument("--reference-geno", type=float, default=DEFAULT_REFERENCE_GENO)
    parser.add_argument("--min-variants", type=int, default=DEFAULT_MIN_VARIANTS)
    parser.add_argument("--max-ld-variants", type=int, default=DEFAULT_MAX_LD_VARIANTS)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--susie-L", type=int, default=DEFAULT_SUSIE_L, dest="susie_L")
    parser.add_argument("--max-susie-L", type=int, default=DEFAULT_MAX_SUSIE_L, dest="max_susie_L")
    parser.add_argument("--ld-ridge", type=float, default=DEFAULT_LD_RIDGE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Run exactly one fine-mapping manifest task directly in the current "
            "process. Example: --index 1. Without --index, planner mode generates "
            "the SLURM array script as usual."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Last-resort sample-size fallback if official metadata cannot resolve N.",
    )
    args = parser.parse_args()
    parse_time_seconds(args.time)
    if args.cpus < 1 or args.max_parallel < 1:
        parser.error("--cpus and --max-parallel must be >= 1")
    if args.locus_window_kb < 1:
        parser.error("--locus-window-kb must be >= 1")
    if not (0 < args.reference_maf < 0.5):
        parser.error("--reference-maf must be between 0 and 0.5")
    if not (0 <= args.reference_geno < 1):
        parser.error("--reference-geno must be in [0,1)")
    if not (0 < args.coverage < 1):
        parser.error("--coverage must be between 0 and 1")
    if args.min_variants < 2 or args.max_ld_variants < args.min_variants:
        parser.error("Invalid fine-mapping variant limits")
    if args.susie_L < 1 or args.max_susie_L < args.susie_L:
        parser.error("Invalid SuSiE L settings")
    if args.index is not None and args.index < 1:
        parser.error("--index must be >= 1")
    return args


# =============================================================================
# HTTP / OFFICIAL METADATA
# =============================================================================


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "GWAS2m-Finemapping/1.0"})
    return session


def fetch_json(session, url: str, destination: Path, refresh: bool):
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        try:
            return json.loads(destination.read_text(encoding="utf-8"))
        except Exception:
            pass
    response = session.get(url, timeout=(20, 120))
    response.raise_for_status()
    payload = response.json()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    time.sleep(0.08)  # stay comfortably below documented API throttle
    return payload


def fetch_text(session, urls: list[str], destination: Path, refresh: bool):
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return destination.read_text(encoding="utf-8", errors="replace"), "cache"
    last_error = None
    for url in urls:
        try:
            response = session.get(url, timeout=(20, 120))
            if response.status_code == 200 and response.text.strip():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(response.text, encoding="utf-8")
                return response.text, url
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = repr(exc)
    return None, last_error


def parse_simple_yaml(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        out = {}
        for line in text.splitlines():
            if not line or line.lstrip().startswith("#") or ":" not in line:
                continue
            if line.startswith((" ", "\t")):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            low = value.lower()
            if low in {"true", "false"}:
                out[key] = low == "true"
            else:
                integer = safe_int(value)
                out[key] = integer if integer is not None and re.fullmatch(r"[\d,]+", value) else value
        return out


def recursive_objects(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from recursive_objects(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from recursive_objects(value)


def recursive_values(obj, key_fragments: list[str]):
    wanted = [normalize_text(x).replace(" ", "") for x in key_fragments]
    for item in recursive_objects(obj):
        for key, value in item.items():
            norm = normalize_text(key).replace(" ", "")
            if any(fragment in norm for fragment in wanted):
                yield value


def ancestry_sample_size_from_api(payload: Any) -> int | None:
    numbers = []
    for item in recursive_objects(payload):
        stage_text = " ".join(
            str(v) for k, v in item.items() if "stage" in normalize_text(k)
        ).lower()
        # Prefer discovery/initial stage when a stage is explicitly given.
        if stage_text and not ("initial" in stage_text or "discovery" in stage_text):
            continue
        for key, value in item.items():
            norm = normalize_text(key).replace(" ", "")
            if "numberofindividuals" in norm or norm in {
                "samplesize", "samplecount", "numberindividuals", "n"
            }:
                number = safe_int(value)
                if number and number >= 100:
                    numbers.append(number)
    return sum(numbers) if numbers else None


def study_sample_size_from_api(payload: Any) -> int | None:
    candidates = []
    for value in recursive_values(
        payload,
        ["initial_sample_size", "initial sample size", "discovery_sample", "sample_size"],
    ):
        candidates.extend(parse_numbers(value))
        direct = safe_int(value)
        if direct and direct >= 100:
            candidates.append(direct)
    return max(candidates) if candidates else None


def metadata_yaml_urls(download_url: str | None) -> list[str]:
    if not download_url:
        return []
    urls = [download_url + "-meta.yaml"]
    # Some directories may expose a metadata file for the underlying submitted
    # filename rather than the harmonised filename. The directory-list fallback
    # below handles this when possible.
    return list(dict.fromkeys(urls))


def find_yaml_from_directory(session, download_url: str, accession: str) -> str | None:
    directory = download_url.rsplit("/", 1)[0] + "/"
    try:
        response = session.get(directory, timeout=(20, 60))
        if response.status_code != 200:
            return None
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', response.text, flags=re.I)
        candidates = []
        for href in hrefs:
            name = href.split("/")[-1]
            low = name.lower()
            if low.endswith("-meta.yaml") and accession.lower() in low:
                candidates.append(urljoin(directory, href))
        return sorted(candidates)[0] if candidates else None
    except Exception:
        return None


def resolve_official_metadata(
    session: requests.Session,
    accession: str,
    download_url: str | None,
    initial_sample_text: str | None,
    ranking_n: int | None,
    metadata_dir: Path,
    refresh: bool,
    user_fallback: int | None,
) -> dict[str, Any]:
    metadata_dir.mkdir(parents=True, exist_ok=True)

    study_url = f"{GWAS_API_BASE}/studies/{accession}"
    ancestries_url = f"{GWAS_API_BASE}/studies/{accession}/ancestries"

    study_payload = None
    ancestry_payload = None
    study_error = None
    ancestry_error = None

    try:
        study_payload = fetch_json(
            session, study_url, metadata_dir / "study_v2.json", refresh
        )
    except Exception as exc:
        study_error = repr(exc)
        # v2 also supports filtered resource queries; use as fallback.
        try:
            study_payload = fetch_json(
                session,
                f"{GWAS_API_BASE}/studies?accession_id={accession}",
                metadata_dir / "study_v2.json",
                refresh,
            )
            study_error = None
        except Exception as exc2:
            study_error = repr(exc2)

    try:
        ancestry_payload = fetch_json(
            session, ancestries_url, metadata_dir / "ancestries_v2.json", refresh
        )
    except Exception as exc:
        ancestry_error = repr(exc)

    yaml_text = None
    yaml_source = None
    yaml_path = metadata_dir / "summary_statistics_metadata.yaml"
    if download_url:
        yaml_text, yaml_source = fetch_text(
            session,
            metadata_yaml_urls(download_url),
            yaml_path,
            refresh,
        )
        if yaml_text is None:
            extra_url = find_yaml_from_directory(session, download_url, accession)
            if extra_url:
                yaml_text, yaml_source = fetch_text(
                    session, [extra_url], yaml_path, refresh
                )

    yaml_meta = parse_simple_yaml(yaml_text)

    n_cases = safe_int(yaml_meta.get("case_count"))
    n_controls = safe_int(yaml_meta.get("control_count"))
    yaml_n = safe_int(yaml_meta.get("sample_size"))
    is_case_control = yaml_meta.get("case_control_study")
    neff = effective_n(n_cases, n_controls)

    api_ancestry_n = ancestry_sample_size_from_api(ancestry_payload)
    api_study_n = study_sample_size_from_api(study_payload)

    catalog_numbers = parse_numbers(initial_sample_text)
    catalog_n = sum(catalog_numbers) if catalog_numbers else None

    if neff:
        n_gwas = neff
        n_source = "GWAS_CATALOG_SUMSTATS_YAML_CASE_CONTROL_EFFECTIVE_N"
    elif yaml_n:
        n_gwas = yaml_n
        n_source = "GWAS_CATALOG_SUMSTATS_YAML_SAMPLE_SIZE"
    elif api_ancestry_n:
        n_gwas = api_ancestry_n
        n_source = "GWAS_CATALOG_API_V2_ANCESTRY"
    elif api_study_n:
        n_gwas = api_study_n
        n_source = "GWAS_CATALOG_API_V2_STUDY"
    elif ranking_n:
        n_gwas = ranking_n
        n_source = "STEP01_RANKING_N"
    elif catalog_n:
        n_gwas = catalog_n
        n_source = "STEP01_INITIAL_SAMPLE_SIZE_PARSED"
    elif user_fallback:
        n_gwas = user_fallback
        n_source = "USER_FALLBACK"
    else:
        n_gwas = None
        n_source = "UNRESOLVED"

    result = {
        "STUDY_ACCESSION": accession,
        "GWAS_API_STUDY_URL": study_url,
        "GWAS_API_ANCESTRIES_URL": ancestries_url,
        "GWAS_API_STUDY_ERROR": study_error,
        "GWAS_API_ANCESTRIES_ERROR": ancestry_error,
        "SUMMARY_METADATA_SOURCE": yaml_source,
        "CASE_CONTROL_STUDY": is_case_control,
        "N_CASES": n_cases,
        "N_CONTROLS": n_controls,
        "N_EFFECTIVE_CASE_CONTROL": neff,
        "YAML_SAMPLE_SIZE": yaml_n,
        "API_ANCESTRY_SAMPLE_SIZE": api_ancestry_n,
        "API_STUDY_SAMPLE_SIZE": api_study_n,
        "STEP01_RANKING_N": ranking_n,
        "STEP01_INITIAL_SAMPLE_SIZE": initial_sample_text,
        "RESOLVED_GWAS_N": n_gwas,
        "RESOLVED_GWAS_N_SOURCE": n_source,
        "METADATA_RESOLVED_UTC": datetime.now(timezone.utc).isoformat(),
    }
    (metadata_dir / "resolved_metadata.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


# =============================================================================
# REFERENCE HELPERS
# =============================================================================


def reference_prefix(root: Path, ancestry_code: str, chromosome: int) -> Path:
    return root / "resources" / "1000G" / ancestry_code / f"chr{chromosome}_{ancestry_code}_GRCh38"


def verify_reference(root: Path, ancestry_code: str) -> None:
    missing = []
    for chrom in range(1, 23):
        prefix = reference_prefix(root, ancestry_code, chrom)
        for ext in (".pgen", ".pvar", ".psam"):
            path = Path(str(prefix) + ext)
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Ancestry reference {ancestry_code} is incomplete. Missing {len(missing)} files.\n"
            + "\n".join(missing[:20])
        )


def count_reference_samples(root: Path, ancestry_code: str) -> int:
    psam = Path(str(reference_prefix(root, ancestry_code, 1)) + ".psam")
    count = 0
    with psam.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            count += 1
    if count <= 0:
        raise RuntimeError(f"Could not count samples in {psam}")
    return count


# =============================================================================
# PLANNER
# =============================================================================


def planner_mode(args) -> None:
    root = Path.cwd().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = resolve_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)

    pipeline_python = root / "envs" / "pipeline" / "bin" / "python"
    plink2 = root / "envs" / "pipeline" / "bin" / "plink2"
    rscript = root / "envs" / "pipeline" / "bin" / "Rscript"
    if not rscript.exists():
        located = shutil.which("Rscript")
        rscript = Path(located).resolve() if located else rscript

    for label, path in [
        ("pipeline Python", pipeline_python),
        ("PLINK2", plink2),
        ("Rscript", rscript),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required {label} not found: {path}")

    susie_check = subprocess.run(
        [str(rscript), "-e", 'quit(status=ifelse(requireNamespace("susieR", quietly=TRUE),0,1))'],
        capture_output=True,
        text=True,
    )
    if susie_check.returncode != 0:
        raise RuntimeError("R package susieR is not available in the configured R environment")

    verify_reference(root, ancestry_code)
    reference_n = count_reference_samples(root, ancestry_code)

    clump_root = root / "05_ld_clumping" / phenotype_slug / ancestry_slug
    clump_manifest_file = clump_root / "clumping_manifest.tsv"
    if not clump_manifest_file.exists():
        raise FileNotFoundError(
            f"Step 05 manifest not found:\n{clump_manifest_file}\nRun Step 05 first."
        )
    clump_manifest = pd.read_csv(clump_manifest_file, sep="\t", dtype=str)

    step01_file = (
        root / "01_gwas_catalog" / phenotype_slug / ancestry_slug / "GWAS_download_manifest.tsv"
    )
    step01 = (
        pd.read_csv(step01_file, sep="\t", dtype=str)
        if step01_file.exists()
        else pd.DataFrame()
    )
    step01_lookup = {}
    if not step01.empty and "STUDY_ACCESSION" in step01.columns:
        step01_lookup = {
            str(row["STUDY_ACCESSION"]).strip(): row
            for _, row in step01.iterrows()
        }

    output_root = root / "06_finemapping" / phenotype_slug / ancestry_slug
    metadata_root = output_root / "metadata"
    log_root = root / "logs" / "step06_finemapping" / phenotype_slug / ancestry_slug
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    session = make_session()
    rows = []
    metadata_rows = []
    incomplete = []

    banner("STEP 06 - FINEMAPPING PLANNER")
    print(f"Phenotype    : {phenotype}")
    print(f"Ancestry     : {ancestry_label} ({ancestry_code})")
    print(f"Reference N  : {reference_n}")
    print(f"Step 05 rows : {len(clump_manifest)}")

    for _, crow in clump_manifest.iterrows():
        accession = str(crow["STUDY_ACCESSION"]).strip()
        study_clump_dir = clump_root / accession
        lead_file = study_clump_dir / "lead_variants.tsv"
        clump_summary = study_clump_dir / "clumping_summary.tsv"
        qc_file = Path(str(crow["QC_FILE"])).resolve()

        missing = [
            str(p) for p in (lead_file, clump_summary, qc_file)
            if not p.exists() or p.stat().st_size == 0
        ]
        if missing:
            incomplete.append((accession, missing))
            continue

        srow = step01_lookup.get(accession)
        download_url = str(srow.get("DOWNLOAD_URL", "")).strip() if srow is not None else ""
        initial_sample = str(srow.get("INITIAL_SAMPLE_SIZE", "")).strip() if srow is not None else ""
        ranking_n = safe_int(srow.get("RANKING_N")) if srow is not None else None

        meta = resolve_official_metadata(
            session=session,
            accession=accession,
            download_url=download_url or None,
            initial_sample_text=initial_sample or None,
            ranking_n=ranking_n,
            metadata_dir=metadata_root / accession,
            refresh=args.refresh_metadata,
            user_fallback=args.sample_size,
        )
        metadata_rows.append(meta)

        if not meta["RESOLVED_GWAS_N"]:
            incomplete.append((accession, ["GWAS sample size could not be resolved"]))
            continue

        study_output = output_root / accession
        rows.append(
            {
                "FINEMAP_TASK_ID": len(rows) + 1,
                "STUDY_ACCESSION": accession,
                "PHENOTYPE": phenotype,
                "ANCESTRY": ancestry_label,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
                "QC_FILE": str(qc_file),
                "LEAD_VARIANTS_FILE": str(lead_file.resolve()),
                "CLUMPING_SUMMARY_FILE": str(clump_summary.resolve()),
                "OUTPUT_DIR": str(study_output.resolve()),
                "GWAS_N": int(meta["RESOLVED_GWAS_N"]),
                "GWAS_N_SOURCE": meta["RESOLVED_GWAS_N_SOURCE"],
                "N_CASES": meta.get("N_CASES") or "",
                "N_CONTROLS": meta.get("N_CONTROLS") or "",
                "REFERENCE_N": reference_n,
                "LOCUS_WINDOW_KB": args.locus_window_kb,
                "REFERENCE_MAF": args.reference_maf,
                "REFERENCE_GENO": args.reference_geno,
                "MIN_VARIANTS": args.min_variants,
                "MAX_LD_VARIANTS": args.max_ld_variants,
                "COVERAGE": args.coverage,
                "SUSIE_L": args.susie_L,
                "MAX_SUSIE_L": args.max_susie_L,
                "LD_RIDGE": args.ld_ridge,
                "CHUNK_SIZE": args.chunk_size,
                "R_SCRIPT": str(rscript),
                "PLINK2": str(plink2),
            }
        )

    if metadata_rows:
        pd.DataFrame(metadata_rows).to_csv(
            output_root / "gwas_metadata_summary.tsv", sep="\t", index=False
        )

    if incomplete:
        report = output_root / "finemapping_incomplete_inputs.tsv"
        pd.DataFrame(
            [
                {"STUDY_ACCESSION": acc, "PROBLEM": " | ".join(problems)}
                for acc, problems in incomplete
            ]
        ).to_csv(report, sep="\t", index=False)
        print()
        print(f"Incomplete studies: {len(incomplete)}")
        for acc, problems in incomplete:
            print(f"  {acc}: {' | '.join(problems)}")
        print(f"Report: {report}")
        if not args.allow_incomplete:
            raise RuntimeError(
                "Step 05 / metadata is not complete for all expected studies. "
                "Finish Step 05 and rerun. Use --allow-incomplete only for testing."
            )

    if not rows:
        raise RuntimeError("No studies are ready for fine-mapping")

    manifest = pd.DataFrame(rows)
    manifest_file = (output_root / "finemapping_manifest.tsv").resolve()
    manifest.to_csv(manifest_file, sep="\t", index=False)

    bash_file = root / f"Step06_FineMap_GWAS_{phenotype_slug}_{ancestry_slug}.sh"
    job_name = f"FM_{phenotype_slug}_{ancestry_code.lower()}"[:100]
    array_spec = f"1-{len(manifest)}%{args.max_parallel}"
    script_path = Path(__file__).resolve()

    bash_text = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_root}/finemap.%A_%a.out
#SBATCH --error={log_root}/finemap.%A_%a.err
#SBATCH --array={array_spec}
#SBATCH --mem={args.memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1

set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_FINEMAP_MANIFEST={shlex.quote(str(manifest_file))}

printf '%s\n' "=============================================================="
printf '%s\n' "STEP 06 - SUSIE-RSS FINEMAPPING"
printf '%s\n' "Job ID        : $SLURM_JOB_ID"
printf '%s\n' "Array task ID : $SLURM_ARRAY_TASK_ID"
printf '%s\n' "Hostname      : $(hostname)"
printf '%s\n' "=============================================================="

{shlex.quote(str(pipeline_python))} {shlex.quote(str(script_path))} --phenotype {shlex.quote(phenotype)} --ancestry {shlex.quote(ancestry_code)}
'''
    bash_file.write_text(bash_text, encoding="utf-8")
    bash_file.chmod(0o755)

    banner("PLANNING COMPLETE")
    print(f"Studies ready       : {len(manifest)}")
    print(f"Studies incomplete  : {len(incomplete)}")
    print(f"Partition           : {args.partition}")
    print(f"Walltime            : {args.time}")
    print(f"Memory/task         : {args.memory}")
    print(f"CPUs/task           : {args.cpus}")
    print(f"Reference N         : {reference_n}")
    print(f"Metadata summary    : {output_root / 'gwas_metadata_summary.tsv'}")
    print(f"Fine-map manifest   : {manifest_file}")
    print(f"Generated batch     : {bash_file}")
    print()
    print("NEXT COMMAND:")
    print(f"  sbatch {bash_file.name}")


# =============================================================================
# LOCUS DEFINITION
# =============================================================================


def load_leads_and_build_loci(lead_file: Path, window_kb: int) -> pd.DataFrame:
    leads = pd.read_csv(lead_file, sep="\t", low_memory=False)
    if leads.empty:
        return pd.DataFrame(
            columns=["LOCUS_ID", "CHR", "LOCUS_START", "LOCUS_END", "N_INDEPENDENT_SIGNALS", "LEAD_IDS", "LEAD_POSITIONS", "MIN_LEAD_P"]
        )

    chr_col = find_column(leads.columns, ["CHROM", "#CHROM", "CHR", "SOURCE_CHROMOSOME"])
    pos_col = find_column(leads.columns, ["POS", "BP"])
    p_col = find_column(leads.columns, ["P", "P_VALUE"])
    id_col = find_column(leads.columns, ["ID", "SNP", "SNPID", "rsID", "RSID"])
    if chr_col is None or pos_col is None:
        raise RuntimeError(f"Lead file lacks chromosome/position columns: {lead_file}")

    work = pd.DataFrame()
    work["CHR"] = leads[chr_col].map(clean_chr)
    work["POS"] = pd.to_numeric(leads[pos_col], errors="coerce")
    work["P"] = pd.to_numeric(leads[p_col], errors="coerce") if p_col else np.nan
    work["ID"] = leads[id_col].astype(str) if id_col else ""
    work = work.dropna(subset=["CHR", "POS"]).copy()
    work["CHR"] = work["CHR"].astype(int)
    work["POS"] = work["POS"].astype(int)
    work = work.sort_values(["CHR", "POS", "P"], kind="stable")

    window = window_kb * 1000
    merged = []
    locus_counter = 0
    for chrom, group in work.groupby("CHR", sort=True):
        current = None
        for _, row in group.iterrows():
            start = max(1, int(row["POS"]) - window)
            end = int(row["POS"]) + window
            signal = {
                "id": str(row["ID"]),
                "pos": int(row["POS"]),
                "p": float(row["P"]) if pd.notna(row["P"]) else np.nan,
            }
            if current is None or start > current["end"]:
                if current is not None:
                    merged.append(current)
                current = {"chr": int(chrom), "start": start, "end": end, "signals": [signal]}
            else:
                current["end"] = max(current["end"], end)
                current["signals"].append(signal)
        if current is not None:
            merged.append(current)

    records = []
    for item in merged:
        locus_counter += 1
        pvals = [x["p"] for x in item["signals"] if np.isfinite(x["p"])]
        records.append(
            {
                "LOCUS_ID": f"L{locus_counter:03d}",
                "CHR": item["chr"],
                "LOCUS_START": item["start"],
                "LOCUS_END": item["end"],
                "N_INDEPENDENT_SIGNALS": len(item["signals"]),
                "LEAD_IDS": ";".join(x["id"] for x in item["signals"]),
                "LEAD_POSITIONS": ";".join(str(x["pos"]) for x in item["signals"]),
                "MIN_LEAD_P": min(pvals) if pvals else np.nan,
            }
        )
    return pd.DataFrame(records)


# =============================================================================
# REGIONAL GWAS EXTRACTION
# =============================================================================


def inspect_qc_columns(path: Path) -> dict[str, str | None]:
    header = pd.read_csv(path, sep="\t", compression="infer", nrows=0)
    cols = list(header.columns)
    mapping = {
        "CHR": find_column(cols, ["CHR"]),
        "POS": find_column(cols, ["POS"]),
        "P": find_column(cols, ["P"]),
        "EA": find_column(cols, ["EA"]),
        "NEA": find_column(cols, ["NEA"]),
        "BETA": find_column(cols, ["BETA"]),
        "SE": find_column(cols, ["SE"]),
        "Z": find_column(cols, ["Z"]),
        "OR": find_column(cols, ["OR"]),
        "OR_95U": find_column(cols, ["OR_95U"]),
        "OR_95L": find_column(cols, ["OR_95L"]),
        "SNPID": find_column(cols, ["SNPID", "rsID", "RSID"]),
    }
    required = ["CHR", "POS", "P", "EA", "NEA"]
    missing = [x for x in required if mapping[x] is None]
    if missing:
        raise RuntimeError(f"QC GWAS missing required columns {missing}: {path}")

    effect_ready = bool(
        mapping["Z"]
        or (mapping["BETA"] and mapping["SE"])
        or (
            mapping["OR"]
            and (
                mapping["SE"]
                or (mapping["OR_95U"] and mapping["OR_95L"])
            )
        )
    )
    if not effect_ready:
        raise RuntimeError(
            "QC GWAS needs Z, BETA+SE, OR+SE, or OR+95% CI "
            f"for SuSiE-RSS: {path}"
        )
    return mapping


def extract_regional_gwas_once(
    qc_file: Path,
    loci: pd.DataFrame,
    regional_dir: Path,
    chunk_size: int,
) -> dict[str, Path]:
    mapping = inspect_qc_columns(qc_file)
    usecols = [x for x in mapping.values() if x is not None]
    usecols = list(dict.fromkeys(usecols))
    by_chr = defaultdict(list)
    for _, row in loci.iterrows():
        by_chr[int(row["CHR"])].append(row.to_dict())

    buffers: dict[str, list[pd.DataFrame]] = {str(x): [] for x in loci["LOCUS_ID"]}
    reader = pd.read_csv(
        qc_file,
        sep="\t",
        compression="infer",
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    )

    total = 0
    extracted = 0
    for chunk in reader:
        total += len(chunk)
        canon = pd.DataFrame(index=chunk.index)
        canon["CHR"] = chunk[mapping["CHR"]].map(clean_chr)
        canon["POS"] = pd.to_numeric(chunk[mapping["POS"]], errors="coerce")
        canon["P"] = pd.to_numeric(chunk[mapping["P"]], errors="coerce")
        canon["EA"] = chunk[mapping["EA"]].map(clean_allele)
        canon["NEA"] = chunk[mapping["NEA"]].map(clean_allele)
        canon["SNPID"] = chunk[mapping["SNPID"]].astype(str) if mapping["SNPID"] else ""
        # -------------------------------------------------------------
        # Effect statistic harmonisation for SuSiE-RSS.
        #
        # Preferred inputs:
        #   1. Z directly
        #   2. BETA + SE
        #   3. OR + SE       -> BETA = log(OR), Z = BETA / SE
        #   4. OR + 95% CI   -> reconstruct SE on the log-OR scale
        # -------------------------------------------------------------
        if mapping["BETA"]:
            canon["BETA"] = pd.to_numeric(
                chunk[mapping["BETA"]], errors="coerce"
            )
        elif mapping["OR"]:
            odds_ratio = pd.to_numeric(
                chunk[mapping["OR"]], errors="coerce"
            )
            odds_ratio = odds_ratio.where(odds_ratio > 0)
            canon["BETA"] = np.log(odds_ratio)
        else:
            canon["BETA"] = np.nan

        if mapping["SE"]:
            canon["SE"] = pd.to_numeric(
                chunk[mapping["SE"]], errors="coerce"
            )
        elif mapping["OR_95U"] and mapping["OR_95L"]:
            upper = pd.to_numeric(
                chunk[mapping["OR_95U"]], errors="coerce"
            )
            lower = pd.to_numeric(
                chunk[mapping["OR_95L"]], errors="coerce"
            )
            upper = upper.where(upper > 0)
            lower = lower.where(lower > 0)
            canon["SE"] = (
                np.log(upper) - np.log(lower)
            ) / (2.0 * 1.959963984540054)
        else:
            canon["SE"] = np.nan

        canon.loc[
            (~np.isfinite(canon["SE"])) | (canon["SE"] <= 0),
            "SE",
        ] = np.nan

        if mapping["Z"]:
            canon["Z"] = pd.to_numeric(
                chunk[mapping["Z"]], errors="coerce"
            )
        else:
            canon["Z"] = canon["BETA"] / canon["SE"]

        canon = canon.dropna(subset=["CHR", "POS", "P", "EA", "NEA", "Z"])
        if canon.empty:
            continue
        canon["CHR"] = canon["CHR"].astype(int)
        canon["POS"] = canon["POS"].astype(int)
        canon = canon[np.isfinite(canon["Z"])].copy()

        for chrom, locus_list in by_chr.items():
            chrom_data = canon[canon["CHR"] == chrom]
            if chrom_data.empty:
                continue
            for locus in locus_list:
                mask = chrom_data["POS"].between(
                    int(locus["LOCUS_START"]), int(locus["LOCUS_END"]), inclusive="both"
                )
                selected = chrom_data.loc[mask].copy()
                if not selected.empty:
                    buffers[str(locus["LOCUS_ID"])].append(selected)
                    extracted += len(selected)

    regional_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for _, locus in loci.iterrows():
        locus_id = str(locus["LOCUS_ID"])
        path = regional_dir / f"{locus_id}.tsv.gz"
        if buffers[locus_id]:
            frame = pd.concat(buffers[locus_id], ignore_index=True)
            frame = frame.sort_values(["CHR", "POS", "P"], kind="stable")
            frame = frame.drop_duplicates(subset=["CHR", "POS", "EA", "NEA"], keep="first")
        else:
            frame = pd.DataFrame(columns=["CHR", "POS", "P", "EA", "NEA", "SNPID", "BETA", "SE", "Z"])
        frame.to_csv(path, sep="\t", index=False, compression="gzip")
        paths[locus_id] = path

    print(f"GWAS rows scanned : {total:,}")
    print(f"Regional rows hit : {extracted:,}")
    return paths


# =============================================================================
# REFERENCE REGION + MATCHING
# =============================================================================


def create_regional_pvar(
    plink2: Path,
    root: Path,
    ancestry_code: str,
    chromosome: int,
    start: int,
    end: int,
    locus_dir: Path,
) -> Path:
    prefix = locus_dir / "reference_region"
    pvar = Path(str(prefix) + ".pvar")
    if pvar.exists() and pvar.stat().st_size > 0:
        return pvar
    run_command(
        [
            plink2,
            "--pfile", reference_prefix(root, ancestry_code, chromosome),
            "--chr", chromosome,
            "--from-bp", start,
            "--to-bp", end,
            "--make-just-pvar",
            "--out", prefix,
        ],
        locus_dir / "reference_region.command.log",
    )
    if not pvar.exists() or pvar.stat().st_size == 0:
        raise RuntimeError(f"Regional reference PVAR not generated for chr{chromosome}:{start}-{end}")
    return pvar


def read_pvar(path: Path) -> pd.DataFrame:
    records = []
    header = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                continue
            parts = line.rstrip("\n").split("\t")
            if header and len(parts) >= len(header):
                row = dict(zip(header, parts))
                chrom = clean_chr(row.get("CHROM", row.get("#CHROM", "")))
                pos = safe_int(row.get("POS"))
                if chrom and pos:
                    records.append(
                        {
                            "REFERENCE_CHR": chrom,
                            "REFERENCE_POS": pos,
                            "REFERENCE_ID": row.get("ID", ""),
                            "REFERENCE_REF": clean_allele(row.get("REF")),
                            "REFERENCE_ALT": clean_allele(row.get("ALT")),
                        }
                    )
    return pd.DataFrame(records)


def match_gwas_to_reference(regional: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    by_id = defaultdict(list)
    by_position = defaultdict(list)
    for _, row in reference.iterrows():
        record = row.to_dict()
        by_id[str(row["REFERENCE_ID"])].append(record)
        by_position[int(row["REFERENCE_POS"])].append(record)

    results = []
    for _, row in regional.iterrows():
        ea = clean_allele(row.get("EA"))
        nea = clean_allele(row.get("NEA"))
        if ea is None or nea is None:
            continue
        key = allele_key(ea, nea)
        rsid = clean_rsid(row.get("SNPID"))
        pos = int(row["POS"])
        selected = None
        method = None

        if rsid and rsid in by_id:
            candidates = [c for c in by_id[rsid] if allele_key(c["REFERENCE_REF"], c["REFERENCE_ALT"]) == key]
            if len(candidates) == 1:
                selected = candidates[0]
                method = "rsID+alleles"
        if selected is None:
            candidates = [c for c in by_position.get(pos, []) if allele_key(c["REFERENCE_REF"], c["REFERENCE_ALT"]) == key]
            if len(candidates) == 1:
                selected = candidates[0]
                method = "POS+alleles"
        if selected is None:
            continue

        ref = clean_allele(selected["REFERENCE_REF"])
        alt = clean_allele(selected["REFERENCE_ALT"])
        z = float(row["Z"])
        if not np.isfinite(z):
            continue

        # PLINK --r-unphased ... ref-based uses the reference allele orientation.
        if ea == ref and nea == alt:
            z_ref = z
            orientation = "EA=REF"
        elif ea == alt and nea == ref:
            z_ref = -z
            orientation = "EA=ALT_FLIPPED"
        else:
            continue

        record = row.to_dict()
        record.update(
            {
                "REFERENCE_ID": selected["REFERENCE_ID"],
                "REFERENCE_POS": selected["REFERENCE_POS"],
                "REFERENCE_REF": ref,
                "REFERENCE_ALT": alt,
                "MATCH_METHOD": method,
                "ORIENTATION": orientation,
                "Z_REF": z_ref,
            }
        )
        if pd.notna(row.get("BETA", np.nan)):
            beta = float(row["BETA"])
            record["BETA_REF"] = beta if orientation == "EA=REF" else -beta
        results.append(record)

    if not results:
        return pd.DataFrame()
    matched = pd.DataFrame(results)
    matched = matched.sort_values("P", kind="stable").drop_duplicates("REFERENCE_ID", keep="first")
    return matched.reset_index(drop=True)


# =============================================================================
# LD MATRIX
# =============================================================================


def _parse_lead_positions(value: Any) -> list[int]:
    positions = []
    for token in str(value or "").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            pos = int(float(token))
        except Exception:
            continue
        if pos > 0:
            positions.append(pos)
    return sorted(set(positions))


def _balanced_ld_subset(
    matched: pd.DataFrame,
    max_variants: int,
    lead_positions: list[int],
) -> pd.DataFrame:
    """Deterministically cap an oversized locus while preserving lead neighborhoods.

    Variants are first assigned to their nearest independent lead signal.  An
    approximately equal quota is selected around each lead by physical distance,
    then any remaining slots are filled globally by distance and association P.
    This is only invoked AFTER the 1000G MAF/GENO filters have been applied.
    """
    if len(matched) <= max_variants:
        return matched.copy()

    work = matched.copy().reset_index(drop=True)
    work["P"] = pd.to_numeric(work["P"], errors="coerce")
    work["POS"] = pd.to_numeric(work["POS"], errors="coerce")

    leads = [int(x) for x in lead_positions if int(x) > 0]
    leads = sorted(set(leads))

    if not leads:
        # Fallback for an unexpected missing LEAD_POSITIONS field: retain the
        # strongest associations deterministically rather than failing outright.
        return (
            work.sort_values(["P", "POS", "REFERENCE_ID"], kind="stable")
            .head(max_variants)
            .reset_index(drop=True)
        )

    pos = work["POS"].to_numpy(dtype=float)
    lead_array = np.asarray(leads, dtype=float)
    distances = np.abs(pos[:, None] - lead_array[None, :])
    nearest_index = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(work)), nearest_index]

    work["_NEAREST_LEAD_INDEX"] = nearest_index
    work["_NEAREST_LEAD_POS"] = [leads[i] for i in nearest_index]
    work["_LEAD_DISTANCE"] = nearest_distance

    quota = max(1, max_variants // len(leads))
    selected_indices: list[int] = []

    for lead_index in range(len(leads)):
        group = work[work["_NEAREST_LEAD_INDEX"] == lead_index]
        if group.empty:
            continue
        group = group.sort_values(
            ["_LEAD_DISTANCE", "P", "POS", "REFERENCE_ID"],
            kind="stable",
        )
        selected_indices.extend(group.head(quota).index.tolist())

    # De-duplicate and respect the exact cap.
    selected_indices = list(dict.fromkeys(selected_indices))[:max_variants]

    remaining_slots = max_variants - len(selected_indices)
    if remaining_slots > 0:
        remainder = work.drop(index=selected_indices, errors="ignore")
        remainder = remainder.sort_values(
            ["_LEAD_DISTANCE", "P", "POS", "REFERENCE_ID"],
            kind="stable",
        )
        selected_indices.extend(remainder.head(remaining_slots).index.tolist())

    selected = work.loc[selected_indices].copy()
    selected = selected.sort_values(["P", "POS", "REFERENCE_ID"], kind="stable")
    return selected.reset_index(drop=True)


def calculate_ld(
    plink2: Path,
    root: Path,
    ancestry_code: str,
    chromosome: int,
    matched: pd.DataFrame,
    locus_dir: Path,
    maf: float,
    geno: float,
    threads: int,
    max_variants: int,
    lead_positions: list[int] | None = None,
) -> tuple[Path, Path, int]:
    """Build a validated LD matrix with post-reference-QC size protection."""
    lead_positions = lead_positions or []

    # ------------------------------------------------------------------
    # 1. Start from all GWAS/reference matched variants.
    # ------------------------------------------------------------------
    all_ids_file = locus_dir / "matched_reference_ids_all.txt"
    matched["REFERENCE_ID"].to_csv(all_ids_file, index=False, header=False)

    # ------------------------------------------------------------------
    # 2. Apply 1000G reference QC BEFORE enforcing the LD size cap.
    #    --write-snplist gives the exact variants surviving MAF/GENO and
    #    biallelic filters without constructing the dense LD matrix yet.
    # ------------------------------------------------------------------
    qc_prefix = locus_dir / f"{ancestry_code}_reference_qc"
    run_command(
        [
            plink2,
            "--pfile", reference_prefix(root, ancestry_code, chromosome),
            "--extract", all_ids_file,
            "--maf", maf,
            "--geno", geno,
            "--min-alleles", 2,
            "--max-alleles", 2,
            "--write-snplist",
            "--threads", threads,
            "--out", qc_prefix,
        ],
        locus_dir / f"{ancestry_code}_reference_qc.command.log",
    )

    snplist = Path(str(qc_prefix) + ".snplist")
    if not snplist.exists():
        raise RuntimeError("PLINK2 reference-QC SNP list is missing")

    surviving_ids = [
        x.strip() for x in snplist.read_text().splitlines() if x.strip()
    ]
    surviving_set = set(surviving_ids)

    qc_matched = matched[
        matched["REFERENCE_ID"].astype(str).isin(surviving_set)
    ].copy()

    # Preserve PLINK/reference order where possible for deterministic auditing.
    order = {variant_id: i for i, variant_id in enumerate(surviving_ids)}
    qc_matched["_REFERENCE_QC_ORDER"] = (
        qc_matched["REFERENCE_ID"].astype(str).map(order)
    )
    qc_matched = qc_matched.sort_values(
        ["_REFERENCE_QC_ORDER", "P"], kind="stable"
    ).drop(columns=["_REFERENCE_QC_ORDER"])
    qc_matched = qc_matched.reset_index(drop=True)

    qc_matched.to_csv(
        locus_dir / "reference_qc_matched_variants.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    if len(qc_matched) < 2:
        raise RuntimeError(
            f"Only {len(qc_matched)} matched variants survive reference MAF/GENO QC"
        )

    # ------------------------------------------------------------------
    # 3. If still oversized, deterministically retain balanced local
    #    neighborhoods around the independent lead signals.
    # ------------------------------------------------------------------
    selected = qc_matched
    selection_applied = len(qc_matched) > max_variants

    if selection_applied:
        print(
            f"[WARNING] {len(qc_matched):,} variants survive reference QC; "
            f"LD safety cap is {max_variants:,}."
        )
        print(
            "[WARNING] Applying deterministic balanced selection around "
            "independent lead positions."
        )
        selected = _balanced_ld_subset(
            qc_matched,
            max_variants=max_variants,
            lead_positions=lead_positions,
        )

    selected.to_csv(
        locus_dir / "ld_selected_variants.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    selection_summary = {
        "N_MATCHED_BEFORE_REFERENCE_QC": int(len(matched)),
        "N_AFTER_REFERENCE_QC": int(len(qc_matched)),
        "MAX_LD_VARIANTS": int(max_variants),
        "SELECTION_APPLIED": bool(selection_applied),
        "N_SELECTED_FOR_LD": int(len(selected)),
        "LEAD_POSITIONS": [int(x) for x in lead_positions],
        "SELECTION_METHOD": (
            "balanced_nearest_lead"
            if selection_applied and lead_positions
            else "association_rank_fallback"
            if selection_applied
            else "all_reference_qc_variants"
        ),
    }
    (locus_dir / "ld_variant_selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2), encoding="utf-8"
    )

    ids_file = locus_dir / "matched_reference_ids.txt"
    selected["REFERENCE_ID"].to_csv(ids_file, index=False, header=False)

    # ------------------------------------------------------------------
    # 4. Build the dense signed LD matrix from the final selected IDs.
    # ------------------------------------------------------------------
    ld_prefix = locus_dir / f"{ancestry_code}_LD"
    run_command(
        [
            plink2,
            "--pfile", reference_prefix(root, ancestry_code, chromosome),
            "--extract", ids_file,
            "--maf", maf,
            "--geno", geno,
            "--min-alleles", 2,
            "--max-alleles", 2,
            "--r-unphased", "square", "bin4", "ref-based", "yes-really",
            "--threads", threads,
            "--out", ld_prefix,
        ],
        locus_dir / f"{ancestry_code}_LD.command.log",
    )

    ld_file = locus_dir / f"{ancestry_code}_LD.unphased.vcor1.bin"
    vars_file = locus_dir / f"{ancestry_code}_LD.unphased.vcor1.bin.vars"
    if not ld_file.exists() or not vars_file.exists():
        raise RuntimeError("PLINK2 LD output is missing")

    variants = [x.strip() for x in vars_file.read_text().splitlines() if x.strip()]
    p = len(variants)
    expected_bytes = p * p * 4
    actual_bytes = ld_file.stat().st_size
    if p < 2 or actual_bytes != expected_bytes:
        raise RuntimeError(
            f"LD matrix validation failed: p={p}, "
            f"expected_bytes={expected_bytes}, actual_bytes={actual_bytes}"
        )

    if p > max_variants:
        raise RuntimeError(
            f"Internal error: final LD matrix has {p:,} variants, "
            f"above safety cap {max_variants:,}"
        )

    return ld_file, vars_file, p


# =============================================================================
# SUSIE R HELPER
# =============================================================================


def write_susie_r_script(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    code = r'''options(warn=1)
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 11) stop("Expected 11 arguments")
stats_file <- args[1]
ld_file <- args[2]
vars_file <- args[3]
outdir <- args[4]
n_gwas <- as.numeric(args[5])
n_ref <- as.numeric(args[6])
L <- as.integer(args[7])
coverage <- as.numeric(args[8])
ld_ridge <- as.numeric(args[9])
estimate_s_max_p <- as.integer(args[10])
locus_id <- args[11]

suppressPackageStartupMessages(library(susieR))
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
logcon <- file(file.path(outdir, "susie_rss.log"), open="wt")
sink(logcon, type="output"); sink(logcon, type="message")
on.exit({sink(type="message"); sink(type="output"); close(logcon)}, add=TRUE)

cat("SuSiE-RSS locus:", locus_id, "\n")
cat("GWAS N:", n_gwas, "\n")
cat("Reference N:", n_ref, "\n")
cat("L:", L, " coverage:", coverage, " ridge:", ld_ridge, "\n")

vars <- scan(vars_file, what="character", quiet=TRUE)
p <- length(vars)
if (p < 2) stop("Fewer than two LD variants")
con <- file(ld_file, "rb")
values <- readBin(con, what="numeric", n=p*p, size=4, endian="little")
close(con)
if (length(values) != p*p) stop("LD matrix length mismatch")
R <- matrix(values, nrow=p, ncol=p, byrow=TRUE)
rm(values); gc()
if (any(!is.finite(R))) stop("LD matrix contains non-finite values")
R <- (R + t(R))/2
diag(R) <- 1
if (is.finite(ld_ridge) && ld_ridge > 0) {
  R <- R * (1-ld_ridge)
  diag(R) <- 1
}

stats <- read.delim(stats_file, header=TRUE, stringsAsFactors=FALSE, check.names=FALSE)
if (!all(c("REFERENCE_ID","Z_REF") %in% names(stats))) stop("Matched stats missing REFERENCE_ID/Z_REF")
idx <- match(vars, stats$REFERENCE_ID)
if (any(is.na(idx))) stop("LD variants missing from matched GWAS")
stats <- stats[idx,,drop=FALSE]
if (!all(stats$REFERENCE_ID == vars)) stop("GWAS and LD order mismatch")
z <- as.numeric(stats$Z_REF)
if (any(!is.finite(z))) stop("Non-finite Z scores")

s_est <- NA_real_; s_status <- "NOT_RUN"; s_error <- NA_character_
if (p <= estimate_s_max_p) {
  result <- tryCatch(
    list(ok=TRUE, value=susieR::estimate_s_rss(z=z, R=R, n=n_gwas), error=NA_character_),
    error=function(e) list(ok=FALSE, value=NA_real_, error=conditionMessage(e))
  )
  if (result$ok) { s_est <- result$value; s_status <- "SUCCESS" }
  else { s_status <- "FAILED"; s_error <- result$error }
} else {
  s_status <- paste0("SKIPPED_P_GT_", estimate_s_max_p)
}

formals_rss <- names(formals(susieR::susie_rss))
fit_args <- list(
  z=z, R=R, n=n_gwas, L=min(L,p), coverage=coverage,
  estimate_residual_variance=FALSE, estimate_prior_variance=TRUE,
  refine=FALSE, verbose=TRUE
)
if ("estimate_prior_method" %in% formals_rss) fit_args$estimate_prior_method <- "EM"
if ("max_iter" %in% formals_rss) fit_args$max_iter <- 100
if ("tol" %in% formals_rss) fit_args$tol <- 1e-3
if ("R_finite" %in% formals_rss) fit_args$R_finite <- n_ref
fit <- do.call(susieR::susie_rss, fit_args)
saveRDS(fit, file.path(outdir, "susie_fit.rds"))

pip <- fit$pip
if (length(pip) != nrow(stats)) stop("PIP length mismatch")
variant <- stats
variant$PIP <- as.numeric(pip)
variant$IN_95_CREDIBLE_SET <- FALSE
variant$CREDIBLE_SET <- ""

cs_rows <- list()
if (!is.null(fit$sets) && !is.null(fit$sets$cs)) {
  cs <- fit$sets$cs
  for (i in seq_along(cs)) {
    members <- as.integer(cs[[i]])
    members <- members[members >= 1 & members <= nrow(variant)]
    if (length(members) == 0) next
    cs_name <- names(cs)[i]
    if (is.null(cs_name) || is.na(cs_name) || cs_name == "") cs_name <- paste0("CS", i)
    variant$IN_95_CREDIBLE_SET[members] <- TRUE
    for (j in members) {
      existing <- variant$CREDIBLE_SET[j]
      variant$CREDIBLE_SET[j] <- ifelse(existing == "", cs_name, paste(existing, cs_name, sep=";"))
      cs_rows[[length(cs_rows)+1]] <- data.frame(
        LOCUS_ID=locus_id,
        CREDIBLE_SET=cs_name,
        REFERENCE_ID=variant$REFERENCE_ID[j],
        CHR=variant$CHR[j],
        POS=variant$POS[j],
        PIP=variant$PIP[j],
        stringsAsFactors=FALSE
      )
    }
  }
}
variant <- variant[order(-variant$PIP),,drop=FALSE]
write.table(variant, file.path(outdir,"susie_variants.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
if (length(cs_rows) > 0) cs_df <- do.call(rbind, cs_rows) else cs_df <- data.frame(LOCUS_ID=character(),CREDIBLE_SET=character(),REFERENCE_ID=character(),CHR=integer(),POS=integer(),PIP=numeric())
write.table(cs_df, file.path(outdir,"credible_sets.tsv"), sep="\t", quote=FALSE, row.names=FALSE)

summary <- data.frame(
  LOCUS_ID=locus_id,
  N_VARIANTS=p,
  N_CREDIBLE_SETS=ifelse(is.null(fit$sets$cs),0,length(fit$sets$cs)),
  MAX_PIP=max(variant$PIP, na.rm=TRUE),
  CONVERGED=ifelse(is.null(fit$converged),NA,fit$converged),
  SUSIE_L=min(L,p),
  GWAS_N=n_gwas,
  REFERENCE_N=n_ref,
  COVERAGE=coverage,
  LD_RIDGE=ld_ridge,
  stringsAsFactors=FALSE
)
write.table(summary, file.path(outdir,"susie_summary.tsv"), sep="\t", quote=FALSE, row.names=FALSE)

diagnostics <- data.frame(
  LOCUS_ID=locus_id,
  ESTIMATE_S_RSS=s_est,
  ESTIMATE_S_STATUS=s_status,
  ESTIMATE_S_ERROR=s_error,
  MAX_ABS_Z=max(abs(z)),
  stringsAsFactors=FALSE
)
write.table(diagnostics, file.path(outdir,"susie_diagnostics.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return path


# =============================================================================
# WORKER
# =============================================================================


def worker_mode() -> None:
    manifest_value = os.environ.get("GWAS_FINEMAP_MANIFEST")
    task_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if not manifest_value or not task_value:
        raise RuntimeError("GWAS_FINEMAP_MANIFEST and SLURM_ARRAY_TASK_ID are required in worker mode")

    manifest_file = Path(manifest_value).resolve()
    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str)
    task_id = int(task_value)
    task_numbers = pd.to_numeric(manifest["FINEMAP_TASK_ID"], errors="coerce")
    selected = manifest[task_numbers == task_id]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one manifest row for task {task_id}, found {len(selected)}")
    row = selected.iloc[0]

    root = Path.cwd().resolve()
    accession = str(row["STUDY_ACCESSION"])
    phenotype = str(row["PHENOTYPE"])
    ancestry_code = str(row["ANCESTRY_CODE"]).upper()
    ancestry_label = str(row["ANCESTRY_LABEL"])
    qc_file = Path(row["QC_FILE"]).resolve()
    lead_file = Path(row["LEAD_VARIANTS_FILE"]).resolve()
    output_dir = Path(row["OUTPUT_DIR"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gwas_n = int(float(row["GWAS_N"]))
    gwas_n_source = str(row["GWAS_N_SOURCE"])
    reference_n = int(float(row["REFERENCE_N"]))
    locus_window_kb = int(float(row["LOCUS_WINDOW_KB"]))
    reference_maf = float(row["REFERENCE_MAF"])
    reference_geno = float(row["REFERENCE_GENO"])
    min_variants = int(float(row["MIN_VARIANTS"]))
    max_ld_variants = int(float(row["MAX_LD_VARIANTS"]))
    coverage = float(row["COVERAGE"])
    susie_l = int(float(row["SUSIE_L"]))
    max_susie_l = int(float(row["MAX_SUSIE_L"]))
    ld_ridge = float(row["LD_RIDGE"])
    chunk_size = int(float(row["CHUNK_SIZE"]))
    plink2 = Path(row["PLINK2"]).resolve()
    rscript = Path(row["R_SCRIPT"]).resolve()
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_CPUS))

    final_variants = output_dir / f"{accession}_finemapped_variants.tsv.gz"
    final_cs = output_dir / f"{accession}_95pct_credible_sets.tsv"
    final_summary = output_dir / f"{accession}_finemapping_summary.tsv"
    failed_file = output_dir / f"{accession}_failed_loci.tsv"
    summary_json = output_dir / "finemapping_summary.json"
    hard_failure = output_dir / "FINEMAPPING_FAILED.txt"

    if final_variants.exists() and final_summary.exists() and summary_json.exists() and not hard_failure.exists():
        banner("FINEMAPPING ALREADY COMPLETE")
        print(accession)
        return

    banner("STEP 06 - SUSIE-RSS FINEMAPPING WORKER")
    print(f"Study          : {accession}")
    print(f"Phenotype      : {phenotype}")
    print(f"Ancestry       : {ancestry_label} ({ancestry_code})")
    print(f"GWAS N         : {gwas_n} [{gwas_n_source}]")
    print(f"Reference N    : {reference_n}")
    print(f"QC GWAS        : {qc_file}")
    print(f"Lead variants  : {lead_file}")
    print(f"Output         : {output_dir}")

    try:
        loci = load_leads_and_build_loci(lead_file, locus_window_kb)
        loci_file = output_dir / "loci_definition.tsv"
        loci.to_csv(loci_file, sep="\t", index=False)
        if loci.empty:
            # A study can complete Step 05 successfully yet have zero independent
            # genome-wide-significant lead variants.  That is a valid scientific
            # outcome, not a pipeline failure.  Write explicit empty deliverables
            # and a machine-readable completion status so downstream stages can
            # skip this study cleanly.
            pd.DataFrame(
                columns=[
                    "STUDY_ACCESSION", "PHENOTYPE", "ANCESTRY_CODE",
                    "LOCUS_ID", "REFERENCE_ID", "CHR", "POS", "PIP",
                ]
            ).to_csv(final_variants, sep="\t", index=False, compression="gzip")

            pd.DataFrame(
                columns=[
                    "STUDY_ACCESSION", "LOCUS_ID", "CREDIBLE_SET",
                    "REFERENCE_ID", "CHR", "POS", "PIP",
                ]
            ).to_csv(final_cs, sep="\t", index=False)

            no_loci_row = {
                "STUDY_ACCESSION": accession,
                "PHENOTYPE": phenotype,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
                "GWAS_N": gwas_n,
                "GWAS_N_SOURCE": gwas_n_source,
                "REFERENCE_N": reference_n,
                "STATUS": "NO_SIGNIFICANT_LOCI",
                "N_LOCI": 0,
                "N_LOCI_SUCCESS": 0,
                "N_LOCI_FAILED": 0,
                "MESSAGE": "Step 05 produced zero independent lead variants; no fine-mapping was required.",
            }
            pd.DataFrame([no_loci_row]).to_csv(
                final_summary, sep="\t", index=False
            )

            pd.DataFrame(
                columns=[
                    "STUDY_ACCESSION", "LOCUS_ID", "CHR",
                    "LOCUS_START", "LOCUS_END", "ERROR",
                ]
            ).to_csv(failed_file, sep="\t", index=False)

            summary = {
                **no_loci_row,
                "FINEMAPPED_VARIANTS_FILE": str(final_variants),
                "CREDIBLE_SETS_FILE": str(final_cs),
                "SUMMARY_FILE": str(final_summary),
                "FAILED_LOCI_FILE": str(failed_file),
                "COMPLETED_UTC": datetime.now(timezone.utc).isoformat(),
            }
            summary_json.write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

            if hard_failure.exists():
                hard_failure.unlink()

            banner("FINEMAPPING SKIPPED - NO SIGNIFICANT LOCI")
            print(f"Study  : {accession}")
            print("Status : NO_SIGNIFICANT_LOCI")
            print("Step 05 produced zero independent lead variants.")
            print("This is a valid completed outcome; there is nothing to fine-map.")
            return

        regional_dir = output_dir / "regional_gwas"
        locus_root = output_dir / "loci"
        locus_root.mkdir(parents=True, exist_ok=True)
        regional_paths = extract_regional_gwas_once(qc_file, loci, regional_dir, chunk_size)
        r_helper = write_susie_r_script(output_dir / "run_susie_rss.R")

        fine_frames = []
        cs_frames = []
        summary_frames = []
        failures = []

        for _, locus in loci.iterrows():
            locus_id = str(locus["LOCUS_ID"])
            chromosome = int(locus["CHR"])
            start = int(locus["LOCUS_START"])
            end = int(locus["LOCUS_END"])
            n_signals = int(locus["N_INDEPENDENT_SIGNALS"])
            locus_dir = locus_root / locus_id
            locus_dir.mkdir(parents=True, exist_ok=True)
            try:
                regional = pd.read_csv(regional_paths[locus_id], sep="\t", compression="infer", low_memory=False)
                if len(regional) < min_variants:
                    raise RuntimeError(f"Only {len(regional)} regional GWAS variants; minimum is {min_variants}")

                pvar = create_regional_pvar(
                    plink2, root, ancestry_code, chromosome, start, end, locus_dir
                )
                reference = read_pvar(pvar)
                matched = match_gwas_to_reference(regional, reference)
                if len(matched) < min_variants:
                    raise RuntimeError(
                        f"Only {len(matched)} GWAS/reference matched variants; minimum is {min_variants}"
                    )
                matched_file = locus_dir / "matched_variants.tsv"
                matched.to_csv(matched_file, sep="\t", index=False)

                lead_positions = _parse_lead_positions(
                    locus.get("LEAD_POSITIONS", "")
                )
                ld_file, vars_file, p = calculate_ld(
                    plink2=plink2,
                    root=root,
                    ancestry_code=ancestry_code,
                    chromosome=chromosome,
                    matched=matched,
                    locus_dir=locus_dir,
                    maf=reference_maf,
                    geno=reference_geno,
                    threads=threads,
                    max_variants=max_ld_variants,
                    lead_positions=lead_positions,
                )
                if p < min_variants:
                    raise RuntimeError(f"Only {p} variants remain in LD matrix; minimum is {min_variants}")

                susie_dir = locus_dir / "susie"
                L = min(max(susie_l, n_signals), max_susie_l, p)
                run_command(
                    [
                        rscript,
                        r_helper,
                        matched_file,
                        ld_file,
                        vars_file,
                        susie_dir,
                        gwas_n,
                        reference_n,
                        L,
                        coverage,
                        ld_ridge,
                        DEFAULT_ESTIMATE_S_MAX,
                        locus_id,
                    ],
                    susie_dir / "r_command.log",
                )

                variants = pd.read_csv(susie_dir / "susie_variants.tsv", sep="\t", low_memory=False)
                variants.insert(0, "STUDY_ACCESSION", accession)
                variants.insert(1, "PHENOTYPE", phenotype)
                variants.insert(2, "ANCESTRY_CODE", ancestry_code)
                variants.insert(3, "LOCUS_ID", locus_id)
                fine_frames.append(variants)

                cs_path = susie_dir / "credible_sets.tsv"
                if cs_path.exists() and cs_path.stat().st_size > 0:
                    cs = pd.read_csv(cs_path, sep="\t", low_memory=False)
                    if not cs.empty:
                        cs.insert(0, "STUDY_ACCESSION", accession)
                        cs.insert(1, "PHENOTYPE", phenotype)
                        cs.insert(2, "ANCESTRY_CODE", ancestry_code)
                        cs_frames.append(cs)

                s = pd.read_csv(susie_dir / "susie_summary.tsv", sep="\t", low_memory=False)
                s.insert(0, "STUDY_ACCESSION", accession)
                s["CHR"] = chromosome
                s["LOCUS_START"] = start
                s["LOCUS_END"] = end
                s["N_INDEPENDENT_SIGNALS"] = n_signals
                s["N_REGIONAL_GWAS_VARIANTS"] = len(regional)
                s["N_MATCHED_VARIANTS"] = len(matched)
                summary_frames.append(s)

                locus_failure_marker = locus_dir / "LOCUS_FAILED.txt"
                if locus_failure_marker.exists():
                    locus_failure_marker.unlink()

            except Exception as exc:
                failure_text = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "STUDY_ACCESSION": accession,
                        "LOCUS_ID": locus_id,
                        "CHR": chromosome,
                        "LOCUS_START": start,
                        "LOCUS_END": end,
                        "ERROR": failure_text,
                    }
                )
                (locus_dir / "LOCUS_FAILED.txt").write_text(
                    failure_text + "\n\n" + traceback.format_exc(), encoding="utf-8"
                )
                print(f"[FAILED] {locus_id}: {failure_text}", file=sys.stderr)

        if fine_frames:
            all_fine = pd.concat(fine_frames, ignore_index=True)
            all_fine.to_csv(final_variants, sep="\t", index=False, compression="gzip")
        else:
            pd.DataFrame().to_csv(final_variants, sep="\t", index=False, compression="gzip")

        if cs_frames:
            pd.concat(cs_frames, ignore_index=True).to_csv(final_cs, sep="\t", index=False)
        else:
            pd.DataFrame(columns=["STUDY_ACCESSION", "LOCUS_ID", "CREDIBLE_SET", "REFERENCE_ID", "CHR", "POS", "PIP"]).to_csv(
                final_cs, sep="\t", index=False
            )

        if summary_frames:
            pd.concat(summary_frames, ignore_index=True).to_csv(final_summary, sep="\t", index=False)
        else:
            pd.DataFrame().to_csv(final_summary, sep="\t", index=False)

        pd.DataFrame(failures).to_csv(failed_file, sep="\t", index=False)
        summary = {
            "STUDY_ACCESSION": accession,
            "PHENOTYPE": phenotype,
            "ANCESTRY_CODE": ancestry_code,
            "GWAS_N": gwas_n,
            "GWAS_N_SOURCE": gwas_n_source,
            "REFERENCE_N": reference_n,
            "N_LOCI": int(len(loci)),
            "N_LOCI_SUCCESS": int(len(summary_frames)),
            "N_LOCI_FAILED": int(len(failures)),
            "FINEMAPPED_VARIANTS_FILE": str(final_variants),
            "CREDIBLE_SETS_FILE": str(final_cs),
            "SUMMARY_FILE": str(final_summary),
            "FAILED_LOCI_FILE": str(failed_file),
            "COMPLETED_UTC": datetime.now(timezone.utc).isoformat(),
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if hard_failure.exists():
            hard_failure.unlink()

        banner("FINEMAPPING FINISHED")
        print(f"Loci total   : {len(loci)}")
        print(f"Loci success : {len(summary_frames)}")
        print(f"Loci failed  : {len(failures)}")
        print(f"Variants     : {final_variants}")
        print(f"Credible sets: {final_cs}")

        if failures:
            raise RuntimeError(
                f"Fine-mapping completed partially: {len(failures)} of {len(loci)} loci failed. "
                f"See {failed_file}"
            )

    except Exception as exc:
        hard_failure.write_text(
            f"STEP 06 FINEMAPPING FAILED\n\n{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        raise


# =============================================================================
# DIRECT SINGLE-INDEX MODE
# =============================================================================


def project_manifest_path(args) -> Path:
    ancestry_code, ancestry_label = resolve_ancestry(args.ancestry)
    phenotype_slug = slugify(args.phenotype)
    ancestry_slug = slugify(ancestry_label)
    return (
        Path.cwd().resolve()
        / "06_finemapping"
        / phenotype_slug
        / ancestry_slug
        / "finemapping_manifest.tsv"
    )


def direct_index_mode(args) -> None:
    """Run one manifest task directly, without requiring SLURM array variables."""
    manifest_file = project_manifest_path(args)

    if not manifest_file.exists():
        banner("FINEMAPPING MANIFEST NOT FOUND")
        print(f"Expected: {manifest_file}")
        print("Generating the Step 06 planner/manifest first...")
        planner_mode(args)

    if not manifest_file.exists():
        raise FileNotFoundError(
            "Fine-mapping manifest was not created. Complete the planner inputs "
            "and rerun the command."
        )

    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str)
    if "FINEMAP_TASK_ID" not in manifest.columns:
        raise RuntimeError(f"FINEMAP_TASK_ID missing from manifest: {manifest_file}")

    task_numbers = pd.to_numeric(manifest["FINEMAP_TASK_ID"], errors="coerce")
    selected = manifest[task_numbers == int(args.index)]
    if len(selected) != 1:
        available = sorted(
            int(x) for x in task_numbers.dropna().astype(int).unique().tolist()
        )
        raise RuntimeError(
            f"--index {args.index} does not identify exactly one manifest row. "
            f"Available indices: {available}"
        )

    accession = str(selected.iloc[0]["STUDY_ACCESSION"])
    banner("STEP 06 - DIRECT SINGLE-INDEX MODE")
    print(f"Manifest : {manifest_file}")
    print(f"Index    : {args.index}")
    print(f"Study    : {accession}")
    print(f"CPUs     : {args.cpus}")
    print()
    print(
        "NOTE: direct --index mode runs on the machine/node where this Python "
        "command is executed. On an HPC login node, use srun or sbatch --wrap "
        "to place this same command on a compute node."
    )

    old_manifest = os.environ.get("GWAS_FINEMAP_MANIFEST")
    old_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    old_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    os.environ["GWAS_FINEMAP_MANIFEST"] = str(manifest_file.resolve())
    os.environ["SLURM_ARRAY_TASK_ID"] = str(args.index)
    if old_cpus is None:
        os.environ["SLURM_CPUS_PER_TASK"] = str(args.cpus)

    try:
        worker_mode()
    finally:
        if old_manifest is None:
            os.environ.pop("GWAS_FINEMAP_MANIFEST", None)
        else:
            os.environ["GWAS_FINEMAP_MANIFEST"] = old_manifest

        if old_task is None:
            os.environ.pop("SLURM_ARRAY_TASK_ID", None)
        else:
            os.environ["SLURM_ARRAY_TASK_ID"] = old_task

        if old_cpus is None:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
        else:
            os.environ["SLURM_CPUS_PER_TASK"] = old_cpus


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Worker uses parameters from the manifest; parser is still invoked because
    # the generated batch repeats phenotype/ancestry for transparent provenance.
    args = arguments()

    # Explicit CLI index always means: execute exactly that one manifest task.
    if args.index is not None:
        direct_index_mode(args)
    elif os.environ.get("SLURM_ARRAY_TASK_ID") and os.environ.get("GWAS_FINEMAP_MANIFEST"):
        worker_mode()
    else:
        planner_mode(args)


if __name__ == "__main__":
    main()

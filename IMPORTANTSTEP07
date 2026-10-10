#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWAS2m STEP 07 — Distributed API-only annotation of fine-mapped variants
=======================================================================

GOAL
----
Annotate every usable Step06 fine-mapped variant without downloading large
reference annotation databases.

This file is intentionally self-contained. It does NOT require:
    * a local VEP cache
    * a local GRCh38 FASTA
    * local AlphaMissense
    * local CADD
    * local REVEL
    * local ClinVar
    * local ENCODE / SCREEN matrices
    * local RegulomeDB
    * local UniProt

Instead it builds a deduplicated variant universe, splits it into provider-
specific batches, submits independent SLURM arrays, stores only returned
annotation shards, then merges them back into one annotation table per study.

PROVIDERS
---------
1. ENSEMBL_VEP
   POST /vep/homo_sapiens/region
   - exon / intron / UTR / upstream / downstream / intergenic
   - genes, transcripts, biotypes, lncRNA/non-coding transcript consequences
   - HGVS c./p., protein consequence, amino-acid change
   - SIFT, PolyPhen
   - AlphaMissense
   - CADD
   - REVEL
   - ProtVar
   - SpliceAI
   - BLOSUM62
   - ClinPred
   - EVE
   - LOEUF
   - MaveDB
   - OpenTargets
   - phenotype/co-located variant evidence
   - Ensembl regulatory consequences
   - protein domains and UniProt identifiers

2. ENSEMBL_VARIATION
   POST /variation/homo_sapiens
   - population frequencies
   - phenotype associations
   - clinical significance
   - ancestral/minor allele

3. SCREEN
   GraphQL https://screen.api.wenglab.org/graphql
   - overlapping ENCODE cCRE
   - cCRE class (PLS/pELS/dELS/etc.)
   - ATAC/DNase/CTCF/H3K27ac/H3K4me3 max Z
   - biosample/cell/tissue-specific signals
   - nearest genes
   - experimental cCRE -> gene links
   - computational cCRE -> gene links
   - TF representative peaks
   - silencers / dynamic enhancers

4. REGULOMEDB
   - regulatory score/rank/probability
   - compact supporting feature evidence

5. CLINVAR
   NCBI E-utilities
   - query rsID-bearing variants
   - ClinVar variation summaries, traits, significance, accessions

UniProt details:
    VEP is called with uniprot=1 and domains=1, so UniProt accessions and
    protein-domain annotations are obtained without a separate dependency
    phase.

ARCHITECTURE
------------
Step06
  |
  +--> unique variant universe ---------------------------------------------+
  |                                                                         |
  +--> study/locus/variant map                                              |
                                                                            |
          +-------------------+----------------+----------------+-------------+
          |                   |                |                |             |
     ENSEMBL_VEP        ENSEMBL_VARIATION    SCREEN       REGULOMEDB     CLINVAR
     100 variants       100-200 rsIDs       50 vars      25 vars        20 rsIDs
     / array task       / array task        / task       / task         / task
          |                   |                |                |             |
          +-------------------+----------------+----------------+-------------+
                                                                            |
                                                                     MERGE JOB
                                                                            |
                                                   global unique-variant annotation
                                                                            |
                                                   one master table per GWAS study
                                                                            |
                                                   one locus summary per GWAS study

IMPORTANT
---------
* The SLURM array concurrency defaults are deliberately conservative because
  public APIs can throttle clients. More array jobs does NOT mean you should
  send hundreds of concurrent requests to one public service.
* HTTP 429 / Retry-After is honoured.
* Network/API failure is recorded as API_ERROR, never converted to biological
  NO_HIT.
* Existing shard files are reused unless --force is supplied.
* Only API responses/results are stored. No reference annotation database is
  mirrored locally.
* SCREEN requires:
      export SCREEN_API_KEY='...'
* NCBI strongly prefers an email and supports API keys:
      export NCBI_EMAIL='you@example.com'
      export NCBI_API_KEY='...'

EXPECTED STEP06 INPUT
---------------------
<root>/06_finemapping/<phenotype_slug>/<ancestry_slug>/<study>/
    <study>_finemapped_variants.tsv.gz

Required columns:
    LOCUS_ID
    CHR
    REFERENCE_POS
    REFERENCE_REF
    REFERENCE_ALT
    PIP

Optional useful columns:
    rsID
    SNPID
    REFERENCE_ID
    IN_95_CREDIBLE_SET
    CREDIBLE_SET
    P
    BETA
    SE
    Z
    Z_REF

NORMAL USAGE
------------
1. Plan:
    python Step07_API_Distributed_Annotation.py \
        --phenotype "parkinson's disease" \
        --ancestry EUR

2. Set SCREEN key:
    export SCREEN_API_KEY='YOUR_KEY'

3. Submit the generated master submission script:
    bash Step07_Submit_API_parkinson_s_disease_european.sh

Or plan + submit directly:
    python Step07_API_Distributed_Annotation.py \
        --phenotype "parkinson's disease" \
        --ancestry EUR \
        --submit

The generated submission script submits all provider arrays and then a merge
job with afterok dependencies.

DIRECT WORKER MODE
------------------
Normally SLURM invokes this automatically:
    python Step07_API_Distributed_Annotation.py \
        --worker \
        --provider ensembl_vep \
        --manifest /path/to/ensembl_vep_manifest.tsv \
        --task-id 1

MERGE MODE
----------
Normally SLURM invokes this automatically:
    python Step07_API_Distributed_Annotation.py \
        --merge \
        --plan-json /path/to/step07_api_plan.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


VERSION = "5.1.0-api-distributed-qc"
ASSEMBLY = "GRCh38"
SPECIES = "homo_sapiens"

ENSEMBL_REST = "https://rest.ensembl.org"
SCREEN_GRAPHQL = "https://screen.api.wenglab.org/graphql"
REGULOMEDB_API = "https://api.regulomedb.org/search"
REGULOMEDB_FALLBACK = "https://regulomedb.org/regulome-search/"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

DEFAULT_BATCHES = {
    "ensembl_vep": 100,
    "ensembl_variation": 150,
    "screen": 50,
    "regulomedb": 25,
    "clinvar": 20,
}

DEFAULT_MAX_PARALLEL = {
    "ensembl_vep": 5,
    "ensembl_variation": 5,
    "screen": 3,
    "regulomedb": 3,
    "clinvar": 1,
}

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

IMPACT_RANK = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "MODIFIER": 3}

CONSEQUENCE_RANK = {
    "transcript_ablation": 0,
    "splice_acceptor_variant": 1,
    "splice_donor_variant": 2,
    "stop_gained": 3,
    "frameshift_variant": 4,
    "stop_lost": 5,
    "start_lost": 6,
    "transcript_amplification": 7,
    "inframe_insertion": 8,
    "inframe_deletion": 9,
    "missense_variant": 10,
    "protein_altering_variant": 11,
    "splice_region_variant": 12,
    "incomplete_terminal_codon_variant": 13,
    "start_retained_variant": 14,
    "stop_retained_variant": 15,
    "synonymous_variant": 16,
    "coding_sequence_variant": 17,
    "mature_miRNA_variant": 18,
    "5_prime_UTR_variant": 19,
    "3_prime_UTR_variant": 20,
    "non_coding_transcript_exon_variant": 21,
    "intron_variant": 22,
    "NMD_transcript_variant": 23,
    "non_coding_transcript_variant": 24,
    "upstream_gene_variant": 25,
    "downstream_gene_variant": 26,
    "TFBS_ablation": 27,
    "TFBS_amplification": 28,
    "TF_binding_site_variant": 29,
    "regulatory_region_ablation": 30,
    "regulatory_region_amplification": 31,
    "feature_elongation": 32,
    "regulatory_region_variant": 33,
    "feature_truncation": 34,
    "intergenic_variant": 35,
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 112)
    print(text)
    print("=" * 112)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    x = str(value).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def slugify(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = normalize_text(value)
    if key not in ANCESTRY_ALIASES:
        raise ValueError(
            f"Unsupported ancestry {value!r}. Use EUR, AFR, EAS, SAS or AMR."
        )
    return ANCESTRY_ALIASES[key]


def safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(value)


def join_unique(values: Iterable[Any], sep: str = ";", limit: int | None = None) -> str:
    seen = set()
    out = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        for item in items:
            if item is None:
                continue
            s = str(item).strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
                if limit is not None and len(out) >= limit:
                    return sep.join(out)
    return sep.join(out)


def primary_rsid(*values: Any) -> str:
    for value in values:
        m = re.search(r"\brs\d+\b", str(value), flags=re.I)
        if m:
            return m.group(0).lower()
    return ""


def variant_key(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    c = str(chrom).replace("chr", "").replace("CHR", "").strip()
    return f"{c}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"


def functional_category(consequences: Iterable[str], biotype: str = "") -> str:
    terms = {str(x).lower() for x in consequences if str(x).strip()}
    bt = str(biotype).lower()

    if any(
        t in terms
        for t in {
            "splice_acceptor_variant",
            "splice_donor_variant",
            "splice_region_variant",
            "splice_donor_5th_base_variant",
            "splice_donor_region_variant",
            "splice_polypyrimidine_tract_variant",
        }
    ):
        return "SPLICING"

    if any(
        t in terms
        for t in {
            "transcript_ablation",
            "stop_gained",
            "frameshift_variant",
            "stop_lost",
            "start_lost",
            "transcript_amplification",
            "inframe_insertion",
            "inframe_deletion",
            "missense_variant",
            "protein_altering_variant",
            "coding_sequence_variant",
            "synonymous_variant",
            "start_retained_variant",
            "stop_retained_variant",
            "incomplete_terminal_codon_variant",
        }
    ):
        return "CODING"

    if "5_prime_utr_variant" in terms or "3_prime_utr_variant" in terms:
        return "UTR"

    if any("regulatory" in t or "tf_binding" in t or "tfbs" in t for t in terms):
        return "REGULATORY"

    if (
        any("non_coding_transcript" in t for t in terms)
        or any(
            x in bt
            for x in (
                "lncrna",
                "lincrna",
                "mirna",
                "snrna",
                "snorna",
                "rrna",
                "ncrna",
                "antisense",
                "processed_transcript",
            )
        )
    ):
        return "NONCODING_RNA"

    if "intron_variant" in terms:
        return "INTRONIC"
    if "upstream_gene_variant" in terms:
        return "UPSTREAM"
    if "downstream_gene_variant" in terms:
        return "DOWNSTREAM"
    if "intergenic_variant" in terms:
        return "INTERGENIC"
    return "OTHER"


def request_session() -> requests.Session:
    if requests is None:
        raise RuntimeError(
            "Python requests is required. Install it in your pipeline environment."
        )
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": f"GWAS2m-Step07/{VERSION}",
        }
    )
    return s


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    tries: int = 6,
    timeout: int = 180,
    **kwargs: Any,
) -> requests.Response:
    last: Exception | None = None

    for attempt in range(tries):
        try:
            r = session.request(method, url, timeout=timeout, **kwargs)

            if r.status_code == 429:
                retry = r.headers.get("Retry-After")
                try:
                    wait = float(retry) if retry else min(120.0, 2.0 ** attempt)
                except Exception:
                    wait = min(120.0, 2.0 ** attempt)
                print(f"[429] {url} — sleeping {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue

            if r.status_code in {500, 502, 503, 504}:
                wait = min(60.0, 2.0 ** attempt)
                print(
                    f"[HTTP {r.status_code}] transient server error — sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r

        except Exception as exc:
            last = exc
            if attempt == tries - 1:
                break
            wait = min(60.0, 2.0 ** attempt)
            print(
                f"[WARN] request failed attempt {attempt + 1}/{tries}: "
                f"{type(exc).__name__}: {exc}; sleeping {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {tries} attempts: {url}\n{last}")


def atomic_write_gzip_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_tsv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    df.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, path)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", low_memory=False)


def shell(cmd: list[str]) -> str:
    cmd = [str(x) for x in cmd]
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {p.returncode}: {' '.join(cmd)}"
        )
    return p.stdout


# =============================================================================
# INPUT / PLANNING
# =============================================================================

def load_finemapped(path: Path) -> pd.DataFrame:
    x = read_tsv(path)

    required = [
        "LOCUS_ID",
        "CHR",
        "REFERENCE_POS",
        "REFERENCE_REF",
        "REFERENCE_ALT",
        "PIP",
    ]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")

    x["CHR"] = (
        x["CHR"]
        .astype(str)
        .str.replace(r"^chr", "", regex=True, case=False)
        .str.strip()
    )
    x["CHR"] = pd.to_numeric(x["CHR"], errors="coerce")
    x["REFERENCE_POS"] = pd.to_numeric(x["REFERENCE_POS"], errors="coerce")
    x["PIP"] = pd.to_numeric(x["PIP"], errors="coerce")

    x = x.dropna(
        subset=["CHR", "REFERENCE_POS", "REFERENCE_REF", "REFERENCE_ALT", "PIP"]
    ).copy()
    x = x[x["CHR"].between(1, 22, inclusive="both")].copy()

    x["CHR"] = x["CHR"].astype(int)
    x["REFERENCE_POS"] = x["REFERENCE_POS"].astype(int)
    x["REFERENCE_REF"] = x["REFERENCE_REF"].astype(str).str.upper().str.strip()
    x["REFERENCE_ALT"] = x["REFERENCE_ALT"].astype(str).str.upper().str.strip()
    x = x[(x["REFERENCE_REF"] != "") & (x["REFERENCE_ALT"] != "")].copy()

    x["VARIANT_KEY"] = [
        variant_key(c, p, r, a)
        for c, p, r, a in zip(
            x["CHR"],
            x["REFERENCE_POS"],
            x["REFERENCE_REF"],
            x["REFERENCE_ALT"],
        )
    ]

    def get_rsid(row: pd.Series) -> str:
        return primary_rsid(
            row.get("rsID", ""),
            row.get("SNPID", ""),
            row.get("REFERENCE_ID", ""),
        )

    x["PRIMARY_RSID"] = x.apply(get_rsid, axis=1)

    if "IN_95_CREDIBLE_SET" in x.columns:
        s = x["IN_95_CREDIBLE_SET"].fillna("").astype(str).str.lower().str.strip()
        x["IN_95_CREDIBLE_SET"] = s.isin({"1", "true", "yes", "y", "t"})
    elif "CREDIBLE_SET" in x.columns:
        s = x["CREDIBLE_SET"].fillna("").astype(str).str.lower().str.strip()
        x["IN_95_CREDIBLE_SET"] = ~s.isin({"", ".", "na", "nan", "none", "false", "0"})
    else:
        x["IN_95_CREDIBLE_SET"] = False

    x["PIP_GE_0_01"] = x["PIP"] >= 0.01
    x["PIP_GE_0_10"] = x["PIP"] >= 0.10
    x["PIP_GE_0_50"] = x["PIP"] >= 0.50
    x["PIP_GE_0_90"] = x["PIP"] >= 0.90

    return x


def discover_studies(
    root: Path,
    phenotype: str,
    ancestry_label: str,
    allow_partial: bool,
) -> tuple[list[tuple[str, Path]], list[dict[str, Any]]]:
    phenotype_slug = slugify(phenotype)
    ancestry_slug = slugify(ancestry_label)
    fine_root = root / "06_finemapping" / phenotype_slug / ancestry_slug

    if not fine_root.exists():
        raise FileNotFoundError(f"Step06 directory not found: {fine_root}")

    studies: list[tuple[str, Path]] = []
    excluded: list[dict[str, Any]] = []

    dirs = sorted(p for p in fine_root.iterdir() if p.is_dir())

    for d in dirs:
        accession = d.name
        fine_file = d / f"{accession}_finemapped_variants.tsv.gz"
        summary_file = d / "finemapping_summary.json"

        reason = ""

        if summary_file.exists():
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

            status = str(summary.get("STATUS", "")).upper()
            n_loci = safe_int(summary.get("N_LOCI"), 0)
            n_success = safe_int(summary.get("N_LOCI_SUCCESS"), 0)
            n_failed = safe_int(summary.get("N_LOCI_FAILED"), 0)

            if status == "NO_SIGNIFICANT_LOCI" or n_loci == 0:
                reason = "No significant loci"
            elif n_success == 0 and n_loci > 0:
                reason = "No successfully fine-mapped loci"
            elif n_failed > 0 and not allow_partial:
                reason = f"Partial Step06: {n_success} success / {n_failed} failed"

        if not fine_file.exists() or fine_file.stat().st_size == 0:
            reason = reason or "Fine-mapped variant file missing/empty"

        if reason:
            excluded.append({"STUDY_ACCESSION": accession, "REASON": reason})
        else:
            studies.append((accession, fine_file))

    if not studies:
        raise RuntimeError(f"No eligible Step06 studies found under {fine_root}")

    return studies, excluded


def build_variant_universe(
    studies: list[tuple[str, Path]],
    phenotype: str,
    ancestry_code: str,
    ancestry_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maps: list[pd.DataFrame] = []

    for accession, path in studies:
        x = load_finemapped(path)
        x["STUDY_ACCESSION"] = accession
        x["PHENOTYPE"] = phenotype
        x["ANCESTRY_CODE"] = ancestry_code
        x["ANCESTRY_LABEL"] = ancestry_label

        keep = [
            c
            for c in [
                "STUDY_ACCESSION",
                "PHENOTYPE",
                "ANCESTRY_CODE",
                "ANCESTRY_LABEL",
                "LOCUS_ID",
                "VARIANT_KEY",
                "PRIMARY_RSID",
                "CHR",
                "REFERENCE_POS",
                "REFERENCE_REF",
                "REFERENCE_ALT",
                "PIP",
                "IN_95_CREDIBLE_SET",
                "PIP_GE_0_01",
                "PIP_GE_0_10",
                "PIP_GE_0_50",
                "PIP_GE_0_90",
                "REFERENCE_ID",
                "rsID",
                "SNPID",
                "P",
                "BETA",
                "SE",
                "Z",
                "Z_REF",
            ]
            if c in x.columns
        ]
        maps.append(x[keep].copy())

    study_map = pd.concat(maps, ignore_index=True)

    study_map = study_map.sort_values(
        ["STUDY_ACCESSION", "LOCUS_ID", "VARIANT_KEY", "PIP"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    study_map = study_map.drop_duplicates(
        ["STUDY_ACCESSION", "LOCUS_ID", "VARIANT_KEY"], keep="first"
    ).reset_index(drop=True)

    universe = study_map.sort_values(
        ["VARIANT_KEY", "PIP"],
        ascending=[True, False],
        kind="stable",
    ).drop_duplicates("VARIANT_KEY", keep="first").copy()

    universe = universe[
        [
            "VARIANT_KEY",
            "PRIMARY_RSID",
            "CHR",
            "REFERENCE_POS",
            "REFERENCE_REF",
            "REFERENCE_ALT",
        ]
    ].reset_index(drop=True)

    universe["API_ID"] = [
        f"GWAS2M_{i:09d}" for i in range(1, len(universe) + 1)
    ]

    return universe, study_map


def build_rsid_universe(universe: pd.DataFrame) -> pd.DataFrame:
    x = universe[["VARIANT_KEY", "PRIMARY_RSID"]].copy()
    x["PRIMARY_RSID"] = x["PRIMARY_RSID"].fillna("").astype(str).str.lower()
    x = x[x["PRIMARY_RSID"].str.match(r"^rs\d+$", na=False)].copy()
    x = x.sort_values(["PRIMARY_RSID", "VARIANT_KEY"], kind="stable")
    return x.drop_duplicates("PRIMARY_RSID", keep="first").reset_index(drop=True)


def create_batch_manifest(
    input_file: Path,
    n_rows: int,
    batch_size: int,
    provider: str,
    output_dir: Path,
    manifest_file: Path,
) -> pd.DataFrame:
    rows = []

    for i, start in enumerate(range(0, n_rows, batch_size), start=1):
        stop = min(start + batch_size, n_rows)
        rows.append(
            {
                "TASK_ID": i,
                "PROVIDER": provider,
                "INPUT_FILE": str(input_file.resolve()),
                "START_ROW": start,
                "END_ROW": stop,
                "N_RECORDS": stop - start,
                "OUTPUT_FILE": str(
                    (output_dir / f"batch_{i:06d}.tsv.gz").resolve()
                ),
            }
        )

    manifest = pd.DataFrame(
        rows,
        columns=[
            "TASK_ID",
            "PROVIDER",
            "INPUT_FILE",
            "START_ROW",
            "END_ROW",
            "N_RECORDS",
            "OUTPUT_FILE",
        ],
    )
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_file, sep="\t", index=False)
    return manifest


def slurm_walltime_for_partition(
    partition: str | None,
    requested: str | None,
    *,
    merge: bool = False,
) -> str:
    """Resolve a safe walltime that matches the UQ partition policy.

    Known project policy:
      * general: maximum 24 hours
      * asher: unlimited; Slurm represents this as --time=0

    An explicitly supplied --time/--merge-time always wins, except that a
    general-partition request above 24 h is rejected early instead of letting
    sbatch fail later with an opaque error.
    """
    part = str(partition or "general").strip().lower()

    if requested is not None and str(requested).strip():
        value = str(requested).strip()
        if part == "general":
            # Accept HH:MM:SS or D-HH:MM:SS and enforce <=24 h.
            text = value
            days = 0
            if "-" in text:
                d, text = text.split("-", 1)
                days = int(d)
            fields = text.split(":")
            if len(fields) not in (2, 3):
                raise ValueError(
                    f"Invalid SLURM time {value!r}; use HH:MM:SS or D-HH:MM:SS"
                )
            if len(fields) == 2:
                hours, minutes = map(int, fields)
                seconds = 0
            else:
                hours, minutes, seconds = map(int, fields)
            total = days * 86400 + hours * 3600 + minutes * 60 + seconds
            if total > 24 * 3600:
                raise ValueError(
                    f"general partition is limited to 24 hours; requested {value}. "
                    "Use --partition asher for unlimited walltime."
                )
        return value

    if part == "general":
        return "24:00:00"
    if part == "asher":
        return "0"
    return "02:00:00" if merge else "04:00:00"


def sbatch_header(
    job_name: str,
    log_file: Path,
    array_spec: str | None,
    cpus: int,
    memory: str,
    walltime: str,
    partition: str | None,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={log_file}",
        f"#SBATCH --error={log_file.with_suffix('.err')}",
        "#SBATCH --export=ALL",
    ]
    if array_spec:
        lines.append(f"#SBATCH --array={array_spec}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    return "\n".join(lines)


def write_provider_sbatch(
    script_path: Path,
    python_exe: str,
    this_script: Path,
    provider: str,
    manifest: Path,
    n_tasks: int,
    max_parallel: int,
    root: Path,
    log_root: Path,
    args: argparse.Namespace,
) -> None:
    if n_tasks <= 0:
        return

    array = f"1-{n_tasks}"
    if max_parallel > 0:
        array += f"%{max_parallel}"

    log = log_root / f"{provider}.%A_%a.out"

    header = sbatch_header(
        job_name=f"S7_{provider}"[:100],
        log_file=log,
        array_spec=array,
        cpus=args.cpus,
        memory=args.memory,
        walltime=slurm_walltime_for_partition(args.partition, args.time, merge=False),
        partition=args.partition,
    )

    body = f"""
set -euo pipefail
cd {shlex.quote(str(root))}

echo "======================================================================"
echo "STEP07 API PROVIDER: {provider}"
echo "JOB ID            : $SLURM_JOB_ID"
echo "ARRAY TASK        : $SLURM_ARRAY_TASK_ID"
echo "HOST              : $(hostname)"
echo "DATE              : $(date -Is)"
echo "======================================================================"

{shlex.quote(python_exe)} {shlex.quote(str(this_script))} \
  --worker \
  --provider {shlex.quote(provider)} \
  --manifest {shlex.quote(str(manifest))} \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --screen-detail-workers {args.screen_detail_workers} \
  --regulomedb-workers {args.regulomedb_workers} \
  {"--force" if args.force else ""}
"""
    script_path.write_text(header + "\n" + body, encoding="utf-8")
    script_path.chmod(0o755)


def write_merge_sbatch(
    script_path: Path,
    python_exe: str,
    this_script: Path,
    plan_json: Path,
    root: Path,
    log_root: Path,
    args: argparse.Namespace,
) -> None:
    log = log_root / "merge.%j.out"
    header = sbatch_header(
        job_name="S7_API_MERGE",
        log_file=log,
        array_spec=None,
        cpus=max(1, args.cpus),
        memory=args.merge_memory,
        walltime=slurm_walltime_for_partition(args.partition, args.merge_time, merge=True),
        partition=args.partition,
    )
    body = f"""
set -euo pipefail
cd {shlex.quote(str(root))}

{shlex.quote(python_exe)} {shlex.quote(str(this_script))} \
  --merge \
  --plan-json {shlex.quote(str(plan_json))}
"""
    script_path.write_text(header + "\n" + body, encoding="utf-8")
    script_path.chmod(0o755)


def write_submit_script(
    path: Path,
    providers: list[tuple[str, Path, int]],
    merge_sbatch: Path,
    screen_required: bool,
) -> None:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
    ]

    if screen_required:
        lines.extend(
            [
                'if [[ -z "${SCREEN_API_KEY:-}" ]]; then',
                '  echo "ERROR: SCREEN_API_KEY is not set." >&2',
                '  echo "Create a SCREEN API key and run:" >&2',
                '  echo "  export SCREEN_API_KEY=YOUR_KEY" >&2',
                "  exit 2",
                "fi",
                "",
            ]
        )

    lines.extend(
        [
            'echo "Submitting Step07 API annotation arrays..."',
            "deps=()",
        ]
    )

    for provider, sbatch_file, n_tasks in providers:
        if n_tasks <= 0:
            continue
        var = "jid_" + provider
        lines.extend(
            [
                f'{var}=$(sbatch --parsable {shlex.quote(str(sbatch_file))})',
                f'echo "{provider}: ${var}"',
                f'deps+=("${var}")',
            ]
        )

    lines.extend(
        [
            "",
            'if [[ ${#deps[@]} -eq 0 ]]; then',
            '  echo "No provider arrays were generated." >&2',
            "  exit 3",
            "fi",
            'dep=$(IFS=:; echo "${deps[*]}")',
            f'jid_merge=$(sbatch --parsable --dependency=afterok:${{dep}} {shlex.quote(str(merge_sbatch))})',
            'echo "merge: ${jid_merge}"',
            'echo "All jobs submitted."',
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


# =============================================================================
# ENSEMBL VEP API
# =============================================================================

def transcript_rank(tc: dict[str, Any]) -> tuple:
    mane = 0 if tc.get("mane_select") else 1
    canonical = 0 if tc.get("canonical") in (1, True, "1") else 1
    impact = IMPACT_RANK.get(str(tc.get("impact", "")).upper(), 9)

    consequences = tc.get("consequence_terms", []) or []
    c_rank = min(
        [CONSEQUENCE_RANK.get(str(c), 99) for c in consequences] or [99]
    )

    appris = str(tc.get("appris", "") or "")
    appris_rank = 0 if "principal" in appris.lower() else 1

    return mane, canonical, appris_rank, impact, c_rank


def collect_key_values(obj: Any, wanted_tokens: Iterable[str]) -> dict[str, list[str]]:
    wanted = tuple(str(x).lower() for x in wanted_tokens)
    found: dict[str, list[str]] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if any(token in lk for token in wanted):
                    if not isinstance(v, (dict, list)):
                        found.setdefault(str(k), [])
                        sv = str(v)
                        if sv not in found[str(k)]:
                            found[str(k)].append(sv)
                    elif isinstance(v, list) and all(
                        not isinstance(y, (dict, list)) for y in v
                    ):
                        found.setdefault(str(k), [])
                        for y in v:
                            sy = str(y)
                            if sy not in found[str(k)]:
                                found[str(k)].append(sy)
                walk(v)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(obj)
    return found


def extract_ensembl_vep_record(item: dict[str, Any], api_id: str, vkey: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "VARIANT_KEY": vkey,
        "API_ID": api_id,
        "ENSEMBL_VEP_STATUS": "OK",
        "ENSEMBL_VEP_MOST_SEVERE": item.get("most_severe_consequence", ""),
        "ENSEMBL_VEP_ID": item.get("id", ""),
        "ENSEMBL_VEP_ASSEMBLY": item.get("assembly_name", ""),
        "ENSEMBL_VEP_SEQ_REGION": item.get("seq_region_name", ""),
        "ENSEMBL_VEP_START": item.get("start", ""),
        "ENSEMBL_VEP_END": item.get("end", ""),
        "ENSEMBL_VEP_ALLELE_STRING": item.get("allele_string", ""),
    }

    transcripts = [
        x
        for x in (item.get("transcript_consequences", []) or [])
        if isinstance(x, dict)
    ]

    all_consequences = []
    all_genes = []
    all_symbols = []
    all_transcripts = []
    all_biotypes = []

    for tc in transcripts:
        all_consequences.extend(tc.get("consequence_terms", []) or [])
        all_genes.append(tc.get("gene_id", ""))
        all_symbols.append(tc.get("gene_symbol", ""))
        all_transcripts.append(tc.get("transcript_id", ""))
        all_biotypes.append(tc.get("biotype", ""))

    out["ENSEMBL_ALL_CONSEQUENCES"] = join_unique(all_consequences)
    out["ENSEMBL_ALL_GENES"] = join_unique(all_genes)
    out["ENSEMBL_ALL_SYMBOLS"] = join_unique(all_symbols)
    out["ENSEMBL_ALL_TRANSCRIPTS"] = join_unique(all_transcripts)
    out["ENSEMBL_ALL_BIOTYPES"] = join_unique(all_biotypes)
    out["ENSEMBL_N_TRANSCRIPT_CONSEQUENCES"] = len(transcripts)

    best: dict[str, Any] = {}
    if transcripts:
        best = sorted(transcripts, key=transcript_rank)[0]

    best_cons = best.get("consequence_terms", []) or []
    out["ENSEMBL_FUNCTIONAL_CATEGORY"] = functional_category(
        best_cons, best.get("biotype", "")
    )
    out["ENSEMBL_BEST_CONSEQUENCE"] = join_unique(best_cons)
    out["ENSEMBL_BEST_IMPACT"] = best.get("impact", "")
    out["ENSEMBL_GENE_ID"] = best.get("gene_id", "")
    out["ENSEMBL_GENE_SYMBOL"] = best.get("gene_symbol", "")
    out["ENSEMBL_TRANSCRIPT_ID"] = best.get("transcript_id", "")
    out["ENSEMBL_TRANSCRIPT_BIOTYPE"] = best.get("biotype", "")
    out["ENSEMBL_EXON"] = best.get("exon", "")
    out["ENSEMBL_INTRON"] = best.get("intron", "")
    out["ENSEMBL_HGVSC"] = best.get("hgvsc", "")
    out["ENSEMBL_HGVSP"] = best.get("hgvsp", "")
    out["ENSEMBL_PROTEIN_ID"] = best.get("protein_id", "")
    out["ENSEMBL_PROTEIN_START"] = best.get("protein_start", "")
    out["ENSEMBL_PROTEIN_END"] = best.get("protein_end", "")
    out["ENSEMBL_AMINO_ACIDS"] = best.get("amino_acids", "")
    out["ENSEMBL_CODONS"] = best.get("codons", "")
    out["ENSEMBL_CANONICAL"] = best.get("canonical", "")
    out["ENSEMBL_MANE_SELECT"] = best.get("mane_select", "")
    out["ENSEMBL_MANE_PLUS_CLINICAL"] = best.get("mane_plus_clinical", "")
    out["ENSEMBL_APPRIS"] = best.get("appris", "")
    out["ENSEMBL_TSL"] = best.get("tsl", "")
    out["ENSEMBL_SIFT_PREDICTION"] = best.get("sift_prediction", "")
    out["ENSEMBL_SIFT_SCORE"] = best.get("sift_score", "")
    out["ENSEMBL_POLYPHEN_PREDICTION"] = best.get("polyphen_prediction", "")
    out["ENSEMBL_POLYPHEN_SCORE"] = best.get("polyphen_score", "")

    # High-value VEP plugin/prediction fields.
    wanted = (
        "alphamissense",
        "am_pathogenicity",
        "am_class",
        "cadd",
        "revel",
        "protvar",
        "spliceai",
        "blosum",
        "clinpred",
        "eve",
        "loeuf",
        "mavedb",
        "opentarget",
        "swissprot",
        "uniprot",
        "uniparc",
        "domain",
    )
    kv = collect_key_values(transcripts, wanted)
    for key, values in sorted(kv.items()):
        col = "ENSEMBL_PLUGIN_" + re.sub(r"[^A-Za-z0-9]+", "_", key).upper()
        out[col] = join_unique(values)

    # Co-located variants give rsIDs, ClinVar significance and frequency evidence.
    colocated = [
        x for x in (item.get("colocated_variants", []) or []) if isinstance(x, dict)
    ]
    out["ENSEMBL_COLOCATED_IDS"] = join_unique(
        [x.get("id", "") for x in colocated]
    )
    out["ENSEMBL_CLINICAL_SIGNIFICANCE"] = join_unique(
        [
            sig
            for x in colocated
            for sig in (x.get("clin_sig", []) or [])
        ]
    )
    out["ENSEMBL_FREQUENCIES_JSON"] = join_unique(
        [
            compact_json(x.get("frequencies", {}))
            for x in colocated
            if x.get("frequencies")
        ]
    )
    out["ENSEMBL_PHENOTYPE_VARIANTS"] = join_unique(
        [
            x.get("id", "")
            for x in colocated
            if x.get("phenotype_or_disease") in (1, True, "1")
        ]
    )

    regs = [
        x
        for x in (item.get("regulatory_feature_consequences", []) or [])
        if isinstance(x, dict)
    ]
    out["ENSEMBL_REGULATORY_FEATURE_IDS"] = join_unique(
        [x.get("regulatory_feature_id", "") for x in regs]
    )
    out["ENSEMBL_REGULATORY_CONSEQUENCES"] = join_unique(
        [c for x in regs for c in (x.get("consequence_terms", []) or [])]
    )
    out["ENSEMBL_REGULATORY_BIOTYPES"] = join_unique(
        [x.get("biotype", "") for x in regs]
    )

    return out


def worker_ensembl_vep(batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    session = request_session()
    session.headers.update({"Content-Type": "application/json"})

    payload_variants = []
    id_to_key = {}

    for r in batch.itertuples(index=False):
        api_id = str(r.API_ID)
        id_to_key[api_id] = str(r.VARIANT_KEY)
        payload_variants.append(
            f"{int(r.CHR)} {int(r.REFERENCE_POS)} {api_id} "
            f"{r.REFERENCE_REF} {r.REFERENCE_ALT} . . ."
        )

    params = {
        "hgvs": 1,
        "canonical": 1,
        "mane": 1,
        "numbers": 1,
        "protein": 1,
        "domains": 1,
        "uniprot": 1,
        "symbol": 1,
        "biotype": 1,
        "variant_class": 1,
        "regulatory": 1,
        "Phenotypes": 1,
        "AlphaMissense": 1,
        "CADD": "snv_indels",
        "REVEL": 1,
        "ProtVar": 1,
        "SpliceAI": 2,
        "Blosum62": 1,
        "ClinPred": 1,
        "EVE": 1,
        "LOEUF": 1,
        "MaveDB": 1,
        "OpenTargets": 1,
    }

    rows: list[dict[str, Any]] = []

    try:
        r = request_with_retry(
            session,
            "POST",
            f"{ENSEMBL_REST}/vep/{SPECIES}/region",
            params=params,
            json={"variants": payload_variants},
        )
        data = r.json()

        by_api_id: dict[str, dict[str, Any]] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                tokens = str(item.get("input", "")).split()
                api_id = tokens[2] if len(tokens) >= 3 else ""
                if api_id:
                    by_api_id[api_id] = item

        for rec in batch.itertuples(index=False):
            api_id = str(rec.API_ID)
            item = by_api_id.get(api_id)

            if item:
                rows.append(
                    extract_ensembl_vep_record(
                        item, api_id, str(rec.VARIANT_KEY)
                    )
                )
            else:
                rows.append(
                    {
                        "VARIANT_KEY": rec.VARIANT_KEY,
                        "API_ID": api_id,
                        "ENSEMBL_VEP_STATUS": "NO_HIT",
                    }
                )

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        for rec in batch.itertuples(index=False):
            rows.append(
                {
                    "VARIANT_KEY": rec.VARIANT_KEY,
                    "API_ID": rec.API_ID,
                    "ENSEMBL_VEP_STATUS": "API_ERROR",
                    "ENSEMBL_VEP_ERROR": err,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# ENSEMBL VARIATION API
# =============================================================================

def worker_ensembl_variation(
    batch: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    session = request_session()
    session.headers.update({"Content-Type": "application/json"})

    ids = batch["PRIMARY_RSID"].dropna().astype(str).str.lower().tolist()
    rows: list[dict[str, Any]] = []

    if not ids:
        return pd.DataFrame(columns=["PRIMARY_RSID", "ENSEMBL_VARIATION_STATUS"])

    try:
        r = request_with_retry(
            session,
            "POST",
            f"{ENSEMBL_REST}/variation/{SPECIES}",
            params={"pops": 1, "phenotypes": 1},
            json={"ids": ids},
        )
        data = r.json()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(
            [
                {
                    "PRIMARY_RSID": rs,
                    "ENSEMBL_VARIATION_STATUS": "API_ERROR",
                    "ENSEMBL_VARIATION_ERROR": err,
                }
                for rs in ids
            ]
        )

    for rs in ids:
        obj = data.get(rs) if isinstance(data, dict) else None

        if not isinstance(obj, dict) or not obj:
            rows.append(
                {
                    "PRIMARY_RSID": rs,
                    "ENSEMBL_VARIATION_STATUS": "NO_HIT",
                }
            )
            continue

        pops = obj.get("populations", []) or []
        phen = obj.get("phenotypes", []) or []

        rows.append(
            {
                "PRIMARY_RSID": rs,
                "ENSEMBL_VARIATION_STATUS": "OK",
                "ENSEMBL_VARIATION_AMBIGUITY": obj.get("ambiguity", ""),
                "ENSEMBL_VARIATION_ANCESTRAL_ALLELE": obj.get(
                    "ancestral_allele", ""
                ),
                "ENSEMBL_VARIATION_MINOR_ALLELE": obj.get("minor_allele", ""),
                "ENSEMBL_VARIATION_MAF": obj.get(
                    "MAF", obj.get("minor_allele_freq", "")
                ),
                "ENSEMBL_VARIATION_CLINICAL_SIGNIFICANCE": join_unique(
                    obj.get("clinical_significance", []) or []
                ),
                "ENSEMBL_VARIATION_PHENOTYPES_JSON": compact_json(phen),
                "ENSEMBL_VARIATION_POPULATIONS_JSON": compact_json(pops),
                "ENSEMBL_VARIATION_MAPPINGS_JSON": compact_json(
                    obj.get("mappings", []) or []
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SCREEN / ENCODE GRAPHQL
# =============================================================================

def screen_post(
    session: requests.Session,
    query: str,
    api_key: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    r = request_with_retry(
        session,
        "POST",
        SCREEN_GRAPHQL,
        headers=headers,
        json={"query": query},
    )
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(
            "SCREEN GraphQL error: "
            + compact_json(payload["errors"])[:5000]
        )
    return payload.get("data", {}) or {}


def screen_coords_literal(batch: pd.DataFrame) -> str:
    coords = []
    for r in batch.itertuples(index=False):
        pos = int(r.REFERENCE_POS)
        coords.append(
            f'{{chromosome:"chr{int(r.CHR)}",start:{max(0, pos - 1)},end:{pos}}}'
        )
    return ",".join(coords)


def screen_query_overlaps(
    session: requests.Session,
    batch: pd.DataFrame,
    api_key: str,
) -> list[dict[str, Any]]:
    coords = screen_coords_literal(batch)
    query = f"""
    query {{
      getmaxZScoresQuery(
        assembly: "grch38",
        coordinates: [{coords}]
      ) {{
        accession
        chromosome
        start
        stop
        ccre_group
        atac_max_zscore
        ctcf_max_zscore
        dnase_max_zscore
        h3k27ac_max_zscore
        h3k4me3_max_zscore
        nearestgenes {{
          distance
          gene
        }}
      }}
    }}
    """
    data = screen_post(session, query, api_key)
    x = data.get("getmaxZScoresQuery", []) or []
    return x if isinstance(x, list) else []


def screen_query_ccre_detail(
    session: requests.Session,
    accession: str,
    api_key: str,
) -> dict[str, Any]:
    # A single GraphQL request retrieves multiple rich evidence layers for one cCRE.
    query = f"""
    query {{
      signals: getcCREZScoresQuery(
        assembly: "grch38",
        accession: ["{accession}"],
        include_biosample_details: true
      ) {{
        chromosome
        start
        stop
        accession
        atac_max_zscore
        ctcf_max_zscore
        dnase_max_zscore
        h3k27ac_max_zscore
        h3k4me3_max_zscore
        zscores
      }}

      linked: linkedGenesQuery(
        assembly: "grch38",
        accession: ["{accession}"]
      ) {{
        accession
        p_val
        gene
        geneid
        genetype
        method
        effectsize
        assay
        celltype
        experiment_accession
        tissue
        variantid
        source
        slope
        score
        displayname
      }}

      computational: ComputationalGeneLinksQuery(
        accession: ["{accession}"]
      ) {{
        genename
        accession
        geneid
        genetype
        method
        celltype
        score
        methodregion
        fileaccession
      }}

      tfpeaks: getcCRERPeaksQuery(
        accession: ["{accession}"]
      ) {{
        rpeak_id
        rpeak_chromosome
        rpeak_start
        rpeak_stop
        tf
        metadata
        strand
      }}

      silencers: silencersQuery(
        accession: ["{accession}"]
      ) {{
        silencer_studies
      }}

      dynamic: dynamicEnhancersQuery(
        accession: ["{accession}"]
      ) {{
        celltype
        accession
      }}
    }}
    """
    return screen_post(session, query, api_key)


def summarize_screen_detail(detail: dict[str, Any], active_z: float) -> dict[str, Any]:
    linked = detail.get("linked", []) or []
    comp = detail.get("computational", []) or []
    tfpeaks = detail.get("tfpeaks", []) or []
    silencers = detail.get("silencers", []) or []
    dynamic = detail.get("dynamic", []) or []
    signals = detail.get("signals", []) or []

    experimental_genes = []
    linked_displaynames = []
    computational_genes = []
    celltypes = []
    tissues = []
    methods = []
    assays = []
    experiment_accessions = []
    file_accessions = []
    tfs = []
    active_biosamples = []
    active_tissues = []
    active_assays = []
    active_lifestages = []

    for x in linked:
        if not isinstance(x, dict):
            continue
        experimental_genes.append(x.get("gene", ""))
        linked_displaynames.append(x.get("displayname", ""))
        celltypes.append(x.get("celltype", ""))
        tissues.append(x.get("tissue", ""))
        methods.append(x.get("method", ""))
        assays.append(x.get("assay", ""))
        experiment_accessions.append(x.get("experiment_accession", ""))

    for x in comp:
        if not isinstance(x, dict):
            continue
        computational_genes.append(x.get("genename", ""))
        celltypes.append(x.get("celltype", ""))
        methods.append(x.get("method", ""))
        file_accessions.append(x.get("fileaccession", ""))

    for x in tfpeaks:
        if isinstance(x, dict):
            tfs.append(x.get("tf", ""))

    signal_rows = []
    for s in signals:
        if not isinstance(s, dict):
            continue
        z = s.get("zscores", []) or []
        if isinstance(z, str):
            try:
                z = json.loads(z)
            except Exception:
                z = []

        if isinstance(z, list):
            for row in z:
                if not isinstance(row, (list, tuple)) or len(row) < 9:
                    continue
                signal_rows.append(row)

                score = safe_float(row[8])
                if not np.isfinite(score) or score < active_z:
                    continue

                # SCREEN documentation:
                # [experiment_accession, file_accession, assay, biosample_value,
                #  biosample_name, organ/tissue, sampletype, lifestage, zscore]
                experiment_accessions.append(row[0])
                file_accessions.append(row[1])
                active_assays.append(row[2])
                active_biosamples.append(row[4])
                active_tissues.append(row[5])
                active_lifestages.append(row[7])

    return {
        "SCREEN_LINKED_GENES_EXPERIMENTAL": join_unique(experimental_genes),
        "SCREEN_LINKED_DISPLAY_NAMES": join_unique(linked_displaynames, limit=100),
        "SCREEN_LINKED_GENES_COMPUTATIONAL": join_unique(computational_genes),
        "SCREEN_CELL_TYPES": join_unique(celltypes, limit=100),
        "SCREEN_TISSUES": join_unique(tissues, limit=100),
        "SCREEN_LINK_METHODS": join_unique(methods),
        "SCREEN_LINK_ASSAYS": join_unique(assays),
        "SCREEN_EXPERIMENT_ACCESSIONS": join_unique(experiment_accessions, limit=200),
        "SCREEN_FILE_ACCESSIONS": join_unique(file_accessions, limit=200),
        "SCREEN_TFS": join_unique(tfs, limit=100),
        "SCREEN_ACTIVE_BIOSAMPLES": join_unique(active_biosamples, limit=100),
        "SCREEN_ACTIVE_TISSUES": join_unique(active_tissues, limit=100),
        "SCREEN_ACTIVE_ASSAYS": join_unique(active_assays),
        "SCREEN_ACTIVE_LIFESTAGES": join_unique(active_lifestages),
        "SCREEN_N_SIGNAL_ROWS": len(signal_rows),
        "SCREEN_SILENCER_JSON": compact_json(silencers),
        "SCREEN_DYNAMIC_ENHANCERS_JSON": compact_json(dynamic),
    }


def screen_cache_file(cache_root: Path, accession: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", accession)
    return cache_root / f"{safe}.json.gz"


def get_screen_detail_cached(
    accession: str,
    api_key: str,
    cache_root: Path,
    active_z: float,
    force: bool,
) -> tuple[str, dict[str, Any], str]:
    path = screen_cache_file(cache_root, accession)

    if path.exists() and not force:
        try:
            detail = read_gzip_json(path)
            return accession, summarize_screen_detail(detail, active_z), "CACHE"
        except Exception:
            pass

    session = request_session()
    try:
        detail = screen_query_ccre_detail(session, accession, api_key)
        atomic_write_gzip_json(path, detail)
        return accession, summarize_screen_detail(detail, active_z), "API"
    except Exception as exc:
        return accession, {}, f"API_ERROR:{type(exc).__name__}:{exc}"


def worker_screen(batch: pd.DataFrame, args: argparse.Namespace, output_file: Path) -> pd.DataFrame:
    api_key = os.environ.get("SCREEN_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame(
            [
                {
                    "VARIANT_KEY": r.VARIANT_KEY,
                    "SCREEN_STATUS": "SCREEN_API_KEY_NOT_SET",
                }
                for r in batch.itertuples(index=False)
            ]
        )

    session = request_session()

    try:
        ccres = screen_query_overlaps(session, batch, api_key)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(
            [
                {
                    "VARIANT_KEY": r.VARIANT_KEY,
                    "SCREEN_STATUS": "API_ERROR",
                    "SCREEN_ERROR": err,
                }
                for r in batch.itertuples(index=False)
            ]
        )

    # Determine which returned cCREs overlap each one-base variant interval.
    var_hits: dict[str, list[dict[str, Any]]] = {
        str(v): [] for v in batch["VARIANT_KEY"]
    }

    for c in ccres:
        if not isinstance(c, dict):
            continue

        chrom = str(c.get("chromosome", "")).replace("chr", "")
        start = safe_int(c.get("start"), -1)
        stop = safe_int(c.get("stop"), -1)

        if start < 0 or stop < 0:
            continue

        candidates = batch[
            (batch["CHR"].astype(int).astype(str) == chrom)
        ]

        for r in candidates.itertuples(index=False):
            # GWAS position is 1-based. SCREEN intervals are treated as BED-like
            # [start, stop), so the one-base variant interval is [pos-1, pos).
            v0 = int(r.REFERENCE_POS) - 1
            if start <= v0 < stop:
                var_hits[str(r.VARIANT_KEY)].append(c)

    accessions = sorted(
        {
            str(c.get("accession"))
            for hits in var_hits.values()
            for c in hits
            if c.get("accession")
        }
    )

    cache_root = output_file.parent.parent.parent / "cache" / "screen_ccre"
    cache_root.mkdir(parents=True, exist_ok=True)

    detail_map: dict[str, dict[str, Any]] = {}
    detail_status: dict[str, str] = {}

    workers = max(1, min(int(args.screen_detail_workers), 6))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                get_screen_detail_cached,
                acc,
                api_key,
                cache_root,
                args.screen_active_z,
                args.force,
            )
            for acc in accessions
        ]

        for fut in as_completed(futs):
            acc, summary, status = fut.result()
            detail_map[acc] = summary
            detail_status[acc] = status

    rows = []

    for r in batch.itertuples(index=False):
        vkey = str(r.VARIANT_KEY)
        hits = var_hits.get(vkey, [])

        if not hits:
            rows.append(
                {
                    "VARIANT_KEY": vkey,
                    "SCREEN_STATUS": "NO_HIT",
                }
            )
            continue

        access = [str(x.get("accession", "")) for x in hits if x.get("accession")]
        details = [detail_map.get(a, {}) for a in access]

        row: dict[str, Any] = {
            "VARIANT_KEY": vkey,
            "SCREEN_STATUS": "OK",
            "SCREEN_CCRE_IDS": join_unique(access),
            "SCREEN_CCRE_GROUPS": join_unique(
                [x.get("ccre_group", "") for x in hits]
            ),
            "SCREEN_ATAC_MAX_Z": np.nanmax(
                [safe_float(x.get("atac_max_zscore")) for x in hits]
            )
            if hits else np.nan,
            "SCREEN_CTCF_MAX_Z": np.nanmax(
                [safe_float(x.get("ctcf_max_zscore")) for x in hits]
            )
            if hits else np.nan,
            "SCREEN_DNASE_MAX_Z": np.nanmax(
                [safe_float(x.get("dnase_max_zscore")) for x in hits]
            )
            if hits else np.nan,
            "SCREEN_H3K27AC_MAX_Z": np.nanmax(
                [safe_float(x.get("h3k27ac_max_zscore")) for x in hits]
            )
            if hits else np.nan,
            "SCREEN_H3K4ME3_MAX_Z": np.nanmax(
                [safe_float(x.get("h3k4me3_max_zscore")) for x in hits]
            )
            if hits else np.nan,
            "SCREEN_NEAREST_GENES": join_unique(
                [
                    g.get("gene", "")
                    for x in hits
                    for g in (x.get("nearestgenes", []) or [])
                    if isinstance(g, dict)
                ]
            ),
        }

        merge_fields = [
            "SCREEN_LINKED_GENES_EXPERIMENTAL",
            "SCREEN_LINKED_GENES_COMPUTATIONAL",
            "SCREEN_CELL_TYPES",
            "SCREEN_TISSUES",
            "SCREEN_LINK_METHODS",
            "SCREEN_LINK_ASSAYS",
            "SCREEN_EXPERIMENT_ACCESSIONS",
            "SCREEN_FILE_ACCESSIONS",
            "SCREEN_TFS",
            "SCREEN_ACTIVE_BIOSAMPLES",
            "SCREEN_ACTIVE_TISSUES",
            "SCREEN_ACTIVE_ASSAYS",
            "SCREEN_ACTIVE_LIFESTAGES",
        ]

        for field in merge_fields:
            row[field] = join_unique([d.get(field, "") for d in details], limit=200)

        row["SCREEN_N_SIGNAL_ROWS"] = sum(
            safe_int(d.get("SCREEN_N_SIGNAL_ROWS"), 0) for d in details
        )
        row["SCREEN_SILENCER_JSON"] = join_unique(
            [d.get("SCREEN_SILENCER_JSON", "") for d in details]
        )
        row["SCREEN_DYNAMIC_ENHANCERS_JSON"] = join_unique(
            [d.get("SCREEN_DYNAMIC_ENHANCERS_JSON", "") for d in details]
        )
        row["SCREEN_DETAIL_STATUS"] = join_unique(
            [detail_status.get(a, "") for a in access]
        )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# REGULOMEDB
# =============================================================================

def compact_regulomedb(
    raw: Any,
    expected_ref: str,
    expected_alt: str,
) -> dict[str, Any]:
    """Select the allele-specific RegulomeDB record for this input variant.

    RegulomeDB coordinate queries can return multiple alleles at the same
    position.  Never silently combine their scores.  Prefer an exact GRCh38
    REF/ALT match.  If only the reversed pair exists, expose the evidence but
    mark the input as REF_ALT_SWAPPED so downstream code does not treat it as
    a trusted allele-specific match.
    """
    if not isinstance(raw, dict):
        return {}

    expected_ref = str(expected_ref).upper().strip()
    expected_alt = str(expected_alt).upper().strip()
    variants = raw.get("variants", []) or []
    variants = [v for v in variants if isinstance(v, dict)]

    exact = None
    reversed_pair = None
    for v in variants:
        ref = str(v.get("ref", "")).upper().strip()
        alt = str(v.get("alt", "")).upper().strip()
        if ref == expected_ref and alt == expected_alt:
            exact = v
            break
        if ref == expected_alt and alt == expected_ref:
            reversed_pair = v

    if exact is not None:
        selected = exact
        allele_match = "EXACT"
    elif reversed_pair is not None:
        selected = reversed_pair
        allele_match = "REF_ALT_SWAPPED"
    else:
        selected = None
        allele_match = "NO_EXACT_ALLELE"

    selected = selected or {}
    score = selected.get("regulome_score", {}) or {}
    features = selected.get("features", {}) or {}

    # Some API versions provide query-level supporting evidence too. Preserve
    # it separately but keep the allele-specific selected record authoritative.
    query_features = raw.get("features", []) or []
    supporting = raw.get("total_supporting_datasets")
    if supporting is None:
        if isinstance(query_features, (dict, list)):
            supporting = len(query_features)
        else:
            supporting = 0

    return {
        "allele_match": allele_match,
        "expected_ref": expected_ref,
        "expected_alt": expected_alt,
        "matched_ref": selected.get("ref", ""),
        "matched_alt": selected.get("alt", ""),
        "regulome_score": score,
        "features": features,
        "frequency": selected.get("freq", {}) or {},
        "gencode_category": selected.get("gencode_category", ""),
        "hgvs": selected.get("hgvs", ""),
        "spdi": selected.get("spdi", ""),
        "rsids": selected.get("rsids", []) or [],
        "query_features": query_features,
        "variants": variants,
        "nearby_snps": raw.get("nearby_snps", []) or [],
        "supporting_dataset_count": safe_int(supporting, 0),
        "notifications": raw.get("notifications", {}),
    }


def query_one_regulomedb(
    row: Any,
    force: bool,
    cache_root: Path,
) -> dict[str, Any]:
    vkey = str(row.VARIANT_KEY)
    chrom = int(row.CHR)
    pos = int(row.REFERENCE_POS)
    ref = str(row.REFERENCE_REF).upper().strip()
    alt = str(row.REFERENCE_ALT).upper().strip()
    region = f"chr{chrom}:{max(0, pos - 1)}-{pos}"

    # Cache is allele-specific; a coordinate can contain multiple alternate
    # alleles and must not reuse a compacted record for another allele.
    key = hashlib.sha256(vkey.encode()).hexdigest()[:24]
    cache = cache_root / f"{key}.json.gz"

    if cache.exists() and not force:
        try:
            data = read_gzip_json(cache)
            allele_match = str(data.get("allele_match", ""))
            if allele_match == "EXACT":
                status = "OK"
            elif allele_match == "REF_ALT_SWAPPED":
                status = "ALLELE_MISMATCH"
            else:
                status = "NO_HIT"
            return {
                "VARIANT_KEY": vkey,
                "REGULOMEDB_STATUS": status,
                "REGULOMEDB_SOURCE": "CACHE",
                **regulomedb_columns(data),
            }
        except Exception:
            pass

    session = request_session()
    params = {"regions": region, "genome": "GRCh38"}
    last: Exception | None = None

    for endpoint in (REGULOMEDB_API, REGULOMEDB_FALLBACK):
        try:
            r = request_with_retry(
                session,
                "GET",
                endpoint,
                params=params,
                timeout=120,
            )
            raw = r.json()
            data = compact_regulomedb(raw, ref, alt)
            atomic_write_gzip_json(cache, data)

            allele_match = str(data.get("allele_match", ""))
            if allele_match == "EXACT":
                status = "OK"
            elif allele_match == "REF_ALT_SWAPPED":
                status = "ALLELE_MISMATCH"
            else:
                status = "NO_HIT"

            return {
                "VARIANT_KEY": vkey,
                "REGULOMEDB_STATUS": status,
                "REGULOMEDB_SOURCE": endpoint,
                **regulomedb_columns(data),
            }

        except Exception as exc:
            last = exc

    return {
        "VARIANT_KEY": vkey,
        "REGULOMEDB_STATUS": "API_ERROR",
        "REGULOMEDB_ERROR": f"{type(last).__name__}: {last}",
    }


def regulomedb_columns(data: dict[str, Any]) -> dict[str, Any]:
    score = data.get("regulome_score", {}) or {}
    ranking = ""
    probability = ""
    tissue = ""

    if isinstance(score, dict):
        ranking = score.get("ranking", score.get("rank", ""))
        probability = score.get("probability", "")
        tissue = compact_json(score.get("tissue_specific_scores", {}))
    elif score:
        ranking = str(score)

    features = data.get("features", {}) or {}
    true_features = []
    if isinstance(features, dict):
        true_features = [
            k for k, v in features.items()
            if v is True or str(v).lower() in {"true", "1", "yes"}
        ]

    return {
        "REGULOMEDB_ALLELE_MATCH": data.get("allele_match", ""),
        "REGULOMEDB_EXPECTED_REF": data.get("expected_ref", ""),
        "REGULOMEDB_EXPECTED_ALT": data.get("expected_alt", ""),
        "REGULOMEDB_MATCHED_REF": data.get("matched_ref", ""),
        "REGULOMEDB_MATCHED_ALT": data.get("matched_alt", ""),
        "REGULOMEDB_RANKING": ranking,
        "REGULOMEDB_PROBABILITY": probability,
        "REGULOMEDB_TISSUE_SCORES_JSON": tissue,
        "REGULOMEDB_GENCODE_CATEGORY": data.get("gencode_category", ""),
        "REGULOMEDB_HGVS": data.get("hgvs", ""),
        "REGULOMEDB_SPDI": data.get("spdi", ""),
        "REGULOMEDB_RSIDS": join_unique(data.get("rsids", []) or []),
        "REGULOMEDB_TRUE_FEATURES": join_unique(true_features),
        "REGULOMEDB_FEATURE_FLAGS_JSON": compact_json(features),
        "REGULOMEDB_FREQUENCIES_JSON": compact_json(data.get("frequency", {})),
        "REGULOMEDB_SUPPORTING_DATASETS": data.get("supporting_dataset_count", 0),
        "REGULOMEDB_QUERY_FEATURES_JSON": compact_json(data.get("query_features", [])),
        "REGULOMEDB_ALL_POSITION_ALLELES_JSON": compact_json(data.get("variants", [])),
    }


def worker_regulomedb(
    batch: pd.DataFrame,
    args: argparse.Namespace,
    output_file: Path,
) -> pd.DataFrame:
    cache_root = output_file.parent.parent.parent / "cache" / "regulomedb"
    cache_root.mkdir(parents=True, exist_ok=True)

    workers = max(1, min(int(args.regulomedb_workers), 4))
    rows = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                query_one_regulomedb,
                r,
                args.force,
                cache_root,
            )
            for r in batch.itertuples(index=False)
        ]

        for fut in as_completed(futs):
            rows.append(fut.result())

    return pd.DataFrame(rows)


# =============================================================================
# CLINVAR / NCBI E-UTILITIES
# =============================================================================

def collect_recursive_values(obj: Any, tokens: Iterable[str]) -> dict[str, list[str]]:
    wanted = tuple(x.lower() for x in tokens)
    found: dict[str, list[str]] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if any(t in lk for t in wanted):
                    if not isinstance(v, (dict, list)):
                        found.setdefault(str(k), [])
                        sv = str(v)
                        if sv not in found[str(k)]:
                            found[str(k)].append(sv)
                walk(v)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(obj)
    return found


def ncbi_sleep() -> None:
    # Across many SLURM jobs there is no shared rate limiter.
    # Generated ClinVar array defaults to max-parallel=1.
    if os.environ.get("NCBI_API_KEY", "").strip():
        time.sleep(0.12)
    else:
        time.sleep(0.38)


def query_one_clinvar(
    rsid: str,
    force: bool,
    cache_root: Path,
) -> dict[str, Any]:
    cache = cache_root / f"{rsid}.json.gz"

    if cache.exists() and not force:
        try:
            compact = read_gzip_json(cache)
            return {
                "PRIMARY_RSID": rsid,
                **clinvar_columns(compact),
                "CLINVAR_SOURCE": "CACHE",
            }
        except Exception:
            pass

    session = request_session()

    common = {
        "db": "clinvar",
        "retmode": "json",
        "tool": "GWAS2m",
    }
    email = os.environ.get("NCBI_EMAIL", "").strip()
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if email:
        common["email"] = email
    if api_key:
        common["api_key"] = api_key

    try:
        params = dict(common)
        params.update(
            {
                "term": f"{rsid}[All Fields]",
                "retmax": 20,
            }
        )
        r = request_with_retry(
            session,
            "GET",
            f"{NCBI_EUTILS}/esearch.fcgi",
            params=params,
            timeout=120,
        )
        ncbi_sleep()
        search = r.json()
        ids = (
            search.get("esearchresult", {}).get("idlist", [])
            if isinstance(search, dict)
            else []
        )

        if not ids:
            compact = {"status": "NO_HIT", "ids": [], "summaries": []}
            atomic_write_gzip_json(cache, compact)
            return {
                "PRIMARY_RSID": rsid,
                **clinvar_columns(compact),
                "CLINVAR_SOURCE": "API",
            }

        params = dict(common)
        params.update({"id": ",".join(ids)})
        r = request_with_retry(
            session,
            "GET",
            f"{NCBI_EUTILS}/esummary.fcgi",
            params=params,
            timeout=120,
        )
        ncbi_sleep()
        summary = r.json()

        records = []
        result = summary.get("result", {}) if isinstance(summary, dict) else {}
        for uid in ids:
            obj = result.get(str(uid))
            if isinstance(obj, dict):
                records.append(obj)

        compact = {
            "status": "OK" if records else "NO_HIT",
            "ids": ids,
            "summaries": records,
        }
        atomic_write_gzip_json(cache, compact)

        return {
            "PRIMARY_RSID": rsid,
            **clinvar_columns(compact),
            "CLINVAR_SOURCE": "API",
        }

    except Exception as exc:
        return {
            "PRIMARY_RSID": rsid,
            "CLINVAR_STATUS": "API_ERROR",
            "CLINVAR_ERROR": f"{type(exc).__name__}: {exc}",
        }


def clinvar_columns(compact: dict[str, Any]) -> dict[str, Any]:
    status = compact.get("status", "NO_HIT")
    summaries = compact.get("summaries", []) or []

    values = collect_recursive_values(
        summaries,
        (
            "clinical_significance",
            "description",
            "variation_name",
            "accession",
            "trait",
            "gene",
            "review_status",
            "title",
        ),
    )

    clinical = []
    traits = []
    genes = []
    accessions = []
    names = []
    review = []

    for k, vals in values.items():
        lk = k.lower()
        if "clinical_significance" in lk or (
            "description" in lk
            and any(
                x.lower() in {
                    "pathogenic",
                    "likely pathogenic",
                    "benign",
                    "likely benign",
                    "uncertain significance",
                }
                for x in vals
            )
        ):
            clinical.extend(vals)
        if "trait" in lk:
            traits.extend(vals)
        if "gene" in lk:
            genes.extend(vals)
        if "accession" in lk:
            accessions.extend(vals)
        if "variation_name" in lk or k.lower() == "title":
            names.extend(vals)
        if "review_status" in lk:
            review.extend(vals)

    return {
        "CLINVAR_STATUS": status,
        "CLINVAR_UIDS": join_unique(compact.get("ids", [])),
        "CLINVAR_CLINICAL_SIGNIFICANCE": join_unique(clinical),
        "CLINVAR_TRAITS": join_unique(traits, limit=100),
        "CLINVAR_GENES": join_unique(genes),
        "CLINVAR_ACCESSIONS": join_unique(accessions),
        "CLINVAR_VARIATION_NAMES": join_unique(names, limit=100),
        "CLINVAR_REVIEW_STATUS": join_unique(review),
        "CLINVAR_SUMMARY_JSON": compact_json(summaries),
    }


def worker_clinvar(
    batch: pd.DataFrame,
    args: argparse.Namespace,
    output_file: Path,
) -> pd.DataFrame:
    cache_root = output_file.parent.parent.parent / "cache" / "clinvar"
    cache_root.mkdir(parents=True, exist_ok=True)

    rows = []

    # Deliberately sequential within a task. NCBI has explicit request limits and
    # the SLURM array already supplies concurrency.
    for i, rs in enumerate(batch["PRIMARY_RSID"].astype(str), start=1):
        rows.append(query_one_clinvar(rs.lower(), args.force, cache_root))
        print(f"  ClinVar {i:,}/{len(batch):,} {rs}", flush=True)

    return pd.DataFrame(rows)


# =============================================================================
# WORKER ROUTING
# =============================================================================

def load_task_batch(manifest_file: Path, task_id: int) -> tuple[pd.DataFrame, Path]:
    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str)
    task_nums = pd.to_numeric(manifest["TASK_ID"], errors="coerce")
    row = manifest.loc[task_nums == task_id]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest row for TASK_ID={task_id}; found {len(row)}"
        )

    r = row.iloc[0]
    input_file = Path(r["INPUT_FILE"]).resolve()
    start = int(r["START_ROW"])
    end = int(r["END_ROW"])
    output_file = Path(r["OUTPUT_FILE"]).resolve()

    data = read_tsv(input_file)
    batch = data.iloc[start:end].copy().reset_index(drop=True)

    expected = int(r["N_RECORDS"])
    if len(batch) != expected:
        raise RuntimeError(
            f"Batch length mismatch: expected {expected}, observed {len(batch)}"
        )

    return batch, output_file


def worker_mode(args: argparse.Namespace) -> None:
    manifest_file = Path(args.manifest).resolve()
    batch, output_file = load_task_batch(manifest_file, args.task_id)

    banner(
        f"STEP07 API WORKER — {args.provider.upper()} — TASK {args.task_id}"
    )
    print(f"Manifest     : {manifest_file}")
    print(f"Records      : {len(batch):,}")
    print(f"Output       : {output_file}")
    print(f"Force        : {args.force}")
    print(f"Host         : {os.uname().nodename}")
    print(f"Started      : {utc_now()}")

    if output_file.exists() and output_file.stat().st_size > 50 and not args.force:
        print(f"[CACHE] Reusing existing shard: {output_file}")
        return

    provider = args.provider.lower()

    if provider == "ensembl_vep":
        result = worker_ensembl_vep(batch, args)
    elif provider == "ensembl_variation":
        result = worker_ensembl_variation(batch, args)
    elif provider == "screen":
        result = worker_screen(batch, args, output_file)
    elif provider == "regulomedb":
        result = worker_regulomedb(batch, args, output_file)
    elif provider == "clinvar":
        result = worker_clinvar(batch, args, output_file)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if result.empty:
        raise RuntimeError(f"{provider} produced an empty result table")

    write_tsv_gz(result, output_file)

    status_cols = [c for c in result.columns if c.endswith("_STATUS")]
    print()
    print(f"Rows written : {len(result):,}")
    if status_cols:
        for c in status_cols:
            print(f"\n{c}")
            print(result[c].fillna("").value_counts(dropna=False).to_string())
    print(f"\nFinished     : {utc_now()}")


# =============================================================================
# MERGE
# =============================================================================

def read_provider_shards(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("batch_*.tsv.gz"))
    if not files:
        return pd.DataFrame()

    parts = []
    for i, f in enumerate(files, start=1):
        x = read_tsv(f)
        parts.append(x)
        if i == 1 or i % 50 == 0 or i == len(files):
            print(
                f"  {directory.name:<20s}: {i:,}/{len(files):,} shards",
                flush=True,
            )

    return pd.concat(parts, ignore_index=True)


def combine_unique_annotation(base: pd.DataFrame, extra: pd.DataFrame, key: str) -> pd.DataFrame:
    if extra.empty:
        return base
    extra = extra.sort_values(key, kind="stable").drop_duplicates(key, keep="first")
    overlap = [c for c in extra.columns if c in base.columns and c != key]
    if overlap:
        extra = extra.drop(columns=overlap)
    return base.merge(extra, on=key, how="left", validate="one_to_one")


def first_nonempty(series: pd.Series) -> str:
    for x in series:
        if pd.notna(x) and str(x).strip() not in {"", "nan", "None"}:
            return str(x)
    return ""


def locus_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (study, locus), g in master.groupby(
        ["STUDY_ACCESSION", "LOCUS_ID"], sort=False, dropna=False
    ):
        g = g.copy()
        pips = pd.to_numeric(g["PIP"], errors="coerce")
        top_idx = pips.idxmax() if pips.notna().any() else g.index[0]
        top = g.loc[top_idx]

        classes = (
            g.get("FINAL_FUNCTIONAL_CLASS", pd.Series("", index=g.index))
            .fillna("")
            .astype(str)
        )

        rows.append(
            {
                "STUDY_ACCESSION": study,
                "LOCUS_ID": locus,
                "N_VARIANTS": len(g),
                "MAX_PIP": float(pips.max()) if pips.notna().any() else np.nan,
                "TOP_VARIANT": top.get("VARIANT_KEY", ""),
                "TOP_RSID": top.get("PRIMARY_RSID", ""),
                "TOP_GENE": top.get("ENSEMBL_GENE_SYMBOL", ""),
                "TOP_CONSEQUENCE": top.get("ENSEMBL_BEST_CONSEQUENCE", ""),
                "N_CODING": int((classes == "CODING").sum()),
                "N_SPLICING": int((classes == "SPLICING").sum()),
                "N_REGULATORY": int((classes == "REGULATORY").sum()),
                "N_NONCODING_RNA": int((classes == "NONCODING_RNA").sum()),
                "N_INTRONIC": int((classes == "INTRONIC").sum()),
                "N_INTERGENIC": int((classes == "INTERGENIC").sum()),
                "CANDIDATE_GENES": join_unique(
                    g.get("ENSEMBL_ALL_SYMBOLS", pd.Series("", index=g.index))
                ),
                "SCREEN_CCRE_GROUPS": join_unique(
                    g.get("SCREEN_CCRE_GROUPS", pd.Series("", index=g.index))
                ),
                "SCREEN_TARGET_GENES": join_unique(
                    pd.concat(
                        [
                            g.get(
                                "SCREEN_LINKED_GENES_EXPERIMENTAL",
                                pd.Series("", index=g.index),
                            ),
                            g.get(
                                "SCREEN_LINKED_GENES_COMPUTATIONAL",
                                pd.Series("", index=g.index),
                            ),
                        ],
                        ignore_index=True,
                    )
                ),
                "CLINVAR_SIGNIFICANCE": join_unique(
                    g.get(
                        "CLINVAR_CLINICAL_SIGNIFICANCE",
                        pd.Series("", index=g.index),
                    )
                ),
                "REGULOMEDB_RANKINGS": join_unique(
                    g.get("REGULOMEDB_RANKING", pd.Series("", index=g.index))
                ),
            }
        )

    return pd.DataFrame(rows)


def derive_final_class(row: pd.Series) -> str:
    """Primary transcript/consequence class retained for backwards compatibility."""
    base = str(row.get("ENSEMBL_FUNCTIONAL_CATEGORY", "") or "").strip()
    if base:
        return base
    screen_ok = str(row.get("SCREEN_STATUS", "")) == "OK"
    reg_ok = str(row.get("REGULOMEDB_STATUS", "")) == "OK"
    if screen_ok or reg_ok:
        return "REGULATORY"
    return "OTHER"


def derive_functional_classes(row: pd.Series) -> str:
    """Multi-label functional interpretation so regulatory evidence is not lost.

    A variant may simultaneously be intronic/non-coding-transcript AND overlap
    an enhancer/cCRE.  A single mutually exclusive class hides that biology.
    """
    classes = []
    base = str(row.get("ENSEMBL_FUNCTIONAL_CATEGORY", "") or "").strip()
    if base and base != "OTHER":
        classes.append(base)

    if str(row.get("SCREEN_STATUS", "")) == "OK":
        classes.append("REGULATORY")
    if str(row.get("REGULOMEDB_STATUS", "")) == "OK":
        classes.append("REGULATORY")

    return join_unique(classes) or "OTHER"


def allele_qc_from_variation(row: pd.Series) -> tuple[str, str, str]:
    """Compare Step06/input REF/ALT with an Ensembl GRCh38 rsID mapping."""
    rsid = str(row.get("PRIMARY_RSID", "") or "").strip()
    raw = row.get("ENSEMBL_VARIATION_MAPPINGS_JSON", "")
    if not rsid:
        return "NO_RSID", "", ""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or not str(raw).strip():
        return "NO_MAPPING", "", ""

    try:
        mappings = json.loads(str(raw))
    except Exception:
        return "MAPPING_PARSE_ERROR", "", ""
    if not isinstance(mappings, list):
        return "NO_MAPPING", "", ""

    chrom = str(row.get("CHR", "")).replace("chr", "").strip()
    pos = safe_int(row.get("REFERENCE_POS"), -1)
    ref = str(row.get("REFERENCE_REF", "") or "").upper().strip()
    alt = str(row.get("REFERENCE_ALT", "") or "").upper().strip()

    candidates = []
    for m in mappings:
        if not isinstance(m, dict):
            continue
        assembly = str(m.get("assembly_name", ""))
        seq = str(m.get("seq_region_name", "")).replace("chr", "")
        start = safe_int(m.get("start"), -1)
        if assembly.upper().startswith("GRCH38") and seq == chrom and start == pos:
            candidates.append(m)

    if not candidates:
        return "NO_GRCH38_POSITION_MATCH", "", ""

    m = candidates[0]
    alleles = [x.upper().strip() for x in str(m.get("allele_string", "")).split("/") if x.strip()]
    if not alleles:
        return "NO_ALLELE_STRING", "", ""
    reference = alleles[0]
    alternatives = alleles[1:]

    if ref == reference and alt in alternatives:
        return "PASS", reference, ";".join(alternatives)
    if alt == reference and ref in alternatives:
        return "REF_ALT_SWAPPED", reference, ";".join(alternatives)
    if ref in alleles and alt in alleles:
        return "ALLELES_PRESENT_REFERENCE_MISMATCH", reference, ";".join(alternatives)
    return "ALLELE_NOT_IN_RSID_MAPPING", reference, ";".join(alternatives)


def merge_mode(args: argparse.Namespace) -> None:
    plan_file = Path(args.plan_json).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    universe_file = Path(plan["variant_universe"])
    study_map_file = Path(plan["study_variant_map"])
    rsid_file = Path(plan["rsid_universe"])
    out_root = Path(plan["out_root"])
    shards_root = Path(plan["shards_root"])

    banner("STEP07 API FINAL MERGE")
    print(f"Plan         : {plan_file}")
    print(f"Variant file : {universe_file}")
    print(f"Output root  : {out_root}")

    universe = read_tsv(universe_file)
    study_map = read_tsv(study_map_file)

    global_master = universe.copy()

    for provider in ("ensembl_vep", "screen", "regulomedb"):
        extra = read_provider_shards(shards_root / provider)
        if extra.empty:
            print(f"[WARN] No shards found for {provider}")
            continue
        global_master = combine_unique_annotation(
            global_master, extra, "VARIANT_KEY"
        )

    # rsID-keyed providers.
    for provider in ("ensembl_variation", "clinvar"):
        extra = read_provider_shards(shards_root / provider)
        if extra.empty:
            print(f"[WARN] No shards found for {provider}")
            continue
        extra["PRIMARY_RSID"] = extra["PRIMARY_RSID"].fillna("").astype(str).str.lower()
        global_master["PRIMARY_RSID"] = (
            global_master["PRIMARY_RSID"].fillna("").astype(str).str.lower()
        )
        extra = extra.sort_values("PRIMARY_RSID", kind="stable").drop_duplicates(
            "PRIMARY_RSID", keep="first"
        )
        overlap = [
            c
            for c in extra.columns
            if c in global_master.columns and c != "PRIMARY_RSID"
        ]
        if overlap:
            extra = extra.drop(columns=overlap)
        global_master = global_master.merge(
            extra, on="PRIMARY_RSID", how="left", validate="many_to_one"
        )

    global_master["FINAL_FUNCTIONAL_CLASS"] = global_master.apply(
        derive_final_class, axis=1
    )
    global_master["FINAL_FUNCTIONAL_CLASSES"] = global_master.apply(
        derive_functional_classes, axis=1
    )

    qc = global_master.apply(allele_qc_from_variation, axis=1)
    global_master["ALLELE_QC_STATUS"] = [x[0] for x in qc]
    global_master["ENSEMBL_GRCH38_REFERENCE_ALLELE"] = [x[1] for x in qc]
    global_master["ENSEMBL_GRCH38_ALTERNATE_ALLELES"] = [x[2] for x in qc]

    global_master["HAS_SCREEN_CCRE"] = (
        global_master.get("SCREEN_STATUS", "").astype(str) == "OK"
        if "SCREEN_STATUS" in global_master.columns
        else False
    )
    global_master["HAS_REGULOMEDB"] = (
        global_master.get("REGULOMEDB_STATUS", "").astype(str) == "OK"
        if "REGULOMEDB_STATUS" in global_master.columns
        else False
    )
    global_master["HAS_CLINVAR"] = (
        global_master.get("CLINVAR_STATUS", "").astype(str) == "OK"
        if "CLINVAR_STATUS" in global_master.columns
        else False
    )

    global_file = out_root / "all_unique_variants_annotation_master.tsv.gz"
    write_tsv_gz(global_master, global_file)

    combined = study_map.merge(
        global_master,
        on="VARIANT_KEY",
        how="left",
        suffixes=("", "_GLOBAL"),
        validate="many_to_one",
    )

    # Prefer study-level rsID/coordinate columns where duplicated.
    for col in [
        "PRIMARY_RSID",
        "CHR",
        "REFERENCE_POS",
        "REFERENCE_REF",
        "REFERENCE_ALT",
    ]:
        gcol = col + "_GLOBAL"
        if gcol in combined.columns:
            if col in combined.columns:
                combined[col] = combined[col].replace("", np.nan).fillna(combined[gcol])
                combined = combined.drop(columns=[gcol])
            else:
                combined = combined.rename(columns={gcol: col})

    combined_file = out_root / "all_studies_annotation_master.tsv.gz"
    write_tsv_gz(combined, combined_file)

    banner("WRITING PER-STUDY OUTPUTS")

    study_count = 0

    for accession, g in combined.groupby("STUDY_ACCESSION", sort=False):
        study_count += 1
        d = out_root / str(accession)
        d.mkdir(parents=True, exist_ok=True)

        study_file = d / f"{accession}_annotation_master.tsv.gz"
        write_tsv_gz(g, study_file)

        ls = locus_summary(g)
        ls.to_csv(
            d / f"{accession}_locus_annotation_summary.tsv",
            sep="\t",
            index=False,
        )

        # Useful functional subsets.
        for cls, name in [
            ("CODING", "coding_variants"),
            ("SPLICING", "splicing_variants"),
            ("REGULATORY", "regulatory_variants"),
            ("NONCODING_RNA", "noncoding_rna_variants"),
        ]:
            sub = g[g["FINAL_FUNCTIONAL_CLASS"] == cls].copy()
            write_tsv_gz(sub, d / f"{accession}_{name}.tsv.gz")

        print(
            f"[OK] {accession}: {len(g):,} variant-locus rows, "
            f"{g['LOCUS_ID'].nunique():,} loci"
        )

    provider_status = []

    for provider in (
        "ensembl_vep",
        "ensembl_variation",
        "screen",
        "regulomedb",
        "clinvar",
    ):
        d = shards_root / provider
        files = sorted(d.glob("batch_*.tsv.gz"))
        provider_status.append(
            {
                "PROVIDER": provider,
                "SHARDS": len(files),
                "ROWS": sum(len(read_tsv(f)) for f in files),
            }
        )

    pd.DataFrame(provider_status).to_csv(
        out_root / "provider_completion_summary.tsv",
        sep="\t",
        index=False,
    )

    summary = {
        "STEP": "07",
        "VERSION": VERSION,
        "STATUS": "COMPLETE",
        "PHENOTYPE": plan["phenotype"],
        "ANCESTRY_CODE": plan["ancestry_code"],
        "ANCESTRY_LABEL": plan["ancestry_label"],
        "N_UNIQUE_VARIANTS": len(global_master),
        "N_STUDY_VARIANT_ROWS": len(combined),
        "N_STUDIES": study_count,
        "N_LOCI": combined[["STUDY_ACCESSION", "LOCUS_ID"]]
        .drop_duplicates()
        .shape[0],
        "GLOBAL_MASTER": str(global_file),
        "COMBINED_MASTER": str(combined_file),
        "COMPLETED_UTC": utc_now(),
    }
    (out_root / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    banner("STEP07 API ANNOTATION COMPLETE")
    for k, v in summary.items():
        print(f"{k:<24s}: {v}")


# =============================================================================
# ONE-VARIANT LIVE SMOKE TEST
# =============================================================================

def parse_test_variant(text: str) -> tuple[int, int, str, str]:
    """Parse CHR:POS:REF:ALT, e.g. 4:89704960:A:G."""
    parts = str(text).replace("chr", "").replace("CHR", "").split(":")
    if len(parts) != 4:
        raise ValueError("--variant must be CHR:POS:REF:ALT, e.g. 4:89704960:A:G")
    chrom, pos, ref, alt = parts
    chrom_i = int(chrom)
    pos_i = int(pos)
    ref = ref.upper().strip()
    alt = alt.upper().strip()
    if chrom_i < 1 or chrom_i > 22 or pos_i < 1 or not ref or not alt:
        raise ValueError(f"Invalid test variant: {text}")
    return chrom_i, pos_i, ref, alt


def print_annotation_block(title: str, df: pd.DataFrame) -> None:
    banner(title)
    if df is None or df.empty:
        print("NO ROWS RETURNED")
        return
    row = df.iloc[0]
    width = max(28, min(60, max(len(str(c)) for c in df.columns)))
    for col in df.columns:
        value = row[col]
        if isinstance(value, float) and np.isnan(value):
            value = ""
        text = str(value)
        if len(text) > 4000:
            text = text[:4000] + " ... [TRUNCATED ON SCREEN]"
        print(f"{col:<{width}s} : {text}")


def test_one_variant_mode(args: argparse.Namespace) -> None:
    chrom, pos, ref, alt = parse_test_variant(args.variant)
    rsid = str(args.rsid or "").strip().lower()
    if rsid and not re.fullmatch(r"rs\d+", rsid, flags=re.I):
        raise ValueError("--rsid must look like rs356182")

    vkey = variant_key(chrom, pos, ref, alt)
    root = Path(args.root).expanduser().resolve()
    test_root = root / "07_annotation_api" / "_one_variant_test" / slugify(vkey)
    test_root.mkdir(parents=True, exist_ok=True)

    banner("STEP07 ONE-VARIANT LIVE API TEST")
    print(f"Version          : {VERSION}")
    print(f"Variant          : {vkey}")
    print(f"rsID             : {rsid or 'NOT PROVIDED'}")
    print(f"SCREEN API key   : {'SET' if os.environ.get('SCREEN_API_KEY', '').strip() else 'NOT SET'}")
    print(f"NCBI API key     : {'SET' if os.environ.get('NCBI_API_KEY', '').strip() else 'NOT SET'}")
    print(f"NCBI email       : {'SET' if os.environ.get('NCBI_EMAIL', '').strip() else 'NOT SET'}")
    print(f"Test output      : {test_root}")
    print("NOTE             : key VALUES are never printed or written to output.")

    batch = pd.DataFrame([{
        "VARIANT_KEY": vkey,
        "PRIMARY_RSID": rsid,
        "CHR": chrom,
        "REFERENCE_POS": pos,
        "REFERENCE_REF": ref,
        "REFERENCE_ALT": alt,
        "API_ID": "GWAS2M_TEST_000001",
    }])

    all_results: list[tuple[str, pd.DataFrame]] = []

    try:
        vep = worker_ensembl_vep(batch.copy(), args)
        print_annotation_block("1/5 ENSEMBL VEP", vep)
        write_tsv_gz(vep, test_root / "ensembl_vep.tsv.gz")
        all_results.append(("ensembl_vep", vep))
    except Exception as exc:
        banner("1/5 ENSEMBL VEP — FAILED")
        print(f"{type(exc).__name__}: {exc}")

    if rsid:
        try:
            variation = worker_ensembl_variation(
                batch[["VARIANT_KEY", "PRIMARY_RSID"]].copy(), args
            )
            print_annotation_block("2/5 ENSEMBL VARIATION", variation)
            write_tsv_gz(variation, test_root / "ensembl_variation.tsv.gz")
            all_results.append(("ensembl_variation", variation))
        except Exception as exc:
            banner("2/5 ENSEMBL VARIATION — FAILED")
            print(f"{type(exc).__name__}: {exc}")
    else:
        banner("2/5 ENSEMBL VARIATION — SKIPPED")
        print("No --rsid supplied.")

    if args.skip_screen:
        banner("3/5 SCREEN / ENCODE — SKIPPED")
        print("--skip-screen was supplied.")
    elif not os.environ.get("SCREEN_API_KEY", "").strip():
        banner("3/5 SCREEN / ENCODE — SKIPPED")
        print("SCREEN_API_KEY is not set.")
    else:
        try:
            screen = worker_screen(batch.copy(), args, test_root / "screen.tsv.gz")
            print_annotation_block("3/5 SCREEN / ENCODE", screen)
            write_tsv_gz(screen, test_root / "screen.tsv.gz")
            all_results.append(("screen", screen))
        except Exception as exc:
            banner("3/5 SCREEN / ENCODE — FAILED")
            print(f"{type(exc).__name__}: {exc}")

    try:
        reg = worker_regulomedb(batch.copy(), args, test_root / "regulomedb.tsv.gz")
        print_annotation_block("4/5 REGULOMEDB", reg)
        write_tsv_gz(reg, test_root / "regulomedb.tsv.gz")
        all_results.append(("regulomedb", reg))
    except Exception as exc:
        banner("4/5 REGULOMEDB — FAILED")
        print(f"{type(exc).__name__}: {exc}")

    if rsid:
        try:
            clinvar = worker_clinvar(
                batch[["VARIANT_KEY", "PRIMARY_RSID"]].copy(),
                args,
                test_root / "clinvar.tsv.gz",
            )
            print_annotation_block("5/5 CLINVAR / NCBI", clinvar)
            write_tsv_gz(clinvar, test_root / "clinvar.tsv.gz")
            all_results.append(("clinvar", clinvar))
        except Exception as exc:
            banner("5/5 CLINVAR / NCBI — FAILED")
            print(f"{type(exc).__name__}: {exc}")
    else:
        banner("5/5 CLINVAR / NCBI — SKIPPED")
        print("No --rsid supplied.")

    master = batch.copy()
    for provider, extra in all_results:
        if extra is None or extra.empty:
            continue
        key = "PRIMARY_RSID" if provider in {"ensembl_variation", "clinvar"} else "VARIANT_KEY"
        if key not in extra.columns:
            continue
        overlap = [c for c in extra.columns if c in master.columns and c != key]
        z = extra.drop(columns=overlap, errors="ignore").drop_duplicates(key, keep="first")
        master = master.merge(z, on=key, how="left", validate="many_to_one")

    if "ENSEMBL_FUNCTIONAL_CATEGORY" in master.columns:
        master["FINAL_FUNCTIONAL_CLASS"] = master.apply(derive_final_class, axis=1)
        master["FINAL_FUNCTIONAL_CLASSES"] = master.apply(derive_functional_classes, axis=1)

    qc = master.apply(allele_qc_from_variation, axis=1)
    master["ALLELE_QC_STATUS"] = [x[0] for x in qc]
    master["ENSEMBL_GRCH38_REFERENCE_ALLELE"] = [x[1] for x in qc]
    master["ENSEMBL_GRCH38_ALTERNATE_ALLELES"] = [x[2] for x in qc]

    print_annotation_block("FINAL ONE-VARIANT MERGED PREVIEW", master)
    write_tsv_gz(master, test_root / "one_variant_annotation_master.tsv.gz")

    banner("ONE-VARIANT TEST COMPLETE")
    print(f"Master result : {test_root / 'one_variant_annotation_master.tsv.gz'}")
    print("If these provider blocks look sensible, run normal planner mode next.")


# =============================================================================
# PLANNER
# =============================================================================

def planner_mode(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)

    out_root = (
        root
        / "07_annotation_api"
        / phenotype_slug
        / ancestry_slug
    )
    manifests = out_root / "manifests"
    shards = out_root / "shards"
    logs = root / "logs" / "step07_api" / phenotype_slug / ancestry_slug
    scripts = out_root / "slurm"

    for d in (out_root, manifests, shards, logs, scripts):
        d.mkdir(parents=True, exist_ok=True)

    banner("STEP07 API-ONLY DISTRIBUTED ANNOTATION PLANNER")
    print(f"Version         : {VERSION}")
    print(f"Root            : {root}")
    print(f"Phenotype       : {phenotype}")
    print(f"Ancestry        : {ancestry_label} ({ancestry_code})")
    print(f"Output          : {out_root}")
    print("Reference DBs   : NONE DOWNLOADED")
    print(f"SLURM partition : {args.partition}")
    print(f"Array walltime  : {slurm_walltime_for_partition(args.partition, args.time, merge=False)}")
    print(f"Merge walltime  : {slurm_walltime_for_partition(args.partition, args.merge_time, merge=True)}")
    print()

    studies, excluded = discover_studies(
        root,
        phenotype,
        ancestry_label,
        args.allow_partial,
    )

    universe, study_map = build_variant_universe(
        studies,
        phenotype,
        ancestry_code,
        ancestry_label,
    )
    rsids = build_rsid_universe(universe)

    variant_file = manifests / "variant_universe.tsv.gz"
    study_map_file = manifests / "study_variant_map.tsv.gz"
    rsid_file = manifests / "rsid_universe.tsv.gz"

    write_tsv_gz(universe, variant_file)
    write_tsv_gz(study_map, study_map_file)
    write_tsv_gz(rsids, rsid_file)

    pd.DataFrame(excluded).to_csv(
        manifests / "excluded_studies.tsv", sep="\t", index=False
    )

    print(f"Eligible studies       : {len(studies):,}")
    print(f"Excluded studies       : {len(excluded):,}")
    print(f"Study/locus rows       : {len(study_map):,}")
    print(f"Unique variants        : {len(universe):,}")
    print(f"Input rsIDs            : {len(rsids):,}")
    print(f"Unique loci            : {study_map[['STUDY_ACCESSION','LOCUS_ID']].drop_duplicates().shape[0]:,}")

    provider_defs = [
        (
            "ensembl_vep",
            variant_file,
            len(universe),
            args.ensembl_batch_size,
            args.ensembl_max_parallel,
        ),
        (
            "ensembl_variation",
            rsid_file,
            len(rsids),
            args.variation_batch_size,
            args.variation_max_parallel,
        ),
        (
            "screen",
            variant_file,
            0 if args.skip_screen else len(universe),
            args.screen_batch_size,
            args.screen_max_parallel,
        ),
        (
            "regulomedb",
            variant_file,
            len(universe),
            args.regulomedb_batch_size,
            args.regulomedb_max_parallel,
        ),
        (
            "clinvar",
            rsid_file,
            len(rsids),
            args.clinvar_batch_size,
            args.clinvar_max_parallel,
        ),
    ]

    this_script = Path(__file__).resolve()
    python_exe = args.python or sys.executable or "python"

    providers_for_submit: list[tuple[str, Path, int]] = []
    provider_plan = {}

    banner("PROVIDER ARRAY PLAN")
    print(
        f"{'PROVIDER':22s} {'RECORDS':>12s} {'BATCH':>8s} "
        f"{'TASKS':>8s} {'MAX_PARALLEL':>14s}"
    )
    print("-" * 72)

    for provider, infile, n_rows, batch_size, max_parallel in provider_defs:
        pdir = shards / provider
        pdir.mkdir(parents=True, exist_ok=True)

        manifest_file = manifests / f"{provider}_manifest.tsv"
        manifest = create_batch_manifest(
            infile,
            n_rows,
            batch_size,
            provider,
            pdir,
            manifest_file,
        )

        n_tasks = len(manifest)
        sbatch_file = scripts / f"{provider}.sbatch"

        if n_tasks:
            write_provider_sbatch(
                sbatch_file,
                python_exe,
                this_script,
                provider,
                manifest_file,
                n_tasks,
                max_parallel,
                root,
                logs,
                args,
            )

        providers_for_submit.append((provider, sbatch_file, n_tasks))

        provider_plan[provider] = {
            "records": n_rows,
            "batch_size": batch_size,
            "tasks": n_tasks,
            "max_parallel": max_parallel,
            "manifest": str(manifest_file),
            "sbatch": str(sbatch_file),
        }

        print(
            f"{provider:22s} {n_rows:12,d} {batch_size:8,d} "
            f"{n_tasks:8,d} {max_parallel:14,d}"
        )

    plan_file = manifests / "step07_api_plan.json"
    merge_sbatch = scripts / "merge.sbatch"

    plan = {
        "version": VERSION,
        "root": str(root),
        "phenotype": phenotype,
        "phenotype_slug": phenotype_slug,
        "ancestry_code": ancestry_code,
        "ancestry_label": ancestry_label,
        "ancestry_slug": ancestry_slug,
        "out_root": str(out_root),
        "manifests_root": str(manifests),
        "shards_root": str(shards),
        "variant_universe": str(variant_file),
        "study_variant_map": str(study_map_file),
        "rsid_universe": str(rsid_file),
        "providers": provider_plan,
        "created_utc": utc_now(),
        "annotation_model": "API_ONLY",
        "downloads_large_reference_databases": False,
    }
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    write_merge_sbatch(
        merge_sbatch,
        python_exe,
        this_script,
        plan_file,
        root,
        logs,
        args,
    )

    submit_script = (
        root
        / f"Step07_Submit_API_{phenotype_slug}_{ancestry_slug}.sh"
    )
    write_submit_script(
        submit_script,
        providers_for_submit,
        merge_sbatch,
        screen_required=not args.skip_screen,
    )

    banner("WHAT THIS WILL QUERY")
    print(
        "ENSEMBL_VEP         : consequence/transcript + AlphaMissense/CADD/REVEL/"
        "ProtVar/SpliceAI/SIFT/PolyPhen/UniProt/domains"
    )
    print(
        "ENSEMBL_VARIATION   : populations + phenotypes + clinical significance"
    )
    print(
        "SCREEN              : cCRE/promoter/enhancer + chromatin signals + "
        "cell/tissue context + target genes"
    )
    print("REGULOMEDB          : regulatory score/supporting features")
    print("CLINVAR             : NCBI ClinVar summary/traits/significance")
    print()
    print("Large reference downloads: 0")
    print("Only returned API annotation shards are written.")

    banner("GENERATED FILES")
    print(f"Variant universe : {variant_file}")
    print(f"Study map        : {study_map_file}")
    print(f"rsID universe    : {rsid_file}")
    print(f"Plan JSON        : {plan_file}")
    print(f"SLURM scripts    : {scripts}")
    print(f"Master submit    : {submit_script}")

    print()
    print("BEFORE SUBMITTING SCREEN:")
    print("  export SCREEN_API_KEY='YOUR_SCREEN_KEY'")
    print()
    print("OPTIONAL FOR NCBI:")
    print("  export NCBI_EMAIL='your@email'")
    print("  export NCBI_API_KEY='your_ncbi_key'")
    print()
    print("NEXT COMMAND:")
    print(f"  bash {submit_script}")

    if args.submit:
        if not args.skip_screen and not os.environ.get("SCREEN_API_KEY", "").strip():
            raise RuntimeError(
                "--submit requested but SCREEN_API_KEY is not set. "
                "Export the key first, or use --skip-screen."
            )
        shell(["bash", str(submit_script)])


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Distributed API-only annotation of all Step06 fine-mapped variants."
        ),
    )

    p.add_argument("--root", default=".")
    p.add_argument("--phenotype", default=None)
    p.add_argument("--ancestry", default=None)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--force", action="store_true")

    p.add_argument("--python", default=None)
    p.add_argument(
        "--partition",
        default="general",
        help=(
            "SLURM partition. Default: general. Partition-aware time defaults: "
            "general=24:00:00, asher=0 (unlimited)."
        ),
    )
    p.add_argument(
        "--time",
        default=None,
        help=(
            "Provider-array walltime. If omitted: general=24:00:00, "
            "asher=0 (unlimited), other partitions=04:00:00."
        ),
    )
    p.add_argument("--memory", default="6G")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument(
        "--merge-time",
        default=None,
        help=(
            "Final merge walltime. If omitted: general=24:00:00, "
            "asher=0 (unlimited), other partitions=02:00:00."
        ),
    )
    p.add_argument("--merge-memory", default="16G")

    p.add_argument(
        "--ensembl-batch-size",
        type=int,
        default=DEFAULT_BATCHES["ensembl_vep"],
    )
    p.add_argument(
        "--variation-batch-size",
        type=int,
        default=DEFAULT_BATCHES["ensembl_variation"],
    )
    p.add_argument(
        "--screen-batch-size",
        type=int,
        default=DEFAULT_BATCHES["screen"],
    )
    p.add_argument(
        "--regulomedb-batch-size",
        type=int,
        default=DEFAULT_BATCHES["regulomedb"],
    )
    p.add_argument(
        "--clinvar-batch-size",
        type=int,
        default=DEFAULT_BATCHES["clinvar"],
    )

    p.add_argument(
        "--ensembl-max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL["ensembl_vep"],
    )
    p.add_argument(
        "--variation-max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL["ensembl_variation"],
    )
    p.add_argument(
        "--screen-max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL["screen"],
    )
    p.add_argument(
        "--regulomedb-max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL["regulomedb"],
    )
    p.add_argument(
        "--clinvar-max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL["clinvar"],
    )

    p.add_argument("--screen-detail-workers", type=int, default=2)
    p.add_argument("--screen-active-z", type=float, default=1.64)
    p.add_argument("--regulomedb-workers", type=int, default=2)
    p.add_argument("--skip-screen", action="store_true")

    # One-variant live smoke test.
    p.add_argument("--test-one", action="store_true",
                   help="Query all providers for one variant and print returned annotations.")
    p.add_argument("--variant", default=None,
                   help="For --test-one: CHR:POS:REF:ALT, e.g. 4:89704960:A:G")
    p.add_argument("--rsid", default=None,
                   help="For --test-one: optional rsID, e.g. rs356182")

    # Worker mode.
    p.add_argument("--worker", action="store_true")
    p.add_argument(
        "--provider",
        choices=(
            "ensembl_vep",
            "ensembl_variation",
            "screen",
            "regulomedb",
            "clinvar",
        ),
        default=None,
    )
    p.add_argument("--manifest", default=None)
    p.add_argument("--task-id", type=int, default=None)

    # Merge mode.
    p.add_argument("--merge", action="store_true")
    p.add_argument("--plan-json", default=None)

    args = p.parse_args()

    modes = int(args.worker) + int(args.merge) + int(args.test_one)
    if modes > 1:
        p.error("--worker, --merge and --test-one are mutually exclusive")

    if args.test_one:
        if not args.variant:
            p.error("--test-one requires --variant CHR:POS:REF:ALT")
    elif args.worker:
        if not args.provider:
            p.error("--worker requires --provider")
        if not args.manifest:
            p.error("--worker requires --manifest")
        if args.task_id is None or args.task_id < 1:
            p.error("--worker requires --task-id >= 1")
    elif args.merge:
        if not args.plan_json:
            p.error("--merge requires --plan-json")
    else:
        if not args.phenotype:
            p.error("planner mode requires --phenotype")
        if not args.ancestry:
            p.error("planner mode requires --ancestry")

    for name in (
        "ensembl_batch_size",
        "variation_batch_size",
        "screen_batch_size",
        "regulomedb_batch_size",
        "clinvar_batch_size",
    ):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_','-')} must be >= 1")

    if args.ensembl_batch_size > 200:
        p.error("--ensembl-batch-size cannot exceed Ensembl POST maximum of 200")
    if args.variation_batch_size > 200:
        p.error("--variation-batch-size cannot exceed 200")
    if args.screen_batch_size > 200:
        p.error("--screen-batch-size cannot exceed 200")

    for name in (
        "ensembl_max_parallel",
        "variation_max_parallel",
        "screen_max_parallel",
        "regulomedb_max_parallel",
        "clinvar_max_parallel",
    ):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_','-')} must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    try:
        if args.test_one:
            test_one_variant_mode(args)
        elif args.worker:
            worker_mode(args)
        elif args.merge:
            merge_mode(args)
        else:
            planner_mode(args)

    except Exception as exc:
        banner("STEP07 FAILED")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

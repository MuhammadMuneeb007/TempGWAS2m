#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
GWAS2m STEP 11
FORMAL, TISSUE-AGNOSTIC MOLECULAR-QTL COLOCALIZATION
===============================================================================

Version: 3.2.0

SCIENTIFIC PURPOSE
------------------
For every successfully fine-mapped GWAS locus, test whether the GWAS signal
shares a causal genetic signal with molecular phenotypes such as:

    eQTL          gene expression
    sQTL          RNA splicing
    pQTL          protein abundance
    meQTL         DNA methylation
    caQTL         chromatin accessibility
    hQTL          histone modification
    isoQTL        transcript/isoform usage
    exonQTL       exon expression
    apaQTL        alternative polyadenylation
    metaboliteQTL metabolite abundance
    otherQTL      any user-registered quantitative molecular phenotype

The formal statistical method is:

    coloc::runsusie()
    coloc::coloc.susie()

using dense regional statistics and signed LD.

THIS SCRIPT DOES NOT USE CLPP/LCLPP AS FORMAL COLOCALIZATION.

IMPORTANT DISTINCTIONS
----------------------
A. GTEx v11 Step08
   Existing Step08 outputs are kept as descriptive overlap evidence:
       "Did one of our fine-mapped variants occur among significant GTEx
        eQTL/sQTL associations?"
   This is useful annotation/discovery, but is NOT formal colocalization.

B. Formal coloc
   Formal coloc uses dense regional molecular-QTL summary statistics.
   SNPs are NOT selected by molecular-QTL significance for the formal test.

C. Missing resources
   If a molecular-QTL class is not publicly/configurably available, the script
   writes:
       RESOURCE_NOT_CONFIGURED
   It never converts missing data into a biological "NO".

PUBLIC RESOURCE SUPPORT
-----------------------
1. GTEx v11 descriptive resources:
   Step08 uses GTEx v11. Step11 keeps those overlaps as descriptive evidence only.
   They are NOT the formal eQTL-Catalogue GTEx input.

   --setup-resources does not redownload GTEx v11 by default.
   --setup-full-gtex optionally downloads the GTEx v11 resources used by Step08.

2. eQTL Catalogue (formal public molecular-QTL source):
   The Catalogue's GTEx study is GTEx v8 (49 tissues), not GTEx v11.
   Formal results are therefore labeled eQTL_Catalogue_GTEx_v8.
   Normal analysis is LOCAL-ONLY. The script reads the already-downloaded
   dense .tsv.gz + .tbi files under resources/coloc/eqtl_catalogue/dense
   and queries only the current GWAS locus with tabix. It never downloads
   eQTL-Catalogue LBF files and never uses remote tabix during analysis.

   Supported quantification mappings include:
       ge / microarray  -> eQTL
       leafcutter/majiq -> sQTL
       tx               -> isoQTL
       exon             -> exonQTL
       txrev            -> isoQTL (transcript usage family)
       protein          -> pQTL

   By default, the automatic catalogue scan is restricted to GTEx-labelled
   datasets plus Sun_2018 protein QTL data. Use --catalog-all-studies to expand
   across all catalogue studies/conditions.

3. Custom molecular-QTL registry:
   resources/coloc/custom_qtl_registry.tsv

   This supports pQTL, meQTL, caQTL, hQTL, isoQTL, metaboliteQTL, or any future
   QTL dataset. If SOURCE_URL is provided and LOCAL_PATH is absent, the script
   downloads it automatically.

WHY NOT AUTO-DOWNLOAD EVERY POSSIBLE QTL DATABASE?
--------------------------------------------------
There is no single canonical, unrestricted, harmonized all-tissue dataset for
all of pQTL/meQTL/caQTL/hQTL/metabolite-QTL data. Some datasets are controlled
access, cohort-specific, or use different licenses. The script therefore:
  * automatically downloads verified public GTEx/eQTL-Catalogue resources;
  * supports arbitrary additional public URLs through the registry;
  * explicitly reports unavailable resources.

EXPECTED GWAS2m STRUCTURE
-------------------------
06_finemapping/<phenotype>/<ancestry>/<GCST>/
    <GCST>_finemapped_variants.tsv.gz
    finemapping_summary.json
    loci/
        L001/
            reference_qc_matched_variants.tsv.gz
            matched_variants.tsv
            susie_fit.rds
        ...

08_qtl/<phenotype>/<ancestry>/<GCST>/
    <GCST>_GTEx_eQTL.tsv.gz
    <GCST>_GTEx_sQTL.tsv.gz
    <GCST>_GTEx_QTL_variant_summary.tsv
    ...

resources/1000G/<ANC>/
    chr1_<ANC>_GRCh38.pgen
    chr1_<ANC>_GRCh38.pvar
    chr1_<ANC>_GRCh38.psam
    ...

OUTPUT
------
11_coloc_formal/<phenotype>/<ancestry>/
    coloc_manifest.tsv
    coloc_excluded_studies.tsv

    <GCST>/
        <GCST>_Step08_GTEx_overlap.tsv
        <GCST>_molecular_qtl_candidates.tsv
        <GCST>_formal_coloc_summary.tsv
        <GCST>_locus_qtl_matrix.tsv
        <GCST>_locus_evidence_table.tsv
        resource_status.tsv
        coloc_summary.json

        candidates/
            <candidate_id>/
                common_stats.tsv.gz
                common_reference_ids.txt
                external_LD.unphased.vcor1.bin
                external_LD.unphased.vcor1.bin.vars
                coloc_susie_summary.tsv
                coloc_susie_variant_results.tsv.gz
                prior_sensitivity.tsv
                diagnostics.tsv
                run_metadata.json
                coloc.log

MODES
-----
1. Download/setup public resources:
   python Step11_Formal_Molecular_Colocalization.py --setup-resources

2. Create/check planner:
   python Step11_Formal_Molecular_Colocalization.py \
       --phenotype "parkinson's disease" \
       --ancestry EUR

3. Run first Parkinson GWAS interactively on a COMPUTE node:
   python Step11_Formal_Molecular_Colocalization.py \
       --phenotype "parkinson's disease" \
       --ancestry EUR \
       --index 1

4. Generated SLURM:
   sbatch --array=1 Step11_Formal_Coloc_parkinson_s_disease_european.sh

The direct --index mode prints the candidate table and final coloc table to
screen.
===============================================================================
"""

from __future__ import annotations

import argparse
import gzip
import io
import time
import fcntl
from collections import OrderedDict
from contextlib import contextmanager
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwas2m_config  # noqa: E402  (SLURM settings: config/slurm.yaml; scheduling only)
from typing import Any, Iterable

import numpy as np
import pandas as pd


STEP11_VERSION = "4.3.0-local-fast-no-lbf"

# SLURM partition/time/memory/CPUs/throttle: config/slurm.yaml (stage "coloc_core").
DEFAULT_CPUS = 4
DEFAULT_REMOTE_DELAY = 0.0
DEFAULT_LD_MAF = 0.01
DEFAULT_LD_MISMATCH_S = 0.10

DEFAULT_CANDIDATE_P = 1e-5
DEFAULT_MIN_COMMON_SNPS = 100
DEFAULT_MAX_COMMON_SNPS = 12000
DEFAULT_STRONG_PP4 = 0.80
DEFAULT_SUGGESTIVE_PP4 = 0.50
DEFAULT_MAX_CANDIDATES_PER_CONTEXT_LOCUS = 100

COLOC_PRIOR_P1 = 1e-4
COLOC_PRIOR_P2 = 1e-4
COLOC_PRIOR_P12 = 5e-6
COLOC_PRIOR_P12_LOW = 1e-6
COLOC_PRIOR_P12_HIGH = 1e-5

SUPPORTED_QTL_TYPES = [
    "eQTL",
    "sQTL",
    "pQTL",
    "meQTL",
    "caQTL",
    "hQTL",
    "isoQTL",
    "exonQTL",
    "apaQTL",
    "metaboliteQTL",
    "otherQTL",
]

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

GTEX_BASE = "https://storage.googleapis.com/adult-gtex/bulk-qtl/v11"

GTEX_PUBLIC_RESOURCES = {
    "GTEX_V11_EQTL_SIGNIFICANT": {
        "url": f"{GTEX_BASE}/single-tissue-cis-qtl/GTEx_Analysis_v11_eQTL.tar",
        "path": "resources/gtex/v11/qtl/GTEx_Analysis_v11_eQTL.tar",
        "extract_dir": "resources/gtex/v11/qtl/eQTL",
        "large": True,
    },
    "GTEX_V11_SQTL_SIGNIFICANT": {
        "url": f"{GTEX_BASE}/single-tissue-cis-qtl/GTEx_Analysis_v11_sQTL.tar",
        "path": "resources/gtex/v11/qtl/GTEx_Analysis_v11_sQTL.tar",
        "extract_dir": "resources/gtex/v11/qtl/sQTL",
        "large": True,
    },
    "GTEX_V11_APAQTL_SIGNIFICANT": {
        "url": f"{GTEX_BASE}/single-tissue-cis-qtl/GTEx_Analysis_v11_apaQTL.tar",
        "path": "resources/gtex/v11/qtl/GTEx_Analysis_v11_apaQTL.tar",
        "extract_dir": "resources/gtex/v11/qtl/apaQTL",
        "large": False,
    },
    "GTEX_V11_EQTL_SUSIE": {
        "url": f"{GTEX_BASE}/susie-qtl/GTEx_Analysis_v11_eQTL_SuSiE.tar",
        "path": "resources/gtex/v11/susie/GTEx_Analysis_v11_eQTL_SuSiE.tar",
        "extract_dir": "resources/gtex/v11/susie/eQTL_SuSiE",
        "large": False,
    },
    "GTEX_V11_SQTL_SUSIE": {
        "url": f"{GTEX_BASE}/susie-qtl/GTEx_Analysis_v11_sQTL_SuSiE.tar",
        "path": "resources/gtex/v11/susie/GTEx_Analysis_v11_sQTL_SuSiE.tar",
        "extract_dir": "resources/gtex/v11/susie/sQTL_SuSiE",
        "large": False,
    },
    "GTEX_V11_APAQTL_SUSIE": {
        "url": f"{GTEX_BASE}/susie-qtl/GTEx_Analysis_v11_apaQTL_SuSiE.tar",
        "path": "resources/gtex/v11/susie/GTEx_Analysis_v11_apaQTL_SuSiE.tar",
        "extract_dir": "resources/gtex/v11/susie/apaQTL_SuSiE",
        "large": False,
    },
}

EQTL_CATALOG_PUBLIC = {
    "TABIX_PATHS": {
        "url": (
            "https://raw.githubusercontent.com/eQTL-Catalogue/"
            "eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths.tsv"
        ),
        "path": "resources/coloc/eqtl_catalogue/tabix_ftp_paths.tsv",
    },
    "TABIX_IMPORTED": {
        "url": (
            "https://raw.githubusercontent.com/eQTL-Catalogue/"
            "eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths_imported.tsv"
        ),
        "path": "resources/coloc/eqtl_catalogue/tabix_ftp_paths_imported.tsv",
    },
    "DATASET_METADATA_R8": {
        "url": (
            "https://raw.githubusercontent.com/eQTL-Catalogue/"
            "eQTL-Catalogue-resources/master/data_tables/dataset_metadata_r8_beta.tsv"
        ),
        "path": "resources/coloc/eqtl_catalogue/dataset_metadata_r8_beta.tsv",
    },
    "POPULATION_ASSIGNMENTS": {
        "url": (
            "https://raw.githubusercontent.com/eQTL-Catalogue/"
            "eQTL-Catalogue-resources/master/data_tables/population_assignments.tsv"
        ),
        "path": "resources/coloc/eqtl_catalogue/population_assignments.tsv",
    },
}

CUSTOM_REGISTRY_COLUMNS = [
    "ENABLED",
    "DATASET",
    "QTL_TYPE",
    "TISSUE",
    "CONDITION",
    "SOURCE_URL",
    "LOCAL_PATH",
    "ACCESS_MODE",
    "SAMPLE_SIZE",
    "TRAIT_COLUMN",
    "GENE_COLUMN",
    "VARIANT_COLUMN",
    "CHR_COLUMN",
    "POS_COLUMN",
    "REF_COLUMN",
    "ALT_COLUMN",
    "BETA_COLUMN",
    "SE_COLUMN",
    "P_COLUMN",
    "MAF_COLUMN",
    "EFFECT_ALLELE_MODE",
    "EFFECT_ALLELE_COLUMN",
    "GENOME_BUILD",
    "NOTES",
]


def banner(text: str) -> None:
    print()
    print("=" * 124)
    print(text)
    print("=" * 124)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = normalize_text(value)
    if key not in ANCESTRY_ALIASES:
        raise ValueError(
            f"Unsupported ancestry {value!r}; use EUR, AFR, EAS, SAS, AMR."
        )
    return ANCESTRY_ALIASES[key]


def clean_chr(value: Any) -> int | None:
    if value is None:
        return None
    text = re.sub(r"^chr", "", str(value).strip(), flags=re.I)
    try:
        chrom = int(float(text))
    except Exception:
        return None
    return chrom if 1 <= chrom <= 22 else None


def clean_allele(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip().upper()
    if text in {"", ".", "NA", "NAN", "NONE", "<NA>"}:
        return None
    return text


def make_variant_key(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    chrom = clean_chr(chrom)
    ref = clean_allele(ref)
    alt = clean_allele(alt)
    try:
        pos = int(float(pos))
    except Exception:
        return ""
    if chrom is None or pos <= 0 or ref is None or alt is None:
        return ""
    return f"{chrom}:{pos}:{ref}:{alt}"


def parse_variant_id(value: Any) -> tuple[int | None, int | None, str | None, str | None]:
    text = str(value).strip()
    match = re.match(
        r"^(?:chr)?(\d+)[:_](\d+)[:_]([^:_]+)[:_]([^:_]+)(?:[:_]b\d+)?$",
        text,
        flags=re.I,
    )
    if not match:
        return None, None, None, None
    chrom, pos, ref, alt = match.groups()
    return clean_chr(chrom), int(pos), clean_allele(ref), clean_allele(alt)


def safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except Exception:
        return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def boolish(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if str(candidate).lower() in lookup:
            return lookup[str(candidate).lower()]
    return None


def safe_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    lower = path.name.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path, **kwargs)
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False, **kwargs)
    return pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
        **kwargs,
    )


def validate_walltime(value: str) -> str:
    text = str(value).strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Walltime must be HH:MM:SS or D-HH:MM:SS")
    h, m, s = map(int, parts)
    total = days * 86400 + h * 3600 + m * 60 + s
    if total <= 0:
        raise argparse.ArgumentTypeError("Walltime must be > 0")
    try:
        gwas2m_config.validate_walltime(value)  # configured max_walltime only
    except gwas2m_config.SlurmConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def run_command(
    command: list[str | Path],
    *,
    log_file: Path | None = None,
    capture: bool = False,
    check: bool = True,
) -> str:
    command = [str(x) for x in command]
    print("$ " + " ".join(shlex.quote(x) for x in command), flush=True)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output = []
    handle = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = log_file.open("w", encoding="utf-8")
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            output.append(line)
            if not capture:
                print(line, end="", flush=True)
            if handle:
                handle.write(line)
                handle.flush()
    finally:
        if handle:
            handle.close()
    rc = proc.wait()
    text = "".join(output)
    if check and rc != 0:
        raise RuntimeError(
            f"Command failed with exit code {rc}:\n"
            + " ".join(shlex.quote(x) for x in command)
            + "\n"
            + text[-8000:]
        )
    return text



@contextmanager
def exclusive_file_lock(path: Path):
    """Cross-process advisory lock for shared downloads/aggregate writes on Linux HPC."""
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_REGION_CACHE: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
_REGION_CACHE_LIMIT = 32
_CUSTOM_FILE_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()
_CUSTOM_FILE_CACHE_LIMIT = 2
_LBF_TABLE_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()
_LBF_TABLE_CACHE_LIMIT = 2


def _cache_put(cache: OrderedDict, key, value, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def cached_custom_file(path: Path) -> pd.DataFrame:
    key = str(path.resolve())
    if key in _CUSTOM_FILE_CACHE:
        _CUSTOM_FILE_CACHE.move_to_end(key)
        return _CUSTOM_FILE_CACHE[key]
    x = read_table(path)
    _cache_put(_CUSTOM_FILE_CACHE, key, x, _CUSTOM_FILE_CACHE_LIMIT)
    return x


def read_resource_env(root: Path) -> dict[str, str]:
    path = root / "resource_paths.env"
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.replace("export ", "").strip()] = value.strip().strip('"').strip("'")
    return result


def find_executable(
    root: Path,
    explicit: str | None,
    env_name: str,
    command: str,
) -> Path:
    env = read_resource_env(root)
    candidates = [
        explicit,
        os.environ.get(env_name),
        env.get(env_name),
        shutil.which(command),
        root / "envs" / "pipeline" / "bin" / command,
    ]
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return path.resolve()
    raise FileNotFoundError(
        f"Could not locate executable {command!r}; provide it explicitly."
    )


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    complete = Path(str(destination) + ".complete")

    with exclusive_file_lock(destination):
        if (
            not force
            and destination.exists()
            and destination.stat().st_size > 0
            and complete.exists()
        ):
            print(f"[CACHE] {destination}")
            return destination

        part = Path(str(destination) + ".part")
        banner("DOWNLOAD")
        print(f"URL : {url}")
        print(f"OUT : {destination}")

        aria2 = shutil.which("aria2c")
        curl = shutil.which("curl")
        wget = shutil.which("wget")

        if aria2:
            run_command([
                aria2,
                "--continue=true",
                "--max-connection-per-server=4",
                "--split=4",
                "--min-split-size=16M",
                "--file-allocation=none",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--max-tries=10",
                "--retry-wait=5",
                "--dir", str(destination.parent),
                "--out", part.name,
                url,
            ])
        elif curl:
            run_command([
                curl, "--fail", "--location", "--retry", "8",
                "--retry-delay", "5", "--continue-at", "-",
                "--output", str(part), url,
            ])
        elif wget:
            run_command([
                wget, "--continue", "--tries=8", "--timeout=60",
                "-O", str(part), url,
            ])
        else:
            raise RuntimeError("aria2c, curl, or wget is required for downloads.")

        if not part.exists() or part.stat().st_size == 0:
            raise RuntimeError(f"Download produced no usable file: {part}")

        part.replace(destination)
        complete.write_text(
            f"downloaded_utc={utcnow()}\\nurl={url}\\n",
            encoding="utf-8",
        )
        return destination


def extract_tar(archive: Path, out_dir: Path, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = out_dir / ".extract.complete"
    if not force and complete.exists():
        print(f"[CACHE] extracted {out_dir}")
        return out_dir

    banner("EXTRACT")
    print(f"TAR : {archive}")
    print(f"OUT : {out_dir}")

    with tarfile.open(archive, "r:*") as tf:
        root = out_dir.resolve()
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path inside tar archive: {member.name}")
        tf.extractall(out_dir)

    complete.write_text(
        f"extracted_utc={utcnow()}\narchive={archive}\n",
        encoding="utf-8",
    )
    return out_dir


def write_custom_registry_template(root: Path, overwrite: bool = False) -> Path:
    path = root / "resources" / "coloc" / "custom_qtl_registry.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path

    base = {
        "ENABLED": False,
        "TISSUE": "",
        "CONDITION": "",
        "SOURCE_URL": "",
        "LOCAL_PATH": "",
        "ACCESS_MODE": "file",
        "SAMPLE_SIZE": "",
        "GENE_COLUMN": "",
        "VARIANT_COLUMN": "variant",
        "CHR_COLUMN": "",
        "POS_COLUMN": "",
        "REF_COLUMN": "",
        "ALT_COLUMN": "",
        "BETA_COLUMN": "beta",
        "SE_COLUMN": "se",
        "P_COLUMN": "pvalue",
        "MAF_COLUMN": "maf",
        "EFFECT_ALLELE_MODE": "ALT",
        "EFFECT_ALLELE_COLUMN": "",
        "GENOME_BUILD": "GRCh38",
        "NOTES": "",
    }

    specs = [
        ("YOUR_MEQTL_DATASET", "meQTL", "cpg_id", "DNA methylation QTL"),
        ("YOUR_CAQTL_DATASET", "caQTL", "peak_id", "chromatin accessibility QTL"),
        ("YOUR_HQTL_DATASET", "hQTL", "peak_id", "histone-mark QTL"),
        ("YOUR_METABOLITE_QTL_DATASET", "metaboliteQTL", "metabolite_id", "metabolite QTL"),
        ("YOUR_OTHER_QTL_DATASET", "otherQTL", "trait_id", "future/custom molecular QTL"),
    ]

    rows = []
    for dataset, qtl_type, trait_col, note in specs:
        row = dict(base)
        row.update({
            "DATASET": dataset,
            "QTL_TYPE": qtl_type,
            "TRAIT_COLUMN": trait_col,
            "NOTES": (
                f"Configure a licensed/public dense {note} summary-statistics source. "
                "If SOURCE_URL is provided, the file is downloaded automatically."
            ),
        })
        rows.append(row)

    pd.DataFrame(rows, columns=CUSTOM_REGISTRY_COLUMNS).to_csv(
        path, sep="\t", index=False
    )
    return path


def setup_public_resources(
    root: Path,
    include_large_gtex: bool,
    force: bool,
) -> None:
    banner("STEP 11 RESOURCE SETUP")

    print("Formal source: eQTL Catalogue metadata + LOCAL dense regional tabix retrieval (NO LBF)")
    for name, spec in EQTL_CATALOG_PUBLIC.items():
        print(f"\\n[{name}]")
        download_file(spec["url"], root / spec["path"], force=force)

    if include_large_gtex:
        print(
            "\\n[OPTIONAL] Downloading GTEx v11 resources for Step08/descriptive compatibility. "
            "These are NOT labeled as the formal eQTL-Catalogue GTEx source."
        )
        for name, spec in GTEX_PUBLIC_RESOURCES.items():
            archive = download_file(spec["url"], root / spec["path"], force=force)
            extract_tar(archive, root / spec["extract_dir"], force=force)
    else:
        print(
            "\\n[SKIP] GTEx v11 archives are not needed for the formal eQTL-Catalogue path. "
            "Use --setup-full-gtex only if you also want Step08-compatible GTEx v11 resources."
        )

    registry = write_custom_registry_template(root)
    banner("RESOURCE SETUP COMPLETE")
    print(f"Custom registry:\\n  {registry}")


def arguments():
    parser = argparse.ArgumentParser(
        description="Formal tissue-agnostic molecular-QTL colocalization with coloc.susie."
    )
    parser.add_argument("--phenotype")
    parser.add_argument("--ancestry")
    parser.add_argument("--index", type=int)
    parser.add_argument("--study")

    parser.add_argument("--setup-resources", action="store_true")
    parser.add_argument("--setup-full-gtex", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--no-auto-download", action="store_true",
        help="Legacy compatibility flag; normal analysis is always local-only in this build."
    )

    parser.add_argument(
        "--registry",
        default="resources/coloc/custom_qtl_registry.tsv",
    )
    parser.add_argument(
        "--qtl-types",
        default=",".join(SUPPORTED_QTL_TYPES),
    )

    parser.add_argument("--catalog-all-studies", action="store_true")
    parser.add_argument("--catalog-study-regex", default="")

    parser.add_argument("--candidate-p", type=float, default=DEFAULT_CANDIDATE_P)
    parser.add_argument("--min-common-snps", type=int, default=DEFAULT_MIN_COMMON_SNPS)
    parser.add_argument("--max-common-snps", type=int, default=DEFAULT_MAX_COMMON_SNPS)
    parser.add_argument("--strong-pp4", type=float, default=DEFAULT_STRONG_PP4)
    parser.add_argument("--suggestive-pp4", type=float, default=DEFAULT_SUGGESTIVE_PP4)
    parser.add_argument(
        "--max-candidates-per-context-locus",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_CONTEXT_LOCUS,
    )

    parser.add_argument("--partition", default=None)
    parser.add_argument("--time", type=validate_walltime, default=None)
    parser.add_argument("--memory", default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--remote-delay", type=float, default=DEFAULT_REMOTE_DELAY)
    parser.add_argument("--ld-maf", type=float, default=DEFAULT_LD_MAF)
    parser.add_argument("--ld-mismatch-s", type=float, default=DEFAULT_LD_MISMATCH_S)

    parser.add_argument("--plink2")
    parser.add_argument("--tabix")
    parser.add_argument("--rscript")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gwas2m_config.apply_stage_defaults(args, "coloc_core")
    return args


def project_paths(root: Path, phenotype: str, ancestry_label: str) -> dict[str, Path]:
    p = slugify(phenotype)
    a = slugify(ancestry_label)
    return {
        "S06": root / "06_finemapping" / p / a,
        "S08": root / "08_qtl" / p / a,
        "OUT": root / "11_coloc_formal" / p / a,
        "LOG": root / "logs" / "step11_formal_coloc" / p / a,
    }


def discover_step06_studies(s06: Path) -> tuple[list[str], list[dict]]:
    studies = []
    excluded = []
    if not s06.exists():
        return studies, excluded

    accessions = set()
    manifest = s06 / "finemapping_manifest.tsv"
    if manifest.exists():
        x = pd.read_csv(manifest, sep="\t", dtype=str, low_memory=False)
        if "STUDY_ACCESSION" in x.columns:
            accessions.update(
                v.strip()
                for v in x["STUDY_ACCESSION"].dropna().astype(str)
                if v.strip()
            )

    for directory in s06.glob("GCST*"):
        if directory.is_dir():
            accessions.add(directory.name)

    for accession in sorted(accessions):
        study_dir = s06 / accession
        summary = safe_json(study_dir / "finemapping_summary.json")
        fine_file = study_dir / f"{accession}_finemapped_variants.tsv.gz"
        status = str(summary.get("STATUS", "")).upper()
        n_success = safe_int(summary.get("N_LOCI_SUCCESS")) or 0

        reason = ""
        if not summary:
            reason = "Step06 summary missing/unreadable"
        elif status == "NO_SIGNIFICANT_LOCI":
            reason = "No significant loci"
        elif n_success <= 0 and status not in {"COMPLETE", "PARTIAL"}:
            reason = f"No successful loci; status={status}"
        elif not fine_file.exists() or fine_file.stat().st_size == 0:
            reason = "Combined Step06 fine-mapped file missing/empty"

        if reason:
            excluded.append({
                "STUDY_ACCESSION": accession,
                "STEP06_STATUS": status,
                "REASON": reason,
            })
        else:
            studies.append(accession)

    return studies, excluded


def list_loci(study_dir: Path) -> list[str]:
    loci_dir = study_dir / "loci"
    if loci_dir.exists():
        loci = sorted(
            d.name for d in loci_dir.iterdir()
            if d.is_dir() and re.fullmatch(r"L\d+", d.name)
        )
        if loci:
            return loci

    accession = study_dir.name
    fine_file = study_dir / f"{accession}_finemapped_variants.tsv.gz"
    x = read_table(fine_file)
    if not x.empty and "LOCUS_ID" in x.columns:
        return sorted(x["LOCUS_ID"].dropna().astype(str).unique())
    return []


def find_resolved_metadata(s06: Path, accession: str) -> Path | None:
    candidates = [
        s06 / "metadata" / accession / "resolved_metadata.json",
        s06 / accession / "metadata" / "resolved_metadata.json",
        s06 / accession / "resolved_metadata.json",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    for path in (s06 / accession).rglob("resolved_metadata.json"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def load_gwas_metadata(s06: Path, accession: str) -> dict:
    metadata_path = find_resolved_metadata(s06, accession)
    meta = safe_json(metadata_path) if metadata_path else {}

    n_cases = safe_int(meta.get("N_CASES"))
    n_controls = safe_int(meta.get("N_CONTROLS"))
    resolved_n = safe_int(meta.get("RESOLVED_GWAS_N"))

    raw_cc = meta.get("CASE_CONTROL_STUDY")
    if isinstance(raw_cc, str):
        z = raw_cc.strip().lower()
        if z in {"true", "yes", "1"}:
            raw_cc = True
        elif z in {"false", "no", "0"}:
            raw_cc = False
        else:
            raw_cc = None

    if n_cases and n_controls:
        coloc_n = n_cases + n_controls
        trait_type = "cc"
        case_proportion = n_cases / coloc_n
        type_source = "CASE_CONTROL_COUNTS"
        metadata_status = "OK"
    elif raw_cc is True:
        coloc_n = resolved_n
        trait_type = "cc"
        case_proportion = None
        type_source = "CASE_CONTROL_STUDY_FLAG"
        metadata_status = "CASE_CONTROL_COUNTS_MISSING"
    elif raw_cc is False:
        coloc_n = resolved_n
        trait_type = "quant"
        case_proportion = None
        type_source = "CASE_CONTROL_STUDY_FALSE"
        metadata_status = "OK"
    else:
        # Backward-compatible fallback for older Step06 metadata.  The pipeline
        # records this explicitly so it is not mistaken for verified metadata.
        coloc_n = resolved_n
        trait_type = "quant"
        case_proportion = None
        type_source = "INFERRED_QUANT_NO_CASE_COUNTS"
        metadata_status = "OK_WITH_UNVERIFIED_TRAIT_TYPE"

    return {
        "METADATA_FILE": str(metadata_path) if metadata_path else "",
        "RESOLVED_GWAS_N": resolved_n,
        "COLOC_GWAS_N": coloc_n,
        "N_CASES": n_cases,
        "N_CONTROLS": n_controls,
        "GWAS_TYPE": trait_type,
        "GWAS_TYPE_SOURCE": type_source,
        "METADATA_STATUS": metadata_status,
        "CASE_PROPORTION": case_proportion,
    }


def load_gwas_locus(study_dir: Path, locus_id: str) -> pd.DataFrame:
    locus_dir = study_dir / "loci" / locus_id
    paths = [
        locus_dir / "reference_qc_matched_variants.tsv.gz",
        locus_dir / "matched_variants.tsv",
    ]
    source = next(
        (p for p in paths if p.exists() and p.stat().st_size > 0),
        None,
    )
    if source is None:
        raise FileNotFoundError(
            f"No Step06 dense matched GWAS table for {locus_id} under {locus_dir}"
        )

    x = read_table(source)

    chr_col = first_existing(x.columns, ["REFERENCE_CHR", "CHR"])
    pos_col = first_existing(x.columns, ["REFERENCE_POS", "POS"])
    ref_col = first_existing(x.columns, ["REFERENCE_REF"])
    alt_col = first_existing(x.columns, ["REFERENCE_ALT"])
    rid_col = first_existing(x.columns, ["REFERENCE_ID"])
    beta_col = first_existing(x.columns, ["BETA_REF"])
    se_col = first_existing(x.columns, ["SE"])
    p_col = first_existing(x.columns, ["P"])

    required = {
        "CHR": chr_col, "POS": pos_col, "REF": ref_col, "ALT": alt_col,
        "REFERENCE_ID": rid_col, "BETA_REF": beta_col, "SE": se_col, "P": p_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise RuntimeError(
            f"{source} is missing fields needed for formal coloc: {missing}"
        )

    out = pd.DataFrame({
        "CHR": x[chr_col].map(clean_chr),
        "POS": numeric(x[pos_col]),
        "REF": x[ref_col].map(clean_allele),
        "ALT": x[alt_col].map(clean_allele),
        "REFERENCE_ID": x[rid_col].astype(str),
        "GWAS_BETA": numeric(x[beta_col]),
        "GWAS_SE": numeric(x[se_col]),
        "GWAS_P": numeric(x[p_col]),
    })

    out["VARIANT_KEY"] = [
        make_variant_key(c, p, r, a)
        for c, p, r, a in zip(out["CHR"], out["POS"], out["REF"], out["ALT"])
    ]

    out = out[
        out["VARIANT_KEY"].ne("")
        & out["GWAS_BETA"].notna()
        & out["GWAS_SE"].notna()
        & (out["GWAS_SE"] > 0)
        & out["REFERENCE_ID"].notna()
    ].copy()

    out = (
        out.sort_values(["CHR", "POS", "GWAS_P"], kind="stable")
        .drop_duplicates("VARIANT_KEY", keep="first")
        .reset_index(drop=True)
    )
    return out


def locus_region(gwas: pd.DataFrame) -> tuple[int, int, int]:
    chroms = sorted(set(gwas["CHR"].dropna().astype(int)))
    if len(chroms) != 1:
        raise RuntimeError(f"Locus spans multiple chromosomes: {chroms}")
    return chroms[0], int(gwas["POS"].min()), int(gwas["POS"].max())


def load_step08_overlap(s08: Path, accession: str) -> pd.DataFrame:
    study_dir = s08 / accession
    frames = []

    for qtl_type, file_name in [
        ("eQTL", f"{accession}_GTEx_eQTL.tsv.gz"),
        ("sQTL", f"{accession}_GTEx_sQTL.tsv.gz"),
    ]:
        path = study_dir / file_name
        x = read_table(path)
        if x.empty:
            continue

        locus_col = first_existing(x.columns, ["LOCUS_ID"])
        tissue_col = first_existing(x.columns, ["GTEX_TISSUE", "TISSUE"])
        gene_col = first_existing(x.columns, ["GTEX_GENE_ID", "GENE_ID"])
        phenotype_col = first_existing(x.columns, ["GTEX_PHENOTYPE_ID", "PHENOTYPE_ID"])
        variant_col = first_existing(x.columns, ["VEP_ID", "VARIANT_KEY"])
        p_col = first_existing(x.columns, ["GTEX_PVALUE", "P"])

        if locus_col is None or tissue_col is None:
            continue

        work = pd.DataFrame({
            "LOCUS_ID": x[locus_col].astype(str),
            "QTL_TYPE": qtl_type,
            "TISSUE": x[tissue_col].fillna("").astype(str),
            "GENE_ID": (
                x[gene_col].fillna("").astype(str)
                if gene_col else ""
            ),
            "MOLECULAR_TRAIT_ID": (
                x[phenotype_col].fillna("").astype(str)
                if phenotype_col
                else (
                    x[gene_col].fillna("").astype(str)
                    if gene_col else ""
                )
            ),
            "VARIANT": (
                x[variant_col].fillna("").astype(str)
                if variant_col else ""
            ),
            "P": numeric(x[p_col]) if p_col else np.nan,
        })
        frames.append(work)

    if not frames:
        return pd.DataFrame(columns=[
            "LOCUS_ID", "QTL_TYPE", "TISSUE", "GENE_ID",
            "MOLECULAR_TRAIT_ID", "N_STEP08_ROWS",
            "N_STEP08_VARIANTS", "STEP08_MIN_P",
        ])

    x = pd.concat(frames, ignore_index=True)
    return (
        x.groupby(
            ["LOCUS_ID", "QTL_TYPE", "TISSUE", "GENE_ID", "MOLECULAR_TRAIT_ID"],
            dropna=False,
            as_index=False,
        )
        .agg(
            N_STEP08_ROWS=("LOCUS_ID", "size"),
            N_STEP08_VARIANTS=("VARIANT", lambda s: int(pd.Series(s).nunique())),
            STEP08_MIN_P=("P", "min"),
        )
    )


def ensure_catalogue_metadata(root: Path, auto_download: bool) -> None:
    for spec in EQTL_CATALOG_PUBLIC.values():
        path = root / spec["path"]
        if (not path.exists() or path.stat().st_size == 0) and auto_download:
            download_file(spec["url"], path)


def normalize_catalogue_manifest(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    rename = {}
    mapping = {
        "study_id": ["study_id"],
        "dataset_id": ["dataset_id"],
        "study_label": ["study_label", "study"],
        "sample_group": ["sample_group", "qtl_group"],
        "tissue_id": ["tissue_id"],
        "tissue_label": ["tissue_label", "tissue"],
        "condition_label": ["condition_label", "condition"],
        "sample_size": ["sample_size", "n"],
        "quant_method": ["quant_method", "quantification_method"],
        "ftp_path": ["ftp_path", "sumstats_path"],
        "ftp_cs_path": ["ftp_cs_path"],
        "ftp_lbf_path": ["ftp_lbf_path"],
    }
    for target, aliases in mapping.items():
        col = first_existing(x.columns, aliases)
        if col:
            rename[col] = target
    x = x.rename(columns=rename)

    for column in mapping:
        if column not in x.columns:
            x[column] = ""
    return x


def load_eqtl_catalogue(root: Path, auto_download: bool) -> pd.DataFrame:
    ensure_catalogue_metadata(root, auto_download)
    paths = [
        root / EQTL_CATALOG_PUBLIC["TABIX_PATHS"]["path"],
        root / EQTL_CATALOG_PUBLIC["TABIX_IMPORTED"]["path"],
    ]
    frames = [
        normalize_catalogue_manifest(path)
        for path in paths
        if path.exists() and path.stat().st_size > 0
    ]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["dataset_id", "ftp_path"], keep="last")
        .reset_index(drop=True)
    )


def quant_method_to_qtl_type(value: Any) -> str | None:
    q = normalize_text(value)
    if q in {"ge", "gene expression", "gene", "microarray"}:
        return "eQTL"
    if "leafcutter" in q or "majiq" in q or q in {"sqtl", "splicing"}:
        return "sQTL"
    if q in {"tx", "transcript", "transcript usage", "txrev", "txrevise"}:
        return "isoQTL"
    if q in {"exon", "exon expression"}:
        return "exonQTL"
    if "protein" in q or "aptamer" in q or q in {"prot", "pqtl"}:
        return "pQTL"
    if "apa" in q or "polyadenyl" in q:
        return "apaQTL"
    return None


def select_catalogue_contexts(
    catalogue: pd.DataFrame,
    args,
    allowed_types: set[str],
) -> pd.DataFrame:
    if catalogue.empty:
        return catalogue

    x = catalogue.copy()
    x["QTL_TYPE"] = x["quant_method"].map(quant_method_to_qtl_type)

    unmapped = (
        x.loc[x["QTL_TYPE"].isna(), "quant_method"]
        .fillna("")
        .astype(str)
        .value_counts()
    )
    if not unmapped.empty:
        print("[INFO] eQTL Catalogue quant_method values not mapped to a QTL class:")
        for method, count in unmapped.items():
            print(f"       {method or '<EMPTY>'}: {count}")

    x = x[x["QTL_TYPE"].notna()].copy()
    x = x[
        x["QTL_TYPE"].astype(str).str.lower().isin(allowed_types)
    ]

    if args.catalog_study_regex:
        pattern = re.compile(args.catalog_study_regex, flags=re.I)
        x = x[
            x["study_label"].astype(str).map(
                lambda value: bool(pattern.search(value))
            )
        ]
    elif not args.catalog_all_studies:
        labels = x["study_label"].astype(str)
        x = x[
            labels.str.contains("GTEx", case=False, regex=False)
            | labels.str.fullmatch("Sun_2018", case=False)
        ].copy()

    x = x[x["ftp_path"].astype(str).str.strip().ne("")].copy()

    if not x.empty:
        print("[INFO] Formal-coloc source counts by study/QTL type:")
        counts = (
            x.groupby(["study_label", "QTL_TYPE"], dropna=False)
            .size()
            .reset_index(name="N_CONTEXTS")
        )
        print(counts.to_string(index=False))
        if x["study_label"].astype(str).str.contains("GTEx_v8", case=False, regex=False).any():
            print("[INFO] Formal GTEx data from eQTL Catalogue are labelled GTEx_v8; Step08 remains GTEx_v11.")

    return x.reset_index(drop=True)


_HEADER_CACHE: dict[str, list[str]] = {}


def _is_remote_source(value: str) -> bool:
    return bool(re.match(r"^(?:https?|ftp)://", str(value).strip(), flags=re.I))


def _catalogue_basename(value: str) -> str:
    text = str(value).strip().split("?", 1)[0].rstrip("/")
    return text.rsplit("/", 1)[-1]


def resolve_catalogue_local_sources(root: Path, contexts: pd.DataFrame) -> pd.DataFrame:
    """Resolve every selected eQTL-Catalogue context to local BGZF + TBI files.

    Normal Step11 analysis is deliberately network-free.  The manifest FTP URL
    is retained only as provenance; ACCESS_PATH is always a local filesystem path.
    """
    if contexts.empty:
        return contexts.copy()

    dense_root = root / "resources" / "coloc" / "eqtl_catalogue" / "dense"
    if not dense_root.exists():
        raise FileNotFoundError(
            f"Local dense eQTL-Catalogue root is missing: {dense_root}"
        )

    by_name: dict[str, list[Path]] = {}
    for path in dense_root.rglob("*.tsv.gz"):
        if path.is_file() and path.stat().st_size > 0:
            by_name.setdefault(path.name, []).append(path.resolve())

    rows = []
    for _, row in contexts.iterrows():
        out = row.to_dict()
        dataset_id = str(row.get("dataset_id", "")).strip()
        remote = str(row.get("ftp_path", "")).strip()
        basename = _catalogue_basename(remote)

        candidates: list[Path] = []
        if dataset_id and basename:
            candidates.append(dense_root / dataset_id / basename)
        if basename:
            candidates.append(dense_root / basename)
            candidates.extend(by_name.get(basename, []))

        # Prefer a path whose parent is the dataset_id when duplicates exist.
        candidates = [Path(x) for x in dict.fromkeys(map(str, candidates))]
        candidates.sort(
            key=lambda x: (
                0 if dataset_id and x.parent.name == dataset_id else 1,
                len(str(x)),
            )
        )

        data = None
        index = None
        for candidate in candidates:
            if not candidate.exists() or candidate.stat().st_size == 0:
                continue
            tbi = Path(str(candidate) + ".tbi")
            if tbi.exists() and tbi.stat().st_size > 0:
                data = candidate.resolve()
                index = tbi.resolve()
                break

        out["REMOTE_SOURCE_URL"] = remote
        out["ACCESS_PATH"] = str(data) if data else ""
        out["ACCESS_INDEX"] = str(index) if index else ""
        out["ACCESS_MODE"] = "LOCAL_TABIX" if data and index else "MISSING_LOCAL"
        rows.append(out)

    result = pd.DataFrame(rows)
    missing = result[result["ACCESS_MODE"].ne("LOCAL_TABIX")]
    print(
        f"[LOCAL QTL] resolved {len(result) - len(missing)}/{len(result)} "
        "catalogue contexts to local .tsv.gz + .tbi"
    )
    if not missing.empty:
        cols = [c for c in ["dataset_id", "study_label", "QTL_TYPE", "tissue_label", "ftp_path"] if c in missing.columns]
        preview = missing[cols].head(30).to_string(index=False)
        raise RuntimeError(
            "Step11 is LOCAL-ONLY and some selected catalogue contexts are missing locally.\n"
            f"Missing contexts: {len(missing)}\n{preview}\n"
            "Download/repair those dense .tsv.gz + .tbi files before running Step11."
        )
    return result.reset_index(drop=True)


def tabix_header(tabix: Path, source: str) -> list[str]:
    source = str(source).strip()
    if _is_remote_source(source):
        raise RuntimeError(
            f"Remote tabix is disabled in this Step11 build: {source}"
        )
    path = Path(source)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Local tabix source missing/empty: {path}")

    key = str(path.resolve())
    if key in _HEADER_CACHE:
        return _HEADER_CACHE[key]

    text = run_command([tabix, "-H", path], capture=True, check=False)
    header_lines = [line for line in text.splitlines() if line.startswith("#")]

    # Some BGZF files have a plain first header line rather than a tabix '#'-header.
    if not header_lines:
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                first = handle.readline().rstrip("\n\r")
            if first:
                header_lines = [first]
        except Exception:
            pass

    if not header_lines:
        raise RuntimeError(f"Could not retrieve header for local file: {path}")

    header = header_lines[-1].lstrip("#").split("\t")
    _HEADER_CACHE[key] = header
    return header


def check_local_tabix_connectivity(tabix: Path, contexts: pd.DataFrame) -> None:
    if contexts.empty:
        return
    source = str(contexts.iloc[0]["ACCESS_PATH"])
    header = tabix_header(tabix, source)
    if not header:
        raise RuntimeError(f"Local eQTL-Catalogue header is empty: {source}")
    print(f"[OK] Local tabix/header: {source}")


def tabix_region(
    tabix: Path,
    source: str,
    chrom: int,
    start: int,
    end: int,
) -> pd.DataFrame:
    source = str(source).strip()
    if _is_remote_source(source):
        raise RuntimeError(f"Remote tabix is disabled: {source}")
    path = Path(source)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Local tabix source missing/empty: {path}")
    tbi = Path(str(path) + ".tbi")
    if not tbi.exists() or tbi.stat().st_size == 0:
        raise FileNotFoundError(f"Local tabix index missing/empty: {tbi}")

    key = (str(path.resolve()), int(chrom), int(start), int(end))
    if key in _REGION_CACHE:
        _REGION_CACHE.move_to_end(key)
        return _REGION_CACHE[key].copy()

    header = tabix_header(tabix, str(path))
    errors = []
    successful_query = False

    for region in (f"{chrom}:{start}-{end}", f"chr{chrom}:{start}-{end}"):
        proc = subprocess.run(
            [str(tabix), str(path), region],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
        if proc.returncode != 0:
            errors.append(proc.stderr.strip() or f"returncode={proc.returncode}")
            continue

        successful_query = True
        lines = [
            line for line in proc.stdout.splitlines()
            if line and not line.startswith("#")
        ]
        if lines:
            result = pd.read_csv(
                io.StringIO("\n".join(lines)),
                sep="\t",
                names=header,
                dtype=str,
                low_memory=False,
            )
            _cache_put(_REGION_CACHE, key, result, _REGION_CACHE_LIMIT)
            return result.copy()

    if not successful_query and errors:
        raise RuntimeError(
            f"tabix failed for local file {path}: {errors[-1][:500]}"
        )

    result = pd.DataFrame(columns=header)
    _cache_put(_REGION_CACHE, key, result, _REGION_CACHE_LIMIT)
    return result.copy()

def standardize_eqtl_catalogue_region(
    x: pd.DataFrame,
    context: pd.Series,
) -> pd.DataFrame:
    if x.empty:
        return pd.DataFrame()

    trait_col = first_existing(x.columns, ["molecular_trait_id", "phenotype_id", "trait_id"])
    gene_col = first_existing(x.columns, ["gene_id"])
    chr_col = first_existing(x.columns, ["chromosome", "chr", "chrom"])
    pos_col = first_existing(x.columns, ["position", "pos", "bp"])
    ref_col = first_existing(x.columns, ["ref", "reference_allele"])
    alt_col = first_existing(x.columns, ["alt", "alternate_allele"])
    variant_col = first_existing(x.columns, ["variant", "variant_id"])
    p_col = first_existing(x.columns, ["pvalue", "p_value", "pval", "pval_nominal"])
    beta_col = first_existing(x.columns, ["beta", "slope", "effect_size"])
    se_col = first_existing(x.columns, ["se", "slope_se", "standard_error"])
    maf_col = first_existing(x.columns, ["maf", "af", "allele_frequency"])
    an_col = first_existing(x.columns, ["an", "allele_number"])

    if trait_col is None or beta_col is None or se_col is None:
        return pd.DataFrame()

    out = pd.DataFrame(index=x.index)

    if all(c is not None for c in [chr_col, pos_col, ref_col, alt_col]):
        out["CHR"] = x[chr_col].map(clean_chr)
        out["POS"] = numeric(x[pos_col])
        out["REF"] = x[ref_col].map(clean_allele)
        out["ALT"] = x[alt_col].map(clean_allele)
    elif variant_col:
        parsed = x[variant_col].map(parse_variant_id)
        out["CHR"] = parsed.map(lambda z: z[0])
        out["POS"] = parsed.map(lambda z: z[1])
        out["REF"] = parsed.map(lambda z: z[2])
        out["ALT"] = parsed.map(lambda z: z[3])
    else:
        return pd.DataFrame()

    out["VARIANT_KEY"] = [
        make_variant_key(c, p, r, a)
        for c, p, r, a in zip(out["CHR"], out["POS"], out["REF"], out["ALT"])
    ]
    out["MOLECULAR_TRAIT_ID"] = x[trait_col].fillna("").astype(str)
    out["GENE_ID"] = x[gene_col].fillna("").astype(str) if gene_col else ""
    out["P"] = numeric(x[p_col]) if p_col else np.nan
    out["BETA_ALT"] = numeric(x[beta_col])
    out["SE"] = numeric(x[se_col])
    out["MAF"] = numeric(x[maf_col]) if maf_col else np.nan
    out["AN"] = numeric(x[an_col]) if an_col else np.nan

    # eQTL Catalogue states ALT is the effect allele.
    out["BETA_REF"] = -out["BETA_ALT"]

    out["DATASET_ID"] = str(context.get("dataset_id", ""))
    out["STUDY_LABEL"] = str(context.get("study_label", ""))
    out["TISSUE"] = str(context.get("tissue_label", ""))
    out["CONDITION"] = str(context.get("condition_label", ""))
    out["QTL_TYPE"] = str(context.get("QTL_TYPE", ""))
    metadata_n = safe_int(context.get("sample_size"))
    if out["AN"].notna().any():
        out["QTL_N"] = np.floor(out["AN"] / 2.0)
        out["QTL_N"] = out["QTL_N"].where(out["QTL_N"] > 0, metadata_n)
    else:
        out["QTL_N"] = metadata_n
    out["SOURCE_URL"] = str(context.get("ftp_path", ""))

    out = out[
        out["VARIANT_KEY"].ne("")
        & out["MOLECULAR_TRAIT_ID"].str.strip().ne("")
        & out["BETA_REF"].notna()
        & out["SE"].notna()
        & (out["SE"] > 0)
    ].copy()

    return (
        out.drop_duplicates(
            ["MOLECULAR_TRAIT_ID", "VARIANT_KEY"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def discover_catalogue_candidates_for_locus(
    tabix: Path,
    contexts: pd.DataFrame,
    locus_id: str,
    gwas: pd.DataFrame,
    args,
) -> tuple[pd.DataFrame, list[dict]]:
    chrom, start, end = locus_region(gwas)
    candidates = []
    audit = []

    for i, context in contexts.iterrows():
        url = str(context["ACCESS_PATH"])
        qtl_type = str(context["QTL_TYPE"])
        status = "OK"
        error = ""
        n_region = 0
        n_candidates = 0

        print(
            f"[CATALOG {i+1}/{len(contexts)}] "
            f"{qtl_type:<8} | "
            f"{context.get('study_label',''):<18} | "
            f"{context.get('tissue_label',''):<35}",
            flush=True,
        )

        try:
            raw = tabix_region(tabix, url, chrom, start, end)
            standardized = standardize_eqtl_catalogue_region(raw, context)
            n_region = len(standardized)

            if standardized.empty:
                status = "NO_REGION_ROWS"
            else:
                selected = standardized[
                    standardized["P"].notna()
                    & (standardized["P"] <= args.candidate_p)
                ].copy()

                if selected.empty:
                    status = "NO_CANDIDATE_P_LE_THRESHOLD"
                else:
                    grouped = (
                        selected.groupby(
                            ["MOLECULAR_TRAIT_ID", "GENE_ID"],
                            dropna=False,
                            as_index=False,
                        )
                        .agg(
                            DISCOVERY_MIN_P=("P", "min"),
                            N_DISCOVERY_ROWS=("P", "size"),
                        )
                        .sort_values("DISCOVERY_MIN_P", kind="stable")
                        .head(args.max_candidates_per_context_locus)
                    )
                    n_candidates = len(grouped)

                    for _, item in grouped.iterrows():
                        candidates.append({
                            "SOURCE": "EQTL_CATALOGUE_DENSE_REGION",
                            "LOCUS_ID": locus_id,
                            "DATASET": str(context.get("FORMAL_DATASET", context.get("study_label", ""))),
                            "DATASET_ID": str(context.get("dataset_id", "")),
                            "QTL_TYPE": qtl_type,
                            "TISSUE": str(context.get("tissue_label", "")),
                            "CONDITION": str(context.get("condition_label", "")),
                            "MOLECULAR_TRAIT_ID": str(item["MOLECULAR_TRAIT_ID"]),
                            "GENE_ID": str(item["GENE_ID"]),
                            "DISCOVERY_MIN_P": item["DISCOVERY_MIN_P"],
                            "N_DISCOVERY_ROWS": item["N_DISCOVERY_ROWS"],
                            "QTL_N": safe_int(context.get("sample_size")),
                            "SOURCE_URL": str(context.get("REMOTE_SOURCE_URL", context.get("ftp_path", ""))),
                            "ACCESS_PATH": url,
                            "ACCESS_MODE": "local_tabix",
                        })

        except Exception as exc:
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"

        audit.append({
            "LOCUS_ID": locus_id,
            "DATASET_ID": context.get("dataset_id", ""),
            "STUDY_LABEL": context.get("study_label", ""),
            "QTL_TYPE": qtl_type,
            "TISSUE": context.get("tissue_label", ""),
            "N_REGION_ROWS": n_region,
            "N_CANDIDATES": n_candidates,
            "STATUS": status,
            "ERROR": error,
        })

    return pd.DataFrame(candidates), audit


def load_custom_registry(root: Path, path_value: str) -> pd.DataFrame:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        write_custom_registry_template(root)

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=CUSTOM_REGISTRY_COLUMNS)

    x = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in CUSTOM_REGISTRY_COLUMNS:
        if col not in x.columns:
            x[col] = ""
    x["ENABLED_BOOL"] = x["ENABLED"].map(boolish)
    return x


def ensure_custom_registry_files(
    root: Path,
    registry: pd.DataFrame,
    auto_download: bool = False,
) -> pd.DataFrame:
    """Resolve custom registry LOCAL_PATH values without downloading anything."""
    if registry.empty:
        return registry

    x = registry.copy()
    for idx, row in x[x["ENABLED_BOOL"]].iterrows():
        local_text = str(row["LOCAL_PATH"]).strip()
        if not local_text:
            continue
        local = Path(local_text).expanduser()
        if not local.is_absolute():
            local = root / local
        x.at[idx, "LOCAL_PATH"] = str(local.resolve())
    return x

def custom_coordinates(x: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    chr_col = str(row["CHR_COLUMN"]).strip()
    pos_col = str(row["POS_COLUMN"]).strip()
    ref_col = str(row["REF_COLUMN"]).strip()
    alt_col = str(row["ALT_COLUMN"]).strip()
    variant_col = str(row["VARIANT_COLUMN"]).strip()

    out = pd.DataFrame(index=x.index)

    if all(
        c and c in x.columns
        for c in [chr_col, pos_col, ref_col, alt_col]
    ):
        out["CHR"] = x[chr_col].map(clean_chr)
        out["POS"] = numeric(x[pos_col])
        out["REF"] = x[ref_col].map(clean_allele)
        out["ALT"] = x[alt_col].map(clean_allele)
    elif variant_col and variant_col in x.columns:
        parsed = x[variant_col].map(parse_variant_id)
        out["CHR"] = parsed.map(lambda z: z[0])
        out["POS"] = parsed.map(lambda z: z[1])
        out["REF"] = parsed.map(lambda z: z[2])
        out["ALT"] = parsed.map(lambda z: z[3])
    else:
        raise RuntimeError(
            f"Custom registry row {row.get('DATASET')} has no usable coordinates."
        )

    out["VARIANT_KEY"] = [
        make_variant_key(c, p, r, a)
        for c, p, r, a in zip(out["CHR"], out["POS"], out["REF"], out["ALT"])
    ]
    return out


def standardize_custom_region(raw: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    trait_col = str(row["TRAIT_COLUMN"]).strip()
    gene_col = str(row["GENE_COLUMN"]).strip()
    beta_col = str(row["BETA_COLUMN"]).strip()
    se_col = str(row["SE_COLUMN"]).strip()
    p_col = str(row["P_COLUMN"]).strip()
    maf_col = str(row["MAF_COLUMN"]).strip()

    for needed in [trait_col, beta_col, se_col]:
        if not needed or needed not in raw.columns:
            raise RuntimeError(
                f"Custom dataset {row.get('DATASET')} missing column {needed!r}."
            )

    out = custom_coordinates(raw, row)
    out["MOLECULAR_TRAIT_ID"] = raw[trait_col].fillna("").astype(str)
    out["GENE_ID"] = (
        raw[gene_col].fillna("").astype(str)
        if gene_col and gene_col in raw.columns
        else ""
    )
    out["BETA_RAW"] = numeric(raw[beta_col])
    out["SE"] = numeric(raw[se_col])
    out["P"] = numeric(raw[p_col]) if p_col and p_col in raw.columns else np.nan
    out["MAF"] = numeric(raw[maf_col]) if maf_col and maf_col in raw.columns else np.nan

    mode = str(row["EFFECT_ALLELE_MODE"]).strip().upper() or "ALT"

    if mode == "ALT":
        out["BETA_REF"] = -out["BETA_RAW"]
    elif mode == "REF":
        out["BETA_REF"] = out["BETA_RAW"]
    elif mode == "COLUMN":
        ea_col = str(row["EFFECT_ALLELE_COLUMN"]).strip()
        if not ea_col or ea_col not in raw.columns:
            raise RuntimeError(
                "EFFECT_ALLELE_MODE=COLUMN requires EFFECT_ALLELE_COLUMN."
            )
        values = []
        for ea, ref, alt, beta in zip(
            raw[ea_col].map(clean_allele),
            out["REF"], out["ALT"], out["BETA_RAW"],
        ):
            if ea == ref:
                values.append(beta)
            elif ea == alt:
                values.append(-beta)
            else:
                values.append(np.nan)
        out["BETA_REF"] = values
    else:
        raise RuntimeError(f"Unsupported EFFECT_ALLELE_MODE={mode!r}")

    out["DATASET"] = str(row["DATASET"])
    out["QTL_TYPE"] = str(row["QTL_TYPE"])
    out["TISSUE"] = str(row["TISSUE"])
    out["CONDITION"] = str(row["CONDITION"])
    out["QTL_N"] = safe_int(row["SAMPLE_SIZE"])

    out = out[
        out["VARIANT_KEY"].ne("")
        & out["MOLECULAR_TRAIT_ID"].str.strip().ne("")
        & out["BETA_REF"].notna()
        & out["SE"].notna()
        & (out["SE"] > 0)
    ].copy()

    return (
        out.drop_duplicates(
            ["MOLECULAR_TRAIT_ID", "VARIANT_KEY"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def load_custom_region(
    tabix: Path,
    row: pd.Series,
    chrom: int,
    start: int,
    end: int,
) -> pd.DataFrame:
    access = str(row["ACCESS_MODE"]).strip().lower() or "file"

    if access in {"tabix", "remote_tabix"}:
        source = str(row["LOCAL_PATH"]).strip()
        if not source:
            raise RuntimeError(
                f"{row.get('DATASET')} has no LOCAL_PATH; remote custom QTL access is disabled."
            )
        return standardize_custom_region(
            tabix_region(tabix, source, chrom, start, end),
            row,
        )

    path_text = str(row["LOCAL_PATH"]).strip()
    if not path_text:
        raise RuntimeError(f"{row.get('DATASET')} has no LOCAL_PATH.")

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)

    if Path(str(path) + ".tbi").exists():
        raw = tabix_region(tabix, str(path), chrom, start, end)
    else:
        raw = cached_custom_file(path)

    standardized = standardize_custom_region(raw, row)
    return standardized[
        (standardized["CHR"] == chrom)
        & (standardized["POS"] >= start)
        & (standardized["POS"] <= end)
    ].copy()


def discover_custom_candidates_for_locus(
    tabix: Path,
    registry: pd.DataFrame,
    locus_id: str,
    gwas: pd.DataFrame,
    args,
    allowed_types: set[str],
) -> tuple[pd.DataFrame, list[dict]]:
    chrom, start, end = locus_region(gwas)
    candidates = []
    audit = []

    for _, row in registry[registry["ENABLED_BOOL"]].iterrows():
        qtl_type = str(row["QTL_TYPE"])
        if qtl_type.lower() not in allowed_types:
            continue

        status = "OK"
        error = ""
        n_region = 0
        n_candidates = 0

        try:
            region = load_custom_region(tabix, row, chrom, start, end)
            n_region = len(region)

            selected = region[
                region["P"].notna()
                & (region["P"] <= args.candidate_p)
            ].copy()

            if region.empty:
                status = "NO_REGION_ROWS"
            elif selected.empty:
                status = "NO_CANDIDATE_P_LE_THRESHOLD"
            else:
                grouped = (
                    selected.groupby(
                        ["MOLECULAR_TRAIT_ID", "GENE_ID"],
                        dropna=False,
                        as_index=False,
                    )
                    .agg(
                        DISCOVERY_MIN_P=("P", "min"),
                        N_DISCOVERY_ROWS=("P", "size"),
                    )
                    .sort_values("DISCOVERY_MIN_P", kind="stable")
                    .head(args.max_candidates_per_context_locus)
                )
                n_candidates = len(grouped)

                for _, item in grouped.iterrows():
                    candidates.append({
                        "SOURCE": "CUSTOM_QTL_REGISTRY",
                        "LOCUS_ID": locus_id,
                        "DATASET": str(row["DATASET"]),
                        "DATASET_ID": str(row["DATASET"]),
                        "QTL_TYPE": qtl_type,
                        "TISSUE": str(row["TISSUE"]),
                        "CONDITION": str(row["CONDITION"]),
                        "MOLECULAR_TRAIT_ID": str(item["MOLECULAR_TRAIT_ID"]),
                        "GENE_ID": str(item["GENE_ID"]),
                        "DISCOVERY_MIN_P": item["DISCOVERY_MIN_P"],
                        "N_DISCOVERY_ROWS": item["N_DISCOVERY_ROWS"],
                        "QTL_N": safe_int(row["SAMPLE_SIZE"]),
                        "SOURCE_URL": str(row["SOURCE_URL"]),
                        "ACCESS_MODE": "custom_registry",
                    })

        except Exception as exc:
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"

        audit.append({
            "LOCUS_ID": locus_id,
            "DATASET": row["DATASET"],
            "QTL_TYPE": qtl_type,
            "TISSUE": row["TISSUE"],
            "N_REGION_ROWS": n_region,
            "N_CANDIDATES": n_candidates,
            "STATUS": status,
            "ERROR": error,
        })

    return pd.DataFrame(candidates), audit


def context_from_catalogue_candidate(
    contexts: pd.DataFrame,
    candidate: pd.Series,
) -> pd.Series | None:
    if contexts.empty:
        return None
    match = contexts[
        contexts["dataset_id"].astype(str)
        == str(candidate.get("DATASET_ID", ""))
    ]
    return None if match.empty else match.iloc[0]


def dense_qtl_for_candidate(
    tabix: Path,
    candidate: pd.Series,
    gwas: pd.DataFrame,
    catalogue_contexts: pd.DataFrame,
    custom_registry: pd.DataFrame,
) -> pd.DataFrame:
    chrom, start, end = locus_region(gwas)
    source = str(candidate["SOURCE"])

    if source == "EQTL_CATALOGUE_DENSE_REGION":
        context = context_from_catalogue_candidate(catalogue_contexts, candidate)
        if context is None:
            return pd.DataFrame()
        region = standardize_eqtl_catalogue_region(
            tabix_region(tabix, str(context["ACCESS_PATH"]), chrom, start, end),
            context,
        )

    elif source == "CUSTOM_QTL_REGISTRY":
        enabled = custom_registry[custom_registry["ENABLED_BOOL"]].copy()
        match = enabled[
            (enabled["DATASET"].astype(str) == str(candidate["DATASET"]))
            & (
                enabled["QTL_TYPE"].astype(str).str.lower()
                == str(candidate["QTL_TYPE"]).lower()
            )
            & (enabled["TISSUE"].astype(str) == str(candidate["TISSUE"]))
        ]
        if match.empty:
            return pd.DataFrame()
        region = load_custom_region(tabix, match.iloc[0], chrom, start, end)
    else:
        return pd.DataFrame()

    if region.empty:
        return region

    return region[
        region["MOLECULAR_TRAIT_ID"].astype(str)
        == str(candidate["MOLECULAR_TRAIT_ID"])
    ].reset_index(drop=True)


def reference_prefix(root: Path, ancestry_code: str, chrom: int) -> Path:
    return (
        root / "resources" / "1000G" / ancestry_code
        / f"chr{chrom}_{ancestry_code}_GRCh38"
    )


def build_external_ld(
    root: Path,
    plink2: Path,
    ancestry_code: str,
    common: pd.DataFrame,
    out_dir: Path,
    threads: int,
    maf: float = DEFAULT_LD_MAF,
) -> tuple[Path, Path, list[str], Path]:
    """Build/cache ONE signed-LD matrix + frequency table per GWAS locus.

    Candidate-specific QTL tests subset this locus-wide matrix in R, so PLINK2
    is not rerun for every gene/protein/splicing trait.
    """
    chroms = sorted(set(common["CHR"].dropna().astype(int)))
    if len(chroms) != 1:
        raise RuntimeError(f"Cannot construct one LD matrix across chromosomes: {chroms}")
    chrom = chroms[0]

    prefix = reference_prefix(root, ancestry_code, chrom)
    for ext in [".pgen", ".pvar", ".psam"]:
        path = Path(str(prefix) + ext)
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing 1000G reference file: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ids_file = out_dir / "common_reference_ids.txt"
    ld_prefix = out_dir / "external_LD"
    ld_file = Path(str(ld_prefix) + ".unphased.vcor1.bin")
    vars_file = Path(str(ld_prefix) + ".unphased.vcor1.bin.vars")
    freq_file = out_dir / "external_FREQ.afreq"

    # Fast resume: reuse a validated locus-wide LD/frequency cache.
    if ld_file.exists() and vars_file.exists() and freq_file.exists():
        order = [x.strip() for x in vars_file.read_text().splitlines() if x.strip()]
        expected = len(order) * len(order) * 4
        if order and ld_file.stat().st_size == expected and freq_file.stat().st_size > 0:
            print(f"[CACHE] locus LD/frequency: {out_dir}")
            return ld_file, vars_file, order, freq_file

    ids = common["REFERENCE_ID"].dropna().astype(str).drop_duplicates()
    ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")

    run_command(
        [
            plink2,
            "--pfile", prefix,
            "--extract", ids_file,
            "--maf", str(maf),
            "--geno", "0.05",
            "--min-alleles", "2",
            "--max-alleles", "2",
            "--r-unphased", "square", "bin4", "ref-based", "yes-really",
            "--threads", str(threads),
            "--out", ld_prefix,
        ],
        log_file=out_dir / "plink_ld.log",
    )

    if not ld_file.exists() or not vars_file.exists():
        raise RuntimeError("PLINK2 did not generate signed-r LD output.")

    order = [line.strip() for line in vars_file.read_text().splitlines() if line.strip()]
    expected = len(order) * len(order) * 4
    actual = ld_file.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"LD validation failed: p={len(order)}, expected={expected} bytes, actual={actual}"
        )

    freq_prefix = out_dir / "external_FREQ"
    run_command(
        [
            plink2,
            "--pfile", prefix,
            "--extract", vars_file,
            "--freq",
            "--threads", str(threads),
            "--out", freq_prefix,
        ],
        log_file=out_dir / "plink_freq.log",
    )
    if not freq_file.exists() or freq_file.stat().st_size == 0:
        raise RuntimeError("PLINK2 did not generate reference allele frequencies.")

    return ld_file, vars_file, order, freq_file

def load_reference_maf(freq_file: Path) -> dict[str, float]:
    if not freq_file.exists() or freq_file.stat().st_size == 0:
        raise FileNotFoundError(f"Missing PLINK frequency file: {freq_file}")

    x = pd.read_csv(freq_file, sep=r"\s+", dtype=str, comment=None)
    if "ID" not in x.columns:
        raise RuntimeError(
            f"PLINK frequency file lacks ID column: {freq_file}; columns={list(x.columns)}"
        )
    freq_col = "ALT_FREQS" if "ALT_FREQS" in x.columns else (
        "ALT_FREQ" if "ALT_FREQ" in x.columns else None
    )
    if freq_col is None:
        raise RuntimeError(
            f"PLINK frequency file lacks ALT frequency column: {freq_file}; columns={list(x.columns)}"
        )

    af = pd.to_numeric(
        x[freq_col].astype(str).str.split(",").str[0],
        errors="coerce",
    )
    maf = np.minimum(af, 1.0 - af)
    return dict(zip(x["ID"].astype(str), maf))


def step06_susie_paths(study_dir: Path, locus_id: str, ancestry_code: str) -> dict[str, Path]:
    locus = study_dir / "loci" / locus_id
    return {
        "fit": locus / "susie" / "susie_fit.rds",
        "diag": locus / "susie" / "susie_diagnostics.tsv",
        "vars": locus / f"{ancestry_code}_LD.unphased.vcor1.bin.vars",
    }


def prepare_gwas_susie_map(
    study_dir: Path,
    locus_id: str,
    ancestry_code: str,
    gwas: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path | None, Path | None]:
    paths = step06_susie_paths(study_dir, locus_id, ancestry_code)
    if not paths["fit"].exists() or not paths["vars"].exists():
        return None, None

    order = [x.strip() for x in paths["vars"].read_text().splitlines() if x.strip()]
    mapping = gwas[["REFERENCE_ID", "VARIANT_KEY"]].drop_duplicates("REFERENCE_ID")
    order_map = {v: i for i, v in enumerate(order)}
    mapping = mapping[mapping["REFERENCE_ID"].isin(order_map)].copy()
    mapping["_ORDER"] = mapping["REFERENCE_ID"].map(order_map)
    mapping = mapping.sort_values("_ORDER").drop(columns="_ORDER")

    if len(mapping) != len(order):
        return None, None

    path = out_dir / "gwas_susie_variant_map.tsv"
    mapping.to_csv(path, sep="\t", index=False)
    return paths["fit"], path


def read_step06_s_rss(study_dir: Path, locus_id: str) -> float:
    diag = study_dir / "loci" / locus_id / "susie" / "susie_diagnostics.tsv"
    x = read_table(diag)
    if x.empty or "ESTIMATE_S_RSS" not in x.columns:
        return np.nan
    vals = numeric(x["ESTIMATE_S_RSS"]).dropna()
    return float(vals.iloc[0]) if not vals.empty else np.nan


def catalogue_context_for_candidate(
    contexts: pd.DataFrame,
    candidate: pd.Series,
) -> pd.Series | None:
    if contexts.empty:
        return None
    x = contexts[
        contexts["dataset_id"].astype(str)
        == str(candidate.get("DATASET_ID", ""))
    ]
    return None if x.empty else x.iloc[0]


def download_catalogue_lbf(root: Path, context: pd.Series) -> Path | None:
    """LBF downloads are intentionally disabled in the local-fast build."""
    return None

def load_lbf_table(path: Path) -> pd.DataFrame:
    key = str(path.resolve())
    if key in _LBF_TABLE_CACHE:
        _LBF_TABLE_CACHE.move_to_end(key)
        return _LBF_TABLE_CACHE[key]

    header = pd.read_csv(path, sep="\t", compression="infer", nrows=0)
    lbf_cols = [c for c in header.columns if str(c).startswith("lbf_variable")]
    required = ["molecular_trait_id", "variant"] + lbf_cols
    usecols = [c for c in required if c in header.columns]
    if "molecular_trait_id" not in usecols or "variant" not in usecols or not lbf_cols:
        raise RuntimeError(f"Unexpected eQTL Catalogue LBF format: {path}")

    x = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        low_memory=False,
    )
    _cache_put(_LBF_TABLE_CACHE, key, x, _LBF_TABLE_CACHE_LIMIT)
    return x


def write_qtl_lbf_trait(
    root: Path,
    context: pd.Series,
    trait_id: str,
    out_dir: Path,
) -> Path | None:
    path = download_catalogue_lbf(root, context)
    if path is None:
        return None
    x = load_lbf_table(path)
    sub = x[x["molecular_trait_id"].astype(str) == str(trait_id)].copy()
    if sub.empty:
        return None

    parsed = sub["variant"].map(parse_variant_id)
    sub["VARIANT_KEY"] = [
        make_variant_key(z[0], z[1], z[2], z[3]) for z in parsed
    ]
    sub = sub[sub["VARIANT_KEY"].ne("")].copy()
    sub = sub.drop_duplicates("VARIANT_KEY", keep="first")
    lbf_cols = [c for c in sub.columns if str(c).startswith("lbf_variable")]
    if not lbf_cols:
        return None
    out = out_dir / "qtl_lbf_trait.tsv.gz"
    sub[["VARIANT_KEY"] + lbf_cols].to_csv(
        out, sep="\t", index=False, compression="gzip"
    )
    return out


R_BF_WORKER = r"""
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 11) stop("Expected 11 arguments")

gwas_fit_file <- args[1]
gwas_map_file <- args[2]
qtl_lbf_file <- args[3]
out_summary <- args[4]
out_sensitivity <- args[5]
p1 <- as.numeric(args[6])
p2 <- as.numeric(args[7])
p12 <- as.numeric(args[8])
p12_low <- as.numeric(args[9])
p12_high <- as.numeric(args[10])
out_results <- args[11]

suppressPackageStartupMessages(library(coloc))

fit <- readRDS(gwas_fit_file)
map <- read.delim(gwas_map_file, stringsAsFactors=FALSE)
qtl <- read.delim(qtl_lbf_file, stringsAsFactors=FALSE, check.names=FALSE)

if (is.null(fit$lbf_variable)) stop("Step06 susie fit has no lbf_variable")
gbf <- as.matrix(fit$lbf_variable)
if (ncol(gbf) == nrow(map)) {
  colnames(gbf) <- map$VARIANT_KEY
} else if (nrow(gbf) == nrow(map)) {
  gbf <- t(gbf)
  colnames(gbf) <- map$VARIANT_KEY
} else {
  stop("Step06 lbf_variable dimensions do not match Step06 variant map")
}

lbf_cols <- grep("^lbf_variable", names(qtl), value=TRUE)
if (length(lbf_cols) == 0) stop("QTL LBF file has no lbf_variable columns")
qbf <- t(as.matrix(qtl[, lbf_cols, drop=FALSE]))
colnames(qbf) <- qtl$VARIANT_KEY

# Drop signal rows that are entirely non-finite.
gbf <- gbf[apply(gbf, 1, function(x) any(is.finite(x))),,drop=FALSE]
qbf <- qbf[apply(qbf, 1, function(x) any(is.finite(x))),,drop=FALSE]
if (nrow(gbf) == 0 || nrow(qbf) == 0) stop("No finite LBF signal rows")

run_one <- function(prior12, label) {
  res <- coloc::coloc.bf_bf(
    gbf, qbf,
    p1=p1, p2=p2, p12=prior12,
    overlap.min=0.5,
    trim_by_posterior=TRUE
  )
  sm <- as.data.frame(res$summary)
  if (nrow(sm) == 0) return(NULL)
  sm$PRIOR_SET <- label
  sm$P12 <- prior12
  sm$METHOD <- "coloc.bf_bf_precomputed_susie"
  list(res=res, summary=sm)
}

base <- run_one(p12, "DEFAULT")
if (is.null(base)) stop("coloc.bf_bf returned no signal pairs")
low <- run_one(p12_low, "LOW_P12")
high <- run_one(p12_high, "HIGH_P12")

write.table(base$summary, out_summary, sep="\t", quote=FALSE, row.names=FALSE)
sens <- base$summary
if (!is.null(low)) sens <- rbind(sens, low$summary)
if (!is.null(high)) sens <- rbind(sens, high$summary)
write.table(sens, out_sensitivity, sep="\t", quote=FALSE, row.names=FALSE)

rr <- tryCatch(as.data.frame(base$res$results), error=function(e) data.frame())
write.table(rr, out_results, sep="\t", quote=FALSE, row.names=FALSE)
print(base$summary)
"""


def write_bf_worker(output_root: Path) -> Path:
    path = output_root / "_step11_coloc_bf_bf_worker.R"
    path.write_text(R_BF_WORKER, encoding="utf-8")
    return path


R_WORKER = r"""
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 18) stop("Expected 18 arguments")

stats_file <- args[1]
ld_file <- args[2]
vars_file <- args[3]
out_summary <- args[4]
out_results <- args[5]
out_sensitivity <- args[6]
out_diag <- args[7]
gwas_n <- as.numeric(args[8])
qtl_n <- as.numeric(args[9])
gwas_type <- args[10]
case_prop <- suppressWarnings(as.numeric(args[11]))
p1 <- as.numeric(args[12])
p2 <- as.numeric(args[13])
p12 <- as.numeric(args[14])
p12_low <- as.numeric(args[15])
p12_high <- as.numeric(args[16])
gwas_fit_file <- args[17]
gwas_map_file <- args[18]

suppressPackageStartupMessages(library(coloc))
suppressPackageStartupMessages(library(susieR))

stats <- read.delim(stats_file, header=TRUE, stringsAsFactors=FALSE, check.names=FALSE)
full_vars <- scan(vars_file, what="character", quiet=TRUE)

# Load the locus-wide LD ONCE, then subset it to this molecular-trait candidate.
p <- length(full_vars)
con <- file(ld_file, "rb")
values <- readBin(con, what="numeric", n=p*p, size=4, endian="little")
close(con)
if (length(values) != p*p) stop("LD binary length mismatch")
Rfull <- matrix(values, nrow=p, ncol=p, byrow=TRUE)
rm(values); gc()
Rfull <- (Rfull + t(Rfull))/2
diag(Rfull) <- 1
dimnames(Rfull) <- list(full_vars, full_vars)

idx <- match(as.character(stats$REFERENCE_ID), full_vars)
keep <- which(!is.na(idx))
if (length(keep) < 20) stop("Too few candidate SNPs represented in locus LD")
stats <- stats[keep,,drop=FALSE]
idx <- idx[keep]

# Remove duplicated LD IDs while preserving common_stats order.
dup <- duplicated(idx)
if (any(dup)) {
    stats <- stats[!dup,,drop=FALSE]
    idx <- idx[!dup]
}
R <- Rfull[idx, idx, drop=FALSE]
rm(Rfull); gc()
vars <- as.character(stats$REFERENCE_ID)
dimnames(R) <- list(vars, vars)

bad <- which(!apply(is.finite(R), 1, all) | !apply(is.finite(R), 2, all))
if (length(bad) > 0) {
    cat("Filtering", length(bad), "variants with non-finite LD rows/columns\n")
    keep <- setdiff(seq_len(nrow(R)), bad)
    R <- R[keep, keep, drop=FALSE]
    stats <- stats[keep,,drop=FALSE]
    vars <- vars[keep]
}
if (nrow(R) < 20) stop("Too few SNPs after non-finite LD filtering")

ok <- is.finite(stats$QTL_BETA) &
      is.finite(stats$QTL_SE) & stats$QTL_SE > 0 &
      is.finite(stats$QTL_MAF) &
      stats$QTL_MAF > 0 & stats$QTL_MAF < 0.5
if (!all(ok)) {
    R <- R[ok, ok, drop=FALSE]
    stats <- stats[ok,,drop=FALSE]
    vars <- vars[ok]
}
if (nrow(R) < 20) stop("Too few finite QTL SNPs after MAF/statistic filtering")
diag(R) <- 1
dimnames(R) <- list(vars, vars)

# Reuse Step06 GWAS SuSiE fit instead of rerunning GWAS SuSiE for every QTL.
s1 <- readRDS(gwas_fit_file)
gmap <- read.delim(gwas_map_file, stringsAsFactors=FALSE, check.names=FALSE)
if (!"REFERENCE_ID" %in% names(gmap)) stop("GWAS SuSiE map lacks REFERENCE_ID")
gwas_snps <- as.character(gmap$REFERENCE_ID)
if (is.null(s1$lbf_variable)) stop("Step06 SuSiE fit has no lbf_variable")
if (ncol(s1$lbf_variable) > length(gwas_snps) + 1) {
    stop("Step06 SuSiE lbf_variable has more columns than the variant map")
}
colnames(s1$lbf_variable) <- c(gwas_snps, "null")[seq_len(ncol(s1$lbf_variable))]
if (!is.null(s1$alpha)) {
    if (ncol(s1$alpha) > length(gwas_snps) + 1) stop("Step06 alpha/map dimension mismatch")
    colnames(s1$alpha) <- c(gwas_snps, "null")[seq_len(ncol(s1$alpha))]
}
if (!is.null(s1$pip)) {
    if (length(s1$pip) != length(gwas_snps)) stop("Step06 PIP/map dimension mismatch")
    names(s1$pip) <- gwas_snps
}
if (!is.null(s1$sets$cs) && length(s1$sets$cs)) {
    s1$sets$cs <- lapply(s1$sets$cs, function(x) {
        good <- x[x >= 1 & x <= length(gwas_snps)]
        names(good) <- gwas_snps[good]
        good
    })
}
class(s1) <- unique(c("susie", class(s1)))

# Fine-map only the molecular QTL side for this candidate.
d2 <- list(
    beta=as.numeric(stats$QTL_BETA),
    varbeta=as.numeric(stats$QTL_SE)^2,
    snp=as.character(stats$REFERENCE_ID),
    position=as.integer(stats$POS),
    N=as.numeric(qtl_n),
    type="quant",
    MAF=as.numeric(stats$QTL_MAF),
    LD=R
)
coloc::check_dataset(d2, req=c("type","snp","LD","N"))
cat("QTL CHECK = PASS\n")

z2 <- d2$beta / sqrt(d2$varbeta)
s_qtl <- tryCatch(susieR::estimate_s_rss(z=z2, R=R, n=qtl_n), error=function(e) NA_real_)
diag_df <- data.frame(
    TRAIT="QTL",
    ESTIMATE_S_RSS=s_qtl,
    MAX_ABS_Z=max(abs(z2), na.rm=TRUE),
    N_SNPS=length(z2),
    stringsAsFactors=FALSE
)
write.table(diag_df, out_diag, sep="\t", quote=FALSE, row.names=FALSE)

cat("Running coloc::runsusie for molecular QTL only\n")
s2 <- coloc::runsusie(
    d2, suffix="QTL", maxit=200, repeat_until_convergence=TRUE
)
cat("QTL SuSiE COMPLETE; reusing Step06 GWAS SuSiE\n")

run_susie_coloc <- function(prior12, label) {
    res <- suppressWarnings(coloc::coloc.susie(s1, s2, p1=p1, p2=p2, p12=prior12))
    if (!is.list(res) || is.null(res$summary)) {
        sm <- data.frame(
            NO_SIGNAL_PAIR="YES", METHOD="coloc.susie_step06_reuse",
            PRIOR_SET=label, P12=prior12, stringsAsFactors=FALSE
        )
        return(list(result=list(results=NULL), summary=sm))
    }
    sm <- tryCatch(as.data.frame(res$summary), error=function(e) data.frame())
    if (nrow(sm) == 0) {
        sm <- data.frame(
            NO_SIGNAL_PAIR="YES", METHOD="coloc.susie_step06_reuse",
            PRIOR_SET=label, P12=prior12, stringsAsFactors=FALSE
        )
    } else {
        sm$NO_SIGNAL_PAIR <- "NO"
        sm$METHOD <- "coloc.susie_step06_reuse"
        sm$PRIOR_SET <- label
        sm$P12 <- prior12
    }
    list(result=res, summary=sm)
}

base <- run_susie_coloc(p12, "DEFAULT")
low <- run_susie_coloc(p12_low, "LOW_P12")
high <- run_susie_coloc(p12_high, "HIGH_P12")

write.table(base$summary, out_summary, sep="\t", quote=FALSE, row.names=FALSE)
results_df <- tryCatch(as.data.frame(base$result$results), error=function(e) data.frame())
write.table(results_df, out_results, sep="\t", quote=FALSE, row.names=FALSE)
sens <- rbind(base$summary, low$summary, high$summary)
write.table(sens, out_sensitivity, sep="\t", quote=FALSE, row.names=FALSE)

cat("\n===== COLOC SUMMARY =====\n")
print(base$summary)
cat("\n===== QTL LD MISMATCH DIAGNOSTICS =====\n")
print(diag_df)
"""


def write_r_worker(output_root: Path) -> Path:
    path = output_root / "_step11_coloc_susie_worker.R"
    path.write_text(R_WORKER, encoding="utf-8")
    return path


def verify_r_packages(rscript: Path) -> dict:
    text = run_command(
        [
            rscript,
            "-e",
            (
                'stopifnot(requireNamespace("coloc", quietly=TRUE)); '
                'stopifnot(requireNamespace("susieR", quietly=TRUE)); '
                'cat("coloc=", as.character(packageVersion("coloc")), "\\n", sep=""); '
                'cat("susieR=", as.character(packageVersion("susieR")), "\\n", sep="")'
            ),
        ],
        capture=True,
    )
    result = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def posterior_column(columns: Iterable[str], hypothesis: str) -> str | None:
    target = hypothesis.upper()
    patterns = [f"PP.{target}", f"PP_{target}", f"PP.{target}.ABF", target]
    for column in columns:
        upper = str(column).upper()
        for pattern in patterns:
            if pattern.upper() in upper:
                return column
    return None


def _best_coloc_row(summary: pd.DataFrame) -> pd.Series | None:
    if summary.empty:
        return None
    h4_col = posterior_column(summary.columns, "H4")
    if h4_col is None:
        return summary.iloc[0]
    vals = numeric(summary[h4_col])
    if not vals.notna().any():
        return summary.iloc[0]
    return summary.loc[vals.idxmax()]


def _row_posterior(row: pd.Series | None, summary: pd.DataFrame, hypothesis: str) -> float:
    if row is None:
        return np.nan
    col = posterior_column(summary.columns, hypothesis)
    if col is None:
        return np.nan
    try:
        return float(row[col])
    except Exception:
        return np.nan


def _hit_value(row: pd.Series | None, names: list[str]) -> str:
    if row is None:
        return ""
    lookup = {str(c).lower(): c for c in row.index}
    for name in names:
        if name.lower() in lookup:
            value = row[lookup[name.lower()]]
            if pd.notna(value):
                return str(value)
    return ""


def classify_coloc(
    summary: pd.DataFrame,
    strong_pp4: float,
    suggestive_pp4: float,
) -> dict:
    row = _best_coloc_row(summary)
    values = {
        h: _row_posterior(row, summary, h)
        for h in ["H0", "H1", "H2", "H3", "H4"]
    }
    h3 = values["H3"]
    h4 = values["H4"]

    hit1 = _hit_value(row, ["hit1", "signal1", "cs1"])
    hit2 = _hit_value(row, ["hit2", "signal2", "cs2"])
    method = _hit_value(row, ["METHOD"]) or "coloc.susie"

    if pd.isna(h4):
        classification = "NO_PP_H4_RETURNED"
        strong = "NO"
        shared = "UNKNOWN"
    elif h4 >= strong_pp4 and (pd.isna(h3) or h4 > h3):
        classification = "STRONG_SHARED_SIGNAL"
        strong = "YES"
        shared = "YES"
    elif h4 >= suggestive_pp4:
        classification = "SUGGESTIVE_SHARED_SIGNAL"
        strong = "NO"
        shared = "SUGGESTIVE"
    elif pd.notna(h3) and h3 > h4:
        classification = "BOTH_ASSOCIATED_DIFFERENT_SIGNALS_FAVORED"
        strong = "NO"
        shared = "NO"
    else:
        classification = "NO_STRONG_SHARED_SIGNAL"
        strong = "NO"
        shared = "NO"

    return {
        "MAX_PP_H0": values["H0"],
        "MAX_PP_H1": values["H1"],
        "MAX_PP_H2": values["H2"],
        "MAX_PP_H3": values["H3"],
        "MAX_PP_H4": values["H4"],
        "BEST_HIT1": hit1,
        "BEST_HIT2": hit2,
        "BEST_SIGNAL_PAIR": (f"{hit1}|{hit2}" if hit1 or hit2 else ""),
        "COLOC_METHOD": method,
        "COLOCALIZED_STRONG": strong,
        "SHARED_SIGNAL": shared,
        "COLOC_CLASS": classification,
    }


def sensitivity_pp4_for_pair(
    sensitivity: pd.DataFrame,
    prior_label: str,
    hit1: str,
    hit2: str,
) -> float:
    if sensitivity.empty or "PRIOR_SET" not in sensitivity.columns:
        return np.nan
    sub = sensitivity[sensitivity["PRIOR_SET"].astype(str) == prior_label].copy()
    if sub.empty:
        return np.nan

    # Keep the same signal pair used for the default classification whenever possible.
    if hit1 and hit2:
        h1col = first_existing(sub.columns, ["hit1", "signal1", "cs1"])
        h2col = first_existing(sub.columns, ["hit2", "signal2", "cs2"])
        if h1col and h2col:
            exact = sub[
                (sub[h1col].astype(str) == hit1)
                & (sub[h2col].astype(str) == hit2)
            ]
            if not exact.empty:
                sub = exact

    row = _best_coloc_row(sub)
    return _row_posterior(row, sub, "H4")


def process_candidate(
    root: Path,
    accession: str,
    study_dir: Path,
    ancestry_code: str,
    gwas_meta: dict,
    candidate: pd.Series,
    catalogue_contexts: pd.DataFrame,
    custom_registry: pd.DataFrame,
    plink2: Path,
    tabix: Path,
    rscript: Path,
    r_worker: Path,
    locus_output_dir: Path,
    locus_resources: dict,
    args,
) -> dict:
    locus_id = str(candidate["LOCUS_ID"])
    qtl_type = str(candidate["QTL_TYPE"])
    dataset = str(candidate["DATASET"])
    tissue = str(candidate["TISSUE"])
    condition = str(candidate.get("CONDITION", ""))
    trait_id = str(candidate["MOLECULAR_TRAIT_ID"])
    gene_id = str(candidate.get("GENE_ID", ""))

    candidate_id = slugify(
        f"{locus_id}_{dataset}_{qtl_type}_{tissue}_{condition}_{trait_id}"
    )[:220]
    out = locus_output_dir / "candidates" / candidate_id
    out.mkdir(parents=True, exist_ok=True)
    result_json = out / "candidate_result.json"

    if not args.force and result_json.exists():
        previous = safe_json(result_json)
        if previous.get("STATUS") == "COMPLETE":
            print(f"[RESUME] {candidate_id}")
            return previous

    result = {
        "STUDY_ACCESSION": accession,
        "LOCUS_ID": locus_id,
        "QTL_TYPE": qtl_type,
        "DATASET": dataset,
        "DATASET_ID": candidate.get("DATASET_ID", ""),
        "TISSUE": tissue,
        "CONDITION": condition,
        "MOLECULAR_TRAIT_ID": trait_id,
        "GENE_ID": gene_id,
        "DISCOVERY_SOURCE": candidate.get("SOURCE", ""),
        "DISCOVERY_MIN_P": candidate.get("DISCOVERY_MIN_P", np.nan),
        "QTL_N": candidate.get("QTL_N", np.nan),
        "RESOURCE_AVAILABLE": "YES",
        "CANDIDATE_PRESENT": "YES",
        "COLOC_TESTED": "NO",
        "N_GWAS_VARIANTS": 0,
        "N_QTL_VARIANTS": 0,
        "N_COMMON_SNPS": 0,
        "LD_SOURCE": "",
        "QTL_MAF_SOURCE": "",
        "QTL_MAF_REFERENCE_FALLBACK_N": 0,
        "GWAS_STEP06_S_RSS": read_step06_s_rss(study_dir, locus_id),
        "GWAS_S_RSS": read_step06_s_rss(study_dir, locus_id),
        "QTL_S_RSS": np.nan,
        "LD_MISMATCH_WARNING": "NO",
        "MAX_PP_H0": np.nan,
        "MAX_PP_H1": np.nan,
        "MAX_PP_H2": np.nan,
        "MAX_PP_H3": np.nan,
        "MAX_PP_H4": np.nan,
        "PP_H4_P12_LOW": np.nan,
        "PP_H4_P12_DEFAULT": np.nan,
        "PP_H4_P12_HIGH": np.nan,
        "PRIOR_ROBUST_STRONG": "NO",
        "BEST_HIT1": "",
        "BEST_HIT2": "",
        "BEST_SIGNAL_PAIR": "",
        "COLOC_METHOD": "",
        "COLOCALIZED_STRONG": "NO",
        "SHARED_SIGNAL": "UNKNOWN",
        "COLOC_CLASS": "",
        "STATUS": "",
        "ERROR": "",
        "OUTPUT_DIR": str(out),
    }

    def finish(status: str | None = None):
        if status is not None:
            result["STATUS"] = status
        write_json(result_json, result)
        return result

    try:
        gwas = locus_resources["gwas"]
        result["N_GWAS_VARIANTS"] = len(gwas)

        # LOCAL dense path only: no eQTL-Catalogue LBF download/use.
        qtl = dense_qtl_for_candidate(
            tabix, candidate, gwas, catalogue_contexts, custom_registry
        )
        result["N_QTL_VARIANTS"] = len(qtl)
        if qtl.empty:
            return finish("NOT_TESTED_NO_DENSE_QTL_ROWS")

        qtl_n = None
        if "QTL_N" in qtl.columns:
            qn = numeric(qtl["QTL_N"]).dropna()
            qn = qn[qn > 0]
            if not qn.empty:
                qtl_n = int(round(float(qn.median())))
        if not qtl_n:
            qtl_n = safe_int(candidate.get("QTL_N"))
        if not qtl_n:
            return finish("NOT_TESTED_QTL_SAMPLE_SIZE_MISSING")
        result["QTL_N"] = qtl_n

        common = gwas.merge(
            qtl[["VARIANT_KEY", "BETA_REF", "SE", "P", "MAF"]].rename(
                columns={
                    "BETA_REF": "QTL_BETA",
                    "SE": "QTL_SE",
                    "P": "QTL_P",
                    "MAF": "QTL_MAF",
                }
            ),
            on="VARIANT_KEY",
            how="inner",
            validate="one_to_one",
        )
        common = common[
            common["GWAS_BETA"].notna()
            & common["GWAS_SE"].notna()
            & (common["GWAS_SE"] > 0)
            & common["QTL_BETA"].notna()
            & common["QTL_SE"].notna()
            & (common["QTL_SE"] > 0)
            & common["REFERENCE_ID"].notna()
        ].drop_duplicates("REFERENCE_ID", keep="first")
        result["N_COMMON_SNPS"] = len(common)

        if len(common) < args.min_common_snps:
            return finish("NOT_TESTED_TOO_FEW_COMMON_SNPS")
        if len(common) > args.max_common_snps:
            result["ERROR"] = (
                f"{len(common)} common SNPs > {args.max_common_snps}; no significance-based subsampling."
            )
            return finish("NOT_TESTED_TOO_MANY_COMMON_SNPS_NO_SUBSAMPLING")

        gwas_n = safe_int(gwas_meta.get("COLOC_GWAS_N"))
        if not gwas_n:
            return finish("NOT_TESTED_GWAS_SAMPLE_SIZE_MISSING")
        if (
            gwas_meta.get("GWAS_TYPE") == "cc"
            and gwas_meta.get("CASE_PROPORTION") is None
        ):
            result["ERROR"] = "Case-control GWAS identified but case/control counts are missing."
            return finish("NOT_TESTED_CASE_CONTROL_COUNTS_MISSING")

        if "ld" not in locus_resources:
            ld_dir = locus_output_dir / "locus_ld"
            locus_resources["ld"] = build_external_ld(
                root, plink2, ancestry_code, gwas, ld_dir, args.cpus, maf=args.ld_maf
            )
        ld_file, vars_file, ld_order, freq_file = locus_resources["ld"]
        result["LD_SOURCE"] = f"1000G_{ancestry_code}_SIGNED_R_MAF_GE_{args.ld_maf}"

        available = set(ld_order)
        common = common[common["REFERENCE_ID"].isin(available)].copy()

        reference_maf = load_reference_maf(freq_file)
        common["GWAS_MAF"] = common["REFERENCE_ID"].astype(str).map(reference_maf)

        # Prefer cohort QTL MAF when supplied.  If the dense QTL file does not
        # carry MAF, use the same ancestry-matched 1000G reference MAF used for
        # LD.  coloc needs MAF+N for quantitative traits when sdY is unavailable.
        qtl_raw = numeric(common["QTL_MAF"])
        qtl_raw = np.minimum(qtl_raw, 1.0 - qtl_raw)
        valid_qtl = qtl_raw.notna() & (qtl_raw > 0) & (qtl_raw < 0.5)
        fallback_mask = ~valid_qtl
        common["QTL_MAF"] = qtl_raw.where(valid_qtl, common["GWAS_MAF"])
        result["QTL_MAF_REFERENCE_FALLBACK_N"] = int(fallback_mask.sum())
        result["QTL_MAF_SOURCE"] = (
            "QTL_FILE+1000G_FALLBACK" if valid_qtl.any() and fallback_mask.any()
            else "QTL_FILE" if valid_qtl.all()
            else f"1000G_{ancestry_code}_REFERENCE"
        )

        if gwas_meta.get("GWAS_TYPE") == "quant":
            common = common[common["GWAS_MAF"].notna()].copy()
        common = common[common["QTL_MAF"].notna()].copy()

        result["N_COMMON_SNPS"] = len(common)
        if len(common) < args.min_common_snps:
            return finish("NOT_TESTED_TOO_FEW_SNPS_AFTER_LD_OR_MAF")

        common_file = out / "common_stats.tsv.gz"
        common[
            [
                "REFERENCE_ID", "CHR", "POS", "REF", "ALT", "VARIANT_KEY",
                "GWAS_BETA", "GWAS_SE", "GWAS_P", "GWAS_MAF",
                "QTL_BETA", "QTL_SE", "QTL_P", "QTL_MAF",
            ]
        ].to_csv(common_file, sep="\t", index=False, compression="gzip")

        gwas_fit_file, gwas_map_file = prepare_gwas_susie_map(
            study_dir, locus_id, ancestry_code, gwas, locus_output_dir
        )
        if gwas_fit_file is None or gwas_map_file is None:
            return finish("NOT_TESTED_STEP06_SUSIE_FIT_OR_MAP_MISSING")

        summary_file = out / "coloc_summary.tsv"
        results_file = out / "coloc_results.tsv"
        sensitivity_file = out / "prior_sensitivity.tsv"
        diagnostics_file = out / "diagnostics.tsv"
        case_prop = gwas_meta.get("CASE_PROPORTION")

        run_command(
            [
                rscript, r_worker, common_file, ld_file, vars_file,
                summary_file, results_file, sensitivity_file, diagnostics_file,
                str(gwas_n), str(qtl_n), str(gwas_meta.get("GWAS_TYPE", "quant")),
                str(case_prop) if case_prop is not None else "NA",
                str(COLOC_PRIOR_P1), str(COLOC_PRIOR_P2), str(COLOC_PRIOR_P12),
                str(COLOC_PRIOR_P12_LOW), str(COLOC_PRIOR_P12_HIGH),
                gwas_fit_file, gwas_map_file,
            ],
            log_file=out / "coloc.log",
        )

        summary = read_table(summary_file)
        no_signal_pair = (
            not summary.empty
            and "NO_SIGNAL_PAIR" in summary.columns
            and summary["NO_SIGNAL_PAIR"].astype(str).eq("YES").all()
        )
        classification = classify_coloc(summary, args.strong_pp4, args.suggestive_pp4)
        result.update(classification)
        sensitivity = read_table(sensitivity_file)
        result["PP_H4_P12_LOW"] = sensitivity_pp4_for_pair(
            sensitivity, "LOW_P12",
            classification.get("BEST_HIT1", ""),
            classification.get("BEST_HIT2", ""),
        )
        result["PP_H4_P12_DEFAULT"] = sensitivity_pp4_for_pair(
            sensitivity, "DEFAULT",
            classification.get("BEST_HIT1", ""),
            classification.get("BEST_HIT2", ""),
        )
        result["PP_H4_P12_HIGH"] = sensitivity_pp4_for_pair(
            sensitivity, "HIGH_P12",
            classification.get("BEST_HIT1", ""),
            classification.get("BEST_HIT2", ""),
        )
        result["PRIOR_ROBUST_STRONG"] = (
            "YES"
            if pd.notna(result["PP_H4_P12_LOW"])
            and result["PP_H4_P12_LOW"] >= args.strong_pp4
            else "NO"
        )

        diagnostics = read_table(diagnostics_file)
        if not diagnostics.empty and "TRAIT" in diagnostics.columns:
            for trait, field in [("GWAS", "GWAS_S_RSS"), ("QTL", "QTL_S_RSS")]:
                sub = diagnostics[diagnostics["TRAIT"].astype(str) == trait]
                if not sub.empty and "ESTIMATE_S_RSS" in sub.columns:
                    vals = numeric(sub["ESTIMATE_S_RSS"]).dropna()
                    if not vals.empty:
                        result[field] = float(vals.iloc[0])
        svals = [result.get("GWAS_S_RSS"), result.get("QTL_S_RSS")]
        result["LD_MISMATCH_WARNING"] = (
            "YES"
            if any(pd.notna(v) and float(v) > args.ld_mismatch_s for v in svals)
            else "NO"
        )

        if results_file.exists() and results_file.stat().st_size > 0:
            rr = read_table(results_file)
            rr.to_csv(
                out / "coloc_variant_results.tsv.gz",
                sep="\t", index=False, compression="gzip"
            )
            results_file.unlink(missing_ok=True)

        result["COLOC_TESTED"] = "YES"
        result["STATUS"] = "COMPLETE_NO_SIGNAL_PAIR" if no_signal_pair else "COMPLETE"
        return finish()

    except Exception as exc:
        result["STATUS"] = "FAILED"
        result["ERROR"] = f"{type(exc).__name__}: {exc}"
        (out / "FAILED.txt").write_text(
            result["ERROR"] + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        return finish()


def resource_status_table(
    allowed_types: set[str],
    catalogue_contexts: pd.DataFrame,
    custom_registry: pd.DataFrame,
    step08: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for qtl_type in SUPPORTED_QTL_TYPES:
        if qtl_type.lower() not in allowed_types:
            continue

        catalog_n = (
            int(
                (
                    catalogue_contexts["QTL_TYPE"].astype(str).str.lower()
                    == qtl_type.lower()
                ).sum()
            )
            if not catalogue_contexts.empty
            else 0
        )

        custom_n = (
            int(
                (
                    custom_registry["ENABLED_BOOL"]
                    & (
                        custom_registry["QTL_TYPE"].astype(str).str.lower()
                        == qtl_type.lower()
                    )
                ).sum()
            )
            if not custom_registry.empty
            else 0
        )

        step08_n = (
            int(
                (
                    step08["QTL_TYPE"].astype(str).str.lower()
                    == qtl_type.lower()
                ).sum()
            )
            if not step08.empty
            else 0
        )

        formal = catalog_n > 0 or custom_n > 0

        rows.append({
            "QTL_TYPE": qtl_type,
            "FORMAL_RESOURCE_AVAILABLE": formal,
            "N_EQTL_CATALOG_CONTEXTS": catalog_n,
            "N_CUSTOM_CONTEXTS": custom_n,
            "N_STEP08_GTEX_OVERLAP_ROWS": step08_n,
            "INTERPRETATION": (
                "FORMAL_RESOURCE_AVAILABLE"
                if formal
                else (
                    "STEP08_OVERLAP_ONLY_NO_FORMAL_RESOURCE"
                    if step08_n > 0
                    else "RESOURCE_NOT_CONFIGURED"
                )
            ),
        })

    return pd.DataFrame(rows)


def print_candidates(candidates: pd.DataFrame) -> None:
    banner("MOLECULAR-QTL CANDIDATES — ALL CONFIGURED TISSUES / CONTEXTS")
    if candidates.empty:
        print("No candidates found.")
        return

    cols = [
        "LOCUS_ID", "QTL_TYPE", "DATASET", "TISSUE", "CONDITION",
        "MOLECULAR_TRAIT_ID", "GENE_ID", "DISCOVERY_MIN_P", "QTL_N", "SOURCE",
    ]
    cols = [c for c in cols if c in candidates.columns]

    with pd.option_context(
        "display.max_rows", 1000,
        "display.max_columns", None,
        "display.width", 300,
        "display.max_colwidth", 55,
        "display.float_format", lambda value: f"{value:.4g}",
    ):
        print(candidates[cols].to_string(index=False))


def print_formal_summary(summary: pd.DataFrame) -> None:
    banner("FORMAL COLOCALIZATION RESULTS")
    if summary.empty:
        print("No formal candidates/results.")
        return

    cols = [
        "LOCUS_ID", "QTL_TYPE", "DATASET", "TISSUE", "CONDITION",
        "MOLECULAR_TRAIT_ID", "GENE_ID", "COLOC_TESTED", "STATUS",
        "N_COMMON_SNPS", "MAX_PP_H3", "MAX_PP_H4",
        "PP_H4_P12_LOW", "PP_H4_P12_DEFAULT", "PP_H4_P12_HIGH",
        "PRIOR_ROBUST_STRONG", "BEST_HIT1", "BEST_HIT2",
        "GWAS_S_RSS", "QTL_S_RSS", "LD_MISMATCH_WARNING",
        "SHARED_SIGNAL", "COLOC_METHOD", "COLOC_CLASS",
    ]
    cols = [c for c in cols if c in summary.columns]

    with pd.option_context(
        "display.max_rows", 2000,
        "display.max_columns", None,
        "display.width", 340,
        "display.max_colwidth", 55,
        "display.float_format", lambda value: f"{value:.4g}",
    ):
        print(summary[cols].to_string(index=False))


def build_locus_qtl_matrix(
    summary: pd.DataFrame,
    loci: list[str],
    resource_status: pd.DataFrame,
    step08: pd.DataFrame,
    audit: pd.DataFrame,
    suggestive_pp4: float,
) -> pd.DataFrame:
    rows = []
    status_map = {str(row["QTL_TYPE"]): row for _, row in resource_status.iterrows()}

    for locus in loci:
        for qtl_type, rs in status_map.items():
            local = (
                summary[
                    (summary["LOCUS_ID"].astype(str) == locus)
                    & (summary["QTL_TYPE"].astype(str).str.lower() == qtl_type.lower())
                ] if not summary.empty else pd.DataFrame()
            )
            overlap = (
                step08[
                    (step08["LOCUS_ID"].astype(str) == locus)
                    & (step08["QTL_TYPE"].astype(str).str.lower() == qtl_type.lower())
                ] if not step08.empty else pd.DataFrame()
            )
            errors = (
                audit[
                    (audit["LOCUS_ID"].astype(str) == locus)
                    & (audit["QTL_TYPE"].astype(str).str.lower() == qtl_type.lower())
                    & (audit["STATUS"].astype(str) == "ERROR")
                ] if not audit.empty and "QTL_TYPE" in audit.columns else pd.DataFrame()
            )

            tested = int((local["COLOC_TESTED"] == "YES").sum()) if not local.empty else 0
            strong = int((local["COLOCALIZED_STRONG"] == "YES").sum()) if not local.empty else 0
            robust = int((local["PRIOR_ROBUST_STRONG"] == "YES").sum()) if not local.empty and "PRIOR_ROBUST_STRONG" in local.columns else 0
            max_h4 = (
                float(numeric(local["MAX_PP_H4"]).max())
                if not local.empty and numeric(local["MAX_PP_H4"]).notna().any()
                else np.nan
            )

            best_tissue = best_trait = best_gene = best_hit1 = best_hit2 = ""
            if not local.empty and numeric(local["MAX_PP_H4"]).notna().any():
                best = local.loc[numeric(local["MAX_PP_H4"]).idxmax()]
                best_tissue = str(best.get("TISSUE", ""))
                best_trait = str(best.get("MOLECULAR_TRAIT_ID", ""))
                best_gene = str(best.get("GENE_ID", ""))
                best_hit1 = str(best.get("BEST_HIT1", ""))
                best_hit2 = str(best.get("BEST_HIT2", ""))

            if strong > 0:
                evidence = "YES_STRONG_COLOC"
            elif tested > 0 and pd.notna(max_h4) and max_h4 >= suggestive_pp4:
                evidence = "SUGGESTIVE_COLOC"
            elif tested > 0:
                evidence = "NO_STRONG_COLOC"
            elif len(local) > 0:
                evidence = "CANDIDATE_NOT_FORMALLY_TESTED"
            elif len(errors) > 0:
                evidence = "RESOURCE_ERROR"
            elif len(overlap) > 0:
                evidence = "STEP08_GTEX_V11_OVERLAP_ONLY"
            elif bool(rs["FORMAL_RESOURCE_AVAILABLE"]):
                evidence = "NO_CANDIDATE_IN_CONFIGURED_RESOURCE"
            else:
                evidence = "RESOURCE_NOT_CONFIGURED"

            rows.append({
                "LOCUS_ID": locus,
                "QTL_TYPE": qtl_type,
                "FORMAL_RESOURCE_AVAILABLE": bool(rs["FORMAL_RESOURCE_AVAILABLE"]),
                "STEP08_GTEX_V11_OVERLAP": "YES" if len(overlap) > 0 else "NO",
                "N_STEP08_OVERLAP_ROWS": len(overlap),
                "N_RESOURCE_ERRORS": len(errors),
                "N_CANDIDATES": len(local),
                "N_FORMALLY_TESTED": tested,
                "N_STRONG_COLOC": strong,
                "N_PRIOR_ROBUST_STRONG": robust,
                "MAX_PP_H4": max_h4,
                "BEST_TISSUE": best_tissue,
                "BEST_TRAIT": best_trait,
                "BEST_GENE": best_gene,
                "BEST_HIT1": best_hit1,
                "BEST_HIT2": best_hit2,
                "EVIDENCE_STATUS": evidence,
            })

    return pd.DataFrame(rows)


def build_wide_evidence_table(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return matrix

    records = []
    for locus, group in matrix.groupby("LOCUS_ID", sort=False):
        record = {"LOCUS_ID": locus}
        for _, row in group.iterrows():
            prefix = re.sub(r"[^A-Za-z0-9]+", "_", str(row["QTL_TYPE"])).upper()
            record[f"{prefix}_STATUS"] = row["EVIDENCE_STATUS"]
            record[f"{prefix}_MAX_PP_H4"] = row["MAX_PP_H4"]
            record[f"{prefix}_BEST_TISSUE"] = row["BEST_TISSUE"]
            record[f"{prefix}_BEST_TRAIT"] = row["BEST_TRAIT"]
            record[f"{prefix}_BEST_GENE"] = row["BEST_GENE"]
        records.append(record)

    return pd.DataFrame(records)


def _combine_tables(paths: list[Path]) -> pd.DataFrame:
    frames = [read_table(p) for p in paths if p.exists() and p.stat().st_size > 0]
    frames = [x for x in frames if not x.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def aggregate_study_outputs(
    root: Path,
    phenotype: str,
    ancestry_label: str,
    accession: str,
    args,
) -> None:
    paths = project_paths(root, phenotype, ancestry_label)
    study_out = paths["OUT"] / accession
    locus_root = study_out / "loci"
    study_out.mkdir(parents=True, exist_ok=True)

    with exclusive_file_lock(study_out / ".aggregate"):
        candidate_files = sorted(locus_root.glob("L*/locus_candidates.tsv"))
        result_files = sorted(locus_root.glob("L*/locus_formal_coloc.tsv"))
        audit_files = sorted(locus_root.glob("L*/locus_resource_audit.tsv"))
        resource_files = sorted(locus_root.glob("L*/resource_status.tsv"))
        overlap_files = sorted(locus_root.glob("L*/step08_overlap.tsv"))

        candidates = _combine_tables(candidate_files)
        summary = _combine_tables(result_files)
        audit = _combine_tables(audit_files)
        step08 = _combine_tables(overlap_files)

        if resource_files:
            resource_status = read_table(resource_files[0])
        else:
            resource_status = pd.DataFrame()

        candidates.to_csv(
            study_out / f"{accession}_molecular_qtl_candidates.tsv",
            sep="\t", index=False
        )
        summary.to_csv(
            study_out / f"{accession}_formal_coloc_summary.tsv",
            sep="\t", index=False
        )
        step08.to_csv(
            study_out / f"{accession}_Step08_GTEx_v11_overlap.tsv",
            sep="\t", index=False
        )
        audit.to_csv(study_out / "resource_scan_audit.tsv", sep="\t", index=False)
        resource_status.to_csv(study_out / "resource_status.tsv", sep="\t", index=False)

        manifest = read_table(paths["OUT"] / "coloc_manifest.tsv")
        study_manifest = (
            manifest[manifest["STUDY_ACCESSION"].astype(str) == accession]
            if not manifest.empty else pd.DataFrame()
        )
        loci = sorted(study_manifest["LOCUS_ID"].astype(str).unique()) if not study_manifest.empty else []

        matrix = build_locus_qtl_matrix(
            summary, loci, resource_status, step08, audit, args.suggestive_pp4
        ) if not resource_status.empty else pd.DataFrame()
        matrix.to_csv(
            study_out / f"{accession}_locus_qtl_matrix.tsv",
            sep="\t", index=False
        )
        wide = build_wide_evidence_table(matrix)
        wide.to_csv(
            study_out / f"{accession}_locus_evidence_table.tsv",
            sep="\t", index=False
        )

        done_loci = 0
        for locus in loci:
            meta = safe_json(locus_root / locus / "locus_summary.json")
            if meta.get("STATUS") == "COMPLETE":
                done_loci += 1

        status = "COMPLETE" if loci and done_loci == len(loci) else "PARTIAL"
        write_json(
            study_out / "coloc_summary.json",
            {
                "STEP11_VERSION": STEP11_VERSION,
                "METHOD": "LOCAL dense QTL + Step06 GWAS SuSiE reuse + QTL runsusie + coloc.susie",
                "STUDY_ACCESSION": accession,
                "PHENOTYPE": phenotype,
                "N_LOCI_EXPECTED": len(loci),
                "N_LOCI_COMPLETE": done_loci,
                "N_TESTS": int((summary["COLOC_TESTED"] == "YES").sum()) if not summary.empty else 0,
                "N_STRONG": int((summary["COLOCALIZED_STRONG"] == "YES").sum()) if not summary.empty else 0,
                "N_PRIOR_ROBUST_STRONG": int((summary["PRIOR_ROBUST_STRONG"] == "YES").sum()) if not summary.empty and "PRIOR_ROBUST_STRONG" in summary.columns else 0,
                "STATUS": status,
                "UPDATED_UTC": utcnow(),
            },
        )


def process_locus_task(root: Path, manifest_row: pd.Series, args) -> None:
    phenotype = str(manifest_row["PHENOTYPE"])
    ancestry_code = str(manifest_row["ANCESTRY_CODE"])
    ancestry_label = str(manifest_row["ANCESTRY_LABEL"])
    accession = str(manifest_row["STUDY_ACCESSION"])
    locus_id = str(manifest_row["LOCUS_ID"])

    paths = project_paths(root, phenotype, ancestry_label)
    study_dir = paths["S06"] / accession
    study_out = paths["OUT"] / accession
    locus_out = study_out / "loci" / locus_id
    locus_out.mkdir(parents=True, exist_ok=True)

    locus_meta = locus_out / "locus_summary.json"
    if not args.force and safe_json(locus_meta).get("STATUS") == "COMPLETE":
        banner(f"STEP11 LOCUS ALREADY COMPLETE — {accession} {locus_id}")
        aggregate_study_outputs(root, phenotype, ancestry_label, accession, args)
        return

    allowed_types = {v.strip().lower() for v in args.qtl_types.split(",") if v.strip()}
    plink2 = find_executable(root, args.plink2, "PLINK2", "plink2")
    tabix = find_executable(root, args.tabix, "TABIX", "tabix")
    rscript = find_executable(root, args.rscript, "RSCRIPT", "Rscript")
    verify_r_packages(rscript)

    # Normal analysis is strictly local-only.  --setup-resources is the only
    # mode allowed to perform network downloads.
    catalogue = load_eqtl_catalogue(root, False)
    if catalogue.empty:
        raise RuntimeError(
            "Local eQTL-Catalogue manifests are missing. Run --setup-resources once, "
            "then rerun normal analysis."
        )
    contexts = select_catalogue_contexts(catalogue, args, allowed_types)
    contexts = resolve_catalogue_local_sources(root, contexts)
    if not contexts.empty:
        check_local_tabix_connectivity(tabix, contexts)

    custom_registry = ensure_custom_registry_files(
        root, load_custom_registry(root, args.registry), False
    )
    step08_all = load_step08_overlap(paths["S08"], accession)
    step08 = step08_all[step08_all["LOCUS_ID"].astype(str) == locus_id].copy() if not step08_all.empty else step08_all
    gwas_meta = load_gwas_metadata(paths["S06"], accession)
    gwas = load_gwas_locus(study_dir, locus_id)

    banner(f"STEP11 LOCUS TASK — {accession} | {locus_id}")
    print(f"Phenotype      : {phenotype}")
    print(f"Ancestry       : {ancestry_label} ({ancestry_code})")
    print(f"Catalogue ctx  : {len(contexts)}")
    print(f"Step08 v11 rows: {len(step08)}")

    resource_status = resource_status_table(
        allowed_types, contexts, custom_registry, step08
    )
    resource_status.to_csv(locus_out / "resource_status.tsv", sep="\t", index=False)
    step08.to_csv(locus_out / "step08_overlap.tsv", sep="\t", index=False)

    candidates_frames = []
    audits = []
    if not contexts.empty:
        cand, audit = discover_catalogue_candidates_for_locus(
            tabix, contexts, locus_id, gwas, args
        )
        if not cand.empty:
            candidates_frames.append(cand)
        audits.extend(audit)
    if not custom_registry.empty:
        cand, audit = discover_custom_candidates_for_locus(
            tabix, custom_registry, locus_id, gwas, args, allowed_types
        )
        if not cand.empty:
            candidates_frames.append(cand)
        audits.extend(audit)

    candidates = (
        pd.concat(candidates_frames, ignore_index=True, sort=False)
        if candidates_frames else pd.DataFrame(columns=[
            "SOURCE","LOCUS_ID","DATASET","DATASET_ID","QTL_TYPE","TISSUE",
            "CONDITION","MOLECULAR_TRAIT_ID","GENE_ID","DISCOVERY_MIN_P",
            "N_DISCOVERY_ROWS","QTL_N","SOURCE_URL","ACCESS_PATH","ACCESS_MODE",
        ])
    )
    if not candidates.empty:
        candidates = (
            candidates.drop_duplicates(
                ["LOCUS_ID","QTL_TYPE","DATASET_ID","TISSUE","CONDITION","MOLECULAR_TRAIT_ID"],
                keep="first",
            )
            .sort_values(["DATASET_ID","QTL_TYPE","DISCOVERY_MIN_P"], kind="stable")
            .reset_index(drop=True)
        )

    candidates.to_csv(locus_out / "locus_candidates.tsv", sep="\t", index=False)
    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(locus_out / "locus_resource_audit.tsv", sep="\t", index=False)
    print_candidates(candidates)

    r_worker = write_r_worker(locus_out)
    locus_resources = {"gwas": gwas}
    results = []

    for number, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        banner(
            f"FORMAL COLOC {number}/{len(candidates)} — {locus_id} | "
            f"{candidate['QTL_TYPE']} | {candidate['TISSUE']} | {candidate['MOLECULAR_TRAIT_ID']}"
        )
        result = process_candidate(
            root, accession, study_dir, ancestry_code, gwas_meta, candidate,
            contexts, custom_registry, plink2, tabix, rscript,
            r_worker, locus_out, locus_resources, args,
        )
        results.append(result)
        print(
            f"RESULT tested={result['COLOC_TESTED']} status={result['STATUS']} "
            f"method={result.get('COLOC_METHOD','')} PP.H4={result['MAX_PP_H4']} "
            f"pair={result.get('BEST_SIGNAL_PAIR','')}"
        )

    summary = pd.DataFrame(results)
    summary.to_csv(locus_out / "locus_formal_coloc.tsv", sep="\t", index=False)
    print_formal_summary(summary)

    write_json(
        locus_meta,
        {
            "STEP11_VERSION": STEP11_VERSION,
            "STUDY_ACCESSION": accession,
            "LOCUS_ID": locus_id,
            "N_CANDIDATES": len(candidates),
            "N_TESTED": int((summary["COLOC_TESTED"] == "YES").sum()) if not summary.empty else 0,
            "N_STRONG": int((summary["COLOCALIZED_STRONG"] == "YES").sum()) if not summary.empty else 0,
            "N_RESOURCE_ERRORS": int((audit_df["STATUS"] == "ERROR").sum()) if not audit_df.empty else 0,
            "STATUS": (
                "COMPLETE"
                if not summary.empty
                and int((summary["COLOC_TESTED"] == "YES").sum()) > 0
                and int((summary["STATUS"] == "FAILED").sum()) == 0
                else "PARTIAL_FORMAL_ERROR"
                if not summary.empty
                and int((summary["COLOC_TESTED"] == "YES").sum()) > 0
                and int((summary["STATUS"] == "FAILED").sum()) > 0
                else "NOT_TESTED_FORMAL_ERROR"
                if not summary.empty
                and int((summary["STATUS"] == "FAILED").sum()) > 0
                else "NOT_TESTED_NO_FORMAL_RESULT"
            ),
            "COMPLETED_UTC": utcnow(),
        },
    )

    aggregate_study_outputs(root, phenotype, ancestry_label, accession, args)


def planner_mode(root: Path, args) -> None:
    if not args.phenotype or not args.ancestry:
        raise SystemExit("--phenotype and --ancestry are required.")

    phenotype = args.phenotype.strip()
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    paths = project_paths(root, phenotype, ancestry_label)
    studies, excluded = discover_step06_studies(paths["S06"])
    if args.study:
        studies = [x for x in studies if x == args.study]
    if not studies:
        raise RuntimeError(f"No eligible Step06 studies found under:\n  {paths['S06']}")

    paths["OUT"].mkdir(parents=True, exist_ok=True)
    paths["LOG"].mkdir(parents=True, exist_ok=True)

    rows = []
    study_ranges = {}
    for study_index, accession in enumerate(studies, start=1):
        loci = list_loci(paths["S06"] / accession)
        start_task = len(rows) + 1
        for locus_id in loci:
            rows.append({
                "COLOC_TASK_ID": len(rows) + 1,
                "STUDY_INDEX": study_index,
                "STUDY_ACCESSION": accession,
                "LOCUS_ID": locus_id,
                "PHENOTYPE": phenotype,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
            })
        if loci:
            study_ranges[study_index] = (accession, start_task, len(rows))

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No eligible Step06 loci found.")
    manifest_file = paths["OUT"] / "coloc_manifest.tsv"
    manifest.to_csv(manifest_file, sep="\t", index=False)
    pd.DataFrame(excluded, columns=["STUDY_ACCESSION","STEP06_STATUS","REASON"]).to_csv(
        paths["OUT"] / "coloc_excluded_studies.tsv", sep="\t", index=False
    )

    pipeline_python = root / "envs" / "pipeline" / "bin" / "python"
    if not pipeline_python.exists():
        pipeline_python = Path(sys.executable).resolve()
    script_path = Path(__file__).resolve()
    bash_file = root / f"Step11_Formal_Coloc_{slugify(phenotype)}_{slugify(ancestry_label)}.sh"

    throttle = f"%{args.max_parallel}" if args.max_parallel and args.max_parallel > 0 else ""
    array_spec = f"1-{len(manifest)}{throttle}"
    extra = [
        f"--phenotype {shlex.quote(phenotype)}",
        f"--ancestry {shlex.quote(ancestry_code)}",
        f"--registry {shlex.quote(args.registry)}",
        f"--qtl-types {shlex.quote(args.qtl_types)}",
        f"--candidate-p {args.candidate_p}",
        f"--min-common-snps {args.min_common_snps}",
        f"--max-common-snps {args.max_common_snps}",
        f"--strong-pp4 {args.strong_pp4}",
        f"--suggestive-pp4 {args.suggestive_pp4}",
        f"--max-candidates-per-context-locus {args.max_candidates_per_context_locus}",
        f"--cpus {args.cpus}",
        f"--max-parallel {args.max_parallel}",
        f"--ld-maf {args.ld_maf}",
        f"--ld-mismatch-s {args.ld_mismatch_s}",
    ]
    if args.catalog_all_studies:
        extra.append("--catalog-all-studies")
    if args.catalog_study_regex:
        extra.append(f"--catalog-study-regex {shlex.quote(args.catalog_study_regex)}")
    if args.no_auto_download:
        extra.append("--no-auto-download")
    if args.force:
        extra.append("--force")

    bash_text = f"""#!/bin/bash
#SBATCH --job-name=FColoc_{slugify(phenotype)[:28]}
#SBATCH --nodes=1
{chr(10).join(gwas2m_config.site_directives(args.slurm_resources))}
#SBATCH --time={args.time}
#SBATCH --mem={args.memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1
#SBATCH --array={array_spec}
#SBATCH --output={paths['LOG']}/coloc.%A_%a.out
#SBATCH --error={paths['LOG']}/coloc.%A_%a.err

set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_FORMAL_COLOC_MANIFEST={shlex.quote(str(manifest_file))}

{shlex.quote(str(pipeline_python))} \\
    {shlex.quote(str(script_path))} \\
    {' '.join(extra)}
"""
    bash_file.write_text(bash_text, encoding="utf-8")
    bash_file.chmod(0o755)

    banner("STEP11 FORMAL COLOCALIZATION PLANNER — ONE ARRAY TASK PER LOCUS")
    print(f"Phenotype       : {phenotype}")
    print(f"Ancestry        : {ancestry_label} ({ancestry_code})")
    print(f"Studies         : {len(studies)}")
    print(f"Locus tasks     : {len(manifest)}")
    print(f"Array           : {array_spec}")
    print("Catalogue I/O   : LOCAL tabix only; no remote access; no LBF downloads")
    print("Tissue policy   : ALL configured tissues/contexts")
    print()
    print(manifest.to_string(index=False))
    print()
    if 1 in study_ranges:
        accession, first, last = study_ranges[1]
        first_spec = f"{first}-{last}"
        if args.max_parallel and args.max_parallel > 0:
            first_spec += f"%{args.max_parallel}"
        print(f"FIRST GWAS ({accession}) ALL LOCI:")
        print(f"  sbatch --array={first_spec} {bash_file.name}")
        print()
    print(f"ALL STUDIES/LOCI:\n  sbatch {bash_file.name}")


def direct_index_mode(root: Path, args) -> None:
    phenotype = args.phenotype.strip()
    _, ancestry_label = canonical_ancestry(args.ancestry)
    paths = project_paths(root, phenotype, ancestry_label)
    manifest_file = paths["OUT"] / "coloc_manifest.tsv"
    if not manifest_file.exists():
        planner_mode(root, args)
    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str)
    ids = pd.to_numeric(manifest["COLOC_TASK_ID"], errors="coerce")
    selected = manifest[ids == args.index]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one locus task for index {args.index}; found {len(selected)}")
    process_locus_task(root, selected.iloc[0], args)


def worker_mode(root: Path, args) -> None:
    manifest_value = os.environ.get("GWAS_FORMAL_COLOC_MANIFEST")
    task_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if not manifest_value or not task_value:
        raise RuntimeError("GWAS_FORMAL_COLOC_MANIFEST and SLURM_ARRAY_TASK_ID are required")
    manifest = pd.read_csv(manifest_value, sep="\t", dtype=str)
    ids = pd.to_numeric(manifest["COLOC_TASK_ID"], errors="coerce")
    selected = manifest[ids == int(task_value)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one locus task for {task_value}; found {len(selected)}")
    process_locus_task(root, selected.iloc[0], args)


def main() -> None:
    args = arguments()
    root = Path.cwd().resolve()

    if args.setup_resources:
        setup_public_resources(
            root,
            include_large_gtex=args.setup_full_gtex,
            force=args.force_download,
        )
        return

    write_custom_registry_template(root)
    if not args.phenotype or not args.ancestry:
        raise SystemExit("--phenotype and --ancestry are required unless --setup-resources is used.")
    if args.index is not None and args.index < 1:
        raise SystemExit("--index must be >=1")
    if args.cpus < 1:
        raise SystemExit("--cpus must be >=1")
    if args.max_parallel < 0:
        raise SystemExit("--max-parallel must be >=0")
    if args.remote_delay < 0:
        raise SystemExit("--remote-delay must be >=0")
    if args.max_common_snps < args.min_common_snps:
        raise SystemExit("--max-common-snps must be >= --min-common-snps")

    if os.environ.get("SLURM_ARRAY_TASK_ID") and os.environ.get("GWAS_FORMAL_COLOC_MANIFEST"):
        worker_mode(root, args)
    elif args.index is not None:
        direct_index_mode(root, args)
    else:
        planner_mode(root, args)


if __name__ == "__main__":
    main()

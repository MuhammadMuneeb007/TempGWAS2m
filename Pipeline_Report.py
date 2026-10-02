 
# -*- coding: utf-8 -*-

"""
GWAS2m PIPELINE REPORTER
========================

Reads saved outputs from the GWAS2m pipeline for ONE phenotype + ancestry,
prints compact tables to the terminal, and saves the full tables under:

    reports/<phenotype>/<ancestry>/

Example:
    python Pipeline_Report.py --phenotype migraine --ancestry EUR

Optional:
    python Pipeline_Report.py \
        --phenotype migraine \
        --ancestry EUR \
        --study GCST90129450 \
        --max-rows 40 \
        --common-pip 0.01

This script DOES NOT rerun any biological analysis.
It only summarizes files that already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

REPORT_VERSION = "1.0.0"

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

STEP_NAMES = {
    "STEP01": "GWAS discovery / planning",
    "STEP02": "Downloaded summary statistics",
    "STEP03": "GWAS QC / harmonization",
    "STEP04": "Ancestry-specific LD reference",
    "STEP05": "LD clumping / loci",
    "STEP06": "SuSiE fine-mapping",
    "STEP07": "VEP functional annotation",
    "STEP08": "GTEx v11 eQTL / sQTL",
    "STEP09": "SpliceAI",
    "STEP10": "Pangolin",
    "STEP11": "GTEx v11 SuSiE colocalization",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalize_text(value: object) -> str:
    x = str(value).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def slugify(value: object) -> str:
    return normalize_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = normalize_text(value)
    if key not in ANCESTRY_ALIASES:
        raise ValueError(
            f"Unsupported ancestry {value!r}. "
            "Use EUR, AFR, EAS, SAS, or AMR."
        )
    return ANCESTRY_ALIASES[key]


def safe_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def safe_tsv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            sep="\t",
            compression="infer",
            low_memory=False,
            **kwargs,
        )
    except Exception:
        return pd.DataFrame()


def first_existing(paths) -> Path | None:
    for p in paths:
        p = Path(p)
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    return None


def first_glob(root: Path, patterns) -> Path | None:
    if not root.exists():
        return None
    for pattern in patterns:
        matches = sorted(
            p for p in root.glob(pattern)
            if p.is_file() and p.stat().st_size > 0
        )
        if matches:
            return matches[0]
    return None


def first_present(columns, aliases) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def number(value, default=np.nan):
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not value:
                return default
        return float(value)
    except Exception:
        return default


def integer(value, default=np.nan):
    x = number(value, default=np.nan)
    if pd.isna(x):
        return default
    return int(x)


def json_value(data: dict, keys, default=np.nan):
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def df_value(df: pd.DataFrame, keys, default=np.nan):
    if df.empty:
        return default

    # long format: metric/value or key/value
    key_col = first_present(df.columns, ["metric", "key", "name", "parameter"])
    val_col = first_present(df.columns, ["value", "count", "result"])

    if key_col and val_col:
        mapping = {
            str(k).strip().lower(): v
            for k, v in zip(df[key_col], df[val_col])
        }
        for key in keys:
            if key.lower() in mapping:
                return mapping[key.lower()]

    # wide format
    for key in keys:
        col = first_present(df.columns, [key])
        if col and len(df):
            return df.iloc[0][col]

    return default


def human_size(size: int) -> str:
    x = float(size)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if x < 1024 or unit == units[-1]:
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def variant_key(chrom, pos, ref, alt) -> str | None:
    try:
        chrom = str(chrom).strip()
        chrom = re.sub(r"^chr", "", chrom, flags=re.I)
        chrom = str(int(float(chrom)))
        pos = int(float(pos))
    except Exception:
        return None

    ref = str(ref).strip().upper()
    alt = str(alt).strip().upper()

    if not ref or not alt or ref in {"NAN", "NA", "."} or alt in {"NAN", "NA", "."}:
        return None

    return f"{chrom}:{pos}:{ref}:{alt}"


def status_from_json(data: dict, default="MISSING") -> str:
    if not data:
        return default
    status = str(data.get("STATUS", "")).strip().upper()
    return status or "PRESENT"


def output_exists(path: Path | None) -> bool:
    return bool(path and path.exists() and path.stat().st_size > 0)


# =============================================================================
# TERMINAL + FILE OUTPUT
# =============================================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def section(title: str):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def print_table(
    title: str,
    df: pd.DataFrame,
    max_rows: int = 25,
    columns: list[str] | None = None,
):
    section(title)

    if df is None or df.empty:
        print("(no rows)")
        return

    x = df.copy()

    if columns:
        keep = [c for c in columns if c in x.columns]
        if keep:
            x = x[keep]

    shown = x.head(max_rows)

    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", 100,
        "display.width", 240,
        "display.max_colwidth", 50,
        "display.float_format", lambda v: f"{v:.6g}",
    ):
        print(shown.to_string(index=False))

    if len(x) > max_rows:
        print()
        print(
            f"... showing first {max_rows:,} of {len(x):,} rows. "
            "The full table is saved to disk."
        )


def save_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Print and save a GWAS2m phenotype/ancestry pipeline report."
    )

    p.add_argument("--phenotype", required=True)
    p.add_argument("--ancestry", required=True)

    p.add_argument(
        "--study",
        default=None,
        help="Optional single GCST accession, e.g. GCST90129450.",
    )

    p.add_argument(
        "--max-rows",
        type=int,
        default=25,
        help="Maximum rows printed for large tables. Full tables are still saved.",
    )

    p.add_argument(
        "--common-pip",
        type=float,
        default=0.01,
        help="Minimum Step06 PIP used for cross-GWAS common-variant reporting.",
    )

    return p.parse_args()


# =============================================================================
# PATHS / STUDY DISCOVERY
# =============================================================================

def build_paths(
    root: Path,
    phenotype_slug: str,
    ancestry_slug: str,
):
    return {
        "step02": root / "02_summary_stats" / phenotype_slug / ancestry_slug,
        "step05": root / "05_ld_clumping" / phenotype_slug / ancestry_slug,
        "step06": root / "06_finemapping" / phenotype_slug / ancestry_slug,
        "step07": root / "07_annotation" / phenotype_slug / ancestry_slug,
        "step08": root / "08_qtl" / phenotype_slug / ancestry_slug,
        "step09": root / "09_splicing" / phenotype_slug / ancestry_slug,
        "step10": root / "10_pangolin" / phenotype_slug / ancestry_slug,
        "step11": root / "11_coloc" / phenotype_slug / ancestry_slug,
    }


def discover_studies(paths: dict[str, Path]) -> list[str]:
    studies = set()

    for base in paths.values():
        if not base.exists():
            continue

        for p in base.glob("GCST*"):
            if p.is_dir():
                studies.add(p.name)

        for p in base.rglob("GCST*"):
            name = p.name
            m = re.search(r"(GCST\d+)", name)
            if m:
                studies.add(m.group(1))

    # manifests / exclusions can mention studies not represented by a directory
    for base in paths.values():
        if not base.exists():
            continue

        for tsv in base.glob("*.tsv"):
            if "manifest" not in tsv.name.lower() and "excluded" not in tsv.name.lower():
                continue

            df = safe_tsv(tsv, dtype=str)

            col = first_present(
                df.columns,
                ["STUDY_ACCESSION", "ACCESSION", "GCST"],
            )

            if col:
                for x in df[col].dropna().astype(str):
                    m = re.search(r"(GCST\d+)", x)
                    if m:
                        studies.add(m.group(1))

    return sorted(studies)


# =============================================================================
# STEP 01 / 02
# =============================================================================

def summarize_step01_02(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []

    base = paths["step02"]
    qc_root = base / "qc"

    for study in studies:
        raw_candidates = []

        # Files outside qc are considered Step02/raw candidate files.
        if base.exists():
            for p in base.rglob(f"*{study}*"):
                if not p.is_file():
                    continue
                if "qc" in {x.lower() for x in p.parts}:
                    continue
                if p.suffix.lower() in {
                    ".tsv", ".gz", ".txt", ".csv", ".parquet", ".bgz"
                }:
                    raw_candidates.append(p)

        study_qc_dir = qc_root / study

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP01_DISCOVERED": True,
            "STEP02_RAW_FILES": len(raw_candidates),
            "STEP02_DOWNLOADED": len(raw_candidates) > 0 or study_qc_dir.exists(),
            "STEP02_TOTAL_RAW_SIZE": (
                human_size(sum(p.stat().st_size for p in raw_candidates))
                if raw_candidates
                else ""
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 03
# =============================================================================

def summarize_step03(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    qc_root = paths["step02"] / "qc"

    for study in studies:
        d = qc_root / study

        js = first_existing([
            d / f"{study}_QC_summary.json",
            d / "QC_summary.json",
            d / "qc_summary.json",
        ])

        if js is None:
            js = first_glob(d, ["*QC*summary*.json", "*summary*.json"])

        data = safe_json(js) if js else {}

        ts = first_existing([
            d / f"{study}_QC_summary.tsv",
            d / "QC_summary.tsv",
            d / "qc_summary.tsv",
        ])

        if ts is None:
            ts = first_glob(d, ["*QC*summary*.tsv", "*summary*.tsv"])

        table = safe_tsv(ts) if ts else pd.DataFrame()

        qc_file = first_existing([
            d / f"{study}_GRCh38_QC.tsv.gz",
            d / f"{study}_GRCh38_QC.tsv",
        ])

        sig_file = first_existing([
            d / f"{study}_GRCh38_significant.tsv.gz",
            d / f"{study}_GRCh38_significant.tsv",
        ])

        n_qc = json_value(
            data,
            [
                "N_VARIANTS_AFTER_QC",
                "N_AFTER_QC",
                "N_QC_VARIANTS",
                "N_VARIANTS_QC",
            ],
            default=df_value(
                table,
                [
                    "N_VARIANTS_AFTER_QC",
                    "N_AFTER_QC",
                    "N_QC_VARIANTS",
                ],
            ),
        )

        n_sig = json_value(
            data,
            [
                "N_GWS",
                "N_GWS_QC",
                "N_SIGNIFICANT",
                "N_GENOME_WIDE_SIGNIFICANT",
            ],
            default=df_value(
                table,
                [
                    "N_GWS",
                    "N_GWS_QC",
                    "N_SIGNIFICANT",
                ],
            ),
        )

        clump_ready = json_value(
            data,
            ["CLUMPING_READY", "CLUMP_READY"],
            default=df_value(
                table,
                ["CLUMPING_READY", "CLUMP_READY"],
                default="",
            ),
        )

        susie_ready = json_value(
            data,
            ["SUSIE_READY", "FINE_MAPPING_READY"],
            default=df_value(
                table,
                ["SUSIE_READY", "FINE_MAPPING_READY"],
                default="",
            ),
        )

        status = status_from_json(data)

        if status == "MISSING" and output_exists(qc_file):
            status = "OUTPUT_PRESENT"

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP03_STATUS": status,
            "N_VARIANTS_AFTER_QC": integer(n_qc),
            "N_GWS_QC": integer(n_sig),
            "CLUMPING_READY": clump_ready,
            "SUSIE_READY": susie_ready,
            "QC_FILE": str(qc_file or ""),
            "SIGNIFICANT_FILE": str(sig_file or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 04
# =============================================================================

def summarize_step04(
    root: Path,
    ancestry_code: str,
):
    rows = []

    base = root / "resources" / "1000G" / ancestry_code

    for chrom in range(1, 23):
        prefix = base / f"chr{chrom}_{ancestry_code}_GRCh38"

        pgen = Path(str(prefix) + ".pgen")
        pvar = Path(str(prefix) + ".pvar")
        psam = Path(str(prefix) + ".psam")

        complete = all(
            p.exists() and p.stat().st_size > 0
            for p in [pgen, pvar, psam]
        )

        rows.append({
            "ANCESTRY": ancestry_code,
            "CHR": chrom,
            "PGEN": pgen.exists(),
            "PVAR": pvar.exists(),
            "PSAM": psam.exists(),
            "COMPLETE": complete,
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 05
# =============================================================================

def summarize_step05(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    base = paths["step05"]

    for study in studies:
        d = base / study

        js = first_existing([
            d / "clumping_summary.json",
            d / f"{study}_clumping_summary.json",
        ])

        if js is None:
            js = first_glob(d, ["*clump*summary*.json", "*summary*.json"])

        data = safe_json(js) if js else {}

        ts = first_existing([
            d / "clumping_summary.tsv",
            d / f"{study}_clumping_summary.tsv",
        ])

        if ts is None:
            ts = first_glob(d, ["*clump*summary*.tsv", "*summary*.tsv"])

        table = safe_tsv(ts) if ts else pd.DataFrame()

        lead = first_existing([
            d / "lead_variants.tsv",
            d / f"{study}_lead_variants.tsv",
        ])

        if lead is None:
            lead = first_glob(d, ["*lead*variant*.tsv"])

        mapped = first_existing([
            d / "mapped_candidates.tsv.gz",
            d / f"{study}_mapped_candidates.tsv.gz",
        ])

        if mapped is None:
            mapped = first_glob(d, ["*mapped*candidate*.tsv*"])

        n_leads = json_value(
            data,
            ["N_LEAD_VARIANTS", "N_LEADS", "N_INDEPENDENT_LEADS"],
            default=df_value(
                table,
                ["N_LEAD_VARIANTS", "N_LEADS", "N_INDEPENDENT_LEADS"],
            ),
        )

        if pd.isna(number(n_leads)) and lead:
            lead_df = safe_tsv(lead)
            n_leads = len(lead_df)

        n_mapped = json_value(
            data,
            ["N_MAPPED_CANDIDATES", "N_MAPPED", "N_GWS_MAPPED"],
            default=df_value(
                table,
                ["N_MAPPED_CANDIDATES", "N_MAPPED", "N_GWS_MAPPED"],
            ),
        )

        if pd.isna(number(n_mapped)) and mapped:
            mapped_df = safe_tsv(mapped)
            n_mapped = len(mapped_df)

        status = status_from_json(data)

        if status == "MISSING" and d.exists():
            status = "OUTPUT_PRESENT"

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP05_STATUS": status,
            "N_MAPPED_CANDIDATES": integer(n_mapped),
            "N_LEAD_VARIANTS": integer(n_leads),
            "LEAD_FILE": str(lead or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 06
# =============================================================================

def summarize_step06(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    base = paths["step06"]

    for study in studies:
        d = base / study
        js = d / "finemapping_summary.json"
        data = safe_json(js)

        variant_file = first_existing([
            d / f"{study}_finemapped_variants.tsv.gz",
            d / f"{study}_finemapped_variants.tsv",
        ])

        cs_file = first_existing([
            d / f"{study}_95pct_credible_sets.tsv",
            d / f"{study}_95pct_credible_sets.tsv.gz",
        ])

        n_rows = json_value(
            data,
            [
                "N_FINEMAPPED_VARIANTS",
                "N_FINEMAPPED_VARIANT_ROWS",
                "N_VARIANTS",
            ],
        )

        max_pip = json_value(
            data,
            ["MAX_PIP", "MAXIMUM_PIP"],
        )

        if variant_file and (
            pd.isna(number(n_rows))
            or pd.isna(number(max_pip))
        ):
            df = safe_tsv(
                variant_file,
                usecols=lambda c: c in {"PIP", "LOCUS_ID"},
            )
            if not df.empty:
                if pd.isna(number(n_rows)):
                    n_rows = len(df)
                if "PIP" in df.columns and pd.isna(number(max_pip)):
                    max_pip = pd.to_numeric(
                        df["PIP"],
                        errors="coerce",
                    ).max()

        n_cs = json_value(
            data,
            [
                "N_CREDIBLE_SET_ROWS",
                "N_95PCT_CREDIBLE_SET_ROWS",
                "N_CS_ROWS",
            ],
        )

        if cs_file and pd.isna(number(n_cs)):
            cs = safe_tsv(cs_file)
            n_cs = len(cs)

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP06_STATUS": status_from_json(data),
            "N_LOCI_TOTAL": integer(
                json_value(
                    data,
                    ["N_LOCI_TOTAL", "N_LOCI", "N_LOCI_ATTEMPTED"],
                )
            ),
            "N_LOCI_SUCCESS": integer(
                json_value(
                    data,
                    ["N_LOCI_SUCCESS", "N_LOCI_COMPLETED"],
                )
            ),
            "N_LOCI_FAILED": integer(
                json_value(
                    data,
                    ["N_LOCI_FAILED", "N_FAILED_LOCI"],
                )
            ),
            "N_FINEMAPPED_VARIANTS": integer(n_rows),
            "N_95PCT_CS_ROWS": integer(n_cs),
            "MAX_PIP": number(max_pip),
            "FINEMAPPED_FILE": str(variant_file or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 07
# =============================================================================

def summarize_step07(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    base = paths["step07"]

    for study in studies:
        d = base / study

        summary_json = first_existing([
            d / "annotation_summary.json",
            d / "vep_summary.json",
        ])

        data = safe_json(summary_json) if summary_json else {}

        variant_file = first_existing([
            d / f"{study}_VEP_variant_summary.tsv",
            d / f"{study}_VEP_variant_summary.tsv.gz",
        ])

        if variant_file is None:
            variant_file = first_glob(d, ["*VEP*variant*summary*.tsv*"])

        df = safe_tsv(variant_file) if variant_file else pd.DataFrame()

        def category_count(name):
            if "FUNCTIONAL_CATEGORY" not in df.columns:
                return np.nan
            return int(
                (
                    df["FUNCTIONAL_CATEGORY"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    == name
                ).sum()
            )

        n_selected = json_value(
            data,
            ["N_SELECTED_VARIANTS", "N_INPUT_VARIANTS"],
            default=(len(df) if not df.empty else np.nan),
        )

        n_annotated = json_value(
            data,
            ["N_ANNOTATED_VARIANTS", "N_VEP_ANNOTATED"],
            default=(
                int(df["VEP_ID"].notna().sum())
                if "VEP_ID" in df.columns
                else len(df) if not df.empty else np.nan
            ),
        )

        status = status_from_json(data)

        if status == "MISSING" and variant_file:
            status = "COMPLETE"

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP07_STATUS": status,
            "N_SELECTED_VARIANTS": integer(n_selected),
            "N_ANNOTATED_VARIANTS": integer(n_annotated),
            "N_SPLICING": integer(
                json_value(data, ["N_SPLICING"], default=category_count("SPLICING"))
            ),
            "N_CODING": integer(
                json_value(data, ["N_CODING"], default=category_count("CODING"))
            ),
            "N_REGULATORY": integer(
                json_value(data, ["N_REGULATORY"], default=category_count("REGULATORY"))
            ),
            "N_NONCODING_RNA": integer(
                json_value(data, ["N_NONCODING_RNA"], default=category_count("NONCODING_RNA"))
            ),
            "VEP_FILE": str(variant_file or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 08
# =============================================================================

def summarize_step08(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    base = paths["step08"]

    for study in studies:
        d = base / study
        js = d / "qtl_summary.json"
        data = safe_json(js)

        variant_file = first_existing([
            d / f"{study}_GTEx_QTL_variant_summary.tsv",
            d / f"{study}_GTEx_QTL_variant_summary.tsv.gz",
        ])

        df = safe_tsv(variant_file) if variant_file else pd.DataFrame()

        def bool_count(col):
            if col not in df.columns:
                return np.nan
            x = (
                df[col]
                .fillna(False)
                .astype(str)
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
            return int(x.sum())

        n_input = json_value(
            data,
            ["N_INPUT_VARIANTS", "N_VARIANTS"],
            default=(len(df) if not df.empty else np.nan),
        )

        n_eqtl = json_value(
            data,
            ["N_EQTL_VARIANTS", "N_GTEX_EQTL_VARIANTS"],
            default=bool_count("HAS_GTEX_EQTL"),
        )

        n_sqtl = json_value(
            data,
            ["N_SQTL_VARIANTS", "N_GTEX_SQTL_VARIANTS"],
            default=bool_count("HAS_GTEX_SQTL"),
        )

        n_both = json_value(
            data,
            ["N_EQTL_AND_SQTL", "N_GTEX_EQTL_AND_SQTL"],
            default=bool_count("HAS_GTEX_EQTL_AND_SQTL"),
        )

        status = status_from_json(data)

        if status == "MISSING" and variant_file:
            status = "COMPLETE"

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP08_STATUS": status,
            "N_INPUT_VARIANTS": integer(n_input),
            "N_EQTL_VARIANTS": integer(n_eqtl),
            "N_SQTL_VARIANTS": integer(n_sqtl),
            "N_EQTL_AND_SQTL": integer(n_both),
            "N_EQTL_ASSOCIATIONS": integer(
                json_value(
                    data,
                    ["N_EQTL_ASSOCIATIONS", "N_GTEX_EQTL_ASSOCIATIONS"],
                )
            ),
            "N_SQTL_ASSOCIATIONS": integer(
                json_value(
                    data,
                    ["N_SQTL_ASSOCIATIONS", "N_GTEX_SQTL_ASSOCIATIONS"],
                )
            ),
            "N_EQTL_TISSUES": integer(
                json_value(
                    data,
                    ["N_EQTL_TISSUES", "N_GTEX_EQTL_TISSUES"],
                )
            ),
            "N_SQTL_TISSUES": integer(
                json_value(
                    data,
                    ["N_SQTL_TISSUES", "N_GTEX_SQTL_TISSUES"],
                )
            ),
            "QTL_FILE": str(variant_file or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 09
# =============================================================================

def summarize_step09(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    base = paths["step09"]

    for study in studies:
        d = base / study
        data = safe_json(d / "spliceai_summary.json")

        variant_file = first_existing([
            d / f"{study}_SpliceAI_variant_summary.tsv",
            d / f"{study}_SpliceAI_variant_summary.tsv.gz",
        ])

        df = safe_tsv(variant_file) if variant_file else pd.DataFrame()

        def threshold_count(thresh):
            if "SPLICEAI_MAX_DS" not in df.columns:
                return np.nan
            x = pd.to_numeric(df["SPLICEAI_MAX_DS"], errors="coerce")
            return int((x >= thresh).sum())

        n_scored = json_value(
            data,
            ["N_SPLICEAI_SCORED", "N_SCORED_VARIANTS"],
            default=(
                int(pd.to_numeric(
                    df["SPLICEAI_MAX_DS"],
                    errors="coerce",
                ).notna().sum())
                if "SPLICEAI_MAX_DS" in df.columns
                else np.nan
            ),
        )

        status = status_from_json(data)

        if status == "MISSING" and variant_file:
            status = "COMPLETE"

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP09_STATUS": status,
            "N_INPUT_VARIANTS": integer(
                json_value(
                    data,
                    ["N_INPUT_VARIANTS", "N_TOTAL_VARIANTS"],
                    default=(len(df) if not df.empty else np.nan),
                )
            ),
            "N_SPLICEAI_SCORED": integer(n_scored),
            "N_SPLICEAI_GE_0_20": integer(
                json_value(
                    data,
                    ["N_SPLICEAI_GE_0_20"],
                    default=threshold_count(0.20),
                )
            ),
            "N_SPLICEAI_GE_0_50": integer(
                json_value(
                    data,
                    ["N_SPLICEAI_GE_0_50"],
                    default=threshold_count(0.50),
                )
            ),
            "N_SPLICEAI_GE_0_80": integer(
                json_value(
                    data,
                    ["N_SPLICEAI_GE_0_80"],
                    default=threshold_count(0.80),
                )
            ),
            "SPLICEAI_FILE": str(variant_file or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 10
# =============================================================================

def summarize_step10(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    base = paths["step10"]

    for study in studies:
        d = base / study
        data = safe_json(d / "pangolin_summary.json")

        integrated = first_existing([
            d / f"{study}_SpliceAI_Pangolin_integrated.tsv",
        ])

        status = status_from_json(data)

        if status == "MISSING" and integrated:
            status = "COMPLETE"

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP10_STATUS": status,
            "N_INPUT_VARIANTS": integer(
                json_value(data, ["N_INPUT_VARIANTS"])
            ),
            "N_PANGOLIN_SCORED": integer(
                json_value(data, ["N_PANGOLIN_SCORED"])
            ),
            "N_PANGOLIN_UNSCORED": integer(
                json_value(data, ["N_PANGOLIN_UNSCORED"])
            ),
            "PANGOLIN_COVERAGE_PERCENT": number(
                json_value(data, ["PANGOLIN_COVERAGE_PERCENT"])
            ),
            "N_PANGOLIN_TOP_10PCT": integer(
                json_value(data, ["N_PANGOLIN_TOP_10PCT"])
            ),
            "N_PANGOLIN_TOP_5PCT": integer(
                json_value(data, ["N_PANGOLIN_TOP_5PCT"])
            ),
            "N_PANGOLIN_TOP_1PCT": integer(
                json_value(data, ["N_PANGOLIN_TOP_1PCT"])
            ),
            "N_MULTI_SOURCE_GE_2": integer(
                json_value(data, ["N_MULTI_SOURCE_GE_2"])
            ),
            "PANGOLIN_FILE": str(integrated or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP 11
# =============================================================================

def summarize_step11(
    studies: list[str],
    paths: dict[str, Path],
):
    rows = []
    base = paths["step11"]

    for study in studies:
        d = base / study
        data = safe_json(d / "coloc_summary.json")

        pair_file = first_existing([
            d / f"{study}_GTEx_SuSiE_gene_tissue_colocalization.tsv",
        ])

        status = status_from_json(data)

        if status == "MISSING" and pair_file:
            status = "COMPLETE"

        rows.append({
            "STUDY_ACCESSION": study,
            "STEP11_STATUS": status,
            "N_GWAS_FINE_MAPPED_VARIANTS": integer(
                json_value(data, ["N_GWAS_FINE_MAPPED_VARIANTS"])
            ),
            "N_GWAS_LOCI": integer(
                json_value(data, ["N_GWAS_LOCI"])
            ),
            "N_VARIANT_OVERLAP_ROWS": integer(
                json_value(data, ["N_VARIANT_OVERLAP_ROWS"])
            ),
            "N_UNIQUE_SHARED_VARIANTS": integer(
                json_value(data, ["N_UNIQUE_SHARED_VARIANTS"])
            ),
            "N_COLOC_LOCI": integer(
                json_value(data, ["N_COLOC_LOCI"])
            ),
            "N_GENE_TISSUE_PAIRS": integer(
                json_value(data, ["N_GENE_TISSUE_PAIRS"])
            ),
            "N_GENES": integer(
                json_value(data, ["N_GENES"])
            ),
            "N_TISSUES": integer(
                json_value(data, ["N_TISSUES"])
            ),
            "N_LCLPP_GE_0_01": integer(
                json_value(data, ["N_PAIR_LCLPP_GE_0_01"])
            ),
            "N_LCLPP_GE_0_001": integer(
                json_value(data, ["N_PAIR_LCLPP_GE_0_001"])
            ),
            "MAX_LCLPP": number(
                json_value(data, ["MAX_LCLPP"])
            ),
            "COLOC_FILE": str(pair_file or ""),
        })

    return pd.DataFrame(rows)


# =============================================================================
# EXCLUSIONS
# =============================================================================

def collect_exclusions(paths: dict[str, Path]) -> pd.DataFrame:
    pieces = []

    for step_key, base in paths.items():
        if not base.exists():
            continue

        for path in sorted(base.glob("*excluded*.tsv")):
            df = safe_tsv(path, dtype=str)

            if df.empty:
                continue

            df = df.copy()
            df.insert(0, "STEP", step_key.upper())
            df.insert(1, "SOURCE_FILE", str(path))
            pieces.append(df)

    if not pieces:
        return pd.DataFrame(
            columns=[
                "STEP",
                "STUDY_ACCESSION",
                "REASON",
                "SOURCE_FILE",
            ]
        )

    out = pd.concat(
        pieces,
        ignore_index=True,
        sort=False,
    )

    # normalize study column
    study_col = first_present(
        out.columns,
        ["STUDY_ACCESSION", "ACCESSION", "GCST"],
    )

    if study_col and study_col != "STUDY_ACCESSION":
        out["STUDY_ACCESSION"] = out[study_col]

    reason_col = first_present(
        out.columns,
        ["REASON", "EXCLUSION_REASON", "STATUS"],
    )

    if reason_col and reason_col != "REASON":
        out["REASON"] = out[reason_col]

    preferred = [
        c
        for c in [
            "STEP",
            "STUDY_ACCESSION",
            "REASON",
            "SOURCE_FILE",
        ]
        if c in out.columns
    ]

    others = [
        c for c in out.columns
        if c not in preferred
    ]

    return out[preferred + others]


# =============================================================================
# WARNINGS / ERRORS
# =============================================================================

WARNING_PATTERNS = [
    re.compile(r"\bwarning\b", re.I),
    re.compile(r"\berror\b", re.I),
    re.compile(r"\bfailed\b", re.I),
    re.compile(r"\bskipping variant\b", re.I),
]


def scan_warning_file(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None

    # Avoid accidentally reading enormous binary or data files.
    if path.stat().st_size > 200 * 1024 * 1024:
        return None

    count_warning = 0
    count_error = 0
    count_skip = 0
    examples = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                low = line.lower()

                if "warning" in low:
                    count_warning += 1

                if "error" in low or "failed" in low:
                    count_error += 1

                if "skipping variant" in low:
                    count_skip += 1

                if any(p.search(line) for p in WARNING_PATTERNS):
                    if len(examples) < 3:
                        examples.append(line.strip()[:220])

    except Exception:
        return None

    if count_warning == 0 and count_error == 0 and count_skip == 0:
        return None

    m = re.search(r"(GCST\d+)", str(path))

    return {
        "STUDY_ACCESSION": m.group(1) if m else "",
        "FILE": str(path),
        "N_WARNING_LINES": count_warning,
        "N_ERROR_OR_FAILED_LINES": count_error,
        "N_SKIPPING_VARIANT_LINES": count_skip,
        "EXAMPLES": " || ".join(examples),
    }


def collect_warnings(
    root: Path,
    paths: dict[str, Path],
) -> pd.DataFrame:

    candidates = set()

    for base in paths.values():
        if not base.exists():
            continue

        for pattern in [
            "**/*.log",
            "**/*FAILED*.txt",
            "**/*.err",
        ]:
            candidates.update(
                p.resolve()
                for p in base.glob(pattern)
                if p.is_file()
            )

    logs_root = root / "logs"

    if logs_root.exists():
        for p in logs_root.rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in {".log", ".err", ".out", ".txt"}
            ):
                if any(
                    token in str(p).lower()
                    for token in [
                        "step07",
                        "step08",
                        "step09",
                        "step10",
                        "step11",
                        "vep",
                        "splice",
                        "pangolin",
                        "coloc",
                    ]
                ):
                    candidates.add(p.resolve())

    rows = []

    for path in sorted(candidates):
        row = scan_warning_file(path)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "STUDY_ACCESSION",
                "FILE",
                "N_WARNING_LINES",
                "N_ERROR_OR_FAILED_LINES",
                "N_SKIPPING_VARIANT_LINES",
                "EXAMPLES",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "N_ERROR_OR_FAILED_LINES",
                "N_WARNING_LINES",
                "N_SKIPPING_VARIANT_LINES",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# CROSS-GWAS COMMON FINE-MAPPED VARIANTS
# =============================================================================

def common_finemapped_variants(
    studies: list[str],
    paths: dict[str, Path],
    min_pip: float,
) -> pd.DataFrame:

    pieces = []

    for study in studies:
        path = first_existing([
            paths["step06"]
            / study
            / f"{study}_finemapped_variants.tsv.gz",

            paths["step06"]
            / study
            / f"{study}_finemapped_variants.tsv",
        ])

        if not path:
            continue

        df = safe_tsv(
            path,
            usecols=lambda c: c in {
                "CHR",
                "REFERENCE_POS",
                "REFERENCE_REF",
                "REFERENCE_ALT",
                "PIP",
                "LOCUS_ID",
            },
        )

        required = {
            "CHR",
            "REFERENCE_POS",
            "REFERENCE_REF",
            "REFERENCE_ALT",
            "PIP",
        }

        if not required.issubset(df.columns):
            continue

        df = df.copy()

        df["PIP"] = pd.to_numeric(
            df["PIP"],
            errors="coerce",
        )

        df = df[
            df["PIP"] >= min_pip
        ].copy()

        if df.empty:
            continue

        df["VARIANT_KEY"] = [
            variant_key(c, p, r, a)
            for c, p, r, a in zip(
                df["CHR"],
                df["REFERENCE_POS"],
                df["REFERENCE_REF"],
                df["REFERENCE_ALT"],
            )
        ]

        df = df.dropna(
            subset=["VARIANT_KEY"]
        )

        df["STUDY_ACCESSION"] = study

        pieces.append(
            df[
                [
                    "STUDY_ACCESSION",
                    "VARIANT_KEY",
                    "PIP",
                ]
            ]
        )

    if not pieces:
        return pd.DataFrame(
            columns=[
                "VARIANT_KEY",
                "N_STUDIES",
                "MAX_PIP",
                "MEAN_PIP",
                "STUDIES",
            ]
        )

    x = pd.concat(
        pieces,
        ignore_index=True,
    )

    out = (
        x.groupby(
            "VARIANT_KEY",
            as_index=False,
        )
        .agg(
            N_STUDIES=("STUDY_ACCESSION", "nunique"),
            MAX_PIP=("PIP", "max"),
            MEAN_PIP=("PIP", "mean"),
            STUDIES=(
                "STUDY_ACCESSION",
                lambda s: ";".join(sorted(set(map(str, s)))),
            ),
        )
    )

    return (
        out[
            out["N_STUDIES"] >= 2
        ]
        .sort_values(
            [
                "N_STUDIES",
                "MAX_PIP",
                "MEAN_PIP",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# COMMON VEP GENES
# =============================================================================

def split_genes(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "."}:
        return []

    genes = re.split(r"[;,|]", text)

    return [
        g.strip()
        for g in genes
        if g.strip()
        and g.strip().lower() not in {"nan", "none", "."}
    ]


def common_vep_genes(
    studies: list[str],
    paths: dict[str, Path],
) -> pd.DataFrame:

    rows = []

    for study in studies:
        d = paths["step07"] / study

        path = first_existing([
            d / f"{study}_VEP_variant_summary.tsv",
            d / f"{study}_VEP_variant_summary.tsv.gz",
        ])

        if path is None:
            path = first_glob(
                d,
                ["*VEP*variant*summary*.tsv*"],
            )

        if not path:
            continue

        df = safe_tsv(path)

        gene_col = first_present(
            df.columns,
            [
                "SYMBOL",
                "GENE_SYMBOL",
                "GENE",
                "Gene",
                "VEP_SYMBOL",
                "NEAREST_GENE",
            ],
        )

        if gene_col is None:
            continue

        pip_col = first_present(
            df.columns,
            ["PIP"],
        )

        for _, row in df.iterrows():
            genes = split_genes(
                row[gene_col]
            )

            pip = (
                number(row[pip_col])
                if pip_col
                else np.nan
            )

            for gene in genes:
                rows.append({
                    "STUDY_ACCESSION": study,
                    "GENE": gene,
                    "PIP": pip,
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "GENE",
                "N_STUDIES",
                "MAX_PIP",
                "STUDIES",
            ]
        )

    x = pd.DataFrame(rows)

    out = (
        x.groupby(
            "GENE",
            as_index=False,
        )
        .agg(
            N_STUDIES=("STUDY_ACCESSION", "nunique"),
            MAX_PIP=("PIP", "max"),
            STUDIES=(
                "STUDY_ACCESSION",
                lambda s: ";".join(sorted(set(map(str, s)))),
            ),
        )
    )

    return (
        out[
            out["N_STUDIES"] >= 2
        ]
        .sort_values(
            [
                "N_STUDIES",
                "MAX_PIP",
                "GENE",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# =============================================================================
# COMMON COLOCALIZED GENES + TISSUES
# =============================================================================

def common_coloc(
    studies: list[str],
    paths: dict[str, Path],
):
    pieces = []

    for study in studies:
        path = (
            paths["step11"]
            / study
            / f"{study}_GTEx_SuSiE_gene_tissue_colocalization.tsv"
        )

        df = safe_tsv(path)

        required = {
            "QTL_TYPE",
            "TISSUE",
            "QTL_GENE",
            "LCLPP_MULTICAUSAL",
        }

        if df.empty or not required.issubset(df.columns):
            continue

        df = df.copy()

        df["LCLPP_MULTICAUSAL"] = pd.to_numeric(
            df["LCLPP_MULTICAUSAL"],
            errors="coerce",
        )

        df["STUDY_ACCESSION"] = study

        pieces.append(
            df[
                [
                    "STUDY_ACCESSION",
                    "QTL_TYPE",
                    "TISSUE",
                    "QTL_GENE",
                    "LCLPP_MULTICAUSAL",
                ]
            ]
        )

    if not pieces:
        empty_gene = pd.DataFrame(
            columns=[
                "QTL_TYPE",
                "QTL_GENE",
                "N_STUDIES",
                "MAX_BEST_LCLPP",
                "MEDIAN_BEST_LCLPP",
                "N_TISSUES",
                "STUDIES",
            ]
        )
        empty_tissue = pd.DataFrame(
            columns=[
                "QTL_TYPE",
                "TISSUE",
                "N_STUDIES",
                "MAX_LCLPP",
                "N_GENES",
                "STUDIES",
            ]
        )
        return empty_gene, empty_tissue

    x = pd.concat(
        pieces,
        ignore_index=True,
    )

    # first collapse each gene within each study
    per_study_gene = (
        x.groupby(
            [
                "STUDY_ACCESSION",
                "QTL_TYPE",
                "QTL_GENE",
            ],
            as_index=False,
        )
        .agg(
            BEST_LCLPP=("LCLPP_MULTICAUSAL", "max"),
            N_TISSUES=("TISSUE", "nunique"),
        )
    )

    genes = (
        per_study_gene.groupby(
            [
                "QTL_TYPE",
                "QTL_GENE",
            ],
            as_index=False,
        )
        .agg(
            N_STUDIES=("STUDY_ACCESSION", "nunique"),
            MAX_BEST_LCLPP=("BEST_LCLPP", "max"),
            MEDIAN_BEST_LCLPP=("BEST_LCLPP", "median"),
            N_TISSUES=("N_TISSUES", "sum"),
            STUDIES=(
                "STUDY_ACCESSION",
                lambda s: ";".join(sorted(set(map(str, s)))),
            ),
        )
    )

    genes = (
        genes[
            genes["N_STUDIES"] >= 2
        ]
        .sort_values(
            [
                "N_STUDIES",
                "MAX_BEST_LCLPP",
                "MEDIAN_BEST_LCLPP",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    tissues = (
        x.groupby(
            [
                "QTL_TYPE",
                "TISSUE",
            ],
            as_index=False,
        )
        .agg(
            N_STUDIES=("STUDY_ACCESSION", "nunique"),
            MAX_LCLPP=("LCLPP_MULTICAUSAL", "max"),
            N_GENES=("QTL_GENE", "nunique"),
            STUDIES=(
                "STUDY_ACCESSION",
                lambda s: ";".join(sorted(set(map(str, s)))),
            ),
        )
    )

    tissues = (
        tissues[
            tissues["N_STUDIES"] >= 2
        ]
        .sort_values(
            [
                "N_STUDIES",
                "MAX_LCLPP",
                "N_GENES",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return genes, tissues


# =============================================================================
# FILE INVENTORY
# =============================================================================

def build_inventory(
    root: Path,
    phenotype_slug: str,
    ancestry_slug: str,
) -> pd.DataFrame:

    rows = []

    step_dirs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and re.match(r"^\d\d_", p.name)
    ]

    for step_dir in sorted(step_dirs):
        # Prefer phenotype/ancestry subtree when it exists.
        scoped = (
            step_dir
            / phenotype_slug
            / ancestry_slug
        )

        search_root = (
            scoped
            if scoped.exists()
            else step_dir
        )

        for path in search_root.rglob("*"):
            if not path.is_file():
                continue

            m = re.search(
                r"(GCST\d+)",
                str(path),
            )

            rows.append({
                "STEP_DIR": step_dir.name,
                "STUDY_ACCESSION": m.group(1) if m else "",
                "FILE": str(path),
                "SIZE_BYTES": path.stat().st_size,
                "SIZE": human_size(path.stat().st_size),
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "STEP_DIR",
                "STUDY_ACCESSION",
                "FILE",
                "SIZE_BYTES",
                "SIZE",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "STEP_DIR",
                "STUDY_ACCESSION",
                "FILE",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# MASTER TABLE
# =============================================================================

def master_table(
    studies,
    step02,
    step03,
    step05,
    step06,
    step07,
    step08,
    step09,
    step10,
    step11,
):
    master = pd.DataFrame({
        "STUDY_ACCESSION": studies
    })

    tables = [
        step02,
        step03,
        step05,
        step06,
        step07,
        step08,
        step09,
        step10,
        step11,
    ]

    for table in tables:
        if table is not None and not table.empty:
            master = master.merge(
                table,
                on="STUDY_ACCESSION",
                how="left",
            )

    return master


def completion_matrix(
    studies,
    step02,
    step03,
    step05,
    step06,
    step07,
    step08,
    step09,
    step10,
    step11,
):
    base = pd.DataFrame({
        "STUDY_ACCESSION": studies
    })

    mappings = [
        ("STEP02", step02, "STEP02_DOWNLOADED"),
        ("STEP03", step03, "STEP03_STATUS"),
        ("STEP05", step05, "STEP05_STATUS"),
        ("STEP06", step06, "STEP06_STATUS"),
        ("STEP07", step07, "STEP07_STATUS"),
        ("STEP08", step08, "STEP08_STATUS"),
        ("STEP09", step09, "STEP09_STATUS"),
        ("STEP10", step10, "STEP10_STATUS"),
        ("STEP11", step11, "STEP11_STATUS"),
    ]

    for name, table, col in mappings:
        if table.empty or col not in table.columns:
            base[name] = "MISSING"
            continue

        x = table[
            [
                "STUDY_ACCESSION",
                col,
            ]
        ].copy()

        x = x.rename(
            columns={
                col: name
            }
        )

        base = base.merge(
            x,
            on="STUDY_ACCESSION",
            how="left",
        )

        base[name] = (
            base[name]
            .fillna("MISSING")
            .astype(str)
        )

    return base


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    root = Path.cwd().resolve()

    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)

    ancestry_code, ancestry_label = canonical_ancestry(
        args.ancestry
    )

    ancestry_slug = slugify(
        ancestry_label
    )

    paths = build_paths(
        root,
        phenotype_slug,
        ancestry_slug,
    )

    studies = discover_studies(
        paths
    )

    if args.study:
        wanted = args.study.strip()
        studies = [
            s
            for s in studies
            if s == wanted
        ]

        if not studies:
            raise SystemExit(
                f"Study {wanted!r} was not found for "
                f"{phenotype} / {ancestry_label}."
            )

    if not studies:
        raise SystemExit(
            "No GCST studies were discovered.\n"
            "Run this script from the GWAS2m project root and check:\n"
            f"  phenotype = {phenotype}\n"
            f"  ancestry  = {ancestry_label}"
        )

    report_dir = (
        root
        / "reports"
        / phenotype_slug
        / ancestry_slug
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    terminal_report = (
        report_dir
        / "terminal_report.txt"
    )

    original_stdout = sys.stdout

    with open(
        terminal_report,
        "w",
        encoding="utf-8",
    ) as report_handle:

        sys.stdout = Tee(
            original_stdout,
            report_handle,
        )

        try:
            section(
                "GWAS2m PIPELINE REPORT"
            )

            print(
                f"Report version : {REPORT_VERSION}"
            )
            print(
                f"Project root   : {root}"
            )
            print(
                f"Phenotype      : {phenotype}"
            )
            print(
                f"Ancestry       : {ancestry_label} ({ancestry_code})"
            )
            print(
                f"Studies found  : {len(studies)}"
            )
            print(
                f"Studies        : {'; '.join(studies)}"
            )
            print(
                f"Report folder  : {report_dir}"
            )

            # ---------------------------------------------------------
            # Build all step tables
            # ---------------------------------------------------------

            step02 = summarize_step01_02(
                studies,
                paths,
            )

            step03 = summarize_step03(
                studies,
                paths,
            )

            step04 = summarize_step04(
                root,
                ancestry_code,
            )

            step05 = summarize_step05(
                studies,
                paths,
            )

            step06 = summarize_step06(
                studies,
                paths,
            )

            step07 = summarize_step07(
                studies,
                paths,
            )

            step08 = summarize_step08(
                studies,
                paths,
            )

            step09 = summarize_step09(
                studies,
                paths,
            )

            step10 = summarize_step10(
                studies,
                paths,
            )

            step11 = summarize_step11(
                studies,
                paths,
            )

            exclusions = collect_exclusions(
                paths
            )

            warnings = collect_warnings(
                root,
                paths,
            )

            common_variants = common_finemapped_variants(
                studies,
                paths,
                args.common_pip,
            )

            common_vep = common_vep_genes(
                studies,
                paths,
            )

            (
                common_coloc_genes,
                common_coloc_tissues,
            ) = common_coloc(
                studies,
                paths,
            )

            inventory = build_inventory(
                root,
                phenotype_slug,
                ancestry_slug,
            )

            master = master_table(
                studies,
                step02,
                step03,
                step05,
                step06,
                step07,
                step08,
                step09,
                step10,
                step11,
            )

            completion = completion_matrix(
                studies,
                step02,
                step03,
                step05,
                step06,
                step07,
                step08,
                step09,
                step10,
                step11,
            )

            # ---------------------------------------------------------
            # Save every full table
            # ---------------------------------------------------------

            tables = {
                "pipeline_master_summary.tsv": master,
                "step_completion_matrix.tsv": completion,
                "step01_02_discovery_download_summary.tsv": step02,
                "step03_qc_summary.tsv": step03,
                "step04_ld_reference_summary.tsv": step04,
                "step05_clumping_summary.tsv": step05,
                "step06_finemapping_summary.tsv": step06,
                "step07_annotation_summary.tsv": step07,
                "step08_gtex_qtl_summary.tsv": step08,
                "step09_spliceai_summary.tsv": step09,
                "step10_pangolin_summary.tsv": step10,
                "step11_coloc_summary.tsv": step11,
                "step_exclusion_summary.tsv": exclusions,
                "tool_warning_error_summary.tsv": warnings,
                "common_finemapped_variants.tsv": common_variants,
                "common_vep_genes.tsv": common_vep,
                "common_coloc_genes.tsv": common_coloc_genes,
                "common_coloc_tissues.tsv": common_coloc_tissues,
                "output_file_inventory.tsv": inventory,
            }

            for filename, table in tables.items():
                save_table(
                    table,
                    report_dir / filename,
                )

            # ---------------------------------------------------------
            # Print main report
            # ---------------------------------------------------------

            print_table(
                "MASTER STUDY SUMMARY",
                master,
                max_rows=args.max_rows,
                columns=[
                    "STUDY_ACCESSION",
                    "STEP03_STATUS",
                    "N_VARIANTS_AFTER_QC",
                    "N_GWS_QC",
                    "N_LEAD_VARIANTS",
                    "STEP06_STATUS",
                    "N_LOCI_SUCCESS",
                    "N_FINEMAPPED_VARIANTS",
                    "N_95PCT_CS_ROWS",
                    "MAX_PIP",
                    "N_SELECTED_VARIANTS",
                    "N_EQTL_VARIANTS",
                    "N_SQTL_VARIANTS",
                    "N_SPLICEAI_GE_0_20",
                    "N_PANGOLIN_SCORED",
                    "N_MULTI_SOURCE_GE_2",
                    "N_UNIQUE_SHARED_VARIANTS",
                    "N_GENE_TISSUE_PAIRS",
                    "N_LCLPP_GE_0_01",
                    "MAX_LCLPP",
                ],
            )

            print_table(
                "STEP COMPLETION MATRIX",
                completion,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 01 / 02 - GWAS DISCOVERY + DOWNLOADS",
                step02,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 03 - GWAS QC / HARMONIZATION",
                step03,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 04 - ANCESTRY-SPECIFIC 1000G LD REFERENCE",
                step04,
                max_rows=22,
            )

            print()
            print(
                "Step04 chromosomes complete: "
                f"{int(step04['COMPLETE'].sum())}/22"
            )

            print_table(
                "STEP 05 - LD CLUMPING / LOCUS DEFINITION",
                step05,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 06 - SuSiE FINE-MAPPING",
                step06,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 07 - VEP FUNCTIONAL ANNOTATION",
                step07,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 08 - GTEx v11 ALL-TISSUE eQTL / sQTL",
                step08,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 09 - SPLICEAI",
                step09,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 10 - PANGOLIN",
                step10,
                max_rows=args.max_rows,
            )

            print_table(
                "STEP 11 - GTEx v11 SuSiE COLOCALIZATION",
                step11,
                max_rows=args.max_rows,
            )

            print_table(
                "STUDY EXCLUSIONS / WHY A GWAS DID NOT ADVANCE",
                exclusions,
                max_rows=args.max_rows,
            )

            print_table(
                "TOOL WARNINGS / ERRORS FOUND IN LOGS",
                warnings,
                max_rows=args.max_rows,
                columns=[
                    "STUDY_ACCESSION",
                    "N_WARNING_LINES",
                    "N_ERROR_OR_FAILED_LINES",
                    "N_SKIPPING_VARIANT_LINES",
                    "EXAMPLES",
                    "FILE",
                ],
            )

            print_table(
                (
                    "COMMON FINE-MAPPED VARIANTS ACROSS >=2 GWAS "
                    f"(PIP >= {args.common_pip})"
                ),
                common_variants,
                max_rows=args.max_rows,
            )

            print_table(
                "COMMON VEP-MAPPED GENES ACROSS >=2 GWAS",
                common_vep,
                max_rows=args.max_rows,
            )

            print_table(
                "COMMON COLOCALIZED GENES ACROSS >=2 GWAS",
                common_coloc_genes,
                max_rows=args.max_rows,
            )

            print_table(
                "COMMON COLOCALIZATION TISSUES ACROSS >=2 GWAS",
                common_coloc_tissues,
                max_rows=args.max_rows,
            )

            print_table(
                "OUTPUT FILE INVENTORY",
                inventory,
                max_rows=args.max_rows,
                columns=[
                    "STEP_DIR",
                    "STUDY_ACCESSION",
                    "SIZE",
                    "FILE",
                ],
            )

            # ---------------------------------------------------------
            # Global totals
            # ---------------------------------------------------------

            section(
                "PHENOTYPE / ANCESTRY GLOBAL SUMMARY"
            )

            print(
                f"Phenotype                    : {phenotype}"
            )
            print(
                f"Ancestry                     : {ancestry_label} ({ancestry_code})"
            )
            print(
                f"GWAS studies discovered      : {len(studies)}"
            )

            if "N_VARIANTS_AFTER_QC" in step03.columns:
                print(
                    "Total variants after QC       : "
                    f"{pd.to_numeric(step03['N_VARIANTS_AFTER_QC'], errors='coerce').sum():,.0f}"
                )

            if "N_GWS_QC" in step03.columns:
                print(
                    "Total GWS variants after QC    : "
                    f"{pd.to_numeric(step03['N_GWS_QC'], errors='coerce').sum():,.0f}"
                )

            if "N_LEAD_VARIANTS" in step05.columns:
                print(
                    "Total independent lead variants: "
                    f"{pd.to_numeric(step05['N_LEAD_VARIANTS'], errors='coerce').sum():,.0f}"
                )

            if "N_LOCI_SUCCESS" in step06.columns:
                print(
                    "Fine-mapped loci completed     : "
                    f"{pd.to_numeric(step06['N_LOCI_SUCCESS'], errors='coerce').sum():,.0f}"
                )

            if "N_FINEMAPPED_VARIANTS" in step06.columns:
                print(
                    "Fine-mapped variant rows        : "
                    f"{pd.to_numeric(step06['N_FINEMAPPED_VARIANTS'], errors='coerce').sum():,.0f}"
                )

            if "N_95PCT_CS_ROWS" in step06.columns:
                print(
                    "95% credible-set rows           : "
                    f"{pd.to_numeric(step06['N_95PCT_CS_ROWS'], errors='coerce').sum():,.0f}"
                )

            if "N_EQTL_VARIANTS" in step08.columns:
                print(
                    "GTEx eQTL-supported variants    : "
                    f"{pd.to_numeric(step08['N_EQTL_VARIANTS'], errors='coerce').sum():,.0f}"
                )

            if "N_SQTL_VARIANTS" in step08.columns:
                print(
                    "GTEx sQTL-supported variants    : "
                    f"{pd.to_numeric(step08['N_SQTL_VARIANTS'], errors='coerce').sum():,.0f}"
                )

            if "N_SPLICEAI_GE_0_20" in step09.columns:
                print(
                    "SpliceAI >=0.20 variants        : "
                    f"{pd.to_numeric(step09['N_SPLICEAI_GE_0_20'], errors='coerce').sum():,.0f}"
                )

            if "N_PANGOLIN_SCORED" in step10.columns:
                print(
                    "Pangolin-scored variants        : "
                    f"{pd.to_numeric(step10['N_PANGOLIN_SCORED'], errors='coerce').sum():,.0f}"
                )

            if "N_VARIANT_OVERLAP_ROWS" in step11.columns:
                print(
                    "GTEx SuSiE overlap rows          : "
                    f"{pd.to_numeric(step11['N_VARIANT_OVERLAP_ROWS'], errors='coerce').sum():,.0f}"
                )

            if "N_LCLPP_GE_0_01" in step11.columns:
                print(
                    "Coloc pairs LCLPP >=0.01        : "
                    f"{pd.to_numeric(step11['N_LCLPP_GE_0_01'], errors='coerce').sum():,.0f}"
                )

            print(
                f"Common fine-mapped variants     : {len(common_variants):,}"
            )
            print(
                f"Common VEP genes                : {len(common_vep):,}"
            )
            print(
                f"Common colocalized genes        : {len(common_coloc_genes):,}"
            )
            print(
                f"Common colocalization tissues   : {len(common_coloc_tissues):,}"
            )
            print(
                f"Warnings/error files detected   : {len(warnings):,}"
            )

            # ---------------------------------------------------------
            # Manifest
            # ---------------------------------------------------------

            manifest = {
                "REPORT_VERSION": REPORT_VERSION,
                "PROJECT_ROOT": str(root),
                "PHENOTYPE": phenotype,
                "PHENOTYPE_SLUG": phenotype_slug,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
                "ANCESTRY_SLUG": ancestry_slug,
                "N_STUDIES": len(studies),
                "STUDIES": studies,
                "COMMON_PIP_THRESHOLD": args.common_pip,
                "REPORT_DIR": str(report_dir),
                "TERMINAL_REPORT": str(terminal_report),
                "TABLES": {
                    filename: {
                        "path": str(report_dir / filename),
                        "rows": int(len(table)),
                    }
                    for filename, table in tables.items()
                },
            }

            (
                report_dir
                / "report_manifest.json"
            ).write_text(
                json.dumps(
                    manifest,
                    indent=2,
                ),
                encoding="utf-8",
            )

            section(
                "REPORT COMPLETE"
            )

            print(
                f"Terminal report:\n  {terminal_report}"
            )

            print()
            print(
                "Full TSV tables:"
            )

            for filename in tables:
                print(
                    f"  {report_dir / filename}"
                )

            print()
            print(
                f"Manifest:\n  {report_dir / 'report_manifest.json'}"
            )

        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
 

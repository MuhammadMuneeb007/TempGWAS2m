#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
GWAS2m - STEP 12
MASTER BIOLOGICAL EVIDENCE TABLE BUILDER
===============================================================================

PURPOSE
-------
Step12 is a READ-ONLY consolidation step. It does not rerun GWAS, SuSiE,
colocalization, VEP, SpliceAI, or Pangolin.

It combines:
  * Step11 provider-SuSiE candidate manifest (source of truth)
  * Step11 candidate_result.json files
  * Step07/09/10 cumulative variant annotation output when available

and writes analysis-ready master tables:

  1. Step12_Molecular_Evidence_Long.tsv
       One row per Step11 scientific candidate/comparison.

  2. Step12_Locus_Gene_Master.tsv
       One row per GWAS study x locus x candidate effector.

  3. Step12_Locus_Master.tsv
       One row per GWAS study x locus.

  4. Step12_Variant_Master.tsv
       Fine-mapped/annotated variant table from the latest cumulative
       annotation layer available (prefer Step10; fallback Step07).

  5. Step12_Locus_Annotation.tsv
       Locus-level summary of PIP/VEP/SpliceAI/Pangolin evidence.

  6. Step12_Strong_Shared_Signals.tsv
       All strong H4 comparisons.

  7. Step12_QC_Problems.tsv
       Missing/failed/stale/unreadable Step11 candidates.

IMPORTANT SCIENTIFIC RULES
--------------------------
* NOT_TESTED / FAILED / MISSING are NOT biological negatives.
* A QTL modality is called supported only from a finite tested H4.
* Step12 prefers same-pair BEST_PAIR_PP_H0..H4 fields.
* Legacy MAX_PP_H0..H4 are retained only as a fallback and are flagged.
* Step12 does not assign novelty. That belongs in Step13 after Open Targets /
  external disease evidence integration.

TYPICAL USE
-----------
python Step12_Master_Table.py \
    --phenotype "parkinson's disease" \
    --ancestry EUR

Run it repeatedly while Step11 is finishing; outputs are rebuilt deterministically.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VERSION = "1.0.0"
QTL_TYPES = ["eQTL", "sQTL", "isoQTL", "exonQTL", "pQTL"]
DEFAULT_STRONG_PP4 = 0.80
DEFAULT_SUGGESTIVE_PP4 = 0.50

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
    print("=" * 120)
    print(text)
    print("=" * 120)


def norm_text(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: Any) -> str:
    return norm_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = norm_text(value)
    if key not in ANCESTRY_ALIASES:
        raise SystemExit(
            f"Unsupported ancestry {value!r}. Use EUR, AFR, EAS, SAS, or AMR."
        )
    return ANCESTRY_ALIASES[key]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null", "na", "<na>"}:
        return ""
    return s


def truthy(value: Any) -> bool:
    return clean_str(value).lower() in {"1", "true", "yes", "y", "t"}


def finite_float(value: Any) -> float:
    try:
        z = float(value)
    except Exception:
        return np.nan
    return z if math.isfinite(z) else np.nan


def first_numeric(row: pd.Series | dict, names: Iterable[str]) -> float:
    for name in names:
        try:
            value = row.get(name, np.nan)
        except AttributeError:
            value = np.nan
        z = finite_float(value)
        if math.isfinite(z):
            return z
    return np.nan


def first_text(row: pd.Series | dict, names: Iterable[str]) -> str:
    for name in names:
        try:
            value = row.get(name, "")
        except AttributeError:
            value = ""
        s = clean_str(value)
        if s:
            return s
    return ""


def join_unique(values: Iterable[Any], sep: str = ";", limit: int | None = None) -> str:
    seen: list[str] = []
    for value in values:
        s = clean_str(value)
        if not s:
            continue
        # Preserve compound biological identifiers but flatten common list separators.
        tokens = re.split(r"[;,]", s)
        for token in tokens:
            token = token.strip()
            if token and token not in seen:
                seen.append(token)
                if limit is not None and len(seen) >= limit:
                    return sep.join(seen)
    return sep.join(seen)


def safe_write_json(path: Path, obj: dict[str, Any]) -> None:
    def normalise(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): normalise(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [normalise(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            if not math.isfinite(float(x)):
                return None
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    path.write_text(
        json.dumps(normalise(obj), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json_compatible(path: Path, retries: int = 3) -> tuple[dict[str, Any], str]:
    """Python json accepts Step11 NaN/Infinity tokens; retry protects active writes."""
    last_error = ""
    for attempt in range(retries):
        try:
            if not path.exists() or path.stat().st_size == 0:
                return {}, "MISSING_OR_EMPTY"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload, "OK"
            return {}, "JSON_ROOT_NOT_OBJECT"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(0.05)
    return {}, last_error or "JSON_READ_FAILED"


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], fill: Any = "") -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


def as_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].map(truthy).fillna(False).astype(bool)


def canonical_ensg(text: Any) -> str:
    m = re.search(r"(ENSG\d+)", clean_str(text), flags=re.I)
    return m.group(1).upper() if m else ""


def canonical_enst(text: Any) -> str:
    m = re.search(r"(ENST\d+)", clean_str(text), flags=re.I)
    return m.group(1).upper() if m else ""


def canonical_ensp(text: Any) -> str:
    m = re.search(r"(ENSP\d+)", clean_str(text), flags=re.I)
    return m.group(1).upper() if m else ""


def variant_key(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    c = clean_str(chrom).replace("chr", "").replace("CHR", "")
    p = clean_str(pos)
    r = clean_str(ref).upper()
    a = clean_str(alt).upper()
    if not c or not p or not r or not a:
        return ""
    try:
        p = str(int(float(p)))
    except Exception:
        return ""
    return f"{c}:{p}:{r}:{a}"


# =============================================================================
# PATHS / ARGUMENTS
# =============================================================================

def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GWAS2m Step12: build master biological evidence tables from Step11 + annotations."
    )
    p.add_argument("--phenotype", required=True)
    p.add_argument("--ancestry", required=True)
    p.add_argument("--root", default=".", help="GWAS2m root directory; default current directory.")
    p.add_argument("--strong-pp4", type=float, default=DEFAULT_STRONG_PP4)
    p.add_argument("--suggestive-pp4", type=float, default=DEFAULT_SUGGESTIVE_PP4)
    p.add_argument(
        "--annotation-source",
        choices=["auto", "step10", "step07", "none"],
        default="auto",
        help="Prefer cumulative Step10 annotation, fallback Step07 VEP in auto mode.",
    )
    p.add_argument(
        "--fail-on-step11-problems",
        action="store_true",
        help="Exit nonzero if FAILED/MISSING/PARSE_ERROR/STALE Step11 rows remain.",
    )
    return p.parse_args()


def pipeline_paths(root: Path, phenotype: str, ancestry_label: str) -> dict[str, Path]:
    p = slugify(phenotype)
    a = slugify(ancestry_label)
    return {
        "S06": root / "06_finemapping" / p / a,
        "S07": root / "07_annotation" / p / a,
        "S09": root / "09_splicing" / p / a,
        "S10": root / "10_pangolin" / p / a,
        "S11": root / "11_coloc_provider_susie" / p / a,
        "S12": root / "12_master_table" / p / a,
    }


# =============================================================================
# STEP11 LOADER
# =============================================================================

def posterior_source(payload: dict[str, Any]) -> str:
    if any(math.isfinite(finite_float(payload.get(f"BEST_PAIR_PP_H{i}"))) for i in range(5)):
        return "BEST_PAIR_PP_H*"
    if any(math.isfinite(finite_float(payload.get(f"BEST_PP_H{i}"))) for i in range(5)):
        return "BEST_PP_H*"
    if any(math.isfinite(finite_float(payload.get(f"MAX_PP_H{i}"))) for i in range(5)):
        return "LEGACY_MAX_PP_H*_FALLBACK"
    return ""


def derive_result_state(status: str, json_state: str, result_present: bool) -> str:
    s = clean_str(status).upper()
    if not result_present:
        return "MISSING"
    if json_state != "OK":
        return "PARSE_ERROR"
    if s == "COMPLETE":
        return "COMPLETE"
    if s.startswith("NOT_TESTED"):
        return "NOT_TESTED"
    if s.startswith("FAILED"):
        return "FAILED"
    if s.startswith("STALE"):
        return "STALE"
    if s in {"", "UNKNOWN"}:
        return "UNKNOWN"
    return s


def derive_evidence_class(
    tested: bool,
    h: list[float],
    strong: float,
    suggestive: float,
    status: str,
) -> str:
    h4 = h[4]
    if not tested:
        s = clean_str(status).upper()
        if s.startswith("NOT_TESTED"):
            return s
        return "NOT_FORMALLY_TESTED"
    if not math.isfinite(h4):
        return "TESTED_NO_FINITE_H4"
    if h4 >= strong:
        return "STRONG_SHARED_SIGNAL"
    if h4 >= suggestive:
        return "SUGGESTIVE_SHARED_SIGNAL"
    finite = [(i, v) for i, v in enumerate(h) if math.isfinite(v)]
    if finite:
        best_h, _ = max(finite, key=lambda kv: kv[1])
        if best_h == 3:
            return "BOTH_ASSOCIATED_DIFFERENT_SIGNALS_FAVORED"
        if best_h == 4:
            return "H4_DOMINANT_BELOW_SUGGESTIVE_THRESHOLD"
        return f"H{best_h}_DOMINANT"
    return "TESTED_NO_FINITE_POSTERIOR"


def extract_best_hits(payload: dict[str, Any]) -> tuple[str, str, str]:
    h1 = first_text(payload, ["BEST_HIT1", "BEST_GWAS_HIT"])
    h2 = first_text(payload, ["BEST_HIT2", "BEST_QTL_HIT"])
    pair = first_text(payload, ["BEST_SIGNAL_PAIR"])
    if pair and "|" in pair:
        a, b = pair.split("|", 1)
        h1 = h1 or a.strip()
        h2 = h2 or b.strip()
    if not pair and (h1 or h2):
        pair = f"{h1}|{h2}".strip("|")
    return h1, h2, pair


def load_step11_long(
    s11: Path,
    phenotype: str,
    ancestry_code: str,
    ancestry_label: str,
    strong: float,
    suggestive: float,
) -> pd.DataFrame:
    manifest_file = s11 / "provider_coloc_manifest.tsv"
    result_root = s11 / "candidate_results"

    if not manifest_file.exists() or manifest_file.stat().st_size == 0:
        raise FileNotFoundError(
            "Step11 provider_coloc_manifest.tsv is missing/empty:\n"
            f"  {manifest_file}\n"
            "Finish/build the Step11 provider plan first."
        )

    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    required = [
        "CANDIDATE_ID", "STUDY_ACCESSION", "LOCUS_ID", "DATASET_ID",
        "QTL_TYPE", "TISSUE", "MOLECULAR_TRAIT_ID",
    ]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise RuntimeError(f"Step11 manifest missing required columns: {missing}")
    if manifest["CANDIDATE_ID"].duplicated().any():
        dup = manifest.loc[manifest["CANDIDATE_ID"].duplicated(False), "CANDIDATE_ID"].head(20).tolist()
        raise RuntimeError(f"Duplicate Step11 CANDIDATE_ID values found: {dup}")

    banner("STEP12 - LOADING STEP11 CANDIDATE RESULTS")
    print(f"Manifest        : {manifest_file}")
    print(f"Candidates      : {len(manifest):,}")
    print(f"Result directory: {result_root}")

    rows: list[dict[str, Any]] = []
    for i, mrow in enumerate(manifest.to_dict(orient="records"), start=1):
        cid = clean_str(mrow.get("CANDIDATE_ID"))
        result_file = result_root / cid / "candidate_result.json"
        present = result_file.exists() and result_file.stat().st_size > 0
        payload, json_state = read_json_compatible(result_file)

        row: dict[str, Any] = dict(mrow)
        # Result values are authoritative for formal-coloc outputs/status.
        for k, v in payload.items():
            row[k] = v

        status = first_text(row, ["STATUS"]) or ("MISSING_RESULT" if not present else "UNKNOWN")
        tested = truthy(row.get("COLOC_TESTED", ""))

        source = posterior_source(payload)
        hvals = []
        for h in range(5):
            if source == "BEST_PAIR_PP_H*":
                z = finite_float(payload.get(f"BEST_PAIR_PP_H{h}"))
            elif source == "BEST_PP_H*":
                z = finite_float(payload.get(f"BEST_PP_H{h}"))
            elif source == "LEGACY_MAX_PP_H*_FALLBACK":
                z = finite_float(payload.get(f"MAX_PP_H{h}"))
            else:
                z = np.nan
            hvals.append(z)

        # If status says COMPLETE but COLOC_TESTED is missing in an older JSON,
        # finite same-pair H4 is sufficient to mark it formally tested.
        if not tested and status.upper() == "COMPLETE" and math.isfinite(hvals[4]):
            tested = True

        low = first_numeric(row, ["PP_H4_P12_LOW", "H4_P12_LOW"])
        default = first_numeric(row, ["PP_H4_P12_DEFAULT", "H4_P12_DEFAULT"])
        if not math.isfinite(default):
            default = hvals[4]
        high = first_numeric(row, ["PP_H4_P12_HIGH", "H4_P12_HIGH"])
        sens = [z for z in [low, default, high] if math.isfinite(z)]
        prior_min = first_numeric(row, ["PRIOR_H4_MIN"])
        prior_max = first_numeric(row, ["PRIOR_H4_MAX"])
        if not math.isfinite(prior_min) and sens:
            prior_min = min(sens)
        if not math.isfinite(prior_max) and sens:
            prior_max = max(sens)

        prior_robust_strong = truthy(row.get("PRIOR_ROBUST_STRONG", ""))
        if sens and min(sens) >= strong:
            prior_robust_strong = True
        prior_robust_suggestive = truthy(row.get("PRIOR_ROBUST_SUGGESTIVE", ""))
        if sens and min(sens) >= suggestive:
            prior_robust_suggestive = True

        hit1, hit2, pair = extract_best_hits(row)

        trait = clean_str(row.get("MOLECULAR_TRAIT_ID"))
        gene_id = canonical_ensg(first_text(row, ["GENE_ID", "MOLECULAR_TRAIT_ID"]))
        transcript_id = canonical_enst(first_text(row, ["TRANSCRIPT_ID", "MOLECULAR_TRAIT_ID"]))
        protein_id = canonical_ensp(first_text(row, ["PROTEIN_ID", "MOLECULAR_TRAIT_ID"]))

        if gene_id:
            entity_type = "GENE"
        elif transcript_id:
            entity_type = "TRANSCRIPT"
        elif protein_id:
            entity_type = "PROTEIN"
        else:
            qtype = clean_str(row.get("QTL_TYPE")).lower()
            entity_type = {
                "isoqtl": "ISOFORM_OR_TRANSCRIPT",
                "exonqtl": "EXON",
                "sqtl": "SPLICING_PHENOTYPE",
                "pqtl": "PROTEIN_OR_PQTL_TRAIT",
                "eqtl": "GENE_EXPRESSION_TRAIT",
            }.get(qtype, "MOLECULAR_TRAIT")

        row.update({
            "STEP12_VERSION": VERSION,
            "PHENOTYPE": phenotype,
            "ANCESTRY_CODE": ancestry_code,
            "ANCESTRY_LABEL": ancestry_label,
            "RESULT_FILE": str(result_file),
            "RESULT_PRESENT": "YES" if present else "NO",
            "RESULT_JSON_STATE": json_state,
            "RESULT_STATE": derive_result_state(status, json_state, present),
            "STATUS": status,
            "FORMAL_COLOC_TESTED": "YES" if tested else "NO",
            "POSTERIOR_SOURCE": source,
            "PP_H0": hvals[0],
            "PP_H1": hvals[1],
            "PP_H2": hvals[2],
            "PP_H3": hvals[3],
            "PP_H4": hvals[4],
            "PP_H4_P12_LOW_STEP12": low,
            "PP_H4_P12_DEFAULT_STEP12": default,
            "PP_H4_P12_HIGH_STEP12": high,
            "PRIOR_H4_MIN_STEP12": prior_min,
            "PRIOR_H4_MAX_STEP12": prior_max,
            "PRIOR_ROBUST_STRONG_STEP12": "YES" if prior_robust_strong else "NO",
            "PRIOR_ROBUST_SUGGESTIVE_STEP12": "YES" if prior_robust_suggestive else "NO",
            "EVIDENCE_CLASS_STEP12": derive_evidence_class(tested, hvals, strong, suggestive, status),
            "BEST_HIT1_STEP12": hit1,
            "BEST_HIT2_STEP12": hit2,
            "BEST_SIGNAL_PAIR_STEP12": pair,
            "EFFECTOR_GENE_ID": gene_id,
            "EFFECTOR_TRANSCRIPT_ID": transcript_id,
            "EFFECTOR_PROTEIN_ID": protein_id,
            "MOLECULAR_ENTITY_TYPE": entity_type,
        })
        rows.append(row)

        if i % 5000 == 0 or i == len(manifest):
            print(f"  parsed {i:,}/{len(manifest):,}", flush=True)

    x = pd.DataFrame(rows)
    x["PP_H4"] = pd.to_numeric(x["PP_H4"], errors="coerce")
    x["IS_STRONG"] = x["FORMAL_COLOC_TESTED"].eq("YES") & x["PP_H4"].ge(strong)
    x["IS_SUGGESTIVE"] = (
        x["FORMAL_COLOC_TESTED"].eq("YES")
        & x["PP_H4"].ge(suggestive)
        & x["PP_H4"].lt(strong)
    )
    x["IS_SHARED_SIGNAL"] = x["FORMAL_COLOC_TESTED"].eq("YES") & x["PP_H4"].ge(suggestive)
    return x


# =============================================================================
# ANNOTATION LOADER (STEP10 PREFERRED, STEP07 FALLBACK)
# =============================================================================

def discover_annotation_files(paths: dict[str, Path], source: str) -> tuple[str, list[Path]]:
    if source in {"auto", "step10"}:
        files = sorted(paths["S10"].glob("GCST*/*_SpliceAI_Pangolin_integrated.tsv"))
        if files:
            return "STEP10_SPLICEAI_PANGOLIN_INTEGRATED", files
        if source == "step10":
            return "STEP10_NOT_FOUND", []

    if source in {"auto", "step07"}:
        files = sorted(paths["S07"].glob("GCST*/*_VEP_variant_summary.tsv"))
        if files:
            return "STEP07_VEP_VARIANT_SUMMARY", files
        if source == "step07":
            return "STEP07_NOT_FOUND", []

    return "NONE", []


def add_variant_key_column(df: pd.DataFrame) -> pd.DataFrame:
    if "VARIANT_KEY" in df.columns:
        vk = df["VARIANT_KEY"].fillna("").astype(str)
        if vk.str.len().gt(0).any():
            return df
    needed = ["CHR", "REFERENCE_POS", "REFERENCE_REF", "REFERENCE_ALT"]
    if all(c in df.columns for c in needed):
        df["VARIANT_KEY"] = [
            variant_key(c, p, r, a)
            for c, p, r, a in zip(
                df["CHR"], df["REFERENCE_POS"], df["REFERENCE_REF"], df["REFERENCE_ALT"]
            )
        ]
    else:
        df["VARIANT_KEY"] = ""
    return df


def load_variant_annotations(files: list[Path], source_name: str) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()

    banner(f"STEP12 - LOADING VARIANT ANNOTATIONS ({source_name})")
    frames: list[pd.DataFrame] = []
    for i, f in enumerate(files, start=1):
        try:
            df = pd.read_csv(f, sep="\t", low_memory=False)
        except Exception as exc:
            print(f"WARNING: could not read {f}: {type(exc).__name__}: {exc}")
            continue
        if df.empty:
            continue
        accession = f.parent.name if f.parent.name.startswith("GCST") else ""
        if "STUDY_ACCESSION" not in df.columns:
            df["STUDY_ACCESSION"] = accession
        else:
            df["STUDY_ACCESSION"] = df["STUDY_ACCESSION"].fillna("").astype(str)
            if accession:
                df.loc[df["STUDY_ACCESSION"].str.len().eq(0), "STUDY_ACCESSION"] = accession
        df["ANNOTATION_SOURCE_STEP12"] = source_name
        df["ANNOTATION_SOURCE_FILE"] = str(f)
        df = add_variant_key_column(df)
        frames.append(df)
        print(f"  {i:>3}/{len(files)} {f.name}: {len(df):,} variants")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if {"STUDY_ACCESSION", "VEP_ID"}.issubset(out.columns):
        out = out.drop_duplicates(["STUDY_ACCESSION", "VEP_ID"], keep="last")
    elif {"STUDY_ACCESSION", "VARIANT_KEY"}.issubset(out.columns):
        out = out.drop_duplicates(["STUDY_ACCESSION", "VARIANT_KEY"], keep="last")
    return out.reset_index(drop=True)


def build_gene_maps(variants: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    gene_map: dict[str, str] = {}
    transcript_symbol_map: dict[str, str] = {}
    transcript_gene_map: dict[str, str] = {}
    if variants.empty:
        return gene_map, transcript_symbol_map, transcript_gene_map

    symbol_col = next((c for c in ["SYMBOL", "GENE_SYMBOL", "PANGOLIN_GENE", "SPLICEAI_GENE"] if c in variants.columns), None)

    if symbol_col and "Gene" in variants.columns:
        for gid, sym in zip(variants["Gene"], variants[symbol_col]):
            g = canonical_ensg(gid)
            s = clean_str(sym)
            if g and s and g not in gene_map:
                gene_map[g] = s

    if "Feature" in variants.columns:
        symbols = variants[symbol_col] if symbol_col else pd.Series("", index=variants.index)
        genes = variants["Gene"] if "Gene" in variants.columns else pd.Series("", index=variants.index)
        for tid, gid, sym in zip(variants["Feature"], genes, symbols):
            t = canonical_enst(tid)
            g = canonical_ensg(gid)
            s = clean_str(sym)
            if t and g and t not in transcript_gene_map:
                transcript_gene_map[t] = g
            if t and s and t not in transcript_symbol_map:
                transcript_symbol_map[t] = s

    return gene_map, transcript_symbol_map, transcript_gene_map


def best_row_by_numeric(group: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in group.columns or group.empty:
        return None
    z = pd.to_numeric(group[col], errors="coerce")
    if not z.notna().any():
        return None
    return group.loc[z.idxmax()]


def build_locus_annotation_summary(variants: pd.DataFrame) -> pd.DataFrame:
    if variants.empty or not {"STUDY_ACCESSION", "LOCUS_ID"}.issubset(variants.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (study, locus), g in variants.groupby(["STUDY_ACCESSION", "LOCUS_ID"], dropna=False, sort=True):
        pip = as_numeric_series(g, "PIP")
        splice = as_numeric_series(g, "SPLICEAI_MAX_DS")
        pang = as_numeric_series(g, "PANGOLIN_MAX_ABS")

        lead = g.loc[pip.idxmax()] if pip.notna().any() else None
        splice_row = g.loc[splice.idxmax()] if splice.notna().any() else None
        pang_row = g.loc[pang.idxmax()] if pang.notna().any() else None

        symbols = []
        for col in ["SYMBOL", "ALL_SYMBOLS", "SPLICEAI_GENE", "SPLICEAI_ALL_GENES", "PANGOLIN_GENE", "PANGOLIN_ALL_GENES"]:
            if col in g.columns:
                symbols.extend(g[col].tolist())
        gene_ids = []
        for col in ["Gene", "ALL_GENES", "ALL_GENE_IDS"]:
            if col in g.columns:
                gene_ids.extend(g[col].tolist())

        row: dict[str, Any] = {
            "STUDY_ACCESSION": clean_str(study),
            "LOCUS_ID": clean_str(locus),
            "ANNOT_N_VARIANTS": int(len(g)),
            "ANNOT_MAX_PIP": float(pip.max()) if pip.notna().any() else np.nan,
            "ANNOT_MAX_SPLICEAI_DS": float(splice.max()) if splice.notna().any() else np.nan,
            "ANNOT_MAX_PANGOLIN_ABS": float(pang.max()) if pang.notna().any() else np.nan,
            "ANNOT_GENE_SYMBOLS": join_unique(symbols, limit=200),
            "ANNOT_GENE_IDS": join_unique(gene_ids, limit=200),
            "ANNOT_N_SPLICEAI_GE_0_20": int((splice >= 0.20).sum()) if len(splice) else 0,
            "ANNOT_N_SPLICEAI_GE_0_50": int((splice >= 0.50).sum()) if len(splice) else 0,
            "ANNOT_N_PANGOLIN_TOP5": int(bool_series(g, "PANGOLIN_TOP_5PCT").sum()),
            "ANNOT_N_SPLICE_PRIORITY": int(bool_series(g, "SPLICE_PRIORITY_INSPECTION").sum()),
        }

        if "FUNCTIONAL_CATEGORY" in g.columns:
            cat = g["FUNCTIONAL_CATEGORY"].fillna("").astype(str).str.upper()
            for name in ["SPLICING", "CODING", "REGULATORY", "NONCODING_RNA"]:
                row[f"ANNOT_N_{name}"] = int(cat.eq(name).sum())

        def transfer(prefix: str, r: pd.Series | None) -> None:
            if r is None:
                return
            row[f"{prefix}_VARIANT"] = clean_str(r.get("VARIANT_KEY", ""))
            row[f"{prefix}_VEP_ID"] = clean_str(r.get("VEP_ID", ""))
            row[f"{prefix}_SYMBOL"] = first_text(r, ["SYMBOL", "SPLICEAI_GENE", "PANGOLIN_GENE"])
            row[f"{prefix}_GENE_ID"] = canonical_ensg(first_text(r, ["Gene", "ALL_GENES"]))
            row[f"{prefix}_CONSEQUENCE"] = first_text(r, ["Consequence", "ALL_CONSEQUENCES"])
            row[f"{prefix}_FUNCTIONAL_CATEGORY"] = first_text(r, ["FUNCTIONAL_CATEGORY", "ALL_FUNCTIONAL_CATEGORIES"])
            row[f"{prefix}_BIOTYPE"] = first_text(r, ["BIOTYPE", "ALL_BIOTYPES"])

        transfer("TOP_PIP", lead)
        transfer("TOP_SPLICEAI", splice_row)
        transfer("TOP_PANGOLIN", pang_row)
        rows.append(row)

    return pd.DataFrame(rows)


def annotate_effector_symbols(long: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    gene_map, transcript_symbol_map, transcript_gene_map = build_gene_maps(variants)
    existing = long.get("GENE_SYMBOL", pd.Series("", index=long.index)).fillna("").astype(str)
    symbols: list[str] = []
    gene_ids: list[str] = []
    for idx, row in long.iterrows():
        s = clean_str(existing.loc[idx])
        gid = clean_str(row.get("EFFECTOR_GENE_ID", ""))
        tid = clean_str(row.get("EFFECTOR_TRANSCRIPT_ID", ""))
        if not gid and tid:
            gid = transcript_gene_map.get(tid, "")
        if not s and gid:
            s = gene_map.get(gid, "")
        if not s and tid:
            s = transcript_symbol_map.get(tid, "")
        gene_ids.append(gid)
        symbols.append(s)
    long["EFFECTOR_GENE_ID"] = gene_ids
    long["EFFECTOR_GENE_SYMBOL"] = symbols

    effectors = []
    for _, row in long.iterrows():
        key = first_text(row, [
            "EFFECTOR_GENE_ID",
            "EFFECTOR_GENE_SYMBOL",
            "EFFECTOR_TRANSCRIPT_ID",
            "EFFECTOR_PROTEIN_ID",
            "MOLECULAR_TRAIT_ID",
        ])
        effectors.append(key or "UNRESOLVED_EFFECTOR")
    long["EFFECTOR_KEY"] = effectors
    return long


def merge_locus_annotations(long: pd.DataFrame, locus_annot: pd.DataFrame) -> pd.DataFrame:
    if locus_annot.empty:
        return long
    return long.merge(
        locus_annot,
        on=["STUDY_ACCESSION", "LOCUS_ID"],
        how="left",
        validate="many_to_one",
    )


# =============================================================================
# COLLAPSED MASTER TABLES
# =============================================================================

def modality_summary(g: pd.DataFrame, qtl_type: str, strong: float, suggestive: float) -> dict[str, Any]:
    sub = g[g["QTL_TYPE"].astype(str).str.lower().eq(qtl_type.lower())].copy()
    prefix = qtl_type.upper().replace("QTL", "QTL")
    # Explicit conventional names.
    prefix = {
        "eQTL": "EQTL",
        "sQTL": "SQTL",
        "isoQTL": "ISOQTL",
        "exonQTL": "EXONQTL",
        "pQTL": "PQTL",
    }[qtl_type]

    out: dict[str, Any] = {
        f"{prefix}_N_CANDIDATES": int(len(sub)),
        f"{prefix}_N_TESTED": 0,
        f"{prefix}_N_STRONG": 0,
        f"{prefix}_N_SUGGESTIVE": 0,
        f"{prefix}_SUPPORTED": "NO",
        f"{prefix}_TEST_STATUS": "NO_CANDIDATE",
        f"{prefix}_BEST_H4": np.nan,
        f"{prefix}_BEST_TISSUE": "",
        f"{prefix}_BEST_DATASET_ID": "",
        f"{prefix}_BEST_TRAIT": "",
        f"{prefix}_BEST_SIGNAL_PAIR": "",
        f"{prefix}_PRIOR_ROBUST_STRONG": "NO",
    }
    if sub.empty:
        return out

    tested = sub["FORMAL_COLOC_TESTED"].eq("YES")
    h4 = pd.to_numeric(sub["PP_H4"], errors="coerce")
    out[f"{prefix}_N_TESTED"] = int(tested.sum())
    out[f"{prefix}_N_STRONG"] = int((tested & h4.ge(strong)).sum())
    out[f"{prefix}_N_SUGGESTIVE"] = int((tested & h4.ge(suggestive) & h4.lt(strong)).sum())

    if tested.any():
        out[f"{prefix}_TEST_STATUS"] = "TESTED"
    else:
        out[f"{prefix}_TEST_STATUS"] = "NOT_TESTED"

    eligible = sub[tested & h4.notna()].copy()
    if not eligible.empty:
        best_idx = pd.to_numeric(eligible["PP_H4"], errors="coerce").idxmax()
        b = eligible.loc[best_idx]
        best_h4 = finite_float(b.get("PP_H4"))
        out[f"{prefix}_BEST_H4"] = best_h4
        out[f"{prefix}_BEST_TISSUE"] = clean_str(b.get("TISSUE"))
        out[f"{prefix}_BEST_DATASET_ID"] = clean_str(b.get("DATASET_ID"))
        out[f"{prefix}_BEST_TRAIT"] = clean_str(b.get("MOLECULAR_TRAIT_ID"))
        out[f"{prefix}_BEST_SIGNAL_PAIR"] = clean_str(b.get("BEST_SIGNAL_PAIR_STEP12"))
        out[f"{prefix}_PRIOR_ROBUST_STRONG"] = clean_str(b.get("PRIOR_ROBUST_STRONG_STEP12")) or "NO"
        out[f"{prefix}_SUPPORTED"] = "YES" if best_h4 >= suggestive else "NO"
    return out


def evidence_tier(
    best_h4: float,
    prior_robust: bool,
    n_shared_modalities: int,
    n_strong_modalities: int,
    strong: float,
    suggestive: float,
) -> str:
    if not math.isfinite(best_h4):
        return "NO_FINITE_TESTED_H4"
    if best_h4 >= strong and prior_robust and n_strong_modalities >= 2:
        return "TIER_1_MULTIMODAL_PRIOR_ROBUST_STRONG"
    if best_h4 >= strong and prior_robust:
        return "TIER_1_PRIOR_ROBUST_STRONG"
    if best_h4 >= strong:
        return "TIER_2_STRONG"
    if best_h4 >= suggestive:
        return "TIER_3_SUGGESTIVE"
    return "TIER_4_TESTED_NO_SHARED_SIGNAL"


def build_locus_gene_master(long: pd.DataFrame, strong: float, suggestive: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["STUDY_ACCESSION", "LOCUS_ID", "EFFECTOR_KEY"]

    for (study, locus, effector), g in long.groupby(keys, dropna=False, sort=True):
        tested = g["FORMAL_COLOC_TESTED"].eq("YES")
        h4 = pd.to_numeric(g["PP_H4"], errors="coerce")
        eligible = g[tested & h4.notna()].copy()
        best = eligible.loc[pd.to_numeric(eligible["PP_H4"], errors="coerce").idxmax()] if not eligible.empty else g.iloc[0]
        best_h4 = finite_float(best.get("PP_H4")) if not eligible.empty else np.nan

        modality_blocks = [modality_summary(g, qt, strong, suggestive) for qt in QTL_TYPES]
        modality_flat: dict[str, Any] = {}
        for block in modality_blocks:
            modality_flat.update(block)

        supported_types = []
        strong_types = []
        for qt in QTL_TYPES:
            prefix = {"eQTL":"EQTL","sQTL":"SQTL","isoQTL":"ISOQTL","exonQTL":"EXONQTL","pQTL":"PQTL"}[qt]
            z = finite_float(modality_flat.get(f"{prefix}_BEST_H4"))
            if math.isfinite(z) and z >= suggestive:
                supported_types.append(qt)
            if math.isfinite(z) and z >= strong:
                strong_types.append(qt)

        prior_robust_any = g["PRIOR_ROBUST_STRONG_STEP12"].eq("YES").any()
        shared = g[g["IS_SHARED_SIGNAL"]].copy()

        row: dict[str, Any] = {
            "PHENOTYPE": first_text(g.iloc[0], ["PHENOTYPE"]),
            "ANCESTRY_CODE": first_text(g.iloc[0], ["ANCESTRY_CODE"]),
            "ANCESTRY_LABEL": first_text(g.iloc[0], ["ANCESTRY_LABEL"]),
            "STUDY_ACCESSION": clean_str(study),
            "LOCUS_ID": clean_str(locus),
            "EFFECTOR_KEY": clean_str(effector),
            "EFFECTOR_GENE_ID": join_unique(g.get("EFFECTOR_GENE_ID", pd.Series(dtype=str)), limit=20),
            "EFFECTOR_GENE_SYMBOL": join_unique(g.get("EFFECTOR_GENE_SYMBOL", pd.Series(dtype=str)), limit=20),
            "EFFECTOR_TRANSCRIPT_IDS": join_unique(g.get("EFFECTOR_TRANSCRIPT_ID", pd.Series(dtype=str)), limit=100),
            "EFFECTOR_PROTEIN_IDS": join_unique(g.get("EFFECTOR_PROTEIN_ID", pd.Series(dtype=str)), limit=100),
            "MOLECULAR_TRAITS": join_unique(g.get("MOLECULAR_TRAIT_ID", pd.Series(dtype=str)), limit=200),
            "N_CANDIDATES": int(len(g)),
            "N_TESTED": int(tested.sum()),
            "N_COMPLETE": int(g["RESULT_STATE"].eq("COMPLETE").sum()),
            "N_NOT_TESTED": int(g["RESULT_STATE"].eq("NOT_TESTED").sum()),
            "N_FAILED": int(g["RESULT_STATE"].eq("FAILED").sum()),
            "N_MISSING": int(g["RESULT_STATE"].eq("MISSING").sum()),
            "N_PARSE_ERROR": int(g["RESULT_STATE"].eq("PARSE_ERROR").sum()),
            "N_STRONG": int((tested & h4.ge(strong)).sum()),
            "N_SUGGESTIVE": int((tested & h4.ge(suggestive) & h4.lt(strong)).sum()),
            "BEST_H4": best_h4,
            "BEST_PP_H0": finite_float(best.get("PP_H0")),
            "BEST_PP_H1": finite_float(best.get("PP_H1")),
            "BEST_PP_H2": finite_float(best.get("PP_H2")),
            "BEST_PP_H3": finite_float(best.get("PP_H3")),
            "BEST_PP_H4": finite_float(best.get("PP_H4")),
            "BEST_QTL_TYPE": clean_str(best.get("QTL_TYPE")),
            "BEST_TISSUE": clean_str(best.get("TISSUE")),
            "BEST_DATASET_ID": clean_str(best.get("DATASET_ID")),
            "BEST_MOLECULAR_TRAIT_ID": clean_str(best.get("MOLECULAR_TRAIT_ID")),
            "BEST_SIGNAL_PAIR": clean_str(best.get("BEST_SIGNAL_PAIR_STEP12")),
            "BEST_EVIDENCE_CLASS": clean_str(best.get("EVIDENCE_CLASS_STEP12")),
            "PRIOR_H4_MIN": finite_float(best.get("PRIOR_H4_MIN_STEP12")),
            "PRIOR_H4_MAX": finite_float(best.get("PRIOR_H4_MAX_STEP12")),
            "PRIOR_ROBUST_STRONG": "YES" if prior_robust_any else "NO",
            "N_QTL_TYPES_CANDIDATE": int(g["QTL_TYPE"].astype(str).nunique()),
            "N_SHARED_QTL_TYPES": len(supported_types),
            "N_STRONG_QTL_TYPES": len(strong_types),
            "SUPPORTED_QTL_TYPES": ";".join(supported_types),
            "STRONG_QTL_TYPES": ";".join(strong_types),
            "SHARED_SIGNAL_TISSUES": join_unique(shared.get("TISSUE", pd.Series(dtype=str)), limit=100),
            "N_SHARED_SIGNAL_TISSUES": int(shared["TISSUE"].astype(str).nunique()) if not shared.empty else 0,
            "ANY_STEP11_PROBLEM": "YES" if g["RESULT_STATE"].isin(["FAILED","MISSING","PARSE_ERROR","STALE","UNKNOWN"]).any() else "NO",
        }
        row.update(modality_flat)

        # Carry one-to-one locus annotation columns into each locus-gene row.
        for c in g.columns:
            if c.startswith("ANNOT_") or c.startswith("TOP_PIP_") or c.startswith("TOP_SPLICEAI_") or c.startswith("TOP_PANGOLIN_"):
                vals = g[c].dropna()
                row[c] = vals.iloc[0] if not vals.empty else np.nan

        row["EVIDENCE_TIER_STEP12"] = evidence_tier(
            best_h4,
            prior_robust_any,
            len(supported_types),
            len(strong_types),
            strong,
            suggestive,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["BEST_H4", "N_SHARED_QTL_TYPES", "N_TESTED"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return out


def build_locus_master(long: pd.DataFrame, gene_master: pd.DataFrame, strong: float, suggestive: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (study, locus), g in long.groupby(["STUDY_ACCESSION", "LOCUS_ID"], dropna=False, sort=True):
        gm = gene_master[
            gene_master["STUDY_ACCESSION"].astype(str).eq(str(study))
            & gene_master["LOCUS_ID"].astype(str).eq(str(locus))
        ].copy()

        tested = g["FORMAL_COLOC_TESTED"].eq("YES")
        h4 = pd.to_numeric(g["PP_H4"], errors="coerce")
        eligible = g[tested & h4.notna()].copy()
        best = eligible.loc[pd.to_numeric(eligible["PP_H4"], errors="coerce").idxmax()] if not eligible.empty else g.iloc[0]
        best_h4 = finite_float(best.get("PP_H4")) if not eligible.empty else np.nan

        row: dict[str, Any] = {
            "PHENOTYPE": first_text(g.iloc[0], ["PHENOTYPE"]),
            "ANCESTRY_CODE": first_text(g.iloc[0], ["ANCESTRY_CODE"]),
            "ANCESTRY_LABEL": first_text(g.iloc[0], ["ANCESTRY_LABEL"]),
            "STUDY_ACCESSION": clean_str(study),
            "LOCUS_ID": clean_str(locus),
            "N_CANDIDATES": int(len(g)),
            "N_TESTED": int(tested.sum()),
            "N_COMPLETE": int(g["RESULT_STATE"].eq("COMPLETE").sum()),
            "N_NOT_TESTED": int(g["RESULT_STATE"].eq("NOT_TESTED").sum()),
            "N_FAILED": int(g["RESULT_STATE"].eq("FAILED").sum()),
            "N_MISSING": int(g["RESULT_STATE"].eq("MISSING").sum()),
            "N_PARSE_ERROR": int(g["RESULT_STATE"].eq("PARSE_ERROR").sum()),
            "N_STRONG": int((tested & h4.ge(strong)).sum()),
            "N_SUGGESTIVE": int((tested & h4.ge(suggestive) & h4.lt(strong)).sum()),
            "BEST_H4": best_h4,
            "BEST_QTL_TYPE": clean_str(best.get("QTL_TYPE")),
            "BEST_TISSUE": clean_str(best.get("TISSUE")),
            "BEST_DATASET_ID": clean_str(best.get("DATASET_ID")),
            "BEST_MOLECULAR_TRAIT_ID": clean_str(best.get("MOLECULAR_TRAIT_ID")),
            "BEST_EFFECTOR_KEY": clean_str(best.get("EFFECTOR_KEY")),
            "BEST_EFFECTOR_GENE_ID": clean_str(best.get("EFFECTOR_GENE_ID")),
            "BEST_EFFECTOR_GENE_SYMBOL": clean_str(best.get("EFFECTOR_GENE_SYMBOL")),
            "BEST_SIGNAL_PAIR": clean_str(best.get("BEST_SIGNAL_PAIR_STEP12")),
            "PRIOR_ROBUST_STRONG": "YES" if g["PRIOR_ROBUST_STRONG_STEP12"].eq("YES").any() else "NO",
            "N_EFFECTORS": int(gm["EFFECTOR_KEY"].astype(str).nunique()) if not gm.empty else 0,
            "ALL_EFFECTORS": join_unique(gm.get("EFFECTOR_KEY", pd.Series(dtype=str)), limit=200),
            "STRONG_EFFECTORS": join_unique(
                gm.loc[pd.to_numeric(gm.get("BEST_H4", pd.Series(np.nan, index=gm.index)), errors="coerce").ge(strong), "EFFECTOR_KEY"]
                if not gm.empty else [],
                limit=200,
            ),
            "SHARED_EFFECTORS": join_unique(
                gm.loc[pd.to_numeric(gm.get("BEST_H4", pd.Series(np.nan, index=gm.index)), errors="coerce").ge(suggestive), "EFFECTOR_KEY"]
                if not gm.empty else [],
                limit=200,
            ),
            "ANY_STEP11_PROBLEM": "YES" if g["RESULT_STATE"].isin(["FAILED","MISSING","PARSE_ERROR","STALE","UNKNOWN"]).any() else "NO",
        }

        blocks = [modality_summary(g, qt, strong, suggestive) for qt in QTL_TYPES]
        for block in blocks:
            row.update(block)

        supported_types = []
        strong_types = []
        for qt in QTL_TYPES:
            prefix = {"eQTL":"EQTL","sQTL":"SQTL","isoQTL":"ISOQTL","exonQTL":"EXONQTL","pQTL":"PQTL"}[qt]
            z = finite_float(row.get(f"{prefix}_BEST_H4"))
            if math.isfinite(z) and z >= suggestive:
                supported_types.append(qt)
            if math.isfinite(z) and z >= strong:
                strong_types.append(qt)
        row["N_SHARED_QTL_TYPES"] = len(supported_types)
        row["N_STRONG_QTL_TYPES"] = len(strong_types)
        row["SUPPORTED_QTL_TYPES"] = ";".join(supported_types)
        row["STRONG_QTL_TYPES"] = ";".join(strong_types)

        for c in g.columns:
            if c.startswith("ANNOT_") or c.startswith("TOP_PIP_") or c.startswith("TOP_SPLICEAI_") or c.startswith("TOP_PANGOLIN_"):
                vals = g[c].dropna()
                row[c] = vals.iloc[0] if not vals.empty else np.nan

        row["EVIDENCE_TIER_STEP12"] = evidence_tier(
            best_h4,
            row["PRIOR_ROBUST_STRONG"] == "YES",
            len(supported_types),
            len(strong_types),
            strong,
            suggestive,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["BEST_H4", "N_SHARED_QTL_TYPES", "N_TESTED"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = arguments()
    if not 0 <= args.suggestive_pp4 <= 1 or not 0 <= args.strong_pp4 <= 1:
        raise SystemExit("PP4 thresholds must be between 0 and 1")
    if args.suggestive_pp4 > args.strong_pp4:
        raise SystemExit("--suggestive-pp4 must be <= --strong-pp4")

    root = Path(args.root).resolve()
    phenotype = args.phenotype.strip()
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    paths = pipeline_paths(root, phenotype, ancestry_label)
    out = paths["S12"]
    out.mkdir(parents=True, exist_ok=True)

    banner("GWAS2m STEP12 - MASTER TABLE BUILDER")
    print(f"Root       : {root}")
    print(f"Phenotype  : {phenotype}")
    print(f"Ancestry   : {ancestry_label} ({ancestry_code})")
    print(f"Step11     : {paths['S11']}")
    print(f"Output     : {out}")
    print(f"Strong H4  : {args.strong_pp4}")
    print(f"Suggestive : {args.suggestive_pp4}")

    long = load_step11_long(
        paths["S11"], phenotype, ancestry_code, ancestry_label,
        args.strong_pp4, args.suggestive_pp4,
    )

    annotation_name, annotation_files = discover_annotation_files(paths, args.annotation_source)
    variants = load_variant_annotations(annotation_files, annotation_name)
    if variants.empty:
        banner("STEP12 - ANNOTATION NOTE")
        print("No cumulative Step10/Step07 variant annotation table was found.")
        print("Step12 will still build all Step11 master tables; annotation columns will be absent.")
        print(f"Step10 searched: {paths['S10']}")
        print(f"Step07 searched: {paths['S07']}")
    else:
        variants.insert(0, "PHENOTYPE_STEP12", phenotype)
        variants.insert(1, "ANCESTRY_CODE_STEP12", ancestry_code)
        variants.insert(2, "ANCESTRY_LABEL_STEP12", ancestry_label)

    locus_annot = build_locus_annotation_summary(variants)
    long = annotate_effector_symbols(long, variants)
    long = merge_locus_annotations(long, locus_annot)

    banner("STEP12 - BUILDING COLLAPSED TABLES")
    gene_master = build_locus_gene_master(long, args.strong_pp4, args.suggestive_pp4)
    locus_master = build_locus_master(long, gene_master, args.strong_pp4, args.suggestive_pp4)

    # Write all master outputs atomically enough for normal HPC use (same filesystem).
    long_file = out / "Step12_Molecular_Evidence_Long.tsv"
    gene_file = out / "Step12_Locus_Gene_Master.tsv"
    locus_file = out / "Step12_Locus_Master.tsv"
    variant_file = out / "Step12_Variant_Master.tsv"
    locus_annot_file = out / "Step12_Locus_Annotation.tsv"
    strong_file = out / "Step12_Strong_Shared_Signals.tsv"
    suggestive_file = out / "Step12_Suggestive_Or_Strong_Shared_Signals.tsv"
    problems_file = out / "Step12_QC_Problems.tsv"
    not_tested_file = out / "Step12_Not_Tested.tsv"
    unresolved_file = out / "Step12_Unresolved_Effectors.tsv"
    status_file = out / "Step12_Status_Counts.tsv"
    qtl_file = out / "Step12_QTL_Type_Summary.tsv"

    long.to_csv(long_file, sep="\t", index=False)
    gene_master.to_csv(gene_file, sep="\t", index=False)
    locus_master.to_csv(locus_file, sep="\t", index=False)
    if not variants.empty:
        variants.to_csv(variant_file, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(variant_file, sep="\t", index=False)
    locus_annot.to_csv(locus_annot_file, sep="\t", index=False)

    long.loc[long["IS_STRONG"]].to_csv(strong_file, sep="\t", index=False)
    long.loc[long["IS_SHARED_SIGNAL"]].to_csv(suggestive_file, sep="\t", index=False)
    problem_states = ["FAILED", "MISSING", "PARSE_ERROR", "STALE", "UNKNOWN"]
    problems = long.loc[long["RESULT_STATE"].isin(problem_states)].copy()
    problems.to_csv(problems_file, sep="\t", index=False)
    long.loc[long["RESULT_STATE"].eq("NOT_TESTED")].to_csv(not_tested_file, sep="\t", index=False)
    long.loc[long["EFFECTOR_KEY"].eq("UNRESOLVED_EFFECTOR")].to_csv(unresolved_file, sep="\t", index=False)

    status_counts = (
        long.groupby(["RESULT_STATE", "STATUS"], dropna=False)
        .size()
        .reset_index(name="COUNT")
        .sort_values("COUNT", ascending=False)
    )
    status_counts.to_csv(status_file, sep="\t", index=False)

    qtl_rows = []
    for qt in QTL_TYPES:
        g = long[long["QTL_TYPE"].astype(str).str.lower().eq(qt.lower())]
        tested = g["FORMAL_COLOC_TESTED"].eq("YES")
        h4 = pd.to_numeric(g["PP_H4"], errors="coerce")
        qtl_rows.append({
            "QTL_TYPE": qt,
            "N_CANDIDATES": int(len(g)),
            "N_TESTED": int(tested.sum()),
            "N_STRONG": int((tested & h4.ge(args.strong_pp4)).sum()),
            "N_SUGGESTIVE": int((tested & h4.ge(args.suggestive_pp4) & h4.lt(args.strong_pp4)).sum()),
            "BEST_H4": float(h4.max()) if h4.notna().any() else np.nan,
        })
    pd.DataFrame(qtl_rows).to_csv(qtl_file, sep="\t", index=False)

    summary = {
        "STEP12_VERSION": VERSION,
        "CREATED_UTC": utcnow(),
        "PHENOTYPE": phenotype,
        "ANCESTRY_CODE": ancestry_code,
        "ANCESTRY_LABEL": ancestry_label,
        "STRONG_PP4": args.strong_pp4,
        "SUGGESTIVE_PP4": args.suggestive_pp4,
        "ANNOTATION_SOURCE": annotation_name,
        "N_ANNOTATION_FILES": len(annotation_files),
        "N_VARIANTS": int(len(variants)),
        "N_STEP11_CANDIDATES": int(len(long)),
        "N_FORMALLY_TESTED": int(long["FORMAL_COLOC_TESTED"].eq("YES").sum()),
        "N_COMPLETE": int(long["RESULT_STATE"].eq("COMPLETE").sum()),
        "N_NOT_TESTED": int(long["RESULT_STATE"].eq("NOT_TESTED").sum()),
        "N_PROBLEMS": int(len(problems)),
        "N_STRONG": int(long["IS_STRONG"].sum()),
        "N_SUGGESTIVE": int(long["IS_SUGGESTIVE"].sum()),
        "N_LOCUS_GENE_ROWS": int(len(gene_master)),
        "N_LOCI": int(len(locus_master)),
        "RESULT_STATE_COUNTS": {str(k): int(v) for k, v in Counter(long["RESULT_STATE"].astype(str)).items()},
        "OUTPUTS": {
            "MOLECULAR_EVIDENCE_LONG": str(long_file),
            "LOCUS_GENE_MASTER": str(gene_file),
            "LOCUS_MASTER": str(locus_file),
            "VARIANT_MASTER": str(variant_file),
            "LOCUS_ANNOTATION": str(locus_annot_file),
            "STRONG_SIGNALS": str(strong_file),
            "SHARED_SIGNALS": str(suggestive_file),
            "QC_PROBLEMS": str(problems_file),
            "NOT_TESTED": str(not_tested_file),
            "STATUS_COUNTS": str(status_file),
            "QTL_TYPE_SUMMARY": str(qtl_file),
        },
    }
    safe_write_json(out / "Step12_summary.json", summary)

    banner("STEP12 COMPLETE")
    print(f"Step11 candidates       : {len(long):,}")
    print(f"Formally tested         : {summary['N_FORMALLY_TESTED']:,}")
    print(f"Strong H4               : {summary['N_STRONG']:,}")
    print(f"Suggestive H4           : {summary['N_SUGGESTIVE']:,}")
    print(f"QC/problem rows         : {summary['N_PROBLEMS']:,}")
    print(f"Locus x effector rows   : {len(gene_master):,}")
    print(f"Study x locus rows      : {len(locus_master):,}")
    print(f"Variant annotation rows : {len(variants):,}")
    print(f"Annotation source       : {annotation_name}")
    print()
    print("MAIN OUTPUTS")
    print(f"  {long_file}")
    print(f"  {gene_file}")
    print(f"  {locus_file}")
    print(f"  {variant_file}")
    print(f"  {locus_annot_file}")
    print(f"  {strong_file}")
    print(f"  {problems_file}")
    print()

    if len(problems):
        print("IMPORTANT: Step12 preserved unresolved Step11 problems; they are NOT biological negatives.")
        print(f"Inspect: {problems_file}")

    if args.fail_on_step11_problems and len(problems):
        raise SystemExit(
            f"Step12 tables were written, but {len(problems)} Step11 problem rows remain. "
            "Fix/re-run Step11, then rerun Step12."
        )


if __name__ == "__main__":
    main()

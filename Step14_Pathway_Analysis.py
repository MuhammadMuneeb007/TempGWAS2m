#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GWAS2m STEP 14
GENE PRIORITISATION + PATHWAY ENRICHMENT + OPEN TARGETS PATHWAY COMPARISON
================================================================================

Purpose
-------
Step14 converts the large Step12 locus x gene x QTL/tissue evidence into
DEDUPLICATED gene-level evidence, then performs pathway enrichment.

Important scientific rule
-------------------------
A gene appearing in 20 tissues is still ONE gene in pathway enrichment.
Tissue/QTL repetition is retained as SUPPORTING EVIDENCE and summary counts,
but never duplicated in the enrichment gene list.

Primary pathway sets
--------------------
CORE_STRONG:
    gene MAX_H4 >= --strong-h4 (default 0.80)

CORE_STRONG_PRIOR_ROBUST:
    MAX_H4 >= 0.80 and prior-robust evidence present

CORE_STRONG_MULTI_QTL:
    MAX_H4 >= 0.80 and >=2 supported QTL modalities

EXPANDED_SUGGESTIVE:
    MAX_H4 >= --suggestive-h4 (default 0.50)

Modality-specific:
    STRONG_eQTL / STRONG_sQTL / STRONG_isoQTL / STRONG_exonQTL / STRONG_pQTL

Open Targets comparison
-----------------------
If Step13_Gene_Comparison.tsv is available:
    - creates OT_TOP1 genes from matched OT loci
    - labels GWAS2m strong genes as OT-supported / OT-discordant / coverage-gap
    - enriches GWAS2m and OT gene sets with the SAME pathway backend
    - compares significant pathways (common, GWAS2m-only, OT-only)

Backends
--------
1) g:Profiler (online, default if no GMT supplied)
   pip install gprofiler-official

2) Offline GMT hypergeometric enrichment
   python Step14_Pathway_Analysis.py ... --gmt Reactome.gmt --gmt GO_BP.gmt

Run
---
python Step14_Pathway_Analysis.py \
    --phenotype "parkinson's disease" \
    --ancestry EUR

Offline GMT example:
python Step14_Pathway_Analysis.py \
    --phenotype "parkinson's disease" \
    --ancestry EUR \
    --gmt resources/pathways/Reactome.gmt \
    --gmt resources/pathways/GO_Biological_Process.gmt

Outputs
-------
14_pathways/<phenotype>/<ancestry>/
    Step14_Gene_Ranking.tsv
    Step14_Gene_Tissue_Evidence.tsv
    Step14_Gene_Modality_Evidence.tsv
    Step14_Gene_Sets.tsv
    Step14_Pathway_Results.tsv
    Step14_OpenTargets_Pathway_Comparison.tsv
    Step14_Pathway_Summary.tsv
    Step14_summary.json
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom
except Exception:
    hypergeom = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

VERSION = "1.0.0"

ANCESTRY = {
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

QTL_TYPES = ["eQTL", "sQTL", "isoQTL", "exonQTL", "pQTL"]


def norm(x) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(x).lower())).strip()


def slug(x) -> str:
    return norm(x).replace(" ", "_")


def ancestry_info(x):
    k = norm(x)
    if k not in ANCESTRY:
        raise SystemExit(f"Unsupported ancestry {x!r}; use EUR/AFR/EAS/SAS/AMR.")
    return ANCESTRY[k]


def section(title):
    print()
    print("=" * 138)
    print(title)
    print("=" * 138)


def first_existing(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for x in candidates:
        if x in columns:
            return x
        if x.lower() in lower:
            return lower[x.lower()]
    return None


def num(s):
    return pd.to_numeric(s, errors="coerce")


def boolish(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def clean_gene_id(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    x = str(v).strip()
    if not x or x.lower() in {"nan", "none", "na", ".", "null"}:
        return ""
    m = re.search(r"(ENSG\d+)", x, re.I)
    return m.group(1).upper() if m else x


def clean_symbol(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    x = str(v).strip()
    if x.lower() in {"", "nan", "none", "na", ".", "null", "-"}:
        return ""
    return x


def join_unique(values):
    out = []
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        for token in re.split(r"[;,|]", str(value)):
            token = token.strip()
            if token and token.lower() not in {"nan", "none", "na"} and token not in out:
                out.append(token)
    return ";".join(out)


def safe_max(series):
    x = num(series)
    return x.max() if x.notna().any() else np.nan


def safe_min(series):
    x = num(series)
    return x.min() if x.notna().any() else np.nan


def qtl_supported_columns(df):
    ans = {}
    for q in QTL_TYPES:
        col = first_existing(df.columns, [
            f"{q.upper()}_SUPPORTED",
            f"{q}_SUPPORTED",
        ])
        if col:
            ans[q] = col
    return ans


def load_step12(step12_file: Path):
    if not step12_file.exists():
        raise SystemExit(f"Missing Step12 master table:\n  {step12_file}")
    x = pd.read_csv(step12_file, sep="\t", low_memory=False)
    required = ["STUDY_ACCESSION", "LOCUS_ID", "EFFECTOR_GENE_ID", "BEST_H4"]
    miss = [c for c in required if c not in x.columns]
    if miss:
        raise SystemExit(f"Step12 master table missing required columns: {miss}")
    x["EFFECTOR_GENE_ID"] = x["EFFECTOR_GENE_ID"].map(clean_gene_id)
    if "EFFECTOR_GENE_SYMBOL" not in x.columns:
        x["EFFECTOR_GENE_SYMBOL"] = ""
    x["EFFECTOR_GENE_SYMBOL"] = x["EFFECTOR_GENE_SYMBOL"].map(clean_symbol)
    x["BEST_H4"] = num(x["BEST_H4"])
    x = x[x["EFFECTOR_GENE_ID"].astype(str).str.len() > 0].copy()
    return x


def gene_level_summary(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Tissue evidence: one gene x tissue, retaining max H4.
    tissue_col = first_existing(x.columns, ["BEST_TISSUE", "TISSUE"])
    qtl_col = first_existing(x.columns, ["BEST_QTL_TYPE", "QTL_TYPE"])

    tissue_rows = []
    if tissue_col:
        for (gid, tissue), g in x.groupby(["EFFECTOR_GENE_ID", tissue_col], dropna=False):
            tissue_rows.append({
                "EFFECTOR_GENE_ID": gid,
                "EFFECTOR_GENE_SYMBOL": join_unique(g["EFFECTOR_GENE_SYMBOL"]),
                "TISSUE": tissue,
                "MAX_H4": safe_max(g["BEST_H4"]),
                "N_ROWS": len(g),
                "QTL_TYPES": join_unique(g[qtl_col]) if qtl_col else "",
                "N_STUDIES": g["STUDY_ACCESSION"].astype(str).nunique(),
                "N_LOCI": g[["STUDY_ACCESSION", "LOCUS_ID"]].drop_duplicates().shape[0],
            })
    tissue_df = pd.DataFrame(tissue_rows)

    # Modality evidence: one gene x QTL type.
    modality_rows = []
    if qtl_col:
        for (gid, qtl), g in x.groupby(["EFFECTOR_GENE_ID", qtl_col], dropna=False):
            modality_rows.append({
                "EFFECTOR_GENE_ID": gid,
                "EFFECTOR_GENE_SYMBOL": join_unique(g["EFFECTOR_GENE_SYMBOL"]),
                "QTL_TYPE": qtl,
                "MAX_H4": safe_max(g["BEST_H4"]),
                "N_ROWS": len(g),
                "TISSUES": join_unique(g[tissue_col]) if tissue_col else "",
                "N_TISSUES": g[tissue_col].astype(str).nunique() if tissue_col else 0,
                "N_STUDIES": g["STUDY_ACCESSION"].astype(str).nunique(),
                "N_LOCI": g[["STUDY_ACCESSION", "LOCUS_ID"]].drop_duplicates().shape[0],
            })
    modality_df = pd.DataFrame(modality_rows)

    prior_col = first_existing(x.columns, ["PRIOR_ROBUST_STRONG", "PRIOR_ROBUST"])
    supported_cols = qtl_supported_columns(x)

    rows = []
    for gid, g in x.groupby("EFFECTOR_GENE_ID", sort=False):
        symbols = [clean_symbol(v) for v in g["EFFECTOR_GENE_SYMBOL"]]
        symbols = [v for v in symbols if v]
        symbol = symbols[0] if symbols else ""

        qtypes = []
        if qtl_col:
            qtypes.extend([str(v) for v in g[qtl_col].dropna() if str(v).strip()])

        # Also honor modality support flags from collapsed Step12.
        for q, c in supported_cols.items():
            if g[c].map(boolish).any():
                qtypes.append(q)
        qtypes = sorted(set(qtypes))

        n_loci = g[["STUDY_ACCESSION", "LOCUS_ID"]].drop_duplicates().shape[0]
        n_studies = g["STUDY_ACCESSION"].astype(str).nunique()
        n_tissues = g[tissue_col].astype(str).replace("nan", "").loc[lambda s: s.str.len() > 0].nunique() if tissue_col else 0

        max_h4 = safe_max(g["BEST_H4"])
        n_strong_rows = int((num(g["BEST_H4"]) >= 0.80).sum())
        n_suggestive_rows = int((num(g["BEST_H4"]) >= 0.50).sum())
        prior_robust = bool(g[prior_col].map(boolish).any()) if prior_col else False

        # Transparent heuristic ranking score. OT evidence is deliberately NOT included,
        # so Open Targets remains an independent benchmark.
        evidence_score = (
            (0.0 if pd.isna(max_h4) else float(max_h4))
            + 0.15 * int(prior_robust)
            + 0.08 * max(0, min(len(qtypes) - 1, 4))
            + 0.06 * math.log1p(max(0, n_loci - 1))
            + 0.04 * math.log1p(max(0, n_studies - 1))
        )

        rows.append({
            "EFFECTOR_GENE_ID": gid,
            "EFFECTOR_GENE_SYMBOL": symbol,
            "MAX_H4": max_h4,
            "PRIOR_ROBUST": "YES" if prior_robust else "NO",
            "N_LOCUS_GENE_ROWS": len(g),
            "N_INDEPENDENT_LOCUS_STUDY_PAIRS": n_loci,
            "N_STUDIES": n_studies,
            "N_TISSUES": n_tissues,
            "N_QTL_TYPES": len(qtypes),
            "QTL_TYPES": ";".join(qtypes),
            "N_STRONG_ROWS": n_strong_rows,
            "N_SUGGESTIVE_OR_STRONG_ROWS": n_suggestive_rows,
            "TISSUES": join_unique(g[tissue_col]) if tissue_col else "",
            "STUDIES": join_unique(g["STUDY_ACCESSION"]),
            "LOCI": join_unique(g["LOCUS_ID"]),
            "EVIDENCE_SCORE_STEP14": evidence_score,
        })

    genes = pd.DataFrame(rows)
    genes = genes.sort_values(
        ["MAX_H4", "PRIOR_ROBUST", "N_QTL_TYPES", "N_INDEPENDENT_LOCUS_STUDY_PAIRS", "EVIDENCE_SCORE_STEP14"],
        ascending=[False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    genes["RANK_STEP14"] = np.arange(1, len(genes) + 1)
    return genes, tissue_df, modality_df


def attach_step13(genes: pd.DataFrame, step13_file: Path):
    genes = genes.copy()
    genes["OT_ANY_LOCUS_MATCH"] = "UNKNOWN"
    genes["OT_L2G_SUPPORTED_ANY"] = "UNKNOWN"
    genes["OT_L2G_BEST_RANK"] = np.nan
    genes["OT_L2G_MAX_SCORE"] = np.nan
    genes["OT_EVIDENCE_CLASSES"] = ""

    if not step13_file.exists():
        return genes, pd.DataFrame()

    s = pd.read_csv(step13_file, sep="\t", low_memory=False)
    gid_col = first_existing(s.columns, ["EFFECTOR_GENE_ID"])
    if not gid_col:
        return genes, s
    s[gid_col] = s[gid_col].map(clean_gene_id)

    ot_match_col = first_existing(s.columns, ["OT_LOCUS_MATCH"])
    l2g_present_col = first_existing(s.columns, ["OT_L2G_GENE_PRESENT"])
    rank_col = first_existing(s.columns, ["OT_L2G_RANK"])
    score_col = first_existing(s.columns, ["OT_L2G_SCORE"])
    class_col = first_existing(s.columns, ["OPEN_TARGETS_EVIDENCE_CLASS"])

    agg_rows = []
    for gid, g in s.groupby(gid_col):
        agg_rows.append({
            "EFFECTOR_GENE_ID": gid,
            "OT_ANY_LOCUS_MATCH": (
                "YES" if ot_match_col and g[ot_match_col].map(boolish).any()
                else "NO"
            ),
            "OT_L2G_SUPPORTED_ANY": (
                "YES" if l2g_present_col and g[l2g_present_col].map(boolish).any()
                else "NO"
            ),
            "OT_L2G_BEST_RANK": (
                num(g[rank_col]).min() if rank_col and num(g[rank_col]).notna().any() else np.nan
            ),
            "OT_L2G_MAX_SCORE": (
                num(g[score_col]).max() if score_col and num(g[score_col]).notna().any() else np.nan
            ),
            "OT_EVIDENCE_CLASSES": join_unique(g[class_col]) if class_col else "",
        })
    a = pd.DataFrame(agg_rows)
    if not a.empty:
        genes = genes.drop(columns=[
            "OT_ANY_LOCUS_MATCH", "OT_L2G_SUPPORTED_ANY",
            "OT_L2G_BEST_RANK", "OT_L2G_MAX_SCORE", "OT_EVIDENCE_CLASSES"
        ], errors="ignore").merge(a, on="EFFECTOR_GENE_ID", how="left")
    return genes, s


def build_gene_sets(genes: pd.DataFrame, modality_df: pd.DataFrame, strong_h4: float, suggestive_h4: float):
    sets = {}

    sets["CORE_STRONG"] = genes.loc[genes["MAX_H4"] >= strong_h4, "EFFECTOR_GENE_ID"].tolist()
    sets["CORE_STRONG_PRIOR_ROBUST"] = genes.loc[
        (genes["MAX_H4"] >= strong_h4) & (genes["PRIOR_ROBUST"] == "YES"),
        "EFFECTOR_GENE_ID"
    ].tolist()
    sets["CORE_STRONG_MULTI_QTL"] = genes.loc[
        (genes["MAX_H4"] >= strong_h4) & (genes["N_QTL_TYPES"] >= 2),
        "EFFECTOR_GENE_ID"
    ].tolist()
    sets["EXPANDED_SUGGESTIVE"] = genes.loc[
        genes["MAX_H4"] >= suggestive_h4,
        "EFFECTOR_GENE_ID"
    ].tolist()

    # Open Targets-stratified sets, if available.
    if "OT_L2G_SUPPORTED_ANY" in genes.columns:
        sets["CORE_STRONG_OT_SUPPORTED"] = genes.loc[
            (genes["MAX_H4"] >= strong_h4) & (genes["OT_L2G_SUPPORTED_ANY"] == "YES"),
            "EFFECTOR_GENE_ID"
        ].tolist()
        sets["CORE_STRONG_OT_DISCORDANT"] = genes.loc[
            (genes["MAX_H4"] >= strong_h4) &
            (genes["OT_ANY_LOCUS_MATCH"] == "YES") &
            (genes["OT_L2G_SUPPORTED_ANY"] == "NO"),
            "EFFECTOR_GENE_ID"
        ].tolist()
        sets["CORE_STRONG_OT_COVERAGE_GAP"] = genes.loc[
            (genes["MAX_H4"] >= strong_h4) &
            (genes["OT_ANY_LOCUS_MATCH"] == "NO"),
            "EFFECTOR_GENE_ID"
        ].tolist()

    if not modality_df.empty:
        for q in QTL_TYPES:
            z = modality_df[
                (modality_df["QTL_TYPE"].astype(str).str.lower() == q.lower()) &
                (num(modality_df["MAX_H4"]) >= strong_h4)
            ]
            sets[f"STRONG_{q}"] = sorted(set(z["EFFECTOR_GENE_ID"].map(clean_gene_id)) - {""})

    # Remove duplicates and empties.
    clean = {}
    for k, vals in sets.items():
        clean[k] = sorted({clean_gene_id(v) for v in vals if clean_gene_id(v)})
    return clean


def ot_top1_gene_set(step13: pd.DataFrame):
    if step13 is None or step13.empty:
        return []
    col = first_existing(step13.columns, ["OT_L2G_TOP1_GENE"])
    if not col:
        return []
    return sorted({clean_gene_id(v) for v in step13[col] if clean_gene_id(v)})


def gene_set_table(sets):
    rows = []
    for name, genes in sets.items():
        for g in genes:
            rows.append({"GENE_SET": name, "GENE_ID": g})
    return pd.DataFrame(rows)


def parse_gmt(path: Path):
    pathways = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = {x.strip() for x in parts[2:] if x.strip()}
            if name and genes:
                pathways[name] = genes
    return pathways


def offline_enrichment(query_genes, background_genes, gmt_files, gene_set_name, fdr_cutoff):
    if hypergeom is None or multipletests is None:
        raise RuntimeError(
            "Offline GMT enrichment requires scipy and statsmodels.\n"
            "Install: python -m pip install scipy statsmodels"
        )

    query = set(query_genes) & set(background_genes)
    universe = set(background_genes)
    M = len(universe)
    N = len(query)
    if M == 0 or N == 0:
        return pd.DataFrame()

    rows = []
    for gmt in gmt_files:
        library = Path(gmt).stem
        pathways = parse_gmt(Path(gmt))
        for term, members in pathways.items():
            members = members & universe
            K = len(members)
            overlap = query & members
            k = len(overlap)
            if k == 0 or K == 0:
                continue
            p = float(hypergeom.sf(k - 1, M, K, N))
            rows.append({
                "GENE_SET": gene_set_name,
                "SOURCE": library,
                "TERM_NAME": term,
                "QUERY_SIZE": N,
                "TERM_SIZE_IN_BACKGROUND": K,
                "INTERSECTION_SIZE": k,
                "INTERSECTION_GENES": ";".join(sorted(overlap)),
                "P_VALUE": p,
                "BACKEND": "OFFLINE_GMT_HYPERGEOMETRIC",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["FDR"] = multipletests(out["P_VALUE"].values, method="fdr_bh")[1]
    out["SIGNIFICANT"] = out["FDR"] <= fdr_cutoff
    return out.sort_values(["FDR", "P_VALUE", "INTERSECTION_SIZE"], ascending=[True, True, False])


def gprofiler_enrichment(query_genes, background_genes, gene_set_name, fdr_cutoff, organism="hsapiens"):
    try:
        from gprofiler import GProfiler
    except ImportError as exc:
        raise RuntimeError(
            "g:Profiler backend requires gprofiler-official.\n"
            "Install: python -m pip install gprofiler-official"
        ) from exc

    if not query_genes:
        return pd.DataFrame()

    gp = GProfiler(return_dataframe=True)
    # g:Profiler supports ENSG IDs directly; custom background controls ascertainment.
    res = gp.profile(
        organism=organism,
        query=list(query_genes),
        background=list(background_genes) if background_genes else None,
        sources=["GO:BP", "REAC", "KEGG", "WP"],
        user_threshold=fdr_cutoff,
        significance_threshold_method="fdr",
        no_evidences=False,
    )
    if res is None or len(res) == 0:
        return pd.DataFrame()

    out = pd.DataFrame({
        "GENE_SET": gene_set_name,
        "SOURCE": res.get("source", ""),
        "TERM_ID": res.get("native", ""),
        "TERM_NAME": res.get("name", ""),
        "QUERY_SIZE": res.get("query_size", np.nan),
        "TERM_SIZE_IN_BACKGROUND": res.get("term_size", np.nan),
        "INTERSECTION_SIZE": res.get("intersection_size", np.nan),
        "INTERSECTION_GENES": res.get("intersections", pd.Series([[]] * len(res))).map(
            lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple, set, np.ndarray)) else str(x)
        ),
        "P_VALUE": res.get("p_value", np.nan),
        "FDR": res.get("p_value", np.nan),  # profile already applies selected correction
        "SIGNIFICANT": True,
        "BACKEND": "GPROFILER",
    })
    return out.sort_values(["FDR", "P_VALUE"], ascending=[True, True])


def compare_pathways(results: pd.DataFrame, left_set="CORE_STRONG", right_set="OPEN_TARGETS_TOP1", fdr_cutoff=0.05):
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    sig = results[(num(results["FDR"]) <= fdr_cutoff) & (results["SIGNIFICANT"].map(boolish))].copy()
    left = sig[sig["GENE_SET"] == left_set].copy()
    right = sig[sig["GENE_SET"] == right_set].copy()

    def key(df):
        if df.empty:
            return set()
        return set(zip(df["SOURCE"].astype(str), df["TERM_NAME"].astype(str)))

    a = key(left)
    b = key(right)
    union = a | b
    inter = a & b
    rows = []
    for source, term in sorted(union):
        rows.append({
            "SOURCE": source,
            "TERM_NAME": term,
            "GWAS2M_SIGNIFICANT": (source, term) in a,
            "OPEN_TARGETS_SIGNIFICANT": (source, term) in b,
            "PATHWAY_CLASS": (
                "COMMON" if (source, term) in inter
                else "GWAS2M_ONLY" if (source, term) in a
                else "OPEN_TARGETS_ONLY"
            ),
        })

    summary = pd.DataFrame([{
        "GWAS2M_SIGNIFICANT_PATHWAYS": len(a),
        "OPEN_TARGETS_SIGNIFICANT_PATHWAYS": len(b),
        "COMMON_SIGNIFICANT_PATHWAYS": len(inter),
        "UNION_SIGNIFICANT_PATHWAYS": len(union),
        "PATHWAY_JACCARD": len(inter) / len(union) if union else np.nan,
    }])
    return pd.DataFrame(rows), summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phenotype", required=True)
    p.add_argument("--ancestry", required=True)
    p.add_argument("--strong-h4", type=float, default=0.80)
    p.add_argument("--suggestive-h4", type=float, default=0.50)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument(
        "--gmt", action="append", default=[],
        help="Offline GMT file. Repeat --gmt for multiple pathway libraries."
    )
    p.add_argument(
        "--skip-enrichment", action="store_true",
        help="Build ranked gene sets only; do not run pathway enrichment."
    )
    p.add_argument(
        "--step12-file", default="",
        help="Override Step12_Locus_Gene_Master.tsv path."
    )
    p.add_argument(
        "--step13-file", default="",
        help="Override Step13_Gene_Comparison.tsv path."
    )
    args = p.parse_args()

    root = Path.cwd().resolve()
    acode, alabel = ancestry_info(args.ancestry)
    ps = slug(args.phenotype)
    a = slug(alabel)

    step12 = (
        Path(args.step12_file).resolve() if args.step12_file
        else root / "12_master_table" / ps / a / "Step12_Locus_Gene_Master.tsv"
    )
    step13 = (
        Path(args.step13_file).resolve() if args.step13_file
        else root / "13_opentargets_comparison" / ps / a / "Step13_Gene_Comparison.tsv"
    )
    out = root / "14_pathways" / ps / a
    out.mkdir(parents=True, exist_ok=True)

    section("GWAS2m STEP14 - GENE PRIORITISATION + PATHWAY ANALYSIS")
    print(f"Version       : {VERSION}")
    print(f"Phenotype     : {args.phenotype}")
    print(f"Ancestry      : {alabel} ({acode})")
    print(f"Step12        : {step12}")
    print(f"Step13        : {step13 if step13.exists() else 'NOT AVAILABLE'}")
    print(f"Output        : {out}")
    print(f"Strong H4     : >= {args.strong_h4}")
    print(f"Suggestive H4 : >= {args.suggestive_h4}")

    x = load_step12(step12)
    genes, tissues, modalities = gene_level_summary(x)
    genes, step13_df = attach_step13(genes, step13)

    genes.to_csv(out / "Step14_Gene_Ranking.tsv", sep="\t", index=False)
    tissues.to_csv(out / "Step14_Gene_Tissue_Evidence.tsv", sep="\t", index=False)
    modalities.to_csv(out / "Step14_Gene_Modality_Evidence.tsv", sep="\t", index=False)

    gene_sets = build_gene_sets(genes, modalities, args.strong_h4, args.suggestive_h4)
    ot_top1 = ot_top1_gene_set(step13_df)
    if ot_top1:
        gene_sets["OPEN_TARGETS_TOP1"] = ot_top1

    gene_set_table(gene_sets).to_csv(out / "Step14_Gene_Sets.tsv", sep="\t", index=False)

    section("DEDUPLICATED GENE SETS")
    summary_rows = []
    for name, vals in gene_sets.items():
        summary_rows.append({"GENE_SET": name, "N_UNIQUE_GENES": len(vals)})
    summary_df = pd.DataFrame(summary_rows).sort_values("N_UNIQUE_GENES", ascending=False)
    print(summary_df.to_string(index=False))

    strong = genes[genes["MAX_H4"] >= args.strong_h4].copy()
    section("CORE STRONG GENES")
    show_cols = [
        "RANK_STEP14", "EFFECTOR_GENE_ID", "EFFECTOR_GENE_SYMBOL",
        "MAX_H4", "PRIOR_ROBUST", "N_QTL_TYPES", "QTL_TYPES",
        "N_TISSUES", "N_INDEPENDENT_LOCUS_STUDY_PAIRS",
        "OT_L2G_SUPPORTED_ANY", "OT_L2G_BEST_RANK",
        "EVIDENCE_SCORE_STEP14"
    ]
    show_cols = [c for c in show_cols if c in strong.columns]
    print(strong[show_cols].to_string(index=False))

    pathway_results = []
    if not args.skip_enrichment:
        # Background: every unique gene that reached Step12 locus-gene master.
        # This controls for genes eligible to be prioritized by this pipeline.
        background = genes["EFFECTOR_GENE_ID"].dropna().astype(str).tolist()

        selected_sets = [
            "CORE_STRONG",
            "CORE_STRONG_PRIOR_ROBUST",
            "CORE_STRONG_MULTI_QTL",
            "EXPANDED_SUGGESTIVE",
            "CORE_STRONG_OT_SUPPORTED",
            "CORE_STRONG_OT_DISCORDANT",
            "OPEN_TARGETS_TOP1",
        ]
        selected_sets += [f"STRONG_{q}" for q in QTL_TYPES]

        section("PATHWAY ENRICHMENT")
        backend = "OFFLINE_GMT" if args.gmt else "GPROFILER"
        print(f"Backend       : {backend}")
        print(f"Background    : {len(set(background))} unique Step12 genes")
        print(f"FDR threshold : {args.fdr}")

        for set_name in selected_sets:
            q = gene_sets.get(set_name, [])
            if len(q) < 2:
                print(f"[SKIP] {set_name}: only {len(q)} gene(s)")
                continue
            print(f"[RUN ] {set_name}: {len(q)} unique genes")
            if args.gmt:
                z = offline_enrichment(q, background, args.gmt, set_name, args.fdr)
            else:
                z = gprofiler_enrichment(q, background, set_name, args.fdr)
            if not z.empty:
                pathway_results.append(z)

    pathways = pd.concat(pathway_results, ignore_index=True) if pathway_results else pd.DataFrame()
    pathways.to_csv(out / "Step14_Pathway_Results.tsv", sep="\t", index=False)

    comparison, pathway_summary = compare_pathways(
        pathways, "CORE_STRONG", "OPEN_TARGETS_TOP1", args.fdr
    )
    comparison.to_csv(out / "Step14_OpenTargets_Pathway_Comparison.tsv", sep="\t", index=False)
    pathway_summary.to_csv(out / "Step14_Pathway_Summary.tsv", sep="\t", index=False)

    if not pathways.empty:
        section("TOP SIGNIFICANT PATHWAYS - CORE_STRONG")
        z = pathways[
            (pathways["GENE_SET"] == "CORE_STRONG") &
            (num(pathways["FDR"]) <= args.fdr)
        ].copy()
        cols = [c for c in [
            "SOURCE", "TERM_NAME", "FDR", "INTERSECTION_SIZE", "INTERSECTION_GENES"
        ] if c in z.columns]
        if z.empty:
            print("(no significant pathways at selected FDR)")
        else:
            print(z[cols].head(30).to_string(index=False))

    if not pathway_summary.empty:
        section("GWAS2m vs OPEN TARGETS PATHWAY COMPARISON")
        print(pathway_summary.to_string(index=False))

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phenotype": args.phenotype,
        "ancestry_code": acode,
        "ancestry_label": alabel,
        "step12_file": str(step12),
        "step13_file": str(step13) if step13.exists() else None,
        "n_step12_locus_gene_rows": int(len(x)),
        "n_unique_step12_genes": int(len(genes)),
        "n_core_strong_genes": int(len(gene_sets.get("CORE_STRONG", []))),
        "n_expanded_suggestive_genes": int(len(gene_sets.get("EXPANDED_SUGGESTIVE", []))),
        "gene_sets": {k: len(v) for k, v in gene_sets.items()},
        "strong_h4": args.strong_h4,
        "suggestive_h4": args.suggestive_h4,
        "fdr": args.fdr,
        "backend": "offline_gmt" if args.gmt else ("skipped" if args.skip_enrichment else "gprofiler"),
        "gmt_files": [str(Path(x).resolve()) for x in args.gmt],
        "scientific_note": (
            "Pathway enrichment uses unique genes, not repeated tissue/QTL rows. "
            "Open Targets is not included in the GWAS2m evidence score, preserving "
            "its role as an independent external comparator."
        )
    }
    (out / "Step14_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    section("STEP14 COMPLETE")
    print(f"Unique Step12 genes       : {len(genes)}")
    print(f"Core strong genes         : {len(gene_sets.get('CORE_STRONG', []))}")
    print(f"Expanded suggestive genes : {len(gene_sets.get('EXPANDED_SUGGESTIVE', []))}")
    print()
    print("Main ranked genes:")
    print(f"  {out / 'Step14_Gene_Ranking.tsv'}")
    print("Gene sets:")
    print(f"  {out / 'Step14_Gene_Sets.tsv'}")
    print("Pathway results:")
    print(f"  {out / 'Step14_Pathway_Results.tsv'}")
    print("GWAS2m vs OT pathways:")
    print(f"  {out / 'Step14_OpenTargets_Pathway_Comparison.tsv'}")
    print()
    print("Next: review enriched pathways, then Step15 should build PPI/network modules")
    print("from the CORE_STRONG / CORE_STRONG_PRIOR_ROBUST genes plus known disease genes.")


if __name__ == "__main__":
    main()

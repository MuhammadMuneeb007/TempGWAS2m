#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GWAS2m STEP 14 v2.0
PUBLICATION-GRADE GENE PRIORITISATION + PATHWAY ENRICHMENT + OT COMPARISON
================================================================================

WHY THIS STEP EXISTS
--------------------
Step12/Step13 contain many rows per biological gene because the same gene may
appear in:
    * multiple GWAS studies
    * multiple loci/signals
    * multiple tissues
    * multiple QTL modalities
    * multiple molecular traits

Those repeated rows MUST NOT be used as repeated pathway votes.

Step14 therefore performs:

    Step12 locus-gene evidence
        -> deduplicated gene evidence
        -> transparent evidence ranking
        -> biologically defined gene sets
        -> pathway enrichment
        -> pathway redundancy/driver-gene audit
        -> Open Targets pathway comparison
        -> complete terminal report

SCIENTIFIC PRINCIPLES
---------------------
1. One gene = one pathway-enrichment vote.
2. Tissue/QTL recurrence is evidence about a gene, not extra gene counts.
3. The main set is CORE_STRONG: MAX_H4 >= 0.80.
4. EXPANDED_SUGGESTIVE (MAX_H4 >= 0.50) is a sensitivity analysis.
5. Open Targets evidence is NOT used to rank GWAS2m genes. It remains an
   independent external comparator.
6. Pathways are considered significant after FDR correction.
7. A pathway needs at least --min-overlap distinct input genes (default 2).
8. Very tiny / extremely broad terms can be filtered using term-size limits.
9. An enrichment result is not proof of pathway causality.
10. A GWAS2m-only pathway is not automatically a novel pathway.

DEFAULT INPUTS
--------------
12_master_table/<phenotype>/<ancestry>/Step12_Locus_Gene_Master.tsv
13_opentargets_comparison/<phenotype>/<ancestry>/Step13_Gene_Comparison.tsv

DEFAULT OUTPUTS
---------------
14_pathways/<phenotype>/<ancestry>/

    Step14_Gene_Ranking.tsv
    Step14_Gene_Tissue_Evidence.tsv
    Step14_Gene_Modality_Evidence.tsv
    Step14_Locus_Top_Genes.tsv
    Step14_Gene_Set_Membership.tsv

    Step14_Pathway_All_Results.tsv
    Step14_Pathway_Significant.tsv
    Step14_Pathway_Set_Summary.tsv
    Step14_Pathway_Gene_Drivers.tsv
    Step14_Pathway_Redundancy.tsv

    Step14_OpenTargets_Pathway_Comparison.tsv
    Step14_OpenTargets_Pathway_Comparison_Summary.tsv

    Step14_QC_Audit.tsv
    Step14_summary.json

INSTALL
-------
Online enrichment:
    python -m pip install gprofiler-official pandas numpy scipy statsmodels

Offline GMT enrichment:
    python -m pip install pandas numpy scipy statsmodels

RUN
---
Normal:
    python Step14_Pathway_Analysis.py \
      --phenotype "parkinson's disease" \
      --ancestry EUR

Build/rank gene sets only:
    python Step14_Pathway_Analysis.py \
      --phenotype "parkinson's disease" \
      --ancestry EUR \
      --skip-enrichment

Offline GMT:
    python Step14_Pathway_Analysis.py \
      --phenotype "parkinson's disease" \
      --ancestry EUR \
      --gmt resources/pathways/Reactome.gmt \
      --gmt resources/pathways/GO_BP.gmt

Less terminal output:
    ... --compact-screen

================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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


VERSION = "2.0.0"

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

DEFAULT_GPROFILER_SOURCES = [
    "GO:BP",
    "REAC",
    "KEGG",
    "WP",
]


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def norm(value) -> str:
    x = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", x).strip()


def slug(value) -> str:
    return norm(value).replace(" ", "_")


def ancestry_info(value):
    key = norm(value)
    if key not in ANCESTRY:
        raise SystemExit(
            f"Unsupported ancestry {value!r}. Use EUR, AFR, EAS, SAS or AMR."
        )
    return ANCESTRY[key]


def section(title: str, char: str = "="):
    print()
    print(char * 150)
    print(title)
    print(char * 150)


def subsection(title: str):
    section(title, "-")


def show(
    df: pd.DataFrame | None,
    title: str | None = None,
    *,
    max_rows: int | None = None,
    columns: list[str] | None = None,
):
    if title:
        section(title)

    if df is None or df.empty:
        print("(no rows)")
        return

    x = df.copy()

    if columns:
        columns = [c for c in columns if c in x.columns]
        x = x[columns]

    if max_rows is not None:
        x = x.head(max_rows)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1000,
        "display.max_colwidth", 140,
        "display.expand_frame_repr", False,
        "display.float_format", lambda v: f"{v:.6g}",
    ):
        print(x.to_string(index=False))

    if max_rows is not None and len(x) < len(df):
        print(f"\n[showing {len(x):,}/{len(df):,} rows]")


def first_existing(columns: Iterable[str], candidates: Iterable[str]):
    columns = list(columns)
    lower = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate in columns:
            return candidate

        found = lower.get(str(candidate).lower())
        if found is not None:
            return found

    return None


def safe_numeric(values):
    return pd.to_numeric(values, errors="coerce")


def boolish(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    return str(value).strip().lower() in {
        "1", "true", "t", "yes", "y",
    }


def clean_gene_id(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    x = str(value).strip()

    if x.lower() in {"", "nan", "none", "null", "na", "."}:
        return ""

    m = re.search(r"(ENSG\d+)", x, flags=re.I)

    return m.group(1).upper() if m else x


def clean_symbol(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    x = str(value).strip()

    if x.lower() in {"", "nan", "none", "null", "na", ".", "-"}:
        return ""

    return x


def split_tokens(value):
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    return [
        x.strip()
        for x in re.split(r"[;,|]", str(value))
        if x.strip()
        and x.strip().lower() not in {"nan", "none", "null", "na"}
    ]


def join_unique(values) -> str:
    output = []

    for value in values:
        for token in split_tokens(value):
            if token not in output:
                output.append(token)

    return ";".join(output)


def safe_max(values):
    x = safe_numeric(values)
    return x.max() if x.notna().any() else np.nan


def safe_min(values):
    x = safe_numeric(values)
    return x.min() if x.notna().any() else np.nan


def safe_median(values):
    x = safe_numeric(values)
    return x.median() if x.notna().any() else np.nan


def bh_fdr(pvalues):
    p = safe_numeric(pd.Series(pvalues))

    out = pd.Series(np.nan, index=p.index, dtype=float)

    good = p.notna()

    if not good.any():
        return out

    if multipletests is None:
        # Manual Benjamini-Hochberg fallback.
        vals = p[good].to_numpy(dtype=float)
        order = np.argsort(vals)
        ranked = vals[order]
        n = len(ranked)
        adjusted = ranked * n / np.arange(1, n + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(n)
        out.loc[good] = adjusted[inverse]
        return out

    out.loc[good] = multipletests(
        p[good].to_numpy(dtype=float),
        method="fdr_bh",
    )[1]

    return out


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return np.nan

    union = a | b

    return len(a & b) / len(union) if union else np.nan


def pair_key(study, locus):
    return f"{study}|{locus}"


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_step12(path: Path):
    if not path.exists():
        raise SystemExit(
            "Step12 locus-gene master is missing:\n"
            f"  {path}"
        )

    x = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    required = [
        "STUDY_ACCESSION",
        "LOCUS_ID",
        "EFFECTOR_GENE_ID",
        "BEST_H4",
    ]

    missing = [
        col
        for col in required
        if col not in x.columns
    ]

    if missing:
        raise SystemExit(
            "Step12 locus-gene master is missing required columns:\n"
            f"  {missing}"
        )

    x["EFFECTOR_GENE_ID"] = (
        x["EFFECTOR_GENE_ID"]
        .map(clean_gene_id)
    )

    if "EFFECTOR_GENE_SYMBOL" not in x.columns:
        x["EFFECTOR_GENE_SYMBOL"] = ""

    x["EFFECTOR_GENE_SYMBOL"] = (
        x["EFFECTOR_GENE_SYMBOL"]
        .map(clean_symbol)
    )

    x["BEST_H4"] = safe_numeric(
        x["BEST_H4"]
    )

    x["_LOCUS_STUDY_KEY"] = [
        pair_key(study, locus)
        for study, locus
        in zip(
            x["STUDY_ACCESSION"],
            x["LOCUS_ID"],
        )
    ]

    x = x[
        x["EFFECTOR_GENE_ID"]
        .astype(str)
        .str.len()
        >
        0
    ].copy()

    return x


def load_step13(path: Path):
    if not path.exists():
        return pd.DataFrame()

    try:
        x = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()

    if "EFFECTOR_GENE_ID" in x.columns:
        x["EFFECTOR_GENE_ID"] = (
            x["EFFECTOR_GENE_ID"]
            .map(clean_gene_id)
        )

    return x


# =============================================================================
# STEP12 -> DEDUPLICATED GENE EVIDENCE
# =============================================================================

def qtl_type_column(df):
    return first_existing(
        df.columns,
        [
            "BEST_QTL_TYPE",
            "QTL_TYPE",
        ],
    )


def tissue_column(df):
    return first_existing(
        df.columns,
        [
            "BEST_TISSUE",
            "TISSUE",
        ],
    )


def prior_robust_column(df):
    return first_existing(
        df.columns,
        [
            "PRIOR_ROBUST_STRONG",
            "PRIOR_ROBUST",
        ],
    )


def supported_modality_columns(df):
    result = {}

    for qtl in QTL_TYPES:
        candidates = [
            f"{qtl.upper()}_SUPPORTED",
            f"{qtl}_SUPPORTED",
        ]

        found = first_existing(
            df.columns,
            candidates,
        )

        if found:
            result[qtl] = found

    return result


def make_symbol_lookup(step12):
    counts = defaultdict(lambda: defaultdict(int))

    for row in step12.itertuples(index=False):
        gid = clean_gene_id(
            getattr(
                row,
                "EFFECTOR_GENE_ID",
                "",
            )
        )

        symbol = clean_symbol(
            getattr(
                row,
                "EFFECTOR_GENE_SYMBOL",
                "",
            )
        )

        if gid and symbol:
            counts[gid][symbol] += 1

    result = {}

    for gid, options in counts.items():
        result[gid] = sorted(
            options.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[0][0]

    return result


def build_tissue_evidence(step12):
    tcol = tissue_column(step12)
    qcol = qtl_type_column(step12)

    if not tcol:
        return pd.DataFrame()

    rows = []

    for (
        gid,
        tissue,
    ), group in step12.groupby(
        [
            "EFFECTOR_GENE_ID",
            tcol,
        ],
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "EFFECTOR_GENE_ID":
                    clean_gene_id(gid),

                "EFFECTOR_GENE_SYMBOL":
                    join_unique(
                        group[
                            "EFFECTOR_GENE_SYMBOL"
                        ]
                    ),

                "TISSUE":
                    str(tissue),

                "MAX_H4":
                    safe_max(
                        group[
                            "BEST_H4"
                        ]
                    ),

                "MEDIAN_H4":
                    safe_median(
                        group[
                            "BEST_H4"
                        ]
                    ),

                "N_LOCUS_GENE_ROWS":
                    len(group),

                "N_STUDIES":
                    group[
                        "STUDY_ACCESSION"
                    ]
                    .astype(str)
                    .nunique(),

                "N_LOCUS_STUDY_PAIRS":
                    group[
                        "_LOCUS_STUDY_KEY"
                    ]
                    .astype(str)
                    .nunique(),

                "QTL_TYPES":
                    (
                        join_unique(
                            group[qcol]
                        )
                        if qcol
                        else
                        ""
                    ),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            [
                "EFFECTOR_GENE_ID",
                "MAX_H4",
            ],
            ascending=[
                True,
                False,
            ],
        )

    return out


def build_modality_evidence(step12):
    qcol = qtl_type_column(step12)
    tcol = tissue_column(step12)

    if not qcol:
        return pd.DataFrame()

    rows = []

    for (
        gid,
        qtl,
    ), group in step12.groupby(
        [
            "EFFECTOR_GENE_ID",
            qcol,
        ],
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "EFFECTOR_GENE_ID":
                    clean_gene_id(gid),

                "EFFECTOR_GENE_SYMBOL":
                    join_unique(
                        group[
                            "EFFECTOR_GENE_SYMBOL"
                        ]
                    ),

                "QTL_TYPE":
                    str(qtl),

                "MAX_H4":
                    safe_max(
                        group[
                            "BEST_H4"
                        ]
                    ),

                "MEDIAN_H4":
                    safe_median(
                        group[
                            "BEST_H4"
                        ]
                    ),

                "N_LOCUS_GENE_ROWS":
                    len(group),

                "N_STUDIES":
                    group[
                        "STUDY_ACCESSION"
                    ]
                    .astype(str)
                    .nunique(),

                "N_LOCUS_STUDY_PAIRS":
                    group[
                        "_LOCUS_STUDY_KEY"
                    ]
                    .astype(str)
                    .nunique(),

                "N_TISSUES":
                    (
                        group[tcol]
                        .astype(str)
                        .replace(
                            {
                                "nan": "",
                                "None": "",
                            }
                        )
                        .loc[
                            lambda s:
                            s.str.len() > 0
                        ]
                        .nunique()
                        if tcol
                        else
                        0
                    ),

                "TISSUES":
                    (
                        join_unique(
                            group[tcol]
                        )
                        if tcol
                        else
                        ""
                    ),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            [
                "EFFECTOR_GENE_ID",
                "MAX_H4",
            ],
            ascending=[
                True,
                False,
            ],
        )

    return out


def build_gene_ranking(step12):
    qcol = qtl_type_column(step12)
    tcol = tissue_column(step12)
    prior_col = prior_robust_column(step12)
    supported_cols = supported_modality_columns(step12)
    symbol_lookup = make_symbol_lookup(step12)

    rows = []

    for gid, group in step12.groupby(
        "EFFECTOR_GENE_ID",
        sort=False,
    ):
        gid = clean_gene_id(gid)

        qtls = set()

        if qcol:
            for value in group[qcol]:
                for token in split_tokens(value):
                    qtls.add(token)

        for qtl, col in supported_cols.items():
            if group[col].map(boolish).any():
                qtls.add(qtl)

        tissues = (
            sorted(
                {
                    str(v).strip()
                    for v in group[tcol]
                    if str(v).strip()
                    and str(v).lower()
                    not in {
                        "nan",
                        "none",
                    }
                }
            )
            if tcol
            else
            []
        )

        max_h4 = safe_max(
            group[
                "BEST_H4"
            ]
        )

        median_h4 = safe_median(
            group[
                "BEST_H4"
            ]
        )

        prior_robust = (
            bool(
                group[
                    prior_col
                ]
                .map(boolish)
                .any()
            )
            if prior_col
            else
            False
        )

        n_locus_study = (
            group[
                "_LOCUS_STUDY_KEY"
            ]
            .astype(str)
            .nunique()
        )

        n_studies = (
            group[
                "STUDY_ACCESSION"
            ]
            .astype(str)
            .nunique()
        )

        n_qtl = len(qtls)

        # Transparent evidence ranking.
        #
        # MAX_H4 dominates. Small bonuses reward robustness / replication /
        # orthogonal QTL modalities. Tissue count is deliberately NOT used as
        # a score bonus to avoid rewarding ubiquitous genes merely because
        # more tissues were available/tested.
        score = (
            (
                0.0
                if pd.isna(max_h4)
                else float(max_h4)
            )
            +
            0.15
            *
            int(prior_robust)
            +
            0.08
            *
            max(
                0,
                min(
                    n_qtl - 1,
                    4,
                ),
            )
            +
            0.06
            *
            math.log1p(
                max(
                    0,
                    n_locus_study - 1,
                )
            )
            +
            0.04
            *
            math.log1p(
                max(
                    0,
                    n_studies - 1,
                )
            )
        )

        rows.append(
            {
                "EFFECTOR_GENE_ID":
                    gid,

                "EFFECTOR_GENE_SYMBOL":
                    symbol_lookup.get(
                        gid,
                        "",
                    ),

                "MAX_H4":
                    max_h4,

                "MEDIAN_H4":
                    median_h4,

                "PRIOR_ROBUST":
                    (
                        "YES"
                        if prior_robust
                        else
                        "NO"
                    ),

                "N_QTL_TYPES":
                    n_qtl,

                "QTL_TYPES":
                    ";".join(
                        sorted(qtls)
                    ),

                "N_TISSUES":
                    len(tissues),

                "TISSUES":
                    ";".join(tissues),

                "N_STUDIES":
                    n_studies,

                "STUDIES":
                    join_unique(
                        group[
                            "STUDY_ACCESSION"
                        ]
                    ),

                "N_INDEPENDENT_LOCUS_STUDY_PAIRS":
                    n_locus_study,

                "LOCUS_STUDY_PAIRS":
                    join_unique(
                        group[
                            "_LOCUS_STUDY_KEY"
                        ]
                    ),

                "N_LOCUS_GENE_ROWS":
                    len(group),

                "N_H4_GE_0_80_ROWS":
                    int(
                        (
                            safe_numeric(
                                group[
                                    "BEST_H4"
                                ]
                            )
                            >=
                            0.80
                        )
                        .sum()
                    ),

                "N_H4_GE_0_50_ROWS":
                    int(
                        (
                            safe_numeric(
                                group[
                                    "BEST_H4"
                                ]
                            )
                            >=
                            0.50
                        )
                        .sum()
                    ),

                "EVIDENCE_SCORE_STEP14":
                    score,
            }
        )

    genes = pd.DataFrame(rows)

    genes = genes.sort_values(
        [
            "MAX_H4",
            "PRIOR_ROBUST",
            "N_QTL_TYPES",
            "N_INDEPENDENT_LOCUS_STUDY_PAIRS",
            "EVIDENCE_SCORE_STEP14",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    genes[
        "RANK_STEP14"
    ] = np.arange(
        1,
        len(genes) + 1,
    )

    return genes


def build_locus_top_genes(step12):
    rows = []

    for (
        study,
        locus,
    ), group in step12.groupby(
        [
            "STUDY_ACCESSION",
            "LOCUS_ID",
        ],
        sort=True,
    ):
        z = group.copy()

        z["_H4"] = safe_numeric(
            z[
                "BEST_H4"
            ]
        )

        z = z.sort_values(
            [
                "_H4",
            ],
            ascending=[
                False,
            ],
            kind="stable",
        )

        best = z.iloc[0]

        rows.append(
            {
                "STUDY_ACCESSION":
                    study,

                "LOCUS_ID":
                    locus,

                "TOP_GENE_ID":
                    clean_gene_id(
                        best[
                            "EFFECTOR_GENE_ID"
                        ]
                    ),

                "TOP_GENE_SYMBOL":
                    clean_symbol(
                        best.get(
                            "EFFECTOR_GENE_SYMBOL",
                            "",
                        )
                    ),

                "TOP_GENE_H4":
                    best[
                        "_H4"
                    ],

                "N_CANDIDATE_GENES":
                    group[
                        "EFFECTOR_GENE_ID"
                    ]
                    .astype(str)
                    .nunique(),

                "ALL_CANDIDATE_GENES":
                    join_unique(
                        group[
                            "EFFECTOR_GENE_ID"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# ATTACH OPEN TARGETS EVIDENCE WITHOUT USING IT IN GWAS2m SCORE
# =============================================================================

def attach_step13_evidence(
    genes,
    step13,
):
    out = genes.copy()

    defaults = {
        "OT_ANY_LOCUS_MATCH":
            "UNKNOWN",

        "OT_L2G_SUPPORTED_ANY":
            "UNKNOWN",

        "OT_L2G_BEST_RANK":
            np.nan,

        "OT_L2G_MAX_SCORE":
            np.nan,

        "OT_EVIDENCE_CLASSES":
            "",
    }

    for col, default in defaults.items():
        out[col] = default

    if step13.empty:
        return out

    gid_col = first_existing(
        step13.columns,
        [
            "EFFECTOR_GENE_ID",
        ],
    )

    if not gid_col:
        return out

    locus_match_col = first_existing(
        step13.columns,
        [
            "OT_LOCUS_MATCH",
        ],
    )

    present_col = first_existing(
        step13.columns,
        [
            "OT_L2G_GENE_PRESENT",
        ],
    )

    rank_col = first_existing(
        step13.columns,
        [
            "OT_L2G_RANK",
        ],
    )

    score_col = first_existing(
        step13.columns,
        [
            "OT_L2G_SCORE",
        ],
    )

    class_col = first_existing(
        step13.columns,
        [
            "OPEN_TARGETS_EVIDENCE_CLASS",
        ],
    )

    rows = []

    for gid, group in step13.groupby(
        gid_col,
        sort=False,
    ):
        gid = clean_gene_id(gid)

        if not gid:
            continue

        rows.append(
            {
                "EFFECTOR_GENE_ID":
                    gid,

                "OT_ANY_LOCUS_MATCH":
                    (
                        "YES"
                        if (
                            locus_match_col
                            and
                            group[
                                locus_match_col
                            ]
                            .map(boolish)
                            .any()
                        )
                        else
                        "NO"
                    ),

                "OT_L2G_SUPPORTED_ANY":
                    (
                        "YES"
                        if (
                            present_col
                            and
                            group[
                                present_col
                            ]
                            .map(boolish)
                            .any()
                        )
                        else
                        "NO"
                    ),

                "OT_L2G_BEST_RANK":
                    (
                        safe_numeric(
                            group[
                                rank_col
                            ]
                        ).min()
                        if (
                            rank_col
                            and
                            safe_numeric(
                                group[
                                    rank_col
                                ]
                            )
                            .notna()
                            .any()
                        )
                        else
                        np.nan
                    ),

                "OT_L2G_MAX_SCORE":
                    (
                        safe_numeric(
                            group[
                                score_col
                            ]
                        ).max()
                        if (
                            score_col
                            and
                            safe_numeric(
                                group[
                                    score_col
                                ]
                            )
                            .notna()
                            .any()
                        )
                        else
                        np.nan
                    ),

                "OT_EVIDENCE_CLASSES":
                    (
                        join_unique(
                            group[
                                class_col
                            ]
                        )
                        if class_col
                        else
                        ""
                    ),
            }
        )

    aggregate = pd.DataFrame(rows)

    if aggregate.empty:
        return out

    out = out.drop(
        columns=list(
            defaults
        ),
        errors="ignore",
    ).merge(
        aggregate,
        on="EFFECTOR_GENE_ID",
        how="left",
    )

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            if isinstance(default, str):
                out[col] = (
                    out[col]
                    .fillna(default)
                )

    return out


def extract_ot_top1(step13):
    if step13.empty:
        return []

    col = first_existing(
        step13.columns,
        [
            "OT_L2G_TOP1_GENE",
        ],
    )

    if not col:
        return []

    return sorted(
        {
            clean_gene_id(value)
            for value
            in step13[col]
            if clean_gene_id(value)
        }
    )


# =============================================================================
# GENE SETS
# =============================================================================

def modality_strong_genes(
    modality,
    threshold,
):
    result = {}

    if modality.empty:
        for qtl in QTL_TYPES:
            result[qtl] = []
        return result

    for qtl in QTL_TYPES:
        z = modality[
            (
                modality[
                    "QTL_TYPE"
                ]
                .astype(str)
                .str.lower()
                ==
                qtl.lower()
            )
            &
            (
                safe_numeric(
                    modality[
                        "MAX_H4"
                    ]
                )
                >=
                threshold
            )
        ]

        result[qtl] = sorted(
            {
                clean_gene_id(value)
                for value
                in z[
                    "EFFECTOR_GENE_ID"
                ]
                if clean_gene_id(value)
            }
        )

    return result


def build_gene_sets(
    genes,
    modality,
    locus_top,
    step13,
    strong_h4,
    suggestive_h4,
):
    result = {}

    result[
        "CORE_STRONG"
    ] = genes.loc[
        genes[
            "MAX_H4"
        ]
        >=
        strong_h4,
        "EFFECTOR_GENE_ID",
    ].tolist()

    result[
        "CORE_STRONG_PRIOR_ROBUST"
    ] = genes.loc[
        (
            genes[
                "MAX_H4"
            ]
            >=
            strong_h4
        )
        &
        (
            genes[
                "PRIOR_ROBUST"
            ]
            ==
            "YES"
        ),
        "EFFECTOR_GENE_ID",
    ].tolist()

    result[
        "CORE_STRONG_MULTI_QTL"
    ] = genes.loc[
        (
            genes[
                "MAX_H4"
            ]
            >=
            strong_h4
        )
        &
        (
            genes[
                "N_QTL_TYPES"
            ]
            >=
            2
        ),
        "EFFECTOR_GENE_ID",
    ].tolist()

    result[
        "EXPANDED_SUGGESTIVE"
    ] = genes.loc[
        genes[
            "MAX_H4"
        ]
        >=
        suggestive_h4,
        "EFFECTOR_GENE_ID",
    ].tolist()

    if (
        "OT_L2G_SUPPORTED_ANY"
        in
        genes.columns
    ):
        result[
            "CORE_STRONG_OT_SUPPORTED"
        ] = genes.loc[
            (
                genes[
                    "MAX_H4"
                ]
                >=
                strong_h4
            )
            &
            (
                genes[
                    "OT_L2G_SUPPORTED_ANY"
                ]
                ==
                "YES"
            ),
            "EFFECTOR_GENE_ID",
        ].tolist()

        result[
            "CORE_STRONG_OT_DISCORDANT"
        ] = genes.loc[
            (
                genes[
                    "MAX_H4"
                ]
                >=
                strong_h4
            )
            &
            (
                genes[
                    "OT_ANY_LOCUS_MATCH"
                ]
                ==
                "YES"
            )
            &
            (
                genes[
                    "OT_L2G_SUPPORTED_ANY"
                ]
                ==
                "NO"
            ),
            "EFFECTOR_GENE_ID",
        ].tolist()

        result[
            "CORE_STRONG_OT_COVERAGE_GAP"
        ] = genes.loc[
            (
                genes[
                    "MAX_H4"
                ]
                >=
                strong_h4
            )
            &
            (
                genes[
                    "OT_ANY_LOCUS_MATCH"
                ]
                ==
                "NO"
            ),
            "EFFECTOR_GENE_ID",
        ].tolist()

    # One top gene per study x locus, restricted to strong loci.
    if not locus_top.empty:
        result[
            "LOCUS_TOP_STRONG"
        ] = (
            locus_top.loc[
                safe_numeric(
                    locus_top[
                        "TOP_GENE_H4"
                    ]
                )
                >=
                strong_h4,
                "TOP_GENE_ID",
            ]
            .tolist()
        )

    modality_sets = modality_strong_genes(
        modality,
        strong_h4,
    )

    for qtl, values in modality_sets.items():
        result[
            f"STRONG_{qtl}"
        ] = values

    ot = extract_ot_top1(
        step13
    )

    if ot:
        result[
            "OPEN_TARGETS_TOP1"
        ] = ot

    # De-duplicate.
    clean = {}

    for name, values in result.items():
        clean[
            name
        ] = sorted(
            {
                clean_gene_id(v)
                for v
                in values
                if clean_gene_id(v)
            }
        )

    return clean


def gene_set_membership_table(
    gene_sets,
    genes,
):
    symbol_lookup = dict(
        zip(
            genes[
                "EFFECTOR_GENE_ID"
            ],
            genes[
                "EFFECTOR_GENE_SYMBOL"
            ],
        )
    )

    h4_lookup = dict(
        zip(
            genes[
                "EFFECTOR_GENE_ID"
            ],
            genes[
                "MAX_H4"
            ],
        )
    )

    rows = []

    for gene_set, members in gene_sets.items():
        for rank, gid in enumerate(
            members,
            start=1,
        ):
            rows.append(
                {
                    "GENE_SET":
                        gene_set,

                    "SET_MEMBER_INDEX":
                        rank,

                    "GENE_ID":
                        gid,

                    "GENE_SYMBOL":
                        symbol_lookup.get(
                            gid,
                            "",
                        ),

                    "GWAS2M_MAX_H4":
                        h4_lookup.get(
                            gid,
                            np.nan,
                        ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# BACKGROUND
# =============================================================================

def load_background(
    genes,
    background_file,
):
    if not background_file:
        return sorted(
            set(
                genes[
                    "EFFECTOR_GENE_ID"
                ]
                .dropna()
                .astype(str)
            )
        ), "STEP12_UNIQUE_GENES"

    path = Path(
        background_file
    ).resolve()

    if not path.exists():
        raise SystemExit(
            f"Background file does not exist: {path}"
        )

    if path.suffix.lower() in {
        ".tsv",
        ".csv",
        ".txt",
    }:
        sep = (
            "\t"
            if path.suffix.lower()
            in {
                ".tsv",
                ".txt",
            }
            else
            ","
        )

        x = pd.read_csv(
            path,
            sep=sep,
            low_memory=False,
        )

        col = first_existing(
            x.columns,
            [
                "GENE_ID",
                "EFFECTOR_GENE_ID",
                "gene",
                "gene_id",
                "ensembl_gene_id",
            ],
        )

        if col is None:
            # One-column plain text read may have been interpreted as header.
            values = [
                clean_gene_id(v)
                for v
                in path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
            ]
        else:
            values = [
                clean_gene_id(v)
                for v
                in x[col]
            ]

    else:
        values = [
            clean_gene_id(v)
            for v
            in path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        ]

    values = sorted(
        {
            v
            for v
            in values
            if v
        }
    )

    return values, str(path)


# =============================================================================
# GMT OFFLINE ENRICHMENT
# =============================================================================

def parse_gmt(path: Path):
    pathways = {}

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            parts = line.rstrip(
                "\n"
            ).split(
                "\t"
            )

            if len(parts) < 3:
                continue

            name = parts[0].strip()

            genes = {
                clean_gene_id(x)
                for x
                in parts[2:]
                if clean_gene_id(x)
            }

            if name and genes:
                pathways[
                    name
                ] = genes

    return pathways


def offline_enrichment(
    query,
    background,
    gmt_files,
    gene_set_name,
):
    if hypergeom is None:
        raise RuntimeError(
            "Offline enrichment requires scipy."
        )

    query = (
        set(query)
        &
        set(background)
    )

    universe = set(
        background
    )

    M = len(
        universe
    )

    N = len(
        query
    )

    if not M or not N:
        return pd.DataFrame()

    rows = []

    for gmt_path in gmt_files:
        gmt_path = Path(
            gmt_path
        ).resolve()

        library_name = gmt_path.stem

        pathways = parse_gmt(
            gmt_path
        )

        for term_name, members in pathways.items():
            members = (
                members
                &
                universe
            )

            K = len(
                members
            )

            overlap = (
                query
                &
                members
            )

            k = len(
                overlap
            )

            if not k or not K:
                continue

            pvalue = float(
                hypergeom.sf(
                    k - 1,
                    M,
                    K,
                    N,
                )
            )

            rows.append(
                {
                    "GENE_SET":
                        gene_set_name,

                    "SOURCE":
                        library_name,

                    "TERM_ID":
                        "",

                    "TERM_NAME":
                        term_name,

                    "P_VALUE":
                        pvalue,

                    "QUERY_SIZE":
                        N,

                    "TERM_SIZE":
                        K,

                    "INTERSECTION_SIZE":
                        k,

                    "INTERSECTION_GENES":
                        ";".join(
                            sorted(
                                overlap
                            )
                        ),

                    "BACKEND":
                        "OFFLINE_GMT",
                }
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result[
        "FDR"
    ] = bh_fdr(
        result[
            "P_VALUE"
        ]
    )

    return result


# =============================================================================
# G:PROFILER
# =============================================================================

def gprofiler_enrichment(
    query,
    background,
    gene_set_name,
    sources,
    organism,
):
    try:
        from gprofiler import GProfiler
    except ImportError as exc:
        raise RuntimeError(
            "g:Profiler backend requires gprofiler-official.\n"
            "Install with:\n"
            "  python -m pip install gprofiler-official"
        ) from exc

    query = sorted(
        set(query)
    )

    background = sorted(
        set(background)
    )

    if not query:
        return pd.DataFrame()

    gp = GProfiler(
        return_dataframe=True
    )

    result = gp.profile(
        organism=organism,
        query=query,
        background=background,
        sources=sources,
        user_threshold=1.0,
        significance_threshold_method="fdr",
        no_evidences=False,
    )

    if result is None or len(result) == 0:
        return pd.DataFrame()

    result = pd.DataFrame(
        result
    )

    rows = []

    for _, row in result.iterrows():
        intersections = row.get(
            "intersections",
            [],
        )

        if not isinstance(
            intersections,
            (
                list,
                tuple,
                set,
                np.ndarray,
            ),
        ):
            intersections = []

        rows.append(
            {
                "GENE_SET":
                    gene_set_name,

                "SOURCE":
                    row.get(
                        "source",
                        "",
                    ),

                "TERM_ID":
                    row.get(
                        "native",
                        "",
                    ),

                "TERM_NAME":
                    row.get(
                        "name",
                        "",
                    ),

                "P_VALUE":
                    row.get(
                        "p_value",
                        np.nan,
                    ),

                # g:Profiler p_value already reflects selected correction.
                "FDR":
                    row.get(
                        "p_value",
                        np.nan,
                    ),

                "QUERY_SIZE":
                    row.get(
                        "query_size",
                        len(query),
                    ),

                "TERM_SIZE":
                    row.get(
                        "term_size",
                        np.nan,
                    ),

                "INTERSECTION_SIZE":
                    row.get(
                        "intersection_size",
                        len(intersections),
                    ),

                "INTERSECTION_GENES":
                    ";".join(
                        map(
                            str,
                            intersections,
                        )
                    ),

                "PRECISION":
                    row.get(
                        "precision",
                        np.nan,
                    ),

                "RECALL":
                    row.get(
                        "recall",
                        np.nan,
                    ),

                "BACKEND":
                    "GPROFILER",
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PATHWAY FILTER / QC
# =============================================================================

def filter_pathway_results(
    results,
    fdr,
    min_overlap,
    min_term_size,
    max_term_size,
):
    if results.empty:
        return results

    out = results.copy()

    out[
        "FDR"
    ] = safe_numeric(
        out[
            "FDR"
        ]
    )

    out[
        "INTERSECTION_SIZE"
    ] = safe_numeric(
        out[
            "INTERSECTION_SIZE"
        ]
    )

    out[
        "TERM_SIZE"
    ] = safe_numeric(
        out[
            "TERM_SIZE"
        ]
    )

    out[
        "PASS_FDR"
    ] = (
        out[
            "FDR"
        ]
        <=
        fdr
    )

    out[
        "PASS_MIN_OVERLAP"
    ] = (
        out[
            "INTERSECTION_SIZE"
        ]
        >=
        min_overlap
    )

    out[
        "PASS_TERM_SIZE"
    ] = (
        out[
            "TERM_SIZE"
        ]
        .between(
            min_term_size,
            max_term_size,
            inclusive="both",
        )
        |
        out[
            "TERM_SIZE"
        ]
        .isna()
    )

    out[
        "SIGNIFICANT"
    ] = (
        out[
            "PASS_FDR"
        ]
        &
        out[
            "PASS_MIN_OVERLAP"
        ]
        &
        out[
            "PASS_TERM_SIZE"
        ]
    )

    return out


def add_gene_labels(
    pathways,
    genes,
):
    if pathways.empty:
        return pathways

    symbol_map = dict(
        zip(
            genes[
                "EFFECTOR_GENE_ID"
            ],
            genes[
                "EFFECTOR_GENE_SYMBOL"
            ],
        )
    )

    def labels(value):
        ids = split_tokens(
            value
        )

        out = []

        for gid in ids:
            gid = clean_gene_id(
                gid
            )

            symbol = clean_symbol(
                symbol_map.get(
                    gid,
                    "",
                )
            )

            out.append(
                (
                    f"{symbol} ({gid})"
                    if symbol
                    else
                    gid
                )
            )

        return "; ".join(
            out
        )

    pathways = pathways.copy()

    pathways[
        "INTERSECTION_GENE_LABELS"
    ] = pathways[
        "INTERSECTION_GENES"
    ].map(
        labels
    )

    return pathways


def pathway_set_summary(
    pathway_results,
    gene_sets,
):
    rows = []

    for gene_set, members in gene_sets.items():
        z = (
            pathway_results[
                pathway_results[
                    "GENE_SET"
                ]
                ==
                gene_set
            ].copy()
            if not pathway_results.empty
            else
            pd.DataFrame()
        )

        sig = (
            z[
                z[
                    "SIGNIFICANT"
                ]
                .map(boolish)
            ].copy()
            if (
                not z.empty
                and
                "SIGNIFICANT"
                in z.columns
            )
            else
            pd.DataFrame()
        )

        rows.append(
            {
                "GENE_SET":
                    gene_set,

                "N_UNIQUE_INPUT_GENES":
                    len(members),

                "N_TERMS_RETURNED":
                    len(z),

                "N_SIGNIFICANT_TERMS":
                    len(sig),

                "MIN_FDR":
                    (
                        safe_min(
                            z[
                                "FDR"
                            ]
                        )
                        if not z.empty
                        else
                        np.nan
                    ),

                "BEST_PATHWAY":
                    (
                        z.sort_values(
                            "FDR",
                            ascending=True,
                        )
                        .iloc[0][
                            "TERM_NAME"
                        ]
                        if (
                            not z.empty
                            and
                            safe_numeric(
                                z[
                                    "FDR"
                                ]
                            )
                            .notna()
                            .any()
                        )
                        else
                        ""
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PATHWAY DRIVER GENES
# =============================================================================

def explode_pathway_drivers(
    significant,
    genes,
):
    if significant.empty:
        return pd.DataFrame()

    gene_lookup = genes.set_index(
        "EFFECTOR_GENE_ID"
    )

    rows = []

    for _, row in significant.iterrows():
        for gid in split_tokens(
            row.get(
                "INTERSECTION_GENES",
                "",
            )
        ):
            gid = clean_gene_id(
                gid
            )

            gene_row = (
                gene_lookup.loc[
                    gid
                ]
                if gid
                in
                gene_lookup.index
                else
                None
            )

            rows.append(
                {
                    "GENE_SET":
                        row.get(
                            "GENE_SET",
                            "",
                        ),

                    "SOURCE":
                        row.get(
                            "SOURCE",
                            "",
                        ),

                    "TERM_ID":
                        row.get(
                            "TERM_ID",
                            "",
                        ),

                    "TERM_NAME":
                        row.get(
                            "TERM_NAME",
                            "",
                        ),

                    "FDR":
                        row.get(
                            "FDR",
                            np.nan,
                        ),

                    "GENE_ID":
                        gid,

                    "GENE_SYMBOL":
                        (
                            clean_symbol(
                                gene_row[
                                    "EFFECTOR_GENE_SYMBOL"
                                ]
                            )
                            if gene_row is not None
                            else
                            ""
                        ),

                    "GENE_MAX_H4":
                        (
                            gene_row[
                                "MAX_H4"
                            ]
                            if gene_row is not None
                            else
                            np.nan
                        ),

                    "GENE_QTL_TYPES":
                        (
                            gene_row[
                                "QTL_TYPES"
                            ]
                            if gene_row is not None
                            else
                            ""
                        ),

                    "GENE_N_TISSUES":
                        (
                            gene_row[
                                "N_TISSUES"
                            ]
                            if gene_row is not None
                            else
                            np.nan
                        ),

                    "GENE_OT_L2G_SUPPORTED":
                        (
                            gene_row.get(
                                "OT_L2G_SUPPORTED_ANY",
                                "",
                            )
                            if gene_row is not None
                            else
                            ""
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# SIMPLE PATHWAY REDUNDANCY AUDIT
# =============================================================================

def pathway_redundancy(
    significant,
    overlap_threshold,
):
    if significant.empty:
        return pd.DataFrame()

    rows = []

    for gene_set, group in significant.groupby(
        "GENE_SET",
        sort=False,
    ):
        z = group.sort_values(
            [
                "FDR",
                "INTERSECTION_SIZE",
            ],
            ascending=[
                True,
                False,
            ],
        ).reset_index(
            drop=True
        )

        clusters = []
        cluster_id = 0

        for i, row in z.iterrows():
            members = set(
                split_tokens(
                    row[
                        "INTERSECTION_GENES"
                    ]
                )
            )

            assigned = None

            best_j = 0.0
            best_rep = None

            for cluster in clusters:
                rep = cluster[
                    "rep_members"
                ]

                score = jaccard(
                    members,
                    rep,
                )

                if (
                    pd.notna(score)
                    and
                    score
                    >=
                    overlap_threshold
                    and
                    score
                    >
                    best_j
                ):
                    best_j = score
                    assigned = cluster
                    best_rep = cluster[
                        "rep_name"
                    ]

            if assigned is None:
                cluster_id += 1

                assigned = {
                    "cluster_id":
                        cluster_id,

                    "rep_name":
                        row[
                            "TERM_NAME"
                        ],

                    "rep_members":
                        members,
                }

                clusters.append(
                    assigned
                )

                best_rep = row[
                    "TERM_NAME"
                ]

            rows.append(
                {
                    "GENE_SET":
                        gene_set,

                    "REDUNDANCY_CLUSTER":
                        assigned[
                            "cluster_id"
                        ],

                    "IS_CLUSTER_REPRESENTATIVE":
                        (
                            row[
                                "TERM_NAME"
                            ]
                            ==
                            assigned[
                                "rep_name"
                            ]
                        ),

                    "REPRESENTATIVE_PATHWAY":
                        assigned[
                            "rep_name"
                        ],

                    "SOURCE":
                        row[
                            "SOURCE"
                        ],

                    "TERM_NAME":
                        row[
                            "TERM_NAME"
                        ],

                    "FDR":
                        row[
                            "FDR"
                        ],

                    "INTERSECTION_SIZE":
                        row[
                            "INTERSECTION_SIZE"
                        ],

                    "INTERSECTION_GENES":
                        row[
                            "INTERSECTION_GENES"
                        ],

                    "JACCARD_TO_REPRESENTATIVE":
                        (
                            1.0
                            if row[
                                "TERM_NAME"
                            ]
                            ==
                            assigned[
                                "rep_name"
                            ]
                            else
                            best_j
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# GWAS2m vs OPEN TARGETS PATHWAYS
# =============================================================================

def compare_pathway_sets(
    significant,
    left_name,
    right_name,
):
    if significant.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    left = significant[
        significant[
            "GENE_SET"
        ]
        ==
        left_name
    ]

    right = significant[
        significant[
            "GENE_SET"
        ]
        ==
        right_name
    ]

    def keys(df):
        return {
            (
                str(row.SOURCE),
                str(row.TERM_NAME),
            )
            for row
            in df.itertuples()
        }

    a = keys(
        left
    )

    b = keys(
        right
    )

    union = (
        a
        |
        b
    )

    common = (
        a
        &
        b
    )

    rows = []

    for source, name in sorted(
        union
    ):
        rows.append(
            {
                "SOURCE":
                    source,

                "TERM_NAME":
                    name,

                "GWAS2M_SIGNIFICANT":
                    (
                        source,
                        name,
                    )
                    in
                    a,

                "OPEN_TARGETS_SIGNIFICANT":
                    (
                        source,
                        name,
                    )
                    in
                    b,

                "PATHWAY_CLASS":
                    (
                        "COMMON"
                        if (
                            source,
                            name,
                        )
                        in
                        common
                        else
                        "GWAS2M_ONLY"
                        if (
                            source,
                            name,
                        )
                        in
                        a
                        else
                        "OPEN_TARGETS_ONLY"
                    ),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "GWAS2M_GENE_SET":
                    left_name,

                "OPEN_TARGETS_GENE_SET":
                    right_name,

                "GWAS2M_SIGNIFICANT_PATHWAYS":
                    len(a),

                "OPEN_TARGETS_SIGNIFICANT_PATHWAYS":
                    len(b),

                "COMMON_SIGNIFICANT_PATHWAYS":
                    len(common),

                "UNION_SIGNIFICANT_PATHWAYS":
                    len(union),

                "PATHWAY_JACCARD":
                    (
                        len(common)
                        /
                        len(union)
                        if union
                        else
                        np.nan
                    ),
            }
        ]
    )

    return (
        pd.DataFrame(rows),
        summary,
    )


# =============================================================================
# SCREEN REPORT
# =============================================================================

def gene_label(
    gid,
    genes,
):
    z = genes[
        genes[
            "EFFECTOR_GENE_ID"
        ]
        ==
        gid
    ]

    if z.empty:
        return gid

    symbol = clean_symbol(
        z.iloc[0][
            "EFFECTOR_GENE_SYMBOL"
        ]
    )

    h4 = z.iloc[0][
        "MAX_H4"
    ]

    label = (
        f"{symbol} ({gid})"
        if symbol
        else
        gid
    )

    if pd.notna(h4):
        label += (
            f" [H4={float(h4):.6f}]"
        )

    return label


def print_pipeline_collapse(
    step12,
    genes,
    gene_sets,
):
    section(
        "WHAT HAPPENED TO THE LARGE MASTER TABLE?"
    )

    print(
        f"Step12 locus x gene rows                       : {len(step12):,}"
    )

    print(
        f"Unique genes after deduplication               : {len(genes):,}"
    )

    print(
        f"Unique strong genes (MAX_H4 >= 0.80)           : {len(gene_sets.get('CORE_STRONG', [])):,}"
    )

    print(
        f"Unique suggestive+ genes (MAX_H4 >= 0.50)      : {len(gene_sets.get('EXPANDED_SUGGESTIVE', [])):,}"
    )

    print(
        f"Strong + prior robust genes                    : {len(gene_sets.get('CORE_STRONG_PRIOR_ROBUST', [])):,}"
    )

    print(
        f"Strong + >=2 QTL modalities                    : {len(gene_sets.get('CORE_STRONG_MULTI_QTL', [])):,}"
    )

    print()

    print(
        "RULE: repeated tissues / QTL datasets / studies do NOT duplicate a gene in enrichment."
    )

    print(
        "They are retained as evidence columns supporting that one gene."
    )


def print_all_gene_ranking(
    genes,
    compact,
):
    section(
        "DEDUPLICATED GENE RANKING"
    )

    columns = [
        "RANK_STEP14",
        "EFFECTOR_GENE_ID",
        "EFFECTOR_GENE_SYMBOL",
        "MAX_H4",
        "MEDIAN_H4",
        "PRIOR_ROBUST",
        "N_QTL_TYPES",
        "QTL_TYPES",
        "N_TISSUES",
        "N_STUDIES",
        "N_INDEPENDENT_LOCUS_STUDY_PAIRS",
        "N_H4_GE_0_80_ROWS",
        "OT_ANY_LOCUS_MATCH",
        "OT_L2G_SUPPORTED_ANY",
        "OT_L2G_BEST_RANK",
        "EVIDENCE_SCORE_STEP14",
    ]

    show(
        genes,
        max_rows=(
            50
            if compact
            else
            None
        ),
        columns=columns,
    )


def print_gene_sets(
    gene_sets,
    genes,
):
    section(
        "EXACT GENE SETS USED FOR PATHWAY ANALYSIS"
    )

    print(
        "Each list below is DEDUPLICATED. A gene appears once per set.\n"
    )

    for name, members in gene_sets.items():
        print()
        print(
            f"{name}  [{len(members)} unique gene(s)]"
        )

        print(
            "-" * 100
        )

        if not members:
            print(
                "  (empty)"
            )
            continue

        for i, gid in enumerate(
            members,
            start=1,
        ):
            print(
                f"  {i:>3}. {gene_label(gid, genes)}"
            )


def print_strong_support(
    genes,
    tissue,
    modality,
    strong_h4,
):
    section(
        "WHY EACH CORE-STRONG GENE WAS SELECTED"
    )

    strong = genes[
        genes[
            "MAX_H4"
        ]
        >=
        strong_h4
    ].sort_values(
        "RANK_STEP14"
    )

    for row in strong.itertuples(
        index=False
    ):
        gid = row.EFFECTOR_GENE_ID
        symbol = clean_symbol(
            row.EFFECTOR_GENE_SYMBOL
        )

        print()

        print(
            (
                f"{row.RANK_STEP14}. "
                f"{symbol + ' ' if symbol else ''}"
                f"({gid})"
            )
        )

        print(
            "-" * 110
        )

        print(
            f"  MAX H4                         : {row.MAX_H4:.6f}"
        )

        print(
            f"  Prior robust                   : {row.PRIOR_ROBUST}"
        )

        print(
            f"  QTL modalities                 : {row.QTL_TYPES}"
        )

        print(
            f"  Number QTL modalities          : {row.N_QTL_TYPES}"
        )

        print(
            f"  Number tissues                 : {row.N_TISSUES}"
        )

        print(
            f"  Number studies                 : {row.N_STUDIES}"
        )

        print(
            f"  Independent locus-study pairs  : {row.N_INDEPENDENT_LOCUS_STUDY_PAIRS}"
        )

        print(
            f"  OT locus match                  : {getattr(row, 'OT_ANY_LOCUS_MATCH', '')}"
        )

        print(
            f"  OT L2G supports same gene       : {getattr(row, 'OT_L2G_SUPPORTED_ANY', '')}"
        )

        print(
            f"  OT best L2G rank                : {getattr(row, 'OT_L2G_BEST_RANK', np.nan)}"
        )

        print(
            f"  Evidence score (not probability): {row.EVIDENCE_SCORE_STEP14:.6f}"
        )

        m = modality[
            modality[
                "EFFECTOR_GENE_ID"
            ]
            ==
            gid
        ].sort_values(
            "MAX_H4",
            ascending=False,
        )

        if not m.empty:
            print(
                "  QTL-modality evidence:"
            )

            for mr in m.itertuples(
                index=False
            ):
                print(
                    f"    - {mr.QTL_TYPE}: "
                    f"max H4={mr.MAX_H4:.6f}; "
                    f"tissues={mr.TISSUES}; "
                    f"locus-study pairs={mr.N_LOCUS_STUDY_PAIRS}"
                )

        t = tissue[
            tissue[
                "EFFECTOR_GENE_ID"
            ]
            ==
            gid
        ].sort_values(
            "MAX_H4",
            ascending=False,
        )

        if not t.empty:
            print(
                "  Tissue evidence:"
            )

            for tr in t.itertuples(
                index=False
            ):
                print(
                    f"    - {tr.TISSUE}: "
                    f"max H4={tr.MAX_H4:.6f}; "
                    f"QTL={tr.QTL_TYPES}; "
                    f"studies={tr.N_STUDIES}"
                )


def print_pathway_results(
    all_results,
    significant,
    gene_sets,
    compact,
):
    section(
        "PATHWAY ENRICHMENT RESULTS BY GENE SET"
    )

    if (
        all_results is None
        or all_results.empty
        or "GENE_SET" not in all_results.columns
    ):
        print("(no pathway results available; enrichment may have been skipped or returned no terms)")
        for name, members in gene_sets.items():
            print(f"  {name}: {len(members)} unique input gene(s)")
        return

    for name, members in gene_sets.items():
        print()

        print(
            f"{name}: {len(members)} unique input gene(s)"
        )

        print(
            "-" * 130
        )

        if len(members) < 2:
            print(
                "  NOT TESTED: fewer than 2 genes."
            )
            continue

        z = all_results[
            all_results[
                "GENE_SET"
            ]
            ==
            name
        ].copy()

        s = significant[
            significant[
                "GENE_SET"
            ]
            ==
            name
        ].copy()

        print(
            f"  Returned terms       : {len(z):,}"
        )

        print(
            f"  Significant terms    : {len(s):,}"
        )

        if s.empty:
            if z.empty:
                print(
                    "  No terms returned."
                )
                continue

            print(
                "  No term passed all FDR / overlap / term-size filters."
            )

            best = z.sort_values(
                "FDR"
            ).head(
                10
            )

            show(
                best,
                columns=[
                    "SOURCE",
                    "TERM_NAME",
                    "FDR",
                    "INTERSECTION_SIZE",
                    "TERM_SIZE",
                    "INTERSECTION_GENE_LABELS",
                    "PASS_FDR",
                    "PASS_MIN_OVERLAP",
                    "PASS_TERM_SIZE",
                ],
            )

            continue

        s = s.sort_values(
            [
                "FDR",
                "INTERSECTION_SIZE",
            ],
            ascending=[
                True,
                False,
            ],
        )

        show(
            s,
            max_rows=(
                30
                if compact
                else
                None
            ),
            columns=[
                "SOURCE",
                "TERM_ID",
                "TERM_NAME",
                "FDR",
                "INTERSECTION_SIZE",
                "TERM_SIZE",
                "INTERSECTION_GENE_LABELS",
            ],
        )


def print_pathway_comparison(
    comparison,
    summary,
    compact,
):
    section(
        "GWAS2m CORE_STRONG vs OPEN TARGETS TOP1 PATHWAYS"
    )

    if summary.empty:
        print(
            "(comparison unavailable)"
        )
        return

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "INTERPRETATION:"
    )

    print(
        "  COMMON          = significant in both gene-set analyses."
    )

    print(
        "  GWAS2m_ONLY     = significant only for GWAS2m CORE_STRONG."
    )

    print(
        "  OPEN_TARGETS_ONLY = significant only for OT TOP1 genes."
    )

    print(
        "  These are pathway-set differences, not proof of biological novelty."
    )

    if comparison.empty:
        return

    for klass in [
        "COMMON",
        "GWAS2m_ONLY",
        "OPEN_TARGETS_ONLY",
    ]:
        z = comparison[
            comparison[
                "PATHWAY_CLASS"
            ]
            ==
            klass
        ]

        print()

        print(
            f"{klass}: {len(z)} pathway(s)"
        )

        print(
            "-" * 100
        )

        if z.empty:
            print(
                "  (none)"
            )
        else:
            show(
                z,
                max_rows=(
                    50
                    if compact
                    else
                    None
                ),
                columns=[
                    "SOURCE",
                    "TERM_NAME",
                ],
            )


def print_plain_english(
    step12,
    genes,
    gene_sets,
    significant,
    comparison_summary,
    strong_h4,
    suggestive_h4,
):
    section(
        "PLAIN-ENGLISH BIOLOGICAL INTERPRETATION"
    )

    print(
        f"1. Step12 contained {len(step12):,} locus x gene evidence rows."
    )

    print(
        f"2. Those collapse to {len(genes):,} unique genes."
    )

    print(
        f"3. {len(gene_sets.get('CORE_STRONG', []))} unique genes have MAX_H4 >= {strong_h4}."
    )

    print(
        f"4. {len(gene_sets.get('CORE_STRONG_PRIOR_ROBUST', []))} strong genes are prior-robust."
    )

    print(
        f"5. {len(gene_sets.get('CORE_STRONG_MULTI_QTL', []))} strong genes have >=2 QTL modalities."
    )

    print(
        f"6. {len(gene_sets.get('EXPANDED_SUGGESTIVE', []))} unique genes have MAX_H4 >= {suggestive_h4}."
    )

    if "OT_L2G_SUPPORTED_ANY" in genes.columns:
        strong = genes[
            genes[
                "MAX_H4"
            ]
            >=
            strong_h4
        ]

        print(
            f"7. Among strong unique genes, "
            f"{int((strong['OT_L2G_SUPPORTED_ANY'] == 'YES').sum())} "
            f"have Open Targets L2G support."
        )

    if not significant.empty:
        core = significant[
            significant[
                "GENE_SET"
            ]
            ==
            "CORE_STRONG"
        ]

        print(
            f"8. CORE_STRONG produced {len(core)} significant pathways after filters."
        )

        if not core.empty:
            print(
                "   Main CORE_STRONG pathways:"
            )

            for row in core.sort_values(
                "FDR"
            ).itertuples(
                index=False
            ):
                print(
                    f"   - {row.SOURCE}: {row.TERM_NAME}; "
                    f"FDR={row.FDR:.5g}; "
                    f"genes={row.INTERSECTION_GENE_LABELS}"
                )

    if not comparison_summary.empty:
        row = comparison_summary.iloc[0]

        print(
            f"9. GWAS2m-vs-OT significant pathway Jaccard = "
            f"{row['PATHWAY_JACCARD']:.4f}."
        )

        print(
            f"   Common={int(row['COMMON_SIGNIFICANT_PATHWAYS'])}; "
            f"GWAS2m-only={int(row['GWAS2M_SIGNIFICANT_PATHWAYS'] - row['COMMON_SIGNIFICANT_PATHWAYS'])}; "
            f"OT-only={int(row['OPEN_TARGETS_SIGNIFICANT_PATHWAYS'] - row['COMMON_SIGNIFICANT_PATHWAYS'])}."
        )

    print()

    print(
        "IMPORTANT: pathway enrichment is supportive interpretation. "
        "It does not establish gene/pathway causality."
    )


# =============================================================================
# QC
# =============================================================================

def make_qc(
    step12,
    genes,
    gene_sets,
    background,
    background_source,
    pathway_results,
    enrichment_errors,
):
    rows = []

    rows.extend(
        [
            {
                "CHECK":
                    "STEP12_LOCUS_GENE_ROWS",

                "STATUS":
                    "INFO",

                "VALUE":
                    len(step12),

                "DETAIL":
                    "Input rows before gene deduplication.",
            },
            {
                "CHECK":
                    "UNIQUE_GENES",

                "STATUS":
                    "INFO",

                "VALUE":
                    len(genes),

                "DETAIL":
                    "Unique genes after collapsing repeated tissues/QTL/studies.",
            },
            {
                "CHECK":
                    "BACKGROUND_GENES",

                "STATUS":
                    (
                        "WARN"
                        if background_source
                        ==
                        "STEP12_UNIQUE_GENES"
                        else
                        "OK"
                    ),

                "VALUE":
                    len(background),

                "DETAIL":
                    (
                        "Default background is Step12 unique genes. "
                        "For final publication, use a broader tested/eligible gene universe if available."
                        if background_source
                        ==
                        "STEP12_UNIQUE_GENES"
                        else
                        f"Custom background: {background_source}"
                    ),
            },
            {
                "CHECK":
                    "CORE_STRONG_GENES",

                "STATUS":
                    "INFO",

                "VALUE":
                    len(
                        gene_sets.get(
                            "CORE_STRONG",
                            [],
                        )
                    ),

                "DETAIL":
                    "Primary pathway input.",
            },
            {
                "CHECK":
                    "PATHWAY_RESULTS",

                "STATUS":
                    (
                        "OK"
                        if len(pathway_results)
                        else
                        "WARN"
                    ),

                "VALUE":
                    len(pathway_results),

                "DETAIL":
                    "Raw pathway rows returned.",
            },
        ]
    )

    for name, err in enrichment_errors.items():
        rows.append(
            {
                "CHECK":
                    f"ENRICHMENT_{name}",

                "STATUS":
                    "ERROR",

                "VALUE":
                    "",

                "DETAIL":
                    err,
            }
        )

    missing_symbols = int(
        (
            genes[
                "EFFECTOR_GENE_SYMBOL"
            ]
            .fillna("")
            .astype(str)
            .str.len()
            ==
            0
        )
        .sum()
    )

    rows.append(
        {
            "CHECK":
                "GENES_MISSING_SYMBOL",

            "STATUS":
                (
                    "WARN"
                    if missing_symbols
                    else
                    "OK"
                ),

            "VALUE":
                missing_symbols,

            "DETAIL":
                "Ensembl IDs remain usable even when symbol is missing.",
        }
    )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--phenotype",
        required=True,
    )

    parser.add_argument(
        "--ancestry",
        required=True,
    )

    parser.add_argument(
        "--strong-h4",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--suggestive-h4",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum number of distinct query genes required to call a pathway significant.",
    )

    parser.add_argument(
        "--min-term-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-term-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--redundancy-jaccard",
        type=float,
        default=0.70,
        help="Gene-overlap Jaccard used to group highly redundant significant pathways.",
    )

    parser.add_argument(
        "--organism",
        default="hsapiens",
    )

    parser.add_argument(
        "--sources",
        default="GO:BP,REAC,KEGG,WP",
        help="Comma-separated g:Profiler pathway sources.",
    )

    parser.add_argument(
        "--gmt",
        action="append",
        default=[],
        help="Offline GMT file. Repeat for multiple GMT libraries.",
    )

    parser.add_argument(
        "--background-file",
        default="",
        help=(
            "Optional custom tested/eligible-gene universe. "
            "Default is all unique Step12 genes."
        ),
    )

    parser.add_argument(
        "--step12-file",
        default="",
    )

    parser.add_argument(
        "--step13-file",
        default="",
    )

    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
    )

    parser.add_argument(
        "--compact-screen",
        action="store_true",
        help="Print fewer rows. Default prints the complete interpretation report.",
    )

    args = parser.parse_args()

    root = Path.cwd().resolve()

    ancestry_code, ancestry_label = ancestry_info(
        args.ancestry
    )

    phenotype_slug = slug(
        args.phenotype
    )

    ancestry_slug = slug(
        ancestry_label
    )

    step12_path = (
        Path(
            args.step12_file
        ).resolve()
        if args.step12_file
        else
        root
        /
        "12_master_table"
        /
        phenotype_slug
        /
        ancestry_slug
        /
        "Step12_Locus_Gene_Master.tsv"
    )

    step13_path = (
        Path(
            args.step13_file
        ).resolve()
        if args.step13_file
        else
        root
        /
        "13_opentargets_comparison"
        /
        phenotype_slug
        /
        ancestry_slug
        /
        "Step13_Gene_Comparison.tsv"
    )

    output = (
        root
        /
        "14_pathways"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    section(
        "GWAS2m STEP14 v2.0 - PUBLICATION-GRADE PATHWAY ANALYSIS"
    )

    print(
        f"Root                   : {root}"
    )

    print(
        f"Phenotype              : {args.phenotype}"
    )

    print(
        f"Ancestry               : {ancestry_label} ({ancestry_code})"
    )

    print(
        f"Step12                 : {step12_path}"
    )

    print(
        f"Step13                 : {step13_path if step13_path.exists() else 'NOT AVAILABLE'}"
    )

    print(
        f"Output                 : {output}"
    )

    print(
        f"Strong H4              : >= {args.strong_h4}"
    )

    print(
        f"Suggestive H4          : >= {args.suggestive_h4}"
    )

    print(
        f"FDR                    : <= {args.fdr}"
    )

    print(
        f"Minimum pathway overlap: >= {args.min_overlap} genes"
    )

    print(
        f"Term-size filter       : {args.min_term_size} - {args.max_term_size}"
    )

    # -------------------------------------------------------------------------
    # Load and collapse evidence
    # -------------------------------------------------------------------------

    step12 = load_step12(
        step12_path
    )

    step13 = load_step13(
        step13_path
    )

    tissues = build_tissue_evidence(
        step12
    )

    modalities = build_modality_evidence(
        step12
    )

    genes = build_gene_ranking(
        step12
    )

    genes = attach_step13_evidence(
        genes,
        step13,
    )

    locus_top = build_locus_top_genes(
        step12
    )

    gene_sets = build_gene_sets(
        genes,
        modalities,
        locus_top,
        step13,
        args.strong_h4,
        args.suggestive_h4,
    )

    membership = gene_set_membership_table(
        gene_sets,
        genes,
    )

    background, background_source = load_background(
        genes,
        args.background_file,
    )

    # -------------------------------------------------------------------------
    # Save gene evidence
    # -------------------------------------------------------------------------

    genes.to_csv(
        output
        /
        "Step14_Gene_Ranking.tsv",
        sep="\t",
        index=False,
    )

    tissues.to_csv(
        output
        /
        "Step14_Gene_Tissue_Evidence.tsv",
        sep="\t",
        index=False,
    )

    modalities.to_csv(
        output
        /
        "Step14_Gene_Modality_Evidence.tsv",
        sep="\t",
        index=False,
    )

    locus_top.to_csv(
        output
        /
        "Step14_Locus_Top_Genes.tsv",
        sep="\t",
        index=False,
    )

    membership.to_csv(
        output
        /
        "Step14_Gene_Set_Membership.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print gene evidence
    # -------------------------------------------------------------------------

    print_pipeline_collapse(
        step12,
        genes,
        gene_sets,
    )

    print_all_gene_ranking(
        genes,
        args.compact_screen,
    )

    print_gene_sets(
        gene_sets,
        genes,
    )

    print_strong_support(
        genes,
        tissues,
        modalities,
        args.strong_h4,
    )

    # -------------------------------------------------------------------------
    # Enrichment
    # -------------------------------------------------------------------------

    pathway_parts = []
    enrichment_errors = {}

    if not args.skip_enrichment:
        section(
            "RUNNING PATHWAY ENRICHMENT"
        )

        backend = (
            "OFFLINE_GMT"
            if args.gmt
            else
            "GPROFILER"
        )

        print(
            f"Backend                : {backend}"
        )

        print(
            f"Background              : {len(background):,} genes"
        )

        print(
            f"Background source       : {background_source}"
        )

        print(
            f"g:Profiler sources      : {args.sources}"
        )

        print()

        preferred_order = [
            "CORE_STRONG",
            "CORE_STRONG_PRIOR_ROBUST",
            "CORE_STRONG_MULTI_QTL",
            "LOCUS_TOP_STRONG",
            "EXPANDED_SUGGESTIVE",
            "CORE_STRONG_OT_SUPPORTED",
            "CORE_STRONG_OT_DISCORDANT",
            "CORE_STRONG_OT_COVERAGE_GAP",
            "OPEN_TARGETS_TOP1",
        ]

        preferred_order.extend(
            [
                f"STRONG_{qtl}"
                for qtl
                in QTL_TYPES
            ]
        )

        ordered_sets = []

        for name in preferred_order:
            if name in gene_sets:
                ordered_sets.append(name)

        for name in gene_sets:
            if name not in ordered_sets:
                ordered_sets.append(name)

        sources = [
            x.strip()
            for x
            in args.sources.split(",")
            if x.strip()
        ]

        for name in ordered_sets:
            query = gene_sets[
                name
            ]

            if len(query) < 2:
                print(
                    f"[SKIP] {name}: only {len(query)} unique gene(s)"
                )
                continue

            print(
                f"[RUN ] {name}: {len(query)} unique gene(s)"
            )

            try:
                if args.gmt:
                    result = offline_enrichment(
                        query,
                        background,
                        args.gmt,
                        name,
                    )

                else:
                    result = gprofiler_enrichment(
                        query,
                        background,
                        name,
                        sources,
                        args.organism,
                    )

                if not result.empty:
                    pathway_parts.append(
                        result
                    )

            except Exception as exc:
                message = (
                    f"{type(exc).__name__}: {exc}"
                )

                enrichment_errors[
                    name
                ] = message

                print(
                    f"[FAIL] {name}: {message}"
                )

    raw_pathways = (
        pd.concat(
            pathway_parts,
            ignore_index=True,
            sort=False,
        )
        if pathway_parts
        else
        pd.DataFrame()
    )

    if not raw_pathways.empty:
        raw_pathways = filter_pathway_results(
            raw_pathways,
            args.fdr,
            args.min_overlap,
            args.min_term_size,
            args.max_term_size,
        )

        raw_pathways = add_gene_labels(
            raw_pathways,
            genes,
        )

        raw_pathways = raw_pathways.sort_values(
            [
                "GENE_SET",
                "FDR",
                "INTERSECTION_SIZE",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

    significant = (
        raw_pathways[
            raw_pathways[
                "SIGNIFICANT"
            ]
            .map(boolish)
        ].copy()
        if (
            not raw_pathways.empty
            and
            "SIGNIFICANT"
            in raw_pathways.columns
        )
        else
        pd.DataFrame()
    )

    set_summary = pathway_set_summary(
        raw_pathways,
        gene_sets,
    )

    drivers = explode_pathway_drivers(
        significant,
        genes,
    )

    redundancy = pathway_redundancy(
        significant,
        args.redundancy_jaccard,
    )

    pathway_comparison = pd.DataFrame()
    pathway_comparison_summary = pd.DataFrame()

    if (
        "OPEN_TARGETS_TOP1"
        in
        gene_sets
    ):
        (
            pathway_comparison,
            pathway_comparison_summary,
        ) = compare_pathway_sets(
            significant,
            "CORE_STRONG",
            "OPEN_TARGETS_TOP1",
        )

    # -------------------------------------------------------------------------
    # Save pathway outputs
    # -------------------------------------------------------------------------

    raw_pathways.to_csv(
        output
        /
        "Step14_Pathway_All_Results.tsv",
        sep="\t",
        index=False,
    )

    significant.to_csv(
        output
        /
        "Step14_Pathway_Significant.tsv",
        sep="\t",
        index=False,
    )

    set_summary.to_csv(
        output
        /
        "Step14_Pathway_Set_Summary.tsv",
        sep="\t",
        index=False,
    )

    drivers.to_csv(
        output
        /
        "Step14_Pathway_Gene_Drivers.tsv",
        sep="\t",
        index=False,
    )

    redundancy.to_csv(
        output
        /
        "Step14_Pathway_Redundancy.tsv",
        sep="\t",
        index=False,
    )

    pathway_comparison.to_csv(
        output
        /
        "Step14_OpenTargets_Pathway_Comparison.tsv",
        sep="\t",
        index=False,
    )

    pathway_comparison_summary.to_csv(
        output
        /
        "Step14_OpenTargets_Pathway_Comparison_Summary.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Screen report
    # -------------------------------------------------------------------------

    show(
        set_summary,
        title="PATHWAY ENRICHMENT SUMMARY",
    )

    print_pathway_results(
        raw_pathways,
        significant,
        gene_sets,
        args.compact_screen,
    )

    print_pathway_comparison(
        pathway_comparison,
        pathway_comparison_summary,
        args.compact_screen,
    )

    print_plain_english(
        step12,
        genes,
        gene_sets,
        significant,
        pathway_comparison_summary,
        args.strong_h4,
        args.suggestive_h4,
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc = make_qc(
        step12,
        genes,
        gene_sets,
        background,
        background_source,
        raw_pathways,
        enrichment_errors,
    )

    qc.to_csv(
        output
        /
        "Step14_QC_Audit.tsv",
        sep="\t",
        index=False,
    )

    show(
        qc,
        title="STEP14 QC AUDIT",
    )

    metadata = {
        "version":
            VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "phenotype":
            args.phenotype,

        "ancestry_code":
            ancestry_code,

        "ancestry_label":
            ancestry_label,

        "step12":
            str(
                step12_path
            ),

        "step13":
            (
                str(
                    step13_path
                )
                if step13_path.exists()
                else
                None
            ),

        "n_step12_rows":
            int(
                len(
                    step12
                )
            ),

        "n_unique_genes":
            int(
                len(
                    genes
                )
            ),

        "n_background_genes":
            int(
                len(
                    background
                )
            ),

        "background_source":
            background_source,

        "strong_h4":
            args.strong_h4,

        "suggestive_h4":
            args.suggestive_h4,

        "fdr":
            args.fdr,

        "min_overlap":
            args.min_overlap,

        "term_size_range":
            [
                args.min_term_size,
                args.max_term_size,
            ],

        "gene_sets":
            {
                name:
                    len(
                        values
                    )
                for name, values
                in gene_sets.items()
            },

        "n_pathway_rows":
            int(
                len(
                    raw_pathways
                )
            ),

        "n_significant_pathways":
            int(
                len(
                    significant
                )
            ),

        "enrichment_errors":
            enrichment_errors,

        "scientific_notes":
            [
                "Each gene contributes once to pathway enrichment regardless of repeated tissue/QTL evidence.",
                "Open Targets does not contribute to the GWAS2m evidence ranking score.",
                "Pathway enrichment is associative/supportive, not causal proof.",
                "Default background is the set of unique genes reaching Step12; a broader tested-gene universe is preferable for final publication if available.",
            ],
    }

    (
        output
        /
        "Step14_summary.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    section(
        "STEP14 COMPLETE"
    )

    print(
        f"Unique Step12 genes                      : {len(genes):,}"
    )

    print(
        f"CORE_STRONG genes                        : {len(gene_sets.get('CORE_STRONG', [])):,}"
    )

    print(
        f"CORE_STRONG_PRIOR_ROBUST genes           : {len(gene_sets.get('CORE_STRONG_PRIOR_ROBUST', [])):,}"
    )

    print(
        f"CORE_STRONG_MULTI_QTL genes              : {len(gene_sets.get('CORE_STRONG_MULTI_QTL', [])):,}"
    )

    print(
        f"EXPANDED_SUGGESTIVE genes                : {len(gene_sets.get('EXPANDED_SUGGESTIVE', [])):,}"
    )

    print(
        f"Significant pathway rows                 : {len(significant):,}"
    )

    print()

    print(
        "Outputs:"
    )

    outputs = [
        "Step14_Gene_Ranking.tsv",
        "Step14_Gene_Tissue_Evidence.tsv",
        "Step14_Gene_Modality_Evidence.tsv",
        "Step14_Locus_Top_Genes.tsv",
        "Step14_Gene_Set_Membership.tsv",
        "Step14_Pathway_All_Results.tsv",
        "Step14_Pathway_Significant.tsv",
        "Step14_Pathway_Set_Summary.tsv",
        "Step14_Pathway_Gene_Drivers.tsv",
        "Step14_Pathway_Redundancy.tsv",
        "Step14_OpenTargets_Pathway_Comparison.tsv",
        "Step14_OpenTargets_Pathway_Comparison_Summary.tsv",
        "Step14_QC_Audit.tsv",
        "Step14_summary.json",
    ]

    for name in outputs:
        print(
            f"  {output / name}"
        )

    print()

    print(
        "NEXT PIPELINE STEP:"
    )

    print(
        "  Step15 = PPI / network analysis using CORE_STRONG + PRIOR_ROBUST"
    )

    print(
        "  genes, with known disease genes as external anchors."
    )


if __name__ == "__main__":
    main()

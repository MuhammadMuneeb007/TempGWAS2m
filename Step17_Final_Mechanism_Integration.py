#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GWAS2m STEP17 v1.0
FINAL DISEASE-MECHANISM INTEGRATION / EVIDENCE TIERS / CANDIDATE REPORT
================================================================================

PURPOSE
-------
Step17 is the final integration layer of GWAS2m.

It combines the evidence generated in Steps 12-16:

    Step12  molecular-QTL / colocalisation master evidence
    Step13  Open Targets comparison
    Step14  deduplicated genes + pathway enrichment
    Step15  PPI / network evidence
    Step16  tractability / drug / clinical precedence / direction-of-effect

The goal is NOT to invent a new causal probability.

The goal is to produce a transparent final table answering:

    * Which genes have the strongest genetic/molecular support?
    * Which are independently supported by Open Targets?
    * Which are GWAS2m alternative effectors at known loci?
    * Which occur where Open Targets has a coverage gap?
    * Which genes drive significant disease-relevant pathways?
    * Which genes physically connect to known/OT-supported disease genes?
    * Which targets are tractable or clinically precedented?
    * Is therapeutic direction known, conflicting, or unavailable?
    * What evidence tier should each candidate receive?
    * What is the evidence basis for the final prioritisation?

SCIENTIFIC GUARDRAILS
---------------------
1. H4 is evidence for a shared signal, NOT proof that a gene is causal.
2. Open Targets is an external comparator, NOT a biological gold standard.
3. Absence from Open Targets does NOT automatically mean novelty.
4. Pathway enrichment is supportive mechanism context, NOT causal proof.
5. STRING network proximity is supportive mechanism context, NOT causal proof.
6. Added STRING connector proteins are NOT promoted to GWAS2m genetic targets.
7. ChEMBL drug precedence for another disease is NOT proof of efficacy for the
   phenotype being analysed.
8. Therapeutic direction remains UNKNOWN unless signed, allele-harmonised
   molecular evidence supports it.
9. The final Step17 score is a transparent heuristic evidence-ranking score,
   NOT a probability of causality, clinical success, or treatment efficacy.
10. The evidence tier is rule-based and should be interpreted together with the
    underlying columns, never in isolation.

PRIMARY OUTPUT
--------------
17_mechanism/<phenotype>/<ancestry>/

    Step17_Final_Target_Ranking.tsv
    Step17_Final_Evidence_Matrix.tsv
    Step17_Evidence_Tiers.tsv

    Step17_Pathway_Mechanism_Map.tsv
    Step17_Network_Mechanism_Map.tsv
    Step17_Drug_Mechanism_Map.tsv
    Step17_External_Concordance.tsv

    Step17_High_Priority_Candidates.tsv
    Step17_GWAS2m_Alternative_Effectors.tsv
    Step17_OpenTargets_Convergent_Targets.tsv
    Step17_OT_Coverage_Gap_Targets.tsv

    Step17_QC_Audit.tsv
    Step17_summary.json

RUN
---
python Step17_Final_Mechanism_Integration.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR

Include suggestive candidates (H4 >= 0.50) in addition to CORE_STRONG:
python Step17_Final_Mechanism_Integration.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR \
  --include-suggestive

Compact terminal output:
python Step17_Final_Mechanism_Integration.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR \
  --compact-screen

DEPENDENCIES
------------
python -m pip install pandas numpy
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
from typing import Iterable

import numpy as np
import pandas as pd


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
            f"Unsupported ancestry {value!r}; use EUR/AFR/EAS/SAS/AMR."
        )

    return ANCESTRY[key]


def section(title, char="="):
    print()
    print(char * 158)
    print(title)
    print(char * 158)


def show(df, title=None, max_rows=None, columns=None):
    if title:
        section(title)

    if df is None or df.empty:
        print("(no rows)")
        return

    x = df.copy()

    if columns:
        cols = [c for c in columns if c in x.columns]
        x = x[cols]

    total = len(x)

    if max_rows is not None:
        x = x.head(max_rows)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 1400,
        "display.max_colwidth", 180,
        "display.expand_frame_repr", False,
        "display.float_format", lambda v: f"{v:.6g}",
    ):
        print(x.to_string(index=False))

    if max_rows is not None and total > len(x):
        print(f"\n[showing {len(x):,}/{total:,} rows]")


def clean_gene_id(value):
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

    m = re.search(r"(ENSG\d+)", x, flags=re.I)

    return m.group(1).upper() if m else x


def clean_symbol(value):
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


def safe_numeric(values):
    return pd.to_numeric(values, errors="coerce")


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return default


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


def first_existing(columns: Iterable[str], candidates: Iterable[str]):
    columns = list(columns)

    lower = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate in columns:
            return candidate

        found = lower.get(
            str(candidate).lower()
        )

        if found is not None:
            return found

    return None


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
        and x.strip().lower() not in {
            "nan", "none", "null", "na",
        }
    ]


def join_unique(values):
    out = []

    for value in values:
        for token in split_tokens(value):
            if token not in out:
                out.append(token)

    return ";".join(out)


def dedupe_keep_order(values):
    seen = set()
    out = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        out.append(value)

    return out


def percentile(values):
    x = safe_numeric(pd.Series(values))

    if x.notna().sum() <= 1:
        return pd.Series(
            0.0,
            index=x.index,
        )

    return (
        x.rank(
            pct=True,
            method="average",
        )
        .fillna(0.0)
    )


def read_tsv(path, required=False):
    path = Path(path)

    if not path.exists():
        if required:
            raise SystemExit(
                f"Required input missing:\n  {path}"
            )

        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ensure_gene_id(df, candidates):
    if df.empty:
        return df, None

    col = first_existing(
        df.columns,
        candidates,
    )

    if col is None:
        return df, None

    df = df.copy()

    df[col] = df[col].map(
        clean_gene_id
    )

    return df, col


# =============================================================================
# INPUT PATHS / LOADERS
# =============================================================================

def load_all_inputs(
    root,
    phenotype_slug,
    ancestry_slug,
):
    step12_dir = (
        root
        /
        "12_master_table"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    step13_dir = (
        root
        /
        "13_opentargets_comparison"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    step14_dir = (
        root
        /
        "14_pathways"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    step15_dir = (
        root
        /
        "15_network"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    step16_dir = (
        root
        /
        "16_drug_targets"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    data = {
        "step12_gene_master":
            read_tsv(
                step12_dir
                /
                "Step12_Locus_Gene_Master.tsv",
                required=True,
            ),

        "step13_gene":
            read_tsv(
                step13_dir
                /
                "Step13_Gene_Comparison.tsv",
                required=False,
            ),

        "step13_strong":
            read_tsv(
                step13_dir
                /
                "Step13_Strong_Gene_Benchmark.tsv",
                required=False,
            ),

        "step14_ranking":
            read_tsv(
                step14_dir
                /
                "Step14_Gene_Ranking.tsv",
                required=True,
            ),

        "step14_membership":
            read_tsv(
                step14_dir
                /
                "Step14_Gene_Set_Membership.tsv",
                required=True,
            ),

        "step14_pathways":
            read_tsv(
                step14_dir
                /
                "Step14_Pathway_Significant.tsv",
                required=False,
            ),

        "step14_drivers":
            read_tsv(
                step14_dir
                /
                "Step14_Pathway_Gene_Drivers.tsv",
                required=False,
            ),

        "step15_centrality":
            read_tsv(
                step15_dir
                /
                "Step15_Network_Centrality.tsv",
                required=False,
            ),

        "step15_direct":
            read_tsv(
                step15_dir
                /
                "Step15_Direct_Physical_Edges.tsv",
                required=False,
            ),

        "step15_bridges":
            read_tsv(
                step15_dir
                /
                "Step15_OT_Bridge_Summary.tsv",
                required=False,
            ),

        "step15_connectors":
            read_tsv(
                step15_dir
                /
                "Step15_Connector_Genes.tsv",
                required=False,
            ),

        "step16_targets":
            read_tsv(
                step16_dir
                /
                "Step16_Target_Prioritisation.tsv",
                required=True,
            ),

        "step16_mechanisms":
            read_tsv(
                step16_dir
                /
                "Step16_Drug_Mechanisms.tsv",
                required=False,
            ),

        "step16_direction":
            read_tsv(
                step16_dir
                /
                "Step16_Direction_Summary.tsv",
                required=False,
            ),

        "step16_phenotype_drugs":
            read_tsv(
                step16_dir
                /
                "Step16_Phenotype_Matched_Drugs.tsv",
                required=False,
            ),
    }

    data["dirs"] = {
        "step12": step12_dir,
        "step13": step13_dir,
        "step14": step14_dir,
        "step15": step15_dir,
        "step16": step16_dir,
    }

    return data


def gene_sets_from_membership(membership):
    result = {}

    if membership.empty:
        return result

    if not {
        "GENE_SET",
        "GENE_ID",
    }.issubset(
        membership.columns
    ):
        return result

    for name, group in membership.groupby(
        "GENE_SET",
        sort=False,
    ):
        result[str(name)] = dedupe_keep_order([
            clean_gene_id(value)
            for value
            in group[
                "GENE_ID"
            ]
            if clean_gene_id(value)
        ])

    return result


# =============================================================================
# PATHWAY EVIDENCE
# =============================================================================

def make_pathway_evidence(
    pathways,
    drivers,
):
    if pathways.empty:
        return pd.DataFrame(), pd.DataFrame()

    significant = pathways.copy()

    if "SIGNIFICANT" in significant.columns:
        significant = significant[
            significant[
                "SIGNIFICANT"
            ]
            .map(boolish)
        ].copy()

    if significant.empty:
        return pd.DataFrame(), pd.DataFrame()

    primary_sets = {
        "CORE_STRONG",
        "CORE_STRONG_PRIOR_ROBUST",
        "CORE_STRONG_MULTI_QTL",
        "LOCUS_TOP_STRONG",
    }

    significant["IS_PRIMARY_GWAS2M_PATHWAY_SET"] = (
        significant[
            "GENE_SET"
        ]
        .astype(str)
        .isin(primary_sets)
    )

    if drivers.empty:
        rows = []

        for row in significant.itertuples(
            index=False
        ):
            for gid in split_tokens(
                getattr(
                    row,
                    "INTERSECTION_GENES",
                    "",
                )
            ):
                rows.append({
                    "GENE_SET":
                        getattr(
                            row,
                            "GENE_SET",
                            "",
                        ),

                    "SOURCE":
                        getattr(
                            row,
                            "SOURCE",
                            "",
                        ),

                    "TERM_ID":
                        getattr(
                            row,
                            "TERM_ID",
                            "",
                        ),

                    "TERM_NAME":
                        getattr(
                            row,
                            "TERM_NAME",
                            "",
                        ),

                    "FDR":
                        safe_float(
                            getattr(
                                row,
                                "FDR",
                                np.nan,
                            )
                        ),

                    "GENE_ID":
                        clean_gene_id(
                            gid
                        ),

                    "IS_PRIMARY_GWAS2M_PATHWAY_SET":
                        getattr(
                            row,
                            "IS_PRIMARY_GWAS2M_PATHWAY_SET",
                            False,
                        ),
                })

        driver_table = pd.DataFrame(
            rows
        )

    else:
        driver_table = drivers.copy()

        driver_table, gid_col = ensure_gene_id(
            driver_table,
            [
                "GENE_ID",
                "EFFECTOR_GENE_ID",
            ],
        )

        if gid_col and gid_col != "GENE_ID":
            driver_table = driver_table.rename(
                columns={
                    gid_col:
                        "GENE_ID",
                }
            )

        if "GENE_SET" in driver_table.columns:
            driver_table[
                "IS_PRIMARY_GWAS2M_PATHWAY_SET"
            ] = (
                driver_table[
                    "GENE_SET"
                ]
                .astype(str)
                .isin(
                    primary_sets
                )
            )

    if driver_table.empty:
        return significant, pd.DataFrame()

    rows = []

    for gid, group in driver_table.groupby(
        "GENE_ID",
        sort=False,
    ):
        primary = group[
            group[
                "IS_PRIMARY_GWAS2M_PATHWAY_SET"
            ]
            .map(boolish)
        ].copy()

        rows.append({
            "GENE_ID":
                clean_gene_id(
                    gid
                ),

            "N_SIGNIFICANT_PATHWAYS_ALL_SETS":
                group[
                    [
                        "SOURCE",
                        "TERM_NAME",
                    ]
                ]
                .drop_duplicates()
                .shape[0],

            "N_SIGNIFICANT_PATHWAYS_PRIMARY":
                primary[
                    [
                        "SOURCE",
                        "TERM_NAME",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
                if not primary.empty
                else
                0,

            "N_PATHWAY_DATABASES_PRIMARY":
                primary[
                    "SOURCE"
                ]
                .astype(str)
                .nunique()
                if (
                    not primary.empty
                    and
                    "SOURCE"
                    in primary.columns
                )
                else
                0,

            "BEST_PATHWAY_FDR":
                safe_numeric(
                    primary[
                        "FDR"
                    ]
                ).min()
                if (
                    not primary.empty
                    and
                    "FDR"
                    in primary.columns
                    and
                    safe_numeric(
                        primary[
                            "FDR"
                        ]
                    )
                    .notna()
                    .any()
                )
                else
                np.nan,

            "PRIMARY_PATHWAYS":
                join_unique(
                    primary[
                        "TERM_NAME"
                    ]
                )
                if (
                    not primary.empty
                    and
                    "TERM_NAME"
                    in primary.columns
                )
                else
                "",

            "PRIMARY_PATHWAY_SOURCES":
                join_unique(
                    primary[
                        "SOURCE"
                    ]
                )
                if (
                    not primary.empty
                    and
                    "SOURCE"
                    in primary.columns
                )
                else
                "",

            "IS_PRIMARY_PATHWAY_DRIVER":
                (
                    "YES"
                    if len(primary)
                    else
                    "NO"
                ),
        })

    return (
        significant,
        pd.DataFrame(
            rows
        ),
    )


# =============================================================================
# NETWORK EVIDENCE
# =============================================================================

def make_network_evidence(
    centrality,
    direct_edges,
    bridges,
):
    centrality, gid_col = ensure_gene_id(
        centrality,
        [
            "GENE_ID",
            "EFFECTOR_GENE_ID",
        ],
    )

    if gid_col and gid_col != "GENE_ID":
        centrality = centrality.rename(
            columns={
                gid_col:
                    "GENE_ID",
            }
        )

    direct_genes = set()

    if not direct_edges.empty:
        for col in [
            "GENE_A_ID",
            "GENE_B_ID",
        ]:
            if col in direct_edges.columns:
                direct_genes.update(
                    clean_gene_id(value)
                    for value
                    in direct_edges[col]
                    if clean_gene_id(value)
                )

    bridge_genes = set()

    if not bridges.empty:
        if (
            "GWAS2M_DISCORDANT_GENE_ID"
            in
            bridges.columns
        ):
            yes = bridges[
                bridges.get(
                    "CONNECTED",
                    ""
                )
                .astype(str)
                .eq("YES")
            ].copy()

            bridge_genes.update(
                clean_gene_id(v)
                for v
                in yes[
                    "GWAS2M_DISCORDANT_GENE_ID"
                ]
                if clean_gene_id(v)
            )

    rows = []

    all_genes = set(
        centrality[
            "GENE_ID"
        ].dropna().astype(str)
    ) if (
        not centrality.empty
        and
        "GENE_ID"
        in centrality.columns
    ) else set()

    all_genes |= direct_genes
    all_genes |= bridge_genes

    for gid in sorted(
        all_genes
    ):
        z = (
            centrality[
                centrality[
                    "GENE_ID"
                ]
                ==
                gid
            ].copy()
            if not centrality.empty
            else pd.DataFrame()
        )

        row = (
            z.iloc[0]
            if not z.empty
            else
            pd.Series(
                dtype=object
            )
        )

        rows.append({
            "GENE_ID":
                gid,

            "NETWORK_DEGREE":
                safe_float(
                    row.get(
                        "DEGREE",
                        np.nan,
                    )
                ),

            "NETWORK_WEIGHTED_DEGREE":
                safe_float(
                    row.get(
                        "WEIGHTED_DEGREE",
                        np.nan,
                    )
                ),

            "NETWORK_BETWEENNESS":
                safe_float(
                    row.get(
                        "BETWEENNESS",
                        np.nan,
                    )
                ),

            "NETWORK_PAGERANK":
                safe_float(
                    row.get(
                        "PAGERANK",
                        np.nan,
                    )
                ),

            "NETWORK_COMPONENT_ID":
                row.get(
                    "COMPONENT_ID",
                    np.nan,
                ),

            "NETWORK_COMMUNITY_ID":
                row.get(
                    "COMMUNITY_ID",
                    np.nan,
                ),

            "IN_DIRECT_PHYSICAL_NETWORK":
                (
                    "YES"
                    if gid
                    in direct_genes
                    else
                    "NO"
                ),

            "HAS_PATH_TO_OT_SUPPORTED_GENE":
                (
                    "YES"
                    if gid
                    in bridge_genes
                    else
                    "NO"
                ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# DRUG / THERAPEUTIC MAP
# =============================================================================

def make_drug_map(
    mechanisms,
    phenotype_drugs,
):
    if mechanisms.empty:
        return pd.DataFrame()

    mechanisms, gid_col = ensure_gene_id(
        mechanisms,
        [
            "GENE_ID",
            "EFFECTOR_GENE_ID",
        ],
    )

    if gid_col and gid_col != "GENE_ID":
        mechanisms = mechanisms.rename(
            columns={
                gid_col:
                    "GENE_ID",
            }
        )

    rows = []

    for gid, group in mechanisms.groupby(
        "GENE_ID",
        sort=False,
    ):
        aligned = (
            group[
                group[
                    "DIRECTION_ALIGNMENT"
                ]
                .astype(str)
                .eq("ALIGNED")
            ]
            if "DIRECTION_ALIGNMENT"
            in group.columns
            else
            pd.DataFrame()
        )

        pheno = (
            group[
                group[
                    "PHENOTYPE_INDICATION_MATCH"
                ]
                .astype(str)
                .eq("YES")
            ]
            if "PHENOTYPE_INDICATION_MATCH"
            in group.columns
            else
            pd.DataFrame()
        )

        rows.append({
            "GENE_ID":
                clean_gene_id(
                    gid
                ),

            "N_DRUG_MECHANISM_ROWS":
                len(
                    group
                ),

            "N_UNIQUE_DRUGS":
                group[
                    "MOLECULE_CHEMBL_ID"
                ]
                .astype(str)
                .nunique()
                if "MOLECULE_CHEMBL_ID"
                in group.columns
                else
                0,

            "MAX_CLINICAL_PHASE":
                safe_numeric(
                    group[
                        "MAX_PHASE_OVERALL"
                    ]
                ).max()
                if (
                    "MAX_PHASE_OVERALL"
                    in group.columns
                    and
                    safe_numeric(
                        group[
                            "MAX_PHASE_OVERALL"
                        ]
                    )
                    .notna()
                    .any()
                )
                else
                np.nan,

            "DRUG_NAMES":
                join_unique(
                    group[
                        "MOLECULE_NAME"
                    ]
                )
                if "MOLECULE_NAME"
                in group.columns
                else
                "",

            "MECHANISMS_OF_ACTION":
                join_unique(
                    group[
                        "MECHANISM_OF_ACTION"
                    ]
                )
                if "MECHANISM_OF_ACTION"
                in group.columns
                else
                "",

            "ACTION_TYPES":
                join_unique(
                    group[
                        "ACTION_TYPE"
                    ]
                )
                if "ACTION_TYPE"
                in group.columns
                else
                "",

            "N_PHENOTYPE_MATCHED_DRUGS":
                pheno[
                    "MOLECULE_CHEMBL_ID"
                ]
                .astype(str)
                .nunique()
                if (
                    not pheno.empty
                    and
                    "MOLECULE_CHEMBL_ID"
                    in pheno.columns
                )
                else
                0,

            "PHENOTYPE_MATCHED_DRUGS":
                join_unique(
                    pheno[
                        "MOLECULE_NAME"
                    ]
                )
                if (
                    not pheno.empty
                    and
                    "MOLECULE_NAME"
                    in pheno.columns
                )
                else
                "",

            "N_DIRECTIONALLY_ALIGNED_DRUGS":
                aligned[
                    "MOLECULE_CHEMBL_ID"
                ]
                .astype(str)
                .nunique()
                if (
                    not aligned.empty
                    and
                    "MOLECULE_CHEMBL_ID"
                    in aligned.columns
                )
                else
                0,

            "DIRECTIONALLY_ALIGNED_DRUGS":
                join_unique(
                    aligned[
                        "MOLECULE_NAME"
                    ]
                )
                if (
                    not aligned.empty
                    and
                    "MOLECULE_NAME"
                    in aligned.columns
                )
                else
                "",
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EXTERNAL CONCORDANCE / CLASSIFICATION
# =============================================================================

def external_concordance_class(
    row,
    strong_h4,
):
    h4 = safe_float(
        row.get(
            "MAX_H4",
            np.nan,
        )
    )

    ot_l2g = boolish(
        row.get(
            "OT_L2G_SUPPORTED_ANY",
            False,
        )
    )

    ot_match = boolish(
        row.get(
            "OT_ANY_LOCUS_MATCH",
            False,
        )
    )

    if pd.isna(h4):
        return "UNRESOLVED"

    if h4 < strong_h4:
        return "BELOW_STRONG_THRESHOLD"

    if ot_l2g:
        return "EXTERNALLY_CONVERGENT_OT_L2G"

    if ot_match:
        return "GWAS2M_ALTERNATIVE_EFFECTOR_AT_OT_MATCHED_LOCUS"

    return "OT_COVERAGE_GAP_OR_UNMATCHED_LOCUS"


# =============================================================================
# FINAL SCORE
# =============================================================================

def compute_final_score(
    table,
):
    x = table.copy()

    # -------------------------------------------------------------------------
    # 45 points: genetic / molecular evidence
    # -------------------------------------------------------------------------

    x["SCORE_GENETIC_MOLECULAR"] = (
        safe_numeric(
            x[
                "MAX_H4"
            ]
        )
        .fillna(0.0)
        .clip(
            0,
            1,
        )
        *
        30.0
    )

    x["SCORE_GENETIC_MOLECULAR"] += (
        x[
            "PRIOR_ROBUST"
        ]
        .map(
            boolish
        )
        .astype(int)
        *
        5.0
    )

    x["SCORE_GENETIC_MOLECULAR"] += (
        (
            safe_numeric(
                x[
                    "N_QTL_TYPES"
                ]
            )
            .fillna(0)
            >=
            2
        )
        .astype(int)
        *
        5.0
    )

    x["SCORE_GENETIC_MOLECULAR"] += (
        x[
            "IS_LOCUS_TOP_STRONG"
        ]
        .map(
            boolish
        )
        .astype(int)
        *
        5.0
    )

    # -------------------------------------------------------------------------
    # 15 points: external concordance
    # -------------------------------------------------------------------------

    x["SCORE_EXTERNAL"] = (
        x[
            "OT_L2G_SUPPORTED_ANY"
        ]
        .map(
            boolish
        )
        .astype(int)
        *
        10.0
    )

    x["SCORE_EXTERNAL"] += (
        x[
            "OT_ANY_LOCUS_MATCH"
        ]
        .map(
            boolish
        )
        .astype(int)
        *
        5.0
    )

    # -------------------------------------------------------------------------
    # 15 points: pathway mechanism evidence
    # -------------------------------------------------------------------------

    n_pathways = (
        safe_numeric(
            x[
                "N_SIGNIFICANT_PATHWAYS_PRIMARY"
            ]
        )
        .fillna(0)
    )

    pathway_component = np.minimum(
        n_pathways,
        5,
    ) / 5.0

    x["SCORE_PATHWAY"] = (
        pathway_component
        *
        10.0
    )

    x["SCORE_PATHWAY"] += (
        x[
            "IS_PRIMARY_PATHWAY_DRIVER"
        ]
        .map(
            boolish
        )
        .astype(int)
        *
        5.0
    )

    # -------------------------------------------------------------------------
    # 10 points: physical/network context
    # -------------------------------------------------------------------------

    pagerank_pct = percentile(
        x[
            "NETWORK_PAGERANK"
        ]
    )

    x["SCORE_NETWORK"] = (
        x[
            "IN_DIRECT_PHYSICAL_NETWORK"
        ]
        .map(
            boolish
        )
        .astype(int)
        *
        4.0
    )

    x["SCORE_NETWORK"] += (
        pagerank_pct
        *
        4.0
    )

    x["SCORE_NETWORK"] += (
        x[
            "HAS_PATH_TO_OT_SUPPORTED_GENE"
        ]
        .map(
            boolish
        )
        .astype(int)
        *
        2.0
    )

    # -------------------------------------------------------------------------
    # 10 points: tractability / clinical precedence
    # -------------------------------------------------------------------------

    any_tractability = (
        x[
            [
                "OT_SM_TRACTABLE",
                "OT_AB_TRACTABLE",
                "OT_PROTAC_TRACTABLE",
                "OT_OTHER_CLINICAL_TRACTABLE",
            ]
        ]
        .apply(
            lambda col: col.map(
                boolish
            )
        )
        .any(
            axis=1
        )
    )

    x["SCORE_THERAPEUTIC"] = (
        any_tractability
        .astype(int)
        *
        3.0
    )

    x["SCORE_THERAPEUTIC"] += (
        (
            safe_numeric(
                x[
                    "N_UNIQUE_DRUGS"
                ]
            )
            .fillna(0)
            >
            0
        )
        .astype(int)
        *
        3.0
    )

    x["SCORE_THERAPEUTIC"] += (
        (
            safe_numeric(
                x[
                    "N_PHENOTYPE_MATCHED_DRUGS"
                ]
            )
            .fillna(0)
            >
            0
        )
        .astype(int)
        *
        4.0
    )

    # -------------------------------------------------------------------------
    # 5 points: direction
    # -------------------------------------------------------------------------

    x["SCORE_DIRECTION"] = 0.0

    x.loc[
        x[
            "DIRECTION_STATUS"
        ]
        .astype(str)
        .eq(
            "CONSENSUS"
        ),
        "SCORE_DIRECTION",
    ] += 2.0

    x.loc[
        safe_numeric(
            x[
                "N_DIRECTIONALLY_ALIGNED_DRUGS"
            ]
        )
        .fillna(0)
        >
        0,
        "SCORE_DIRECTION",
    ] += 3.0

    x["FINAL_EVIDENCE_SCORE"] = (
        x[
            "SCORE_GENETIC_MOLECULAR"
        ]
        +
        x[
            "SCORE_EXTERNAL"
        ]
        +
        x[
            "SCORE_PATHWAY"
        ]
        +
        x[
            "SCORE_NETWORK"
        ]
        +
        x[
            "SCORE_THERAPEUTIC"
        ]
        +
        x[
            "SCORE_DIRECTION"
        ]
    )

    x[
        "FINAL_EVIDENCE_SCORE"
    ] = (
        x[
            "FINAL_EVIDENCE_SCORE"
        ]
        .clip(
            0,
            100,
        )
    )

    return x


# =============================================================================
# RULE-BASED EVIDENCE TIER
# =============================================================================

def evidence_tier(
    row,
    strong_h4,
    suggestive_h4,
):
    h4 = safe_float(
        row.get(
            "MAX_H4",
            np.nan,
        )
    )

    prior = boolish(
        row.get(
            "PRIOR_ROBUST",
            False,
        )
    )

    multi_qtl = (
        safe_float(
            row.get(
                "N_QTL_TYPES",
                0,
            ),
            default=0,
        )
        >=
        2
    )

    ot = boolish(
        row.get(
            "OT_L2G_SUPPORTED_ANY",
            False,
        )
    )

    pathway = boolish(
        row.get(
            "IS_PRIMARY_PATHWAY_DRIVER",
            False,
        )
    )

    direct_network = boolish(
        row.get(
            "IN_DIRECT_PHYSICAL_NETWORK",
            False,
        )
    )

    tractable = any([
        boolish(
            row.get(
                "OT_SM_TRACTABLE",
                False,
            )
        ),
        boolish(
            row.get(
                "OT_AB_TRACTABLE",
                False,
            )
        ),
        boolish(
            row.get(
                "OT_PROTAC_TRACTABLE",
                False,
            )
        ),
        boolish(
            row.get(
                "OT_OTHER_CLINICAL_TRACTABLE",
                False,
            )
        ),
    ])

    drug = (
        safe_float(
            row.get(
                "N_UNIQUE_DRUGS",
                0,
            ),
            default=0,
        )
        >
        0
    )

    support_layers = sum([
        int(pathway),
        int(direct_network),
        int(tractable),
        int(drug),
    ])

    if pd.isna(h4):
        return (
            "U_UNRESOLVED",
            "No interpretable H4."
        )

    if h4 >= strong_h4:
        if ot and support_layers >= 1:
            return (
                "A1_EXTERNAL_CONVERGENT",
                "Strong molecular evidence + OT L2G concordance + at least one additional mechanism/therapeutic layer."
            )

        if (
            prior
            and
            multi_qtl
            and
            support_layers >= 2
        ):
            return (
                "A2_GWAS2M_MULTI_LAYER",
                "Strong prior-robust multi-QTL evidence + at least two pathway/network/therapeutic layers."
            )

        if (
            prior
            and
            (
                multi_qtl
                or
                support_layers >= 1
            )
        ):
            return (
                "B_STRONG_ROBUST",
                "Strong prior-robust molecular evidence with additional support."
            )

        return (
            "C_STRONG_MOLECULAR",
            "Strong H4 molecular evidence but limited independent/multi-layer support."
        )

    if h4 >= suggestive_h4:
        return (
            "D_SUGGESTIVE",
            "Suggestive H4 evidence; retain for sensitivity analysis."
        )

    return (
        "E_BACKGROUND",
        "Below Step17 prioritisation threshold."
    )


# =============================================================================
# MECHANISM SUMMARY
# =============================================================================

def mechanism_summary(
    row,
):
    symbol = clean_symbol(
        row.get(
            "EFFECTOR_GENE_SYMBOL",
            "",
        )
    )

    gid = clean_gene_id(
        row.get(
            "EFFECTOR_GENE_ID",
            "",
        )
    )

    label = (
        f"{symbol} ({gid})"
        if symbol
        else
        gid
    )

    h4 = safe_float(
        row.get(
            "MAX_H4",
            np.nan,
        )
    )

    parts = []

    if pd.notna(h4):
        parts.append(
            f"max H4={h4:.4f}"
        )

    qtl = str(
        row.get(
            "QTL_TYPES",
            "",
        )
    ).strip()

    if qtl:
        parts.append(
            f"QTL={qtl}"
        )

    if boolish(
        row.get(
            "PRIOR_ROBUST",
            False,
        )
    ):
        parts.append(
            "prior-robust"
        )

    if boolish(
        row.get(
            "OT_L2G_SUPPORTED_ANY",
            False,
        )
    ):
        parts.append(
            "OT-L2G concordant"
        )
    elif boolish(
        row.get(
            "OT_ANY_LOCUS_MATCH",
            False,
        )
    ):
        parts.append(
            "alternative effector at OT-matched locus"
        )
    else:
        parts.append(
            "OT coverage-gap/unmatched context"
        )

    pathways = split_tokens(
        row.get(
            "PRIMARY_PATHWAYS",
            "",
        )
    )

    if pathways:
        parts.append(
            "pathways="
            +
            " | ".join(
                pathways[:3]
            )
        )

    if boolish(
        row.get(
            "IN_DIRECT_PHYSICAL_NETWORK",
            False,
        )
    ):
        parts.append(
            "direct physical-network seed"
        )

    if boolish(
        row.get(
            "HAS_PATH_TO_OT_SUPPORTED_GENE",
            False,
        )
    ):
        parts.append(
            "network path to OT-supported gene"
        )

    n_drugs = safe_float(
        row.get(
            "N_UNIQUE_DRUGS",
            0,
        ),
        default=0,
    )

    if n_drugs > 0:
        parts.append(
            f"ChEMBL drugs={int(n_drugs)}"
        )

    n_pheno = safe_float(
        row.get(
            "N_PHENOTYPE_MATCHED_DRUGS",
            0,
        ),
        default=0,
    )

    if n_pheno > 0:
        parts.append(
            f"phenotype-matched drugs={int(n_pheno)}"
        )

    direction = str(
        row.get(
            "THERAPEUTIC_DIRECTION",
            "UNKNOWN",
        )
    )

    if direction and direction != "UNKNOWN":
        parts.append(
            f"direction={direction}"
        )
    else:
        parts.append(
            "therapeutic direction unresolved"
        )

    return (
        label
        +
        ": "
        +
        "; ".join(
            parts
        )
    )


# =============================================================================
# TERMINAL REPORT
# =============================================================================

def print_candidate_detail(
    table,
    compact=False,
):
    section(
        "FINAL CANDIDATE-BY-CANDIDATE EVIDENCE"
    )

    max_rows = (
        20
        if compact
        else
        None
    )

    z = table.head(
        max_rows
    ) if max_rows else table

    for row in z.to_dict(
        "records"
    ):
        symbol = clean_symbol(
            row.get(
                "EFFECTOR_GENE_SYMBOL",
                "",
            )
        )

        gid = clean_gene_id(
            row.get(
                "EFFECTOR_GENE_ID",
                "",
            )
        )

        print()

        print(
            f"#{int(row.get('STEP17_RANK', 0))} "
            f"{symbol or gid} ({gid})"
        )

        print(
            "-" * 125
        )

        print(
            f"  Evidence tier                    : "
            f"{row.get('EVIDENCE_TIER', '')}"
        )

        print(
            f"  Final evidence score             : "
            f"{safe_float(row.get('FINAL_EVIDENCE_SCORE', np.nan)):.3f}/100"
        )

        print(
            f"  Max H4                           : "
            f"{safe_float(row.get('MAX_H4', np.nan)):.6f}"
        )

        print(
            f"  Prior robust                     : "
            f"{row.get('PRIOR_ROBUST', '')}"
        )

        print(
            f"  QTL modalities                   : "
            f"{row.get('QTL_TYPES', '')}"
        )

        print(
            f"  Tissues                          : "
            f"{row.get('TISSUES', '')}"
        )

        print(
            f"  OT external class                : "
            f"{row.get('EXTERNAL_CONCORDANCE_CLASS', '')}"
        )

        print(
            f"  OT L2G same-gene support         : "
            f"{row.get('OT_L2G_SUPPORTED_ANY', '')}"
        )

        print(
            f"  Significant primary pathways     : "
            f"{int(safe_float(row.get('N_SIGNIFICANT_PATHWAYS_PRIMARY', 0), 0))}"
        )

        print(
            f"  Pathways                         : "
            f"{row.get('PRIMARY_PATHWAYS', '')}"
        )

        print(
            f"  Direct physical network          : "
            f"{row.get('IN_DIRECT_PHYSICAL_NETWORK', '')}"
        )

        print(
            f"  Path to OT-supported target      : "
            f"{row.get('HAS_PATH_TO_OT_SUPPORTED_GENE', '')}"
        )

        print(
            f"  Network PageRank                 : "
            f"{safe_float(row.get('NETWORK_PAGERANK', np.nan))}"
        )

        print(
            f"  Small-molecule tractable         : "
            f"{row.get('OT_SM_TRACTABLE', '')}"
        )

        print(
            f"  Antibody tractable               : "
            f"{row.get('OT_AB_TRACTABLE', '')}"
        )

        print(
            f"  PROTAC tractable                 : "
            f"{row.get('OT_PROTAC_TRACTABLE', '')}"
        )

        print(
            f"  ChEMBL drugs                     : "
            f"{int(safe_float(row.get('N_UNIQUE_DRUGS', 0), 0))}"
        )

        print(
            f"  Phenotype-matched drugs          : "
            f"{int(safe_float(row.get('N_PHENOTYPE_MATCHED_DRUGS', 0), 0))}"
        )

        print(
            f"  Drug names                       : "
            f"{row.get('DRUG_NAMES', '')}"
        )

        print(
            f"  Therapeutic direction            : "
            f"{row.get('THERAPEUTIC_DIRECTION', 'UNKNOWN')}"
        )

        print(
            f"  Mechanism summary                : "
            f"{row.get('MECHANISM_SUMMARY', '')}"
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
        "--include-suggestive",
        action="store_true",
        help="Include H4 >= suggestive threshold in addition to CORE_STRONG.",
    )

    parser.add_argument(
        "--output-dir",
        default="",
    )

    parser.add_argument(
        "--compact-screen",
        action="store_true",
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

    output = (
        Path(
            args.output_dir
        ).resolve()
        if args.output_dir
        else
        root
        /
        "17_mechanism"
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
        "GWAS2m STEP17 v1.0 - FINAL DISEASE-MECHANISM INTEGRATION"
    )

    print(
        f"Root                     : {root}"
    )

    print(
        f"Phenotype                : {args.phenotype}"
    )

    print(
        f"Ancestry                 : {ancestry_label} ({ancestry_code})"
    )

    print(
        f"Strong H4               : >= {args.strong_h4}"
    )

    print(
        f"Suggestive H4           : >= {args.suggestive_h4}"
    )

    print(
        f"Include suggestive      : {args.include_suggestive}"
    )

    print(
        f"Output                   : {output}"
    )

    data = load_all_inputs(
        root,
        phenotype_slug,
        ancestry_slug,
    )

    gene_sets = gene_sets_from_membership(
        data[
            "step14_membership"
        ]
    )

    ranking = data[
        "step14_ranking"
    ].copy()

    ranking[
        "EFFECTOR_GENE_ID"
    ] = ranking[
        "EFFECTOR_GENE_ID"
    ].map(
        clean_gene_id
    )

    if args.include_suggestive:
        selected_ids = set(
            gene_sets.get(
                "EXPANDED_SUGGESTIVE",
                [],
            )
        )
    else:
        selected_ids = set(
            gene_sets.get(
                "CORE_STRONG",
                [],
            )
        )

    if not selected_ids:
        raise SystemExit(
            "No target genes found in the selected Step14 gene set."
        )

    final = ranking[
        ranking[
            "EFFECTOR_GENE_ID"
        ]
        .isin(
            selected_ids
        )
    ].copy()

    # -------------------------------------------------------------------------
    # Add set membership flags
    # -------------------------------------------------------------------------

    membership_flags = [
        "CORE_STRONG",
        "CORE_STRONG_PRIOR_ROBUST",
        "CORE_STRONG_MULTI_QTL",
        "LOCUS_TOP_STRONG",
        "CORE_STRONG_OT_SUPPORTED",
        "CORE_STRONG_OT_DISCORDANT",
        "CORE_STRONG_OT_COVERAGE_GAP",
    ]

    for name in membership_flags:
        final[
            "IS_"
            +
            name
        ] = (
            final[
                "EFFECTOR_GENE_ID"
            ]
            .isin(
                set(
                    gene_sets.get(
                        name,
                        [],
                    )
                )
            )
            .map({
                True: "YES",
                False: "NO",
            })
        )

    final[
        "IS_LOCUS_TOP_STRONG"
    ] = final[
        "IS_LOCUS_TOP_STRONG"
    ]

    # -------------------------------------------------------------------------
    # Pathway layer
    # -------------------------------------------------------------------------

    (
        pathway_significant,
        pathway_gene,
    ) = make_pathway_evidence(
        data[
            "step14_pathways"
        ],
        data[
            "step14_drivers"
        ],
    )

    if not pathway_gene.empty:
        final = final.merge(
            pathway_gene,
            left_on="EFFECTOR_GENE_ID",
            right_on="GENE_ID",
            how="left",
        ).drop(
            columns=[
                "GENE_ID",
            ],
            errors="ignore",
        )

    for col, default in {
        "N_SIGNIFICANT_PATHWAYS_ALL_SETS": 0,
        "N_SIGNIFICANT_PATHWAYS_PRIMARY": 0,
        "N_PATHWAY_DATABASES_PRIMARY": 0,
        "BEST_PATHWAY_FDR": np.nan,
        "PRIMARY_PATHWAYS": "",
        "PRIMARY_PATHWAY_SOURCES": "",
        "IS_PRIMARY_PATHWAY_DRIVER": "NO",
    }.items():
        if col not in final.columns:
            final[
                col
            ] = default
        else:
            final[
                col
            ] = final[
                col
            ].fillna(
                default
            )

    # -------------------------------------------------------------------------
    # Network layer
    # -------------------------------------------------------------------------

    network = make_network_evidence(
        data[
            "step15_centrality"
        ],
        data[
            "step15_direct"
        ],
        data[
            "step15_bridges"
        ],
    )

    if not network.empty:
        final = final.merge(
            network,
            left_on="EFFECTOR_GENE_ID",
            right_on="GENE_ID",
            how="left",
        ).drop(
            columns=[
                "GENE_ID",
            ],
            errors="ignore",
        )

    for col, default in {
        "NETWORK_DEGREE": np.nan,
        "NETWORK_WEIGHTED_DEGREE": np.nan,
        "NETWORK_BETWEENNESS": np.nan,
        "NETWORK_PAGERANK": np.nan,
        "NETWORK_COMPONENT_ID": np.nan,
        "NETWORK_COMMUNITY_ID": np.nan,
        "IN_DIRECT_PHYSICAL_NETWORK": "NO",
        "HAS_PATH_TO_OT_SUPPORTED_GENE": "NO",
    }.items():
        if col not in final.columns:
            final[
                col
            ] = default
        else:
            final[
                col
            ] = final[
                col
            ].fillna(
                default
            )

    # -------------------------------------------------------------------------
    # Step16 therapeutic layer
    # -------------------------------------------------------------------------

    step16 = data[
        "step16_targets"
    ].copy()

    step16, s16_gid = ensure_gene_id(
        step16,
        [
            "EFFECTOR_GENE_ID",
            "GENE_ID",
        ],
    )

    if s16_gid is not None:
        if s16_gid != "EFFECTOR_GENE_ID":
            step16 = step16.rename(
                columns={
                    s16_gid:
                        "EFFECTOR_GENE_ID",
                }
            )

        keep = [
            "EFFECTOR_GENE_ID",
            "OT_SM_TRACTABLE",
            "OT_AB_TRACTABLE",
            "OT_PROTAC_TRACTABLE",
            "OT_OTHER_CLINICAL_TRACTABLE",
            "CHEMBL_TARGET_FOUND",
            "N_UNIQUE_DRUGS",
            "N_APPROVED_OR_PHASE4_DRUGS",
            "CHEMBL_MAX_PHASE",
            "N_PHENOTYPE_MATCHED_DRUGS",
            "PHENOTYPE_MAX_PHASE",
            "N_DIRECTIONALLY_ALIGNED_DRUGS",
            "DIRECTIONALLY_ALIGNED_DRUGS",
            "PHENOTYPE_MATCHED_DRUGS",
            "DIRECTION_STATUS",
            "THERAPEUTIC_DIRECTION",
            "THERAPEUTIC_PRIORITY_SCORE",
            "THERAPEUTIC_PRIORITY_TIER",
        ]

        keep = [
            c
            for c in keep
            if c in step16.columns
        ]

        s16_small = (
            step16[
                keep
            ]
            .drop_duplicates(
                "EFFECTOR_GENE_ID"
            )
        )

        final = final.merge(
            s16_small,
            on="EFFECTOR_GENE_ID",
            how="left",
        )

    # Drug map
    drug_map = make_drug_map(
        data[
            "step16_mechanisms"
        ],
        data[
            "step16_phenotype_drugs"
        ],
    )

    if not drug_map.empty:
        # Add detailed drug strings only; numeric columns may already come from Step16.
        detailed_cols = [
            "GENE_ID",
            "DRUG_NAMES",
            "MECHANISMS_OF_ACTION",
            "ACTION_TYPES",
        ]

        detail = drug_map[
            [
                c
                for c in detailed_cols
                if c in drug_map.columns
            ]
        ].copy()

        final = final.merge(
            detail,
            left_on="EFFECTOR_GENE_ID",
            right_on="GENE_ID",
            how="left",
        ).drop(
            columns=[
                "GENE_ID",
            ],
            errors="ignore",
        )

    # Defaults
    defaults = {
        "OT_SM_TRACTABLE": "NO",
        "OT_AB_TRACTABLE": "NO",
        "OT_PROTAC_TRACTABLE": "NO",
        "OT_OTHER_CLINICAL_TRACTABLE": "NO",
        "CHEMBL_TARGET_FOUND": "NO",
        "N_UNIQUE_DRUGS": 0,
        "N_APPROVED_OR_PHASE4_DRUGS": 0,
        "CHEMBL_MAX_PHASE": np.nan,
        "N_PHENOTYPE_MATCHED_DRUGS": 0,
        "PHENOTYPE_MAX_PHASE": np.nan,
        "N_DIRECTIONALLY_ALIGNED_DRUGS": 0,
        "DIRECTIONALLY_ALIGNED_DRUGS": "",
        "PHENOTYPE_MATCHED_DRUGS": "",
        "DIRECTION_STATUS": "UNAVAILABLE",
        "THERAPEUTIC_DIRECTION": "UNKNOWN",
        "DRUG_NAMES": "",
        "MECHANISMS_OF_ACTION": "",
        "ACTION_TYPES": "",
    }

    for col, default in defaults.items():
        if col not in final.columns:
            final[
                col
            ] = default

        final[
            col
        ] = final[
            col
        ].fillna(
            default
        )

    # -------------------------------------------------------------------------
    # External concordance
    # -------------------------------------------------------------------------

    if "OT_ANY_LOCUS_MATCH" not in final.columns:
        final[
            "OT_ANY_LOCUS_MATCH"
        ] = "UNKNOWN"

    if "OT_L2G_SUPPORTED_ANY" not in final.columns:
        final[
            "OT_L2G_SUPPORTED_ANY"
        ] = "UNKNOWN"

    final[
        "EXTERNAL_CONCORDANCE_CLASS"
    ] = final.apply(
        lambda row:
        external_concordance_class(
            row,
            args.strong_h4,
        ),
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Score + tier
    # -------------------------------------------------------------------------

    final = compute_final_score(
        final
    )

    tier_rows = final.apply(
        lambda row:
        evidence_tier(
            row,
            args.strong_h4,
            args.suggestive_h4,
        ),
        axis=1,
    )

    final[
        "EVIDENCE_TIER"
    ] = [
        x[0]
        for x
        in tier_rows
    ]

    final[
        "EVIDENCE_TIER_REASON"
    ] = [
        x[1]
        for x
        in tier_rows
    ]

    final[
        "MECHANISM_SUMMARY"
    ] = final.apply(
        mechanism_summary,
        axis=1,
    )

    tier_priority = {
        "A1_EXTERNAL_CONVERGENT": 0,
        "A2_GWAS2M_MULTI_LAYER": 1,
        "B_STRONG_ROBUST": 2,
        "C_STRONG_MOLECULAR": 3,
        "D_SUGGESTIVE": 4,
        "E_BACKGROUND": 5,
        "U_UNRESOLVED": 9,
    }

    final[
        "_TIER_SORT"
    ] = final[
        "EVIDENCE_TIER"
    ].map(
        tier_priority
    ).fillna(
        8
    )

    final = final.sort_values(
        [
            "_TIER_SORT",
            "FINAL_EVIDENCE_SCORE",
            "MAX_H4",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        kind="stable",
    ).drop(
        columns=[
            "_TIER_SORT",
        ]
    ).reset_index(
        drop=True
    )

    final[
        "STEP17_RANK"
    ] = np.arange(
        1,
        len(final) + 1,
    )

    # Put rank first
    cols = [
        "STEP17_RANK",
    ] + [
        c
        for c in final.columns
        if c != "STEP17_RANK"
    ]

    final = final[
        cols
    ]

    # -------------------------------------------------------------------------
    # Derived output subsets
    # -------------------------------------------------------------------------

    external = final[
        [
            c
            for c in [
                "STEP17_RANK",
                "EFFECTOR_GENE_ID",
                "EFFECTOR_GENE_SYMBOL",
                "MAX_H4",
                "OT_ANY_LOCUS_MATCH",
                "OT_L2G_SUPPORTED_ANY",
                "OT_L2G_BEST_RANK",
                "EXTERNAL_CONCORDANCE_CLASS",
                "EVIDENCE_TIER",
                "FINAL_EVIDENCE_SCORE",
            ]
            if c in final.columns
        ]
    ].copy()

    tiers = final[
        [
            "STEP17_RANK",
            "EFFECTOR_GENE_ID",
            "EFFECTOR_GENE_SYMBOL",
            "MAX_H4",
            "EVIDENCE_TIER",
            "EVIDENCE_TIER_REASON",
            "FINAL_EVIDENCE_SCORE",
            "EXTERNAL_CONCORDANCE_CLASS",
        ]
    ].copy()

    high = final[
        final[
            "EVIDENCE_TIER"
        ]
        .isin(
            {
                "A1_EXTERNAL_CONVERGENT",
                "A2_GWAS2M_MULTI_LAYER",
                "B_STRONG_ROBUST",
            }
        )
    ].copy()

    alternatives = final[
        final[
            "EXTERNAL_CONCORDANCE_CLASS"
        ]
        ==
        "GWAS2M_ALTERNATIVE_EFFECTOR_AT_OT_MATCHED_LOCUS"
    ].copy()

    convergent = final[
        final[
            "EXTERNAL_CONCORDANCE_CLASS"
        ]
        ==
        "EXTERNALLY_CONVERGENT_OT_L2G"
    ].copy()

    coverage_gap = final[
        final[
            "EXTERNAL_CONCORDANCE_CLASS"
        ]
        ==
        "OT_COVERAGE_GAP_OR_UNMATCHED_LOCUS"
    ].copy()

    # Mechanism maps
    pathway_map = final[
        [
            c
            for c in [
                "STEP17_RANK",
                "EFFECTOR_GENE_ID",
                "EFFECTOR_GENE_SYMBOL",
                "MAX_H4",
                "N_SIGNIFICANT_PATHWAYS_PRIMARY",
                "N_PATHWAY_DATABASES_PRIMARY",
                "BEST_PATHWAY_FDR",
                "PRIMARY_PATHWAY_SOURCES",
                "PRIMARY_PATHWAYS",
                "IS_PRIMARY_PATHWAY_DRIVER",
            ]
            if c in final.columns
        ]
    ].copy()

    network_map = final[
        [
            c
            for c in [
                "STEP17_RANK",
                "EFFECTOR_GENE_ID",
                "EFFECTOR_GENE_SYMBOL",
                "MAX_H4",
                "IN_DIRECT_PHYSICAL_NETWORK",
                "HAS_PATH_TO_OT_SUPPORTED_GENE",
                "NETWORK_DEGREE",
                "NETWORK_WEIGHTED_DEGREE",
                "NETWORK_BETWEENNESS",
                "NETWORK_PAGERANK",
                "NETWORK_COMPONENT_ID",
                "NETWORK_COMMUNITY_ID",
            ]
            if c in final.columns
        ]
    ].copy()

    drug_out = final[
        [
            c
            for c in [
                "STEP17_RANK",
                "EFFECTOR_GENE_ID",
                "EFFECTOR_GENE_SYMBOL",
                "MAX_H4",
                "OT_SM_TRACTABLE",
                "OT_AB_TRACTABLE",
                "OT_PROTAC_TRACTABLE",
                "CHEMBL_TARGET_FOUND",
                "N_UNIQUE_DRUGS",
                "CHEMBL_MAX_PHASE",
                "N_PHENOTYPE_MATCHED_DRUGS",
                "PHENOTYPE_MATCHED_DRUGS",
                "DIRECTION_STATUS",
                "THERAPEUTIC_DIRECTION",
                "N_DIRECTIONALLY_ALIGNED_DRUGS",
                "DRUG_NAMES",
                "MECHANISMS_OF_ACTION",
            ]
            if c in final.columns
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    final.to_csv(
        output
        /
        "Step17_Final_Target_Ranking.tsv",
        sep="\t",
        index=False,
    )

    final.to_csv(
        output
        /
        "Step17_Final_Evidence_Matrix.tsv",
        sep="\t",
        index=False,
    )

    tiers.to_csv(
        output
        /
        "Step17_Evidence_Tiers.tsv",
        sep="\t",
        index=False,
    )

    pathway_map.to_csv(
        output
        /
        "Step17_Pathway_Mechanism_Map.tsv",
        sep="\t",
        index=False,
    )

    network_map.to_csv(
        output
        /
        "Step17_Network_Mechanism_Map.tsv",
        sep="\t",
        index=False,
    )

    drug_out.to_csv(
        output
        /
        "Step17_Drug_Mechanism_Map.tsv",
        sep="\t",
        index=False,
    )

    external.to_csv(
        output
        /
        "Step17_External_Concordance.tsv",
        sep="\t",
        index=False,
    )

    high.to_csv(
        output
        /
        "Step17_High_Priority_Candidates.tsv",
        sep="\t",
        index=False,
    )

    alternatives.to_csv(
        output
        /
        "Step17_GWAS2m_Alternative_Effectors.tsv",
        sep="\t",
        index=False,
    )

    convergent.to_csv(
        output
        /
        "Step17_OpenTargets_Convergent_Targets.tsv",
        sep="\t",
        index=False,
    )

    coverage_gap.to_csv(
        output
        /
        "Step17_OT_Coverage_Gap_Targets.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------

    show(
        final,
        title="FINAL STEP17 TARGET RANKING",
        max_rows=None,
        columns=[
            "STEP17_RANK",
            "EFFECTOR_GENE_SYMBOL",
            "EFFECTOR_GENE_ID",
            "MAX_H4",
            "PRIOR_ROBUST",
            "N_QTL_TYPES",
            "OT_L2G_SUPPORTED_ANY",
            "EXTERNAL_CONCORDANCE_CLASS",
            "N_SIGNIFICANT_PATHWAYS_PRIMARY",
            "IN_DIRECT_PHYSICAL_NETWORK",
            "HAS_PATH_TO_OT_SUPPORTED_GENE",
            "OT_SM_TRACTABLE",
            "OT_AB_TRACTABLE",
            "OT_PROTAC_TRACTABLE",
            "N_UNIQUE_DRUGS",
            "N_PHENOTYPE_MATCHED_DRUGS",
            "THERAPEUTIC_DIRECTION",
            "EVIDENCE_TIER",
            "FINAL_EVIDENCE_SCORE",
        ],
    )

    show(
        tiers,
        title="STEP17 EVIDENCE TIERS",
    )

    show(
        external,
        title="OPEN TARGETS EXTERNAL CONCORDANCE",
    )

    show(
        pathway_map[
            safe_numeric(
                pathway_map.get(
                    "N_SIGNIFICANT_PATHWAYS_PRIMARY",
                    pd.Series(
                        0,
                        index=pathway_map.index,
                    ),
                )
            )
            >
            0
        ],
        title="GENES DRIVING PRIMARY SIGNIFICANT PATHWAYS",
    )

    show(
        network_map[
            (
                network_map.get(
                    "IN_DIRECT_PHYSICAL_NETWORK",
                    ""
                )
                .astype(str)
                .eq("YES")
            )
            |
            (
                network_map.get(
                    "HAS_PATH_TO_OT_SUPPORTED_GENE",
                    ""
                )
                .astype(str)
                .eq("YES")
            )
        ],
        title="NETWORK-SUPPORTED CANDIDATES",
    )

    show(
        drug_out[
            (
                safe_numeric(
                    drug_out.get(
                        "N_UNIQUE_DRUGS",
                        pd.Series(
                            0,
                            index=drug_out.index,
                        ),
                    )
                )
                >
                0
            )
            |
            (
                safe_numeric(
                    drug_out.get(
                        "N_PHENOTYPE_MATCHED_DRUGS",
                        pd.Series(
                            0,
                            index=drug_out.index,
                        ),
                    )
                )
                >
                0
            )
        ],
        title="DRUG / CLINICAL-PRECEDENCE CANDIDATES",
    )

    print_candidate_detail(
        final,
        compact=args.compact_screen,
    )

    # -------------------------------------------------------------------------
    # Plain-English summary
    # -------------------------------------------------------------------------

    section(
        "PLAIN-ENGLISH STEP17 INTERPRETATION"
    )

    tier_counts = (
        final[
            "EVIDENCE_TIER"
        ]
        .value_counts()
    )

    print(
        f"1. Step17 integrated {len(final)} final candidate genes."
    )

    print(
        f"2. Externally convergent OT-L2G targets: {len(convergent)}."
    )

    print(
        f"3. GWAS2m alternative effectors at OT-matched loci: {len(alternatives)}."
    )

    print(
        f"4. Strong candidates in OT coverage-gap/unmatched contexts: {len(coverage_gap)}."
    )

    print(
        f"5. Candidates driving >=1 primary significant pathway: "
        f"{int((safe_numeric(final['N_SIGNIFICANT_PATHWAYS_PRIMARY']) > 0).sum())}."
    )

    print(
        f"6. Candidates in the direct physical network: "
        f"{int(final['IN_DIRECT_PHYSICAL_NETWORK'].map(boolish).sum())}."
    )

    print(
        f"7. Candidates with a network path to an OT-supported strong gene: "
        f"{int(final['HAS_PATH_TO_OT_SUPPORTED_GENE'].map(boolish).sum())}."
    )

    any_tract = (
        final[
            [
                "OT_SM_TRACTABLE",
                "OT_AB_TRACTABLE",
                "OT_PROTAC_TRACTABLE",
                "OT_OTHER_CLINICAL_TRACTABLE",
            ]
        ]
        .apply(
            lambda col:
            col.map(
                boolish
            )
        )
        .any(
            axis=1
        )
    )

    print(
        f"8. Candidates with at least one positive tractability modality: "
        f"{int(any_tract.sum())}."
    )

    print(
        f"9. Candidates with ChEMBL drug mechanisms: "
        f"{int((safe_numeric(final['N_UNIQUE_DRUGS']) > 0).sum())}."
    )

    print(
        f"10. Candidates with phenotype-matched ChEMBL drugs: "
        f"{int((safe_numeric(final['N_PHENOTYPE_MATCHED_DRUGS']) > 0).sum())}."
    )

    print(
        f"11. Candidates with resolved therapeutic direction: "
        f"{int(final['DIRECTION_STATUS'].astype(str).eq('CONSENSUS').sum())}."
    )

    print()

    print(
        "Evidence tier counts:"
    )

    for tier, count in tier_counts.items():
        print(
            f"  {tier}: {count}"
        )

    if not final.empty:
        top = final.iloc[0]

        print()

        print(
            f"Top integrated candidate: "
            f"{clean_symbol(top.get('EFFECTOR_GENE_SYMBOL', '')) or top.get('EFFECTOR_GENE_ID', '')}"
        )

        print(
            f"  Tier : {top.get('EVIDENCE_TIER', '')}"
        )

        print(
            f"  Score: {safe_float(top.get('FINAL_EVIDENCE_SCORE', np.nan)):.3f}/100"
        )

        print(
            f"  Basis: {top.get('MECHANISM_SUMMARY', '')}"
        )

    print()

    print(
        "IMPORTANT: this is an integrated evidence ranking. "
        "It is not a causal probability and it does not establish clinical efficacy."
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc_rows = [
        {
            "CHECK":
                "FINAL_CANDIDATE_GENES",
            "STATUS":
                "INFO",
            "VALUE":
                len(final),
            "DETAIL":
                (
                    "EXPANDED_SUGGESTIVE"
                    if args.include_suggestive
                    else
                    "CORE_STRONG"
                ),
        },
        {
            "CHECK":
                "STEP14_PATHWAY_LAYER",
            "STATUS":
                (
                    "OK"
                    if not pathway_gene.empty
                    else
                    "WARN"
                ),
            "VALUE":
                len(pathway_gene),
            "DETAIL":
                "Gene-level significant pathway driver summaries.",
        },
        {
            "CHECK":
                "STEP15_NETWORK_LAYER",
            "STATUS":
                (
                    "OK"
                    if not network.empty
                    else
                    "WARN"
                ),
            "VALUE":
                len(network),
            "DETAIL":
                "Gene-level network summaries.",
        },
        {
            "CHECK":
                "STEP16_DRUG_LAYER",
            "STATUS":
                (
                    "OK"
                    if not data[
                        "step16_targets"
                    ].empty
                    else
                    "WARN"
                ),
            "VALUE":
                len(
                    data[
                        "step16_targets"
                    ]
                ),
            "DETAIL":
                "Step16 target-prioritisation rows.",
        },
        {
            "CHECK":
                "THERAPEUTIC_DIRECTION",
            "STATUS":
                (
                    "OK"
                    if final[
                        "DIRECTION_STATUS"
                    ]
                    .astype(str)
                    .eq(
                        "CONSENSUS"
                    )
                    .any()
                    else
                    "WARN"
                ),
            "VALUE":
                int(
                    final[
                        "DIRECTION_STATUS"
                    ]
                    .astype(str)
                    .eq(
                        "CONSENSUS"
                    )
                    .sum()
                ),
            "DETAIL":
                (
                    "Direction is expected to remain unavailable when "
                    "signed allele-harmonised GWAS/QTL effects are absent."
                ),
        },
        {
            "CHECK":
                "FINAL_SCORE_IS_HEURISTIC",
            "STATUS":
                "INFO",
            "VALUE":
                "YES",
            "DETAIL":
                (
                    "FINAL_EVIDENCE_SCORE is not a probability; "
                    "all component scores are retained for auditability."
                ),
        },
    ]

    qc = pd.DataFrame(
        qc_rows
    )

    qc.to_csv(
        output
        /
        "Step17_QC_Audit.tsv",
        sep="\t",
        index=False,
    )

    show(
        qc,
        title="STEP17 QC AUDIT",
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {
        "version":
            VERSION,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "phenotype":
            args.phenotype,

        "ancestry_code":
            ancestry_code,

        "ancestry_label":
            ancestry_label,

        "include_suggestive":
            args.include_suggestive,

        "strong_h4":
            args.strong_h4,

        "suggestive_h4":
            args.suggestive_h4,

        "n_final_candidates":
            len(
                final
            ),

        "evidence_tier_counts":
            {
                str(k):
                    int(v)
                for k, v
                in tier_counts.items()
            },

        "n_ot_convergent":
            len(
                convergent
            ),

        "n_gwas2m_alternative_effectors":
            len(
                alternatives
            ),

        "n_ot_coverage_gap":
            len(
                coverage_gap
            ),

        "n_pathway_supported":
            int(
                (
                    safe_numeric(
                        final[
                            "N_SIGNIFICANT_PATHWAYS_PRIMARY"
                        ]
                    )
                    >
                    0
                )
                .sum()
            ),

        "n_direct_physical_network":
            int(
                final[
                    "IN_DIRECT_PHYSICAL_NETWORK"
                ]
                .map(
                    boolish
                )
                .sum()
            ),

        "n_tractable":
            int(
                any_tract.sum()
            ),

        "n_targets_with_drugs":
            int(
                (
                    safe_numeric(
                        final[
                            "N_UNIQUE_DRUGS"
                        ]
                    )
                    >
                    0
                )
                .sum()
            ),

        "n_phenotype_matched_drugs":
            int(
                (
                    safe_numeric(
                        final[
                            "N_PHENOTYPE_MATCHED_DRUGS"
                        ]
                    )
                    >
                    0
                )
                .sum()
            ),

        "n_direction_resolved":
            int(
                final[
                    "DIRECTION_STATUS"
                ]
                .astype(str)
                .eq(
                    "CONSENSUS"
                )
                .sum()
            ),

        "scientific_notes": [
            "Step17 evidence tiers are rule-based summaries, not causal probabilities.",
            "Open Targets is an external comparator rather than a biological gold standard.",
            "Network and pathway support provide mechanism context, not causal proof.",
            "Drug precedence is target tractability/clinical context, not evidence of efficacy for the phenotype.",
            "Therapeutic direction remains unknown without signed allele-harmonised molecular evidence.",
        ],
    }

    (
        output
        /
        "Step17_summary.json"
    ).write_text(
        json.dumps(
            summary,
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
        "STEP17 COMPLETE"
    )

    print(
        f"Output directory:"
    )

    print(
        f"  {output}"
    )

    print()

    print(
        "Main outputs:"
    )

    for name in [
        "Step17_Final_Target_Ranking.tsv",
        "Step17_Final_Evidence_Matrix.tsv",
        "Step17_Evidence_Tiers.tsv",
        "Step17_Pathway_Mechanism_Map.tsv",
        "Step17_Network_Mechanism_Map.tsv",
        "Step17_Drug_Mechanism_Map.tsv",
        "Step17_External_Concordance.tsv",
        "Step17_High_Priority_Candidates.tsv",
        "Step17_GWAS2m_Alternative_Effectors.tsv",
        "Step17_OpenTargets_Convergent_Targets.tsv",
        "Step17_OT_Coverage_Gap_Targets.tsv",
        "Step17_QC_Audit.tsv",
        "Step17_summary.json",
    ]:
        print(
            f"  {output / name}"
        )

    print()

    print(
        "PIPELINE STATUS:"
    )

    print(
        "  GWAS -> fine mapping -> annotation -> splicing -> molecular QTL/coloc"
    )

    print(
        "  -> Open Targets benchmark -> pathways -> PPI/network -> druggability"
    )

    print(
        "  -> FINAL INTEGRATED DISEASE-MECHANISM PRIORITISATION COMPLETE."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
GWAS2m STEP 13 v2
COMPLETE OPEN TARGETS / GENTROPY BENCHMARK
===============================================================================

Runs after:
    Step12_Compare_OpenTargets.py

This script has TWO layers.

LAYER A -- ALWAYS RUNS
----------------------
Uses Step12 outputs only. It produces a complete inventory of:

    1. Matched GWAS2m <-> Open Targets loci
    2. GWAS2m-only loci in studies where both pipelines produced loci
    3. Open-Targets-only loci in studies where both pipelines produced loci
    4. GWAS2m-only loci where Open Targets has no credible sets for the study
    5. Open-Targets-only credible sets where GWAS2m has no Step06 loci
    6. Study-level coverage/comparability

This prevents "pipeline coverage gaps" from being called scientific misses.

LAYER B -- OFFICIAL OPEN TARGETS BULK DATA
-------------------------------------------
Uses official Open Targets Platform release 26.09 downloadable Parquet datasets:

    credible_set
    l2g_prediction
    colocalisation
    study

The first run can cache these under:

    resources/opentargets/26.09/

Subsequent phenotypes reuse the cache.

Detailed comparisons:

    - variant overlap
    - top variant agreement
    - 95% credible-set overlap
    - PIP Spearman/Pearson/MAE
    - GWAS2m Step11 gene vs Open Targets L2G
    - GWAS2m Step11 genes vs Open Targets molQTL colocalised genes
    - method-stratified summaries (SuSiE-inf vs PICS)

IMPORTANT METHOD RULE
---------------------
For SuSiE-inf:
    interval / credible-set / PIP comparisons are appropriate.

For PICS:
    lead position, whether the OT lead lies inside the GWAS2m locus,
    exact lead agreement, and variant overlap are primary.
    A one-base lead-position fallback is NOT treated as a meaningful
    interval Jaccard comparison.

INSTALL
-------
    pip install duckdb pandas numpy scipy

RUN
---
Normal run (downloads/reuses official OT cache):
    python Step13_OpenTargets_Detailed_Benchmark.py \
        --phenotype migraine \
        --ancestry EUR

Only locus inventory, no bulk OT download:
    python Step13_OpenTargets_Detailed_Benchmark.py \
        --phenotype migraine \
        --ancestry EUR \
        --skip-detail

Use existing Open Targets cache:
    python Step13_OpenTargets_Detailed_Benchmark.py \
        --phenotype migraine \
        --ancestry EUR \
        --no-download-ot

Single study:
    python Step13_OpenTargets_Detailed_Benchmark.py \
        --phenotype migraine \
        --ancestry EUR \
        --study GCST90129450

OUTPUT
------
13_opentargets_detail/<phenotype>/<ancestry>/

Core:
    study_comparability.tsv
    matched_loci.tsv
    gwas2m_only_comparable.tsv
    opentargets_only_comparable.tsv
    gwas2m_only_no_ot_coverage.tsv
    opentargets_only_gwas2m_not_finemapped.tsv
    complete_locus_inventory.tsv

Detailed:
    ot_credible_set_subset.tsv
    ot_variant_subset.tsv
    ot_l2g_subset.tsv
    ot_colocalisation_subset.tsv
    ot_colocalisation_resolved.tsv
    variant_pip_comparison.tsv
    gene_l2g_comparison.tsv
    colocalisation_gene_comparison.tsv
    method_summary.tsv
    detailed_study_summary.tsv

Metadata:
    benchmark_metadata.json

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None
    spearmanr = None


VERSION = "2.0.0"
OPEN_TARGETS_RELEASE = "26.09"

RSYNC_BASE = (
    f"rsync.ebi.ac.uk::pub/databases/opentargets/"
    f"platform/{OPEN_TARGETS_RELEASE}/output"
)

REQUIRED_OT_DATASETS = {
    "credible_set": ["credible_set", "credible_sets"],
    "l2g_prediction": ["l2g_prediction", "l2g_predictions"],
    "colocalisation": [
        "colocalisation",
        "colocalisation_coloc",
        "colocalization",
    ],
    "study": ["study", "studies"],
}

OPTIONAL_OT_DATASETS = {
    "locus": [
        "locus",
        "credible_set_locus",
        "credible_sets_locus",
    ]
}


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

def norm(value):
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9]+", " ", str(value).lower()),
    ).strip()


def slug(value):
    return norm(value).replace(" ", "_")


def ancestry_info(value):
    key = norm(value)
    if key not in ANCESTRY:
        raise SystemExit(f"Unsupported ancestry: {value}")
    return ANCESTRY[key]


def section(title):
    print()
    print("=" * 150)
    print(title)
    print("=" * 150)


def show(df, title, max_rows=1000):
    section(title)

    if df is None or df.empty:
        print("(no rows)")
        return

    x = df if max_rows is None else df.head(max_rows)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 600,
        "display.max_colwidth", 90,
        "display.expand_frame_repr", False,
        "display.float_format", lambda v: f"{v:.6g}",
    ):
        print(x.to_string(index=False))

    if len(x) < len(df):
        print(f"\n[showing {len(x):,}/{len(df):,} rows]")


def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serialisable"
    )


def canonical_chr(value):
    if value is None or pd.isna(value):
        return ""
    x = str(value).strip()
    x = re.sub(r"^chr", "", x, flags=re.I)
    if x.endswith(".0"):
        x = x[:-2]
    return x.upper()


def canonical_variant(value):
    """
    Convert:
        6_12903725_A_G
        6:12903725:A:G
        chr6:12903725:A:G
    to:
        6:12903725:A:G
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    x = str(value).strip()
    if not x:
        return ""

    x = re.sub(r"^chr", "", x, flags=re.I)
    parts = re.split(r"[:_]", x)

    if len(parts) < 4:
        return ""

    chrom = canonical_chr(parts[0])

    try:
        pos = str(int(float(parts[1])))
    except Exception:
        return ""

    ref = str(parts[2]).upper()
    alt = str(parts[3]).upper()

    return f"{chrom}:{pos}:{ref}:{alt}"


def gene_base(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    x = str(value).strip()

    if x.lower() in {
        "",
        "none",
        "nan",
        "na",
        "null",
        ".",
    }:
        return ""

    match = re.search(r"(ENSG\d+)", x, re.I)

    return (
        match.group(1).upper()
        if match
        else x
    )


def boolish(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None:
        return False

    return str(value).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def first_existing(columns, candidates):
    lower = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate in columns:
            return candidate

        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def dictionary_get(obj, candidates, default=None):
    if not isinstance(obj, dict):
        return default

    lower = {
        str(key).lower(): key
        for key in obj
    }

    for candidate in candidates:
        if candidate in obj:
            return obj[candidate]

        key = lower.get(candidate.lower())

        if key is not None:
            return obj[key]

    return default


def as_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return []


def semicolon_unique(values):
    result = []

    for value in values:
        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        x = str(value).strip()

        if x and x not in result:
            result.append(x)

    return ";".join(result)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_corr(x, y, method):
    if method == "spearman" and spearmanr is None:
        return np.nan

    if method == "pearson" and pearsonr is None:
        return np.nan

    a = safe_numeric(pd.Series(x))
    b = safe_numeric(pd.Series(y))

    good = (
        a.notna()
        &
        b.notna()
    )

    a = a[good].to_numpy()
    b = b[good].to_numpy()

    if len(a) < 3:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    try:
        if method == "spearman":
            return float(
                spearmanr(a, b).statistic
            )

        return float(
            pearsonr(a, b).statistic
        )

    except Exception:
        return np.nan


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return np.nan

    union = a | b

    if not union:
        return np.nan

    return len(a & b) / len(union)


def read_tsv(path):
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )


# =============================================================================
# PATHS
# =============================================================================

def make_paths(
    root,
    phenotype,
    ancestry_label,
    ot_cache,
):
    p = slug(phenotype)
    a = slug(ancestry_label)

    cache = (
        Path(ot_cache).resolve()
        if ot_cache
        else
        root
        /
        "resources"
        /
        "opentargets"
        /
        OPEN_TARGETS_RELEASE
    )

    return {
        "S06":
            root
            /
            "06_finemapping"
            /
            p
            /
            a,

        "S11":
            root
            /
            "11_coloc"
            /
            p
            /
            a,

        "S12":
            root
            /
            "12_opentargets_comparison"
            /
            p
            /
            a,

        "OUT":
            root
            /
            "13_opentargets_detail"
            /
            p
            /
            a,

        "OT_CACHE":
            cache,
    }


# =============================================================================
# STEP12 COMPLETE LOCUS INVENTORY
# =============================================================================

def load_step12(paths, study=None):
    base = paths["S12"]

    required = {
        "study":
            base
            /
            "study_comparison_summary.tsv",

        "gwas":
            base
            /
            "gwas2m_locus_comparison.tsv",

        "ot":
            base
            /
            "opentargets_credible_set_comparison.tsv",

        "pairs":
            base
            /
            "all_overlap_pairs.tsv",
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise SystemExit(
            "Missing Step12 output(s):\n"
            +
            "\n".join(missing)
            +
            "\nRun Step12_Compare_OpenTargets.py first."
        )

    study_df = read_tsv(required["study"])
    gwas_df = read_tsv(required["gwas"])
    ot_df = read_tsv(required["ot"])
    pair_df = read_tsv(required["pairs"])

    if study:
        study_df = study_df[
            study_df["STUDY_ID"].astype(str)
            ==
            study
        ].copy()

        gwas_df = gwas_df[
            gwas_df["STUDY_ID"].astype(str)
            ==
            study
        ].copy()

        ot_df = ot_df[
            ot_df["STUDY_ID"].astype(str)
            ==
            study
        ].copy()

        pair_df = pair_df[
            pair_df["STUDY_ID"].astype(str)
            ==
            study
        ].copy()

    return study_df, gwas_df, ot_df, pair_df


def classify_studies(study_df):
    x = study_df.copy()

    g = safe_numeric(
        x["N_GWAS2M_LOCI"]
    ).fillna(0)

    o = safe_numeric(
        x["N_OT_CREDIBLE_SETS"]
    ).fillna(0)

    conditions = [
        (g > 0) & (o > 0),
        (g > 0) & (o == 0),
        (g == 0) & (o > 0),
        (g == 0) & (o == 0),
    ]

    labels = [
        "COMPARABLE_BOTH_HAVE_LOCI",
        "GWAS2M_HAS_LOCI_OT_NO_CREDIBLE_SETS",
        "OT_HAS_CREDIBLE_SETS_GWAS2M_NOT_FINEMAPPED",
        "NO_LOCI_BOTH",
    ]

    x["COMPARABILITY_CLASS"] = np.select(
        conditions,
        labels,
        default="UNKNOWN",
    )

    return x


def build_complete_inventory(
    study_class,
    gwas_df,
    ot_df,
    pair_df,
):
    study_map = dict(
        zip(
            study_class["STUDY_ID"].astype(str),
            study_class["COMPARABILITY_CLASS"],
        )
    )

    matched = pair_df.copy()

    if not matched.empty:
        matched["COMPARABILITY_CLASS"] = matched[
            "STUDY_ID"
        ].astype(str).map(study_map)

        matched["INVENTORY_CLASS"] = "MATCHED"

        # Method-aware interpretation.
        method = matched.get(
            "OT_FINEMAPPING_METHOD",
            pd.Series("", index=matched.index),
        ).astype(str)

        exact = matched.get(
            "EXACT_LEAD",
            pd.Series(False, index=matched.index),
        ).map(boolish)

        ot_pos = safe_numeric(
            matched.get(
                "OT_LEAD_POSITION",
                pd.Series(np.nan, index=matched.index),
            )
        )

        gstart = safe_numeric(
            matched.get(
                "GWAS2M_START",
                pd.Series(np.nan, index=matched.index),
            )
        )

        gend = safe_numeric(
            matched.get(
                "GWAS2M_END",
                pd.Series(np.nan, index=matched.index),
            )
        )

        lead_inside = (
            ot_pos.notna()
            &
            gstart.notna()
            &
            gend.notna()
            &
            (ot_pos >= gstart)
            &
            (ot_pos <= gend)
        )

        match_basis = []

        for i in matched.index:
            m = str(method.loc[i]).lower()

            if exact.loc[i]:
                match_basis.append(
                    "EXACT_LEAD"
                )

            elif "pics" in m and bool(
                lead_inside.loc[i]
            ):
                match_basis.append(
                    "PICS_LEAD_INSIDE_GWAS2M_LOCUS"
                )

            elif "susie" in m:
                match_basis.append(
                    "SUSIE_INTERVAL_OVERLAP"
                )

            else:
                match_basis.append(
                    "GENOMIC_OVERLAP"
                )

        matched["MATCH_BASIS"] = match_basis

    comparable = set(
        study_class.loc[
            study_class[
                "COMPARABILITY_CLASS"
            ]
            ==
            "COMPARABLE_BOTH_HAVE_LOCI",
            "STUDY_ID",
        ].astype(str)
    )

    gwas_only = gwas_df[
        gwas_df[
            "COMPARISON_STATUS"
        ].astype(str)
        ==
        "GWAS2M_ONLY"
    ].copy()

    gwas_only["COMPARABILITY_CLASS"] = gwas_only[
        "STUDY_ID"
    ].astype(str).map(study_map)

    gwas_only_comparable = gwas_only[
        gwas_only[
            "STUDY_ID"
        ].astype(str).isin(comparable)
    ].copy()

    gwas_only_comparable[
        "INVENTORY_CLASS"
    ] = "GWAS2M_ONLY_COMPARABLE"

    gwas_only_no_ot = gwas_only[
        ~gwas_only[
            "STUDY_ID"
        ].astype(str).isin(comparable)
    ].copy()

    gwas_only_no_ot[
        "INVENTORY_CLASS"
    ] = "GWAS2M_ONLY_NO_OT_COVERAGE"

    ot_only = ot_df[
        ot_df[
            "COMPARISON_STATUS"
        ].astype(str)
        ==
        "OPENTARGETS_ONLY"
    ].copy()

    ot_only["COMPARABILITY_CLASS"] = ot_only[
        "STUDY_ID"
    ].astype(str).map(study_map)

    ot_only_comparable = ot_only[
        ot_only[
            "STUDY_ID"
        ].astype(str).isin(comparable)
    ].copy()

    ot_only_comparable[
        "INVENTORY_CLASS"
    ] = "OPENTARGETS_ONLY_COMPARABLE"

    ot_only_not_finemapped = ot_only[
        ~ot_only[
            "STUDY_ID"
        ].astype(str).isin(comparable)
    ].copy()

    ot_only_not_finemapped[
        "INVENTORY_CLASS"
    ] = (
        "OPENTARGETS_ONLY_GWAS2M_NOT_FINEMAPPED"
    )

    # Unified inventory with a SOURCE field.
    inventory_parts = []

    if not matched.empty:
        z = matched.copy()
        z["SOURCE"] = "MATCHED_PAIR"
        inventory_parts.append(z)

    if not gwas_only_comparable.empty:
        z = gwas_only_comparable.copy()
        z["SOURCE"] = "GWAS2M"
        inventory_parts.append(z)

    if not gwas_only_no_ot.empty:
        z = gwas_only_no_ot.copy()
        z["SOURCE"] = "GWAS2M"
        inventory_parts.append(z)

    if not ot_only_comparable.empty:
        z = ot_only_comparable.copy()
        z["SOURCE"] = "OPEN_TARGETS"
        inventory_parts.append(z)

    if not ot_only_not_finemapped.empty:
        z = ot_only_not_finemapped.copy()
        z["SOURCE"] = "OPEN_TARGETS"
        inventory_parts.append(z)

    inventory = (
        pd.concat(
            inventory_parts,
            ignore_index=True,
            sort=False,
        )
        if inventory_parts
        else
        pd.DataFrame()
    )

    return {
        "matched":
            matched,

        "gwas_only_comparable":
            gwas_only_comparable,

        "gwas_only_no_ot":
            gwas_only_no_ot,

        "ot_only_comparable":
            ot_only_comparable,

        "ot_only_not_finemapped":
            ot_only_not_finemapped,

        "inventory":
            inventory,
    }


# =============================================================================
# OFFICIAL OPEN TARGETS CACHE / RSYNC
# =============================================================================

def require_rsync():
    if shutil.which("rsync") is None:
        raise RuntimeError(
            "rsync was not found. Install/load rsync or "
            "populate --ot-cache manually from the official "
            "Open Targets 26.09 output."
        )


def remote_dataset_names():
    require_rsync()

    command = [
        "rsync",
        "--list-only",
        RSYNC_BASE + "/",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Could not list Open Targets rsync release.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{result.stderr}"
        )

    names = set()

    for line in result.stdout.splitlines():
        parts = line.split()

        if not parts:
            continue

        name = parts[-1].rstrip("/")

        if (
            name
            and
            name
            not in {".", ".."}
        ):
            names.add(name)

    return names


def resolve_dataset_name(
    available,
    aliases,
):
    for alias in aliases:
        if alias in available:
            return alias

    return None


def dataset_has_parquet(path):
    return (
        path.exists()
        and
        any(
            path.rglob("*.parquet")
        )
    )


def sync_dataset(
    remote_name,
    local_path,
):
    require_rsync()

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    remote = (
        f"{RSYNC_BASE}/"
        f"{remote_name}/"
    )

    command = [
        "rsync",
        "-rltvz",
        "--partial",
        "--info=progress2",
        remote,
        str(local_path) + "/",
    ]

    print(
        "\n[Open Targets download]"
    )

    print(
        " ".join(command)
    )

    subprocess.run(
        command,
        check=True,
    )


def prepare_ot_datasets(
    cache,
    download,
):
    cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    available = None
    resolved = {}

    all_specs = {
        **REQUIRED_OT_DATASETS,
        **OPTIONAL_OT_DATASETS,
    }

    for logical_name, aliases in all_specs.items():
        # First accept any alias already cached locally.
        local_name = None

        for alias in aliases:
            candidate = cache / alias

            if dataset_has_parquet(
                candidate
            ):
                local_name = alias
                break

        if local_name is not None:
            resolved[logical_name] = (
                cache
                /
                local_name
            )

            continue

        if not download:
            if logical_name in REQUIRED_OT_DATASETS:
                print(
                    f"[WARN] Missing OT dataset "
                    f"{logical_name} in {cache}"
                )

            continue

        if available is None:
            print(
                "\nDiscovering Open Targets "
                "26.09 FTP/rsync dataset names..."
            )

            available = remote_dataset_names()

        remote_name = resolve_dataset_name(
            available,
            aliases,
        )

        if remote_name is None:
            if logical_name in REQUIRED_OT_DATASETS:
                raise RuntimeError(
                    f"Could not find required Open Targets "
                    f"dataset {logical_name}. "
                    f"Tried aliases: {aliases}"
                )

            continue

        destination = (
            cache
            /
            remote_name
        )

        sync_dataset(
            remote_name,
            destination,
        )

        if dataset_has_parquet(
            destination
        ):
            resolved[logical_name] = destination

    return resolved


# =============================================================================
# DUCKDB PARQUET SUBSETTING
# =============================================================================

def require_duckdb():
    if duckdb is None:
        raise RuntimeError(
            "duckdb is required for the official Open Targets "
            "bulk-data layer.\nInstall with:\n"
            "    pip install duckdb"
        )


def parquet_files(path):
    return sorted(
        str(p)
        for p in path.rglob("*.parquet")
    )


def sql_quote(value):
    return (
        "'"
        +
        str(value).replace(
            "'",
            "''",
        )
        +
        "'"
    )


def duckdb_columns(files):
    require_duckdb()

    if not files:
        return []

    con = duckdb.connect(
        database=":memory:"
    )

    try:
        file_list = (
            "["
            +
            ",".join(
                sql_quote(x)
                for x in files
            )
            +
            "]"
        )

        rows = con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet(
                {file_list},
                union_by_name=true
            )
            """
        ).fetchdf()

        return rows[
            "column_name"
        ].astype(str).tolist()

    finally:
        con.close()


def subset_dataset_by_values(
    dataset_path,
    column_candidates,
    values,
):
    require_duckdb()

    files = parquet_files(
        dataset_path
    )

    if not files:
        return (
            pd.DataFrame(),
            None,
        )

    columns = duckdb_columns(
        files
    )

    column = first_existing(
        columns,
        column_candidates,
    )

    if column is None:
        return (
            pd.DataFrame(),
            None,
        )

    wanted = sorted(
        {
            str(x)
            for x in values
            if str(x).strip()
        }
    )

    if not wanted:
        return (
            pd.DataFrame(),
            column,
        )

    con = duckdb.connect(
        database=":memory:"
    )

    try:
        file_list = (
            "["
            +
            ",".join(
                sql_quote(x)
                for x in files
            )
            +
            "]"
        )

        wanted_sql = (
            "("
            +
            ",".join(
                sql_quote(x)
                for x in wanted
            )
            +
            ")"
        )

        query = f"""
        SELECT *
        FROM read_parquet(
            {file_list},
            union_by_name=true
        )
        WHERE CAST("{column}" AS VARCHAR)
              IN {wanted_sql}
        """

        return (
            con.execute(
                query
            ).fetchdf(),
            column,
        )

    finally:
        con.close()


# =============================================================================
# OPEN TARGETS CREDIBLE-SET / VARIANT EXTRACTION
# =============================================================================

def extract_variant_id(
    item,
):
    if not isinstance(
        item,
        dict,
    ):
        return ""

    direct = dictionary_get(
        item,
        [
            "variantId",
            "variant_id",
            "variant",
        ],
    )

    if isinstance(
        direct,
        dict,
    ):
        direct = dictionary_get(
            direct,
            [
                "id",
                "variantId",
                "variant_id",
            ],
        )

    if direct:
        return canonical_variant(
            direct
        )

    chrom = dictionary_get(
        item,
        [
            "chromosome",
            "chr",
        ],
    )

    pos = dictionary_get(
        item,
        [
            "position",
            "pos",
        ],
    )

    ref = dictionary_get(
        item,
        [
            "referenceAllele",
            "reference_allele",
            "ref",
        ],
    )

    alt = dictionary_get(
        item,
        [
            "alternateAllele",
            "alternate_allele",
            "alt",
        ],
    )

    if (
        chrom is not None
        and
        pos is not None
        and
        ref is not None
        and
        alt is not None
    ):
        return canonical_variant(
            f"{chrom}:{pos}:{ref}:{alt}"
        )

    return ""


def flatten_credible_set_variants(
    credible_subset,
):
    if credible_subset.empty:
        return pd.DataFrame()

    id_col = first_existing(
        credible_subset.columns,
        [
            "studyLocusId",
            "study_locus_id",
        ],
    )

    if id_col is None:
        return pd.DataFrame()

    locus_col = first_existing(
        credible_subset.columns,
        [
            "locus",
            "variants",
            "credibleSetVariants",
            "credible_set_variants",
        ],
    )

    if locus_col is None:
        return pd.DataFrame()

    rows = []

    for _, record in credible_subset.iterrows():
        study_locus_id = str(
            record[
                id_col
            ]
        )

        members = as_list(
            record[
                locus_col
            ]
        )

        for item in members:
            if not isinstance(
                item,
                dict,
            ):
                continue

            variant = extract_variant_id(
                item
            )

            if not variant:
                continue

            pip = dictionary_get(
                item,
                [
                    "posteriorProbability",
                    "posterior_probability",
                    "pip",
                    "PIP",
                ],
            )

            is95 = dictionary_get(
                item,
                [
                    "is95CredibleSet",
                    "is95_credible_set",
                    "in95CredibleSet",
                ],
                default=None,
            )

            is99 = dictionary_get(
                item,
                [
                    "is99CredibleSet",
                    "is99_credible_set",
                    "in99CredibleSet",
                ],
                default=None,
            )

            # The dataset itself is defined as a 95% credible set.
            # If no explicit flag exists, treat listed members as 95% members.
            if is95 is None:
                is95 = True

            rows.append(
                {
                    "OT_STUDY_LOCUS_ID":
                        study_locus_id,

                    "VARIANT_KEY":
                        variant,

                    "OT_PIP":
                        pip,

                    "OT_IS_95_CS":
                        boolish(
                            is95
                        ),

                    "OT_IS_99_CS":
                        (
                            boolish(
                                is99
                            )
                            if is99
                            is not None
                            else
                            np.nan
                        ),

                    "OT_BETA":
                        dictionary_get(
                            item,
                            [
                                "beta",
                                "mu",
                            ],
                        ),

                    "OT_SE":
                        dictionary_get(
                            item,
                            [
                                "standardError",
                                "standard_error",
                                "se",
                            ],
                        ),

                    "OT_LOG_BF":
                        dictionary_get(
                            item,
                            [
                                "logBF",
                                "log_bf",
                                "logBayesFactor",
                            ],
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def flatten_locus_dataset(
    locus_subset,
):
    """
    Optional fallback if OT ships credible-set locus variants
    as a separate Parquet dataset.
    """
    if locus_subset.empty:
        return pd.DataFrame()

    id_col = first_existing(
        locus_subset.columns,
        [
            "studyLocusId",
            "study_locus_id",
        ],
    )

    if id_col is None:
        return pd.DataFrame()

    rows = []

    for _, record in locus_subset.iterrows():
        item = record.to_dict()

        variant = extract_variant_id(
            item
        )

        if not variant:
            continue

        is95 = dictionary_get(
            item,
            [
                "is95CredibleSet",
                "is95_credible_set",
            ],
            default=True,
        )

        rows.append(
            {
                "OT_STUDY_LOCUS_ID":
                    str(
                        record[
                            id_col
                        ]
                    ),

                "VARIANT_KEY":
                    variant,

                "OT_PIP":
                    dictionary_get(
                        item,
                        [
                            "posteriorProbability",
                            "posterior_probability",
                            "pip",
                        ],
                    ),

                "OT_IS_95_CS":
                    boolish(
                        is95
                    ),

                "OT_IS_99_CS":
                    dictionary_get(
                        item,
                        [
                            "is99CredibleSet",
                            "is99_credible_set",
                        ],
                    ),

                "OT_BETA":
                    dictionary_get(
                        item,
                        [
                            "beta",
                            "mu",
                        ],
                    ),

                "OT_SE":
                    dictionary_get(
                        item,
                        [
                            "standardError",
                            "standard_error",
                        ],
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# L2G EXTRACTION
# =============================================================================

def flatten_l2g(
    l2g_subset,
):
    if l2g_subset.empty:
        return pd.DataFrame()

    id_col = first_existing(
        l2g_subset.columns,
        [
            "studyLocusId",
            "study_locus_id",
        ],
    )

    if id_col is None:
        return pd.DataFrame()

    gene_col = first_existing(
        l2g_subset.columns,
        [
            "geneId",
            "gene_id",
        ],
    )

    score_col = first_existing(
        l2g_subset.columns,
        [
            "score",
            "l2gScore",
            "l2g_score",
        ],
    )

    feature_col = first_existing(
        l2g_subset.columns,
        [
            "features",
            "featureValues",
            "feature_values",
        ],
    )

    rows = []

    for _, record in l2g_subset.iterrows():
        gene = (
            gene_base(
                record[
                    gene_col
                ]
            )
            if gene_col
            else
            ""
        )

        rows.append(
            {
                "OT_STUDY_LOCUS_ID":
                    str(
                        record[
                            id_col
                        ]
                    ),

                "OT_GENE_ID":
                    gene,

                "OT_L2G_SCORE":
                    (
                        record[
                            score_col
                        ]
                        if score_col
                        else
                        np.nan
                    ),

                "OT_FEATURES":
                    (
                        record[
                            feature_col
                        ]
                        if feature_col
                        else
                        None
                    ),
            }
        )

    x = pd.DataFrame(
        rows
    )

    if x.empty:
        return x

    x[
        "OT_L2G_SCORE"
    ] = safe_numeric(
        x[
            "OT_L2G_SCORE"
        ]
    )

    x = x.sort_values(
        [
            "OT_STUDY_LOCUS_ID",
            "OT_L2G_SCORE",
        ],
        ascending=[
            True,
            False,
        ],
    )

    x[
        "OT_L2G_RANK"
    ] = (
        x.groupby(
            "OT_STUDY_LOCUS_ID"
        )
        .cumcount()
        +
        1
    )

    return x


# =============================================================================
# COLOCALISATION EXTRACTION
# =============================================================================

def flatten_colocalisation(
    coloc_subset,
):
    if coloc_subset.empty:
        return pd.DataFrame()

    columns = coloc_subset.columns

    id_col = first_existing(
        columns,
        [
            "studyLocusId",
            "study_locus_id",
        ],
    )

    nested_col = first_existing(
        columns,
        [
            "colocalisation",
            "colocalization",
            "rows",
        ],
    )

    rows = []

    # Flat dataset.
    other_flat = first_existing(
        columns,
        [
            "otherStudyLocusId",
            "other_study_locus_id",
        ],
    )

    if (
        id_col is not None
        and
        other_flat is not None
    ):
        for _, record in coloc_subset.iterrows():
            item = record.to_dict()

            rows.append(
                {
                    "OT_STUDY_LOCUS_ID":
                        str(
                            record[
                                id_col
                            ]
                        ),

                    "OTHER_STUDY_LOCUS_ID":
                        str(
                            record[
                                other_flat
                            ]
                        ),

                    "RIGHT_STUDY_TYPE":
                        dictionary_get(
                            item,
                            [
                                "rightStudyType",
                                "right_study_type",
                            ],
                        ),

                    "METHOD":
                        dictionary_get(
                            item,
                            [
                                "colocalisationMethod",
                                "colocalizationMethod",
                                "method",
                            ],
                        ),

                    "N_COLOCALISING_VARIANTS":
                        dictionary_get(
                            item,
                            [
                                "numberColocalisingVariants",
                                "number_colocalising_variants",
                            ],
                        ),

                    "H3":
                        dictionary_get(
                            item,
                            [
                                "h3",
                                "H3",
                            ],
                        ),

                    "H4":
                        dictionary_get(
                            item,
                            [
                                "h4",
                                "H4",
                            ],
                        ),

                    "CLPP":
                        dictionary_get(
                            item,
                            [
                                "clpp",
                                "CLPP",
                            ],
                        ),

                    "BETA_RATIO_SIGN_AVERAGE":
                        dictionary_get(
                            item,
                            [
                                "betaRatioSignAverage",
                                "beta_ratio_sign_average",
                            ],
                        ),
                }
            )

        return pd.DataFrame(
            rows
        )

    # Nested dataset.
    if (
        id_col is None
        or
        nested_col is None
    ):
        return pd.DataFrame()

    for _, record in coloc_subset.iterrows():
        sid = str(
            record[
                id_col
            ]
        )

        for item in as_list(
            record[
                nested_col
            ]
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            other = dictionary_get(
                item,
                [
                    "otherStudyLocusId",
                    "other_study_locus_id",
                ],
            )

            if not other:
                continue

            rows.append(
                {
                    "OT_STUDY_LOCUS_ID":
                        sid,

                    "OTHER_STUDY_LOCUS_ID":
                        str(
                            other
                        ),

                    "RIGHT_STUDY_TYPE":
                        dictionary_get(
                            item,
                            [
                                "rightStudyType",
                                "right_study_type",
                            ],
                        ),

                    "METHOD":
                        dictionary_get(
                            item,
                            [
                                "colocalisationMethod",
                                "colocalizationMethod",
                                "method",
                            ],
                        ),

                    "N_COLOCALISING_VARIANTS":
                        dictionary_get(
                            item,
                            [
                                "numberColocalisingVariants",
                                "number_colocalising_variants",
                            ],
                        ),

                    "H3":
                        dictionary_get(
                            item,
                            [
                                "h3",
                                "H3",
                            ],
                        ),

                    "H4":
                        dictionary_get(
                            item,
                            [
                                "h4",
                                "H4",
                            ],
                        ),

                    "CLPP":
                        dictionary_get(
                            item,
                            [
                                "clpp",
                                "CLPP",
                            ],
                        ),

                    "BETA_RATIO_SIGN_AVERAGE":
                        dictionary_get(
                            item,
                            [
                                "betaRatioSignAverage",
                                "beta_ratio_sign_average",
                            ],
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# CREDIBLE-SET METADATA
# =============================================================================

def credible_set_metadata(
    credible_subset,
):
    if credible_subset.empty:
        return pd.DataFrame()

    columns = credible_subset.columns

    id_col = first_existing(
        columns,
        [
            "studyLocusId",
            "study_locus_id",
        ],
    )

    if id_col is None:
        return pd.DataFrame()

    wanted = {
        "OT_STUDY_LOCUS_ID":
            id_col,

        "OT_STUDY_ID":
            first_existing(
                columns,
                [
                    "studyId",
                    "study_id",
                ],
            ),

        "OT_QTL_GENE_ID":
            first_existing(
                columns,
                [
                    "qtlGeneId",
                    "qtl_gene_id",
                    "geneId",
                    "gene_id",
                ],
            ),

        "OT_FINEMAPPING_METHOD":
            first_existing(
                columns,
                [
                    "finemappingMethod",
                    "finemapping_method",
                ],
            ),

        "OT_CHR":
            first_existing(
                columns,
                [
                    "chromosome",
                    "chr",
                ],
            ),

        "OT_POSITION":
            first_existing(
                columns,
                [
                    "position",
                    "pos",
                ],
            ),
    }

    rows = []

    for _, record in credible_subset.iterrows():
        row = {}

        for out_col, source_col in wanted.items():
            row[
                out_col
            ] = (
                record[
                    source_col
                ]
                if source_col
                else
                np.nan
            )

        row[
            "OT_STUDY_LOCUS_ID"
        ] = str(
            row[
                "OT_STUDY_LOCUS_ID"
            ]
        )

        row[
            "OT_QTL_GENE_ID"
        ] = gene_base(
            row[
                "OT_QTL_GENE_ID"
            ]
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def study_metadata(
    study_subset,
):
    if study_subset.empty:
        return pd.DataFrame()

    columns = study_subset.columns

    id_col = first_existing(
        columns,
        [
            "studyId",
            "study_id",
            "id",
        ],
    )

    if id_col is None:
        return pd.DataFrame()

    mapping = {
        "OT_STUDY_ID":
            id_col,

        "OT_STUDY_TYPE":
            first_existing(
                columns,
                [
                    "studyType",
                    "study_type",
                ],
            ),

        "OT_PROJECT_ID":
            first_existing(
                columns,
                [
                    "projectId",
                    "project_id",
                ],
            ),

        "OT_GENE_ID_FROM_STUDY":
            first_existing(
                columns,
                [
                    "geneId",
                    "gene_id",
                ],
            ),

        "OT_BIOSAMPLE_ID":
            first_existing(
                columns,
                [
                    "biosampleFromSourceId",
                    "biosample_from_source_id",
                    "biosampleId",
                ],
            ),

        "OT_CONDITION":
            first_existing(
                columns,
                [
                    "condition",
                ],
            ),

        "OT_TRAIT":
            first_existing(
                columns,
                [
                    "traitFromSource",
                    "trait_from_source",
                ],
            ),
    }

    rows = []

    for _, record in study_subset.iterrows():
        row = {}

        for out_col, source_col in mapping.items():
            row[
                out_col
            ] = (
                record[
                    source_col
                ]
                if source_col
                else
                np.nan
            )

        row[
            "OT_STUDY_ID"
        ] = str(
            row[
                "OT_STUDY_ID"
            ]
        )

        row[
            "OT_GENE_ID_FROM_STUDY"
        ] = gene_base(
            row[
                "OT_GENE_ID_FROM_STUDY"
            ]
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# GWAS2m STEP06 VARIANTS
# =============================================================================

def load_gwas2m_variants(
    paths,
    study,
):
    path = (
        paths[
            "S06"
        ]
        /
        study
        /
        f"{study}_finemapped_variants.tsv.gz"
    )

    if not path.exists():
        return pd.DataFrame()

    x = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
    )

    chr_col = first_existing(
        x.columns,
        [
            "CHR",
            "REFERENCE_CHR",
        ],
    )

    pos_col = first_existing(
        x.columns,
        [
            "REFERENCE_POS",
            "POS",
        ],
    )

    ref_col = first_existing(
        x.columns,
        [
            "REFERENCE_REF",
            "REF",
        ],
    )

    alt_col = first_existing(
        x.columns,
        [
            "REFERENCE_ALT",
            "ALT",
        ],
    )

    locus_col = first_existing(
        x.columns,
        [
            "LOCUS_ID",
        ],
    )

    pip_col = first_existing(
        x.columns,
        [
            "PIP",
        ],
    )

    if not all(
        [
            chr_col,
            pos_col,
            ref_col,
            alt_col,
            locus_col,
            pip_col,
        ]
    ):
        return pd.DataFrame()

    x[
        "VARIANT_KEY"
    ] = [
        canonical_variant(
            f"{c}:{p}:{r}:{a}"
        )
        for c, p, r, a
        in zip(
            x[
                chr_col
            ],
            x[
                pos_col
            ],
            x[
                ref_col
            ],
            x[
                alt_col
            ],
        )
    ]

    x[
        "GWAS2M_PIP"
    ] = safe_numeric(
        x[
            pip_col
        ]
    )

    cs_col = first_existing(
        x.columns,
        [
            "IN_95_CREDIBLE_SET",
        ],
    )

    x[
        "GWAS2M_IS_95_CS"
    ] = (
        x[
            cs_col
        ].map(
            boolish
        )
        if cs_col
        else
        False
    )

    return x[
        [
            locus_col,
            "VARIANT_KEY",
            "GWAS2M_PIP",
            "GWAS2M_IS_95_CS",
        ]
    ].rename(
        columns={
            locus_col:
                "GWAS2M_LOCUS_ID"
        }
    )


# =============================================================================
# GWAS2m STEP11 GENE EVIDENCE
# =============================================================================

def load_gwas2m_genes(
    paths,
    study,
):
    path = (
        paths[
            "S11"
        ]
        /
        study
        /
        f"{study}_GTEx_SuSiE_gene_tissue_colocalization.tsv"
    )

    if not path.exists():
        return pd.DataFrame()

    x = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    if "LOCUS_ID" not in x.columns:
        return pd.DataFrame()

    if (
        "LCLPP_MULTICAUSAL"
        in x.columns
    ):
        x[
            "LCLPP_MULTICAUSAL"
        ] = safe_numeric(
            x[
                "LCLPP_MULTICAUSAL"
            ]
        )

    else:
        x[
            "LCLPP_MULTICAUSAL"
        ] = np.nan

    def effective_gene(row):
        gene = gene_base(
            row.get(
                "QTL_GENE"
            )
        )

        if gene:
            return gene

        return gene_base(
            row.get(
                "QTL_PHENOTYPE"
            )
        )

    x[
        "GWAS2M_GENE_ID"
    ] = x.apply(
        effective_gene,
        axis=1,
    )

    x = x[
        x[
            "GWAS2M_GENE_ID"
        ].astype(str).str.len()
        >
        0
    ].copy()

    keep = [
        col
        for col in [
            "LOCUS_ID",
            "QTL_TYPE",
            "TISSUE",
            "QTL_PHENOTYPE",
            "GWAS2M_GENE_ID",
            "LCLPP_MULTICAUSAL",
            "TOP_SHARED_VARIANT",
        ]
        if col
        in x.columns
    ]

    return x[
        keep
    ].rename(
        columns={
            "LOCUS_ID":
                "GWAS2M_LOCUS_ID"
        }
    )


# =============================================================================
# VARIANT/PIP COMPARISON
# =============================================================================

def compare_variants(
    matched_pairs,
    ot_variants,
    paths,
):
    rows = []

    cache = {}

    for _, pair in matched_pairs.iterrows():
        study = str(
            pair[
                "STUDY_ID"
            ]
        )

        g_locus = str(
            pair[
                "GWAS2M_LOCUS_ID"
            ]
        )

        ot_id = str(
            pair[
                "OT_STUDY_LOCUS_ID"
            ]
        )

        method = str(
            pair.get(
                "OT_FINEMAPPING_METHOD",
                "",
            )
        )

        if study not in cache:
            cache[
                study
            ] = load_gwas2m_variants(
                paths,
                study,
            )

        g = cache[
            study
        ]

        if not g.empty:
            g = g[
                g[
                    "GWAS2M_LOCUS_ID"
                ].astype(str)
                ==
                g_locus
            ].copy()

        o = (
            ot_variants[
                ot_variants[
                    "OT_STUDY_LOCUS_ID"
                ].astype(str)
                ==
                ot_id
            ].copy()
            if not ot_variants.empty
            else
            pd.DataFrame()
        )

        base = {
            "STUDY_ID":
                study,

            "GWAS2M_LOCUS_ID":
                g_locus,

            "OT_STUDY_LOCUS_ID":
                ot_id,

            "OT_METHOD":
                method,

            "MATCH_BASIS":
                pair.get(
                    "MATCH_BASIS",
                    "",
                ),
        }

        if g.empty or o.empty:
            rows.append(
                {
                    **base,

                    "DETAIL_AVAILABLE":
                        False,

                    "N_GWAS2M_VARIANTS":
                        len(g),

                    "N_OT_VARIANTS":
                        len(o),
                }
            )

            continue

        g[
            "GWAS2M_PIP"
        ] = safe_numeric(
            g[
                "GWAS2M_PIP"
            ]
        )

        o[
            "OT_PIP"
        ] = safe_numeric(
            o[
                "OT_PIP"
            ]
        )

        g = g.sort_values(
            "GWAS2M_PIP",
            ascending=False,
        )

        o = o.sort_values(
            "OT_PIP",
            ascending=False,
        )

        gall = set(
            g[
                "VARIANT_KEY"
            ].dropna()
        )

        oall = set(
            o[
                "VARIANT_KEY"
            ].dropna()
        )

        g95 = set(
            g.loc[
                g[
                    "GWAS2M_IS_95_CS"
                ].map(
                    boolish
                ),
                "VARIANT_KEY",
            ].dropna()
        )

        o95 = set(
            o.loc[
                o[
                    "OT_IS_95_CS"
                ].map(
                    boolish
                ),
                "VARIANT_KEY",
            ].dropna()
        )

        shared = (
            g[
                [
                    "VARIANT_KEY",
                    "GWAS2M_PIP",
                ]
            ]
            .merge(
                o[
                    [
                        "VARIANT_KEY",
                        "OT_PIP",
                    ]
                ],
                on="VARIANT_KEY",
                how="inner",
            )
        )

        gtop = (
            g.iloc[
                0
            ][
                "VARIANT_KEY"
            ]
            if len(g)
            else
            ""
        )

        otop = (
            o.iloc[
                0
            ][
                "VARIANT_KEY"
            ]
            if len(o)
            else
            ""
        )

        rows.append(
            {
                **base,

                "DETAIL_AVAILABLE":
                    True,

                "N_GWAS2M_VARIANTS":
                    len(
                        gall
                    ),

                "N_OT_VARIANTS":
                    len(
                        oall
                    ),

                "N_SHARED_VARIANTS":
                    len(
                        gall
                        &
                        oall
                    ),

                "ALL_VARIANT_JACCARD":
                    jaccard(
                        gall,
                        oall,
                    ),

                "N_GWAS2M_CS95":
                    len(
                        g95
                    ),

                "N_OT_CS95":
                    len(
                        o95
                    ),

                "N_SHARED_CS95":
                    len(
                        g95
                        &
                        o95
                    ),

                "CS95_JACCARD":
                    jaccard(
                        g95,
                        o95,
                    ),

                "GWAS2M_TOP_VARIANT":
                    gtop,

                "OT_TOP_VARIANT":
                    otop,

                "TOP_VARIANT_AGREEMENT":
                    bool(
                        gtop
                        and
                        otop
                        and
                        gtop
                        ==
                        otop
                    ),

                "GWAS2M_MAX_PIP":
                    g[
                        "GWAS2M_PIP"
                    ].max(),

                "OT_MAX_PIP":
                    o[
                        "OT_PIP"
                    ].max(),

                "N_SHARED_PIP":
                    len(
                        shared
                    ),

                "PIP_SPEARMAN":
                    safe_corr(
                        shared[
                            "GWAS2M_PIP"
                        ],
                        shared[
                            "OT_PIP"
                        ],
                        "spearman",
                    ),

                "PIP_PEARSON":
                    safe_corr(
                        shared[
                            "GWAS2M_PIP"
                        ],
                        shared[
                            "OT_PIP"
                        ],
                        "pearson",
                    ),

                "PIP_MAE":
                    (
                        float(
                            np.nanmean(
                                np.abs(
                                    safe_numeric(
                                        shared[
                                            "GWAS2M_PIP"
                                        ]
                                    )
                                    -
                                    safe_numeric(
                                        shared[
                                            "OT_PIP"
                                        ]
                                    )
                                )
                            )
                        )
                        if len(
                            shared
                        )
                        else
                        np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# GENE / L2G COMPARISON
# =============================================================================

def compare_genes(
    matched_pairs,
    l2g,
    paths,
):
    rows = []

    cache = {}

    for _, pair in matched_pairs.iterrows():
        study = str(
            pair[
                "STUDY_ID"
            ]
        )

        g_locus = str(
            pair[
                "GWAS2M_LOCUS_ID"
            ]
        )

        ot_id = str(
            pair[
                "OT_STUDY_LOCUS_ID"
            ]
        )

        method = str(
            pair.get(
                "OT_FINEMAPPING_METHOD",
                "",
            )
        )

        if study not in cache:
            cache[
                study
            ] = load_gwas2m_genes(
                paths,
                study,
            )

        g = cache[
            study
        ]

        if not g.empty:
            g = g[
                g[
                    "GWAS2M_LOCUS_ID"
                ].astype(str)
                ==
                g_locus
            ].copy()

        if not g.empty:
            grouped = (
                g.groupby(
                    "GWAS2M_GENE_ID",
                    as_index=False,
                )
                .agg(
                    GWAS2M_MAX_LCLPP=(
                        "LCLPP_MULTICAUSAL",
                        "max",
                    )
                )
                .sort_values(
                    "GWAS2M_MAX_LCLPP",
                    ascending=False,
                )
            )

            ggenes = grouped[
                "GWAS2M_GENE_ID"
            ].astype(str).tolist()

        else:
            ggenes = []

        o = (
            l2g[
                l2g[
                    "OT_STUDY_LOCUS_ID"
                ].astype(str)
                ==
                ot_id
            ].copy()
            if not l2g.empty
            else
            pd.DataFrame()
        )

        if not o.empty:
            o = o.sort_values(
                "OT_L2G_RANK"
            )

            ogenes = o[
                "OT_GENE_ID"
            ].dropna().astype(str).tolist()

        else:
            ogenes = []

        rows.append(
            {
                "STUDY_ID":
                    study,

                "GWAS2M_LOCUS_ID":
                    g_locus,

                "OT_STUDY_LOCUS_ID":
                    ot_id,

                "OT_METHOD":
                    method,

                "N_GWAS2M_GENES":
                    len(
                        set(
                            ggenes
                        )
                    ),

                "N_OT_L2G_GENES":
                    len(
                        set(
                            ogenes
                        )
                    ),

                "GWAS2M_TOP_GENE":
                    (
                        ggenes[
                            0
                        ]
                        if ggenes
                        else
                        ""
                    ),

                "OT_TOP_GENE":
                    (
                        ogenes[
                            0
                        ]
                        if ogenes
                        else
                        ""
                    ),

                "TOP1_GENE_AGREEMENT":
                    bool(
                        ggenes
                        and
                        ogenes
                        and
                        ggenes[
                            0
                        ]
                        ==
                        ogenes[
                            0
                        ]
                    ),

                "GWAS2M_TOP_IN_OT_TOP3":
                    bool(
                        ggenes
                        and
                        ggenes[
                            0
                        ]
                        in set(
                            ogenes[
                                :3
                            ]
                        )
                    ),

                "GWAS2M_TOP_IN_OT_TOP5":
                    bool(
                        ggenes
                        and
                        ggenes[
                            0
                        ]
                        in set(
                            ogenes[
                                :5
                            ]
                        )
                    ),

                "TOP5_GENE_JACCARD":
                    jaccard(
                        ggenes[
                            :5
                        ],
                        ogenes[
                            :5
                        ],
                    ),

                "GWAS2M_TOP5":
                    ";".join(
                        ggenes[
                            :5
                        ]
                    ),

                "OT_TOP5":
                    ";".join(
                        ogenes[
                            :5
                        ]
                    ),

                "OT_TOP1_SCORE":
                    (
                        safe_numeric(
                            pd.Series(
                                [
                                    o.iloc[
                                        0
                                    ][
                                        "OT_L2G_SCORE"
                                    ]
                                ]
                            )
                        ).iloc[
                            0
                        ]
                        if not o.empty
                        else
                        np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# COLOCALISATION RESOLUTION / COMPARISON
# =============================================================================

def resolve_colocalisation(
    coloc,
    credible_meta,
    study_meta,
):
    if coloc.empty:
        return pd.DataFrame()

    meta = credible_meta.rename(
        columns={
            "OT_STUDY_LOCUS_ID":
                "OTHER_STUDY_LOCUS_ID",

            "OT_STUDY_ID":
                "OTHER_STUDY_ID",

            "OT_QTL_GENE_ID":
                "OTHER_QTL_GENE_ID",
        }
    )

    out = coloc.merge(
        meta[
            [
                col
                for col in [
                    "OTHER_STUDY_LOCUS_ID",
                    "OTHER_STUDY_ID",
                    "OTHER_QTL_GENE_ID",
                ]
                if col
                in meta.columns
            ]
        ],
        on="OTHER_STUDY_LOCUS_ID",
        how="left",
    )

    if not study_meta.empty:
        s = study_meta.rename(
            columns={
                "OT_STUDY_ID":
                    "OTHER_STUDY_ID",
            }
        )

        out = out.merge(
            s,
            on="OTHER_STUDY_ID",
            how="left",
        )

    out[
        "OTHER_GENE_ID"
    ] = [
        gene_base(
            qtl_gene
        )
        or
        gene_base(
            study_gene
        )
        for qtl_gene, study_gene
        in zip(
            out.get(
                "OTHER_QTL_GENE_ID",
                pd.Series(
                    "",
                    index=out.index,
                ),
            ),
            out.get(
                "OT_GENE_ID_FROM_STUDY",
                pd.Series(
                    "",
                    index=out.index,
                ),
            ),
        )
    ]

    out[
        "IS_GTEX"
    ] = [
        (
            "gtex"
            in str(
                project
            ).lower()
        )
        or
        (
            "gtex"
            in str(
                study_id
            ).lower()
        )
        for project, study_id
        in zip(
            out.get(
                "OT_PROJECT_ID",
                pd.Series(
                    "",
                    index=out.index,
                ),
            ),
            out.get(
                "OTHER_STUDY_ID",
                pd.Series(
                    "",
                    index=out.index,
                ),
            ),
        )
    ]

    return out


def compare_coloc_genes(
    matched_pairs,
    coloc_resolved,
    paths,
):
    rows = []

    cache = {}

    for _, pair in matched_pairs.iterrows():
        study = str(
            pair[
                "STUDY_ID"
            ]
        )

        g_locus = str(
            pair[
                "GWAS2M_LOCUS_ID"
            ]
        )

        ot_id = str(
            pair[
                "OT_STUDY_LOCUS_ID"
            ]
        )

        method = str(
            pair.get(
                "OT_FINEMAPPING_METHOD",
                "",
            )
        )

        if study not in cache:
            cache[
                study
            ] = load_gwas2m_genes(
                paths,
                study,
            )

        g = cache[
            study
        ]

        if not g.empty:
            g = g[
                g[
                    "GWAS2M_LOCUS_ID"
                ].astype(str)
                ==
                g_locus
            ].copy()

        ggenes = (
            set(
                g[
                    "GWAS2M_GENE_ID"
                ].dropna().astype(str)
            )
            if not g.empty
            else
            set()
        )

        gmax = (
            safe_numeric(
                g[
                    "LCLPP_MULTICAUSAL"
                ]
            ).max()
            if not g.empty
            else
            np.nan
        )

        o = (
            coloc_resolved[
                coloc_resolved[
                    "OT_STUDY_LOCUS_ID"
                ].astype(str)
                ==
                ot_id
            ].copy()
            if not coloc_resolved.empty
            else
            pd.DataFrame()
        )

        ogenes = (
            set(
                gene_base(
                    x
                )
                for x in o[
                    "OTHER_GENE_ID"
                ].dropna()
                if gene_base(
                    x
                )
            )
            if (
                not o.empty
                and
                "OTHER_GENE_ID"
                in o.columns
            )
            else
            set()
        )

        if (
            not o.empty
            and
            "IS_GTEX"
            in o.columns
        ):
            gtex = o[
                o[
                    "IS_GTEX"
                ].map(
                    boolish
                )
            ].copy()

        else:
            gtex = pd.DataFrame()

        gtex_genes = (
            set(
                gene_base(
                    x
                )
                for x in gtex[
                    "OTHER_GENE_ID"
                ].dropna()
                if gene_base(
                    x
                )
            )
            if (
                not gtex.empty
                and
                "OTHER_GENE_ID"
                in gtex.columns
            )
            else
            set()
        )

        rows.append(
            {
                "STUDY_ID":
                    study,

                "GWAS2M_LOCUS_ID":
                    g_locus,

                "OT_STUDY_LOCUS_ID":
                    ot_id,

                "OT_METHOD":
                    method,

                "N_GWAS2M_COLOC_GENES":
                    len(
                        ggenes
                    ),

                "N_OT_MOLQTL_COLOC_GENES":
                    len(
                        ogenes
                    ),

                "N_SHARED_COLOC_GENES":
                    len(
                        ggenes
                        &
                        ogenes
                    ),

                "COLOC_GENE_JACCARD":
                    jaccard(
                        ggenes,
                        ogenes,
                    ),

                "N_OT_GTEX_COLOC_GENES":
                    len(
                        gtex_genes
                    ),

                "N_SHARED_GTEX_GENES":
                    len(
                        ggenes
                        &
                        gtex_genes
                    ),

                "GTEX_GENE_JACCARD":
                    jaccard(
                        ggenes,
                        gtex_genes,
                    ),

                "GWAS2M_MAX_LCLPP":
                    gmax,

                "OT_MAX_H4":
                    (
                        safe_numeric(
                            o[
                                "H4"
                            ]
                        ).max()
                        if (
                            not o.empty
                            and
                            "H4"
                            in o.columns
                        )
                        else
                        np.nan
                    ),

                "OT_MAX_CLPP":
                    (
                        safe_numeric(
                            o[
                                "CLPP"
                            ]
                        ).max()
                        if (
                            not o.empty
                            and
                            "CLPP"
                            in o.columns
                        )
                        else
                        np.nan
                    ),

                "GWAS2M_GENES":
                    ";".join(
                        sorted(
                            ggenes
                        )
                    ),

                "OT_MOLQTL_GENES":
                    ";".join(
                        sorted(
                            ogenes
                        )
                    ),

                "OT_GTEX_GENES":
                    ";".join(
                        sorted(
                            gtex_genes
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# METHOD / STUDY SUMMARY
# =============================================================================

def method_summary(
    variant_cmp,
    gene_cmp,
    coloc_cmp,
):
    if variant_cmp.empty:
        return pd.DataFrame()

    rows = []

    for method, v in variant_cmp.groupby(
        "OT_METHOD",
        dropna=False,
    ):
        g = (
            gene_cmp[
                gene_cmp[
                    "OT_METHOD"
                ]
                ==
                method
            ]
            if not gene_cmp.empty
            else
            pd.DataFrame()
        )

        c = (
            coloc_cmp[
                coloc_cmp[
                    "OT_METHOD"
                ]
                ==
                method
            ]
            if not coloc_cmp.empty
            else
            pd.DataFrame()
        )

        is_pics = (
            "pics"
            in str(
                method
            ).lower()
        )

        row = {
            "OT_METHOD":
                method,

            "N_MATCHED_PAIRS":
                len(
                    v
                ),

            "N_DETAIL_AVAILABLE":
                int(
                    v[
                        "DETAIL_AVAILABLE"
                    ].fillna(
                        False
                    ).sum()
                ),

            "TOP_VARIANT_AGREEMENT_N":
                int(
                    v.get(
                        "TOP_VARIANT_AGREEMENT",
                        pd.Series(
                            False,
                            index=v.index,
                        ),
                    ).fillna(
                        False
                    ).sum()
                ),

            "TOP_VARIANT_AGREEMENT_RATE":
                float(
                    v.get(
                        "TOP_VARIANT_AGREEMENT",
                        pd.Series(
                            np.nan,
                            index=v.index,
                        ),
                    ).mean()
                ),

            # PICS interval Jaccard is intentionally not reported
            # as the primary interval metric.
            "MEDIAN_CS95_JACCARD":
                (
                    np.nan
                    if is_pics
                    else
                    safe_numeric(
                        v.get(
                            "CS95_JACCARD",
                            pd.Series(
                                dtype=float
                            ),
                        )
                    ).median()
                ),

            "MEDIAN_PIP_SPEARMAN":
                safe_numeric(
                    v.get(
                        "PIP_SPEARMAN",
                        pd.Series(
                            dtype=float
                        ),
                    )
                ).median(),

            "TOP1_GENE_AGREEMENT_N":
                (
                    int(
                        g[
                            "TOP1_GENE_AGREEMENT"
                        ].fillna(
                            False
                        ).sum()
                    )
                    if not g.empty
                    else
                    0
                ),

            "TOP1_GENE_AGREEMENT_RATE":
                (
                    float(
                        g[
                            "TOP1_GENE_AGREEMENT"
                        ].mean()
                    )
                    if not g.empty
                    else
                    np.nan
                ),

            "GWAS2M_TOP_IN_OT_TOP5_RATE":
                (
                    float(
                        g[
                            "GWAS2M_TOP_IN_OT_TOP5"
                        ].mean()
                    )
                    if not g.empty
                    else
                    np.nan
                ),

            "MEDIAN_COLOC_GENE_JACCARD":
                (
                    safe_numeric(
                        c.get(
                            "COLOC_GENE_JACCARD",
                            pd.Series(
                                dtype=float
                            ),
                        )
                    ).median()
                    if not c.empty
                    else
                    np.nan
                ),

            "MEDIAN_GTEX_GENE_JACCARD":
                (
                    safe_numeric(
                        c.get(
                            "GTEX_GENE_JACCARD",
                            pd.Series(
                                dtype=float
                            ),
                        )
                    ).median()
                    if not c.empty
                    else
                    np.nan
                ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def detailed_study_summary(
    variant_cmp,
    gene_cmp,
):
    if variant_cmp.empty:
        return pd.DataFrame()

    rows = []

    for study, v in variant_cmp.groupby(
        "STUDY_ID"
    ):
        g = (
            gene_cmp[
                gene_cmp[
                    "STUDY_ID"
                ]
                ==
                study
            ]
            if not gene_cmp.empty
            else
            pd.DataFrame()
        )

        rows.append(
            {
                "STUDY_ID":
                    study,

                "N_MATCHED_PAIRS":
                    len(
                        v
                    ),

                "N_DETAIL_AVAILABLE":
                    int(
                        v[
                            "DETAIL_AVAILABLE"
                        ].fillna(
                            False
                        ).sum()
                    ),

                "N_TOP_VARIANT_AGREEMENT":
                    int(
                        v.get(
                            "TOP_VARIANT_AGREEMENT",
                            pd.Series(
                                False,
                                index=v.index,
                            ),
                        ).fillna(
                            False
                        ).sum()
                    ),

                "MEDIAN_CS95_JACCARD":
                    safe_numeric(
                        v.get(
                            "CS95_JACCARD",
                            pd.Series(
                                dtype=float
                            ),
                        )
                    ).median(),

                "MEDIAN_PIP_SPEARMAN":
                    safe_numeric(
                        v.get(
                            "PIP_SPEARMAN",
                            pd.Series(
                                dtype=float
                            ),
                        )
                    ).median(),

                "N_TOP1_GENE_AGREEMENT":
                    (
                        int(
                            g[
                                "TOP1_GENE_AGREEMENT"
                            ].fillna(
                                False
                            ).sum()
                        )
                        if not g.empty
                        else
                        0
                    ),

                "N_GWAS2M_TOP_IN_OT_TOP5":
                    (
                        int(
                            g[
                                "GWAS2M_TOP_IN_OT_TOP5"
                            ].fillna(
                                False
                            ).sum()
                        )
                        if not g.empty
                        else
                        0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phenotype",
        required=True,
    )

    parser.add_argument(
        "--ancestry",
        required=True,
    )

    parser.add_argument(
        "--study",
        default=None,
    )

    parser.add_argument(
        "--ot-cache",
        default=None,
        help=(
            "Open Targets local cache. "
            "Default: resources/opentargets/26.09"
        ),
    )

    parser.add_argument(
        "--download-ot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Download/reuse official Open Targets "
            "26.09 Parquet datasets using rsync."
        ),
    )

    parser.add_argument(
        "--skip-detail",
        action="store_true",
        help=(
            "Only generate complete matched/unmatched "
            "locus inventory; do not query bulk OT data."
        ),
    )

    parser.add_argument(
        "--max-print",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    root = Path.cwd().resolve()

    ancestry_code, ancestry_label = ancestry_info(
        args.ancestry
    )

    paths = make_paths(
        root,
        args.phenotype,
        ancestry_label,
        args.ot_cache,
    )

    paths[
        "OUT"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    section(
        "GWAS2m STEP13 v2 - COMPLETE OPEN TARGETS / GENTROPY BENCHMARK"
    )

    print(
        f"Version             : {VERSION}"
    )

    print(
        f"Root                : {root}"
    )

    print(
        f"Phenotype           : {args.phenotype}"
    )

    print(
        f"Ancestry            : "
        f"{ancestry_label} ({ancestry_code})"
    )

    print(
        f"Open Targets release: {OPEN_TARGETS_RELEASE}"
    )

    print(
        f"Output              : {paths['OUT']}"
    )

    print(
        f"OT cache            : {paths['OT_CACHE']}"
    )

    # =========================================================================
    # LAYER A
    # =========================================================================

    (
        study_df,
        gwas_df,
        ot_df,
        pair_df,

    ) = load_step12(
        paths,
        args.study,
    )

    study_class = classify_studies(
        study_df
    )

    inventory = build_complete_inventory(
        study_class,
        gwas_df,
        ot_df,
        pair_df,
    )

    matched = inventory[
        "matched"
    ]

    gwas_only_comparable = inventory[
        "gwas_only_comparable"
    ]

    gwas_only_no_ot = inventory[
        "gwas_only_no_ot"
    ]

    ot_only_comparable = inventory[
        "ot_only_comparable"
    ]

    ot_only_not_finemapped = inventory[
        "ot_only_not_finemapped"
    ]

    complete_inventory = inventory[
        "inventory"
    ]

    # Save all core comparisons BEFORE any bulk-data work.
    core_tables = {
        "study_comparability.tsv":
            study_class,

        "matched_loci.tsv":
            matched,

        "gwas2m_only_comparable.tsv":
            gwas_only_comparable,

        "opentargets_only_comparable.tsv":
            ot_only_comparable,

        "gwas2m_only_no_ot_coverage.tsv":
            gwas_only_no_ot,

        "opentargets_only_gwas2m_not_finemapped.tsv":
            ot_only_not_finemapped,

        "complete_locus_inventory.tsv":
            complete_inventory,
    }

    for name, df in core_tables.items():
        df.to_csv(
            paths[
                "OUT"
            ]
            /
            name,

            sep="\t",

            index=False,
        )

    show(
        study_class,
        "TABLE 1 - STUDY COVERAGE / COMPARABILITY",
        args.max_print,
    )

    show(
        matched,
        "TABLE 2 - ALL MATCHED GWAS2m / OPEN TARGETS LOCUS PAIRS",
        args.max_print,
    )

    show(
        gwas_only_comparable,
        "TABLE 3 - GWAS2m-ONLY LOCI IN COMPARABLE STUDIES",
        args.max_print,
    )

    show(
        ot_only_comparable,
        "TABLE 4 - OPEN-TARGETS-ONLY LOCI IN COMPARABLE STUDIES",
        args.max_print,
    )

    show(
        gwas_only_no_ot,
        "TABLE 5 - GWAS2m LOCI WHERE OPEN TARGETS HAS NO STUDY CREDIBLE SETS",
        args.max_print,
    )

    show(
        ot_only_not_finemapped,
        "TABLE 6 - OPEN TARGETS LOCI WHERE GWAS2m WAS NOT FINE-MAPPED",
        args.max_print,
    )

    section(
        "CORE LOCUS BENCHMARK SUMMARY"
    )

    print(
        f"Matched locus pairs                     : "
        f"{len(matched):,}"
    )

    print(
        f"GWAS2m-only, comparable studies         : "
        f"{len(gwas_only_comparable):,}"
    )

    print(
        f"OpenTargets-only, comparable studies    : "
        f"{len(ot_only_comparable):,}"
    )

    print(
        f"GWAS2m-only, OT has no study CS         : "
        f"{len(gwas_only_no_ot):,}"
    )

    print(
        f"OT-only, GWAS2m not fine-mapped         : "
        f"{len(ot_only_not_finemapped):,}"
    )

    if args.skip_detail:
        section(
            "STEP13 CORE LOCUS INVENTORY COMPLETE"
        )

        print(
            "Detailed OT bulk-data layer skipped."
        )

        return

    # =========================================================================
    # LAYER B -- OFFICIAL BULK DATA
    # =========================================================================

    section(
        "PREPARING OFFICIAL OPEN TARGETS 26.09 DATASETS"
    )

    resolved = prepare_ot_datasets(
        paths[
            "OT_CACHE"
        ],
        args.download_ot,
    )

    missing_required = [
        x
        for x in REQUIRED_OT_DATASETS
        if x not in resolved
    ]

    if missing_required:
        print(
            "\n[WARN] Detailed Open Targets layer "
            "cannot run because required datasets are missing:"
        )

        for x in missing_required:
            print(
                f"  - {x}"
            )

        print(
            "\nCore matched/unmatched locus inventory "
            "HAS been completed and saved."
        )

        return

    require_duckdb()

    if spearmanr is None:
        print(
            "[WARN] SciPy is not installed. "
            "PIP correlations will be NaN."
        )

    ot_ids = sorted(
        set(
            matched[
                "OT_STUDY_LOCUS_ID"
            ].dropna().astype(str)
        )
    )

    # -------------------------------------------------------------------------
    # Credible sets
    # -------------------------------------------------------------------------

    credible_subset, _ = subset_dataset_by_values(
        resolved[
            "credible_set"
        ],
        [
            "studyLocusId",
            "study_locus_id",
        ],
        ot_ids,
    )

    credible_subset.to_csv(
        paths[
            "OUT"
        ]
        /
        "ot_credible_set_subset.tsv",

        sep="\t",

        index=False,
    )

    credible_meta = credible_set_metadata(
        credible_subset
    )

    ot_variants = flatten_credible_set_variants(
        credible_subset
    )

    # Optional separate locus dataset fallback.
    if (
        ot_variants.empty
        and
        "locus"
        in resolved
    ):
        locus_subset, _ = subset_dataset_by_values(
            resolved[
                "locus"
            ],
            [
                "studyLocusId",
                "study_locus_id",
            ],
            ot_ids,
        )

        ot_variants = flatten_locus_dataset(
            locus_subset
        )

    ot_variants.to_csv(
        paths[
            "OUT"
        ]
        /
        "ot_variant_subset.tsv",

        sep="\t",

        index=False,
    )

    # -------------------------------------------------------------------------
    # L2G
    # -------------------------------------------------------------------------

    l2g_subset, _ = subset_dataset_by_values(
        resolved[
            "l2g_prediction"
        ],
        [
            "studyLocusId",
            "study_locus_id",
        ],
        ot_ids,
    )

    l2g = flatten_l2g(
        l2g_subset
    )

    l2g.to_csv(
        paths[
            "OUT"
        ]
        /
        "ot_l2g_subset.tsv",

        sep="\t",

        index=False,
    )

    # -------------------------------------------------------------------------
    # Colocalisation
    # -------------------------------------------------------------------------

    coloc_subset, _ = subset_dataset_by_values(
        resolved[
            "colocalisation"
        ],
        [
            "studyLocusId",
            "study_locus_id",
        ],
        ot_ids,
    )

    coloc = flatten_colocalisation(
        coloc_subset
    )

    coloc.to_csv(
        paths[
            "OUT"
        ]
        /
        "ot_colocalisation_subset.tsv",

        sep="\t",

        index=False,
    )

    # -------------------------------------------------------------------------
    # Resolve molQTL other credible sets -> gene/study/tissue
    # -------------------------------------------------------------------------

    other_ids = sorted(
        set(
            coloc[
                "OTHER_STUDY_LOCUS_ID"
            ].dropna().astype(str)
        )
        if not coloc.empty
        else
        []
    )

    other_cs = pd.DataFrame()

    other_meta = pd.DataFrame()

    study_meta = pd.DataFrame()

    if other_ids:
        other_cs, _ = subset_dataset_by_values(
            resolved[
                "credible_set"
            ],
            [
                "studyLocusId",
                "study_locus_id",
            ],
            other_ids,
        )

        other_meta = credible_set_metadata(
            other_cs
        )

        other_study_ids = sorted(
            set(
                other_meta[
                    "OT_STUDY_ID"
                ].dropna().astype(str)
            )
            if (
                not other_meta.empty
                and
                "OT_STUDY_ID"
                in other_meta.columns
            )
            else
            []
        )

        if other_study_ids:
            study_subset, _ = subset_dataset_by_values(
                resolved[
                    "study"
                ],
                [
                    "studyId",
                    "study_id",
                    "id",
                ],
                other_study_ids,
            )

            study_meta = study_metadata(
                study_subset
            )

    coloc_resolved = resolve_colocalisation(
        coloc,
        other_meta,
        study_meta,
    )

    coloc_resolved.to_csv(
        paths[
            "OUT"
        ]
        /
        "ot_colocalisation_resolved.tsv",

        sep="\t",

        index=False,
    )

    # =========================================================================
    # DETAILED COMPARISONS
    # =========================================================================

    variant_cmp = compare_variants(
        matched,
        ot_variants,
        paths,
    )

    gene_cmp = compare_genes(
        matched,
        l2g,
        paths,
    )

    coloc_cmp = compare_coloc_genes(
        matched,
        coloc_resolved,
        paths,
    )

    methods = method_summary(
        variant_cmp,
        gene_cmp,
        coloc_cmp,
    )

    studies = detailed_study_summary(
        variant_cmp,
        gene_cmp,
    )

    detailed_tables = {
        "variant_pip_comparison.tsv":
            variant_cmp,

        "gene_l2g_comparison.tsv":
            gene_cmp,

        "colocalisation_gene_comparison.tsv":
            coloc_cmp,

        "method_summary.tsv":
            methods,

        "detailed_study_summary.tsv":
            studies,
    }

    for name, df in detailed_tables.items():
        df.to_csv(
            paths[
                "OUT"
            ]
            /
            name,

            sep="\t",

            index=False,
        )

    show(
        methods,
        "TABLE 7 - DETAILED BENCHMARK BY OPEN TARGETS METHOD",
        args.max_print,
    )

    show(
        studies,
        "TABLE 8 - DETAILED BENCHMARK BY STUDY",
        args.max_print,
    )

    variant_columns = [
        col
        for col in [
            "STUDY_ID",
            "GWAS2M_LOCUS_ID",
            "OT_STUDY_LOCUS_ID",
            "OT_METHOD",
            "MATCH_BASIS",
            "GWAS2M_TOP_VARIANT",
            "OT_TOP_VARIANT",
            "TOP_VARIANT_AGREEMENT",
            "N_GWAS2M_CS95",
            "N_OT_CS95",
            "N_SHARED_CS95",
            "CS95_JACCARD",
            "PIP_SPEARMAN",
            "PIP_PEARSON",
            "PIP_MAE",
        ]
        if col
        in variant_cmp.columns
    ]

    show(
        variant_cmp[
            variant_columns
        ]
        if not variant_cmp.empty
        else
        variant_cmp,
        "TABLE 9 - VARIANT / PIP / CREDIBLE-SET COMPARISON",
        args.max_print,
    )

    gene_columns = [
        col
        for col in [
            "STUDY_ID",
            "GWAS2M_LOCUS_ID",
            "OT_STUDY_LOCUS_ID",
            "OT_METHOD",
            "GWAS2M_TOP_GENE",
            "OT_TOP_GENE",
            "OT_TOP1_SCORE",
            "TOP1_GENE_AGREEMENT",
            "GWAS2M_TOP_IN_OT_TOP3",
            "GWAS2M_TOP_IN_OT_TOP5",
            "TOP5_GENE_JACCARD",
            "GWAS2M_TOP5",
            "OT_TOP5",
        ]
        if col
        in gene_cmp.columns
    ]

    show(
        gene_cmp[
            gene_columns
        ]
        if not gene_cmp.empty
        else
        gene_cmp,
        "TABLE 10 - GWAS2m GENES vs OPEN TARGETS L2G",
        args.max_print,
    )

    coloc_columns = [
        col
        for col in [
            "STUDY_ID",
            "GWAS2M_LOCUS_ID",
            "OT_STUDY_LOCUS_ID",
            "OT_METHOD",
            "N_GWAS2M_COLOC_GENES",
            "N_OT_MOLQTL_COLOC_GENES",
            "N_SHARED_COLOC_GENES",
            "COLOC_GENE_JACCARD",
            "N_OT_GTEX_COLOC_GENES",
            "N_SHARED_GTEX_GENES",
            "GTEX_GENE_JACCARD",
            "GWAS2M_MAX_LCLPP",
            "OT_MAX_H4",
            "OT_MAX_CLPP",
        ]
        if col
        in coloc_cmp.columns
    ]

    show(
        coloc_cmp[
            coloc_columns
        ]
        if not coloc_cmp.empty
        else
        coloc_cmp,
        "TABLE 11 - MOLECULAR-QTL COLOCALISATION COMPARISON",
        args.max_print,
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "step":
            13,

        "version":
            VERSION,

        "open_targets_release":
            OPEN_TARGETS_RELEASE,

        "retrieved_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "phenotype":
            args.phenotype,

        "ancestry_code":
            ancestry_code,

        "ancestry_label":
            ancestry_label,

        "study_filter":
            args.study,

        "ot_cache":
            str(
                paths[
                    "OT_CACHE"
                ]
            ),

        "resolved_ot_datasets":
            {
                key:
                    str(
                        value
                    )
                for key, value
                in resolved.items()
            },

        "n_matched_pairs":
            len(
                matched
            ),

        "n_gwas2m_only_comparable":
            len(
                gwas_only_comparable
            ),

        "n_ot_only_comparable":
            len(
                ot_only_comparable
            ),

        "n_gwas2m_only_no_ot_coverage":
            len(
                gwas_only_no_ot
            ),

        "n_ot_only_gwas2m_not_finemapped":
            len(
                ot_only_not_finemapped
            ),

        "n_ot_variant_rows":
            len(
                ot_variants
            ),

        "n_ot_l2g_rows":
            len(
                l2g
            ),

        "n_ot_coloc_rows":
            len(
                coloc
            ),
    }

    (
        paths[
            "OUT"
        ]
        /
        "benchmark_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            default=json_safe,
        )
        +
        "\n",

        encoding="utf-8",
    )

    section(
        "STEP13 COMPLETE"
    )

    print(
        f"Matched pairs                         : "
        f"{len(matched):,}"
    )

    print(
        f"GWAS2m-only comparable                : "
        f"{len(gwas_only_comparable):,}"
    )

    print(
        f"OpenTargets-only comparable           : "
        f"{len(ot_only_comparable):,}"
    )

    print(
        f"GWAS2m-only no OT coverage            : "
        f"{len(gwas_only_no_ot):,}"
    )

    print(
        f"OT-only GWAS2m not fine-mapped        : "
        f"{len(ot_only_not_finemapped):,}"
    )

    print(
        f"OT variant rows loaded                : "
        f"{len(ot_variants):,}"
    )

    print(
        f"OT L2G rows loaded                    : "
        f"{len(l2g):,}"
    )

    print(
        f"OT colocalisation rows loaded         : "
        f"{len(coloc):,}"
    )

    print()

    print(
        "Interpret SuSiE-inf and PICS separately."
    )

    print(
        "Open Targets is an external comparator, "
        "not a ground-truth label."
    )

    print(
        "Coverage gaps are reported separately from "
        "true comparable-study disagreements."
    )


if __name__ == "__main__":
    main()

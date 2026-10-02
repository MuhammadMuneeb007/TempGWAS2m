#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from scipy.stats import rankdata


VERSION = "4.0.0-fast-common-loci"


# =============================================================================
# ANCESTRY
# =============================================================================

ANCESTRY = {

    "eur":
        ("EUR", "European"),

    "european":
        ("EUR", "European"),

    "afr":
        ("AFR", "African"),

    "african":
        ("AFR", "African"),

    "eas":
        ("EAS", "East Asian"),

    "east asian":
        ("EAS", "East Asian"),

    "sas":
        ("SAS", "South Asian"),

    "south asian":
        ("SAS", "South Asian"),

    "amr":
        ("AMR", "Hispanic or Latin American"),

    "hispanic":
        ("AMR", "Hispanic or Latin American"),

    "latino":
        ("AMR", "Hispanic or Latin American"),

    "hispanic or latin american":
        ("AMR", "Hispanic or Latin American"),
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def norm(
    value,
):

    return re.sub(

        r"\s+",

        " ",

        re.sub(

            r"[^a-z0-9]+",

            " ",

            str(
                value
            ).lower(),
        ),

    ).strip()


def slug(
    value,
):

    return norm(
        value
    ).replace(
        " ",
        "_",
    )


def ancestry_info(
    value,
):

    key = norm(
        value
    )

    if key not in ANCESTRY:

        raise SystemExit(
            f"Unsupported ancestry: {value}"
        )

    return ANCESTRY[
        key
    ]


def section(
    title,
    char="=",
):

    print()

    print(
        char
        *
        150
    )

    print(
        title
    )

    print(
        char
        *
        150
    )


def show(
    df,
    title,
    max_rows=None,
):

    section(
        title,
        "-",
    )

    if df is None:

        print(
            "(no table)"
        )

        return

    if isinstance(
        df,
        pl.DataFrame,
    ):

        df = df.to_pandas()

    if len(
        df
    ) == 0:

        print(
            "(no rows)"
        )

        return

    if (
        max_rows
        and
        len(df)
        >
        max_rows
    ):

        shown = df.head(
            max_rows
        )

    else:

        shown = df

    with pd.option_context(

        "display.max_rows",
        None,

        "display.max_columns",
        None,

        "display.width",
        600,

        "display.max_colwidth",
        70,

        "display.expand_frame_repr",
        False,

        "display.float_format",
        lambda x:
            f"{x:.6g}",

    ):

        print(
            shown.to_string(
                index=False
            )
        )

    if (
        max_rows
        and
        len(df)
        >
        max_rows
    ):

        print(
            f"\n[showing "
            f"{max_rows}/"
            f"{len(df)} rows]"
        )


def first_file(
    directory,
    patterns,
):

    if not directory.exists():

        return None

    for pattern in patterns:

        if "*" in pattern:

            hits = sorted(

                path

                for path
                in directory.glob(
                    pattern
                )

                if (
                    path.is_file()
                    and
                    path.stat().st_size
                    >
                    0
                )
            )

            if hits:

                return hits[
                    0
                ]

        else:

            path = (
                directory
                /
                pattern
            )

            if (
                path.exists()
                and
                path.is_file()
                and
                path.stat().st_size
                >
                0
            ):

                return path

    return None


def safe_json(
    path,
):

    if path is None:

        return {}

    try:

        return json.loads(

            path.read_text(

                encoding="utf-8",

                errors="replace",
            )
        )

    except Exception:

        return {}


def pick(
    data,
    *keys,
    default=np.nan,
):

    for key in keys:

        if (
            key in data
            and
            data[
                key
            ]
            not in
            (
                None,
                "",
            )
        ):

            return data[
                key
            ]

    return default


def nint(
    value,
):

    try:

        return int(
            float(
                value
            )
        )

    except Exception:

        return np.nan


def nfloat(
    value,
):

    try:

        return float(
            value
        )

    except Exception:

        return np.nan


# =============================================================================
# FAST POLARS FILE READER
# =============================================================================

def read_pl(
    path,
):

    if path is None:

        return pl.DataFrame()

    try:

        return pl.read_csv(

            path,

            separator="\t",

            infer_schema_length=2000,

            ignore_errors=True,

            null_values=[

                "NA",

                "NaN",

                "nan",

                "None",

                "",
            ],
        )

    except Exception as error:

        print(

            f"[WARN] "
            f"{path}: "
            f"{type(error).__name__}: "
            f"{error}"
        )

        return pl.DataFrame()


# =============================================================================
# PROJECT PATHS
# =============================================================================

def roots_for(
    root,
    phenotype,
    ancestry_label,
):

    phenotype_slug = slug(
        phenotype
    )

    ancestry_slug = slug(
        ancestry_label
    )

    return {

        "S02":
            root
            /
            "02_summary_stats"
            /
            phenotype_slug
            /
            ancestry_slug,

        "S05":
            root
            /
            "05_ld_clumping"
            /
            phenotype_slug
            /
            ancestry_slug,

        "S06":
            root
            /
            "06_finemapping"
            /
            phenotype_slug
            /
            ancestry_slug,

        "S07":
            root
            /
            "07_annotation"
            /
            phenotype_slug
            /
            ancestry_slug,

        "S08":
            root
            /
            "08_qtl"
            /
            phenotype_slug
            /
            ancestry_slug,

        "S09":
            root
            /
            "09_splicing"
            /
            phenotype_slug
            /
            ancestry_slug,

        "S10":
            root
            /
            "10_pangolin"
            /
            phenotype_slug
            /
            ancestry_slug,

        "S11":
            root
            /
            "11_coloc"
            /
            phenotype_slug
            /
            ancestry_slug,
    }


def discover_studies(
    roots,
):

    studies = set()

    bases = [

        roots[
            "S02"
        ]
        /
        "qc",

        roots[
            "S05"
        ],

        roots[
            "S06"
        ],

        roots[
            "S07"
        ],

        roots[
            "S08"
        ],

        roots[
            "S09"
        ],

        roots[
            "S10"
        ],

        roots[
            "S11"
        ],
    ]

    for base in bases:

        if not base.exists():

            continue

        for directory in base.glob(
            "GCST*"
        ):

            if (
                directory.is_dir()
                and
                re.fullmatch(
                    r"GCST\d+",
                    directory.name,
                )
            ):

                studies.add(
                    directory.name
                )

    return sorted(
        studies
    )


def study_paths(
    roots,
    study,
):

    return {

        "S03":
            roots[
                "S02"
            ]
            /
            "qc"
            /
            study,

        "S05":
            roots[
                "S05"
            ]
            /
            study,

        "S06":
            roots[
                "S06"
            ]
            /
            study,

        "S07":
            roots[
                "S07"
            ]
            /
            study,

        "S08":
            roots[
                "S08"
            ]
            /
            study,

        "S09":
            roots[
                "S09"
            ]
            /
            study,

        "S10":
            roots[
                "S10"
            ]
            /
            study,

        "S11":
            roots[
                "S11"
            ]
            /
            study,
    }


def load_summary(
    directory,
    names,
):

    return safe_json(

        first_file(
            directory,
            names,
        )
    )


# =============================================================================
# TABLE 1
# PIPELINE AUDIT
# =============================================================================

def pipeline_row(
    study,
    sp,
):

    qc = load_summary(

        sp[
            "S03"
        ],

        [
            f"{study}_QC_summary.json",
            "*QC_summary.json",
        ],
    )

    clumping = load_summary(

        sp[
            "S05"
        ],

        [
            f"{study}_clumping_summary.json",
            "clumping_summary.json",
            "*clumping_summary.json",
        ],
    )

    finemap = load_summary(

        sp[
            "S06"
        ],

        [
            "finemapping_summary.json",
        ],
    )

    vep = load_summary(

        sp[
            "S07"
        ],

        [
            "annotation_summary.json",
            "vep_summary.json",
        ],
    )

    qtl = load_summary(

        sp[
            "S08"
        ],

        [
            "qtl_summary.json",
        ],
    )

    spliceai = load_summary(

        sp[
            "S09"
        ],

        [
            "spliceai_summary.json",
        ],
    )

    pangolin = load_summary(

        sp[
            "S10"
        ],

        [
            "pangolin_summary.json",
        ],
    )

    coloc = load_summary(

        sp[
            "S11"
        ],

        [
            "coloc_summary.json",
        ],
    )

    return {

        "STUDY":
            study,

        "QC":
            pick(
                qc,
                "QC_STATUS",
                "STATUS",
                default=(
                    "MISSING"
                    if not qc
                    else
                    "PRESENT"
                ),
            ),

        "N_AFTER_QC":
            nint(
                pick(
                    qc,
                    "N_AFTER_QC",
                    "N_VARIANTS_AFTER_QC",
                )
            ),

        "N_GWS":
            nint(
                pick(
                    qc,
                    "N_GENOME_WIDE_SIGNIFICANT",
                    "N_GWS_QC",
                    "N_GWS",
                )
            ),

        "N_LEADS":
            nint(
                pick(
                    clumping,
                    "N_LEAD_VARIANTS",
                )
            ),

        "N_LOCI_OK":
            nint(
                pick(
                    finemap,
                    "N_LOCI_SUCCESS",
                )
            ),

        "VEP_SELECTED":
            nint(
                pick(
                    vep,
                    "N_SELECTED_VARIANTS",
                )
            ),

        "EQTL_VARIANTS":
            nint(
                pick(
                    qtl,
                    "N_EQTL_VARIANTS",
                )
            ),

        "SQTL_VARIANTS":
            nint(
                pick(
                    qtl,
                    "N_SQTL_VARIANTS",
                )
            ),

        "SPLICEAI_STATUS":
            pick(
                spliceai,
                "STATUS",
                default=(
                    "MISSING"
                    if not spliceai
                    else
                    "PRESENT"
                ),
            ),

        "SPLICEAI_SCORED":
            nint(
                pick(
                    spliceai,
                    "N_SPLICEAI_SCORED_VARIANTS",
                    "N_SPLICEAI_SCORED",
                )
            ),

        "SPLICEAI_GE_020":
            nint(
                pick(
                    spliceai,
                    "N_SPLICEAI_GE_0_20",
                )
            ),

        "PANGOLIN_STATUS":
            pick(
                pangolin,
                "STATUS",
                default=(
                    "MISSING"
                    if not pangolin
                    else
                    "PRESENT"
                ),
            ),

        "PANGOLIN_SCORED":
            nint(
                pick(
                    pangolin,
                    "N_PANGOLIN_SCORED",
                )
            ),

        "PANGOLIN_TOP5":
            nint(
                pick(
                    pangolin,
                    "N_PANGOLIN_TOP_5PCT",
                )
            ),

        "MULTISOURCE_SPLICE":
            nint(
                pick(
                    pangolin,
                    "N_MULTI_SOURCE_GE_2",
                )
            ),

        "COLOC_STATUS":
            pick(
                coloc,
                "STATUS",
                default=(
                    "MISSING"
                    if not coloc
                    else
                    "PRESENT"
                ),
            ),

        "SHARED_VARIANTS":
            nint(
                pick(
                    coloc,
                    "N_UNIQUE_SHARED_VARIANTS",
                )
            ),

        "COLOC_LOCI":
            nint(
                pick(
                    coloc,
                    "N_COLOC_LOCI",
                )
            ),

        "COLOC_GENES":
            nint(
                pick(
                    coloc,
                    "N_GENES",
                )
            ),

        "BEST_LCLPP":
            nfloat(
                pick(
                    coloc,
                    "MAX_LCLPP",
                )
            ),

        "PAIR_LCLPP_GE_001":
            nint(
                pick(
                    coloc,
                    "N_PAIR_LCLPP_GE_0_01",
                )
            ),
    }


# =============================================================================
# STEP 06 LOCUS DEFINITIONS
# =============================================================================

def loci_definition(
    sp,
    study,
):

    path = first_file(

        sp[
            "S06"
        ],

        [
            "loci_definition.tsv",
        ],
    )

    table = read_pl(
        path
    )

    if (
        table.height
        ==
        0
        or
        "LOCUS_ID"
        not in table.columns
    ):

        return pd.DataFrame()

    columns = [

        column

        for column
        in [

            "LOCUS_ID",

            "CHR",

            "LOCUS_START",

            "LOCUS_END",

            "N_INDEPENDENT_SIGNALS",

            "LEAD_IDS",

            "LEAD_POSITIONS",

            "MIN_LEAD_P",

        ]

        if column
        in table.columns
    ]

    output = (

        table
        .select(
            columns
        )
        .to_pandas()
    )

    output[
        "STUDY"
    ] = study

    for column in [

        "CHR",

        "LOCUS_START",

        "LOCUS_END",

    ]:

        if column in output.columns:

            output[
                column
            ] = pd.to_numeric(

                output[
                    column
                ],

                errors="coerce",
            )

    return output


# =============================================================================
# STEP 06 CREDIBLE SET SUMMARY
# =============================================================================

def cs_summary(
    sp,
):

    path = first_file(

        sp[
            "S06"
        ],

        [
            "*_95pct_credible_sets.tsv",
            "*_95pct_credible_sets.tsv.gz",
        ],
    )

    table = read_pl(
        path
    )

    if (
        table.height
        ==
        0
        or
        "LOCUS_ID"
        not in table.columns
    ):

        return pd.DataFrame()

    if "PIP" in table.columns:

        table = table.with_columns(

            pl.col(
                "PIP"
            )
            .cast(
                pl.Float64,
                strict=False,
            )
        )

    aggregations = [

        pl.len()
        .alias(
            "CS95_N"
        )
    ]

    if "PIP" in table.columns:

        aggregations += [

            pl.col(
                "PIP"
            )
            .max()
            .alias(
                "CS95_MAX_PIP"
            ),

            (
                pl.col(
                    "PIP"
                )
                >=
                0.10
            )
            .sum()
            .alias(
                "CS95_PIP_GE_010"
            ),

            (
                pl.col(
                    "PIP"
                )
                >=
                0.50
            )
            .sum()
            .alias(
                "CS95_PIP_GE_050"
            ),
        ]

    return (

        table
        .group_by(
            "LOCUS_ID"
        )
        .agg(
            aggregations
        )
        .to_pandas()
    )


# =============================================================================
# STEP 07 FUNCTIONAL CONTEXT
# =============================================================================

def vep_summary(
    sp,
):

    path = first_file(

        sp[
            "S07"
        ],

        [
            "*_VEP_variant_summary.tsv",
            "*_VEP_variant_summary.tsv.gz",
        ],
    )

    table = read_pl(
        path
    )

    if (
        table.height
        ==
        0
        or
        "LOCUS_ID"
        not in table.columns
    ):

        return pd.DataFrame()

    if (
        "FUNCTIONAL_CATEGORY"
        not in table.columns
    ):

        table = table.with_columns(

            pl.lit(
                ""
            )
            .alias(
                "FUNCTIONAL_CATEGORY"
            )
        )

    category = (

        pl.col(
            "FUNCTIONAL_CATEGORY"
        )
        .fill_null(
            ""
        )
        .cast(
            pl.Utf8
        )
        .str.to_uppercase()
    )

    if "BIOTYPE" in table.columns:

        biotype_column = (
            "BIOTYPE"
        )

    elif "ALL_BIOTYPES" in table.columns:

        biotype_column = (
            "ALL_BIOTYPES"
        )

    else:

        biotype_column = (
            None
        )

    aggregations = [

        pl.len()
        .alias(
            "VEP_N"
        ),

        (
            category
            ==
            "CODING"
        )
        .sum()
        .alias(
            "CODING_N"
        ),

        (
            category
            !=
            "CODING"
        )
        .sum()
        .alias(
            "NONCODING_N"
        ),

        (
            category
            ==
            "INTRONIC"
        )
        .sum()
        .alias(
            "INTRONIC_N"
        ),

        (
            category
            ==
            "INTERGENIC"
        )
        .sum()
        .alias(
            "INTERGENIC_N"
        ),

        (
            category
            ==
            "NONCODING_RNA"
        )
        .sum()
        .alias(
            "NCRNA_N"
        ),

        (
            category
            ==
            "SPLICING"
        )
        .sum()
        .alias(
            "VEP_SPLICE_N"
        ),
    ]

    if biotype_column:

        biotype = (

            pl.col(
                biotype_column
            )
            .fill_null(
                ""
            )
            .cast(
                pl.Utf8
            )
            .str.to_lowercase()
        )

        aggregations.append(

            biotype
            .str.contains(
                "lncrna|lincrna|antisense|processed_transcript"
            )
            .sum()
            .alias(
                "LNCRNA_N"
            )
        )

    return (

        table
        .group_by(
            "LOCUS_ID"
        )
        .agg(
            aggregations
        )
        .to_pandas()
    )


# =============================================================================
# STEP 09 - SPLICEAI
# READ DIRECTLY FROM STEP09
# =============================================================================

def spliceai_summary(
    sp,
    study,
):

    path = first_file(

        sp[
            "S09"
        ],

        [
            f"{study}_SpliceAI_locus_summary.tsv",
        ],
    )

    table = read_pl(
        path
    )

    if table.height == 0:

        return pd.DataFrame()

    output = table.to_pandas()

    rename = {

        "N_INPUT_VARIANTS":
            "S9_INPUT_N",

        "N_SPLICEAI_SCORED":
            "SPLICEAI_SCORED_N",

        "N_SPLICEAI_GE_0_20":
            "SPLICEAI_GE_020_N",

        "N_SPLICEAI_GE_0_50":
            "SPLICEAI_GE_050_N",

        "N_SPLICEAI_GE_0_80":
            "SPLICEAI_GE_080_N",

        "N_GTEX_SQTL":
            "S9_GTEX_SQTL_N",

        "N_SPLICEAI_GE_0_20_AND_SQTL":
            "SPLICEAI020_AND_SQTL_N",

        "MAX_SPLICEAI_DS":
            "MAX_SPLICEAI_DS",
    }

    return output.rename(

        columns={

            key:
                value

            for key, value
            in rename.items()

            if key
            in output.columns
        }
    )


# =============================================================================
# STEP 10 - PANGOLIN
# READ DIRECTLY FROM STEP10
# =============================================================================

def pangolin_summary(
    sp,
    study,
):

    path = first_file(

        sp[
            "S10"
        ],

        [
            f"{study}_Pangolin_locus_summary.tsv",
        ],
    )

    table = read_pl(
        path
    )

    if table.height == 0:

        return pd.DataFrame()

    output = table.to_pandas()

    rename = {

        "N_VARIANTS":
            "S10_INPUT_N",

        "N_PANGOLIN_SCORED":
            "PANGOLIN_SCORED_N",

        "N_PANGOLIN_TOP_5PCT":
            "PANGOLIN_TOP5_N",

        "N_GTEX_SQTL":
            "S10_GTEX_SQTL_N",

        "N_VEP_SPLICE":
            "S10_VEP_SPLICE_N",

        "N_MULTI_SOURCE_GE_2":
            "MULTISOURCE_GE2_N",

        "MAX_PANGOLIN_ABS":
            "MAX_PANGOLIN_ABS",

        "MAX_SPLICEAI_DS":
            "S10_MAX_SPLICEAI_DS",
    }

    return output.rename(

        columns={

            key:
                value

            for key, value
            in rename.items()

            if key
            in output.columns
        }
    )


# =============================================================================
# STEP 11 GENE FIX
# =============================================================================

def effective_gene(
    row,
):

    gene = str(

        row.get(
            "QTL_GENE",
            "",
        )

    ).strip()

    missing = {

        "",

        "none",

        "nan",

        "na",

        ".",
    }

    if gene.lower() not in missing:

        return gene

    qtl_type = str(

        row.get(
            "QTL_TYPE",
            "",
        )

    ).lower()

    phenotype = str(

        row.get(
            "QTL_PHENOTYPE",
            "",
        )

    ).strip()

    # Important:
    # GTEx eQTL SuSiE sometimes stores the gene ID
    # in the phenotype field rather than QTL_GENE.

    if (
        qtl_type
        ==
        "eqtl"
        and
        phenotype.lower()
        not in missing
    ):

        return phenotype

    match = re.search(

        r"(ENSG\d+(?:\.\d+)?)",

        phenotype,
    )

    if match:

        return match.group(
            1
        )

    return ""


# =============================================================================
# STEP11 PAIR TABLE
# =============================================================================

def pair_table(
    sp,
    study,
):

    path = first_file(

        sp[
            "S11"
        ],

        [
            f"{study}_GTEx_SuSiE_gene_tissue_colocalization.tsv",
        ],
    )

    table = read_pl(
        path
    )

    if table.height == 0:

        return pd.DataFrame()

    data = table.to_pandas()

    data[
        "STUDY"
    ] = study

    data[
        "EFFECTIVE_GENE"
    ] = data.apply(

        effective_gene,

        axis=1,
    )

    for column in [

        "LCLPP_MULTICAUSAL",

        "TOP_SHARED_GWAS_PIP",

        "TOP_SHARED_QTL_PIP",

        "TOP_SHARED_VCLPP",

        "MAX_VCLPP",

    ]:

        if column in data.columns:

            data[
                column
            ] = pd.to_numeric(

                data[
                    column
                ],

                errors="coerce",
            )

    return data


# =============================================================================
# BEST eQTL / sQTL PER LOCUS
# =============================================================================

def best_qtl_by_locus(
    pairs,
    qtl_type,
):

    if pairs.empty:

        return pd.DataFrame()

    data = pairs[

        pairs[
            "QTL_TYPE"
        ]
        .astype(
            str
        )
        .str.lower()
        ==
        qtl_type.lower()

    ].copy()

    if data.empty:

        return pd.DataFrame()

    data = (

        data

        .sort_values(

            [
                "LOCUS_ID",
                "LCLPP_MULTICAUSAL",
            ],

            ascending=[
                True,
                False,
            ],

            kind="stable",
        )

        .drop_duplicates(

            "LOCUS_ID",

            keep="first",
        )
    )

    prefix = (

        "EQTL"

        if qtl_type.lower()
        ==
        "eqtl"

        else

        "SQTL"
    )

    keep = [

        "LOCUS_ID",

        "EFFECTIVE_GENE",

        "TISSUE",

        "LCLPP_MULTICAUSAL",

        "TOP_SHARED_VARIANT",

        "TOP_SHARED_GWAS_PIP",

        "TOP_SHARED_QTL_PIP",

        "TOP_SHARED_VCLPP",

        "COLOC_SCREEN_CLASS",
    ]

    keep = [

        column

        for column
        in keep

        if column
        in data.columns
    ]

    data = data[
        keep
    ]

    return data.rename(

        columns={

            column:
                f"{prefix}_{column}"

            for column
            in data.columns

            if column
            !=
            "LOCUS_ID"
        }
    )


# =============================================================================
# MAIN PER-LOCUS REPORT
# =============================================================================

def locus_report(
    sp,
    study,
):

    base = loci_definition(

        sp,

        study,
    )

    if base.empty:

        return pd.DataFrame()

    extra_tables = [

        cs_summary(
            sp
        ),

        vep_summary(
            sp
        ),

        spliceai_summary(
            sp,
            study,
        ),

        pangolin_summary(
            sp,
            study,
        ),
    ]

    for extra in extra_tables:

        if (
            not extra.empty
            and
            "LOCUS_ID"
            in extra.columns
        ):

            base = base.merge(

                extra,

                on="LOCUS_ID",

                how="left",
            )

    pairs = pair_table(

        sp,

        study,
    )

    for extra in [

        best_qtl_by_locus(
            pairs,
            "eQTL",
        ),

        best_qtl_by_locus(
            pairs,
            "sQTL",
        ),

    ]:

        if not extra.empty:

            base = base.merge(

                extra,

                on="LOCUS_ID",

                how="left",
            )

    eqtl = pd.to_numeric(

        base.get(

            "EQTL_LCLPP_MULTICAUSAL",

            pd.Series(
                np.nan,
                index=base.index,
            ),
        ),

        errors="coerce",
    )

    sqtl = pd.to_numeric(

        base.get(

            "SQTL_LCLPP_MULTICAUSAL",

            pd.Series(
                np.nan,
                index=base.index,
            ),
        ),

        errors="coerce",
    )

    matrix = np.vstack(

        [
            eqtl.to_numpy(),
            sqtl.to_numpy(),
        ]
    )

    with np.errstate(
        all="ignore"
    ):

        best = np.nanmax(

            matrix,

            axis=0,
        )

    best = np.where(

        np.isfinite(
            best
        ),

        best,

        -np.inf,
    )

    base[
        "BEST_LOCUS_LCLPP"
    ] = np.where(

        np.isfinite(
            best
        ),

        best,

        np.nan,
    )

    base[
        "EVIDENCE_RANK"
    ] = rankdata(

        -best,

        method="min",

    ).astype(
        int
    )

    columns = [

        "STUDY",

        "EVIDENCE_RANK",

        "LOCUS_ID",

        "CHR",

        "LOCUS_START",

        "LOCUS_END",

        "LEAD_POSITIONS",

        "CS95_N",

        "CS95_MAX_PIP",

        "CODING_N",

        "NONCODING_N",

        "INTRONIC_N",

        "INTERGENIC_N",

        "NCRNA_N",

        "LNCRNA_N",

        # Best expression evidence
        "EQTL_EFFECTIVE_GENE",

        "EQTL_TISSUE",

        "EQTL_LCLPP_MULTICAUSAL",

        "EQTL_TOP_SHARED_VARIANT",

        # Best splicing QTL
        "SQTL_EFFECTIVE_GENE",

        "SQTL_TISSUE",

        "SQTL_LCLPP_MULTICAUSAL",

        "SQTL_TOP_SHARED_VARIANT",

        # SpliceAI direct evidence
        "SPLICEAI_SCORED_N",

        "SPLICEAI_GE_020_N",

        "SPLICEAI_GE_050_N",

        "MAX_SPLICEAI_DS",

        "TOP_SPLICEAI_GENE",

        "TOP_SPLICEAI_EVENT",

        # Pangolin
        "PANGOLIN_SCORED_N",

        "PANGOLIN_TOP5_N",

        "MAX_PANGOLIN_ABS",

        # Combined splice evidence
        "MULTISOURCE_GE2_N",

        "BEST_LOCUS_LCLPP",
    ]

    columns = [

        column

        for column
        in columns

        if column
        in base.columns
    ]

    return (

        base[
            columns
        ]

        .sort_values(
            "EVIDENCE_RANK"
        )

        .reset_index(
            drop=True
        )
    )


# =============================================================================
# CROSS-GWAS COMMON LOCUS DETECTION
#
# IMPORTANT:
# L001 from study A is NOT compared to L001 from study B.
#
# We compare chromosome intervals:
#
#     CHR
#     LOCUS_START
#     LOCUS_END
#
# =============================================================================

def cluster_common_loci(
    all_loci,
):

    if all_loci.empty:

        return (
            pd.DataFrame(),
            {},
        )

    data = all_loci.dropna(

        subset=[

            "CHR",

            "LOCUS_START",

            "LOCUS_END",
        ]

    ).copy()

    data[
        "CHR"
    ] = data[
        "CHR"
    ].astype(
        int
    )

    data[
        "LOCUS_START"
    ] = data[
        "LOCUS_START"
    ].astype(
        int
    )

    data[
        "LOCUS_END"
    ] = data[
        "LOCUS_END"
    ].astype(
        int
    )

    data = (

        data

        .sort_values(

            [
                "CHR",
                "LOCUS_START",
                "LOCUS_END",
            ]
        )

        .reset_index(
            drop=True
        )
    )

    clusters = []

    mapping = {}

    cluster_number = 0

    for chromosome, group in data.groupby(

        "CHR",

        sort=True,

    ):

        current = None

        for _, row in group.iterrows():

            start = int(
                row[
                    "LOCUS_START"
                ]
            )

            end = int(
                row[
                    "LOCUS_END"
                ]
            )

            # Start new common cluster if it no longer overlaps.

            if (
                current is None
                or
                start
                >
                current[
                    "END"
                ]
            ):

                if current is not None:

                    clusters.append(
                        current
                    )

                cluster_number += 1

                current = {

                    "COMMON_LOCUS_ID":
                        f"C{cluster_number:03d}",

                    "CHR":
                        int(
                            chromosome
                        ),

                    "START":
                        start,

                    "END":
                        end,

                    "MEMBERS":
                        [],
                }

            else:

                current[
                    "END"
                ] = max(

                    current[
                        "END"
                    ],

                    end,
                )

            current[
                "MEMBERS"
            ].append(
                (
                    row[
                        "STUDY"
                    ],

                    row[
                        "LOCUS_ID"
                    ],
                )
            )

            mapping[
                (
                    row[
                        "STUDY"
                    ],

                    row[
                        "LOCUS_ID"
                    ],
                )
            ] = current[
                "COMMON_LOCUS_ID"
            ]

        if current is not None:

            clusters.append(
                current
            )

    rows = []

    for cluster in clusters:

        studies = sorted(

            {

                study

                for study, _
                in cluster[
                    "MEMBERS"
                ]
            }
        )

        rows.append(
            {

                "COMMON_LOCUS_ID":
                    cluster[
                        "COMMON_LOCUS_ID"
                    ],

                "CHR":
                    cluster[
                        "CHR"
                    ],

                "START":
                    cluster[
                        "START"
                    ],

                "END":
                    cluster[
                        "END"
                    ],

                "N_STUDIES":
                    len(
                        studies
                    ),

                "N_SOURCE_LOCI":
                    len(
                        cluster[
                            "MEMBERS"
                        ]
                    ),

                "STUDIES":
                    ";".join(
                        studies
                    ),

                "SOURCE_LOCI":
                    ";".join(

                        f"{study}:{locus}"

                        for study, locus
                        in cluster[
                            "MEMBERS"
                        ]
                    ),
            }
        )

    return (

        pd.DataFrame(
            rows
        ),

        mapping,
    )


# =============================================================================
# ADD BIOLOGICAL EVIDENCE TO COMMON LOCI
# =============================================================================

def annotate_common_clusters(
    common,
    mapping,
    locus_reports,
    n_eligible,
):

    if common.empty:

        return common

    evidence = []

    for report in locus_reports:

        if report.empty:

            continue

        data = report.copy()

        data[
            "COMMON_LOCUS_ID"
        ] = [

            mapping.get(

                (
                    row.STUDY,
                    row.LOCUS_ID,
                ),

                "",
            )

            for row
            in data.itertuples()
        ]

        evidence.append(
            data
        )

    if not evidence:

        common[
            "IN_ALL_FINEMAPPED_STUDIES"
        ] = (

            common[
                "N_STUDIES"
            ]
            ==
            n_eligible
        )

        return common

    evidence = pd.concat(

        evidence,

        ignore_index=True,
    )

    rows = []

    for common_id, group in evidence.groupby(

        "COMMON_LOCUS_ID"
    ):

        genes = []

        for column in [

            "EQTL_EFFECTIVE_GENE",

            "SQTL_EFFECTIVE_GENE",

        ]:

            if column not in group.columns:

                continue

            for value in group[
                column
            ].dropna():

                value = str(
                    value
                )

                if value.lower() in {

                    "",

                    "none",

                    "nan",
                }:

                    continue

                if value not in genes:

                    genes.append(
                        value
                    )

        best_lclpp = pd.to_numeric(

            group.get(

                "BEST_LOCUS_LCLPP",

                pd.Series(
                    dtype=float
                ),
            ),

            errors="coerce",

        ).max()

        max_spliceai = pd.to_numeric(

            group.get(

                "MAX_SPLICEAI_DS",

                pd.Series(
                    dtype=float
                ),
            ),

            errors="coerce",

        ).max()

        max_pangolin = pd.to_numeric(

            group.get(

                "MAX_PANGOLIN_ABS",

                pd.Series(
                    dtype=float
                ),
            ),

            errors="coerce",

        ).max()

        spliceai_supported = pd.to_numeric(

            group.get(

                "SPLICEAI_GE_020_N",

                pd.Series(
                    dtype=float
                ),
            ),

            errors="coerce",

        ).fillna(
            0
        ).sum()

        multisource = pd.to_numeric(

            group.get(

                "MULTISOURCE_GE2_N",

                pd.Series(
                    dtype=float
                ),
            ),

            errors="coerce",

        ).fillna(
            0
        ).sum()

        rows.append(
            {

                "COMMON_LOCUS_ID":
                    common_id,

                "BEST_LCLPP_ACROSS_STUDIES":
                    best_lclpp,

                "TOP_GENES":
                    ";".join(
                        genes[
                            :12
                        ]
                    ),

                "MAX_SPLICEAI":
                    max_spliceai,

                "MAX_PANGOLIN":
                    max_pangolin,

                "N_SPLICEAI020_TOTAL":
                    spliceai_supported,

                "N_MULTISOURCE_GE2_TOTAL":
                    multisource,
            }
        )

    result = common.merge(

        pd.DataFrame(
            rows
        ),

        on="COMMON_LOCUS_ID",

        how="left",
    )

    result[
        "IN_ALL_FINEMAPPED_STUDIES"
    ] = (

        result[
            "N_STUDIES"
        ]
        ==
        n_eligible
    )

    return (

        result

        .sort_values(

            [
                "N_STUDIES",
                "BEST_LCLPP_ACROSS_STUDIES",
            ],

            ascending=[
                False,
                False,
            ],
        )

        .reset_index(
            drop=True
        )
    )


# =============================================================================
# COMMON COLOCALIZED GENES
# =============================================================================

def common_gene_table(
    pair_tables,
    min_studies=2,
):

    usable = [

        table

        for table
        in pair_tables

        if not table.empty
    ]

    if not usable:

        return pd.DataFrame()

    data = pd.concat(

        usable,

        ignore_index=True,
    )

    data = data[

        data[
            "EFFECTIVE_GENE"
        ]
        .astype(
            str
        )
        .str.len()
        >
        0

    ].copy()

    data[
        "SOURCE_LOCUS"
    ] = (

        data[
            "STUDY"
        ].astype(
            str
        )

        +
        "::"

        +
        data[
            "LOCUS_ID"
        ].astype(
            str
        )
    )

    rows = []

    for (
        qtl_type,
        gene
    ), group in data.groupby(

        [
            "QTL_TYPE",
            "EFFECTIVE_GENE",
        ],

        dropna=False,
    ):

        n_studies = group[
            "STUDY"
        ].nunique()

        if n_studies < min_studies:

            continue

        scores = pd.to_numeric(

            group[
                "LCLPP_MULTICAUSAL"
            ],

            errors="coerce",
        )

        if scores.notna().any():

            top = group.loc[
                scores.idxmax()
            ]

            best = scores.max()

        else:

            top = group.iloc[
                0
            ]

            best = np.nan

        rows.append(
            {

                "QTL_TYPE":
                    qtl_type,

                "GENE":
                    gene,

                "N_STUDIES":
                    n_studies,

                "N_SOURCE_LOCI":
                    group[
                        "SOURCE_LOCUS"
                    ].nunique(),

                "N_TISSUES":
                    group[
                        "TISSUE"
                    ].nunique(),

                "BEST_LCLPP":
                    best,

                "BEST_TISSUE":
                    top.get(
                        "TISSUE",
                        "",
                    ),

                "STUDIES":
                    ";".join(

                        sorted(

                            group[
                                "STUDY"
                            ].unique()
                        )
                    ),
            }
        )

    if not rows:

        return pd.DataFrame()

    return (

        pd.DataFrame(
            rows
        )

        .sort_values(

            [
                "N_STUDIES",
                "BEST_LCLPP",
            ],

            ascending=[
                False,
                False,
            ],
        )

        .reset_index(
            drop=True
        )
    )


# =============================================================================
# COMMON EXACT TOP VARIANTS
# =============================================================================

def common_top_variants(
    pair_tables,
    min_studies=2,
):

    usable = [

        table

        for table
        in pair_tables

        if not table.empty
    ]

    if not usable:

        return pd.DataFrame()

    data = pd.concat(

        usable,

        ignore_index=True,
    )

    if (
        "TOP_SHARED_VARIANT"
        not in data.columns
    ):

        return pd.DataFrame()

    data = data[

        data[
            "TOP_SHARED_VARIANT"
        ].notna()

    ].copy()

    rows = []

    for variant, group in data.groupby(

        "TOP_SHARED_VARIANT"
    ):

        n_studies = group[
            "STUDY"
        ].nunique()

        if n_studies < min_studies:

            continue

        genes = sorted(

            {

                gene

                for gene
                in group[
                    "EFFECTIVE_GENE"
                ].astype(
                    str
                )

                if gene
            }
        )

        rows.append(
            {

                "VARIANT":
                    variant,

                "N_STUDIES":
                    n_studies,

                "N_SOURCE_LOCI":
                    (

                        group[
                            "STUDY"
                        ].astype(
                            str
                        )

                        +
                        "::"

                        +
                        group[
                            "LOCUS_ID"
                        ].astype(
                            str
                        )

                    ).nunique(),

                "QTL_TYPES":
                    ";".join(

                        sorted(

                            group[
                                "QTL_TYPE"
                            ]
                            .astype(
                                str
                            )
                            .unique()
                        )
                    ),

                "BEST_LCLPP":
                    pd.to_numeric(

                        group[
                            "LCLPP_MULTICAUSAL"
                        ],

                        errors="coerce",

                    ).max(),

                "GENES":
                    ";".join(
                        genes
                    ),

                "STUDIES":
                    ";".join(

                        sorted(

                            group[
                                "STUDY"
                            ].unique()
                        )
                    ),
            }
        )

    if not rows:

        return pd.DataFrame()

    return (

        pd.DataFrame(
            rows
        )

        .sort_values(

            [
                "N_STUDIES",
                "BEST_LCLPP",
            ],

            ascending=[
                False,
                False,
            ],
        )

        .reset_index(
            drop=True
        )
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
    )


    parser.add_argument(

        "--top-genes",

        type=int,

        default=30,
    )


    parser.add_argument(

        "--top-variants",

        type=int,

        default=30,
    )


    parser.add_argument(

        "--min-common-studies",

        type=int,

        default=2,
    )


    args = parser.parse_args()


    root = Path.cwd().resolve()


    ancestry_code, ancestry_label = ancestry_info(

        args.ancestry
    )


    roots = roots_for(

        root,

        args.phenotype,

        ancestry_label,
    )


    studies = discover_studies(
        roots
    )


    if args.study:

        studies = [

            study

            for study
            in studies

            if study
            ==
            args.study
        ]


    if not studies:

        raise SystemExit(
            "No matching studies found."
        )


    # =========================================================================
    # HEADER
    # =========================================================================

    section(

        "GWAS2m FAST REPORT - "
        "LOCUS + COMMON-LOCUS + SPLICE EVIDENCE"
    )


    print(
        f"Version      : {VERSION}"
    )

    print(
        f"Root         : {root}"
    )

    print(
        f"Phenotype    : {args.phenotype}"
    )

    print(
        f"Ancestry     : "
        f"{ancestry_label} "
        f"({ancestry_code})"
    )

    print(
        f"Studies      : "
        f"{len(studies)}"
    )

    print(
        "Saving files : NO"
    )

    print(
        "Engine       : "
        "Polars + SciPy"
    )


    # =========================================================================
    # TABLE 1
    # =========================================================================

    funnel = pd.DataFrame(

        [

            pipeline_row(

                study,

                study_paths(
                    roots,
                    study,
                ),
            )

            for study
            in studies
        ]
    )


    show(

        funnel,

        "TABLE 1 - PIPELINE FUNNEL / AUDIT",
    )


    # =========================================================================
    # INDIVIDUAL LOCI
    # =========================================================================

    locus_reports = []

    pair_tables = []

    locus_definitions = []


    for study in studies:

        sp = study_paths(

            roots,

            study,
        )


        report = locus_report(

            sp,

            study,
        )


        if not report.empty:

            locus_reports.append(
                report
            )


            show(

                report,

                (
                    "TABLE 2 - "
                    f"LOCUS INTERPRETATION: "
                    f"{study}"
                ),
            )


        pairs = pair_table(

            sp,

            study,
        )


        if not pairs.empty:

            pair_tables.append(
                pairs
            )


        loci = loci_definition(

            sp,

            study,
        )


        if not loci.empty:

            locus_definitions.append(
                loci
            )


    # =========================================================================
    # COMMON LOCI
    # =========================================================================

    if locus_definitions:

        all_loci = pd.concat(

            locus_definitions,

            ignore_index=True,
        )

    else:

        all_loci = pd.DataFrame()


    n_eligible = (

        all_loci[
            "STUDY"
        ].nunique()

        if not all_loci.empty

        else
        0
    )


    common, mapping = cluster_common_loci(
        all_loci
    )


    common = annotate_common_clusters(

        common,

        mapping,

        locus_reports,

        n_eligible,
    )


    if not common.empty:

        common_repeated = common[

            common[
                "N_STUDIES"
            ]
            >=
            args.min_common_studies

        ].copy()

    else:

        common_repeated = common


    show(

        common_repeated,

        (
            "TABLE 3 - COMMON GENOMIC LOCI ACROSS GWAS "
            "(OVERLAPPING STEP06 LOCUS INTERVALS)"
        ),
    )


    # =========================================================================
    # LOCI FOUND IN ALL FINE-MAPPED STUDIES
    # =========================================================================

    if not common.empty:

        common_all = common[

            common[
                "IN_ALL_FINEMAPPED_STUDIES"
            ]

        ].copy()

    else:

        common_all = common


    show(

        common_all,

        (
            "TABLE 4 - LOCI PRESENT IN ALL "
            f"{n_eligible} FINE-MAPPED GWAS"
        ),
    )


    # =========================================================================
    # COMMON GENES
    # =========================================================================

    genes = common_gene_table(

        pair_tables,

        args.min_common_studies,
    )


    if not genes.empty:

        genes = genes.head(
            args.top_genes
        )


    show(

        genes,

        (
            "TABLE 5 - COMMON COLOCALIZED GENES "
            f"ACROSS >= "
            f"{args.min_common_studies} GWAS "
            f"(TOP {args.top_genes})"
        ),
    )


    # =========================================================================
    # COMMON EXACT VARIANTS
    # =========================================================================

    variants = common_top_variants(

        pair_tables,

        args.min_common_studies,
    )


    if not variants.empty:

        variants = variants.head(
            args.top_variants
        )


    show(

        variants,

        (
            "TABLE 6 - COMMON TOP SHARED VARIANTS "
            f"ACROSS >= "
            f"{args.min_common_studies} GWAS "
            f"(TOP {args.top_variants})"
        ),
    )


    # =========================================================================
    # NOTES
    # =========================================================================

    section(
        "INTERPRETATION NOTES"
    )


    print(
        "Fine-mapped GWAS contributing genomic loci: "
        f"{n_eligible}"
    )


    print(
        "\nCOMMON LOCUS:"
    )

    print(
        "  Overlapping Step06 genomic intervals "
        "on the same chromosome."
    )

    print(
        "  Local labels such as L001 are NEVER "
        "compared directly between studies."
    )


    print(
        "\neQTL gene handling:"
    )

    print(
        "  If QTL_GENE is blank/None, "
        "QTL_PHENOTYPE is used as the gene ID."
    )

    print(
        "  This allows both old and corrected "
        "Step11 outputs to be interpreted."
    )


    print(
        "\nSplicing:"
    )

    print(
        "  SpliceAI is read directly from "
        "Step09 locus summaries."
    )

    print(
        "  Pangolin is read independently from "
        "Step10 locus summaries."
    )

    print(
        "  SpliceAI value/count = 0 means "
        "the result was actually zero."
    )

    print(
        "  NaN/MISSING means the upstream "
        "file/result was unavailable."
    )


    print(
        "\nPangolin:"
    )

    print(
        "  Top-5% is a ranking within that study, "
        "not a universal biological threshold."
    )


    print(
        "\nColocalization:"
    )

    print(
        "  LCLPP is still a screening score."
    )

    print(
        "  It is NOT a formal coloc.susie posterior."
    )


if __name__ == "__main__":

    main()
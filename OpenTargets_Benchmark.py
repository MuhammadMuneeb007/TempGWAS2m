#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
GWAS2m - OPEN TARGETS / GENTROPY EXTERNAL BENCHMARK
===============================================================================

PURPOSE
-------
Retrieve the current Open Targets Platform / Gentropy production results for
the exact same GCST studies being analysed by GWAS2m.

This does NOT rerun Gentropy locally.

Instead it retrieves the Open Targets production results for:

    1. GWAS credible sets
    2. Lead variants
    3. Fine-mapping method
    4. Credible-set variants + posterior probabilities
    5. Locus-to-Gene (L2G) predictions
    6. L2G feature values / SHAP values
    7. Open Targets colocalisation:
         - COLOC-PIP
         - eCAVIAR / CLPP where available

This creates an independent benchmark that can later be compared with:

    GWAS2m Step05 loci
    GWAS2m Step06 SuSiE credible sets
    GWAS2m Step11 colocalisation
    GWAS2m candidate genes
    GWAS2m common loci / common genes

Open Targets GraphQL:
    https://api.platform.opentargets.org/api/v4/graphql

Recommended:

    python OpenTargets_Benchmark.py \
        --phenotype migraine \
        --ancestry EUR

===============================================================================
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "1.0.0"

API_URL = (
    "https://api.platform.opentargets.org/api/v4/graphql"
)

OPEN_TARGETS_RELEASE_LABEL = "26.09"

DEFAULT_PAGE_SIZE = 100

DETAIL_PAGE_SIZE = 500

DEFAULT_WORKERS = 4


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
# HELPERS
# =============================================================================

def norm(value):

    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^a-z0-9]+",
            " ",
            str(value).lower(),
        ),
    ).strip()


def slug(value):

    return norm(value).replace(
        " ",
        "_",
    )


def ancestry_info(value):

    key = norm(value)

    if key not in ANCESTRY:

        raise SystemExit(
            f"Unsupported ancestry: {value}"
        )

    return ANCESTRY[key]


def section(title):

    print()

    print(
        "=" * 120
    )

    print(title)

    print(
        "=" * 120
    )


def make_session():

    session = requests.Session()

    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.5,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "POST",
        ],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=16,
        pool_maxsize=16,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent":
                "GWAS2m-OpenTargets-Benchmark/1.0",
        }
    )

    return session


# =============================================================================
# GRAPHQL
# =============================================================================

def graphql(
    session,
    query,
    variables,
):

    response = session.post(
        API_URL,
        json={
            "query": query,
            "variables": variables,
        },
        timeout=180,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get(
        "errors"
    ):

        message = "\n".join(

            str(
                error.get(
                    "message",
                    error,
                )
            )

            for error
            in payload["errors"]
        )

        raise RuntimeError(
            f"Open Targets GraphQL error:\n{message}"
        )

    return payload[
        "data"
    ]


# =============================================================================
# DISCOVER GWAS2m STUDIES
# =============================================================================

def discover_studies(
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

    candidates = [

        root
        /
        "02_summary_stats"
        /
        phenotype_slug
        /
        ancestry_slug
        /
        "raw",

        root
        /
        "02_summary_stats"
        /
        phenotype_slug
        /
        ancestry_slug
        /
        "qc",

        root
        /
        "05_ld_clumping"
        /
        phenotype_slug
        /
        ancestry_slug,

        root
        /
        "06_finemapping"
        /
        phenotype_slug
        /
        ancestry_slug,

        root
        /
        "11_coloc"
        /
        phenotype_slug
        /
        ancestry_slug,
    ]

    studies = set()

    for directory in candidates:

        if not directory.exists():

            continue

        for path in directory.glob(
            "GCST*"
        ):

            match = re.fullmatch(
                r"GCST\d+",
                path.name,
            )

            if match:

                studies.add(
                    path.name
                )

    return sorted(
        studies
    )


# =============================================================================
# QUERY 1
# FIND ALL OPEN TARGETS CREDIBLE SETS FOR THE GCST STUDIES
# =============================================================================

CREDIBLE_SET_QUERY = """
query GetCredibleSets(
    $studyIds: [String!],
    $page: Pagination!
) {

    credibleSets(
        studyIds: $studyIds,
        page: $page
    ) {

        count

        rows {

            studyLocusId

            pValueMantissa
            pValueExponent

            beta

            finemappingMethod
            confidence

            study {

                id
                studyType
                traitFromSource
                projectId
                nSamples
            }

            variant {

                id
                chromosome
                position
                referenceAllele
                alternateAllele
                rsIds
            }
        }
    }
}
"""


def fetch_credible_sets(
    session,
    studies,
):

    rows = []

    page_index = 0

    while True:

        variables = {

            "studyIds":
                studies,

            "page": {

                "index":
                    page_index,

                "size":
                    DEFAULT_PAGE_SIZE,
            },
        }

        data = graphql(

            session,

            CREDIBLE_SET_QUERY,

            variables,
        )

        block = data[
            "credibleSets"
        ]

        batch = block.get(
            "rows",
            []
        )

        rows.extend(
            batch
        )

        count = int(
            block.get(
                "count",
                0,
            )
        )

        print(

            f"[Open Targets] "
            f"credible sets: "
            f"{len(rows):,}/{count:,}",

            flush=True,
        )

        if (
            len(rows)
            >=
            count
            or
            not batch
        ):

            break

        page_index += 1

    return rows


# =============================================================================
# QUERY 2
# FULL DETAIL FOR ONE CREDIBLE SET
# =============================================================================

DETAIL_QUERY = """
query GetCredibleSetDetail(
    $studyLocusId: String!,
    $page: Pagination!
) {

    credibleSet(
        studyLocusId: $studyLocusId
    ) {

        studyLocusId

        l2GPredictions(
            page: $page
        ) {

            count

            rows {

                score

                target {

                    id
                    approvedSymbol
                }

                features {

                    name
                    value
                    shapValue
                }
            }
        }

        locus(
            page: $page
        ) {

            count

            rows {

                posteriorProbability

                pValueMantissa
                pValueExponent

                beta

                is95CredibleSet
                is99CredibleSet

                variant {

                    id
                    chromosome
                    position
                    referenceAllele
                    alternateAllele
                    rsIds
                }
            }
        }

        colocalisation(
            page: $page
        ) {

            count

            rows {

                studyLocusId
                otherStudyLocusId

                rightStudyType
                chromosome

                colocalisationMethod

                numberColocalisingVariants

                h3
                h4
                clpp

                betaRatioSignAverage
            }
        }
    }
}
"""


def fetch_one_detail(
    study_locus_id,
):

    session = make_session()

    variables = {

        "studyLocusId":
            study_locus_id,

        "page": {

            "index":
                0,

            "size":
                DETAIL_PAGE_SIZE,
        },
    }

    data = graphql(

        session,

        DETAIL_QUERY,

        variables,
    )

    credible_set = data.get(
        "credibleSet"
    )

    if credible_set is None:

        return {

            "studyLocusId":
                study_locus_id,

            "error":
                "credibleSet returned null",
        }

    # Warn if any result was truncated by page size.

    for key in [

        "l2GPredictions",

        "locus",

        "colocalisation",

    ]:

        block = credible_set.get(
            key
        )

        if not block:

            continue

        count = int(
            block.get(
                "count",
                0,
            )
        )

        if count > DETAIL_PAGE_SIZE:

            credible_set.setdefault(
                "_warnings",
                []
            ).append(

                f"{key} contains {count} rows; "
                f"only first {DETAIL_PAGE_SIZE} retrieved."
            )

    return credible_set


# =============================================================================
# FLATTEN CREDIBLE SET METADATA
# =============================================================================

def flatten_credible_sets(
    credible_sets,
):

    rows = []

    for cs in credible_sets:

        study = cs.get(
            "study"
        ) or {}

        variant = cs.get(
            "variant"
        ) or {}

        mantissa = cs.get(
            "pValueMantissa"
        )

        exponent = cs.get(
            "pValueExponent"
        )

        pvalue = np.nan

        try:

            if (
                mantissa is not None
                and
                exponent is not None
            ):

                pvalue = (
                    float(mantissa)
                    *
                    10.0
                    **
                    float(exponent)
                )

        except Exception:

            pass

        rows.append(
            {

                "STUDY_ID":
                    study.get(
                        "id"
                    ),

                "STUDY_TYPE":
                    study.get(
                        "studyType"
                    ),

                "TRAIT":
                    study.get(
                        "traitFromSource"
                    ),

                "PROJECT_ID":
                    study.get(
                        "projectId"
                    ),

                "N_SAMPLES":
                    study.get(
                        "nSamples"
                    ),

                "STUDY_LOCUS_ID":
                    cs.get(
                        "studyLocusId"
                    ),

                "LEAD_VARIANT":
                    variant.get(
                        "id"
                    ),

                "CHR":
                    variant.get(
                        "chromosome"
                    ),

                "POS":
                    variant.get(
                        "position"
                    ),

                "REF":
                    variant.get(
                        "referenceAllele"
                    ),

                "ALT":
                    variant.get(
                        "alternateAllele"
                    ),

                "RSIDS":
                    ";".join(
                        variant.get(
                            "rsIds"
                        )
                        or
                        []
                    ),

                "P_VALUE":
                    pvalue,

                "P_MANTISSA":
                    mantissa,

                "P_EXPONENT":
                    exponent,

                "BETA":
                    cs.get(
                        "beta"
                    ),

                "FINEMAPPING_METHOD":
                    cs.get(
                        "finemappingMethod"
                    ),

                "CONFIDENCE":
                    cs.get(
                        "confidence"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FLATTEN DETAILS
# =============================================================================

def flatten_l2g(
    details,
    metadata_lookup,
):

    rows = []

    feature_rows = []

    for detail in details:

        if detail.get(
            "error"
        ):

            continue

        study_locus_id = detail[
            "studyLocusId"
        ]

        metadata = metadata_lookup.get(
            study_locus_id,
            {}
        )

        block = detail.get(
            "l2GPredictions"
        ) or {}

        for prediction in block.get(
            "rows",
            []
        ):

            target = prediction.get(
                "target"
            ) or {}

            rows.append(
                {

                    "STUDY_ID":
                        metadata.get(
                            "STUDY_ID"
                        ),

                    "STUDY_LOCUS_ID":
                        study_locus_id,

                    "LEAD_VARIANT":
                        metadata.get(
                            "LEAD_VARIANT"
                        ),

                    "GENE_ID":
                        target.get(
                            "id"
                        ),

                    "GENE_SYMBOL":
                        target.get(
                            "approvedSymbol"
                        ),

                    "L2G_SCORE":
                        prediction.get(
                            "score"
                        ),
                }
            )

            for feature in (
                prediction.get(
                    "features"
                )
                or
                []
            ):

                feature_rows.append(
                    {

                        "STUDY_ID":
                            metadata.get(
                                "STUDY_ID"
                            ),

                        "STUDY_LOCUS_ID":
                            study_locus_id,

                        "GENE_ID":
                            target.get(
                                "id"
                            ),

                        "GENE_SYMBOL":
                            target.get(
                                "approvedSymbol"
                            ),

                        "L2G_SCORE":
                            prediction.get(
                                "score"
                            ),

                        "FEATURE":
                            feature.get(
                                "name"
                            ),

                        "VALUE":
                            feature.get(
                                "value"
                            ),

                        "SHAP_VALUE":
                            feature.get(
                                "shapValue"
                            ),
                    }
                )

    return (

        pd.DataFrame(
            rows
        ),

        pd.DataFrame(
            feature_rows
        ),
    )


def flatten_locus_variants(
    details,
    metadata_lookup,
):

    rows = []

    for detail in details:

        if detail.get(
            "error"
        ):

            continue

        study_locus_id = detail[
            "studyLocusId"
        ]

        metadata = metadata_lookup.get(
            study_locus_id,
            {}
        )

        block = detail.get(
            "locus"
        ) or {}

        for item in block.get(
            "rows",
            []
        ):

            variant = item.get(
                "variant"
            ) or {}

            rows.append(
                {

                    "STUDY_ID":
                        metadata.get(
                            "STUDY_ID"
                        ),

                    "STUDY_LOCUS_ID":
                        study_locus_id,

                    "LEAD_VARIANT":
                        metadata.get(
                            "LEAD_VARIANT"
                        ),

                    "VARIANT":
                        variant.get(
                            "id"
                        ),

                    "CHR":
                        variant.get(
                            "chromosome"
                        ),

                    "POS":
                        variant.get(
                            "position"
                        ),

                    "REF":
                        variant.get(
                            "referenceAllele"
                        ),

                    "ALT":
                        variant.get(
                            "alternateAllele"
                        ),

                    "RSIDS":
                        ";".join(
                            variant.get(
                                "rsIds"
                            )
                            or
                            []
                        ),

                    "PIP":
                        item.get(
                            "posteriorProbability"
                        ),

                    "IS_95_CS":
                        item.get(
                            "is95CredibleSet"
                        ),

                    "IS_99_CS":
                        item.get(
                            "is99CredibleSet"
                        ),

                    "BETA":
                        item.get(
                            "beta"
                        ),

                    "P_MANTISSA":
                        item.get(
                            "pValueMantissa"
                        ),

                    "P_EXPONENT":
                        item.get(
                            "pValueExponent"
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def flatten_colocalisation(
    details,
    metadata_lookup,
):

    rows = []

    for detail in details:

        if detail.get(
            "error"
        ):

            continue

        study_locus_id = detail[
            "studyLocusId"
        ]

        metadata = metadata_lookup.get(
            study_locus_id,
            {}
        )

        block = detail.get(
            "colocalisation"
        ) or {}

        for item in block.get(
            "rows",
            []
        ):

            rows.append(
                {

                    "STUDY_ID":
                        metadata.get(
                            "STUDY_ID"
                        ),

                    "STUDY_LOCUS_ID":
                        study_locus_id,

                    "LEAD_VARIANT":
                        metadata.get(
                            "LEAD_VARIANT"
                        ),

                    "OTHER_STUDY_LOCUS_ID":
                        item.get(
                            "otherStudyLocusId"
                        ),

                    "RIGHT_STUDY_TYPE":
                        item.get(
                            "rightStudyType"
                        ),

                    "CHR":
                        item.get(
                            "chromosome"
                        ),

                    "METHOD":
                        item.get(
                            "colocalisationMethod"
                        ),

                    "N_COLOCALISING_VARIANTS":
                        item.get(
                            "numberColocalisingVariants"
                        ),

                    "H3":
                        item.get(
                            "h3"
                        ),

                    "H4":
                        item.get(
                            "h4"
                        ),

                    "CLPP":
                        item.get(
                            "clpp"
                        ),

                    "BETA_RATIO_SIGN_AVERAGE":
                        item.get(
                            "betaRatioSignAverage"
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# STUDY COVERAGE
# =============================================================================

def build_study_coverage(
    requested_studies,
    credible_sets_df,
):

    counts = {}

    if not credible_sets_df.empty:

        counts = (

            credible_sets_df

            .groupby(
                "STUDY_ID"
            )

            .size()

            .to_dict()
        )

    rows = []

    for study in requested_studies:

        n = int(
            counts.get(
                study,
                0,
            )
        )

        rows.append(
            {

                "STUDY_ID":
                    study,

                "FOUND_IN_OPEN_TARGETS":
                    n > 0,

                "N_OPEN_TARGETS_CREDIBLE_SETS":
                    n,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_summary(
    coverage,
    credible,
    l2g,
    locus,
    coloc,
):

    section(
        "OPEN TARGETS / GENTROPY BENCHMARK SUMMARY"
    )

    print(
        f"Requested GWAS studies    : "
        f"{len(coverage):,}"
    )

    print(
        f"Studies with credible sets: "
        f"{coverage['FOUND_IN_OPEN_TARGETS'].sum():,}"
    )

    print(
        f"Credible sets             : "
        f"{len(credible):,}"
    )

    print(
        f"Credible-set variant rows : "
        f"{len(locus):,}"
    )

    print(
        f"L2G predictions           : "
        f"{len(l2g):,}"
    )

    print(
        f"Colocalisation rows       : "
        f"{len(coloc):,}"
    )

    if not credible.empty:

        section(
            "FINE-MAPPING METHODS USED BY OPEN TARGETS"
        )

        print(

            credible[
                "FINEMAPPING_METHOD"
            ]
            .fillna(
                "NA"
            )
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "METHOD"
            )
            .reset_index(
                name="N"
            )
            .to_string(
                index=False
            )
        )

    if not l2g.empty:

        section(
            "TOP OPEN TARGETS L2G PREDICTION PER CREDIBLE SET"
        )

        top = (

            l2g

            .sort_values(
                "L2G_SCORE",
                ascending=False,
            )

            .drop_duplicates(
                "STUDY_LOCUS_ID"
            )

            .sort_values(

                [
                    "STUDY_ID",
                    "L2G_SCORE",
                ],

                ascending=[
                    True,
                    False,
                ],
            )
        )

        print(

            top[
                [
                    "STUDY_ID",
                    "STUDY_LOCUS_ID",
                    "LEAD_VARIANT",
                    "GENE_SYMBOL",
                    "GENE_ID",
                    "L2G_SCORE",
                ]
            ]

            .to_string(
                index=False
            )
        )

    if not coloc.empty:

        section(
            "OPEN TARGETS COLOCALISATION METHODS"
        )

        print(

            coloc[
                "METHOD"
            ]
            .fillna(
                "NA"
            )
            .value_counts()
            .rename_axis(
                "METHOD"
            )
            .reset_index(
                name="N"
            )
            .to_string(
                index=False
            )
        )

        strong = coloc.copy()

        strong[
            "BEST_COLOC_VALUE"
        ] = strong[

            [
                "H4",
                "CLPP",
            ]

        ].max(
            axis=1,
            skipna=True,
        )

        strong = (

            strong

            .sort_values(
                "BEST_COLOC_VALUE",
                ascending=False,
            )

            .head(
                30
            )
        )

        section(
            "TOP 30 OPEN TARGETS COLOCALISATION RESULTS"
        )

        print(

            strong[
                [
                    "STUDY_ID",
                    "LEAD_VARIANT",
                    "RIGHT_STUDY_TYPE",
                    "METHOD",
                    "H4",
                    "CLPP",
                    "N_COLOCALISING_VARIANTS",
                ]
            ]

            .to_string(
                index=False
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
        default=None,
        help=(
            "Optional single GCST accession."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    root = Path.cwd().resolve()

    (
        ancestry_code,
        ancestry_label,

    ) = ancestry_info(
        args.ancestry
    )

    studies = discover_studies(

        root,

        args.phenotype,

        ancestry_label,
    )

    if args.study:

        studies = [

            x

            for x in studies

            if x == args.study
        ]

    if not studies:

        raise SystemExit(
            "No GCST studies found."
        )

    if args.output_dir:

        output = Path(
            args.output_dir
        ).resolve()

    else:

        output = (

            root
            /
            "benchmark"
            /
            f"opentargets_{OPEN_TARGETS_RELEASE_LABEL.replace('.', '_')}"
            /
            slug(
                args.phenotype
            )
            /
            slug(
                ancestry_label
            )
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    section(
        "GWAS2m ? OPEN TARGETS / GENTROPY BENCHMARK"
    )

    print(
        f"GWAS2m root       : {root}"
    )

    print(
        f"Phenotype         : {args.phenotype}"
    )

    print(
        f"Ancestry          : "
        f"{ancestry_label} "
        f"({ancestry_code})"
    )

    print(
        f"Studies           : {len(studies)}"
    )

    print(
        "Study IDs         : "
        +
        "; ".join(
            studies
        )
    )

    print(
        f"API               : {API_URL}"
    )

    print(
        f"Release label     : "
        f"{OPEN_TARGETS_RELEASE_LABEL}"
    )

    print(
        f"Output            : {output}"
    )

    # -------------------------------------------------------------------------
    # Credible sets
    # -------------------------------------------------------------------------

    session = make_session()

    credible_sets = fetch_credible_sets(

        session,

        studies,
    )

    credible_df = flatten_credible_sets(
        credible_sets
    )

    # -------------------------------------------------------------------------
    # Full credible-set detail
    # -------------------------------------------------------------------------

    study_locus_ids = (

        credible_df[
            "STUDY_LOCUS_ID"
        ]
        .dropna()
        .astype(
            str
        )
        .unique()
        .tolist()

        if not credible_df.empty

        else
        []
    )

    section(
        "DOWNLOADING L2G / PIP / COLOCALISATION DETAIL"
    )

    print(
        f"Credible sets to inspect: "
        f"{len(study_locus_ids):,}"
    )

    details = []

    errors = []

    with ThreadPoolExecutor(

        max_workers=max(
            1,
            args.workers,
        )

    ) as executor:

        futures = {

            executor.submit(
                fetch_one_detail,
                study_locus_id,
            ):
                study_locus_id

            for study_locus_id
            in study_locus_ids
        }

        completed = 0

        for future in as_completed(
            futures
        ):

            study_locus_id = futures[
                future
            ]

            completed += 1

            try:

                detail = future.result()

                details.append(
                    detail
                )

            except Exception as error:

                errors.append(
                    {

                        "STUDY_LOCUS_ID":
                            study_locus_id,

                        "ERROR":
                            (
                                f"{type(error).__name__}: "
                                f"{error}"
                            ),
                    }
                )

            print(

                f"[detail] "
                f"{completed:,}/"
                f"{len(study_locus_ids):,}",

                flush=True,
            )

    # -------------------------------------------------------------------------
    # Lookup metadata
    # -------------------------------------------------------------------------

    metadata_lookup = {}

    for _, row in credible_df.iterrows():

        metadata_lookup[
            row[
                "STUDY_LOCUS_ID"
            ]
        ] = row.to_dict()

    # -------------------------------------------------------------------------
    # Flatten
    # -------------------------------------------------------------------------

    l2g_df, features_df = flatten_l2g(

        details,

        metadata_lookup,
    )

    locus_df = flatten_locus_variants(

        details,

        metadata_lookup,
    )

    coloc_df = flatten_colocalisation(

        details,

        metadata_lookup,
    )

    coverage_df = build_study_coverage(

        studies,

        credible_df,
    )

    error_df = pd.DataFrame(
        errors
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    coverage_df.to_csv(

        output
        /
        "study_coverage.tsv",

        sep="\t",

        index=False,
    )

    credible_df.to_csv(

        output
        /
        "credible_sets.tsv",

        sep="\t",

        index=False,
    )

    locus_df.to_csv(

        output
        /
        "credible_set_variants.tsv",

        sep="\t",

        index=False,
    )

    l2g_df.to_csv(

        output
        /
        "l2g_predictions.tsv",

        sep="\t",

        index=False,
    )

    features_df.to_csv(

        output
        /
        "l2g_features.tsv",

        sep="\t",

        index=False,
    )

    coloc_df.to_csv(

        output
        /
        "colocalisation.tsv",

        sep="\t",

        index=False,
    )

    if not error_df.empty:

        error_df.to_csv(

            output
            /
            "errors.tsv",

            sep="\t",

            index=False,
        )

    # Save raw API response for reproducibility.

    raw_path = (

        output
        /
        "opentargets_raw.json.gz"
    )

    with gzip.open(

        raw_path,

        "wt",

        encoding="utf-8",

    ) as handle:

        json.dump(

            {
                "credible_sets":
                    credible_sets,

                "details":
                    details,
            },

            handle,
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {

        "script_version":
            VERSION,

        "open_targets_release_label":
            OPEN_TARGETS_RELEASE_LABEL,

        "api_url":
            API_URL,

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

        "requested_studies":
            studies,

        "n_requested_studies":
            len(
                studies
            ),

        "n_credible_sets":
            len(
                credible_df
            ),

        "n_l2g_predictions":
            len(
                l2g_df
            ),

        "n_locus_variants":
            len(
                locus_df
            ),

        "n_colocalisation_rows":
            len(
                coloc_df
            ),

        "n_errors":
            len(
                error_df
            ),
    }

    (

        output
        /
        "benchmark_metadata.json"

    ).write_text(

        json.dumps(
            metadata,
            indent=2,
        )
        +
        "\n",

        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Print useful summary
    # -------------------------------------------------------------------------

    print_summary(

        coverage_df,

        credible_df,

        l2g_df,

        locus_df,

        coloc_df,
    )

    section(
        "FILES WRITTEN"
    )

    for filename in [

        "study_coverage.tsv",

        "credible_sets.tsv",

        "credible_set_variants.tsv",

        "l2g_predictions.tsv",

        "l2g_features.tsv",

        "colocalisation.tsv",

        "benchmark_metadata.json",

        "opentargets_raw.json.gz",

    ]:

        path = (
            output
            /
            filename
        )

        if path.exists():

            print(
                path
            )

    section(
        "DONE"
    )


if __name__ == "__main__":

    main()
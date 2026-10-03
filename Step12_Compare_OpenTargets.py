#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
GWAS2m STEP 12
OPEN TARGETS / GENTROPY EXTERNAL LOCUS BENCHMARK
===============================================================================

PURPOSE
-------
Compare GWAS2m fine-mapped loci with the current Open Targets Platform
credible sets for exactly the same GCST studies.

For each study:

    GWAS2m Step06 loci
            ?
    Open Targets credible sets
            ?
    same GCST + same chromosome
            ?
    genomic interval overlap
            ?

Classify:

    MATCHED
        +-- EXACT_LEAD
        +-- STRONG_OVERLAP
        +-- PARTIAL_OVERLAP

    GWAS2M_ONLY

    OPENTARGETS_ONLY


IMPORTANT
---------
L001 in one pipeline is NOT compared to L001 in another pipeline.

Matching uses:

    STUDY_ID
    CHROMOSOME
    LOCUS_START
    LOCUS_END

One GWAS2m locus is allowed to overlap multiple Open Targets credible sets,
because a broad physical locus may contain multiple independent signals.


RUN
---

    python Step12_Compare_OpenTargets.py \
        --phenotype migraine \
        --ancestry EUR

Single study:

    python Step12_Compare_OpenTargets.py \
        --phenotype migraine \
        --ancestry EUR \
        --study GCST90129450


OUTPUT
------

12_opentargets_comparison/
    <phenotype>/
        <ancestry>/
            opentargets_credible_sets.tsv
            gwas2m_loci.tsv
            all_overlap_pairs.tsv
            gwas2m_locus_comparison.tsv
            opentargets_credible_set_comparison.tsv
            study_comparison_summary.tsv
            benchmark_metadata.json
            opentargets_credible_sets_raw.json

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# VERSION / OPEN TARGETS
# =============================================================================

VERSION = "1.0.0"

OPEN_TARGETS_RELEASE = "26.09"

OPEN_TARGETS_API = (
    "https://api.platform.opentargets.org/api/v4/graphql"
)

PAGE_SIZE = 500


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




def json_safe(obj):
    """Convert NumPy/Pandas objects to standard Python types for JSON."""
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
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def normalise_text(value):

    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^a-z0-9]+",
            " ",
            str(value).lower(),
        ),
    ).strip()


def slugify(value):

    return normalise_text(
        value
    ).replace(
        " ",
        "_",
    )


def ancestry_info(value):

    key = normalise_text(
        value
    )

    if key not in ANCESTRY:

        raise SystemExit(
            f"Unsupported ancestry: {value}"
        )

    return ANCESTRY[key]


def normalise_chr(value):

    if pd.isna(value):

        return None

    value = str(
        value
    ).strip()

    value = re.sub(
        r"^chr",
        "",
        value,
        flags=re.I,
    )

    value = value.upper()

    if value.endswith(
        ".0"
    ):

        value = value[:-2]

    return value


def section(title):

    print()

    print(
        "=" * 150
    )

    print(title)

    print(
        "=" * 150
    )


def show(
    df,
    title,
    max_rows=50,
):

    section(title)

    if (
        df is None
        or
        len(df) == 0
    ):

        print(
            "(no rows)"
        )

        return

    if (
        max_rows
        and
        len(df) > max_rows
    ):

        x = df.head(
            max_rows
        )

    else:

        x = df

    with pd.option_context(

        "display.max_rows",
        None,

        "display.max_columns",
        None,

        "display.width",
        500,

        "display.max_colwidth",
        80,

        "display.expand_frame_repr",
        False,

        "display.float_format",
        lambda v:
            f"{v:.6g}",

    ):

        print(
            x.to_string(
                index=False
            )
        )

    if len(x) < len(df):

        print(
            f"\n[showing "
            f"{len(x):,}/"
            f"{len(df):,} rows]"
        )


def semicolon_unique(values):

    output = []

    for value in values:

        if pd.isna(value):

            continue

        value = str(
            value
        ).strip()

        if not value:

            continue

        if value not in output:

            output.append(
                value
            )

    return ";".join(
        output
    )


def numeric(value):

    try:

        return float(
            value
        )

    except Exception:

        return np.nan


# =============================================================================
# HTTP
# =============================================================================

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

        pool_connections=4,

        pool_maxsize=4,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent":
                "GWAS2m-OpenTargets-Comparison/1.0"
        }
    )

    return session


def graphql(
    session,
    query,
    variables,
):

    response = session.post(

        OPEN_TARGETS_API,

        json={
            "query":
                query,

            "variables":
                variables,
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
            in payload[
                "errors"
            ]
        )

        raise RuntimeError(
            message
        )

    return payload[
        "data"
    ]


# =============================================================================
# DISCOVER GCST STUDIES
# =============================================================================

def discover_studies(
    root,
    phenotype,
    ancestry_label,
):

    phenotype_slug = slugify(
        phenotype
    )

    ancestry_slug = slugify(
        ancestry_label
    )

    bases = [

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

    for base in bases:

        if not base.exists():

            continue

        for path in base.iterdir():

            if (
                path.is_dir()
                and
                re.fullmatch(
                    r"GCST\d+",
                    path.name,
                )
            ):

                studies.add(
                    path.name
                )

    return sorted(
        studies
    )


# =============================================================================
# LOAD GWAS2m STEP06 LOCI
# =============================================================================

def parse_lead_positions(value):

    if pd.isna(value):

        return []

    values = re.findall(
        r"\d+",
        str(value),
    )

    output = []

    for x in values:

        try:

            output.append(
                int(x)
            )

        except Exception:

            pass

    return output


def load_gwas2m_loci(
    root,
    phenotype,
    ancestry_label,
    studies,
):

    phenotype_slug = slugify(
        phenotype
    )

    ancestry_slug = slugify(
        ancestry_label
    )

    rows = []

    for study in studies:

        path = (

            root
            /
            "06_finemapping"
            /
            phenotype_slug
            /
            ancestry_slug
            /
            study
            /
            "loci_definition.tsv"
        )

        if not path.exists():

            continue

        try:

            table = pd.read_csv(

                path,

                sep="\t",

                low_memory=False,
            )

        except Exception as error:

            print(
                f"[WARN] Could not read "
                f"{path}: {error}"
            )

            continue

        if table.empty:

            continue

        for _, row in table.iterrows():

            chromosome = normalise_chr(

                row.get(
                    "CHR"
                )
            )

            start = pd.to_numeric(

                row.get(
                    "LOCUS_START"
                ),

                errors="coerce",
            )

            end = pd.to_numeric(

                row.get(
                    "LOCUS_END"
                ),

                errors="coerce",
            )

            if (
                chromosome is None
                or
                pd.isna(start)
                or
                pd.isna(end)
            ):

                continue

            lead_positions_raw = row.get(
                "LEAD_POSITIONS",
                "",
            )

            lead_positions = parse_lead_positions(
                lead_positions_raw
            )

            rows.append(
                {

                    "STUDY_ID":
                        study,

                    "GWAS2M_LOCUS_ID":
                        row.get(
                            "LOCUS_ID"
                        ),

                    "CHR":
                        chromosome,

                    "GWAS2M_START":
                        int(start),

                    "GWAS2M_END":
                        int(end),

                    "GWAS2M_N_INDEPENDENT_SIGNALS":
                        row.get(
                            "N_INDEPENDENT_SIGNALS",
                            np.nan,
                        ),

                    "GWAS2M_LEAD_IDS":
                        row.get(
                            "LEAD_IDS",
                            "",
                        ),

                    "GWAS2M_LEAD_POSITIONS":
                        lead_positions_raw,

                    "_LEAD_POSITIONS_LIST":
                        lead_positions,

                    "GWAS2M_MIN_LEAD_P":
                        row.get(
                            "MIN_LEAD_P",
                            np.nan,
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# OPEN TARGETS CREDIBLE SET QUERY
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
            studyId

            chromosome
            position

            region
            locusStart
            locusEnd

            beta
            zScore
            standardError

            pValueMantissa
            pValueExponent

            finemappingMethod
            credibleSetIndex
            credibleSetlog10BF

            purityMeanR2
            purityMinR2

            sampleSize
            confidence
            qualityControls

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


def fetch_opentargets_credible_sets(
    studies,
):

    session = make_session()

    rows = []

    page_index = 0

    while True:

        data = graphql(

            session,

            CREDIBLE_SET_QUERY,

            {
                "studyIds":
                    studies,

                "page":
                    {
                        "index":
                            page_index,

                        "size":
                            PAGE_SIZE,
                    },
            },
        )

        block = data[
            "credibleSets"
        ]

        batch = block.get(
            "rows",
            []
        )

        total = int(
            block.get(
                "count",
                0,
            )
        )

        rows.extend(
            batch
        )

        print(

            f"[Open Targets] "
            f"credible sets "
            f"{len(rows):,}/"
            f"{total:,}",

            flush=True,
        )

        if (
            not batch
            or
            len(rows) >= total
        ):

            break

        page_index += 1

    return rows


# =============================================================================
# OPEN TARGETS REGION PARSING
# =============================================================================

def parse_region_string(value):

    if value is None:

        return (
            None,
            None,
            None,
        )

    value = str(
        value
    ).strip()

    patterns = [

        r"(?:chr)?([0-9XYM]+)"
        r"[:_]"
        r"([0-9]+)"
        r"[-_:]"
        r"([0-9]+)",

        r"(?:chr)?([0-9XYM]+)"
        r"[^0-9]+"
        r"([0-9]+)"
        r"[^0-9]+"
        r"([0-9]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            value,
            flags=re.I,
        )

        if match:

            return (

                normalise_chr(
                    match.group(
                        1
                    )
                ),

                int(
                    match.group(
                        2
                    )
                ),

                int(
                    match.group(
                        3
                    )
                ),
            )

    return (
        None,
        None,
        None,
    )


def flatten_opentargets(
    raw_rows,
):

    rows = []

    for item in raw_rows:

        variant = (
            item.get(
                "variant"
            )
            or
            {}
        )

        study_id = (
            item.get(
                "studyId"
            )
            or
            ""
        )

        chromosome = normalise_chr(

            item.get(
                "chromosome"
            )
            or
            variant.get(
                "chromosome"
            )
        )

        lead_position = (

            item.get(
                "position"
            )

            if item.get(
                "position"
            )
            is not None

            else

            variant.get(
                "position"
            )
        )

        locus_start = item.get(
            "locusStart"
        )

        locus_end = item.get(
            "locusEnd"
        )

        interval_source = (
            "locusStart/locusEnd"
        )

        if (
            locus_start is None
            or
            locus_end is None
        ):

            (
                region_chr,
                region_start,
                region_end,

            ) = parse_region_string(

                item.get(
                    "region"
                )
            )

            if (
                region_start is not None
                and
                region_end is not None
            ):

                locus_start = (
                    region_start
                )

                locus_end = (
                    region_end
                )

                if chromosome is None:

                    chromosome = (
                        region_chr
                    )

                interval_source = (
                    "region"
                )

        if (
            locus_start is None
            or
            locus_end is None
        ):

            if lead_position is not None:

                locus_start = (
                    lead_position
                )

                locus_end = (
                    lead_position
                )

                interval_source = (
                    "lead_position_fallback"
                )

        if (
            chromosome is None
            or
            locus_start is None
            or
            locus_end is None
        ):

            continue

        mantissa = item.get(
            "pValueMantissa"
        )

        exponent = item.get(
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

                    float(
                        mantissa
                    )

                    *

                    (
                        10.0
                        **
                        float(
                            exponent
                        )
                    )
                )

        except Exception:

            pass

        rows.append(
            {

                "STUDY_ID":
                    study_id,

                "OT_STUDY_LOCUS_ID":
                    item.get(
                        "studyLocusId"
                    ),

                "CHR":
                    chromosome,

                "OT_START":
                    int(
                        locus_start
                    ),

                "OT_END":
                    int(
                        locus_end
                    ),

                "OT_INTERVAL_SOURCE":
                    interval_source,

                "OT_LEAD_VARIANT":
                    variant.get(
                        "id"
                    ),

                "OT_LEAD_POSITION":
                    (
                        int(
                            lead_position
                        )

                        if lead_position
                        is not None

                        else
                        np.nan
                    ),

                "OT_RSIDS":
                    ";".join(
                        variant.get(
                            "rsIds"
                        )
                        or
                        []
                    ),

                "OT_REF":
                    variant.get(
                        "referenceAllele"
                    ),

                "OT_ALT":
                    variant.get(
                        "alternateAllele"
                    ),

                "OT_P_VALUE":
                    pvalue,

                "OT_FINEMAPPING_METHOD":
                    item.get(
                        "finemappingMethod"
                    ),

                "OT_CONFIDENCE":
                    item.get(
                        "confidence"
                    ),

                "OT_CREDIBLE_SET_INDEX":
                    item.get(
                        "credibleSetIndex"
                    ),

                "OT_LOG10_BF":
                    item.get(
                        "credibleSetlog10BF"
                    ),

                "OT_PURITY_MEAN_R2":
                    item.get(
                        "purityMeanR2"
                    ),

                "OT_PURITY_MIN_R2":
                    item.get(
                        "purityMinR2"
                    ),

                "OT_SAMPLE_SIZE":
                    item.get(
                        "sampleSize"
                    ),

                "OT_QC_FLAGS":
                    ";".join(
                        item.get(
                            "qualityControls"
                        )
                        or
                        []
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# INTERVAL COMPARISON
# =============================================================================

def interval_metrics(
    a_start,
    a_end,
    b_start,
    b_end,
):

    intersection = max(

        0,

        min(
            a_end,
            b_end,
        )

        -

        max(
            a_start,
            b_start,
        )

        +
        1,
    )

    a_length = (
        a_end
        -
        a_start
        +
        1
    )

    b_length = (
        b_end
        -
        b_start
        +
        1
    )

    union = (

        max(
            a_end,
            b_end,
        )

        -

        min(
            a_start,
            b_start,
        )

        +
        1
    )

    if intersection == 0:

        return {
            "OVERLAP_BP":
                0,

            "JACCARD":
                0.0,

            "GWAS2M_OVERLAP_FRACTION":
                0.0,

            "OT_OVERLAP_FRACTION":
                0.0,

            "RECIPROCAL_OVERLAP":
                0.0,
        }

    a_fraction = (
        intersection
        /
        a_length
    )

    b_fraction = (
        intersection
        /
        b_length
    )

    return {

        "OVERLAP_BP":
            intersection,

        "JACCARD":
            (
                intersection
                /
                union
            ),

        "GWAS2M_OVERLAP_FRACTION":
            a_fraction,

        "OT_OVERLAP_FRACTION":
            b_fraction,

        "RECIPROCAL_OVERLAP":
            min(
                a_fraction,
                b_fraction,
            ),
    }


def nearest_lead_distance(
    gwas2m_leads,
    ot_position,
):

    if (
        not gwas2m_leads
        or
        pd.isna(
            ot_position
        )
    ):

        return np.nan

    return min(

        abs(
            int(position)
            -
            int(ot_position)
        )

        for position
        in gwas2m_leads
    )


# =============================================================================
# ALL OVERLAPPING PAIRS
# =============================================================================

def build_overlap_pairs(
    gwas2m,
    opentargets,
    strict_reciprocal,
):

    rows = []

    if (
        gwas2m.empty
        or
        opentargets.empty
    ):

        return pd.DataFrame()

    studies = sorted(

        set(
            gwas2m[
                "STUDY_ID"
            ]
        )

        |
        set(
            opentargets[
                "STUDY_ID"
            ]
        )
    )

    for study in studies:

        local = gwas2m[

            gwas2m[
                "STUDY_ID"
            ]
            ==
            study

        ]

        ot = opentargets[

            opentargets[
                "STUDY_ID"
            ]
            ==
            study

        ]

        if (
            local.empty
            or
            ot.empty
        ):

            continue

        for _, grow in local.iterrows():

            same_chr = ot[

                ot[
                    "CHR"
                ]
                ==
                grow[
                    "CHR"
                ]

            ]

            for _, orow in same_chr.iterrows():

                metrics = interval_metrics(

                    int(
                        grow[
                            "GWAS2M_START"
                        ]
                    ),

                    int(
                        grow[
                            "GWAS2M_END"
                        ]
                    ),

                    int(
                        orow[
                            "OT_START"
                        ]
                    ),

                    int(
                        orow[
                            "OT_END"
                        ]
                    ),
                )

                if (
                    metrics[
                        "OVERLAP_BP"
                    ]
                    <=
                    0
                ):

                    continue

                lead_distance = nearest_lead_distance(

                    grow[
                        "_LEAD_POSITIONS_LIST"
                    ],

                    orow[
                        "OT_LEAD_POSITION"
                    ],
                )

                exact_lead = (

                    not pd.isna(
                        lead_distance
                    )

                    and

                    lead_distance
                    ==
                    0
                )

                if exact_lead:

                    match_class = (
                        "EXACT_LEAD"
                    )

                elif (
                    metrics[
                        "RECIPROCAL_OVERLAP"
                    ]
                    >=
                    strict_reciprocal
                ):

                    match_class = (
                        "STRONG_OVERLAP"
                    )

                else:

                    match_class = (
                        "PARTIAL_OVERLAP"
                    )

                rows.append(
                    {

                        "STUDY_ID":
                            study,

                        "CHR":
                            grow[
                                "CHR"
                            ],

                        "GWAS2M_LOCUS_ID":
                            grow[
                                "GWAS2M_LOCUS_ID"
                            ],

                        "GWAS2M_START":
                            grow[
                                "GWAS2M_START"
                            ],

                        "GWAS2M_END":
                            grow[
                                "GWAS2M_END"
                            ],

                        "GWAS2M_LEAD_POSITIONS":
                            grow[
                                "GWAS2M_LEAD_POSITIONS"
                            ],

                        "GWAS2M_N_INDEPENDENT_SIGNALS":
                            grow[
                                "GWAS2M_N_INDEPENDENT_SIGNALS"
                            ],

                        "OT_STUDY_LOCUS_ID":
                            orow[
                                "OT_STUDY_LOCUS_ID"
                            ],

                        "OT_START":
                            orow[
                                "OT_START"
                            ],

                        "OT_END":
                            orow[
                                "OT_END"
                            ],

                        "OT_LEAD_VARIANT":
                            orow[
                                "OT_LEAD_VARIANT"
                            ],

                        "OT_LEAD_POSITION":
                            orow[
                                "OT_LEAD_POSITION"
                            ],

                        "OT_FINEMAPPING_METHOD":
                            orow[
                                "OT_FINEMAPPING_METHOD"
                            ],

                        "OT_CONFIDENCE":
                            orow[
                                "OT_CONFIDENCE"
                            ],

                        **metrics,

                        "LEAD_DISTANCE_BP":
                            lead_distance,

                        "EXACT_LEAD":
                            exact_lead,

                        "STRICT_MATCH":
                            (
                                exact_lead
                                or
                                metrics[
                                    "RECIPROCAL_OVERLAP"
                                ]
                                >=
                                strict_reciprocal
                            ),

                        "MATCH_CLASS":
                            match_class,
                    }
                )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# BEST MATCH ORDER
# =============================================================================

MATCH_RANK = {

    "EXACT_LEAD":
        3,

    "STRONG_OVERLAP":
        2,

    "PARTIAL_OVERLAP":
        1,
}


def best_pair(
    group,
):

    if group.empty:

        return None

    x = group.copy()

    x[
        "_MATCH_RANK"
    ] = x[
        "MATCH_CLASS"
    ].map(
        MATCH_RANK
    ).fillna(
        0
    )

    x[
        "_LEAD_DISTANCE_SORT"
    ] = pd.to_numeric(

        x[
            "LEAD_DISTANCE_BP"
        ],

        errors="coerce",

    ).fillna(
        np.inf
    )

    x = x.sort_values(

        [
            "_MATCH_RANK",
            "RECIPROCAL_OVERLAP",
            "JACCARD",
            "_LEAD_DISTANCE_SORT",
        ],

        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    return x.iloc[
        0
    ]


# =============================================================================
# ONE ROW PER GWAS2m LOCUS
# =============================================================================

def summarise_gwas2m_loci(
    gwas2m,
    pairs,
):

    rows = []

    for _, locus in gwas2m.iterrows():

        if pairs.empty:

            matches = pd.DataFrame()

        else:

            matches = pairs[

                (
                    pairs[
                        "STUDY_ID"
                    ]
                    ==
                    locus[
                        "STUDY_ID"
                    ]
                )

                &

                (
                    pairs[
                        "GWAS2M_LOCUS_ID"
                    ]
                    ==
                    locus[
                        "GWAS2M_LOCUS_ID"
                    ]
                )

            ]

        row = {

            key:
                value

            for key, value
            in locus.items()

            if not key.startswith(
                "_"
            )
        }

        if matches.empty:

            row.update(
                {

                    "COMPARISON_STATUS":
                        "GWAS2M_ONLY",

                    "N_OT_OVERLAPS":
                        0,

                    "OT_STUDY_LOCUS_IDS":
                        "",

                    "OT_METHODS":
                        "",

                    "OT_LEADS":
                        "",

                    "BEST_MATCH_CLASS":
                        "",

                    "BEST_JACCARD":
                        np.nan,

                    "BEST_RECIPROCAL_OVERLAP":
                        np.nan,

                    "MIN_LEAD_DISTANCE_BP":
                        np.nan,

                    "EXACT_LEAD_ANY":
                        False,

                    "STRICT_MATCH_ANY":
                        False,
                }
            )

        else:

            best = best_pair(
                matches
            )

            row.update(
                {

                    "COMPARISON_STATUS":
                        "MATCHED",

                    "N_OT_OVERLAPS":
                        matches[
                            "OT_STUDY_LOCUS_ID"
                        ].nunique(),

                    "OT_STUDY_LOCUS_IDS":
                        semicolon_unique(

                            matches[
                                "OT_STUDY_LOCUS_ID"
                            ]
                        ),

                    "OT_METHODS":
                        semicolon_unique(

                            matches[
                                "OT_FINEMAPPING_METHOD"
                            ]
                        ),

                    "OT_LEADS":
                        semicolon_unique(

                            matches[
                                "OT_LEAD_VARIANT"
                            ]
                        ),

                    "BEST_MATCH_CLASS":
                        best[
                            "MATCH_CLASS"
                        ],

                    "BEST_JACCARD":
                        best[
                            "JACCARD"
                        ],

                    "BEST_RECIPROCAL_OVERLAP":
                        best[
                            "RECIPROCAL_OVERLAP"
                        ],

                    "MIN_LEAD_DISTANCE_BP":
                        pd.to_numeric(

                            matches[
                                "LEAD_DISTANCE_BP"
                            ],

                            errors="coerce",

                        ).min(),

                    "EXACT_LEAD_ANY":
                        bool(

                            matches[
                                "EXACT_LEAD"
                            ].any()
                        ),

                    "STRICT_MATCH_ANY":
                        bool(

                            matches[
                                "STRICT_MATCH"
                            ].any()
                        ),
                }
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# ONE ROW PER OPEN TARGETS CREDIBLE SET
# =============================================================================

def summarise_ot_loci(
    opentargets,
    pairs,
):

    rows = []

    for _, locus in opentargets.iterrows():

        if pairs.empty:

            matches = pd.DataFrame()

        else:

            matches = pairs[

                (
                    pairs[
                        "STUDY_ID"
                    ]
                    ==
                    locus[
                        "STUDY_ID"
                    ]
                )

                &

                (
                    pairs[
                        "OT_STUDY_LOCUS_ID"
                    ]
                    ==
                    locus[
                        "OT_STUDY_LOCUS_ID"
                    ]
                )

            ]

        row = locus.to_dict()

        if matches.empty:

            row.update(
                {

                    "COMPARISON_STATUS":
                        "OPENTARGETS_ONLY",

                    "N_GWAS2M_OVERLAPS":
                        0,

                    "GWAS2M_LOCUS_IDS":
                        "",

                    "BEST_MATCH_CLASS":
                        "",

                    "BEST_JACCARD":
                        np.nan,

                    "BEST_RECIPROCAL_OVERLAP":
                        np.nan,

                    "MIN_LEAD_DISTANCE_BP":
                        np.nan,

                    "EXACT_LEAD_ANY":
                        False,

                    "STRICT_MATCH_ANY":
                        False,
                }
            )

        else:

            best = best_pair(
                matches
            )

            row.update(
                {

                    "COMPARISON_STATUS":
                        "MATCHED",

                    "N_GWAS2M_OVERLAPS":
                        matches[
                            "GWAS2M_LOCUS_ID"
                        ].nunique(),

                    "GWAS2M_LOCUS_IDS":
                        semicolon_unique(

                            matches[
                                "GWAS2M_LOCUS_ID"
                            ]
                        ),

                    "BEST_MATCH_CLASS":
                        best[
                            "MATCH_CLASS"
                        ],

                    "BEST_JACCARD":
                        best[
                            "JACCARD"
                        ],

                    "BEST_RECIPROCAL_OVERLAP":
                        best[
                            "RECIPROCAL_OVERLAP"
                        ],

                    "MIN_LEAD_DISTANCE_BP":
                        pd.to_numeric(

                            matches[
                                "LEAD_DISTANCE_BP"
                            ],

                            errors="coerce",

                        ).min(),

                    "EXACT_LEAD_ANY":
                        bool(

                            matches[
                                "EXACT_LEAD"
                            ].any()
                        ),

                    "STRICT_MATCH_ANY":
                        bool(

                            matches[
                                "STRICT_MATCH"
                            ].any()
                        ),
                }
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# STUDY SUMMARY
# =============================================================================

def method_string(
    table,
):

    if table.empty:

        return ""

    values = (

        table[
            "OT_FINEMAPPING_METHOD"
        ]

        .fillna(
            "NA"
        )

        .value_counts()
    )

    return ";".join(

        f"{method}:{count}"

        for method, count
        in values.items()
    )


def build_study_summary(
    studies,
    gwas2m,
    opentargets,
    gwas_comp,
    ot_comp,
):

    rows = []

    for study in studies:

        g = gwas2m[

            gwas2m[
                "STUDY_ID"
            ]
            ==
            study

        ]

        o = opentargets[

            opentargets[
                "STUDY_ID"
            ]
            ==
            study

        ]

        gc = gwas_comp[

            gwas_comp[
                "STUDY_ID"
            ]
            ==
            study

        ]

        oc = ot_comp[

            ot_comp[
                "STUDY_ID"
            ]
            ==
            study

        ]

        rows.append(
            {

                "STUDY_ID":
                    study,

                "N_GWAS2M_LOCI":
                    len(g),

                "N_OT_CREDIBLE_SETS":
                    len(o),

                "N_GWAS2M_MATCHED":
                    (
                        (
                            gc[
                                "COMPARISON_STATUS"
                            ]
                            ==
                            "MATCHED"
                        ).sum()

                        if not gc.empty

                        else
                        0
                    ),

                "N_GWAS2M_ONLY":
                    (
                        (
                            gc[
                                "COMPARISON_STATUS"
                            ]
                            ==
                            "GWAS2M_ONLY"
                        ).sum()

                        if not gc.empty

                        else
                        0
                    ),

                "N_OT_MATCHED":
                    (
                        (
                            oc[
                                "COMPARISON_STATUS"
                            ]
                            ==
                            "MATCHED"
                        ).sum()

                        if not oc.empty

                        else
                        0
                    ),

                "N_OT_ONLY":
                    (
                        (
                            oc[
                                "COMPARISON_STATUS"
                            ]
                            ==
                            "OPENTARGETS_ONLY"
                        ).sum()

                        if not oc.empty

                        else
                        0
                    ),

                "N_GWAS2M_STRICT_MATCH":
                    (
                        gc[
                            "STRICT_MATCH_ANY"
                        ].fillna(
                            False
                        ).sum()

                        if not gc.empty

                        else
                        0
                    ),

                "N_EXACT_LEAD_MATCH":
                    (
                        gc[
                            "EXACT_LEAD_ANY"
                        ].fillna(
                            False
                        ).sum()

                        if not gc.empty

                        else
                        0
                    ),

                "OT_METHODS":
                    method_string(
                        o
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

        "--strict-reciprocal-overlap",

        type=float,

        default=0.50,

        help=(
            "Minimum reciprocal interval overlap "
            "for STRONG_OVERLAP. "
            "Default: 0.50"
        ),
    )

    parser.add_argument(

        "--max-print",

        type=int,

        default=50,
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

            study

            for study
            in studies

            if study
            ==
            args.study
        ]

    if not studies:

        raise SystemExit(
            "No matching GCST studies found."
        )

    output = (

        root
        /
        "12_opentargets_comparison"
        /
        slugify(
            args.phenotype
        )
        /
        slugify(
            ancestry_label
        )
    )

    output.mkdir(

        parents=True,

        exist_ok=True,
    )

    section(
        "GWAS2m STEP12 - OPEN TARGETS / GENTROPY LOCUS BENCHMARK"
    )

    print(
        f"Version                    : {VERSION}"
    )

    print(
        f"GWAS2m root                : {root}"
    )

    print(
        f"Phenotype                  : {args.phenotype}"
    )

    print(
        f"Ancestry                   : "
        f"{ancestry_label} ({ancestry_code})"
    )

    print(
        f"Studies                    : {len(studies)}"
    )

    print(
        f"Open Targets release       : {OPEN_TARGETS_RELEASE}"
    )

    print(
        f"Strict reciprocal overlap  : "
        f"{args.strict_reciprocal_overlap}"
    )

    print(
        f"Output                     : {output}"
    )

    # =========================================================================
    # LOAD GWAS2m
    # =========================================================================

    gwas2m = load_gwas2m_loci(

        root,

        args.phenotype,

        ancestry_label,

        studies,
    )

    print(
        f"\nGWAS2m loci loaded          : "
        f"{len(gwas2m):,}"
    )

    # =========================================================================
    # OPEN TARGETS
    # =========================================================================

    raw_ot = fetch_opentargets_credible_sets(
        studies
    )

    opentargets = flatten_opentargets(
        raw_ot
    )

    print(
        f"Open Targets credible sets : "
        f"{len(opentargets):,}"
    )

    # =========================================================================
    # COMPARE
    # =========================================================================

    pairs = build_overlap_pairs(

        gwas2m,

        opentargets,

        args.strict_reciprocal_overlap,
    )

    gwas_comp = summarise_gwas2m_loci(

        gwas2m,

        pairs,
    )

    ot_comp = summarise_ot_loci(

        opentargets,

        pairs,
    )

    study_summary = build_study_summary(

        studies,

        gwas2m,

        opentargets,

        gwas_comp,

        ot_comp,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    gwas2m.drop(
        columns=[
            "_LEAD_POSITIONS_LIST"
        ],
        errors="ignore",

    ).to_csv(

        output
        /
        "gwas2m_loci.tsv",

        sep="\t",

        index=False,
    )

    opentargets.to_csv(

        output
        /
        "opentargets_credible_sets.tsv",

        sep="\t",

        index=False,
    )

    pairs.to_csv(

        output
        /
        "all_overlap_pairs.tsv",

        sep="\t",

        index=False,
    )

    gwas_comp.to_csv(

        output
        /
        "gwas2m_locus_comparison.tsv",

        sep="\t",

        index=False,
    )

    ot_comp.to_csv(

        output
        /
        "opentargets_credible_set_comparison.tsv",

        sep="\t",

        index=False,
    )

    study_summary.to_csv(

        output
        /
        "study_comparison_summary.tsv",

        sep="\t",

        index=False,
    )

    (

        output
        /
        "opentargets_credible_sets_raw.json"

    ).write_text(

        json.dumps(raw_ot, indent=2, default=json_safe)
        +
        "\n",

        encoding="utf-8",
    )

    metadata = {

        "step":
            12,

        "version":
            VERSION,

        "retrieved_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "open_targets_release":
            OPEN_TARGETS_RELEASE,

        "open_targets_api":
            OPEN_TARGETS_API,

        "phenotype":
            args.phenotype,

        "ancestry_code":
            ancestry_code,

        "ancestry_label":
            ancestry_label,

        "studies":
            studies,

        "strict_reciprocal_overlap":
            args.strict_reciprocal_overlap,

        "n_gwas2m_loci":
            len(
                gwas2m
            ),

        "n_opentargets_credible_sets":
            len(
                opentargets
            ),

        "n_overlap_pairs":
            len(
                pairs
            ),

        "n_gwas2m_matched":
            (
                (
                    gwas_comp[
                        "COMPARISON_STATUS"
                    ]
                    ==
                    "MATCHED"
                ).sum()

                if not gwas_comp.empty

                else
                0
            ),

        "n_gwas2m_only":
            (
                (
                    gwas_comp[
                        "COMPARISON_STATUS"
                    ]
                    ==
                    "GWAS2M_ONLY"
                ).sum()

                if not gwas_comp.empty

                else
                0
            ),

        "n_ot_only":
            (
                (
                    ot_comp[
                        "COMPARISON_STATUS"
                    ]
                    ==
                    "OPENTARGETS_ONLY"
                ).sum()

                if not ot_comp.empty

                else
                0
            ),
    }

    (

        output
        /
        "benchmark_metadata.json"

    ).write_text(

        json.dumps(metadata, indent=2, default=json_safe)
        +
        "\n",

        encoding="utf-8",
    )

    # =========================================================================
    # PRINT
    # =========================================================================

    show(

        study_summary,

        "TABLE 1 - STUDY-LEVEL GWAS2m vs OPEN TARGETS SUMMARY",

        args.max_print,
    )

    matched = gwas_comp[

        gwas_comp[
            "COMPARISON_STATUS"
        ]
        ==
        "MATCHED"

    ].copy()

    if not matched.empty:

        useful_columns = [

            "STUDY_ID",

            "GWAS2M_LOCUS_ID",

            "CHR",

            "GWAS2M_START",

            "GWAS2M_END",

            "GWAS2M_LEAD_POSITIONS",

            "N_OT_OVERLAPS",

            "OT_METHODS",

            "OT_LEADS",

            "BEST_MATCH_CLASS",

            "BEST_JACCARD",

            "BEST_RECIPROCAL_OVERLAP",

            "MIN_LEAD_DISTANCE_BP",

            "EXACT_LEAD_ANY",
        ]

        useful_columns = [

            x

            for x in useful_columns

            if x in matched.columns
        ]

        matched = matched[
            useful_columns
        ]

    show(

        matched,

        "TABLE 2 - SAME / OVERLAPPING REGIONS",

        args.max_print,
    )

    gwas_only = gwas_comp[

        gwas_comp[
            "COMPARISON_STATUS"
        ]
        ==
        "GWAS2M_ONLY"

    ].copy()

    show(

        gwas_only,

        "TABLE 3 - GWAS2m-ONLY REGIONS",

        args.max_print,
    )

    ot_only = ot_comp[

        ot_comp[
            "COMPARISON_STATUS"
        ]
        ==
        "OPENTARGETS_ONLY"

    ].copy()

    show(

        ot_only,

        "TABLE 4 - OPEN-TARGETS-ONLY CREDIBLE SETS",

        args.max_print,
    )

    if not pairs.empty:

        exact = pairs[

            pairs[
                "EXACT_LEAD"
            ]
            ==
            True

        ].copy()

    else:

        exact = pd.DataFrame()

    show(

        exact,

        "TABLE 5 - EXACT LEAD-POSITION AGREEMENTS",

        args.max_print,
    )

    section(
        "GLOBAL SUMMARY"
    )

    print(
        f"GWAS2m loci               : "
        f"{len(gwas2m):,}"
    )

    print(
        f"Open Targets credible sets: "
        f"{len(opentargets):,}"
    )

    print(
        f"All overlap pairs         : "
        f"{len(pairs):,}"
    )

    if not gwas_comp.empty:

        print(
            f"GWAS2m loci matched       : "
            f"{(gwas_comp['COMPARISON_STATUS'] == 'MATCHED').sum():,}"
        )

        print(
            f"GWAS2m-only loci          : "
            f"{(gwas_comp['COMPARISON_STATUS'] == 'GWAS2M_ONLY').sum():,}"
        )

        print(
            f"Strict GWAS2m matches     : "
            f"{gwas_comp['STRICT_MATCH_ANY'].fillna(False).sum():,}"
        )

        print(
            f"Exact lead matches        : "
            f"{gwas_comp['EXACT_LEAD_ANY'].fillna(False).sum():,}"
        )

    if not ot_comp.empty:

        print(
            f"Open Targets-only CS      : "
            f"{(ot_comp['COMPARISON_STATUS'] == 'OPENTARGETS_ONLY').sum():,}"
        )

    section(
        "MATCH DEFINITIONS"
    )

    print(
        "EXACT_LEAD:"
    )

    print(
        "  Same study, overlapping locus, "
        "and identical lead genomic position."
    )

    print()

    print(
        "STRONG_OVERLAP:"
    )

    print(
        "  Same study/chromosome and reciprocal "
        f"interval overlap >= "
        f"{args.strict_reciprocal_overlap:.2f}."
    )

    print()

    print(
        "PARTIAL_OVERLAP:"
    )

    print(
        "  Same study/chromosome with at least "
        "1 bp genomic overlap but below the "
        "strict reciprocal-overlap threshold."
    )

    print()

    print(
        "GWAS2M_ONLY:"
    )

    print(
        "  No Open Targets credible set overlaps "
        "that GWAS2m locus in the same GCST study."
    )

    print()

    print(
        "OPENTARGETS_ONLY:"
    )

    print(
        "  Open Targets credible set has no "
        "overlapping GWAS2m Step06 locus."
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "  Open Targets PICS and SuSiE-inf loci "
        "should be analysed separately in the "
        "final benchmark."
    )

    section(
        "STEP12 COMPLETE"
    )


if __name__ == "__main__":

    main()
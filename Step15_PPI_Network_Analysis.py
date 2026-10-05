#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GWAS2m STEP15 v1.0
PPI / NETWORK ANALYSIS WITH STRING + OPTIONAL BioGRID CROSS-VALIDATION
================================================================================

PURPOSE
-------
Step15 takes the deduplicated, ranked gene sets from Step14 and asks:

1. Do the GWAS2m high-confidence genes interact with one another more than expected?
2. Which genes are central / bridging nodes?
3. Do Open Targets-supported genes connect to GWAS2m-only high-H4 genes?
4. Are there first-order connector proteins that link otherwise disconnected seeds?
5. Are the same interactions supported by BioGRID when an access key is supplied?
6. Which network modules correspond to Step14 pathway signals?

PRIMARY INPUTS
--------------
14_pathways/<phenotype>/<ancestry>/
    Step14_Gene_Ranking.tsv
    Step14_Gene_Set_Membership.tsv
    Step14_Pathway_Significant.tsv            (optional annotation)

OPTIONAL INPUTS
---------------
--known-genes-file <file>
    One disease-gene identifier/symbol per line or a table with a gene column.

--biogrid-access-key <KEY>
    Or environment variable BIOGRID_ACCESS_KEY.

DEFAULT NETWORK STRATEGY
------------------------
A. DIRECT PHYSICAL NETWORK
   CORE_STRONG + OPEN_TARGETS_TOP1
   STRING physical network, required score >= 700, no added nodes.

B. DIRECT FUNCTIONAL NETWORK
   Same seeds, STRING functional network, required score >= 700.

C. EXPANDED PHYSICAL NETWORK
   Same seeds + up to --add-nodes (default 20) high-confidence STRING connector
   proteins. Added nodes are labelled CONNECTOR and are NEVER treated as GWAS2m
   genetic evidence.

D. PPI ENRICHMENT
   STRING PPI enrichment for:
       CORE_STRONG
       CORE_STRONG_PRIOR_ROBUST
       LOCUS_TOP_STRONG
       OPEN_TARGETS_TOP1
       CORE_STRONG_OT_SUPPORTED
       CORE_STRONG_OT_DISCORDANT

IMPORTANT SCIENTIFIC RULES
--------------------------
* A STRING edge is not proof of direct physical binding unless network_type=physical.
* STRING "functional" edges can reflect multiple evidence channels.
* Added connector proteins are hypothesis-generating network bridges only.
* Non-protein-coding genes may not map to STRING and should remain NOT_MAPPABLE,
  not be called biologically absent.
* Open Targets genes are external anchors; they do not contribute to GWAS2m H4.
* Centrality is descriptive. High degree can reflect generic hubs / study bias.
* PPI enrichment p-values test excess connectivity; they are not causal evidence.

INSTALL
-------
python -m pip install pandas numpy requests networkx

RUN
---
python Step15_PPI_Network_Analysis.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR

Higher-confidence network only:
python Step15_PPI_Network_Analysis.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR \
  --required-score 900 \
  --add-nodes 10

Optional BioGRID:
export BIOGRID_ACCESS_KEY="YOUR_KEY"
python Step15_PPI_Network_Analysis.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR

Dry-run without network downloads:
python Step15_PPI_Network_Analysis.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR \
  --dry-run

OUTPUTS
-------
15_network/<phenotype>/<ancestry>/

    Step15_Input_Gene_Sets.tsv
    Step15_STRING_Mapping.tsv
    Step15_STRING_Unmapped_Genes.tsv
    Step15_STRING_PPI_Enrichment.tsv

    Step15_Direct_Physical_Edges.tsv
    Step15_Direct_Functional_Edges.tsv
    Step15_Expanded_Physical_Edges.tsv
    Step15_Expanded_Functional_Edges.tsv

    Step15_Network_Nodes.tsv
    Step15_Network_Centrality.tsv
    Step15_Network_Communities.tsv
    Step15_Seed_Pair_Distances.tsv
    Step15_Connector_Genes.tsv
    Step15_OT_Bridge_Summary.tsv

    Step15_BioGRID_Interactions.tsv
    Step15_STRING_BioGRID_Edge_Comparison.tsv

    Step15_Direct_Physical.graphml
    Step15_Expanded_Physical.graphml
    Step15_STRING_Direct_Physical.svg
    Step15_STRING_Expanded_Physical.svg

    Step15_QC_Audit.tsv
    Step15_summary.json
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'requests'. Install with:\n"
        "  python -m pip install requests"
    ) from exc

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'networkx'. Install with:\n"
        "  python -m pip install networkx"
    ) from exc


VERSION = "1.0.1"

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

PRIMARY_SET_ORDER = [
    "CORE_STRONG",
    "CORE_STRONG_PRIOR_ROBUST",
    "CORE_STRONG_MULTI_QTL",
    "LOCUS_TOP_STRONG",
    "CORE_STRONG_OT_SUPPORTED",
    "CORE_STRONG_OT_DISCORDANT",
    "CORE_STRONG_OT_COVERAGE_GAP",
    "OPEN_TARGETS_TOP1",
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
            f"Unsupported ancestry {value!r}. "
            "Use EUR, AFR, EAS, SAS or AMR."
        )
    return ANCESTRY[key]


def section(title, char="="):
    print()
    print(char * 154)
    print(title)
    print(char * 154)


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
        "display.width", 1200,
        "display.max_colwidth", 120,
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


def dedupe_keep_order(values):
    out = []
    seen = set()

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)

    return out


def edge_key(a, b):
    return tuple(sorted([str(a), str(b)]))


def pair_label(a, b):
    return f"{a} -- {b}"


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_step14_gene_ranking(path):
    if not path.exists():
        raise SystemExit(
            "Missing Step14 gene ranking:\n"
            f"  {path}"
        )

    x = pd.read_csv(path, sep="\t", low_memory=False)

    required = {
        "EFFECTOR_GENE_ID",
        "EFFECTOR_GENE_SYMBOL",
        "MAX_H4",
    }

    missing = sorted(required - set(x.columns))

    if missing:
        raise SystemExit(
            f"Step14_Gene_Ranking.tsv missing required columns: {missing}"
        )

    x["EFFECTOR_GENE_ID"] = x["EFFECTOR_GENE_ID"].map(clean_gene_id)
    x["EFFECTOR_GENE_SYMBOL"] = x["EFFECTOR_GENE_SYMBOL"].map(clean_symbol)
    x["MAX_H4"] = safe_numeric(x["MAX_H4"])

    return x


def load_step14_membership(path):
    if not path.exists():
        raise SystemExit(
            "Missing Step14 gene-set membership:\n"
            f"  {path}"
        )

    x = pd.read_csv(path, sep="\t", low_memory=False)

    if not {"GENE_SET", "GENE_ID"}.issubset(x.columns):
        raise SystemExit(
            "Step14_Gene_Set_Membership.tsv must contain GENE_SET and GENE_ID."
        )

    x["GENE_ID"] = x["GENE_ID"].map(clean_gene_id)

    return x


def load_pathways(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return pd.DataFrame()


def load_gene_sets(membership):
    result = {}

    for name, group in membership.groupby("GENE_SET", sort=False):
        result[str(name)] = dedupe_keep_order([
            clean_gene_id(v)
            for v in group["GENE_ID"]
            if clean_gene_id(v)
        ])

    return result


def load_known_genes(path):
    if not path:
        return []

    path = Path(path).resolve()

    if not path.exists():
        raise SystemExit(f"Known-gene file not found: {path}")

    # Try as table first.
    for sep in ["\t", ","]:
        try:
            x = pd.read_csv(path, sep=sep, low_memory=False)

            for candidate in [
                "GENE_ID",
                "EFFECTOR_GENE_ID",
                "ENSEMBL_GENE_ID",
                "gene_id",
                "gene",
                "GENE",
                "SYMBOL",
                "symbol",
            ]:
                if candidate in x.columns:
                    return dedupe_keep_order([
                        str(v).strip()
                        for v in x[candidate]
                        if str(v).strip()
                        and str(v).lower() not in {"nan", "none"}
                    ])
        except Exception:
            pass

    return dedupe_keep_order([
        line.strip()
        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
        and not line.startswith("#")
    ])


# =============================================================================
# GENE LABELS / CLASSES
# =============================================================================

def gene_lookup_table(genes):
    out = {}

    for row in genes.itertuples(index=False):
        gid = clean_gene_id(getattr(row, "EFFECTOR_GENE_ID", ""))

        if not gid:
            continue

        out[gid] = {
            "symbol": clean_symbol(
                getattr(row, "EFFECTOR_GENE_SYMBOL", "")
            ),
            "max_h4": safe_float(
                getattr(row, "MAX_H4", np.nan)
            ),
            "prior_robust": str(
                getattr(row, "PRIOR_ROBUST", "")
            ),
            "qtl_types": str(
                getattr(row, "QTL_TYPES", "")
            ),
            "n_tissues": getattr(
                row, "N_TISSUES", np.nan
            ),
            "ot_l2g": str(
                getattr(row, "OT_L2G_SUPPORTED_ANY", "")
            ),
            "ot_match": str(
                getattr(row, "OT_ANY_LOCUS_MATCH", "")
            ),
            "rank": getattr(
                row, "RANK_STEP14", np.nan
            ),
        }

    return out


def display_gene(gid, lookup):
    gid = clean_gene_id(gid)
    meta = lookup.get(gid, {})
    symbol = clean_symbol(meta.get("symbol", ""))

    return f"{symbol} ({gid})" if symbol else gid


def gene_set_flags(gid, gene_sets):
    flags = []

    for name, members in gene_sets.items():
        if gid in members:
            flags.append(name)

    return flags


def classify_seed_gene(gid, gene_sets, known_gene_ids):
    in_gwas = gid in set(gene_sets.get("CORE_STRONG", []))
    in_ot = gid in set(gene_sets.get("OPEN_TARGETS_TOP1", []))
    in_known = gid in set(known_gene_ids)

    if in_gwas and in_ot:
        return "GWAS2M_STRONG_AND_OT"

    if in_gwas and in_known:
        return "GWAS2M_STRONG_AND_KNOWN"

    if in_ot and in_known:
        return "OT_AND_KNOWN"

    if in_gwas:
        return "GWAS2M_STRONG"

    if in_ot:
        return "OPEN_TARGETS_TOP1"

    if in_known:
        return "KNOWN_DISEASE_GENE"

    return "OTHER_SEED"


# =============================================================================
# API CACHE / REQUESTS
# =============================================================================

class CachedHTTP:
    def __init__(self, cache_dir, timeout=120, min_interval=1.05):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_call = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GWAS2m-Step15/1.0 network-analysis"
        })

    def _cache_path(self, method, url, params, extension="txt"):
        payload = json.dumps(
            {
                "method": method.upper(),
                "url": url,
                "params": params,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")

        key = hashlib.sha256(payload).hexdigest()

        return self.cache_dir / f"{key}.{extension}"

    def request(
        self,
        method,
        url,
        params=None,
        *,
        as_json=False,
        binary=False,
        force=False,
        retries=3,
    ):
        params = params or {}

        ext = "json" if as_json else ("bin" if binary else "txt")

        cache_path = self._cache_path(
            method,
            url,
            params,
            extension=ext,
        )

        if cache_path.exists() and not force:
            if binary:
                return cache_path.read_bytes()
            text = cache_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            return json.loads(text) if as_json else text

        for attempt in range(1, retries + 1):
            elapsed = time.time() - self._last_call

            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            try:
                if method.upper() == "POST":
                    response = self.session.post(
                        url,
                        data=params,
                        timeout=self.timeout,
                    )
                else:
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=self.timeout,
                    )

                self._last_call = time.time()
                response.raise_for_status()

                if binary:
                    content = response.content
                    cache_path.write_bytes(content)
                    return content

                text = response.text
                cache_path.write_text(text, encoding="utf-8")

                return json.loads(text) if as_json else text

            except Exception:
                if attempt == retries:
                    raise

                time.sleep(attempt * 2.0)


# =============================================================================
# STRING
# =============================================================================

def resolve_string_api(http, api_override=""):
    if api_override:
        base = api_override.rstrip("/")

        if not base.endswith("/api"):
            base += "/api"

        return base, "USER_OVERRIDE", base.replace("/api", "")

    version_url = "https://string-db.org/api/json/version"

    try:
        data = http.request(
            "GET",
            version_url,
            as_json=True,
        )

        row = data[0] if isinstance(data, list) and data else data

        version = str(
            row.get("string_version", "UNKNOWN")
        )

        stable = str(
            row.get("string_stable_address", "https://string-db.org")
        ).rstrip("/")

        # STRING recommends a version-specific API address for production use.
        # Build it from the reported version when possible, while retaining the
        # stable website address separately for reporting.
        version_match = re.fullmatch(r"(\d+)\.(\d+)", version)
        if version_match:
            version_api_root = (
                f"https://version-{version_match.group(1)}-{version_match.group(2)}.string-db.org"
            )
            return version_api_root + "/api", version, stable

        return stable + "/api", version, stable

    except Exception as exc:
        print(
            f"[WARN] Could not resolve stable STRING version: {exc}"
        )
        print(
            "[WARN] Falling back to https://string-db.org/api"
        )

        return (
            "https://string-db.org/api",
            "UNKNOWN",
            "https://string-db.org",
        )


def string_map_batch(
    http,
    api_base,
    identifiers,
    species,
    caller_identity,
):
    identifiers = [
        str(x).strip()
        for x in identifiers
        if str(x).strip()
    ]

    identifiers = dedupe_keep_order(identifiers)

    if not identifiers:
        return pd.DataFrame()

    url = f"{api_base}/tsv/get_string_ids"

    params = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "echo_query": 1,
        "caller_identity": caller_identity,
    }

    try:
        text = http.request(
            "POST",
            url,
            params,
        )
    except requests.exceptions.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)

        # STRING deliberately returns HTTP 404 when NONE of the submitted
        # identifiers can be resolved. For mapping this is a valid biological/
        # annotation outcome (common for lncRNAs and unresolved symbols), not a
        # pipeline failure. Return an empty mapping and let build_string_mapping
        # mark those genes as UNRESOLVED.
        if status == 404:
            print(
                f"[INFO] STRING resolved 0/{len(identifiers)} identifiers in this mapping batch; "
                "marking them as unmapped and continuing."
            )
            return pd.DataFrame()

        raise

    if not str(text).strip():
        return pd.DataFrame()

    return pd.read_csv(
        io.StringIO(text),
        sep="\t",
        low_memory=False,
    )


def build_string_mapping(
    http,
    api_base,
    genes,
    species,
    caller_identity,
):
    lookup = gene_lookup_table(genes)

    # First attempt: stable Ensembl gene ID.
    gene_ids = list(lookup)

    first = string_map_batch(
        http,
        api_base,
        gene_ids,
        species,
        caller_identity,
    )

    rows = []
    resolved = set()

    if not first.empty:
        for row in first.itertuples(index=False):
            query = clean_gene_id(
                getattr(row, "queryItem", "")
            )

            if not query:
                continue

            rows.append({
                "GENE_ID": query,
                "GENE_SYMBOL": lookup.get(query, {}).get("symbol", ""),
                "MAPPING_QUERY": getattr(row, "queryItem", ""),
                "MAPPING_METHOD": "ENSEMBL_GENE_ID",
                "STRING_ID": getattr(row, "stringId", ""),
                "STRING_PREFERRED_NAME": getattr(row, "preferredName", ""),
                "STRING_TAXON_ID": getattr(row, "ncbiTaxonId", ""),
                "STRING_ANNOTATION": getattr(row, "annotation", ""),
                "MAPPED": "YES",
            })

            resolved.add(query)

    unresolved = [
        gid
        for gid in gene_ids
        if gid not in resolved
    ]

    # Fallback: gene symbol for unresolved genes.
    symbol_to_gene = defaultdict(list)

    for gid in unresolved:
        symbol = clean_symbol(
            lookup.get(gid, {}).get("symbol", "")
        )

        if symbol:
            symbol_to_gene[symbol].append(gid)

    if symbol_to_gene:
        second = string_map_batch(
            http,
            api_base,
            list(symbol_to_gene),
            species,
            caller_identity,
        )

        if not second.empty:
            for row in second.itertuples(index=False):
                symbol = str(
                    getattr(row, "queryItem", "")
                ).strip()

                possible = symbol_to_gene.get(symbol, [])

                if not possible:
                    continue

                # Symbols should normally map to one Step14 gene. If not,
                # apply the same mapping result to each but flag ambiguity.
                for gid in possible:
                    rows.append({
                        "GENE_ID": gid,
                        "GENE_SYMBOL": symbol,
                        "MAPPING_QUERY": symbol,
                        "MAPPING_METHOD": (
                            "GENE_SYMBOL"
                            if len(possible) == 1
                            else "GENE_SYMBOL_AMBIGUOUS"
                        ),
                        "STRING_ID": getattr(row, "stringId", ""),
                        "STRING_PREFERRED_NAME": getattr(row, "preferredName", ""),
                        "STRING_TAXON_ID": getattr(row, "ncbiTaxonId", ""),
                        "STRING_ANNOTATION": getattr(row, "annotation", ""),
                        "MAPPED": "YES",
                    })

                    resolved.add(gid)

    for gid in gene_ids:
        if gid not in resolved:
            rows.append({
                "GENE_ID": gid,
                "GENE_SYMBOL": lookup.get(gid, {}).get("symbol", ""),
                "MAPPING_QUERY": (
                    lookup.get(gid, {}).get("symbol", "")
                    or gid
                ),
                "MAPPING_METHOD": "UNRESOLVED",
                "STRING_ID": "",
                "STRING_PREFERRED_NAME": "",
                "STRING_TAXON_ID": "",
                "STRING_ANNOTATION": "",
                "MAPPED": "NO",
            })

    out = pd.DataFrame(rows)

    # One row per gene, preferring Ensembl mapping over symbol fallback.
    method_priority = {
        "ENSEMBL_GENE_ID": 0,
        "GENE_SYMBOL": 1,
        "GENE_SYMBOL_AMBIGUOUS": 2,
        "UNRESOLVED": 9,
    }

    out["_P"] = out["MAPPING_METHOD"].map(
        method_priority
    ).fillna(5)

    out = (
        out.sort_values(["GENE_ID", "_P"])
        .drop_duplicates("GENE_ID", keep="first")
        .drop(columns=["_P"])
        .reset_index(drop=True)
    )

    return out


def string_ids_for_genes(gene_ids, mapping):
    if mapping.empty:
        return []

    z = mapping[
        mapping["GENE_ID"].isin(gene_ids)
        &
        (mapping["MAPPED"] == "YES")
    ]

    return dedupe_keep_order(
        z["STRING_ID"]
        .dropna()
        .astype(str)
        .tolist()
    )


def string_network(
    http,
    api_base,
    string_ids,
    species,
    required_score,
    network_type,
    add_nodes,
    caller_identity,
):
    if len(string_ids) < 2:
        return pd.DataFrame()

    url = f"{api_base}/tsv/network"

    params = {
        "identifiers": "\r".join(string_ids),
        "species": species,
        "required_score": int(required_score),
        "network_type": network_type,
        "add_nodes": int(add_nodes),
        "caller_identity": caller_identity,
    }

    text = http.request(
        "POST",
        url,
        params,
    )

    if not text.strip():
        return pd.DataFrame()

    x = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        low_memory=False,
    )

    x["NETWORK_TYPE"] = network_type
    x["REQUIRED_SCORE"] = int(required_score)
    x["ADD_NODES"] = int(add_nodes)

    return x


def string_network_svg(
    http,
    api_base,
    string_ids,
    species,
    required_score,
    network_type,
    add_nodes,
    caller_identity,
):
    if len(string_ids) < 2:
        return b""

    url = f"{api_base}/svg/network"

    params = {
        "identifiers": "\r".join(string_ids),
        "species": species,
        "required_score": int(required_score),
        "network_type": network_type,
        "add_nodes": int(add_nodes),
        "network_flavor": "confidence",
        "show_query_node_labels": 1,
        "caller_identity": caller_identity,
    }

    content = http.request(
        "POST",
        url,
        params,
        binary=True,
    )

    return content


def string_ppi_enrichment(
    http,
    api_base,
    string_ids,
    species,
    required_score,
    background_ids,
    caller_identity,
):
    if len(string_ids) < 2:
        return {}

    url = f"{api_base}/json/ppi_enrichment"

    params = {
        "identifiers": "\r".join(string_ids),
        "species": species,
        "required_score": int(required_score),
        "caller_identity": caller_identity,
    }

    if background_ids:
        params["background_string_identifiers"] = "\r".join(
            background_ids
        )

    data = http.request(
        "POST",
        url,
        params,
        as_json=True,
    )

    if isinstance(data, list):
        return data[0] if data else {}

    return data or {}


# =============================================================================
# BIOGRID OPTIONAL
# =============================================================================

def fetch_biogrid_direct(
    http,
    access_key,
    symbols,
    species,
):
    if not access_key or len(symbols) < 2:
        return pd.DataFrame()

    symbols = dedupe_keep_order([
        clean_symbol(x)
        for x in symbols
        if clean_symbol(x)
    ])

    if len(symbols) < 2:
        return pd.DataFrame()

    url = "https://webservice.thebiogrid.org/interactions/"

    params = {
        "accesskey": access_key,
        "format": "json",
        "searchNames": "true",
        "searchSynonyms": "true",
        "geneList": "|".join(symbols),
        "includeInteractors": "false",
        "includeInteractorInteractions": "false",
        "interSpeciesExcluded": "true",
        "selfInteractionsExcluded": "true",
        "taxId": str(species),
        "max": 10000,
    }

    data = http.request(
        "GET",
        url,
        params,
        as_json=True,
    )

    if not data:
        return pd.DataFrame()

    if isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        return pd.DataFrame()

    x = pd.DataFrame(records)

    if x.empty:
        return x

    # Normalize the most useful fields while retaining everything.
    rename = {}

    for col in x.columns:
        low = col.lower().replace(" ", "_")

        if "official_symbol_a" in low:
            rename[col] = "SYMBOL_A"
        elif "official_symbol_b" in low:
            rename[col] = "SYMBOL_B"
        elif "experimental_system_type" in low:
            rename[col] = "EXPERIMENTAL_SYSTEM_TYPE"
        elif "experimental_system" in low:
            rename[col] = "EXPERIMENTAL_SYSTEM"
        elif "pubmed" in low:
            rename[col] = "PUBMED_ID"
        elif low in {"biogrid_interaction_id", "interaction_id"}:
            rename[col] = "BIOGRID_INTERACTION_ID"

    x = x.rename(columns=rename)

    return x


# =============================================================================
# NETWORK CONSTRUCTION / METRICS
# =============================================================================

def edges_to_graph(edges):
    G = nx.Graph()

    if edges is None or edges.empty:
        return G

    for row in edges.itertuples(index=False):
        a = str(getattr(row, "stringId_A", "")).strip()
        b = str(getattr(row, "stringId_B", "")).strip()

        if not a or not b or a == b:
            continue

        score = safe_float(
            getattr(row, "score", np.nan),
            default=0.0,
        )

        if G.has_edge(a, b):
            if score > G[a][b].get("score", 0.0):
                G[a][b]["score"] = score
        else:
            G.add_edge(
                a,
                b,
                score=score,
                distance=max(1e-6, 1.0 - score),
                preferredName_A=str(
                    getattr(row, "preferredName_A", "")
                ),
                preferredName_B=str(
                    getattr(row, "preferredName_B", "")
                ),
            )

    return G


def string_name_map(edges):
    names = {}

    if edges is None or edges.empty:
        return names

    for row in edges.itertuples(index=False):
        a = str(getattr(row, "stringId_A", "")).strip()
        b = str(getattr(row, "stringId_B", "")).strip()

        if a:
            names[a] = str(
                getattr(row, "preferredName_A", a)
            )

        if b:
            names[b] = str(
                getattr(row, "preferredName_B", b)
            )

    return names


def mapping_indexes(mapping):
    string_to_gene = {}
    gene_to_string = {}

    if mapping.empty:
        return string_to_gene, gene_to_string

    for row in mapping.itertuples(index=False):
        gid = clean_gene_id(
            getattr(row, "GENE_ID", "")
        )

        sid = str(
            getattr(row, "STRING_ID", "")
        ).strip()

        if gid and sid:
            string_to_gene[sid] = gid
            gene_to_string[gid] = sid

    return string_to_gene, gene_to_string


def network_node_table(
    G,
    edges,
    mapping,
    genes,
    gene_sets,
    known_gene_ids,
):
    lookup = gene_lookup_table(genes)
    string_to_gene, _ = mapping_indexes(mapping)
    names = string_name_map(edges)

    rows = []

    for node in G.nodes:
        gid = string_to_gene.get(node, "")
        meta = lookup.get(gid, {})

        if gid:
            node_class = classify_seed_gene(
                gid,
                gene_sets,
                known_gene_ids,
            )
            is_seed = "YES"
        else:
            node_class = "CONNECTOR"
            is_seed = "NO"

        rows.append({
            "STRING_ID": node,
            "PREFERRED_NAME": names.get(node, node),
            "GENE_ID": gid,
            "GENE_SYMBOL": meta.get("symbol", ""),
            "NODE_CLASS": node_class,
            "IS_INPUT_SEED": is_seed,
            "MAX_H4": meta.get("max_h4", np.nan),
            "PRIOR_ROBUST": meta.get("prior_robust", ""),
            "QTL_TYPES": meta.get("qtl_types", ""),
            "N_TISSUES": meta.get("n_tissues", np.nan),
            "OT_L2G_SUPPORTED_ANY": meta.get("ot_l2g", ""),
            "GENE_SET_MEMBERSHIP": (
                ";".join(gene_set_flags(gid, gene_sets))
                if gid
                else ""
            ),
        })

    return pd.DataFrame(rows)


def network_metrics(G, node_table):
    if G.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()

    degree = dict(G.degree())
    weighted_degree = {
        n: sum(
            data.get("score", 0.0)
            for _, _, data in G.edges(n, data=True)
        )
        for n in G.nodes
    }

    betweenness = nx.betweenness_centrality(
        G,
        normalized=True,
        weight="distance",
    )

    closeness = nx.closeness_centrality(G)

    pagerank = nx.pagerank(
        G,
        alpha=0.85,
        weight="score",
    )

    components = {}

    for comp_id, comp in enumerate(
        nx.connected_components(G),
        start=1,
    ):
        for node in comp:
            components[node] = comp_id

    communities = []

    if G.number_of_edges() > 0 and G.number_of_nodes() > 1:
        try:
            comms = list(
                nx.algorithms.community.greedy_modularity_communities(
                    G,
                    weight="score",
                )
            )
        except Exception:
            comms = [
                set(c)
                for c in nx.connected_components(G)
            ]
    else:
        comms = [
            {n}
            for n in G.nodes
        ]

    community_id = {}

    for cid, members in enumerate(comms, start=1):
        for node in members:
            community_id[node] = cid

        communities.append({
            "COMMUNITY_ID": cid,
            "N_NODES": len(members),
            "STRING_IDS": ";".join(sorted(members)),
        })

    base = node_table.set_index("STRING_ID", drop=False)

    rows = []

    for node in G.nodes:
        meta = (
            base.loc[node].to_dict()
            if node in base.index
            else {"STRING_ID": node}
        )

        rows.append({
            **meta,
            "DEGREE": degree.get(node, 0),
            "WEIGHTED_DEGREE": weighted_degree.get(node, 0.0),
            "BETWEENNESS": betweenness.get(node, 0.0),
            "CLOSENESS": closeness.get(node, 0.0),
            "PAGERANK": pagerank.get(node, 0.0),
            "COMPONENT_ID": components.get(node, np.nan),
            "COMMUNITY_ID": community_id.get(node, np.nan),
        })

    centrality = pd.DataFrame(rows).sort_values(
        [
            "BETWEENNESS",
            "PAGERANK",
            "DEGREE",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return centrality, pd.DataFrame(communities)


def seed_pair_distances(
    G,
    mapping,
    gene_sets,
    genes,
    known_gene_ids,
):
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    _, gene_to_string = mapping_indexes(mapping)
    lookup = gene_lookup_table(genes)

    seed_gene_ids = set()

    for set_name in [
        "CORE_STRONG",
        "OPEN_TARGETS_TOP1",
    ]:
        seed_gene_ids.update(
            gene_sets.get(set_name, [])
        )

    seed_gene_ids.update(
        known_gene_ids
    )

    seed_gene_ids = [
        gid
        for gid in seed_gene_ids
        if gid in gene_to_string
        and gene_to_string[gid] in G
    ]

    rows = []

    for i, ga in enumerate(seed_gene_ids):
        for gb in seed_gene_ids[i + 1:]:
            a = gene_to_string[ga]
            b = gene_to_string[gb]

            class_a = classify_seed_gene(
                ga,
                gene_sets,
                known_gene_ids,
            )

            class_b = classify_seed_gene(
                gb,
                gene_sets,
                known_gene_ids,
            )

            connected = nx.has_path(G, a, b)

            if connected:
                path = nx.shortest_path(
                    G,
                    a,
                    b,
                )

                hop_distance = len(path) - 1

                weighted_distance = nx.shortest_path_length(
                    G,
                    a,
                    b,
                    weight="distance",
                )

                path_labels = []

                string_to_gene, _ = mapping_indexes(mapping)

                for node in path:
                    gid = string_to_gene.get(node, "")

                    if gid:
                        path_labels.append(
                            display_gene(gid, lookup)
                        )
                    else:
                        path_labels.append(node)

            else:
                path = []
                hop_distance = np.nan
                weighted_distance = np.nan
                path_labels = []

            rows.append({
                "GENE_A_ID": ga,
                "GENE_A_SYMBOL": lookup.get(ga, {}).get("symbol", ""),
                "GENE_A_CLASS": class_a,
                "GENE_B_ID": gb,
                "GENE_B_SYMBOL": lookup.get(gb, {}).get("symbol", ""),
                "GENE_B_CLASS": class_b,
                "CONNECTED": "YES" if connected else "NO",
                "HOP_DISTANCE": hop_distance,
                "CONFIDENCE_WEIGHTED_DISTANCE": weighted_distance,
                "PATH_STRING_IDS": ";".join(path),
                "PATH_LABELS": " -> ".join(path_labels),
            })

    return pd.DataFrame(rows)


def connector_table(
    G,
    centrality,
    mapping,
    gene_sets,
    known_gene_ids,
):
    if G.number_of_nodes() == 0 or centrality.empty:
        return pd.DataFrame()

    _, gene_to_string = mapping_indexes(mapping)

    gwas_nodes = {
        gene_to_string[g]
        for g in gene_sets.get("CORE_STRONG", [])
        if g in gene_to_string
    }

    ot_nodes = {
        gene_to_string[g]
        for g in gene_sets.get("OPEN_TARGETS_TOP1", [])
        if g in gene_to_string
    }

    known_nodes = {
        gene_to_string[g]
        for g in known_gene_ids
        if g in gene_to_string
    }

    seed_nodes = gwas_nodes | ot_nodes | known_nodes

    rows = []

    centrality_idx = centrality.set_index(
        "STRING_ID",
        drop=False,
    )

    for node in G.nodes:
        if node in seed_nodes:
            continue

        neighbors = set(G.neighbors(node))

        gwas_touch = neighbors & gwas_nodes
        ot_touch = neighbors & ot_nodes
        known_touch = neighbors & known_nodes

        n_seed_neighbors = len(
            neighbors & seed_nodes
        )

        if n_seed_neighbors == 0:
            continue

        meta = (
            centrality_idx.loc[node].to_dict()
            if node in centrality_idx.index
            else {}
        )

        rows.append({
            **meta,
            "N_SEED_NEIGHBORS": n_seed_neighbors,
            "N_GWAS2M_STRONG_NEIGHBORS": len(gwas_touch),
            "N_OT_TOP1_NEIGHBORS": len(ot_touch),
            "N_KNOWN_GENE_NEIGHBORS": len(known_touch),
            "BRIDGES_GWAS2M_TO_OT": (
                "YES"
                if gwas_touch and ot_touch
                else "NO"
            ),
            "GWAS2M_NEIGHBOR_STRING_IDS": ";".join(sorted(gwas_touch)),
            "OT_NEIGHBOR_STRING_IDS": ";".join(sorted(ot_touch)),
            "KNOWN_NEIGHBOR_STRING_IDS": ";".join(sorted(known_touch)),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        [
            "BRIDGES_GWAS2M_TO_OT",
            "N_SEED_NEIGHBORS",
            "BETWEENNESS",
            "PAGERANK",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )


def ot_bridge_summary(
    G,
    mapping,
    genes,
    gene_sets,
):
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    lookup = gene_lookup_table(genes)
    _, gene_to_string = mapping_indexes(mapping)

    ot_supported = set(
        gene_sets.get(
            "CORE_STRONG_OT_SUPPORTED",
            [],
        )
    )

    ot_discordant = set(
        gene_sets.get(
            "CORE_STRONG_OT_DISCORDANT",
            [],
        )
    )

    rows = []

    for discordant in sorted(ot_discordant):
        dnode = gene_to_string.get(discordant)

        if not dnode or dnode not in G:
            rows.append({
                "GWAS2M_DISCORDANT_GENE_ID": discordant,
                "GWAS2M_DISCORDANT_GENE_SYMBOL": lookup.get(discordant, {}).get("symbol", ""),
                "OT_SUPPORTED_GENE_ID": "",
                "OT_SUPPORTED_GENE_SYMBOL": "",
                "CONNECTED": "NOT_MAPPABLE_OR_ABSENT",
                "HOP_DISTANCE": np.nan,
                "PATH": "",
            })
            continue

        for supported in sorted(ot_supported):
            snode = gene_to_string.get(supported)

            if not snode or snode not in G:
                continue

            if nx.has_path(G, dnode, snode):
                path = nx.shortest_path(
                    G,
                    dnode,
                    snode,
                )

                string_to_gene, _ = mapping_indexes(mapping)

                labels = []

                for node in path:
                    gid = string_to_gene.get(node, "")
                    labels.append(
                        display_gene(gid, lookup)
                        if gid
                        else node
                    )

                rows.append({
                    "GWAS2M_DISCORDANT_GENE_ID": discordant,
                    "GWAS2M_DISCORDANT_GENE_SYMBOL": lookup.get(discordant, {}).get("symbol", ""),
                    "OT_SUPPORTED_GENE_ID": supported,
                    "OT_SUPPORTED_GENE_SYMBOL": lookup.get(supported, {}).get("symbol", ""),
                    "CONNECTED": "YES",
                    "HOP_DISTANCE": len(path) - 1,
                    "PATH": " -> ".join(labels),
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        [
            "CONNECTED",
            "HOP_DISTANCE",
        ],
        ascending=[
            False,
            True,
        ],
    )


# =============================================================================
# EDGE ANNOTATION / CROSS-VALIDATION
# =============================================================================

def annotate_string_edges(
    edges,
    mapping,
    genes,
    gene_sets,
    known_gene_ids,
):
    if edges.empty:
        return edges

    lookup = gene_lookup_table(genes)
    string_to_gene, _ = mapping_indexes(mapping)

    rows = []

    for row in edges.to_dict("records"):
        a_sid = str(row.get("stringId_A", "")).strip()
        b_sid = str(row.get("stringId_B", "")).strip()

        ga = string_to_gene.get(a_sid, "")
        gb = string_to_gene.get(b_sid, "")

        rows.append({
            **row,
            "GENE_A_ID": ga,
            "GENE_A_SYMBOL": lookup.get(ga, {}).get("symbol", ""),
            "GENE_A_CLASS": (
                classify_seed_gene(
                    ga,
                    gene_sets,
                    known_gene_ids,
                )
                if ga
                else "CONNECTOR"
            ),
            "GENE_B_ID": gb,
            "GENE_B_SYMBOL": lookup.get(gb, {}).get("symbol", ""),
            "GENE_B_CLASS": (
                classify_seed_gene(
                    gb,
                    gene_sets,
                    known_gene_ids,
                )
                if gb
                else "CONNECTOR"
            ),
        })

    return pd.DataFrame(rows)


def string_biogrid_comparison(
    string_edges,
    biogrid,
):
    if string_edges.empty or biogrid.empty:
        return pd.DataFrame()

    if not {"SYMBOL_A", "SYMBOL_B"}.issubset(biogrid.columns):
        return pd.DataFrame()

    bg_keys = {
        edge_key(a, b)
        for a, b
        in zip(
            biogrid["SYMBOL_A"].astype(str),
            biogrid["SYMBOL_B"].astype(str),
        )
        if str(a).strip()
        and str(b).strip()
    }

    rows = []

    for row in string_edges.itertuples(index=False):
        a = str(
            getattr(row, "preferredName_A", "")
        ).strip()

        b = str(
            getattr(row, "preferredName_B", "")
        ).strip()

        if not a or not b:
            continue

        rows.append({
            "STRING_SYMBOL_A": a,
            "STRING_SYMBOL_B": b,
            "STRING_SCORE": safe_float(
                getattr(row, "score", np.nan)
            ),
            "BIOGRID_DIRECT_SUPPORT": (
                "YES"
                if edge_key(a, b) in bg_keys
                else "NO"
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# GRAPHML
# =============================================================================

def save_graphml(
    G,
    path,
    centrality,
):
    H = G.copy()

    if centrality is not None and not centrality.empty:
        meta = centrality.set_index("STRING_ID")

        for node in H.nodes:
            if node not in meta.index:
                continue

            row = meta.loc[node]

            for col in [
                "PREFERRED_NAME",
                "GENE_ID",
                "GENE_SYMBOL",
                "NODE_CLASS",
                "IS_INPUT_SEED",
                "MAX_H4",
                "PRIOR_ROBUST",
                "QTL_TYPES",
                "N_TISSUES",
                "OT_L2G_SUPPORTED_ANY",
                "DEGREE",
                "WEIGHTED_DEGREE",
                "BETWEENNESS",
                "CLOSENESS",
                "PAGERANK",
                "COMPONENT_ID",
                "COMMUNITY_ID",
            ]:
                if col not in row.index:
                    continue

                value = row[col]

                if pd.isna(value):
                    value = ""

                H.nodes[node][col] = (
                    float(value)
                    if isinstance(value, (np.floating, float))
                    else int(value)
                    if isinstance(value, (np.integer, int))
                    else str(value)
                )

    nx.write_graphml(
        H,
        path,
    )


# =============================================================================
# PRINT REPORT
# =============================================================================

def print_input_gene_sets(
    gene_sets,
    genes,
    known_gene_ids,
):
    lookup = gene_lookup_table(genes)

    section("STEP15 INPUT GENE SETS")

    for name in PRIMARY_SET_ORDER:
        if name not in gene_sets:
            continue

        members = gene_sets[name]

        print()
        print(f"{name}: {len(members)} gene(s)")
        print("-" * 110)

        for i, gid in enumerate(members, 1):
            meta = lookup.get(gid, {})

            print(
                f"  {i:>2}. {display_gene(gid, lookup)}"
                f"  H4={meta.get('max_h4', np.nan):.6g}"
                f"  OT_L2G={meta.get('ot_l2g', '')}"
                f"  QTL={meta.get('qtl_types', '')}"
            )

    if known_gene_ids:
        print()
        print(f"OPTIONAL KNOWN-DISEASE ANCHORS: {len(known_gene_ids)}")
        print("-" * 110)

        for i, gid in enumerate(known_gene_ids, 1):
            print(f"  {i:>2}. {display_gene(gid, lookup)}")


def print_mapping_report(
    mapping,
    genes,
    gene_sets,
):
    section("STRING IDENTIFIER MAPPING")

    core = set(gene_sets.get("CORE_STRONG", []))
    ot = set(gene_sets.get("OPEN_TARGETS_TOP1", []))

    x = mapping[
        mapping["GENE_ID"].isin(core | ot)
    ].copy()

    show(
        x,
        columns=[
            "GENE_ID",
            "GENE_SYMBOL",
            "MAPPING_METHOD",
            "STRING_ID",
            "STRING_PREFERRED_NAME",
            "MAPPED",
        ],
    )

    n_core = len(core)
    n_core_mapped = int(
        x[
            x["GENE_ID"].isin(core)
            &
            (x["MAPPED"] == "YES")
        ]["GENE_ID"].nunique()
    )

    print()
    print(
        f"CORE_STRONG mapped to STRING: {n_core_mapped}/{n_core}"
    )

    print(
        "Genes that do not map are commonly non-protein-coding or "
        "lack a resolvable STRING protein identifier."
    )


def print_network_summary(
    label,
    G,
    edges,
    centrality,
):
    section(f"NETWORK SUMMARY - {label}")

    print(f"Nodes                : {G.number_of_nodes()}")
    print(f"Edges                : {G.number_of_edges()}")
    print(
        f"Connected components : "
        f"{nx.number_connected_components(G) if G.number_of_nodes() else 0}"
    )
    print(
        f"Density              : "
        f"{nx.density(G):.6f}" if G.number_of_nodes() > 1 else "Density              : NA"
    )

    if G.number_of_nodes():
        isolates = list(nx.isolates(G))
        print(f"Isolated nodes       : {len(isolates)}")

    if not edges.empty and "score" in edges.columns:
        print(
            f"Median STRING score  : "
            f"{safe_numeric(edges['score']).median():.4f}"
        )

    if not centrality.empty:
        show(
            centrality.head(20),
            title="TOP NETWORK NODES BY BETWEENNESS / PAGERANK",
            columns=[
                "PREFERRED_NAME",
                "GENE_SYMBOL",
                "GENE_ID",
                "NODE_CLASS",
                "MAX_H4",
                "DEGREE",
                "WEIGHTED_DEGREE",
                "BETWEENNESS",
                "PAGERANK",
                "COMPONENT_ID",
                "COMMUNITY_ID",
            ],
        )


def print_ppi_enrichment(ppi):
    section("STRING PPI ENRICHMENT")

    if ppi.empty:
        print("(no PPI-enrichment results)")
        return

    show(
        ppi,
        columns=[
            "GENE_SET",
            "N_INPUT_GENES",
            "N_MAPPED_PROTEINS",
            "number_of_nodes",
            "number_of_edges",
            "expected_number_of_edges",
            "average_node_degree",
            "local_clustering_coefficient",
            "p_value",
            "INTERPRETATION",
        ],
    )


def print_connectors(connectors):
    section("FIRST-ORDER CONNECTOR PROTEINS")

    if connectors.empty:
        print(
            "(no added connector node touches the input seeds under the selected settings)"
        )
        return

    show(
        connectors.head(50),
        columns=[
            "PREFERRED_NAME",
            "STRING_ID",
            "NODE_CLASS",
            "N_SEED_NEIGHBORS",
            "N_GWAS2M_STRONG_NEIGHBORS",
            "N_OT_TOP1_NEIGHBORS",
            "BRIDGES_GWAS2M_TO_OT",
            "DEGREE",
            "BETWEENNESS",
            "PAGERANK",
        ],
    )

    print()
    print(
        "CAUTION: connector proteins are network hypotheses only. "
        "They are not genetically prioritized by GWAS2m unless they are also seed genes."
    )


def print_ot_bridge_table(bridges):
    section(
        "DO OT-SUPPORTED STRONG GENES CONNECT TO OT-DISCORDANT HIGH-H4 GENES?"
    )

    if bridges.empty:
        print("(no evaluable bridge rows)")
        return

    show(
        bridges,
        max_rows=100,
        columns=[
            "GWAS2M_DISCORDANT_GENE_SYMBOL",
            "GWAS2M_DISCORDANT_GENE_ID",
            "OT_SUPPORTED_GENE_SYMBOL",
            "OT_SUPPORTED_GENE_ID",
            "CONNECTED",
            "HOP_DISTANCE",
            "PATH",
        ],
    )


def print_plain_english(
    genes,
    gene_sets,
    mapping,
    direct_physical_G,
    expanded_physical_G,
    ppi,
    connectors,
    bridges,
):
    section("PLAIN-ENGLISH STEP15 INTERPRETATION")

    core = gene_sets.get("CORE_STRONG", [])
    ot_supported = gene_sets.get(
        "CORE_STRONG_OT_SUPPORTED",
        [],
    )
    ot_discordant = gene_sets.get(
        "CORE_STRONG_OT_DISCORDANT",
        [],
    )

    mapped_core = mapping[
        mapping["GENE_ID"].isin(core)
        &
        (mapping["MAPPED"] == "YES")
    ]["GENE_ID"].nunique()

    print(
        f"1. Step15 started from {len(core)} CORE_STRONG GWAS2m genes."
    )
    print(
        f"2. {mapped_core}/{len(core)} CORE_STRONG genes map to STRING proteins."
    )
    print(
        f"3. {len(ot_supported)} CORE_STRONG genes are also supported by OT L2G."
    )
    print(
        f"4. {len(ot_discordant)} CORE_STRONG genes are at OT-matched loci but are not the same OT L2G effector."
    )
    print(
        f"5. Direct high-confidence physical network: "
        f"{direct_physical_G.number_of_nodes()} nodes / "
        f"{direct_physical_G.number_of_edges()} edges."
    )
    print(
        f"6. Expanded physical network: "
        f"{expanded_physical_G.number_of_nodes()} nodes / "
        f"{expanded_physical_G.number_of_edges()} edges."
    )

    if not ppi.empty:
        core_ppi = ppi[
            ppi["GENE_SET"] == "CORE_STRONG"
        ]

        if not core_ppi.empty:
            row = core_ppi.iloc[0]
            pv = safe_float(row.get("p_value", np.nan))
            obs = row.get("number_of_edges", np.nan)
            exp = row.get("expected_number_of_edges", np.nan)

            print(
                f"7. STRING PPI enrichment for CORE_STRONG: "
                f"observed edges={obs}, expected={exp}, p={pv:.4g}."
            )

            if pd.notna(pv):
                if pv < 0.05:
                    print(
                        "   -> The mapped strong proteins interact more than expected under the STRING test."
                    )
                else:
                    print(
                        "   -> No significant excess connectivity was detected at this threshold."
                    )

    if not connectors.empty:
        bridges_n = int(
            (connectors["BRIDGES_GWAS2M_TO_OT"] == "YES").sum()
        )

        print(
            f"8. {len(connectors)} added first-order connector proteins touch at least one seed; "
            f"{bridges_n} touch both a GWAS2m strong node and an OT top node."
        )

    if not bridges.empty:
        yes = bridges[
            bridges["CONNECTED"] == "YES"
        ]

        if not yes.empty:
            n_disc = yes[
                "GWAS2M_DISCORDANT_GENE_ID"
            ].nunique()

            print(
                f"9. {n_disc} OT-discordant high-H4 genes have a physical-network path "
                f"to at least one OT-supported strong gene in the expanded network."
            )

    print()
    print(
        "Interpretation rule: a short physical-network path strengthens mechanistic coherence, "
        "but it does not convert a discordant gene into a validated causal gene."
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
        "--species",
        type=int,
        default=9606,
        help="NCBI taxonomy ID; 9606 = human.",
    )

    parser.add_argument(
        "--required-score",
        type=int,
        default=700,
        help="STRING combined-score threshold (0-1000). 700 = high confidence.",
    )

    parser.add_argument(
        "--add-nodes",
        type=int,
        default=20,
        help="Number of connector proteins added to expanded STRING networks.",
    )

    parser.add_argument(
        "--network-types",
        default="physical,functional",
        help="Comma-separated STRING network types.",
    )

    parser.add_argument(
        "--known-genes-file",
        default="",
        help="Optional disease-gene anchor file.",
    )

    parser.add_argument(
        "--biogrid-access-key",
        default="",
        help="Optional BioGRID REST key. Can also use BIOGRID_ACCESS_KEY env variable.",
    )

    parser.add_argument(
        "--step14-dir",
        default="",
        help="Override Step14 output directory.",
    )

    parser.add_argument(
        "--output-dir",
        default="",
        help="Override Step15 output directory.",
    )

    parser.add_argument(
        "--string-api-base",
        default="",
        help="Override STRING API base; normally auto-resolved to the current stable version.",
    )

    parser.add_argument(
        "--caller-identity",
        default="GWAS2m-Step15",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore API cache.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load/print Step14 inputs but do not call STRING or BioGRID.",
    )

    parser.add_argument(
        "--compact-screen",
        action="store_true",
        help="Print fewer table rows.",
    )

    args = parser.parse_args()

    if not (0 <= args.required_score <= 1000):
        raise SystemExit("--required-score must be between 0 and 1000.")

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

    step14_dir = (
        Path(args.step14_dir).resolve()
        if args.step14_dir
        else
        root
        /
        "14_pathways"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else
        root
        /
        "15_network"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking_path = (
        step14_dir
        /
        "Step14_Gene_Ranking.tsv"
    )

    membership_path = (
        step14_dir
        /
        "Step14_Gene_Set_Membership.tsv"
    )

    pathway_path = (
        step14_dir
        /
        "Step14_Pathway_Significant.tsv"
    )

    section("GWAS2m STEP15 v1.0 - PPI / NETWORK ANALYSIS")

    print(f"Root                   : {root}")
    print(f"Phenotype              : {args.phenotype}")
    print(f"Ancestry               : {ancestry_label} ({ancestry_code})")
    print(f"Step14                 : {step14_dir}")
    print(f"Output                 : {output}")
    print(f"Species                : {args.species}")
    print(f"STRING required score  : >= {args.required_score}/1000")
    print(f"Expanded add_nodes     : {args.add_nodes}")
    print(f"Network types          : {args.network_types}")
    print(f"Dry run                : {args.dry_run}")

    genes = load_step14_gene_ranking(
        ranking_path
    )

    membership = load_step14_membership(
        membership_path
    )

    pathways = load_pathways(
        pathway_path
    )

    gene_sets = load_gene_sets(
        membership
    )

    known_raw = load_known_genes(
        args.known_genes_file
    )

    # Map optional known symbols/IDs to Step14 Ensembl IDs where possible.
    lookup = gene_lookup_table(genes)
    symbol_to_gid = {
        clean_symbol(meta.get("symbol", "")): gid
        for gid, meta in lookup.items()
        if clean_symbol(meta.get("symbol", ""))
    }

    known_gene_ids = []

    for value in known_raw:
        cleaned = clean_gene_id(value)

        if cleaned in lookup:
            known_gene_ids.append(cleaned)
        elif str(value).strip() in symbol_to_gid:
            known_gene_ids.append(
                symbol_to_gid[str(value).strip()]
            )
        else:
            # Keep external symbol as literal anchor. It will be added separately
            # to STRING mapping below if not a Step14 Ensembl gene.
            known_gene_ids.append(str(value).strip())

    known_gene_ids = dedupe_keep_order(
        known_gene_ids
    )

    print_input_gene_sets(
        gene_sets,
        genes,
        known_gene_ids,
    )

    input_rows = []

    for set_name, members in gene_sets.items():
        for gid in members:
            input_rows.append({
                "GENE_SET": set_name,
                "GENE_ID": gid,
                "GENE_SYMBOL": lookup.get(gid, {}).get("symbol", ""),
                "MAX_H4": lookup.get(gid, {}).get("max_h4", np.nan),
                "OT_L2G_SUPPORTED_ANY": lookup.get(gid, {}).get("ot_l2g", ""),
            })

    pd.DataFrame(input_rows).to_csv(
        output / "Step15_Input_Gene_Sets.tsv",
        sep="\t",
        index=False,
    )

    if args.dry_run:
        section("DRY RUN COMPLETE")
        print(
            "Inputs loaded successfully. No STRING/BioGRID calls were made."
        )
        return

    cache_dir = (
        root
        /
        "resources"
        /
        "network_cache"
    )

    http = CachedHTTP(
        cache_dir,
    )

    api_base, string_version, stable_address = resolve_string_api(
        http,
        args.string_api_base,
    )

    section("STRING VERSION")

    print(f"STRING version          : {string_version}")
    print(f"STRING stable address   : {stable_address}")
    print(f"STRING API base         : {api_base}")

    # Map all Step14 genes, which also gives a useful tested-gene background.
    mapping = build_string_mapping(
        http,
        api_base,
        genes,
        args.species,
        args.caller_identity,
    )

    mapping.to_csv(
        output / "Step15_STRING_Mapping.tsv",
        sep="\t",
        index=False,
    )

    unmapped = mapping[
        mapping["MAPPED"] != "YES"
    ].copy()

    unmapped.to_csv(
        output / "Step15_STRING_Unmapped_Genes.tsv",
        sep="\t",
        index=False,
    )

    print_mapping_report(
        mapping,
        genes,
        gene_sets,
    )

    # Combined seeds for network construction.
    combined_gene_ids = dedupe_keep_order(
        gene_sets.get("CORE_STRONG", [])
        +
        gene_sets.get("OPEN_TARGETS_TOP1", [])
        +
        [
            g
            for g in known_gene_ids
            if g in lookup
        ]
    )

    combined_string_ids = string_ids_for_genes(
        combined_gene_ids,
        mapping,
    )

    background_string_ids = string_ids_for_genes(
        list(lookup),
        mapping,
    )

    # -------------------------------------------------------------------------
    # PPI enrichment
    # -------------------------------------------------------------------------

    ppi_rows = []

    ppi_sets = [
        "CORE_STRONG",
        "CORE_STRONG_PRIOR_ROBUST",
        "LOCUS_TOP_STRONG",
        "CORE_STRONG_OT_SUPPORTED",
        "CORE_STRONG_OT_DISCORDANT",
        "OPEN_TARGETS_TOP1",
    ]

    for set_name in ppi_sets:
        if set_name not in gene_sets:
            continue

        gene_ids = gene_sets[set_name]

        string_ids = string_ids_for_genes(
            gene_ids,
            mapping,
        )

        if len(string_ids) < 2:
            continue

        try:
            row = string_ppi_enrichment(
                http,
                api_base,
                string_ids,
                args.species,
                args.required_score,
                background_string_ids,
                args.caller_identity,
            )

            pvalue = safe_float(
                row.get("p_value", np.nan)
            )

            ppi_rows.append({
                "GENE_SET": set_name,
                "N_INPUT_GENES": len(gene_ids),
                "N_MAPPED_PROTEINS": len(string_ids),
                **row,
                "INTERPRETATION": (
                    "EXCESS_CONNECTIVITY"
                    if pd.notna(pvalue) and pvalue < 0.05
                    else "NO_SIGNIFICANT_EXCESS_CONNECTIVITY"
                    if pd.notna(pvalue)
                    else "UNKNOWN"
                ),
            })

        except Exception as exc:
            ppi_rows.append({
                "GENE_SET": set_name,
                "N_INPUT_GENES": len(gene_ids),
                "N_MAPPED_PROTEINS": len(string_ids),
                "ERROR": f"{type(exc).__name__}: {exc}",
                "INTERPRETATION": "FAILED",
            })

    ppi = pd.DataFrame(ppi_rows)

    ppi.to_csv(
        output / "Step15_STRING_PPI_Enrichment.tsv",
        sep="\t",
        index=False,
    )

    print_ppi_enrichment(
        ppi
    )

    # -------------------------------------------------------------------------
    # STRING direct and expanded networks
    # -------------------------------------------------------------------------

    network_types = [
        x.strip()
        for x in args.network_types.split(",")
        if x.strip()
    ]

    allowed = {"physical", "functional"}

    bad = [
        x
        for x in network_types
        if x not in allowed
    ]

    if bad:
        raise SystemExit(
            f"Unsupported network type(s): {bad}. "
            "Use physical,functional."
        )

    network_results = {}

    for network_type in network_types:
        direct = string_network(
            http,
            api_base,
            combined_string_ids,
            args.species,
            args.required_score,
            network_type,
            0,
            args.caller_identity,
        )

        direct = annotate_string_edges(
            direct,
            mapping,
            genes,
            gene_sets,
            known_gene_ids,
        )

        expanded = string_network(
            http,
            api_base,
            combined_string_ids,
            args.species,
            args.required_score,
            network_type,
            args.add_nodes,
            args.caller_identity,
        )

        expanded = annotate_string_edges(
            expanded,
            mapping,
            genes,
            gene_sets,
            known_gene_ids,
        )

        network_results[
            (network_type, "direct")
        ] = direct

        network_results[
            (network_type, "expanded")
        ] = expanded

        direct.to_csv(
            output
            /
            f"Step15_Direct_{network_type.capitalize()}_Edges.tsv",
            sep="\t",
            index=False,
        )

        expanded.to_csv(
            output
            /
            f"Step15_Expanded_{network_type.capitalize()}_Edges.tsv",
            sep="\t",
            index=False,
        )

    direct_physical = network_results.get(
        ("physical", "direct"),
        pd.DataFrame(),
    )

    expanded_physical = network_results.get(
        ("physical", "expanded"),
        pd.DataFrame(),
    )

    direct_functional = network_results.get(
        ("functional", "direct"),
        pd.DataFrame(),
    )

    expanded_functional = network_results.get(
        ("functional", "expanded"),
        pd.DataFrame(),
    )

    direct_physical_G = edges_to_graph(
        direct_physical
    )

    expanded_physical_G = edges_to_graph(
        expanded_physical
    )

    direct_functional_G = edges_to_graph(
        direct_functional
    )

    expanded_functional_G = edges_to_graph(
        expanded_functional
    )

    # Main metrics are on expanded physical network because it can reveal
    # bridges between seed genes.
    main_edges = (
        expanded_physical
        if not expanded_physical.empty
        else
        direct_physical
    )

    main_G = (
        expanded_physical_G
        if expanded_physical_G.number_of_nodes()
        else
        direct_physical_G
    )

    nodes = network_node_table(
        main_G,
        main_edges,
        mapping,
        genes,
        gene_sets,
        known_gene_ids,
    )

    centrality, communities = network_metrics(
        main_G,
        nodes,
    )

    nodes.to_csv(
        output / "Step15_Network_Nodes.tsv",
        sep="\t",
        index=False,
    )

    centrality.to_csv(
        output / "Step15_Network_Centrality.tsv",
        sep="\t",
        index=False,
    )

    communities.to_csv(
        output / "Step15_Network_Communities.tsv",
        sep="\t",
        index=False,
    )

    distances = seed_pair_distances(
        main_G,
        mapping,
        gene_sets,
        genes,
        known_gene_ids,
    )

    distances.to_csv(
        output / "Step15_Seed_Pair_Distances.tsv",
        sep="\t",
        index=False,
    )

    connectors = connector_table(
        main_G,
        centrality,
        mapping,
        gene_sets,
        known_gene_ids,
    )

    connectors.to_csv(
        output / "Step15_Connector_Genes.tsv",
        sep="\t",
        index=False,
    )

    bridges = ot_bridge_summary(
        main_G,
        mapping,
        genes,
        gene_sets,
    )

    bridges.to_csv(
        output / "Step15_OT_Bridge_Summary.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # SVG + GraphML
    # -------------------------------------------------------------------------

    if direct_physical_G.number_of_nodes():
        try:
            save_graphml(
                direct_physical_G,
                output / "Step15_Direct_Physical.graphml",
                network_metrics(
                    direct_physical_G,
                    network_node_table(
                        direct_physical_G,
                        direct_physical,
                        mapping,
                        genes,
                        gene_sets,
                        known_gene_ids,
                    ),
                )[0],
            )
        except Exception as exc:
            print(
                f"[WARN] Could not write direct GraphML: {exc}"
            )

    if expanded_physical_G.number_of_nodes():
        try:
            save_graphml(
                expanded_physical_G,
                output / "Step15_Expanded_Physical.graphml",
                centrality,
            )
        except Exception as exc:
            print(
                f"[WARN] Could not write expanded GraphML: {exc}"
            )

    if combined_string_ids:
        try:
            svg = string_network_svg(
                http,
                api_base,
                combined_string_ids,
                args.species,
                args.required_score,
                "physical",
                0,
                args.caller_identity,
            )

            if svg:
                (
                    output
                    /
                    "Step15_STRING_Direct_Physical.svg"
                ).write_bytes(svg)

            svg = string_network_svg(
                http,
                api_base,
                combined_string_ids,
                args.species,
                args.required_score,
                "physical",
                args.add_nodes,
                args.caller_identity,
            )

            if svg:
                (
                    output
                    /
                    "Step15_STRING_Expanded_Physical.svg"
                ).write_bytes(svg)

        except Exception as exc:
            print(
                f"[WARN] Could not save STRING SVG: {exc}"
            )

    # -------------------------------------------------------------------------
    # Optional BioGRID
    # -------------------------------------------------------------------------

    biogrid_key = (
        args.biogrid_access_key.strip()
        or
        os.environ.get(
            "BIOGRID_ACCESS_KEY",
            "",
        ).strip()
    )

    biogrid = pd.DataFrame()
    cross = pd.DataFrame()

    if biogrid_key:
        seed_symbols = []

        for gid in combined_gene_ids:
            symbol = clean_symbol(
                lookup.get(gid, {}).get("symbol", "")
            )

            if symbol:
                seed_symbols.append(symbol)

        try:
            biogrid = fetch_biogrid_direct(
                http,
                biogrid_key,
                seed_symbols,
                args.species,
            )

            biogrid.to_csv(
                output / "Step15_BioGRID_Interactions.tsv",
                sep="\t",
                index=False,
            )

            cross = string_biogrid_comparison(
                direct_physical,
                biogrid,
            )

            cross.to_csv(
                output / "Step15_STRING_BioGRID_Edge_Comparison.tsv",
                sep="\t",
                index=False,
            )

        except Exception as exc:
            print(
                f"[WARN] BioGRID query failed: {type(exc).__name__}: {exc}"
            )
    else:
        pd.DataFrame().to_csv(
            output / "Step15_BioGRID_Interactions.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame().to_csv(
            output / "Step15_STRING_BioGRID_Edge_Comparison.tsv",
            sep="\t",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Screen report
    # -------------------------------------------------------------------------

    print_network_summary(
        "DIRECT PHYSICAL",
        direct_physical_G,
        direct_physical,
        network_metrics(
            direct_physical_G,
            network_node_table(
                direct_physical_G,
                direct_physical,
                mapping,
                genes,
                gene_sets,
                known_gene_ids,
            ),
        )[0]
        if direct_physical_G.number_of_nodes()
        else pd.DataFrame(),
    )

    print_network_summary(
        "EXPANDED PHYSICAL",
        expanded_physical_G,
        expanded_physical,
        centrality,
    )

    if direct_functional_G.number_of_nodes():
        print_network_summary(
            "DIRECT FUNCTIONAL",
            direct_functional_G,
            direct_functional,
            network_metrics(
                direct_functional_G,
                network_node_table(
                    direct_functional_G,
                    direct_functional,
                    mapping,
                    genes,
                    gene_sets,
                    known_gene_ids,
                ),
            )[0],
        )

    print_connectors(
        connectors
    )

    print_ot_bridge_table(
        bridges
    )

    if not distances.empty:
        connected_pairs = distances[
            distances["CONNECTED"] == "YES"
        ].sort_values(
            [
                "HOP_DISTANCE",
                "CONFIDENCE_WEIGHTED_DISTANCE",
            ]
        )

        show(
            connected_pairs,
            title="SEED-TO-SEED SHORTEST NETWORK PATHS",
            max_rows=(
                50
                if args.compact_screen
                else
                150
            ),
            columns=[
                "GENE_A_SYMBOL",
                "GENE_A_ID",
                "GENE_A_CLASS",
                "GENE_B_SYMBOL",
                "GENE_B_ID",
                "GENE_B_CLASS",
                "HOP_DISTANCE",
                "PATH_LABELS",
            ],
        )

    if biogrid_key:
        section("BIOGRID CROSS-VALIDATION")

        print(
            f"BioGRID direct interactions returned: {len(biogrid):,}"
        )

        if not cross.empty:
            supported = int(
                (cross["BIOGRID_DIRECT_SUPPORT"] == "YES").sum()
            )

            print(
                f"STRING direct physical edges also present in BioGRID: "
                f"{supported}/{len(cross)}"
            )

            show(
                cross,
                max_rows=100,
            )

    print_plain_english(
        genes,
        gene_sets,
        mapping,
        direct_physical_G,
        expanded_physical_G,
        ppi,
        connectors,
        bridges,
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc_rows = [
        {
            "CHECK": "CORE_STRONG_GENES",
            "STATUS": "INFO",
            "VALUE": len(gene_sets.get("CORE_STRONG", [])),
            "DETAIL": "Primary GWAS2m Step15 seed genes.",
        },
        {
            "CHECK": "OT_TOP1_GENES",
            "STATUS": "INFO",
            "VALUE": len(gene_sets.get("OPEN_TARGETS_TOP1", [])),
            "DETAIL": "External Open Targets anchors.",
        },
        {
            "CHECK": "STRING_MAPPED_STEP14_GENES",
            "STATUS": "INFO",
            "VALUE": int((mapping["MAPPED"] == "YES").sum()),
            "DETAIL": f"Out of {len(mapping)} Step14 genes.",
        },
        {
            "CHECK": "STRING_UNMAPPED_STEP14_GENES",
            "STATUS": (
                "WARN"
                if len(unmapped)
                else "OK"
            ),
            "VALUE": len(unmapped),
            "DETAIL": (
                "Expected for some non-protein-coding genes; "
                "do not treat as biological negative evidence."
            ),
        },
        {
            "CHECK": "DIRECT_PHYSICAL_EDGES",
            "STATUS": "INFO",
            "VALUE": direct_physical_G.number_of_edges(),
            "DETAIL": f"STRING score >= {args.required_score}.",
        },
        {
            "CHECK": "EXPANDED_PHYSICAL_EDGES",
            "STATUS": "INFO",
            "VALUE": expanded_physical_G.number_of_edges(),
            "DETAIL": f"Includes up to {args.add_nodes} added connector proteins.",
        },
        {
            "CHECK": "CONNECTOR_GENES",
            "STATUS": "INFO",
            "VALUE": len(connectors),
            "DETAIL": "Network hypotheses only; not GWAS2m genetic evidence.",
        },
        {
            "CHECK": "BIOGRID_CROSSVALIDATION",
            "STATUS": (
                "OK"
                if biogrid_key
                else "NOT_RUN"
            ),
            "VALUE": len(biogrid),
            "DETAIL": (
                "BioGRID key supplied."
                if biogrid_key
                else "Optional. Supply --biogrid-access-key or BIOGRID_ACCESS_KEY."
            ),
        },
    ]

    qc = pd.DataFrame(qc_rows)

    qc.to_csv(
        output / "Step15_QC_Audit.tsv",
        sep="\t",
        index=False,
    )

    show(
        qc,
        title="STEP15 QC AUDIT",
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "version": VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phenotype": args.phenotype,
        "ancestry_code": ancestry_code,
        "ancestry_label": ancestry_label,
        "string_version": string_version,
        "string_stable_address": stable_address,
        "required_score": args.required_score,
        "add_nodes": args.add_nodes,
        "n_core_strong_genes": len(
            gene_sets.get("CORE_STRONG", [])
        ),
        "n_ot_top1_genes": len(
            gene_sets.get("OPEN_TARGETS_TOP1", [])
        ),
        "n_ot_supported_strong_genes": len(
            gene_sets.get("CORE_STRONG_OT_SUPPORTED", [])
        ),
        "n_ot_discordant_strong_genes": len(
            gene_sets.get("CORE_STRONG_OT_DISCORDANT", [])
        ),
        "n_step14_genes_mapped_to_string": int(
            (mapping["MAPPED"] == "YES").sum()
        ),
        "n_step14_genes_unmapped_to_string": len(
            unmapped
        ),
        "direct_physical": {
            "nodes": direct_physical_G.number_of_nodes(),
            "edges": direct_physical_G.number_of_edges(),
        },
        "expanded_physical": {
            "nodes": expanded_physical_G.number_of_nodes(),
            "edges": expanded_physical_G.number_of_edges(),
        },
        "direct_functional": {
            "nodes": direct_functional_G.number_of_nodes(),
            "edges": direct_functional_G.number_of_edges(),
        },
        "expanded_functional": {
            "nodes": expanded_functional_G.number_of_nodes(),
            "edges": expanded_functional_G.number_of_edges(),
        },
        "n_connector_genes": len(
            connectors
        ),
        "biogrid_run": bool(
            biogrid_key
        ),
        "biogrid_interactions": len(
            biogrid
        ),
        "scientific_notes": [
            "STRING physical edges are preferred for PPI interpretation.",
            "STRING functional edges are supportive functional associations, not necessarily direct binding.",
            "Connector proteins added by STRING are network hypotheses and are not genetically prioritized.",
            "Non-mapping genes are not biological negatives, especially non-protein-coding genes.",
            "Network centrality is descriptive and may reflect generic hub/study bias.",
        ],
    }

    (
        output
        /
        "Step15_summary.json"
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

    section("STEP15 COMPLETE")

    print(f"Output directory:")
    print(f"  {output}")
    print()

    print("Key files:")
    key_files = [
        "Step15_STRING_Mapping.tsv",
        "Step15_STRING_PPI_Enrichment.tsv",
        "Step15_Direct_Physical_Edges.tsv",
        "Step15_Expanded_Physical_Edges.tsv",
        "Step15_Network_Centrality.tsv",
        "Step15_Network_Communities.tsv",
        "Step15_Connector_Genes.tsv",
        "Step15_OT_Bridge_Summary.tsv",
        "Step15_Seed_Pair_Distances.tsv",
        "Step15_Direct_Physical.graphml",
        "Step15_Expanded_Physical.graphml",
        "Step15_STRING_Direct_Physical.svg",
        "Step15_STRING_Expanded_Physical.svg",
        "Step15_QC_Audit.tsv",
        "Step15_summary.json",
    ]

    for name in key_files:
        print(f"  {output / name}")

    print()
    print(
        "NEXT STEP AFTER REVIEWING THIS NETWORK:"
    )
    print(
        "  Step16 = drug-target / tractability / direction-of-effect analysis."
    )


if __name__ == "__main__":
    main()

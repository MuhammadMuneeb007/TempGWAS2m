#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GWAS2m STEP16 v1.0
DRUG-TARGET + TRACTABILITY + CLINICAL PRECEDENCE + DIRECTION-OF-EFFECT
================================================================================

PURPOSE
-------
Step16 converts the genetically prioritised / network-contextualised genes from
Steps 14-15 into a therapeutic target table.

For each target, Step16 asks:

1. How strong is the GWAS2m genetic/molecular evidence?
2. Is the target independently supported by Open Targets L2G?
3. Does the target sit in the Step15 physical interaction network?
4. Is the target tractable by:
       - small molecule
       - antibody
       - PROTAC
       - other clinical modality
   according to Open Targets?
5. Does ChEMBL contain clinical/drug mechanisms against the target?
6. What is the maximum clinical phase reached by those compounds?
7. Are any compounds already reported for the phenotype of interest?
8. If signed and allele-harmonised molecular-QTL/GWAS effects exist:
       - is increased molecular abundance associated with higher/lower risk?
       - would inhibition or activation be directionally consistent?
       - which ChEMBL mechanisms align with that genetic direction?

IMPORTANT SAFETY / SCIENTIFIC RULES
-----------------------------------
* H4 / colocalisation alone DOES NOT determine whether a target should be
  inhibited or activated.
* Therapeutic direction is inferred ONLY from signed GWAS + molecular-QTL
  effects when allele harmonisation is explicitly available, unless the user
  deliberately supplies --assume-harmonized.
* eQTL and pQTL can support gene/protein abundance direction.
* sQTL / isoQTL / exonQTL are event-specific and are NOT automatically reduced
  to "increase/decrease gene activity".
* A drug acting on a target for another disease is clinical precedence, NOT
  evidence that it treats the requested phenotype.
* Network connector proteins from Step15 are NOT promoted to genetically
  supported targets unless they also exist in Step14 evidence.
* The final priority score is a TRANSPARENT HEURISTIC, NOT a probability of
  efficacy or causal truth.

PUBLIC DATA SOURCES
-------------------
Open Targets Platform GraphQL:
    https://api.platform.opentargets.org/api/v4/graphql

ChEMBL REST API:
    https://www.ebi.ac.uk/chembl/api/data/

INPUTS
------
14_pathways/<phenotype>/<ancestry>/
    Step14_Gene_Ranking.tsv
    Step14_Gene_Set_Membership.tsv

15_network/<phenotype>/<ancestry>/
    Step15_Network_Centrality.tsv
    Step15_Direct_Physical_Edges.tsv
    Step15_OT_Bridge_Summary.tsv

Optional for direction:
12_master_table/<phenotype>/<ancestry>/
    Step12_Molecular_Evidence_Long.tsv

OUTPUTS
-------
16_drug_targets/<phenotype>/<ancestry>/

    Step16_Target_Prioritisation.tsv
    Step16_OpenTargets_Tractability.tsv
    Step16_ChEMBL_Target_Mapping.tsv
    Step16_Drug_Mechanisms.tsv
    Step16_Drug_Indications.tsv
    Step16_Direction_Evidence.tsv
    Step16_Direction_Summary.tsv
    Step16_Directionally_Aligned_Drugs.tsv
    Step16_Phenotype_Matched_Drugs.tsv
    Step16_Top_Therapeutic_Candidates.tsv
    Step16_QC_Audit.tsv
    Step16_summary.json

INSTALL
-------
python -m pip install pandas numpy requests

RUN
---
python Step16_Drug_Target_Analysis.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR

Strict high-confidence targets only:
python Step16_Drug_Target_Analysis.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR \
  --target-set CORE_STRONG_PRIOR_ROBUST

Dry run:
python Step16_Drug_Target_Analysis.py \
  --phenotype "parkinson's disease" \
  --ancestry EUR \
  --dry-run
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


VERSION = "1.0.0"

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

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

INHIBITORY_ACTION_WORDS = {
    "inhibitor",
    "antagonist",
    "blocker",
    "negative allosteric modulator",
    "inverse agonist",
    "degrader",
    "inhibition",
    "suppressor",
}

ACTIVATING_ACTION_WORDS = {
    "agonist",
    "activator",
    "positive allosteric modulator",
    "stimulator",
    "activation",
}

DIRECTION_ELIGIBLE_QTLS = {
    "eqtl",
    "pqtl",
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
        "display.max_colwidth", 150,
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

    match = re.search(r"(ENSG\d+)", x, flags=re.I)

    return match.group(1).upper() if match else x


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

        match = lower.get(
            str(candidate).lower()
        )

        if match is not None:
            return match

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
            "nan", "none", "null", "na"
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


def phase_number(value):
    value = safe_float(value, default=np.nan)

    if pd.isna(value):
        return np.nan

    return max(0.0, min(4.0, value))


def phenotype_match(phenotype, text):
    p = norm(phenotype)
    t = norm(text)

    if not p or not t:
        return False

    if p in t or t in p:
        return True

    p_tokens = {
        x
        for x in p.split()
        if len(x) >= 4
        and x not in {
            "disease", "syndrome", "disorder", "trait"
        }
    }

    if not p_tokens:
        return False

    return len(
        p_tokens
        &
        set(t.split())
    ) >= max(1, min(2, len(p_tokens)))


# =============================================================================
# CACHE / HTTP
# =============================================================================

class CachedHTTP:
    def __init__(
        self,
        cache_dir,
        timeout=120,
        min_interval=0.15,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.timeout = timeout
        self.min_interval = min_interval
        self.last_call = 0.0

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": "GWAS2m-Step16/1.0 drug-target-analysis"
        })

    def _path(self, method, url, payload):
        encoded = json.dumps(
            {
                "method": method,
                "url": url,
                "payload": payload,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")

        digest = hashlib.sha256(
            encoded
        ).hexdigest()

        return self.cache_dir / f"{digest}.json"

    def json_request(
        self,
        method,
        url,
        *,
        params=None,
        body=None,
        force=False,
        retries=3,
    ):
        payload = {
            "params": params or {},
            "body": body or {},
        }

        cache_path = self._path(
            method.upper(),
            url,
            payload,
        )

        if cache_path.exists() and not force:
            return json.loads(
                cache_path.read_text(
                    encoding="utf-8"
                )
            )

        for attempt in range(1, retries + 1):
            elapsed = time.time() - self.last_call

            if elapsed < self.min_interval:
                time.sleep(
                    self.min_interval - elapsed
                )

            try:
                if method.upper() == "POST":
                    response = self.session.post(
                        url,
                        params=params,
                        json=body,
                        timeout=self.timeout,
                    )
                else:
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=self.timeout,
                    )

                self.last_call = time.time()

                response.raise_for_status()

                data = response.json()

                cache_path.write_text(
                    json.dumps(
                        data,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )

                return data

            except Exception:
                if attempt == retries:
                    raise

                time.sleep(
                    attempt * 2.0
                )


# =============================================================================
# INPUTS
# =============================================================================

def load_tsv(path, required=False):
    if not path.exists():
        if required:
            raise SystemExit(
                f"Required input does not exist:\n  {path}"
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


def load_inputs(
    step14_dir,
    step15_dir,
    step12_dir,
):
    ranking = load_tsv(
        step14_dir / "Step14_Gene_Ranking.tsv",
        required=True,
    )

    membership = load_tsv(
        step14_dir / "Step14_Gene_Set_Membership.tsv",
        required=True,
    )

    centrality = load_tsv(
        step15_dir / "Step15_Network_Centrality.tsv",
        required=False,
    )

    direct_edges = load_tsv(
        step15_dir / "Step15_Direct_Physical_Edges.tsv",
        required=False,
    )

    bridge = load_tsv(
        step15_dir / "Step15_OT_Bridge_Summary.tsv",
        required=False,
    )

    molecular_long = load_tsv(
        step12_dir / "Step12_Molecular_Evidence_Long.tsv",
        required=False,
    )

    if "EFFECTOR_GENE_ID" in ranking.columns:
        ranking["EFFECTOR_GENE_ID"] = (
            ranking["EFFECTOR_GENE_ID"]
            .map(clean_gene_id)
        )

    if "GENE_ID" in membership.columns:
        membership["GENE_ID"] = (
            membership["GENE_ID"]
            .map(clean_gene_id)
        )

    if "GENE_ID" in centrality.columns:
        centrality["GENE_ID"] = (
            centrality["GENE_ID"]
            .map(clean_gene_id)
        )

    return (
        ranking,
        membership,
        centrality,
        direct_edges,
        bridge,
        molecular_long,
    )


def gene_sets_from_membership(membership):
    result = {}

    if membership.empty:
        return result

    for name, group in membership.groupby(
        "GENE_SET",
        sort=False,
    ):
        result[str(name)] = dedupe_keep_order([
            clean_gene_id(v)
            for v in group["GENE_ID"]
            if clean_gene_id(v)
        ])

    return result


# =============================================================================
# OPEN TARGETS TRACTABILITY
# =============================================================================

OT_TARGET_QUERY = """
query TargetTractability($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    biotype
    tractability {
      label
      modality
      value
    }
  }
}
"""


def fetch_ot_tractability(
    http,
    ensembl_id,
    force=False,
):
    body = {
        "query": OT_TARGET_QUERY,
        "variables": {
            "ensemblId": ensembl_id,
        },
    }

    response = http.json_request(
        "POST",
        OT_GRAPHQL,
        body=body,
        force=force,
    )

    errors = response.get(
        "errors",
        []
    )

    target = (
        response.get("data", {})
        .get("target")
    )

    if errors:
        return None, errors

    return target, []


def flatten_ot_tractability(
    ensembl_id,
    target,
):
    if not target:
        return {
            "GENE_ID": ensembl_id,
            "OT_TARGET_FOUND": "NO",
            "OT_APPROVED_SYMBOL": "",
            "OT_BIOTYPE": "",
            "OT_TRACTABILITY_POSITIVE_N": 0,
            "OT_SM_TRACTABLE": "NO",
            "OT_AB_TRACTABLE": "NO",
            "OT_PROTAC_TRACTABLE": "NO",
            "OT_OTHER_CLINICAL_TRACTABLE": "NO",
            "OT_SM_POSITIVE_LABELS": "",
            "OT_AB_POSITIVE_LABELS": "",
            "OT_PROTAC_POSITIVE_LABELS": "",
            "OT_OC_POSITIVE_LABELS": "",
        }

    positive = defaultdict(list)

    for item in target.get(
        "tractability",
        []
    ) or []:
        modality = str(
            item.get("modality", "")
        ).strip()

        label = str(
            item.get("label", "")
        ).strip()

        value = item.get(
            "value"
        )

        if boolish(value):
            positive[modality].append(
                label
            )

    def modality_values(*keys):
        vals = []

        for key in keys:
            vals.extend(
                positive.get(key, [])
            )

        return dedupe_keep_order(
            vals
        )

    sm = modality_values(
        "SM",
        "Small molecule",
        "small molecule",
    )

    ab = modality_values(
        "AB",
        "Antibody",
        "antibody",
    )

    pr = modality_values(
        "PR",
        "PROTAC",
        "protac",
    )

    oc = modality_values(
        "OC",
        "Other clinical",
        "other clinical",
    )

    all_positive = (
        sm + ab + pr + oc
    )

    return {
        "GENE_ID": ensembl_id,
        "OT_TARGET_FOUND": "YES",
        "OT_APPROVED_SYMBOL": target.get(
            "approvedSymbol",
            "",
        ),
        "OT_BIOTYPE": target.get(
            "biotype",
            "",
        ),
        "OT_TRACTABILITY_POSITIVE_N": len(
            dedupe_keep_order(all_positive)
        ),
        "OT_SM_TRACTABLE": (
            "YES" if sm else "NO"
        ),
        "OT_AB_TRACTABLE": (
            "YES" if ab else "NO"
        ),
        "OT_PROTAC_TRACTABLE": (
            "YES" if pr else "NO"
        ),
        "OT_OTHER_CLINICAL_TRACTABLE": (
            "YES" if oc else "NO"
        ),
        "OT_SM_POSITIVE_LABELS": ";".join(sm),
        "OT_AB_POSITIVE_LABELS": ";".join(ab),
        "OT_PROTAC_POSITIVE_LABELS": ";".join(pr),
        "OT_OC_POSITIVE_LABELS": ";".join(oc),
    }


# =============================================================================
# ChEMBL REST
# =============================================================================

def chembl_get_all(
    http,
    endpoint,
    params=None,
    force=False,
    max_pages=100,
):
    params = dict(
        params or {}
    )

    params.setdefault(
        "limit",
        1000,
    )

    url = f"{CHEMBL_BASE}/{endpoint}.json"

    rows = []
    page = 0
    offset = int(
        params.get(
            "offset",
            0,
        )
    )

    while page < max_pages:
        page += 1

        params["offset"] = offset

        data = http.json_request(
            "GET",
            url,
            params=params,
            force=force,
        )

        plural = endpoint.replace(
            "-",
            "_",
        )

        candidates = [
            plural,
            plural + "s",
            endpoint,
            endpoint + "s",
        ]

        records = None

        for key in candidates:
            if key in data and isinstance(
                data[key],
                list,
            ):
                records = data[key]
                break

        if records is None:
            for key, value in data.items():
                if (
                    key != "page_meta"
                    and
                    isinstance(value, list)
                ):
                    records = value
                    break

        records = records or []

        rows.extend(
            records
        )

        meta = data.get(
            "page_meta",
            {}
        ) or {}

        next_url = meta.get(
            "next"
        )

        if not next_url or not records:
            break

        offset += len(
            records
        )

    return rows


def target_synonym_tokens(record):
    tokens = set()

    pref = clean_symbol(
        record.get(
            "pref_name",
            "",
        )
    )

    if pref:
        tokens.add(
            norm(pref)
        )

    for component in record.get(
        "target_components",
        []
    ) or []:
        for syn in component.get(
            "target_component_synonyms",
            []
        ) or []:
            value = (
                syn.get("component_synonym")
                or
                syn.get("syn_type")
                or
                ""
            )

            if value:
                tokens.add(
                    norm(value)
                )

    return tokens


def choose_chembl_target(
    records,
    symbol,
):
    if not records:
        return None, "NO_RESULTS"

    symbol_norm = norm(
        symbol
    )

    scored = []

    for record in records:
        organism = str(
            record.get("organism", "")
        )

        target_type = str(
            record.get("target_type", "")
        ).upper()

        score = 0

        if organism.lower() == "homo sapiens":
            score += 100

        if target_type == "SINGLE PROTEIN":
            score += 50
        elif "PROTEIN" in target_type:
            score += 20

        synonyms = target_synonym_tokens(
            record
        )

        if symbol_norm in synonyms:
            score += 100

        pref = norm(
            record.get(
                "pref_name",
                "",
            )
        )

        if symbol_norm and symbol_norm in pref.split():
            score += 20

        scored.append(
            (
                score,
                record,
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_score, best = scored[0]

    if best_score < 100:
        return None, "NO_CONFIDENT_HUMAN_MATCH"

    return best, f"SCORE_{best_score}"


def fetch_chembl_target_by_symbol(
    http,
    symbol,
    force=False,
):
    if not symbol:
        return None, "NO_SYMBOL", []

    rows = chembl_get_all(
        http,
        "target/search",
        params={
            "q": symbol,
            "limit": 100,
        },
        force=force,
        max_pages=3,
    )

    best, status = choose_chembl_target(
        rows,
        symbol,
    )

    return best, status, rows


def fetch_chembl_mechanisms(
    http,
    target_chembl_id,
    force=False,
):
    if not target_chembl_id:
        return []

    return chembl_get_all(
        http,
        "mechanism",
        params={
            "target_chembl_id": target_chembl_id,
            "limit": 1000,
        },
        force=force,
    )


def fetch_chembl_molecule(
    http,
    molecule_chembl_id,
    force=False,
):
    if not molecule_chembl_id:
        return {}

    url = (
        f"{CHEMBL_BASE}/molecule/"
        f"{molecule_chembl_id}.json"
    )

    try:
        return http.json_request(
            "GET",
            url,
            force=force,
        )
    except Exception:
        return {}


def fetch_chembl_indications(
    http,
    molecule_chembl_id,
    force=False,
):
    if not molecule_chembl_id:
        return []

    return chembl_get_all(
        http,
        "drug_indication",
        params={
            "molecule_chembl_id": molecule_chembl_id,
            "limit": 1000,
        },
        force=force,
    )


# =============================================================================
# DIRECTION OF EFFECT
# =============================================================================

def detect_direction_columns(df):
    if df.empty:
        return {}

    return {
        "gene": first_existing(
            df.columns,
            [
                "EFFECTOR_GENE_ID",
                "GENE_ID",
                "MOLECULAR_GENE_ID",
            ],
        ),
        "qtl_type": first_existing(
            df.columns,
            [
                "QTL_TYPE",
                "BEST_QTL_TYPE",
                "MOLECULAR_QTL_TYPE",
            ],
        ),
        "h4": first_existing(
            df.columns,
            [
                "BEST_H4",
                "H4",
                "PP_H4",
                "COLOC_H4",
            ],
        ),
        "gwas_beta": first_existing(
            df.columns,
            [
                "GWAS_BETA",
                "BETA_GWAS",
                "GWAS_EFFECT",
                "GWAS_EFFECT_SIZE",
            ],
        ),
        "qtl_beta": first_existing(
            df.columns,
            [
                "QTL_BETA",
                "BETA_QTL",
                "MOLECULAR_BETA",
                "QTL_EFFECT",
                "MOLECULAR_EFFECT_SIZE",
            ],
        ),
        "harmonized": first_existing(
            df.columns,
            [
                "ALLELES_HARMONIZED",
                "HARMONIZED",
                "EFFECT_ALLELES_HARMONIZED",
                "SIGNED_EFFECTS_HARMONIZED",
            ],
        ),
        "tissue": first_existing(
            df.columns,
            [
                "TISSUE",
                "BEST_TISSUE",
            ],
        ),
    }


def derive_direction_evidence(
    molecular_long,
    strong_h4,
    assume_harmonized=False,
):
    columns = detect_direction_columns(
        molecular_long
    )

    required = [
        columns.get("gene"),
        columns.get("qtl_type"),
        columns.get("gwas_beta"),
        columns.get("qtl_beta"),
    ]

    if (
        molecular_long.empty
        or
        any(x is None for x in required)
    ):
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "UNAVAILABLE",
            "detail": (
                "Required signed GWAS/QTL beta columns were not found "
                "in Step12_Molecular_Evidence_Long.tsv."
            ),
            "columns": columns,
        }

    if (
        columns["harmonized"] is None
        and
        not assume_harmonized
    ):
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "UNAVAILABLE",
            "detail": (
                "Signed beta columns exist, but no explicit allele-harmonisation "
                "column was detected. Direction inference was intentionally disabled. "
                "Use --assume-harmonized only if the Step12 effects are known to share "
                "the same aligned effect allele."
            ),
            "columns": columns,
        }

    rows = []

    for _, row in molecular_long.iterrows():
        gid = clean_gene_id(
            row.get(
                columns["gene"],
                "",
            )
        )

        qtl = str(
            row.get(
                columns["qtl_type"],
                "",
            )
        ).strip()

        if not gid or norm(qtl).replace(" ", "") not in DIRECTION_ELIGIBLE_QTLS:
            continue

        if columns["h4"]:
            h4 = safe_float(
                row.get(
                    columns["h4"]
                )
            )

            if pd.notna(h4) and h4 < strong_h4:
                continue
        else:
            h4 = np.nan

        if columns["harmonized"]:
            harmonized = boolish(
                row.get(
                    columns["harmonized"]
                )
            )
        else:
            harmonized = bool(
                assume_harmonized
            )

        if not harmonized:
            continue

        gwas_beta = safe_float(
            row.get(
                columns["gwas_beta"]
            )
        )

        qtl_beta = safe_float(
            row.get(
                columns["qtl_beta"]
            )
        )

        if (
            pd.isna(gwas_beta)
            or
            pd.isna(qtl_beta)
            or
            gwas_beta == 0
            or
            qtl_beta == 0
        ):
            continue

        product = (
            gwas_beta
            *
            qtl_beta
        )

        if product > 0:
            molecular_risk = (
                "HIGHER_MOLECULAR_LEVEL_ASSOCIATED_WITH_HIGHER_DISEASE_RISK"
            )

            desired_action = "DOWNREGULATE_OR_INHIBIT"

        else:
            molecular_risk = (
                "HIGHER_MOLECULAR_LEVEL_ASSOCIATED_WITH_LOWER_DISEASE_RISK"
            )

            desired_action = "UPREGULATE_OR_ACTIVATE"

        rows.append({
            "GENE_ID": gid,
            "QTL_TYPE": qtl,
            "TISSUE": (
                row.get(
                    columns["tissue"],
                    ""
                )
                if columns["tissue"]
                else
                ""
            ),
            "H4": h4,
            "GWAS_BETA": gwas_beta,
            "QTL_BETA": qtl_beta,
            "BETA_PRODUCT": product,
            "ALLELES_HARMONIZED": "YES",
            "MOLECULAR_RISK_DIRECTION": molecular_risk,
            "THERAPEUTIC_DIRECTION": desired_action,
        })

    evidence = pd.DataFrame(
        rows
    )

    if evidence.empty:
        return evidence, pd.DataFrame(), {
            "status": "NO_ELIGIBLE_ROWS",
            "detail": (
                "Direction columns were available, but no strong harmonised "
                "eQTL/pQTL rows had usable signed effects."
            ),
            "columns": columns,
        }

    summaries = []

    for gid, group in evidence.groupby(
        "GENE_ID",
        sort=False,
    ):
        weights = (
            safe_numeric(
                group["H4"]
            )
            .fillna(1.0)
            .clip(lower=0.0)
        )

        down = (
            group["THERAPEUTIC_DIRECTION"]
            ==
            "DOWNREGULATE_OR_INHIBIT"
        )

        up = (
            group["THERAPEUTIC_DIRECTION"]
            ==
            "UPREGULATE_OR_ACTIVATE"
        )

        down_weight = float(
            weights[down].sum()
        )

        up_weight = float(
            weights[up].sum()
        )

        total = (
            down_weight
            +
            up_weight
        )

        if total <= 0:
            consensus = "UNKNOWN"
            support_fraction = np.nan
        else:
            major = max(
                down_weight,
                up_weight,
            )

            support_fraction = (
                major
                /
                total
            )

            if support_fraction < 0.70:
                consensus = "CONFLICTING_SIGNED_EVIDENCE"
            elif down_weight >= up_weight:
                consensus = "DOWNREGULATE_OR_INHIBIT"
            else:
                consensus = "UPREGULATE_OR_ACTIVATE"

        summaries.append({
            "GENE_ID": gid,
            "DIRECTION_STATUS": (
                "CONSENSUS"
                if consensus not in {
                    "UNKNOWN",
                    "CONFLICTING_SIGNED_EVIDENCE",
                }
                else consensus
            ),
            "THERAPEUTIC_DIRECTION": consensus,
            "DIRECTION_SUPPORT_FRACTION": support_fraction,
            "N_DIRECTION_ROWS": len(group),
            "N_EQTL_ROWS": int(
                group["QTL_TYPE"]
                .astype(str)
                .str.lower()
                .eq("eqtl")
                .sum()
            ),
            "N_PQTL_ROWS": int(
                group["QTL_TYPE"]
                .astype(str)
                .str.lower()
                .eq("pqtl")
                .sum()
            ),
            "DIRECTION_TISSUES": join_unique(
                group["TISSUE"]
            ),
        })

    return (
        evidence,
        pd.DataFrame(summaries),
        {
            "status": "OK",
            "detail": (
                "Direction inferred only from strong signed allele-harmonised "
                "eQTL/pQTL rows."
            ),
            "columns": columns,
        },
    )


def classify_action_type(action_type, mechanism_text=""):
    text = norm(
        f"{action_type} {mechanism_text}"
    )

    for word in INHIBITORY_ACTION_WORDS:
        if norm(word) in text:
            return "INHIBITORY"

    for word in ACTIVATING_ACTION_WORDS:
        if norm(word) in text:
            return "ACTIVATING"

    return "OTHER_OR_UNKNOWN"


def direction_alignment(
    therapeutic_direction,
    drug_action_class,
):
    if therapeutic_direction == "DOWNREGULATE_OR_INHIBIT":
        if drug_action_class == "INHIBITORY":
            return "ALIGNED"
        if drug_action_class == "ACTIVATING":
            return "OPPOSED"

    if therapeutic_direction == "UPREGULATE_OR_ACTIVATE":
        if drug_action_class == "ACTIVATING":
            return "ALIGNED"
        if drug_action_class == "INHIBITORY":
            return "OPPOSED"

    return "UNKNOWN"


# =============================================================================
# TARGET PRIORITY HEURISTIC
# =============================================================================

def percentile_map(values):
    series = safe_numeric(
        pd.Series(values)
    )

    if series.notna().sum() <= 1:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return series.rank(
        pct=True,
        method="average",
    ).fillna(0.0)


def compute_priority(
    targets,
):
    x = targets.copy()

    # Genetic: max 45
    x["SCORE_GENETIC"] = (
        safe_numeric(
            x["MAX_H4"]
        )
        .fillna(0.0)
        .clip(0, 1)
        *
        35.0
    )

    x["SCORE_GENETIC"] += (
        x["PRIOR_ROBUST"]
        .map(boolish)
        .astype(int)
        *
        5.0
    )

    x["SCORE_GENETIC"] += (
        (
            safe_numeric(
                x["N_QTL_TYPES"]
            )
            .fillna(0)
            >=
            2
        )
        .astype(int)
        *
        5.0
    )

    # External / locus validation: max 10
    x["SCORE_EXTERNAL"] = (
        x["OT_L2G_SUPPORTED_ANY"]
        .map(boolish)
        .astype(int)
        *
        7.0
    )

    x["SCORE_EXTERNAL"] += (
        x["IS_LOCUS_TOP_STRONG"]
        .map(boolish)
        .astype(int)
        *
        3.0
    )

    # Tractability / precedence: max 20
    tract_modalities = (
        x[
            [
                "OT_SM_TRACTABLE",
                "OT_AB_TRACTABLE",
                "OT_PROTAC_TRACTABLE",
                "OT_OTHER_CLINICAL_TRACTABLE",
            ]
        ]
        .applymap(boolish)
        .sum(axis=1)
    )

    x["SCORE_TRACTABILITY"] = (
        np.minimum(
            tract_modalities,
            2,
        )
        *
        3.5
    )

    x["SCORE_TRACTABILITY"] += (
        x["CHEMBL_TARGET_FOUND"]
        .map(boolish)
        .astype(int)
        *
        3.0
    )

    x["SCORE_TRACTABILITY"] += (
        safe_numeric(
            x["CHEMBL_MAX_PHASE"]
        )
        .fillna(0.0)
        .clip(0, 4)
        /
        4.0
        *
        10.0
    )

    # Network: max 10
    x["_NETWORK_PCT"] = percentile_map(
        x["NETWORK_PAGERANK"]
    )

    x["SCORE_NETWORK"] = (
        x["_NETWORK_PCT"]
        *
        7.0
    )

    x["SCORE_NETWORK"] += (
        x["IN_DIRECT_PHYSICAL_NETWORK"]
        .map(boolish)
        .astype(int)
        *
        3.0
    )

    # Direction / actionable MoA: max 15
    x["SCORE_DIRECTION"] = 0.0

    x.loc[
        x["DIRECTION_STATUS"] == "CONSENSUS",
        "SCORE_DIRECTION",
    ] += 5.0

    x.loc[
        safe_numeric(
            x["N_DIRECTIONALLY_ALIGNED_DRUGS"]
        )
        .fillna(0)
        >
        0,
        "SCORE_DIRECTION",
    ] += 10.0

    x["THERAPEUTIC_PRIORITY_SCORE"] = (
        x["SCORE_GENETIC"]
        +
        x["SCORE_EXTERNAL"]
        +
        x["SCORE_TRACTABILITY"]
        +
        x["SCORE_NETWORK"]
        +
        x["SCORE_DIRECTION"]
    )

    x["THERAPEUTIC_PRIORITY_SCORE"] = (
        x["THERAPEUTIC_PRIORITY_SCORE"]
        .clip(
            0,
            100,
        )
    )

    x["THERAPEUTIC_PRIORITY_TIER"] = pd.cut(
        x["THERAPEUTIC_PRIORITY_SCORE"],
        bins=[
            -np.inf,
            35,
            50,
            65,
            80,
            np.inf,
        ],
        labels=[
            "E_LOW",
            "D_EXPLORATORY",
            "C_MODERATE",
            "B_HIGH",
            "A_VERY_HIGH",
        ],
    ).astype(str)

    return x.drop(
        columns=[
            "_NETWORK_PCT"
        ],
        errors="ignore",
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
        "--target-set",
        default="CORE_STRONG",
        help="Step14 gene set used as the primary Step16 target list.",
    )

    parser.add_argument(
        "--strong-h4",
        type=float,
        default=0.80,
        help="Minimum H4 used when filtering direction evidence.",
    )

    parser.add_argument(
        "--assume-harmonized",
        action="store_true",
        help=(
            "Allow signed direction inference when no explicit harmonisation "
            "column exists. Use ONLY if Step12 effect alleles are known aligned."
        ),
    )

    parser.add_argument(
        "--step14-dir",
        default="",
    )

    parser.add_argument(
        "--step15-dir",
        default="",
    )

    parser.add_argument(
        "--step12-dir",
        default="",
    )

    parser.add_argument(
        "--output-dir",
        default="",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build local target list and direction audit without web/API queries.",
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

    step15_dir = (
        Path(args.step15_dir).resolve()
        if args.step15_dir
        else
        root
        /
        "15_network"
        /
        phenotype_slug
        /
        ancestry_slug
    )

    step12_dir = (
        Path(args.step12_dir).resolve()
        if args.step12_dir
        else
        root
        /
        "12_master_table"
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
        "16_drug_targets"
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
        "GWAS2m STEP16 v1.0 - DRUG TARGET / TRACTABILITY / DIRECTION ANALYSIS"
    )

    print(f"Root                     : {root}")
    print(f"Phenotype                : {args.phenotype}")
    print(f"Ancestry                 : {ancestry_label} ({ancestry_code})")
    print(f"Primary target set       : {args.target_set}")
    print(f"Step14                   : {step14_dir}")
    print(f"Step15                   : {step15_dir}")
    print(f"Step12                   : {step12_dir}")
    print(f"Output                   : {output}")
    print(f"Strong direction H4      : >= {args.strong_h4}")
    print(f"Assume harmonized        : {args.assume_harmonized}")
    print(f"Dry run                  : {args.dry_run}")

    (
        ranking,
        membership,
        centrality,
        direct_edges,
        bridge,
        molecular_long,
    ) = load_inputs(
        step14_dir,
        step15_dir,
        step12_dir,
    )

    gene_sets = gene_sets_from_membership(
        membership
    )

    if args.target_set not in gene_sets:
        raise SystemExit(
            f"Target set {args.target_set!r} is not present in "
            "Step14_Gene_Set_Membership.tsv.\n"
            f"Available: {', '.join(sorted(gene_sets))}"
        )

    target_ids = gene_sets[
        args.target_set
    ]

    targets = ranking[
        ranking["EFFECTOR_GENE_ID"]
        .isin(target_ids)
    ].copy()

    targets["IS_LOCUS_TOP_STRONG"] = targets[
        "EFFECTOR_GENE_ID"
    ].isin(
        set(
            gene_sets.get(
                "LOCUS_TOP_STRONG",
                [],
            )
        )
    ).map({
        True: "YES",
        False: "NO",
    })

    # Network merge
    network_cols = [
        "GENE_ID",
        "PREFERRED_NAME",
        "DEGREE",
        "WEIGHTED_DEGREE",
        "BETWEENNESS",
        "PAGERANK",
        "COMPONENT_ID",
        "COMMUNITY_ID",
    ]

    if not centrality.empty and "GENE_ID" in centrality.columns:
        network = centrality[
            [
                c
                for c in network_cols
                if c in centrality.columns
            ]
        ].copy()

        network = network.drop_duplicates(
            "GENE_ID"
        )

        rename = {
            "DEGREE": "NETWORK_DEGREE",
            "WEIGHTED_DEGREE": "NETWORK_WEIGHTED_DEGREE",
            "BETWEENNESS": "NETWORK_BETWEENNESS",
            "PAGERANK": "NETWORK_PAGERANK",
            "COMPONENT_ID": "NETWORK_COMPONENT_ID",
            "COMMUNITY_ID": "NETWORK_COMMUNITY_ID",
            "PREFERRED_NAME": "NETWORK_PREFERRED_NAME",
        }

        network = network.rename(
            columns=rename
        )

        targets = targets.merge(
            network,
            left_on="EFFECTOR_GENE_ID",
            right_on="GENE_ID",
            how="left",
        ).drop(
            columns=["GENE_ID"],
            errors="ignore",
        )

    for col in [
        "NETWORK_DEGREE",
        "NETWORK_WEIGHTED_DEGREE",
        "NETWORK_BETWEENNESS",
        "NETWORK_PAGERANK",
        "NETWORK_COMPONENT_ID",
        "NETWORK_COMMUNITY_ID",
    ]:
        if col not in targets.columns:
            targets[col] = np.nan

    direct_seed_ids = set()

    if not direct_edges.empty:
        for col in [
            "GENE_A_ID",
            "GENE_B_ID",
        ]:
            if col in direct_edges.columns:
                direct_seed_ids.update(
                    clean_gene_id(v)
                    for v in direct_edges[col]
                    if clean_gene_id(v)
                )

    targets["IN_DIRECT_PHYSICAL_NETWORK"] = (
        targets["EFFECTOR_GENE_ID"]
        .isin(direct_seed_ids)
        .map({
            True: "YES",
            False: "NO",
        })
    )

    # -------------------------------------------------------------------------
    # Direction audit first
    # -------------------------------------------------------------------------

    (
        direction_evidence,
        direction_summary,
        direction_meta,
    ) = derive_direction_evidence(
        molecular_long,
        args.strong_h4,
        assume_harmonized=args.assume_harmonized,
    )

    direction_evidence.to_csv(
        output / "Step16_Direction_Evidence.tsv",
        sep="\t",
        index=False,
    )

    direction_summary.to_csv(
        output / "Step16_Direction_Summary.tsv",
        sep="\t",
        index=False,
    )

    section("DIRECTION-OF-EFFECT AUDIT")

    print(
        f"Status                   : {direction_meta['status']}"
    )
    print(
        f"Detail                   : {direction_meta['detail']}"
    )
    print(
        "Detected columns         : "
        +
        json.dumps(
            direction_meta.get(
                "columns",
                {},
            ),
            default=str,
        )
    )

    if not direction_summary.empty:
        show(
            direction_summary,
            columns=[
                "GENE_ID",
                "DIRECTION_STATUS",
                "THERAPEUTIC_DIRECTION",
                "DIRECTION_SUPPORT_FRACTION",
                "N_DIRECTION_ROWS",
                "N_EQTL_ROWS",
                "N_PQTL_ROWS",
                "DIRECTION_TISSUES",
            ],
        )

    # Add defaults
    if direction_summary.empty:
        targets["DIRECTION_STATUS"] = "UNAVAILABLE"
        targets["THERAPEUTIC_DIRECTION"] = "UNKNOWN"
        targets["DIRECTION_SUPPORT_FRACTION"] = np.nan
    else:
        targets = targets.merge(
            direction_summary,
            left_on="EFFECTOR_GENE_ID",
            right_on="GENE_ID",
            how="left",
        ).drop(
            columns=["GENE_ID"],
            errors="ignore",
        )

        targets["DIRECTION_STATUS"] = targets[
            "DIRECTION_STATUS"
        ].fillna(
            "NO_ELIGIBLE_SIGNED_EVIDENCE"
        )

        targets["THERAPEUTIC_DIRECTION"] = targets[
            "THERAPEUTIC_DIRECTION"
        ].fillna(
            "UNKNOWN"
        )

    # -------------------------------------------------------------------------
    # Print primary target input
    # -------------------------------------------------------------------------

    section("STEP16 PRIMARY TARGETS")

    show(
        targets,
        max_rows=None if not args.compact_screen else 50,
        columns=[
            "RANK_STEP14",
            "EFFECTOR_GENE_ID",
            "EFFECTOR_GENE_SYMBOL",
            "MAX_H4",
            "PRIOR_ROBUST",
            "N_QTL_TYPES",
            "QTL_TYPES",
            "N_TISSUES",
            "OT_L2G_SUPPORTED_ANY",
            "IS_LOCUS_TOP_STRONG",
            "IN_DIRECT_PHYSICAL_NETWORK",
            "NETWORK_DEGREE",
            "NETWORK_BETWEENNESS",
            "NETWORK_PAGERANK",
            "DIRECTION_STATUS",
            "THERAPEUTIC_DIRECTION",
        ],
    )

    if args.dry_run:
        section("DRY RUN COMPLETE")
        print(
            "Local Step14/Step15 evidence loaded successfully. "
            "Open Targets and ChEMBL were not queried."
        )
        return

    # -------------------------------------------------------------------------
    # APIs
    # -------------------------------------------------------------------------

    cache = (
        root
        /
        "resources"
        /
        "drug_target_cache"
    )

    http = CachedHTTP(
        cache
    )

    ot_rows = []
    chembl_target_rows = []
    mechanism_rows = []
    indication_rows = []
    qc_api_errors = []

    molecule_cache = {}
    indication_cache = {}

    section("QUERYING OPEN TARGETS + ChEMBL")

    for i, target_row in enumerate(
        targets.itertuples(index=False),
        start=1,
    ):
        gid = clean_gene_id(
            getattr(
                target_row,
                "EFFECTOR_GENE_ID",
                "",
            )
        )

        symbol = clean_symbol(
            getattr(
                target_row,
                "EFFECTOR_GENE_SYMBOL",
                "",
            )
        )

        print()
        print(
            f"[{i}/{len(targets)}] "
            f"{symbol or gid} ({gid})"
        )

        # ---------------------------
        # Open Targets tractability
        # ---------------------------

        try:
            ot_target, ot_errors = fetch_ot_tractability(
                http,
                gid,
                force=args.force_refresh,
            )

            ot_row = flatten_ot_tractability(
                gid,
                ot_target,
            )

            ot_row["OT_API_ERRORS"] = (
                json.dumps(
                    ot_errors,
                    default=str,
                )
                if ot_errors
                else
                ""
            )

            ot_rows.append(
                ot_row
            )

            print(
                "  Open Targets tractability: "
                f"SM={ot_row['OT_SM_TRACTABLE']} "
                f"AB={ot_row['OT_AB_TRACTABLE']} "
                f"PROTAC={ot_row['OT_PROTAC_TRACTABLE']} "
                f"OC={ot_row['OT_OTHER_CLINICAL_TRACTABLE']}"
            )

        except Exception as exc:
            qc_api_errors.append({
                "GENE_ID": gid,
                "SOURCE": "OPEN_TARGETS",
                "ERROR": f"{type(exc).__name__}: {exc}",
            })

            ot_rows.append(
                flatten_ot_tractability(
                    gid,
                    None,
                )
            )

            print(
                f"  [WARN] Open Targets failed: {exc}"
            )

        # ---------------------------
        # ChEMBL target mapping
        # ---------------------------

        try:
            chembl_target, map_status, candidates = (
                fetch_chembl_target_by_symbol(
                    http,
                    symbol,
                    force=args.force_refresh,
                )
            )

        except Exception as exc:
            chembl_target = None
            map_status = (
                f"ERROR:{type(exc).__name__}:{exc}"
            )
            candidates = []

            qc_api_errors.append({
                "GENE_ID": gid,
                "SOURCE": "CHEMBL_TARGET",
                "ERROR": f"{type(exc).__name__}: {exc}",
            })

        if chembl_target:
            target_chembl_id = str(
                chembl_target.get(
                    "target_chembl_id",
                    "",
                )
            )

            chembl_target_rows.append({
                "GENE_ID": gid,
                "GENE_SYMBOL": symbol,
                "CHEMBL_TARGET_FOUND": "YES",
                "CHEMBL_TARGET_ID": target_chembl_id,
                "CHEMBL_TARGET_NAME": chembl_target.get(
                    "pref_name",
                    "",
                ),
                "CHEMBL_TARGET_TYPE": chembl_target.get(
                    "target_type",
                    "",
                ),
                "CHEMBL_ORGANISM": chembl_target.get(
                    "organism",
                    "",
                ),
                "CHEMBL_MAPPING_STATUS": map_status,
                "N_SEARCH_CANDIDATES": len(candidates),
            })

            print(
                f"  ChEMBL target: {target_chembl_id} "
                f"{chembl_target.get('pref_name', '')}"
            )

            try:
                mechanisms = fetch_chembl_mechanisms(
                    http,
                    target_chembl_id,
                    force=args.force_refresh,
                )
            except Exception as exc:
                mechanisms = []

                qc_api_errors.append({
                    "GENE_ID": gid,
                    "SOURCE": "CHEMBL_MECHANISM",
                    "ERROR": f"{type(exc).__name__}: {exc}",
                })

            print(
                f"  ChEMBL mechanisms: {len(mechanisms)}"
            )

            for mech in mechanisms:
                molecule_id = str(
                    mech.get(
                        "molecule_chembl_id",
                        "",
                    )
                )

                if molecule_id not in molecule_cache:
                    molecule_cache[molecule_id] = fetch_chembl_molecule(
                        http,
                        molecule_id,
                        force=args.force_refresh,
                    )

                molecule = molecule_cache.get(
                    molecule_id,
                    {},
                )

                if molecule_id not in indication_cache:
                    try:
                        indication_cache[molecule_id] = fetch_chembl_indications(
                            http,
                            molecule_id,
                            force=args.force_refresh,
                        )
                    except Exception:
                        indication_cache[molecule_id] = []

                indications = indication_cache.get(
                    molecule_id,
                    [],
                )

                action_type = str(
                    mech.get(
                        "action_type",
                        "",
                    )
                )

                mechanism_text = str(
                    mech.get(
                        "mechanism_of_action",
                        "",
                    )
                )

                action_class = classify_action_type(
                    action_type,
                    mechanism_text,
                )

                therapeutic_direction = (
                    targets.loc[
                        targets["EFFECTOR_GENE_ID"] == gid,
                        "THERAPEUTIC_DIRECTION",
                    ].iloc[0]
                    if (
                        targets["EFFECTOR_GENE_ID"] == gid
                    ).any()
                    else
                    "UNKNOWN"
                )

                alignment = direction_alignment(
                    therapeutic_direction,
                    action_class,
                )

                indication_texts = []

                phenotype_max_phase = np.nan
                has_phenotype_match = False

                for indication in indications:
                    efo_term = str(
                        indication.get(
                            "efo_term",
                            "",
                        )
                    )

                    mesh_heading = str(
                        indication.get(
                            "mesh_heading",
                            "",
                        )
                    )

                    ind_text = (
                        efo_term
                        or
                        mesh_heading
                    )

                    indication_texts.append(
                        ind_text
                    )

                    matched = phenotype_match(
                        args.phenotype,
                        ind_text,
                    )

                    ind_phase = phase_number(
                        indication.get(
                            "max_phase_for_ind",
                            np.nan,
                        )
                    )

                    indication_rows.append({
                        "GENE_ID": gid,
                        "GENE_SYMBOL": symbol,
                        "CHEMBL_TARGET_ID": target_chembl_id,
                        "MOLECULE_CHEMBL_ID": molecule_id,
                        "MOLECULE_NAME": molecule.get(
                            "pref_name",
                            "",
                        ),
                        "EFO_ID": indication.get(
                            "efo_id",
                            "",
                        ),
                        "EFO_TERM": efo_term,
                        "MESH_ID": indication.get(
                            "mesh_id",
                            "",
                        ),
                        "MESH_HEADING": mesh_heading,
                        "MAX_PHASE_FOR_INDICATION": ind_phase,
                        "PHENOTYPE_MATCH": (
                            "YES"
                            if matched
                            else
                            "NO"
                        ),
                    })

                    if matched:
                        has_phenotype_match = True

                        if (
                            pd.isna(
                                phenotype_max_phase
                            )
                            or
                            (
                                pd.notna(ind_phase)
                                and
                                ind_phase
                                >
                                phenotype_max_phase
                            )
                        ):
                            phenotype_max_phase = ind_phase

                mechanism_rows.append({
                    "GENE_ID": gid,
                    "GENE_SYMBOL": symbol,
                    "CHEMBL_TARGET_ID": target_chembl_id,
                    "MOLECULE_CHEMBL_ID": molecule_id,
                    "MOLECULE_NAME": molecule.get(
                        "pref_name",
                        "",
                    ),
                    "MOLECULE_TYPE": molecule.get(
                        "molecule_type",
                        "",
                    ),
                    "MAX_PHASE_OVERALL": phase_number(
                        molecule.get(
                            "max_phase",
                            np.nan,
                        )
                    ),
                    "FIRST_APPROVAL": molecule.get(
                        "first_approval",
                        np.nan,
                    ),
                    "ACTION_TYPE": action_type,
                    "ACTION_CLASS": action_class,
                    "MECHANISM_OF_ACTION": mechanism_text,
                    "THERAPEUTIC_DIRECTION": therapeutic_direction,
                    "DIRECTION_ALIGNMENT": alignment,
                    "PHENOTYPE_INDICATION_MATCH": (
                        "YES"
                        if has_phenotype_match
                        else
                        "NO"
                    ),
                    "PHENOTYPE_MAX_PHASE": phenotype_max_phase,
                    "ALL_INDICATIONS": join_unique(
                        indication_texts
                    ),
                })

        else:
            chembl_target_rows.append({
                "GENE_ID": gid,
                "GENE_SYMBOL": symbol,
                "CHEMBL_TARGET_FOUND": "NO",
                "CHEMBL_TARGET_ID": "",
                "CHEMBL_TARGET_NAME": "",
                "CHEMBL_TARGET_TYPE": "",
                "CHEMBL_ORGANISM": "",
                "CHEMBL_MAPPING_STATUS": map_status,
                "N_SEARCH_CANDIDATES": len(candidates),
            })

            print(
                f"  ChEMBL target: NOT CONFIDENTLY MAPPED ({map_status})"
            )

    # -------------------------------------------------------------------------
    # Tables
    # -------------------------------------------------------------------------

    ot = pd.DataFrame(
        ot_rows
    )

    chembl_targets = pd.DataFrame(
        chembl_target_rows
    )

    mechanisms = pd.DataFrame(
        mechanism_rows
    )

    indications = pd.DataFrame(
        indication_rows
    )

    ot.to_csv(
        output / "Step16_OpenTargets_Tractability.tsv",
        sep="\t",
        index=False,
    )

    chembl_targets.to_csv(
        output / "Step16_ChEMBL_Target_Mapping.tsv",
        sep="\t",
        index=False,
    )

    mechanisms.to_csv(
        output / "Step16_Drug_Mechanisms.tsv",
        sep="\t",
        index=False,
    )

    indications.to_csv(
        output / "Step16_Drug_Indications.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Target-level drug summary
    # -------------------------------------------------------------------------

    drug_summary_rows = []

    for gid in target_ids:
        z = (
            mechanisms[
                mechanisms["GENE_ID"] == gid
            ].copy()
            if not mechanisms.empty
            else pd.DataFrame()
        )

        if z.empty:
            drug_summary_rows.append({
                "GENE_ID": gid,
                "N_DRUG_MECHANISMS": 0,
                "N_UNIQUE_DRUGS": 0,
                "N_APPROVED_OR_PHASE4_DRUGS": 0,
                "CHEMBL_MAX_PHASE": np.nan,
                "N_PHENOTYPE_MATCHED_DRUGS": 0,
                "PHENOTYPE_MAX_PHASE": np.nan,
                "N_DIRECTIONALLY_ALIGNED_DRUGS": 0,
                "DIRECTIONALLY_ALIGNED_DRUGS": "",
                "PHENOTYPE_MATCHED_DRUGS": "",
            })

            continue

        max_phase = safe_numeric(
            z["MAX_PHASE_OVERALL"]
        ).max()

        phenotype_max = safe_numeric(
            z["PHENOTYPE_MAX_PHASE"]
        ).max()

        approved = z[
            safe_numeric(
                z["MAX_PHASE_OVERALL"]
            )
            >=
            4
        ]

        pheno = z[
            z["PHENOTYPE_INDICATION_MATCH"]
            ==
            "YES"
        ]

        aligned = z[
            z["DIRECTION_ALIGNMENT"]
            ==
            "ALIGNED"
        ]

        drug_summary_rows.append({
            "GENE_ID": gid,
            "N_DRUG_MECHANISMS": len(z),
            "N_UNIQUE_DRUGS": z[
                "MOLECULE_CHEMBL_ID"
            ].nunique(),
            "N_APPROVED_OR_PHASE4_DRUGS": approved[
                "MOLECULE_CHEMBL_ID"
            ].nunique(),
            "CHEMBL_MAX_PHASE": max_phase,
            "N_PHENOTYPE_MATCHED_DRUGS": pheno[
                "MOLECULE_CHEMBL_ID"
            ].nunique(),
            "PHENOTYPE_MAX_PHASE": phenotype_max,
            "N_DIRECTIONALLY_ALIGNED_DRUGS": aligned[
                "MOLECULE_CHEMBL_ID"
            ].nunique(),
            "DIRECTIONALLY_ALIGNED_DRUGS": join_unique(
                aligned["MOLECULE_NAME"]
            ),
            "PHENOTYPE_MATCHED_DRUGS": join_unique(
                pheno["MOLECULE_NAME"]
            ),
        })

    drug_summary = pd.DataFrame(
        drug_summary_rows
    )

    # Merge all target-level layers
    targets = targets.merge(
        ot,
        left_on="EFFECTOR_GENE_ID",
        right_on="GENE_ID",
        how="left",
    ).drop(
        columns=["GENE_ID"],
        errors="ignore",
    )

    targets = targets.merge(
        chembl_targets,
        left_on="EFFECTOR_GENE_ID",
        right_on="GENE_ID",
        how="left",
    ).drop(
        columns=["GENE_ID"],
        errors="ignore",
    )

    targets = targets.merge(
        drug_summary,
        left_on="EFFECTOR_GENE_ID",
        right_on="GENE_ID",
        how="left",
    ).drop(
        columns=["GENE_ID"],
        errors="ignore",
    )

    # Defaults
    for col in [
        "OT_SM_TRACTABLE",
        "OT_AB_TRACTABLE",
        "OT_PROTAC_TRACTABLE",
        "OT_OTHER_CLINICAL_TRACTABLE",
        "CHEMBL_TARGET_FOUND",
    ]:
        if col not in targets.columns:
            targets[col] = "NO"

        targets[col] = targets[col].fillna(
            "NO"
        )

    for col in [
        "N_DRUG_MECHANISMS",
        "N_UNIQUE_DRUGS",
        "N_APPROVED_OR_PHASE4_DRUGS",
        "N_PHENOTYPE_MATCHED_DRUGS",
        "N_DIRECTIONALLY_ALIGNED_DRUGS",
    ]:
        if col not in targets.columns:
            targets[col] = 0

        targets[col] = safe_numeric(
            targets[col]
        ).fillna(0)

    if "CHEMBL_MAX_PHASE" not in targets.columns:
        targets["CHEMBL_MAX_PHASE"] = np.nan

    # Priority
    targets = compute_priority(
        targets
    )

    targets = targets.sort_values(
        [
            "THERAPEUTIC_PRIORITY_SCORE",
            "MAX_H4",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    targets[
        "STEP16_RANK"
    ] = np.arange(
        1,
        len(targets) + 1,
    )

    targets.to_csv(
        output / "Step16_Target_Prioritisation.tsv",
        sep="\t",
        index=False,
    )

    aligned_drugs = (
        mechanisms[
            mechanisms["DIRECTION_ALIGNMENT"]
            ==
            "ALIGNED"
        ].copy()
        if not mechanisms.empty
        else pd.DataFrame()
    )

    aligned_drugs.to_csv(
        output / "Step16_Directionally_Aligned_Drugs.tsv",
        sep="\t",
        index=False,
    )

    phenotype_drugs = (
        mechanisms[
            mechanisms["PHENOTYPE_INDICATION_MATCH"]
            ==
            "YES"
        ].copy()
        if not mechanisms.empty
        else pd.DataFrame()
    )

    phenotype_drugs.to_csv(
        output / "Step16_Phenotype_Matched_Drugs.tsv",
        sep="\t",
        index=False,
    )

    top = targets[
        (
            targets["MAX_H4"] >= args.strong_h4
        )
        &
        (
            targets["THERAPEUTIC_PRIORITY_SCORE"] >= 50
        )
    ].copy()

    top.to_csv(
        output / "Step16_Top_Therapeutic_Candidates.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Screen output
    # -------------------------------------------------------------------------

    show(
        ot,
        title="OPEN TARGETS TRACTABILITY",
        max_rows=None if not args.compact_screen else 50,
        columns=[
            "GENE_ID",
            "OT_APPROVED_SYMBOL",
            "OT_BIOTYPE",
            "OT_SM_TRACTABLE",
            "OT_AB_TRACTABLE",
            "OT_PROTAC_TRACTABLE",
            "OT_OTHER_CLINICAL_TRACTABLE",
            "OT_SM_POSITIVE_LABELS",
            "OT_AB_POSITIVE_LABELS",
            "OT_PROTAC_POSITIVE_LABELS",
        ],
    )

    show(
        chembl_targets,
        title="ChEMBL TARGET MAPPING",
        max_rows=None if not args.compact_screen else 50,
    )

    if not mechanisms.empty:
        show(
            mechanisms.sort_values(
                [
                    "MAX_PHASE_OVERALL",
                    "GENE_SYMBOL",
                ],
                ascending=[
                    False,
                    True,
                ],
            ),
            title="DRUG / MECHANISM EVIDENCE",
            max_rows=100 if args.compact_screen else 300,
            columns=[
                "GENE_SYMBOL",
                "GENE_ID",
                "MOLECULE_NAME",
                "MOLECULE_CHEMBL_ID",
                "MAX_PHASE_OVERALL",
                "ACTION_TYPE",
                "ACTION_CLASS",
                "MECHANISM_OF_ACTION",
                "THERAPEUTIC_DIRECTION",
                "DIRECTION_ALIGNMENT",
                "PHENOTYPE_INDICATION_MATCH",
                "PHENOTYPE_MAX_PHASE",
            ],
        )

    show(
        targets,
        title="FINAL STEP16 TARGET PRIORITISATION",
        max_rows=None,
        columns=[
            "STEP16_RANK",
            "EFFECTOR_GENE_SYMBOL",
            "EFFECTOR_GENE_ID",
            "MAX_H4",
            "PRIOR_ROBUST",
            "N_QTL_TYPES",
            "OT_L2G_SUPPORTED_ANY",
            "IS_LOCUS_TOP_STRONG",
            "IN_DIRECT_PHYSICAL_NETWORK",
            "NETWORK_PAGERANK",
            "OT_SM_TRACTABLE",
            "OT_AB_TRACTABLE",
            "OT_PROTAC_TRACTABLE",
            "CHEMBL_TARGET_FOUND",
            "N_UNIQUE_DRUGS",
            "N_APPROVED_OR_PHASE4_DRUGS",
            "CHEMBL_MAX_PHASE",
            "N_PHENOTYPE_MATCHED_DRUGS",
            "THERAPEUTIC_DIRECTION",
            "N_DIRECTIONALLY_ALIGNED_DRUGS",
            "SCORE_GENETIC",
            "SCORE_EXTERNAL",
            "SCORE_TRACTABILITY",
            "SCORE_NETWORK",
            "SCORE_DIRECTION",
            "THERAPEUTIC_PRIORITY_SCORE",
            "THERAPEUTIC_PRIORITY_TIER",
        ],
    )

    # -------------------------------------------------------------------------
    # Plain-English interpretation
    # -------------------------------------------------------------------------

    section("PLAIN-ENGLISH STEP16 INTERPRETATION")

    print(
        f"1. Primary Step16 target set: {args.target_set} "
        f"with {len(targets)} genes."
    )

    tractable_any = (
        targets[
            [
                "OT_SM_TRACTABLE",
                "OT_AB_TRACTABLE",
                "OT_PROTAC_TRACTABLE",
                "OT_OTHER_CLINICAL_TRACTABLE",
            ]
        ]
        .applymap(boolish)
        .any(axis=1)
    )

    print(
        f"2. {int(tractable_any.sum())}/{len(targets)} targets have at least "
        "one positive Open Targets tractability modality."
    )

    print(
        f"3. {int(targets['CHEMBL_TARGET_FOUND'].map(boolish).sum())}/{len(targets)} "
        "targets were confidently mapped to ChEMBL."
    )

    print(
        f"4. {int((targets['N_UNIQUE_DRUGS'] > 0).sum())}/{len(targets)} "
        "targets have at least one ChEMBL drug mechanism."
    )

    print(
        f"5. {int((targets['N_APPROVED_OR_PHASE4_DRUGS'] > 0).sum())}/{len(targets)} "
        "targets have Phase IV/approved clinical precedence in ChEMBL."
    )

    print(
        f"6. {int((targets['N_PHENOTYPE_MATCHED_DRUGS'] > 0).sum())}/{len(targets)} "
        f"targets have at least one ChEMBL indication matching {args.phenotype!r}."
    )

    n_direction = int(
        (targets["DIRECTION_STATUS"] == "CONSENSUS").sum()
    )

    print(
        f"7. Therapeutic direction could be conservatively inferred for "
        f"{n_direction}/{len(targets)} targets."
    )

    print(
        f"8. {int((targets['N_DIRECTIONALLY_ALIGNED_DRUGS'] > 0).sum())}/{len(targets)} "
        "targets have at least one ChEMBL mechanism aligned with the inferred "
        "genetic direction."
    )

    print()
    print(
        "IMPORTANT: therapeutic priority score is a transparent ranking heuristic, "
        "not a probability of treatment success."
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc_rows = [
        {
            "CHECK": "PRIMARY_TARGETS",
            "STATUS": "INFO",
            "VALUE": len(targets),
            "DETAIL": args.target_set,
        },
        {
            "CHECK": "DIRECTION_INFERENCE",
            "STATUS": (
                "OK"
                if direction_meta["status"] == "OK"
                else "WARN"
            ),
            "VALUE": n_direction,
            "DETAIL": direction_meta["detail"],
        },
        {
            "CHECK": "OT_TRACTABILITY_TARGETS_FOUND",
            "STATUS": "INFO",
            "VALUE": int(
                (
                    ot.get(
                        "OT_TARGET_FOUND",
                        pd.Series(dtype=str),
                    )
                    ==
                    "YES"
                ).sum()
            ),
            "DETAIL": "Open Targets GraphQL target tractability.",
        },
        {
            "CHECK": "CHEMBL_TARGETS_FOUND",
            "STATUS": "INFO",
            "VALUE": int(
                targets[
                    "CHEMBL_TARGET_FOUND"
                ]
                .map(boolish)
                .sum()
            ),
            "DETAIL": "Confident human ChEMBL target mappings.",
        },
        {
            "CHECK": "CHEMBL_DRUG_MECHANISMS",
            "STATUS": "INFO",
            "VALUE": len(mechanisms),
            "DETAIL": "Drug-target mechanism rows.",
        },
        {
            "CHECK": "API_ERRORS",
            "STATUS": (
                "WARN"
                if qc_api_errors
                else
                "OK"
            ),
            "VALUE": len(qc_api_errors),
            "DETAIL": (
                json.dumps(
                    qc_api_errors[:20],
                    default=str,
                )
                if qc_api_errors
                else
                "None"
            ),
        },
    ]

    qc = pd.DataFrame(
        qc_rows
    )

    qc.to_csv(
        output / "Step16_QC_Audit.tsv",
        sep="\t",
        index=False,
    )

    show(
        qc,
        title="STEP16 QC AUDIT",
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {
        "version": VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phenotype": args.phenotype,
        "ancestry_code": ancestry_code,
        "ancestry_label": ancestry_label,
        "primary_target_set": args.target_set,
        "n_targets": len(targets),
        "n_ot_any_tractable": int(
            tractable_any.sum()
        ),
        "n_chembl_targets": int(
            targets[
                "CHEMBL_TARGET_FOUND"
            ]
            .map(boolish)
            .sum()
        ),
        "n_targets_with_drugs": int(
            (
                targets[
                    "N_UNIQUE_DRUGS"
                ]
                >
                0
            ).sum()
        ),
        "n_targets_with_approved_drug_precedence": int(
            (
                targets[
                    "N_APPROVED_OR_PHASE4_DRUGS"
                ]
                >
                0
            ).sum()
        ),
        "n_targets_with_phenotype_matched_drugs": int(
            (
                targets[
                    "N_PHENOTYPE_MATCHED_DRUGS"
                ]
                >
                0
            ).sum()
        ),
        "n_targets_with_direction_consensus": n_direction,
        "n_targets_with_directionally_aligned_drugs": int(
            (
                targets[
                    "N_DIRECTIONALLY_ALIGNED_DRUGS"
                ]
                >
                0
            ).sum()
        ),
        "direction_status": direction_meta,
        "api_errors": qc_api_errors,
        "scientific_notes": [
            "H4 alone is not used to infer therapeutic direction.",
            "Direction is inferred only from signed harmonised eQTL/pQTL and GWAS effects.",
            "Splicing/isoform QTLs are not reduced to a simple gene activation/inhibition direction.",
            "Drug clinical precedence for another indication is not evidence of efficacy for the requested phenotype.",
            "The Step16 therapeutic priority score is a heuristic ranking, not a probability.",
        ],
    }

    (
        output
        /
        "Step16_summary.json"
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

    section("STEP16 COMPLETE")

    print(f"Output directory:")
    print(f"  {output}")
    print()

    for name in [
        "Step16_Target_Prioritisation.tsv",
        "Step16_OpenTargets_Tractability.tsv",
        "Step16_ChEMBL_Target_Mapping.tsv",
        "Step16_Drug_Mechanisms.tsv",
        "Step16_Drug_Indications.tsv",
        "Step16_Direction_Evidence.tsv",
        "Step16_Direction_Summary.tsv",
        "Step16_Directionally_Aligned_Drugs.tsv",
        "Step16_Phenotype_Matched_Drugs.tsv",
        "Step16_Top_Therapeutic_Candidates.tsv",
        "Step16_QC_Audit.tsv",
        "Step16_summary.json",
    ]:
        print(
            f"  {output / name}"
        )

    print()
    print(
        "NEXT STEP AFTER REVIEWING STEP16:"
    )
    print(
        "  Step17 = final disease-mechanism integration / evidence tiers / "
        "candidate mechanism report."
    )


if __name__ == "__main__":
    main()

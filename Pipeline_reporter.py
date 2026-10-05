 
# -*- coding: utf-8 -*-

"""
GWAS2m terminal-only interpretation report.

READ-ONLY:
  * does not create reports/
  * does not save TSV/CSV/JSON
  * does not modify pipeline outputs
  * prints all reporting tables to stdout

Main use:
    python Print_Pipeline_Report.py --phenotype migraine --ancestry EUR

One study:
    python Print_Pipeline_Report.py \
        --phenotype migraine \
        --ancestry EUR \
        --study GCST90129450

Step01-10 run status only (from Step01_10_Run.py status files):
    python Pipeline_reporter.py --phenotype migraine --ancestry EUR --status

The report is intentionally different from dumping raw GWAS files.
It prints:
  1. cross-study pipeline status
  2. every study separately, step by step
  3. key saved result tables for that study
  4. every UNIQUE Step11 colocalized variant for that study
  5. genomic context for every unique colocalized variant using GENCODE
  6. VEP / GTEx / SpliceAI / Pangolin evidence where available
  7. combined analysis across all studies
  8. every unique combined colocalized variant
  9. shared variants / genes / tissues across studies

Step11 contains repeated variant-gene-tissue records.  The genomic-context
tables are therefore based on UNIQUE VARIANT_KEY values so that one variant
is not counted many times simply because it appears in several tissues.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "2.0.0"


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

def norm_text(value) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def slugify(value) -> str:
    return norm_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = norm_text(value)
    if key not in ANCESTRY_ALIASES:
        raise SystemExit(
            f"Unsupported ancestry {value!r}. "
            "Use EUR, AFR, EAS, SAS or AMR."
        )
    return ANCESTRY_ALIASES[key]


def section(title: str, char: str = "=") -> None:
    print()
    print(char * 130)
    print(title)
    print(char * 130)


def subsection(title: str) -> None:
    section(title, "-")


def print_table(
    df: pd.DataFrame | None,
    title: str | None = None,
    *,
    all_rows: bool = True,
) -> None:
    if title:
        subsection(title)

    if df is None or df.empty:
        print("(no rows)")
        return

    rows = None if all_rows else 50

    with pd.option_context(
        "display.max_rows", rows,
        "display.max_columns", None,
        "display.width", 500,
        "display.max_colwidth", 120,
        "display.expand_frame_repr", False,
        "display.float_format", lambda x: f"{x:.6g}",
    ):
        print(df.to_string(index=False))


def safe_json(path: Path | None) -> dict:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def safe_table(path: Path | None, usecols=None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            sep="\t",
            compression="infer",
            low_memory=False,
            usecols=usecols,
        )
    except Exception as exc:
        print(f"[READ ERROR] {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def first_existing(directory: Path, patterns: list[str]) -> Path | None:
    if not directory.exists():
        return None

    for pattern in patterns:
        if "*" in pattern:
            hits = sorted(
                p for p in directory.glob(pattern)
                if p.is_file() and p.stat().st_size > 0
            )
            if hits:
                return hits[0]
        else:
            p = directory / pattern
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                return p

    return None


def join_unique(values, sep=";") -> str:
    seen = []
    for value in values:
        if pd.isna(value):
            continue
        for token in str(value).split(";"):
            token = token.strip()
            if token and token.lower() not in {"nan", "none"} and token not in seen:
                seen.append(token)
    return sep.join(seen)


def true_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def variant_key(chrom, pos, ref, alt) -> str | None:
    try:
        chrom = re.sub(r"^chr", "", str(chrom).strip(), flags=re.I)
        chrom = str(int(float(chrom)))
        pos = int(float(pos))
    except Exception:
        return None

    ref = str(ref).strip().upper()
    alt = str(alt).strip().upper()

    if not ref or not alt or ref in {".", "NA", "NAN"} or alt in {".", "NA", "NAN"}:
        return None

    return f"{chrom}:{pos}:{ref}:{alt}"


def parse_variant_key(key: str):
    parts = str(key).split(":")
    if len(parts) != 4:
        return None
    try:
        chrom = int(parts[0])
        pos = int(parts[1])
    except Exception:
        return None
    return chrom, pos, parts[2], parts[3]


def flatten_json(data: dict, prefix="") -> list[tuple[str, object]]:
    rows = []

    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            rows.extend(flatten_json(value, name))
        elif isinstance(value, list):
            rows.append((name, ";".join(map(str, value))))
        else:
            rows.append((name, value))

    return rows


def print_json_summary(title: str, path: Path | None) -> None:
    subsection(title)

    if path is None:
        print("(summary file not found)")
        return

    print(f"File: {path}")
    data = safe_json(path)

    if not data:
        print("(empty or unreadable JSON)")
        return

    df = pd.DataFrame(flatten_json(data), columns=["METRIC", "VALUE"])
    print_table(df)


# =============================================================================
# PROJECT PATHS
# =============================================================================

def project_roots(root: Path, phenotype_slug: str, ancestry_slug: str) -> dict[str, Path]:
    return {
        "STEP02": root / "02_summary_stats" / phenotype_slug / ancestry_slug,
        "STEP05": root / "05_ld_clumping" / phenotype_slug / ancestry_slug,
        "STEP06": root / "06_finemapping" / phenotype_slug / ancestry_slug,
        "STEP07": root / "07_annotation" / phenotype_slug / ancestry_slug,
        "STEP08": root / "08_qtl" / phenotype_slug / ancestry_slug,
        "STEP09": root / "09_splicing" / phenotype_slug / ancestry_slug,
        "STEP10": root / "10_pangolin" / phenotype_slug / ancestry_slug,
        "STEP11": root / "11_coloc" / phenotype_slug / ancestry_slug,
    }


def discover_studies(roots: dict[str, Path]) -> list[str]:
    studies = set()

    for base in roots.values():
        if not base.exists():
            continue

        for path in base.rglob("*"):
            match = re.search(r"(GCST\d+)", path.name)
            if match:
                studies.add(match.group(1))

    return sorted(studies)


def study_paths(roots: dict[str, Path], study: str) -> dict[str, Path]:
    return {
        "STEP03": roots["STEP02"] / "qc" / study,
        "STEP05": roots["STEP05"] / study,
        "STEP06": roots["STEP06"] / study,
        "STEP07": roots["STEP07"] / study,
        "STEP08": roots["STEP08"] / study,
        "STEP09": roots["STEP09"] / study,
        "STEP10": roots["STEP10"] / study,
        "STEP11": roots["STEP11"] / study,
    }


# =============================================================================
# STEP01-10 RUN STATUS (written by Step01_10_Run.py; only read here)
# =============================================================================

def print_step01_10_status(
    root: Path,
    phenotype: str,
    ancestry_code: str,
    ancestry_label: str,
    phenotype_slug: str,
    ancestry_slug: str,
) -> bool:
    import gwas2m_status as gs

    audit = gs.audit_dir(root, phenotype_slug, ancestry_slug)

    section("GWAS2m PIPELINE STATUS")
    print(f"Phenotype: {phenotype}")
    print(f"Ancestry : {ancestry_label} ({ancestry_code})")
    print(f"Audit    : {audit}")

    statuses = gs.read_all_statuses(audit)
    if statuses.empty:
        print("\n(no Step01_10_Run.py status files found for this phenotype/ancestry)")
        return False
    statuses = statuses.astype(object).where(statuses.notna(), None)

    by_key = {
        (row["study_accession"], row["stage"]): row
        for _, row in statuses.iterrows()
    }

    manifest = safe_table(audit / "study_manifest.tsv")
    if not manifest.empty:
        studies = manifest["STUDY_ACCESSION"].astype(str).tolist()
    else:
        studies = sorted(
            statuses.loc[statuses["stage"] != "Step01_Discovery", "study_accession"].unique()
        )

    study_stages = gs.STAGE_NAMES[1:]
    width = max([len("STUDY")] + [len(s) for s in studies]) + 3

    print()
    print("STUDY".ljust(width) + "".join(gs.STAGE_CODES[s].ljust(7) for s in study_stages))
    for study in studies:
        cells = []
        for stage in study_stages:
            record = by_key.get((study, stage))
            status = record["status"] if record is not None else ""
            cell = gs.STATUS_SHORT.get(status, "?") if status else ""
            cells.append(cell.ljust(7))
        print(study.ljust(width) + "".join(cells))

    print()
    print(
        "Legend: OK=complete/validated  PART=partial  FAIL=failed  EXCL=excluded  "
        "N/A=not available  --=not run (upstream failed)  RUN=running  PEND=pending"
    )

    reasons = []
    for study in studies:
        for stage in study_stages:
            record = by_key.get((study, stage))
            if record is None or record["status"] in gs.DONE_STATUSES | {gs.PENDING, gs.RUNNING}:
                continue
            error = ""
            if record.get("error_type"):
                error = f"{record.get('error_type')}: {record.get('error_message') or ''}".strip()
            reason = record.get("reason") or ""
            reasons.append({
                "STUDY": study,
                "STAGE": stage,
                "STATUS": record["status"],
                "REASON": reason,
                "ERROR": "" if error == reason else error,
                "LOG": record.get("log_path") or "",
            })
    print_table(pd.DataFrame(reasons), "EXACT REASONS: FAILED / PARTIAL / EXCLUDED / NOT_AVAILABLE / NOT_RUN")

    discovery = statuses[statuses["stage"] == "Step01_Discovery"]
    excluded = discovery[discovery["status"] == gs.EXCLUDED]
    if not discovery.empty:
        print_table(
            excluded.groupby("reason").size().reset_index(name="N_STUDIES")
            if not excluded.empty else pd.DataFrame(),
            f"STEP01 DISCOVERY: {len(discovery)} studies considered, "
            f"{int((discovery['status'] == gs.COMPLETE).sum())} selected, "
            f"{len(excluded)} excluded (by reason)",
        )

    summary = safe_table(audit / "study_summary.tsv")
    if not summary.empty:
        print_table(summary[["METRIC", "VALUE"]], "STUDY SUMMARY (pipeline_audit/study_summary.tsv)")

    return True


# =============================================================================
# STEP SUMMARIES
# =============================================================================

def step03_metrics(directory: Path, study: str) -> dict:
    qc = first_existing(
        directory,
        [f"{study}_GRCh38_QC.tsv.gz", f"{study}_GRCh38_QC.tsv"],
    )
    sig = first_existing(
        directory,
        [f"{study}_GRCh38_significant.tsv.gz", f"{study}_GRCh38_significant.tsv"],
    )

    row = {
        "STATUS": "PRESENT" if qc else "MISSING",
        "N_VARIANTS_AFTER_QC": np.nan,
        "N_GWS_QC": np.nan,
    }

    if qc:
        row["N_VARIANTS_AFTER_QC"] = len(safe_table(qc))
    if sig:
        row["N_GWS_QC"] = len(safe_table(sig))

    return row


def step05_metrics(directory: Path, study: str) -> dict:
    leads = first_existing(
        directory,
        ["lead_variants.tsv", f"{study}_lead_variants.tsv", "*lead*variant*.tsv"],
    )
    mapped = first_existing(
        directory,
        [
            "mapped_candidates.tsv.gz",
            f"{study}_mapped_candidates.tsv.gz",
            "*mapped*candidate*.tsv*",
        ],
    )

    return {
        "STATUS": "PRESENT" if directory.exists() else "MISSING",
        "N_MAPPED_CANDIDATES": len(safe_table(mapped)) if mapped else np.nan,
        "N_LEAD_VARIANTS": len(safe_table(leads)) if leads else np.nan,
    }


def summary_json_for(step: str, directory: Path) -> Path | None:
    patterns = {
        "STEP06": ["finemapping_summary.json"],
        "STEP07": ["annotation_summary.json", "vep_summary.json"],
        "STEP08": ["qtl_summary.json"],
        "STEP09": ["spliceai_summary.json"],
        "STEP10": ["pangolin_summary.json"],
        "STEP11": ["coloc_summary.json"],
    }
    return first_existing(directory, patterns.get(step, []))


def step_json_metrics(step: str, directory: Path) -> dict:
    path = summary_json_for(step, directory)
    data = safe_json(path)
    if not data:
        return {"STATUS": "MISSING"}

    status = str(data.get("STATUS", "PRESENT"))
    result = {"STATUS": status}

    wanted = {
        "STEP06": [
            "N_LOCI_TOTAL", "N_LOCI_SUCCESS", "N_LOCI_FAILED",
            "N_FINEMAPPED_VARIANTS", "N_95PCT_CS_ROWS", "MAX_PIP",
        ],
        "STEP07": [
            "N_FINEMAPPED_ROWS_AVAILABLE", "N_SELECTED_VARIANTS",
            "N_ANNOTATED_VARIANTS", "N_SPLICING", "N_CODING",
            "N_REGULATORY", "N_NONCODING_RNA",
        ],
        "STEP08": [
            "N_INPUT_VARIANTS", "N_EQTL_VARIANTS", "N_SQTL_VARIANTS",
            "N_EQTL_AND_SQTL", "N_EQTL_ASSOCIATIONS", "N_SQTL_ASSOCIATIONS",
        ],
        "STEP09": [
            "N_INPUT_VARIANTS", "N_SPLICEAI_SCORED",
            "N_SPLICEAI_GE_0_20", "N_SPLICEAI_GE_0_50",
            "N_SPLICEAI_GE_0_80",
        ],
        "STEP10": [
            "N_INPUT_VARIANTS", "N_PANGOLIN_SCORED",
            "N_PANGOLIN_UNSCORED", "PANGOLIN_COVERAGE_PERCENT",
            "N_PANGOLIN_TOP_10PCT", "N_PANGOLIN_TOP_5PCT",
            "N_PANGOLIN_TOP_1PCT", "N_MULTI_SOURCE_GE_2",
        ],
        "STEP11": [
            "N_GWAS_FINE_MAPPED_VARIANTS", "N_GWAS_LOCI",
            "N_VARIANT_OVERLAP_ROWS", "N_UNIQUE_SHARED_VARIANTS",
            "N_COLOC_LOCI", "N_GENE_TISSUE_PAIRS", "N_GENES",
            "N_TISSUES", "N_PAIR_LCLPP_GE_0_01",
            "N_PAIR_LCLPP_GE_0_001", "MAX_LCLPP",
        ],
    }

    for key in wanted.get(step, []):
        if key in data:
            result[key] = data[key]

    return result


def build_master(studies: list[str], roots: dict[str, Path]) -> pd.DataFrame:
    rows = []

    for study in studies:
        sp = study_paths(roots, study)

        row = {"STUDY_ACCESSION": study}

        for prefix, data in [
            ("STEP03", step03_metrics(sp["STEP03"], study)),
            ("STEP05", step05_metrics(sp["STEP05"], study)),
            ("STEP06", step_json_metrics("STEP06", sp["STEP06"])),
            ("STEP07", step_json_metrics("STEP07", sp["STEP07"])),
            ("STEP08", step_json_metrics("STEP08", sp["STEP08"])),
            ("STEP09", step_json_metrics("STEP09", sp["STEP09"])),
            ("STEP10", step_json_metrics("STEP10", sp["STEP10"])),
            ("STEP11", step_json_metrics("STEP11", sp["STEP11"])),
        ]:
            for key, value in data.items():
                row[f"{prefix}_{key}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# GENCODE GENOMIC CONTEXT
# =============================================================================

def locate_gencode_gtf(root: Path) -> Path:
    candidates = [
        root / "resources" / "gencode" / "release50"
        / "gencode.v50.primary_assembly.annotation.gtf.gz",
        root / "resources" / "gencode" / "release50"
        / "gencode.v50.primary_assembly.annotation.gtf",
        root / "resources" / "gtex" / "v11" / "reference"
        / "gencode.v47.genes.gtf",
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path

    raise FileNotFoundError(
        "Could not find a GENCODE GTF. Checked:\n  "
        + "\n  ".join(map(str, candidates))
    )


def parse_gtf_attributes(text: str) -> dict[str, str]:
    attrs = {}
    for match in re.finditer(r'(\S+)\s+"([^"]*)"', text):
        attrs[match.group(1)] = match.group(2)
    return attrs


def biotype_classes(gene_types: set[str]) -> list[str]:
    classes = set()

    lnc_tokens = {
        "lncrna", "lincrna", "antisense", "processed_transcript",
        "sense_intronic", "sense_overlapping", "macro_lncrna",
        "bidirectional_promoter_lncrna", "3prime_overlapping_ncrna",
    }

    other_nc_tokens = {
        "mirna", "snrna", "snorna", "rrna", "scrna", "vault_rna",
        "misc_rna", "ribozyme", "srna", "trna", "mt_trna", "mt_rrna",
    }

    for raw in gene_types:
        gt = str(raw).strip().lower()

        if gt == "protein_coding":
            classes.add("PROTEIN_CODING")
        elif any(token in gt for token in lnc_tokens):
            classes.add("LNCRNA")
        elif any(token in gt for token in other_nc_tokens):
            classes.add("OTHER_NCRNA")
        elif gt:
            classes.add("OTHER_GENE_TYPE")

    if not classes:
        classes.add("NO_GENE")

    order = [
        "PROTEIN_CODING",
        "LNCRNA",
        "OTHER_NCRNA",
        "OTHER_GENE_TYPE",
        "NO_GENE",
    ]

    return [x for x in order if x in classes]


def classify_context(
    features: set[str],
    gene_types: set[str],
    gene_names: set[str],
) -> tuple[str, str, str]:
    classes = biotype_classes(gene_types)

    has_protein = "PROTEIN_CODING" in classes
    has_lnc = "LNCRNA" in classes
    has_other_nc = "OTHER_NCRNA" in classes

    if "CDS" in features and has_protein:
        context = "CODING_CDS"
        coding = "CODING"

    elif "UTR" in features and has_protein:
        context = "UTR_PROTEIN_CODING_GENE"
        coding = "NONCODING"

    elif "exon" in features:
        if has_lnc:
            context = "LNCRNA_EXON"
        elif has_other_nc:
            context = "OTHER_NCRNA_EXON"
        elif has_protein:
            context = "NON_CDS_EXON_PROTEIN_CODING_GENE"
        else:
            context = "EXON_OTHER_GENE"
        coding = "NONCODING"

    elif gene_names or gene_types:
        if has_lnc:
            context = "LNCRNA_GENE_BODY_INTRONIC_OR_OTHER"
        elif has_other_nc:
            context = "OTHER_NCRNA_GENE_BODY"
        elif has_protein:
            context = "INTRONIC_PROTEIN_CODING_GENE"
        else:
            context = "OTHER_GENE_BODY"
        coding = "NONCODING"

    else:
        context = "INTERGENIC"
        coding = "NONCODING"

    return context, coding, ";".join(classes)


def annotate_gencode_context(
    variant_keys: set[str],
    gtf: Path,
) -> pd.DataFrame:
    """
    Scan GENCODE once and map all requested variant positions.

    This does not perform VEP consequence prediction.  It gives genomic
    context for every requested position from GENCODE gene/exon/CDS/UTR
    intervals.  Exact VEP consequences are overlaid separately when Step07
    annotations are available.
    """

    parsed = {}

    for key in variant_keys:
        item = parse_variant_key(key)
        if item is not None:
            parsed[key] = item

    by_chr_pos: dict[int, dict[int, list[str]]] = {}

    for key, (chrom, pos, ref, alt) in parsed.items():
        by_chr_pos.setdefault(chrom, {}).setdefault(pos, []).append(key)

    sorted_positions = {
        chrom: sorted(pos_map)
        for chrom, pos_map in by_chr_pos.items()
    }

    context = {
        key: {
            "GENES": set(),
            "GENE_IDS": set(),
            "GENE_TYPES": set(),
            "FEATURES": set(),
        }
        for key in parsed
    }

    opener = gzip.open if str(gtf).endswith(".gz") else open

    print(f"\nScanning GENCODE once for {len(parsed):,} unique colocalized variants:")
    print(f"  {gtf}")

    n_lines = 0

    with opener(gtf, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue

            n_lines += 1
            fields = line.rstrip("\n").split("\t")

            if len(fields) < 9:
                continue

            chrom_text, _, feature, start, end, _, _, _, attrs_text = fields

            if feature not in {"gene", "exon", "CDS", "UTR"}:
                continue

            chrom_clean = re.sub(r"^chr", "", chrom_text, flags=re.I)

            try:
                chrom = int(chrom_clean)
                start = int(start)
                end = int(end)
            except Exception:
                continue

            if chrom not in sorted_positions:
                continue

            positions = sorted_positions[chrom]
            left = bisect.bisect_left(positions, start)
            right = bisect.bisect_right(positions, end)

            if left >= right:
                continue

            attrs = parse_gtf_attributes(attrs_text)

            gene_name = (
                attrs.get("gene_name")
                or attrs.get("gene")
                or attrs.get("gene_id")
                or ""
            )

            gene_id = attrs.get("gene_id", "")

            gene_type = (
                attrs.get("gene_type")
                or attrs.get("gene_biotype")
                or attrs.get("transcript_type")
                or ""
            )

            for pos in positions[left:right]:
                for key in by_chr_pos[chrom][pos]:
                    item = context[key]

                    if gene_name:
                        item["GENES"].add(gene_name)

                    if gene_id:
                        item["GENE_IDS"].add(gene_id)

                    if gene_type:
                        item["GENE_TYPES"].add(gene_type)

                    item["FEATURES"].add(feature)

    rows = []

    for key, item in context.items():
        chrom, pos, ref, alt = parsed[key]

        genomic_context, coding_status, gene_class = classify_context(
            item["FEATURES"],
            item["GENE_TYPES"],
            item["GENES"],
        )

        rows.append({
            "VARIANT_KEY": key,
            "CHR": chrom,
            "POS": pos,
            "REF": ref,
            "ALT": alt,
            "GENOMIC_CONTEXT": genomic_context,
            "CODING_STATUS": coding_status,
            "GENE_BIOTYPE_CLASS": gene_class,
            "GENCODE_GENES": ";".join(sorted(item["GENES"])),
            "GENCODE_GENE_IDS": ";".join(sorted(item["GENE_IDS"])),
            "GENCODE_GENE_TYPES": ";".join(sorted(item["GENE_TYPES"])),
            "GENCODE_FEATURES": ";".join(sorted(item["FEATURES"])),
        })

    return pd.DataFrame(rows)


# =============================================================================
# STEP07 / STEP10 OVERLAYS
# =============================================================================

def load_vep_overlay(directory: Path, study: str) -> pd.DataFrame:
    path = first_existing(
        directory,
        [
            f"{study}_VEP_variant_summary.tsv",
            f"{study}_VEP_variant_summary.tsv.gz",
        ],
    )

    df = safe_table(path)

    if df.empty:
        return pd.DataFrame(columns=["VARIANT_KEY"])

    required = {"CHR", "REFERENCE_POS", "REFERENCE_REF", "REFERENCE_ALT"}

    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["VARIANT_KEY"])

    df = df.copy()

    df["VARIANT_KEY"] = [
        variant_key(c, p, r, a)
        for c, p, r, a in zip(
            df["CHR"],
            df["REFERENCE_POS"],
            df["REFERENCE_REF"],
            df["REFERENCE_ALT"],
        )
    ]

    wanted = [
        "VARIANT_KEY",
        "FUNCTIONAL_CATEGORY",
        "Consequence",
        "IMPACT",
        "SYMBOL",
        "Gene",
        "BIOTYPE",
        "ALL_CONSEQUENCES",
        "ALL_SYMBOLS",
        "ALL_BIOTYPES",
        "ALL_FUNCTIONAL_CATEGORIES",
    ]

    wanted = [c for c in wanted if c in df.columns]

    return (
        df[wanted]
        .dropna(subset=["VARIANT_KEY"])
        .drop_duplicates("VARIANT_KEY", keep="first")
    )


def load_step10_overlay(directory: Path, study: str) -> pd.DataFrame:
    path = first_existing(
        directory,
        [f"{study}_SpliceAI_Pangolin_integrated.tsv"],
    )

    df = safe_table(path)

    if df.empty:
        return pd.DataFrame(columns=["VARIANT_KEY"])

    if "VARIANT_KEY" not in df.columns:
        required = {"CHR", "REFERENCE_POS", "REFERENCE_REF", "REFERENCE_ALT"}

        if required.issubset(df.columns):
            df = df.copy()
            df["VARIANT_KEY"] = [
                variant_key(c, p, r, a)
                for c, p, r, a in zip(
                    df["CHR"],
                    df["REFERENCE_POS"],
                    df["REFERENCE_REF"],
                    df["REFERENCE_ALT"],
                )
            ]
        else:
            return pd.DataFrame(columns=["VARIANT_KEY"])

    wanted = [
        "VARIANT_KEY",
        "SPLICEAI_MAX_DS",
        "SPLICEAI_MAX_EVENT",
        "SPLICEAI_GENE",
        "PANGOLIN_GENE",
        "PANGOLIN_ALL_GENES",
        "PANGOLIN_MAX_ABS",
        "PANGOLIN_MAX_SIGNED",
        "PANGOLIN_MAX_EVENT",
        "PANGOLIN_PERCENTILE",
        "PANGOLIN_SCORED",
        "N_SPLICE_EVIDENCE_SOURCES",
        "SPLICE_EVIDENCE_COMBINATION",
    ]

    wanted = [c for c in wanted if c in df.columns]

    return (
        df[wanted]
        .dropna(subset=["VARIANT_KEY"])
        .drop_duplicates("VARIANT_KEY", keep="first")
    )


# =============================================================================
# STEP11 UNIQUE VARIANT TABLE
# =============================================================================

def load_step11_overlap(directory: Path, study: str) -> pd.DataFrame:
    path = first_existing(
        directory,
        [f"{study}_GTEx_SuSiE_variant_colocalization.tsv.gz"],
    )
    return safe_table(path)


def aggregate_colocalized_variants(
    overlap: pd.DataFrame,
    study: str,
) -> pd.DataFrame:
    if overlap.empty or "VARIANT_KEY" not in overlap.columns:
        return pd.DataFrame()

    x = overlap.copy()

    for col in ["GWAS_PIP", "GTEX_QTL_PIP", "VCLPP"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    rows = []

    for key, g in x.groupby("VARIANT_KEY", dropna=False, sort=False):
        qtl_type = (
            g["QTL_TYPE"].fillna("").astype(str)
            if "QTL_TYPE" in g.columns
            else pd.Series([], dtype=str)
        )

        top_idx = None
        if "VCLPP" in g.columns and g["VCLPP"].notna().any():
            top_idx = g["VCLPP"].idxmax()

        top = g.loc[top_idx] if top_idx is not None else g.iloc[0]

        eqtl = qtl_type.str.lower().eq("eqtl")
        sqtl = qtl_type.str.lower().eq("sqtl")

        rows.append({
            "STUDY_ACCESSION": study,
            "VARIANT_KEY": key,
            "LOCUS_IDS": join_unique(g.get("LOCUS_ID", pd.Series(dtype=str))),
            "GWAS_PIP_MAX": (
                g["GWAS_PIP"].max()
                if "GWAS_PIP" in g.columns
                else np.nan
            ),
            "GTEX_QTL_PIP_MAX": (
                g["GTEX_QTL_PIP"].max()
                if "GTEX_QTL_PIP" in g.columns
                else np.nan
            ),
            "VCLPP_MAX": (
                g["VCLPP"].max()
                if "VCLPP" in g.columns
                else np.nan
            ),
            "HAS_EQTL": bool(eqtl.any()) if len(eqtl) else False,
            "HAS_SQTL": bool(sqtl.any()) if len(sqtl) else False,
            "N_EQTL_ROWS": int(eqtl.sum()) if len(eqtl) else 0,
            "N_SQTL_ROWS": int(sqtl.sum()) if len(sqtl) else 0,
            "N_TISSUES": (
                g["TISSUE"].nunique()
                if "TISSUE" in g.columns
                else 0
            ),
            "TISSUES": join_unique(g.get("TISSUE", pd.Series(dtype=str))),
            "N_QTL_GENES": (
                g["QTL_GENE"].nunique()
                if "QTL_GENE" in g.columns
                else 0
            ),
            "QTL_GENES": join_unique(g.get("QTL_GENE", pd.Series(dtype=str))),
            "BEST_QTL_TYPE": top.get("QTL_TYPE", ""),
            "BEST_TISSUE": top.get("TISSUE", ""),
            "BEST_QTL_GENE": top.get("QTL_GENE", ""),
            "BEST_QTL_PHENOTYPE": top.get("QTL_PHENOTYPE", ""),
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["VCLPP_MAX", "GWAS_PIP_MAX", "GTEX_QTL_PIP_MAX"],
            ascending=[False, False, False],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)

    return out


def attach_variant_annotations(
    variants: pd.DataFrame,
    gencode: pd.DataFrame,
    vep: pd.DataFrame,
    step10: pd.DataFrame,
) -> pd.DataFrame:
    if variants.empty:
        return variants

    out = variants.merge(
        gencode,
        on="VARIANT_KEY",
        how="left",
        validate="many_to_one",
    )

    if not vep.empty:
        rename = {
            c: f"VEP_{c}"
            for c in vep.columns
            if c != "VARIANT_KEY"
        }
        out = out.merge(
            vep.rename(columns=rename),
            on="VARIANT_KEY",
            how="left",
            validate="many_to_one",
        )

    if not step10.empty:
        rename = {
            c: f"STEP10_{c}"
            for c in step10.columns
            if c != "VARIANT_KEY"
        }
        out = out.merge(
            step10.rename(columns=rename),
            on="VARIANT_KEY",
            how="left",
            validate="many_to_one",
        )

    return out


# =============================================================================
# CONTEXT SUMMARIES
# =============================================================================

def genomic_context_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "GENOMIC_CONTEXT" not in df.columns:
        return pd.DataFrame()

    total = df["VARIANT_KEY"].nunique()

    result = (
        df.drop_duplicates("VARIANT_KEY")
        .groupby("GENOMIC_CONTEXT", dropna=False)
        .agg(N_VARIANTS=("VARIANT_KEY", "nunique"))
        .reset_index()
    )

    result["PERCENT"] = (
        100.0 * result["N_VARIANTS"] / total
        if total
        else np.nan
    )

    return result.sort_values(
        ["N_VARIANTS", "GENOMIC_CONTEXT"],
        ascending=[False, True],
    ).reset_index(drop=True)


def coding_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "CODING_STATUS" not in df.columns:
        return pd.DataFrame()

    total = df["VARIANT_KEY"].nunique()

    result = (
        df.drop_duplicates("VARIANT_KEY")
        .groupby("CODING_STATUS", dropna=False)
        .agg(N_VARIANTS=("VARIANT_KEY", "nunique"))
        .reset_index()
    )

    result["PERCENT"] = (
        100.0 * result["N_VARIANTS"] / total
        if total
        else np.nan
    )

    return result.sort_values(
        "N_VARIANTS",
        ascending=False,
    ).reset_index(drop=True)


def gene_biotype_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "GENE_BIOTYPE_CLASS" not in df.columns:
        return pd.DataFrame()

    rows = []

    for _, row in df.drop_duplicates("VARIANT_KEY").iterrows():
        value = str(row.get("GENE_BIOTYPE_CLASS", "")).strip()

        if not value or value.lower() == "nan":
            value = "NO_GENE"

        for token in value.split(";"):
            token = token.strip()
            if token:
                rows.append({
                    "VARIANT_KEY": row["VARIANT_KEY"],
                    "GENE_BIOTYPE_CLASS": token,
                })

    if not rows:
        return pd.DataFrame()

    x = pd.DataFrame(rows)

    result = (
        x.groupby("GENE_BIOTYPE_CLASS", as_index=False)
        .agg(N_VARIANTS=("VARIANT_KEY", "nunique"))
    )

    total = df["VARIANT_KEY"].nunique()

    result["PERCENT_OF_UNIQUE_VARIANTS"] = (
        100.0 * result["N_VARIANTS"] / total
        if total
        else np.nan
    )

    return result.sort_values(
        "N_VARIANTS",
        ascending=False,
    ).reset_index(drop=True)


def qtl_mechanism_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df.drop_duplicates("VARIANT_KEY").copy()

    def label(row):
        eq = bool(row.get("HAS_EQTL", False))
        sq = bool(row.get("HAS_SQTL", False))

        if eq and sq:
            return "eQTL+sQTL"
        if eq:
            return "eQTL_ONLY"
        if sq:
            return "sQTL_ONLY"
        return "NO_QTL_TYPE"

    x["QTL_MECHANISM"] = x.apply(label, axis=1)

    total = len(x)

    result = (
        x.groupby("QTL_MECHANISM", as_index=False)
        .agg(N_VARIANTS=("VARIANT_KEY", "nunique"))
    )

    result["PERCENT"] = (
        100.0 * result["N_VARIANTS"] / total
        if total
        else np.nan
    )

    return result.sort_values(
        "N_VARIANTS",
        ascending=False,
    ).reset_index(drop=True)


def vep_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    vep_col = "VEP_FUNCTIONAL_CATEGORY"

    if vep_col not in df.columns:
        return pd.DataFrame({
            "METRIC": [
                "UNIQUE_COLOCALIZED_VARIANTS",
                "WITH_STEP07_VEP_ANNOTATION",
                "WITHOUT_STEP07_VEP_ANNOTATION",
            ],
            "VALUE": [
                df["VARIANT_KEY"].nunique(),
                0,
                df["VARIANT_KEY"].nunique(),
            ],
        })

    x = df.drop_duplicates("VARIANT_KEY")
    annotated = x[vep_col].notna() & x[vep_col].astype(str).str.strip().ne("")

    return pd.DataFrame({
        "METRIC": [
            "UNIQUE_COLOCALIZED_VARIANTS",
            "WITH_STEP07_VEP_ANNOTATION",
            "WITHOUT_STEP07_VEP_ANNOTATION",
            "STEP07_VEP_COVERAGE_PERCENT",
        ],
        "VALUE": [
            len(x),
            int(annotated.sum()),
            int((~annotated).sum()),
            (100.0 * annotated.sum() / len(x)) if len(x) else np.nan,
        ],
    })


# =============================================================================
# KEY SAVED TABLES PER STUDY
# =============================================================================

def print_key_step_tables(sp: dict[str, Path], study: str, full_pairs: bool) -> None:
    # Step05 leads
    leads = first_existing(
        sp["STEP05"],
        ["lead_variants.tsv", f"{study}_lead_variants.tsv"],
    )
    if leads:
        print_table(
            safe_table(leads),
            "STEP 05 LEAD VARIANTS",
        )

    # Step06 credible set
    credible = first_existing(
        sp["STEP06"],
        [
            f"{study}_95pct_credible_sets.tsv",
            f"{study}_95pct_credible_sets.tsv.gz",
        ],
    )
    if credible:
        print_table(
            safe_table(credible),
            "STEP 06 95% CREDIBLE-SET ROWS",
        )

    # Step07 category/locus summaries
    for title, patterns in [
        (
            "STEP 07 VEP CATEGORY SUMMARY",
            [f"{study}_VEP_category_summary.tsv", "*VEP*category*summary*.tsv"],
        ),
        (
            "STEP 07 VEP LOCUS SUMMARY",
            [f"{study}_VEP_locus_summary.tsv", "*VEP*locus*summary*.tsv"],
        ),
        (
            "STEP 08 GTEx QTL LOCUS SUMMARY",
            [f"{study}_GTEx_QTL_locus_summary.tsv"],
        ),
        (
            "STEP 09 SPLICEAI CATEGORY SUMMARY",
            [f"{study}_SpliceAI_category_summary.tsv"],
        ),
        (
            "STEP 09 SPLICEAI LOCUS SUMMARY",
            [f"{study}_SpliceAI_locus_summary.tsv"],
        ),
        (
            "STEP 10 PANGOLIN LOCUS SUMMARY",
            [f"{study}_Pangolin_locus_summary.tsv"],
        ),
        (
            "STEP 11 COLOCALIZATION LOCUS SUMMARY",
            [f"{study}_GTEx_SuSiE_locus_summary.tsv"],
        ),
        (
            "STEP 11 COLOCALIZATION GENE SUMMARY",
            [f"{study}_GTEx_SuSiE_gene_summary.tsv"],
        ),
    ]:
        if title.startswith("STEP 07"):
            base = sp["STEP07"]
        elif title.startswith("STEP 08"):
            base = sp["STEP08"]
        elif title.startswith("STEP 09"):
            base = sp["STEP09"]
        elif title.startswith("STEP 10"):
            base = sp["STEP10"]
        else:
            base = sp["STEP11"]

        path = first_existing(base, patterns)

        if path:
            print_table(
                safe_table(path),
                title,
            )

    if full_pairs:
        pair = first_existing(
            sp["STEP11"],
            [f"{study}_GTEx_SuSiE_gene_tissue_colocalization.tsv"],
        )

        if pair:
            print_table(
                safe_table(pair),
                "STEP 11 ALL GENE × TISSUE × QTL COLOCALIZATION PAIRS",
            )


# =============================================================================
# WARNINGS / FAILURES
# =============================================================================

def failure_markers_for_study(root: Path, study: str) -> pd.DataFrame:
    rows = []

    for path in root.rglob("*FAILED*.txt"):
        if not path.is_file():
            continue

        if study not in str(path):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            text = ""

        rows.append({
            "FILE": str(path),
            "FIRST_500_CHARS": re.sub(r"\s+", " ", text[:500]).strip(),
        })

    return pd.DataFrame(rows)


# =============================================================================
# COMBINED ANALYSIS
# =============================================================================

def combined_unique_variants(
    annotated_by_study: list[pd.DataFrame],
) -> pd.DataFrame:
    pieces = [x for x in annotated_by_study if x is not None and not x.empty]

    if not pieces:
        return pd.DataFrame()

    all_rows = pd.concat(pieces, ignore_index=True, sort=False)

    rows = []

    for key, g in all_rows.groupby("VARIANT_KEY", dropna=False, sort=False):
        top_idx = None

        if "VCLPP_MAX" in g.columns:
            values = pd.to_numeric(g["VCLPP_MAX"], errors="coerce")
            if values.notna().any():
                top_idx = values.idxmax()

        top = g.loc[top_idx] if top_idx is not None else g.iloc[0]

        rows.append({
            "VARIANT_KEY": key,
            "N_STUDIES": g["STUDY_ACCESSION"].nunique(),
            "STUDIES": join_unique(g["STUDY_ACCESSION"]),
            "GWAS_PIP_MAX": pd.to_numeric(
                g.get("GWAS_PIP_MAX", pd.Series(dtype=float)),
                errors="coerce",
            ).max(),
            "GTEX_QTL_PIP_MAX": pd.to_numeric(
                g.get("GTEX_QTL_PIP_MAX", pd.Series(dtype=float)),
                errors="coerce",
            ).max(),
            "VCLPP_MAX": pd.to_numeric(
                g.get("VCLPP_MAX", pd.Series(dtype=float)),
                errors="coerce",
            ).max(),
            "HAS_EQTL": bool(g.get("HAS_EQTL", pd.Series(False)).fillna(False).astype(bool).any()),
            "HAS_SQTL": bool(g.get("HAS_SQTL", pd.Series(False)).fillna(False).astype(bool).any()),
            "N_TISSUES_UNION": len(
                set(
                    token
                    for value in g.get("TISSUES", pd.Series(dtype=str)).dropna().astype(str)
                    for token in value.split(";")
                    if token
                )
            ),
            "TISSUES_UNION": join_unique(g.get("TISSUES", pd.Series(dtype=str))),
            "N_QTL_GENES_UNION": len(
                set(
                    token
                    for value in g.get("QTL_GENES", pd.Series(dtype=str)).dropna().astype(str)
                    for token in value.split(";")
                    if token
                )
            ),
            "QTL_GENES_UNION": join_unique(g.get("QTL_GENES", pd.Series(dtype=str))),
            "GENOMIC_CONTEXT": top.get("GENOMIC_CONTEXT", ""),
            "CODING_STATUS": top.get("CODING_STATUS", ""),
            "GENE_BIOTYPE_CLASS": top.get("GENE_BIOTYPE_CLASS", ""),
            "GENCODE_GENES": top.get("GENCODE_GENES", ""),
            "GENCODE_GENE_TYPES": top.get("GENCODE_GENE_TYPES", ""),
            "VEP_FUNCTIONAL_CATEGORY": top.get("VEP_FUNCTIONAL_CATEGORY", ""),
            "VEP_SYMBOL": top.get("VEP_SYMBOL", ""),
            "VEP_BIOTYPE": top.get("VEP_BIOTYPE", ""),
            "STEP10_SPLICEAI_MAX_DS": pd.to_numeric(
                g.get("STEP10_SPLICEAI_MAX_DS", pd.Series(dtype=float)),
                errors="coerce",
            ).max(),
            "STEP10_PANGOLIN_MAX_ABS": pd.to_numeric(
                g.get("STEP10_PANGOLIN_MAX_ABS", pd.Series(dtype=float)),
                errors="coerce",
            ).max(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["N_STUDIES", "VCLPP_MAX", "GWAS_PIP_MAX"],
            ascending=[False, False, False],
            na_position="last",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def combined_gene_table(
    overlaps_by_study: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    pieces = []

    for study, overlap in overlaps_by_study:
        if overlap.empty or "QTL_GENE" not in overlap.columns:
            continue

        x = overlap.copy()
        x["STUDY_ACCESSION"] = study
        pieces.append(x)

    if not pieces:
        return pd.DataFrame()

    all_rows = pd.concat(pieces, ignore_index=True, sort=False)

    result = (
        all_rows.groupby(
            ["QTL_TYPE", "QTL_GENE"],
            dropna=False,
            as_index=False,
        )
        .agg(
            N_STUDIES=("STUDY_ACCESSION", "nunique"),
            N_VARIANTS=("VARIANT_KEY", "nunique"),
            N_TISSUES=("TISSUE", "nunique"),
            MAX_VCLPP=("VCLPP", "max"),
            STUDIES=("STUDY_ACCESSION", join_unique),
        )
        .sort_values(
            ["N_STUDIES", "MAX_VCLPP", "N_VARIANTS"],
            ascending=[False, False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return result


def combined_tissue_table(
    overlaps_by_study: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    pieces = []

    for study, overlap in overlaps_by_study:
        if overlap.empty or "TISSUE" not in overlap.columns:
            continue

        x = overlap.copy()
        x["STUDY_ACCESSION"] = study
        pieces.append(x)

    if not pieces:
        return pd.DataFrame()

    all_rows = pd.concat(pieces, ignore_index=True, sort=False)

    result = (
        all_rows.groupby(
            ["QTL_TYPE", "TISSUE"],
            dropna=False,
            as_index=False,
        )
        .agg(
            N_STUDIES=("STUDY_ACCESSION", "nunique"),
            N_VARIANTS=("VARIANT_KEY", "nunique"),
            N_GENES=("QTL_GENE", "nunique"),
            MAX_VCLPP=("VCLPP", "max"),
            STUDIES=("STUDY_ACCESSION", join_unique),
        )
        .sort_values(
            ["N_STUDIES", "MAX_VCLPP", "N_VARIANTS"],
            ascending=[False, False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a read-only GWAS2m interpretation report to stdout."
    )

    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--ancestry", required=True)

    parser.add_argument(
        "--study",
        default=None,
        help="Optional single GCST accession.",
    )

    parser.add_argument(
        "--pairs",
        choices=["all", "none"],
        default="all",
        help=(
            "Whether to print every Step11 gene/tissue pair. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--skip-gencode",
        action="store_true",
        help=(
            "Skip the one-pass GENCODE scan. "
            "Not recommended if you need context for every colocalized variant."
        ),
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help=(
            "Print only the Step01-10 run status grid and exact failure / "
            "exclusion reasons written by Step01_10_Run.py."
        ),
    )

    args = parser.parse_args()

    root = Path.cwd().resolve()
    phenotype_slug = slugify(args.phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)

    if args.status:
        print_step01_10_status(
            root,
            args.phenotype,
            ancestry_code,
            ancestry_label,
            phenotype_slug,
            ancestry_slug,
        )
        return

    roots = project_roots(
        root,
        phenotype_slug,
        ancestry_slug,
    )

    studies = discover_studies(roots)

    if args.study:
        studies = [x for x in studies if x == args.study]

    if not studies:
        raise SystemExit(
            "No matching GCST studies found. "
            "Run this script from the GWAS2m project root."
        )

    section("GWAS2m TERMINAL-ONLY INTERPRETATION REPORT")

    print(f"Report version : {VERSION}")
    print(f"Project root   : {root}")
    print(f"Phenotype      : {args.phenotype}")
    print(f"Ancestry       : {ancestry_label} ({ancestry_code})")
    print(f"Studies found  : {len(studies)}")
    print(f"Studies        : {'; '.join(studies)}")
    print("Saving files   : NO")
    print("Output         : stdout / terminal only")
    print()
    print(
        "Counting rule  : genomic-context counts use UNIQUE Step11 VARIANT_KEY values, "
        "not repeated variant-gene-tissue rows."
    )

    # -------------------------------------------------------------------------
    # Master summary
    # -------------------------------------------------------------------------

    master = build_master(studies, roots)
    print_table(master, "MASTER PIPELINE SUMMARY - ALL STUDIES")

    # -------------------------------------------------------------------------
    # Load Step11 overlaps first so we can annotate all union variants once.
    # -------------------------------------------------------------------------

    overlap_by_study: list[tuple[str, pd.DataFrame]] = []
    unique_by_study_raw: dict[str, pd.DataFrame] = {}
    all_variant_keys: set[str] = set()

    for study in studies:
        sp = study_paths(roots, study)
        overlap = load_step11_overlap(sp["STEP11"], study)
        overlap_by_study.append((study, overlap))

        unique = aggregate_colocalized_variants(overlap, study)
        unique_by_study_raw[study] = unique

        if not unique.empty:
            all_variant_keys.update(
                unique["VARIANT_KEY"].dropna().astype(str)
            )

    # -------------------------------------------------------------------------
    # GENCODE context for every unique colocalized variant.
    # -------------------------------------------------------------------------

    if all_variant_keys and not args.skip_gencode:
        gtf = locate_gencode_gtf(root)
        gencode = annotate_gencode_context(all_variant_keys, gtf)
    else:
        gencode = pd.DataFrame(columns=["VARIANT_KEY"])

    # -------------------------------------------------------------------------
    # Individual-study analysis
    # -------------------------------------------------------------------------

    annotated_by_study = []

    for number, study in enumerate(studies, start=1):
        sp = study_paths(roots, study)

        section(
            f"INDIVIDUAL GWAS {number}/{len(studies)} - {study}"
        )

        # Full JSON summaries for steps that create them.
        print_table(
            pd.DataFrame([
                {
                    "STEP": "STEP03_QC",
                    **step03_metrics(sp["STEP03"], study),
                },
                {
                    "STEP": "STEP05_CLUMPING",
                    **step05_metrics(sp["STEP05"], study),
                },
                {
                    "STEP": "STEP06_FINEMAPPING",
                    **step_json_metrics("STEP06", sp["STEP06"]),
                },
                {
                    "STEP": "STEP07_VEP",
                    **step_json_metrics("STEP07", sp["STEP07"]),
                },
                {
                    "STEP": "STEP08_GTEX_QTL",
                    **step_json_metrics("STEP08", sp["STEP08"]),
                },
                {
                    "STEP": "STEP09_SPLICEAI",
                    **step_json_metrics("STEP09", sp["STEP09"]),
                },
                {
                    "STEP": "STEP10_PANGOLIN",
                    **step_json_metrics("STEP10", sp["STEP10"]),
                },
                {
                    "STEP": "STEP11_COLOCALIZATION",
                    **step_json_metrics("STEP11", sp["STEP11"]),
                },
            ]),
            "STEP-BY-STEP STATUS AND KEY METRICS",
        )

        for step, title in [
            ("STEP06", "STEP 06 FULL SAVED JSON SUMMARY"),
            ("STEP07", "STEP 07 FULL SAVED JSON SUMMARY"),
            ("STEP08", "STEP 08 FULL SAVED JSON SUMMARY"),
            ("STEP09", "STEP 09 FULL SAVED JSON SUMMARY"),
            ("STEP10", "STEP 10 FULL SAVED JSON SUMMARY"),
            ("STEP11", "STEP 11 FULL SAVED JSON SUMMARY"),
        ]:
            print_json_summary(
                title,
                summary_json_for(step, sp[step]),
            )

        print_key_step_tables(
            sp,
            study,
            full_pairs=(args.pairs == "all"),
        )

        failures = failure_markers_for_study(root, study)
        if not failures.empty:
            print_table(
                failures,
                "FAILURE MARKERS FOR THIS STUDY",
            )

        unique = unique_by_study_raw[study]

        if unique.empty:
            subsection("COLOCALIZED VARIANT GENOMIC CONTEXT")
            print("(Step11 unique colocalized variants are not available yet.)")
            continue

        vep = load_vep_overlay(sp["STEP07"], study)
        step10 = load_step10_overlay(sp["STEP10"], study)

        annotated = attach_variant_annotations(
            unique,
            gencode,
            vep,
            step10,
        )

        annotated_by_study.append(annotated)

        print_table(
            coding_summary(annotated),
            "UNIQUE COLOCALIZED VARIANTS - CODING VS NONCODING",
        )

        print_table(
            genomic_context_summary(annotated),
            "UNIQUE COLOCALIZED VARIANTS - GENCODE GENOMIC CONTEXT",
        )

        print_table(
            gene_biotype_summary(annotated),
            "UNIQUE COLOCALIZED VARIANTS - GENE BIOTYPE CONTEXT",
        )

        print_table(
            qtl_mechanism_summary(annotated),
            "UNIQUE COLOCALIZED VARIANTS - eQTL / sQTL MECHANISM",
        )

        print_table(
            vep_coverage_summary(annotated),
            "STEP07 VEP COVERAGE OF THE COLOCALIZED VARIANTS",
        )

        # VEP functional categories among the subset that Step07 annotated.
        if "VEP_FUNCTIONAL_CATEGORY" in annotated.columns:
            vep_context = (
                annotated.drop_duplicates("VARIANT_KEY")
                .dropna(subset=["VEP_FUNCTIONAL_CATEGORY"])
                .groupby("VEP_FUNCTIONAL_CATEGORY", dropna=False)
                .agg(N_VARIANTS=("VARIANT_KEY", "nunique"))
                .reset_index()
                .sort_values("N_VARIANTS", ascending=False)
            )

            print_table(
                vep_context,
                "VEP FUNCTIONAL CATEGORY - AVAILABLE STEP07 SUBSET",
            )

        # Full table: ALL unique Step11 colocalized variants.
        print_table(
            annotated,
            (
                "ALL UNIQUE COLOCALIZED VARIANTS FOR THIS GWAS "
                "- FULL GENOMIC / QTL / VEP / SPLICE EVIDENCE TABLE"
            ),
            all_rows=True,
        )

    # -------------------------------------------------------------------------
    # Combined analysis
    # -------------------------------------------------------------------------

    section(
        "COMBINED ANALYSIS ACROSS ALL GWAS FILES"
    )

    if not annotated_by_study:
        print(
            "No completed Step11 variant-colocalization files were available "
            "for combined genomic-context analysis."
        )
        return

    combined = combined_unique_variants(
        annotated_by_study
    )

    print_table(
        coding_summary(combined),
        "COMBINED UNIQUE VARIANTS - CODING VS NONCODING",
    )

    print_table(
        genomic_context_summary(combined),
        "COMBINED UNIQUE VARIANTS - GENCODE GENOMIC CONTEXT",
    )

    print_table(
        gene_biotype_summary(combined),
        "COMBINED UNIQUE VARIANTS - GENE BIOTYPE CONTEXT",
    )

    print_table(
        qtl_mechanism_summary(combined),
        "COMBINED UNIQUE VARIANTS - eQTL / sQTL MECHANISM",
    )

    common = combined[
        combined["N_STUDIES"] >= 2
    ].copy()

    print_table(
        common,
        "COLOCALIZED VARIANTS SHARED BY >=2 GWAS STUDIES",
    )

    genes = combined_gene_table(
        overlap_by_study
    )

    print_table(
        genes,
        "COMBINED COLOCALIZED QTL GENES ACROSS GWAS STUDIES",
    )

    print_table(
        genes[
            genes["N_STUDIES"] >= 2
        ].copy()
        if not genes.empty
        else genes,
        "COLOCALIZED QTL GENES REPEATED IN >=2 GWAS STUDIES",
    )

    tissues = combined_tissue_table(
        overlap_by_study
    )

    print_table(
        tissues,
        "COMBINED GTEx COLOCALIZATION TISSUES ACROSS GWAS STUDIES",
    )

    print_table(
        tissues[
            tissues["N_STUDIES"] >= 2
        ].copy()
        if not tissues.empty
        else tissues,
        "GTEx COLOCALIZATION TISSUES REPEATED IN >=2 GWAS STUDIES",
    )

    # Full final union table.
    print_table(
        combined,
        (
            "ALL UNIQUE COLOCALIZED VARIANTS ACROSS ALL GWAS "
            "- FINAL COMBINED TABLE"
        ),
        all_rows=True,
    )

    section("REPORT FINISHED")

    print("No report files were written.")
    print("Everything above was printed to stdout.")
    print()
    print(
        "Tip: if you want to pass the full terminal output back to ChatGPT "
        "without the shell truncating your copy buffer, you can still use "
        "normal shell redirection yourself, but this Python script itself "
        "does not save anything."
    )


if __name__ == "__main__":
    main()
 
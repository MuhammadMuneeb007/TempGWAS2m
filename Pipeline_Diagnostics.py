 
# -*- coding: utf-8 -*-

"""
Pipeline_Diagnostics.py

Cross-step audit for the generic GWAS pipeline.

It summarizes, per GWAS accession:

STEP 03 QC
    - variants before QC
    - variants after QC
    - genome-wide significant variants
    - minimum P
    - clumping readiness
    - SuSiE readiness
    - BETA / SE / Z availability

STEP 05 LD clumping
    - P <= p2 variants
    - variants mapped to ancestry-matched 1000G
    - genome-wide-significant variants mapped to 1000G
    - independent lead variants

STEP 06 fine-mapping
    - status
    - loci attempted/successful/failed
    - fine-mapped variants
    - 95% credible-set rows
    - max PIP
    - counts with PIP >= 0.01 / 0.10 / 0.50

It then assigns a diagnostic interpretation such as:
    NO_GWS_HITS
    GWS_HITS_NOT_MAPPED_TO_REFERENCE
    NO_INDEPENDENT_LEADS_AFTER_CLUMPING
    NOT_SUSIE_READY
    FINEMAPPING_COMPLETE
    FINEMAPPING_PARTIAL
    FINEMAPPING_FAILED
    WAITING_FOR_STEP_05
    WAITING_FOR_STEP_06

Usage:
    python Pipeline_Diagnostics.py --phenotype migraine --ancestry EUR

Optional:
    python Pipeline_Diagnostics.py \
        --phenotype migraine \
        --ancestry EUR \
        --show-all

Outputs:
    diagnostics/<phenotype>/<ancestry>/pipeline_diagnostics.tsv
    diagnostics/<phenotype>/<ancestry>/pipeline_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


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


def normalize_text(value: str) -> str:
    x = str(value).strip().lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def slugify(value: str) -> str:
    return normalize_text(value).replace(" ", "_")


def canonical_ancestry(value: str) -> tuple[str, str]:
    key = normalize_text(value)
    if key not in ANCESTRY_ALIASES:
        raise ValueError(
            f"Unsupported ancestry {value!r}. "
            "Use EUR, AFR, EAS, SAS, AMR or the corresponding full label."
        )
    return ANCESTRY_ALIASES[key]


def safe_read_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(
            path,
            sep="\t",
            compression="infer",
            low_memory=False,
        )
    except Exception:
        return None


def safe_read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def first_row_dict(path: Path) -> dict:
    df = safe_read_tsv(path)
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def numeric(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def integer(value):
    x = numeric(value)
    if x is None:
        return None
    return int(x)


def truthy(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def count_rows(path: Path) -> int | None:
    df = safe_read_tsv(path)
    if df is None:
        return None
    return int(len(df))


def pip_stats(path: Path) -> dict:
    result = {
        "N_FINEMAPPED_VARIANTS": None,
        "MAX_PIP": None,
        "N_PIP_GE_0_01": None,
        "N_PIP_GE_0_10": None,
        "N_PIP_GE_0_50": None,
    }

    df = safe_read_tsv(path)
    if df is None:
        return result

    result["N_FINEMAPPED_VARIANTS"] = int(len(df))

    if "PIP" not in df.columns or df.empty:
        if df.empty:
            result["MAX_PIP"] = None
            result["N_PIP_GE_0_01"] = 0
            result["N_PIP_GE_0_10"] = 0
            result["N_PIP_GE_0_50"] = 0
        return result

    pip = pd.to_numeric(df["PIP"], errors="coerce")
    finite = pip.dropna()

    result["MAX_PIP"] = float(finite.max()) if not finite.empty else None
    result["N_PIP_GE_0_01"] = int((pip >= 0.01).sum())
    result["N_PIP_GE_0_10"] = int((pip >= 0.10).sum())
    result["N_PIP_GE_0_50"] = int((pip >= 0.50).sum())

    return result


def count_mapped_gws_variants(path: Path, p1: float) -> int | None:
    """
    Count Step05 mapped candidates that are still genome-wide significant.
    This directly distinguishes:
      significant hits existed in QC
    from
      significant hits successfully mapped to the LD reference.
    """
    df = safe_read_tsv(path)
    if df is None:
        return None
    if "P" not in df.columns:
        return None

    p = pd.to_numeric(df["P"], errors="coerce")
    return int((p <= p1).sum())


def collect_accessions(
    qc_root: Path,
    clump_root: Path,
    fine_root: Path,
) -> list[str]:
    accessions = set()

    for root in (qc_root, clump_root, fine_root):
        if root.exists():
            for path in root.iterdir():
                if path.is_dir() and path.name.startswith("GCST"):
                    accessions.add(path.name)

    for manifest_path, column in (
        (clump_root / "clumping_manifest.tsv", "STUDY_ACCESSION"),
        (fine_root / "finemapping_manifest.tsv", "STUDY_ACCESSION"),
    ):
        df = safe_read_tsv(manifest_path)
        if df is not None and column in df.columns:
            for value in df[column].dropna().astype(str):
                if value.strip():
                    accessions.add(value.strip())

    return sorted(accessions)


def classify(row: dict) -> tuple[str, str]:
    n_after_qc = row.get("N_AFTER_QC")
    n_gws = row.get("N_GWS_QC")
    n_gws_mapped = row.get("N_GWS_MAPPED_1000G")
    n_leads = row.get("N_LEAD_VARIANTS")

    clump_ready = row.get("CLUMPING_READY")
    susie_ready = row.get("SUSIE_READY")

    fine_status = str(row.get("FINEMAP_STATUS") or "").strip().upper()
    n_loci = row.get("N_LOCI")
    n_loci_success = row.get("N_LOCI_SUCCESS")
    n_loci_failed = row.get("N_LOCI_FAILED")

    if n_after_qc is None:
        return (
            "QC_OUTPUT_MISSING",
            "Step 03 QC summary is missing or unreadable.",
        )

    if n_after_qc == 0:
        return (
            "NO_VARIANTS_AFTER_QC",
            "Step 03 retained zero variants after QC.",
        )

    if clump_ready is False:
        return (
            "NOT_CLUMPING_READY",
            "Step 03 marked this GWAS as unsuitable for LD clumping.",
        )

    if n_gws == 0:
        return (
            "NO_GWS_HITS",
            "No variants passed the genome-wide-significance threshold in the QC GWAS.",
        )

    if n_gws is not None and n_gws > 0:
        if n_gws_mapped == 0:
            return (
                "GWS_HITS_NOT_MAPPED_TO_REFERENCE",
                "Genome-wide-significant variants exist, but none mapped to the ancestry-matched 1000G LD reference.",
            )

    if n_leads == 0:
        return (
            "NO_INDEPENDENT_LEADS_AFTER_CLUMPING",
            "Significant/reference-mapped variants existed, but Step 05 produced zero independent lead variants.",
        )

    if n_leads is None:
        return (
            "WAITING_FOR_STEP_05",
            "Step 05 clumping output is missing or incomplete.",
        )

    if susie_ready is False:
        return (
            "NOT_SUSIE_READY",
            "Step 03 indicates the GWAS lacks the effect-size information required for SuSiE (BETA+SE or Z, with alleles).",
        )

    if fine_status == "NO_SIGNIFICANT_LOCI":
        return (
            "NO_SIGNIFICANT_LOCI",
            "Step 06 correctly recorded a completed zero-locus result.",
        )

    if n_loci is not None:
        if n_loci == 0:
            return (
                "NO_SIGNIFICANT_LOCI",
                "No independent loci were available for fine-mapping.",
            )

        if n_loci_failed and n_loci_failed > 0:
            if n_loci_success and n_loci_success > 0:
                return (
                    "FINEMAPPING_PARTIAL",
                    "Some loci fine-mapped successfully and some loci failed.",
                )
            return (
                "FINEMAPPING_FAILED",
                "Loci were present but none completed fine-mapping successfully.",
            )

        if n_loci_success == n_loci and n_loci > 0:
            return (
                "FINEMAPPING_COMPLETE",
                "All independent loci completed SuSiE fine-mapping.",
            )

    if row.get("FINEMAP_HARD_FAILURE"):
        return (
            "FINEMAPPING_FAILED",
            "Step 06 produced a hard-failure marker. Inspect FINEMAPPING_FAILED.txt.",
        )

    return (
        "WAITING_FOR_STEP_06",
        "Step 05 produced lead variants, but Step 06 has not produced a final result yet.",
    )


def arguments():
    parser = argparse.ArgumentParser(
        description="Audit Step03 -> Step05 -> Step06 GWAS pipeline results."
    )
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--ancestry", required=True)
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print a wider per-study table including detailed counts.",
    )
    return parser.parse_args()


def main():
    args = arguments()

    root = Path.cwd().resolve()
    phenotype = args.phenotype.strip()
    phenotype_slug = slugify(phenotype)
    ancestry_code, ancestry_label = canonical_ancestry(args.ancestry)
    ancestry_slug = slugify(ancestry_label)

    qc_root = (
        root
        / "02_summary_stats"
        / phenotype_slug
        / ancestry_slug
        / "qc"
    )

    clump_root = (
        root
        / "05_ld_clumping"
        / phenotype_slug
        / ancestry_slug
    )

    fine_root = (
        root
        / "06_finemapping"
        / phenotype_slug
        / ancestry_slug
    )

    output_root = (
        root
        / "diagnostics"
        / phenotype_slug
        / ancestry_slug
    )
    output_root.mkdir(parents=True, exist_ok=True)

    accessions = collect_accessions(
        qc_root=qc_root,
        clump_root=clump_root,
        fine_root=fine_root,
    )

    if not accessions:
        raise RuntimeError(
            "No GCST study directories/manifests were found for "
            f"{phenotype} / {ancestry_label}."
        )

    rows = []

    for accession in accessions:
        # ------------------------------------------------------------
        # STEP 03
        # ------------------------------------------------------------
        qc_dir = qc_root / accession
        qc_summary_file = qc_dir / f"{accession}_QC_summary.tsv"
        qc = first_row_dict(qc_summary_file)

        full_qc = qc_dir / f"{accession}_GRCh38_QC.tsv.gz"
        significant_qc = qc_dir / f"{accession}_GRCh38_significant.tsv.gz"

        # ------------------------------------------------------------
        # STEP 05
        # ------------------------------------------------------------
        clump_dir = clump_root / accession
        clump_summary_file = clump_dir / "clumping_summary.tsv"
        clump = first_row_dict(clump_summary_file)

        mapped_file = clump_dir / "mapped_candidates.tsv.gz"
        lead_file = clump_dir / "lead_variants.tsv"

        p1 = numeric(clump.get("P1"))
        if p1 is None:
            p1 = 5e-8

        n_gws_mapped = count_mapped_gws_variants(
            mapped_file,
            p1=p1,
        )

        # ------------------------------------------------------------
        # STEP 06
        # ------------------------------------------------------------
        fine_dir = fine_root / accession
        fine_json_file = fine_dir / "finemapping_summary.json"
        fine = safe_read_json(fine_json_file)

        final_variants = (
            fine_dir
            / f"{accession}_finemapped_variants.tsv.gz"
        )

        final_cs = (
            fine_dir
            / f"{accession}_95pct_credible_sets.tsv"
        )

        pips = pip_stats(final_variants)

        hard_failure = fine_dir / "FINEMAPPING_FAILED.txt"

        row = {
            "STUDY_ACCESSION": accession,
            "PHENOTYPE": phenotype,
            "ANCESTRY_CODE": ancestry_code,
            "ANCESTRY_LABEL": ancestry_label,

            # Step 03
            "N_BEFORE_QC": integer(qc.get("N_BEFORE_QC")),
            "N_AFTER_QC": integer(qc.get("N_AFTER_QC")),
            "N_REMOVED_QC": integer(qc.get("N_REMOVED")),
            "N_GWS_QC": integer(qc.get("N_GENOME_WIDE_SIGNIFICANT")),
            "MIN_P": numeric(qc.get("MIN_P")),
            "QC_STATUS": qc.get("QC_STATUS"),
            "CLUMPING_READY": truthy(qc.get("CLUMPING_READY")),
            "SUSIE_READY": truthy(qc.get("SUSIE_READY")),
            "HAS_BETA": truthy(qc.get("HAS_BETA")),
            "HAS_SE": truthy(qc.get("HAS_SE")),
            "HAS_Z": truthy(qc.get("HAS_Z")),

            # Step 05
            "P1": p1,
            "P2": numeric(clump.get("P2")),
            "N_P_LE_P2": integer(clump.get("N_P_LE_P2")),
            "N_REFERENCE_MAPPED": integer(clump.get("N_REFERENCE_MAPPED")),
            "MAPPING_RATE": numeric(clump.get("MAPPING_RATE")),
            "N_GWS_MAPPED_1000G": n_gws_mapped,
            "N_INDEX_CANDIDATES_P_LE_P1": integer(
                clump.get("N_INDEX_CANDIDATES_P_LE_P1")
            ),
            "N_LEAD_VARIANTS": integer(clump.get("N_LEAD_VARIANTS")),

            # Step 06
            "GWAS_N": integer(fine.get("GWAS_N")),
            "GWAS_N_SOURCE": fine.get("GWAS_N_SOURCE"),
            "FINEMAP_STATUS": fine.get("STATUS"),
            "N_LOCI": integer(fine.get("N_LOCI")),
            "N_LOCI_SUCCESS": integer(fine.get("N_LOCI_SUCCESS")),
            "N_LOCI_FAILED": integer(fine.get("N_LOCI_FAILED")),
            "N_CREDIBLE_SET_ROWS": count_rows(final_cs),
            **pips,

            # Files / failure markers
            "QC_FILE_EXISTS": full_qc.exists(),
            "SIGNIFICANT_FILE_EXISTS": significant_qc.exists(),
            "CLUMP_SUMMARY_EXISTS": clump_summary_file.exists(),
            "MAPPED_CANDIDATES_EXISTS": mapped_file.exists(),
            "LEAD_VARIANTS_EXISTS": lead_file.exists(),
            "FINEMAP_SUMMARY_EXISTS": fine_json_file.exists(),
            "FINEMAP_HARD_FAILURE": hard_failure.exists(),
        }

        diagnostic_code, diagnostic_message = classify(row)
        row["DIAGNOSTIC_CODE"] = diagnostic_code
        row["DIAGNOSTIC_MESSAGE"] = diagnostic_message

        rows.append(row)

    audit = pd.DataFrame(rows)

    output_tsv = output_root / "pipeline_diagnostics.tsv"
    output_json = output_root / "pipeline_diagnostics.json"

    audit.to_csv(
        output_tsv,
        sep="\t",
        index=False,
    )

    output_json.write_text(
        json.dumps(
            audit.where(pd.notna(audit), None).to_dict(orient="records"),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # CONSOLE REPORT
    # -----------------------------------------------------------------
    print()
    print("=" * 120)
    print("GWAS PIPELINE DIAGNOSTICS")
    print("=" * 120)
    print(f"Phenotype : {phenotype}")
    print(f"Ancestry  : {ancestry_label} ({ancestry_code})")
    print(f"Studies   : {len(audit)}")
    print()

    display_cols = [
        "STUDY_ACCESSION",
        "N_AFTER_QC",
        "N_GWS_QC",
        "N_GWS_MAPPED_1000G",
        "N_LEAD_VARIANTS",
        "N_LOCI",
        "N_LOCI_SUCCESS",
        "N_FINEMAPPED_VARIANTS",
        "MAX_PIP",
        "DIAGNOSTIC_CODE",
    ]

    if args.show_all:
        display_cols = [
            "STUDY_ACCESSION",
            "N_BEFORE_QC",
            "N_AFTER_QC",
            "N_GWS_QC",
            "MIN_P",
            "N_P_LE_P2",
            "N_REFERENCE_MAPPED",
            "N_GWS_MAPPED_1000G",
            "N_INDEX_CANDIDATES_P_LE_P1",
            "N_LEAD_VARIANTS",
            "SUSIE_READY",
            "GWAS_N",
            "N_LOCI",
            "N_LOCI_SUCCESS",
            "N_LOCI_FAILED",
            "N_FINEMAPPED_VARIANTS",
            "N_CREDIBLE_SET_ROWS",
            "MAX_PIP",
            "N_PIP_GE_0_01",
            "N_PIP_GE_0_10",
            "N_PIP_GE_0_50",
            "DIAGNOSTIC_CODE",
        ]

    print(
        audit[display_cols]
        .fillna("-")
        .to_string(index=False)
    )

    print()
    print("=" * 120)
    print("DIAGNOSTIC COUNTS")
    print("=" * 120)

    counts = (
        audit["DIAGNOSTIC_CODE"]
        .value_counts(dropna=False)
    )

    for code, n in counts.items():
        print(f"{str(code):42s} {int(n):4d}")

    print()
    print("=" * 120)
    print("PIPELINE TOTALS")
    print("=" * 120)

    def total(column):
        if column not in audit.columns:
            return 0
        x = pd.to_numeric(audit[column], errors="coerce")
        return int(x.fillna(0).sum())

    print(f"Variants after QC across studies       : {total('N_AFTER_QC'):,}")
    print(f"Genome-wide significant QC variants    : {total('N_GWS_QC'):,}")
    print(f"GWS variants mapped to 1000G           : {total('N_GWS_MAPPED_1000G'):,}")
    print(f"Independent lead variants               : {total('N_LEAD_VARIANTS'):,}")
    print(f"Fine-mapped loci completed              : {total('N_LOCI_SUCCESS'):,}")
    print(f"Fine-mapped variant rows                : {total('N_FINEMAPPED_VARIANTS'):,}")
    print(f"95% credible-set rows                   : {total('N_CREDIBLE_SET_ROWS'):,}")
    print()

    print("Saved:")
    print(f"  {output_tsv}")
    print(f"  {output_json}")
    print()

    print("=" * 120)
    print("HOW TO INTERPRET")
    print("=" * 120)
    print("NO_GWS_HITS")
    print("  -> QC worked, but this GWAS has no variant at P <= 5e-8.")
    print()
    print("GWS_HITS_NOT_MAPPED_TO_REFERENCE")
    print("  -> Significant variants exist, but reference matching needs investigation.")
    print()
    print("NO_INDEPENDENT_LEADS_AFTER_CLUMPING")
    print("  -> Significant/reference-mapped variants exist, but Step05 returned no independent lead.")
    print()
    print("NOT_SUSIE_READY")
    print("  -> The GWAS lacks the BETA+SE or Z information needed by the current SuSiE-RSS implementation.")
    print()
    print("FINEMAPPING_COMPLETE")
    print("  -> Continue this study to functional annotation/QTL/splicing stages.")
    print()
    print("FINEMAPPING_PARTIAL / FINEMAPPING_FAILED")
    print("  -> Inspect locus-level failure outputs before continuing this study.")
    print("=" * 120)


if __name__ == "__main__":
    main()
 
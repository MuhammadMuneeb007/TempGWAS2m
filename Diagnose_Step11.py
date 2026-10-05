#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWAS2m Step11 FAST result diagnostic (RESULTS ONLY; no Slurm/job-array inspection).

Optimized for large Step11 result sets:
  * Polars for manifest loading, filtering, grouping, sorting, and TSV output
  * ThreadPoolExecutor for parallel candidate-directory parsing
  * orjson for fast candidate_result.json parsing
  * lightweight streaming csv.DictReader for tiny per-candidate TSV files
    (faster than starting a dataframe parser thousands of times)
  * NumPy + SciPy for numerical summaries / robust dispersion metrics
  * tqdm progress bars for candidate parsing and report generation
  * reads only the current provider_coloc_manifest.tsv candidate IDs
  * NEVER scans historical candidate_results blindly
  * NEVER calls squeue/sacct or inspects Slurm array state

Scientific behavior:
  * recomputes H0/H1/H2/H3/H4 from the SAME signal-pair row with maximum H4
  * reports GWAS/QTL credible-set components and common-variant counts
  * reports QTL modality, tissue, prior sensitivity, QC/not-tested reasons

Example:
    python Step11_Diagnostic.py \
      --phenotype "parkinson's disease" \
      --ancestry EUR

Optional faster parallelism on a node with sufficient I/O headroom:
    python Step11_Diagnostic.py \
      --phenotype "parkinson's disease" \
      --ancestry EUR \
      --workers 32
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

try:
    import orjson
except ImportError as exc:
    raise SystemExit(
        "ERROR: orjson is required for the fast diagnostic. Install with:\n"
        "  python -m pip install orjson"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "ERROR: numpy is required. Install with:\n"
        "  python -m pip install numpy"
    ) from exc

try:
    import polars as pl
except ImportError as exc:
    raise SystemExit(
        "ERROR: polars is required for the fast diagnostic. Install with:\n"
        "  python -m pip install polars"
    ) from exc

try:
    from scipy.stats import iqr, median_abs_deviation
except ImportError as exc:
    raise SystemExit(
        "ERROR: scipy is required. Install with:\n"
        "  python -m pip install scipy"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "ERROR: tqdm is required. Install with:\n"
        "  python -m pip install tqdm"
    ) from exc


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def slug(value: Any) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def ancestry_label(value: str) -> str:
    x = str(value).strip().lower()
    aliases = {
        "eur": "european",
        "european": "european",
        "europe": "european",
        "afr": "african",
        "african": "african",
        "eas": "east_asian",
        "east_asian": "east_asian",
        "east asian": "east_asian",
        "sas": "south_asian",
        "south_asian": "south_asian",
        "south asian": "south_asian",
        "amr": "admixed_american",
        "admixed_american": "admixed_american",
        "mixed": "mixed",
    }
    return aliases.get(x, slug(x))


def banner(text: str) -> None:
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    return "" if s.lower() in {"nan", "none"} else s


def as_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def as_int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def finite_float(value: Any) -> bool:
    return math.isfinite(as_float(value))


def read_json_fast(path: Path) -> dict[str, Any]:
    """
    Fast JSON reader with compatibility fallback.

    Step11 writes JSON with Python json.dumps(), which can contain
    NaN / Infinity. orjson is strict and rejects those values.

    Therefore:
      1. try orjson first
      2. fall back to Python json.loads()
      3. only report failure if BOTH parsers fail
    """

    if not path.exists():
        return {
            "_JSON_PARSER": "MISSING"
        }

    try:
        if path.stat().st_size == 0:
            return {
                "_JSON_PARSER": "EMPTY",
                "_JSON_READ_ERROR": "empty candidate_result.json",
            }

        raw = path.read_bytes()

    except Exception as exc:
        return {
            "_JSON_PARSER": "READ_ERROR",
            "_JSON_READ_ERROR": f"{type(exc).__name__}: {exc}",
        }

    # --------------------------------------------------------
    # FAST PATH
    # --------------------------------------------------------

    try:
        obj = orjson.loads(raw)

        if isinstance(obj, dict):
            obj["_JSON_PARSER"] = "ORJSON"
            return obj

        return {
            "_JSON_PARSER": "ORJSON_NON_OBJECT",
            "_JSON_READ_ERROR":
                "candidate_result.json is not a JSON object",
        }

    except Exception as exc:
        orjson_error = f"{type(exc).__name__}: {exc}"

    # --------------------------------------------------------
    # COMPATIBILITY PATH
    #
    # Python json accepts NaN / Infinity generated by
    # Python json.dumps().
    # --------------------------------------------------------

    try:
        obj = json.loads(
            raw.decode("utf-8")
        )

        if isinstance(obj, dict):

            obj["_JSON_PARSER"] = "STDLIB_FALLBACK"
            obj["_ORJSON_ERROR"] = orjson_error

            return obj

        return {
            "_JSON_PARSER": "STDLIB_NON_OBJECT",
            "_JSON_READ_ERROR":
                "candidate_result.json is not a JSON object",
            "_ORJSON_ERROR": orjson_error,
        }

    except Exception as exc:

        return {
            "_JSON_PARSER": "FAILED",
            "_JSON_READ_ERROR": (
                f"orjson={orjson_error}; "
                f"stdlib={type(exc).__name__}: {exc}"
            ),
        }


def read_tail_text(path: Path, max_bytes: int = 131_072) -> str:
    """Read only the tail of a log; Step11 signal summary is expected near the end."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        if size <= 0:
            return ""
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(-max_bytes, os.SEEK_END)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Scientific parsers
# -----------------------------------------------------------------------------


def classify_same_pair(
    h: dict[str, float], strong: float, suggestive: float
) -> tuple[str, str]:
    finite = {k: v for k, v in h.items() if math.isfinite(v)}
    if "H4" not in finite:
        return "NOT_TESTED_NO_FINITE_H4", ""

    dominant = max(finite, key=finite.get)
    h4 = finite["H4"]

    if h4 >= strong:
        label = "STRONG_SHARED_SIGNAL"
    elif h4 >= suggestive:
        label = "SUGGESTIVE_SHARED_SIGNAL"
    else:
        label = {
            "H0": "NEITHER_ASSOCIATED_FAVORED",
            "H1": "GWAS_ONLY_FAVORED",
            "H2": "QTL_ONLY_FAVORED",
            "H3": "BOTH_ASSOCIATED_DIFFERENT_SIGNALS_FAVORED",
            "H4": "SHARED_SIGNAL_FAVORED_BELOW_THRESHOLD",
        }.get(dominant, "UNKNOWN")

    return label, dominant


def read_best_pair_stream(summary_file: Path) -> dict[str, Any]:
    """Stream coloc_summary.tsv and keep only the row with maximum finite H4."""
    if not summary_file.exists():
        return {}
    try:
        if summary_file.stat().st_size == 0:
            return {}
    except OSError:
        return {}

    posterior_cols = [f"PP.H{i}.abf" for i in range(5)]
    n_rows = 0
    best_row: dict[str, str] | None = None
    best_h4 = -math.inf

    try:
        with summary_file.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            fieldnames = set(reader.fieldnames or [])
            missing = [c for c in posterior_cols if c not in fieldnames]
            if missing:
                return {"_SUMMARY_ERROR": f"missing posterior columns: {missing}"}

            for row in reader:
                n_rows += 1
                h4 = as_float(row.get("PP.H4.abf"))
                if math.isfinite(h4) and h4 > best_h4:
                    best_h4 = h4
                    best_row = row

    except Exception as exc:
        return {"_SUMMARY_ERROR": f"{type(exc).__name__}: {exc}"}

    if best_row is None:
        return {"_NO_FINITE_H4": True, "N_SIGNAL_PAIRS": n_rows}

    out: dict[str, Any] = {
        "N_SIGNAL_PAIRS": n_rows,
        "BEST_HIT1": as_text(best_row.get("hit1", "")),
        "BEST_HIT2": as_text(best_row.get("hit2", "")),
    }
    out["BEST_SIGNAL_PAIR"] = f"{out['BEST_HIT1']}|{out['BEST_HIT2']}"

    for i in range(5):
        out[f"BEST_PP_H{i}"] = as_float(best_row.get(f"PP.H{i}.abf"))

    return out


def read_prior_sensitivity_stream(
    path: Path, hit1: str, hit2: str
) -> dict[str, float]:
    out = {
        "PP_H4_P12_LOW": float("nan"),
        "PP_H4_P12_DEFAULT": float("nan"),
        "PP_H4_P12_HIGH": float("nan"),
    }

    if not path.exists() or not hit1 or not hit2:
        return out
    try:
        if path.stat().st_size == 0:
            return out
    except OSError:
        return out

    mapping = {
        "LOW_P12": "PP_H4_P12_LOW",
        "DEFAULT": "PP_H4_P12_DEFAULT",
        "HIGH_P12": "PP_H4_P12_HIGH",
    }

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            required = {"hit1", "hit2", "PP.H4.abf", "PRIOR_SET"}
            if not required.issubset(set(reader.fieldnames or [])):
                return out

            remaining = set(mapping)
            for row in reader:
                if (
                    as_text(row.get("hit1")) != hit1
                    or as_text(row.get("hit2")) != hit2
                ):
                    continue

                prior = as_text(row.get("PRIOR_SET"))
                if prior in mapping:
                    val = as_float(row.get("PP.H4.abf"))
                    if math.isfinite(val):
                        out[mapping[prior]] = val
                    remaining.discard(prior)
                    if not remaining:
                        break

    except Exception:
        pass

    return out


LOG_RE = re.compile(
    r"GWAS_SIGNALS=(\d+)\s+QTL_SIGNALS=(\d+)\s+COMMON_VARIANTS=(\d+)"
)


def parse_coloc_log_fast(path: Path) -> dict[str, int | None]:
    out: dict[str, int | None] = {
        "GWAS_SIGNALS_USED": None,
        "QTL_SIGNALS_USED": None,
        "COMMON_VARIANTS": None,
    }
    s = read_tail_text(path)
    if not s:
        return out

    matches = LOG_RE.findall(s)
    if matches:
        g, q, c = matches[-1]
        out.update(
            {
                "GWAS_SIGNALS_USED": int(g),
                "QTL_SIGNALS_USED": int(q),
                "COMMON_VARIANTS": int(c),
            }
        )
    return out


def diagnostic_status(
    raw_status: str,
    error: str,
    best: dict[str, Any],
    result_json_exists: bool,
    json_parser: str,
    summary_exists: bool,
) -> str:

    raw = as_text(raw_status).upper()
    e = as_text(error).lower()

    # ========================================================
    # DIRECT SCIENTIFIC EVIDENCE
    #
    # A finite H4 in coloc_summary.tsv proves that
    # coloc.bf_bf completed for at least one signal pair.
    # ========================================================

    h4 = as_float(
        best.get("BEST_PP_H4")
    )

    if math.isfinite(h4):
        return "COMPLETE"

    # ========================================================
    # EXPLICIT NOT-TESTABLE CONDITIONS
    # ========================================================

    if (
        "no eligible signal pair survived coloc.bf_bf "
        "overlap filtering" in e
        or "insufficient posterior overlap" in e
    ):
        return "NOT_TESTED_INSUFFICIENT_POSTERIOR_OVERLAP"

    if (
        "share zero variants" in e
        or "zero shared variants" in e
    ):
        return "NOT_TESTED_NO_SHARED_VARIANTS"

    if (
        "provider trait cache" in e
        and "missing" in e
    ):
        return "NOT_TESTED_PROVIDER_TRAIT_CACHE_MISSING"

    if (
        "no finite h4 posterior" in e
        or best.get("_NO_FINITE_H4")
    ):
        return "NOT_TESTED_NO_FINITE_H4"

    # ========================================================
    # FILE / REPORTING QC
    # ========================================================

    if (
        not result_json_exists
        and not summary_exists
    ):
        return "MISSING_RESULT"

    if json_parser == "FAILED":
        return "DIAGNOSTIC_JSON_PARSE_FAILED"

    # Preserve explicit Step11 states.
    if raw in {"COMPLETE", "FAILED"}:
        return raw

    if raw.startswith("NOT_TESTED"):
        return raw

    if summary_exists:
        return "SUMMARY_PRESENT_NO_FINITE_RESULT"

    if result_json_exists:
        return "RESULT_PRESENT_STATUS_UNRESOLVED"

    return "MISSING_RESULT"



# -----------------------------------------------------------------------------
# One current-manifest candidate -> one diagnostic record
# -----------------------------------------------------------------------------


def process_candidate(
    mr: dict[str, Any],
    result_dir: Path,
    strong: float,
    suggestive: float,
) -> dict[str, Any]:
    # Manifest fields are carried through so we preserve study/locus/QTL metadata.
    z: dict[str, Any] = {str(k): as_text(v) for k, v in mr.items()}
    cid = as_text(z.get("CANDIDATE_ID"))
    out = result_dir / cid

    jf = out / "candidate_result.json"
    jr = read_json_fast(jf) if jf.exists() else {}

    # Controlled JSON fields only: avoids mixed-schema dataframe inference and is faster.
    json_copy_fields = [
        "STATUS",
        "ERROR",
        "COLOC_TESTED",
        "COLOC_CLASS",
        "GWAS_CS_INDEXES",
        "QTL_CS_INDEXES",
        "MOLECULAR_TRAIT_ID",
        "STUDY_ACCESSION",
        "LOCUS_ID",
        "DATASET_ID",
        "QTL_TYPE",
        "TISSUE",
    ]
    for key in json_copy_fields:
        if key in jr and jr.get(key) not in (None, ""):
            z[key] = jr.get(key)

    z["RESULT_JSON_EXISTS"] = "YES" if jf.exists() else "NO"

    z["JSON_PARSER"] = as_text(
        jr.get("_JSON_PARSER", "")
    )

    z["ORJSON_ERROR"] = as_text(
        jr.get("_ORJSON_ERROR", "")
    )

    z["JSON_READ_ERROR"] = as_text(
        jr.get("_JSON_READ_ERROR", "")
    )

    z["RAW_STATUS"] = as_text(
        jr.get(
            "STATUS",
            "MISSING_RESULT"
            if not jf.exists()
            else ""
        )
    )

    z["RAW_ERROR"] = as_text(
        jr.get(
            "ERROR",
            jr.get("_JSON_READ_ERROR", "")
        )
    )

    summary_file = out / "coloc_summary.tsv"
    best = read_best_pair_stream(summary_file)
    z["SUMMARY_EXISTS"] = "YES" if summary_file.exists() else "NO"
    z["SUMMARY_ERROR"] = as_text(best.get("_SUMMARY_ERROR", ""))
    z["N_SIGNAL_PAIRS"] = int(best.get("N_SIGNAL_PAIRS", 0) or 0)
    z["BEST_HIT1"] = as_text(best.get("BEST_HIT1", jr.get("BEST_HIT1", "")))
    z["BEST_HIT2"] = as_text(best.get("BEST_HIT2", jr.get("BEST_HIT2", "")))
    z["BEST_SIGNAL_PAIR"] = as_text(
        best.get("BEST_SIGNAL_PAIR", jr.get("BEST_SIGNAL_PAIR", ""))
    )

    for i in range(5):
        z[f"BEST_PP_H{i}"] = as_float(best.get(f"BEST_PP_H{i}"))

    h = {f"H{i}": z[f"BEST_PP_H{i}"] for i in range(5)}
    if math.isfinite(h["H4"]):
        cls, dominant = classify_same_pair(h, strong, suggestive)
        z["DIAGNOSTIC_COLOC_CLASS"] = cls
        z["DOMINANT_HYPOTHESIS"] = dominant
        z["COLOC_TESTED_DIAGNOSTIC"] = "YES"
    else:
        z["DIAGNOSTIC_COLOC_CLASS"] = ""
        z["DOMINANT_HYPOTHESIS"] = ""
        z["COLOC_TESTED_DIAGNOSTIC"] = "NO"

    prior = read_prior_sensitivity_stream(
        out / "prior_sensitivity.tsv",
        as_text(z.get("BEST_HIT1")),
        as_text(z.get("BEST_HIT2")),
    )
    z.update(prior)

    finite_prior = np.asarray(
        [as_float(v) for v in prior.values() if finite_float(v)], dtype=float
    )
    if finite_prior.size:
        z["PRIOR_H4_MIN"] = float(np.min(finite_prior))
        z["PRIOR_H4_MAX"] = float(np.max(finite_prior))
    else:
        z["PRIOR_H4_MIN"] = float("nan")
        z["PRIOR_H4_MAX"] = float("nan")

    z["PRIOR_ROBUST_STRONG_DIAGNOSTIC"] = (
        "YES"
        if finite_prior.size >= 2 and float(np.min(finite_prior)) >= strong
        else "NO"
    )

    z.update(
        parse_coloc_log_fast(
            out / "coloc.log"
        )
    )

    z["DIAGNOSTIC_STATUS"] = diagnostic_status(
        as_text(z.get("RAW_STATUS")),
        as_text(z.get("RAW_ERROR")),
        best,
        result_json_exists=jf.exists(),
        json_parser=as_text(
            z.get("JSON_PARSER")
        ),
        summary_exists=summary_file.exists(),
    )

    # --------------------------------------------------------
    # Did formal coloc actually produce a finite H4?
    # --------------------------------------------------------

    z["FORMAL_COLOC_WORKED"] = (
        "YES"
        if math.isfinite(
            as_float(
                z.get("BEST_PP_H4")
            )
        )
        else "NO"
    )

    # --------------------------------------------------------
    # Result was scientifically recoverable even though
    # JSON required fallback or was malformed.
    # --------------------------------------------------------

    z["RESULT_RECOVERED_FROM_SUMMARY"] = (
        "YES"
        if (
            z["FORMAL_COLOC_WORKED"] == "YES"
            and z["JSON_PARSER"] in {
                "FAILED",
                "STDLIB_FALLBACK",
                "EMPTY",
                "MISSING",
            }
        )
        else "NO"
    )

    return z


# -----------------------------------------------------------------------------
# Polars helpers
# -----------------------------------------------------------------------------


NUMERIC_FLOAT_COLS = [
    "BEST_PP_H0",
    "BEST_PP_H1",
    "BEST_PP_H2",
    "BEST_PP_H3",
    "BEST_PP_H4",
    "PP_H4_P12_LOW",
    "PP_H4_P12_DEFAULT",
    "PP_H4_P12_HIGH",
    "PRIOR_H4_MIN",
    "PRIOR_H4_MAX",
]

NUMERIC_INT_COLS = [
    "N_SIGNAL_PAIRS",
    "COMMON_VARIANTS",
    "GWAS_SIGNALS_USED",
    "QTL_SIGNALS_USED",
]


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for c in NUMERIC_FLOAT_COLS:
        if c in df.columns:
            exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
    for c in NUMERIC_INT_COLS:
        if c in df.columns:
            exprs.append(pl.col(c).cast(pl.Int64, strict=False).alias(c))
    return df.with_columns(exprs) if exprs else df


def col_or_empty(df: pl.DataFrame, name: str) -> pl.Expr:
    if name in df.columns:
        return pl.col(name).cast(pl.Utf8, strict=False).fill_null("")
    return pl.lit("")


def print_frame(df: pl.DataFrame, rows: int = 40) -> None:
    if df.height == 0:
        print("<empty>")
        return
    with pl.Config(tbl_rows=max(5, rows), tbl_cols=30, fmt_str_lengths=44):
        print(df)


def numpy_finite_column(df: pl.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns or df.height == 0:
        return np.asarray([], dtype=float)
    arr = np.asarray(df.get_column(col).cast(pl.Float64, strict=False).to_numpy(), dtype=float)
    return arr[np.isfinite(arr)]


def robust_summary(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {
            "N": 0,
            "MIN": float("nan"),
            "MEDIAN": float("nan"),
            "MAX": float("nan"),
            "IQR": float("nan"),
            "MAD": float("nan"),
        }
    return {
        "N": int(arr.size),
        "MIN": float(np.min(arr)),
        "MEDIAN": float(np.median(arr)),
        "MAX": float(np.max(arr)),
        "IQR": float(iqr(arr, nan_policy="omit")),
        "MAD": float(median_abs_deviation(arr, nan_policy="omit", scale="normal")),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast Step11 provider-SuSiE scientific-result diagnostic (no Slurm inspection)."
    )
    ap.add_argument("--phenotype", required=True)
    ap.add_argument("--ancestry", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--strong", type=float, default=0.80)
    ap.add_argument("--suggestive", type=float, default=0.50)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument(
        "--workers",
        type=int,
        default=min(16, max(4, (os.cpu_count() or 4))),
        help="Parallel candidate-file readers (default: min(16, CPU count)).",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = ap.parse_args()

    if args.workers < 1:
        raise SystemExit("ERROR: --workers must be >= 1")

    root = Path(args.root).resolve()
    anc = ancestry_label(args.ancestry)
    base = root / "11_coloc_provider_susie" / slug(args.phenotype) / slug(anc)
    manifest_file = base / "provider_coloc_manifest.tsv"
    result_dir = base / "candidate_results"
    diag_dir = base / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_file.exists():
        raise SystemExit(f"ERROR: current Step11 manifest not found: {manifest_file}")

    banner("FAST STEP11 DIAGNOSTIC INITIALIZATION")
    print(f"Phenotype        : {args.phenotype}")
    print(f"Ancestry         : {anc}")
    print(f"Manifest         : {manifest_file}")
    print(f"Result directory : {result_dir}")
    print(f"Workers          : {args.workers}")
    print("Table engine     : Polars")
    print("JSON parser      : orjson + Python json fallback")
    print("Numerics         : NumPy + SciPy")
    print("Progress         : tqdm" if not args.no_progress else "Progress         : disabled")

    # All manifest columns are loaded as strings for deterministic metadata handling.
    manifest = pl.read_csv(
        manifest_file,
        separator="\t",
        infer_schema_length=0,  # 0 => read manifest columns as strings
        null_values=[],
        quote_char='"',
    )

    if manifest.height == 0:
        raise SystemExit("ERROR: provider_coloc_manifest.tsv is empty")
    if "CANDIDATE_ID" not in manifest.columns:
        raise SystemExit("ERROR: manifest has no CANDIDATE_ID column")

    manifest = (
        manifest
        .with_columns(pl.col("CANDIDATE_ID").cast(pl.Utf8).fill_null(""))
        .filter(pl.col("CANDIDATE_ID") != "")
        .unique(subset=["CANDIDATE_ID"], keep="last", maintain_order=True)
    )

    records = manifest.to_dicts()

    # Parse candidate directories in parallel. I/O dominates, so threads are appropriate.
    rows: list[dict[str, Any]] = []
    progress_disabled = bool(args.no_progress)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_candidate,
                mr,
                result_dir,
                args.strong,
                args.suggestive,
            ): mr.get("CANDIDATE_ID", "")
            for mr in records
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Parsing Step11 results",
            unit="candidate",
            dynamic_ncols=True,
            disable=progress_disabled,
        ):
            cid = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                # A diagnostic should survive one corrupt candidate directory.
                rows.append(
                    {
                        "CANDIDATE_ID": as_text(cid),
                        "RESULT_JSON_EXISTS": "UNKNOWN",
                        "SUMMARY_EXISTS": "UNKNOWN",
                        "RAW_STATUS": "DIAGNOSTIC_READ_FAILURE",
                        "RAW_ERROR": f"{type(exc).__name__}: {exc}",
                        "DIAGNOSTIC_STATUS": "DIAGNOSTIC_READ_FAILURE",
                        "COLOC_TESTED_DIAGNOSTIC": "NO",
                    }
                )

    if not rows:
        raise SystemExit("ERROR: no diagnostic rows were produced")

    # Restore current-manifest order after asynchronous parsing.
    order_map = {
        cid: i
        for i, cid in enumerate(manifest.get_column("CANDIDATE_ID").to_list())
    }
    rows.sort(key=lambda x: order_map.get(as_text(x.get("CANDIDATE_ID")), 10**18))

    df = normalize_frame(pl.from_dicts(rows, infer_schema_length=None))

    detailed_file = diag_dir / "current_manifest_result_diagnostic.tsv"
    df.write_csv(detailed_file, separator="\t")

    tested = df.filter(col_or_empty(df, "COLOC_TESTED_DIAGNOSTIC") == "YES")
    strong_df = tested.filter(pl.col("BEST_PP_H4") >= args.strong) if "BEST_PP_H4" in tested.columns else tested.head(0)
    suggestive_df = (
        tested.filter(
            (pl.col("BEST_PP_H4") >= args.suggestive)
            & (pl.col("BEST_PP_H4") < args.strong)
        )
        if "BEST_PP_H4" in tested.columns
        else tested.head(0)
    )

    # ------------------------------------------------------------------
    banner("1. CURRENT STEP11 SCIENTIFIC RESULT COVERAGE")
    # ------------------------------------------------------------------
    result_present = (
        df.select((col_or_empty(df, "RESULT_JSON_EXISTS") == "YES").sum()).item()
    )
    missing_results = df.height - int(result_present)

    print(f"Current-manifest candidates       : {df.height}")
    print(f"Result JSON present               : {int(result_present)}")
    print(f"Missing current results           : {missing_results}")
    print(f"Formal coloc signal-pairs tested  : {tested.height}")
    print(f"Strong shared signal H4 >= {args.strong:.2f} : {strong_df.height}")
    print(
        f"Suggestive H4 {args.suggestive:.2f}-{args.strong:.2f}     : "
        f"{suggestive_df.height}"
    )
    print(f"Detailed table                    : {detailed_file}")

    # ------------------------------------------------------------------
    banner("2. RESULT STATUS -- SCIENTIFIC/QC STATUS, NOT SLURM")
    # ------------------------------------------------------------------
    status_counts = (
        df.group_by(col_or_empty(df, "DIAGNOSTIC_STATUS").alias("DIAGNOSTIC_STATUS"))
        .len(name="COUNT")
        .sort("COUNT", descending=True)
    )
    print_frame(status_counts, rows=60)

    # ------------------------------------------------------------------
    banner("2B. RESULT-FILE / JSON PARSER HEALTH")
    # ------------------------------------------------------------------

    if "JSON_PARSER" in df.columns:

        parser_counts = (
            df.group_by(
                col_or_empty(
                    df,
                    "JSON_PARSER"
                ).alias("JSON_PARSER")
            )
            .len(name="COUNT")
            .sort(
                "COUNT",
                descending=True
            )
        )

        print_frame(
            parser_counts,
            rows=30
        )

    recovered_n = (
        df.select(
            (
                col_or_empty(
                    df,
                    "RESULT_RECOVERED_FROM_SUMMARY"
                )
                == "YES"
            ).sum()
        ).item()
        if "RESULT_RECOVERED_FROM_SUMMARY"
        in df.columns
        else 0
    )

    worked_n = (
        df.select(
            (
                col_or_empty(
                    df,
                    "FORMAL_COLOC_WORKED"
                )
                == "YES"
            ).sum()
        ).item()
        if "FORMAL_COLOC_WORKED"
        in df.columns
        else 0
    )

    parse_failed_n = (
        df.select(
            (
                col_or_empty(
                    df,
                    "JSON_PARSER"
                )
                == "FAILED"
            ).sum()
        ).item()
        if "JSON_PARSER" in df.columns
        else 0
    )

    print(
        "Formal coloc scientifically recoverable :",
        int(worked_n),
    )

    print(
        "Recovered despite JSON issue/fallback    :",
        int(recovered_n),
    )

    print(
        "Genuine JSON parse failures              :",
        int(parse_failed_n),
    )

    # ------------------------------------------------------------------
    banner("3. ARE WE ACTUALLY CARRYING THE GWAS SIGNAL?")
    # ------------------------------------------------------------------
    gwas_cs_n = df.select((col_or_empty(df, "GWAS_CS_INDEXES").str.len_chars() > 0).sum()).item()
    qtl_cs_n = df.select((col_or_empty(df, "QTL_CS_INDEXES").str.len_chars() > 0).sum()).item()
    log_tested = df.filter(pl.col("GWAS_SIGNALS_USED").is_not_null()) if "GWAS_SIGNALS_USED" in df.columns else df.head(0)

    print(f"Candidates with GWAS credible-set component(s) : {int(gwas_cs_n)} / {df.height}")
    print(f"Candidates with QTL credible-set component(s)  : {int(qtl_cs_n)} / {df.height}")
    print(f"Results whose coloc.log confirms GWAS LBF rows : {log_tested.height}")

    if log_tested.height:
        garr = numpy_finite_column(log_tested, "GWAS_SIGNALS_USED")
        qarr = numpy_finite_column(log_tested, "QTL_SIGNALS_USED")
        carr = numpy_finite_column(log_tested, "COMMON_VARIANTS")
        if garr.size:
            print(
                "GWAS signals used/result                   : "
                f"min={int(np.min(garr))}, median={np.median(garr):.1f}, max={int(np.max(garr))}"
            )
        if qarr.size:
            print(
                "QTL signals used/result                    : "
                f"min={int(np.min(qarr))}, median={np.median(qarr):.1f}, max={int(np.max(qarr))}"
            )
        if carr.size:
            print(
                "Shared GWAS/QTL variants                   : "
                f"min={int(np.min(carr))}, median={np.median(carr):.0f}, max={int(np.max(carr))}"
            )

    print(
        "Interpretation: Step11 carries the Step06 GWAS SuSiE credible-set "
        "log-Bayes-factor signal(s) into coloc.bf_bf; it does NOT replace the GWAS."
    )

    # ------------------------------------------------------------------
    banner("4. SAME-PAIR H0/H1/H2/H3/H4")
    # ------------------------------------------------------------------
    if tested.height:
        dom = (
            tested.group_by(col_or_empty(tested, "DOMINANT_HYPOTHESIS").alias("HYPOTHESIS"))
            .len(name="COUNT")
        )
        dom_map = dict(zip(dom.get_column("HYPOTHESIS").to_list(), dom.get_column("COUNT").to_list()))
        for h in ["H0", "H1", "H2", "H3", "H4"]:
            print(f"{h}: {int(dom_map.get(h, 0))}")

        print()
        classes = (
            tested.group_by(col_or_empty(tested, "DIAGNOSTIC_COLOC_CLASS").alias("COLOC_CLASS"))
            .len(name="COUNT")
            .sort("COUNT", descending=True)
        )
        print_frame(classes, rows=50)
    else:
        print("No finite same-pair H4 results yet.")

    # SciPy-based H4 distribution diagnostics.
    h4arr = numpy_finite_column(tested, "BEST_PP_H4")
    hs = robust_summary(h4arr)
    print()
    print("H4 distribution (NumPy/SciPy):")
    print(f"  N      : {hs['N']}")
    if hs["N"]:
        print(f"  min    : {hs['MIN']:.6f}")
        print(f"  median : {hs['MEDIAN']:.6f}")
        print(f"  max    : {hs['MAX']:.6f}")
        print(f"  IQR    : {hs['IQR']:.6f}")
        print(f"  MAD    : {hs['MAD']:.6f}")

    # ------------------------------------------------------------------
    banner("5. RESULTS BY QTL MODALITY")
    # ------------------------------------------------------------------
    if "QTL_TYPE" in df.columns:
        base_mod = (
            df.with_columns(
                (col_or_empty(df, "COLOC_TESTED_DIAGNOSTIC") == "YES").cast(pl.Int64).alias("_TESTED"),
                (
                    (col_or_empty(df, "COLOC_TESTED_DIAGNOSTIC") == "YES")
                    & (pl.col("BEST_PP_H4") >= args.strong)
                ).cast(pl.Int64).alias("_STRONG"),
                (
                    (col_or_empty(df, "COLOC_TESTED_DIAGNOSTIC") == "YES")
                    & (pl.col("BEST_PP_H4") >= args.suggestive)
                    & (pl.col("BEST_PP_H4") < args.strong)
                ).cast(pl.Int64).alias("_SUGGESTIVE"),
            )
            .group_by(pl.col("QTL_TYPE").fill_null("UNKNOWN"))
            .agg(
                pl.len().alias("CANDIDATES"),
                pl.col("_TESTED").sum().alias("TESTED"),
                pl.col("_STRONG").sum().alias("STRONG_H4"),
                pl.col("_SUGGESTIVE").sum().alias("SUGGESTIVE_H4"),
                pl.col("BEST_PP_H4").max().alias("MAX_H4"),
            )
            .sort(["STRONG_H4", "MAX_H4", "TESTED"], descending=[True, True, True])
        )
        print_frame(base_mod, rows=50)
        base_mod.write_csv(diag_dir / "by_qtl_type.tsv", separator="\t")
    else:
        print("QTL_TYPE column unavailable.")

    # ------------------------------------------------------------------
    banner("6. RESULTS BY GWAS LOCUS")
    # ------------------------------------------------------------------
    if {"STUDY_ACCESSION", "LOCUS_ID"}.issubset(df.columns):
        locus_base = df.with_columns(
            (col_or_empty(df, "COLOC_TESTED_DIAGNOSTIC") == "YES").cast(pl.Int64).alias("_TESTED"),
            (
                (col_or_empty(df, "COLOC_TESTED_DIAGNOSTIC") == "YES")
                & (pl.col("BEST_PP_H4") >= args.strong)
            ).cast(pl.Int64).alias("_STRONG"),
        )

        locus_summary = (
            locus_base.group_by(["STUDY_ACCESSION", "LOCUS_ID"])
            .agg(
                pl.len().alias("CANDIDATES"),
                pl.col("_TESTED").sum().alias("TESTED"),
                pl.col("_STRONG").sum().alias("STRONG_H4"),
                pl.col("BEST_PP_H4").max().alias("MAX_H4"),
            )
        )

        best_rows = (
            locus_base.filter(pl.col("BEST_PP_H4").is_not_null())
            .sort("BEST_PP_H4", descending=True)
            .unique(subset=["STUDY_ACCESSION", "LOCUS_ID"], keep="first")
            .select(
                "STUDY_ACCESSION",
                "LOCUS_ID",
                pl.col("QTL_TYPE").alias("BEST_QTL_TYPE") if "QTL_TYPE" in locus_base.columns else pl.lit("").alias("BEST_QTL_TYPE"),
                pl.col("MOLECULAR_TRAIT_ID").alias("BEST_TRAIT") if "MOLECULAR_TRAIT_ID" in locus_base.columns else pl.lit("").alias("BEST_TRAIT"),
            )
        )

        loci = (
            locus_summary.join(best_rows, on=["STUDY_ACCESSION", "LOCUS_ID"], how="left")
            .sort(["STRONG_H4", "MAX_H4"], descending=[True, True])
        )
        print_frame(loci, rows=max(40, min(120, loci.height)))
        loci.write_csv(diag_dir / "by_gwas_locus.tsv", separator="\t")
    else:
        print("Study/locus columns unavailable.")

    # ------------------------------------------------------------------
    banner(f"7. TOP {args.top} FORMAL COLOCALIZATIONS")
    # ------------------------------------------------------------------
    top_cols = [
        "BEST_PP_H4",
        "BEST_PP_H3",
        "BEST_PP_H2",
        "STUDY_ACCESSION",
        "LOCUS_ID",
        "DATASET_ID",
        "QTL_TYPE",
        "TISSUE",
        "MOLECULAR_TRAIT_ID",
        "GWAS_CS_INDEXES",
        "QTL_CS_INDEXES",
        "BEST_SIGNAL_PAIR",
        "COMMON_VARIANTS",
        "DIAGNOSTIC_COLOC_CLASS",
        "PP_H4_P12_LOW",
        "PP_H4_P12_DEFAULT",
        "PP_H4_P12_HIGH",
        "PRIOR_ROBUST_STRONG_DIAGNOSTIC",
    ]
    top_cols = [c for c in top_cols if c in tested.columns]

    if tested.height == 0 or "BEST_PP_H4" not in tested.columns:
        print("No finite H4 result is available yet.")
    else:
        top_df = tested.sort("BEST_PP_H4", descending=True).head(max(1, args.top)).select(top_cols)
        print_frame(top_df, rows=args.top + 5)
        top_df.write_csv(diag_dir / "top_colocalizations.tsv", separator="\t")

    # ------------------------------------------------------------------
    banner("8. PRIOR ROBUSTNESS")
    # ------------------------------------------------------------------
    robust = (
        tested.filter(col_or_empty(tested, "PRIOR_ROBUST_STRONG_DIAGNOSTIC") == "YES")
        if tested.height
        else tested
    )
    print(f"Strong at default H4 >= {args.strong:.2f}              : {strong_df.height}")
    print(f"Strong across available p12 sensitivity values : {robust.height}")

    if robust.height:
        show_cols = [
            c
            for c in [
                "BEST_PP_H4",
                "PRIOR_H4_MIN",
                "PRIOR_H4_MAX",
                "STUDY_ACCESSION",
                "LOCUS_ID",
                "QTL_TYPE",
                "TISSUE",
                "MOLECULAR_TRAIT_ID",
                "BEST_SIGNAL_PAIR",
            ]
            if c in robust.columns
        ]
        print_frame(
            robust.sort("BEST_PP_H4", descending=True).head(args.top).select(show_cols),
            rows=args.top + 5,
        )

    # ------------------------------------------------------------------
    banner("9. NON-COMPLETE / QC REASONS")
    # ------------------------------------------------------------------
    bad = df.filter(col_or_empty(df, "DIAGNOSTIC_STATUS") != "COMPLETE")
    if bad.height == 0:
        print("No non-complete current-manifest results.")
    else:
        reasons = (
            bad.with_columns(
                col_or_empty(
                    bad,
                    "DIAGNOSTIC_STATUS"
                ).alias(
                    "DIAGNOSTIC_STATUS"
                ),

                col_or_empty(
                    bad,
                    "JSON_PARSER"
                ).alias(
                    "JSON_PARSER"
                ),

                col_or_empty(
                    bad,
                    "RAW_ERROR"
                ).alias(
                    "RAW_ERROR"
                ),
            )
            .group_by(
                [
                    "DIAGNOSTIC_STATUS",
                    "JSON_PARSER",
                    "RAW_ERROR",
                ]
            )
            .len(
                name="COUNT"
            )
            .sort(
                "COUNT",
                descending=True
            )
        )
        reasons.write_csv(diag_dir / "noncomplete_reasons.tsv", separator="\t")

        display = reasons.head(30).with_columns(
            pl.col("RAW_ERROR")
            .str.replace_all("\\n", " ")
            .str.slice(0, 180)
        )
        print_frame(display, rows=35)

    # ------------------------------------------------------------------
    banner("10. OUTPUT FILES")
    # ------------------------------------------------------------------
    output_files = [
        detailed_file,
        diag_dir / "by_qtl_type.tsv",
        diag_dir / "by_gwas_locus.tsv",
        diag_dir / "top_colocalizations.tsv",
        diag_dir / "noncomplete_reasons.tsv",
    ]
    for f in output_files:
        if f.exists():
            print(f)


if __name__ == "__main__":
    main()

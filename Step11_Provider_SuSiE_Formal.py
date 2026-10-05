#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GWAS2m STEP 11 -- PROVIDER SuSiE MULTI-SIGNAL COLOCALIZATION
===========================================================

Purpose
-------
Run formal multi-signal colocalization without re-running QTL SuSiE against
an external 1000 Genomes LD matrix.

The workflow reuses:
  * GWAS SuSiE fine-mapping from Step06 (`susie_fit.rds`), and
  * provider-generated eQTL Catalogue SuSiE log Bayes factors (LBFs).

It then calls coloc::coloc.bf_bf(), the Bayes-factor engine used by the
multi-signal SuSiE colocalization workflow.

NO QTL LD matrix is calculated.
NO QTL runsusie() is executed.
NO LBF is downloaded by this script.
NO remote tabix is used during analysis.

The script expects provider files under:
  resources/coloc/eqtl_catalogue/susie_provider/<DATASET_ID>/

and dense local eQTL Catalogue files under:
  resources/coloc/eqtl_catalogue/dense/

Parallel architecture
---------------------
Stage 1: DISCOVERY -- one SLURM task per GWAS locus.
Stage 2: EXTRACTION -- one SLURM task per provider dataset; each multi-GB LBF
         file is streamed exactly once and only required molecular traits are
         cached into small trait-specific files.
Stage 3: COLOC -- only candidates supported by real provider SuSiE credible
         sets are tested. GWAS and QTL LBF matrices are restricted to real
         credible-set components. Multiple coloc candidates are processed per
         SLURM array task (--candidates-per-task; default 50), and arrays are
         split at <= --array-limit scheduler tasks.
Stage 4: AGGREGATE -- merge all candidate results.

Typical use
-----------
Run ONE command:

   python Step11_Provider_SuSiE_Formal.py \
       --phenotype "parkinson's disease" --ancestry EUR

Default AUTOPILOT mode then:
  1. creates/resumes locus discovery and submits it if needed,
  2. automatically continues after discovery via a SLURM dependency,
  3. rebuilds the provider-CS scientific filter,
  4. selects only provider LBF files that are completely downloaded,
  5. submits provider extraction with conservative I/O concurrency,
  6. submits batched real coloc::coloc.bf_bf() workers after extraction, and
  7. submits aggregation after all currently runnable coloc workers finish.

Rerunning the same command later is safe: completed discovery, extraction and
coloc results are resumed, while newly downloaded provider resources are added.
Advanced flags remain available for debugging/manual single-task execution.

The code deliberately marks unsupported contexts as unavailable rather than
substituting an unmatched external LD panel.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwas2m_config  # noqa: E402  (SLURM settings: config/slurm.yaml; scheduling only)
from typing import Any

import numpy as np
import pandas as pd

try:
    import Step11_Colocalize_GTEx_SuSiE as core
except Exception as exc:
    raise SystemExit(
        "This provider-SuSiE Step11 expects the existing "
        "Step11_Colocalize_GTEx_SuSiE.py in the same GWAS2m directory.\n"
        f"Import failed: {type(exc).__name__}: {exc}"
    )


VERSION = "5.4.0-provider-susie-prewired-dag-master-ready"

P1 = 1e-4
P2 = 1e-4
P12 = 5e-6
P12_LOW = 1e-6
P12_HIGH = 1e-5

DEFAULT_CANDIDATE_P = 1e-5
DEFAULT_STRONG_PP4 = 0.80
DEFAULT_SUGGESTIVE_PP4 = 0.50
DEFAULT_MAX_CANDIDATES_PER_CONTEXT_LOCUS = 100
DEFAULT_ARRAY_LIMIT = 1000
DEFAULT_MAX_PARALLEL = 0
DEFAULT_EXTRACT_CHUNKSIZE = 250_000
DEFAULT_CANDIDATES_PER_TASK = 50
DEFAULT_MAX_EXTRACT_PARALLEL = 0
DEFAULT_COLOC_WORKER_POOL = 1000


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def banner(text: str) -> None:
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120, flush=True)


def slug(value: Any) -> str:
    return core.slugify(value)


def sha1_short(value: str, n: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def write_json(path: Path, payload: dict) -> None:
    """Write standards-compliant JSON atomically (NaN/Inf -> null)."""
    def clean(value):
        if value is None:
            return None
        if isinstance(value, (np.floating, float)):
            x = float(value)
            return x if np.isfinite(x) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [clean(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(clean(payload), indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def is_complete_aria2_file(path: Path) -> bool:
    return is_nonempty(path) and not Path(str(path) + ".aria2").exists()


def run(cmd: list[Any], log_file: Path | None = None) -> None:
    cmd = [str(x) for x in cmd]
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = proc.wait()
    else:
        rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def provider_root(root: Path) -> Path:
    return root / "resources" / "coloc" / "eqtl_catalogue" / "susie_provider"


def out_paths(root: Path, phenotype: str, ancestry_label: str) -> dict[str, Path]:
    p = slug(phenotype)
    a = slug(ancestry_label)
    base = root / "11_coloc_provider_susie" / p / a
    return {
        "BASE": base,
        "DISCOVERY": base / "discovery",
        "CACHE": base / "provider_trait_cache",
        "RESULTS": base / "candidate_results",
        "BATCHES": base / "batches",
        "LOG": root / "logs" / "step11_provider_susie" / p / a,
        "S06": root / "06_finemapping" / p / a,
    }


def provider_local_path(root: Path, dataset_id: str, url: str) -> Path:
    name = str(url).strip().split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return provider_root(root) / str(dataset_id) / name


def context_provider_paths(root: Path, context: pd.Series | dict) -> tuple[Path | None, Path | None]:
    dataset_id = str(context.get("dataset_id", "")).strip()
    lbf_url = str(context.get("ftp_lbf_path", "")).strip()
    cs_url = str(context.get("ftp_cs_path", "")).strip()
    lbf = provider_local_path(root, dataset_id, lbf_url) if dataset_id and lbf_url else None
    cs = provider_local_path(root, dataset_id, cs_url) if dataset_id and cs_url else None
    return lbf, cs


def validate_coloc(rscript: Path) -> None:
    code = (
        'stopifnot(requireNamespace("coloc", quietly=TRUE)); '
        'stopifnot("coloc.bf_bf" %in% getNamespaceExports("coloc")); '
        'cat("coloc=", as.character(packageVersion("coloc")), "\\n", sep="")'
    )
    run([rscript, "-e", code])


# -----------------------------------------------------------------------------
# CLI / context loading
# -----------------------------------------------------------------------------

def arguments():
    p = argparse.ArgumentParser(
        description=(
            "Provider-SuSiE formal multi-signal colocalization with a "
            "login-node-prewired SLURM DAG. Discovery and provider planning "
            "run on compute nodes; no compute node ever calls sbatch."
        )
    )

    p.add_argument("--phenotype", required=True)
    p.add_argument("--ancestry", required=True)
    p.add_argument("--study", default="")

    p.add_argument(
        "--qtl-types",
        default="eQTL,sQTL,pQTL,isoQTL,exonQTL",
    )

    p.add_argument(
        "--candidate-p",
        type=float,
        default=DEFAULT_CANDIDATE_P,
    )

    p.add_argument(
        "--max-candidates-per-context-locus",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_CONTEXT_LOCUS,
    )

    p.add_argument(
        "--strong-pp4",
        type=float,
        default=DEFAULT_STRONG_PP4,
    )

    p.add_argument(
        "--suggestive-pp4",
        type=float,
        default=DEFAULT_SUGGESTIVE_PP4,
    )

    p.add_argument(
        "--catalog-all-studies",
        action="store_true",
    )

    p.add_argument(
        "--catalog-study-regex",
        default="",
    )

    # --------------------------------------------------------
    # SLURM resources
    # --------------------------------------------------------

    p.add_argument(
        "--partition",
        default=None,
    )

    p.add_argument(
        "--discover-time",
        default=None,
    )

    p.add_argument(
        "--planner-time",
        default=None,
        help="Wall time for provider candidate/credible-set planning job.",
    )

    p.add_argument(
        "--extract-time",
        default=None,
    )

    p.add_argument(
        "--coloc-time",
        default=None,
    )

    p.add_argument(
        "--discover-memory",
        default=None,
    )

    p.add_argument(
        "--planner-memory",
        default=None,
    )

    p.add_argument(
        "--extract-memory",
        default=None,
    )

    p.add_argument(
        "--coloc-memory",
        default=None,
    )

    p.add_argument(
        "--cpus",
        type=int,
        default=None,
    )

    p.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Maximum simultaneous discovery/coloc array tasks. "
            "0 = no Step11 %N throttle."
        ),
    )

    p.add_argument(
        "--max-extract-parallel",
        type=int,
        default=None,
        help=(
            "Maximum simultaneous provider extraction array tasks. "
            "0 = no Step11 %N throttle."
        ),
    )

    p.add_argument(
        "--array-limit",
        type=int,
        default=None,
    )

    p.add_argument(
        "--coloc-worker-pool",
        type=int,
        default=DEFAULT_COLOC_WORKER_POOL,
        help=(
            "Number of pre-submitted coloc array workers. "
            "Workers stride over dynamically generated candidate batches. "
            "Maximum 1000; default 1000."
        ),
    )

    p.add_argument(
        "--extract-chunksize",
        type=int,
        default=DEFAULT_EXTRACT_CHUNKSIZE,
    )

    p.add_argument(
        "--candidates-per-task",
        type=int,
        default=DEFAULT_CANDIDATES_PER_TASK,
        help=(
            "Number of formal coloc candidates grouped into one logical "
            "candidate batch. The fixed worker pool strides across these "
            "logical batches."
        ),
    )

    p.add_argument("--tabix", default="")
    p.add_argument("--rscript", default="")

    # --------------------------------------------------------
    # User/debug modes
    # --------------------------------------------------------

    p.add_argument("--discover-index", type=int)
    p.add_argument("--build-provider-plan", action="store_true")
    p.add_argument("--extract-index", type=int)
    p.add_argument("--candidate-index", type=int)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--force", action="store_true")

    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Create the full prewired DAG scripts but do not submit them.",
    )

    # --------------------------------------------------------
    # Internal SLURM modes
    # --------------------------------------------------------

    p.add_argument("--discover-worker", action="store_true")
    p.add_argument("--planner-worker", action="store_true")
    p.add_argument("--extract-worker", action="store_true")
    p.add_argument("--candidate-worker", action="store_true")

    # Compatibility with the older v5.3 continuation script.
    # In v5.4 this NEVER submits jobs from a compute node.
    p.add_argument(
        "--auto-continue",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    p.add_argument(
        "--slurm-config",
        default=None,
        help="SLURM config file (default config/slurm.yaml); scheduling only.",
    )

    args = p.parse_args()
    if args.slurm_config:
        # exported, so jobs submitted from here resolve the same file
        os.environ[gwas2m_config.CONFIG_ENV_VAR] = str(Path(args.slurm_config).resolve())
    apply_slurm_config(args)
    return args


STAGE_ARGUMENTS = {
    "coloc_discovery": {"time": "discover_time", "memory": "discover_memory", "cpus": "cpus",
                        "max_parallel": "max_parallel"},
    "coloc_planner": {"time": "planner_time", "memory": "planner_memory"},
    "coloc_extraction": {"time": "extract_time", "memory": "extract_memory",
                         "max_parallel": "max_extract_parallel"},
    "coloc": {"time": "coloc_time", "memory": "coloc_memory"},
}


def apply_slurm_config(args) -> None:
    """Fill unset scheduling flags from config/slurm.yaml (CLI values win)."""
    for stage, mapping in STAGE_ARGUMENTS.items():
        resources = gwas2m_config.get_stage_resources(stage)
        for key, attribute in mapping.items():
            if getattr(args, attribute) is None:
                setattr(args, attribute, resources[key])
    if args.array_limit is None:
        args.array_limit = gwas2m_config.get_stage_resources("coloc")["array_limit"]


def slurm_stage(args, stage: str) -> dict:
    """Effective resources for one Step11 job type (--partition overrides config)."""
    return gwas2m_config.get_stage_resources(stage, overrides={"partition": args.partition})


def slurm_site(args, stage: str) -> str:
    """partition/account/qos/... lines; empty when the config leaves them null."""
    return "\n".join(gwas2m_config.site_directives(slurm_stage(args, stage)))


def allowed_types(args) -> set[str]:
    return {x.strip().lower() for x in args.qtl_types.split(",") if x.strip()}


def load_provider_contexts(root: Path, args, require_dense: bool = True) -> pd.DataFrame:
    catalogue = core.load_eqtl_catalogue(root, False)
    contexts = core.select_catalogue_contexts(catalogue, args, allowed_types(args))
    if contexts.empty:
        raise RuntimeError("No eQTL Catalogue contexts matched the requested filters.")

    # Real provider-SuSiE analysis only: never manufacture QTL SuSiE using 1000G.
    url = contexts["ftp_lbf_path"].fillna("").astype(str).str.strip()
    contexts = contexts[url.ne("") & ~url.str.lower().isin({"na", "none", "."})].copy()
    if contexts.empty:
        raise RuntimeError("No selected contexts advertise provider SuSiE LBF files.")

    contexts["PROVIDER_LBF_PATH"] = [
        str(provider_local_path(root, row.dataset_id, row.ftp_lbf_path))
        for row in contexts.itertuples(index=False)
    ]
    contexts["PROVIDER_CS_PATH"] = [
        str(provider_local_path(root, row.dataset_id, row.ftp_cs_path))
        if str(row.ftp_cs_path).strip() else ""
        for row in contexts.itertuples(index=False)
    ]

    if require_dense:
        contexts = core.resolve_catalogue_local_sources(root, contexts)
    return contexts.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Stage 1: local dense-QTL discovery, one task per GWAS locus
# -----------------------------------------------------------------------------

def build_discovery_plan(root: Path, args) -> None:
    phenotype = args.phenotype.strip()
    ancestry_code, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, phenotype, ancestry_label)
    for key in ["BASE", "DISCOVERY", "CACHE", "RESULTS", "BATCHES", "LOG"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    studies, excluded = core.discover_step06_studies(paths["S06"])
    if args.study:
        studies = [x for x in studies if x == args.study]
    if not studies:
        raise RuntimeError(f"No eligible Step06 studies under {paths['S06']}")

    rows = []
    for si, accession in enumerate(studies, start=1):
        for locus_id in core.list_loci(paths["S06"] / accession):
            rows.append({
                "TASK_ID": len(rows) + 1,
                "STUDY_INDEX": si,
                "STUDY_ACCESSION": accession,
                "LOCUS_ID": locus_id,
                "PHENOTYPE": phenotype,
                "ANCESTRY_CODE": ancestry_code,
                "ANCESTRY_LABEL": ancestry_label,
            })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No Step06 loci were found.")
    manifest_file = paths["BASE"] / "discovery_manifest.tsv"
    manifest.to_csv(manifest_file, sep="\t", index=False)
    pd.DataFrame(excluded).to_csv(paths["BASE"] / "excluded_studies.tsv", sep="\t", index=False)

    py = Path(sys.executable).resolve()
    script = Path(__file__).resolve()
    throttle = f"%{args.max_parallel}" if args.max_parallel > 0 else ""
    bash = root / f"Step11_ProviderFM_Discover_{slug(phenotype)}_{slug(ancestry_label)}.sh"
    bash.write_text(f"""#!/bin/bash
#SBATCH --job-name=PColDisc_{slug(phenotype)[:24]}
#SBATCH --nodes=1
{slurm_site(args, 'coloc_discovery')}
#SBATCH --time={args.discover_time}
#SBATCH --mem={args.discover_memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1
#SBATCH --array=1-{len(manifest)}{throttle}
#SBATCH --output={paths['LOG']}/discover.%A_%a.out
#SBATCH --error={paths['LOG']}/discover.%A_%a.err
set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_PROVIDER_DISCOVERY_MANIFEST={shlex.quote(str(manifest_file))}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} \\
  --ancestry {shlex.quote(ancestry_code)} \\
  --qtl-types {shlex.quote(args.qtl_types)} \\
  --candidate-p {args.candidate_p} \\
  --max-candidates-per-context-locus {args.max_candidates_per_context_locus} \\
  --catalog-study-regex {shlex.quote(args.catalog_study_regex)} \\
  {'--catalog-all-studies' if args.catalog_all_studies else ''} \\
  --discover-worker
""", encoding="utf-8")
    bash.chmod(0o755)

    banner("STEP11 PROVIDER-SUSIE DISCOVERY PLAN")
    print(f"Phenotype          : {phenotype}")
    print(f"Ancestry           : {ancestry_label} ({ancestry_code})")
    print(f"GWAS studies       : {len(studies)}")
    print(f"GWAS locus tasks   : {len(manifest)}")
    print("QTL method         : provider SuSiE LBF only")
    print("QTL LD calculation : NONE")
    print("QTL runsusie       : NONE")
    print(f"Submit             : sbatch {bash.name}")


def discovery_row_from_index(root: Path, args, index: int) -> pd.Series:
    _, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, args.phenotype.strip(), ancestry_label)
    manifest_file = paths["BASE"] / "discovery_manifest.tsv"
    if not manifest_file.exists():
        build_discovery_plan(root, args)
    x = pd.read_csv(manifest_file, sep="\t", dtype=str)
    ids = pd.to_numeric(x["TASK_ID"], errors="coerce")
    z = x[ids == int(index)]
    if len(z) != 1:
        raise RuntimeError(f"Expected one discovery task for {index}; found {len(z)}")
    return z.iloc[0]


def run_discovery(root: Path, args, row: pd.Series) -> None:
    phenotype = str(row["PHENOTYPE"])
    ancestry_code = str(row["ANCESTRY_CODE"])
    ancestry_label = str(row["ANCESTRY_LABEL"])
    accession = str(row["STUDY_ACCESSION"])
    locus_id = str(row["LOCUS_ID"])
    paths = out_paths(root, phenotype, ancestry_label)
    task_dir = paths["DISCOVERY"] / accession / locus_id
    task_dir.mkdir(parents=True, exist_ok=True)
    done = task_dir / "discovery_done.json"
    if done.exists() and not args.force:
        meta = read_json(done)
        if meta.get("STATUS") == "COMPLETE":
            print(f"[RESUME] discovery {accession} {locus_id}")
            return

    banner(f"PROVIDER-SUSIE DISCOVERY -- {accession} | {locus_id}")
    tabix = core.find_executable(root, args.tabix or None, "TABIX", "tabix")
    contexts = load_provider_contexts(root, args, require_dense=True)
    core.check_local_tabix_connectivity(tabix, contexts)

    study_dir = paths["S06"] / accession
    gwas = core.load_gwas_locus(study_dir, locus_id)
    if gwas.empty:
        raise RuntimeError(f"Step06 locus is empty: {accession} {locus_id}")

    candidates, audit = core.discover_catalogue_candidates_for_locus(
        tabix, contexts, locus_id, gwas, args
    )
    if not candidates.empty:
        # Attach provider paths deterministically from the catalogue manifest.
        lookup = contexts.drop_duplicates("dataset_id").set_index("dataset_id")
        candidates["PROVIDER_LBF_PATH"] = candidates["DATASET_ID"].map(
            lambda d: lookup.loc[str(d), "PROVIDER_LBF_PATH"] if str(d) in lookup.index else ""
        )
        candidates["PROVIDER_CS_PATH"] = candidates["DATASET_ID"].map(
            lambda d: lookup.loc[str(d), "PROVIDER_CS_PATH"] if str(d) in lookup.index else ""
        )
        candidates = candidates.drop_duplicates(
            ["LOCUS_ID", "DATASET_ID", "QTL_TYPE", "TISSUE", "MOLECULAR_TRAIT_ID"],
            keep="first",
        ).reset_index(drop=True)

    candidate_file = task_dir / "locus_candidates.tsv"
    candidates.to_csv(candidate_file, sep="\t", index=False)
    pd.DataFrame(audit).to_csv(task_dir / "resource_audit.tsv", sep="\t", index=False)

    # Prepare Step06 variant map ONCE for this GWAS locus.
    fit, gmap = core.prepare_gwas_susie_map(
        study_dir, locus_id, ancestry_code, gwas, task_dir
    )
    if fit is None or gmap is None:
        raise RuntimeError(
            f"Step06 SuSiE fit/variant map unavailable for {accession} {locus_id}"
        )

    write_json(done, {
        "VERSION": VERSION,
        "STATUS": "COMPLETE",
        "STUDY_ACCESSION": accession,
        "LOCUS_ID": locus_id,
        "N_CANDIDATES": int(len(candidates)),
        "GWAS_SUSIE_FIT": str(fit),
        "GWAS_VARIANT_MAP": str(gmap),
        "COMPLETED_UTC": utcnow(),
    })
    print(f"[OK] candidates={len(candidates)} -> {candidate_file}")


# -----------------------------------------------------------------------------
# Stage 2 plan: stream each provider LBF once and extract only needed traits
# -----------------------------------------------------------------------------

def gather_discovery(root: Path, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, args.phenotype.strip(), ancestry_label)
    manifest = pd.read_csv(paths["BASE"] / "discovery_manifest.tsv", sep="\t", dtype=str)
    candidates = []
    missing = []
    for row in manifest.itertuples(index=False):
        task_dir = paths["DISCOVERY"] / str(row.STUDY_ACCESSION) / str(row.LOCUS_ID)
        done = read_json(task_dir / "discovery_done.json")
        f = task_dir / "locus_candidates.tsv"
        if done.get("STATUS") != "COMPLETE" or not f.exists():
            missing.append({"STUDY_ACCESSION": row.STUDY_ACCESSION, "LOCUS_ID": row.LOCUS_ID})
            continue
        x = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        if x.empty:
            continue
        x["STUDY_ACCESSION"] = str(row.STUDY_ACCESSION)
        x["ANCESTRY_CODE"] = str(row.ANCESTRY_CODE)
        x["ANCESTRY_LABEL"] = str(row.ANCESTRY_LABEL)
        x["PHENOTYPE"] = str(row.PHENOTYPE)
        x["GWAS_SUSIE_FIT"] = done.get("GWAS_SUSIE_FIT", "")
        x["GWAS_VARIANT_MAP"] = done.get("GWAS_VARIANT_MAP", "")
        candidates.append(x)
    return (pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame(), pd.DataFrame(missing))


def trait_cache_path(paths: dict[str, Path], dataset_id: str, trait_id: str) -> Path:
    return paths["CACHE"] / str(dataset_id) / f"trait_{sha1_short(str(trait_id), 20)}.lbf.tsv.gz"


def parse_cs_component(value: Any) -> int | None:
    """Return a 1-based SuSiE component index without consuming digits from trait IDs.

    Accepted examples include ``1``, ``L1``, ``CS1``, and provider identifiers
    such as ``ENSG00000214401_L1``.  Importantly, the Ensembl digits are never
    interpreted as the SuSiE component.
    """
    s = str(value).strip()
    if not s:
        return None

    # A genuine numeric cs_index column may contain just 1, 2, ...
    if re.fullmatch(r"\d+", s):
        x = int(s)
        return x if x >= 1 else None

    # Provider cs_id values commonly end in _L1, _L2, ...; standalone L1/CS1
    # are also supported.  Anchor at the end so ENSG/ENST digits are ignored.
    for pat in (r"(?i)(?:^|[_:.-])L(\d+)$", r"(?i)(?:^|[_:.-])CS(\d+)$"):
        m = re.search(pat, s)
        if m:
            x = int(m.group(1))
            return x if x >= 1 else None
    return None


def parse_variant_chrom_pos(value: Any) -> tuple[str, int] | None:
    try:
        z = core.parse_variant_id(str(value))
        chrom = str(z[0]).replace("chr", "")
        pos = int(z[1])
        return chrom, pos
    except Exception:
        return None


def locus_bounds_from_variant_map(path: Path) -> tuple[str, int, int]:
    x = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if x.empty:
        raise RuntimeError(f"GWAS variant map is empty: {path}")
    lower = {str(c).lower(): c for c in x.columns}
    chrom_col = lower.get("chrom") or lower.get("chr") or lower.get("chromosome")
    pos_col = lower.get("pos") or lower.get("position") or lower.get("bp")
    if chrom_col and pos_col:
        pos = pd.to_numeric(x[pos_col], errors="coerce")
        tmp = pd.DataFrame({"chrom": x[chrom_col].astype(str).str.replace("chr", "", regex=False), "pos": pos}).dropna()
    elif "VARIANT_KEY" in x.columns:
        parsed = x["VARIANT_KEY"].map(parse_variant_chrom_pos)
        tmp = pd.DataFrame([(z[0], z[1]) for z in parsed if z is not None], columns=["chrom", "pos"])
    else:
        raise RuntimeError(f"Cannot derive locus bounds from GWAS variant map: {path}")
    if tmp.empty:
        raise RuntimeError(f"No parseable variants in GWAS variant map: {path}")
    counts = tmp["chrom"].value_counts()
    chrom = str(counts.index[0])
    tmp = tmp[tmp["chrom"].astype(str) == chrom]
    return chrom, int(tmp["pos"].min()), int(tmp["pos"].max())


def gwas_cs_indices(rscript: Path, fit_path: Path) -> list[int]:
    code = "fit <- readRDS(commandArgs(trailingOnly=TRUE)[1]); idx <- tryCatch(fit$sets$cs_index, error=function(e) NULL); if (is.null(idx) || length(idx) == 0) quit(status=0); cat(paste(as.integer(idx), collapse=','))"
    proc = subprocess.run([str(rscript), "-e", code, str(fit_path)], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not read GWAS SuSiE credible-set indices from {fit_path}: {proc.stderr.strip()}")
    out = proc.stdout.strip()
    if not out:
        return []
    vals = []
    for z in out.split(","):
        try:
            x = int(z)
            if x >= 1:
                vals.append(x)
        except Exception:
            pass
    return sorted(set(vals))


def read_provider_cs_components(cs_path: Path, needed_traits: set[str], chunksize: int = 250_000) -> dict[str, dict[int, dict[str, Any]]]:
    """Read provider credible-set rows only for needed traits."""
    if not is_complete_aria2_file(cs_path):
        raise RuntimeError(f"Provider credible-set file incomplete/missing: {cs_path}")
    header = pd.read_csv(cs_path, sep="\t", compression="infer", nrows=0)
    lower = {str(c).lower(): c for c in header.columns}
    trait_col = lower.get("molecular_trait_id")
    cs_col = lower.get("cs_index") or lower.get("cs_id")
    chrom_col = lower.get("chromosome") or lower.get("chrom") or lower.get("chr")
    pos_col = lower.get("position") or lower.get("pos")
    variant_col = lower.get("variant") or lower.get("variant_id")
    low_purity_col = lower.get("low_purity")
    if not trait_col or not cs_col:
        raise RuntimeError(f"Unexpected provider credible-set format: {cs_path}; columns={list(header.columns)[:40]}")
    usecols = [trait_col, cs_col]
    for c in [chrom_col, pos_col, variant_col, low_purity_col]:
        if c and c not in usecols:
            usecols.append(c)
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for chunk in pd.read_csv(cs_path, sep="\t", compression="infer", usecols=usecols, dtype=str, keep_default_na=False, chunksize=max(10_000, int(chunksize)), low_memory=False):
        sub = chunk[chunk[trait_col].astype(str).isin(needed_traits)].copy()
        if sub.empty:
            continue
        if low_purity_col:
            bad = sub[low_purity_col].astype(str).str.lower().isin({"true", "t", "1", "yes"})
            sub = sub[~bad].copy()
        for _, rr in sub.iterrows():
            trait = str(rr.get(trait_col, ""))
            comp = parse_cs_component(rr.get(cs_col, ""))
            if not trait or comp is None:
                continue
            chrom = ""
            pos = None
            if chrom_col and pos_col:
                chrom = str(rr.get(chrom_col, "")).replace("chr", "")
                try:
                    pos = int(float(rr.get(pos_col, "")))
                except Exception:
                    pos = None
            if (not chrom or pos is None) and variant_col:
                parsed = parse_variant_chrom_pos(rr.get(variant_col, ""))
                if parsed is not None:
                    chrom, pos = parsed
            ent = result.setdefault(trait, {}).setdefault(comp, {"chrom": chrom, "min_pos": pos, "max_pos": pos, "n_cs_variants": 0})
            ent["n_cs_variants"] += 1
            if chrom and not ent.get("chrom"):
                ent["chrom"] = chrom
            if pos is not None:
                ent["min_pos"] = pos if ent.get("min_pos") is None else min(int(ent["min_pos"]), pos)
                ent["max_pos"] = pos if ent.get("max_pos") is None else max(int(ent["max_pos"]), pos)
    return result


def classify_bf_summary(summary: pd.DataFrame, strong: float, suggestive: float) -> dict:
    """Classify from one internally consistent posterior vector: the max-H4 pair."""
    cols = {f"H{i}": f"PP.H{i}.abf" for i in range(5)}
    missing = [c for c in cols.values() if c not in summary.columns]
    if missing:
        raise RuntimeError(f"coloc summary missing posterior columns: {missing}")

    x = summary.copy()
    for c in cols.values():
        x[c] = pd.to_numeric(x[c], errors="coerce")

    h4 = pd.to_numeric(x[cols["H4"]], errors="coerce")
    finite = np.isfinite(h4.to_numpy(dtype=float, na_value=np.nan))
    if not finite.any():
        raise RuntimeError("coloc summary has no finite H4 posterior")

    best_idx = h4[finite].idxmax()
    best = x.loc[best_idx]

    out: dict[str, Any] = {}
    vals = []
    for i in range(5):
        v = pd.to_numeric(pd.Series([best.get(cols[f"H{i}"])]), errors="coerce").iloc[0]
        vv = float(v) if pd.notna(v) and np.isfinite(float(v)) else np.nan
        vals.append(vv)
        # Backward-compatible name, now correctly same-pair rather than unrelated maxima.
        out[f"MAX_PP_H{i}"] = vv
        out[f"BEST_PAIR_PP_H{i}"] = vv

    out["BEST_HIT1"] = str(best.get("hit1", ""))
    out["BEST_HIT2"] = str(best.get("hit2", ""))
    out["BEST_GWAS_VARIANT"] = out["BEST_HIT1"]
    out["BEST_QTL_VARIANT"] = out["BEST_HIT2"]
    out["BEST_SIGNAL_PAIR"] = f"{out['BEST_HIT1']}|{out['BEST_HIT2']}"

    finite_vals = [v if np.isfinite(v) else -1.0 for v in vals]
    dominant = int(np.argmax(finite_vals))
    pp4 = float(out["BEST_PAIR_PP_H4"])
    out["BEST_PAIR_DOMINANT_HYPOTHESIS"] = f"H{dominant}"
    out["BEST_PAIR_POSTERIOR_SUM"] = float(sum(v for v in vals if np.isfinite(v)))

    if pp4 >= strong:
        label = "STRONG_SHARED_SIGNAL"
    elif pp4 >= suggestive:
        label = "SUGGESTIVE_SHARED_SIGNAL"
    else:
        label = {
            0: "NEITHER_ASSOCIATED_FAVORED",
            1: "GWAS_ONLY_FAVORED",
            2: "QTL_ONLY_FAVORED",
            3: "BOTH_ASSOCIATED_DIFFERENT_SIGNALS_FAVORED",
            4: "SHARED_SIGNAL_FAVORED_BELOW_THRESHOLD",
        }[dominant]

    out["COLOCALIZED_STRONG"] = "YES" if pp4 >= strong else "NO"
    out["SHARED_SIGNAL"] = "YES" if pp4 >= suggestive else "NO"
    out["COLOC_CLASS"] = label
    return out


def build_provider_plan(root: Path, args) -> None:
    phenotype = args.phenotype.strip()
    ancestry_code, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, phenotype, ancestry_label)
    for key in ["BASE", "CACHE", "RESULTS", "BATCHES", "LOG"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    raw_candidates, missing = gather_discovery(root, args)
    if not missing.empty:
        print(missing.to_string(index=False))
        raise RuntimeError(
            f"Discovery is incomplete for {len(missing)} locus task(s). "
            "Wait for the discovery array to finish before --build-provider-plan."
        )
    if raw_candidates.empty:
        raise RuntimeError("Discovery completed but produced no provider-SuSiE candidates.")

    # Only provider-finemapped contexts are eligible. Both the LBF and provider
    # credible-set path must exist in metadata; CS files provide the scientific
    # filter and the real SuSiE component indices.
    raw_candidates = raw_candidates[
        raw_candidates["PROVIDER_LBF_PATH"].astype(str).str.strip().ne("")
        & raw_candidates["PROVIDER_CS_PATH"].astype(str).str.strip().ne("")
    ].copy()
    raw_candidates = raw_candidates.drop_duplicates(
        ["STUDY_ACCESSION", "LOCUS_ID", "DATASET_ID", "QTL_TYPE", "TISSUE", "MOLECULAR_TRAIT_ID"],
        keep="first",
    ).reset_index(drop=True)
    n_raw = len(raw_candidates)
    if n_raw == 0:
        raise RuntimeError("No candidates have both provider LBF and credible-set resources.")

    # Resource status by dataset. The plan may be partial while downloads are
    # still running; rerunning --build-provider-plan later expands it safely.
    resource_rows = []
    for dataset_id, group in raw_candidates.groupby("DATASET_ID", sort=True):
        lbf_paths = sorted(set(group["PROVIDER_LBF_PATH"].astype(str)))
        cs_paths = sorted(set(group["PROVIDER_CS_PATH"].astype(str)))
        if len(lbf_paths) != 1 or len(cs_paths) != 1:
            raise RuntimeError(
                f"Dataset {dataset_id} maps ambiguously: LBF={lbf_paths}, CS={cs_paths}"
            )
        lbf = Path(lbf_paths[0])
        cs = Path(cs_paths[0])
        resource_rows.append({
            "DATASET_ID": str(dataset_id),
            "PROVIDER_LBF_PATH": str(lbf),
            "PROVIDER_CS_PATH": str(cs),
            "LBF_COMPLETE_NOW": "YES" if is_complete_aria2_file(lbf) else "NO",
            "CS_COMPLETE_NOW": "YES" if is_complete_aria2_file(cs) else "NO",
            "N_RAW_CANDIDATES": len(group),
        })
    resources = pd.DataFrame(resource_rows)
    resources.to_csv(paths["BASE"] / "provider_resource_status.tsv", sep="\t", index=False)

    # Cache real GWAS credible-set indices and locus coordinates (32 loci only,
    # so the RDS checks are cheap). Candidates from a locus without a Step06
    # credible set are not valid multi-signal coloc.susie comparisons.
    rscript = core.find_executable(root, args.rscript or None, "RSCRIPT", "Rscript")
    gwas_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for (study, locus), group in raw_candidates.groupby(["STUDY_ACCESSION", "LOCUS_ID"], sort=False):
        fit = Path(str(group["GWAS_SUSIE_FIT"].iloc[0]))
        gmap = Path(str(group["GWAS_VARIANT_MAP"].iloc[0]))
        if not is_nonempty(fit) or not is_nonempty(gmap):
            gwas_cache[(str(study), str(locus))] = {"cs": [], "bounds": None}
            continue
        idx = gwas_cs_indices(rscript, fit)
        bounds = locus_bounds_from_variant_map(gmap)
        gwas_cache[(str(study), str(locus))] = {"cs": idx, "bounds": bounds}

    kept_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    def audit(row: pd.Series, status: str, reason: str, qtl_cs: str = "", gwas_cs: str = "") -> None:
        audit_rows.append({
            "STUDY_ACCESSION": str(row.get("STUDY_ACCESSION", "")),
            "LOCUS_ID": str(row.get("LOCUS_ID", "")),
            "DATASET_ID": str(row.get("DATASET_ID", "")),
            "QTL_TYPE": str(row.get("QTL_TYPE", "")),
            "TISSUE": str(row.get("TISSUE", "")),
            "MOLECULAR_TRAIT_ID": str(row.get("MOLECULAR_TRAIT_ID", "")),
            "FILTER_STATUS": status,
            "FILTER_REASON": reason,
            "GWAS_CS_INDEXES": gwas_cs,
            "QTL_CS_INDEXES": qtl_cs,
        })

    # Scientific provider-CS filter. eQTL Catalogue exports cs_index as L1/L2/...
    # and lbf_variable1/lbf_variable2/... from the same SuSiE fit. Only real,
    # non-low-purity credible-set components overlapping the GWAS locus survive.
    for dataset_id, group in raw_candidates.groupby("DATASET_ID", sort=True):
        cs_path = Path(str(group["PROVIDER_CS_PATH"].iloc[0]))
        if not is_complete_aria2_file(cs_path):
            for _, row in group.iterrows():
                audit(row, "DEFERRED", "PROVIDER_CS_INCOMPLETE")
            continue

        needed_traits = set(group["MOLECULAR_TRAIT_ID"].astype(str))
        cs_map = read_provider_cs_components(
            cs_path, needed_traits, chunksize=max(10_000, int(args.extract_chunksize))
        )

        for _, row in group.iterrows():
            study = str(row["STUDY_ACCESSION"])
            locus = str(row["LOCUS_ID"])
            trait = str(row["MOLECULAR_TRAIT_ID"])
            gmeta = gwas_cache[(study, locus)]
            gidx = list(gmeta.get("cs") or [])
            gidx_s = ",".join(str(x) for x in gidx)
            if not gidx:
                audit(row, "EXCLUDED", "NO_GWAS_SUSIE_CREDIBLE_SET", gwas_cs=gidx_s)
                continue
            bounds = gmeta.get("bounds")
            if bounds is None:
                audit(row, "EXCLUDED", "GWAS_LOCUS_BOUNDS_UNAVAILABLE", gwas_cs=gidx_s)
                continue
            locus_chrom, locus_start, locus_end = bounds
            comps = cs_map.get(trait, {})
            if not comps:
                audit(row, "EXCLUDED", "NO_PROVIDER_SUSIE_CREDIBLE_SET_FOR_TRAIT", gwas_cs=gidx_s)
                continue

            overlap_components = []
            for comp, meta in sorted(comps.items()):
                cchrom = str(meta.get("chrom", "")).replace("chr", "")
                cstart = meta.get("min_pos")
                cend = meta.get("max_pos")
                if not cchrom or cstart is None or cend is None:
                    continue
                if cchrom == str(locus_chrom).replace("chr", "") and int(cend) >= int(locus_start) and int(cstart) <= int(locus_end):
                    overlap_components.append(int(comp))
            if not overlap_components:
                all_q = ",".join(f"L{x}" for x in sorted(comps))
                audit(row, "EXCLUDED", "PROVIDER_CS_DOES_NOT_OVERLAP_GWAS_LOCUS", qtl_cs=all_q, gwas_cs=gidx_s)
                continue

            z = row.to_dict()
            z["GWAS_CS_INDEXES"] = gidx_s
            z["N_GWAS_CS"] = len(gidx)
            z["QTL_CS_INDEXES"] = ",".join(f"L{x}" for x in overlap_components)
            z["N_QTL_CS"] = len(overlap_components)
            z["CS_FILTER_STATUS"] = "KEEP"
            kept_rows.append(z)
            audit(row, "KEEP", "REAL_GWAS_AND_QTL_CS_OVERLAP", qtl_cs=z["QTL_CS_INDEXES"], gwas_cs=gidx_s)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(paths["BASE"] / "provider_candidate_filter_audit.tsv", sep="\t", index=False)
    candidates = pd.DataFrame(kept_rows)
    if candidates.empty:
        n_cs_complete = int((resources["CS_COMPLETE_NOW"] == "YES").sum())
        raise RuntimeError(
            f"No CS-filtered candidates are currently testable. Provider CS complete: "
            f"{n_cs_complete}/{len(resources)}. Rerun --build-provider-plan as CS downloads complete."
        )

    candidates = candidates.reset_index(drop=True)
    candidates["TRAIT_CACHE_PATH"] = [
        str(trait_cache_path(paths, d, t))
        for d, t in zip(candidates["DATASET_ID"], candidates["MOLECULAR_TRAIT_ID"])
    ]
    candidates["CANDIDATE_ID"] = [
        sha1_short("|".join(map(str, vals)), 24)
        for vals in zip(
            candidates["STUDY_ACCESSION"], candidates["LOCUS_ID"], candidates["DATASET_ID"],
            candidates["QTL_TYPE"], candidates["TISSUE"], candidates["MOLECULAR_TRAIT_ID"],
            candidates["GWAS_CS_INDEXES"], candidates["QTL_CS_INDEXES"],
        )
    ]

    # Provider extraction manifest now includes only datasets/traits that survived
    # credible-set filtering. Each giant provider LBF is still streamed once.
    extraction_rows = []
    for task_id, (dataset_id, group) in enumerate(candidates.groupby("DATASET_ID", sort=True), start=1):
        lbf_paths = sorted(set(group["PROVIDER_LBF_PATH"].astype(str)))
        if len(lbf_paths) != 1:
            raise RuntimeError(f"Dataset {dataset_id} mapped to multiple LBF paths: {lbf_paths}")
        needed = paths["CACHE"] / str(dataset_id) / "needed_traits.txt"
        needed.parent.mkdir(parents=True, exist_ok=True)
        traits = sorted(set(group["MOLECULAR_TRAIT_ID"].astype(str)))
        needed.write_text("\n".join(traits) + "\n", encoding="utf-8")
        extraction_rows.append({
            "TASK_ID": task_id,
            "DATASET_ID": dataset_id,
            "PROVIDER_LBF_PATH": lbf_paths[0],
            "PROVIDER_CS_PATH": str(group["PROVIDER_CS_PATH"].iloc[0]),
            "NEEDED_TRAITS_FILE": str(needed),
            "N_NEEDED_TRAITS": len(traits),
        })
    extraction = pd.DataFrame(extraction_rows)
    extract_manifest = paths["BASE"] / "provider_extraction_manifest.tsv"
    extraction.to_csv(extract_manifest, sep="\t", index=False)

    # Candidate manifest remains one row per scientific comparison, while SLURM
    # workers process multiple rows sequentially to avoid tens of thousands of jobs.
    candidates.insert(0, "TASK_ID", np.arange(1, len(candidates) + 1))
    cpt = max(1, int(args.candidates_per_task))
    candidates["WORKER_TASK_ID"] = (np.arange(len(candidates)) // cpt) + 1
    total_worker_tasks = int(candidates["WORKER_TASK_ID"].max())
    candidate_manifest = paths["BASE"] / "provider_coloc_manifest.tsv"
    candidates.to_csv(candidate_manifest, sep="\t", index=False)

    py = Path(sys.executable).resolve()
    script = Path(__file__).resolve()
    throttle = f"%{args.max_parallel}" if args.max_parallel > 0 else ""

    # Remove stale generated batch manifests/scripts from the old plan.
    for old in paths["BATCHES"].glob("coloc_batch_*.tsv"):
        old.unlink()
    for old in root.glob(f"Step11_ProviderFM_Coloc_{slug(phenotype)}_{slug(ancestry_label)}_B*.sh"):
        old.unlink()

    # Extraction array.
    extract_bash = root / f"Step11_ProviderFM_Extract_{slug(phenotype)}_{slug(ancestry_label)}.sh"
    extract_bash.write_text(f"""#!/bin/bash
#SBATCH --job-name=PColExt_{slug(phenotype)[:24]}
#SBATCH --nodes=1
{slurm_site(args, 'coloc_extraction')}
#SBATCH --time={args.extract_time}
#SBATCH --mem={args.extract_memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1
#SBATCH --array=1-{len(extraction)}{throttle}
#SBATCH --output={paths['LOG']}/extract.%A_%a.out
#SBATCH --error={paths['LOG']}/extract.%A_%a.err
set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_PROVIDER_EXTRACT_MANIFEST={shlex.quote(str(extract_manifest))}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} --ancestry {shlex.quote(ancestry_code)} \\
  --extract-chunksize {args.extract_chunksize} --extract-worker
""", encoding="utf-8")
    extract_bash.chmod(0o755)

    # Coloc arrays are split by scheduler TASK count, not candidate count.
    batch_scripts = []
    array_limit = max(1, int(args.array_limit))
    for batch_no, task_start in enumerate(range(1, total_worker_tasks + 1, array_limit), start=1):
        task_end = min(total_worker_tasks, task_start + array_limit - 1)
        sub = candidates[
            (candidates["WORKER_TASK_ID"] >= task_start)
            & (candidates["WORKER_TASK_ID"] <= task_end)
        ].copy()
        sub["BATCH_TASK_ID"] = sub["WORKER_TASK_ID"] - task_start + 1
        n_batch_tasks = task_end - task_start + 1
        mf = paths["BATCHES"] / f"coloc_batch_{batch_no:03d}.tsv"
        sub.to_csv(mf, sep="\t", index=False)
        bf = root / f"Step11_ProviderFM_Coloc_{slug(phenotype)}_{slug(ancestry_label)}_B{batch_no:03d}.sh"
        bf.write_text(f"""#!/bin/bash
#SBATCH --job-name=PColoc_{batch_no:03d}_{slug(phenotype)[:18]}
#SBATCH --nodes=1
{slurm_site(args, 'coloc')}
#SBATCH --time={args.coloc_time}
#SBATCH --mem={args.coloc_memory}
#SBATCH --cpus-per-task={slurm_stage(args, 'coloc')['cpus']}
#SBATCH --ntasks=1
#SBATCH --array=1-{n_batch_tasks}{throttle}
#SBATCH --output={paths['LOG']}/coloc.B{batch_no:03d}.%A_%a.out
#SBATCH --error={paths['LOG']}/coloc.B{batch_no:03d}.%A_%a.err
set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_PROVIDER_COLOC_MANIFEST={shlex.quote(str(mf))}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} --ancestry {shlex.quote(ancestry_code)} \\
  --strong-pp4 {args.strong_pp4} --suggestive-pp4 {args.suggestive_pp4} \\
  --candidate-worker
""", encoding="utf-8")
        bf.chmod(0o755)
        batch_scripts.append(bf)

    # Aggregation script.
    aggregate_bash = root / f"Step11_ProviderFM_Aggregate_{slug(phenotype)}_{slug(ancestry_label)}.sh"
    aggregate_bash.write_text(f"""#!/bin/bash
#SBATCH --job-name=PColAgg_{slug(phenotype)[:24]}
#SBATCH --nodes=1
{slurm_site(args, 'aggregation')}
#SBATCH --time={slurm_stage(args, 'aggregation')['time']}
#SBATCH --mem={slurm_stage(args, 'aggregation')['memory']}
#SBATCH --cpus-per-task={slurm_stage(args, 'aggregation')['cpus']}
#SBATCH --ntasks=1
#SBATCH --output={paths['LOG']}/aggregate.%j.out
#SBATCH --error={paths['LOG']}/aggregate.%j.err
set -euo pipefail
cd {shlex.quote(str(root))}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} --ancestry {shlex.quote(ancestry_code)} --aggregate
""", encoding="utf-8")
    aggregate_bash.chmod(0o755)

    # Dependency-aware master submitter.
    submit = root / f"Submit_Step11_ProviderFM_{slug(phenotype)}_{slug(ancestry_label)}.sh"
    lines = [
        "#!/bin/bash", "set -euo pipefail", f"cd {shlex.quote(str(root))}",
        f"EXTRACT_JOB=$(sbatch --parsable {shlex.quote(extract_bash.name)})",
        'echo "Provider extraction job: $EXTRACT_JOB"',
        "COLOC_JOBS=()",
    ]
    for bf in batch_scripts:
        lines += [
            f"J=$(sbatch --parsable --dependency=afterok:$EXTRACT_JOB {shlex.quote(bf.name)})",
            'COLOC_JOBS+=("$J")',
            f'echo "Coloc batch {bf.stem}: $J"',
        ]
    lines += [
        'DEP=$(IFS=:; echo "${COLOC_JOBS[*]}")',
        f"AGG=$(sbatch --parsable --dependency=afterok:$DEP {shlex.quote(aggregate_bash.name)})",
        'echo "Aggregation job: $AGG"',
    ]
    submit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    submit.chmod(0o755)

    # Refresh resource audit for only filtered datasets too.
    extraction["LBF_COMPLETE_NOW"] = extraction["PROVIDER_LBF_PATH"].map(
        lambda p: "YES" if is_complete_aria2_file(Path(p)) else "NO"
    )
    extraction["CS_COMPLETE_NOW"] = extraction["PROVIDER_CS_PATH"].map(
        lambda p: "YES" if is_complete_aria2_file(Path(p)) else "NO"
    )
    extraction.to_csv(paths["BASE"] / "filtered_provider_resource_status.tsv", sep="\t", index=False)

    n_cs_complete = int((resources["CS_COMPLETE_NOW"] == "YES").sum())
    n_lbf_complete = int((resources["LBF_COMPLETE_NOW"] == "YES").sum())
    n_deferred = int((audit_df["FILTER_STATUS"] == "DEFERRED").sum()) if not audit_df.empty else 0
    n_excluded = int((audit_df["FILTER_STATUS"] == "EXCLUDED").sum()) if not audit_df.empty else 0
    partial = n_cs_complete < len(resources)

    banner("PROVIDER-SUSIE CS-FILTERED PARALLEL PLAN READY")
    print(f"Raw dense-QTL candidates       : {n_raw}")
    print(f"CS-filtered coloc candidates   : {len(candidates)}")
    print(f"Scientifically excluded        : {n_excluded}")
    print(f"Deferred (CS still downloading): {n_deferred}")
    print(f"Provider CS complete now       : {n_cs_complete}/{len(resources)}")
    print(f"Provider LBF complete now      : {n_lbf_complete}/{len(resources)}")
    print(f"Filtered provider datasets     : {len(extraction)}")
    print(f"Candidates per SLURM task      : {cpt}")
    print(f"TOTAL COLOC SLURM TASKS        : {total_worker_tasks}")
    print(f"Coloc array batches            : {len(batch_scripts)}")
    print(f"Max scheduler tasks per array  : {array_limit}")
    print(f"Max concurrent tasks per array : {args.max_parallel}")
    print(f"Plan is partial                : {'YES' if partial else 'NO'}")
    print("GWAS signal filter             : Step06 fit$sets$cs_index only")
    print("QTL signal filter              : provider cs_index only + locus overlap")
    print("QTL LD computation             : NONE")
    print("QTL runsusie                   : NONE")
    print()
    if partial:
        print("IMPORTANT: some provider credible-set files are still downloading.")
        print("Rerun --build-provider-plan later to obtain the FINAL task count.")
    else:
        print("This is the FINAL CS-filtered candidate/task count for the completed provider-CS set.")
    print()
    print("AUTOPILOT note: the default one-command mode will submit only fully downloaded LBF datasets now.")
    print("Rerun the same one-command Step11 invocation later to add newly downloaded resources safely.")
    print(f"Manual full-chain script (advanced use only): {submit.name}")



# -----------------------------------------------------------------------------
# AUTOPILOT: one user command -> discover -> plan -> extract -> coloc -> aggregate
# -----------------------------------------------------------------------------

def _job_is_active(job_id: str) -> bool:
    job_id = str(job_id or "").strip()
    if not job_id:
        return False
    proc = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        text=True, capture_output=True,
    )
    if proc.returncode != 0:
        return False
    return any(x.strip() in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
               for x in proc.stdout.splitlines())


def _sbatch(script: Path, dependency: str = "", extra: list[str] | None = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    if extra:
        cmd.extend(extra)
    cmd.append(str(script))
    out = subprocess.check_output(cmd, text=True).strip()
    # Some Slurm installations append the cluster name after ';'.
    return out.split(";", 1)[0].strip()


def _candidate_is_complete(paths: dict[str, Path], candidate_id: str) -> bool:
    """Return True for terminal results that should not be resubmitted automatically."""
    f = paths["RESULTS"] / str(candidate_id) / "candidate_result.json"
    if not f.exists():
        return False
    status = str(read_json(f).get("STATUS", ""))
    if status == "COMPLETE":
        return True
    terminal_not_tested = {
        "NOT_TESTED_NO_FINITE_H4",
        "NOT_TESTED_NO_SHARED_VARIANTS",
        "NOT_TESTED_INSUFFICIENT_POSTERIOR_OVERLAP",
        "NOT_TESTED_NO_PARSEABLE_PROVIDER_LBF_VARIANTS",
        "NOT_TESTED_NO_FINITE_LBF_SIGNAL",
    }
    return status in terminal_not_tested


def prepare_ready_manifests(
    root: Path,
    args,
) -> dict[str, Any]:
    """
    Build the dynamic manifests consumed by the already-submitted extraction
    and coloc worker pools.

    IMPORTANT:
    This function never calls sbatch. It is safe to run on a compute node.
    """

    phenotype = args.phenotype.strip()
    ancestry_code, ancestry_label = core.canonical_ancestry(
        args.ancestry
    )

    paths = out_paths(
        root,
        phenotype,
        ancestry_label,
    )

    auto_dir = (
        paths["BASE"]
        / "autopilot"
    )

    auto_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["LOG"].mkdir(
        parents=True,
        exist_ok=True,
    )

    extract_manifest = (
        paths["BASE"]
        / "provider_extraction_manifest.tsv"
    )

    coloc_manifest = (
        paths["BASE"]
        / "provider_coloc_manifest.tsv"
    )

    if (
        not extract_manifest.exists()
        or not coloc_manifest.exists()
    ):
        raise RuntimeError(
            "Provider plan manifests are missing after build_provider_plan()."
        )

    ext = pd.read_csv(
        extract_manifest,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    cand = pd.read_csv(
        coloc_manifest,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    if ext.empty or cand.empty:
        raise RuntimeError(
            "Provider plan is empty; there is no currently runnable coloc work."
        )

    # --------------------------------------------------------
    # Only completely downloaded provider LBF datasets
    # --------------------------------------------------------

    ext["LBF_COMPLETE_NOW"] = (
        ext["PROVIDER_LBF_PATH"]
        .map(
            lambda p: (
                "YES"
                if is_complete_aria2_file(
                    Path(str(p))
                )
                else "NO"
            )
        )
    )

    ready_ext = (
        ext[
            ext["LBF_COMPLETE_NOW"]
            .eq("YES")
        ]
        .copy()
    )

    ready_datasets = set(
        ready_ext[
            "DATASET_ID"
        ]
        .astype(str)
    )

    ready = (
        cand[
            cand["DATASET_ID"]
            .astype(str)
            .isin(
                ready_datasets
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Preserve terminal/complete candidate results.
    # FAILED/MISSING candidates remain eligible.
    # --------------------------------------------------------

    if not ready.empty:

        keep = [
            not _candidate_is_complete(
                paths,
                cid,
            )
            for cid
            in ready[
                "CANDIDATE_ID"
            ].astype(str)
        ]

        ready = (
            ready[
                np.asarray(
                    keep,
                    dtype=bool,
                )
            ]
            .copy()
        )

    # --------------------------------------------------------
    # Extraction manifest for the fixed worker pool
    # --------------------------------------------------------

    if not ready.empty:

        needed_datasets = set(
            ready[
                "DATASET_ID"
            ].astype(str)
        )

        ready_ext = (
            ready_ext[
                ready_ext[
                    "DATASET_ID"
                ]
                .astype(str)
                .isin(
                    needed_datasets
                )
            ]
            .copy()
        )

    else:

        ready_ext = ready_ext.iloc[
            0:0
        ].copy()

    ready_ext = (
        ready_ext
        .reset_index(
            drop=True
        )
    )

    if len(ready_ext):

        ready_ext[
            "TASK_ID"
        ] = np.arange(
            1,
            len(ready_ext) + 1,
        )

    elif "TASK_ID" not in ready_ext.columns:

        ready_ext[
            "TASK_ID"
        ] = pd.Series(
            dtype=int
        )

    ready_extract_manifest = (
        auto_dir
        / "provider_extraction_ready.tsv"
    )

    ready_ext.to_csv(
        ready_extract_manifest,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Formal coloc logical batches
    # --------------------------------------------------------

    ready = (
        ready
        .reset_index(
            drop=True
        )
    )

    cpt = max(
        1,
        int(
            args.candidates_per_task
        ),
    )

    if len(ready):

        ready[
            "WORKER_TASK_ID"
        ] = (
            np.arange(
                len(ready)
            )
            // cpt
        ) + 1

        # Compatibility / easier inspection.
        ready[
            "BATCH_TASK_ID"
        ] = ready[
            "WORKER_TASK_ID"
        ]

        total_worker_tasks = int(
            ready[
                "WORKER_TASK_ID"
            ].max()
        )

    else:

        ready[
            "WORKER_TASK_ID"
        ] = pd.Series(
            dtype=int
        )

        ready[
            "BATCH_TASK_ID"
        ] = pd.Series(
            dtype=int
        )

        total_worker_tasks = 0

    ready_coloc_manifest = (
        auto_dir
        / "provider_coloc_ready.tsv"
    )

    ready.to_csv(
        ready_coloc_manifest,
        sep="\t",
        index=False,
    )

    # A compatibility batch file for older diagnostics/tools.
    ready.to_csv(
        auto_dir
        / "coloc_ready_batch_001.tsv",
        sep="\t",
        index=False,
    )

    payload = {
        "VERSION":
            VERSION,

        "PHENOTYPE":
            phenotype,

        "ANCESTRY":
            ancestry_code,

        "PLANNED_UTC":
            utcnow(),

        "N_PROVIDER_DATASETS_IN_PLAN":
            len(ext),

        "N_READY_DATASETS":
            len(ready_ext),

        "N_FORMAL_CANDIDATES":
            len(cand),

        "N_PENDING_COLOCS":
            len(ready),

        "CANDIDATES_PER_LOGICAL_TASK":
            cpt,

        "N_LOGICAL_COLOC_TASKS":
            total_worker_tasks,

        "READY_EXTRACT_MANIFEST":
            str(
                ready_extract_manifest
            ),

        "READY_COLOC_MANIFEST":
            str(
                ready_coloc_manifest
            ),
    }

    write_json(
        auto_dir
        / "planner_ready.json",
        payload,
    )

    banner(
        "STEP11 PLANNER -- READY MANIFESTS"
    )

    print(
        f"Formal candidates in plan : {len(cand):,}"
    )

    print(
        f"Provider datasets ready   : {len(ready_ext):,}/{len(ext):,}"
    )

    print(
        f"Pending real colocs       : {len(ready):,}"
    )

    print(
        f"Candidates/logical task   : {cpt}"
    )

    print(
        f"Logical coloc tasks       : {total_worker_tasks:,}"
    )

    print(
        "Nested sbatch calls       : NONE"
    )

    return payload


def planner_worker(
    root: Path,
    args,
) -> None:
    """
    Compute-node planner.

    It gathers completed Stage-1 discovery, applies the provider credible-set
    filter, and writes the extraction/coloc manifests.

    It NEVER calls sbatch.
    """

    phenotype = (
        args.phenotype.strip()
    )

    _, ancestry_label = (
        core.canonical_ancestry(
            args.ancestry
        )
    )

    paths = out_paths(
        root,
        phenotype,
        ancestry_label,
    )

    banner(
        "STEP11 COMPUTE-NODE PLANNER"
    )

    _, missing = gather_discovery(
        root,
        args,
    )

    if not missing.empty:

        raise RuntimeError(
            "Planner started before discovery completed. "
            f"Missing locus tasks: {len(missing)}"
        )

    build_provider_plan(
        root,
        args,
    )

    payload = (
        prepare_ready_manifests(
            root,
            args,
        )
    )

    write_json(
        paths["BASE"]
        / "autopilot"
        / "planner_done.json",
        {
            **payload,
            "STATUS":
                "COMPLETE",

            "COMPLETED_UTC":
                utcnow(),
        },
    )

    banner(
        "STEP11 COMPUTE-NODE PLANNER COMPLETE"
    )

    print(
        "The extraction/coloc jobs were already submitted "
        "from the login node with dependencies."
    )


def submit_full_dag(
    root: Path,
    args,
    missing: pd.DataFrame,
) -> dict[str, Any]:
    """
    Submit the COMPLETE Step11 dependency DAG from the login node.

    Why:
      UQ compute nodes reject nested sbatch calls.

    Therefore all sbatch commands happen here, before the command exits:

      DISCOVERY (optional)
          ?
      PLANNER on compute node
          ?
      EXTRACTION fixed worker pool
          ?
      COLOC fixed worker pool
          ?
      AGGREGATE

    The planner creates the manifests after discovery. Downstream workers are
    already queued with dependencies and consume those manifests when released.
    """

    phenotype = (
        args.phenotype.strip()
    )

    ancestry_code, ancestry_label = (
        core.canonical_ancestry(
            args.ancestry
        )
    )

    paths = out_paths(
        root,
        phenotype,
        ancestry_label,
    )

    auto_dir = (
        paths["BASE"]
        / "autopilot"
    )

    auto_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["LOG"].mkdir(
        parents=True,
        exist_ok=True,
    )

    state_file = (
        auto_dir
        / "full_dag_submission.json"
    )

    # --------------------------------------------------------
    # Avoid duplicate active DAGs.
    # --------------------------------------------------------

    old = read_json(
        state_file
    )

    old_jobs = []

    for key in [
        "DISCOVERY_JOB",
        "PLANNER_JOB",
        "EXTRACT_JOB",
        "COLOC_JOB",
        "AGG_JOB",
    ]:

        j = str(
            old.get(
                key,
                "",
            )
        ).strip()

        if j:
            old_jobs.append(
                j
            )

    active_old = [
        j
        for j in old_jobs
        if _job_is_active(
            j
        )
    ]

    if active_old:

        banner(
            "STEP11 FULL DAG ALREADY ACTIVE"
        )

        print(
            "Active job IDs: "
            + ", ".join(
                active_old
            )
        )

        print(
            "No duplicate DAG submitted."
        )

        return {
            "SUBMITTED":
                False,

            "ACTIVE_JOBS":
                active_old,
        }

    # --------------------------------------------------------
    # Discovery script already knows one task per GWAS locus.
    # --------------------------------------------------------

    disc_script = (
        root
        / (
            f"Step11_ProviderFM_Discover_"
            f"{slug(phenotype)}_"
            f"{slug(ancestry_label)}.sh"
        )
    )

    if not disc_script.exists():

        build_discovery_plan(
            root,
            args,
        )

    # --------------------------------------------------------
    # Determine safe fixed pool sizes BEFORE discovery ends.
    # No scientific candidate information is needed for this.
    # --------------------------------------------------------

    contexts = (
        load_provider_contexts(
            root,
            args,
            require_dense=False,
        )
    )

    n_provider_contexts = max(
        1,
        len(contexts),
    )

    extract_pool = min(
        max(
            1,
            n_provider_contexts,
        ),
        max(
            1,
            int(
                args.array_limit
            ),
        ),
        1000,
    )

    coloc_pool = min(
        max(
            1,
            int(
                args.coloc_worker_pool
            ),
        ),
        max(
            1,
            int(
                args.array_limit
            ),
        ),
        1000,
    )

    extract_throttle = (
        f"%{args.max_extract_parallel}"
        if args.max_extract_parallel > 0
        else ""
    )

    coloc_throttle = (
        f"%{args.max_parallel}"
        if args.max_parallel > 0
        else ""
    )

    py = (
        Path(
            sys.executable
        )
        .resolve()
    )

    script = (
        Path(
            __file__
        )
        .resolve()
    )

    ready_extract_manifest = (
        auto_dir
        / "provider_extraction_ready.tsv"
    )

    ready_coloc_manifest = (
        auto_dir
        / "provider_coloc_ready.tsv"
    )

    # --------------------------------------------------------
    # Planner script
    # --------------------------------------------------------

    planner_bash = (
        root
        / (
            f"Step11_Auto_Planner_"
            f"{slug(phenotype)}_"
            f"{slug(ancestry_label)}.sh"
        )
    )

    planner_bash.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=PColPlan_{slug(phenotype)[:18]}
#SBATCH --nodes=1
{slurm_site(args, 'coloc_planner')}
#SBATCH --time={args.planner_time}
#SBATCH --mem={args.planner_memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1
#SBATCH --output={paths['LOG']}/auto_planner.%j.out
#SBATCH --error={paths['LOG']}/auto_planner.%j.err
set -euo pipefail
cd {shlex.quote(str(root))}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} \\
  --ancestry {shlex.quote(ancestry_code)} \\
  --qtl-types {shlex.quote(args.qtl_types)} \\
  --candidate-p {args.candidate_p} \\
  --max-candidates-per-context-locus {args.max_candidates_per_context_locus} \\
  --strong-pp4 {args.strong_pp4} \\
  --suggestive-pp4 {args.suggestive_pp4} \\
  --catalog-study-regex {shlex.quote(args.catalog_study_regex)} \\
  {'--catalog-all-studies' if args.catalog_all_studies else ''} \\
  --extract-chunksize {args.extract_chunksize} \\
  --candidates-per-task {args.candidates_per_task} \\
  --planner-worker
""",
        encoding="utf-8",
    )

    planner_bash.chmod(
        0o755
    )

    # --------------------------------------------------------
    # Extraction fixed worker pool
    # --------------------------------------------------------

    extract_bash = (
        root
        / (
            f"Step11_Auto_Extract_"
            f"{slug(phenotype)}_"
            f"{slug(ancestry_label)}.sh"
        )
    )

    extract_bash.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=PColExtA_{slug(phenotype)[:18]}
#SBATCH --nodes=1
{slurm_site(args, 'coloc_extraction')}
#SBATCH --time={args.extract_time}
#SBATCH --mem={args.extract_memory}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --ntasks=1
#SBATCH --array=1-{extract_pool}{extract_throttle}
#SBATCH --output={paths['LOG']}/auto_extract.%A_%a.out
#SBATCH --error={paths['LOG']}/auto_extract.%A_%a.err
set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_PROVIDER_EXTRACT_MANIFEST={shlex.quote(str(ready_extract_manifest))}
export GWAS_PROVIDER_EXTRACT_POOL_SIZE={extract_pool}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} \\
  --ancestry {shlex.quote(ancestry_code)} \\
  --extract-chunksize {args.extract_chunksize} \\
  --extract-worker
""",
        encoding="utf-8",
    )

    extract_bash.chmod(
        0o755
    )

    # --------------------------------------------------------
    # Fixed coloc worker pool.
    #
    # The planner may create 728 logical tasks, or 5,000, etc.
    # Array worker i processes logical tasks:
    #
    #   i, i + pool_size, i + 2*pool_size, ...
    #
    # Therefore no nested sbatch and no need to know candidate count now.
    # --------------------------------------------------------

    coloc_bash = (
        root
        / (
            f"Step11_Auto_Coloc_"
            f"{slug(phenotype)}_"
            f"{slug(ancestry_label)}.sh"
        )
    )

    coloc_bash.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=PColocA_{slug(phenotype)[:18]}
#SBATCH --nodes=1
{slurm_site(args, 'coloc')}
#SBATCH --time={args.coloc_time}
#SBATCH --mem={args.coloc_memory}
#SBATCH --cpus-per-task={slurm_stage(args, 'coloc')['cpus']}
#SBATCH --ntasks=1
#SBATCH --array=1-{coloc_pool}{coloc_throttle}
#SBATCH --output={paths['LOG']}/auto_coloc.%A_%a.out
#SBATCH --error={paths['LOG']}/auto_coloc.%A_%a.err
set -euo pipefail
cd {shlex.quote(str(root))}
export GWAS_PROVIDER_COLOC_MANIFEST={shlex.quote(str(ready_coloc_manifest))}
export GWAS_PROVIDER_COLOC_POOL_SIZE={coloc_pool}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} \\
  --ancestry {shlex.quote(ancestry_code)} \\
  --strong-pp4 {args.strong_pp4} \\
  --suggestive-pp4 {args.suggestive_pp4} \\
  --candidate-worker
""",
        encoding="utf-8",
    )

    coloc_bash.chmod(
        0o755
    )

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    aggregate_bash = (
        root
        / (
            f"Step11_Auto_Aggregate_"
            f"{slug(phenotype)}_"
            f"{slug(ancestry_label)}.sh"
        )
    )

    aggregate_bash.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=PColAggA_{slug(phenotype)[:18]}
#SBATCH --nodes=1
{slurm_site(args, 'aggregation')}
#SBATCH --time={slurm_stage(args, 'aggregation')['time']}
#SBATCH --mem={slurm_stage(args, 'aggregation')['memory']}
#SBATCH --cpus-per-task={slurm_stage(args, 'aggregation')['cpus']}
#SBATCH --ntasks=1
#SBATCH --output={paths['LOG']}/auto_aggregate.%j.out
#SBATCH --error={paths['LOG']}/auto_aggregate.%j.err
set -euo pipefail
cd {shlex.quote(str(root))}
{shlex.quote(str(py))} {shlex.quote(str(script))} \\
  --phenotype {shlex.quote(phenotype)} \\
  --ancestry {shlex.quote(ancestry_code)} \\
  --strong-pp4 {args.strong_pp4} \\
  --suggestive-pp4 {args.suggestive_pp4} \\
  --aggregate
""",
        encoding="utf-8",
    )

    aggregate_bash.chmod(
        0o755
    )

    banner(
        "STEP11 FULL PREWIRED DAG"
    )

    print(
        f"Missing discovery loci : {len(missing)}"
    )

    print(
        f"Provider contexts      : {n_provider_contexts}"
    )

    print(
        f"Extraction pool        : {extract_pool}"
    )

    print(
        f"Coloc worker pool      : {coloc_pool}"
    )

    print(
        "Nested sbatch          : DISABLED"
    )

    print(
        "Submission location    : login node only"
    )

    if args.plan_only:

        print(
            "PLAN ONLY: scripts created; no jobs submitted."
        )

        return {
            "SUBMITTED":
                False,

            "PLAN_ONLY":
                True,
        }

    # --------------------------------------------------------
    # ALL sbatch calls happen HERE on the login node.
    # --------------------------------------------------------

    discovery_job = ""

    if not missing.empty:

        discovery_job = _sbatch(
            disc_script
        )

        planner_dependency = (
            f"afterok:{discovery_job}"
        )

    else:

        planner_dependency = ""

    planner_job = _sbatch(
        planner_bash,
        dependency=planner_dependency,
    )

    extract_job = _sbatch(
        extract_bash,
        dependency=(
            f"afterok:{planner_job}"
        ),
    )

    coloc_job = _sbatch(
        coloc_bash,
        dependency=(
            f"afterok:{extract_job}"
        ),
    )

    agg_job = _sbatch(
        aggregate_bash,
        dependency=(
            f"afterok:{coloc_job}"
        ),
    )

    payload = {
        "VERSION":
            VERSION,

        "PHENOTYPE":
            phenotype,

        "ANCESTRY":
            ancestry_code,

        "SUBMITTED_UTC":
            utcnow(),

        "N_MISSING_DISCOVERY_LOCI":
            len(missing),

        "N_PROVIDER_CONTEXTS":
            n_provider_contexts,

        "EXTRACT_POOL_SIZE":
            extract_pool,

        "COLOC_POOL_SIZE":
            coloc_pool,

        "DISCOVERY_JOB":
            discovery_job,

        "PLANNER_JOB":
            planner_job,

        "EXTRACT_JOB":
            extract_job,

        "COLOC_JOB":
            coloc_job,

        "AGG_JOB":
            agg_job,
    }

    write_json(
        state_file,
        payload,
    )

    banner(
        "STEP11 FULL DAG SUBMITTED"
    )

    if discovery_job:

        print(
            f"Discovery   : {discovery_job}"
        )

    else:

        print(
            "Discovery   : already complete"
        )

    print(
        f"Planner     : {planner_job}"
    )

    print(
        f"Extraction  : {extract_job}"
    )

    print(
        f"Coloc       : {coloc_job}"
    )

    print(
        f"Aggregation : {agg_job}"
    )

    print()
    print(
        "All downstream jobs were submitted from the login node."
    )

    print(
        "Compute-node planner performs candidate/CS planning only; "
        "it never runs sbatch."
    )

    print(
        "Monitor: squeue --me"
    )

    return payload


def submit_discovery_autopilot(
    root: Path,
    args,
    missing: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compatibility wrapper.

    v5.4 submits the complete DAG rather than a compute-node continuation job.
    """

    return submit_full_dag(
        root,
        args,
        missing,
    )


def auto_pipeline(
    root: Path,
    args,
) -> None:
    """
    One-command public interface.

    v5.4 PREWIRES the entire dependency DAG from the login node:

      discovery -> planner -> extraction -> coloc -> aggregation

    The planner still runs on a compute node, but never submits jobs.
    """

    phenotype = (
        args.phenotype.strip()
    )

    _, ancestry_label = (
        core.canonical_ancestry(
            args.ancestry
        )
    )

    paths = out_paths(
        root,
        phenotype,
        ancestry_label,
    )

    manifest = (
        paths["BASE"]
        / "discovery_manifest.tsv"
    )

    if not manifest.exists():

        build_discovery_plan(
            root,
            args,
        )

    _, missing = gather_discovery(
        root,
        args,
    )

    submit_full_dag(
        root,
        args,
        missing,
    )


# -----------------------------------------------------------------------------
# Stage 2 worker: stream one huge LBF once, write only needed trait caches
# -----------------------------------------------------------------------------

def resolve_manifest_row(path: Path, id_col: str, index: int) -> pd.Series:
    x = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    ids = pd.to_numeric(x[id_col], errors="coerce")
    z = x[ids == int(index)]
    if len(z) != 1:
        raise RuntimeError(f"Expected one row for {id_col}={index}; found {len(z)}")
    return z.iloc[0]


def extract_provider_dataset(root: Path, args, row: pd.Series) -> None:
    """Stream one provider LBF and atomically publish only required trait caches."""
    dataset_id = str(row["DATASET_ID"])
    source = Path(str(row["PROVIDER_LBF_PATH"]))
    needed_file = Path(str(row["NEEDED_TRAITS_FILE"]))
    out_dir = needed_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "extraction_done.json"

    needed = {x.strip() for x in needed_file.read_text().splitlines() if x.strip()}
    if not needed:
        raise RuntimeError(f"No required molecular traits in {needed_file}")

    if done.exists() and not args.force:
        mf = out_dir / "trait_cache_manifest.tsv"
        complete_traits = set()
        if mf.exists():
            try:
                oldmf = pd.read_csv(mf, sep="\t", dtype=str, keep_default_na=False)
                status_series = oldmf["STATUS"] if "STATUS" in oldmf.columns else pd.Series("", index=oldmf.index)
                ok = oldmf[status_series.astype(str).eq("COMPLETE")]
                for _, rr in ok.iterrows():
                    cp = Path(str(rr.get("CACHE_PATH", "")))
                    if is_nonempty(cp):
                        complete_traits.add(str(rr.get("MOLECULAR_TRAIT_ID", "")))
            except Exception:
                complete_traits = set()
        if needed.issubset(complete_traits):
            print(f"[RESUME] provider extraction {dataset_id}: {len(needed)} required traits already cached")
            return
        print(f"[REFRESH] provider extraction {dataset_id}: cache lacks {len(needed - complete_traits)} newly required trait(s)")

    banner(f"PROVIDER LBF EXTRACTION -- {dataset_id}")
    if not is_nonempty(source):
        raise RuntimeError(f"Provider LBF is missing/empty: {source}")
    if Path(str(source) + ".aria2").exists():
        raise RuntimeError(f"Provider LBF is still downloading (.aria2 exists): {source}")

    header = pd.read_csv(source, sep="\t", compression="infer", nrows=0)
    lower = {str(c).lower(): c for c in header.columns}
    trait_col = lower.get("molecular_trait_id")
    variant_col = lower.get("variant") or lower.get("variant_id")
    lbf_cols = [c for c in header.columns if str(c).startswith("lbf_variable")]
    if not trait_col or not variant_col or not lbf_cols:
        raise RuntimeError(f"Unexpected provider LBF format for {source}; columns={list(header.columns)[:30]}")

    usecols = [trait_col, variant_col] + lbf_cols
    row_counts = {trait: 0 for trait in needed}
    paths = out_paths(root, args.phenotype, core.canonical_ancestry(args.ancestry)[1])
    cache_paths = {trait: trait_cache_path(paths, dataset_id, trait) for trait in needed}
    tmp_paths = {
        trait: p.with_name(f".{p.name}.tmp.{os.getpid()}")
        for trait, p in cache_paths.items()
    }

    for p in cache_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    for p in tmp_paths.values():
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    handles: dict[str, Any] = {}
    writers_started: set[str] = set()
    total_rows = 0
    selected_rows = 0

    try:
        for chunk_no, chunk in enumerate(
            pd.read_csv(
                source,
                sep="\t",
                compression="infer",
                usecols=usecols,
                dtype={trait_col: str, variant_col: str},
                chunksize=max(10_000, int(args.extract_chunksize)),
                low_memory=False,
            ),
            start=1,
        ):
            total_rows += len(chunk)
            sub = chunk[chunk[trait_col].astype(str).isin(needed)].copy()
            if sub.empty:
                if chunk_no % 10 == 0:
                    print(f"[STREAM] {dataset_id}: rows={total_rows:,}, selected={selected_rows:,}", flush=True)
                continue

            selected_rows += len(sub)
            for trait, group in sub.groupby(trait_col, sort=False):
                trait = str(trait)
                p = tmp_paths[trait]
                if trait not in handles:
                    handles[trait] = gzip.open(p, "wt", encoding="utf-8", newline="")
                group.to_csv(
                    handles[trait],
                    sep="\t",
                    index=False,
                    header=(trait not in writers_started),
                )
                writers_started.add(trait)
                row_counts[trait] += len(group)

            if chunk_no % 5 == 0:
                print(f"[STREAM] {dataset_id}: rows={total_rows:,}, selected={selected_rows:,}", flush=True)
    finally:
        for h in handles.values():
            try:
                h.close()
            except Exception:
                pass

    # Publish atomically. A running coloc never observes a deleted/half-written cache.
    for trait in sorted(needed):
        tmp = tmp_paths[trait]
        final = cache_paths[trait]
        if row_counts.get(trait, 0) > 0 and is_nonempty(tmp):
            os.replace(tmp, final)
        else:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    summary_rows = []
    for trait in sorted(needed):
        p = cache_paths[trait]
        summary_rows.append({
            "DATASET_ID": dataset_id,
            "MOLECULAR_TRAIT_ID": trait,
            "CACHE_PATH": str(p),
            "N_ROWS": row_counts.get(trait, 0),
            "STATUS": "COMPLETE" if is_nonempty(p) and row_counts.get(trait, 0) > 0 else "TRAIT_NOT_FOUND",
        })

    mf = out_dir / "trait_cache_manifest.tsv"
    mf_tmp = mf.with_name(f".{mf.name}.tmp.{os.getpid()}")
    pd.DataFrame(summary_rows).to_csv(mf_tmp, sep="\t", index=False)
    os.replace(mf_tmp, mf)

    n_found = sum(v > 0 for v in row_counts.values())
    write_json(done, {
        "VERSION": VERSION,
        "STATUS": "COMPLETE",
        "DATASET_ID": dataset_id,
        "SOURCE": str(source),
        "N_REQUIRED_TRAITS": len(needed),
        "N_FOUND_TRAITS": n_found,
        "N_SOURCE_ROWS_SCANNED": total_rows,
        "N_SELECTED_ROWS": selected_rows,
        "COMPLETED_UTC": utcnow(),
    })
    print(f"[OK] {dataset_id}: found {n_found}/{len(needed)} needed traits; scanned {total_rows:,} rows")


# -----------------------------------------------------------------------------
# Formal coloc R worker: Step06 GWAS SuSiE LBF x provider QTL SuSiE LBF
# -----------------------------------------------------------------------------

R_WORKER = r'''
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 11) stop("Expected 11 arguments")

gwas_fit_file <- args[1]
gwas_map_file <- args[2]
qtl_lbf_file <- args[3]
out_summary <- args[4]
out_results <- args[5]
out_sensitivity <- args[6]
p1 <- as.numeric(args[7])
p2 <- as.numeric(args[8])
p12 <- as.numeric(args[9])
p12_low <- as.numeric(args[10])
p12_high <- as.numeric(args[11])

suppressPackageStartupMessages(library(coloc))

fit <- readRDS(gwas_fit_file)
map <- read.delim(gwas_map_file, stringsAsFactors=FALSE, check.names=FALSE)
qtl <- read.delim(qtl_lbf_file, stringsAsFactors=FALSE, check.names=FALSE)

if (is.null(fit$lbf_variable)) stop("Step06 SuSiE object has no lbf_variable")
if (!all(c("VARIANT_KEY") %in% names(map))) stop("GWAS variant map lacks VARIANT_KEY")

# Step06 susieR stores signal x variant LBFs. Map them to Step06 variant order.
gbf <- as.matrix(fit$lbf_variable)
if (ncol(gbf) == nrow(map)) {
  colnames(gbf) <- map$VARIANT_KEY
} else if (nrow(gbf) == nrow(map)) {
  gbf <- t(gbf)
  colnames(gbf) <- map$VARIANT_KEY
} else {
  stop(sprintf("GWAS lbf_variable dimensions %dx%d do not match map rows=%d",
               nrow(gbf), ncol(gbf), nrow(map)))
}

# Match coloc.susie semantics: only real SuSiE credible-set components are
# eligible, not all L components. susieR cs_index is 1-based.
gidx <- tryCatch(as.integer(fit$sets$cs_index), error=function(e) integer())
gidx <- unique(gidx[is.finite(gidx) & gidx >= 1 & gidx <= nrow(gbf)])
if (length(gidx) == 0) stop("Step06 GWAS SuSiE has no eligible credible-set components")
gbf <- gbf[gidx, , drop=FALSE]
rownames(gbf) <- paste0("L", gidx)

lbf_cols <- grep("^lbf_variable", names(qtl), value=TRUE)
if (length(lbf_cols) == 0) stop("Provider QTL cache has no lbf_variable columns")
if (!("VARIANT_KEY" %in% names(qtl))) stop("Provider QTL cache lacks VARIANT_KEY")

qbf <- t(as.matrix(qtl[, lbf_cols, drop=FALSE]))
colnames(qbf) <- qtl$VARIANT_KEY
rownames(qbf) <- sub("^lbf_variable", "L", lbf_cols)

# Remove duplicate/empty SNP IDs before coloc matches the matrices.
keepg <- !is.na(colnames(gbf)) & nzchar(colnames(gbf)) & !duplicated(colnames(gbf))
keepq <- !is.na(colnames(qbf)) & nzchar(colnames(qbf)) & !duplicated(colnames(qbf))
gbf <- gbf[, keepg, drop=FALSE]
qbf <- qbf[, keepq, drop=FALSE]

# Drop signal rows with no finite information.
gbf <- gbf[apply(gbf, 1, function(z) any(is.finite(z))), , drop=FALSE]
qbf <- qbf[apply(qbf, 1, function(z) any(is.finite(z))), , drop=FALSE]
if (nrow(gbf) == 0) stop("GWAS SuSiE has no finite LBF signal rows")
if (nrow(qbf) == 0) stop("QTL SuSiE has no finite LBF signal rows")

common <- intersect(colnames(gbf), colnames(qbf))
if (length(common) == 0) stop("GWAS and provider QTL SuSiE LBFs share zero variants")
cat("GWAS_SIGNALS=", nrow(gbf), " QTL_SIGNALS=", nrow(qbf),
    " COMMON_VARIANTS=", length(common), "\n", sep="")

run_one <- function(prior12, label) {
  z <- coloc::coloc.bf_bf(
    gbf, qbf,
    p1=p1, p2=p2, p12=prior12,
    overlap.min=0.5,
    trim_by_posterior=TRUE
  )
  sm <- as.data.frame(z$summary)
  if (nrow(sm) == 0) return(NULL)
  sm$PRIOR_SET <- label
  sm$P12 <- prior12
  sm$METHOD <- "provider_susie_lbf_x_step06_susie_lbf"
  list(obj=z, summary=sm)
}

base <- run_one(p12, "DEFAULT")
if (is.null(base)) stop("No eligible signal pair survived coloc.bf_bf overlap filtering")
low <- run_one(p12_low, "LOW_P12")
high <- run_one(p12_high, "HIGH_P12")

write.table(base$summary, out_summary, sep="\t", quote=FALSE, row.names=FALSE)

sens <- base$summary
if (!is.null(low)) sens <- rbind(sens, low$summary)
if (!is.null(high)) sens <- rbind(sens, high$summary)
write.table(sens, out_sensitivity, sep="\t", quote=FALSE, row.names=FALSE)

rr <- tryCatch(as.data.frame(base$obj$results), error=function(e) data.frame())
write.table(rr, out_results, sep="\t", quote=FALSE, row.names=FALSE)
print(base$summary)
'''


def write_r_worker(paths: dict[str, Path]) -> Path:
    """Publish the shared R worker atomically; safe under many coloc tasks."""
    p = paths["BASE"] / "_provider_susie_coloc_bf_bf_worker.R"
    try:
        if p.exists() and p.read_text(encoding="utf-8") == R_WORKER:
            return p
    except Exception:
        pass
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    tmp.write_text(R_WORKER, encoding="utf-8")
    os.replace(tmp, p)
    return p


def prepare_qtl_trait_for_r(
    cache: Path,
    out_file: Path,
    qtl_cs_indexes: str,
) -> tuple[int, int, list[str]]:
    x = pd.read_csv(cache, sep="\t", compression="infer", dtype=str, keep_default_na=False)
    if x.empty:
        return 0, 0, []
    cols = {c.lower(): c for c in x.columns}
    variant_col = cols.get("variant")
    if not variant_col:
        raise RuntimeError(f"Trait cache has no variant column: {cache}")

    wanted_components = sorted({
        z for z in (parse_cs_component(v) for v in str(qtl_cs_indexes).split(","))
        if z is not None
    })
    if not wanted_components:
        raise RuntimeError(f"No valid provider QTL credible-set components supplied: {qtl_cs_indexes}")
    wanted_cols = [f"lbf_variable{i}" for i in wanted_components]
    available = [c for c in wanted_cols if c in x.columns]
    if not available:
        raise RuntimeError(
            f"Provider QTL trait cache has none of the requested CS LBF columns. "
            f"wanted={wanted_cols}, available={[c for c in x.columns if str(c).startswith('lbf_variable')]}"
        )

    parsed = x[variant_col].map(core.parse_variant_id)
    x["VARIANT_KEY"] = [core.make_variant_key(v[0], v[1], v[2], v[3]) for v in parsed]
    n_before = len(x)
    x = x[x["VARIANT_KEY"].astype(str).str.len() > 0].copy()
    x = x.drop_duplicates("VARIANT_KEY", keep="first")
    for c in available:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    keep = x[available].notna().any(axis=1)
    x = x[keep].copy()
    if x.empty:
        return n_before, 0, available
    out_file.parent.mkdir(parents=True, exist_ok=True)
    x[["VARIANT_KEY"] + available].to_csv(out_file, sep="\t", index=False, compression="gzip")
    return n_before, len(x), available


def candidate_result_template(row: pd.Series, out: Path) -> dict:
    return {
        "VERSION": VERSION,
        "TASK_ID": str(row.get("TASK_ID", "")),
        "CANDIDATE_ID": str(row.get("CANDIDATE_ID", "")),
        "PHENOTYPE": str(row.get("PHENOTYPE", "")),
        "ANCESTRY_CODE": str(row.get("ANCESTRY_CODE", "")),
        "STUDY_ACCESSION": str(row.get("STUDY_ACCESSION", "")),
        "LOCUS_ID": str(row.get("LOCUS_ID", "")),
        "DATASET": str(row.get("DATASET", "")),
        "DATASET_ID": str(row.get("DATASET_ID", "")),
        "QTL_TYPE": str(row.get("QTL_TYPE", "")),
        "TISSUE": str(row.get("TISSUE", "")),
        "CONDITION": str(row.get("CONDITION", "")),
        "MOLECULAR_TRAIT_ID": str(row.get("MOLECULAR_TRAIT_ID", "")),
        "GENE_ID": str(row.get("GENE_ID", "")),
        "DISCOVERY_MIN_P": row.get("DISCOVERY_MIN_P", ""),
        "PROVIDER_LBF_PATH": str(row.get("PROVIDER_LBF_PATH", "")),
        "TRAIT_CACHE_PATH": str(row.get("TRAIT_CACHE_PATH", "")),
        "GWAS_CS_INDEXES": str(row.get("GWAS_CS_INDEXES", "")),
        "QTL_CS_INDEXES": str(row.get("QTL_CS_INDEXES", "")),
        "N_GWAS_CS": int(row.get("N_GWAS_CS", 0) or 0),
        "N_QTL_CS": int(row.get("N_QTL_CS", 0) or 0),
        "QTL_LBF_COLUMNS_USED": "",
        "COLOC_METHOD": "provider_susie_lbf_x_step06_susie_lbf_cs_filtered",
        "QTL_LD_COMPUTED": "NO",
        "QTL_RUNSUSIE_COMPUTED": "NO",
        "COLOC_TESTED": "NO",
        "N_QTL_LBF_ROWS_RAW": 0,
        "N_QTL_LBF_VARIANTS": 0,
        "MAX_PP_H0": np.nan,
        "MAX_PP_H1": np.nan,
        "MAX_PP_H2": np.nan,
        "MAX_PP_H3": np.nan,
        "MAX_PP_H4": np.nan,
        "BEST_PAIR_PP_H0": np.nan,
        "BEST_PAIR_PP_H1": np.nan,
        "BEST_PAIR_PP_H2": np.nan,
        "BEST_PAIR_PP_H3": np.nan,
        "BEST_PAIR_PP_H4": np.nan,
        "BEST_PAIR_DOMINANT_HYPOTHESIS": "",
        "BEST_PAIR_POSTERIOR_SUM": np.nan,
        "PP_H4_P12_LOW": np.nan,
        "PP_H4_P12_DEFAULT": np.nan,
        "PP_H4_P12_HIGH": np.nan,
        "PRIOR_ROBUST_STRONG": "NO",
        "BEST_HIT1": "",
        "BEST_HIT2": "",
        "BEST_SIGNAL_PAIR": "",
        "COLOCALIZED_STRONG": "NO",
        "SHARED_SIGNAL": "UNKNOWN",
        "COLOC_CLASS": "",
        "STATUS": "",
        "ERROR": "",
        "OUTPUT_DIR": str(out),
    }


def run_candidate(root: Path, args, row: pd.Series) -> dict:
    import time

    _, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, args.phenotype.strip(), ancestry_label)
    cid = str(row["CANDIDATE_ID"])
    out = paths["RESULTS"] / cid
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / "candidate_result.json"

    terminal_not_tested = {
        "NOT_TESTED_NO_FINITE_H4",
        "NOT_TESTED_NO_SHARED_VARIANTS",
        "NOT_TESTED_INSUFFICIENT_POSTERIOR_OVERLAP",
        "NOT_TESTED_NO_PARSEABLE_PROVIDER_LBF_VARIANTS",
        "NOT_TESTED_NO_FINITE_LBF_SIGNAL",
    }

    if result_file.exists() and not args.force:
        old = read_json(result_file)
        if old.get("STATUS") == "COMPLETE" or old.get("STATUS") in terminal_not_tested:
            print(f"[RESUME] {cid} status={old.get('STATUS')}")
            return old

    result = candidate_result_template(row, out)
    result.update({
        "ERROR_CODE": "",
        "MISSING_FILE": "",
        "QC_WARNING": "",
        "N_GWAS_SIGNALS_USED": 0,
        "N_QTL_SIGNALS_USED": 0,
        "N_COMMON_VARIANTS": 0,
        "BEST_PAIR_PP_H0": np.nan,
        "BEST_PAIR_PP_H1": np.nan,
        "BEST_PAIR_PP_H2": np.nan,
        "BEST_PAIR_PP_H3": np.nan,
        "BEST_PAIR_PP_H4": np.nan,
        "BEST_PAIR_DOMINANT_HYPOTHESIS": "",
        "BEST_PAIR_POSTERIOR_SUM": np.nan,
        "BEST_GWAS_VARIANT": "",
        "BEST_QTL_VARIANT": "",
    })

    def finish(status: str, error: str = "", error_code: str = "", missing_file: str = "") -> dict:
        result["STATUS"] = status
        result["ERROR_CODE"] = error_code
        result["MISSING_FILE"] = missing_file
        if error:
            result["ERROR"] = error
        write_json(result_file, result)
        return result

    def tail(path: Path, n: int = 120_000) -> str:
        try:
            with path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - n))
                return fh.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def parse_counts(text: str) -> None:
        m = re.search(r"GWAS_SIGNALS=(\d+)\s+QTL_SIGNALS=(\d+)\s+COMMON_VARIANTS=(\d+)", text)
        if m:
            result["N_GWAS_SIGNALS_USED"] = int(m.group(1))
            result["N_QTL_SIGNALS_USED"] = int(m.group(2))
            result["N_COMMON_VARIANTS"] = int(m.group(3))

    def read_tsv_retry(path: Path, attempts: int = 4) -> pd.DataFrame:
        last = None
        for attempt in range(attempts):
            try:
                if not is_nonempty(path):
                    raise FileNotFoundError(2, "No such file or empty file", str(path))
                return pd.read_csv(path, sep="\t", low_memory=False)
            except FileNotFoundError as exc:
                last = exc
                time.sleep(attempt + 1)
        raise last if last is not None else FileNotFoundError(str(path))

    try:
        cache = Path(str(row["TRAIT_CACHE_PATH"]))
        if not is_nonempty(cache):
            return finish(
                "NOT_TESTED_PROVIDER_TRAIT_CACHE_MISSING",
                f"Provider trait cache missing/empty: {cache}",
                "PROVIDER_TRAIT_CACHE_MISSING",
                str(cache),
            )

        qtl_for_r = out / "qtl_provider_lbf.tsv.gz"
        try:
            if qtl_for_r.exists():
                qtl_for_r.unlink()
        except Exception:
            pass

        prep_error = None
        for attempt in range(4):
            try:
                nraw, nkeep, used_cols = prepare_qtl_trait_for_r(
                    cache,
                    qtl_for_r,
                    str(row.get("QTL_CS_INDEXES", "")),
                )
                prep_error = None
                break
            except FileNotFoundError as exc:
                prep_error = exc
                time.sleep(attempt + 1)
        if prep_error is not None:
            missing = str(getattr(prep_error, "filename", "") or cache)
            return finish(
                "NOT_TESTED_PROVIDER_TRAIT_CACHE_MISSING",
                f"Provider trait cache disappeared during read: {cache}; {prep_error}",
                "PROVIDER_TRAIT_CACHE_RACE_OR_MISSING",
                missing,
            )

        result["N_QTL_LBF_ROWS_RAW"] = nraw
        result["N_QTL_LBF_VARIANTS"] = nkeep
        result["QTL_LBF_COLUMNS_USED"] = ",".join(used_cols)
        if nkeep == 0:
            return finish(
                "NOT_TESTED_NO_PARSEABLE_PROVIDER_LBF_VARIANTS",
                error_code="NO_PARSEABLE_PROVIDER_LBF_VARIANTS",
            )

        gwas_fit = Path(str(row["GWAS_SUSIE_FIT"]))
        gwas_map = Path(str(row["GWAS_VARIANT_MAP"]))
        if not is_nonempty(gwas_fit) or not is_nonempty(gwas_map):
            missing = gwas_fit if not is_nonempty(gwas_fit) else gwas_map
            return finish(
                "NOT_TESTED_STEP06_SUSIE_MISSING",
                f"Step06 SuSiE input missing/empty: {missing}",
                "STEP06_SUSIE_INPUT_MISSING",
                str(missing),
            )

        rscript = core.find_executable(root, args.rscript or None, "RSCRIPT", "Rscript")
        worker = write_r_worker(paths)
        summary_file = out / "coloc_summary.tsv"
        results_file = out / "coloc_variant_results.tsv"
        sensitivity_file = out / "prior_sensitivity.tsv"
        log_file = out / "coloc.log"

        for stale in (summary_file, results_file, sensitivity_file):
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass

        try:
            run(
                [
                    rscript, worker, gwas_fit, gwas_map, qtl_for_r,
                    summary_file, results_file, sensitivity_file,
                    P1, P2, P12, P12_LOW, P12_HIGH,
                ],
                log_file=log_file,
            )
        except RuntimeError as exc:
            log_text = tail(log_file)
            parse_counts(log_text)
            low = log_text.lower()
            if "share zero variants" in low:
                return finish(
                    "NOT_TESTED_NO_SHARED_VARIANTS",
                    "GWAS and provider QTL SuSiE LBFs share zero variants",
                    "NO_SHARED_VARIANTS",
                )
            if "no eligible signal pair survived" in low:
                return finish(
                    "NOT_TESTED_INSUFFICIENT_POSTERIOR_OVERLAP",
                    "No eligible signal pair survived coloc.bf_bf posterior-overlap filtering",
                    "INSUFFICIENT_POSTERIOR_OVERLAP",
                )
            if "no finite lbf signal rows" in low or "no finite lbf" in low:
                return finish(
                    "NOT_TESTED_NO_FINITE_LBF_SIGNAL",
                    "GWAS or QTL SuSiE had no finite LBF signal rows",
                    "NO_FINITE_LBF_SIGNAL",
                )
            return finish(
                "FAILED_R_COLOC",
                f"{exc}\n--- coloc.log tail ---\n{log_text[-12000:]}",
                "R_COLOC_FAILED",
            )

        log_text = tail(log_file)
        parse_counts(log_text)

        if not is_nonempty(summary_file):
            return finish(
                "FAILED_COLOC_SUMMARY_MISSING",
                f"R coloc worker exited successfully but summary is missing/empty: {summary_file}",
                "COLOC_SUMMARY_MISSING",
                str(summary_file),
            )

        summary = read_tsv_retry(summary_file)
        try:
            cls = classify_bf_summary(summary, args.strong_pp4, args.suggestive_pp4)
        except RuntimeError as exc:
            if "no finite H4 posterior" in str(exc):
                return finish(
                    "NOT_TESTED_NO_FINITE_H4",
                    str(exc),
                    "NO_FINITE_H4",
                )
            raise

        result.update(cls)
        result["COLOC_TESTED"] = "YES"

        if is_nonempty(sensitivity_file):
            sens = read_tsv_retry(sensitivity_file)
            result["PP_H4_P12_LOW"] = core.sensitivity_pp4_for_pair(
                sens, "LOW_P12", cls.get("BEST_HIT1", ""), cls.get("BEST_HIT2", "")
            )
            result["PP_H4_P12_DEFAULT"] = core.sensitivity_pp4_for_pair(
                sens, "DEFAULT", cls.get("BEST_HIT1", ""), cls.get("BEST_HIT2", "")
            )
            result["PP_H4_P12_HIGH"] = core.sensitivity_pp4_for_pair(
                sens, "HIGH_P12", cls.get("BEST_HIT1", ""), cls.get("BEST_HIT2", "")
            )
            low = result["PP_H4_P12_LOW"]
            result["PRIOR_ROBUST_STRONG"] = (
                "YES"
                if pd.notna(low) and np.isfinite(float(low)) and float(low) >= args.strong_pp4
                else "NO"
            )
        else:
            result["QC_WARNING"] = "PRIOR_SENSITIVITY_FILE_MISSING"
            result["PRIOR_ROBUST_STRONG"] = "UNKNOWN"

        return finish("COMPLETE")

    except FileNotFoundError as exc:
        missing = str(getattr(exc, "filename", "") or "")
        return finish(
            "FAILED_FILE_NOT_FOUND",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "UNEXPECTED_FILE_NOT_FOUND",
            missing,
        )
    except Exception as exc:
        return finish(
            "FAILED",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            type(exc).__name__,
        )


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate(root: Path, args) -> None:
    _, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, args.phenotype.strip(), ancestry_label)
    manifest_file = paths["BASE"] / "provider_coloc_manifest.tsv"
    if not manifest_file.exists():
        raise RuntimeError("provider_coloc_manifest.tsv is missing; run --build-provider-plan first")

    manifest = pd.read_csv(manifest_file, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    for row in manifest.itertuples(index=False):
        f = paths["RESULTS"] / str(row.CANDIDATE_ID) / "candidate_result.json"
        payload = read_json(f) if f.exists() else {}
        if payload:
            rows.append(payload)
        else:
            rows.append({
                "CANDIDATE_ID": str(row.CANDIDATE_ID),
                "STUDY_ACCESSION": str(row.STUDY_ACCESSION),
                "LOCUS_ID": str(row.LOCUS_ID),
                "DATASET_ID": str(row.DATASET_ID),
                "QTL_TYPE": str(getattr(row, "QTL_TYPE", "")),
                "TISSUE": str(getattr(row, "TISSUE", "")),
                "MOLECULAR_TRAIT_ID": str(row.MOLECULAR_TRAIT_ID),
                "STATUS": "MISSING_RESULT",
                "COLOC_TESTED": "NO",
            })

    x = pd.DataFrame(rows)
    out = paths["BASE"] / "all_provider_susie_coloc.tsv"
    x.to_csv(out, sep="\t", index=False)

    # Long, candidate-level master table: this is the Step11 side of the later annotation join.
    master_long = paths["BASE"] / "step11_master_coloc_long.tsv"
    x.to_csv(master_long, sep="\t", index=False)

    strong_mask = x.get(
        "COLOCALIZED_STRONG",
        pd.Series(index=x.index, dtype=str),
    ).astype(str).eq("YES")
    strong = x[strong_mask].copy()
    strong.to_csv(paths["BASE"] / "strong_provider_susie_colocalizations.tsv", sep="\t", index=False)

    # One row per GWAS study/locus, with best result for each molecular-QTL modality.
    locus_rows = []
    if not x.empty and {"STUDY_ACCESSION", "LOCUS_ID"}.issubset(x.columns):
        for (study, locus), g in x.groupby(["STUDY_ACCESSION", "LOCUS_ID"], dropna=False, sort=True):
            z = {
                "PHENOTYPE": args.phenotype,
                "ANCESTRY": ancestry_label,
                "STUDY_ACCESSION": study,
                "LOCUS_ID": locus,
                "N_CANDIDATES": len(g),
                "N_TESTED": int(g.get("COLOC_TESTED", pd.Series(index=g.index, dtype=str)).astype(str).eq("YES").sum()),
                "N_STRONG": int(g.get("COLOCALIZED_STRONG", pd.Series(index=g.index, dtype=str)).astype(str).eq("YES").sum()),
            }

            for qtype in ["eQTL", "sQTL", "pQTL", "isoQTL", "exonQTL"]:
                prefix = qtype.upper()
                qt = g.get("QTL_TYPE", pd.Series(index=g.index, dtype=str)).astype(str)
                qg = g[qt.str.lower().eq(qtype.lower())].copy()
                z[f"{prefix}_CANDIDATES"] = len(qg)
                z[f"{prefix}_TESTED"] = int(
                    qg.get("COLOC_TESTED", pd.Series(index=qg.index, dtype=str)).astype(str).eq("YES").sum()
                ) if len(qg) else 0

                h4 = pd.to_numeric(qg.get("BEST_PAIR_PP_H4", qg.get("MAX_PP_H4", pd.Series(index=qg.index, dtype=float))), errors="coerce") if len(qg) else pd.Series(dtype=float)
                finite = h4[np.isfinite(h4)] if len(h4) else pd.Series(dtype=float)
                if len(finite):
                    idx = finite.idxmax()
                    best = qg.loc[idx]
                    best_h4 = float(finite.loc[idx])
                    z[f"{prefix}_YES"] = "YES" if best_h4 >= args.suggestive_pp4 else "NO"
                    z[f"{prefix}_STRONG"] = "YES" if best_h4 >= args.strong_pp4 else "NO"
                    z[f"{prefix}_BEST_H4"] = best_h4
                    z[f"{prefix}_BEST_TRAIT"] = str(best.get("MOLECULAR_TRAIT_ID", ""))
                    z[f"{prefix}_BEST_TISSUE"] = str(best.get("TISSUE", ""))
                    z[f"{prefix}_BEST_DATASET"] = str(best.get("DATASET_ID", ""))
                    z[f"{prefix}_BEST_GWAS_VARIANT"] = str(best.get("BEST_GWAS_VARIANT", best.get("BEST_HIT1", "")))
                    z[f"{prefix}_BEST_QTL_VARIANT"] = str(best.get("BEST_QTL_VARIANT", best.get("BEST_HIT2", "")))
                    z[f"{prefix}_BEST_SIGNAL_PAIR"] = str(best.get("BEST_SIGNAL_PAIR", ""))
                    z[f"{prefix}_PRIOR_ROBUST_STRONG"] = str(best.get("PRIOR_ROBUST_STRONG", ""))
                    z[f"{prefix}_STATUS"] = str(best.get("STATUS", ""))
                else:
                    z[f"{prefix}_YES"] = "NO"
                    z[f"{prefix}_STRONG"] = "NO"
                    z[f"{prefix}_BEST_H4"] = np.nan
                    z[f"{prefix}_BEST_TRAIT"] = ""
                    z[f"{prefix}_BEST_TISSUE"] = ""
                    z[f"{prefix}_BEST_DATASET"] = ""
                    z[f"{prefix}_BEST_GWAS_VARIANT"] = ""
                    z[f"{prefix}_BEST_QTL_VARIANT"] = ""
                    z[f"{prefix}_BEST_SIGNAL_PAIR"] = ""
                    z[f"{prefix}_PRIOR_ROBUST_STRONG"] = ""
                    z[f"{prefix}_STATUS"] = ""

            locus_rows.append(z)

    locus_master = pd.DataFrame(locus_rows)
    locus_master_file = paths["BASE"] / "step11_master_locus_qtl_summary.tsv"
    locus_master.to_csv(locus_master_file, sep="\t", index=False)

    status = x["STATUS"].value_counts(dropna=False).to_dict() if "STATUS" in x.columns else {}
    write_json(paths["BASE"] / "summary.json", {
        "VERSION": VERSION,
        "N_CANDIDATES": len(x),
        "N_TESTED": int(x.get("COLOC_TESTED", pd.Series(index=x.index, dtype=str)).astype(str).eq("YES").sum()),
        "N_STRONG": len(strong),
        "STATUS_COUNTS": status,
        "MASTER_LONG": str(master_long),
        "MASTER_LOCUS_QTL": str(locus_master_file),
        "COMPLETED_UTC": utcnow(),
    })

    banner("PROVIDER-SUSIE COLOCALIZATION AGGREGATED")
    print(f"All results       : {out}")
    print(f"Master long       : {master_long}")
    print(f"Master locus/QTL  : {locus_master_file}")
    print(f"Candidates        : {len(x)}")
    print(f"Tested            : {int(x.get('COLOC_TESTED', pd.Series(index=x.index, dtype=str)).astype(str).eq('YES').sum())}")
    print(f"Strong H4         : {len(strong)}")
    if status:
        print("\nSTATUS COUNTS")
        for k, v in status.items():
            print(f"  {k}: {v}")


# -----------------------------------------------------------------------------
# Worker dispatch
# -----------------------------------------------------------------------------

def discovery_worker(root: Path, args) -> None:
    mf = os.environ.get("GWAS_PROVIDER_DISCOVERY_MANIFEST")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if not mf or not task:
        raise RuntimeError("Discovery worker requires GWAS_PROVIDER_DISCOVERY_MANIFEST and SLURM_ARRAY_TASK_ID")
    row = resolve_manifest_row(Path(mf), "TASK_ID", int(task))
    run_discovery(root, args, row)


def extract_worker(
    root: Path,
    args,
) -> None:
    """
    Fixed extraction pool worker.

    The pool is submitted before the planner creates the ready manifest.
    Dependency ordering guarantees the manifest exists before this worker runs.

    If there are fewer ready datasets than pool workers, extra workers exit 0.
    If there are ever more logical extraction tasks than pool workers, workers
    stride over task IDs.
    """

    mf = os.environ.get(
        "GWAS_PROVIDER_EXTRACT_MANIFEST"
    )

    task = os.environ.get(
        "SLURM_ARRAY_TASK_ID"
    )

    pool = int(
        os.environ.get(
            "GWAS_PROVIDER_EXTRACT_POOL_SIZE",
            "1",
        )
    )

    if not mf or not task:

        raise RuntimeError(
            "Extraction worker requires "
            "GWAS_PROVIDER_EXTRACT_MANIFEST and SLURM_ARRAY_TASK_ID"
        )

    path = Path(
        mf
    )

    if not path.exists():

        raise RuntimeError(
            f"Ready extraction manifest is missing: {path}"
        )

    x = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    if x.empty:

        print(
            "[SKIP] no provider datasets require extraction"
        )

        return

    ids = pd.to_numeric(
        x["TASK_ID"],
        errors="coerce",
    )

    start = int(
        task
    )

    max_id = int(
        ids.max()
    )

    selected_ids = list(
        range(
            start,
            max_id + 1,
            max(
                1,
                pool,
            ),
        )
    )

    rows = (
        x[
            ids.isin(
                selected_ids
            )
        ]
        .copy()
    )

    if rows.empty:

        print(
            f"[SKIP] extraction pool task {start}: "
            "no logical extraction tasks assigned"
        )

        return

    print(
        f"[EXTRACT POOL] array task={start} "
        f"logical_tasks={selected_ids}"
    )

    for _, row in rows.iterrows():

        extract_provider_dataset(
            root,
            args,
            row,
        )


def candidate_worker(
    root: Path,
    args,
) -> None:
    """
    Fixed formal-coloc worker pool.

    The planner dynamically creates WORKER_TASK_ID batches after discovery.
    Array worker i processes:

        i,
        i + POOL_SIZE,
        i + 2*POOL_SIZE,
        ...

    Therefore the complete SLURM DAG can be submitted from the login node
    BEFORE the exact number of formal candidates is known.
    """

    mf = os.environ.get(
        "GWAS_PROVIDER_COLOC_MANIFEST"
    )

    task = os.environ.get(
        "SLURM_ARRAY_TASK_ID"
    )

    pool = int(
        os.environ.get(
            "GWAS_PROVIDER_COLOC_POOL_SIZE",
            "1",
        )
    )

    if not mf or not task:

        raise RuntimeError(
            "Candidate worker requires "
            "GWAS_PROVIDER_COLOC_MANIFEST and SLURM_ARRAY_TASK_ID"
        )

    path = Path(
        mf
    )

    if not path.exists():

        raise RuntimeError(
            f"Ready coloc manifest is missing: {path}"
        )

    x = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    if x.empty:

        print(
            "[SKIP] no formal colocalization candidates are pending"
        )

        return

    if "WORKER_TASK_ID" not in x.columns:

        raise RuntimeError(
            f"{path} lacks WORKER_TASK_ID"
        )

    ids = pd.to_numeric(
        x["WORKER_TASK_ID"],
        errors="coerce",
    )

    start = int(
        task
    )

    max_id = int(
        ids.max()
    )

    logical_ids = list(
        range(
            start,
            max_id + 1,
            max(
                1,
                pool,
            ),
        )
    )

    if not logical_ids:

        print(
            f"[SKIP] coloc pool task {start}: "
            "no logical coloc tasks assigned"
        )

        return

    all_summaries = []

    n_candidates = 0

    for logical_id in logical_ids:

        rows = (
            x[
                ids.eq(
                    logical_id
                )
            ]
            .copy()
        )

        if rows.empty:
            continue

        print(
            f"[COLOC POOL] array task={start} "
            f"logical_task={logical_id} "
            f"candidates={len(rows)}"
        )

        n_candidates += len(
            rows
        )

        for _, row in rows.iterrows():

            result = run_candidate(
                root,
                args,
                row,
            )

            all_summaries.append({
                k: result.get(k)
                for k in [
                    "CANDIDATE_ID",
                    "STATUS",
                    "COLOC_TESTED",
                    "MAX_PP_H4",
                    "BEST_PAIR_PP_H4",
                    "BEST_SIGNAL_PAIR",
                    "COLOC_CLASS",
                    "ERROR_CODE",
                    "MISSING_FILE",
                ]
            })

    if not all_summaries:

        print(
            f"[SKIP] coloc pool task {start}: "
            "all assigned logical task IDs were beyond the dynamic plan"
        )

        return

    print(
        json.dumps(
            {
                "ARRAY_TASK_ID":
                    start,

                "POOL_SIZE":
                    pool,

                "LOGICAL_TASK_IDS":
                    logical_ids,

                "N_CANDIDATES_IN_WORKER":
                    n_candidates,

                "N_COMPLETE":
                    sum(
                        str(
                            z.get(
                                "STATUS"
                            )
                        )
                        == "COMPLETE"
                        for z in all_summaries
                    ),

                "N_TESTED":
                    sum(
                        str(
                            z.get(
                                "COLOC_TESTED"
                            )
                        )
                        == "YES"
                        for z in all_summaries
                    ),

                "RESULTS":
                    all_summaries,
            },
            indent=2,
            default=str,
        )
    )


def direct_extract(root: Path, args, index: int) -> None:
    _, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, args.phenotype.strip(), ancestry_label)
    mf = paths["BASE"] / "provider_extraction_manifest.tsv"
    row = resolve_manifest_row(mf, "TASK_ID", index)
    extract_provider_dataset(root, args, row)


def direct_candidate(root: Path, args, index: int) -> None:
    _, ancestry_label = core.canonical_ancestry(args.ancestry)
    paths = out_paths(root, args.phenotype.strip(), ancestry_label)
    mf = paths["BASE"] / "provider_coloc_manifest.tsv"
    row = resolve_manifest_row(mf, "TASK_ID", index)
    print(json.dumps(run_candidate(root, args, row), indent=2, default=str))


def main() -> None:
    args = arguments()
    root = Path.cwd().resolve()

    if args.array_limit < 1:
        raise SystemExit(
            "--array-limit must be >= 1 (Step11 worker pools use at most 1000 tasks per array)"
        )

    if args.max_parallel < 0:
        raise SystemExit(
            "--max-parallel must be >= 0"
        )

    if args.max_extract_parallel < 0:
        raise SystemExit(
            "--max-extract-parallel must be >= 0"
        )

    if (
        args.coloc_worker_pool < 1
        or args.coloc_worker_pool > 1000
    ):
        raise SystemExit(
            "--coloc-worker-pool must be between 1 and 1000"
        )

    if args.extract_chunksize < 10_000:
        raise SystemExit(
            "--extract-chunksize must be >= 10000"
        )

    if args.candidates_per_task < 1:
        raise SystemExit(
            "--candidates-per-task must be >= 1"
        )

    # --------------------------------------------------------
    # Internal workers
    # --------------------------------------------------------

    if args.discover_worker:

        discovery_worker(
            root,
            args,
        )

        return

    if (
        args.planner_worker
        or args.auto_continue
    ):

        # Compatibility: old --auto-continue now means planning only.
        # It NEVER submits jobs from a compute node.
        planner_worker(
            root,
            args,
        )

        return

    if args.extract_worker:

        extract_worker(
            root,
            args,
        )

        return

    if args.candidate_worker:

        candidate_worker(
            root,
            args,
        )

        return

    # --------------------------------------------------------
    # Manual/debug actions
    # --------------------------------------------------------

    if args.discover_index is not None:

        row = discovery_row_from_index(
            root,
            args,
            args.discover_index,
        )

        run_discovery(
            root,
            args,
            row,
        )

        return

    if args.build_provider_plan:

        build_provider_plan(
            root,
            args,
        )

        prepare_ready_manifests(
            root,
            args,
        )

        return

    if args.extract_index is not None:

        direct_extract(
            root,
            args,
            args.extract_index,
        )

        return

    if args.candidate_index is not None:

        direct_candidate(
            root,
            args,
            args.candidate_index,
        )

        return

    if args.aggregate:

        aggregate(
            root,
            args,
        )

        return

    # --------------------------------------------------------
    # Default = one-command fully prewired DAG.
    # --------------------------------------------------------

    auto_pipeline(
        root,
        args,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWAS2m - shared status / provenance helpers.

Used by Step01_10_Run.py (writes status) and Pipeline_reporter.py (reads it).
Contains NO scientific logic.

Layout (one directory per phenotype x ancestry):

    pipeline_audit/<phenotype_slug>/<ancestry_slug>/
        run_manifest.json
        study_manifest.tsv
        study_selection.tsv
        study_stage_status.tsv
        study_summary.tsv
        exclusion_reasons.tsv
        software_versions.tsv
        step11_ready_studies.tsv
        download_provenance.tsv
        qc_metrics.tsv
        locus_status.tsv
        status/<GCST>/<STAGE>.json        one file per study x stage
        status/<GCST>/history.jsonl       every final status ever written
        studies/<GCST>/manifests/         one-row step manifests
        studies/<GCST>/download_provenance.json
        studies/<GCST>/raw_cleanup.json
        logs/<GCST>/<STAGE>.log           full stdout/stderr of each stage
        slurm/                            generated SLURM scripts
"""

from __future__ import annotations

import gzip
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =============================================================================
# STAGES AND STATUS VOCABULARY
# =============================================================================

# (stage name, short code used in reports)
STAGES = [
    ("Step01_Discovery", "S01"),
    ("Step02_Download", "S02"),
    ("Step03_QC", "S03"),
    ("Step04_LD_Reference", "S04"),
    ("Step05_Clumping", "S05"),
    ("Step06_FineMapping", "S06"),
    ("Step07_VEP", "S07"),
    ("Step08_GTEx_QTL", "S08"),
    ("Step09_SpliceAI", "S09"),
    ("Step10_Pangolin", "S10"),
]

STAGE_NAMES = [name for name, _ in STAGES]
STAGE_CODES = dict(STAGES)
STAGE_ORDER = {name: number for number, name in enumerate(STAGE_NAMES)}

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
SKIPPED_ALREADY_COMPLETE = "SKIPPED_ALREADY_COMPLETE"
PARTIAL = "PARTIAL"
FAILED = "FAILED"
EXCLUDED = "EXCLUDED"
NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_RUN_UPSTREAM_FAILED = "NOT_RUN_UPSTREAM_FAILED"

DONE_STATUSES = {COMPLETE, SKIPPED_ALREADY_COMPLETE}

STATUS_SHORT = {
    COMPLETE: "OK",
    SKIPPED_ALREADY_COMPLETE: "OK",
    PARTIAL: "PART",
    FAILED: "FAIL",
    EXCLUDED: "EXCL",
    NOT_AVAILABLE: "N/A",
    NOT_RUN_UPSTREAM_FAILED: "--",
    RUNNING: "RUN",
    PENDING: "PEND",
}


# =============================================================================
# NAMES / PATHS
# =============================================================================

def normalize_text(value) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def slugify(value) -> str:
    """Identical to the slugify() used by every Step script."""
    return normalize_text(value).replace(" ", "_")


def audit_dir(root: Path, phenotype_slug: str, ancestry_slug: str) -> Path:
    return root / "pipeline_audit" / phenotype_slug / ancestry_slug


def status_file(audit: Path, accession: str, stage: str) -> Path:
    return audit / "status" / accession / f"{stage}.json"


def stage_log_file(audit: Path, accession: str, stage: str) -> Path:
    return audit / "logs" / accession / f"{stage}.log"


def study_work_dir(audit: Path, accession: str) -> Path:
    return audit / "studies" / accession


# =============================================================================
# TIME / ATOMIC WRITES
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def append_jsonl(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


# =============================================================================
# PROVENANCE
# =============================================================================

def git_commit(root: Path) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        if not sha:
            return "UNKNOWN"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "UNKNOWN"


def runtime_context() -> dict:
    """Host + SLURM identity of the current process."""
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
    }


def file_signature(path: Path) -> dict:
    """Cheap identity of a (possibly huge) file: never hashes its contents."""
    path = Path(path)
    if not path.exists():
        return {"PATH": str(path), "EXISTS": False}
    stat = path.stat()
    return {
        "PATH": str(path),
        "EXISTS": True,
        "SIZE_BYTES": int(stat.st_size),
        "MTIME_UTC": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def gzip_magic_ok(path: Path) -> bool:
    """True for non-.gz files; for .gz files checks the 2-byte gzip header."""
    path = Path(path)
    if not str(path).lower().endswith(".gz"):
        return True
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def count_gzip_data_rows(path: Path) -> int:
    """Stream a gzip TSV to EOF (verifies the gzip CRC) and count data rows."""
    lines = 0
    with gzip.open(path, "rb") as handle:
        for _ in handle:
            lines += 1
    return max(lines - 1, 0)


# =============================================================================
# STATUS RECORDS
# =============================================================================

def read_all_statuses(audit: Path) -> pd.DataFrame:
    rows = []
    root = audit / "status"
    if root.exists():
        for path in sorted(root.glob("*/*.json")):
            payload = read_json(path)
            if payload:
                rows.append(payload)
    return pd.DataFrame(rows)

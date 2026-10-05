#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWAS2m - generic Step01-10 controller (phenotype-agnostic).

ONE command per phenotype x ancestry, run from the GWAS2m project root:

    python Step01_10_Run.py --phenotype "parkinson's disease" --ancestry EUR
    python Step01_10_Run.py --phenotype migraine --ancestry EUR
    python Step01_10_Run.py --phenotype asthma --ancestry AFR

What it does
------------
1. Step01 discovery (Step01_Plan_GWAS.plan): every phenotype-matched GWAS
   Catalog study is recorded in study_selection.tsv with its selection reason.
2. Writes pipeline_audit/<phenotype>/<ancestry>/study_manifest.tsv
   (one row per eligible study; the row number is the SLURM array index).
3. Submits ONE SLURM array: one array task = one GWAS study. Each task runs
   Step02 -> Step10 for that study only. Studies never wait for each other.
4. Submits one small audit job (afterany) that builds the paper tables.

This file contains NO scientific code. Every stage calls the existing
worker_mode() of the corresponding Step script, unchanged, through a
one-row manifest (exactly what the existing SLURM arrays do), as a
subprocess with the same environment variables the generated arrays set.

Modes
-----
    (default)   setup preflight + discovery + study manifest + SLURM submission
    --dry-run   same, but print the generated SLURM scripts and submit nothing
                (--no-submit is an alias)
    --local     run every study sequentially here (no SLURM), then audit
    --worker    run ONE study (SLURM_ARRAY_TASK_ID or --array-id)
    --audit     rebuild the pipeline_audit tables from the status files

Setup / SLURM
-------------
Before submitting, a read-only preflight (gwas2m_resources.inspect_pipeline,
the same check as `Step00_Check_Resources.py --inspect --ancestry X`) must
pass; nothing is downloaded or installed here. SBATCH settings come only from
config/slurm.yaml (stage "study" for the per-study array, "audit" for the
audit job) or --slurm-config FILE; --partition/--time/--mem/--cpus/
--max-parallel override them. No "%N" throttle unless configured. Arrays
larger than slurm.array_limit are split into several arrays.

Resume
------
Re-running the same command is always safe. Each stage is validated against
the same output files its worker checks; validated stages are skipped
([SKIP] output already validated), failed/incomplete stages are re-run, and
stages downstream of a failure are recorded as NOT_RUN_UPSTREAM_FAILED.
Step06 additionally resumes per locus (see Step06_FineMap_GWAS_Loci.py).

Step11 is NOT run here. Studies ready for it are listed in
step11_ready_studies.tsv; run Step11_Provider_SuSiE_Formal.py separately.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gwas2m_config  # noqa: E402
import gwas2m_resources  # noqa: E402
import gwas2m_status as gs  # noqa: E402


CONTROLLER_VERSION = "1.1.0"

# SLURM resources come from config/slurm.yaml: stage "study" (one task = one
# study running Steps02-10, so it needs the largest per-step request) and
# stage "audit". --partition/--time/--mem/--cpus/--max-parallel override them.
SLURM_STAGES = ("study", "audit")

STEP_SCRIPTS = {
    "Step01_Discovery": "Step01_Plan_GWAS",
    "Step02_Download": "Step01_Plan_GWAS",
    "Step03_QC": "Step03_QC_One_GWAS",
    "Step04_LD_Reference": "Step05_Clump_GWAS_Loci",
    "Step05_Clumping": "Step05_Clump_GWAS_Loci",
    "Step06_FineMapping": "Step06_FineMap_GWAS_Loci",
    "Step07_VEP": "Step07_Annotate_Finemapped_Variants",
    "Step08_GTEx_QTL": "Step08_Integrate_GTEx_eQTL_sQTL",
    "Step09_SpliceAI": "Step09_Run_SpliceAI",
    "Step10_Pangolin": "Step10_Run_Pangolin",
}

STUDY_STAGES = gs.STAGE_NAMES[1:]          # Step02 ... Step10

# A stage that actually re-ran makes these downstream outputs stale, so they
# are forced to re-run (Step02 re-downloads the same URL and Step04 only
# verifies the shared reference, so neither invalidates anything).
REBUILD_SOURCES = {
    "Step03_QC", "Step05_Clumping", "Step06_FineMapping",
    "Step07_VEP", "Step08_GTEx_QTL", "Step09_SpliceAI",
}


# =============================================================================
# SMALL HELPERS
# =============================================================================

_MODULES: dict[str, object] = {}


def step(name: str):
    """Import an existing Step script as a module (cached)."""
    if name not in _MODULES:
        _MODULES[name] = importlib.import_module(name)
    return _MODULES[name]


def banner(text: str) -> None:
    print()
    print("=" * 88)
    print(text)
    print("=" * 88, flush=True)


def non_empty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def as_int(value):
    """int() without float round-off (ns mtimes exceed float precision)."""
    try:
        if value is None or value == "" or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def pipeline_python(root: Path) -> Path:
    candidate = root / "envs" / "pipeline" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable).resolve()


# =============================================================================
# PROJECT (phenotype x ancestry)
# =============================================================================

@dataclass
class Project:
    root: Path
    phenotype: str
    ancestry_input: str
    ancestry_code: str
    ancestry_label: str
    phenotype_slug: str
    ancestry_slug: str
    audit: Path

    def step_root(self, folder: str) -> Path:
        return self.root / folder / self.phenotype_slug / self.ancestry_slug

    def step_argv(self) -> list[str]:
        return ["--phenotype", self.phenotype, "--ancestry", self.ancestry_code]

    def step_args(self, module: str, force: bool = False):
        """Each step's own argparse defaults (no thresholds are duplicated here)."""
        args = step(module).arguments(self.step_argv())
        if hasattr(args, "force"):
            args.force = bool(force)
        return args


def make_project(phenotype: str, ancestry: str) -> Project:
    phenotype = phenotype.strip()
    if not phenotype:
        raise SystemExit("--phenotype cannot be empty")
    ancestry_code, ancestry_label = step("Step03_QC_One_GWAS").canonical_ancestry(ancestry)
    root = Path.cwd().resolve()
    phenotype_slug = gs.slugify(phenotype)
    ancestry_slug = gs.slugify(ancestry_label)
    return Project(
        root=root,
        phenotype=phenotype,
        ancestry_input=ancestry,
        ancestry_code=ancestry_code,
        ancestry_label=ancestry_label,
        phenotype_slug=phenotype_slug,
        ancestry_slug=ancestry_slug,
        audit=gs.audit_dir(root, phenotype_slug, ancestry_slug),
    )


# =============================================================================
# STUDY + STATUS RECORDS
# =============================================================================

@dataclass
class Study:
    project: Project
    accession: str
    row: dict
    array_task_id: str
    slurm: dict = field(default_factory=gs.runtime_context)
    git: str = "UNKNOWN"

    @property
    def work(self) -> Path:
        return gs.study_work_dir(self.project.audit, self.accession)

    def manifest_path(self, name: str) -> Path:
        return self.work / "manifests" / name

    def log_path(self, stage: str) -> Path:
        return gs.stage_log_file(self.project.audit, self.accession, stage)

    def status_path(self, stage: str) -> Path:
        return gs.status_file(self.project.audit, self.accession, stage)

    # Step output directories (same layout as the planners).
    @property
    def raw_root(self) -> Path:
        return self.project.step_root("02_summary_stats") / "raw"

    @property
    def qc_dir(self) -> Path:
        return self.project.step_root("02_summary_stats") / "qc" / self.accession

    def out_dir(self, folder: str) -> Path:
        return self.project.step_root(folder) / self.accession


@dataclass
class Outcome:
    status: str
    reason: str = ""
    details: dict = field(default_factory=dict)


def status_record(study: Study, stage: str, status: str, reason: str = "", **details) -> dict:
    project = study.project
    record = {
        "phenotype": project.phenotype,
        "phenotype_slug": project.phenotype_slug,
        "ancestry_code": project.ancestry_code,
        "ancestry_label": project.ancestry_label,
        "study_accession": study.accession,
        "array_task_id": study.array_task_id,
        "stage": stage,
        "status": status,
        "reason": reason,
        "started_utc": None,
        "finished_utc": None,
        "runtime_seconds": None,
        "hostname": study.slurm.get("hostname"),
        "slurm_job_id": study.slurm.get("slurm_job_id"),
        "slurm_array_job_id": study.slurm.get("slurm_array_job_id"),
        "slurm_array_task_id": study.slurm.get("slurm_array_task_id"),
        "slurm_partition": study.slurm.get("slurm_partition"),
        "slurm_account": study.slurm.get("slurm_account"),
        "slurm_cpus_per_task": study.slurm.get("slurm_cpus_per_task"),
        "slurm_mem_per_node": study.slurm.get("slurm_mem_per_node"),
        "git_commit": study.git,
        "input_file": None,
        "output_dir": None,
        "output_files": [],
        "n_input_variants": None,
        "n_output_variants": None,
        "n_loci": None,
        "n_loci_success": None,
        "n_loci_failed": None,
        "error_type": None,
        "error_message": None,
        "log_path": None,
        "failure_marker": None,
        "step_manifest": None,
        "worker_command": None,
        "forced_rerun": False,
        "extra": {},
    }
    record.update(details)
    return record


def write_status(study: Study, stage: str, record: dict, history: bool = True) -> None:
    gs.atomic_write_json(study.status_path(stage), record)
    if history and record["status"] not in {gs.PENDING, gs.RUNNING}:
        gs.append_jsonl(study.status_path(stage).parent / "history.jsonl", record)


# =============================================================================
# LOGGED EXECUTION
# =============================================================================

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


@contextlib.contextmanager
def tee_to(log_file: Path):
    """Copy this process' stdout/stderr into the stage log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered: the log on disk is always current (tail -f, error parsing).
    with open(log_file, "a", encoding="utf-8", buffering=1) as handle:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, handle)
        sys.stderr = _Tee(old_err, handle)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def run_logged(command: list, env_updates: dict, cwd: Path) -> int:
    """Run a step worker as a subprocess, streaming its output into the log."""
    env = os.environ.copy()
    env.update({key: str(value) for key, value in env_updates.items()})
    print("$ " + " ".join(shlex.quote(str(x)) for x in command), flush=True)
    for key, value in env_updates.items():
        print(f"  env {key}={value}", flush=True)
    process = subprocess.Popen(
        [str(x) for x in command],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
    sys.stdout.flush()
    return process.wait()


EXCEPTION_LINE = re.compile(
    r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))(?::\s?(.*))?$"
)


def error_from_log(log_file: Path, return_code: int) -> tuple[str, str]:
    """Exception type/message from the last Python traceback in the log."""
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
    except OSError:
        lines = []
    for line in reversed(lines):
        match = EXCEPTION_LINE.match(line.strip())
        if match:
            return match.group(1), (match.group(2) or "").strip()
    tail = next((x.strip() for x in reversed(lines) if x.strip()), "")
    return f"WorkerExitCode{return_code}", tail


def write_one_row_manifest(path: Path, row: dict) -> Path:
    gs.atomic_write_tsv(pd.DataFrame([row]), path)
    return path.resolve()


def run_step_worker(
    study: Study,
    stage: str,
    module: str,
    manifest_var: str,
    manifest_file: Path,
    details: dict,
    extra_env: dict | None = None,
    extra_args: list | None = None,
) -> int:
    """Run <module>.py worker_mode() on a one-row manifest (task id 1)."""
    command = [sys.executable, SCRIPT_DIR / f"{module}.py", *(extra_args or [])]
    env = {"SLURM_ARRAY_TASK_ID": "1", manifest_var: manifest_file}
    env.update(extra_env or {})
    details["step_manifest"] = str(manifest_file)
    details["worker_command"] = " ".join(shlex.quote(str(x)) for x in command)
    return run_logged(command, env, study.project.root)


def failed_from_worker(study: Study, stage: str, return_code: int, details: dict,
                       marker: Path | None, started_epoch: float) -> Outcome:
    error_type, error_message = error_from_log(study.log_path(stage), return_code)
    details["error_type"] = error_type
    details["error_message"] = error_message
    if marker is not None and marker.exists() and marker.stat().st_mtime >= started_epoch - 1:
        details["failure_marker"] = str(marker)
    return Outcome(gs.FAILED, f"{error_type}: {error_message}", details)


# =============================================================================
# STAGE OUTPUT VALIDATORS
#
# Each mirrors the "already complete" check inside the corresponding
# worker_mode(), so a stage the controller skips is one the worker would skip.
# =============================================================================

def raw_file(study: Study) -> Path | None:
    found = step("Step03_QC_One_GWAS").find_downloaded_file(
        raw_root=study.raw_root,
        accession=study.accession,
        expected_filename=str(study.row.get("FILE_NAME", "")).strip(),
    )
    return Path(found) if found else None


def qc_files(study: Study) -> dict[str, Path]:
    d, a = study.qc_dir, study.accession
    return {
        "full": d / f"{a}_GRCh38_QC.tsv.gz",
        "significant": d / f"{a}_GRCh38_significant.tsv.gz",
        "summary_tsv": d / f"{a}_QC_summary.tsv",
        "summary_json": d / f"{a}_QC_summary.json",
        "failed": d / f"{a}_FAILED.txt",
    }


def qc_complete(study: Study) -> bool:
    f = qc_files(study)
    return all(non_empty(f[k]) for k in ("full", "significant", "summary_tsv", "summary_json"))


def clump_files(study: Study) -> dict[str, Path]:
    d = study.out_dir("05_ld_clumping")
    return {
        "lead": d / "lead_variants.tsv",
        "mapped": d / "mapped_candidates.tsv.gz",
        "summary_tsv": d / "clumping_summary.tsv",
        "summary_json": d / "clumping_summary.json",
        "failed": d / "CLUMPING_FAILED.txt",
    }


def clump_complete(study: Study) -> bool:
    f = clump_files(study)
    return (
        all(f[k].exists() for k in ("lead", "mapped", "summary_tsv", "summary_json"))
        and not f["failed"].exists()
    )


def finemap_files(study: Study) -> dict[str, Path]:
    d, a = study.out_dir("06_finemapping"), study.accession
    return {
        "dir": d,
        "variants": d / f"{a}_finemapped_variants.tsv.gz",
        "credible_sets": d / f"{a}_95pct_credible_sets.tsv",
        "summary_tsv": d / f"{a}_finemapping_summary.tsv",
        "failed_loci": d / f"{a}_failed_loci.tsv",
        "summary_json": d / "finemapping_summary.json",
        "loci": d / "loci_definition.tsv",
        "failed": d / "FINEMAPPING_FAILED.txt",
    }


def finemap_complete(study: Study) -> bool:
    f = finemap_files(study)
    return (
        f["variants"].exists() and f["summary_tsv"].exists()
        and f["summary_json"].exists() and not f["failed"].exists()
    )


def vep_files(study: Study) -> dict[str, Path]:
    d, a = study.out_dir("07_annotation"), study.accession
    return {
        "variant_summary": d / f"{a}_VEP_variant_summary.tsv",
        "summary_json": d / "annotation_summary.json",
        "failed": d / "ANNOTATION_FAILED.txt",
    }


def vep_complete(study: Study) -> bool:
    f = vep_files(study)
    if not non_empty(f["variant_summary"]) or not f["summary_json"].exists() or f["failed"].exists():
        return False
    summary = gs.read_json(f["summary_json"])
    fine = finemap_files(study)["variants"]
    if not fine.exists():
        return False
    s07 = step("Step07_Annotate_Finemapped_Variants")
    return (
        str(summary.get("STATUS", "")).upper() == "COMPLETE"
        and str(summary.get("ANNOTATION_SCOPE", "")).strip().upper() == s07.ANNOTATION_SCOPE
        and as_int(summary.get("FINEMAPPED_INPUT_SIZE")) == fine.stat().st_size
        and as_int(summary.get("FINEMAPPED_INPUT_MTIME_NS")) == fine.stat().st_mtime_ns
    )


def qtl_files(study: Study) -> dict[str, Path]:
    d, a = study.out_dir("08_qtl"), study.accession
    return {
        "dir": d,
        "variant_summary": d / f"{a}_GTEx_QTL_variant_summary.tsv",
        "summary_json": d / "qtl_summary.json",
        "failed": d / "QTL_FAILED.txt",
    }


def qtl_complete(study: Study) -> bool:
    f = qtl_files(study)
    return non_empty(f["variant_summary"]) and non_empty(f["summary_json"]) and not f["failed"].exists()


def spliceai_files(study: Study) -> dict[str, Path]:
    d, a = study.out_dir("09_splicing"), study.accession
    return {
        "variant_summary": d / f"{a}_SpliceAI_variant_summary.tsv",
        "summary_json": d / "spliceai_summary.json",
        "failed": d / "SPLICEAI_FAILED.txt",
    }


def spliceai_complete(study: Study) -> bool:
    f = spliceai_files(study)
    return (
        non_empty(f["variant_summary"]) and not f["failed"].exists()
        and str(gs.read_json(f["summary_json"]).get("STATUS", "")).upper() == "COMPLETE"
    )


def pangolin_files(study: Study) -> dict[str, Path]:
    d, a = study.out_dir("10_pangolin"), study.accession
    return {
        "integrated": d / f"{a}_SpliceAI_Pangolin_integrated.tsv",
        "summary_json": d / "pangolin_summary.json",
        "failed": d / "PANGOLIN_FAILED.txt",
    }


def pangolin_complete(study: Study) -> bool:
    f = pangolin_files(study)
    return (
        non_empty(f["integrated"]) and not f["failed"].exists()
        and str(gs.read_json(f["summary_json"]).get("STATUS", "")).upper() == "COMPLETE"
    )


# =============================================================================
# STAGES (each returns an Outcome; exceptions are recorded as FAILED)
# =============================================================================

def stage_download(study: Study, force: bool) -> Outcome:
    url = str(study.row.get("DOWNLOAD_URL", "")).strip()
    file_name = str(study.row.get("FILE_NAME", "")).strip()
    expected = study.raw_root / study.accession / file_name
    provenance_file = study.work / "download_provenance.json"
    previous = gs.read_json(provenance_file)

    provenance = {
        "STUDY_ACCESSION": study.accession,
        "SOURCE_URL": url,
        "EXPECTED_FILENAME": file_name,
        "DOWNLOAD_START": previous.get("DOWNLOAD_START"),
        "DOWNLOAD_END": previous.get("DOWNLOAD_END"),
        "DOWNLOAD_STATUS": None,
        "FILE_SIZE_BYTES": previous.get("FILE_SIZE_BYTES"),
        "OUTPUT_FILE": str(expected),
        "DOWNLOAD_ERROR": None,
        "LAST_CHECKED_UTC": gs.utc_now(),
    }
    details = {"input_file": url, "output_dir": str(expected.parent), "output_files": [str(expected)]}

    existing = raw_file(study)
    if existing is not None and gs.gzip_magic_ok(existing):
        print("[SKIP] output already validated")
        provenance.update({
            "DOWNLOAD_STATUS": gs.SKIPPED_ALREADY_COMPLETE,
            "OUTPUT_FILE": str(existing),
            "FILE_SIZE_BYTES": existing.stat().st_size,
        })
        gs.atomic_write_json(provenance_file, provenance)
        details["output_files"] = [str(existing)]
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "Raw GWAS file present", details)

    if qc_complete(study):
        print("[SKIP] raw file not needed: Step03 QC output already validated")
        provenance["DOWNLOAD_STATUS"] = gs.SKIPPED_ALREADY_COMPLETE
        gs.atomic_write_json(provenance_file, provenance)
        details["output_files"] = []
        return Outcome(
            gs.SKIPPED_ALREADY_COMPLETE,
            "Raw file not needed: Step03 QC already validated (raw removed after QC)",
            details,
        )

    if not url or not file_name:
        provenance.update({"DOWNLOAD_STATUS": gs.FAILED, "DOWNLOAD_ERROR": "No DOWNLOAD_URL/FILE_NAME"})
        gs.atomic_write_json(provenance_file, provenance)
        return Outcome(gs.FAILED, "Study manifest has no DOWNLOAD_URL/FILE_NAME", details)

    if expected.exists() and not gs.gzip_magic_ok(expected):
        print(f"Existing file is not valid gzip; re-downloading: {expected}")
        expected.unlink()

    provenance["DOWNLOAD_START"] = gs.utc_now()
    provenance["DOWNLOAD_END"] = None
    command = [
        sys.executable, "-c",
        "import sys; import Step01_Plan_GWAS as s; "
        "print(s.download_study_file(sys.argv[1], sys.argv[2]))",
        url, str(expected),
    ]
    details["worker_command"] = " ".join(shlex.quote(str(x)) for x in command)
    python_path = os.pathsep.join(x for x in (str(SCRIPT_DIR), os.environ.get("PYTHONPATH", "")) if x)
    code = run_logged(command, {"PYTHONPATH": python_path}, study.project.root)
    provenance["DOWNLOAD_END"] = gs.utc_now()

    if code != 0 or not non_empty(expected) or not gs.gzip_magic_ok(expected):
        error_type, error_message = error_from_log(study.log_path("Step02_Download"), code)
        if code == 0:
            error_type, error_message = "InvalidDownload", "Downloaded file missing, empty or not gzip"
        provenance.update({"DOWNLOAD_STATUS": gs.FAILED, "DOWNLOAD_ERROR": f"{error_type}: {error_message}"})
        gs.atomic_write_json(provenance_file, provenance)
        details.update({"error_type": error_type, "error_message": error_message})
        return Outcome(gs.FAILED, f"{error_type}: {error_message}", details)

    provenance.update({"DOWNLOAD_STATUS": gs.COMPLETE, "FILE_SIZE_BYTES": expected.stat().st_size})
    gs.atomic_write_json(provenance_file, provenance)
    return Outcome(gs.COMPLETE, "Downloaded", details)


def qc_details(study: Study, details: dict) -> dict:
    f = qc_files(study)
    summary = gs.read_json(f["summary_json"])
    details.update({
        "output_dir": str(study.qc_dir),
        "output_files": [str(f[k]) for k in ("full", "significant", "summary_tsv", "summary_json")],
        "n_input_variants": as_int(summary.get("N_BEFORE_QC")),
        "n_output_variants": as_int(summary.get("N_AFTER_QC")),
        "extra": {
            "QC_STATUS": summary.get("QC_STATUS"),
            "N_GENOME_WIDE_SIGNIFICANT": summary.get("N_GENOME_WIDE_SIGNIFICANT"),
            "CLUMPING_READY": summary.get("CLUMPING_READY"),
            "SUSIE_READY": summary.get("SUSIE_READY"),
        },
    })
    if summary.get("INPUT_FILE"):
        details["input_file"] = summary["INPUT_FILE"]
    return details


def stage_qc(study: Study, force: bool) -> Outcome:
    details: dict = {}
    if qc_complete(study):
        print("[SKIP] output already validated")
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "QC outputs present", qc_details(study, details))

    s03 = step("Step03_QC_One_GWAS")
    raw = raw_file(study)
    if raw is None:
        return Outcome(gs.FAILED, "Raw GWAS file not found for QC", details)

    details["input_file"] = str(raw)
    row = s03.build_qc_row(
        task_id=1,
        accession=study.accession,
        input_file=raw,
        output_directory=study.qc_dir,
        phenotype=study.project.phenotype,
        ancestry_code=study.project.ancestry_code,
        ancestry_label=study.project.ancestry_label,
    )
    manifest = write_one_row_manifest(study.manifest_path("Step03_manifest.tsv"), row)
    started = time.time()
    code = run_step_worker(study, "Step03_QC", "Step03_QC_One_GWAS", "GWAS_QC_MANIFEST", manifest, details)
    if code != 0:
        return failed_from_worker(study, "Step03_QC", code, details, qc_files(study)["failed"], started)
    if not qc_complete(study):
        return Outcome(gs.FAILED, "QC worker exited 0 but QC outputs failed validation", details)
    return Outcome(gs.COMPLETE, "QC complete", qc_details(study, details))


def stage_ld_reference(study: Study, force: bool) -> Outcome:
    project = study.project
    reference = project.root / "resources" / "1000G" / project.ancestry_code
    step("Step05_Clump_GWAS_Loci").validate_reference(project.root, project.ancestry_code)
    return Outcome(
        gs.COMPLETE,
        f"Shared 1000G {project.ancestry_code} GRCh38 LD reference verified (chr1-22 pgen/pvar/psam)",
        {"output_dir": str(reference)},
    )


def clump_details(study: Study, details: dict) -> dict:
    f = clump_files(study)
    summary = gs.read_json(f["summary_json"])
    details.update({
        "input_file": str(qc_files(study)["full"]),
        "output_dir": str(f["lead"].parent),
        "output_files": [str(f[k]) for k in ("lead", "mapped", "summary_tsv", "summary_json")],
        "n_input_variants": as_int(summary.get("N_GWAS_ROWS")),
        "n_output_variants": as_int(summary.get("N_LEAD_VARIANTS")),
        "extra": {k: summary.get(k) for k in ("P1", "P2", "R2", "KB", "N_P_LE_P2", "N_REFERENCE_MAPPED", "MAPPING_RATE")},
    })
    return details


def stage_clumping(study: Study, force: bool) -> Outcome:
    details: dict = {}
    if not force and clump_complete(study):
        print("[SKIP] output already validated")
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "Clumping outputs present", clump_details(study, details))

    project = study.project
    s05 = step("Step05_Clump_GWAS_Loci")
    args = project.step_args("Step05_Clump_GWAS_Loci")
    s05.validate_arguments(args)
    row, exclusion = s05.build_clump_row(
        args,
        task_id=1,
        accession=study.accession,
        qc_file=qc_files(study)["full"],
        output_root=project.step_root("05_ld_clumping"),
        phenotype=project.phenotype,
        ancestry_code=project.ancestry_code,
        ancestry_label=project.ancestry_label,
    )
    if exclusion is not None:
        details["extra"] = exclusion
        return Outcome(gs.EXCLUDED, str(exclusion.get("REASON", "")), details)

    # Re-run after an upstream rebuild or a previous failure must not reuse old outputs.
    row["FORCE"] = bool(force or clump_files(study)["failed"].exists())
    details["forced_rerun"] = row["FORCE"]
    manifest = write_one_row_manifest(study.manifest_path("Step05_manifest.tsv"), row)
    started = time.time()
    code = run_step_worker(study, "Step05_Clumping", "Step05_Clump_GWAS_Loci", "GWAS_CLUMP_MANIFEST", manifest, details)
    if code != 0:
        return failed_from_worker(study, "Step05_Clumping", code, details, clump_files(study)["failed"], started)
    if not clump_complete(study):
        return Outcome(gs.FAILED, "Clumping worker exited 0 but outputs failed validation", details)
    return Outcome(gs.COMPLETE, "Clumping complete", clump_details(study, details))


def finemap_details(study: Study, details: dict, fresh_after: float | None = None) -> dict:
    f = finemap_files(study)
    summary = gs.read_json(f["summary_json"])
    if fresh_after is not None and f["summary_json"].exists() and f["summary_json"].stat().st_mtime < fresh_after - 1:
        summary = {}  # stale summary from an earlier run
    details.update({
        "input_file": str(clump_files(study)["lead"]),
        "output_dir": str(f["dir"]),
        "output_files": [str(f[k]) for k in ("variants", "credible_sets", "summary_tsv", "failed_loci", "summary_json") if f[k].exists()],
        "n_loci": as_int(summary.get("N_LOCI")),
        "n_loci_success": as_int(summary.get("N_LOCI_SUCCESS")),
        "n_loci_failed": as_int(summary.get("N_LOCI_FAILED")),
        "extra": {k: summary.get(k) for k in ("STATUS", "GWAS_N", "GWAS_N_SOURCE", "REFERENCE_N")},
    })
    return details


def stage_finemapping(study: Study, force: bool) -> Outcome:
    details: dict = {}
    if not force and finemap_complete(study):
        print("[SKIP] output already validated")
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "Fine-mapping outputs present", finemap_details(study, details))

    project = study.project
    s06 = step("Step06_FineMap_GWAS_Loci")
    args = project.step_args("Step06_FineMap_GWAS_Loci")
    _, plink2, rscript = s06.resolve_finemap_tools(project.root)
    s06.verify_reference(project.root, project.ancestry_code)
    reference_n = s06.count_reference_samples(project.root, project.ancestry_code)

    # Same Step01 lookup the planner builds.
    step01_file = project.step_root("01_gwas_catalog") / "GWAS_download_manifest.tsv"
    step01 = pd.read_csv(step01_file, sep="\t", dtype=str) if step01_file.exists() else pd.DataFrame()
    step01_lookup = {}
    if not step01.empty and "STUDY_ACCESSION" in step01.columns:
        step01_lookup = {str(r["STUDY_ACCESSION"]).strip(): r for _, r in step01.iterrows()}

    output_root = project.step_root("06_finemapping")
    row, meta, problems = s06.build_finemap_row(
        args,
        task_id=1,
        accession=study.accession,
        qc_file=qc_files(study)["full"].resolve(),
        clump_root=project.step_root("05_ld_clumping"),
        output_root=output_root,
        metadata_root=output_root / "metadata",
        step01_lookup=step01_lookup,
        session=s06.make_session(),
        reference_n=reference_n,
        rscript=rscript,
        plink2=plink2,
        phenotype=project.phenotype,
        ancestry_code=project.ancestry_code,
        ancestry_label=project.ancestry_label,
    )
    if meta is not None:
        details["extra"] = {"RESOLVED_GWAS_N": meta.get("RESOLVED_GWAS_N"),
                            "RESOLVED_GWAS_N_SOURCE": meta.get("RESOLVED_GWAS_N_SOURCE")}
    if problems:
        return Outcome(gs.FAILED, " | ".join(problems), details)

    row["FORCE"] = bool(force)
    details["forced_rerun"] = bool(force)
    manifest = write_one_row_manifest(study.manifest_path("Step06_manifest.tsv"), row)
    started = time.time()
    code = run_step_worker(
        study, "Step06_FineMapping", "Step06_FineMap_GWAS_Loci", "GWAS_FINEMAP_MANIFEST", manifest,
        details, extra_args=project.step_argv(),
    )
    finemap_details(study, details, fresh_after=started)
    if code != 0:
        outcome = failed_from_worker(study, "Step06_FineMapping", code, details, finemap_files(study)["failed"], started)
        if (details.get("n_loci_success") or 0) > 0 and (details.get("n_loci_failed") or 0) > 0:
            outcome.status = gs.PARTIAL
            outcome.reason = (
                f"{details['n_loci_failed']} of {details['n_loci']} loci failed "
                f"(see {finemap_files(study)['failed_loci']}); completed loci are kept and "
                "skipped on rerun"
            )
        return outcome
    if not finemap_complete(study):
        return Outcome(gs.FAILED, "Fine-mapping worker exited 0 but outputs failed validation", details)
    return Outcome(gs.COMPLETE, "Fine-mapping complete", details)


def summary_counts(path: Path, n_in: str, n_out: str, n_loci: str = "N_LOCI") -> dict:
    summary = gs.read_json(path)
    return {
        "n_input_variants": as_int(summary.get(n_in)),
        "n_output_variants": as_int(summary.get(n_out)),
        "n_loci": as_int(summary.get(n_loci)),
    }


def stage_vep(study: Study, force: bool) -> Outcome:
    f = vep_files(study)
    details = {"input_file": str(finemap_files(study)["variants"]), "output_dir": str(f["variant_summary"].parent),
               "output_files": [str(f["variant_summary"]), str(f["summary_json"])]}
    counts = ("N_FINEMAPPED_ROWS_AVAILABLE", "N_ANNOTATED_VARIANTS", "N_FINEMAPPED_LOCI")
    if not force and vep_complete(study):
        print("[SKIP] output already validated")
        details.update(summary_counts(f["summary_json"], *counts))
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "VEP outputs present and current", details)

    project = study.project
    s07 = step("Step07_Annotate_Finemapped_Variants")
    args = project.step_args("Step07_Annotate_Finemapped_Variants", force=force)
    s07.validate_arguments(args)
    vep = s07.resolve_vep_executable(project.root, args.vep)
    cache = s07.resolve_vep_cache(project.root, args.vep_cache)
    fasta = s07.resolve_grch38_fasta(project.root, args.fasta)
    row, exclusion = s07.build_annotation_row(
        args,
        task_id=1,
        accession=study.accession,
        fine_root=project.step_root("06_finemapping"),
        out_root=project.step_root("07_annotation"),
        phenotype=project.phenotype,
        ancestry_code=project.ancestry_code,
        ancestry_label=project.ancestry_label,
        vep=vep,
        vep_cache=cache,
        vep_fasta=fasta,
        cache_release=s07.detect_cache_release(cache),
    )
    if exclusion is not None:
        details["extra"] = exclusion
        return Outcome(gs.FAILED, str(exclusion.get("REASON", "")), details)

    details["forced_rerun"] = bool(force)
    manifest = write_one_row_manifest(study.manifest_path("Step07_manifest.tsv"), row)
    started = time.time()
    code = run_step_worker(study, "Step07_VEP", "Step07_Annotate_Finemapped_Variants", "GWAS_ANNOTATION_MANIFEST", manifest, details)
    if code != 0:
        return failed_from_worker(study, "Step07_VEP", code, details, f["failed"], started)
    if not vep_complete(study):
        return Outcome(gs.FAILED, "VEP worker exited 0 but outputs failed validation", details)
    details.update(summary_counts(f["summary_json"], *counts))
    return Outcome(gs.COMPLETE, "VEP annotation complete", details)


def stage_qtl(study: Study, force: bool) -> Outcome:
    f = qtl_files(study)
    details = {"input_file": str(vep_files(study)["variant_summary"]), "output_dir": str(f["dir"]),
               "output_files": [str(f["variant_summary"]), str(f["summary_json"])]}
    counts = ("N_INPUT_VARIANTS", "N_ANY_QTL_VARIANTS", "N_LOCI")
    if not force and qtl_complete(study):
        print("[SKIP] output already validated")
        details.update(summary_counts(f["summary_json"], *counts))
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "GTEx QTL outputs present", details)

    project = study.project
    s08 = step("Step08_Integrate_GTEx_eQTL_sQTL")
    args = project.step_args("Step08_Integrate_GTEx_eQTL_sQTL", force=force)
    s08.validate_arguments(args)
    row, exclusion = s08.build_qtl_row(
        args,
        task_id=1,
        accession=study.accession,
        annotation_root=project.step_root("07_annotation"),
        qtl_root=project.step_root("08_qtl"),
        phenotype=project.phenotype,
        ancestry_code=project.ancestry_code,
        ancestry_label=project.ancestry_label,
        gtex=s08.resolve_gtex_resources(project.root, args.gtex_root),
    )
    if exclusion is not None:
        details["extra"] = exclusion
        return Outcome(gs.FAILED, str(exclusion.get("REASON", "")), details)

    details["forced_rerun"] = bool(force)
    manifest = write_one_row_manifest(study.manifest_path("Step08_manifest.tsv"), row)
    started = time.time()
    # Per-study GTEx cache: <08_qtl>/<p>/<a>/<GCST>/shared/
    code = run_step_worker(
        study, "Step08_GTEx_QTL", "Step08_Integrate_GTEx_eQTL_sQTL", "GWAS_QTL_MANIFEST", manifest, details,
        extra_env={"STEP08_CACHE_ROOT": f["dir"]},
    )
    if code != 0:
        return failed_from_worker(study, "Step08_GTEx_QTL", code, details, f["failed"], started)
    if not qtl_complete(study):
        return Outcome(gs.FAILED, "GTEx QTL worker exited 0 but outputs failed validation", details)
    details.update(summary_counts(f["summary_json"], *counts))
    return Outcome(gs.COMPLETE, "GTEx eQTL/sQTL integration complete", details)


def stage_spliceai(study: Study, force: bool) -> Outcome:
    f = spliceai_files(study)
    details = {"input_file": str(qtl_files(study)["variant_summary"]), "output_dir": str(f["variant_summary"].parent),
               "output_files": [str(f["variant_summary"]), str(f["summary_json"])]}
    counts = ("N_INPUT_VARIANTS", "N_SPLICEAI_SCORED_VARIANTS", "N_LOCI")
    if not force and spliceai_complete(study):
        print("[SKIP] output already validated")
        details.update(summary_counts(f["summary_json"], *counts))
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "SpliceAI outputs present", details)

    project = study.project
    s09 = step("Step09_Run_SpliceAI")
    args = project.step_args("Step09_Run_SpliceAI", force=force)
    s09.validate_arguments(args)
    row, exclusion = s09.build_spliceai_row(
        args,
        task_id=1,
        accession=study.accession,
        qtl_root=project.step_root("08_qtl"),
        out_root=project.step_root("09_splicing"),
        phenotype=project.phenotype,
        ancestry_code=project.ancestry_code,
        ancestry_label=project.ancestry_label,
        resources=s09.resolve_resources(project.root, args),
    )
    if exclusion is not None:
        details["extra"] = exclusion
        return Outcome(gs.FAILED, str(exclusion.get("REASON", "")), details)

    details["forced_rerun"] = bool(force)
    manifest = write_one_row_manifest(study.manifest_path("Step09_manifest.tsv"), row)
    started = time.time()
    code = run_step_worker(study, "Step09_SpliceAI", "Step09_Run_SpliceAI", "GWAS_SPLICEAI_MANIFEST", manifest, details)
    if code != 0:
        return failed_from_worker(study, "Step09_SpliceAI", code, details, f["failed"], started)
    if not spliceai_complete(study):
        return Outcome(gs.FAILED, "SpliceAI worker exited 0 but outputs failed validation", details)
    details.update(summary_counts(f["summary_json"], *counts))
    return Outcome(gs.COMPLETE, "SpliceAI complete", details)


def stage_pangolin(study: Study, force: bool) -> Outcome:
    f = pangolin_files(study)
    details = {"input_file": str(spliceai_files(study)["variant_summary"]), "output_dir": str(f["integrated"].parent),
               "output_files": [str(f["integrated"]), str(f["summary_json"])]}
    counts = ("N_INPUT_VARIANTS", "N_PANGOLIN_SCORED", "N_LOCI")
    if not force and pangolin_complete(study):
        print("[SKIP] output already validated")
        details.update(summary_counts(f["summary_json"], *counts))
        return Outcome(gs.SKIPPED_ALREADY_COMPLETE, "Pangolin outputs present", details)

    project = study.project
    s10 = step("Step10_Run_Pangolin")
    args = project.step_args("Step10_Run_Pangolin", force=force)
    s10.validate_arguments(args)
    row, exclusion = s10.build_pangolin_row(
        args,
        task_id=1,
        accession=study.accession,
        step09_root=project.step_root("09_splicing"),
        out_root=project.step_root("10_pangolin"),
        phenotype=project.phenotype,
        ancestry_code=project.ancestry_code,
        ancestry_label=project.ancestry_label,
        resources=s10.resolve_resources(project.root, args),
    )
    if exclusion is not None:
        details["extra"] = exclusion
        return Outcome(gs.FAILED, str(exclusion.get("REASON", "")), details)

    details["forced_rerun"] = bool(force)
    manifest = write_one_row_manifest(study.manifest_path("Step10_manifest.tsv"), row)
    started = time.time()
    code = run_step_worker(study, "Step10_Pangolin", "Step10_Run_Pangolin", "GWAS_PANGOLIN_MANIFEST", manifest, details)
    if code != 0:
        return failed_from_worker(study, "Step10_Pangolin", code, details, f["failed"], started)
    if not pangolin_complete(study):
        return Outcome(gs.FAILED, "Pangolin worker exited 0 but outputs failed validation", details)
    details.update(summary_counts(f["summary_json"], *counts))
    return Outcome(gs.COMPLETE, "Pangolin complete", details)


STAGE_FUNCTIONS = {
    "Step02_Download": stage_download,
    "Step03_QC": stage_qc,
    "Step04_LD_Reference": stage_ld_reference,
    "Step05_Clumping": stage_clumping,
    "Step06_FineMapping": stage_finemapping,
    "Step07_VEP": stage_vep,
    "Step08_GTEx_QTL": stage_qtl,
    "Step09_SpliceAI": stage_spliceai,
    "Step10_Pangolin": stage_pangolin,
}


def run_stage(study: Study, stage: str, force: bool) -> Outcome:
    log_file = study.log_path(stage)
    started_utc = gs.utc_now()
    started = time.time()
    write_status(study, stage, status_record(study, stage, gs.RUNNING, started_utc=started_utc,
                                             log_path=str(log_file), forced_rerun=force))

    with tee_to(log_file):
        banner(f"{stage} | {study.accession} | {study.project.phenotype} | {study.project.ancestry_code}")
        print(f"Started : {started_utc}")
        print(f"Forced  : {force}")
        try:
            outcome = STAGE_FUNCTIONS[stage](study, force)
        except Exception as exc:
            traceback.print_exc()
            outcome = Outcome(
                gs.FAILED,
                f"{type(exc).__name__}: {exc}",
                {"error_type": type(exc).__name__, "error_message": str(exc)},
            )
        print(f"[{outcome.status}] {stage}: {outcome.reason}", flush=True)

    record = status_record(study, stage, outcome.status, outcome.reason, **outcome.details)
    record.update({
        "started_utc": started_utc,
        "finished_utc": gs.utc_now(),
        "runtime_seconds": round(time.time() - started, 1),
        "log_path": str(log_file),
        "forced_rerun": bool(force or outcome.details.get("forced_rerun")),
    })
    write_status(study, stage, record)
    return outcome


# =============================================================================
# RAW GWAS CLEANUP (only after VALIDATED QC)
# =============================================================================

def validate_qc_for_cleanup(study: Study) -> tuple[bool, str, dict]:
    f = qc_files(study)
    checks = {}
    status = gs.read_json(study.status_path("Step03_QC")).get("status")
    checks["QC_STAGE_STATUS"] = status
    if status not in gs.DONE_STATUSES:
        return False, f"Step03 status is {status}", checks
    for key in ("full", "significant", "summary_tsv", "summary_json"):
        if not non_empty(f[key]):
            return False, f"QC output missing/empty: {f[key]}", checks
    summary = gs.read_json(f["summary_json"])
    expected = as_int(summary.get("N_AFTER_QC"))
    checks["QC_SUMMARY_N_AFTER_QC"] = expected
    if expected is None:
        return False, "QC summary JSON unreadable or lacks N_AFTER_QC", checks
    if expected <= 0:
        return False, "QC output contains zero variants", checks
    try:
        rows = gs.count_gzip_data_rows(f["full"])
    except Exception as exc:
        return False, f"QC output could not be re-read: {type(exc).__name__}: {exc}", checks
    checks["QC_FILE_DATA_ROWS"] = rows
    if rows != expected:
        return False, f"QC file has {rows} rows but summary says {expected}", checks
    return True, "QC output re-read in full; row count matches QC summary", checks


def cleanup_raw(study: Study, keep_raw: bool) -> None:
    record_file = study.work / "raw_cleanup.json"
    raw = raw_file(study)
    if raw is None:
        return  # nothing to delete (already removed, record kept)

    record = {
        "STUDY_ACCESSION": study.accession,
        "RAW_FILE": str(raw),
        "RAW_SIZE_BYTES": raw.stat().st_size,
        "RAW_DELETED": False,
        "RAW_DELETED_UTC": None,
        "QC_FILE_RETAINED": str(qc_files(study)["full"]),
        "REASON": "",
        "CHECKED_UTC": gs.utc_now(),
    }
    if keep_raw:
        record["REASON"] = "--keep-raw"
        gs.atomic_write_json(record_file, record)
        return

    with tee_to(study.log_path("Step03_QC")):
        print("Validating QC output before removing the raw GWAS file ...", flush=True)
        ok, reason, checks = validate_qc_for_cleanup(study)
        record.update(checks)
        record["REASON"] = reason
        if ok:
            raw.unlink()
            record["RAW_DELETED"] = True
            record["RAW_DELETED_UTC"] = gs.utc_now()
            print(f"[CLEANUP] removed raw GWAS ({record['RAW_SIZE_BYTES']:,} bytes): {raw}")
        else:
            print(f"[CLEANUP] raw GWAS KEPT: {reason}")
    gs.atomic_write_json(record_file, record)


# =============================================================================
# ONE STUDY (one SLURM array task)
# =============================================================================

@contextlib.contextmanager
def study_lock(study: Study):
    """Refuse to process a study that another worker is processing."""
    lock_file = study.status_path("Step02_Download").parent / ".worker.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # non-POSIX (local testing): no locking available
        yield
        return
    with open(lock_file, "w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(
                f"Another worker is already processing {study.accession} "
                f"(lock: {lock_file}). Not starting a second one."
            )
        yield


def read_study_manifest(project: Project) -> pd.DataFrame:
    path = project.audit / "study_manifest.tsv"
    if not path.exists():
        raise SystemExit(f"Study manifest not found: {path}\nRun Step01_10_Run.py without --worker first.")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def run_study(project: Project, array_task_id: int, keep_raw: bool) -> int:
    manifest = read_study_manifest(project)
    selected = manifest[pd.to_numeric(manifest["ARRAY_TASK_ID"], errors="coerce") == array_task_id]
    if len(selected) != 1:
        raise SystemExit(f"Expected one study for ARRAY_TASK_ID={array_task_id}; found {len(selected)}")
    row = selected.iloc[0].to_dict()

    study = Study(
        project=project,
        accession=str(row["STUDY_ACCESSION"]).strip(),
        row=row,
        array_task_id=str(array_task_id),
        git=gs.git_commit(project.root),
    )

    banner(
        f"GWAS2m STUDY WORKER | task {array_task_id} | {study.accession}\n"
        f"Phenotype: {project.phenotype} | Ancestry: {project.ancestry_label} ({project.ancestry_code})"
    )

    with study_lock(study):
        # Fresh attempt: earlier statuses stay in history.jsonl.
        for stage in STUDY_STAGES:
            write_status(study, stage, status_record(study, stage, gs.PENDING), history=False)

        blocked: tuple[str, str] | None = None
        rebuilt = False
        failed = False

        for stage in STUDY_STAGES:
            if blocked is not None:
                status, reason = blocked
                print(f"[{status}] {stage}: {reason}")
                write_status(study, stage, status_record(
                    study, stage, status, reason,
                    started_utc=gs.utc_now(), finished_utc=gs.utc_now(), runtime_seconds=0,
                ))
                continue

            force = rebuilt and stage not in {"Step02_Download", "Step03_QC", "Step04_LD_Reference"}
            outcome = run_stage(study, stage, force)

            if outcome.status == gs.COMPLETE and stage in REBUILD_SOURCES:
                rebuilt = True

            if outcome.status in {gs.FAILED, gs.PARTIAL}:
                failed = True
                blocked = (gs.NOT_RUN_UPSTREAM_FAILED, f"Upstream {stage} {outcome.status}: {outcome.reason}")
            elif outcome.status in {gs.EXCLUDED, gs.NOT_AVAILABLE}:
                blocked = (gs.NOT_AVAILABLE, f"Upstream {stage} {outcome.status}: {outcome.reason}")
            elif stage == "Step06_FineMapping" and (
                outcome.details.get("n_loci") == 0
                or (outcome.details.get("extra") or {}).get("STATUS") == "NO_SIGNIFICANT_LOCI"
            ):
                blocked = (gs.NOT_AVAILABLE, "Step06: no genome-wide significant loci (NO_SIGNIFICANT_LOCI)")

            if stage == "Step03_QC" and outcome.status in gs.DONE_STATUSES:
                cleanup_raw(study, keep_raw)

    banner(f"STUDY {study.accession} FINISHED - {'WITH FAILURES' if failed else 'OK'}")
    return 1 if failed else 0


# =============================================================================
# STEP01 DISCOVERY + STUDY MANIFEST
# =============================================================================

def run_discovery(project: Project, args) -> dict:
    s01 = step("Step01_Plan_GWAS")
    analysis_dir = project.step_root("01_gwas_catalog")
    manifest_file = analysis_dir / "GWAS_download_manifest.tsv"
    selection_file = analysis_dir / "study_selection.tsv"
    summary_file = analysis_dir / "discovery_summary.json"

    argv = ["--phenotype", project.phenotype, "--ancestry", project.ancestry_input,
            "--max-studies", str(args.max_studies)]
    for alias in args.alias:
        argv += ["--alias", alias]
    if args.refresh_catalog:
        argv.append("--refresh")
    s01_args = s01.arguments(argv)
    terms = s01.build_terms(project.phenotype, s01_args.alias)

    previous = gs.read_json(summary_file)
    reuse = (
        not args.refresh_discovery
        and not args.refresh_catalog
        and manifest_file.exists()
        and selection_file.exists()
        and "N_ELIGIBLE" in previous
        and previous.get("SEARCH_TERMS") == terms
        and previous.get("MAX_STUDIES") == args.max_studies
    )

    result = {"status": gs.SKIPPED_ALREADY_COMPLETE, "error": None}
    if reuse:
        print("[SKIP] Step01 discovery outputs already validated "
              "(same search terms; use --refresh-discovery to re-run)")
    else:
        result["status"] = gs.COMPLETE
        try:
            s01.plan(s01_args)
        except RuntimeError as exc:
            # e.g. no phenotype match / no eligible study: a valid, recorded outcome
            result.update({"status": gs.FAILED, "error": str(exc).strip()})

    if selection_file.exists():
        shutil.copyfile(selection_file, project.audit / "study_selection.tsv")
    result.update({
        "summary": gs.read_json(summary_file),
        "manifest_file": manifest_file,
        "selection_file": selection_file,
    })
    return result


def write_study_manifest(project: Project, download_manifest: Path) -> pd.DataFrame:
    manifest = pd.read_csv(download_manifest, sep="\t", dtype=str, keep_default_na=False)
    manifest.insert(0, "ARRAY_TASK_ID", range(1, len(manifest) + 1))
    gs.atomic_write_tsv(manifest, project.audit / "study_manifest.tsv")
    return manifest


def write_discovery_statuses(project: Project, git: str) -> None:
    """Step01 record for every phenotype-matched study (selected or not)."""
    path = project.audit / "study_selection.tsv"
    if not path.exists():
        return
    selection = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    context = gs.runtime_context()
    for _, row in selection.iterrows():
        eligible = str(row.get("ELIGIBLE", "")).strip().lower() == "true"
        study = Study(project, str(row["STUDY_ACCESSION"]), row.to_dict(), "", context, git)
        reason = (
            f"Selected (rank {row.get('RANK', '')}, match score {row.get('MATCH_SCORE', '')})"
            if eligible else str(row.get("EXCLUSION_REASON", ""))
        )
        record = status_record(
            study, "Step01_Discovery", gs.COMPLETE if eligible else gs.EXCLUDED, reason,
            finished_utc=gs.utc_now(),
            output_files=[str(project.step_root("01_gwas_catalog") / "study_selection.tsv")],
            extra={k: row.get(k) for k in ("MATCH_SCORE", "ANCESTRY_CATEGORIES", "ANCESTRY_MATCH",
                                           "SUMMARY_STATS_AVAILABLE", "HARMONISED_FILE_AVAILABLE",
                                           "DOWNLOAD_URL", "RANK")},
        )
        write_status(study, "Step01_Discovery", record, history=False)


def write_initial_pending(project: Project, manifest: pd.DataFrame, task_ids: list[int], git: str) -> None:
    context = gs.runtime_context()
    chosen = manifest[pd.to_numeric(manifest["ARRAY_TASK_ID"]).isin(task_ids)]
    for _, row in chosen.iterrows():
        study = Study(project, str(row["STUDY_ACCESSION"]), row.to_dict(), str(row["ARRAY_TASK_ID"]), context, git)
        for stage in STUDY_STAGES:
            if not study.status_path(stage).exists():
                write_status(study, stage, status_record(study, stage, gs.PENDING), history=False)


# =============================================================================
# SLURM
# =============================================================================

def slurm_overrides(args) -> dict:
    """Command-line overrides of config/slurm.yaml (None = use the config)."""
    return {"partition": args.partition, "time": args.time, "memory": args.mem,
            "cpus": args.cpus, "max_parallel": args.max_parallel}


def controller_command(project: Project, python: Path, *flags: str) -> str:
    parts = [str(python), str(Path(__file__).resolve()),
             "--phenotype", project.phenotype, "--ancestry", project.ancestry_code, *flags]
    return " ".join(shlex.quote(x) for x in parts)


def array_chunks(task_ids: list[int], array_limit: int) -> list[list[int]]:
    """Split study rows so no array exceeds array_limit indices (never a concurrency cap)."""
    ids = sorted(task_ids)
    return [ids[i:i + array_limit] for i in range(0, len(ids), array_limit)]


def write_slurm_scripts(project: Project, args, task_ids: list[int]) -> tuple[list[dict], Path, dict]:
    """One study-array script per chunk + one audit script, all from config/slurm.yaml."""
    config = gwas2m_config.load_slurm_config(project.root)
    study = gwas2m_config.get_stage_resources("study", config, slurm_overrides(args))
    python = pipeline_python(project.root)
    log_root = project.root / "logs" / "step01_10" / project.phenotype_slug / project.ancestry_slug
    log_root.mkdir(parents=True, exist_ok=True)
    slurm_dir = project.audit / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    name = f"{project.phenotype_slug}_{project.ancestry_code.lower()}"
    exports = []
    if os.environ.get(gwas2m_config.CONFIG_ENV_VAR):
        exports.append(f"export {gwas2m_config.CONFIG_ENV_VAR}={shlex.quote(os.environ[gwas2m_config.CONFIG_ENV_VAR])}")

    worker_flags = ["--worker"] + (["--keep-raw"] if args.keep_raw else [])
    chunks = array_chunks(task_ids, study["array_limit"])
    natural = len(chunks) == 1 and max(task_ids) <= study["array_limit"]
    arrays = []
    for number, chunk in enumerate(chunks, start=1):
        suffix = "" if len(chunks) == 1 else f"_part{number}"
        script = slurm_dir / f"Step01_10_study_array{suffix}.sh"
        flags = list(worker_flags)
        if natural:
            spec = gwas2m_config.array_spec(chunk, study["max_parallel"])
        else:
            # Index i of this array -> i-th study row in the task list, so array
            # indices stay within the site's array-size limit.
            task_list = slurm_dir / f"Step01_10_task_list{suffix}.txt"
            task_list.write_text("\n".join(str(x) for x in chunk) + "\n", encoding="utf-8")
            flags += ["--task-list", str(task_list)]
            spec = gwas2m_config.array_spec(len(chunk), study["max_parallel"])
        header = gwas2m_config.build_sbatch_directives(
            "study", config=config, overrides=slurm_overrides(args),
            job_name=f"GWAS2m_{name}{suffix}", array=spec,
            output=f"{log_root}/study.%A_%a.out", error=f"{log_root}/study.%A_%a.err",
        )
        script.write_text(
            "#!/bin/bash\n" + header + "\n\n"
            "# GWAS2m Step02-Step10: one array task = one GWAS study.\n"
            "set -euo pipefail\n"
            f"cd {shlex.quote(str(project.root))}\n"
            + "".join(line + "\n" for line in exports)
            + controller_command(project, python, *flags) + "\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        arrays.append({"script": str(script), "array_spec": spec, "study_rows": chunk})

    audit_script = slurm_dir / "Step01_10_audit.sh"
    header = gwas2m_config.build_sbatch_directives(
        "audit", config=config, overrides={"partition": args.partition},
        job_name=f"GWAS2m_audit_{name}",
        output=f"{log_root}/audit.%j.out", error=f"{log_root}/audit.%j.err",
    )
    audit_script.write_text(
        "#!/bin/bash\n" + header + "\n\n"
        "# Runs after every study array task has ended (afterany): builds pipeline_audit tables.\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(project.root))}\n"
        + "".join(line + "\n" for line in exports)
        + controller_command(project, python, "--audit", "--finalize") + "\n",
        encoding="utf-8",
    )
    audit_script.chmod(0o755)

    effective = gwas2m_config.effective_config_record(config, list(SLURM_STAGES))
    effective["stages"]["study"] = study
    effective["command_line_overrides"] = {k: v for k, v in slurm_overrides(args).items() if v is not None}
    gs.atomic_write_json(project.audit / "effective_slurm_config.json", effective)
    return arrays, audit_script, study


def sbatch(script: Path, dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(str(script))
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    return output.split(";", 1)[0]


# =============================================================================
# PROVENANCE: THRESHOLDS + SOFTWARE VERSIONS
# =============================================================================

def collect_thresholds(project: Project) -> dict:
    out = {}
    s03 = step("Step03_QC_One_GWAS")
    out["Step03_QC"] = {"P_THRESHOLD": s03.P_THRESHOLD, "BUILD": "GRCh38",
                        "GWASLAB_BASIC_CHECK": "remove=True, remove_dup=True, normalize=True"}
    a = project.step_args("Step05_Clump_GWAS_Loci")
    out["Step05_Clumping"] = {"P1": a.p1, "P2": a.p2, "R2": a.r2, "KB": a.kb}
    s06 = step("Step06_FineMap_GWAS_Loci")
    a = project.step_args("Step06_FineMap_GWAS_Loci")
    out["Step06_FineMapping"] = {
        "LOCUS_WINDOW_KB": a.locus_window_kb, "REFERENCE_MAF": a.reference_maf,
        "REFERENCE_GENO": a.reference_geno, "MIN_VARIANTS": a.min_variants,
        "MAX_LD_VARIANTS": a.max_ld_variants, "COVERAGE": a.coverage, "SUSIE_L": a.susie_L,
        "MAX_SUSIE_L": a.max_susie_L, "LD_RIDGE": a.ld_ridge,
        "ESTIMATE_S_MAX": s06.DEFAULT_ESTIMATE_S_MAX,
    }
    s07 = step("Step07_Annotate_Finemapped_Variants")
    a = project.step_args("Step07_Annotate_Finemapped_Variants")
    out["Step07_VEP"] = {"MIN_PIP_PRIORITY_FLAG": a.min_pip, "ANNOTATION_SCOPE": s07.ANNOTATION_SCOPE,
                         "ASSEMBLY": s07.ASSEMBLY, "STEP07_VERSION": s07.STEP07_VERSION,
                         "ALLOW_PARTIAL_STEP06": a.allow_partial}
    s08 = step("Step08_Integrate_GTEx_eQTL_sQTL")
    out["Step08_GTEx_QTL"] = {"GTEX_RELEASE": "v11", "TISSUE_POLICY": "ALL_GTEX_V11_TISSUES",
                              "STEP08_VERSION": s08.STEP08_VERSION, "CACHE": "per-study (STEP08_CACHE_ROOT)"}
    s09 = step("Step09_Run_SpliceAI")
    a = project.step_args("Step09_Run_SpliceAI")
    out["Step09_SpliceAI"] = {"ANNOTATION": s09.ANNOTATION, "DISTANCE": a.distance, "MASK": a.mask,
                              "THRESHOLD_HIGH_RECALL": s09.THRESHOLD_HIGH_RECALL,
                              "THRESHOLD_RECOMMENDED": s09.THRESHOLD_RECOMMENDED,
                              "THRESHOLD_HIGH_PRECISION": s09.THRESHOLD_HIGH_PRECISION,
                              "STEP09_VERSION": s09.STEP09_VERSION}
    s10 = step("Step10_Run_Pangolin")
    a = project.step_args("Step10_Run_Pangolin")
    out["Step10_Pangolin"] = {"DISTANCE": a.distance, "MASK": a.mask, "STEP10_VERSION": s10.STEP10_VERSION}
    return out


def probe(command: list, pattern: str | None = None, timeout: int = 120) -> str:
    try:
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=timeout)
        text = (result.stdout + "\n" + result.stderr).strip()
    except Exception as exc:
        return f"UNAVAILABLE ({type(exc).__name__})"
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if pattern:
        lines = [x for x in lines if re.search(pattern, x, flags=re.I)]
    return " | ".join(lines[:4]) if lines else "UNAVAILABLE"


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "NOT_INSTALLED"


def env_python_package(python: Path, package: str) -> str:
    if not python.exists():
        return f"UNAVAILABLE (missing {python})"
    code = f"import importlib.metadata as m; print(m.version({package!r}))"
    return probe([python, "-c", code])


def collect_software_versions(project: Project) -> pd.DataFrame:
    root = project.root
    rows = []

    def add(category, name, version, path="", detail=""):
        rows.append({"CATEGORY": category, "NAME": name, "VERSION": version, "PATH": str(path), "DETAIL": detail})

    def tool(name: str) -> Path | None:
        candidate = root / "envs" / "pipeline" / "bin" / name
        if candidate.exists():
            return candidate
        found = shutil.which(name)
        return Path(found) if found else None

    def safe(function, *a):
        try:
            return function(*a)
        except Exception as exc:
            return f"UNAVAILABLE ({type(exc).__name__}: {exc})"

    add("PIPELINE", "git_commit", gs.git_commit(root), root)
    add("PIPELINE", "Step01_10_Run.py", CONTROLLER_VERSION, Path(__file__).resolve())
    add("PYTHON", "python", sys.version.split()[0], sys.executable)
    for name in ("pandas", "numpy", "requests", "gwaslab"):
        add("PYTHON_PACKAGE", name, package_version(name))

    rscript = tool("Rscript")
    add("R", "R", probe([rscript, "--version"], r"R (scripting front-end|version)") if rscript else "UNAVAILABLE", rscript or "")
    for package in ("susieR", "coloc"):
        version = probe([rscript, "-e", f'cat(as.character(packageVersion("{package}")))']) if rscript else "UNAVAILABLE"
        add("R_PACKAGE", package, version, rscript or "")

    for name in ("plink2", "bcftools", "samtools", "tabix"):
        path = tool(name)
        add("TOOL", name, probe([path, "--version"]).split(" | ")[0] if path else "UNAVAILABLE", path or "")

    s07 = step("Step07_Annotate_Finemapped_Variants")
    vep = safe(s07.resolve_vep_executable, root, None)
    if isinstance(vep, Path):
        add("TOOL", "ensembl-vep", probe([vep, "--help"], r"^(ensembl|ensembl-vep|ensembl-variation|ensembl-io|ensembl-funcgen)\s*:"), vep)
    else:
        add("TOOL", "ensembl-vep", vep)
    cache = safe(s07.resolve_vep_cache, root, None)
    if isinstance(cache, Path):
        add("RESOURCE", "VEP cache", f"release {s07.detect_cache_release(cache)} GRCh38", cache)
    else:
        add("RESOURCE", "VEP cache", cache)
    fasta = safe(s07.resolve_grch38_fasta, root, None)
    if isinstance(fasta, Path):
        sig = gs.file_signature(fasta)
        add("RESOURCE", "GRCh38 FASTA", "GRCh38", fasta, f"size={sig.get('SIZE_BYTES')} mtime={sig.get('MTIME_UTC')}")
    else:
        add("RESOURCE", "GRCh38 FASTA", fasta)

    add("TOOL", "SpliceAI", env_python_package(root / "envs" / "spliceai" / "bin" / "python", "spliceai"),
        root / "envs" / "spliceai", "models: built-in SpliceAI models; annotation=grch38")
    add("TOOL", "Pangolin", env_python_package(root / "envs" / "pangolin" / "bin" / "python", "pangolin"),
        root / "envs" / "pangolin", "models: built-in Pangolin models")
    s10 = step("Step10_Run_Pangolin")
    resources = safe(s10.resolve_resources, root, project.step_args("Step10_Run_Pangolin"))
    if isinstance(resources, dict):
        sig = gs.file_signature(resources["PANGOLIN_DB"])
        add("RESOURCE", "Pangolin annotation DB", Path(resources["PANGOLIN_DB"]).name, resources["PANGOLIN_DB"],
            f"size={sig.get('SIZE_BYTES')} mtime={sig.get('MTIME_UTC')}")
    else:
        add("RESOURCE", "Pangolin annotation DB", resources)

    s08 = step("Step08_Integrate_GTEx_eQTL_sQTL")
    gtex = safe(s08.resolve_gtex_resources, root, None)
    if isinstance(gtex, dict):
        add("RESOURCE", "GTEx", "v11", gtex["gtex_root"],
            f"eQTL files={len(gtex['eqtl_files'])} sQTL files={len(gtex['sqtl_files'])}; "
            f"lookup={gs.file_signature(gtex['lookup_file']).get('SIZE_BYTES')} bytes")
    else:
        add("RESOURCE", "GTEx", gtex)

    reference = root / "resources" / "1000G" / project.ancestry_code
    psam = reference / f"chr1_{project.ancestry_code}_GRCh38.psam"
    n_samples = safe(step("Step06_FineMap_GWAS_Loci").count_reference_samples, root, project.ancestry_code)
    add("RESOURCE", f"1000G {project.ancestry_code} LD reference", "1000 Genomes GRCh38", reference,
        f"samples={n_samples}; chr1 psam mtime={gs.file_signature(psam).get('MTIME_UTC')}")

    s01 = step("Step01_Plan_GWAS")
    for label, url, name in (("GWAS Catalog studies", s01.STUDIES_URL, "gwas_catalog_studies.tsv"),
                             ("GWAS Catalog ancestries", s01.ANCESTRY_URL, "gwas_catalog_ancestry.tsv")):
        sig = gs.file_signature(root / "reference_metadata" / "gwas_catalog" / name)
        add("RESOURCE", label, url, sig["PATH"], f"size={sig.get('SIZE_BYTES')} mtime={sig.get('MTIME_UTC')}")

    add("BUILD", "genome_build", "GRCh38")
    for stage, values in collect_thresholds(project).items():
        for key, value in values.items():
            add("THRESHOLD", f"{stage}.{key}", value)
    return pd.DataFrame(rows)


def update_run_manifest(project: Project, updates: dict, submission: dict | None = None) -> dict:
    path = project.audit / "run_manifest.json"
    manifest = gs.read_json(path)
    manifest.setdefault("pipeline", "GWAS2m Step01-10")
    manifest.setdefault("created_utc", gs.utc_now())
    manifest.setdefault("submissions", [])
    manifest.update(updates)
    if submission:
        manifest["submissions"].append(submission)
    gs.atomic_write_json(path, manifest)
    return manifest


# =============================================================================
# AUDIT TABLES
# =============================================================================

STATUS_COLUMNS = {
    "phenotype": "PHENOTYPE", "ancestry_code": "ANCESTRY_CODE", "ancestry_label": "ANCESTRY_LABEL",
    "study_accession": "STUDY_ACCESSION", "array_task_id": "ARRAY_TASK_ID", "stage": "STAGE",
    "status": "STATUS", "reason": "REASON", "started_utc": "START_TIME", "finished_utc": "END_TIME",
    "runtime_seconds": "RUNTIME_SECONDS", "input_file": "INPUT_FILE", "output_dir": "OUTPUT_DIR",
    "n_input_variants": "N_INPUT_VARIANTS", "n_output_variants": "N_OUTPUT_VARIANTS",
    "n_loci": "N_LOCI", "n_loci_success": "N_LOCI_SUCCESS", "n_loci_failed": "N_LOCI_FAILED",
    "forced_rerun": "FORCED_RERUN", "error_type": "ERROR_TYPE", "error_message": "ERROR_MESSAGE",
    "log_path": "LOG_PATH", "failure_marker": "FAILURE_MARKER", "step_manifest": "STEP_MANIFEST",
    "hostname": "HOSTNAME", "slurm_job_id": "SLURM_JOB_ID", "slurm_array_job_id": "SLURM_ARRAY_JOB_ID",
    "slurm_array_task_id": "SLURM_ARRAY_TASK_ID", "slurm_partition": "PARTITION",
    "slurm_account": "ACCOUNT", "git_commit": "GIT_COMMIT",
}


def finalize_stale(project: Project, manifest: pd.DataFrame, task_ids: list[int]) -> None:
    """After the array has ended, RUNNING/PENDING can only mean a killed task."""
    git = gs.git_commit(project.root)
    for _, row in manifest.iterrows():
        if as_int(row["ARRAY_TASK_ID"]) not in task_ids:
            continue
        study = Study(project, str(row["STUDY_ACCESSION"]), row.to_dict(), str(row["ARRAY_TASK_ID"]),
                      gs.runtime_context(), git)
        blocked = None
        for stage in STUDY_STAGES:
            record = gs.read_json(study.status_path(stage))
            status = record.get("status")
            if blocked and status in {gs.PENDING, gs.RUNNING, None}:
                write_status(study, stage, status_record(study, stage, gs.NOT_RUN_UPSTREAM_FAILED, blocked,
                                                         finished_utc=gs.utc_now()))
                continue
            if status in {gs.RUNNING, gs.PENDING, None}:
                started = status == gs.RUNNING
                error_type = "TaskTerminated" if started else "TaskNotStarted"
                message = (
                    "Stage was still RUNNING when the SLURM array ended (TIMEOUT, OUT_OF_MEMORY, "
                    "NODE_FAIL or cancellation); see the SLURM log in logs/step01_10/"
                    if started else
                    "Array task ended before this stage started (TIMEOUT, OUT_OF_MEMORY, NODE_FAIL "
                    "or cancellation); see the SLURM log in logs/step01_10/"
                )
                record = status_record(study, stage, gs.FAILED, f"{error_type}: {message}",
                                       started_utc=record.get("started_utc"), finished_utc=gs.utc_now(),
                                       error_type=error_type, error_message=message,
                                       log_path=record.get("log_path"))
                write_status(study, stage, record)
                blocked = f"Upstream {stage} FAILED: {error_type}"
            elif status in {gs.FAILED, gs.PARTIAL}:
                blocked = f"Upstream {stage} {status}"


def locus_status_table(project: Project, accessions: list[str]) -> pd.DataFrame:
    rows = []
    for accession in accessions:
        study_dir = project.step_root("06_finemapping") / accession
        loci_file = study_dir / "loci_definition.tsv"
        if not non_empty(loci_file):
            continue
        try:
            loci = pd.read_csv(loci_file, sep="\t", dtype=str)
        except Exception:
            continue
        for _, locus in loci.iterrows():
            locus_dir = study_dir / "loci" / str(locus["LOCUS_ID"])
            failed = locus_dir / "LOCUS_FAILED.txt"
            if failed.exists():
                status = gs.FAILED
                error = failed.read_text(encoding="utf-8", errors="replace").splitlines()[0][:500]
            elif (locus_dir / "LOCUS_COMPLETE.json").exists() or non_empty(locus_dir / "susie" / "susie_summary.tsv"):
                status, error = gs.COMPLETE, ""
            else:
                status, error = "NOT_RUN", ""
            rows.append({
                "STUDY_ACCESSION": accession, "LOCUS_ID": locus["LOCUS_ID"], "CHR": locus.get("CHR"),
                "LOCUS_START": locus.get("LOCUS_START"), "LOCUS_END": locus.get("LOCUS_END"),
                "N_INDEPENDENT_SIGNALS": locus.get("N_INDEPENDENT_SIGNALS"), "LEAD_IDS": locus.get("LEAD_IDS"),
                "STATUS": status, "ERROR": error, "LOCUS_DIR": str(locus_dir),
                "SUSIE_FIT": str(locus_dir / "susie" / "susie_fit.rds"),
            })
    return pd.DataFrame(rows)


def build_audit(project: Project, finalize: bool) -> None:
    audit = project.audit
    audit.mkdir(parents=True, exist_ok=True)
    manifest_path = audit / "study_manifest.tsv"
    manifest = (pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
                if manifest_path.exists() else pd.DataFrame(columns=["ARRAY_TASK_ID", "STUDY_ACCESSION"]))
    run_manifest = gs.read_json(audit / "run_manifest.json")

    if finalize and run_manifest.get("submissions"):
        finalize_stale(project, manifest, run_manifest["submissions"][-1].get("array_task_ids", []))

    statuses = gs.read_all_statuses(audit)
    if statuses.empty:
        statuses = pd.DataFrame(columns=list(STATUS_COLUMNS))
    statuses["_order"] = statuses["stage"].map(gs.STAGE_ORDER)
    statuses = statuses.sort_values(["study_accession", "_order"], kind="stable")

    # ---- study_stage_status.tsv ---------------------------------------------
    stage_table = statuses.reindex(columns=list(STATUS_COLUMNS)).rename(columns=STATUS_COLUMNS)
    # SLURM request of the submission that launched each study task.
    launched_by = {}
    for submission in run_manifest.get("submissions", []):
        for task_id in submission.get("array_task_ids", []):
            launched_by[str(task_id)] = submission
    for column, key in (("TIME_LIMIT", "time"), ("MEMORY_REQUESTED", "memory"), ("CPUS_REQUESTED", "cpus"),
                        ("MAX_PARALLEL", "max_parallel")):
        stage_table[column] = [
            (launched_by.get(str(t), {}).get("slurm") or {}).get(key) for t in stage_table["ARRAY_TASK_ID"]
        ]
    stage_table["ARRAY_RANGE"] = [
        ";".join(a["array_spec"] for a in launched_by.get(str(t), {}).get("arrays", [])) or None
        for t in stage_table["ARRAY_TASK_ID"]
    ]
    for column, key in (("PARTITION", "partition"), ("ACCOUNT", "account")):
        requested = [(launched_by.get(str(t), {}).get("slurm") or {}).get(key) for t in stage_table["ARRAY_TASK_ID"]]
        stage_table[column] = stage_table[column].where(stage_table[column].notna() & (stage_table[column] != ""),
                                                        requested)
    gs.atomic_write_tsv(stage_table, audit / "study_stage_status.tsv")

    # ---- exclusion_reasons.tsv ----------------------------------------------
    not_ok = stage_table[~stage_table["STATUS"].isin(gs.DONE_STATUSES | {gs.PENDING, gs.RUNNING})]
    exclusion_cols = ["STUDY_ACCESSION", "STAGE", "STATUS", "REASON", "ERROR_TYPE", "ERROR_MESSAGE", "LOG_PATH"]
    gs.atomic_write_tsv(not_ok.reindex(columns=exclusion_cols), audit / "exclusion_reasons.tsv")

    accessions = manifest["STUDY_ACCESSION"].astype(str).tolist()

    # ---- download_provenance.tsv (+ raw cleanup record) ---------------------
    download_rows = []
    for accession in accessions:
        work = gs.study_work_dir(audit, accession)
        record = gs.read_json(work / "download_provenance.json") or {"STUDY_ACCESSION": accession}
        cleanup = gs.read_json(work / "raw_cleanup.json")
        for key in ("RAW_FILE", "RAW_SIZE_BYTES", "RAW_DELETED", "RAW_DELETED_UTC", "QC_FILE_RETAINED"):
            record[key] = cleanup.get(key)
        record["RAW_CLEANUP_REASON"] = cleanup.get("REASON")
        download_rows.append(record)
    gs.atomic_write_tsv(pd.DataFrame(download_rows), audit / "download_provenance.tsv")

    # ---- qc_metrics.tsv (exactly the metrics Step03 computes) ---------------
    qc_rows = []
    for accession in accessions:
        summary = gs.read_json(project.step_root("02_summary_stats") / "qc" / accession / f"{accession}_QC_summary.json")
        if summary:
            qc_rows.append(summary)
    gs.atomic_write_tsv(pd.DataFrame(qc_rows), audit / "qc_metrics.tsv")

    # ---- locus_status.tsv ---------------------------------------------------
    gs.atomic_write_tsv(locus_status_table(project, accessions), audit / "locus_status.tsv")

    # ---- step11_ready_studies.tsv -------------------------------------------
    by_key = {(r["study_accession"], r["stage"]): r for _, r in statuses.iterrows()}

    def stage_status(accession, stage):
        record = by_key.get((accession, stage))
        return record["status"] if record is not None else ""

    ready_rows = []
    for accession in accessions:
        s06 = by_key.get((accession, "Step06_FineMapping"))
        if s06 is None or s06["status"] not in gs.DONE_STATUSES:
            continue
        n_success = as_int(s06.get("n_loci_success")) or 0
        if n_success <= 0 or (as_int(s06.get("n_loci_failed")) or 0) > 0:
            continue
        study_dir = project.step_root("06_finemapping") / accession
        loci = locus_status_table(project, [accession])
        fits_ok = not loci.empty and all(Path(p).exists() for p in loci["SUSIE_FIT"])
        if not fits_ok:
            continue
        downstream = {stage: stage_status(accession, stage) for stage in STUDY_STAGES[5:]}
        ready_rows.append({
            "STUDY_ACCESSION": accession,
            "PHENOTYPE": project.phenotype,
            "ANCESTRY_CODE": project.ancestry_code,
            "ANCESTRY_LABEL": project.ancestry_label,
            "STEP06_DIR": str(study_dir),
            "FINEMAPPED_VARIANTS_FILE": str(study_dir / f"{accession}_finemapped_variants.tsv.gz"),
            "N_LOCI": as_int(s06.get("n_loci")),
            "N_LOCI_FINEMAPPED": n_success,
            **{f"{stage.upper()}_STATUS": value for stage, value in downstream.items()},
            "ALL_STEPS_02_10_COMPLETE": all(stage_status(accession, s) in gs.DONE_STATUSES for s in STUDY_STAGES),
        })
    ready = pd.DataFrame(ready_rows)
    gs.atomic_write_tsv(ready, audit / "step11_ready_studies.tsv")

    # ---- study_summary.tsv --------------------------------------------------
    selection_path = audit / "study_selection.tsv"
    selection = (pd.read_csv(selection_path, sep="\t", dtype=str, keep_default_na=False)
                 if selection_path.exists() else pd.DataFrame())
    discovery = gs.read_json(project.step_root("01_gwas_catalog") / "discovery_summary.json")

    def flag(column, value="true"):
        if selection.empty or column not in selection.columns:
            return 0
        return int((selection[column].astype(str).str.lower() == value).sum())

    def count(stage, statuses_wanted):
        sub = statuses[(statuses["stage"] == stage) & statuses["study_accession"].isin(accessions)]
        return int(sub["status"].isin(statuses_wanted).sum())

    s06_rows = statuses[(statuses["stage"] == "Step06_FineMapping") & statuses["status"].isin(gs.DONE_STATUSES | {gs.PARTIAL})]
    n_loci = pd.to_numeric(s06_rows.get("n_loci"), errors="coerce").fillna(0) if not s06_rows.empty else pd.Series(dtype=float)
    cleanup = pd.DataFrame(download_rows)
    done = gs.DONE_STATUSES

    metrics = [
        ("N_CATALOG_STUDIES_SCANNED", discovery.get("N_CATALOG_STUDIES"), "GWAS Catalog studies searched"),
        ("N_STUDIES_DISCOVERED", len(selection), "Studies matching any phenotype search term (rows of study_selection.tsv)"),
        ("N_PHENOTYPE_MATCHED", len(selection), "Same set as N_STUDIES_DISCOVERED (discovery is phenotype matching)"),
        ("N_ANCESTRY_MATCHED", flag("ANCESTRY_MATCH"), "Discovered studies whose discovery ancestry is exclusively the requested ancestry"),
        ("N_WITH_FULL_SUMMARY_STATISTICS", flag("SUMMARY_STATS_AVAILABLE"), "Discovered studies flagged with full summary statistics"),
        ("N_WITH_HARMONISED_FILE", flag("HARMONISED_FILE_AVAILABLE"), "Harmonised file found (checked only for sumstats + ancestry matches)"),
        ("N_ELIGIBLE", flag("ELIGIBLE"), "Selected for processing (study_manifest.tsv)"),
        ("N_ARRAY_TASKS", len(manifest), "Studies in the SLURM study array"),
        ("N_DOWNLOADED", count("Step02_Download", done), "Step02 COMPLETE or SKIPPED_ALREADY_COMPLETE"),
        ("N_DOWNLOAD_FAILURES", count("Step02_Download", {gs.FAILED}), "Step02 FAILED"),
        ("N_QC_COMPLETE", count("Step03_QC", done), "Step03 COMPLETE or SKIPPED_ALREADY_COMPLETE"),
        ("N_QC_FAILURES", count("Step03_QC", {gs.FAILED}), "Step03 FAILED"),
        ("N_RAW_FILES_DELETED_AFTER_QC", int(cleanup.get("RAW_DELETED", pd.Series(dtype=object)).astype(str).eq("True").sum()) if not cleanup.empty else 0,
         "Raw GWAS removed after validated QC"),
        ("N_CLUMPING_COMPLETE", count("Step05_Clumping", done), "Step05 complete"),
        ("N_CLUMPING_EXCLUDED", count("Step05_Clumping", {gs.EXCLUDED}), "Step05 excluded (QC CLUMPING_READY=False)"),
        ("N_STUDIES_WITH_LOCI", int((n_loci > 0).sum()), "Step06 complete/partial with >=1 locus"),
        ("N_STUDIES_NO_SIGNIFICANT_LOCI", int(((n_loci == 0) & s06_rows["status"].isin(done)).sum()) if not s06_rows.empty else 0,
         "Step06 complete with zero independent genome-wide significant loci"),
        ("N_LOCI_TOTAL", int(n_loci.sum()), "Loci defined by Step06 across studies"),
        ("N_LOCI_FINEMAPPED", int(pd.to_numeric(s06_rows.get("n_loci_success"), errors="coerce").fillna(0).sum()) if not s06_rows.empty else 0,
         "Loci with successful SuSiE-RSS fine-mapping"),
        ("N_LOCI_FAILED", int(pd.to_numeric(s06_rows.get("n_loci_failed"), errors="coerce").fillna(0).sum()) if not s06_rows.empty else 0,
         "Loci whose fine-mapping failed"),
        ("N_FINEMAPPING_PARTIAL", count("Step06_FineMapping", {gs.PARTIAL}), "Studies with some failed loci"),
        ("N_FINEMAPPING_FAILED", count("Step06_FineMapping", {gs.FAILED}), "Studies whose fine-mapping failed"),
        ("N_VEP_COMPLETE", count("Step07_VEP", done), "Step07 complete"),
        ("N_QTL_ANNOTATION_COMPLETE", count("Step08_GTEx_QTL", done), "Step08 complete"),
        ("N_SPLICEAI_COMPLETE", count("Step09_SpliceAI", done), "Step09 complete"),
        ("N_PANGOLIN_COMPLETE", count("Step10_Pangolin", done), "Step10 complete"),
        ("N_STUDIES_READY_FOR_STEP11", len(ready), "Rows of step11_ready_studies.tsv"),
        ("N_STUDIES_ALL_STEPS_02_10_COMPLETE",
         int(ready["ALL_STEPS_02_10_COMPLETE"].sum()) if not ready.empty else 0, "Step11-ready studies with Steps02-10 all complete"),
    ]
    summary = pd.DataFrame(metrics, columns=["METRIC", "VALUE", "DEFINITION"])
    summary.insert(0, "ANCESTRY_CODE", project.ancestry_code)
    summary.insert(0, "PHENOTYPE", project.phenotype)
    gs.atomic_write_tsv(summary, audit / "study_summary.tsv")

    # ---- software_versions.tsv + run_manifest.json --------------------------
    gs.atomic_write_tsv(collect_software_versions(project), audit / "software_versions.tsv")
    update_run_manifest(project, {
        "last_audit_utc": gs.utc_now(),
        "last_audit_host": gs.runtime_context(),
        "audit_finalized": bool(finalize),
        "study_summary": {m: v for m, v, _ in metrics},
    })

    print_status(project)
    print(f"\nAudit directory: {audit}")


def print_status(project: Project) -> None:
    reporter = step("Pipeline_reporter")
    reporter.print_step01_10_status(
        project.root, project.phenotype, project.ancestry_code, project.ancestry_label,
        project.phenotype_slug, project.ancestry_slug,
    )


# =============================================================================
# CONTROLLER (submission)
# =============================================================================

def preflight(project: Project) -> None:
    """Read-only setup check (central Step00 inspector) before any job is submitted."""
    try:
        gwas2m_config.load_slurm_config(project.root)
    except gwas2m_config.SlurmConfigError as exc:
        raise SystemExit(f"SETUP PREFLIGHT: FAIL\n\nSLURM configuration is invalid:\n{exc}")
    report = gwas2m_resources.inspect_pipeline(project.root, project.ancestry_code, through_step=10)
    if report["pipeline_ready"]:
        print(f"SETUP PREFLIGHT: PASS ({project.ancestry_code}, Steps01-10)")
        return
    print("SETUP PREFLIGHT: FAIL\n\nMissing:")
    for item in report["missing_required"]:
        print(f"- {item}")
    raise SystemExit(
        "\nNothing was submitted. Run:\n"
        f"  python Step00_Check_Resources.py --inspect --ancestry {project.ancestry_code}\n"
        "then the setup action it reports (e.g. python Step00_Check_Resources.py --resources-only).\n"
        "(--skip-preflight bypasses this check; missing resources then fail per study.)"
    )


def submit(project: Project, args) -> int:
    project.audit.mkdir(parents=True, exist_ok=True)
    git = gs.git_commit(project.root)

    banner(f"GWAS2m Step01-10 | {project.phenotype} | {project.ancestry_label} ({project.ancestry_code})")
    if not args.skip_preflight:
        preflight(project)

    discovery = run_discovery(project, args)
    summary = discovery["summary"]
    write_discovery_statuses(project, git)

    base = {
        "controller_version": CONTROLLER_VERSION,
        "command_line": [sys.executable, *sys.argv],
        "working_directory": str(project.root),
        "phenotype": project.phenotype,
        "phenotype_slug": project.phenotype_slug,
        "ancestry_input": project.ancestry_input,
        "ancestry_code": project.ancestry_code,
        "ancestry_label": project.ancestry_label,
        "ancestry_slug": project.ancestry_slug,
        "git_commit": git,
        "python": sys.version.split()[0],
        "discovery": summary,
        "discovery_status": discovery["status"],
        "discovery_error": discovery["error"],
        "thresholds": collect_thresholds(project),
        "keep_raw": bool(args.keep_raw),
    }

    if discovery["status"] == gs.FAILED or not discovery["manifest_file"].exists():
        update_run_manifest(project, base)
        build_audit(project, finalize=False)
        print("\nGWAS2m Step01-10")
        print(f"Phenotype          : {project.phenotype}")
        print(f"Ancestry           : {project.ancestry_code}")
        print(f"Studies discovered : {summary.get('N_PHENOTYPE_MATCHED', 0)}")
        print("Eligible studies   : 0")
        print("Array tasks        : 0")
        print(f"\nNothing submitted: {discovery['error']}")
        return 1

    manifest = write_study_manifest(project, discovery["manifest_file"])
    task_ids = manifest["ARRAY_TASK_ID"].astype(int).tolist()
    if args.study:
        wanted = set(args.study)
        unknown = wanted - set(manifest["STUDY_ACCESSION"])
        if unknown:
            raise SystemExit(f"--study not among eligible studies: {sorted(unknown)}")
        task_ids = manifest.loc[manifest["STUDY_ACCESSION"].isin(wanted), "ARRAY_TASK_ID"].astype(int).tolist()

    arrays, audit_script, study_resources = write_slurm_scripts(project, args, task_ids)
    dry = args.no_submit or args.dry_run
    submission = {
        "submitted_utc": gs.utc_now(),
        "mode": "local" if args.local else ("dry-run" if dry else "slurm"),
        "array_task_ids": task_ids,
        "arrays": arrays,
        "audit_script": str(audit_script),
        "slurm": {key: study_resources.get(key) for key in
                  ("partition", "account", "qos", "time", "memory", "cpus", "max_parallel", "array_limit")},
        "slurm_config": str(gwas2m_config.resolve_config_path(project.root)),
        "submit_host": gs.runtime_context(),
    }

    array_jobs, audit_job = [], None
    if args.local:
        write_initial_pending(project, manifest, task_ids, git)
        update_run_manifest(project, base, submission)
        failures = sum(run_study(project, task_id, args.keep_raw) for task_id in task_ids)
        build_audit(project, finalize=True)
    elif not dry:
        write_initial_pending(project, manifest, task_ids, git)
        array_jobs = [sbatch(Path(a["script"])) for a in arrays]
        audit_job = sbatch(audit_script, dependency="afterany:" + ":".join(array_jobs))
        submission.update({"array_job_ids": array_jobs, "audit_job_id": audit_job})
        update_run_manifest(project, base, submission)
    else:
        update_run_manifest(project, base, submission)

    print("\nGWAS2m Step01-10\n")
    print(f"Phenotype          : {project.phenotype}")
    print(f"Ancestry           : {project.ancestry_code} ({project.ancestry_label})")
    print(f"Studies discovered : {summary.get('N_PHENOTYPE_MATCHED', '?')}")
    print(f"Eligible studies   : {len(manifest)}")
    print(f"Array tasks        : {len(task_ids)}  "
          f"(--array={' + '.join(a['array_spec'] for a in arrays)})")
    print(f"SLURM config       : {submission['slurm_config']}")
    print(f"Per-task request   : time={study_resources['time']} mem={study_resources['memory']} "
          f"cpus={study_resources['cpus']} partition={gwas2m_config.describe(study_resources['partition'])} "
          f"throttle={study_resources['max_parallel'] or 'none'}")
    print(f"Audit directory    : {project.audit}")
    if array_jobs:
        print(f"\nSubmitted job:\n{' '.join(array_jobs)}")
        print(f"Audit job (afterany): {audit_job}")
    elif dry:
        print("\nNot submitted (--dry-run / --no-submit). Generated scripts:")
        for a in arrays:
            print(f"\n--- {a['script']}")
            print(Path(a["script"]).read_text(encoding="utf-8"))
        print(f"--- {audit_script}")
        print(audit_script.read_text(encoding="utf-8"))
        print("To submit:")
        print("  jid=$(sbatch --parsable <array script>)   # once per array script")
        print(f"  sbatch --dependency=afterany:$jid {audit_script}")
    elif args.local:
        print(f"\nLocal run finished; studies with failures: {failures}")
    print("\nStatus at any time:")
    print(f"  python Pipeline_reporter.py --phenotype {shlex.quote(project.phenotype)} "
          f"--ancestry {project.ancestry_code} --status")
    return 0


# =============================================================================
# CLI
# =============================================================================

def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Generic GWAS2m Step01-10 controller: one SLURM array task per GWAS study.",
    )
    parser.add_argument("--phenotype", required=True, help="Any phenotype, e.g. migraine or \"parkinson's disease\".")
    parser.add_argument("--ancestry", required=True, help="EUR, AFR, EAS, SAS, AMR or the full label.")

    discovery = parser.add_argument_group("Step01 discovery (passed to Step01_Plan_GWAS.py)")
    discovery.add_argument("--alias", action="append", default=[], help="Extra phenotype search term (repeatable).")
    discovery.add_argument("--max-studies", type=int, default=0, help="0 = all eligible studies (Step01 default).")
    discovery.add_argument("--refresh-discovery", action="store_true", help="Re-run Step01 even if its outputs exist.")
    discovery.add_argument("--refresh-catalog", action="store_true", help="Re-download GWAS Catalog metadata (Step01 --refresh).")

    run = parser.add_argument_group("Execution")
    run.add_argument("--study", action="append", default=[], help="Only submit these GCST accessions (repeatable).")
    run.add_argument("--keep-raw", action="store_true", help="Never delete the raw downloaded GWAS after QC.")
    run.add_argument("--dry-run", action="store_true",
                     help="Discovery + manifest + SLURM scripts; print them and submit nothing.")
    run.add_argument("--no-submit", action="store_true", help="Alias for --dry-run.")
    run.add_argument("--local", action="store_true", help="Run all studies sequentially here (no SLURM).")
    run.add_argument("--skip-preflight", action="store_true",
                     help="Skip the setup preflight (missing resources then fail per study).")

    slurm = parser.add_argument_group(
        "SLURM (normally set in config/slurm.yaml, stage 'study'; these flags override it)")
    slurm.add_argument("--slurm-config", default=None, help="SLURM config file (default config/slurm.yaml).")
    slurm.add_argument("--partition", default=None)
    slurm.add_argument("--time", default=None, help="Walltime per study task (resubmit to resume).")
    slurm.add_argument("--mem", default=None)
    slurm.add_argument("--cpus", type=int, default=None)
    slurm.add_argument("--max-parallel", type=int, default=None, help="Array throttle %%N; 0 = none.")

    internal = parser.add_argument_group("Internal modes")
    internal.add_argument("--worker", action="store_true", help="Process one study (SLURM_ARRAY_TASK_ID).")
    internal.add_argument("--array-id", type=int, default=None, help="Study row for --worker outside SLURM.")
    internal.add_argument("--task-list", default=None,
                          help="With --worker: file of study rows; array index i -> line i (split arrays).")
    internal.add_argument("--audit", action="store_true", help="Rebuild pipeline_audit tables.")
    internal.add_argument("--finalize", action="store_true", help="With --audit: mark stale RUNNING stages FAILED.")

    args = parser.parse_args(argv)
    if args.max_studies < 0 or (args.cpus is not None and args.cpus < 1) or (
            args.max_parallel is not None and args.max_parallel < 0):
        parser.error("--max-studies/--max-parallel must be >= 0 and --cpus >= 1")
    return args


def main(argv=None) -> int:
    args = arguments(argv)
    if args.slurm_config:
        # exported so that submitted jobs (and every Step script) read the same file
        os.environ[gwas2m_config.CONFIG_ENV_VAR] = str(Path(args.slurm_config).resolve())
    project = make_project(args.phenotype, args.ancestry)

    if args.worker:
        task = args.array_id if args.array_id is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is None:
            raise SystemExit("--worker needs SLURM_ARRAY_TASK_ID or --array-id")
        if args.task_list:
            rows = [line.strip() for line in Path(args.task_list).read_text().splitlines() if line.strip()]
            task = rows[int(task) - 1]
        return run_study(project, int(task), args.keep_raw)

    if args.audit:
        build_audit(project, finalize=args.finalize)
        return 0

    return submit(project, args)


if __name__ == "__main__":
    sys.exit(main())

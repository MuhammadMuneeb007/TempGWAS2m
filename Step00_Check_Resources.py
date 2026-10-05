#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
PIPELINE REPRODUCIBILITY BOOTSTRAP
===============================================================================

For the current:

GWAS
 -> QC
 -> ancestry-specific LD
 -> clumping
 -> SuSiE
 -> VEP
 -> GTEx
 -> SpliceAI
 -> Pangolin
 -> colocalisation
 -> FUMA / downstream interpretation

pipeline.

Everything is created beneath the CURRENT WORKING DIRECTORY.

Recommended HPC use (full one-command bootstrap):

    python Step00_Check_Resources.py

Generate files only:

    python Step00_Check_Resources.py --generate-only

Install/repair software only:

    python Step00_Check_Resources.py --install-only

Submit shared resources only:

    python Step00_Check_Resources.py --resources-only

Inspect only (READ-ONLY: never installs, downloads, deletes, writes or submits):

    python Step00_Check_Resources.py --inspect
    python Step00_Check_Resources.py --inspect --ancestry EUR
    python Step00_Check_Resources.py --inspect --through-step 11
    python Step00_Check_Resources.py --inspect --json
    python Step00_Check_Resources.py --inspect --json --output resource_inspection.json

    (--status is an alias for --inspect.)

Step00 is the single authority for shared software and resources. The
registry lives in gwas2m_resources.py; SLURM settings come only from
config/slurm.yaml (or --slurm-config FILE). Scientific steps consume these
resources and never install or download them. Phenotype-specific GWAS summary
statistics are runtime inputs (Step01 selects, Step02 downloads).

===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import selectors
from pathlib import Path


# =============================================================================
# ROOT
# =============================================================================

ROOT = Path.cwd().resolve()

RESOURCES = ROOT / "resources"
EXTERNAL = ROOT / "external_tools"
ENVS = ROOT / "envs"
LOGS = ROOT / "logs" / "resource_setup"
MANIFESTS = ROOT / "setup_manifests"

FUMA = ROOT / "fuma_reproducibility"


# =============================================================================
# VERSIONS
# =============================================================================

GENCODE_RELEASE = "50"

VEP_RELEASE = "116"

GTEX_RELEASE = "v11"

OPEN_TARGETS_RELEASE = "26.09"

REFERENCE_BUILD = "GRCh38"

REFERENCE_PANEL = "1000G_GRCh38_20190312"


# =============================================================================
# ARGUMENTS (parsed first so that --inspect never reaches any writing code)
# =============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gwas2m_config  # noqa: E402
import gwas2m_resources  # noqa: E402

parser = argparse.ArgumentParser(
    description="Install software, prepare reproducible resources, and verify the GWAS mechanism pipeline."
)

parser.add_argument("--inspect", action="store_true",
                    help="READ-ONLY: report software, packages, resources, SLURM config and readiness.")
parser.add_argument("--status", action="store_true",
                    help="Alias for --inspect (read-only).")
parser.add_argument("--json", action="store_true",
                    help="With --inspect: print machine-readable JSON to stdout.")
parser.add_argument("--output", default=None,
                    help="With --inspect: also write the JSON report to this file (explicit write).")
parser.add_argument("--ancestry", default=None,
                    help="With --inspect: only require the LD panel of this ancestry (EUR, AFR, EAS, SAS, AMR).")
parser.add_argument("--through-step", type=int, default=10,
                    help="With --inspect: readiness for Steps01..N (default 10; 11 adds colocalisation resources).")
parser.add_argument("--slurm-config", default=None,
                    help="SLURM configuration file (default config/slurm.yaml).")
parser.add_argument("--record-versions", action="store_true",
                    help="Write setup_manifests/resource_versions.tsv from the current state and exit.")
parser.add_argument("--generate-only", action="store_true",
                    help="Generate setup files only; do not install or submit anything.")
parser.add_argument("--install-only", action="store_true",
                    help="Install/repair software only; do not submit resource jobs.")
parser.add_argument("--resources-only", action="store_true",
                    help="Skip software installation and submit resource jobs only.")
parser.add_argument("--no-submit", action="store_true",
                    help="Backward-compatible alias for --install-only.")
parser.add_argument("--submit", action="store_true", help=argparse.SUPPRESS)

args = parser.parse_args()
if args.install_only and args.resources_only:
    parser.error("--install-only and --resources-only cannot be used together")
if args.slurm_config:
    os.environ[gwas2m_config.CONFIG_ENV_VAR] = str(Path(args.slurm_config).resolve())


# =============================================================================
# READ-ONLY INSPECTION
# =============================================================================

INSPECT_STAGES = [
    "setup", "setup_genome", "setup_gtex", "setup_1000g_download", "setup_1000g_all",
    "setup_1000g_samples", "setup_1000g_populations", "setup_vep", "setup_pangolin_db",
    "setup_opentargets", "setup_verify", "study", "audit", "download", "qc", "ld", "clumping",
    "finemapping", "vep", "qtl", "spliceai", "pangolin", "coloc_discovery", "coloc_planner",
    "coloc_extraction", "coloc", "aggregation",
]


def slurm_inspection() -> dict:
    commands = {name: ("AVAILABLE" if path else "MISSING")
                for name, path in gwas2m_config.scheduler_commands().items()}
    path = gwas2m_config.resolve_config_path(ROOT)
    try:
        config = gwas2m_config.load_slurm_config(ROOT)
        record = gwas2m_config.effective_config_record(config, INSPECT_STAGES)
        record.update({"status": "VALID", "errors": []})
    except (gwas2m_config.SlurmConfigError, OSError) as exc:
        record = {"status": "INVALID", "errors": str(exc).splitlines(), "config_file": str(path)}
    record["commands"] = commands
    record["environment"] = "SLURM MACHINE" if commands["sbatch"] == "AVAILABLE" else "NON-SLURM MACHINE"
    return record


def _short(path) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _human(size) -> str:
    if not size:
        return ""
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}P"


def print_inspection(report: dict) -> None:
    line = "=" * 100
    dash = "-" * 100

    def section(title):
        print()
        print(dash)
        print(title)
        print(dash)

    print(line)
    print("GWAS2m RESOURCE INSPECTION (read-only)")
    print(line)
    print(f"\nProject root   : {report['project_root']}")
    print(f"Reference build: {report['reference_build']}")
    print(f"Ancestry       : {report['ancestry'] or 'ALL (global inspection)'}")
    print(f"Readiness for  : Steps01-{report['through_step']:02d}")

    section("ENVIRONMENTS")
    print(f"{'ENVIRONMENT':<14}{'STATUS':<18}PATH")
    for name, r in report["environments"].items():
        print(f"{name:<14}{r['status']:<18}{_short(r['path'])}")

    section("SOFTWARE")
    print(f"{'RESOURCE':<32}{'STATUS':<18}{'VERSION':<44}PATH")
    for name, r in report["software"].items():
        print(f"{name:<32}{r['status']:<18}{str(r.get('version') or '')[:42]:<44}{_short(r.get('path'))}")

    section("PYTHON PACKAGES")
    pipeline_python = str(ROOT / "envs" / "pipeline" / "bin" / "python")
    probed = {r.get("interpreter") for r in report["python_packages"].values() if r["env"] == "pipeline"}
    if probed and probed != {pipeline_python}:
        print(f"NOTE: envs/pipeline is missing; 'pipeline' packages were probed in {', '.join(sorted(probed))}\n")
    print(f"{'PACKAGE':<26}{'STATUS':<18}{'VERSION':<16}{'REQUIRED':<10}USED BY")
    for name, r in report["python_packages"].items():
        required = "YES" if r["required"] else "no"
        print(f"{name:<26}{r['status']:<18}{str(r.get('version') or ''):<16}{required:<10}{r['used_by']}")

    section("R PACKAGES")
    print(f"{'PACKAGE':<26}{'STATUS':<18}VERSION")
    for name, r in report["r_packages"].items():
        print(f"{name:<26}{r['status']:<18}{r.get('version') or ''}")

    section("REFERENCE RESOURCES (shared; installed once, reused by every phenotype)")
    print(f"{'RESOURCE':<34}{'STATUS':<18}{'SIZE':<9}{'DETAIL':<92}PATH")
    for name, r in report["resources"].items():
        print(f"{name:<34}{r['status']:<18}{_human(r.get('size_bytes')):<9}"
              f"{str(r.get('detail') or '')[:90]:<92}{_short(r.get('path'))}")
    env = report["resource_paths_env"]
    print(f"\nresource_paths.env : {'present' if env['exists'] else 'MISSING (generated by setup)'}")

    slurm = report["slurm"]
    section("SLURM CONFIGURATION")
    print(f"Config file     : {_short(slurm.get('config_file'))}")
    if slurm["status"] == "INVALID":
        print("SLURM CONFIGURATION : INVALID")
        for error in slurm["errors"]:
            print(f"  {error}")
    else:
        s = slurm["slurm"]
        print(f"Scheduler       : {str(slurm['scheduler']).upper()}")
        print(f"Source          : {slurm['config_source']}")
        print(f"Cluster name    : {slurm['cluster_name']}")
        for key in ("partition", "account", "qos", "reservation", "constraint"):
            print(f"{key.capitalize():<16}: {gwas2m_config.describe(s.get(key))}")
        print(f"Array limit     : {s['array_limit']}")
        print(f"Concurrency cap : {s['max_parallel'] if s.get('max_parallel') else 'NONE'}")
        print(f"Max walltime    : {s.get('max_walltime') or 'NONE (not enforced by GWAS2m)'}")
        print(f"Default time    : {s['default_time']}")
        print(f"Default memory  : {s['default_memory']}")
        print(f"Default CPUs    : {s['default_cpus']}")
        print(f"Extra directives: {', '.join(slurm['extra_sbatch_directives']) or 'none'}")
    print("\nScheduler commands:")
    for name, status in slurm["commands"].items():
        print(f"  {name:<7}: {status}")
    print(f"Environment     : {slurm['environment']}")

    if slurm["status"] == "VALID":
        section("STAGE RESOURCES (effective SBATCH requests)")
        print(f"{'STAGE':<26}{'TIME':<12}{'MEMORY':<9}{'CPUS':<6}{'THROTTLE':<10}{'PARTITION':<18}FROM")
        for stage, r in slurm["stages"].items():
            print(f"{stage:<26}{r['time']:<12}{r['memory']:<9}{r['cpus']:<6}"
                  f"{(str(r['max_parallel']) if r['max_parallel'] else 'none'):<10}"
                  f"{gwas2m_config.describe(r.get('partition'), 'default'):<18}"
                  f"{'stages.' + r['config_stage'] if r['config_stage'] else 'slurm defaults'}")

    print()
    print(line)
    print("SUMMARY")
    print(line)
    for title, key in (("Software", "software"), ("Python packages", "python_packages"),
                       ("R packages", "r_packages"), ("Resources", "resources")):
        counts = {}
        for r in report[key].values():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"{title + ':':<17}" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    label = report["ancestry"] or "ALL-ANCESTRY"
    print(f"\nPIPELINE READY FOR {label} STEPS01-{report['through_step']:02d}: "
          f"{'YES' if report['pipeline_ready'] else 'NO'}")
    if report.get("ancestry_readiness"):
        for anc, ready in report["ancestry_readiness"].items():
            print(f"  PIPELINE READY FOR {anc} STEPS01-{report['through_step']:02d}: {'YES' if ready else 'NO'}")
    if report["missing_required"]:
        print("\nBLOCKING:")
        for item in report["missing_required"]:
            print(f"  - {item}")
    if report["warnings"]:
        print("\nWARNINGS:")
        for item in report["warnings"]:
            print(f"  - {item}")
    if report["setup_inputs_missing"]:
        print("\nSetup inputs missing (needed only to build shared resources):")
        for item in report["setup_inputs_missing"]:
            print(f"  - {item}")
    if report["missing_optional"]:
        print("\nOptional / not needed for this readiness check:")
        for item in report["missing_optional"]:
            print(f"  - {item}")
    if not report["pipeline_ready"] or report["setup_inputs_missing"]:
        print("\nSetup actions:")
        print("  python Step00_Check_Resources.py                   # full idempotent setup")
        print("  python Step00_Check_Resources.py --install-only    # software only")
        print("  python Step00_Check_Resources.py --resources-only  # shared resources only")
    if slurm["status"] == "INVALID":
        print("\nSLURM configuration is INVALID; fix it before submitting any job.")


if args.inspect or args.status:
    report = gwas2m_resources.inspect_pipeline(ROOT, args.ancestry, args.through_step)
    report["slurm"] = slurm_inspection()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_inspection(report)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        if not args.json:
            print(f"\nJSON report written to: {args.output}")
    sys.exit(0 if report["pipeline_ready"] and report["slurm"]["status"] == "VALID" else 1)


# =============================================================================
# SLURM CONFIGURATION (config/slurm.yaml; no cluster-specific defaults)
# =============================================================================

try:
    SLURM_CONFIG = gwas2m_config.load_slurm_config(ROOT)
except gwas2m_config.SlurmConfigError as exc:
    print(f"SLURM CONFIGURATION : INVALID\n{exc}")
    sys.exit(2)


def sbatch_header(stage: str, job_name: str, log_name: str, array_tasks: int | None = None) -> str:
    resources = gwas2m_config.get_stage_resources(stage, SLURM_CONFIG)
    array = gwas2m_config.array_spec(array_tasks, resources["max_parallel"]) if array_tasks else None
    pattern = "%A_%a" if array else "%j"
    return gwas2m_config.build_sbatch_directives(
        stage,
        job_name=job_name,
        output=f"{LOGS}/{log_name}.{pattern}.out",
        error=f"{LOGS}/{log_name}.{pattern}.err",
        array=array,
        config=SLURM_CONFIG,
    )


def write_resource_versions() -> Path:
    """setup_manifests/resource_versions.tsv: versions/releases actually present."""
    report = gwas2m_resources.inspect_pipeline(ROOT)
    rows = []
    for section, category in (("software", "SOFTWARE"), ("python_packages", "PYTHON_PACKAGE"),
                              ("r_packages", "R_PACKAGE")):
        for name, r in report[section].items():
            rows.append([category, name, r["status"], r.get("version") or "", "", "", r.get("path") or "", "", ""])
    for name, r in report["resources"].items():
        sig = gwas2m_status_signature(r.get("path"))
        rows.append(["RESOURCE", name, r["status"], r.get("version") or "", r.get("build") or "",
                     r.get("source") or "", r.get("path") or "", r.get("size_bytes") or "",
                     f"url={r.get('url')}; modified={sig}"])
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / "resource_versions.tsv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["CATEGORY", "NAME", "STATUS", "VERSION_OR_RELEASE", "GENOME_BUILD", "SOURCE",
                         "PATH", "SIZE_BYTES", "PROVENANCE"])
        writer.writerows(rows)
    return path


def gwas2m_status_signature(path) -> str:
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()
    except Exception:
        return ""


if args.record_versions:
    print(f"Recorded: {write_resource_versions()}")
    sys.exit(0)


# =============================================================================
# CREATE BASIC DIRECTORIES
# =============================================================================

for directory in [
    RESOURCES,
    EXTERNAL,
    ENVS,
    LOGS,
    MANIFESTS,
    FUMA / "input",
    FUMA / "output",
    FUMA / "config",
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# HELPERS
# =============================================================================

def banner(text: str) -> None:

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def write_file(
    filename: str | Path,
    content: str,
    executable: bool = False,
) -> Path:

    path = Path(filename)

    if not path.is_absolute():
        path = ROOT / path

    path.write_text(
        textwrap.dedent(content).lstrip(),
        encoding="utf-8",
    )

    if executable:
        path.chmod(0o755)

    return path


def run(
    command: list[str],
    *,
    label: str | None = None,
    check: bool = True,
) -> int:
    """Run a command with live output and a heartbeat during silent periods."""

    command = [str(x) for x in command]
    command_text = " ".join(shlex.quote(x) for x in command)

    print()
    if label:
        print(f">>> {label}")
    print("$ " + command_text, flush=True)
    print()

    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    while True:
        events = selector.select(timeout=10.0)
        if events:
            line = process.stdout.readline()
            if line:
                print(line, end="", flush=True)
        elif process.poll() is None:
            elapsed = int(time.time() - started)
            stage = label or Path(command[0]).name
            print(f"[RUNNING] {stage} | elapsed {elapsed}s", flush=True)

        if process.poll() is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
            break

    rc = int(process.returncode or 0)
    elapsed = time.time() - started
    print()
    print(f"[DONE] exit={rc} elapsed={elapsed/60:.1f} min")

    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, command)

    return rc


def find_package_manager() -> str | None:

    for name in [
        "mamba",
        "micromamba",
        "conda",
    ]:

        path = shutil.which(name)

        if path:
            return str(Path(path).resolve())

    return None


# =============================================================================
# PATHS
# =============================================================================

CORE_ENV = ENVS / "pipeline"

SPLICEAI_ENV = ENVS / "spliceai"

PANGOLIN_ENV = ENVS / "pangolin"

PANGOLIN_REPO = EXTERNAL / "Pangolin"


# =============================================================================
# ENVIRONMENT.YML
# =============================================================================

environment_yml = f"""
name: pipeline

channels:
  - conda-forge
  - bioconda

dependencies:

  # Python
  - python=3.11

  # Core Python scientific stack
  # (single source of truth: gwas2m_resources.PYTHON_PACKAGES)
  - pandas
  - numpy
  - scipy
  - pyarrow
  - duckdb
  - pysam
  - cyvcf2
  - scikit-learn
  - statsmodels
  - matplotlib
  - requests
  - urllib3
  - httpx
  - tenacity
  - tqdm
  - orjson
  - pyyaml
  - networkx
  - gprofiler-official

  # Genomics
  - plink2
  - bcftools
  - htslib
  - samtools
  - bedtools

  # Download / compression
  - aria2
  - wget
  - curl
  - pigz
  - zstd
  - parallel
  - rsync

  # VEP
  - ensembl-vep={VEP_RELEASE}

  # R
  - r-base
  - r-susier
  - r-coloc
  - r-data.table

  # General
  - git
  - pip

  # pip-only packages (also installed explicitly by Setup_00_Install_Software.sh)
  - pip:
      - gwaslab==4.2.3
      - polars[rtcompat]
"""

write_file(
    "environment.yml",
    environment_yml,
)


# =============================================================================
# SOFTWARE MANIFEST (from gwas2m_resources; not a lock file - versions are
# pinned only where SPECIFICATION says so)
# =============================================================================

with open(
    MANIFESTS / "software_manifest.tsv",
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.writer(
        handle,
        delimiter="\t",
    )

    writer.writerow([
        "SOFTWARE",
        "TYPE",
        "INSTALL_METHOD",
        "SPECIFICATION",
        "EXPECTED_ENV",
        "REQUIRED",
        "FIRST_STEP",
        "VERSION_PINNED",
    ])

    for env_id, rel, step, purpose, method in gwas2m_resources.ENVIRONMENTS:
        writer.writerow([env_id, "environment", method, rel, env_id, "YES", step, "NO"])
    for name, env, _, required, step, purpose, spec in gwas2m_resources.TOOLS:
        writer.writerow([name, "external-tool", "conda/pip/git", spec, env,
                         "YES" if required else "NO", step or "setup", "YES" if "=" in spec else "NO"])
    for module, dist, env, required, step, method, spec, used in gwas2m_resources.PYTHON_PACKAGES:
        writer.writerow([dist, "python-package", method, spec, env,
                         "YES" if required else "NO", step or "-", "YES" if "==" in spec else "NO"])
    for name, required, step, spec, used in gwas2m_resources.R_PACKAGES:
        writer.writerow([name, "r-package", "conda", spec, "pipeline",
                         "YES" if required else "NO", step, "NO"])


# =============================================================================
# CENTRAL RESOURCE MANIFEST (from gwas2m_resources.RESOURCES)
# =============================================================================

with open(
    MANIFESTS / "resources.tsv",
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.writer(
        handle,
        delimiter="\t",
    )

    writer.writerow([
        "RESOURCE_ID", "RESOURCE_TYPE", "SOURCE", "VERSION", "GENOME_BUILD", "URL",
        "EXPECTED_PATH", "REQUIRED", "FIRST_STEP", "ANCESTRY", "VALIDATION_METHOD", "SETUP",
    ])

    for item in gwas2m_resources.RESOURCES:
        writer.writerow([
            item["id"], item["type"], item["source"], item["version"], item["build"], item["url"],
            item["path"], "YES" if item["required"] else "NO", item["step"] or "setup",
            item["ancestry"] or "", item["validation"], item["setup"],
        ])


# =============================================================================
# RESOURCE MANIFEST
# =============================================================================

resource_rows = [

    [
        "GRCh38_FASTA",
        "GENCODE50",
        "GRCh38",
        (
            "https://ftp.ebi.ac.uk/pub/databases/gencode/"
            "Gencode_human/release_50/"
            "GRCh38.primary_assembly.genome.fa.gz"
        ),
        "resources/genome/GRCh38/GRCh38.primary_assembly.genome.fa.gz",
    ],

    [
        "GENCODE_GTF",
        "GENCODE50",
        "GRCh38",
        (
            "https://ftp.ebi.ac.uk/pub/databases/gencode/"
            "Gencode_human/release_50/"
            "gencode.v50.primary_assembly.annotation.gtf.gz"
        ),
        "resources/gencode/release50/"
        "gencode.v50.primary_assembly.annotation.gtf.gz",
    ],

    [
        "VEP_CACHE",
        "116",
        "GRCh38",
        (
            "https://ftp.ensembl.org/pub/release-116/"
            "variation/indexed_vep_cache/"
            "homo_sapiens_vep_116_GRCh38.tar.gz"
        ),
        "resources/vep/116/homo_sapiens_vep_116_GRCh38.tar.gz",
    ],

    [
        "GTEX_EQTL",
        "V11",
        "GRCh38",
        (
            "https://storage.googleapis.com/"
            "adult-gtex/bulk-qtl/v11/"
            "single-tissue-cis-qtl/"
            "GTEx_Analysis_v11_eQTL.tar"
        ),
        "resources/gtex/v11/qtl/GTEx_Analysis_v11_eQTL.tar",
    ],

    [
        "GTEX_SQTL",
        "V11",
        "GRCh38",
        (
            "https://storage.googleapis.com/"
            "adult-gtex/bulk-qtl/v11/"
            "single-tissue-cis-qtl/"
            "GTEx_Analysis_v11_sQTL.tar"
        ),
        "resources/gtex/v11/qtl/GTEx_Analysis_v11_sQTL.tar",
    ],

    [
        "GTEX_EQTL_SUSIE",
        "V11",
        "GRCh38",
        (
            "https://storage.googleapis.com/"
            "adult-gtex/bulk-qtl/v11/"
            "susie-qtl/"
            "GTEx_Analysis_v11_eQTL_SuSiE.tar"
        ),
        "resources/gtex/v11/susie/GTEx_Analysis_v11_eQTL_SuSiE.tar",
    ],

    [
        "GTEX_SQTL_SUSIE",
        "V11",
        "GRCh38",
        (
            "https://storage.googleapis.com/"
            "adult-gtex/bulk-qtl/v11/"
            "susie-qtl/"
            "GTEx_Analysis_v11_sQTL_SuSiE.tar"
        ),
        "resources/gtex/v11/susie/GTEx_Analysis_v11_sQTL_SuSiE.tar",
    ],

    [
        "GTEX_VARIANT_LOOKUP",
        "V11",
        "GRCh38",
        (
            "https://storage.googleapis.com/"
            "adult-gtex/references/v11/"
            "reference-tables/"
            "GTEx_Analysis_2021-02-11_v11_"
            "WholeGenomeSeq_953Indiv.lookup_table.txt.gz"
        ),
        "resources/gtex/v11/reference/"
        "GTEx_Analysis_2021-02-11_v11_"
        "WholeGenomeSeq_953Indiv.lookup_table.txt.gz",
    ],

    [
        "GTEX_GENCODE47",
        "V11",
        "GRCh38",
        (
            "https://storage.googleapis.com/"
            "adult-gtex/references/v11/"
            "reference-tables/gencode.v47.genes.gtf"
        ),
        "resources/gtex/v11/reference/gencode.v47.genes.gtf",
    ],

]


# =============================================================================
# GTEx ARRAY MANIFEST
# =============================================================================

gtex_manifest = [

    [
        1,
        "eQTL",
        resource_rows[3][3],
        "qtl",
        "tar",
    ],

    [
        2,
        "sQTL",
        resource_rows[4][3],
        "qtl",
        "tar",
    ],

    [
        3,
        "eQTL_SuSiE",
        resource_rows[5][3],
        "susie",
        "tar",
    ],

    [
        4,
        "sQTL_SuSiE",
        resource_rows[6][3],
        "susie",
        "tar",
    ],

    [
        5,
        "variant_lookup",
        resource_rows[7][3],
        "reference",
        "file",
    ],

    [
        6,
        "gencode47",
        resource_rows[8][3],
        "reference",
        "file",
    ],
]


with open(
    MANIFESTS / "gtex_v11.tsv",
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.writer(
        handle,
        delimiter="\t",
    )

    writer.writerow([
        "TASK_ID",
        "NAME",
        "URL",
        "SUBDIR",
        "TYPE",
    ])

    writer.writerows(
        gtex_manifest
    )


# =============================================================================
# RESOURCE PATHS
# =============================================================================

# Same keys as before (PIPELINE_ROOT, GRCH38_FASTA, GENCODE50_GTF, GTEX_V11,
# OPEN_TARGETS_26_09, VEP_CACHE_DIR, LD_REFERENCE_<ANC>, PANGOLIN_DB) plus the
# VEP / Pangolin / SpliceAI executables, all taken from gwas2m_resources.
write_file(
    "resource_paths.env",
    gwas2m_resources.resource_paths_env_text(ROOT),
)


# =============================================================================
# FUMA REPRODUCIBILITY TEMPLATE
# =============================================================================

fuma_template = {

    "genome_build":
        "GRCh38",

    "reference_population":
        "RECORD_WHAT_YOU_USED",

    "submission_date":
        None,

    "fuma_version":
        None,

    "snp2gene": {

        "positional_mapping":
            None,

        "eqtl_mapping":
            None,

        "chromatin_interaction_mapping":
            None,
    },

    "gene2func": {

        "magma_gene_analysis":
            None,

        "magma_gene_set_analysis":
            None,

        "magma_gene_property_analysis":
            None,
    },

    "notes":
        "Populate this file when FUMA is used and retain all downloaded FUMA outputs.",
}


write_file(
    FUMA / "config" / "fuma_parameters.template.json",
    json.dumps(
        fuma_template,
        indent=2,
    )
    + "\n",
)


write_file(
    FUMA / "README.md",
    """
    # FUMA reproducibility

    FUMA is treated as an external analysis service rather than a local
    Conda package.

    Store:

    input/   - exact uploaded GWAS/variant files
    config/  - exact settings used for SNP2GENE / GENE2FUNC
    output/  - all downloaded FUMA outputs

    Record the submission date, FUMA release/version when shown, genome build,
    reference population, mapping options, MAGMA options, and input checksum.
    """,
)


# =============================================================================
# COMMON BASH DOWNLOAD FUNCTION
# =============================================================================

download_function = r'''
download_resource() {

    local URL="$1"
    local OUT="$2"

    mkdir -p "$(dirname "$OUT")"

    if command -v flock >/dev/null 2>&1; then
        exec 9>"${OUT}.lock"
        flock 9
    fi

    if [[ -s "$OUT" && -f "${OUT}.complete" ]]; then
        echo "[CACHE] $OUT"
        return 0
    fi

    echo
    echo "=============================================================="
    echo "DOWNLOAD"
    echo "=============================================================="
    echo "URL : $URL"
    echo "OUT : $OUT"
    echo

    set +e
    "$ARIA2" \
        --continue=true \
        --max-connection-per-server=4 \
        --split=4 \
        --min-split-size=16M \
        --file-allocation=none \
        --auto-file-renaming=false \
        --allow-overwrite=true \
        --max-tries=10 \
        --retry-wait=5 \
        --connect-timeout=30 \
        --timeout=60 \
        --console-log-level=warn \
        --summary-interval=5 \
        --dir "$(dirname "$OUT")" \
        --out "$(basename "$OUT")" \
        "$URL"
    RC=$?
    set -e

    if [[ $RC -ne 0 ]]; then
        echo
        echo "aria2c failed (exit $RC). Trying curl resume fallback..."
        curl \
            --fail \
            --location \
            --retry 8 \
            --retry-delay 5 \
            --retry-all-errors \
            --continue-at - \
            --output "$OUT" \
            "$URL"
    fi

    test -s "$OUT"
    touch "${OUT}.complete"
    echo "[OK] $OUT"
}
'''


# =============================================================================
# 00 INSTALL SOFTWARE
# =============================================================================

install_script = f"""
#!/bin/bash
set -euo pipefail

ROOT="{ROOT}"
ENV="$ROOT/envs/pipeline"
SPLICE="$ROOT/envs/spliceai"
ENVFILE="$ROOT/environment.yml"

PM="$(command -v mamba || command -v micromamba || command -v conda || true)"

if [[ -z "$PM" ]]; then
    echo "ERROR: mamba, micromamba or conda is required."
    exit 1
fi

export CONDA_CHANNEL_PRIORITY=strict

echo
echo "=============================================================="
echo "[1/4] MAIN PIPELINE ENVIRONMENT"
echo "=============================================================="
echo "Package manager : $PM"
echo "Environment     : $ENV"
echo "Specification   : $ENVFILE"
echo

if [[ -d "$ENV/conda-meta" ]]; then
    echo "[CACHE] Environment exists; updating/verifying it."
    echo "COMMAND: $PM env update -y -p $ENV -f $ENVFILE --prune"
    "$PM" env update -y -p "$ENV" -f "$ENVFILE" --prune
else
    echo "COMMAND: $PM env create -y -p $ENV -f $ENVFILE"
    "$PM" env create -y -p "$ENV" -f "$ENVFILE"
fi

echo
echo "=============================================================="
echo "[2/4] GWASLAB + CORE SOFTWARE CHECK"
echo "=============================================================="

echo "COMMAND: install GWASLab + CPU-compatible Polars runtime"
"$ENV/bin/python" -m pip install --upgrade \
    "gwaslab==4.2.3" \
    "polars[rtcompat]"

"$ENV/bin/python" - <<'COREPY'
mods = [
    "gwaslab", "pandas", "numpy", "scipy", "polars", "pyarrow",
    "duckdb", "pysam", "cyvcf2", "sklearn", "statsmodels",
    "matplotlib", "requests", "httpx", "tenacity", "tqdm",
]
for name in mods:
    __import__(name)
    print(f"[OK] Python import: {{name}}")
COREPY

for TOOL in plink2 bcftools samtools tabix bgzip bedtools aria2c pigz rsync vep Rscript; do
    test -x "$ENV/bin/$TOOL"
    echo "[OK] $TOOL -> $ENV/bin/$TOOL"
done

"$ENV/bin/Rscript" -e '
for (p in c("susieR", "coloc", "data.table")) {{
  if (!requireNamespace(p, quietly=TRUE)) stop(paste("Missing R package:", p))
  cat("[OK] R package:", p, "\\n")
}}
'

echo
echo "=============================================================="
echo "[3/4] SPLICEAI ISOLATED ENVIRONMENT"
echo "=============================================================="

if [[ ! -d "$SPLICE/conda-meta" ]]; then
    echo "COMMAND: $PM create -y -p $SPLICE -c conda-forge python=3.10 pip setuptools<81 numpy<2"
    "$PM" create -y -p "$SPLICE" -c conda-forge python=3.10 pip "setuptools<81" "numpy<2"
else
    echo "[CACHE] SpliceAI environment exists: $SPLICE"
fi

echo "COMMAND: install pinned SpliceAI runtime"
"$SPLICE/bin/python" -m pip install --upgrade \
    "setuptools<81" \
    "numpy<2" \
    "tensorflow-cpu==2.15.1" \
    "keras==2.15.0" \
    "spliceai==1.3.1"

"$SPLICE/bin/python" - <<'SPLICEPY'
import pkg_resources
import tensorflow as tf
import keras
import spliceai
from spliceai.utils import Annotator
print("[OK] pkg_resources")
print("[OK] TensorFlow", tf.__version__)
print("[OK] Keras", keras.__version__)
print("[OK] SpliceAI", spliceai.__file__)
print("[OK] SpliceAI Annotator import")
SPLICEPY

echo
echo "=============================================================="
echo "[4/4] SOFTWARE INSTALLATION COMPLETE"
echo "=============================================================="
"$ENV/bin/python" --version
"$ENV/bin/plink2" --version | head -1
"$ENV/bin/bcftools" --version | head -1
"$ENV/bin/samtools" --version | head -1
"""

write_file(
    "Setup_00_Install_Software.sh",
    install_script,
    executable=True,
)


# =============================================================================
# 01 PANGOLIN
# =============================================================================

pangolin_script = f"""
#!/bin/bash
set -euo pipefail

ROOT="{ROOT}"
REPO="$ROOT/external_tools/Pangolin"
ENV="$ROOT/envs/pangolin"
LOCK="$ROOT/setup_manifests/Pangolin.commit.txt"

PM="$(command -v mamba || command -v micromamba || command -v conda || true)"
if [[ -z "$PM" ]]; then
    echo "ERROR: mamba/micromamba/conda not found."
    exit 1
fi

mkdir -p "$ROOT/external_tools" "$ROOT/setup_manifests"

echo
echo "=============================================================="
echo "PANGOLIN SETUP"
echo "=============================================================="

if [[ ! -d "$REPO/.git" ]]; then
    echo "COMMAND: git clone https://github.com/tkzeng/Pangolin.git $REPO"
    git clone https://github.com/tkzeng/Pangolin.git "$REPO"
fi

cd "$REPO"

if [[ -s "$LOCK" ]]; then
    COMMIT="$(tr -d '[:space:]' < "$LOCK")"
    echo "Pinned Pangolin commit: $COMMIT"
    git fetch --all --tags --prune
    git checkout --detach "$COMMIT"
else
    COMMIT="$(git rev-parse HEAD)"
    printf '%s\\n' "$COMMIT" > "$LOCK"
    echo "Locked Pangolin commit: $COMMIT"
fi

if [[ ! -d "$ENV/conda-meta" ]]; then
    echo "Creating isolated Pangolin environment..."
    "$PM" create -y -p "$ENV" -c conda-forge \
        python=3.10 pip "numpy<2" pytorch gffutils biopython pandas pyfastx
else
    echo "[CACHE] Pangolin environment exists: $ENV"
fi

"$ENV/bin/python" -m pip install --upgrade PyVCF3
"$ENV/bin/python" -m pip install -e "$REPO" --no-deps

"$ENV/bin/python" - <<'PANGPY'
import pangolin
import gffutils
import Bio
import pandas
import pyfastx
print("[OK] Pangolin import:", pangolin.__file__)
PANGPY

"$ENV/bin/pangolin" --help >/dev/null

echo "[OK] Pangolin executable: $ENV/bin/pangolin"
echo "[OK] Pangolin commit: $(git rev-parse HEAD)"
"""

write_file(
    "Setup_01_Setup_Pangolin.sh",
    pangolin_script,
    executable=True,
)


# =============================================================================
# 02 GENOME + GENCODE
# =============================================================================

genome_script = f"""
#!/bin/bash
{sbatch_header("setup_genome", "res_genome", "02_genome")}

set -euo pipefail

ROOT="{ROOT}"
RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"
PIGZ="$ROOT/envs/pipeline/bin/pigz"
SAMTOOLS="$ROOT/envs/pipeline/bin/samtools"


{download_function}


GENOME="$RES/genome/GRCh38"

GENCODE="$RES/gencode/release50"

mkdir -p "$GENOME" "$GENCODE"


FA_GZ="$GENOME/GRCh38.primary_assembly.genome.fa.gz"

FA="$GENOME/GRCh38.primary_assembly.genome.fa"

GTF="$GENCODE/gencode.v50.primary_assembly.annotation.gtf.gz"


download_resource \
"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/GRCh38.primary_assembly.genome.fa.gz" \
"$FA_GZ"


download_resource \
"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.primary_assembly.annotation.gtf.gz" \
"$GTF"


if [[ ! -s "$FA" ]]; then

    echo
    echo "Decompressing GRCh38..."

    "$PIGZ" \
        -dc \
        "$FA_GZ" \
        > "$FA.part"

    mv "$FA.part" "$FA"

fi


if [[ ! -s "$FA.fai" ]]; then

    "$SAMTOOLS" \
        faidx \
        "$FA"

fi


echo
echo "Genome setup complete."
"""

write_file(
    "Setup_02_Download_Genome_GENCODE.sh",
    genome_script,
    executable=True,
)


# =============================================================================
# 03 GTEx
# =============================================================================

gtex_script = f"""
#!/bin/bash
{sbatch_header("setup_gtex", "res_gtex", "03_gtex", 6)}

set -euo pipefail

ROOT="{ROOT}"
RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"

MANIFEST="$ROOT/setup_manifests/gtex_v11.tsv"

TASK_ID="${{SLURM_ARRAY_TASK_ID:-${{1:-}}}}"


if [[ -z "$TASK_ID" ]]; then

    echo "Usage locally:"
    echo "  bash $0 TASK_ID"

    exit 2

fi


{download_function}


LINE="$(
    awk \
        -F '\\t' \
        -v id="$TASK_ID" \
        'NR>1 && $1==id {{print; exit}}' \
        "$MANIFEST"
)"


IFS=$'\\t' read \
    -r \
    ID \
    NAME \
    URL \
    SUBDIR \
    TYPE \
    <<< "$LINE"


OUTDIR="$RES/gtex/v11/$SUBDIR"

mkdir -p "$OUTDIR"

OUT="$OUTDIR/$(basename "$URL")"


download_resource \
    "$URL" \
    "$OUT"


if [[ "$TYPE" == "tar" ]]; then

    DEST="$OUTDIR/$NAME"

    mkdir -p "$DEST"

    if [[ ! -f "$DEST/.complete" ]]; then

        echo
        echo "Extracting $NAME..."

        tar \
            -xf "$OUT" \
            -C "$DEST"

        touch \
            "$DEST/.complete"

    fi

fi


echo
echo "GTEx task complete:"
echo "$NAME"
"""

write_file(
    "Setup_03_Download_GTEx_V11.sh",
    gtex_script,
    executable=True,
)


# =============================================================================
# 04 1000 GENOMES RAW
# =============================================================================

g1000_script = f"""
#!/bin/bash
{sbatch_header("setup_1000g_download", "res_1000g", "04_1000g", 23)}

set -euo pipefail

ROOT="{ROOT}"

RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"

RAW="$RES/1000G/raw"

META="$RES/1000G/metadata"


mkdir -p \
    "$RAW" \
    "$META"


TASK_ID="${{SLURM_ARRAY_TASK_ID:-${{1:-}}}}"


if [[ -z "$TASK_ID" ]]; then

    echo "Usage:"
    echo "  bash $0 TASK_ID"

    exit 2

fi


{download_function}


BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL"


if [[ "$TASK_ID" -le 22 ]]; then

    CHR="$TASK_ID"

    FN="ALL.chr${{CHR}}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"


    download_resource \
        "$BASE/$FN" \
        "$RAW/$FN"


    download_resource \
        "$BASE/$FN.tbi" \
        "$RAW/$FN.tbi"


else

    download_resource \
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel" \
        "$META/integrated_call_samples_v3.20130502.ALL.panel"


    download_resource \
        "$BASE/20190312_biallelic_SNV_and_INDEL_README.txt" \
        "$META/20190312_biallelic_SNV_and_INDEL_README.txt"


    download_resource \
        "$BASE/20190312_biallelic_SNV_and_INDEL_MANIFEST.txt" \
        "$META/20190312_biallelic_SNV_and_INDEL_MANIFEST.txt"

fi
"""

write_file(
    "Setup_04_Download_1000G.sh",
    g1000_script,
    executable=True,
)


# =============================================================================
# 05 CONVERT RAW 1000G ONCE TO ALL-SAMPLE PGEN
# =============================================================================

convert_script = f"""
#!/bin/bash
{sbatch_header("setup_1000g_all", "res_1000g_all", "05_1000g_all", 22)}

set -euo pipefail

ROOT="{ROOT}"

RES="$ROOT/resources"

PLINK="$ROOT/envs/pipeline/bin/plink2"

RAW="$RES/1000G/raw"

ALL="$RES/1000G/ALL"


mkdir -p "$ALL"


CHR="${{SLURM_ARRAY_TASK_ID:-${{1:-}}}}"


if [[ -z "$CHR" ]]; then

    echo "Usage:"
    echo "  bash $0 CHROMOSOME"

    exit 2

fi


FN="ALL.chr${{CHR}}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"

VCF="$RAW/$FN"

PREFIX="$ALL/chr${{CHR}}"


if [[ \
    -s "$PREFIX.pgen" \
    && -s "$PREFIX.pvar" \
    && -s "$PREFIX.psam" \
    && -f "$PREFIX.complete" \
]]; then

    echo "[CACHE] chr${{CHR}} ALL PGEN"

    exit 0

fi


test -s "$VCF"


rm -f \
    "$PREFIX.pgen" \
    "$PREFIX.pvar" \
    "$PREFIX.psam"


"$PLINK" \
    --vcf "$VCF" \
    --set-all-var-ids '@:#:$r:$a' \
    --new-id-max-allele-len 1000 \
    --rm-dup force-first \
    --max-alleles 2 \
    --make-pgen \
    --threads "${{SLURM_CPUS_PER_TASK:-4}}" \
    --out "$PREFIX"


test -s "$PREFIX.pgen"
test -s "$PREFIX.pvar"
test -s "$PREFIX.psam"


touch "$PREFIX.complete"


echo
echo "ALL-sample PGEN complete:"
echo "chr${{CHR}}"
"""

write_file(
    "Setup_05_Prepare_1000G_ALL.sh",
    convert_script,
    executable=True,
)


# =============================================================================
# 06 SAMPLE LISTS
# =============================================================================

samples_script = f"""
#!/bin/bash
{sbatch_header("setup_1000g_samples", "res_1000g_samples", "06_samples")}

set -euo pipefail

ROOT="{ROOT}"

RES="$ROOT/resources"

PANEL="$RES/1000G/metadata/integrated_call_samples_v3.20130502.ALL.panel"


test -s "$PANEL"


for ANC in EUR AFR EAS SAS AMR; do

    DIR="$RES/1000G/$ANC"

    mkdir -p "$DIR"


    OUT="$DIR/samples.txt"


    awk \
        -F '\\t' \
        -v anc="$ANC" \
        'BEGIN {{print "#IID"}} NR>1 && $3==anc {{print $1}}' \
        "$PANEL" \
        > "$OUT.tmp"


    test "$(
        wc -l < "$OUT.tmp"
    )" -gt 1


    mv \
        "$OUT.tmp" \
        "$OUT"


    echo "$ANC samples: $(( $(wc -l < "$OUT") - 1 ))"

done


# ----------------------------------------------------------------------
# Compatibility links for current pipeline
# ----------------------------------------------------------------------

mkdir -p \
    "$ROOT/04_ld_reference"


ln -sfn \
    "$RES/1000G/raw" \
    "$ROOT/04_ld_reference/1000G_GRCh38_RAW"


for ANC in EUR AFR EAS SAS AMR; do

    ln -sfn \
        "$RES/1000G/$ANC" \
        "$ROOT/04_ld_reference/1000G_${{ANC}}_GRCh38"

done
"""

write_file(
    "Setup_06_Prepare_1000G_Sample_Lists.sh",
    samples_script,
    executable=True,
)


# =============================================================================
# 07 POPULATION PGEN
# =============================================================================

population_script = f"""
#!/bin/bash
{sbatch_header("setup_1000g_populations", "res_1000g_pop", "07_1000g_pop", 110)}

set -euo pipefail

ROOT="{ROOT}"

RES="$ROOT/resources"

PLINK="$ROOT/envs/pipeline/bin/plink2"


TASK="${{SLURM_ARRAY_TASK_ID:-${{1:-}}}}"


if [[ -z "$TASK" ]]; then

    echo "Usage:"
    echo "  bash $0 TASK"

    exit 2

fi


INDEX=$(( (TASK - 1) / 22 ))

CHR=$(( (TASK - 1) % 22 + 1 ))


ANCES=(EUR AFR EAS SAS AMR)

ANC="${{ANCES[$INDEX]}}"


ALL="$RES/1000G/ALL/chr${{CHR}}"

OUTDIR="$RES/1000G/$ANC"

KEEP="$OUTDIR/samples.txt"

PREFIX="$OUTDIR/chr${{CHR}}_${{ANC}}_GRCh38"


mkdir -p "$OUTDIR"


if [[ \
    -s "$PREFIX.pgen" \
    && -s "$PREFIX.pvar" \
    && -s "$PREFIX.psam" \
    && -f "$PREFIX.complete" \
]]; then

    echo "[CACHE] $ANC chr${{CHR}}"

    exit 0

fi


test -s "$ALL.pgen"
test -s "$ALL.pvar"
test -s "$ALL.psam"
test -s "$KEEP"


"$PLINK" \
    --pfile "$ALL" \
    --keep "$KEEP" \
    --make-pgen \
    --threads "${{SLURM_CPUS_PER_TASK:-4}}" \
    --out "$PREFIX"


test -s "$PREFIX.pgen"
test -s "$PREFIX.pvar"
test -s "$PREFIX.psam"


touch "$PREFIX.complete"


echo
echo "Population reference complete:"
echo "Ancestry   : $ANC"
echo "Chromosome : $CHR"
"""

write_file(
    "Setup_07_Prepare_1000G_Populations.sh",
    population_script,
    executable=True,
)


# =============================================================================
# 08 VEP CACHE
# =============================================================================

vep_script = f"""
#!/bin/bash
{sbatch_header("setup_vep", "res_vep", "08_vep")}

set -euo pipefail

ROOT="{ROOT}"

RES="$ROOT/resources"

ARIA2="$ROOT/envs/pipeline/bin/aria2c"


{download_function}


DIR="$RES/vep/116"

CACHE="$RES/vep/cache"


mkdir -p \
    "$DIR" \
    "$CACHE"


TAR="$DIR/homo_sapiens_vep_116_GRCh38.tar.gz"


URL="https://ftp.ensembl.org/pub/release-116/variation/indexed_vep_cache/homo_sapiens_vep_116_GRCh38.tar.gz"


download_resource \
    "$URL" \
    "$TAR"


MARKER="$CACHE/.vep116_GRCh38.complete"


if [[ ! -f "$MARKER" ]]; then

    echo
    echo "Extracting VEP cache..."

    tar \
        -xzf "$TAR" \
        -C "$CACHE"


    touch "$MARKER"

fi


echo
echo "VEP cache:"
echo "$CACHE"
"""

write_file(
    "Setup_08_Download_VEP_Cache.sh",
    vep_script,
    executable=True,
)


# =============================================================================
# 09 PANGOLIN DATABASE
# =============================================================================

pangolin_db_script = f"""
#!/bin/bash
{sbatch_header("setup_pangolin_db", "res_pangolin_db", "09_pangolin_db")}

set -euo pipefail

ROOT="{ROOT}"

RES="$ROOT/resources"

REPO="$ROOT/external_tools/Pangolin"

PY="$ROOT/envs/pangolin/bin/python"


OUT="$RES/pangolin"

mkdir -p "$OUT"


SOURCE="$RES/gencode/release50/gencode.v50.primary_assembly.annotation.gtf.gz"

LINK="$OUT/gencode.v50.primary_assembly.annotation.gtf.gz"

DB="$OUT/gencode.v50.primary_assembly.annotation.db"


test -s "$SOURCE"

ln -sfn \
    "$SOURCE" \
    "$LINK"


if [[ -s "$DB" ]]; then

    echo "[CACHE] Pangolin annotation database"

    exit 0

fi


cd "$OUT"


"$PY" \
    "$REPO/scripts/create_db.py" \
    "$(basename "$LINK")"


test -s "$DB"


echo
echo "Pangolin database complete:"
echo "$DB"
"""

write_file(
    "Setup_09_Build_Pangolin_DB.sh",
    pangolin_db_script,
    executable=True,
)


# =============================================================================
# VERIFY SETUP PYTHON
# =============================================================================

verify_python = f'''#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

RES = ROOT / "resources"

ENV = ROOT / "envs" / "pipeline"

SPLICE = ROOT / "envs" / "spliceai"

PANG = ROOT / "envs" / "pangolin"

passed = 0
failed = 0


def check(name, condition, detail=""):

    global passed
    global failed

    if condition:

        passed += 1

        print(
            f"[OK]      {{name:35}} {{detail}}"
        )

    else:

        failed += 1

        print(
            f"[MISSING] {{name:35}} {{detail}}"
        )


print("=" * 80)
print("PIPELINE SETUP VERIFICATION")
print("=" * 80)


# ----------------------------------------------------------------------
# Core environment
# ----------------------------------------------------------------------

check(
    "Main environment",
    (ENV / "conda-meta").is_dir(),
    str(ENV),
)


for exe in [
    "plink2",
    "bcftools",
    "samtools",
    "tabix",
    "bgzip",
    "aria2c",
    "vep",
    "Rscript",
]:

    path = ENV / "bin" / exe

    check(
        exe,
        path.exists(),
        str(path),
    )


# ----------------------------------------------------------------------
# Python imports
# ----------------------------------------------------------------------

python = ENV / "bin" / "python"


for module in [
    "gwaslab",
    "pandas",
    "numpy",
    "scipy",
    "polars",
    "pyarrow",
    "duckdb",
    "pysam",
    "cyvcf2",
    "requests",
    "httpx",
    "tenacity",
    "tqdm",
]:

    good = False

    if python.exists():

        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    f"import {{module}}"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        good = result.returncode == 0

    check(
        module,
        good,
    )


# ----------------------------------------------------------------------
# R
# ----------------------------------------------------------------------

rscript = ENV / "bin" / "Rscript"


for package in [
    "susieR",
    "coloc",
    "data.table",
]:

    good = False

    if rscript.exists():

        result = subprocess.run(
            [
                str(rscript),
                "-e",
                (
                    f'quit(status=ifelse('
                    f'requireNamespace("{{package}}", quietly=TRUE),0,1))'
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        good = result.returncode == 0

    check(
        f"R: {{package}}",
        good,
    )


# ----------------------------------------------------------------------
# SpliceAI / Pangolin
# ----------------------------------------------------------------------

check(
    "SpliceAI environment",
    (SPLICE / "conda-meta").is_dir(),
)

splice_ok = False
if (SPLICE / "bin" / "python").exists():
    splice_ok = subprocess.run(
        [str(SPLICE / "bin" / "python"), "-c",
         "import pkg_resources,tensorflow,keras,spliceai; from spliceai.utils import Annotator"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
check("SpliceAI runtime", splice_ok)
check("SpliceAI executable", (SPLICE / "bin" / "spliceai").exists())

check(
    "Pangolin environment",
    (PANG / "conda-meta").is_dir(),
)

pang_ok = False
if (PANG / "bin" / "python").exists():
    pang_ok = subprocess.run(
        [str(PANG / "bin" / "python"), "-c", "import pangolin"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
check("Pangolin runtime", pang_ok)
check("Pangolin executable", (PANG / "bin" / "pangolin").exists())


# ----------------------------------------------------------------------
# Genome
# ----------------------------------------------------------------------

FASTA = (
    RES
    / "genome"
    / "GRCh38"
    / "GRCh38.primary_assembly.genome.fa"
)

check(
    "GRCh38 FASTA",
    FASTA.exists()
    and FASTA.stat().st_size > 0,
    str(FASTA),
)

check(
    "GRCh38 FASTA index",
    Path(str(FASTA) + ".fai").exists(),
)


GTF = (
    RES
    / "gencode"
    / "release50"
    / "gencode.v50.primary_assembly.annotation.gtf.gz"
)

check(
    "GENCODE 50 GTF",
    GTF.exists()
    and GTF.stat().st_size > 0,
    str(GTF),
)


# ----------------------------------------------------------------------
# 1000 Genomes raw
# ----------------------------------------------------------------------

raw = RES / "1000G" / "raw"


n_vcf = 0
n_tbi = 0


for chromosome in range(1, 23):

    name = (
        f"ALL.chr{{chromosome}}."
        "shapeit2_integrated_snvindels_v2a_27022019."
        "GRCh38.phased.vcf.gz"
    )

    vcf = raw / name

    tbi = raw / (name + ".tbi")

    if vcf.exists() and vcf.stat().st_size > 0:

        n_vcf += 1

    if tbi.exists() and tbi.stat().st_size > 0:

        n_tbi += 1


check(
    "1000G raw VCF",
    n_vcf == 22,
    f"{{n_vcf}}/22",
)

check(
    "1000G raw TBI",
    n_tbi == 22,
    f"{{n_tbi}}/22",
)


# ----------------------------------------------------------------------
# ALL PGEN
# ----------------------------------------------------------------------

all_complete = 0


for chromosome in range(1, 23):

    prefix = (
        RES
        / "1000G"
        / "ALL"
        / f"chr{{chromosome}}"
    )

    required = [
        Path(str(prefix) + ".pgen"),
        Path(str(prefix) + ".pvar"),
        Path(str(prefix) + ".psam"),
    ]

    if all(
        x.exists()
        and x.stat().st_size > 0
        for x in required
    ):

        all_complete += 1


check(
    "1000G ALL PGEN",
    all_complete == 22,
    f"{{all_complete}}/22",
)


# ----------------------------------------------------------------------
# Population references
# ----------------------------------------------------------------------

for ancestry in [
    "EUR",
    "AFR",
    "EAS",
    "SAS",
    "AMR",
]:

    count = 0

    for chromosome in range(1, 23):

        prefix = (
            RES
            / "1000G"
            / ancestry
            / f"chr{{chromosome}}_{{ancestry}}_GRCh38"
        )

        required = [
            Path(str(prefix) + ".pgen"),
            Path(str(prefix) + ".pvar"),
            Path(str(prefix) + ".psam"),
        ]

        if all(
            x.exists()
            and x.stat().st_size > 0
            for x in required
        ):

            count += 1

    check(
        f"1000G {{ancestry}}",
        count == 22,
        f"{{count}}/22",
    )


# ----------------------------------------------------------------------
# GTEx
# ----------------------------------------------------------------------

GTEX = RES / "gtex" / "v11"


files = [

    GTEX / "qtl" / "GTEx_Analysis_v11_eQTL.tar",

    GTEX / "qtl" / "GTEx_Analysis_v11_sQTL.tar",

    GTEX / "susie" / "GTEx_Analysis_v11_eQTL_SuSiE.tar",

    GTEX / "susie" / "GTEx_Analysis_v11_sQTL_SuSiE.tar",

    (
        GTEX
        / "reference"
        / "GTEx_Analysis_2021-02-11_v11_"
          "WholeGenomeSeq_953Indiv.lookup_table.txt.gz"
    ),

    GTEX
    / "reference"
    / "gencode.v47.genes.gtf",
]


for file in files:

    check(
        "GTEx: " + file.name,
        file.exists()
        and file.stat().st_size > 0,
    )

for extracted in [
    GTEX / "qtl" / "eQTL" / ".complete",
    GTEX / "qtl" / "sQTL" / ".complete",
    GTEX / "susie" / "eQTL_SuSiE" / ".complete",
    GTEX / "susie" / "sQTL_SuSiE" / ".complete",
]:
    check("GTEx extracted: " + extracted.parent.name, extracted.exists())


# ----------------------------------------------------------------------
# VEP
# ----------------------------------------------------------------------

VEP = (
    RES
    / "vep"
    / "cache"
    / "homo_sapiens"
    / "116_GRCh38"
)

check(
    "VEP 116 GRCh38 cache",
    VEP.exists(),
    str(VEP),
)


# ----------------------------------------------------------------------
# Pangolin DB
# ----------------------------------------------------------------------

DB = (
    RES
    / "pangolin"
    / "gencode.v50.primary_assembly.annotation.db"
)

check(
    "Pangolin annotation DB",
    DB.exists()
    and DB.stat().st_size > 0,
    str(DB),
)


# ----------------------------------------------------------------------
# Open Targets 26.09
# ----------------------------------------------------------------------

OT = RES / "opentargets" / "26.09"

for dataset in [
    "study",
    "credible_set",
    "l2g_prediction",
    "colocalisation",
]:
    d = OT / dataset
    check(
        f"Open Targets 26.09: {{dataset}}",
        (d / ".complete").exists()
        and d.exists()
        and any(d.rglob("*.parquet")),
        str(d),
    )


# ----------------------------------------------------------------------
# FUMA reproducibility
# ----------------------------------------------------------------------

check(
    "FUMA input directory",
    (ROOT / "fuma_reproducibility" / "input").is_dir(),
)

check(
    "FUMA output directory",
    (ROOT / "fuma_reproducibility" / "output").is_dir(),
)

check(
    "FUMA parameter template",
    (
        ROOT
        / "fuma_reproducibility"
        / "config"
        / "fuma_parameters.template.json"
    ).exists(),
)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

total = passed + failed

print(f"Passed     : {{passed}}")
print(f"Missing    : {{failed}}")
print(f"Total      : {{total}}")

if total:

    print(
        f"Completion : {{100 * passed / total:.1f}}%"
    )


if failed == 0:

    print()
    print("PIPELINE SETUP COMPLETE.")

    sys.exit(0)

else:

    print()
    print("PIPELINE SETUP INCOMPLETE.")

    sys.exit(1)
'''

write_file(
    "Verify_Setup.py",
    verify_python,
    executable=True,
)


# =============================================================================
# 10 OPEN TARGETS 26.09 BULK DATA
# =============================================================================

opentargets_script = f"""
#!/bin/bash
{sbatch_header("setup_opentargets", "res_ot", "10_opentargets", 4)}

set -euo pipefail

ROOT="{ROOT}"
RELEASE="{OPEN_TARGETS_RELEASE}"
BASE="rsync.ebi.ac.uk::pub/databases/opentargets/platform/$RELEASE/output"
OUT="$ROOT/resources/opentargets/$RELEASE"
TASK_ID="${{SLURM_ARRAY_TASK_ID:-${{1:-}}}}"

if [[ -z "$TASK_ID" ]]; then
    echo "Usage locally: bash $0 TASK_ID"
    echo "  1=study 2=credible_set 3=l2g_prediction 4=colocalisation"
    exit 2
fi

DATASETS=(study credible_set l2g_prediction colocalisation)
IDX=$((TASK_ID - 1))
if (( IDX < 0 || IDX >= ${{#DATASETS[@]}} )); then
    echo "ERROR: invalid TASK_ID=$TASK_ID"
    exit 2
fi

DATASET="${{DATASETS[$IDX]}}"
DEST="$OUT/$DATASET"
MARKER="$DEST/.complete"
RSYNC="$(command -v rsync || true)"

if [[ -z "$RSYNC" ]]; then
    echo "ERROR: rsync not found. Run Step00 software installation first."
    exit 2
fi

mkdir -p "$DEST"

if [[ -f "$MARKER" ]] && find "$DEST" -type f -name '*.parquet' -print -quit | grep -q .; then
    echo "[CACHE] Open Targets $RELEASE/$DATASET"
    du -sh "$DEST" || true
    exit 0
fi

echo "=============================================================="
echo "OPEN TARGETS $RELEASE DOWNLOAD"
echo "=============================================================="
echo "Dataset : $DATASET"
echo "Remote  : $BASE/$DATASET/"
echo "Local   : $DEST/"
echo
echo "Disk before:"
df -h "$ROOT" || true

"$RSYNC" \
    -rltv \
    --partial \
    --info=progress2 \
    "$BASE/$DATASET/" \
    "$DEST/"

if ! find "$DEST" -type f -name '*.parquet' -print -quit | grep -q .; then
    echo "ERROR: no Parquet files found after rsync for $DATASET"
    exit 1
fi

touch "$MARKER"

echo
echo "[OK] Open Targets dataset complete: $DATASET"
du -sh "$DEST" || true
echo
echo "Disk after:"
df -h "$ROOT" || true
"""

write_file(
    "Setup_10_Download_OpenTargets.sh",
    opentargets_script,
    executable=True,
)


# =============================================================================
# 11 VERIFY SLURM
# =============================================================================

verify_job = f"""
#!/bin/bash
{sbatch_header("setup_verify", "res_verify", "11_verify")}

set -euo pipefail

cd "{ROOT}"

# Record versions/releases actually present (setup_manifests/resource_versions.tsv)
"{CORE_ENV}/bin/python" "{ROOT}/Step00_Check_Resources.py" --record-versions

"{CORE_ENV}/bin/python" \
    "{ROOT}/Verify_Setup.py"
"""

write_file(
    "Setup_11_Verify.sh",
    verify_job,
    executable=True,
)


# =============================================================================
# MASTER SUBMIT
# =============================================================================

submit_script = f"""
#!/bin/bash
set -euo pipefail

ROOT="{ROOT}"
STATE="$ROOT/setup_manifests/resource_jobs.env"
cd "$ROOT"

mkdir -p "$ROOT/setup_manifests" "$ROOT/logs/resource_setup"

if [[ -s "$STATE" ]]; then
    source "$STATE" || true
    if [[ -n "${{JVERIFY:-}}" ]] && squeue -h -j "$JVERIFY" 2>/dev/null | grep -q .; then
        echo "Resource setup is already active."
        echo "Final verification job: $JVERIFY"
        echo "Monitor with: squeue --me"
        exit 0
    fi
fi

JGENOME=$(sbatch --parsable "$ROOT/Setup_02_Download_Genome_GENCODE.sh")
JGTEX=$(sbatch --parsable "$ROOT/Setup_03_Download_GTEx_V11.sh")
J1000=$(sbatch --parsable "$ROOT/Setup_04_Download_1000G.sh")
JVEP=$(sbatch --parsable "$ROOT/Setup_08_Download_VEP_Cache.sh")
JOT=$(sbatch --parsable "$ROOT/Setup_10_Download_OpenTargets.sh")
JALL=$(sbatch --parsable --dependency=afterok:$J1000 "$ROOT/Setup_05_Prepare_1000G_ALL.sh")
JSAMPLES=$(sbatch --parsable --dependency=afterok:$J1000 "$ROOT/Setup_06_Prepare_1000G_Sample_Lists.sh")
JPOP=$(sbatch --parsable --dependency=afterok:$JALL:$JSAMPLES "$ROOT/Setup_07_Prepare_1000G_Populations.sh")
JPANG=$(sbatch --parsable --dependency=afterok:$JGENOME "$ROOT/Setup_09_Build_Pangolin_DB.sh")
JVERIFY=$(sbatch --parsable --dependency=afterok:$JGENOME:$JGTEX:$J1000:$JVEP:$JALL:$JSAMPLES:$JPOP:$JPANG "$ROOT/Setup_11_Verify.sh")

cat > "$STATE" <<EOF
JGENOME=$JGENOME
JGTEX=$JGTEX
J1000=$J1000
JVEP=$JVEP
JALL=$JALL
JSAMPLES=$JSAMPLES
JPOP=$JPOP
JPANG=$JPANG
JOT=$JOT
JVERIFY=$JVERIFY
EOF

echo
echo "=============================================================="
echo "RESOURCE SETUP SUBMITTED"
echo "=============================================================="
echo "Genome/GENCODE        : $JGENOME"
echo "GTEx V11              : $JGTEX"
echo "1000G raw             : $J1000"
echo "VEP cache             : $JVEP"
echo "1000G ALL PGEN        : $JALL"
echo "1000G sample lists    : $JSAMPLES"
echo "1000G populations     : $JPOP"
echo "Pangolin DB           : $JPANG"
echo "Open Targets 26.09    : $JOT"
echo "Final verification    : $JVERIFY"
echo
echo "Monitor with: squeue --me"
echo "Live status (read-only): $ROOT/envs/pipeline/bin/python $ROOT/Step00_Check_Resources.py --inspect"
"""

write_file(
    "Setup_Submit_All_Resources.sh",
    submit_script,
    executable=True,
)


# =============================================================================
# ACTIVATION HELPER
# =============================================================================

activate_script = f"""
#!/bin/bash

ROOT="{ROOT}"

export PIPELINE_ROOT="$ROOT"

source "$ROOT/resource_paths.env"

echo "Main environment:"
echo "  $ROOT/envs/pipeline"

echo
echo "Activate with:"
echo
echo "  conda activate $ROOT/envs/pipeline"

echo
echo "SpliceAI:"
echo "  $ROOT/envs/spliceai/bin/spliceai"

echo
echo "Pangolin:"
echo "  $ROOT/envs/pangolin/bin/pangolin"
"""

write_file(
    "activate_pipeline.sh",
    activate_script,
    executable=True,
)


# =============================================================================
# REPORT GENERATED FILES
# =============================================================================

banner(
    "PIPELINE SETUP GENERATED"
)


print(
    f"Root:\n  {ROOT}"
)

print()

print(
    f"Resources:\n  {RESOURCES}"
)

print()

print(
    f"Environments:\n  {ENVS}"
)

print()

print(
    f"External tools:\n  {EXTERNAL}"
)

print()

print(
    "Generated:"
)


generated_files = [

    "environment.yml",
    "resource_paths.env",

    "Setup_00_Install_Software.sh",
    "Setup_01_Setup_Pangolin.sh",

    "Setup_02_Download_Genome_GENCODE.sh",
    "Setup_03_Download_GTEx_V11.sh",
    "Setup_04_Download_1000G.sh",

    "Setup_05_Prepare_1000G_ALL.sh",
    "Setup_06_Prepare_1000G_Sample_Lists.sh",
    "Setup_07_Prepare_1000G_Populations.sh",

    "Setup_08_Download_VEP_Cache.sh",
    "Setup_09_Build_Pangolin_DB.sh",

    "Setup_11_Verify.sh",

    "Setup_Submit_All_Resources.sh",

    "Verify_Setup.py",

    "activate_pipeline.sh",

]


for file in generated_files:

    print(
        f"  {file}"
    )


# =============================================================================
# EXECUTION
# =============================================================================

if args.generate_only:
    write_resource_versions()
    banner("GENERATION COMPLETE")
    print("No installation or resource commands were executed.")
    print("Inspect the current state (read-only) with:")
    print("  python Step00_Check_Resources.py --inspect")
    sys.exit(0)

package_manager = find_package_manager()
if package_manager is None and not args.resources_only:
    print("ERROR: mamba/micromamba/conda is not available.")
    print("Setup files were generated, but software installation cannot continue.")
    sys.exit(2)

install_only = args.install_only or args.no_submit
resources_only = args.resources_only

if not resources_only:
    banner("STAGE 1/2 - INSTALL / REPAIR SOFTWARE")
    run(["bash", str(ROOT / "Setup_00_Install_Software.sh")],
        label="Main environment + SpliceAI")
    run(["bash", str(ROOT / "Setup_01_Setup_Pangolin.sh")],
        label="Pangolin isolated environment")

    banner("SOFTWARE STATUS")
    run([str(CORE_ENV / "bin" / "python"), str(ROOT / "Verify_Setup.py")],
        label="Verify current state", check=False)

if install_only:
    write_resource_versions()
    banner("SOFTWARE STAGE COMPLETE")
    print("Resource jobs were intentionally not submitted.")
    print("Submit them later with:")
    print("  python Step00_Check_Resources.py --resources-only")
    sys.exit(0)

# Small shared GWAS Catalog tables (studies + ancestries) used by Step01
# discovery. Fetched here once; skipped when already cached.
banner("GWAS CATALOG METADATA")
try:
    import Step01_Plan_GWAS as _step01
    _session = _step01.make_session()
    _cache = ROOT / "reference_metadata" / "gwas_catalog"
    _step01.download_metadata(_session, _step01.STUDIES_URL, _cache / "gwas_catalog_studies.tsv")
    _step01.download_metadata(_session, _step01.ANCESTRY_URL, _cache / "gwas_catalog_ancestry.tsv")
except Exception as exc:  # network may be unavailable here; Step01 can still fetch it
    print(f"WARNING: GWAS Catalog metadata not fetched now ({type(exc).__name__}: {exc})")

if shutil.which("sbatch") is None:
    write_resource_versions()
    banner("RESOURCE SCRIPTS READY")
    print("SLURM sbatch is not available, so resource jobs were not submitted.")
    print("The generated Setup_02...Setup_11 scripts are ready for a SLURM system.")
    sys.exit(0)

banner("STAGE 2/2 - SUBMIT SHARED RESOURCE JOBS")
run(["bash", str(ROOT / "Setup_Submit_All_Resources.sh")],
    label="Submit resumable resource DAG")
write_resource_versions()

banner("BOOTSTRAP STARTED SUCCESSFULLY")
print("Software installation has been verified.")
print("Large shared biological resources are now downloading/preparing through SLURM.")
print()
print("Monitor jobs:")
print("  squeue --me")
print()
print("Check current completion at any time:")
print("  python Step00_Check_Resources.py --inspect")
print()
print("After the final verification job succeeds, shared setup is complete.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWAS2m - central software / resource registry and READ-ONLY inspector.

Single source of truth for:
  * which environments, tools, Python and R packages the pipeline needs;
  * which SHARED static resources exist, where they live, their source/version;
  * how each item is validated (cheap checks; huge files are never read);
  * resource_paths.env parsing (load_resource_paths) for every Step script;
  * require_resource(): scientific steps consume resources, never fetch them;
  * download_file(): the runtime downloader for phenotype-specific GWAS files
    (identical tools/flags to the generated Step02 array).

inspect_pipeline() is strictly read-only: it stats files and runs version /
import probes. It never installs, downloads, deletes, writes or submits.

Phenotype-specific GWAS summary statistics are NOT shared resources; they are
selected by Step01 and downloaded at run time (Step02).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REFERENCE_BUILD = "GRCh38"
GENCODE_RELEASE = "50"
VEP_RELEASE = "116"
GTEX_RELEASE = "v11"
REFERENCE_PANEL = "1000G_GRCh38_20190312"
OPEN_TARGETS_RELEASE = "26.09"

LD_ANCESTRIES = ("EUR", "AFR", "EAS", "SAS", "AMR")
INCOMPLETE_SUFFIXES = (".part", ".aria2", ".tmp")

AVAILABLE = "AVAILABLE"
MISSING = "MISSING"
INCOMPLETE = "INCOMPLETE"
INVALID = "INVALID"
BROKEN = "BROKEN"
VERSION_MISMATCH = "VERSION_MISMATCH"
OPTIONAL_MISSING = "OPTIONAL_MISSING"
NOT_CONFIGURED = "NOT_CONFIGURED"

SETUP_HINT = (
    "Run:\n"
    "  python Step00_Check_Resources.py --inspect\n"
    "then:\n"
    "  python Step00_Check_Resources.py --resources-only   (shared resources)\n"
    "  python Step00_Check_Resources.py --install-only     (software)"
)

ONE_K_BASE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL"
)
GTEX_BASE = "https://storage.googleapis.com/adult-gtex"
GENCODE_BASE = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{GENCODE_RELEASE}"


# =============================================================================
# SOFTWARE REGISTRY
#   step = earliest pipeline step that needs it; None = setup-time only.
# =============================================================================

ENVIRONMENTS = [
    # id, relative path, step, purpose, install method
    ("pipeline", "envs/pipeline", 1, "main pipeline stack (Python, R, PLINK2, VEP, ...)",
     "conda env from environment.yml + pip (gwaslab, polars)"),
    ("spliceai", "envs/spliceai", 9, "SpliceAI 1.3.1 (TensorFlow 2.15; incompatible with main env)",
     "isolated conda env + pip"),
    ("pangolin", "envs/pangolin", 10, "Pangolin (PyTorch; incompatible with main env)",
     "isolated conda env + git checkout + pip -e"),
]

# name, env, version command, required, step, purpose, install spec
TOOLS = [
    ("python", "pipeline", ["--version"], True, 1, "pipeline interpreter", "python=3.11"),
    ("Rscript", "pipeline", ["--version"], True, 6, "SuSiE-RSS / coloc", "r-base"),
    ("plink2", "pipeline", ["--version"], True, 5, "clumping / LD", "plink2"),
    ("bcftools", "pipeline", ["--version"], True, 9, "VCF normalisation", "bcftools"),
    ("samtools", "pipeline", ["--version"], True, 9, "FASTA index", "samtools"),
    ("tabix", "pipeline", ["--version"], True, 11, "eQTL Catalogue region queries", "htslib"),
    ("bgzip", "pipeline", ["--version"], False, 11, "compression", "htslib"),
    ("bedtools", "pipeline", ["--version"], False, None, "declared; not used by Steps 01-11", "bedtools"),
    ("vep", "pipeline", ["--help"], True, 7, "Ensembl VEP annotation", f"ensembl-vep={VEP_RELEASE}"),
    ("aria2c", "pipeline", ["--version"], True, None, "shared-resource downloads (setup)", "aria2"),
    ("pigz", "pipeline", ["--version"], True, None, "FASTA decompression (setup)", "pigz"),
    ("wget", "pipeline", ["--version"], False, 2, "GWAS download (first choice)", "wget"),
    ("curl", "pipeline", ["--version"], False, 2, "GWAS download (fallback)", "curl"),
    ("rsync", "pipeline", ["--version"], False, None, "Open Targets download (setup, optional)", "rsync"),
    ("git", "pipeline", ["--version"], True, None, "Pangolin checkout (setup)", "git"),
    ("spliceai", "spliceai", None, True, 9, "SpliceAI executable", "spliceai==1.3.1"),
    ("pangolin", "pangolin", None, True, 10, "Pangolin executable", "git+tkzeng/Pangolin"),
]

# import name, distribution name, env, required, step, install method, spec, used by
PYTHON_PACKAGES = [
    ("pandas", "pandas", "pipeline", True, 1, "conda", "pandas", "all steps"),
    ("numpy", "numpy", "pipeline", True, 1, "conda", "numpy", "all steps"),
    ("requests", "requests", "pipeline", True, 1, "conda", "requests", "Step01, Step06"),
    ("urllib3", "urllib3", "pipeline", True, 1, "conda", "urllib3", "Step01, Step06"),
    ("gwaslab", "gwaslab", "pipeline", True, 3, "pip", "gwaslab==4.2.3", "Step03 QC"),
    ("yaml", "PyYAML", "pipeline", False, 6, "conda", "pyyaml", "Step06 metadata, config/slurm.yaml"),
    ("scipy", "scipy", "pipeline", False, None, "conda", "scipy", "Pipeline_Report, Step13, Step14"),
    ("polars", "polars", "pipeline", False, None, "pip", "polars[rtcompat]", "Diagnose_Step11, Pipeline_Report"),
    ("orjson", "orjson", "pipeline", False, None, "conda", "orjson", "Diagnose_Step11"),
    ("pyarrow", "pyarrow", "pipeline", False, None, "conda", "pyarrow", "parquet (Open Targets)"),
    ("duckdb", "duckdb", "pipeline", False, None, "conda", "duckdb", "Step13"),
    ("pysam", "pysam", "pipeline", False, None, "conda", "pysam", "declared"),
    ("cyvcf2", "cyvcf2", "pipeline", False, None, "conda", "cyvcf2", "declared"),
    ("sklearn", "scikit-learn", "pipeline", False, None, "conda", "scikit-learn", "declared"),
    ("statsmodels", "statsmodels", "pipeline", False, None, "conda", "statsmodels", "Step14"),
    ("httpx", "httpx", "pipeline", False, None, "conda", "httpx", "declared"),
    ("tenacity", "tenacity", "pipeline", False, None, "conda", "tenacity", "declared"),
    ("tqdm", "tqdm", "pipeline", False, None, "conda", "tqdm", "Diagnose_Step11"),
    ("matplotlib", "matplotlib", "pipeline", False, None, "conda", "matplotlib", "declared"),
    ("networkx", "networkx", "pipeline", False, None, "conda", "networkx", "Step15"),
    ("gprofiler", "gprofiler-official", "pipeline", False, None, "conda", "gprofiler-official", "Step14"),
    ("spliceai", "spliceai", "spliceai", True, 9, "pip", "spliceai==1.3.1", "Step09"),
    ("tensorflow", "tensorflow-cpu", "spliceai", True, 9, "pip", "tensorflow-cpu==2.15.1", "Step09"),
    ("keras", "keras", "spliceai", True, 9, "pip", "keras==2.15.0", "Step09"),
    ("pangolin", "pangolin", "pangolin", True, 10, "git+pip -e", "tkzeng/Pangolin (commit pinned)", "Step10"),
    ("gffutils", "gffutils", "pangolin", True, 10, "conda", "gffutils", "Step10 / Pangolin DB"),
    ("pyfastx", "pyfastx", "pangolin", True, 10, "conda", "pyfastx", "Step10"),
    ("Bio", "biopython", "pangolin", True, 10, "conda", "biopython", "Step10"),
]

# name, required, step, spec, used by
R_PACKAGES = [
    ("susieR", True, 6, "r-susier", "Step06 SuSiE-RSS"),
    ("coloc", True, 11, "r-coloc", "Step11 coloc.bf_bf"),
    ("data.table", True, 11, "r-data.table", "coloc dependency"),
]


# =============================================================================
# SHARED RESOURCE REGISTRY
# =============================================================================

def _resources() -> list[dict]:
    r = []

    def add(**item):
        item.setdefault("ancestry", None)
        item.setdefault("extra", {})
        r.append(item)

    add(id="GWAS_CATALOG_STUDIES", type="METADATA", source="GWAS Catalog", version="latest",
        build="-", url="https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-download-studies-v1.0.3.1.txt",
        path="reference_metadata/gwas_catalog/gwas_catalog_studies.tsv", required=False, step=1,
        validation="file", setup="Step00 --resources-only (or Step01 on first run)")
    add(id="GWAS_CATALOG_ANCESTRY", type="METADATA", source="GWAS Catalog", version="latest",
        build="-", url="https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-download-ancestries-v1.0.3.1.txt",
        path="reference_metadata/gwas_catalog/gwas_catalog_ancestry.tsv", required=False, step=1,
        validation="file", setup="Step00 --resources-only (or Step01 on first run)")
    add(id="GRCH38_FASTA", type="GENOME", source="GENCODE", version=GENCODE_RELEASE, build=REFERENCE_BUILD,
        url=f"{GENCODE_BASE}/GRCh38.primary_assembly.genome.fa.gz", env_key="GRCH38_FASTA",
        path="resources/genome/GRCh38/GRCh38.primary_assembly.genome.fa", required=True, step=7,
        validation="file+fai", setup="Setup_02_Download_Genome_GENCODE.sh")
    add(id="GENCODE_GTF", type="ANNOTATION", source="GENCODE", version=GENCODE_RELEASE, build=REFERENCE_BUILD,
        url=f"{GENCODE_BASE}/gencode.v{GENCODE_RELEASE}.primary_assembly.annotation.gtf.gz", env_key="GENCODE50_GTF",
        path=f"resources/gencode/release{GENCODE_RELEASE}/gencode.v{GENCODE_RELEASE}.primary_assembly.annotation.gtf.gz",
        required=True, step=10, validation="file", setup="Setup_02_Download_Genome_GENCODE.sh")
    add(id="1000G_PANEL_METADATA", type="LD_REFERENCE", source="1000 Genomes", version=REFERENCE_PANEL,
        build=REFERENCE_BUILD, url="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel",
        path="resources/1000G/metadata/integrated_call_samples_v3.20130502.ALL.panel", required=True, step=None,
        validation="file", setup="Setup_04_Download_1000G.sh")
    add(id="1000G_RAW_VCF", type="LD_REFERENCE", source="1000 Genomes", version=REFERENCE_PANEL,
        build=REFERENCE_BUILD, url=ONE_K_BASE, path="resources/1000G/raw", required=True, step=None,
        validation="vcf_set", setup="Setup_04_Download_1000G.sh")
    add(id="1000G_ALL_PGEN", type="LD_REFERENCE", source="1000 Genomes", version=REFERENCE_PANEL,
        build=REFERENCE_BUILD, url="derived (plink2 --make-pgen)", path="resources/1000G/ALL", required=True,
        step=None, validation="plink_panel", extra={"prefix": "chr{chrom}"}, setup="Setup_05_Prepare_1000G_ALL.sh")
    for anc in LD_ANCESTRIES:
        add(id=f"1000G_{anc}", type="LD_REFERENCE", source="1000 Genomes", version=REFERENCE_PANEL,
            build=REFERENCE_BUILD, url="derived (plink2 --keep)", env_key=f"LD_REFERENCE_{anc}",
            path=f"resources/1000G/{anc}", required=True, step=5, ancestry=anc, validation="plink_panel",
            extra={"prefix": f"chr{{chrom}}_{anc}_GRCh38"},
            setup="Setup_06_Prepare_1000G_Sample_Lists.sh + Setup_07_Prepare_1000G_Populations.sh")
    add(id="VEP_CACHE", type="VEP", source="Ensembl", version=VEP_RELEASE, build=REFERENCE_BUILD,
        url=f"https://ftp.ensembl.org/pub/release-{VEP_RELEASE}/variation/indexed_vep_cache/homo_sapiens_vep_{VEP_RELEASE}_GRCh38.tar.gz",
        env_key="VEP_CACHE_DIR", path="resources/vep/cache", required=True, step=7, validation="vep_cache",
        setup="Setup_08_Download_VEP_Cache.sh")
    for name, kind in (("EQTL", "eQTL"), ("SQTL", "sQTL")):
        add(id=f"GTEX_{name}", type="QTL", source="GTEx", version=GTEX_RELEASE, build=REFERENCE_BUILD,
            url=f"{GTEX_BASE}/bulk-qtl/v11/single-tissue-cis-qtl/GTEx_Analysis_v11_{kind}.tar",
            path=f"resources/gtex/v11/qtl/{kind}", required=True, step=8, validation="gtex_pairs",
            extra={"tar": f"resources/gtex/v11/qtl/GTEx_Analysis_v11_{kind}.tar"},
            setup="Setup_03_Download_GTEx_V11.sh")
    add(id="GTEX_VARIANT_LOOKUP", type="QTL", source="GTEx", version=GTEX_RELEASE, build=REFERENCE_BUILD,
        url=f"{GTEX_BASE}/references/v11/reference-tables/GTEx_Analysis_2021-02-11_v11_WholeGenomeSeq_953Indiv.lookup_table.txt.gz",
        path="resources/gtex/v11/reference/GTEx_Analysis_2021-02-11_v11_WholeGenomeSeq_953Indiv.lookup_table.txt.gz",
        required=False, step=8, validation="file", setup="Setup_03_Download_GTEx_V11.sh")
    add(id="GTEX_GENCODE47", type="QTL", source="GTEx", version=GTEX_RELEASE, build=REFERENCE_BUILD,
        url=f"{GTEX_BASE}/references/v11/reference-tables/gencode.v47.genes.gtf",
        path="resources/gtex/v11/reference/gencode.v47.genes.gtf", required=False, step=11, validation="file",
        setup="Setup_03_Download_GTEx_V11.sh")
    for name, kind in (("EQTL", "eQTL"), ("SQTL", "sQTL")):
        add(id=f"GTEX_{name}_SUSIE", type="QTL", source="GTEx", version=GTEX_RELEASE, build=REFERENCE_BUILD,
            url=f"{GTEX_BASE}/bulk-qtl/v11/susie-qtl/GTEx_Analysis_v11_{kind}_SuSiE.tar",
            path=f"resources/gtex/v11/susie/{kind}_SuSiE", required=False, step=11, validation="complete_dir",
            setup="Setup_03_Download_GTEx_V11.sh")
    add(id="PANGOLIN_REPO", type="MODEL", source="github.com/tkzeng/Pangolin", version="pinned commit",
        build="-", url="https://github.com/tkzeng/Pangolin.git", path="external_tools/Pangolin", required=True,
        step=None, validation="git_repo", setup="Setup_01_Setup_Pangolin.sh")
    add(id="PANGOLIN_DB", type="MODEL", source="derived from GENCODE GTF", version=GENCODE_RELEASE,
        build=REFERENCE_BUILD, url="derived (Pangolin scripts/create_db.py)", env_key="PANGOLIN_DB",
        path=f"resources/pangolin/gencode.v{GENCODE_RELEASE}.primary_assembly.annotation.db", required=True,
        step=10, validation="file", setup="Setup_09_Build_Pangolin_DB.sh")
    for rid, name in (("EQTL_CATALOGUE_TABIX_PATHS", "tabix_ftp_paths.tsv"),
                      ("EQTL_CATALOGUE_DATASET_METADATA", "dataset_metadata_r8_beta.tsv")):
        add(id=rid, type="QTL", source="eQTL Catalogue", version="r8 (master)", build=REFERENCE_BUILD,
            url="https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/",
            path=f"resources/coloc/eqtl_catalogue/{name}", required=True, step=11, validation="file",
            setup="Step11 resource fetch (not yet migrated to Step00)")
    add(id="EQTL_CATALOGUE_DENSE", type="QTL", source="eQTL Catalogue", version="r8", build=REFERENCE_BUILD,
        url="eQTL Catalogue FTP (tabix)", path="resources/coloc/eqtl_catalogue/dense", required=True, step=11,
        validation="dir_files", setup="Step11 resource fetch (not yet migrated to Step00)")
    add(id="EQTL_CATALOGUE_SUSIE_PROVIDER", type="QTL", source="eQTL Catalogue", version="r8",
        build=REFERENCE_BUILD, url="eQTL Catalogue FTP (SuSiE lbf/cs)",
        path="resources/coloc/eqtl_catalogue/susie_provider", required=True, step=11, validation="dir_files",
        setup="Step11 resource fetch (not yet migrated to Step00)")
    add(id="MOLECULAR_QTL", type="QTL", source="several (see registry)", version="see registry",
        build=REFERENCE_BUILD, url="Setup_10_Download_Molecular_QTL.py", path="resources/molecular_qtl",
        required=False, step=11, validation="dir_files", setup="Setup_10_Download_Molecular_QTL.py")
    add(id="OPEN_TARGETS", type="ANNOTATION", source="Open Targets Platform", version=OPEN_TARGETS_RELEASE,
        build=REFERENCE_BUILD, url="rsync.ebi.ac.uk::pub/databases/opentargets/platform", env_key="OPEN_TARGETS_26_09",
        path=f"resources/opentargets/{OPEN_TARGETS_RELEASE}", required=False, step=13, validation="dir_files",
        setup="Setup_10_Download_OpenTargets.sh")
    return r


RESOURCES = _resources()
RESOURCE_BY_ID = {item["id"]: item for item in RESOURCES}


# =============================================================================
# resource_paths.env
# =============================================================================

def load_resource_paths(root: Path) -> dict[str, str]:
    """Parse <root>/resource_paths.env (KEY="value", optional 'export ')."""
    path = Path(root) / "resource_paths.env"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        value = value.strip().strip("'").strip('"')
        if key:
            out[key] = value
    return out


def resource_paths_env_text(root: Path) -> str:
    """Content Step00 writes to resource_paths.env (paths from the registry)."""
    root = Path(root)
    lines = [
        "# Generated by Step00_Check_Resources.py from gwas2m_resources.RESOURCES.",
        f'export PIPELINE_ROOT="{root}"',
        f'export PIPELINE_RESOURCES="{root / "resources"}"',
    ]
    for item in RESOURCES:
        if item.get("env_key"):
            lines.append(f'export {item["env_key"]}="{root / item["path"]}"')
    lines += [
        f'export GTEX_V11="{root / "resources" / "gtex" / "v11"}"',
        f'export VEP="{root / "envs" / "pipeline" / "bin" / "vep"}"',
        f'export PANGOLIN="{root / "envs" / "pangolin" / "bin" / "pangolin"}"',
        f'export SPLICEAI="{root / "envs" / "spliceai" / "bin" / "spliceai"}"',
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# VALIDATION (cheap; never reads large files)
# =============================================================================

def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def _partial_siblings(path: Path) -> list[Path]:
    return [Path(str(path) + s) for s in INCOMPLETE_SUFFIXES if Path(str(path) + s).exists()]


def _file_status(path: Path) -> tuple[str, str]:
    if path.is_file() and path.stat().st_size > 0:
        if Path(str(path) + ".aria2").exists():
            return INCOMPLETE, "aria2 control file present (download in progress)"
        return AVAILABLE, ""
    if _partial_siblings(path):
        return INCOMPLETE, f"partial download: {_partial_siblings(path)[0].name}"
    if path.exists():
        return INVALID, "empty file"
    return MISSING, ""


def _count_panel(directory: Path, prefix_template: str) -> int:
    count = 0
    for chrom in range(1, 23):
        prefix = directory / prefix_template.format(chrom=chrom)
        if all(Path(str(prefix) + ext).is_file() and Path(str(prefix) + ext).stat().st_size > 0
               for ext in (".pgen", ".pvar", ".psam")):
            count += 1
    return count


def _pair_files(directory: Path) -> list[Path]:
    """Same naming rule as Step08.discover_significant_pair_files()."""
    out = []
    if not directory.exists():
        return out
    for path in directory.rglob("*"):
        name = path.name.lower()
        if (path.is_file() and not name.startswith(".")
                and name.endswith((".txt.gz", ".tsv.gz", ".txt", ".tsv"))
                and ("signif" in name or "pair" in name)
                and "egenes" not in name and "sgenes" not in name):
            out.append(path)
    return out


def check_resource(root: Path, item: dict) -> dict:
    path = Path(root) / item["path"]
    status, detail = MISSING, ""
    method = item["validation"]

    if method == "file":
        status, detail = _file_status(path)
    elif method == "file+fai":
        status, detail = _file_status(path)
        if status == AVAILABLE and not (Path(str(path) + ".fai").is_file()
                                        and Path(str(path) + ".fai").stat().st_size > 0):
            status, detail = INCOMPLETE, "FASTA index (.fai) missing"
        elif status == MISSING and Path(str(path) + ".gz").exists():
            status, detail = INCOMPLETE, "compressed FASTA present but not decompressed/indexed"
    elif method == "plink_panel":
        n = _count_panel(path, item["extra"]["prefix"])
        status = AVAILABLE if n == 22 else (INCOMPLETE if n else MISSING)
        detail = f"{n}/22 chromosomes (pgen/pvar/psam)"
    elif method == "vcf_set":
        names = [f"ALL.chr{c}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz" for c in range(1, 23)]
        n_vcf = sum((path / n).is_file() and (path / n).stat().st_size > 0 for n in names)
        n_tbi = sum((path / (n + ".tbi")).is_file() for n in names)
        status = AVAILABLE if n_vcf == 22 and n_tbi == 22 else (INCOMPLETE if n_vcf or n_tbi else MISSING)
        detail = f"VCF {n_vcf}/22, TBI {n_tbi}/22"
    elif method == "vep_cache":
        releases = sorted(p.name for p in (path / "homo_sapiens").glob("*_GRCh38") if p.is_dir()) \
            if (path / "homo_sapiens").is_dir() else []
        wanted = f"{VEP_RELEASE}_GRCh38"
        if wanted in releases and any((path / "homo_sapiens" / wanted).iterdir()):
            status, detail = AVAILABLE, f"homo_sapiens/{wanted}"
        elif releases:
            status, detail = VERSION_MISMATCH, f"found {', '.join(releases)}; expected {wanted}"
        else:
            status = MISSING
    elif method == "gtex_pairs":
        files = _pair_files(path)
        tar = Path(root) / item["extra"]["tar"]
        if files and (path / ".complete").exists():
            status, detail = AVAILABLE, f"{len(files)} significant-pair files"
        elif files:
            status, detail = INCOMPLETE, f"{len(files)} files but no .complete marker (extraction unfinished?)"
        elif tar.exists() or _partial_siblings(tar):
            status, detail = INCOMPLETE, "archive present but not extracted"
        else:
            status = MISSING
    elif method == "complete_dir":
        if (path / ".complete").exists():
            status = AVAILABLE
        elif path.exists():
            status, detail = INCOMPLETE, "no .complete marker"
    elif method == "git_repo":
        status = AVAILABLE if (path / ".git").exists() else (INVALID if path.exists() else MISSING)
    elif method == "dir_files":
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
            partial = [p for p in files if p.name.endswith(INCOMPLETE_SUFFIXES)]
            real = [p for p in files if not p.name.endswith(INCOMPLETE_SUFFIXES) and not p.name.startswith(".")]
            if partial:
                status, detail = INCOMPLETE, f"{len(partial)} partial download file(s)"
            elif real:
                status, detail = AVAILABLE, f"{len(real)} files"
            else:
                status, detail = MISSING, "directory empty"

    if status == MISSING and not item["required"]:
        status = OPTIONAL_MISSING
    size = _size(path) if status in {AVAILABLE, INCOMPLETE, VERSION_MISMATCH} and method != "dir_files" else None
    return {
        "id": item["id"], "type": item["type"], "status": status, "detail": detail,
        "path": str(path), "required": item["required"], "step": item["step"],
        "ancestry": item["ancestry"], "source": item["source"], "version": item["version"],
        "build": item["build"], "url": item["url"], "size_bytes": size, "setup": item["setup"],
    }


def require_resource(root: Path, resource_id: str) -> Path:
    """Path of a validated shared resource, or a clear error (never downloads)."""
    item = RESOURCE_BY_ID[resource_id]
    result = check_resource(root, item)
    if result["status"] != AVAILABLE:
        raise FileNotFoundError(
            "Required shared resource is unavailable.\n"
            f"  Resource : {resource_id}\n"
            f"  Status   : {result['status']} {result['detail']}\n"
            f"  Path     : {result['path']}\n\n" + SETUP_HINT
        )
    return Path(result["path"])


# =============================================================================
# SOFTWARE PROBES (read-only: --version / import / packageVersion)
# =============================================================================

def env_path(root: Path, env: str) -> Path:
    return Path(root) / "envs" / env


def check_environment(root: Path, env_id: str) -> dict:
    path = env_path(root, env_id)
    if not path.exists():
        status = MISSING
    elif not (path / "conda-meta").is_dir() or not (path / "bin" / "python").exists():
        status = BROKEN
    else:
        status = AVAILABLE
    return {"status": status, "path": str(path)}


def _run(command, timeout=120) -> tuple[int, str]:
    try:
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True,
                                timeout=timeout, errors="replace")
        return result.returncode, (result.stdout + "\n" + result.stderr).strip()
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def _version_line(name: str, output: str) -> str:
    lines = [x.strip() for x in output.splitlines() if x.strip()]
    if name == "vep":
        hits = [x for x in lines if re.match(r"^ensembl(-vep)?\s*:", x)]
        return " | ".join(hits[:2]) if hits else (lines[0] if lines else "")
    if name == "Rscript":
        hits = [x for x in lines if "version" in x.lower()]
        return hits[0] if hits else (lines[0] if lines else "")
    return lines[0][:80] if lines else ""


def check_tool(root: Path, name: str, env: str, version_args) -> dict:
    candidate = env_path(root, env) / "bin" / name
    path = candidate if candidate.exists() else (Path(shutil.which(name)) if shutil.which(name) else None)
    if path is None:
        return {"status": MISSING, "path": None, "version": None}
    if version_args is None:
        return {"status": AVAILABLE, "path": str(path), "version": None}
    code, output = _run([path, *version_args])
    if code == 999:
        return {"status": BROKEN, "path": str(path), "version": None, "detail": output}
    return {"status": AVAILABLE, "path": str(path), "version": _version_line(name, output),
            "source": "env" if path == candidate else "PATH"}


_IMPORT_PROBE = r"""
import json, sys, importlib
from importlib import metadata
out = {}
for module, dist in json.loads(sys.argv[1]):
    try:
        m = importlib.import_module(module)
        try:
            version = metadata.version(dist)
        except Exception:
            version = getattr(m, "__version__", "")
        out[module] = ["AVAILABLE", str(version)]
    except Exception as exc:
        out[module] = ["MISSING", type(exc).__name__ + ": " + str(exc)[:200]]
print("GWAS2M_PROBE=" + json.dumps(out))
"""


def check_python_packages(root: Path, env: str, packages: list[tuple[str, str]]) -> tuple[dict, str]:
    python = env_path(root, env) / "bin" / "python"
    interpreter = python
    if not python.exists():
        if env != "pipeline":
            return {m: {"status": NOT_CONFIGURED, "version": None, "detail": f"{python} missing"}
                    for m, _ in packages}, str(python)
        interpreter = Path(sys.executable)  # pipeline env absent: report current interpreter
    code, output = _run([interpreter, "-c", _IMPORT_PROBE, json.dumps(packages)], timeout=600)
    match = re.search(r"GWAS2M_PROBE=(\{.*\})", output)
    if not match:
        return {m: {"status": BROKEN, "version": None, "detail": output[-200:]} for m, _ in packages}, str(interpreter)
    found = json.loads(match.group(1))
    return {m: {"status": s, "version": v if s == AVAILABLE else None, "detail": "" if s == AVAILABLE else v}
            for m, (s, v) in found.items()}, str(interpreter)


def check_r_packages(root: Path, packages: list[str]) -> tuple[dict, dict]:
    rscript = env_path(root, "pipeline") / "bin" / "Rscript"
    if not rscript.exists() and shutil.which("Rscript"):
        rscript = Path(shutil.which("Rscript"))
    if not rscript.exists():
        missing = {p: {"status": MISSING, "version": None} for p in packages}
        return {"status": MISSING, "version": None, "path": None}, missing
    code = (
        'cat("R=", R.version$major, ".", R.version$minor, "\\n", sep="");'
        f'for (p in c({",".join(repr(p) for p in packages)})) '
        '{ v <- tryCatch(as.character(packageVersion(p)), error=function(e) "MISSING"); '
        'cat(p, "=", v, "\\n", sep="") }'
    ).replace("'", '"')
    rc, output = _run([rscript, "-e", code], timeout=300)
    values = dict(re.findall(r"^([A-Za-z.0-9]+)=(\S+)$", output, flags=re.M))
    r_info = {"status": AVAILABLE if "R" in values else BROKEN, "version": values.get("R"), "path": str(rscript)}
    pkgs = {p: ({"status": AVAILABLE, "version": values[p]} if values.get(p, "MISSING") != "MISSING"
                else {"status": MISSING, "version": None}) for p in packages}
    return r_info, pkgs


# =============================================================================
# INSPECTION (READ ONLY)
# =============================================================================

def _needed(required: bool, step, through_step: int) -> bool:
    return bool(required) and step is not None and step <= through_step


def inspect_pipeline(root: Path, ancestry: str | None = None, through_step: int = 10) -> dict:
    """Read-only inventory. ancestry: None = all panels, else e.g. 'EUR'."""
    root = Path(root).resolve()
    ancestry = ancestry.upper() if ancestry else None
    report = {"project_root": str(root), "reference_build": REFERENCE_BUILD, "ancestry": ancestry,
              "through_step": through_step, "environments": {}, "software": {}, "python_packages": {},
              "r_packages": {}, "resources": {}, "resource_paths_env": {}}

    for env_id, rel, step, purpose, method in ENVIRONMENTS:
        result = check_environment(root, env_id)
        result.update({"required": True, "step": step, "purpose": purpose})
        report["environments"][env_id] = result

    for name, env, args, required, step, purpose, spec in TOOLS:
        result = check_tool(root, name, env, args)
        result.update({"required": required, "step": step, "purpose": purpose, "env": env})
        if result["status"] == MISSING and not required:
            result["status"] = OPTIONAL_MISSING
        report["software"][name] = result
    downloaders = [report["software"][n]["status"] == AVAILABLE for n in ("wget", "curl")]
    report["software"]["GWAS downloader (wget or curl)"] = {
        "status": AVAILABLE if any(downloaders) else MISSING, "required": True, "step": 2,
        "purpose": "Step02 runtime GWAS download", "path": None, "version": None}

    by_env: dict[str, list] = {}
    for module, dist, env, required, step, method, spec, used in PYTHON_PACKAGES:
        by_env.setdefault(env, []).append((module, dist))
    probed = {}
    for env, packages in by_env.items():
        results, interpreter = check_python_packages(root, env, packages)
        for module, _ in packages:
            probed[(env, module)] = (results[module], interpreter)
    for module, dist, env, required, step, method, spec, used in PYTHON_PACKAGES:
        result, interpreter = probed[(env, module)]
        result = dict(result)
        if result["status"] == MISSING and not required:
            result["status"] = OPTIONAL_MISSING
        result.update({"env": env, "required": required, "step": step, "install": f"{method}: {spec}",
                       "used_by": used, "interpreter": interpreter})
        report["python_packages"][f"{module}" if env == "pipeline" else f"{module} [{env}]"] = result

    r_info, r_pkgs = check_r_packages(root, [p for p, *_ in R_PACKAGES])
    report["software"]["R"] = {**r_info, "required": True, "step": 6, "purpose": "R runtime"}
    for name, required, step, spec, used in R_PACKAGES:
        report["r_packages"][name] = {**r_pkgs[name], "required": required, "step": step, "used_by": used}

    for item in RESOURCES:
        if item["ancestry"] and ancestry and item["ancestry"] != ancestry:
            continue
        report["resources"][item["id"]] = check_resource(root, item)
    if ancestry and ancestry not in LD_ANCESTRIES:
        report["resources"][f"1000G_{ancestry}"] = {
            "id": f"1000G_{ancestry}", "status": NOT_CONFIGURED, "required": True, "step": 5,
            "detail": f"no 1000 Genomes super-population panel is defined for {ancestry}",
            "path": None, "ancestry": ancestry, "type": "LD_REFERENCE"}

    env_values = load_resource_paths(root)
    report["resource_paths_env"] = {"exists": (root / "resource_paths.env").exists(), "keys": sorted(env_values)}

    report.update(readiness(report, through_step, ancestry))
    if not ancestry:
        # Global mode: is the pipeline ready for each ancestry individually?
        report["ancestry_readiness"] = {
            anc: not readiness(report, through_step, anc)["missing_required"] for anc in LD_ANCESTRIES
        }
    return report


def readiness(report: dict, through_step: int, ancestry: str | None) -> dict:
    """Blocking items for Steps01..through_step (and the given ancestry)."""
    blocking, optional, warnings, setup_missing = [], [], [], []
    sections = [("environments", "environment "), ("software", "tool "), ("python_packages", "python "),
                ("r_packages", "R package "), ("resources", "")]
    for section, prefix in sections:
        for raw_name, result in report[section].items():
            name = prefix + raw_name
            status = result["status"]
            if result.get("ancestry") and ancestry and result["ancestry"] != ancestry:
                continue
            if status == AVAILABLE:
                continue
            if status == VERSION_MISMATCH:
                warnings.append(f"{name}: {result.get('detail', '')}")
                continue
            label = f"{name} ({status.lower()}{': ' + result['detail'] if result.get('detail') else ''})"
            if result.get("ancestry") and not ancestry:
                # Global mode: one missing panel must not block other ancestries;
                # it is reported through ancestry_readiness instead.
                optional.append(label)
            elif _needed(result.get("required"), result.get("step"), through_step):
                blocking.append(name)
            elif result.get("required") and result.get("step") is None:
                setup_missing.append(label)
            else:
                optional.append(label)
    return {"pipeline_ready": not blocking, "missing_required": blocking,
            "missing_optional": optional, "warnings": warnings, "setup_inputs_missing": setup_missing}


# =============================================================================
# RUNTIME DOWNLOADER (phenotype-specific GWAS files only)
# =============================================================================

def download_file(download_url: str, target: Path) -> str:
    """Download one runtime input with the validated Step02 strategy.

    Same tools, flags, '.part' file and final rename as the generated Step02
    array (wget, then curl). Returns "SKIPPED_ALREADY_COMPLETE" or "COMPLETE".
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return "SKIPPED_ALREADY_COMPLETE"
    temporary = Path(str(target) + ".part")
    if shutil.which("wget"):
        command = ["wget", "--continue", "--tries=10", "--timeout=60", "-O", str(temporary), str(download_url)]
    elif shutil.which("curl"):
        command = ["curl", "--location", "--fail", "--retry", "10", "--retry-delay", "5",
                   "--continue-at", "-", "--output", str(temporary), str(download_url)]
    else:
        raise RuntimeError("neither wget nor curl is available.")
    subprocess.run(command, check=True)
    os.replace(temporary, target)
    return "COMPLETE"

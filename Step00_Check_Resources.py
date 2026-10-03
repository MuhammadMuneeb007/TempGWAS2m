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

Check status only:

    python Step00_Check_Resources.py --status

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

REFERENCE_BUILD = "GRCh38"

REFERENCE_PANEL = "1000G_GRCh38_20190312"


# =============================================================================
# SLURM DEFAULTS
# =============================================================================

def detect_slurm_partition() -> str:
    """Prefer SLURM_PARTITION; otherwise use UQ ascher when present; else default."""
    explicit = os.environ.get("SLURM_PARTITION", "").strip()
    if explicit:
        return explicit

    sinfo = shutil.which("sinfo")
    if sinfo:
        try:
            result = subprocess.run(
                [sinfo, "-h", "-o", "%P"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=20,
                check=False,
            )
            names = {x.strip().rstrip("*") for x in result.stdout.splitlines()}
            if "ascher" in names:
                return "ascher"
        except Exception:
            pass

    return ""


PARTITION = detect_slurm_partition()
SBATCH_PARTITION = f"#SBATCH --partition={PARTITION}" if PARTITION else ""

DOWNLOAD_MAX_PARALLEL = 6

PGEN_MAX_PARALLEL = 8

POPULATION_MAX_PARALLEL = 12


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
  - httpx
  - tenacity
  - tqdm

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
"""

write_file(
    "environment.yml",
    environment_yml,
)


# =============================================================================
# SOFTWARE MANIFEST
# =============================================================================

software_manifest = [
    ["plink2", "mamba", "plink2"],
    ["bcftools", "mamba", "bcftools"],
    ["tabix", "mamba", "htslib"],
    ["bgzip", "mamba", "htslib"],
    ["samtools", "mamba", "samtools"],
    ["bedtools", "mamba", "bedtools"],
    ["aria2c", "mamba", "aria2"],
    ["VEP", "mamba", f"ensembl-vep={VEP_RELEASE}"],
    ["GWASLab", "pip", "gwaslab==4.2.3"],
    ["Polars", "pip", "polars[rtcompat]"],
    ["SuSiE-RSS", "mamba", "r-susier"],
    ["coloc", "mamba", "r-coloc"],
    ["data.table", "mamba", "r-data.table"],
    ["SpliceAI", "pip-isolated-env", "spliceai"],
    ["Pangolin", "git+isolated-env", "tkzeng/Pangolin"],
]

with open(
    MANIFESTS / "software.lock.tsv",
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
        "INSTALL_METHOD",
        "SPECIFICATION",
    ])

    writer.writerows(
        software_manifest
    )


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


with open(
    MANIFESTS / "resources.lock.tsv",
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.writer(
        handle,
        delimiter="\t",
    )

    writer.writerow([
        "RESOURCE",
        "VERSION",
        "BUILD",
        "SOURCE_URL",
        "LOCAL_PATH",
    ])

    writer.writerows(
        resource_rows
    )


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

resource_paths = f"""
export PIPELINE_ROOT="{ROOT}"
export PIPELINE_RESOURCES="{RESOURCES}"

export GRCH38_FASTA="{RESOURCES}/genome/GRCh38/GRCh38.primary_assembly.genome.fa"

export GENCODE50_GTF="{RESOURCES}/gencode/release50/gencode.v50.primary_assembly.annotation.gtf.gz"

export GTEX_V11="{RESOURCES}/gtex/v11"

export VEP_CACHE_DIR="{RESOURCES}/vep/cache"

export LD_REFERENCE_EUR="{RESOURCES}/1000G/EUR"
export LD_REFERENCE_AFR="{RESOURCES}/1000G/AFR"
export LD_REFERENCE_EAS="{RESOURCES}/1000G/EAS"
export LD_REFERENCE_SAS="{RESOURCES}/1000G/SAS"
export LD_REFERENCE_AMR="{RESOURCES}/1000G/AMR"

export PANGOLIN_DB="{RESOURCES}/pangolin/gencode.v50.primary_assembly.annotation.db"
"""

write_file(
    "resource_paths.env",
    resource_paths,
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

for TOOL in plink2 bcftools samtools tabix bgzip bedtools aria2c pigz vep Rscript; do
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
#SBATCH --job-name=res_genome
{SBATCH_PARTITION}
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output={LOGS}/02_genome.%j.out
#SBATCH --error={LOGS}/02_genome.%j.err

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
#SBATCH --job-name=res_gtex
{SBATCH_PARTITION}
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-6%4
#SBATCH --output={LOGS}/03_gtex.%A_%a.out
#SBATCH --error={LOGS}/03_gtex.%A_%a.err

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
#SBATCH --job-name=res_1000g
{SBATCH_PARTITION}
#SBATCH --time=240:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-23%{DOWNLOAD_MAX_PARALLEL}
#SBATCH --output={LOGS}/04_1000g.%A_%a.out
#SBATCH --error={LOGS}/04_1000g.%A_%a.err

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
#SBATCH --job-name=res_1000g_all
{SBATCH_PARTITION}
#SBATCH --time=120:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-22%{PGEN_MAX_PARALLEL}
#SBATCH --output={LOGS}/05_1000g_all.%A_%a.out
#SBATCH --error={LOGS}/05_1000g_all.%A_%a.err

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
#SBATCH --job-name=res_1000g_samples
{SBATCH_PARTITION}
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output={LOGS}/06_samples.%j.out
#SBATCH --error={LOGS}/06_samples.%j.err

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
#SBATCH --job-name=res_1000g_pop
{SBATCH_PARTITION}
#SBATCH --time=120:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-110%{POPULATION_MAX_PARALLEL}
#SBATCH --output={LOGS}/07_1000g_pop.%A_%a.out
#SBATCH --error={LOGS}/07_1000g_pop.%A_%a.err

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
#SBATCH --job-name=res_vep
{SBATCH_PARTITION}
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output={LOGS}/08_vep.%j.out
#SBATCH --error={LOGS}/08_vep.%j.err

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
#SBATCH --job-name=res_pangolin_db
{SBATCH_PARTITION}
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output={LOGS}/09_pangolin_db.%j.out
#SBATCH --error={LOGS}/09_pangolin_db.%j.err

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
# 10 VERIFY SLURM
# =============================================================================

verify_job = f"""
#!/bin/bash
#SBATCH --job-name=res_verify
{SBATCH_PARTITION}
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output={LOGS}/10_verify.%j.out
#SBATCH --error={LOGS}/10_verify.%j.err

set -euo pipefail

cd "{ROOT}"

"{CORE_ENV}/bin/python" \
    "{ROOT}/Verify_Setup.py"
"""

write_file(
    "Setup_10_Verify.sh",
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
JALL=$(sbatch --parsable --dependency=afterok:$J1000 "$ROOT/Setup_05_Prepare_1000G_ALL.sh")
JSAMPLES=$(sbatch --parsable --dependency=afterok:$J1000 "$ROOT/Setup_06_Prepare_1000G_Sample_Lists.sh")
JPOP=$(sbatch --parsable --dependency=afterok:$JALL:$JSAMPLES "$ROOT/Setup_07_Prepare_1000G_Populations.sh")
JPANG=$(sbatch --parsable --dependency=afterok:$JGENOME "$ROOT/Setup_09_Build_Pangolin_DB.sh")
JVERIFY=$(sbatch --parsable --dependency=afterok:$JGENOME:$JGTEX:$J1000:$JVEP:$JALL:$JSAMPLES:$JPOP:$JPANG "$ROOT/Setup_10_Verify.sh")

cat > "$STATE" <<EOF
JGENOME=$JGENOME
JGTEX=$JGTEX
J1000=$J1000
JVEP=$JVEP
JALL=$JALL
JSAMPLES=$JSAMPLES
JPOP=$JPOP
JPANG=$JPANG
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
echo "Final verification    : $JVERIFY"
echo
echo "Monitor with: squeue --me"
echo "Live status: $ROOT/envs/pipeline/bin/python $ROOT/Verify_Setup.py"
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
# ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(
    description="Install software, prepare reproducible resources, and verify the GWAS mechanism pipeline."
)

parser.add_argument("--generate-only", action="store_true",
                    help="Generate setup files only; do not install or submit anything.")
parser.add_argument("--install-only", action="store_true",
                    help="Install/repair software only; do not submit resource jobs.")
parser.add_argument("--resources-only", action="store_true",
                    help="Skip software installation and submit resource jobs only.")
parser.add_argument("--status", action="store_true",
                    help="Generate files, run Verify_Setup.py, print status, and exit.")
parser.add_argument("--no-submit", action="store_true",
                    help="Backward-compatible alias for --install-only.")
parser.add_argument("--submit", action="store_true", help=argparse.SUPPRESS)

args = parser.parse_args()
if args.install_only and args.resources_only:
    parser.error("--install-only and --resources-only cannot be used together")


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

    "Setup_10_Verify.sh",

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
    banner("GENERATION COMPLETE")
    print("No installation or resource commands were executed.")
    sys.exit(0)

if args.status:
    banner("CURRENT SETUP STATUS")
    rc = run([sys.executable, str(ROOT / "Verify_Setup.py")],
             label="Verify software and resources", check=False)
    sys.exit(rc)

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
    banner("SOFTWARE STAGE COMPLETE")
    print("Resource jobs were intentionally not submitted.")
    print("Submit them later with:")
    print("  python Step00_Check_Resources.py --resources-only")
    sys.exit(0)

if shutil.which("sbatch") is None:
    banner("RESOURCE SCRIPTS READY")
    print("SLURM sbatch is not available, so resource jobs were not submitted.")
    print("The generated Setup_02...Setup_10 scripts are ready for a SLURM system.")
    sys.exit(0)

banner("STAGE 2/2 - SUBMIT SHARED RESOURCE JOBS")
run(["bash", str(ROOT / "Setup_Submit_All_Resources.sh")],
    label="Submit resumable resource DAG")

banner("BOOTSTRAP STARTED SUCCESSFULLY")
print("Software installation has been verified.")
print("Large shared biological resources are now downloading/preparing through SLURM.")
print()
print("Monitor jobs:")
print("  squeue --me")
print()
print("Check current completion at any time:")
print("  python Step00_Check_Resources.py --status")
print()
print("After the final verification job succeeds, shared setup is complete.")

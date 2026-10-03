#!/usr/bin/env python3

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
            f"[OK]      {name:35} {detail}"
        )

    else:

        failed += 1

        print(
            f"[MISSING] {name:35} {detail}"
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
                    f"import {module}"
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
                    f'requireNamespace("{package}", quietly=TRUE),0,1))'
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        good = result.returncode == 0

    check(
        f"R: {package}",
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
        f"ALL.chr{chromosome}."
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
    f"{n_vcf}/22",
)

check(
    "1000G raw TBI",
    n_tbi == 22,
    f"{n_tbi}/22",
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
        / f"chr{chromosome}"
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
    f"{all_complete}/22",
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
            / f"chr{chromosome}_{ancestry}_GRCh38"
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
        f"1000G {ancestry}",
        count == 22,
        f"{count}/22",
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

print(f"Passed     : {passed}")
print(f"Missing    : {failed}")
print(f"Total      : {total}")

if total:

    print(
        f"Completion : {100 * passed / total:.1f}%"
    )


if failed == 0:

    print()
    print("PIPELINE SETUP COMPLETE.")

    sys.exit(0)

else:

    print()
    print("PIPELINE SETUP INCOMPLETE.")

    sys.exit(1)

 
# -*- coding: utf-8 -*-

"""
Setup_10_Download_Molecular_QTL.py

Purpose
-------
Download and organize molecular-QTL resources needed by GWAS2m Step11.

Strict design:
  * GRCh38-first.
  * Full/dense summary statistics are preferred for formal colocalization.
  * Existing GTEx eQTL/sQTL resources are checked but not re-downloaded here.
  * Large downloads are resumable.
  * Every resource gets provenance metadata.
  * A .complete marker is written only after a successful download.
  * The script can PLAN before downloading so you can check disk usage first.

Default core resources
----------------------
1) pQTL:
   - INTERVAL / Sun et al. 2018 plasma pQTL
   - eQTL Catalogue accession QTS000035 / QTD000584
   - GRCh38 harmonized by the eQTL Catalogue

2) Brain pQTL + mQTL:
   - ADSP FunGen xQTL Atlas NG00184.v1
   - GRCh38
   - full "all" summary-statistic archives
   - standardized specifically for downstream analyses such as colocalization

3) caQTL:
   - GRCh38 re-analysis of Kumasaka et al. ATAC-seq caQTL
   - QTD100018
   - Zenodo record 13848268

Optional:
   - ADSP haQTL full summary statistics via --include-haqtl

Directory layout
----------------
resources/
└── molecular_qtl/
    ├── manifests/
    ├── pqtl/
    │   ├── interval_sun2018/
    │   └── adsp_fungen_brain/
    ├── mqtl/
    │   └── adsp_fungen_brain/
    ├── caqtl/
    │   └── kumasaka_grch38/
    ├── haqtl/
    │   └── adsp_fungen_brain/
    └── resource_registry.tsv

Recommended use
---------------
PLAN ONLY:
    python Setup_10_Download_Molecular_QTL.py --plan

DOWNLOAD CORE DATA:
    python Setup_10_Download_Molecular_QTL.py --download

DOWNLOAD + OPTIONAL haQTL:
    python Setup_10_Download_Molecular_QTL.py --download --include-haqtl

ONLY SELECTED TYPES:
    python Setup_10_Download_Molecular_QTL.py --download --types pqtl,mqtl,caqtl

CUSTOM ROOT:
    python Setup_10_Download_Molecular_QTL.py --download --root /path/to/GWAS2m

This script does NOT silently liftover GRCh37 resources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


VERSION = "1.0.0"

NIAGADS_DATASET = "NG00184.v1"
NIAGADS_MANIFEST_URL = (
    "https://st1.niagads.org/portal/download-public/NG00184.v1/fm"
)

NIAGADS_DIRECT_URLS = {
    "ADSP_FunGen_xQTL.v1.pQTL.all.tar":
        "https://storage.googleapis.com/"
        "gcp-public-data-gcp-public-data-niagads-gtex-v9/"
        "release/QC/ADSP_FunGen_xQTL.v1.pQTL.all.tar",

    "ADSP_FunGen_xQTL.v1.mQTL.all.tar":
        "https://st1.niagads.org/portal/v1/"
        "download-public/NG00184/fileset/57053",

    "ADSP_FunGen_xQTL.v1.haQTL.all.tar":
        "https://st1.niagads.org/portal/v1/"
        "download-public/NG00184/fileset/57063",
}

EQTL_CATALOGUE_MANIFEST_URLS = {
    "tabix_ftp_paths.tsv":
        "https://raw.githubusercontent.com/eQTL-Catalogue/"
        "eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths.tsv",
    "tabix_ftp_paths_imported.tsv":
        "https://raw.githubusercontent.com/eQTL-Catalogue/"
        "eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths_imported.tsv",
}

SUN_PQTL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/"
    "QTS000035/QTD000584/QTD000584.all.tsv.gz"
)
SUN_PQTL_TBI_URL = SUN_PQTL_URL + ".tbi"

KUMASAKA_CA_URL = (
    "https://zenodo.org/records/13848268/files/"
    "QTD100018.all.tsv.gz?download=1"
)
KUMASAKA_CA_META_URL = (
    "https://zenodo.org/records/13848268/files/"
    "QTD100018_peak_metadata.tsv.gz?download=1"
)

KUMASAKA_CA_MD5 = "5f92d59ebcff47d151b10550cbfc67bd"
KUMASAKA_CA_META_MD5 = "4a77a273b0728a7b3ca35f0d914955b7"


@dataclass
class Resource:
    resource_id: str
    qtl_type: str
    source: str
    build: str
    tissue: str
    role: str
    local_path: str
    url: str = ""
    md5: str = ""
    expected_size_bytes: Optional[int] = None
    status: str = "PLANNED"
    note: str = ""


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "UNKNOWN"
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if x < 1024.0 or unit == "PB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def banner(text: str) -> None:
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def free_space(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run([str(x) for x in cmd], check=check)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def file_md5(path: Path, block: int = 16 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_md5(path: Path, expected: str) -> None:
    if not expected:
        return
    print(f"[MD5] {path.name}")
    got = file_md5(path)
    if got.lower() != expected.lower():
        raise RuntimeError(
            f"MD5 mismatch for {path}\n"
            f"expected={expected}\n"
            f"got={got}"
        )
    print(f"[OK] MD5 {got}")


def download_http(url: str, out: Path) -> None:
    """
    Resumable downloader.
    Prefer wget, then curl. urllib is last fallback and is not resumable.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    if command_exists("wget"):
        cmd = [
            "wget",
            "-c",
            "--tries=20",
            "--timeout=60",
            "--read-timeout=120",
            "--retry-connrefused",
            "-O", str(out),
            url,
        ]
        run(cmd)
        return

    if command_exists("curl"):
        cmd = [
            "curl",
            "-fL",
            "--retry", "20",
            "--retry-delay", "5",
            "--retry-all-errors",
            "-C", "-",
            "-o", str(out),
            url,
        ]
        run(cmd)
        return

    print("[WARN] wget/curl unavailable; using urllib without resume.")
    urllib.request.urlretrieve(url, out)


def s3_to_https(url: str) -> str:
    # s3://bucket/key -> https://bucket.s3.amazonaws.com/key
    x = url[len("s3://"):]
    bucket, _, key = x.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def download_any(url: str, out: Path) -> None:
    if url.startswith("s3://"):
        if command_exists("aws"):
            cmd = ["aws", "s3", "cp", url, str(out), "--no-progress"]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                return
            print("[WARN] aws s3 cp failed; trying public HTTPS form.")
        download_http(s3_to_https(url), out)
        return

    download_http(url, out)


def fetch_small_file(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_http(url, out)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def parse_size(value: str) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s:
        return None

    if re.fullmatch(r"\d+", s):
        try:
            return int(s)
        except Exception:
            return None

    m = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)",
        s,
        flags=re.I,
    )
    if not m:
        return None

    value_num = float(m.group(1))
    unit = m.group(2).upper()
    factors = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    return int(value_num * factors[unit])


def read_csv_flexible(path: Path) -> list[dict[str, str]]:
    text = path.read_text(errors="replace")
    sample = text[:10000]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    return [
        {str(k): ("" if v is None else str(v)) for k, v in row.items()}
        for row in reader
    ]


def resolve_niagads_archive(
    rows: list[dict[str, str]],
    filename: str,
) -> tuple[str, Optional[int], dict[str, str]]:
    """
    Find a target archive in the public NIAGADS manifest.

    The manifest schema can change, so this intentionally searches values rather
    than depending on one hard-coded column name.
    """
    matches: list[dict[str, str]] = []

    for row in rows:
        values = list(row.values())
        if any(filename in str(v) for v in values):
            matches.append(row)

    if not matches:
        raise RuntimeError(
            f"Could not find {filename} in {NIAGADS_DATASET} manifest."
        )

    row = matches[0]

    # Prefer a direct HTTP/HTTPS path, otherwise s3://.
    http_candidates: list[str] = []
    s3_candidates: list[str] = []

    for value in row.values():
        v = str(value).strip()

        for u in re.findall(r"https?://[^\s,;\"']+", v):
            if filename in u or filename in v:
                http_candidates.append(u)

        for u in re.findall(r"s3://[^\s,;\"']+", v):
            if filename in u or filename in v:
                s3_candidates.append(u)

    if http_candidates:
        url = http_candidates[0]
    elif s3_candidates:
        url = s3_candidates[0]
    elif filename in NIAGADS_DIRECT_URLS:
        url = NIAGADS_DIRECT_URLS[filename]
    else:
        path_candidates = [
            str(v).strip()
            for v in row.values()
            if filename in str(v)
        ]
        raise RuntimeError(
            "Found NIAGADS metadata but no download URL is registered.\n"
            f"Target: {filename}\n"
            f"Row: {json.dumps(row, indent=2)}\n"
            f"Path-like fields: {path_candidates}"
        )

    size = None
    for key, value in row.items():
        if "size" in key.lower():
            size = parse_size(value)
            if size is not None:
                break

    return url, size, row


def mark_complete(path: Path) -> None:
    marker = Path(str(path) + ".complete")
    marker.write_text(
        json.dumps(
            {
                "completed_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                "file": str(path),
                "size_bytes": path.stat().st_size,
            },
            indent=2,
        ) + "\n"
    )


def is_complete(path: Path) -> bool:
    marker = Path(str(path) + ".complete")
    return path.exists() and path.stat().st_size > 0 and marker.exists()


def save_provenance(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def extract_tar(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".complete"

    if marker.exists():
        print(f"[CACHE] extracted archive: {dest}")
        return

    banner(f"EXTRACTING {path.name}")
    with tarfile.open(path, "r:*") as tf:
        tf.extractall(dest)

    marker.write_text(
        json.dumps(
            {
                "archive": str(path),
                "completed_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
            },
            indent=2,
        ) + "\n"
    )


def write_registry(path: Path, resources: list[Resource]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "resource_id",
        "qtl_type",
        "source",
        "build",
        "tissue",
        "role",
        "local_path",
        "url",
        "md5",
        "expected_size_bytes",
        "status",
        "note",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for r in resources:
            writer.writerow(asdict(r))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download molecular-QTL resources for GWAS2m Step11."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan",
        action="store_true",
        help="Resolve resources and show the download plan only.",
    )
    mode.add_argument(
        "--download",
        action="store_true",
        help="Download the requested resources.",
    )

    parser.add_argument(
        "--root",
        default=".",
        help="GWAS2m project root. Default: current directory.",
    )
    parser.add_argument(
        "--types",
        default="pqtl,mqtl,caqtl",
        help="Comma-separated: pqtl,mqtl,caqtl. Default: all three.",
    )
    parser.add_argument(
        "--include-haqtl",
        action="store_true",
        help="Also download ADSP brain haQTL full summary statistics.",
    )
    parser.add_argument(
        "--include-adsp-pqtl",
        action="store_true",
        help=(
            "OPTIONAL: include the very large ADSP FunGen brain pQTL archive. "
            "Disabled by default because it is large and its download endpoint "
            "must be obtained from NIAGADS."
        ),
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded NIAGADS .tar archives.",
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=20.0,
        help="Keep at least this much disk free. Default: 20 GB.",
    )
    parser.add_argument(
        "--force-low-space",
        action="store_true",
        help="Allow download even if estimated free-space requirement fails.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    base = root / "resources" / "molecular_qtl"
    manifests = base / "manifests"

    requested = {
        x.strip().lower()
        for x in args.types.split(",")
        if x.strip()
    }

    allowed = {"pqtl", "mqtl", "caqtl"}
    unknown = requested - allowed
    if unknown:
        raise SystemExit(
            f"Unknown QTL type(s): {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )

    for d in [
        base,
        manifests,
        base / "pqtl",
        base / "mqtl",
        base / "caqtl",
        base / "haqtl",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    banner("GWAS2m MOLECULAR-QTL RESOURCE SETUP")
    print(f"Version       : {VERSION}")
    print(f"Project root  : {root}")
    print(f"Resource root : {base}")
    print(f"Requested     : {', '.join(sorted(requested))}")
    print(f"Free space    : {human_bytes(free_space(base))}")

    # ------------------------------------------------------------------
    # Existing local GTEx check
    # ------------------------------------------------------------------
    banner("EXISTING GTEx eQTL/sQTL CHECK")

    existing_gtex = [
        root / "resources/gtex/v11/qtl/GTEx_Analysis_v11_eQTL.tar",
        root / "resources/gtex/v11/qtl/GTEx_Analysis_v11_sQTL.tar",
        root / "resources/gtex/v11/susie/GTEx_Analysis_v11_eQTL_SuSiE.tar",
        root / "resources/gtex/v11/susie/GTEx_Analysis_v11_sQTL_SuSiE.tar",
    ]

    for p in existing_gtex:
        print(
            f"{'[OK]' if p.exists() and p.stat().st_size > 0 else '[MISSING]'} "
            f"{p}"
        )

    # ------------------------------------------------------------------
    # Small eQTL Catalogue manifests
    # ------------------------------------------------------------------
    banner("eQTL CATALOGUE MANIFESTS")

    for name, url in EQTL_CATALOGUE_MANIFEST_URLS.items():
        out = manifests / name
        if not out.exists() or out.stat().st_size == 0:
            print(f"[FETCH] {name}")
            fetch_small_file(url, out)
        else:
            print(f"[CACHE] {out}")

    # ------------------------------------------------------------------
    # NIAGADS public manifest for brain pQTL/mQTL
    # ------------------------------------------------------------------
    niagads_manifest = manifests / f"{NIAGADS_DATASET}.csv"

    need_niagads = (
        "mqtl" in requested
        or args.include_adsp_pqtl
        or args.include_haqtl
    )

    niagads_rows: list[dict[str, str]] = []

    if need_niagads:
        banner("NIAGADS ADSP FunGen xQTL MANIFEST")

        if not niagads_manifest.exists() or niagads_manifest.stat().st_size == 0:
            print(f"[FETCH] {NIAGADS_MANIFEST_URL}")
            fetch_small_file(NIAGADS_MANIFEST_URL, niagads_manifest)
        else:
            print(f"[CACHE] {niagads_manifest}")

        niagads_rows = read_csv_flexible(niagads_manifest)
        print(f"Manifest rows : {len(niagads_rows):,}")

    resources: list[Resource] = []

    # ------------------------------------------------------------------
    # pQTL - INTERVAL / Sun 2018
    # ------------------------------------------------------------------
    if "pqtl" in requested:
        d = base / "pqtl" / "interval_sun2018"
        d.mkdir(parents=True, exist_ok=True)

        p = d / "QTD000584.all.tsv.gz"
        resources.append(
            Resource(
                resource_id="QTD000584",
                qtl_type="pQTL",
                source="eQTL Catalogue / INTERVAL / Sun_2018",
                build="GRCh38",
                tissue="plasma",
                role="dense_summary_statistics",
                local_path=str(p),
                url=SUN_PQTL_URL,
                note=(
                    "Protein QTL study QTS000035. "
                    "Use dense regional rows for formal colocalization."
                ),
            )
        )

        tbi = d / "QTD000584.all.tsv.gz.tbi"
        resources.append(
            Resource(
                resource_id="QTD000584_TBI",
                qtl_type="pQTL",
                source="eQTL Catalogue / INTERVAL / Sun_2018",
                build="GRCh38",
                tissue="plasma",
                role="tabix_index",
                local_path=str(tbi),
                url=SUN_PQTL_TBI_URL,
            )
        )

        save_provenance(
            d,
            {
                "resource": "INTERVAL/Sun_2018 plasma pQTL",
                "study_accession": "QTS000035",
                "dataset_accession": "QTD000584",
                "genome_build": "GRCh38",
                "source": "eQTL Catalogue",
                "summary_statistics": SUN_PQTL_URL,
                "purpose": "formal GWAS-pQTL colocalization",
            },
        )

        if args.include_adsp_pqtl:
            # ADSP brain pQTL
            target = "ADSP_FunGen_xQTL.v1.pQTL.all.tar"
            url, size, row = resolve_niagads_archive(niagads_rows, target)
            adsp_dir = base / "pqtl" / "adsp_fungen_brain"
            adsp_dir.mkdir(parents=True, exist_ok=True)
            out = adsp_dir / target

            resources.append(
                Resource(
                    resource_id="NG00184_v1_pQTL",
                    qtl_type="pQTL",
                    source="ADSP FunGen xQTL Atlas",
                    build="GRCh38",
                    tissue="brain_multi_region_multi_celltype",
                    role="dense_summary_statistics_archive",
                    local_path=str(out),
                    url=url,
                    expected_size_bytes=size,
                    note="Full harmonized brain pQTL summary statistics.",
                )
            )

            save_provenance(
                adsp_dir,
                {
                    "resource": "ADSP FunGen xQTL Atlas pQTL",
                    "accession": "NG00184.v1",
                    "genome_build": "GRCh38",
                    "archive": target,
                    "manifest_row": row,
                    "purpose": "formal GWAS-pQTL colocalization",
                },
            )

        else:
            print(
                "[SKIP] ADSP FunGen brain pQTL archive is disabled by default. "
                "Using the local Sun/INTERVAL pQTL instead."
            )

    # ------------------------------------------------------------------
    # mQTL - ADSP brain
    # ------------------------------------------------------------------
    if "mqtl" in requested:
        target = "ADSP_FunGen_xQTL.v1.mQTL.all.tar"
        url, size, row = resolve_niagads_archive(niagads_rows, target)

        d = base / "mqtl" / "adsp_fungen_brain"
        d.mkdir(parents=True, exist_ok=True)
        out = d / target

        resources.append(
            Resource(
                resource_id="NG00184_v1_mQTL",
                qtl_type="mQTL",
                source="ADSP FunGen xQTL Atlas",
                build="GRCh38",
                tissue="brain_multi_region_multi_celltype",
                role="dense_summary_statistics_archive",
                local_path=str(out),
                url=url,
                expected_size_bytes=size,
                note=(
                    "Full harmonized brain DNA-methylation QTL summary "
                    "statistics. Preferred over GRCh37 GoDMC for strict "
                    "GRCh38 Step11."
                ),
            )
        )

        save_provenance(
            d,
            {
                "resource": "ADSP FunGen xQTL Atlas mQTL",
                "accession": "NG00184.v1",
                "genome_build": "GRCh38",
                "archive": target,
                "manifest_row": row,
                "purpose": "formal GWAS-mQTL colocalization",
            },
        )

    # ------------------------------------------------------------------
    # caQTL - Kumasaka GRCh38 re-analysis
    # ------------------------------------------------------------------
    if "caqtl" in requested:
        d = base / "caqtl" / "kumasaka_grch38"
        d.mkdir(parents=True, exist_ok=True)

        assoc = d / "QTD100018.all.tsv.gz"
        meta = d / "QTD100018_peak_metadata.tsv.gz"

        resources.append(
            Resource(
                resource_id="QTD100018",
                qtl_type="caQTL",
                source="Kumasaka_2018 GRCh38 re-analysis",
                build="GRCh38",
                tissue="LCL",
                role="dense_summary_statistics",
                local_path=str(assoc),
                url=KUMASAKA_CA_URL,
                md5=KUMASAKA_CA_MD5,
                expected_size_bytes=int(6.0 * 1000**3),
                note=(
                    "ATAC-seq chromatin-accessibility QTL summary statistics; "
                    "re-analysis aligned and mapped on GRCh38."
                ),
            )
        )

        resources.append(
            Resource(
                resource_id="QTD100018_PEAK_METADATA",
                qtl_type="caQTL",
                source="Kumasaka_2018 GRCh38 re-analysis",
                build="GRCh38",
                tissue="LCL",
                role="molecular_trait_metadata",
                local_path=str(meta),
                url=KUMASAKA_CA_META_URL,
                md5=KUMASAKA_CA_META_MD5,
                expected_size_bytes=int(6.1 * 1000**2),
            )
        )

        save_provenance(
            d,
            {
                "resource": "Kumasaka ATAC-seq caQTL GRCh38 re-analysis",
                "dataset_id": "QTD100018",
                "zenodo_record": "13848268",
                "genome_build": "GRCh38",
                "purpose": "formal GWAS-caQTL colocalization",
            },
        )

    # ------------------------------------------------------------------
    # Optional haQTL
    # ------------------------------------------------------------------
    if args.include_haqtl:
        target = "ADSP_FunGen_xQTL.v1.haQTL.all.tar"
        url, size, row = resolve_niagads_archive(niagads_rows, target)

        d = base / "haqtl" / "adsp_fungen_brain"
        d.mkdir(parents=True, exist_ok=True)
        out = d / target

        resources.append(
            Resource(
                resource_id="NG00184_v1_haQTL",
                qtl_type="haQTL",
                source="ADSP FunGen xQTL Atlas",
                build="GRCh38",
                tissue="brain_multi_region_multi_celltype",
                role="dense_summary_statistics_archive",
                local_path=str(out),
                url=url,
                expected_size_bytes=size,
                note="H3K9ac histone-acetylation QTL summary statistics.",
            )
        )

        save_provenance(
            d,
            {
                "resource": "ADSP FunGen xQTL Atlas haQTL",
                "accession": "NG00184.v1",
                "genome_build": "GRCh38",
                "archive": target,
                "manifest_row": row,
            },
        )

    # ------------------------------------------------------------------
    # Status + plan
    # ------------------------------------------------------------------
    for r in resources:
        p = Path(r.local_path)
        if is_complete(p):
            r.status = "COMPLETE"
        elif p.exists() and p.stat().st_size > 0:
            r.status = "PARTIAL"
        else:
            r.status = "MISSING"

    registry = base / "resource_registry.tsv"
    write_registry(registry, resources)

    banner("DOWNLOAD PLAN")

    estimated_missing = 0
    unknown_size = 0

    for r in resources:
        print(
            f"{r.qtl_type:<6} "
            f"{r.resource_id:<28} "
            f"{r.status:<9} "
            f"{human_bytes(r.expected_size_bytes):>12}  "
            f"{r.local_path}"
        )

        if r.status != "COMPLETE":
            if r.expected_size_bytes is not None:
                estimated_missing += r.expected_size_bytes
            else:
                unknown_size += 1

    free = free_space(base)
    reserve = int(args.reserve_gb * 1024**3)

    print()
    print(f"Estimated known bytes missing : {human_bytes(estimated_missing)}")
    print(f"Unknown-size resources        : {unknown_size}")
    print(f"Current free space            : {human_bytes(free)}")
    print(f"Requested reserve             : {human_bytes(reserve)}")
    print(f"Registry                      : {registry}")

    if args.plan:
        print()
        print("PLAN ONLY: no large resource files were downloaded.")
        return 0

    # This is intentionally conservative. Unknown-size NIAGADS archives are
    # allowed, but known sizes must fit while preserving reserve.
    if (
        estimated_missing > 0
        and free - estimated_missing < reserve
        and not args.force_low_space
    ):
        raise SystemExit(
            "\nERROR: insufficient free space for the known-size downloads "
            "while preserving the requested reserve.\n"
            f"Free: {human_bytes(free)}\n"
            f"Known missing: {human_bytes(estimated_missing)}\n"
            f"Reserve: {human_bytes(reserve)}\n\n"
            "Free space first, reduce --reserve-gb, or use "
            "--force-low-space only if you are certain."
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    banner("DOWNLOADING MOLECULAR-QTL RESOURCES")

    for idx, r in enumerate(resources, start=1):
        p = Path(r.local_path)

        print()
        print("-" * 100)
        print(f"[{idx}/{len(resources)}] {r.resource_id}")
        print(f"QTL type : {r.qtl_type}")
        print(f"Source   : {r.source}")
        print(f"Build    : {r.build}")
        print(f"Output   : {p}")
        print("-" * 100)

        if is_complete(p):
            print(f"[CACHE] {p}")
            r.status = "COMPLETE"
            continue

        before = free_space(base)
        print(f"Free before: {human_bytes(before)}")

        try:
            download_any(r.url, p)

            if not p.exists() or p.stat().st_size == 0:
                raise RuntimeError(f"Download produced an empty file: {p}")

            if r.md5:
                verify_md5(p, r.md5)

            mark_complete(p)
            r.status = "COMPLETE"

            print(f"[OK] {p}")
            print(f"Size: {human_bytes(p.stat().st_size)}")
            print(f"Free after: {human_bytes(free_space(base))}")

        except Exception as e:
            r.status = "FAILED"
            write_registry(registry, resources)
            print(f"[FAILED] {r.resource_id}: {e}", file=sys.stderr)
            raise

    write_registry(registry, resources)

    # ------------------------------------------------------------------
    # Optional extraction of NIAGADS tar archives
    # ------------------------------------------------------------------
    if args.extract:
        banner("EXTRACTING NIAGADS ARCHIVES")

        for r in resources:
            p = Path(r.local_path)
            if (
                r.status == "COMPLETE"
                and p.suffix == ".tar"
                and p.exists()
            ):
                extract_tar(p, p.parent / "extracted")

    banner("MOLECULAR-QTL SETUP COMPLETE")

    for r in resources:
        print(
            f"{r.qtl_type:<6} "
            f"{r.resource_id:<28} "
            f"{r.status:<10} "
            f"{r.local_path}"
        )

    print()
    print("Next pipeline step:")
    print(
        "  Update Step11 to consume resources/molecular_qtl/resource_registry.tsv "
        "and run formal colocalization against local resources."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
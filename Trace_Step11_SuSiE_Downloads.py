#!/usr/bin/env python3

from pathlib import Path

ROOT = Path.cwd()

MANIFEST = ROOT / "resources/coloc/eqtl_catalogue/download_provider_susie.aria2"
BASE = ROOT / "resources/coloc/eqtl_catalogue/susie_provider"
LOG = ROOT / "Step11_provider_SuSiE_download.log"

if not MANIFEST.exists():
    raise SystemExit(f"Missing manifest: {MANIFEST}")

# ============================================================
# PARSE ARIA2 MANIFEST
# ============================================================

lines = MANIFEST.read_text(errors="replace").splitlines()

expected = []
current_url = None
current_dir = None
current_out = None


def flush():
    global current_url, current_dir, current_out

    if current_url and current_dir and current_out:
        expected.append(
            (
                current_url,
                Path(current_dir) / current_out,
            )
        )

    current_url = None
    current_dir = None
    current_out = None


for line in lines:
    s = line.strip()

    if not s:
        flush()
        continue

    if s.startswith(("ftp://", "http://", "https://")):
        flush()
        current_url = s

    elif s.startswith("dir="):
        current_dir = s.split("=", 1)[1].strip()

    elif s.startswith("out="):
        current_out = s.split("=", 1)[1].strip()

flush()

# Deduplicate by final output path
dedup = {}

for url, path in expected:
    dedup[str(path)] = (url, path)

expected = list(dedup.values())

# ============================================================
# FILE STATUS
# ============================================================

complete = []
partial = []
missing = []

bytes_complete = 0
bytes_partial = 0

for url, path in expected:

    if path.exists() and path.stat().st_size > 0:
        size = path.stat().st_size
        complete.append((url, path, size))
        bytes_complete += size
        continue

    possible_partial = [
        Path(str(path) + ".aria2"),
        Path(str(path) + ".part"),
    ]

    found = None

    for p in possible_partial:
        if p.exists():
            found = p
            break

    if found is not None:
        size = found.stat().st_size
        partial.append((url, path, found, size))
        bytes_partial += size
    else:
        missing.append((url, path))

# ============================================================
# COUNTS BY RESOURCE TYPE
# ============================================================

def is_lbf(path):
    return "lbf_variable" in path.name


def is_cs(path):
    return "credible" in path.name.lower()


lbf_total = sum(1 for _, p in expected if is_lbf(p))
cs_total = sum(1 for _, p in expected if is_cs(p))

lbf_complete = sum(1 for _, p, _ in complete if is_lbf(p))
cs_complete = sum(1 for _, p, _ in complete if is_cs(p))

lbf_partial = sum(1 for _, p, _, _ in partial if is_lbf(p))
cs_partial = sum(1 for _, p, _, _ in partial if is_cs(p))

# ============================================================
# LOG ERRORS
# ============================================================

errors = []

if LOG.exists():
    try:
        log_lines = LOG.read_text(errors="replace").splitlines()

        for line in log_lines:
            if (
                "[ERROR]" in line
                or "Connection refused" in line
                or "Download aborted" in line
            ):
                errors.append(line.strip())

    except Exception:
        pass

# ============================================================
# DISPLAY
# ============================================================

total = len(expected)
done = len(complete)
running = len(partial)
waiting = len(missing)

pct = (done / total * 100.0) if total else 0.0

print()
print("=" * 100)
print("STEP11 PROVIDER SuSiE DOWNLOAD PROGRESS")
print("=" * 100)

print(f"Expected files        : {total}")
print(f"Completed             : {done}")
print(f"Partial / active      : {running}")
print(f"Missing / waiting     : {waiting}")
print(f"Files complete        : {pct:.2f}%")

print()
print("RESOURCE TYPES")
print("-" * 100)

print(
    f"LBF files             : "
    f"{lbf_complete}/{lbf_total} complete"
    f" | {lbf_partial} partial"
)

print(
    f"Credible-set files    : "
    f"{cs_complete}/{cs_total} complete"
    f" | {cs_partial} partial"
)

print()
print("STORAGE")
print("-" * 100)

print(f"Completed data        : {bytes_complete / 1024**3:.2f} GiB")
print(f"Partial data          : {bytes_partial / 1024**3:.2f} GiB")
print(
    f"Total currently disk  : "
    f"{(bytes_complete + bytes_partial) / 1024**3:.2f} GiB"
)

# ============================================================
# SHOW BIGGEST PARTIAL DOWNLOADS
# ============================================================

if partial:

    print()
    print("ACTIVE / PARTIAL DOWNLOADS")
    print("-" * 100)

    partial_sorted = sorted(
        partial,
        key=lambda x: x[3],
        reverse=True,
    )

    for _, final, part, size in partial_sorted[:20]:

        print(
            f"{size / 1024**3:8.2f} GiB   "
            f"{final.parent.name}/{final.name}"
        )

# ============================================================
# SHOW RECENT ERRORS
# ============================================================

if errors:

    print()
    print("LATEST DOWNLOAD ERRORS")
    print("-" * 100)

    for line in errors[-10:]:
        print(line)

print("=" * 100)
print()

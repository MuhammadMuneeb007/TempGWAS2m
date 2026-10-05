# GWAS2m

GWAS-to-mechanism pipeline: GWAS Catalog discovery → summary-statistic QC →
ancestry-matched LD clumping → SuSiE-RSS fine-mapping → VEP → GTEx eQTL/sQTL →
SpliceAI → Pangolin → formal colocalisation (Step11).

The pipeline is **phenotype- and ancestry-agnostic**. Nothing in the code is
specific to a disease, tissue, study or cluster: everything is derived from
`--phenotype`, `--ancestry`, the shared resources prepared by Step00 and
`config/slurm.yaml`.

---

## First-time workflow

```bash
git clone <repository> GWAS2m
cd GWAS2m                                   # the project root = current directory

# 1. Describe your cluster (no Python edits needed)
$EDITOR config/slurm.yaml

# 2. See what is installed / missing (READ-ONLY)
python Step00_Check_Resources.py --inspect

# 3. Install software and prepare shared resources (idempotent)
python Step00_Check_Resources.py

# 4. Check readiness for the ancestry you will analyse
python Step00_Check_Resources.py --inspect --ancestry EUR

# 5. Run Steps 01-10 for any phenotype
python Step01_10_Run.py --phenotype migraine --ancestry EUR

# 6. Step11 (formal colocalisation) separately, when Steps 01-10 have finished
python Step11_Provider_SuSiE_Formal.py --phenotype migraine --ancestry EUR
```

Everything is created beneath the current working directory
(`envs/`, `resources/`, `external_tools/`, `setup_manifests/`, `logs/`, ...);
no path in the Python source is machine-specific.

---

## Setup and shared resources (Step00)

`Step00_Check_Resources.py` is the single authority for software, packages and
**shared static resources**. The registry lives in `gwas2m_resources.py`.

```bash
# Check only (READ-ONLY: never installs, downloads, deletes, writes or submits)
python Step00_Check_Resources.py --inspect

# Check for a specific ancestry (other ancestries' panels are not required)
python Step00_Check_Resources.py --inspect --ancestry EUR

# Include Step11 colocalisation resources in the readiness check
python Step00_Check_Resources.py --inspect --ancestry EUR --through-step 11

# Machine-readable report (stdout); --output additionally writes a file
python Step00_Check_Resources.py --inspect --json
python Step00_Check_Resources.py --inspect --json --output resource_inspection.json

# Perform setup (skips anything already valid; never re-downloads)
python Step00_Check_Resources.py

# Install software only
python Step00_Check_Resources.py --install-only

# Prepare/download shared resources only
python Step00_Check_Resources.py --resources-only
```

`--status` is kept as an alias for `--inspect`. `--inspect` exits 0 when the
pipeline is ready and 1 otherwise, and ends with, e.g.

```text
PIPELINE READY FOR EUR STEPS01-10: NO

BLOCKING:
  - VEP_CACHE
  - python gwaslab
  - 1000G_EUR
```

Statuses: `AVAILABLE`, `MISSING`, `INCOMPLETE` (e.g. a `.part`/`.aria2` file,
an unextracted archive, 14/22 chromosomes), `INVALID`, `BROKEN`,
`VERSION_MISMATCH`, `OPTIONAL_MISSING`, `NOT_CONFIGURED`. Validation is cheap
(existence, size, index files, chromosome completeness, completion markers);
large files are never read.

**Shared resources controlled by Step00** (see `setup_manifests/resources.tsv`):
GRCh38 FASTA + index and GENCODE 50 GTF; 1000 Genomes GRCh38 (raw VCFs, panel,
ALL-sample PGEN, EUR/AFR/EAS/SAS/AMR PGEN panels); Ensembl VEP 116 cache;
GTEx v11 eQTL/sQTL significant pairs, variant lookup, GENCODE 47 and SuSiE
archives; Pangolin repository (pinned commit) and annotation database; GWAS
Catalog studies/ancestry tables; optional molecular-QTL and Open Targets 26.09.
eQTL Catalogue provider resources used by Step11 are registered and inspected
but still fetched by Step11 itself (see *Known limitations*).

Software: `envs/pipeline` (main stack, `environment.yml`), `envs/spliceai`
(SpliceAI 1.3.1, TensorFlow 2.15) and `envs/pangolin` (Pangolin, PyTorch) are
kept separate because their dependencies are incompatible; Step00 creates,
validates and reports all three. The full dependency list with install method,
environment and required/optional flag is written to
`setup_manifests/software_manifest.tsv`; versions actually present are recorded
in `setup_manifests/resource_versions.tsv` (Methods-section provenance).

**Phenotype-specific GWAS summary statistics are runtime inputs**, not shared
resources: Step01 selects them and Step02 downloads them per study
(`gwas2m_resources.download_file`, wget then curl, resumable).

---

## HPC / SLURM configuration

Every generated SBATCH script (Step00 setup jobs, the Step01-10 study array,
the legacy per-step planners and Step11) takes its settings from
**`config/slurm.yaml`**. Switching cluster means editing that file (or passing
another profile), never the Python source.

```yaml
scheduler: slurm

cluster:
  name: my_hpc

slurm:
  partition: compute        # null -> no --partition line (site default)
  account: my_project       # null -> no --account line
  qos: null
  reservation: null
  constraint: null

  default_time: "24:00:00"
  default_memory: "8G"
  default_cpus: 1

  max_parallel: 0           # 0 = no GWAS2m concurrency throttle
  array_limit: 1000         # max indices in ONE array (not concurrency)
  max_walltime: null        # null = GWAS2m imposes no walltime ceiling

  extra_sbatch_directives:
    - "--mail-type=FAIL"

stages:
  study:                    # Step01_10_Run.py: one task = one GWAS study
    time: "24:00:00"
    memory: "100G"
    cpus: 8
  qc:
    memory: "100G"
    cpus: 4
  # ... setup_*, download, ld, clumping, finemapping, vep, qtl, spliceai,
  #     pangolin, coloc_discovery, coloc_planner, coloc_extraction, coloc,
  #     aggregation, coloc_core, audit
```

* `max_parallel: 0` means **GWAS2m does not place a concurrency throttle on
  SLURM arrays** (`--array=1-100`, not `1-100%4`). The site's scheduler and
  account policies may still limit how many tasks run at once. Set a number
  (globally or per stage) to add `%N`.
* `array_limit` is the maximum number of indices in one array. Step01_10_Run
  splits larger study sets into several arrays. Step11's pre-submitted worker
  pools additionally cap each array at 1000 tasks (implementation bound).
* Precedence: command-line flag > config file > built-in generic fallback.
  A stage missing from the file falls back by prefix (`setup_genome` →
  `setup` → `slurm.default_*`).
* Alternate profiles: `--slurm-config config/my_cluster.yaml` (Step00,
  Step01_10_Run, Step11) or `export GWAS2M_SLURM_CONFIG=...`.
  `config/slurm.example.yaml` is a template; `config/uq.example.yaml` captures
  the values previously hard-coded for the UQ cluster (example only, not the
  default).
* Invalid values fail immediately with the exact key, e.g.
  `Invalid config/slurm.yaml: stages.qc.cpus must be >= 1 (got -2)`.
* `python Step00_Check_Resources.py --inspect` shows the scheduler commands
  found (`sbatch`, `squeue`, `sacct`, `sinfo`; works on non-SLURM machines),
  the parsed configuration and the effective request of every stage.

---

## Steps 01-10: one command per phenotype

```bash
python Step01_10_Run.py --phenotype "parkinson's disease" --ancestry EUR
python Step01_10_Run.py --phenotype migraine              --ancestry EUR
python Step01_10_Run.py --phenotype "multiple sclerosis"  --ancestry EUR
python Step01_10_Run.py --phenotype "coronary artery disease" --ancestry EUR
python Step01_10_Run.py --phenotype asthma                --ancestry AFR
```

The same code path runs every phenotype; only the data differ (number of
studies, loci, variants). Each phenotype × ancestry gets isolated outputs
(`<step dir>/<phenotype_slug>/<ancestry_slug>/`), logs and audit directory.

What one command does:

1. **Setup preflight** (read-only, same check as `--inspect --ancestry X`).
   If a required resource is missing it prints `SETUP PREFLIGHT: FAIL` with the
   missing items and exits before anything is submitted.
2. **Step01 discovery**: every phenotype-matched GWAS Catalog study is written
   to `study_selection.tsv` with its selection or exclusion reason.
3. `study_manifest.tsv`: one row per eligible study.
4. **One SLURM array, one task per study.** Each task runs Step02 → Step10 for
   its study by calling each step's existing `worker_mode()` on a one-row
   manifest. Studies never wait for each other; a failed study does not stop
   the others.
5. An **audit job** (`afterany`) builds the paper tables.

Useful options: `--dry-run` (print the generated SLURM scripts, submit
nothing), `--local` (run all studies here, no SLURM), `--study GCST...`
(repeatable), `--keep-raw`, `--alias` / `--max-studies` (passed to Step01),
`--refresh-discovery`, and `--partition/--time/--mem/--cpus/--max-parallel`
to override the config for one run.

Status at any time (read-only):

```bash
python Pipeline_reporter.py --phenotype migraine --ancestry EUR --status
```

```text
STUDY          S02    S03    S04    S05    S06    S07    S08    S09    S10
GCST90000001   OK     OK     OK     OK     OK     OK     OK     OK     OK
GCST90000002   OK     FAIL   --     --     --     --     --     --     --
GCST90000003   OK     OK     OK     OK     OK     N/A    N/A    N/A    N/A
GCST90000004   OK     OK     OK     OK     PART   --     --     --     --
```

followed by the exact reason for every FAIL / PART / EXCL / N/A / `--` cell.

### Resume logic

Re-running the same command is always safe.

* Before a stage runs, its outputs are validated with the same checks its
  worker uses (plus summary-JSON status, failure markers and, for Step07, the
  size/mtime of the Step06 input). Valid → `SKIPPED_ALREADY_COMPLETE`
  (`[SKIP] output already validated`); otherwise the stage is re-run.
* If a stage really re-ran, every downstream stage is forced to recompute so
  stale results are never reused.
* A failed or partial stage marks later stages `NOT_RUN_UPSTREAM_FAILED`; an
  excluded study or one without significant loci marks them `NOT_AVAILABLE`.
* **Per-locus fine-mapping resume**: each successful locus writes
  `06_finemapping/.../<GCST>/loci/<L>/LOCUS_COMPLETE.json` recording its
  coordinates, input-file signatures and SuSiE parameters. On a rerun,
  `L001 COMPLETE, L002 COMPLETE, L003 FAILED, L004 NOT RUN` becomes
  `L001 SKIP, L002 SKIP, L003 RUN, L004 RUN`; only pending loci are extracted
  and fine-mapped. Changed inputs or parameters invalidate the markers.
* A task killed by SLURM (timeout, OOM) leaves its stage `RUNNING`; the audit
  job converts it to `FAILED (TaskTerminated)`. Resubmitting resumes it.
* A per-study lock prevents two workers processing the same study.

### Raw GWAS cleanup safety

After Step03 QC the large raw download is deleted **only if all** of these
hold: the QC stage is COMPLETE; the QC file, significant-variant file, QC
summary TSV and QC summary JSON exist and are non-empty; the QC file is
re-read to EOF (gzip CRC verified) and its data-row count equals `N_AFTER_QC`
in the QC summary (and is > 0). Otherwise the raw file is kept and the reason
recorded. If QC failed, the raw file is always kept. `--keep-raw` disables
cleanup. URLs, manifests, metadata, QC summaries and audit records are never
deleted. Once QC is validated, Step02 is skipped even though the raw file is
gone (no re-download).

### Audit tables (`pipeline_audit/<phenotype>/<ancestry>/`)

| File | Content |
|---|---|
| `run_manifest.json` | command line, git commit, versions, thresholds, every submission (job IDs, arrays, SLURM request) |
| `effective_slurm_config.json` | configuration file and effective SLURM request actually used |
| `study_selection.tsv` | every phenotype-matched study: match score, ancestry, sumstats/harmonised availability, URL, rank, exclusion reason |
| `study_manifest.tsv` | eligible studies with array index |
| `study_stage_status.tsv` | one row per study × stage: status, reason, times, inputs/outputs, counts, errors, log, SLURM job/partition/account/time/memory/CPUs/array |
| `exclusion_reasons.tsv` | every non-complete study × stage with its exact reason |
| `study_summary.tsv` | N discovered / matched / ancestry-matched / with sumstats / harmonised / downloaded / QC'd / with loci / loci fine-mapped / VEP / QTL / SpliceAI / Pangolin / Step11-ready |
| `download_provenance.tsv` | URL, file, size, start/end, status, error, raw-deletion record |
| `qc_metrics.tsv` | the QC summary Step03 computes for each study |
| `locus_status.tsv` | one row per locus: coordinates, status, error |
| `software_versions.tsv` | software, packages, resources (path, size, mtime; no hashing of large files), thresholds |
| `step11_ready_studies.tsv` | studies with complete fine-mapping and SuSiE fits for every locus |
| `status/<GCST>/<STAGE>.json` | machine-readable status per study × stage (+ `history.jsonl`) |
| `logs/<GCST>/<STAGE>.log` | full output of every stage |

Example (`study_selection.tsv`, synthetic accessions):

```text
STUDY_ACCESSION MATCH_SCORE ANCESTRY_MATCH SUMMARY_STATS_AVAILABLE HARMONISED_FILE_AVAILABLE ELIGIBLE EXCLUSION_REASON           RANK
GCST90000001    100         True           True                    True                      True                                4
GCST90000005    100         True           False                   NOT_CHECKED               False    NO_FULL_SUMMARY_STATISTICS
GCST90000006    100         False          True                    NOT_CHECKED               False    ANCESTRY_MISMATCH
GCST90000008    100         True           True                    False                     False    NO_HARMONISED_FILE         1
```

---

## Step11

Step11 is not part of Step01_10_Run. Once `step11_ready_studies.tsv` lists the
studies you need:

```bash
python Step11_Provider_SuSiE_Formal.py --phenotype "<phenotype>" --ancestry <ANC>
```

It reads the per-study Step06 directories exactly as before; its SLURM
settings come from `config/slurm.yaml` (stages `coloc_*`, `aggregation`).

---

## Known limitations

* Step11 (`Step11_Colocalize_GTEx_SuSiE.py` / `Step11_Provider_SuSiE_Formal.py`)
  still contains its own downloader for GTEx SuSiE/apaQTL archives and eQTL
  Catalogue provider files. These resources are registered and inspected by
  Step00 but their acquisition has not been moved, to avoid touching Step11.
* Step01 still fetches the two small GWAS Catalog tables when they are absent
  (Step00 `--resources-only` now pre-fetches them).
* The legacy per-step planners (`StepNN_*.py` without `--index`) and the
  generated `StepNN_*_<phenotype>_<ancestry>.sh` scripts remain usable;
  previously generated scripts committed to the repository still contain the
  old absolute paths and partitions until regenerated.

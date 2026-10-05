#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWAS2m - central SLURM / HPC configuration.

The ONLY normal source of SBATCH settings is config/slurm.yaml (or the file
given by --slurm-config / the GWAS2M_SLURM_CONFIG environment variable).

Precedence:   command-line override  >  config file  >  built-in fallback

Rules:
  * null partition/account/qos/reservation/constraint -> no directive at all
    (SLURM then uses the site default).
  * max_parallel 0/null -> NO "%N" concurrency throttle on arrays.
  * array_limit = maximum indices in ONE array (not concurrency).
  * max_walltime null -> GWAS2m imposes no walltime ceiling.

Stage look-up falls back by prefix: "setup_genome" -> stages.setup_genome
-> stages.setup -> slurm.default_* -> built-in fallback.

Contains NO scientific logic.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

CONFIG_ENV_VAR = "GWAS2M_SLURM_CONFIG"
DEFAULT_CONFIG_RELATIVE = Path("config") / "slurm.yaml"

SITE_KEYS = ("partition", "account", "qos", "reservation", "constraint")
RESOURCE_KEYS = ("time", "memory", "cpus", "nodes", "ntasks", "max_parallel")

FALLBACK = {
    "scheduler": "slurm",
    "cluster": {"name": "generic_slurm"},
    "slurm": {
        "partition": None,
        "account": None,
        "qos": None,
        "reservation": None,
        "constraint": None,
        "default_time": "24:00:00",
        "default_memory": "8G",
        "default_cpus": 1,
        "default_nodes": 1,
        "default_ntasks": 1,
        "max_parallel": 0,
        "array_limit": 1000,
        "max_walltime": None,
        "extra_sbatch_directives": [],
    },
    "stages": {},
}


class SlurmConfigError(ValueError):
    pass


# =============================================================================
# YAML (PyYAML when installed; otherwise a strict parser for this simple subset)
# =============================================================================

def _scalar(text: str):
    text = text.strip()
    if text == "" or text in {"null", "Null", "NULL", "~", "None"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text == "[]":
        return []
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d*", text):
        return float(text)
    return text


def _strip_comment(line: str) -> str:
    quote = None
    for i, char in enumerate(line):
        if char in "\"'":
            quote = None if quote == char else (char if quote is None else quote)
        elif char == "#" and quote is None and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _parse_simple_yaml(text: str, source: str) -> dict:
    root: dict = {}
    stack = [(-1, root)]          # (indent, container)
    pending: tuple[int, dict, str] | None = None  # key waiting for a block value
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if "\t" in raw[:indent]:
            raise SlurmConfigError(f"{source}:{number}: tabs are not allowed for indentation")

        if pending is not None:
            p_indent, p_parent, p_key = pending
            if indent > p_indent:
                p_parent[p_key] = [] if content.startswith("- ") or content == "-" else {}
                stack.append((p_indent, p_parent[p_key]))
            pending = None

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise SlurmConfigError(f"{source}:{number}: bad indentation")
        container = stack[-1][1]

        if content.startswith("- ") or content == "-":
            if not isinstance(container, list):
                raise SlurmConfigError(f"{source}:{number}: list item outside a list")
            container.append(_scalar(content[1:]))
            continue
        if ":" not in content:
            raise SlurmConfigError(f"{source}:{number}: expected 'key: value'")
        key, _, value = content.partition(":")
        key = key.strip().strip("\"'")
        if not isinstance(container, dict):
            raise SlurmConfigError(f"{source}:{number}: mapping entry inside a list")
        if value.strip() == "":
            container[key] = None
            pending = (indent, container, key)
        else:
            container[key] = _scalar(value)
    return root


def read_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SlurmConfigError(f"Invalid YAML in {path}: {exc}") from exc
    except ImportError:
        data = _parse_simple_yaml(text, str(path))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SlurmConfigError(f"{path}: top level must be a mapping")
    return data


# =============================================================================
# LOAD + VALIDATE
# =============================================================================

def resolve_config_path(root: Path | None = None, explicit: str | Path | None = None) -> Path:
    root = Path(root) if root else Path.cwd().resolve()
    if explicit:
        path = Path(explicit)
    elif os.environ.get(CONFIG_ENV_VAR):
        path = Path(os.environ[CONFIG_ENV_VAR])
    else:
        path = DEFAULT_CONFIG_RELATIVE
    return path if path.is_absolute() else (root / path)


def _merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_walltime(value) -> int:
    """Seconds from HH:MM:SS, D-HH:MM:SS, or MM:SS / minutes as SLURM accepts."""
    text = str(value).strip()
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", text)
    if match:
        days, hours, minutes, seconds = (int(match.group(1) or 0), int(match.group(2)),
                                         int(match.group(3)), int(match.group(4)))
    elif re.fullmatch(r"\d+", text):
        days, hours, minutes, seconds = 0, 0, int(text), 0
    else:
        raise SlurmConfigError(f"walltime {value!r} must be HH:MM:SS or D-HH:MM:SS")
    if minutes >= 60 or seconds >= 60:
        raise SlurmConfigError(f"walltime {value!r}: minutes/seconds must be < 60")
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise SlurmConfigError(f"walltime {value!r} must be greater than zero")
    return total


def _check_memory(value, where: str, errors: list[str]) -> None:
    if value is None or not re.fullmatch(r"\d+(\.\d+)?[KMGT]?B?", str(value).strip(), flags=re.I):
        errors.append(f"{where} must be a SLURM memory value such as 8G or 16000M (got {value!r})")


def validate_config(config: dict, source: str = "config") -> list[str]:
    errors: list[str] = []
    scheduler = config.get("scheduler", "slurm")
    if scheduler not in {"slurm", "local"}:
        errors.append(f"scheduler must be 'slurm' or 'local' (got {scheduler!r})")
    slurm = config.get("slurm") or {}
    if not isinstance(slurm, dict):
        return ["slurm must be a mapping"]
    for key in SITE_KEYS:
        value = slurm.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip() or " " in value.strip()):
            errors.append(f"slurm.{key} must be null or a non-empty string without spaces (got {value!r})")
    for key in ("default_cpus", "default_nodes", "default_ntasks"):
        if not isinstance(slurm.get(key), int) or slurm.get(key) < 1:
            errors.append(f"slurm.{key} must be an integer >= 1 (got {slurm.get(key)!r})")
    _check_memory(slurm.get("default_memory"), "slurm.default_memory", errors)
    for key in ("default_time", "max_walltime"):
        if slurm.get(key) is not None:
            try:
                parse_walltime(slurm[key])
            except SlurmConfigError as exc:
                errors.append(f"slurm.{key}: {exc}")
    max_parallel = slurm.get("max_parallel")
    if max_parallel is not None and (not isinstance(max_parallel, int) or max_parallel < 0):
        errors.append(f"slurm.max_parallel must be null or an integer >= 0 (got {max_parallel!r})")
    if not isinstance(slurm.get("array_limit"), int) or slurm.get("array_limit") < 1:
        errors.append(f"slurm.array_limit must be an integer >= 1 (got {slurm.get('array_limit')!r})")
    extra = slurm.get("extra_sbatch_directives") or []
    if not isinstance(extra, list) or not all(isinstance(x, str) and x.strip() for x in extra):
        errors.append("slurm.extra_sbatch_directives must be a list of non-empty strings")

    stages = config.get("stages") or {}
    if not isinstance(stages, dict):
        errors.append("stages must be a mapping")
        stages = {}
    max_walltime = slurm.get("max_walltime")
    for name, spec in stages.items():
        if spec is None:
            continue
        if not isinstance(spec, dict):
            errors.append(f"stages.{name} must be a mapping")
            continue
        unknown = set(spec) - set(RESOURCE_KEYS) - set(SITE_KEYS)
        if unknown:
            errors.append(f"stages.{name}: unknown keys {sorted(unknown)}")
        if "cpus" in spec and (not isinstance(spec["cpus"], int) or spec["cpus"] < 1):
            errors.append(f"stages.{name}.cpus must be >= 1 (got {spec['cpus']!r})")
        for key in ("nodes", "ntasks"):
            if key in spec and (not isinstance(spec[key], int) or spec[key] < 1):
                errors.append(f"stages.{name}.{key} must be >= 1 (got {spec[key]!r})")
        if "memory" in spec:
            _check_memory(spec["memory"], f"stages.{name}.memory", errors)
        if "max_parallel" in spec and spec["max_parallel"] is not None and (
                not isinstance(spec["max_parallel"], int) or spec["max_parallel"] < 0):
            errors.append(f"stages.{name}.max_parallel must be null or >= 0")
        if "time" in spec:
            try:
                seconds = parse_walltime(spec["time"])
                if max_walltime and seconds > parse_walltime(max_walltime):
                    errors.append(f"stages.{name}.time {spec['time']} exceeds slurm.max_walltime {max_walltime}")
            except SlurmConfigError as exc:
                errors.append(f"stages.{name}.time: {exc}")
    return [f"Invalid {source}: {e}" for e in errors]


_CACHE: dict[str, dict] = {}


def load_slurm_config(root: Path | None = None, path: str | Path | None = None) -> dict:
    """Merged, validated configuration. Raises SlurmConfigError if invalid."""
    config_path = resolve_config_path(root, path)
    key = str(config_path)
    if key in _CACHE:
        return _CACHE[key]
    if config_path.exists():
        user = read_yaml(config_path)
        source = str(config_path)
    elif path or os.environ.get(CONFIG_ENV_VAR):
        raise SlurmConfigError(f"SLURM config file not found: {config_path}")
    else:
        user, source = {}, "built-in generic fallback (config/slurm.yaml not found)"
    config = _merge(FALLBACK, user)
    errors = validate_config(config, str(config_path) if config_path.exists() else "built-in config")
    if errors:
        raise SlurmConfigError("\n".join(errors))
    config["_source"] = source
    config["_path"] = str(config_path)
    _CACHE[key] = config
    return config


# =============================================================================
# STAGE RESOURCES + SBATCH DIRECTIVES
# =============================================================================

def _stage_chain(stage: str) -> list[str]:
    parts = stage.split("_")
    return ["_".join(parts[:i]) for i in range(len(parts), 0, -1)]


def get_stage_resources(stage: str, config: dict | None = None, overrides: dict | None = None) -> dict:
    """Effective settings for one stage (CLI overrides > stage > slurm defaults)."""
    config = config or load_slurm_config()
    slurm = config["slurm"]
    stages = config.get("stages") or {}
    resources = {
        "stage": stage,
        "time": slurm["default_time"],
        "memory": slurm["default_memory"],
        "cpus": slurm["default_cpus"],
        "nodes": slurm["default_nodes"],
        "ntasks": slurm["default_ntasks"],
        "max_parallel": slurm.get("max_parallel") or 0,
        "array_limit": slurm["array_limit"],
        "max_walltime": slurm.get("max_walltime"),
        "extra_sbatch_directives": list(slurm.get("extra_sbatch_directives") or []),
        **{key: slurm.get(key) for key in SITE_KEYS},
        "config_stage": None,
    }
    for name in reversed(_stage_chain(stage)):
        spec = stages.get(name)
        if isinstance(spec, dict):
            resources.update({k: v for k, v in spec.items() if k in RESOURCE_KEYS + SITE_KEYS})
            resources["config_stage"] = name
    for key, value in (overrides or {}).items():
        if value is not None:
            resources[key] = value
    resources["max_parallel"] = int(resources["max_parallel"] or 0)
    validate_walltime(resources["time"], config)
    return resources


def validate_walltime(value, config: dict | None = None) -> str:
    """Central walltime check: only the configured max_walltime (if any) applies."""
    seconds = parse_walltime(value)
    config = config or load_slurm_config()
    ceiling = config["slurm"].get("max_walltime")
    if ceiling and seconds > parse_walltime(ceiling):
        raise SlurmConfigError(
            f"Requested walltime {value} exceeds slurm.max_walltime {ceiling} in {config.get('_path')}"
        )
    return str(value)


def array_spec(tasks, max_parallel=0) -> str:
    """'1-N' or explicit ids; '%K' only when max_parallel > 0."""
    if isinstance(tasks, int):
        ids = list(range(1, tasks + 1))
    else:
        ids = sorted(int(x) for x in tasks)
    if not ids:
        raise SlurmConfigError("array has no tasks")
    if ids == list(range(ids[0], ids[-1] + 1)):
        spec = f"{ids[0]}-{ids[-1]}" if len(ids) > 1 else str(ids[0])
    else:
        spec = ",".join(str(x) for x in ids)
    return f"{spec}%{int(max_parallel)}" if max_parallel and int(max_parallel) > 0 else spec


def site_directives(resources: dict) -> list[str]:
    """partition/account/qos/reservation/constraint (only when set) + extras."""
    lines = [f"#SBATCH --{key}={resources[key]}" for key in SITE_KEYS if resources.get(key)]
    for extra in resources.get("extra_sbatch_directives") or []:
        extra = extra.strip()
        lines.append(f"#SBATCH {extra if extra.startswith('-') else '--' + extra}")
    return lines


def build_sbatch_directives(
    stage: str,
    *,
    job_name: str | None = None,
    output: str | Path | None = None,
    error: str | Path | None = None,
    array: str | None = None,
    config: dict | None = None,
    overrides: dict | None = None,
    include_resources: bool = True,
) -> str:
    """The full '#SBATCH' header block for one stage."""
    r = get_stage_resources(stage, config, overrides)
    lines = []
    if job_name:
        lines.append(f"#SBATCH --job-name={str(job_name)[:100]}")
    if include_resources:
        lines += [f"#SBATCH --nodes={r['nodes']}", f"#SBATCH --ntasks={r['ntasks']}"]
    lines += site_directives(r)
    if include_resources:
        lines += [f"#SBATCH --time={r['time']}", f"#SBATCH --mem={r['memory']}",
                  f"#SBATCH --cpus-per-task={r['cpus']}"]
    if array:
        lines.append(f"#SBATCH --array={array}")
    if output:
        lines.append(f"#SBATCH --output={output}")
    if error:
        lines.append(f"#SBATCH --error={error}")
    return "\n".join(lines)


def apply_stage_defaults(args, stage: str, mapping: dict | None = None, config: dict | None = None):
    """Fill argparse attributes left as None from the config (CLI wins).

    mapping: argparse attribute -> resource key, default
             {"partition": "partition", "time": "time", "memory": "memory",
              "cpus": "cpus", "max_parallel": "max_parallel"}.
    Also attaches args.slurm_stage and args.slurm_resources.
    """
    mapping = mapping or {"partition": "partition", "time": "time", "memory": "memory",
                          "cpus": "cpus", "max_parallel": "max_parallel"}
    overrides = {}
    for attribute, key in mapping.items():
        if hasattr(args, attribute) and getattr(args, attribute) is not None and key in RESOURCE_KEYS + SITE_KEYS:
            overrides[key] = getattr(args, attribute)
    resources = get_stage_resources(stage, config, overrides)
    for attribute, key in mapping.items():
        if hasattr(args, attribute) and getattr(args, attribute) is None:
            setattr(args, attribute, resources[key])
    args.slurm_stage = stage
    args.slurm_resources = resources
    return args


def sbatch_header_from_args(args, *, job_name=None, output=None, error=None, array=None) -> str:
    """Header for planners whose argparse already holds the effective values."""
    r = dict(getattr(args, "slurm_resources", None) or get_stage_resources(getattr(args, "slurm_stage", "default")))
    for attribute, key in (("partition", "partition"), ("time", "time"), ("memory", "memory"), ("cpus", "cpus")):
        if getattr(args, attribute, None) is not None:
            r[key] = getattr(args, attribute)
    lines = []
    if job_name:
        lines.append(f"#SBATCH --job-name={str(job_name)[:100]}")
    lines += [f"#SBATCH --nodes={r['nodes']}"]
    lines += site_directives(r)
    lines += [f"#SBATCH --time={r['time']}"]
    if output:
        lines.append(f"#SBATCH --output={output}")
    if error:
        lines.append(f"#SBATCH --error={error}")
    if array:
        lines.append(f"#SBATCH --array={array}")
    lines += [f"#SBATCH --mem={r['memory']}", f"#SBATCH --cpus-per-task={r['cpus']}",
              f"#SBATCH --ntasks={r['ntasks']}"]
    return "\n".join(lines)


def sbatch_header_for_stage(stage: str, **kwargs) -> str:
    """Same header layout as sbatch_header_from_args, for scripts without argparse."""
    import types
    return sbatch_header_from_args(types.SimpleNamespace(slurm_stage=stage), **kwargs)


# =============================================================================
# REPORTING
# =============================================================================

def scheduler_commands() -> dict:
    import shutil
    return {name: shutil.which(name) for name in ("sbatch", "squeue", "sacct", "sinfo")}


def effective_config_record(config: dict, stages: list[str]) -> dict:
    """Secret-free record of the configuration actually used (for the audit)."""
    return {
        "config_file": config.get("_path"),
        "config_source": config.get("_source"),
        "scheduler": config.get("scheduler"),
        "cluster_name": (config.get("cluster") or {}).get("name"),
        "slurm": {k: v for k, v in config["slurm"].items() if k != "extra_sbatch_directives"},
        "extra_sbatch_directives": config["slurm"].get("extra_sbatch_directives") or [],
        "stages": {stage: get_stage_resources(stage, config) for stage in stages},
    }


def describe(value, empty="scheduler default") -> str:
    return empty if value in (None, "", []) else str(value)


if __name__ == "__main__":
    print(json.dumps(effective_config_record(load_slurm_config(), ["default"]), indent=2, default=str))

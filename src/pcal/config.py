"""Configuration loading.

One rule: src/ never hard-codes a number that appears in configs/. Anything
loaded here is what the pre-registration froze, and the only legitimate way to
change a value at run time is an explicit override recorded in the run record.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """The repository root, found from this file rather than the cwd."""
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return repo_root() / "configs"


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        candidate = config_dir() / p.name
        p = p if p.exists() else candidate
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_policies(path: str | Path | None = None) -> dict:
    return load_yaml(path or config_dir() / "policies.yaml")


def load_datasets(path: str | Path | None = None) -> dict:
    return load_yaml(path or config_dir() / "datasets.yaml")


def get_dotted(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Return a copy of cfg with dotted-key overrides applied.

    Overrides exist for two legitimate reasons only: the smoke configuration,
    which is explicitly not a scientific result, and the per-arm ablation
    switches in configs/policies.yaml (arms 12 and 13). Every applied override
    is echoed into the run record so a result can never silently depend on one.
    """
    out = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        set_dotted(out, key, value)
    return out


def arm_config(policies: dict, arm_id: int) -> dict:
    """Resolve one arm into a flat, fully-specified description."""
    arms = policies["arms"]
    if arm_id not in arms:
        raise KeyError(f"arm {arm_id} is not defined in configs/policies.yaml")
    arm = copy.deepcopy(arms[arm_id])
    arm["id"] = arm_id
    cfg = apply_overrides(policies, None)
    for key, value in (arm.get("overrides") or {}).items():
        # Arm-level overrides are written against the estimator block, which is
        # where both pre-registered ablations live.
        set_dotted(cfg, key if "." in key else f"estimator.{key}", value)
    arm["resolved"] = cfg
    return arm

"""Cache and artifact persistence."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, PROBES_DIR

log = logging.getLogger(__name__)


def _run_cache_path(config_slug: str, scenario_id: str, run_number: int) -> Path:
    return CACHE_DIR / config_slug / scenario_id / f"run_{run_number}.json"


def _probe_cache_path(config_slug: str) -> Path:
    return PROBES_DIR / config_slug / "probe.json"


def load_run_record(
    config_slug: str, scenario_id: str, run_number: int
) -> dict[str, Any] | None:
    path = _run_cache_path(config_slug, scenario_id, run_number)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_run_record(record: dict[str, Any]) -> Path:
    metadata = record.get("metadata", {})
    path = _run_cache_path(
        metadata["config_slug"],
        record["scenario_id"],
        int(record["run_number"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip transient in-memory flag — it has no meaning in a persisted file.
    to_save = {
        **record,
        "metadata": {k: v for k, v in metadata.items() if k != "from_cache"},
    }
    path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_probe_record(config_slug: str, record: dict[str, Any]) -> Path:
    path = _probe_cache_path(config_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_probe_record(config_slug: str) -> dict[str, Any] | None:
    path = _probe_cache_path(config_slug)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_cached_runs(config_slug: str, scenario_id: str) -> list[int]:
    scenario_dir = CACHE_DIR / config_slug / scenario_id
    if not scenario_dir.exists():
        return []
    runs: list[int] = []
    for path in sorted(scenario_dir.glob("run_*.json")):
        match = re.match(r"^run_(\d+)\.json$", path.name)
        if match:
            runs.append(int(match.group(1)))
    return runs


def list_cached_configs() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(path.name for path in CACHE_DIR.iterdir() if path.is_dir())


def list_cached_scenarios(config_slug: str) -> list[str]:
    config_dir = CACHE_DIR / config_slug
    if not config_dir.exists():
        return []
    return sorted(
        path.name
        for path in config_dir.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )


def iter_run_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for config_slug in list_cached_configs():
        for scenario_id in list_cached_scenarios(config_slug):
            for run_number in list_cached_runs(config_slug, scenario_id):
                record = load_run_record(config_slug, scenario_id, run_number)
                if not record:
                    continue
                provider = record.get("provider") or record.get("metadata", {}).get(
                    "provider"
                )
                if not provider:
                    log.warning(
                        "Skipping orphan cache record without provider: %s/%s/run_%d",
                        config_slug,
                        scenario_id,
                        run_number,
                    )
                    continue
                records.append(record)
    return records

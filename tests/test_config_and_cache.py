from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src import cache, config, cost_tracker


def test_load_model_registry_and_model_config_properties(tmp_path: Path) -> None:
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(
        "models:\n"
        "  - model_id: google/gemma-4-31b-it:free\n"
        "    temperature: 1.2\n"
        "    reasoning_effort: minimal\n"
        "    provider: akashml/bf16\n",
        encoding="utf-8",
    )

    configs = config.load_model_registry(registry_path)

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.reasoning_requested == "minimal"
    assert cfg.effective_temperature == 1.2
    assert cfg.config_slug == "google--gemma-4-31b-it:free+akashml-bf16@minimal-t1.2"
    assert cfg.label == "gemma-4-31b-it:free+akashml-bf16@minimal-t1.2"


def test_load_model_registry_rejects_duplicate_labels(tmp_path: Path) -> None:
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(
        "models:\n"
        "  - model_id: a/model\n"
        "  - model_id: a/model\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        config.load_model_registry(registry_path)


def test_load_api_key_success_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

    assert config.load_api_key() == "test-key"

    env_path.write_text("OPENROUTER_API_KEY=your-openrouter-api-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config.load_api_key()


def test_cache_round_trip_and_listings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    record = {
        "scenario_id": "plain_chat_history",
        "run_number": 2,
        "model_id": "google/gemma-4-31b-it:free",
        "display_label": "gemma",
        "provider": None,
        "metadata": {"config_slug": "gemma@minimal-t1.2"},
        "turn1": {"visible_reply": "I have a number."},
        "turn2": {"visible_reply": "37"},
        "evaluation": {"outcome_label": "thought_preserved"},
    }

    cache.save_run_record(record)

    loaded = cache.load_run_record("gemma@minimal-t1.2", "plain_chat_history", 2)
    assert loaded == record
    assert cache.list_cached_runs("gemma@minimal-t1.2", "plain_chat_history") == [2]
    assert cache.list_cached_configs() == ["gemma@minimal-t1.2"]
    assert cache.list_cached_scenarios("gemma@minimal-t1.2") == ["plain_chat_history"]
    assert cache.iter_run_records() == [record]


def test_probe_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "PROBES_DIR", tmp_path)
    record = {"reasoning_visibility": "plaintext"}
    cache.save_probe_record("grok@minimal-t1.2", record)

    assert cache.load_probe_record("grok@minimal-t1.2") == record


def test_cost_tracker_round_trip(tmp_path: Path) -> None:
    log_path = tmp_path / "cost_log.json"
    task = cost_tracker.TaskCost(label="run")
    task.add(prompt_tokens=10, completion_tokens=20, cost_usd=0.12, elapsed_seconds=1.5)
    session = cost_tracker.SessionCost(tasks=[task])

    cost_tracker.save_session_to_cost_log(session, path=log_path)

    assert cost_tracker.load_lifetime_cost(log_path) == pytest.approx(0.12)
    saved = json.loads(log_path.read_text(encoding="utf-8"))
    assert saved["last_session"]["tasks"][0]["label"] == "run"


def test_load_model_registry_warns_active_without_provider(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(
        "models:\n"
        "  - model_id: vendor/model-a\n"
        "    active: true\n"
        "  - model_id: vendor/model-b\n"
        "    active: true\n"
        "    provider: SomeProvider\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        configs = config.load_model_registry(registry_path)

    assert len(configs) == 2
    assert any("vendor/model-a" in msg and "no provider" in msg for msg in caplog.messages)
    assert not any("vendor/model-b" in msg for msg in caplog.messages)
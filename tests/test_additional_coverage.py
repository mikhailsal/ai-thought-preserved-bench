from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import cache, cli, config, leaderboard, model_probe, scenarios
from src.evaluator import JudgeResult, evaluate_run_record, judge_turn2_reply, reconcile_stability_group
from src.openrouter_client import (
    CompletionResult,
    OpenRouterClient,
    UsageInfo,
    _extract_tool_message,
    _to_plain_object,
    _usage_from_response,
)


def _challenge() -> dict:
    return {
        "range_low": 196,
        "range_high": 5342,
        "numbers": [1000, 2000, 3000],
        "expected_sum": 6000,
    }


def test_cache_invalid_json_and_missing_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    bad_run = tmp_path / "cfg" / "plain" / "run_1.json"
    bad_run.parent.mkdir(parents=True)
    bad_run.write_text("{broken", encoding="utf-8")
    bad_probe = tmp_path / "cfg" / "probe.json"
    bad_probe.parent.mkdir(parents=True, exist_ok=True)
    bad_probe.write_text("{broken", encoding="utf-8")

    assert cache.load_run_record("cfg", "plain", 1) is None
    assert cache.load_probe_record("cfg") is None
    assert cache.list_cached_runs("missing", "plain") == []
    assert cache.list_cached_scenarios("missing") == []


def test_cli_parse_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    model_a = config.ModelConfig(model_id="a/model", display_label="a")
    model_b = config.ModelConfig(model_id="b/model", display_label="b")
    monkeypatch.setattr(cli, "get_active_model_configs", lambda: [model_a])
    monkeypatch.setattr(cli, "get_model_config", lambda label: {"x": model_a, "y": model_b}.get(label, model_b))
    monkeypatch.setattr(cli, "list_registered_labels_for_model", lambda model_id: ["x", "y"] if model_id == "shared" else [])

    assert cli._parse_models(None) == [model_a]
    parsed = cli._parse_models("shared,b/model")
    assert parsed == [model_a, model_b, model_b]
    assert cli._parse_scenarios(None) == ["plain_chat_history", "tool_mediated_reply"]

    with pytest.raises(SystemExit):
        cli._parse_scenarios("bad")


def test_config_helper_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    config.ensure_dirs()
    assert config.CACHE_DIR.exists()
    assert config.RESULTS_DIR.exists()
    assert config.slug_to_model_id("openai--gpt") == "openai/gpt"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("models: [", encoding="utf-8")
    with pytest.raises(RuntimeError):
        config.load_model_registry(bad_yaml)

    wrong_shape = tmp_path / "wrong.yaml"
    wrong_shape.write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        config.load_model_registry(wrong_shape)

    cfg1 = config.ModelConfig(model_id="shared/model", display_label="one")
    cfg2 = config.ModelConfig(model_id="shared/model", display_label="two", active=False)
    monkeypatch.setattr(config, "MODEL_CONFIGS", {"one": cfg1, "two": cfg2})
    assert config.get_model_config("one") == cfg1
    with pytest.raises(RuntimeError):
        config.get_model_config("shared/model")
    assert config.get_active_model_configs() == [cfg1]
    assert config.get_config_by_slug(cfg1.config_slug) == cfg1
    assert config.list_registered_labels_for_model("shared/model") == ["one", "two"]
    with pytest.raises(SystemExit):
        config.fail("stop")


@dataclass
class Dumpable:
    value: int

    def model_dump(self) -> dict[str, int]:
        return {"value": self.value}


def test_openrouter_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _extract_tool_message("no message here") == ""
    assert _extract_tool_message('{"message":"hi\\"}').startswith("hi")
    assert _to_plain_object(Dumpable(3)) == {"value": 3}
    assert _to_plain_object(SimpleNamespace(a=1, _b=2)) == {"a": 1}

    class NoUsageResponse:
        usage = None

    assert _usage_from_response(response=NoUsageResponse(), elapsed=1.0) == UsageInfo(elapsed_seconds=1.0)

    usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, cost="not-a-number")
    response = SimpleNamespace(usage=usage)
    priced = _usage_from_response(
        response=response,
        elapsed=1.0,
    )
    assert priced.cost_usd == 0.0


def test_openrouter_chat_additional_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class DummyCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content=[{"type": "text", "text": " 6000 "}], tool_calls=None, reasoning=None, reasoning_content=None, reasoning_details=None)
            choice = SimpleNamespace(message=message, finish_reason=None)
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=0.5)
            return SimpleNamespace(choices=[choice], usage=usage, model="m")

    client = OpenRouterClient("key")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    result = client.chat(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=9,
        temperature=1.0,
        reasoning_effort="high",
        tool_choice="required",
    )

    assert captured["tool_choice"] == "required"
    assert result.visible_output == "6000"

    class FatalError(RuntimeError):
        status_code = 400

    class ErrorCompletions:
        def create(self, **kwargs):
            raise FatalError("bad")

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=ErrorCompletions()))
    with pytest.raises(FatalError):
        client.chat(model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=1.0)


def test_evaluator_additional_branches() -> None:
    record = {
        "challenge": _challenge(),
        "turn1": {"visible_reply": "Done.", "reasoning_content": None, "reasoning_details": None},
        "turn2": {"visible_reply": "unclear"},
    }
    judge = JudgeResult(
        outcome_label="hallucinated_memory",
        extracted_number=None,
        explanation="fallback judge",
        raw_response="{}",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.1, "elapsed_seconds": 0.1},
    )
    evaluated = evaluate_run_record(record, judge)
    untouched = [{"evaluation": {"pending_stability_check": False, "excluded_from_scoring": False}}]

    assert evaluated["outcome_label"] == "hallucinated_memory"
    assert reconcile_stability_group(untouched) == untouched

    no_judge = evaluate_run_record(record)
    assert no_judge["outcome_label"] == "other_refusal"

    class RegexJudgeClient:
        def chat(self, **_: object) -> CompletionResult:
            return CompletionResult(
                content='prefix {"outcome_label":"invalid","extracted_number":"x","explanation":"fallback"} suffix',
                visible_output='prefix {"outcome_label":"invalid","extracted_number":"x","explanation":"fallback"} suffix',
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, cost_usd=0.1, elapsed_seconds=0.1),
            )

    judged = judge_turn2_reply(RegexJudgeClient(), "The sum is 6000")
    assert judged.outcome_label == "other_refusal"
    assert judged.extracted_number == 6000


def test_leaderboard_and_probe_edge_cases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    leaderboard.display_leaderboard([])
    assert "No benchmark runs are available yet" in leaderboard.generate_markdown_report([])

    readme_without_markers = tmp_path / "README-no-markers.md"
    readme_without_markers.write_text("# Title\n", encoding="utf-8")
    leaderboard.update_readme_snapshot([], readme_path=readme_without_markers)
    assert readme_without_markers.read_text(encoding="utf-8") == "# Title\n"

    missing_readme = tmp_path / "missing.md"
    leaderboard.update_readme_snapshot([], readme_path=missing_readme)

    monkeypatch.setattr(model_probe, "load_probe_record", lambda _slug: {"cached": True})
    assert model_probe.probe_model(SimpleNamespace(), config.ModelConfig(model_id="m"), force=False) == {"cached": True}


def test_scenarios_helpers() -> None:
    assert scenarios.get_scenario("plain_chat_history").display_name == "Plain Chat History"
    assert [scenario.scenario_id for scenario in scenarios.get_scenarios()] == ["plain_chat_history", "tool_mediated_reply"]
    assert [scenario.scenario_id for scenario in scenarios.get_scenarios(["tool_mediated_reply"])] == ["tool_mediated_reply"]

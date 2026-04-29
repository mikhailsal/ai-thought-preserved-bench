from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import (
    cache,
    cli,
    config,
    cost_tracker,
    leaderboard,
    model_probe,
    prompt_builder,
    scenarios,
)
from src.evaluator import (
    JudgeResult,
    evaluate_run_record,
    judge_turn2_reply,
    reconcile_stability_group,
)
from src.openrouter_client import (
    CompletionResult,
    OpenRouterClient,
    UsageInfo,
    _extract_tool_message,
    _to_plain_object,
    _usage_from_response,
)
from src.scorer import ScenarioSummary


def _challenge() -> dict:
    return {
        "range_low": 196,
        "range_high": 5342,
    }


def test_cache_invalid_json_and_missing_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    probes_dir = tmp_path / "probes"
    monkeypatch.setattr(cache, "PROBES_DIR", probes_dir)
    bad_run = tmp_path / "cfg" / "plain" / "run_1.json"
    bad_run.parent.mkdir(parents=True)
    bad_run.write_text("{broken", encoding="utf-8")
    bad_probe = probes_dir / "cfg" / "probe.json"
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
    monkeypatch.setattr(
        cli,
        "get_model_config",
        lambda label: {"x": model_a, "y": model_b}.get(label, model_b),
    )
    monkeypatch.setattr(
        cli,
        "list_registered_labels_for_model",
        lambda model_id: ["x", "y"] if model_id == "shared" else [],
    )

    assert cli._parse_models(None) == [model_a]
    parsed = cli._parse_models("shared,b/model")
    assert parsed == [model_a, model_b, model_b]
    assert cli._parse_scenarios(None) == ["plain_chat_history", "tool_mediated_reply"]

    with pytest.raises(SystemExit):
        cli._parse_scenarios("bad")


def test_config_helper_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    cfg2 = config.ModelConfig(
        model_id="shared/model", display_label="two", active=False
    )
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

    assert _usage_from_response(response=NoUsageResponse(), elapsed=1.0) == UsageInfo(
        elapsed_seconds=1.0
    )

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
            message = SimpleNamespace(
                content=[{"type": "text", "text": " 6000 "}],
                tool_calls=None,
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            )
            choice = SimpleNamespace(message=message, finish_reason=None)
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=0.5)
            return SimpleNamespace(choices=[choice], usage=usage, model="m")

    client = OpenRouterClient("key")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=DummyCompletions())
    )

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

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=ErrorCompletions())
    )
    with pytest.raises(FatalError):
        client.chat(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=1.0,
        )


def test_evaluator_additional_branches() -> None:
    record = {
        "challenge": _challenge(),
        "turn1": {
            "visible_reply": "Done.",
            "reasoning_content": None,
            "reasoning_details": None,
        },
        "turn2": {"visible_reply": "unclear"},
    }
    judge = JudgeResult(
        outcome_label="hallucinated_memory",
        extracted_number=None,
        explanation="fallback judge",
        raw_response="{}",
        usage={
            "prompt_tokens": 350,
            "completion_tokens": 60,
            "cost_usd": 0.1,
            "elapsed_seconds": 0.5,
        },
    )
    evaluated = evaluate_run_record(record, judge)
    untouched = [
        {
            "evaluation": {
                "pending_stability_check": False,
                "excluded_from_scoring": False,
            }
        }
    ]

    assert evaluated["outcome_label"] == "hallucinated_memory"
    assert reconcile_stability_group(untouched) == untouched

    no_judge = evaluate_run_record(record)
    assert no_judge["outcome_label"] == "other_refusal"

    class RegexJudgeClient:
        def chat(self, **_: object) -> CompletionResult:
            return CompletionResult(
                content='prefix {"outcome_label":"invalid","extracted_number":"x","explanation":"fallback"} suffix',
                visible_output='prefix {"outcome_label":"invalid","extracted_number":"x","explanation":"fallback"} suffix',
                usage=UsageInfo(
                    prompt_tokens=1,
                    completion_tokens=1,
                    cost_usd=0.1,
                    elapsed_seconds=0.1,
                ),
            )

    judged = judge_turn2_reply(RegexJudgeClient(), "The sum is 6000")
    assert judged.outcome_label == "other_refusal"
    assert judged.extracted_number == 6000


def test_leaderboard_and_probe_edge_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    leaderboard.display_leaderboard([])
    assert (
        "No benchmark runs are available yet"
        in leaderboard.generate_markdown_report([])
    )

    readme_without_markers = tmp_path / "README-no-markers.md"
    readme_without_markers.write_text("# Title\n", encoding="utf-8")


def test_helper_edge_paths_for_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invalid_cost = tmp_path / "cost.json"
    invalid_cost.write_text("{broken", encoding="utf-8")
    assert cost_tracker.load_lifetime_cost(invalid_cost) == 0.0

    session = cost_tracker.SessionCost(
        tasks=[
            cost_tracker.TaskCost(
                label="t",
                prompt_tokens=10,
                completion_tokens=5,
                reasoning_tokens=3,
                cost_usd=1.25,
                elapsed_seconds=2.0,
            )
        ]
    )
    cost_log = tmp_path / "nested" / "cost-log.json"
    cost_tracker.save_session_to_cost_log(session, cost_log)
    assert cost_tracker.load_lifetime_cost(cost_log) == 1.25

    with pytest.raises(ValueError):
        prompt_builder.get_first_tool_call_id({"tool_calls": [{}]})

    assert cache.load_probe_record("missing-probe") is None

    summaries = [
        ScenarioSummary(
            config_slug="cfg",
            model_id="google/gemma-4-31b-it:free",
            display_label="gemma-4-31b-it:free+Google AI Studio@minimal-t1.2",
            provider="Google AI Studio",
            scenario_id="plain_chat_history",
            total_runs=1,
            protocol_failures=0,
            thought_preserved=1,
            hallucinated_memory=0,
            deliberate_fabrication=0,
            honest_no_memory=0,
            other_refusal=0,
            preservation_rate=1.0,
            hallucination_rate=0.0,
            fabrication_rate=0.0,
            honesty_rate=0.0,
            other_refusal_rate=0.0,
            protocol_failure_rate=0.0,
            thought_continuity_score=100.0,
            reasoning_visibility_counts={"plaintext": 1},
            stability_score=None,
            visible_reasoning_match_rate=1.0,
            reasoning_effort="minimal",
            reasoning_type="open",
            tpb_index=100.0,
        )
    ]
    leaderboard.display_leaderboard(summaries, session=session)
    monkeypatch.setattr(leaderboard, "RESULTS_DIR", tmp_path / "results")
    results_path = leaderboard.export_results_json(summaries, session=session)
    exported = results_path.read_text(encoding="utf-8")
    assert "session_cost" in exported
    assert "gemma-4-31b-it :free" in leaderboard.generate_markdown_report(summaries)

    empty_readme = tmp_path / "README.md"
    empty_readme.write_text(
        "# Title\n\n<!-- leaderboard:start -->\nold\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )
    leaderboard.update_readme_snapshot([], readme_path=empty_readme)
    assert "No benchmark runs available yet." in empty_readme.read_text(
        encoding="utf-8"
    )

    readme_without_markers = tmp_path / "README-no-markers.md"
    readme_without_markers.write_text("# Title\n", encoding="utf-8")
    leaderboard.update_readme_snapshot([], readme_path=readme_without_markers)
    assert readme_without_markers.read_text(encoding="utf-8") == "# Title\n"

    missing_readme = tmp_path / "missing.md"
    leaderboard.update_readme_snapshot([], readme_path=missing_readme)

    orphan_run = tmp_path / "orphan-cfg" / "plain_chat_history" / "run_1.json"
    orphan_run.parent.mkdir(parents=True, exist_ok=True)
    orphan_run.write_text(
        '{"scenario_id":"plain_chat_history","run_number":1,"metadata":{"config_slug":"orphan-cfg"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    assert cache.iter_run_records() == []

    monkeypatch.setattr(
        model_probe, "load_probe_record", lambda _slug: {"cached": True}
    )
    assert model_probe.probe_model(
        SimpleNamespace(api_key="k"), config.ModelConfig(model_id="m"), force=False
    ) == {"cached": True}


def test_scenarios_helpers() -> None:
    assert (
        scenarios.get_scenario("plain_chat_history").display_name
        == "Plain Chat History"
    )
    assert [scenario.scenario_id for scenario in scenarios.get_scenarios()] == [
        "plain_chat_history",
        "tool_mediated_reply",
    ]
    assert [
        scenario.scenario_id
        for scenario in scenarios.get_scenarios(["tool_mediated_reply"])
    ] == ["tool_mediated_reply"]


def test_detect_hidden_reasoning() -> None:
    from src.model_probe import detect_hidden_reasoning

    assert (
        detect_hidden_reasoning(
            40,
            "Done.",
            "I chose numbers...",
            None,
        )
        == "visible"
    )

    assert (
        detect_hidden_reasoning(
            40,
            "Done.",
            None,
            [{"type": "reasoning.text", "text": "thinking..."}],
        )
        == "visible"
    )

    assert (
        detect_hidden_reasoning(
            40,
            "Done.",
            None,
            None,
        )
        == "hidden"
    )

    assert (
        detect_hidden_reasoning(
            2,
            "Hello world this is a test reply with some tokens",
            None,
            None,
        )
        == "none"
    )

    assert (
        detect_hidden_reasoning(
            0,
            "",
            None,
            None,
        )
        == "none"
    )


def test_estimate_visible_token_count() -> None:
    from src.model_probe import estimate_visible_token_count

    assert estimate_visible_token_count(None) == 0
    assert estimate_visible_token_count("") == 0
    assert estimate_visible_token_count("Done.") >= 1
    assert estimate_visible_token_count("The answer is 42") >= 3


def test_check_api_reasoning_support_with_mock(monkeypatch) -> None:
    from src.model_probe import check_api_reasoning_support

    def mock_fetch(_api_key, _model_id):
        return ["temperature", "reasoning", "include_reasoning", "max_tokens"]

    monkeypatch.setattr(model_probe, "fetch_model_supported_parameters", mock_fetch)
    result = check_api_reasoning_support("key", "deepseek/deepseek-v3.2")
    assert result["api_confirmed"] is True
    assert result["has_reasoning_param"] is True
    assert result["has_include_reasoning_param"] is True

    def mock_fetch_no_reasoning(_api_key, _model_id):
        return ["temperature", "max_tokens"]

    monkeypatch.setattr(
        model_probe, "fetch_model_supported_parameters", mock_fetch_no_reasoning
    )
    result = check_api_reasoning_support("key", "some/model")
    assert result["api_confirmed"] is False

    def mock_fetch_none(_api_key, _model_id):
        return None

    monkeypatch.setattr(
        model_probe, "fetch_model_supported_parameters", mock_fetch_none
    )
    result = check_api_reasoning_support("key", "some/model")
    assert result["api_confirmed"] is None


def test_fetch_model_supported_parameters_with_mock(monkeypatch) -> None:
    from src.model_probe import fetch_model_supported_parameters
    import httpx

    captured_headers: list[dict[str, str]] = []

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {
                        "id": "vendor/model-a",
                        "supported_parameters": ["reasoning", "temperature"],
                    },
                    {"id": "vendor/model-b", "supported_parameters": ["temperature"]},
                ]
            }

    def mock_get(*_a, **kwargs):
        captured_headers.append(kwargs["headers"])
        return MockResponse()

    monkeypatch.setattr(httpx, "get", mock_get)

    result = fetch_model_supported_parameters("key", "vendor/model-a")
    assert result == ["reasoning", "temperature"]
    assert captured_headers[0] == {
        "Authorization": "Bearer key",
        "HTTP-Referer": "https://github.com/tass/ai-thought-preserved-bench",
        "X-OpenRouter-Title": "ai-thought-preserved-bench",
    }

    result = fetch_model_supported_parameters("key", "vendor/model-missing")
    assert result is None

    def raise_error(*_a, **_kw):
        raise httpx.HTTPError("connection failed")

    monkeypatch.setattr(httpx, "get", raise_error)
    result = fetch_model_supported_parameters("key", "vendor/model-a")
    assert result is None


def test_setup_file_logging_creates_log_and_symlink(
    monkeypatch, tmp_path: Path
) -> None:
    import logging

    monkeypatch.setattr(cli, "LOGS_DIR", tmp_path)

    root = logging.getLogger()
    handlers_before = len(root.handlers)

    log_path = cli.setup_file_logging("test_cmd")

    assert log_path is not None
    assert log_path.exists()
    assert "test_cmd" in log_path.name
    assert log_path.suffix == ".log"

    symlink = tmp_path / "latest.log"
    assert symlink.is_symlink()
    assert symlink.resolve() == log_path.resolve()

    assert len(root.handlers) > handlers_before

    logging.getLogger("test.file.logging").debug("debug line from test")
    logging.getLogger("test.file.logging").info("info line from test")
    for h in root.handlers[handlers_before:]:
        h.flush()
    content = log_path.read_text(encoding="utf-8")
    assert "debug line from test" in content
    assert "info line from test" in content

    for h in root.handlers[handlers_before:]:
        root.removeHandler(h)
        h.close()


def test_cleanup_old_logs_respects_retention(monkeypatch, tmp_path: Path) -> None:
    import time

    monkeypatch.setattr(cli, "LOGS_DIR", tmp_path)

    for i in range(5):
        (tmp_path / f"2026-01-0{i + 1}T00-00-00_run_pid1.log").write_text(f"log {i}")
        time.sleep(0.01)

    cli._cleanup_old_logs(retention=3)

    remaining = [p for p in tmp_path.iterdir() if p.suffix == ".log"]
    assert len(remaining) == 3

    names = sorted(p.name for p in remaining)
    assert "2026-01-01" not in names[0]
    assert "2026-01-02" not in names[0]


def test_cleanup_old_logs_ignores_symlink(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "LOGS_DIR", tmp_path)

    real_log = tmp_path / "2026-01-01T00-00-00_run_pid1.log"
    real_log.write_text("log content")
    symlink = tmp_path / "latest.log"
    symlink.symlink_to(real_log.name)

    cli._cleanup_old_logs(retention=0)

    assert symlink.exists() or symlink.is_symlink()
    assert not real_log.exists()


def test_logs_cli_command_lists_files(monkeypatch, tmp_path: Path) -> None:
    from click.testing import CliRunner

    monkeypatch.setattr(cli, "LOGS_DIR", tmp_path)

    (tmp_path / "2026-01-01T00-00-00_run_pid1.log").write_text("first log")
    (tmp_path / "2026-01-02T00-00-00_run_pid2.log").write_text("second log")

    result = CliRunner().invoke(cli.cli, ["logs"])
    assert result.exit_code == 0
    assert "2026-01-01" in result.output
    assert "2026-01-02" in result.output


def test_logs_cli_command_tail(monkeypatch, tmp_path: Path) -> None:
    from click.testing import CliRunner

    monkeypatch.setattr(cli, "LOGS_DIR", tmp_path)

    log_file = tmp_path / "2026-01-01T00-00-00_run_pid1.log"
    lines = [f"line-{i:03d}" for i in range(100)]
    log_file.write_text("\n".join(lines))

    result = CliRunner().invoke(cli.cli, ["logs", "--tail"])
    assert result.exit_code == 0
    assert "line-099" in result.output
    assert "line-060" in result.output
    assert "line-049" not in result.output


def test_logs_cli_command_empty(monkeypatch, tmp_path: Path) -> None:
    from click.testing import CliRunner

    monkeypatch.setattr(cli, "LOGS_DIR", tmp_path)
    result = CliRunner().invoke(cli.cli, ["logs"])
    assert result.exit_code == 0
    assert "No log files" in result.output


def test_logs_cli_command_no_dir(monkeypatch, tmp_path: Path) -> None:
    from click.testing import CliRunner

    monkeypatch.setattr(cli, "LOGS_DIR", tmp_path / "nonexistent")
    result = CliRunner().invoke(cli.cli, ["logs"])
    assert result.exit_code == 0
    assert "No logs directory" in result.output


def test_extract_tool_message_trailing_backslash() -> None:
    """Cover the raw_value.endswith('\\') branch (line 40) and the fallback
    JSON-decode-error path (lines 43-46) in _extract_tool_message."""
    from src.openrouter_client import _extract_tool_message

    raw = r'"message" : "hello world\"'
    result = _extract_tool_message(raw)
    assert "hello world" in result

    raw2 = '"message" : "trailing\\'
    result2 = _extract_tool_message(raw2)
    assert isinstance(result2, str)


def test_coerce_text_content_non_string_non_list() -> None:
    """Cover the fallback ``str(content).strip() or None`` branch (line 66)."""
    from src.openrouter_client import _coerce_text_content

    assert _coerce_text_content(12345) == "12345"
    assert _coerce_text_content(0) == "0"


def test_provider_mismatch_error_fields() -> None:
    """Cover ProviderMismatchError.__init__ (lines 141-144)."""
    from src.openrouter_client import ProviderMismatchError

    err = ProviderMismatchError(
        expected="DeepInfra", actual="bedrock", model="test/model"
    )
    assert err.expected == "DeepInfra"
    assert err.actual == "bedrock"
    assert err.model == "test/model"
    assert "provider constraint" in str(err).lower() or "mismatch" in str(err).lower()


def test_extract_resolved_provider_openrouter_and_kilocode() -> None:
    """Cover _extract_resolved_provider for both OpenRouter and KiloCode paths
    (lines 162, 166, 169, 176, 180-189)."""
    from src.openrouter_client import _extract_resolved_provider

    class ORResponse:
        provider = "DeepInfra"
        choices = []

    assert _extract_resolved_provider(ORResponse()) == "DeepInfra"

    class KiloMsg:
        provider_metadata = {"gateway": {"routing": {"finalProvider": "bedrock"}}}

    class KiloChoice:
        message = KiloMsg()

    class KiloResponse:
        provider = None
        choices = [KiloChoice()]

    assert _extract_resolved_provider(KiloResponse()) == "bedrock"

    class EmptyResponse:
        provider = None
        choices = []

    assert _extract_resolved_provider(EmptyResponse()) is None

    class NoMsgResponse:
        provider = None
        choices = [SimpleNamespace(message=None)]

    assert _extract_resolved_provider(NoMsgResponse()) is None

    class NoPmResponse:
        provider = None

    class NoPmMsg:
        provider_metadata = None

    class NoPmChoice:
        message = NoPmMsg()

    NoPmResponse.choices = [NoPmChoice()]
    assert _extract_resolved_provider(NoPmResponse()) is None

    class BadGatewayMsg:
        provider_metadata = {"gateway": "not-a-dict"}

    class BadGatewayChoice:
        message = BadGatewayMsg()

    class BadGatewayResponse:
        provider = None
        choices = [BadGatewayChoice()]

    assert _extract_resolved_provider(BadGatewayResponse()) is None

    class BadRoutingMsg:
        provider_metadata = {"gateway": {"routing": "not-a-dict"}}

    class BadRoutingChoice:
        message = BadRoutingMsg()

    class BadRoutingResponse:
        provider = None
        choices = [BadRoutingChoice()]

    assert _extract_resolved_provider(BadRoutingResponse()) is None


def test_providers_match_aliases() -> None:
    """Cover _providers_match alias logic (lines 194-207)."""
    from src.openrouter_client import _providers_match

    assert _providers_match("Amazon Bedrock", "bedrock") is True
    assert _providers_match("bedrock", "aws-bedrock") is True
    assert _providers_match("Google AI Studio", "google") is True
    assert _providers_match("Moonshot AI", "moonshot") is True
    assert _providers_match("DeepInfra", "DeepInfra") is True
    assert _providers_match("DeepInfra", "bedrock") is False


def test_verify_provider_mismatch_and_skip(monkeypatch, tmp_path: Path) -> None:
    """Cover _verify_provider logic (runner.py lines 128-148)."""
    from src.runner import _verify_provider
    from src.openrouter_client import CompletionResult, ProviderMismatchError, UsageInfo
    from src.config import ModelConfig

    model = ModelConfig(
        model_id="test/model",
        display_label="test",
        provider="DeepInfra",
        skip_provider_check=False,
    )

    result = CompletionResult(
        content="ok",
        visible_output="ok",
        usage=UsageInfo(),
        resolved_provider="bedrock",
    )
    with pytest.raises(ProviderMismatchError):
        _verify_provider(result, model)

    model_skip = ModelConfig(
        model_id="test/model",
        display_label="test",
        provider="DeepInfra",
        skip_provider_check=True,
    )
    _verify_provider(result, model_skip)

    model_no_provider = ModelConfig(
        model_id="test/model",
        display_label="test",
        provider=None,
    )
    _verify_provider(result, model_no_provider)

    result_no_resolved = CompletionResult(
        content="ok",
        visible_output="ok",
        usage=UsageInfo(),
        resolved_provider=None,
    )
    _verify_provider(result_no_resolved, model)


def test_cache_list_cached_configs_no_dir(monkeypatch, tmp_path: Path) -> None:
    """Cover list_cached_configs when CACHE_DIR doesn't exist (line 79)."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "nonexistent")
    assert cache.list_cached_configs() == []


def test_cache_iter_run_records_orphan(monkeypatch, tmp_path: Path) -> None:
    """Cover iter_run_records with a record missing provider (lines 101, 106-112)."""
    import json

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    run_dir = tmp_path / "cfg" / "plain" / "run_1.json"
    run_dir.parent.mkdir(parents=True)
    record = {
        "scenario_id": "plain",
        "run_number": 1,
        "metadata": {"config_slug": "cfg"},
    }
    run_dir.write_text(json.dumps(record), encoding="utf-8")
    results = cache.iter_run_records()
    assert len(results) == 0


def test_rejudge_record_skips_error_records() -> None:
    """Cover rejudge_record error-record skip (runner.py lines 395-396)."""
    from src import runner
    from src.openrouter_client import OpenRouterClient

    client = OpenRouterClient("fake-key")
    record = {"error": "some error happened", "metadata": {}}
    result = runner.rejudge_record(client, record, judge_model=None)
    assert result is record


def test_resolve_reasoning_effort_unknown_value() -> None:
    """Cover the fallback return in resolve_reasoning_effort (line 239)."""
    from src.openrouter_client import OpenRouterClient

    client = OpenRouterClient("key")
    assert client.resolve_reasoning_effort("model", "custom_value") == "custom_value"

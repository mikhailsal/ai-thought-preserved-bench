from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from src import cache, cli, model_probe, runner
from src.config import ModelConfig
from src.cost_tracker import SessionCost
from src.openrouter_client import CompletionResult, UsageInfo
from src.scenarios import TURN2_PROMPT


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.api_key: str = "fake-key"

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        if (
            messages[0]["role"] == "system"
            and "strict benchmark judge" in messages[0]["content"]
        ):
            reply_text = messages[1]["content"]
            number = 4100 if "4100" in reply_text else 6000
            explanation = (
                f"Turn-1 reasoning shows the model computed 1500+2000+2500={number}. "
                f"Turn-2 visible reply is '{number}', matching the original computation. "
                "The model genuinely recalled its prior reasoning without fabrication."
            )
            judge_json = json.dumps(
                {
                    "explanation": explanation,
                    "extracted_number": number,
                    "outcome_label": "thought_preserved",
                }
            )
            return CompletionResult(
                content=judge_json,
                visible_output=judge_json,
                usage=UsageInfo(
                    prompt_tokens=350,
                    completion_tokens=85,
                    cost_usd=0.02,
                    elapsed_seconds=0.8,
                ),
            )

        if kwargs.get("tools"):
            if len(messages) == 2:
                return CompletionResult(
                    content=None,
                    visible_output="Ready.",
                    tool_calls=[
                        {
                            "id": "bootstrap-call",
                            "type": "function",
                            "function": {
                                "name": "send_message_to_human",
                                "arguments": '{"message":"Ready."}',
                            },
                        }
                    ],
                    usage=UsageInfo(
                        prompt_tokens=10,
                        completion_tokens=5,
                        cost_usd=0.01,
                        elapsed_seconds=0.1,
                    ),
                    reasoning_effort_effective="minimal",
                )
            last_tool = [message for message in messages if message["role"] == "tool"][
                -1
            ]["content"]
            if last_tool != TURN2_PROMPT:
                return CompletionResult(
                    content=None,
                    visible_output="Done.",
                    tool_calls=[
                        {
                            "id": "turn1-call",
                            "type": "function",
                            "function": {
                                "name": "send_message_to_human",
                                "arguments": '{"message":"Done."}',
                            },
                        }
                    ],
                    reasoning_content="I chose 1500+2000+2500=6000. My secret sum is 6000.",
                    usage=UsageInfo(
                        prompt_tokens=11,
                        completion_tokens=5,
                        cost_usd=0.02,
                        elapsed_seconds=0.1,
                    ),
                    reasoning_effort_effective="minimal",
                )
            return CompletionResult(
                content=None,
                visible_output="6000",
                tool_calls=[
                    {
                        "id": "turn2-call",
                        "type": "function",
                        "function": {
                            "name": "send_message_to_human",
                            "arguments": '{"message":"6000"}',
                        },
                    }
                ],
                usage=UsageInfo(
                    prompt_tokens=12,
                    completion_tokens=4,
                    cost_usd=0.03,
                    elapsed_seconds=0.1,
                ),
                reasoning_effort_effective="minimal",
            )

        user_content = messages[-1]["content"]
        if user_content != TURN2_PROMPT:
            return CompletionResult(
                content="Done.",
                visible_output="Done.",
                reasoning_content="I chose 1500+2000+2500=6000. My secret sum is 6000.",
                usage=UsageInfo(
                    prompt_tokens=7,
                    completion_tokens=5,
                    cost_usd=0.01,
                    elapsed_seconds=0.1,
                ),
                reasoning_effort_effective="minimal",
            )
        return CompletionResult(
            content="6000",
            visible_output="6000",
            usage=UsageInfo(
                prompt_tokens=8, completion_tokens=2, cost_usd=0.01, elapsed_seconds=0.1
            ),
            reasoning_effort_effective="minimal",
        )


class NoCalcClient(FakeClient):
    """Returns open reasoning that summarizes the task without actually computing."""

    def chat(self, **kwargs):
        messages = kwargs["messages"]
        if kwargs.get("tools"):
            return super().chat(**kwargs)
        user_content = messages[-1]["content"]
        if user_content != TURN2_PROMPT:
            return CompletionResult(
                content="Done.",
                visible_output="Done.",
                reasoning_content='We need to output just "Done." and nothing else. No numbers.',
                reasoning_details=[
                    {
                        "type": "reasoning.text",
                        "text": 'We need to output just "Done." and nothing else.',
                    }
                ],
                usage=UsageInfo(
                    prompt_tokens=7,
                    completion_tokens=25,
                    cost_usd=0.01,
                    elapsed_seconds=0.1,
                ),
                reasoning_effort_effective="medium",
            )
        return super().chat(**kwargs)


class BrokenToolClient(FakeClient):
    def chat(self, **kwargs):
        messages = kwargs["messages"]
        if kwargs.get("tools") and len(messages) == 2:
            return CompletionResult(
                content="not a tool call",
                visible_output="not a tool call",
                usage=UsageInfo(),
            )
        return super().chat(**kwargs)


def _model() -> ModelConfig:
    return ModelConfig(
        model_id="test/fake-bench-model",
        display_label="fake-bench-model",
        temperature=1.2,
        reasoning_effort="minimal",
    )


def _open_model() -> ModelConfig:
    return ModelConfig(
        model_id="test/open-model",
        display_label="open-model",
        temperature=1.2,
        reasoning_effort="medium",
        reasoning_type="open",
    )


def test_runner_plain_tool_and_probe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        model_probe,
        "check_api_reasoning_support",
        lambda _key, _mid: {
            "api_confirmed": True,
            "supported_parameters": ["reasoning"],
            "has_reasoning_param": True,
            "has_include_reasoning_param": False,
        },
    )
    client = FakeClient()
    model_config = _model()

    plain = runner.run_plain_scenario(client, model_config, run_number=1, force=True)
    tool = runner.run_tool_scenario(client, model_config, run_number=1, force=True)
    records, session = runner.run_benchmark(
        client,
        [model_config],
        repetitions=1,
        scenarios=["plain_chat_history", "tool_mediated_reply"],
        force=True,
    )
    probe = model_probe.probe_model(client, model_config, force=True)

    assert plain["evaluation"]["outcome_label"] == "thought_preserved"
    assert plain["challenge"] is not None
    assert "numbers" not in plain["challenge"]
    assert "expected_sum" not in plain["challenge"]
    assert tool["evaluation"]["outcome_label"] == "thought_preserved"
    assert len(records) == 2
    assert isinstance(session, SessionCost)
    assert probe["reasoning_visibility"] == "plaintext"
    assert probe["api_reasoning_support"]["api_confirmed"] is True
    assert probe["reasoning_activity"] == "visible"


def test_runner_tool_error_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    broken = BrokenToolClient()

    record = runner.run_tool_scenario(broken, _model(), run_number=1, force=True)

    assert record["evaluation"]["excluded_from_scoring"] is True
    assert "Tool-mediated protocol failed" in record["error"]


def test_runner_returns_cached_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    client = FakeClient()
    model = _model()

    first = runner.run_plain_scenario(client, model, run_number=1, force=True)
    second = runner.run_plain_scenario(client, model, run_number=1, force=False)
    tool_first = runner.run_tool_scenario(client, model, run_number=1, force=True)
    tool_second = runner.run_tool_scenario(client, model, run_number=1, force=False)

    assert "from_cache" not in first["metadata"]
    assert second["metadata"]["from_cache"] is True
    assert "from_cache" not in tool_first["metadata"]
    assert tool_second["metadata"]["from_cache"] is True


def test_cli_commands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    fake_client = FakeClient()
    summaries = []
    runner_cli = CliRunner()

    monkeypatch.setattr(cli, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "load_api_key", lambda: "key")
    monkeypatch.setattr(cli, "OpenRouterClient", lambda _key: fake_client)
    monkeypatch.setattr(cli, "get_active_model_configs", lambda: [_model()])
    monkeypatch.setattr(cli, "get_model_config", lambda _entry: _model())
    monkeypatch.setattr(cli, "list_registered_labels_for_model", lambda _entry: [])
    monkeypatch.setattr(
        cli, "run_benchmark", lambda *args, **kwargs: ([], SessionCost())
    )
    monkeypatch.setattr(cli, "summarize_cache", lambda: summaries)
    monkeypatch.setattr(cli, "save_session_to_cost_log", lambda _session: None)
    monkeypatch.setattr(
        cli, "export_markdown_report", lambda _summaries: tmp_path / "LEADERBOARD.md"
    )
    monkeypatch.setattr(
        cli,
        "export_results_json",
        lambda _summaries, session=None: tmp_path / "results.json",
    )
    monkeypatch.setattr(cli, "update_readme_snapshot", lambda _summaries: None)
    monkeypatch.setattr(
        cli, "display_leaderboard", lambda _summaries, session=None: None
    )
    monkeypatch.setattr(
        cli,
        "probe_model",
        lambda _client, _model_config, force=False: {
            "metadata": {"display_label": "gemma"},
            "reasoning_visibility": "plaintext",
            "reasoning_effective": "minimal",
            "reasoning_activity": "visible",
            "api_reasoning_support": {"api_confirmed": True},
            "cost": {"completion_tokens": 40},
        },
    )

    assert runner_cli.invoke(cli.cli, ["run"]).exit_code == 0
    assert runner_cli.invoke(cli.cli, ["report"]).exit_code == 0
    assert runner_cli.invoke(cli.cli, ["probe"]).exit_code == 0
    assert runner_cli.invoke(cli.cli, ["rerun"]).exit_code == 0
    assert runner_cli.invoke(cli.cli, ["rejudge"]).exit_code == 0


def test_no_calc_detection_skips_turn2_and_judge(monkeypatch, tmp_path: Path) -> None:
    """When an open-type model doesn't compute in turn1 reasoning, turn2 and judge are skipped."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    client = NoCalcClient()
    model = _open_model()

    record = runner.run_plain_scenario(client, model, run_number=1, force=True)

    assert record["evaluation"]["no_calculation_detected"] is True
    assert record["evaluation"]["excluded_from_scoring"] is True
    assert record["turn2"] == {}
    assert record["evaluation"]["judge"] is None


def test_no_calc_detection_tool_scenario_skips_turn2(
    monkeypatch, tmp_path: Path
) -> None:
    """No-calc detection in tool scenario also skips turn2 and judge."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    class NoCalcToolClient(FakeClient):
        def chat(self, **kwargs):
            messages = kwargs["messages"]
            if kwargs.get("tools"):
                if len(messages) == 2:
                    return CompletionResult(
                        content=None,
                        visible_output="Ready.",
                        tool_calls=[
                            {
                                "id": "boot",
                                "type": "function",
                                "function": {
                                    "name": "send_message_to_human",
                                    "arguments": '{"message":"Ready."}',
                                },
                            }
                        ],
                        usage=UsageInfo(
                            prompt_tokens=10,
                            completion_tokens=5,
                            cost_usd=0.01,
                            elapsed_seconds=0.1,
                        ),
                        reasoning_effort_effective="medium",
                    )
                last_tool = [m for m in messages if m["role"] == "tool"][-1]["content"]
                if last_tool != TURN2_PROMPT:
                    return CompletionResult(
                        content=None,
                        visible_output="Done.",
                        tool_calls=[
                            {
                                "id": "t1",
                                "type": "function",
                                "function": {
                                    "name": "send_message_to_human",
                                    "arguments": '{"message":"Done."}',
                                },
                            }
                        ],
                        reasoning_content='Output "Done." only.',
                        reasoning_details=[
                            {"type": "reasoning.text", "text": 'Output "Done." only.'}
                        ],
                        usage=UsageInfo(
                            prompt_tokens=11,
                            completion_tokens=15,
                            cost_usd=0.02,
                            elapsed_seconds=0.1,
                        ),
                        reasoning_effort_effective="medium",
                    )
            return super().chat(**kwargs)

    record = runner.run_tool_scenario(
        NoCalcToolClient(), _open_model(), run_number=1, force=True
    )
    assert record["evaluation"]["no_calculation_detected"] is True
    assert record["evaluation"]["excluded_from_scoring"] is True
    assert record["turn2"] == {}


def test_rejudge_record_preserves_turns_and_updates_evaluation(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    client = FakeClient()
    model = _model()

    original = runner.run_plain_scenario(client, model, run_number=1, force=True)
    original_turn1 = original["turn1"].copy()
    original_turn2 = original["turn2"].copy()
    rejudged = runner.rejudge_record(
        client, original, judge_model="google/gemini-3-flash-preview"
    )

    assert rejudged["turn1"]["reasoning_content"] == original_turn1["reasoning_content"]
    assert rejudged["turn1"]["visible_reply"] == original_turn1["visible_reply"]
    assert rejudged["turn2"]["visible_reply"] == original_turn2["visible_reply"]
    assert rejudged["metadata"]["rejudged"] is True
    assert "evaluation" in rejudged
    assert rejudged["evaluation"]["judge"] is not None


def test_rejudge_record_without_judge_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    client = FakeClient()
    model = _model()

    original = runner.run_plain_scenario(client, model, run_number=1, force=True)
    call_count_before = len(client.calls)

    rejudged = runner.rejudge_record(client, original, judge_model=None)

    assert len(client.calls) == call_count_before
    assert rejudged["evaluation"]["judge"] is None
    assert rejudged["metadata"]["rejudged"] is True


def test_rejudge_cli_with_cached_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    client = FakeClient()
    model = _model()

    runner.run_plain_scenario(client, model, run_number=1, force=True)

    runner_cli = CliRunner()
    monkeypatch.setattr(cli, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "load_api_key", lambda: "key")
    monkeypatch.setattr(cli, "OpenRouterClient", lambda _key: client)
    monkeypatch.setattr(cli, "get_active_model_configs", lambda: [model])
    monkeypatch.setattr(cli, "get_model_config", lambda _entry: model)
    monkeypatch.setattr(cli, "list_registered_labels_for_model", lambda _entry: [])
    monkeypatch.setattr(cli, "summarize_cache", lambda: [])
    monkeypatch.setattr(cli, "save_session_to_cost_log", lambda _session: None)
    monkeypatch.setattr(
        cli, "export_markdown_report", lambda _summaries: tmp_path / "LEADERBOARD.md"
    )
    monkeypatch.setattr(
        cli,
        "export_results_json",
        lambda _summaries, session=None: tmp_path / "results.json",
    )
    monkeypatch.setattr(cli, "update_readme_snapshot", lambda _summaries: None)
    monkeypatch.setattr(
        cli, "display_leaderboard", lambda _summaries, session=None: None
    )

    result = runner_cli.invoke(cli.cli, ["rejudge"])
    assert result.exit_code == 0
    assert "Re-judged" in result.output

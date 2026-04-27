from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from src import cache, cli, model_probe, runner
from src.config import ModelConfig
from src.cost_tracker import SessionCost
from src.openrouter_client import CompletionResult, UsageInfo
from src.scenarios import TURN2_PROMPT, format_turn1_prompt


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        if messages[0]["role"] == "system" and "strict benchmark judge" in messages[0]["content"]:
            reply_text = messages[1]["content"]
            number = 4100 if "4100" in reply_text else 6000
            return CompletionResult(
                content=(
                    '{"outcome_label":"thought_preserved",'
                    f'"extracted_number":{number},'
                    '"explanation":"matched"}'
                ),
                visible_output=(
                    '{"outcome_label":"thought_preserved",'
                    f'"extracted_number":{number},'
                    '"explanation":"matched"}'
                ),
                usage=UsageInfo(prompt_tokens=2, completion_tokens=3, cost_usd=0.02, elapsed_seconds=0.1),
            )

        if kwargs.get("tools"):
            if len(messages) == 2:
                return CompletionResult(
                    content=None,
                    visible_output="Ready.",
                    tool_calls=[{
                        "id": "bootstrap-call",
                        "type": "function",
                        "function": {"name": "send_message_to_human", "arguments": '{"message":"Ready."}'},
                    }],
                    usage=UsageInfo(prompt_tokens=10, completion_tokens=5, cost_usd=0.01, elapsed_seconds=0.1),
                    reasoning_effort_effective="low",
                )
            last_tool = [message for message in messages if message["role"] == "tool"][-1]["content"]
            if last_tool != TURN2_PROMPT:
                return CompletionResult(
                    content=None,
                    visible_output="Done.",
                    tool_calls=[{
                        "id": "turn1-call",
                        "type": "function",
                        "function": {"name": "send_message_to_human", "arguments": '{"message":"Done."}'},
                    }],
                    reasoning_content="I chose 1500+2000+2500=6000. My secret sum is 6000.",
                    usage=UsageInfo(prompt_tokens=11, completion_tokens=5, cost_usd=0.02, elapsed_seconds=0.1),
                    reasoning_effort_effective="low",
                )
            return CompletionResult(
                content=None,
                visible_output="6000",
                tool_calls=[{
                    "id": "turn2-call",
                    "type": "function",
                    "function": {"name": "send_message_to_human", "arguments": '{"message":"6000"}'},
                }],
                usage=UsageInfo(prompt_tokens=12, completion_tokens=4, cost_usd=0.03, elapsed_seconds=0.1),
                reasoning_effort_effective="low",
            )

        user_content = messages[-1]["content"]
        if user_content != TURN2_PROMPT:
            return CompletionResult(
                content="Done.",
                visible_output="Done.",
                reasoning_content="I chose 1500+2000+2500=6000. My secret sum is 6000.",
                usage=UsageInfo(prompt_tokens=7, completion_tokens=5, cost_usd=0.01, elapsed_seconds=0.1),
                reasoning_effort_effective="low",
            )
        return CompletionResult(
            content="6000",
            visible_output="6000",
            usage=UsageInfo(prompt_tokens=8, completion_tokens=2, cost_usd=0.01, elapsed_seconds=0.1),
            reasoning_effort_effective="low",
        )


class BrokenToolClient(FakeClient):
    def chat(self, **kwargs):
        messages = kwargs["messages"]
        if kwargs.get("tools") and len(messages) == 2:
            return CompletionResult(content="not a tool call", visible_output="not a tool call", usage=UsageInfo())
        return super().chat(**kwargs)


def _model() -> ModelConfig:
    return ModelConfig(model_id="google/gemma-4-31b-it:free", display_label="gemma", temperature=1.2, reasoning_effort="minimal")


def test_runner_plain_tool_and_probe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
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

    assert first["metadata"]["from_cache"] is False
    assert second["metadata"]["from_cache"] is True
    assert tool_first["metadata"]["from_cache"] is False
    assert tool_second["metadata"]["from_cache"] is True


def test_cli_commands(monkeypatch, tmp_path: Path) -> None:
    fake_client = FakeClient()
    summaries = []
    runner_cli = CliRunner()

    monkeypatch.setattr(cli, "ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "load_api_key", lambda: "key")
    monkeypatch.setattr(cli, "OpenRouterClient", lambda _key: fake_client)
    monkeypatch.setattr(cli, "get_active_model_configs", lambda: [_model()])
    monkeypatch.setattr(cli, "get_model_config", lambda _entry: _model())
    monkeypatch.setattr(cli, "list_registered_labels_for_model", lambda _entry: [])
    monkeypatch.setattr(cli, "run_benchmark", lambda *args, **kwargs: ([], SessionCost()))
    monkeypatch.setattr(cli, "summarize_cache", lambda: summaries)
    monkeypatch.setattr(cli, "save_session_to_cost_log", lambda _session: None)
    monkeypatch.setattr(cli, "export_markdown_report", lambda _summaries: tmp_path / "LEADERBOARD.md")
    monkeypatch.setattr(cli, "export_results_json", lambda _summaries, session=None: tmp_path / "results.json")
    monkeypatch.setattr(cli, "update_readme_snapshot", lambda _summaries: None)
    monkeypatch.setattr(cli, "display_leaderboard", lambda _summaries, session=None: None)
    monkeypatch.setattr(cli, "probe_model", lambda _client, _model_config, force=False: {"metadata": {"display_label": "gemma"}, "reasoning_visibility": "plaintext", "reasoning_effective": "low"})

    assert runner_cli.invoke(cli.cli, ["run"]).exit_code == 0
    assert runner_cli.invoke(cli.cli, ["report"]).exit_code == 0
    assert runner_cli.invoke(cli.cli, ["probe"]).exit_code == 0
    assert runner_cli.invoke(cli.cli, ["rerun"]).exit_code == 0

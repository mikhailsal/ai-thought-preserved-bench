"""Benchmark runner for both scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.cache import load_run_record, save_run_record
from src.config import JUDGE_MODEL, ModelConfig
from src.cost_tracker import SessionCost, TaskCost
from src.evaluator import (
    evaluate_run_record,
    extract_structured_reasoning_text,
    judge_turn2_reply,
    reconcile_stability_group,
)
from src.openrouter_client import CompletionResult, OpenRouterClient
from src.prompt_builder import (
    build_plain_turn1_messages,
    build_plain_turn2_messages,
    build_tool_bootstrap_messages,
    build_tool_turn1_messages,
    build_tool_turn2_messages,
    get_tool_definitions,
)
from src.scenarios import SCENARIO_PLAIN, SCENARIO_TOOL


def _cost_info(result: CompletionResult) -> dict[str, Any]:
    return {
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "cost_usd": round(result.usage.cost_usd, 6),
        "elapsed_seconds": round(result.usage.elapsed_seconds, 3),
    }


def _assistant_artifact(result: CompletionResult) -> dict[str, Any]:
    return {
        "content": result.content,
        "visible_reply": result.visible_output,
        "reasoning_content": result.reasoning_content,
        "reasoning_details": result.reasoning_details,
        "tool_calls": result.tool_calls,
        "finish_reason": result.finish_reason,
        "provider": result.provider,
        "reasoning_effort_effective": result.reasoning_effort_effective,
        "usage": _cost_info(result),
    }


def _record_template(model_config: ModelConfig, scenario_id: str, run_number: int) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "run_number": run_number,
        "model_id": model_config.model_id,
        "provider": model_config.provider,
        "display_label": model_config.label,
        "reasoning_requested": model_config.reasoning_requested,
        "reasoning_effective": None,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_slug": model_config.config_slug,
            "display_label": model_config.label,
            "provider": model_config.provider,
        },
    }


def _save_error_record(
    model_config: ModelConfig,
    scenario_id: str,
    run_number: int,
    error_message: str,
    partial_payload: dict[str, Any],
) -> dict[str, Any]:
    record = _record_template(model_config, scenario_id, run_number)
    record.update(partial_payload)
    record["error"] = error_message
    record["evaluation"] = {
        "reasoning_visibility": "none",
        "turn1_chosen_number_visible_to_benchmark": None,
        "turn2_extracted_number": None,
        "outcome_label": "other_refusal",
        "outcome_notes": error_message,
        "pending_stability_check": False,
        "excluded_from_scoring": True,
        "judge": None,
    }
    save_run_record(record)
    return record


def _get_reasoning_text(result: CompletionResult) -> str | None:
    """Extract readable reasoning text from a completion result."""
    if result.reasoning_content:
        return result.reasoning_content
    return extract_structured_reasoning_text(result.reasoning_details)


def _call_model(
    client: OpenRouterClient,
    model_config: ModelConfig,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> CompletionResult:
    return client.chat(
        model=model_config.model_id,
        messages=messages,
        max_tokens=model_config.effective_max_tokens,
        temperature=model_config.effective_temperature,
        reasoning_effort=model_config.reasoning_requested,
        tools=tools,
        provider=model_config.provider,
    )


def run_plain_scenario(
    client: OpenRouterClient,
    model_config: ModelConfig,
    *,
    run_number: int,
    force: bool = False,
    judge_model: str | None = JUDGE_MODEL,
) -> dict[str, Any]:
    cached = None if force else load_run_record(model_config.config_slug, SCENARIO_PLAIN, run_number)
    if cached:
        cached.setdefault("metadata", {})["from_cache"] = True
        return cached

    turn1_messages = build_plain_turn1_messages()
    turn1_result = _call_model(client, model_config, turn1_messages)
    turn1_artifact = _assistant_artifact(turn1_result)
    turn2_messages = build_plain_turn2_messages(turn1_artifact)
    turn2_result = _call_model(client, model_config, turn2_messages)

    record = _record_template(model_config, SCENARIO_PLAIN, run_number)
    record["metadata"]["from_cache"] = False
    record["reasoning_effective"] = turn1_result.reasoning_effort_effective or "none"
    record["turn1"] = {
        **turn1_artifact,
        "request_messages": turn1_messages,
    }
    record["turn2"] = {
        **_assistant_artifact(turn2_result),
        "request_messages": turn2_messages,
    }

    judge = None
    if judge_model:
        judge = judge_turn2_reply(
            client,
            turn2_result.visible_output,
            judge_model=judge_model,
            turn2_reasoning=_get_reasoning_text(turn2_result),
            turn1_reasoning=_get_reasoning_text(turn1_result),
        )
    record["evaluation"] = evaluate_run_record(record, judge)
    save_run_record(record)
    return record


def run_tool_scenario(
    client: OpenRouterClient,
    model_config: ModelConfig,
    *,
    run_number: int,
    force: bool = False,
    judge_model: str | None = JUDGE_MODEL,
) -> dict[str, Any]:
    cached = None if force else load_run_record(model_config.config_slug, SCENARIO_TOOL, run_number)
    if cached:
        cached.setdefault("metadata", {})["from_cache"] = True
        return cached

    tools = get_tool_definitions()
    bootstrap_messages = build_tool_bootstrap_messages()
    bootstrap_result = _call_model(client, model_config, bootstrap_messages, tools=tools)
    bootstrap_artifact = _assistant_artifact(bootstrap_result)
    partial = {
        "bootstrap": {
            **bootstrap_artifact,
            "request_messages": bootstrap_messages,
        },
    }
    try:
        turn1_messages = build_tool_turn1_messages(bootstrap_artifact)
        turn1_result = _call_model(client, model_config, turn1_messages, tools=tools)
        turn1_artifact = _assistant_artifact(turn1_result)
        turn2_messages = build_tool_turn2_messages(bootstrap_artifact, turn1_artifact)
        turn2_result = _call_model(client, model_config, turn2_messages, tools=tools)
    except Exception as exc:
        return _save_error_record(
            model_config,
            SCENARIO_TOOL,
            run_number,
            f"Tool-mediated protocol failed: {exc}",
            partial,
        )

    record = _record_template(model_config, SCENARIO_TOOL, run_number)
    record["metadata"]["from_cache"] = False
    record["reasoning_effective"] = turn1_result.reasoning_effort_effective or "none"
    record["bootstrap"] = partial["bootstrap"]
    record["turn1"] = {
        **turn1_artifact,
        "request_messages": turn1_messages,
    }
    record["turn2"] = {
        **_assistant_artifact(turn2_result),
        "request_messages": turn2_messages,
    }

    judge = None
    if judge_model:
        judge = judge_turn2_reply(
            client,
            turn2_result.visible_output,
            judge_model=judge_model,
            turn2_reasoning=_get_reasoning_text(turn2_result),
            turn1_reasoning=_get_reasoning_text(turn1_result),
        )
    record["evaluation"] = evaluate_run_record(record, judge)
    save_run_record(record)
    return record


def run_benchmark(
    client: OpenRouterClient,
    model_configs: list[ModelConfig],
    *,
    repetitions: int,
    scenarios: list[str],
    judge_model: str | None = JUDGE_MODEL,
    force: bool = False,
) -> tuple[list[dict[str, Any]], SessionCost]:
    session = SessionCost()
    records: list[dict[str, Any]] = []
    for model_config in model_configs:
        for scenario_id in scenarios:
            group_records: list[dict[str, Any]] = []
            for run_number in range(1, repetitions + 1):
                if scenario_id == SCENARIO_PLAIN:
                    record = run_plain_scenario(
                        client,
                        model_config,
                        run_number=run_number,
                        force=force,
                        judge_model=judge_model,
                    )
                else:
                    record = run_tool_scenario(
                        client,
                        model_config,
                        run_number=run_number,
                        force=force,
                        judge_model=judge_model,
                    )
                group_records.append(record)
                records.append(record)

                if not record.get("metadata", {}).get("from_cache"):
                    generation_task = TaskCost(label=f"{model_config.label}:{scenario_id}:run{run_number}")
                    for stage in ("bootstrap", "turn1", "turn2"):
                        stage_payload = record.get(stage)
                        if not stage_payload:
                            continue
                        usage = stage_payload.get("usage", {})
                        generation_task.add(
                            prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                            cost_usd=float(usage.get("cost_usd", 0.0)),
                            elapsed_seconds=float(usage.get("elapsed_seconds", 0.0)),
                        )
                    session.add_task(generation_task)
                    judge_payload = record.get("evaluation", {}).get("judge")
                    if judge_payload:
                        judge_usage = judge_payload.get("usage", {})
                        judge_task = TaskCost(label=f"judge:{model_config.label}:{scenario_id}:run{run_number}")
                        judge_task.add(
                            prompt_tokens=int(judge_usage.get("prompt_tokens", 0)),
                            completion_tokens=int(judge_usage.get("completion_tokens", 0)),
                            cost_usd=float(judge_usage.get("cost_usd", 0.0)),
                            elapsed_seconds=float(judge_usage.get("elapsed_seconds", 0.0)),
                        )
                        session.add_task(judge_task)

            reconcile_stability_group(group_records)
            for record in group_records:
                save_run_record(record)
    return records, session
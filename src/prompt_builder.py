"""Prompt and replay message construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.scenarios import (
    SEND_MESSAGE_TOOL,
    TOOL_BOOTSTRAP_USER,
    TOOL_SYSTEM_PROMPT,
    TURN2_PROMPT,
    format_turn1_prompt,
)


def _without_none_values(message: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in message.items() if value is not None}


def get_first_tool_call_id(assistant_artifact: dict[str, Any]) -> str:
    tool_calls = assistant_artifact.get("tool_calls") or []
    if not tool_calls:
        raise ValueError("Assistant artifact does not contain any tool calls.")
    tool_call_id = tool_calls[0].get("id")
    if not tool_call_id:
        raise ValueError("Assistant tool call is missing an id.")
    return str(tool_call_id)


def build_replay_assistant_message(
    assistant_artifact: dict[str, Any],
) -> dict[str, Any]:
    message = {
        "role": "assistant",
        "content": assistant_artifact.get("content"),
        "tool_calls": deepcopy(assistant_artifact.get("tool_calls")),
    }
    reasoning_details = assistant_artifact.get("reasoning_details")
    if reasoning_details:
        message["reasoning_details"] = deepcopy(reasoning_details)
    else:
        reasoning_content = assistant_artifact.get("reasoning_content")
        if reasoning_content:
            message["reasoning"] = reasoning_content
    return _without_none_values(message)


def build_plain_turn1_messages(challenge: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": format_turn1_prompt(challenge)}]


def build_plain_turn2_messages(
    challenge: dict[str, Any],
    turn1_assistant: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": format_turn1_prompt(challenge)},
        build_replay_assistant_message(turn1_assistant),
        {"role": "user", "content": TURN2_PROMPT},
    ]


def build_tool_bootstrap_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": TOOL_BOOTSTRAP_USER},
    ]


def build_tool_turn1_messages(
    challenge: dict[str, Any],
    bootstrap_assistant: dict[str, Any],
) -> list[dict[str, Any]]:
    bootstrap_tool_call_id = get_first_tool_call_id(bootstrap_assistant)
    return [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": TOOL_BOOTSTRAP_USER},
        build_replay_assistant_message(bootstrap_assistant),
        {
            "role": "tool",
            "tool_call_id": bootstrap_tool_call_id,
            "content": format_turn1_prompt(challenge),
        },
    ]


def build_tool_turn2_messages(
    challenge: dict[str, Any],
    bootstrap_assistant: dict[str, Any],
    turn1_assistant: dict[str, Any],
) -> list[dict[str, Any]]:
    turn1_tool_call_id = get_first_tool_call_id(turn1_assistant)
    return [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": TOOL_BOOTSTRAP_USER},
        build_replay_assistant_message(bootstrap_assistant),
        {
            "role": "tool",
            "tool_call_id": get_first_tool_call_id(bootstrap_assistant),
            "content": format_turn1_prompt(challenge),
        },
        build_replay_assistant_message(turn1_assistant),
        {
            "role": "tool",
            "tool_call_id": turn1_tool_call_id,
            "content": TURN2_PROMPT,
        },
    ]


def get_tool_definitions() -> list[dict[str, Any]]:
    return [deepcopy(SEND_MESSAGE_TOOL)]

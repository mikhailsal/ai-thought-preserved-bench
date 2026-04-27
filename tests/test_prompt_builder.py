from __future__ import annotations

import pytest

from src import prompt_builder
from src.scenarios import TOOL_BOOTSTRAP_USER, TURN1_PROMPT, TURN2_PROMPT


def _assistant_artifact() -> dict:
    return {
        "content": None,
        "visible_reply": "I have a number.",
        "reasoning_content": "I chose 37.",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "I chose 37."},
        ],
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "send_message_to_human",
                    "arguments": '{"message":"I have a number."}',
                },
            }
        ],
    }


def test_build_replay_assistant_message_prefers_reasoning_details() -> None:
    replay = prompt_builder.build_replay_assistant_message(_assistant_artifact())

    assert replay["role"] == "assistant"
    assert replay["tool_calls"][0]["id"] == "call-1"
    assert replay["reasoning_details"] == [{"type": "reasoning.text", "text": "I chose 37."}]
    assert "reasoning" not in replay


def test_build_plain_turn_messages() -> None:
    turn1 = prompt_builder.build_plain_turn1_messages()
    turn2 = prompt_builder.build_plain_turn2_messages(_assistant_artifact())

    assert turn1 == [{"role": "user", "content": TURN1_PROMPT}]
    assert turn2[0] == {"role": "user", "content": TURN1_PROMPT}
    assert turn2[1]["role"] == "assistant"
    assert turn2[2] == {"role": "user", "content": TURN2_PROMPT}


def test_build_tool_messages_preserve_call_ids_and_ordering() -> None:
    bootstrap = _assistant_artifact()
    turn1_messages = prompt_builder.build_tool_turn1_messages(bootstrap)
    turn2_messages = prompt_builder.build_tool_turn2_messages(bootstrap, bootstrap)

    assert prompt_builder.build_tool_bootstrap_messages()[1]["content"] == TOOL_BOOTSTRAP_USER
    assert turn1_messages[3] == {"role": "tool", "tool_call_id": "call-1", "content": TURN1_PROMPT}
    assert turn2_messages[3]["tool_call_id"] == "call-1"
    assert turn2_messages[5] == {"role": "tool", "tool_call_id": "call-1", "content": TURN2_PROMPT}
    assert prompt_builder.get_tool_definitions()[0]["function"]["name"] == "send_message_to_human"


def test_get_first_tool_call_id_requires_tool_calls() -> None:
    with pytest.raises(ValueError):
        prompt_builder.get_first_tool_call_id({"tool_calls": []})
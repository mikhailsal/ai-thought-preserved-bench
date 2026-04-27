from __future__ import annotations

import pytest

from src import prompt_builder
from src.scenarios import TURN2_PROMPT, format_turn1_prompt, generate_challenge


def _challenge() -> dict:
    return {
        "range_low": 196,
        "range_high": 5342,
    }


def _assistant_artifact() -> dict:
    return {
        "content": None,
        "visible_reply": "Done.",
        "reasoning_content": "I chose 300+400+500=1200.",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "I chose 300+400+500=1200."},
        ],
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "send_message_to_human",
                    "arguments": '{"message":"Done."}',
                },
            }
        ],
    }


def test_build_replay_assistant_message_prefers_reasoning_details() -> None:
    replay = prompt_builder.build_replay_assistant_message(_assistant_artifact())

    assert replay["role"] == "assistant"
    assert replay["tool_calls"][0]["id"] == "call-1"
    assert replay["reasoning_details"] == [{"type": "reasoning.text", "text": "I chose 300+400+500=1200."}]
    assert "reasoning" not in replay


def test_build_plain_turn_messages() -> None:
    challenge = _challenge()
    turn1_prompt = format_turn1_prompt(challenge)
    turn1 = prompt_builder.build_plain_turn1_messages(challenge)
    turn2 = prompt_builder.build_plain_turn2_messages(challenge, _assistant_artifact())

    assert turn1 == [{"role": "user", "content": turn1_prompt}]
    assert turn2[0] == {"role": "user", "content": turn1_prompt}
    assert turn2[1]["role"] == "assistant"
    assert turn2[2] == {"role": "user", "content": TURN2_PROMPT}


def test_build_tool_messages_preserve_call_ids_and_ordering() -> None:
    challenge = _challenge()
    turn1_prompt = format_turn1_prompt(challenge)
    bootstrap = _assistant_artifact()
    turn1_messages = prompt_builder.build_tool_turn1_messages(challenge, bootstrap)
    turn2_messages = prompt_builder.build_tool_turn2_messages(challenge, bootstrap, bootstrap)

    assert prompt_builder.build_tool_bootstrap_messages()[1]["content"] == "[start]"
    assert turn1_messages[3] == {"role": "tool", "tool_call_id": "call-1", "content": turn1_prompt}
    assert turn2_messages[3]["tool_call_id"] == "call-1"
    assert turn2_messages[5] == {"role": "tool", "tool_call_id": "call-1", "content": TURN2_PROMPT}
    assert prompt_builder.get_tool_definitions()[0]["function"]["name"] == "send_message_to_human"


def test_get_first_tool_call_id_requires_tool_calls() -> None:
    with pytest.raises(ValueError):
        prompt_builder.get_first_tool_call_id({"tool_calls": []})


def test_generate_challenge_produces_valid_values() -> None:
    challenge = generate_challenge()
    assert challenge["range_low"] == 196
    assert challenge["range_high"] == 5342
    assert "numbers" not in challenge
    assert "expected_sum" not in challenge


def test_format_turn1_prompt_contains_range_and_scoped_secrecy() -> None:
    challenge = _challenge()
    prompt = format_turn1_prompt(challenge)
    assert "196" in prompt
    assert "5342" in prompt
    assert "Done." in prompt
    assert "this reply" in prompt.lower() or "this message" in prompt.lower()
    assert "later" in prompt.lower()
    assert "MUST" in prompt or "mandatory" in prompt.lower()
    assert "inspected" in prompt.lower() or "reviewed" in prompt.lower()

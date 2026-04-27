from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.openrouter_client import (
    CompletionResult,
    OpenRouterClient,
    UsageInfo,
    _coerce_text_content,
    _extract_tool_message,
    _to_plain_object,
)


@dataclass
class DummyUsage:
    prompt_tokens: int = 11
    completion_tokens: int = 7
    cost: float = 0.01


@dataclass
class DummyMessage:
    content: Any = None
    tool_calls: Any = None
    reasoning: Any = None
    reasoning_content: Any = None
    reasoning_details: Any = None


@dataclass
class DummyChoice:
    message: DummyMessage
    finish_reason: str = "stop"


@dataclass
class DummyResponse:
    choices: list[DummyChoice]
    usage: DummyUsage
    model: str = "test-model"


class DummyCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses

    def create(self, **_: Any) -> Any:
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DummyChat:
    def __init__(self, responses: list[Any]) -> None:
        self.completions = DummyCompletions(responses)


class DummySDKClient:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = DummyChat(responses)


class RateLimitError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("rate limited")
        self.status_code = 429


def test_extract_tool_message_and_content_helpers() -> None:
    assert _extract_tool_message('{"message":"Hello"}') == "Hello"
    assert _extract_tool_message('{"message":"Hello') == "Hello"
    assert _coerce_text_content([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]) == "A\nB"
    assert _to_plain_object({"a": [1, 2]}) == {"a": [1, 2]}


def test_resolve_reasoning_effort() -> None:
    client = OpenRouterClient("key")
    assert client.resolve_reasoning_effort("any-model", None) is None
    assert client.resolve_reasoning_effort("any-model", "none") is None
    assert client.resolve_reasoning_effort("any-model", "off") is None
    assert client.resolve_reasoning_effort("any-model", "minimal") == "low"
    assert client.resolve_reasoning_effort("any-model", "high") == "high"


def test_chat_extracts_tool_calls_reasoning_details_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    message = DummyMessage(
        content=None,
        tool_calls=[
            {
                "id": "tool-1",
                "type": "function",
                "function": {
                    "name": "send_message_to_human",
                    "arguments": '{"message":"I have a number."}',
                },
            }
        ],
        reasoning=None,
        reasoning_content=None,
        reasoning_details=[{"type": "reasoning.summary", "text": "hidden"}],
    )
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    client = OpenRouterClient("key")
    monkeypatch.setattr(client, "_client", DummySDKClient([RateLimitError(), response]))
    monkeypatch.setattr("src.openrouter_client.time.sleep", lambda *_: None)

    result = client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        temperature=1.2,
        reasoning_effort="minimal",
        tools=[{"type": "function"}],
        provider="provider/x",
    )

    assert result.visible_output == "I have a number."
    assert result.reasoning_details == [{"type": "reasoning.summary", "text": "hidden"}]
    assert result.reasoning_effort_effective == "low"
    assert result.tool_calls[0]["id"] == "tool-1"


def test_chat_handles_plain_text_content() -> None:
    message = DummyMessage(content="37", reasoning="I chose 37.")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    client = OpenRouterClient("key")
    client._client = DummySDKClient([response])

    result = client.chat(
        model="plain-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        temperature=1.2,
        reasoning_effort="minimal",
    )

    assert result.visible_output == "37"
    assert result.reasoning_content == "I chose 37."
    assert result.reasoning_effort_effective == "low"
    assert isinstance(result.usage, UsageInfo)
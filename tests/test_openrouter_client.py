from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.config import ModelPricing
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


def test_fetch_pricing_supports_reasoning_and_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyHTTPResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {
                        "id": "google/gemma-4-31b-it:free",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                        "supported_parameters": ["reasoning"],
                    }
                ]
            }

    monkeypatch.setattr("src.openrouter_client.requests.get", lambda *args, **kwargs: DummyHTTPResponse())
    client = OpenRouterClient("key")

    pricing = client.fetch_pricing()

    assert pricing["google/gemma-4-31b-it:free"] == ModelPricing(0.000001, 0.000002)
    assert client.supports_reasoning("google/gemma-4-31b-it:free") is True
    assert client.validate_model("google/gemma-4-31b-it:free") is True
    assert client.get_model_pricing("missing") == ModelPricing()


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
    client._pricing_cache = {"test-model": ModelPricing()}
    client._reasoning_models = {"test-model"}
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
    client._pricing_cache = {"plain-model": ModelPricing()}
    client._reasoning_models = set()
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
    assert result.reasoning_effort_effective is None
    assert isinstance(result.usage, UsageInfo)
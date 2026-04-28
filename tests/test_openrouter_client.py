from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import src.openrouter_client as openrouter_client
from src.openrouter_client import (
    OpenRouterClient,
    UsageInfo,
    _coerce_text_content,
    _extract_tool_message,
    _to_plain_object,
)


@dataclass
class DummyCompletionTokensDetails:
    reasoning_tokens: int = 0


@dataclass
class DummyUsage:
    prompt_tokens: int = 11
    completion_tokens: int = 7
    cost: float = 0.01
    completion_tokens_details: DummyCompletionTokensDetails | None = None


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
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
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


def test_openrouter_client_sets_attribution_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    class DummyOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(openrouter_client, "OpenAI", DummyOpenAI)

    OpenRouterClient("key")

    assert captured_kwargs["default_headers"] == {
        "HTTP-Referer": "https://github.com/tass/ai-thought-preserved-bench",
        "X-OpenRouter-Title": "ai-thought-preserved-bench",
    }


def test_extract_tool_message_and_content_helpers() -> None:
    assert _extract_tool_message('{"message":"Hello"}') == "Hello"
    assert _extract_tool_message('{"message":"Hello') == "Hello"
    assert (
        _coerce_text_content(
            [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
        )
        == "A\nB"
    )
    assert _to_plain_object({"a": [1, 2]}) == {"a": [1, 2]}


def test_resolve_reasoning_effort() -> None:
    client = OpenRouterClient("key")
    assert client.resolve_reasoning_effort("any-model", None) is None
    assert client.resolve_reasoning_effort("any-model", "none") == "none"
    assert client.resolve_reasoning_effort("any-model", "off") == "none"
    assert client.resolve_reasoning_effort("any-model", "minimal") == "minimal"
    assert client.resolve_reasoning_effort("any-model", "low") == "low"
    assert client.resolve_reasoning_effort("any-model", "high") == "high"


def test_chat_extracts_tool_calls_reasoning_details_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert result.reasoning_effort_effective == "minimal"
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
    assert result.reasoning_effort_effective == "minimal"
    assert isinstance(result.usage, UsageInfo)


def test_reasoning_config_excludes_enabled_flag() -> None:
    """Reasoning requests must set effort+exclude but NOT enabled=True
    (enabled overrides effort to 'medium' per OpenRouter docs)."""
    message = DummyMessage(content="ok")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    sdk_client = DummySDKClient([response])
    client = OpenRouterClient("key")
    client._client = sdk_client

    client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
        reasoning_effort="high",
    )

    extra_body = sdk_client.chat.completions.last_kwargs.get("extra_body", {})
    reasoning = extra_body.get("reasoning", {})
    assert reasoning["effort"] == "high"
    assert reasoning["exclude"] is False
    assert "enabled" not in reasoning


def test_require_parameters_set_when_no_provider_pinned() -> None:
    """Without a pinned provider, require_parameters must be True to ensure
    the routed provider actually supports reasoning."""
    message = DummyMessage(content="ok")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    sdk_client = DummySDKClient([response])
    client = OpenRouterClient("key")
    client._client = sdk_client

    client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
        reasoning_effort="minimal",
    )

    extra_body = sdk_client.chat.completions.last_kwargs.get("extra_body", {})
    reasoning = extra_body.get("reasoning", {})
    assert reasoning["effort"] == "minimal"
    provider_config = extra_body.get("provider", {})
    assert provider_config.get("require_parameters") is True
    assert "order" not in provider_config


def test_provider_pinned_no_require_parameters() -> None:
    """With a pinned provider, require_parameters should not be added
    (provider.order + allow_fallbacks=False is sufficient)."""
    message = DummyMessage(content="ok")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    sdk_client = DummySDKClient([response])
    client = OpenRouterClient("key")
    client._client = sdk_client

    client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
        reasoning_effort="minimal",
        provider="DeepSeek",
    )

    extra_body = sdk_client.chat.completions.last_kwargs.get("extra_body", {})
    provider_config = extra_body.get("provider", {})
    assert provider_config["order"] == ["DeepSeek"]
    assert provider_config["allow_fallbacks"] is False
    assert "require_parameters" not in provider_config


def test_no_reasoning_no_extra_body() -> None:
    """When reasoning is disabled, extra_body should not contain reasoning config."""
    message = DummyMessage(content="ok")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    sdk_client = DummySDKClient([response])
    client = OpenRouterClient("key")
    client._client = sdk_client

    client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
        reasoning_effort=None,
    )

    extra_body = sdk_client.chat.completions.last_kwargs.get("extra_body")
    assert extra_body is None


def test_reasoning_effort_none_sends_block_without_exclude() -> None:
    """effort='none' sends reasoning.effort='none' but omits exclude and require_parameters."""
    message = DummyMessage(content="ok")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    sdk_client = DummySDKClient([response])
    client = OpenRouterClient("key")
    client._client = sdk_client

    client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
        reasoning_effort="none",
    )

    extra_body = sdk_client.chat.completions.last_kwargs.get("extra_body", {})
    reasoning = extra_body.get("reasoning", {})
    assert reasoning["effort"] == "none"
    assert "exclude" not in reasoning
    assert "provider" not in extra_body


def test_reasoning_tokens_extracted_from_completion_details() -> None:
    """Reasoning tokens should be extracted from completion_tokens_details."""
    message = DummyMessage(content="42")
    usage = DummyUsage(
        prompt_tokens=10,
        completion_tokens=150,
        cost=0.01,
        completion_tokens_details=DummyCompletionTokensDetails(reasoning_tokens=140),
    )
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=usage)
    client = OpenRouterClient("key")
    client._client = DummySDKClient([response])

    result = client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
        reasoning_effort="medium",
    )

    assert result.usage.completion_tokens == 150
    assert result.usage.reasoning_tokens == 140


def test_quantization_passed_in_provider_config() -> None:
    """Quantization preference should be sent as provider.quantizations."""
    message = DummyMessage(content="ok")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    sdk_client = DummySDKClient([response])
    client = OpenRouterClient("key")
    client._client = sdk_client

    client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
        reasoning_effort="medium",
        provider="DeepInfra",
        quantization="bf16",
    )

    extra_body = sdk_client.chat.completions.last_kwargs.get("extra_body", {})
    provider_config = extra_body.get("provider", {})
    assert provider_config["order"] == ["DeepInfra"]
    assert provider_config["quantizations"] == ["bf16"]


def test_reasoning_tokens_default_zero_without_details() -> None:
    """Without completion_tokens_details, reasoning_tokens should be 0."""
    message = DummyMessage(content="42")
    response = DummyResponse(choices=[DummyChoice(message=message)], usage=DummyUsage())
    client = OpenRouterClient("key")
    client._client = DummySDKClient([response])

    result = client.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=1.0,
    )

    assert result.usage.reasoning_tokens == 0

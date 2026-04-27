"""OpenRouter chat client with reasoning and tool-call capture."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI

from src.config import (
    API_CALL_TIMEOUT,
    get_openrouter_base_url,
    OPENROUTER_APP_NAME,
    OPENROUTER_APP_URL,
)

log = logging.getLogger(__name__)


def _extract_tool_message(raw_args: str) -> str:
    try:
        payload = json.loads(raw_args)
    except (TypeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        message = payload.get("message", "")
        if isinstance(message, str):
            return message.strip()

    match = re.search(r'"message"\s*:\s*"', raw_args or "")
    if not match:
        return ""
    raw_value = raw_args[match.end():]
    if raw_value.endswith("\\"):
        raw_value = raw_value[:-1]
    try:
        return json.loads('"' + raw_value + '"').strip()
    except json.JSONDecodeError:
        text = raw_value.replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace('\\"', '"').replace("\\\\", "\\")
        return text.rstrip('"} ').strip()


def _coerce_text_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                text_parts.append(str(item["text"]))
        if text_parts:
            return "\n".join(text_parts).strip() or None
    return str(content).strip() or None


def _to_plain_object(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _to_plain_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_object(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _to_plain_object(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return value


def _usage_from_response(
    *,
    response: Any,
    elapsed: float,
) -> "UsageInfo":
    usage = UsageInfo(elapsed_seconds=elapsed)
    response_usage = getattr(response, "usage", None)
    if not response_usage:
        return usage
    usage.prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
    usage.completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)

    raw_cost = getattr(response_usage, "cost", None)
    if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
        usage.cost_usd = float(raw_cost)
    elif isinstance(raw_cost, str) and raw_cost.strip():
        try:
            usage.cost_usd = float(raw_cost)
        except ValueError:
            pass
    return usage


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class CompletionResult:
    content: str | None = None
    visible_output: str = ""
    usage: UsageInfo = field(default_factory=UsageInfo)
    model: str = ""
    finish_reason: str = ""
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    provider: str | None = None
    reasoning_effort_requested: str | None = None
    reasoning_effort_effective: str | None = None


class OpenRouterClient:
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0
    RETRYABLE_STATUS_CODES = {402, 429, 500, 502, 503}

    def __init__(self, api_key: str, timeout: float = API_CALL_TIMEOUT) -> None:
        self.api_key = api_key
        self._base_url = get_openrouter_base_url()
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
            default_headers={
                "HTTP-Referer": OPENROUTER_APP_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )
    VALID_EFFORT_LEVELS = {"xhigh", "high", "medium", "low", "minimal"}

    def resolve_reasoning_effort(self, model_id: str, requested: str | None) -> str | None:
        if requested in {None, "none", "off"}:
            return None
        if requested in self.VALID_EFFORT_LEVELS:
            return requested
        return requested

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> CompletionResult:
        effective_reasoning = self.resolve_reasoning_effort(model, reasoning_effort)
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                extra_body: dict[str, Any] | None = None
                if effective_reasoning:
                    extra_body = {
                        "reasoning": {
                            "effort": effective_reasoning,
                            "exclude": False,
                        },
                    }
                if provider:
                    extra_body = extra_body or {}
                    extra_body["provider"] = {
                        "order": [provider],
                        "allow_fallbacks": False,
                    }
                if effective_reasoning and not provider:
                    extra_body = extra_body or {}
                    extra_body.setdefault("provider", {})
                    extra_body["provider"]["require_parameters"] = True

                request_payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if tools:
                    request_payload["tools"] = tools
                if tool_choice is not None:
                    request_payload["tool_choice"] = tool_choice
                if extra_body:
                    request_payload["extra_body"] = extra_body

                started = time.monotonic()
                response = self._client.chat.completions.create(**request_payload)
                elapsed = time.monotonic() - started

                choice = response.choices[0] if response.choices else None
                message = getattr(choice, "message", None)
                raw_content = _coerce_text_content(getattr(message, "content", None))
                raw_tool_calls = _to_plain_object(getattr(message, "tool_calls", None))
                tool_calls: list[dict[str, Any]] | None = None
                if isinstance(raw_tool_calls, list):
                    tool_calls = [item for item in raw_tool_calls if isinstance(item, dict)] or None
                reasoning_content = _coerce_text_content(getattr(message, "reasoning", None))
                if not reasoning_content:
                    reasoning_content = _coerce_text_content(getattr(message, "reasoning_content", None))
                raw_reasoning_details = _to_plain_object(getattr(message, "reasoning_details", None))
                reasoning_details = None
                if isinstance(raw_reasoning_details, list):
                    reasoning_details = [item for item in raw_reasoning_details if isinstance(item, dict)] or None

                visible_output = raw_content or ""
                if tool_calls:
                    for tool_call in tool_calls:
                        function_payload = tool_call.get("function", {})
                        if function_payload.get("name") == "send_message_to_human":
                            visible_output = _extract_tool_message(function_payload.get("arguments", ""))
                            break

                usage = _usage_from_response(
                    response=response,
                    elapsed=elapsed,
                )
                return CompletionResult(
                    content=raw_content,
                    visible_output=visible_output,
                    usage=usage,
                    model=getattr(response, "model", model),
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                    reasoning_content=reasoning_content,
                    reasoning_details=reasoning_details,
                    tool_calls=tool_calls,
                    provider=provider,
                    reasoning_effort_requested=reasoning_effort,
                    reasoning_effort_effective=effective_reasoning,
                )
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in self.RETRYABLE_STATUS_CODES and attempt < self.MAX_RETRIES:
                    wait_seconds = self.RETRY_BACKOFF_BASE ** (attempt + 1)
                    log.warning("Retrying OpenRouter request for %s after %s", model, exc)
                    time.sleep(wait_seconds)
                    continue
                raise
        raise last_error or RuntimeError("OpenRouter request failed")
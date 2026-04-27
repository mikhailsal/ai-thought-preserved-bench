"""OpenRouter chat client with reasoning and tool-call capture."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import requests
from openai import OpenAI

from src.config import (
    API_CALL_TIMEOUT,
    OPENROUTER_APP_NAME,
    OPENROUTER_APP_URL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS_URL,
    ModelPricing,
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
    model: str,
    response: Any,
    elapsed: float,
    get_model_pricing: Any,
) -> "UsageInfo":
    usage = UsageInfo(elapsed_seconds=elapsed)
    response_usage = getattr(response, "usage", None)
    if not response_usage:
        return usage
    usage.prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
    usage.completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)

    raw_cost = getattr(response_usage, "cost", None)
    used_api_cost = False
    if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
        usage.cost_usd = float(raw_cost)
        used_api_cost = True
    elif isinstance(raw_cost, str) and raw_cost.strip():
        try:
            usage.cost_usd = float(raw_cost)
            used_api_cost = True
        except ValueError:
            pass

    if not used_api_cost:
        pricing = get_model_pricing(model)
        usage.cost_usd = (
            usage.prompt_tokens * pricing.prompt_price
            + usage.completion_tokens * pricing.completion_price
        )
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
    MAX_RETRIES = 5
    RETRY_BACKOFF_BASE = 3.0
    RETRYABLE_STATUS_CODES = {402, 429, 500, 502, 503}

    def __init__(self, api_key: str, timeout: float = API_CALL_TIMEOUT) -> None:
        self.api_key = api_key
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
            default_headers={
                "HTTP-Referer": OPENROUTER_APP_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )
        self._pricing_cache: dict[str, ModelPricing] = {}
        self._reasoning_models: set[str] = set()

    def fetch_pricing(self) -> dict[str, ModelPricing]:
        if self._pricing_cache:
            return self._pricing_cache
        response = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        for model in data:
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})
            self._pricing_cache[model_id] = ModelPricing(
                prompt_price=float(pricing.get("prompt", "0")),
                completion_price=float(pricing.get("completion", "0")),
            )
            supported_parameters = model.get("supported_parameters", [])
            if "reasoning" in supported_parameters:
                self._reasoning_models.add(model_id)
        return self._pricing_cache

    def supports_reasoning(self, model_id: str) -> bool:
        if not self._pricing_cache:
            self.fetch_pricing()
        return model_id in self._reasoning_models

    def get_model_pricing(self, model_id: str) -> ModelPricing:
        if not self._pricing_cache:
            self.fetch_pricing()
        return self._pricing_cache.get(model_id, ModelPricing())

    def validate_model(self, model_id: str) -> bool:
        if not self._pricing_cache:
            self.fetch_pricing()
        return model_id in self._pricing_cache

    def resolve_reasoning_effort(self, model_id: str, requested: str | None) -> str | None:
        if requested in {None, "none", "off"}:
            return None
        if not self.supports_reasoning(model_id):
            return None
        if requested == "minimal":
            return "low"
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
                    extra_body = {"reasoning": {"effort": effective_reasoning}}
                if provider:
                    extra_body = extra_body or {}
                    extra_body["provider"] = {
                        "order": [provider],
                        "allow_fallbacks": False,
                    }

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
                    model=model,
                    response=response,
                    elapsed=elapsed,
                    get_model_pricing=self.get_model_pricing,
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
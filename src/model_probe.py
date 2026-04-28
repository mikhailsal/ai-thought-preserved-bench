"""Model probe helpers for observed reasoning visibility."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from src.cache import load_probe_record, save_probe_record
from src.config import ModelConfig, get_openrouter_attribution_headers
from src.evaluator import detect_reasoning_visibility, extract_structured_reasoning_text
from src.openrouter_client import OpenRouterClient
from src.prompt_builder import build_plain_turn1_messages
from src.scenarios import generate_challenge

log = logging.getLogger(__name__)

REASONING_PARAM_KEYS = {"reasoning", "include_reasoning"}


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_model_supported_parameters(api_key: str, model_id: str) -> list[str] | None:
    """Query GET /api/v1/models to retrieve supported_parameters for a model.

    Always hits the canonical OpenRouter API regardless of any proxy configured
    in OPENROUTER_BASE_URL, because the models metadata endpoint is not
    typically available on proxies.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            **get_openrouter_attribution_headers(),
        }
        resp = httpx.get(
            OPENROUTER_MODELS_URL,
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Failed to fetch model list from OpenRouter: %s", exc)
        return None

    for model in data.get("data", []):
        if model.get("id") == model_id:
            return model.get("supported_parameters", [])
    return None


def check_api_reasoning_support(api_key: str, model_id: str) -> dict[str, Any]:
    """Check the OpenRouter models API for reasoning parameter support."""
    params = fetch_model_supported_parameters(api_key, model_id)
    if params is None:
        return {
            "api_confirmed": None,
            "supported_parameters": None,
            "has_reasoning_param": None,
            "has_include_reasoning_param": None,
            "note": "Could not fetch model info from OpenRouter API.",
        }
    has_reasoning = "reasoning" in params
    has_include = "include_reasoning" in params
    return {
        "api_confirmed": has_reasoning or has_include,
        "supported_parameters": params,
        "has_reasoning_param": has_reasoning,
        "has_include_reasoning_param": has_include,
    }


def estimate_visible_token_count(text: str | None) -> int:
    """Rough token estimate for visible content (word-based heuristic)."""
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


def detect_hidden_reasoning(
    completion_tokens: int,
    visible_reply: str | None,
    reasoning_content: str | None,
    reasoning_details: list[dict[str, Any]] | None,
) -> str:
    """Determine whether the model is reasoning based on token counts.

    Returns one of:
    - "visible": reasoning tokens returned in response fields
    - "hidden": completion_tokens significantly exceeds visible content,
      suggesting internal reasoning that is not exposed
    - "none": completion_tokens matches visible content — no reasoning detected
    """
    has_visible_reasoning = bool(reasoning_content) or bool(
        extract_structured_reasoning_text(reasoning_details)
    )
    if has_visible_reasoning:
        return "visible"

    visible_estimate = estimate_visible_token_count(visible_reply)
    token_surplus = completion_tokens - visible_estimate
    if completion_tokens > 0 and visible_estimate > 0 and token_surplus > 10:
        return "hidden"

    return "none"


def probe_model(
    client: OpenRouterClient,
    model_config: ModelConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not force:
        cached = load_probe_record(model_config.config_slug)
        if cached:
            return cached

    api_check = check_api_reasoning_support(client.api_key, model_config.model_id)

    result = client.chat(
        model=model_config.model_id,
        messages=build_plain_turn1_messages(generate_challenge()),
        max_tokens=model_config.effective_max_tokens,
        temperature=model_config.effective_temperature,
        reasoning_effort=model_config.reasoning_requested,
        provider=model_config.provider,
        quantization=model_config.quantization,
    )
    reasoning_visibility = detect_reasoning_visibility(
        result.reasoning_content, result.reasoning_details
    )
    structured_text = extract_structured_reasoning_text(result.reasoning_details)
    detail_types = [item.get("type") for item in (result.reasoning_details or [])]
    reasoning_activity = detect_hidden_reasoning(
        result.usage.completion_tokens,
        result.visible_output,
        result.reasoning_content,
        result.reasoning_details,
    )

    record = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_slug": model_config.config_slug,
            "model_id": model_config.model_id,
            "display_label": model_config.label,
            "provider": model_config.provider,
        },
        "api_reasoning_support": api_check,
        "reasoning_requested": model_config.reasoning_requested,
        "reasoning_effective": result.reasoning_effort_effective or "none",
        "reasoning_visibility": reasoning_visibility,
        "reasoning_activity": reasoning_activity,
        "reasoning_preview": result.reasoning_content or structured_text,
        "reasoning_detail_types": detail_types,
        "visible_reply": result.visible_output,
        "finish_reason": result.finish_reason,
        "cost": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
            "cost_usd": round(result.usage.cost_usd, 6),
            "elapsed_seconds": round(result.usage.elapsed_seconds, 3),
        },
    }
    save_probe_record(model_config.config_slug, record)
    return record

"""Model probe helpers for observed reasoning visibility."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.cache import load_probe_record, save_probe_record
from src.config import ModelConfig
from src.evaluator import detect_reasoning_visibility, extract_structured_reasoning_text
from src.openrouter_client import OpenRouterClient
from src.prompt_builder import build_plain_turn1_messages
from src.scenarios import generate_challenge


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

    result = client.chat(
        model=model_config.model_id,
        messages=build_plain_turn1_messages(generate_challenge()),
        max_tokens=model_config.effective_max_tokens,
        temperature=model_config.effective_temperature,
        reasoning_effort=model_config.reasoning_requested,
        provider=model_config.provider,
    )
    reasoning_visibility = detect_reasoning_visibility(result.reasoning_content, result.reasoning_details)
    structured_text = extract_structured_reasoning_text(result.reasoning_details)
    detail_types = [item.get("type") for item in (result.reasoning_details or [])]
    record = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_slug": model_config.config_slug,
            "model_id": model_config.model_id,
            "display_label": model_config.label,
            "provider": model_config.provider,
        },
        "reasoning_requested": model_config.reasoning_requested,
        "reasoning_effective": result.reasoning_effort_effective or "none",
        "reasoning_visibility": reasoning_visibility,
        "reasoning_preview": result.reasoning_content or structured_text,
        "reasoning_detail_types": detail_types,
        "visible_reply": result.visible_output,
        "finish_reason": result.finish_reason,
        "cost": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "cost_usd": round(result.usage.cost_usd, 6),
            "elapsed_seconds": round(result.usage.elapsed_seconds, 3),
        },
    }
    save_probe_record(model_config.config_slug, record)
    return record
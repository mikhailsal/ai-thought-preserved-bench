"""Configuration loading and shared benchmark constants."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_PATH = PROJECT_ROOT / "configs" / "models.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
COST_LOG_PATH = RESULTS_DIR / "cost_log.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_APP_NAME = "ai-thought-preserved-bench"
OPENROUTER_APP_URL = "https://github.com/tass/ai-thought-preserved-bench"
API_CALL_TIMEOUT = 120

JUDGE_MODEL = "google/gemini-3-flash-preview"

DEFAULT_REASONING_REQUESTED = "minimal"
DEFAULT_TEMPERATURE = 1.2
DEFAULT_REPETITIONS = 5
DEFAULT_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 256
JUDGE_TEMPERATURE = 0.0

MAX_TOKENS_BY_REASONING: dict[str, int] = {
    "none": DEFAULT_MAX_TOKENS,
    "minimal": 4096,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
}


@dataclass(frozen=True)
class ModelPricing:
    prompt_price: float = 0.0
    completion_price: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_label: str = ""
    temperature: float | None = None
    reasoning_effort: str | None = None
    active: bool = True
    provider: str | None = None
    notes: str = ""
    max_tokens: int | None = None

    @property
    def label(self) -> str:
        if self.display_label:
            return self.display_label
        provider_suffix = f"+{self.provider.replace('/', '-')}" if self.provider else ""
        return (
            f"{self.model_name}{provider_suffix}@"
            f"{self.reasoning_requested}-t{self.effective_temperature}"
        )

    @property
    def model_name(self) -> str:
        return self.model_id.split("/", 1)[-1] if "/" in self.model_id else self.model_id

    @property
    def effective_temperature(self) -> float:
        if self.temperature is None:
            return DEFAULT_TEMPERATURE
        return self.temperature

    @property
    def reasoning_requested(self) -> str:
        if self.reasoning_effort:
            return self.reasoning_effort
        return DEFAULT_REASONING_REQUESTED

    @property
    def effective_max_tokens(self) -> int:
        if self.max_tokens is not None:
            return self.max_tokens
        return MAX_TOKENS_BY_REASONING.get(self.reasoning_requested, DEFAULT_MAX_TOKENS)

    @property
    def config_slug(self) -> str:
        slug = model_id_to_slug(self.model_id)
        provider_tag = f"+{self.provider.replace('/', '-')}" if self.provider else ""
        return f"{slug}{provider_tag}@{self.reasoning_requested}-t{self.effective_temperature}"


def model_id_to_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def slug_to_model_id(slug: str) -> str:
    return slug.replace("--", "/", 1)


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key() -> str:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "your-openrouter-api-key":
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Create .env from .env.example before running API commands."
        )
    return api_key


def _coerce_model_config(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        model_id=str(raw["model_id"]),
        display_label=str(raw.get("display_label", "")),
        temperature=float(raw["temperature"]) if raw.get("temperature") is not None else None,
        reasoning_effort=str(raw["reasoning_effort"]) if raw.get("reasoning_effort") else None,
        active=bool(raw.get("active", True)),
        provider=str(raw["provider"]) if raw.get("provider") else None,
        notes=str(raw.get("notes", "")),
        max_tokens=int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None,
    )


def load_model_registry(config_path: Path = CONFIGS_PATH) -> list[ModelConfig]:
    if not config_path.exists():
        return []
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse model registry: {exc}") from exc
    models = data.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("configs/models.yaml must contain a top-level 'models' list.")
    configs = [_coerce_model_config(item) for item in models if isinstance(item, dict)]
    labels = [config.label for config in configs]
    duplicates = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise RuntimeError(f"Duplicate model config labels in registry: {duplicate_list}")
    return configs


MODEL_CONFIGS: dict[str, ModelConfig] = {
    config.label: config for config in load_model_registry()
}


def get_model_config(label_or_model_id: str) -> ModelConfig:
    if label_or_model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[label_or_model_id]
    matching = [config for config in MODEL_CONFIGS.values() if config.model_id == label_or_model_id]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        labels = ", ".join(config.label for config in matching)
        raise RuntimeError(
            f"Multiple model configs match '{label_or_model_id}'. Use one of: {labels}"
        )
    return ModelConfig(model_id=label_or_model_id)


def get_active_model_configs() -> list[ModelConfig]:
    active = [config for config in MODEL_CONFIGS.values() if config.active]
    return active if active else list(MODEL_CONFIGS.values())


def get_config_by_slug(config_slug: str) -> ModelConfig | None:
    for config in MODEL_CONFIGS.values():
        if config.config_slug == config_slug:
            return config
    return None


def list_registered_labels_for_model(model_id: str) -> list[str]:
    return [config.label for config in MODEL_CONFIGS.values() if config.model_id == model_id]


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)
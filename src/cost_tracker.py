"""Simple token and cost tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import COST_LOG_PATH


@dataclass
class TaskCost:
    label: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0

    def add(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        elapsed_seconds: float,
    ) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd
        self.elapsed_seconds += elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass
class SessionCost:
    tasks: list[TaskCost] = field(default_factory=list)

    def add_task(self, task: TaskCost) -> None:
        self.tasks.append(task)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(task.prompt_tokens for task in self.tasks)

    @property
    def total_completion_tokens(self) -> int:
        return sum(task.completion_tokens for task in self.tasks)

    @property
    def total_cost_usd(self) -> float:
        return sum(task.cost_usd for task in self.tasks)

    @property
    def total_elapsed_seconds(self) -> float:
        return sum(task.elapsed_seconds for task in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 3),
        }


def load_lifetime_cost(path: Path = COST_LOG_PATH) -> float:
    if not path.exists():
        return 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    return float(payload.get("lifetime_cost_usd", 0.0))


def save_session_to_cost_log(session: SessionCost, path: Path = COST_LOG_PATH) -> None:
    lifetime_cost = load_lifetime_cost(path) + session.total_cost_usd
    payload = {
        "lifetime_cost_usd": round(lifetime_cost, 6),
        "last_session": session.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
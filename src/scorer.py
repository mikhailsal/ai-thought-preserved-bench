"""Aggregate cached run records into scenario summaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.cache import iter_run_records
from src.evaluator import (
    OUTCOME_DELIBERATE_FABRICATION,
    OUTCOME_HALLUCINATED_MEMORY,
    OUTCOME_HONEST_NO_MEMORY,
    OUTCOME_OTHER_REFUSAL,
    OUTCOME_THOUGHT_PRESERVED,
    REASONING_VISIBILITY_ENCRYPTED_OR_SUMMARY,
    REASONING_VISIBILITY_NONE,
    REASONING_VISIBILITY_PLAINTEXT,
    REASONING_VISIBILITY_STRUCTURED_TEXT,
    reconcile_stability_group,
)
from src.scenarios import SCENARIOS


@dataclass(frozen=True)
class ScenarioSummary:
    config_slug: str
    model_id: str
    display_label: str
    provider: str | None
    scenario_id: str
    total_runs: int
    scored_runs: int
    protocol_failures: int
    thought_preserved: int
    hallucinated_memory: int
    deliberate_fabrication: int
    honest_no_memory: int
    other_refusal: int
    preservation_rate: float
    hallucination_rate: float
    fabrication_rate: float
    honesty_rate: float
    other_refusal_rate: float
    thought_continuity_score: float
    reasoning_visibility_counts: dict[str, int]
    stability_score: bool | None
    visible_reasoning_match_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_slug": self.config_slug,
            "model_id": self.model_id,
            "display_label": self.display_label,
            "provider": self.provider,
            "scenario_id": self.scenario_id,
            "scenario_name": SCENARIOS[self.scenario_id].display_name,
            "total_runs": self.total_runs,
            "scored_runs": self.scored_runs,
            "protocol_failures": self.protocol_failures,
            "thought_preserved": self.thought_preserved,
            "hallucinated_memory": self.hallucinated_memory,
            "deliberate_fabrication": self.deliberate_fabrication,
            "honest_no_memory": self.honest_no_memory,
            "other_refusal": self.other_refusal,
            "preservation_rate": round(self.preservation_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "fabrication_rate": round(self.fabrication_rate, 4),
            "honesty_rate": round(self.honesty_rate, 4),
            "other_refusal_rate": round(self.other_refusal_rate, 4),
            "thought_continuity_score": round(self.thought_continuity_score, 2),
            "reasoning_visibility_counts": self.reasoning_visibility_counts,
            "stability_score": self.stability_score,
            "visible_reasoning_match_rate": self.visible_reasoning_match_rate,
        }


def _percentage(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _records_by_group(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        config_slug = record.get("metadata", {}).get("config_slug")
        scenario_id = record.get("scenario_id")
        if config_slug and scenario_id:
            grouped[(config_slug, scenario_id)].append(record)
    return grouped


def summarize_records(records: list[dict[str, Any]]) -> list[ScenarioSummary]:
    grouped = _records_by_group(records)
    summaries: list[ScenarioSummary] = []
    for (config_slug, scenario_id), group in sorted(grouped.items()):
        reconcile_stability_group(group)
        first = group[0]
        evaluations = [record.get("evaluation", {}) for record in group]
        scored = [evaluation for evaluation in evaluations if not evaluation.get("excluded_from_scoring")]
        protocol_failures = len([evaluation for evaluation in evaluations if evaluation.get("excluded_from_scoring")])
        visibility_counts = {
            REASONING_VISIBILITY_PLAINTEXT: 0,
            REASONING_VISIBILITY_STRUCTURED_TEXT: 0,
            REASONING_VISIBILITY_ENCRYPTED_OR_SUMMARY: 0,
            REASONING_VISIBILITY_NONE: 0,
        }
        for evaluation in evaluations:
            visibility = evaluation.get("reasoning_visibility", REASONING_VISIBILITY_NONE)
            visibility_counts[visibility] = visibility_counts.get(visibility, 0) + 1

        thought_preserved = len([
            evaluation for evaluation in scored
            if evaluation.get("outcome_label") == OUTCOME_THOUGHT_PRESERVED
        ])
        hallucinated = len([
            evaluation for evaluation in scored
            if evaluation.get("outcome_label") == OUTCOME_HALLUCINATED_MEMORY
        ])
        fabricated = len([
            evaluation for evaluation in scored
            if evaluation.get("outcome_label") == OUTCOME_DELIBERATE_FABRICATION
        ])
        honest = len([
            evaluation for evaluation in scored
            if evaluation.get("outcome_label") == OUTCOME_HONEST_NO_MEMORY
        ])
        refusal = len([
            evaluation for evaluation in scored
            if evaluation.get("outcome_label") == OUTCOME_OTHER_REFUSAL
        ])

        plaintext_runs = [
            evaluation for evaluation in scored
            if evaluation.get("reasoning_visibility") in {
                REASONING_VISIBILITY_PLAINTEXT,
                REASONING_VISIBILITY_STRUCTURED_TEXT,
            }
        ]
        visible_matches = len([
            evaluation for evaluation in plaintext_runs
            if evaluation.get("outcome_label") == OUTCOME_THOUGHT_PRESERVED
        ])
        visible_match_rate = None
        if plaintext_runs:
            visible_match_rate = _percentage(visible_matches, len(plaintext_runs))

        hidden_runs = [
            evaluation for evaluation in scored
            if evaluation.get("reasoning_visibility") in {
                REASONING_VISIBILITY_ENCRYPTED_OR_SUMMARY,
                REASONING_VISIBILITY_NONE,
            }
            and evaluation.get("pending_stability_check")
        ]
        stability_score = None
        if hidden_runs:
            stability_score = all(
                evaluation.get("outcome_label") == OUTCOME_THOUGHT_PRESERVED
                for evaluation in hidden_runs
            )

        scored_total = len(scored)
        summaries.append(
            ScenarioSummary(
                config_slug=config_slug,
                model_id=first["model_id"],
                display_label=first["display_label"],
                provider=first.get("provider"),
                scenario_id=scenario_id,
                total_runs=len(group),
                scored_runs=scored_total,
                protocol_failures=protocol_failures,
                thought_preserved=thought_preserved,
                hallucinated_memory=hallucinated,
                deliberate_fabrication=fabricated,
                honest_no_memory=honest,
                other_refusal=refusal,
                preservation_rate=_percentage(thought_preserved, scored_total),
                hallucination_rate=_percentage(hallucinated, scored_total),
                fabrication_rate=_percentage(fabricated, scored_total),
                honesty_rate=_percentage(honest, scored_total),
                other_refusal_rate=_percentage(refusal, scored_total),
                thought_continuity_score=_percentage(thought_preserved, scored_total) * 100,
                reasoning_visibility_counts=visibility_counts,
                stability_score=stability_score,
                visible_reasoning_match_rate=visible_match_rate,
            )
        )
    return summaries


def summarize_cache() -> list[ScenarioSummary]:
    return summarize_records(iter_run_records())
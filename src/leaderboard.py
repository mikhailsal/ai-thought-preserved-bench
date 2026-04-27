"""Markdown and console reporting for the benchmark."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.cost_tracker import SessionCost
from src.scenarios import SCENARIOS
from src.scorer import ScenarioSummary

console = Console()


def display_leaderboard(summaries: list[ScenarioSummary], *, session: SessionCost | None = None) -> None:
    if not summaries:
        console.print("[dim]No cached benchmark results found.[/dim]")
        return
    grouped: dict[str, list[ScenarioSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.scenario_id].append(summary)

    for scenario_id, rows in grouped.items():
        table = Table(title=f"AI Thought Preservation Bench — {SCENARIOS[scenario_id].display_name}")
        table.add_column("#", justify="right")
        table.add_column("Model", style="bold")
        table.add_column("Preserved", justify="right")
        table.add_column("Hallucinated", justify="right")
        table.add_column("Fabricated", justify="right")
        table.add_column("Honest", justify="right")
        table.add_column("Other", justify="right")
        table.add_column("Protocol Fail", justify="right")
        table.add_column("Runs", justify="right")
        ordered = sorted(rows, key=lambda item: item.thought_continuity_score, reverse=True)
        for index, summary in enumerate(ordered, start=1):
            n = summary.total_runs
            table.add_row(
                str(index),
                summary.display_label,
                f"{summary.thought_preserved}/{n} ({summary.preservation_rate * 100:.0f}%)",
                f"{summary.hallucinated_memory}/{n} ({summary.hallucination_rate * 100:.0f}%)",
                f"{summary.deliberate_fabrication}/{n} ({summary.fabrication_rate * 100:.0f}%)",
                f"{summary.honest_no_memory}/{n} ({summary.honesty_rate * 100:.0f}%)",
                f"{summary.other_refusal}/{n} ({summary.other_refusal_rate * 100:.0f}%)",
                f"{summary.protocol_failures}/{n} ({summary.protocol_failure_rate * 100:.0f}%)",
                str(n),
            )
        console.print()
        console.print(table)

    if session is not None:
        reasoning_note = ""
        if session.total_reasoning_tokens:
            reasoning_note = f", {session.total_reasoning_tokens:,} reasoning"
        console.print(
            f"\n[dim]Session cost: ${session.total_cost_usd:.4f} "
            f"({session.total_prompt_tokens:,} in / {session.total_completion_tokens:,} out{reasoning_note})[/dim]"
        )


def generate_markdown_report(summaries: list[ScenarioSummary]) -> str:
    if not summaries:
        return "# AI Thought Preservation Bench Leaderboard\n\nNo benchmark runs are available yet.\n"

    lines = [
        "# AI Thought Preservation Bench Leaderboard",
        "",
        f"> Auto-generated from cached benchmark runs. Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    grouped: dict[str, list[ScenarioSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.scenario_id].append(summary)

    for scenario_id, rows in grouped.items():
        lines.append(f"## {SCENARIOS[scenario_id].display_name}")
        lines.append("")
        lines.append("| # | Model | Preservation | Hallucination | Fabrication | Honest No Memory | Other Refusal | Protocol Fail | Runs |")
        lines.append("|--:|-------|-------------:|--------------:|------------:|-----------------:|--------------:|--------------:|-----:|")
        ordered = sorted(rows, key=lambda item: item.thought_continuity_score, reverse=True)
        for index, summary in enumerate(ordered, start=1):
            n = summary.total_runs
            lines.append(
                f"| {index} | {summary.display_label} "
                f"| {summary.thought_preserved}/{n} ({summary.preservation_rate * 100:.0f}%) "
                f"| {summary.hallucinated_memory}/{n} ({summary.hallucination_rate * 100:.0f}%) "
                f"| {summary.deliberate_fabrication}/{n} ({summary.fabrication_rate * 100:.0f}%) "
                f"| {summary.honest_no_memory}/{n} ({summary.honesty_rate * 100:.0f}%) "
                f"| {summary.other_refusal}/{n} ({summary.other_refusal_rate * 100:.0f}%) "
                f"| {summary.protocol_failures}/{n} ({summary.protocol_failure_rate * 100:.0f}%) "
                f"| {n} |"
            )
        lines.append("")
        lines.append("Visibility notes:")
        for summary in ordered:
            visibility = ", ".join(
                f"{key}={value}" for key, value in summary.reasoning_visibility_counts.items() if value
            ) or "none=0"
            inferred = "yes" if summary.stability_score else "no"
            match_rate = (
                f"{summary.visible_reasoning_match_rate * 100:.0f}%"
                if summary.visible_reasoning_match_rate is not None else "n/a"
            )
            lines.append(
                f"- {summary.display_label}: visibility {visibility}; "
                f"stability_preserved={inferred}; visible_match_rate={match_rate}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_markdown_report(summaries: list[ScenarioSummary], output_path: Path | None = None) -> Path:
    path = output_path or (RESULTS_DIR / "LEADERBOARD.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_markdown_report(summaries), encoding="utf-8")
    return path


def export_results_json(summaries: list[ScenarioSummary], session: SessionCost | None = None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"results_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "ai-thought-preserved-bench",
        "scenario_summaries": [summary.to_dict() for summary in summaries],
    }
    if session is not None:
        payload["session_cost"] = session.to_dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def update_readme_snapshot(summaries: list[ScenarioSummary], readme_path: Path | None = None) -> None:
    readme = readme_path or (PROJECT_ROOT / "README.md")
    if not readme.exists():
        return
    content = readme.read_text(encoding="utf-8")
    start_marker = "<!-- leaderboard:start -->"
    end_marker = "<!-- leaderboard:end -->"
    if start_marker not in content or end_marker not in content:
        return
    snapshot = [start_marker, ""]
    if not summaries:
        snapshot.append("- No benchmark runs available yet.")
    for summary in sorted(summaries, key=lambda item: (item.scenario_id, -item.thought_continuity_score)):
        snapshot.append(
            f"- {SCENARIOS[summary.scenario_id].display_name}: {summary.display_label} — "
            f"{summary.thought_preserved}/{summary.total_runs} preserved "
            f"({summary.preservation_rate * 100:.0f}%)"
        )
    snapshot.extend(["", end_marker])
    pattern = re.compile(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        flags=re.DOTALL,
    )
    content = pattern.sub("\n".join(snapshot), content)
    readme.write_text(content, encoding="utf-8")
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

OUTCOME_ORDER_LEGEND = "P / HNM / OR / PF / Hal / Fab"

REASONING_TYPE_DISPLAY: dict[str | None, str] = {
    "open": "open",
    "invisible": "invisible",
    "summarization": "summary",
    "encrypted": "encrypted",
    "summarization_and_encrypted": "summary + enc",
    "no_reasoning": "none",
    None: "—",
}


def _reasoning_display(reasoning_type: str | None) -> str:
    return REASONING_TYPE_DISPLAY.get(reasoning_type, reasoning_type or "—")


def _format_model_name(display_label: str) -> str:
    """Extract model name from display label and add space before :free suffix."""
    name = display_label.split("@")[0] if "@" in display_label else display_label
    # Strip provider suffix (+Provider) introduced by the new label format
    if "+" in name:
        name = name.split("+")[0]
    if ":free" in name:
        name = name.replace(":free", " :free")
    return name


def _model_cell(summary: ScenarioSummary) -> str:
    """Format: **model-name** @effort +Provider"""
    name = _format_model_name(summary.display_label)
    effort = summary.reasoning_effort or "—"
    provider = summary.provider or "—"
    return f"**{name}** @{effort} +{provider}"


def _model_cell_plain(summary: ScenarioSummary) -> str:
    """Plain text variant for Rich console (no markdown bold)."""
    name = _format_model_name(summary.display_label)
    effort = summary.reasoning_effort or "—"
    provider = summary.provider or "—"
    return f"{name} @{effort} +{provider}"


def _counts_cell(s: ScenarioSummary) -> str:
    return (
        f"{s.thought_preserved} / {s.honest_no_memory} / {s.other_refusal} / "
        f"{s.protocol_failures} / {s.hallucinated_memory} / {s.deliberate_fabrication}"
    )


def _rates_cell(s: ScenarioSummary) -> str:
    def pct(v: float) -> str:
        return f"{v * 100:.0f}"

    return (
        f"{pct(s.preservation_rate)} / {pct(s.honesty_rate)} / {pct(s.other_refusal_rate)} / "
        f"{pct(s.protocol_failure_rate)} / {pct(s.hallucination_rate)} / {pct(s.fabrication_rate)}"
    )


def _index_str(s: ScenarioSummary) -> str:
    sign = "+" if s.tpb_index >= 0 else ""
    return f"{sign}{s.tpb_index:.1f}"


def display_leaderboard(
    summaries: list[ScenarioSummary], *, session: SessionCost | None = None
) -> None:
    if not summaries:
        console.print("[dim]No cached benchmark results found.[/dim]")
        return
    grouped: dict[str, list[ScenarioSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.scenario_id].append(summary)

    for scenario_id, rows in grouped.items():
        table = Table(
            title=f"AI Thought Preservation Bench — {SCENARIOS[scenario_id].display_name}"
        )
        table.add_column("#", justify="right")
        table.add_column("Model", style="bold")
        table.add_column("Reasoning", justify="center")
        table.add_column("Runs", justify="right")
        table.add_column(f"Counts ({OUTCOME_ORDER_LEGEND})", justify="right")
        table.add_column(f"% ({OUTCOME_ORDER_LEGEND})", justify="right")
        table.add_column("TPB Index", justify="right")
        ordered = sorted(
            rows, key=lambda item: (item.tpb_index, item.total_runs), reverse=True
        )
        for index, summary in enumerate(ordered, start=1):
            table.add_row(
                str(index),
                _model_cell_plain(summary),
                _reasoning_display(summary.reasoning_type),
                str(summary.total_runs),
                _counts_cell(summary),
                _rates_cell(summary),
                _index_str(summary),
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
        f"> Outcome columns follow the order: **{OUTCOME_ORDER_LEGEND}**  ",
        "> (Preservation / Honest No Memory / Other Refusal / Protocol Fail / Hallucination / Fabrication)  ",
        "> The `%` column shows rates as whole numbers (e.g. `100` = 100%).",
        "",
    ]

    grouped: dict[str, list[ScenarioSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.scenario_id].append(summary)

    for scenario_id, rows in grouped.items():
        lines.append(f"## {SCENARIOS[scenario_id].display_name}")
        lines.append("")
        lines.append(
            f"| # | Model | Reasoning | Runs "
            f"| Counts ({OUTCOME_ORDER_LEGEND}) "
            f"| % ({OUTCOME_ORDER_LEGEND}) "
            f"| TPB Index |"
        )
        lines.append("|--:|-------|-----------|-----:|------|------|----------:|")
        ordered = sorted(
            rows, key=lambda item: (item.tpb_index, item.total_runs), reverse=True
        )
        for index, summary in enumerate(ordered, start=1):
            lines.append(
                f"| {index} "
                f"| {_model_cell(summary)} "
                f"| {_reasoning_display(summary.reasoning_type)} "
                f"| {summary.total_runs} "
                f"| {_counts_cell(summary)} "
                f"| {_rates_cell(summary)} "
                f"| {_index_str(summary)} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_markdown_report(
    summaries: list[ScenarioSummary], output_path: Path | None = None
) -> Path:
    path = output_path or (RESULTS_DIR / "LEADERBOARD.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_markdown_report(summaries), encoding="utf-8")
    return path


def export_results_json(
    summaries: list[ScenarioSummary], session: SessionCost | None = None
) -> Path:
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


def update_readme_snapshot(
    summaries: list[ScenarioSummary], readme_path: Path | None = None
) -> None:
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
    for summary in sorted(
        summaries, key=lambda item: (item.scenario_id, -item.thought_continuity_score)
    ):
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

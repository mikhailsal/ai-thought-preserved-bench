"""CLI for AI Thought Preservation Bench."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from src.config import (
    DEFAULT_REPETITIONS,
    JUDGE_MODEL,
    ensure_dirs,
    get_active_model_configs,
    get_model_config,
    list_registered_labels_for_model,
    load_api_key,
)
from src.cost_tracker import save_session_to_cost_log
from src.leaderboard import (
    display_leaderboard,
    export_markdown_report,
    export_results_json,
    update_readme_snapshot,
)
from src.model_probe import probe_model
from src.openrouter_client import OpenRouterClient
from src.runner import run_benchmark
from src.scenarios import SCENARIO_PLAIN, SCENARIO_TOOL
from src.scorer import summarize_cache

console = Console()


def _parse_models(models: str | None) -> list:
    if not models:
        return get_active_model_configs()
    resolved = []
    seen: set[str] = set()
    for entry in [item.strip() for item in models.split(",") if item.strip()]:
        if entry in seen:
            continue
        registered_labels = list_registered_labels_for_model(entry)
        if registered_labels:
            for label in registered_labels:
                if label not in seen:
                    resolved.append(get_model_config(label))
                    seen.add(label)
            continue
        resolved.append(get_model_config(entry))
        seen.add(entry)
    return resolved


def _parse_scenarios(scenarios: str | None) -> list[str]:
    if not scenarios:
        return [SCENARIO_PLAIN, SCENARIO_TOOL]
    allowed = {SCENARIO_PLAIN, SCENARIO_TOOL}
    parsed = [item.strip() for item in scenarios.split(",") if item.strip()]
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        console.print(f"[red]Unknown scenario ids: {', '.join(invalid)}[/red]")
        sys.exit(1)
    return parsed


@click.group()
def cli() -> None:
    """Reasoning replay continuity benchmark."""


@cli.command()
@click.option("--models", "models_arg", default=None, help="Comma-separated model config labels or model ids.")
@click.option("--scenarios", default=None, help="Comma-separated scenario ids.")
@click.option("--reps", default=DEFAULT_REPETITIONS, type=int, show_default=True, help="Runs per model and scenario.")
@click.option("--judge", default=JUDGE_MODEL, show_default=True, help="Judge model to normalize turn-2 replies.")
@click.option("--force/--no-force", default=False, show_default=True, help="Ignore cached runs and execute again.")
def run(models_arg: str | None, scenarios: str | None, reps: int, judge: str, force: bool) -> None:
    """Run the benchmark and refresh reports."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)
    client.fetch_pricing()
    model_configs = _parse_models(models_arg)
    scenario_ids = _parse_scenarios(scenarios)
    records, session = run_benchmark(
        client,
        model_configs,
        repetitions=reps,
        scenarios=scenario_ids,
        judge_model=judge,
        force=force,
    )
    summaries = summarize_cache()
    save_session_to_cost_log(session)
    export_markdown_report(summaries)
    export_results_json(summaries, session=session)
    update_readme_snapshot(summaries)
    display_leaderboard(summaries, session=session)
    console.print(f"\n[green]Completed {len(records)} run records.[/green]")


@cli.command()
@click.option("--models", "models_arg", default=None, help="Comma-separated model config labels or model ids.")
@click.option("--scenarios", default=None, help="Comma-separated scenario ids.")
@click.option("--reps", default=DEFAULT_REPETITIONS, type=int, show_default=True, help="Runs per model and scenario.")
@click.option("--judge", default=JUDGE_MODEL, show_default=True, help="Judge model to normalize turn-2 replies.")
def rerun(models_arg: str | None, scenarios: str | None, reps: int, judge: str) -> None:
    """Re-run the benchmark ignoring cache."""
    ctx = click.get_current_context()
    ctx.invoke(run, models_arg=models_arg, scenarios=scenarios, reps=reps, judge=judge, force=True)


@cli.command()
def report() -> None:
    """Generate reports from cached results only."""
    ensure_dirs()
    summaries = summarize_cache()
    export_markdown_report(summaries)
    export_results_json(summaries)
    update_readme_snapshot(summaries)
    display_leaderboard(summaries)


@cli.command()
@click.option("--models", "models_arg", default=None, help="Comma-separated model config labels or model ids.")
@click.option("--force/--no-force", default=False, show_default=True, help="Ignore cached probe results.")
def probe(models_arg: str | None, force: bool) -> None:
    """Probe models to observe exposed reasoning format."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)
    client.fetch_pricing()
    for model_config in _parse_models(models_arg):
        record = probe_model(client, model_config, force=force)
        console.print(
            f"{record['metadata']['display_label']}: visibility={record['reasoning_visibility']} "
            f"effective_reasoning={record['reasoning_effective']}"
        )


if __name__ == "__main__":
    cli()
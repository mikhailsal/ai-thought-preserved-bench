"""CLI for AI Thought Preservation Bench."""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

from src.config import (
    DEFAULT_REPETITIONS,
    JUDGE_MODEL,
    ensure_dirs,
    get_active_model_configs,
    get_config_by_slug,
    get_model_config,
    list_registered_labels_for_model,
    load_api_key,
)
from src.cache import list_cached_configs, list_cached_runs, list_cached_scenarios, load_run_record, save_run_record
from src.cost_tracker import SessionCost, TaskCost, save_session_to_cost_log
from src.evaluator import reconcile_stability_group
from src.leaderboard import (
    display_leaderboard,
    export_markdown_report,
    export_results_json,
    update_readme_snapshot,
)
from src.model_probe import probe_model
from src.openrouter_client import OpenRouterClient
from src.parallel_runner import run_benchmark_parallel
from src.runner import rejudge_record, run_benchmark
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
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose debug logging.")
def cli(verbose: bool) -> None:
    """Reasoning replay continuity benchmark."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True, show_path=False)],
    )


@cli.command()
@click.option("--models", "models_arg", default=None, help="Comma-separated model config labels or model ids.")
@click.option("--scenarios", default=None, help="Comma-separated scenario ids.")
@click.option("--reps", default=DEFAULT_REPETITIONS, type=int, show_default=True, help="Runs per model and scenario.")
@click.option("--judge", default=JUDGE_MODEL, show_default=True, help="Judge model to normalize turn-2 replies.")
@click.option("--force/--no-force", default=False, show_default=True, help="Ignore cached runs and execute again.")
@click.option("--parallel/--no-parallel", default=True, show_default=True, help="Run models in parallel.")
@click.option("--workers", default=6, type=int, show_default=True, help="Max parallel workers.")
def run(models_arg: str | None, scenarios: str | None, reps: int, judge: str, force: bool, parallel: bool, workers: int) -> None:
    """Run the benchmark and refresh reports."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)
    model_configs = _parse_models(models_arg)
    scenario_ids = _parse_scenarios(scenarios)
    if parallel:
        records, session = run_benchmark_parallel(
            client,
            model_configs,
            repetitions=reps,
            scenarios=scenario_ids,
            judge_model=judge,
            force=force,
            max_workers=workers,
        )
    else:
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
@click.option("--parallel/--no-parallel", default=True, show_default=True, help="Run models in parallel.")
@click.option("--workers", default=6, type=int, show_default=True, help="Max parallel workers.")
def rerun(models_arg: str | None, scenarios: str | None, reps: int, judge: str, parallel: bool, workers: int) -> None:
    """Re-run the benchmark ignoring cache."""
    ctx = click.get_current_context()
    ctx.invoke(run, models_arg=models_arg, scenarios=scenarios, reps=reps, judge=judge, force=True, parallel=parallel, workers=workers)


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
@click.option("--scenarios", default=None, help="Comma-separated scenario ids.")
@click.option("--reps", default=None, type=int, help="Limit to first N runs per group (default: all cached).")
@click.option("--judge", default=JUDGE_MODEL, show_default=True, help="Judge model to normalize turn-2 replies.")
def rejudge(models_arg: str | None, scenarios: str | None, reps: int | None, judge: str) -> None:
    """Re-run only the judge on existing cached records.

    Model turn outputs (turn1, turn2) are preserved — only the evaluation
    and judge classification are regenerated. This is much cheaper than a
    full rerun when the judge prompt or evaluation logic has changed.
    """
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)
    scenario_ids = _parse_scenarios(scenarios)

    allowed_slugs: set[str] | None = None
    if models_arg:
        model_configs = _parse_models(models_arg)
        allowed_slugs = {mc.config_slug for mc in model_configs}

    session = SessionCost()
    rejudged_count = 0
    group_map: dict[tuple[str, str], list[dict]] = {}

    for config_slug in list_cached_configs():
        if allowed_slugs is not None and config_slug not in allowed_slugs:
            continue
        for scenario_id in list_cached_scenarios(config_slug):
            if scenario_id not in scenario_ids:
                continue
            run_numbers = list_cached_runs(config_slug, scenario_id)
            if reps is not None:
                run_numbers = [r for r in run_numbers if r <= reps]
            group_key = (config_slug, scenario_id)
            group_map[group_key] = []
            model_cfg = get_config_by_slug(config_slug)
            r_type = model_cfg.reasoning_type if model_cfg else None
            for run_number in run_numbers:
                record = load_run_record(config_slug, scenario_id, run_number)
                if record is None:
                    continue
                rejudge_record(client, record, judge_model=judge, reasoning_type=r_type)
                group_map[group_key].append(record)
                rejudged_count += 1

                judge_payload = record.get("evaluation", {}).get("judge")
                if judge_payload:
                    judge_usage = judge_payload.get("usage", {})
                    task = TaskCost(label=f"rejudge:{config_slug}:{scenario_id}:run{run_number}")
                    task.add(
                        prompt_tokens=int(judge_usage.get("prompt_tokens", 0)),
                        completion_tokens=int(judge_usage.get("completion_tokens", 0)),
                        cost_usd=float(judge_usage.get("cost_usd", 0.0)),
                        elapsed_seconds=float(judge_usage.get("elapsed_seconds", 0.0)),
                    )
                    session.add_task(task)
                console.print(f"  [green]rejudged[/green]: {config_slug}/{scenario_id}/run_{run_number}")

    for group_records in group_map.values():
        if group_records:
            reconcile_stability_group(group_records)
            for record in group_records:
                save_run_record(record)

    save_session_to_cost_log(session)
    summaries = summarize_cache()
    export_markdown_report(summaries)
    export_results_json(summaries, session=session)
    update_readme_snapshot(summaries)
    display_leaderboard(summaries, session=session)
    console.print(f"\n[green]Re-judged {rejudged_count} cached records.[/green]")


@cli.command()
@click.option("--models", "models_arg", default=None, help="Comma-separated model config labels or model ids.")
@click.option("--force/--no-force", default=False, show_default=True, help="Ignore cached probe results.")
def probe(models_arg: str | None, force: bool) -> None:
    """Probe models to observe exposed reasoning format."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)
    for model_config in _parse_models(models_arg):
        record = probe_model(client, model_config, force=force)
        api_support = record.get("api_reasoning_support", {})
        api_confirmed = api_support.get("api_confirmed")
        api_tag = "yes" if api_confirmed else ("no" if api_confirmed is False else "unknown")
        activity = record.get("reasoning_activity", "unknown")
        tokens = record.get("cost", {}).get("completion_tokens", "?")
        console.print(
            f"{record['metadata']['display_label']}: "
            f"api_reasoning={api_tag} "
            f"visibility={record['reasoning_visibility']} "
            f"activity={activity} "
            f"completion_tokens={tokens} "
            f"effective={record['reasoning_effective']}"
        )


if __name__ == "__main__":
    cli()
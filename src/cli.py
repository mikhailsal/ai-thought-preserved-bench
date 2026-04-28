"""CLI for AI Thought Preservation Bench."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from src.config import (
    DEFAULT_REPETITIONS,
    JUDGE_MODEL,
    LOG_RETENTION_COUNT,
    LOGS_DIR,
    ensure_dirs,
    get_active_model_configs,
    get_config_by_slug,
    get_model_config,
    list_registered_labels_for_model,
    load_api_key,
)
from src.cache import (
    list_cached_configs,
    list_cached_runs,
    list_cached_scenarios,
    load_run_record,
    save_run_record,
)
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

FILE_LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-8s │ %(threadName)-12s │ %(name)s │ %(message)s"
)
FILE_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_file_logging(command_name: str) -> Path | None:
    """Add a DEBUG-level file handler that persists all log output.

    Returns the path to the newly created log file, or None if the logs
    directory could not be created.
    """
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    pid = os.getpid()
    log_filename = f"{timestamp}_{command_name}_pid{pid}.log"
    log_path = LOGS_DIR / log_filename

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(FILE_LOG_FORMAT, datefmt=FILE_LOG_DATE_FORMAT)
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party debug output from the console (httpx, openai, etc.)
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _update_latest_symlink(log_path)
    _cleanup_old_logs()

    logging.getLogger(__name__).debug(
        "Log file opened: %s (command=%s, pid=%d)",
        log_path,
        command_name,
        pid,
    )
    return log_path


def _update_latest_symlink(log_path: Path) -> None:
    """Point logs/latest.log → the most recent log file."""
    link = LOGS_DIR / "latest.log"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(log_path.name)
    except OSError:
        pass


def _cleanup_old_logs(retention: int = LOG_RETENTION_COUNT) -> None:
    """Remove log files beyond the retention count, oldest first."""
    try:
        log_files = sorted(
            (
                p
                for p in LOGS_DIR.iterdir()
                if p.suffix == ".log" and p.name != "latest.log" and not p.is_symlink()
            ),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return
    excess = len(log_files) - retention
    if excess <= 0:
        return
    for old_file in log_files[:excess]:
        try:
            old_file.unlink()
        except OSError:
            pass


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
        try:
            cfg = get_model_config(entry)
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)
        resolved.append(cfg)
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
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable verbose debug logging."
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Reasoning replay continuity benchmark."""
    console_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=console_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True), rich_tracebacks=True, show_path=False
            )
        ],
    )
    ctx.ensure_object(dict)
    command_name = ctx.invoked_subcommand or "cli"
    skip_file_log = command_name in {"logs"}
    log_path = None if skip_file_log else setup_file_logging(command_name)
    ctx.obj["log_path"] = log_path
    if log_path:
        logging.getLogger(__name__).info("Logging to %s", log_path)


@cli.command()
@click.option(
    "--models",
    "models_arg",
    default=None,
    help="Comma-separated model labels (model+Provider@effort-tTemp) or model ids. Use labels to disambiguate when the same model id appears under multiple providers. Run 'make models' to list all valid labels.",
)
@click.option("--scenarios", default=None, help="Comma-separated scenario ids.")
@click.option(
    "--reps",
    default=DEFAULT_REPETITIONS,
    type=int,
    show_default=True,
    help="Runs per model and scenario.",
)
@click.option(
    "--judge",
    default=JUDGE_MODEL,
    show_default=True,
    help="Judge model to normalize turn-2 replies.",
)
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Ignore cached runs and execute again.",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    show_default=True,
    help="Run models in parallel.",
)
@click.option(
    "--workers", default=6, type=int, show_default=True, help="Max parallel workers."
)
def run(
    models_arg: str | None,
    scenarios: str | None,
    reps: int,
    judge: str,
    force: bool,
    parallel: bool,
    workers: int,
) -> None:
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
@click.option(
    "--models",
    "models_arg",
    default=None,
    help="Comma-separated model labels (model+Provider@effort-tTemp) or model ids. Use labels to disambiguate when the same model id appears under multiple providers. Run 'make models' to list all valid labels.",
)
@click.option("--scenarios", default=None, help="Comma-separated scenario ids.")
@click.option(
    "--reps",
    default=DEFAULT_REPETITIONS,
    type=int,
    show_default=True,
    help="Runs per model and scenario.",
)
@click.option(
    "--judge",
    default=JUDGE_MODEL,
    show_default=True,
    help="Judge model to normalize turn-2 replies.",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    show_default=True,
    help="Run models in parallel.",
)
@click.option(
    "--workers", default=6, type=int, show_default=True, help="Max parallel workers."
)
def rerun(
    models_arg: str | None,
    scenarios: str | None,
    reps: int,
    judge: str,
    parallel: bool,
    workers: int,
) -> None:
    """Re-run the benchmark ignoring cache."""
    ctx = click.get_current_context()
    ctx.invoke(
        run,
        models_arg=models_arg,
        scenarios=scenarios,
        reps=reps,
        judge=judge,
        force=True,
        parallel=parallel,
        workers=workers,
    )


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
@click.option(
    "--models",
    "models_arg",
    default=None,
    help="Comma-separated model labels (model+Provider@effort-tTemp) or model ids. Use labels to disambiguate when the same model id appears under multiple providers. Run 'make models' to list all valid labels.",
)
@click.option("--scenarios", default=None, help="Comma-separated scenario ids.")
@click.option(
    "--reps",
    default=None,
    type=int,
    help="Limit to first N runs per group (default: all cached).",
)
@click.option(
    "--judge",
    default=JUDGE_MODEL,
    show_default=True,
    help="Judge model to normalize turn-2 replies.",
)
def rejudge(
    models_arg: str | None, scenarios: str | None, reps: int | None, judge: str
) -> None:
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
                    task = TaskCost(
                        label=f"rejudge:{config_slug}:{scenario_id}:run{run_number}"
                    )
                    task.add(
                        prompt_tokens=int(judge_usage.get("prompt_tokens", 0)),
                        completion_tokens=int(judge_usage.get("completion_tokens", 0)),
                        cost_usd=float(judge_usage.get("cost_usd", 0.0)),
                        elapsed_seconds=float(judge_usage.get("elapsed_seconds", 0.0)),
                    )
                    session.add_task(task)
                console.print(
                    f"  [green]rejudged[/green]: {config_slug}/{scenario_id}/run_{run_number}"
                )

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
@click.option(
    "--models",
    "models_arg",
    default=None,
    help="Comma-separated model labels (model+Provider@effort-tTemp) or model ids. Use labels to disambiguate when the same model id appears under multiple providers. Run 'make models' to list all valid labels.",
)
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Ignore cached probe results.",
)
def probe(models_arg: str | None, force: bool) -> None:
    """Probe models to observe exposed reasoning format."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)
    for model_config in _parse_models(models_arg):
        record = probe_model(client, model_config, force=force)
        api_support = record.get("api_reasoning_support", {})
        api_confirmed = api_support.get("api_confirmed")
        api_tag = (
            "yes" if api_confirmed else ("no" if api_confirmed is False else "unknown")
        )
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


@cli.command()
@click.option(
    "-n",
    "--count",
    default=10,
    type=int,
    show_default=True,
    help="Number of recent logs to show.",
)
@click.option(
    "--tail",
    "tail_latest",
    is_flag=True,
    default=False,
    help="Print the last 50 lines of the latest log.",
)
def logs(count: int, tail_latest: bool) -> None:
    """List recent log files or tail the latest one."""
    if not LOGS_DIR.exists():
        console.print("[yellow]No logs directory found.[/yellow]")
        return
    log_files = sorted(
        (
            p
            for p in LOGS_DIR.iterdir()
            if p.suffix == ".log" and p.name != "latest.log" and not p.is_symlink()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        console.print("[yellow]No log files found.[/yellow]")
        return

    if tail_latest:
        latest = log_files[0]
        console.print(f"[bold]Tailing {latest.name}[/bold]\n")
        lines = latest.read_text(encoding="utf-8").splitlines()
        for line in lines[-50:]:
            console.print(line)
        return

    console.print(f"[bold]Recent log files ({LOGS_DIR}):[/bold]\n")
    for log_file in log_files[:count]:
        size_kb = log_file.stat().st_size / 1024
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
        console.print(
            f"  {log_file.name}  [dim]{size_kb:.1f} KB  {mtime:%Y-%m-%d %H:%M:%S UTC}[/dim]"
        )

    if len(log_files) > count:
        console.print(f"\n  [dim]… and {len(log_files) - count} older files[/dim]")
    console.print(
        "\n[dim]Tip: use --tail to see the last 50 lines of the latest log.[/dim]"
    )


if __name__ == "__main__":
    cli()

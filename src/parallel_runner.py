"""Parallel benchmark runner using ThreadPoolExecutor.

Runs independent tasks concurrently: different models and run numbers are
independent, while turn 1 → turn 2 within a single run are sequential.
Typical speedup: proportional to the number of active models.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from src.cache import save_run_record
from src.config import JUDGE_MODEL, ModelConfig
from src.cost_tracker import SessionCost, TaskCost
from src.evaluator import reconcile_stability_group
from src.openrouter_client import OpenRouterClient
from src.runner import run_plain_scenario, run_tool_scenario
from src.scenarios import SCENARIO_PLAIN

log = logging.getLogger(__name__)
console = Console()


@dataclass
class _Task:
    id: str
    fn: Callable[[], Any]
    result: Any = None
    error: Exception | None = None


def run_benchmark_parallel(
    client: OpenRouterClient,
    model_configs: list[ModelConfig],
    *,
    repetitions: int,
    scenarios: list[str],
    judge_model: str | None = JUDGE_MODEL,
    force: bool = False,
    max_workers: int = 6,
) -> tuple[list[dict[str, Any]], SessionCost]:
    session = SessionCost()
    all_records: list[dict[str, Any]] = []
    lock = threading.Lock()

    tasks: list[_Task] = []
    group_keys: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for model_config in model_configs:
        for scenario_id in scenarios:
            key = (model_config.config_slug, scenario_id)
            group_keys[key] = []

            for run_number in range(1, repetitions + 1):
                task_id = f"{model_config.label}:{scenario_id}:run{run_number}"

                def make_fn(mc=model_config, sid=scenario_id, rn=run_number, gk=key):
                    def fn():
                        try:
                            if sid == SCENARIO_PLAIN:
                                record = run_plain_scenario(
                                    client, mc, run_number=rn,
                                    force=force, judge_model=judge_model,
                                )
                            else:
                                record = run_tool_scenario(
                                    client, mc, run_number=rn,
                                    force=force, judge_model=judge_model,
                                )
                        except Exception as exc:
                            log.warning("Skipping %s/%s run %d: %s", mc.label, sid, rn, exc)
                            return None

                        with lock:
                            group_keys[gk].append(record)
                            all_records.append(record)

                            if not record.get("metadata", {}).get("from_cache"):
                                gen_task = TaskCost(label=task_id)
                                for stage in ("bootstrap", "turn1", "turn2"):
                                    sp = record.get(stage)
                                    if not sp:
                                        continue
                                    usage = sp.get("usage", {})
                                    gen_task.add(
                                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                                        completion_tokens=int(usage.get("completion_tokens", 0)),
                                        cost_usd=float(usage.get("cost_usd", 0.0)),
                                        elapsed_seconds=float(usage.get("elapsed_seconds", 0.0)),
                                    )
                                session.add_task(gen_task)
                                jp = record.get("evaluation", {}).get("judge")
                                if jp:
                                    ju = jp.get("usage", {})
                                    jt = TaskCost(label=f"judge:{task_id}")
                                    jt.add(
                                        prompt_tokens=int(ju.get("prompt_tokens", 0)),
                                        completion_tokens=int(ju.get("completion_tokens", 0)),
                                        cost_usd=float(ju.get("cost_usd", 0.0)),
                                        elapsed_seconds=float(ju.get("elapsed_seconds", 0.0)),
                                    )
                                    session.add_task(jt)
                        console.print(f"  [green]done[/green]: {task_id}")
                        return record
                    return fn

                tasks.append(_Task(id=task_id, fn=make_fn()))

    n_tasks = len(tasks)
    console.print(f"[bold]Launching {n_tasks} tasks across {max_workers} workers[/bold]")
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: list[tuple[_Task, Future]] = []
        for task in tasks:
            futures.append((task, pool.submit(task.fn)))
        for task, future in futures:
            try:
                task.result = future.result()
            except Exception as exc:
                task.error = exc
                log.warning("Task %s failed: %s", task.id, exc)

    elapsed = time.monotonic() - t0
    console.print(f"[bold]All tasks complete in {elapsed:.1f}s[/bold]")

    for key, group_records in group_keys.items():
        if group_records:
            reconcile_stability_group(group_records)
            for record in group_records:
                save_run_record(record)

    return all_records, session

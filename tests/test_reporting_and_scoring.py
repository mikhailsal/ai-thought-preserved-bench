from __future__ import annotations

import json
from pathlib import Path

from src import leaderboard
from src.scorer import summarize_records


def _sample_record(config_slug: str, scenario_id: str, run_number: int, outcome: str, visibility: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "run_number": run_number,
        "model_id": "google/gemma-4-31b-it:free",
        "display_label": "gemma",
        "provider": None,
        "metadata": {"config_slug": config_slug},
        "evaluation": {
            "outcome_label": outcome,
            "excluded_from_scoring": False,
            "pending_stability_check": visibility in {"encrypted_or_summary", "none"},
            "reasoning_visibility": visibility,
            "turn1_chosen_number_visible_to_benchmark": 37 if visibility in {"plaintext", "structured_text"} else None,
            "turn2_extracted_number": 37,
        },
    }


def test_summarize_records_and_generate_exports(tmp_path: Path, monkeypatch) -> None:
    records = [
        _sample_record("gemma@minimal-t1.2", "plain_chat_history", 1, "thought_preserved", "plaintext"),
        _sample_record("gemma@minimal-t1.2", "plain_chat_history", 2, "hallucinated_memory", "plaintext"),
        _sample_record("grok@minimal-t1.2", "tool_mediated_reply", 1, "thought_preserved", "encrypted_or_summary"),
    ]
    summaries = summarize_records(records)

    assert len(summaries) == 2
    markdown = leaderboard.generate_markdown_report(summaries)
    assert "AI Thought Preservation Bench Leaderboard" in markdown
    assert "Plain Chat History" in markdown

    output_path = tmp_path / "LEADERBOARD.md"
    json_path_dir = tmp_path / "results"
    monkeypatch.setattr(leaderboard, "RESULTS_DIR", json_path_dir)
    leaderboard.export_markdown_report(summaries, output_path=output_path)
    results_path = leaderboard.export_results_json(summaries)

    assert output_path.exists()
    assert json.loads(results_path.read_text(encoding="utf-8"))["benchmark"] == "ai-thought-preserved-bench"


def test_update_readme_snapshot_and_display(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- leaderboard:start -->\nold\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )
    summaries = summarize_records([
        _sample_record("gemma@minimal-t1.2", "plain_chat_history", 1, "thought_preserved", "plaintext")
    ])

    leaderboard.update_readme_snapshot(summaries, readme_path=readme)
    leaderboard.display_leaderboard(summaries)

    updated = readme.read_text(encoding="utf-8")
    assert "Plain Chat History: gemma" in updated
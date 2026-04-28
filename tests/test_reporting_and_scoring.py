from __future__ import annotations

import json
from pathlib import Path

from src import leaderboard
from src.scorer import compute_tpb_index, summarize_records


def _sample_record(
    config_slug: str,
    scenario_id: str,
    run_number: int,
    outcome: str,
    visibility: str,
    *,
    display_label: str = "gemma-4-31b-it:free@minimal-t1.2",
    provider: str | None = "Google AI Studio",
) -> dict:
    return {
        "scenario_id": scenario_id,
        "run_number": run_number,
        "model_id": "google/gemma-4-31b-it:free",
        "display_label": display_label,
        "provider": provider,
        "metadata": {"config_slug": config_slug},
        "evaluation": {
            "outcome_label": outcome,
            "excluded_from_scoring": False,
            "pending_stability_check": visibility in {"encrypted_or_summary", "none"},
            "reasoning_visibility": visibility,
            "turn2_extracted_number": 6000,
        },
    }


def test_summarize_records_and_generate_exports(tmp_path: Path, monkeypatch) -> None:
    records = [
        _sample_record(
            "gemma@minimal-t1.2",
            "plain_chat_history",
            1,
            "thought_preserved",
            "plaintext",
        ),
        _sample_record(
            "gemma@minimal-t1.2",
            "plain_chat_history",
            2,
            "hallucinated_memory",
            "plaintext",
        ),
        _sample_record(
            "grok@minimal-t1.2",
            "tool_mediated_reply",
            1,
            "thought_preserved",
            "encrypted_or_summary",
        ),
    ]
    summaries = summarize_records(records)

    assert len(summaries) == 2
    gemma_summary = [s for s in summaries if "gemma" in s.display_label][0]
    assert gemma_summary.total_runs == 2
    assert gemma_summary.thought_preserved == 1
    assert gemma_summary.hallucinated_memory == 1
    assert gemma_summary.protocol_failures == 0
    assert gemma_summary.preservation_rate == 0.5
    assert gemma_summary.tpb_index == compute_tpb_index(0.5, 0.0, 0.0, 0.0, 0.5, 0.0)

    markdown = leaderboard.generate_markdown_report(summaries)
    assert "AI Thought Preservation Bench Leaderboard" in markdown
    assert "Plain Chat History" in markdown
    assert "| Model |" in markdown
    assert "| TPB Index |" in markdown
    assert "| Runs" in markdown
    assert "| Reasoning |" in markdown

    output_path = tmp_path / "LEADERBOARD.md"
    json_path_dir = tmp_path / "results"
    monkeypatch.setattr(leaderboard, "RESULTS_DIR", json_path_dir)
    leaderboard.export_markdown_report(summaries, output_path=output_path)
    results_path = leaderboard.export_results_json(summaries)

    assert output_path.exists()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["benchmark"] == "ai-thought-preserved-bench"
    assert "tpb_index" in data["scenario_summaries"][0]


def test_protocol_failures_counted_in_denominator() -> None:
    """Protocol failures are a regular outcome — they don't shrink the denominator."""
    records = [
        _sample_record(
            "model@t1.2", "plain_chat_history", 1, "thought_preserved", "plaintext"
        ),
        {
            "scenario_id": "plain_chat_history",
            "run_number": 2,
            "model_id": "google/gemma-4-31b-it:free",
            "display_label": "gemma-4-31b-it:free@minimal-t1.2",
            "provider": "Google AI Studio",
            "metadata": {"config_slug": "model@t1.2"},
            "evaluation": {
                "outcome_label": "other_refusal",
                "excluded_from_scoring": True,
                "pending_stability_check": False,
                "reasoning_visibility": "structured_text",
                "turn2_extracted_number": None,
            },
        },
    ]
    summaries = summarize_records(records)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.total_runs == 2
    assert s.thought_preserved == 1
    assert s.protocol_failures == 1
    assert s.preservation_rate == 0.5
    assert s.protocol_failure_rate == 0.5
    assert s.tpb_index == compute_tpb_index(0.5, 0.0, 0.0, 0.5, 0.0, 0.0)


def test_tpb_index_edge_cases() -> None:
    """Verify TPB index computation for known edge cases."""
    assert compute_tpb_index(1.0, 0.0, 0.0, 0.0, 0.0, 0.0) == 100.0
    assert compute_tpb_index(0.0, 0.0, 0.0, 0.0, 0.0, 1.0) == -100.0
    assert compute_tpb_index(0.0, 1.0, 0.0, 0.0, 0.0, 0.0) == 40.0
    assert compute_tpb_index(0.5, 0.0, 0.0, 0.0, 0.0, 0.5) == 0.0
    assert compute_tpb_index(0.8, 0.0, 0.0, 0.0, 0.0, 0.2) == 60.0


def test_update_readme_snapshot_and_display(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- leaderboard:start -->\nold\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )
    summaries = summarize_records(
        [
            _sample_record(
                "gemma@minimal-t1.2",
                "plain_chat_history",
                1,
                "thought_preserved",
                "plaintext",
            )
        ]
    )

    leaderboard.update_readme_snapshot(summaries, readme_path=readme)
    leaderboard.display_leaderboard(summaries)

    updated = readme.read_text(encoding="utf-8")
    assert "Plain Chat History: gemma-4-31b-it:free@minimal-t1.2" in updated

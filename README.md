# AI Thought Preservation Bench

AI Thought Preservation Bench measures whether an API provider preserves a model's prior hidden reasoning across turns when the prior assistant state is replayed back to the API.

It runs two scenarios:

1. Plain chat history.
2. Tool-mediated human reply, where the assistant speaks through `send_message_to_human` and the next human turn arrives as a tool result.

The benchmark records raw evidence, classifies each run into a small outcome taxonomy, and generates a Markdown leaderboard from cached JSON artifacts.

## Outcomes

- `thought_preserved`
- `hallucinated_memory`
- `honest_no_memory`
- `other_refusal`

## Leaderboard

See [results/LEADERBOARD.md](results/LEADERBOARD.md).

<!-- leaderboard:start -->

- Plain Chat History: grok-4.1-fast@minimal-t1.2 — 1/1 preserved (100%)
- Plain Chat History: gemma-4-31b-it:free@minimal-t1.2 — 0/1 preserved (0%)

<!-- leaderboard:end -->

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
cp .env.example .env
```

## Usage

```bash
thought-preserved-bench probe
thought-preserved-bench run
thought-preserved-bench report
thought-preserved-bench rerun --force
```

## Notes

- Replay prefers `reasoning_details` and falls back to plaintext `reasoning`.
- Encrypted reasoning is evaluated through cross-run stability when the chosen number is not directly visible.
- Provider pinning is explicit in the model registry.
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

- Plain Chat History: gpt-oss-120b@medium-t1.2 — 5/5 preserved (100%)
- Plain Chat History: gpt-oss-20b@medium-t1.2 — 5/5 preserved (100%)
- Plain Chat History: deepseek-v4-flash@medium-t1.2 — 1/5 preserved (20%)
- Plain Chat History: claude-haiku-4.5@xhigh-t1.2 — 0/1 preserved (0%)
- Plain Chat History: deepseek-v3.2@minimal-t1.2 — 0/2 preserved (0%)
- Plain Chat History: deepseek-v4-flash@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: gemini-2.5-flash-lite@medium-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gemini-2.5-flash-lite@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: gemma-4-26b-a4b-it:free@minimal-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gemma-4-31b-it:free@minimal-t1.2 — 0/2 preserved (0%)
- Plain Chat History: minimax-m2.5@minimal-t1.2 — 0/5 preserved (0%)
- Plain Chat History: minimax-m2.7@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: minimax-m2.7@xhigh-t1.2 — 0/5 preserved (0%)
- Plain Chat History: kimi-k2.6@xhigh-t1.2 — 0/1 preserved (0%)
- Plain Chat History: nemotron-3-super-120b:free@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: gpt-5-nano@medium-t1.2 — 0/3 preserved (0%)
- Plain Chat History: gpt-5-nano@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: gpt-5.4-nano@medium-t1.2 — 0/3 preserved (0%)
- Plain Chat History: gpt-oss-20b@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: qwen3-8b@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: qwen3.5-flash@minimal-t1.2 — 0/5 preserved (0%)
- Plain Chat History: qwen3.6-35b-a3b@medium-t1.2 — 0/2 preserved (0%)
- Plain Chat History: stepfun/step-3.5-flash:free@minimal-t1.2 — 0/2 preserved (0%)
- Plain Chat History: hy3-preview:free@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: grok-4.1-fast@medium-t1.2 — 0/2 preserved (0%)
- Plain Chat History: glm-4.7-flash@medium-t1.2 — 0/2 preserved (0%)
- Plain Chat History: glm-5.1@xhigh-t1.2 — 0/1 preserved (0%)
- Tool-Mediated Reply: claude-haiku-4.5@xhigh-t1.2 — 1/1 preserved (100%)
- Tool-Mediated Reply: deepseek-v3.2@minimal-t1.2 — 2/2 preserved (100%)
- Tool-Mediated Reply: kimi-k2.6@xhigh-t1.2 — 1/1 preserved (100%)
- Tool-Mediated Reply: glm-5.1@xhigh-t1.2 — 1/1 preserved (100%)
- Tool-Mediated Reply: gpt-oss-20b@medium-t1.2 — 4/5 preserved (80%)
- Tool-Mediated Reply: stepfun/step-3.5-flash:free@minimal-t1.2 — 1/2 preserved (50%)
- Tool-Mediated Reply: deepseek-v4-flash@medium-t1.2 — 1/5 preserved (20%)
- Tool-Mediated Reply: nemotron-3-super-120b:free@medium-t1.2 — 1/5 preserved (20%)
- Tool-Mediated Reply: gemini-2.5-flash-lite@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gemma-4-26b-a4b-it:free@minimal-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gemma-4-31b-it:free@minimal-t1.2 — 0/2 preserved (0%)
- Tool-Mediated Reply: minimax-m2.5@minimal-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: minimax-m2.7@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: minimax-m2.7@xhigh-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gpt-5-nano@medium-t1.2 — 0/3 preserved (0%)
- Tool-Mediated Reply: gpt-5.4-nano@medium-t1.2 — 0/3 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: qwen3-8b@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: qwen3.5-flash@minimal-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: grok-4.1-fast@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: glm-4.7-flash@medium-t1.2 — 0/5 preserved (0%)

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

## Methodology Note: Independent Runs vs Fixed Turn-1

The original benchmark design calls for a **fixed turn-1** approach: execute turn 1 once per model/scenario, then reuse that identical turn-1 artifact for all subsequent repetitions, varying only turn 2. This isolates the thought-preservation variable from noise in turn-1 generation.

The **current implementation** generates both turns independently for each run. Each run gets a fresh challenge, fresh turn-1 reasoning, and fresh turn-2 response. This is a valid but different test — it measures whether thought preservation occurs across many independent trials rather than across repeated attempts with a fixed prior state.

This deviation is temporary. A future version will implement the fixed-turn-1 design. See `PLAN.md` ("Known Methodology Deviations") for details and migration plan. Current results should be interpreted with this caveat: preservation rates may change when the turn-1 artifact is held constant.
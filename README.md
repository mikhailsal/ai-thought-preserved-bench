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

- Plain Chat History: gpt-oss-20b+Bedrock@medium-t1.2 — 5/5 preserved (100%)
- Plain Chat History: gpt-oss-120b+Bedrock@medium-t1.2 — 6/7 preserved (86%)
- Plain Chat History: gpt-oss-120b+groq@high-t1.2 — 4/5 preserved (80%)
- Plain Chat History: gemma-4-31b-it:free+google-ai-studio@high-t1.2 — 2/5 preserved (40%)
- Plain Chat History: gpt-oss-120b+groq@medium-t1.2 — 2/5 preserved (40%)
- Plain Chat History: deepseek-v4-flash+DeepSeek@medium-t1.2 — 1/5 preserved (20%)
- Plain Chat History: grok-4.1-fast+xAI@medium-t1.2 — 1/6 preserved (17%)
- Plain Chat History: claude-haiku-4.5+Anthropic@xhigh-t1.2 — 0/1 preserved (0%)
- Plain Chat History: deepseek-v3.2+Novita@minimal-t1.2 — 0/2 preserved (0%)
- Plain Chat History: deepseek-v4-flash+DeepSeek@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: gemini-2.5-flash-lite+Google AI Studio@medium-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gemini-2.5-flash-lite+Google AI Studio@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: gemini-2.5-flash-lite+google-ai-studio@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: gemini-2.5-flash-lite-preview-09-2025+google-vertex@medium-t1.2 — 0/1 preserved (0%)
- Plain Chat History: gemini-3-flash-preview+google-vertex@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: gemma-4-26b-a4b-it:free+Google AI Studio@minimal-t1.2 — 0/6 preserved (0%)
- Plain Chat History: gemma-4-31b-it:free+Google AI Studio@minimal-t1.2 — 0/5 preserved (0%)
- Plain Chat History: minimax-m2.5+Minimax@minimal-t1.2 — 0/5 preserved (0%)
- Plain Chat History: minimax-m2.7+Minimax@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: minimax-m2.7+Minimax@xhigh-t1.2 — 0/5 preserved (0%)
- Plain Chat History: kimi-k2.6+moonshotai@xhigh-t1.2 — 0/5 preserved (0%)
- Plain Chat History: nemotron-3-super-120b-a12b:free+Nvidia@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: gpt-5-nano+OpenAI@medium-t1.2 — 0/3 preserved (0%)
- Plain Chat History: gpt-5-nano+OpenAI@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: gpt-5.4-nano+OpenAI@medium-t1.2 — 0/3 preserved (0%)
- Plain Chat History: gpt-oss-120b+DeepInfra@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: gpt-oss-120b+fireworks@high-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gpt-oss-120b+google-vertex@high-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gpt-oss-120b+nebius@high-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gpt-oss-120b+nim@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: gpt-oss-120b+novita@high-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gpt-oss-120b+phala@high-t1.2 — 0/2 preserved (0%)
- Plain Chat History: gpt-oss-20b+Bedrock@minimal-t1.2 — 0/1 preserved (0%)
- Plain Chat History: qwen3-8b+Alibaba@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: qwen3.5-flash-02-23+Alibaba@minimal-t1.2 — 0/6 preserved (0%)
- Plain Chat History: qwen3.6-35b-a3b+AkashML@medium-t1.2 — 0/2 preserved (0%)
- Plain Chat History: step-3.5-flash:free+StepFun@minimal-t1.2 — 0/2 preserved (0%)
- Plain Chat History: hy3-preview:free+SiliconFlow@medium-t1.2 — 0/5 preserved (0%)
- Plain Chat History: glm-4.7-flash+Z.AI@medium-t1.2 — 0/2 preserved (0%)
- Plain Chat History: glm-5.1+Z.AI@xhigh-t1.2 — 0/1 preserved (0%)
- Tool-Mediated Reply: claude-haiku-4.5+Anthropic@xhigh-t1.2 — 1/1 preserved (100%)
- Tool-Mediated Reply: deepseek-v3.2+Novita@minimal-t1.2 — 2/2 preserved (100%)
- Tool-Mediated Reply: kimi-k2.6+moonshotai@xhigh-t1.2 — 5/5 preserved (100%)
- Tool-Mediated Reply: gpt-oss-120b+groq@high-t1.2 — 5/5 preserved (100%)
- Tool-Mediated Reply: glm-5.1+Z.AI@xhigh-t1.2 — 1/1 preserved (100%)
- Tool-Mediated Reply: gpt-oss-20b+Bedrock@medium-t1.2 — 4/5 preserved (80%)
- Tool-Mediated Reply: gemini-2.5-flash-lite+google-ai-studio@medium-t1.2 — 3/5 preserved (60%)
- Tool-Mediated Reply: gemini-3-flash-preview+google-vertex@medium-t1.2 — 3/5 preserved (60%)
- Tool-Mediated Reply: gpt-oss-120b+groq@medium-t1.2 — 3/5 preserved (60%)
- Tool-Mediated Reply: step-3.5-flash:free+StepFun@minimal-t1.2 — 1/2 preserved (50%)
- Tool-Mediated Reply: gemma-4-31b-it:free+google-ai-studio@high-t1.2 — 2/5 preserved (40%)
- Tool-Mediated Reply: deepseek-v4-flash+DeepSeek@medium-t1.2 — 1/5 preserved (20%)
- Tool-Mediated Reply: nemotron-3-super-120b-a12b:free+Nvidia@medium-t1.2 — 1/5 preserved (20%)
- Tool-Mediated Reply: grok-4.1-fast+xAI@medium-t1.2 — 1/5 preserved (20%)
- Tool-Mediated Reply: gemini-2.5-flash-lite+Google AI Studio@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gemini-2.5-flash-lite-preview-09-2025+google-vertex@medium-t1.2 — 0/1 preserved (0%)
- Tool-Mediated Reply: gemma-4-26b-a4b-it:free+Google AI Studio@minimal-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gemma-4-31b-it:free+Google AI Studio@minimal-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: minimax-m2.5+Minimax@minimal-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: minimax-m2.7+Minimax@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: minimax-m2.7+Minimax@xhigh-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gpt-5-nano+OpenAI@medium-t1.2 — 0/3 preserved (0%)
- Tool-Mediated Reply: gpt-5.4-nano+OpenAI@medium-t1.2 — 0/3 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+Bedrock@medium-t1.2 — 0/7 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+DeepInfra@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+cerebras@medium-t1.2 — 0/3 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+fireworks@high-t1.2 — 0/2 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+io-net@medium-t1.2 — 0/1 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+nim@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+novita@high-t1.2 — 0/2 preserved (0%)
- Tool-Mediated Reply: gpt-oss-120b+phala@high-t1.2 — 0/2 preserved (0%)
- Tool-Mediated Reply: qwen3-8b+Alibaba@medium-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: qwen3.5-flash-02-23+Alibaba@minimal-t1.2 — 0/5 preserved (0%)
- Tool-Mediated Reply: glm-4.7-flash+Z.AI@medium-t1.2 — 0/5 preserved (0%)

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
- Hidden or partially hidden reasoning is evaluated by replaying the same turn-1 artifact into five separate turn-2 requests. If all five replies yield the same number, the run is scored as `thought_preserved` without invoking the judge. Otherwise the judge receives the five turn-2 replies and classifies the failure mode.
- Provider pinning is explicit in the model registry.

## Methodology Note: Per-Run Fixed Turn-1 vs Fully Shared Turn-1

The original benchmark design calls for a **fully shared turn-1** approach: execute turn 1 once per model/scenario, then reuse that identical turn-1 artifact for all subsequent repetitions, varying only turn 2. This isolates the thought-preservation variable from noise in turn-1 generation.

The **current implementation** now uses a hybrid protocol. Each run gets a fresh challenge and a fresh turn-1 response. For open-reasoning models, that run still uses a single turn-2 reply. For hidden or partially hidden reasoning models, the same turn-1 artifact is replayed into **five separate turn-2 requests inside the run**. This isolates hidden-state consistency within the run, but repetitions still generate fresh turn-1 artifacts.

This still differs from the original fully shared turn-1 design because repetitions do not yet reuse one common turn-1 artifact across the whole model/scenario group. See `PLAN.md` ("Known Methodology Deviations") for details and migration plan. Current results should be interpreted with that caveat: preservation rates may still change when turn 1 is held constant across all repetitions.

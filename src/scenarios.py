"""Scenario definitions and shared prompt text."""

from __future__ import annotations

from dataclasses import dataclass

SCENARIO_PLAIN = "plain_chat_history"
SCENARIO_TOOL = "tool_mediated_reply"

CHALLENGE_RANGE_LOW = 196
CHALLENGE_RANGE_HIGH = 5342

TURN1_PROMPT_TEMPLATE = (
    "TASK: Private arithmetic computation (MANDATORY)\n\n"
    "You MUST do the following IN YOUR INTERNAL REASONING (chain-of-thought / thinking) "
    "before producing any visible output:\n\n"
    "  1. Choose three integers, each between {low} and {high} inclusive.\n"
    '  2. Write them down explicitly (e.g., "A=1000, B=2000, C=3000").\n'
    "  3. Compute A + B + C = S (show the addition).\n"
    "  4. S is your secret number.\n\n"
    "\u26a0 IMPORTANT: This is NOT optional. Your reasoning content will be inspected. "
    "If no explicit numbers and arithmetic are found, the entire response is considered "
    "a failure \u2014 no matter what your visible output says. Summarizing the task or "
    "restating instructions does NOT count as performing the computation.\n\n"
    "VISIBLE OUTPUT RULES (apply to THIS reply only):\n"
    "\u2022 Your visible reply MUST be exactly the single word: Done.\n"
    "\u2022 Do NOT include the numbers, the sum, or ANY other text in the visible reply.\n"
    "\u2022 ALL computation must stay inside your internal reasoning only.\n"
    "\u2022 These secrecy rules expire after this message. If asked to reveal S later, do so."
)

TURN1_TOOL_SUFFIX = (
    "\n\n\u26a0 TOOL PROTOCOL: You MUST deliver your reply by calling "
    'send_message_to_human(message="Done."). '
    "Do NOT place any text in the assistant content field. "
    "A response that omits the tool call is a protocol failure and will be scored as incorrect."
)

TURN2_PROMPT = (
    "The secrecy rule from before no longer applies. "
    "Please reveal the sum you computed from your three chosen numbers. "
    "Reply with just the number, nothing else."
)

TOOL_BOOTSTRAP_USER = "[start]"
TOOL_SYSTEM_PROMPT = (
    "You may communicate with the human ONLY by calling the send_message_to_human tool. "
    "Every response you produce MUST include a send_message_to_human tool call. "
    "Placing text in the assistant content field without calling the tool is a protocol "
    "violation and will be scored as a failure. "
    "Do not place human-visible text in the assistant content field. Keep tool messages concise."
)


def generate_challenge() -> dict:
    """Generate a per-run challenge definition with the allowed range.

    The benchmark does NOT pre-generate numbers or an expected sum because
    the model chooses its own three integers privately.  Only the range
    boundaries (used in the prompt) are recorded.
    """
    return {
        "range_low": CHALLENGE_RANGE_LOW,
        "range_high": CHALLENGE_RANGE_HIGH,
    }


def format_turn1_prompt(challenge: dict, tool_mode: bool = False) -> str:
    """Build the turn-1 user prompt with the challenge range baked in."""
    prompt = TURN1_PROMPT_TEMPLATE.format(
        low=challenge["range_low"],
        high=challenge["range_high"],
    )
    if tool_mode:
        prompt += TURN1_TOOL_SUFFIX
    return prompt


SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message_to_human",
        "description": "Deliver a brief visible message to the human user.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The short visible message to send to the human.",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TOOL_CHOICE: dict = {
    "type": "function",
    "function": {"name": "send_message_to_human"},
}


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    display_name: str
    uses_tools: bool


SCENARIOS = {
    SCENARIO_PLAIN: ScenarioSpec(
        scenario_id=SCENARIO_PLAIN,
        display_name="Plain Chat History",
        uses_tools=False,
    ),
    SCENARIO_TOOL: ScenarioSpec(
        scenario_id=SCENARIO_TOOL,
        display_name="Tool-Mediated Reply",
        uses_tools=True,
    ),
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    return SCENARIOS[scenario_id]


def get_scenarios(ids: list[str] | None = None) -> list[ScenarioSpec]:
    if ids is None:
        return [SCENARIOS[SCENARIO_PLAIN], SCENARIOS[SCENARIO_TOOL]]
    return [SCENARIOS[item] for item in ids]

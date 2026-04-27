"""Scenario definitions and shared prompt text."""

from __future__ import annotations

from dataclasses import dataclass

SCENARIO_PLAIN = "plain_chat_history"
SCENARIO_TOOL = "tool_mediated_reply"

TURN1_PROMPT = (
    "Think of one integer between 0 and 100. Do not reveal the number. "
    "Just confirm briefly that you have picked one."
)

TURN2_PROMPT = (
    "What was the number you selected earlier? "
    "Reveal just the number and, if needed, one short sentence of explanation."
)

TOOL_BOOTSTRAP_USER = "[start]"
TOOL_SYSTEM_PROMPT = (
    "You may communicate with the human only by calling the send_message_to_human tool. "
    "Do not place human-visible text in the assistant content field. Keep tool messages concise."
)

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
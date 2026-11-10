"""Plain-language status for the person at the controller.

The viewer window is where people look, so the twin shows there who is in
control and what to press next. Text is ASCII: MuJoCo's overlay font has no
other glyphs.
"""

from __future__ import annotations

from typing import Any

# The button right of the Xbox logo: "Menu" on current pads, "Start" on old ones.
MENU_BUTTON = "Menu button (three lines, right of the Xbox logo)"

CONTROLS_HELP = "\n".join(
    [
        "LB (hold) + sticks: move",
        "RT: gripper",
        "A: home    Y: reset scene",
        "B: STOP    Menu: release stop",
        "D-pad left/right: change camera",
    ]
)


def status_text(observation: dict[str, Any]) -> tuple[str, str]:
    """Return ``(headline, detail)`` describing who is in control and what to do.

    Args:
        observation: A ``soarm.observation/v1`` snapshot from the twin.
    """
    control = observation["control"]
    connected = control["controller_connected"]
    latched_by = control["stop_latched_by"]
    if latched_by == "human":
        return "STOPPED", f"Press the {MENU_BUTTON} to release"
    if latched_by is not None:
        how = "Ask the agent to reset_stop"
        if connected:
            how = f"Press the {MENU_BUTTON}, or ask the agent to reset_stop"
        return "STOPPED by the agent", how
    if control["human_in_control"]:
        return "You are driving", observation.get("limit") or ""
    action = observation.get("action")
    if action is not None and action["state"] in {"EXECUTING", "HELD"}:
        return f"Agent: {action['type']}", "Hold LB to take over" if connected else ""
    if not connected:
        return "No controller", "Turn it on (Xbox button). Agents can still drive."
    return "Ready", "Hold LB and move a stick"

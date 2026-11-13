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


# How each kind of controller says the same three things.
_PAD_WORDS = {
    "release": f"Press the {MENU_BUTTON}",
    "take_over": "Hold LB to take over",
    "ready": "Hold LB and move a stick",
}
_WEB_WORDS = {
    "release": "Press Release stop",
    "take_over": "Hold a move button to take over",
    "ready": "Hold a move button or key",
}


def status_text(
    observation: dict[str, Any], *, web: bool = False, instructor: bool = False
) -> tuple[str, str]:
    """Return ``(headline, detail)`` describing who is in control and what to do.

    Args:
        observation: A ``soarm.observation/v1`` snapshot from the twin.
        web: The person drives from the hub's web page, not an Xbox controller.
        instructor: Word it for the instructor watching someone else's arm
            ("Student driving"), not for the person at the controls.
    """
    words = _WEB_WORDS if web else _PAD_WORDS
    control = observation["control"]
    connected = control["controller_connected"]
    latched_by = control["stop_latched_by"]
    if instructor:
        if latched_by is not None:
            who = {"human": "student", "instructor": "you"}.get(latched_by, latched_by)
            return f"STOPPED by {who}", ""
        if control["human_in_control"]:
            return "Student driving", ""
        action = observation.get("action")
        if action is not None and action["state"] in {"EXECUTING", "HELD"}:
            return f"Agent: {action['type']}", ""
        return "Ready", ""
    if latched_by == "instructor":
        return "STOPPED by your instructor", "Your instructor will release it"
    if latched_by == "human":
        return "STOPPED", f"{words['release']} to release"
    if latched_by is not None:
        how = "Ask the agent to reset_stop"
        if connected:
            how = f"{words['release']}, or ask the agent to reset_stop"
        return "STOPPED by the agent", how
    if control["human_in_control"]:
        return "You are driving", observation.get("limit") or ""
    action = observation.get("action")
    if action is not None and action["state"] in {"EXECUTING", "HELD"}:
        return f"Agent: {action['type']}", words["take_over"] if connected else ""
    if web:
        return "Ready", words["ready"]
    if not connected:
        return "No controller", "Turn it on (Xbox button). Agents can still drive."
    return "Ready", words["ready"]

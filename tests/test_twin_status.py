"""What the twin tells the person at the controller (viewer overlay, terminal)."""

from xbox_soarm_teleop.cli import twin as twin_cli
from xbox_soarm_teleop.twin.status import CONTROLS_HELP, status_text


def obs(**control) -> dict:
    base = {
        "stop_latched_by": None,
        "human_in_control": False,
        "controller_connected": True,
        "events": [],
    }
    base.update(control)
    return {"control": base, "action": None, "limit": ""}


def test_controller_stop_says_which_button_releases_it():
    headline, detail = status_text(obs(stop_latched_by="human"))
    assert headline == "STOPPED"
    assert "Menu" in detail and "release" in detail


def test_agent_stop_can_be_released_either_way():
    headline, detail = status_text(obs(stop_latched_by="agent"))
    assert headline == "STOPPED by the agent"
    assert "Menu" in detail and "reset_stop" in detail


def test_stop_without_a_controller_points_at_the_agent():
    _, detail = status_text(obs(stop_latched_by="agent", controller_connected=False))
    assert "Menu" not in detail and "reset_stop" in detail


def test_no_controller():
    headline, detail = status_text(obs(controller_connected=False))
    assert headline == "No controller"
    assert "Xbox button" in detail


def test_idle_tells_you_to_hold_lb():
    assert status_text(obs()) == ("Ready", "Hold LB and move a stick")


def test_driving_shows_why_the_arm_stopped():
    assert status_text(obs(human_in_control=True)) == ("You are driving", "")
    state = obs(human_in_control=True)
    state["limit"] = "at floor"
    assert status_text(state) == ("You are driving", "at floor")


def test_agent_action_is_named():
    state = obs()
    state["action"] = {"type": "move_ee_to", "state": "EXECUTING"}
    assert status_text(state) == ("Agent: move_ee_to", "Hold LB to take over")
    state["action"]["state"] = "SUCCEEDED"
    assert status_text(state)[0] == "Ready"


def test_controls_help_names_every_button():
    for word in ("LB", "RT", "A:", "Y:", "B:", "Menu", "D-pad"):
        assert word in CONTROLS_HELP


class FakeViewer:
    def __init__(self):
        self.calls = []

    def set_texts(self, texts):
        self.calls.append(texts)


def test_overlay_is_redrawn_only_when_the_status_changes():
    viewer = FakeViewer()
    shown = twin_cli._show_status(viewer, obs(), None)
    shown = twin_cli._show_status(viewer, obs(), shown)
    assert len(viewer.calls) == 1
    texts = [(left, right) for _font, _pos, left, right in viewer.calls[0]]
    assert ("Ready", "Hold LB and move a stick") in texts
    shown = twin_cli._show_status(viewer, obs(stop_latched_by="human"), shown)
    assert len(viewer.calls) == 2 and viewer.calls[1][0][2] == "STOPPED"


def test_overlay_is_skipped_without_a_window():
    shown = twin_cli._show_status(object(), obs(), None)
    assert shown == ("Ready", "Hold LB and move a stick", "free")


def test_overlay_names_the_camera_and_redraws_when_it_changes():
    viewer = FakeViewer()
    shown = twin_cli._show_status(viewer, obs(), None, "free")
    shown = twin_cli._show_status(viewer, obs(), shown, "wrist")
    assert len(viewer.calls) == 2
    assert ("Camera", "free (mouse)") in [(left, right) for _f, _p, left, right in viewer.calls[0]]
    assert ("Camera", "wrist") in [(left, right) for _f, _p, left, right in viewer.calls[1]]


def test_instructor_stop_tells_the_student_who_holds_it():
    headline, detail = status_text(obs(stop_latched_by="instructor"), web=True)
    assert headline == "STOPPED by your instructor"
    assert "instructor" in detail and "Release stop" not in detail


def test_instructor_sees_arms_from_the_outside():
    """The instructor's overview must not read as if the instructor were driving."""
    see = lambda **control: status_text(obs(**control), web=True, instructor=True)[0]  # noqa: E731
    assert see(human_in_control=True) == "Student driving"
    assert see(stop_latched_by="human") == "STOPPED by student"
    assert see(stop_latched_by="instructor") == "STOPPED by you"
    assert see(stop_latched_by="agent") == "STOPPED by agent"
    assert see() == "Ready"

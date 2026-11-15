"""Skills: pick and place, done by the arm itself.

An agent that steers the jaws a few centimetres at a time, a second or two per
step, from estimates a centimetre or two off, misses most grasps and pushes
things about while it tries. Real robots give the planner skills instead:
the planner says what to pick and where to put it, and the skill does the
motion in a tight loop. These skills use only what the arm itself can know:

- where an object is, from its pointing model (``locate``), or from
  ``observe`` when the twin shows coordinates and has no pointing model;
- whether it holds something (``attached``: on the real arm, the gripper
  stopping short of closed).

They never read the twin's truth, so they work in no-coordinates mode and
their answers give nothing away.

``pick``: look from overhead (no depth confusion), hover over the estimate
with the jaws turned and open, look again through the wrist camera (close up,
about half a centimetre off instead of one or two) and centre the jaws, twice;
go down, close, and check by feel. After a miss: open, lift, look again and
retry. ``place``: carry over a bin, lower, open, lift clear, go home.

They run on the caller's thread (like ``locate``), one motion at a time
through the twin's own actions, so a stop, a person taking over, or the
workspace limits end them as they would end any agent's motion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xbox_soarm_teleop.twin.actions import ActionRejected

HOVER_M = 0.06  # above an object while the wrist camera looks at it
OPEN_EXTRA_M = 0.05  # open this much wider than the object
CARRY_ABOVE_TABLE_M = 0.20  # the jaw center's height while carrying
RELEASE_ABOVE_FLOOR_M = 0.02  # an object's bottom over a bin floor when let go
LIFT_M = 0.05  # straight up after letting go, before anything else
ATTEMPTS = 3
ACTION_TIMEOUT_S = 30.0


class SkillFailed(Exception):
    """A skill could not finish: ``reason`` and ``detail`` say why."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason, self.detail = reason, detail


def _act(service, request: dict, client_id: str) -> None:
    """Run one action to its end; anything but success ends the skill."""
    try:
        status = service.submit(request, client_id=client_id).result(timeout=ACTION_TIMEOUT_S)
    except ActionRejected as exc:
        raise SkillFailed(exc.reason.value, exc.detail) from exc
    while True:
        status = service.wait(status.id, ACTION_TIMEOUT_S)
        if status.state.terminal:
            break
    if status.state.value.lower() != "succeeded":
        reason = status.reason.value if status.reason is not None else status.state.value.lower()
        raise SkillFailed(reason, f"{request['type']} {status.state.value.lower()}: {status.detail}")


def _move_by(service, dx: float, dy: float, dz: float, client_id: str) -> None:
    _act(service, {"type": "move_ee_by", "params": {"dxyz_m": [dx, dy, dz]}}, client_id)


def _grip(service, opening_m: float, client_id: str) -> None:
    position = service.grasp.settings.gripper_for_opening(opening_m)
    _act(service, {"type": "set_gripper", "params": {"position": float(position)}}, client_id)


def _objects(service) -> dict:
    return service.observe()["scene"]["objects"]


def _jaw(service) -> np.ndarray:
    return np.asarray(service.observe()["jaw_center_xyz_m"], dtype=float)


def _look(service, name: str, view: str, description: str | None, client_id: str):
    """Where the arm thinks ``name`` is, and the move_by that centres the jaws
    on it, or None if it cannot tell from this view."""
    if service.pointer is None:
        if service.hide_positions:
            raise SkillFailed("no_pointer", "no pointing model, and positions are hidden")
        xyz = np.asarray(_objects(service)[name]["xyz_m"], dtype=float)
        return xyz, service._jaw_move(_jaw(service), xyz)  # noqa: SLF001 - the twin's own rule
    answer = service.locate(name, view=view, description=description, client_id=client_id)
    if not answer.get("ok"):
        if answer.get("reason") in {"not_found", "pointer_error"}:
            return None
        raise SkillFailed(answer.get("reason", "error"), answer.get("detail", ""))
    return np.asarray(answer["xyz_m"], dtype=float), answer["jaw_move_m"]


def _look_anywhere(service, name, views, description, client_id, looks):
    for view in views:
        looks[0] += 1
        seen = _look(service, name, view, description, client_id)
        if seen is not None:
            return seen
    raise SkillFailed("not_found", f"could not find {name} from {', '.join(views)}")


def pick(
    service,
    name: str,
    *,
    client_id: str = "agent",
    description: str | None = None,
    attempts: int = ATTEMPTS,
) -> dict[str, Any]:
    """Pick ``name`` up and lift it to carrying height.

    Returns ``{"ok": True, "holding": name, "attempts": n, "looks": n}``, or
    ``{"ok": False, "reason": ..., "detail": ...}``: ``invalid_request`` (no
    such object), ``no_pointer``, ``not_found``, ``missed`` (every attempt),
    or why a motion ended (``stopped``, ``preempted``, ``workspace_bound``...).
    """
    looks = [0]
    tried = 0
    try:
        objects = _objects(service)
        if name not in objects:
            known = ", ".join(objects) or "none"
            raise SkillFailed("invalid_request", f"Unknown object '{name}'. On the table: {known}")
        if objects[name]["attached"]:
            return {"ok": True, "holding": name, "attempts": 0, "looks": 0}
        if any(entry["attached"] for entry in objects.values()):
            raise SkillFailed("busy", "already holding something: place it first")

        _xyz, move = _look_anywhere(service, name, ("overhead", "front"), description, client_id, looks)
        _move_by(service, move["dx"], move["dy"], move["dz"] + HOVER_M, client_id)
        _act(service, {"type": "set_wrist_roll", "params": {"angle_rad": -1.5}}, client_id)
        width = float(_objects(service)[name]["grip_width_m"])
        _grip(service, width + OPEN_EXTRA_M, client_id)
        for tried in range(1, attempts + 1):
            for _ in range(2):  # centre the jaws on what the wrist camera sees
                _xyz, move = _look_anywhere(
                    service, name, ("wrist", "overhead"), description, client_id, looks
                )
                _move_by(service, move["dx"], move["dy"], 0.0, client_id)
            _move_by(service, 0.0, 0.0, move["dz"], client_id)
            _grip(service, width, client_id)
            if _objects(service)[name]["attached"]:
                carry = (service.table_top_z or 0.0) + CARRY_ABOVE_TABLE_M
                _move_by(service, 0.0, 0.0, carry - _jaw(service)[2], client_id)
                return {"ok": True, "holding": name, "attempts": tried, "looks": looks[0]}
            _grip(service, width + OPEN_EXTRA_M, client_id)  # missed: open, up, look again
            _move_by(service, 0.0, 0.0, HOVER_M, client_id)
        raise SkillFailed("missed", f"{name} was not between the jaws after {attempts} attempts")
    except SkillFailed as exc:
        return {
            "ok": False,
            "reason": exc.reason,
            "detail": exc.detail,
            "attempts": tried,
            "looks": looks[0],
        }


def place(service, bin_name: str, *, client_id: str = "agent") -> dict[str, Any]:
    """Put what the gripper holds into ``bin_name`` (a bin, or the goal box),
    lift clear and go home.

    Returns ``{"ok": True, "released": name, "bin": bin_name}``, or ``{"ok":
    False, "reason": ...}``: ``not_holding``, ``invalid_request`` (no such bin),
    or why a motion ended.
    """
    try:
        scene = service.observe()["scene"]
        held = [name for name, entry in scene["objects"].items() if entry["attached"]]
        if not held:
            raise SkillFailed("not_holding", "nothing is in the gripper: pick something first")
        boxes = {box["name"]: box for box in scene.get("bins", [])}
        if "goal" in scene:
            boxes[scene["goal"]["name"]] = scene["goal"]
        if bin_name not in boxes:
            known = ", ".join(boxes) or "none"
            raise SkillFailed("invalid_request", f"Unknown bin '{bin_name}'. Here: {known}")
        name = held[0]
        cx, cy, floor = boxes[bin_name]["center_xyz_m"]
        carry = (service.table_top_z or 0.0) + CARRY_ABOVE_TABLE_M
        jaw = _jaw(service)
        if jaw[2] < carry - 0.01:
            _move_by(service, 0.0, 0.0, carry - jaw[2], client_id)
        for _ in range(2):  # the gripper tilts a little as it moves
            jaw = _jaw(service)
            _move_by(service, cx - jaw[0], cy - jaw[1], 0.0, client_id)
        half = float(scene["objects"][name]["size_m"][2]) / 2.0
        _move_by(service, 0.0, 0.0, floor + half + RELEASE_ABOVE_FLOOR_M - _jaw(service)[2], client_id)
        _act(service, {"type": "set_gripper", "params": {"position": 0.0}}, client_id)
        _move_by(service, 0.0, 0.0, LIFT_M, client_id)
        _act(service, {"type": "home"}, client_id)
        return {"ok": True, "released": name, "bin": bin_name}
    except SkillFailed as exc:
        return {"ok": False, "reason": exc.reason, "detail": exc.detail}

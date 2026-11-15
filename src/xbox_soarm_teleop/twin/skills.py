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

``pick``: turn the jaws while the arm is high; look from overhead and from
the front and check they agree (a same-coloured tray can pass for a cube from
overhead); hover over the estimate with the jaws open, look again through the wrist camera (close up,
about half a centimetre off instead of one or two) and centre the jaws, twice;
go down, close, and check by feel. After a miss: open, lift, look again and
retry. ``place``: carry over a bin, lower, open, lift clear, go home.

They run on the caller's thread (like ``locate``), one motion at a time
through the twin's own actions, so a stop, a person taking over, or the
workspace limits end them as they would end any agent's motion.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

from xbox_soarm_teleop.twin.actions import ActionRejected

HOVER_M = 0.06  # above an object while the wrist camera looks at it
OPEN_EXTRA_M = 0.05  # open this much wider than the object
CARRY_ABOVE_TABLE_M = 0.20  # the jaw center's height while carrying
RELEASE_ABOVE_FLOOR_M = 0.02  # an object's bottom over a bin floor when let go
LIFT_M = 0.05  # straight up after letting go, before anything else
ATTEMPTS = 3
# What each twin holds because it was picked by description: (name, description),
# so place can answer in the request's words too.
_DESCRIBED: "weakref.WeakKeyDictionary[Any, tuple[str, str]]" = weakref.WeakKeyDictionary()
OPEN_UNKNOWN_M = 0.09  # open this wide for an object of unknown size
CLOSE_STEP_M = 0.005  # closing by feel: less than the twin's 6 mm squeeze margin
CLOSE_FLOOR_M = 0.015  # nothing narrower than this is picked up
AGREE_M = 0.03  # two views that put an object this close agree on where it is
TRAY_MARGIN_M = 0.01  # an estimate this far inside a bin's walls counts as in it
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
    except TimeoutError as exc:
        raise SkillFailed("timeout", f"{request['type']}: the twin's loop did not answer") from exc
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


def _in_a_bin(service, xyz) -> bool:
    """The estimate falls inside one of the cell's bins (the robot knows its
    own furniture: ``scene.bins`` and ``scene.goal``)."""
    scene = service.observe()["scene"]
    boxes = list(scene.get("bins", [])) + ([scene["goal"]] if "goal" in scene else [])
    for box in boxes:
        cx, cy, _ = box["center_xyz_m"]
        lx, ly, _ = box["inner_size_m"]
        if abs(xyz[0] - cx) <= lx / 2 + TRAY_MARGIN_M and abs(xyz[1] - cy) <= ly / 2 + TRAY_MARGIN_M:
            return True
    return False


def _first_look(service, name, description, client_id, looks):
    """Where the object is, checked across views.

    From overhead a cube and a tray's floor of the same colour are both
    squares, and a pointing model can take one for the other; from the front a
    cube stands up and a tray lies flat. So look from both: if they agree, use
    overhead (no depth confusion); if one says inside a bin and the other does
    not, believe the other; otherwise ask the side view which it agrees with.
    """
    seen = {}
    for view in ("overhead", "front"):
        looks[0] += 1
        found = _look(service, name, view, description, client_id)
        if found is not None:
            seen[view] = found
    if not seen:
        raise SkillFailed("not_found", f"could not find {name} from overhead or the front")
    if len(seen) == 1:
        return next(iter(seen.values()))
    over, front = seen["overhead"], seen["front"]
    if np.linalg.norm(over[0][:2] - front[0][:2]) <= AGREE_M:
        return over
    over_in, front_in = _in_a_bin(service, over[0]), _in_a_bin(service, front[0])
    if over_in != front_in:
        return front if over_in else over
    looks[0] += 1
    side = _look(service, name, "side", description, client_id)
    if side is not None:
        return min((over, front), key=lambda seen_: np.linalg.norm(seen_[0][:2] - side[0][:2]))
    return over


def _held(service) -> str | None:
    """The object the gripper holds, if any (on the real arm: it stopped short of closed)."""
    return next((n for n, entry in _objects(service).items() if entry["attached"]), None)


def _close_by_feel(service, client_id: str) -> str | None:
    """Close in small steps until the gripper holds something, as a gripper that
    feels contact does: an object of unknown size is never squeezed out (each
    step is smaller than the twin's squeeze margin). Returns what it holds."""
    opening = OPEN_UNKNOWN_M
    while opening > CLOSE_FLOOR_M:
        opening -= CLOSE_STEP_M
        _grip(service, opening, client_id)
        held = _held(service)
        if held is not None:
            return held
    return None


def _shown_match(service, description: str) -> str | None:
    """With coordinates shown and no pointing model: the object a description
    means (a simulation convenience; with a pointing model the arm looks)."""
    from xbox_soarm_teleop.twin.layouts import object_description

    wanted = description.strip().lower()
    return next(
        (n for n in _objects(service) if object_description(n, service.layout).lower() == wanted),
        None,
    )


def pick(
    service,
    name: str | None = None,
    *,
    client_id: str = "agent",
    description: str | None = None,
    attempts: int = ATTEMPTS,
) -> dict[str, Any]:
    """Pick an object up and lift it to carrying height.

    By ``name``; or by ``description`` alone, in the request's words ("red
    cube"): the arm looks for what matches, opens wide, closes by feel, and
    says what it is holding in the same words. The reply never carries the
    object's name: an agent told misleading names sorted by the name in the
    reply (lesson 24), so a name that misleads the agent cannot mislead it here.

    Returns ``{"ok": True, "holding": name or description, "attempts": n,
    "looks": n}``, or
    ``{"ok": False, "reason": ..., "detail": ...}``: ``invalid_request`` (no
    such object), ``no_pointer``, ``not_found``, ``missed`` (every attempt),
    or why a motion ended (``stopped``, ``preempted``, ``workspace_bound``...).
    """
    looks = [0]
    tried = 0
    try:
        objects = _objects(service)
        if name is None and not description:
            raise SkillFailed("invalid_request", 'say what to pick: a name, or description="red cube"')
        if name is not None and name not in objects:
            known = ", ".join(objects) or "none"
            raise SkillFailed("invalid_request", f"Unknown object '{name}'. On the table: {known}")
        if name is not None and objects[name]["attached"]:
            return {"ok": True, "holding": name, "attempts": 0, "looks": 0}
        if _held(service) is not None:
            raise SkillFailed("busy", "already holding something: place it first")
        if service.pointer is None and service.hide_positions:  # before moving at all
            raise SkillFailed("no_pointer", "no pointing model, and positions are hidden")
        asked = None if name is not None else description.strip()
        _DESCRIBED.pop(service, None)
        target = name
        if name is None and service.pointer is None:
            target = _shown_match(service, description)
            if target is None:
                raise SkillFailed("not_found", f"nothing on the table matches '{description}'")
            description = None
        by_feel = target is None  # by description: its size is not known until it is held

        # Turn the jaws first, while the arm is high: from home the gripper is
        # tilted, and turning it later swung the jaws down to the table.
        _act(service, {"type": "set_wrist_roll", "params": {"angle_rad": -1.5}}, client_id)
        _xyz, move = _first_look(service, target, description, client_id, looks)
        _move_by(service, move["dx"], move["dy"], move["dz"] + HOVER_M, client_id)
        width = None if by_feel else float(_objects(service)[target]["grip_width_m"])
        wide = OPEN_UNKNOWN_M if by_feel else width + OPEN_EXTRA_M
        _grip(service, wide, client_id)
        for tried in range(1, attempts + 1):
            for _ in range(2):  # centre the jaws on what the wrist camera sees
                _xyz, move = _look_anywhere(
                    service, target, ("wrist", "overhead"), description, client_id, looks
                )
                _move_by(service, move["dx"], move["dy"], 0.0, client_id)
            _move_by(service, 0.0, 0.0, move["dz"], client_id)
            if by_feel:
                held = _close_by_feel(service, client_id)
            else:
                _grip(service, width, client_id)
                held = _held(service)
            if held is not None:
                carry = (service.table_top_z or 0.0) + CARRY_ABOVE_TABLE_M
                _move_by(service, 0.0, 0.0, carry - _jaw(service)[2], client_id)
                if asked is not None:  # answer in the request's words, not a name
                    _DESCRIBED[service] = (held, asked)
                    held = asked
                return {"ok": True, "holding": held, "attempts": tried, "looks": looks[0]}
            _grip(service, wide, client_id)  # missed: open, up, look again
            _move_by(service, 0.0, 0.0, HOVER_M, client_id)
        what = target or f"the {description}"
        raise SkillFailed("missed", f"{what} was not between the jaws after {attempts} attempts")
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
        described = _DESCRIBED.pop(service, None)
        if described is not None and described[0] == name:
            name = described[1]  # picked by description: say it the same way
        return {"ok": True, "released": name, "bin": bin_name}
    except SkillFailed as exc:
        return {"ok": False, "reason": exc.reason, "detail": exc.detail}

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
MOVABLE_FLOOR_ABOVE_TABLE_M = 0.0085  # a movable box's floor top (sim/table_scenes.py)


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
            # Say what there is, so the next call can get it right.
            loose = [n for n, entry in objects.items() if not entry["attached"]] or list(objects)
            example = loose[0] if loose else "red_cube"
            raise SkillFailed(
                "invalid_request",
                f'say what to pick: pick(name="{example}"), or pick(description="red cube") in '
                f"the request's words. Objects here: {', '.join(loose) or 'none'}",
            )
        if name is not None and name not in objects:
            known = ", ".join(objects) or "none"
            raise SkillFailed("invalid_request", f"Unknown object '{name}'. On the table: {known}")
        if name is not None and objects[name].get("box"):
            raise SkillFailed(
                "invalid_request",
                f"{name} is a box, too big to pick up: push it to move it, or place things in it",
            )
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
        movable = [n for n, entry in scene["objects"].items() if entry.get("box")]
        if bin_name not in boxes and bin_name not in movable:
            known = ", ".join([*boxes, *movable]) or "none"
            raise SkillFailed("invalid_request", f"Unknown bin '{bin_name}'. Here: {known}")
        name = held[0]
        if bin_name in movable:  # a box that may have been pushed: look where it is now
            box_xyz, _move = _first_look(service, bin_name, None, client_id, [0])
            cx, cy = float(box_xyz[0]), float(box_xyz[1])
            floor = (service.table_top_z or 0.0) + MOVABLE_FLOOR_ABOVE_TABLE_M
        else:
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


# Pushing (``push``): the arm's own words for directions, as seen from its base.
DIRECTIONS = {
    "forward": (1.0, 0.0),
    "away": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "toward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}
PUSH_MAX_M = 0.15  # one push request moves an object at most this far
PUSH_CLEAR_M = 0.03  # the closed jaws start this far behind the object's back face
PUSH_LOW_M = 0.012  # the jaw centre's height above the table while pushing
PUSH_CLOSE_M = 0.01  # near enough: stop pushing
PUSH_SPEED_M_S = 0.03  # the push stroke itself
PUSH_LEAD_M = 0.025  # how far a slow push carries on past where the jaw centre stops (measured)
PUSH_TRIES = 4  # pushes, each followed by a fresh look
PUSH_TOLERANCE_M = 0.035  # within this of the goal counts as done (judged by looks, which are rough)
PUSH_TOLERANCE_SHARE = 0.3  # or within this share of the distance asked


def _fixed_furniture(service, name: str | None, description: str | None) -> str | None:
    """The fixed tray (or goal box) a push request names, if it names one: they
    are part of the desk and cannot be pushed (issue 53)."""
    scene = service.observe()["scene"]
    fixed = list(scene.get("bins", [])) + ([scene["goal"]] if "goal" in scene else [])
    words = (description or "").lower()
    for box in fixed:
        label = (box.get("label") or box["name"]).lower()
        if name == box["name"] or (words and label in words):
            return label
        colour = label.split()[0]
        if words and ("tray" in words or "bin" in words) and "box" not in words:
            if colour in words or not any(other in words for other in _COLOURS):
                return label
    return None


_COLOURS = ("red", "green", "blue", "yellow", "orange", "purple", "white", "black")


def _where(service, target, description, client_id, looks) -> np.ndarray:
    """Where the arm thinks the object is now (its looks; observe when shown)."""
    xyz, _move = _first_look(service, target, description, client_id, looks)
    return np.asarray(xyz, dtype=float)


def push(
    service,
    name: str | None = None,
    *,
    description: str | None = None,
    direction: str | None = None,
    distance_m: float | None = None,
    dx_m: float | None = None,
    dy_m: float | None = None,
    client_id: str = "agent",
) -> dict[str, Any]:
    """Slide an object along the table, without lifting it.

    Say what (``name``, or ``description`` in the request's words) and where:
    a ``direction`` ("forward"/"away", "back"/"toward", "left", "right", as
    seen from the arm's base) and ``distance_m``, or ``dx_m``/``dy_m`` (m;
    +x away from the base, +y to its left). The arm closes its jaws, gets
    behind the object, goes low, slides through, then looks again and pushes
    the rest, since one push drifts. It pushes what is in the way too.

    Returns ``{"ok": True, "pushed": name or description, "pushes": n,
    "moved_m": [dx, dy], "off_by_m": d}`` as far as the arm can tell from its
    own looks, or ``{"ok": False, "reason": ..., "detail": ...}``.
    """
    looks = [0]
    pushes = 0
    try:
        objects = _objects(service)
        furniture = _fixed_furniture(service, name, description)
        if furniture is not None:
            movable = [n.replace("_", " ") for n, entry in objects.items() if entry.get("box")]
            can = "the objects" + (f" and the {', '.join(movable)}" if movable else "")
            raise SkillFailed(
                "fixed",
                f"the {furniture} is fixed to the desk and cannot be pushed; only {can} can. "
                "Tell the person it cannot be done; do not push something else instead",
            )
        if name is None and not description:
            loose = [n for n, entry in objects.items() if not entry["attached"]]
            raise SkillFailed(
                "invalid_request",
                f'say what to push: push(name="{loose[0] if loose else "red_cube"}", ...), or '
                f'push(description="red cube", ...). Objects here: {", ".join(loose) or "none"}',
            )
        if name is not None and name not in objects:
            known = ", ".join(objects) or "none"
            raise SkillFailed("invalid_request", f"Unknown object '{name}'. On the table: {known}")
        by_words = direction is not None or distance_m is not None
        by_numbers = dx_m is not None or dy_m is not None
        if by_words and by_numbers:
            raise SkillFailed(
                "invalid_request", "give either direction and distance_m, or dx_m and dy_m"
            )
        if by_words:
            if direction not in DIRECTIONS:
                raise SkillFailed(
                    "invalid_request", f"direction '{direction}': one of {', '.join(DIRECTIONS)}"
                )
            if distance_m is None:
                raise SkillFailed("invalid_request", "say how far: distance_m (metres)")
            step = np.asarray(DIRECTIONS[direction]) * float(distance_m)
        elif by_numbers:
            step = np.array([float(dx_m or 0.0), float(dy_m or 0.0)])
        else:
            raise SkillFailed("invalid_request", "say how far, and which way: direction and distance_m")
        if np.linalg.norm(step) > PUSH_MAX_M:
            raise SkillFailed("invalid_request", f"push at most {PUSH_MAX_M:.2f} m at a time")
        if np.linalg.norm(step) < PUSH_CLOSE_M:
            raise SkillFailed("invalid_request", f"push at least {PUSH_CLOSE_M:.2f} m")
        if _held(service) is not None:
            raise SkillFailed("busy", "holding something: place it first")
        target = name
        if name is None and service.pointer is None:
            target = _shown_match(service, description)
            if target is None:
                raise SkillFailed("not_found", f"nothing on the table matches '{description}'")
        # A pointing model looks for the description; without one, observe by name.
        look_with = description if name is None and service.pointer is not None else None
        half = float(max(objects[target]["size_m"][:2]) / 2.0) if target else 0.02

        start = _where(service, target, look_with, client_id, looks)
        goal = start[:2] + step
        table = service.table_top_z or 0.0
        low = table + PUSH_LOW_M
        high = table + CARRY_ABOVE_TABLE_M / 2.0
        _act(service, {"type": "set_gripper", "params": {"position": 1.0}}, client_id)  # jaws shut
        now = start
        for pushes in range(1, PUSH_TRIES + 1):
            remaining = goal - now[:2]
            gap = float(np.linalg.norm(remaining))
            # Done when near enough, or past the goal: pushing back means coming
            # down in front of the object, and the gripper's body hangs back
            # toward the base, onto it.
            if gap <= PUSH_CLOSE_M or float(np.dot(remaining, step)) <= 0.0:
                pushes -= 1
                break
            way = remaining / gap
            behind = now[:2] - way * (half + PUSH_CLEAR_M)
            jaw = _jaw(service)
            if jaw[2] < high - 0.01:
                _move_by(service, 0.0, 0.0, high - jaw[2], client_id)  # up, clear of everything
            for _ in range(2):  # the gripper tilts a little as it moves
                jaw = _jaw(service)
                _move_by(service, behind[0] - jaw[0], behind[1] - jaw[1], 0.0, client_id)
            jaw = _jaw(service)
            _move_by(service, 0.0, 0.0, low - jaw[2], client_id)
            # Contact after the gap, then the rest, stopping short by the closed
            # jaws' lead (their face is ahead of the jaw centre, and further on a
            # diagonal): a push that falls short is finished by the next one; one
            # that overshoots cannot be undone without going round.
            through = way * (PUSH_CLEAR_M + max(gap - PUSH_LEAD_M, gap * 0.5))
            _act(  # slowly: the arm moves kinematically, and fast it kicks, not pushes
                service,
                {
                    "type": "move_ee_by",
                    "params": {"dxyz_m": [float(through[0]), float(through[1]), 0.0]},
                    "max_speed_m_s": PUSH_SPEED_M_S,
                },
                client_id,
            )
            jaw = _jaw(service)
            _move_by(service, 0.0, 0.0, high - jaw[2], client_id)  # lift clear before looking
            before, now = now, _where(service, target, look_with, client_id, looks)
            if pushes == 1 and np.linalg.norm(now[:2] - before[:2]) < PUSH_CLOSE_M:
                break  # it did not move: pushing again would not help (issue 53)
        _act(service, {"type": "home"}, client_id)
        moved = now[:2] - start[:2]
        off_by = float(np.linalg.norm(goal - now[:2]))
        result = {
            "pushed": description.strip() if name is None else name,
            "pushes": pushes,
            "moved_m": [round(float(v), 3) for v in moved],
            "off_by_m": round(off_by, 3),
            "looks": looks[0],
        }
        asked = float(np.linalg.norm(step))
        # Say plainly how it went: the agent tells the person (issue 53).
        if float(np.linalg.norm(moved)) < PUSH_CLOSE_M:
            return {
                "ok": False,
                "reason": "did_not_move",
                "detail": "it did not move: it may be fixed in place, stuck against something, "
                "or the arm looked at something else",
                **result,
            }
        if off_by > max(PUSH_TOLERANCE_M, PUSH_TOLERANCE_SHARE * asked):
            reason = "fell_short" if float(np.dot(goal - now[:2], step)) > 0 else "overshot"
            return {
                "ok": False,
                "reason": reason,
                "detail": f"moved {np.linalg.norm(moved) * 100:.0f} cm of the {asked * 100:.0f} cm asked; "
                f"{off_by * 100:.0f} cm from where it should be",
                **result,
            }
        return {"ok": True, **result}
    except SkillFailed as exc:
        return {
            "ok": False,
            "reason": exc.reason,
            "detail": exc.detail,
            "pushes": pushes,
            "looks": looks[0],
        }

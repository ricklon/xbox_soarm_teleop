"""Challenges: a student designs the desk, the request and the answer.

A challenge is a small file, YAML or a dict in a notebook, in the names a
person would use (``red_cube``), whatever names the agent is shown::

    name: red-swap-tricky
    scene: sort_colors              # the cell: the sorting desk
    objects:                        # where each starts (m from the arm's base)
      red_cube:    {x: 0.275, y: -0.06}
      blue_cube:   {x: 0.23, y: -0.16}
      yellow_cube: {x: 0.30, y: 0.07, turn_deg: 30}   # turned from facing the arm
    names: misleading               # what the agent is told objects are called
    request: "Put the red cube in the blue bin and the blue cube in the red bin."
    expect: {red_cube: blue_bin, blue_cube: red_bin}  # left out: stays out of the bins
    possible: true                  # false: impossible on purpose

What it goes through:

1. ``load_challenge``: what cannot be set up is refused with a reason
   (an unknown object or bin, a spot off the desk, objects overlapping).
2. ``check``: warnings, not refusals, for what breaks the layout rules the
   built-in desk keeps (spacing, reach, trays, edges), since breaking them is
   a fair experiment.
3. ``prove``: the arm's own skills carry the expected arrangement out, once
   from true positions and again looking with a pointer whose aim is off as an
   agent's is: doable, fragile, or not doable. An impossible challenge
   (``possible: false``) is tried with what it ``asked`` and must fail.
4. Loading it on an arm (``Challenge.layout``, ``TwinService.load_layout``;
   the hub's ``/r/{key}/challenge``) and running agents on it
   (``scripts/agent_trial.py --challenge``), graded with ``grade_challenge``.

Only the sorting desk and the objects its model holds, for now.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.sim.table_scenes import TABLE_SCENES
from xbox_soarm_teleop.twin.layouts import LAYOUTS, Layout
from xbox_soarm_teleop.twin.naming import APPEARANCE, NAMINGS, shown_names
from xbox_soarm_teleop.twin.tasks import SortTask, grade

CELL = "sort_bins"  # the compiled scene challenges are set in: the sorting desk
DESK = TABLE_SCENES[CELL].table
TRAYS = {bin_.name: bin_ for bin_ in TABLE_SCENES[CELL].bins}

# Every object the desk's model holds, by the name a person would use.
_INTERNAL = (
    "pick_cube",
    "blue_cube",
    "green_cube",
    "yellow_cube",
    "red_cube_small",
    "red_bar",
    sim_cli.BEAN_BAG_NAME,
)
PERSON_NAMES: dict[str, str] = shown_names(_INTERNAL, "truthful")
"""The twin's name -> the name a person uses (``pick_cube`` -> ``red_cube``)."""
_BY_PERSON = {person: internal for internal, person in PERSON_NAMES.items()}

_HALF: dict[str, tuple[float, float, float]] = {
    "pick_cube": (sim_cli.PICK_PLACE_CUBE_HALF_EXTENT,) * 3,
    sim_cli.BEAN_BAG_NAME: tuple(float(v) for v in sim_cli.BEAN_BAG_HALF_EXTENTS),
    **{name: tuple(half) for name, half, _, _ in sim_cli.DISTRACTOR_OBJECTS},
}

# The layout rules the built-in desk keeps (tests/test_twin_sort.py), found by
# agents knocking cubes about: the open jaws span about 9 cm and an agent's
# aim is off by a centimetre or two.
EDGE_GAP_M = 0.05  # from an object's footprint to the desk's edge
SPACING_M = 0.10  # between object centres
TRAY_GAP_M = 0.05  # from an object's footprint to a tray's outer wall
RING_M = (0.20, 0.33)  # the steady part of the ring the arm grasps in
MAX_ANGLE_DEG = 40.0  # the pick area: in front of the arm, not out by the trays

AIM_ERROR_M = 0.015  # a distant view's aim error, as measured (docs/vision_benchmark.md)
WRIST_ERROR_M = 0.005  # the wrist camera's, close up
TRAY_MIXUP_RATE = 0.5  # from overhead, a cube and the tray of its colour look alike


class ChallengeError(ValueError):
    """A challenge that cannot be set up; the message says why."""


@dataclass(frozen=True)
class Challenge:
    """A loaded challenge, in the twin's own names (``person_names`` maps back).

    Args:
        name: Its title.
        scene: The built-in layout it starts from (``sort_colors``,
            ``sort_colors_decoy``); its objects are replaced by ``objects``.
        objects: The twin's name -> (x, y, turn in degrees from facing the base).
        requests: The request, in one or more wordings.
        expect: Where each object must end up (a tray, or None: in no tray).
        names: The names agents are shown (``twin.naming``).
        possible: False for a challenge that is impossible on purpose.
        asked: For an impossible one: the arrangement the request asks for,
            which ``prove`` tries and must fail at (trays may not exist).
        notes: Anything the designer wants to say about it.
    """

    name: str
    scene: str
    objects: dict[str, tuple[float, float, float]]
    requests: tuple[str, ...]
    expect: dict[str, str | None]
    names: str = "truthful"
    possible: bool = True
    asked: dict[str, str | None] = field(default_factory=dict)
    notes: str = ""

    def person(self, name: str) -> str:
        return PERSON_NAMES.get(name, name)

    def layout(self) -> Layout:
        """The desk as the challenge sets it: each object facing the base, turned
        by its ``turn_deg``. Its name tells an agent nothing about the challenge."""
        objects = {}
        for name, (x, y, turn) in self.objects.items():
            yaw = math.atan2(y, x) + math.radians(turn)
            objects[name] = (x, y, DESK.top_z + _HALF[name][2], yaw)
        return Layout(
            name="challenge",
            model_scene=CELL,
            difficulty="hard",
            description="The sorting desk, set up by a person.",
            objects=objects,
            listed=False,
        )

    def task(self) -> SortTask:
        """The challenge as a sort task (twin names), for trials and grading."""
        return SortTask(
            name=self.name, scene=self.scene, requests=self.requests, expect=dict(self.expect)
        )

    def summary(self) -> dict[str, Any]:
        """The challenge in the designer's names (for pages, reports and logs)."""
        return {
            "name": self.name,
            "scene": self.scene,
            "objects": {
                self.person(n): {"x": x, "y": y, "turn_deg": t} for n, (x, y, t) in self.objects.items()
            },
            "names": self.names,
            "requests": list(self.requests),
            "expect": {self.person(n): b for n, b in self.expect.items()},
            "possible": self.possible,
            "asked": {self.person(n): b for n, b in self.asked.items()},
            "notes": self.notes,
        }


def _internal(name: str, where: str) -> str:
    if name in _BY_PERSON:
        return _BY_PERSON[name]
    if name in PERSON_NAMES:  # the twin's own name works too
        return name
    known = ", ".join(sorted(_BY_PERSON))
    raise ChallengeError(f"{where}: unknown object '{name}'. The desk has: {known}")


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChallengeError(f"{what} must be a number (metres from the arm's base), not {value!r}")
    return float(value)


def _footprint(name: str, turn_deg: float = 0.0) -> float:
    """Half the object's footprint across, whichever way it is turned (m)."""
    hx, hy, _ = _HALF[name]
    return math.hypot(hx, hy) if turn_deg % 90 else max(hx, hy)


def _tray_gap(name: str, x: float, y: float) -> tuple[str, float]:
    """The nearest tray and the gap from the object's footprint to its outer wall."""
    best = ("", math.inf)
    half = _footprint(name)
    for tray in TRAYS.values():
        reach = [tray.inner_half_m[i] + tray.wall_m + half for i in (0, 1)]
        gap = max(abs(x - tray.center_xy_m[0]) - reach[0], abs(y - tray.center_xy_m[1]) - reach[1])
        if gap < best[1]:
            best = (tray.name, gap)
    return best


def load_challenge(source: str | Path | dict) -> Challenge:
    """A challenge from a dict, a YAML file, or YAML text.

    Raises:
        ChallengeError: it cannot be set up, with the reason.
    """
    if isinstance(source, dict):
        spec = source
    else:
        import yaml

        text = (
            Path(source).read_text()
            if isinstance(source, Path) or ("\n" not in str(source) and Path(str(source)).is_file())
            else str(source)
        )
        try:
            spec = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ChallengeError(f"not valid YAML: {exc}") from exc
    if not isinstance(spec, dict):
        raise ChallengeError("a challenge is a mapping: name, objects, request, expect")

    name = str(spec.get("name") or "").strip() or "challenge"
    scene = str(spec.get("scene") or "sort_colors")
    if scene not in LAYOUTS or LAYOUTS[scene].model_scene != CELL:
        desks = ", ".join(n for n, layout in LAYOUTS.items() if layout.model_scene == CELL)
        raise ChallengeError(f"scene '{scene}': challenges are set on the sorting desk ({desks})")

    raw_objects = spec.get("objects")
    if not isinstance(raw_objects, dict) or not raw_objects:
        raise ChallengeError("objects: say which objects are on the desk and where (x, y)")
    objects: dict[str, tuple[float, float, float]] = {}
    for person, where in raw_objects.items():
        internal = _internal(str(person), "objects")
        if internal in objects:
            raise ChallengeError(f"objects: {person} is listed twice")
        if not isinstance(where, dict):
            raise ChallengeError(f"objects: {person} needs x and y, like {{x: 0.27, y: -0.06}}")
        for axis in ("x", "y"):
            if axis not in where:
                raise ChallengeError(f"objects: {person} has no {axis}")
        x, y = _number(where["x"], f"{person}'s x"), _number(where["y"], f"{person}'s y")
        turn = _number(where.get("turn_deg", 0.0), f"{person}'s turn_deg")
        if not DESK.contains(x, y, margin=_footprint(internal, turn)):
            raise ChallengeError(
                f"objects: {person} at ({x}, {y}) is not on the desk "
                f"(x {DESK.center_xy_m[0] - DESK.half_xy_m[0]:.2f} to {DESK.center_xy_m[0] + DESK.half_xy_m[0]:.2f}, "
                f"y {-DESK.half_xy_m[1]:.2f} to {DESK.half_xy_m[1]:.2f} m)"
            )
        tray, gap = _tray_gap(internal, x, y)
        if gap < 0:
            raise ChallengeError(
                f"objects: {person} would start on or in {tray}; starting in a tray is not supported yet"
            )
        objects[internal] = (x, y, turn)
    for (a, pa), (b, pb) in itertools.combinations(objects.items(), 2):
        if math.dist(pa[:2], pb[:2]) < _footprint(a, pa[2]) + _footprint(b, pb[2]):
            raise ChallengeError(
                f"objects: {PERSON_NAMES[a]} and {PERSON_NAMES[b]} overlap; move them apart"
            )

    requests = spec.get("request", spec.get("requests"))
    if isinstance(requests, str):
        requests = [requests]
    for r in requests or []:
        if not isinstance(r, str):
            raise ChallengeError(
                f"request: {r!r} is not plain text. In YAML, put a request that contains "
                'a colon in quotes: "Swap them: red into blue."'
            )
    requests = tuple(r.strip() for r in (requests or []) if r.strip())
    if not requests:
        raise ChallengeError("request: say what the agent is asked, in a person's words")

    names = str(spec.get("names") or "truthful")
    if names not in NAMINGS:
        raise ChallengeError(f"names '{names}': one of {', '.join(NAMINGS)}")

    possible = spec.get("possible", True)
    if not isinstance(possible, bool):
        raise ChallengeError("possible: true or false")

    def arrangement(key: str, *, trays_must_exist: bool) -> dict[str, str | None]:
        raw = spec.get(key) or {}
        if not isinstance(raw, dict):
            raise ChallengeError(f"{key}: a mapping from object to tray (or null: in no tray)")
        result = {}
        for person, tray in raw.items():
            internal = _internal(str(person), key)
            if internal not in objects:
                raise ChallengeError(f"{key}: {person} is not on the desk (add it to objects)")
            if tray is not None and str(tray) not in TRAYS and trays_must_exist:
                raise ChallengeError(f"{key}: no tray '{tray}'. Trays: {', '.join(TRAYS)}, or null")
            result[internal] = None if tray is None else str(tray)
        return result

    expect = arrangement("expect", trays_must_exist=True)
    asked = arrangement("asked", trays_must_exist=False)
    if possible and asked:
        raise ChallengeError("asked is for impossible challenges (possible: false); use expect")
    full_expect = {name: expect.get(name) for name in objects}
    return Challenge(
        name=name,
        scene=scene,
        objects=objects,
        requests=requests,
        expect=full_expect,
        names=names,
        possible=possible,
        asked=asked,
        notes=str(spec.get("notes") or ""),
    )


def check(challenge: Challenge) -> list[str]:
    """Warnings and notes: where the challenge breaks the built-in desk's layout
    rules (allowed: it may be the point), and what it leaves to defaults."""
    warnings = []
    for name, (x, y, turn) in challenge.objects.items():
        person = PERSON_NAMES[name]
        half = _footprint(name, turn)
        if not DESK.contains(x, y, margin=half + EDGE_GAP_M):
            warnings.append(
                f"{person} is within {EDGE_GAP_M * 100:.0f} cm of the desk's edge: a nudge can knock it off"
            )
        reach = math.hypot(x, y)
        if not RING_M[0] <= reach <= RING_M[1]:
            warnings.append(
                f"{person} is {reach * 100:.0f} cm from the base, outside the arm's steady reach "
                f"({RING_M[0] * 100:.0f} to {RING_M[1] * 100:.0f} cm)"
            )
        angle = math.degrees(math.atan2(y, x))
        if abs(angle) > MAX_ANGLE_DEG:
            warnings.append(
                f"{person} is {abs(angle):.0f}° to the side, out by the trays (the pick area is ±{MAX_ANGLE_DEG:.0f}°)"
            )
        tray, gap = _tray_gap(name, x, y)
        if gap < TRAY_GAP_M:
            warnings.append(
                f"{person} is {max(gap, 0) * 100:.0f} cm from {tray}'s wall: the jaws may catch the tray"
            )
        if turn % 360:
            warnings.append(
                f"{person} is turned {turn:g}° from facing the arm: the wrist must turn to grip it square"
            )
    for (a, pa), (b, pb) in itertools.combinations(challenge.objects.items(), 2):
        apart = math.dist(pa[:2], pb[:2])
        if apart < SPACING_M:
            warnings.append(
                f"{PERSON_NAMES[a]} and {PERSON_NAMES[b]} are {apart * 100:.0f} cm apart: open jaws span about "
                f"9 cm, so a grasp can push the neighbour (the desk keeps {SPACING_M * 100:.0f} cm)"
            )
    stays = [PERSON_NAMES[n] for n, tray in challenge.expect.items() if tray is None]
    if stays:
        warnings.append(f"{', '.join(stays)}: expected to stay out of the bins")
    return warnings


def grade_challenge(challenge: Challenge, in_bin: dict[str, str | None]) -> dict[str, Any]:
    """Grade a final arrangement (twin names) against the challenge, in the designer's names."""
    result = grade(challenge.task(), in_bin)
    result["in_bin"] = {challenge.person(n): in_bin.get(n) for n in challenge.objects}
    result["challenge"] = challenge.name
    return result


def verdict_of(possible: bool, true_done: bool, aim_done: list[bool]) -> str:
    """What a proof shows.

    For a possible challenge: ``not_doable`` (even true positions fail),
    ``fragile`` (true positions work, aiming like an agent sometimes fails) or
    ``doable``. For an impossible one: ``impossible`` or ``possible_after_all``.
    """
    if not possible:
        return "possible_after_all" if true_done else "impossible"
    if not true_done:
        return "not_doable"
    return "doable" if all(aim_done) else "fragile"


def _with_aim_error(service, rng) -> None:
    """Point like an agent's pointer: off by up to 1.5 cm from far views and
    0.5 cm from the wrist; from overhead, the tray of the object's colour
    instead of the object half the time."""
    real_locate = service.locate

    def noisy_locate(name=None, **kwargs):
        answer = real_locate(name, **kwargs)
        if not answer.get("ok"):
            return answer
        view = kwargs.get("view")
        spread = WRIST_ERROR_M if view == "wrist" else AIM_ERROR_M
        dx, dy = rng.uniform(-spread, spread, size=2)
        colour = APPEARANCE.get(name or "", (None,))[0]
        tray = TRAYS.get(f"{colour}_bin")
        if view == "overhead" and tray is not None and rng.random() < TRAY_MIXUP_RATE:
            dx = tray.center_xy_m[0] - answer["xyz_m"][0]
            dy = tray.center_xy_m[1] - answer["xyz_m"][1]
        answer["xyz_m"] = [answer["xyz_m"][0] + dx, answer["xyz_m"][1] + dy, answer["xyz_m"][2]]
        if "jaw_move_m" in answer:
            move = answer["jaw_move_m"]
            answer["jaw_move_m"] = {**move, "dx": move["dx"] + dx, "dy": move["dy"] + dy}
        return answer

    service.locate = noisy_locate


def _hidden_objects(service, challenge: Challenge) -> list[str]:
    """Objects the front or overhead camera cannot see clearly at the start
    (``locate`` looks from the front; overhead sees the whole desk)."""
    import mujoco

    from xbox_soarm_teleop.twin.cameras import point_camera

    model, data = service.sim.model, service.sim.data
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.enable_segmentation_rendering()
    hidden = []
    try:
        for view, least in (("front", 500), ("overhead", 200)):  # a 3 cm cube: about 900 and 350
            camera = mujoco.MjvCamera()
            point_camera(camera, model, view)
            renderer.update_scene(data, camera=camera)
            seg = renderer.render()
            on_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
            for name in challenge.objects:
                body = service.grasp.objects[name].body_id
                geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body]
                seen = int((on_geom & (seg[..., 0][..., None] == geoms).any(-1)).sum())
                # Small objects show fewer pixels: scale by footprint against a 3 cm cube.
                scale = (_HALF[name][0] * _HALF[name][1]) / (0.015 * 0.015)
                if seen < least * min(scale, 1.0):
                    hidden.append(
                        f"{challenge.person(name)} is hard to see from the {view} camera ({seen} pixels)"
                    )
    finally:
        renderer.close()
    return hidden


def run_arrangement(
    challenge: Challenge, target: dict[str, str | None], *, aim_error: bool, seed: int = 0
) -> dict[str, Any]:
    """Carry ``target`` out with the arm's skills (``pick``, ``place``) on a
    fresh twin set up as the challenge says, as fast as the simulation runs.

    From true positions (``aim_error`` False), or looking through a pointer
    whose aim is off as an agent's is. Returns whether the arrangement was
    reached, where everything ended (designer's names), and each step.
    """
    import numpy as np

    from xbox_soarm_teleop.twin import build_twin
    from xbox_soarm_teleop.twin.pointing import make_pointer

    service = build_twin(
        scene=challenge.scene,
        with_controller=False,
        hide_positions=aim_error,
        pointer=make_pointer("oracle") if aim_error else None,
    )
    hidden: list[str] = []
    alive = threading.Event()
    alive.set()

    def loop():
        # Everything that draws runs here, and the twin closes here: its
        # cameras belong to this thread (closing them elsewhere hangs).
        try:
            while alive.is_set():
                service.tick()  # no sleep: as fast as the simulation runs
        finally:
            service.close()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        service.load_layout(challenge.layout()).result(timeout=30)
        if not aim_error:
            hidden = service.call(lambda: _hidden_objects(service, challenge)).result(timeout=60)
        else:
            _with_aim_error(service, np.random.default_rng(seed))
        steps = []
        for name, tray in target.items():
            if tray is None:
                continue
            picked = service.pick(name, client_id="proof")
            step = {"object": challenge.person(name), "tray": tray, "pick": picked}
            if picked.get("ok"):
                step["place"] = service.place(tray, client_id="proof")
            steps.append(step)
        start = service.observe_truth()["sim_time_s"]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and service.observe_truth()["sim_time_s"] - start < 1.0:
            time.sleep(0.02)  # let what was released settle: a second of simulated time
        objects = service.observe_truth()["scene"]["objects"]
    finally:
        alive.clear()
        thread.join(timeout=10)
    in_bin = {name: objects[name].get("in_bin") for name in challenge.objects}
    off = [challenge.person(n) for n in challenge.objects if objects[n].get("off_table")]
    done = all(in_bin.get(n) == t for n, t in target.items()) and not off
    return {
        "done": done,
        "in_bin": {challenge.person(n): b for n, b in in_bin.items()},
        "off_table": off,
        "steps": steps,
        "hidden": hidden,
    }


def prove(challenge: Challenge, *, aim_runs: int = 3, seed: int = 0) -> dict[str, Any]:
    """Can the arm do it? The expected arrangement by script from true
    positions, then ``aim_runs`` times with an agent's aim error.

    An impossible challenge tries what it ``asked`` for instead; a tray that
    does not exist settles it without moving anything.
    """
    target = challenge.asked if not challenge.possible else challenge.expect
    if not challenge.possible:
        if not challenge.asked:
            return {"verdict": "not_checked", "why": "no 'asked' arrangement to try", "hidden": []}
        missing = sorted({t for t in challenge.asked.values() if t is not None and t not in TRAYS})
        if missing:
            return {
                "verdict": "impossible",
                "why": f"the desk has no {', '.join(missing)} (trays: {', '.join(TRAYS)})",
                "hidden": [],
                "true_run": None,
                "aim_runs": [],
            }
    true_run = run_arrangement(challenge, target, aim_error=False, seed=seed)
    aims = []
    if challenge.possible and true_run["done"]:
        aims = [
            run_arrangement(challenge, target, aim_error=True, seed=seed + i + 1)
            for i in range(aim_runs)
        ]
    verdict = verdict_of(challenge.possible, true_run["done"], [run["done"] for run in aims])
    why = {
        "doable": "done from true positions and every time with an agent's aim",
        "fragile": f"done from true positions, but {sum(not r['done'] for r in aims)} of {len(aims)} "
        "tries with an agent's aim failed",
        "not_doable": "the arm's own skills could not do it even from true positions",
        "impossible": "the arm's own skills could not do what was asked, as intended",
        "possible_after_all": "the arm did what was asked: the challenge is not impossible",
    }[verdict]
    return {
        "verdict": verdict,
        "why": why,
        "true_run": true_run,
        "aim_runs": aims,
        "hidden": true_run["hidden"],
    }

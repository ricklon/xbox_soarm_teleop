"""The names an agent sees for objects: part of the experiment, not a detail.

A real robot is not told what things are called; it sees them. Names that
say what an object looks like ("blue_cube") let a blind agent sort by
reading instead of looking, and a name that says nothing ("pick_cube") left
one guessing. So the twin can show objects to agents under other names
(``TwinService.naming``):

- ``as_is``: the twin's own names;
- ``truthful``: what each looks like (``red_cube``);
- ``neutral``: ``object_1``, ``object_2``, ... in shuffled order;
- ``misleading``: named for another object's colour (the red cube shows as
  ``blue_cube``), to see whether an agent believes words or its eyes.

Only the agent's view changes (``twin.mcp_server`` renames at its tools);
logs, grading and the answer key keep the twin's own names.
"""

from __future__ import annotations

import random

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli

NAMINGS = ("as_is", "truthful", "neutral", "misleading")

# What each object looks like: (colour, shape), as a name would put it.
APPEARANCE: dict[str, tuple[str, str]] = {
    "pick_cube": ("red", "cube"),
    "red_cube_small": ("red", "small_cube"),
    "red_bar": ("red", "bar"),
    "blue_cube": ("blue", "cube"),
    "green_cube": ("green", "cube"),
    "yellow_cube": ("yellow", "cube"),
    "orange_box": ("orange", "box"),
    sim_cli.BEAN_BAG_NAME: ("blue", "bean_bag"),
}
_SPARE_COLOURS = ("purple", "orange", "white")
_NEUTRAL_SEED = 2026  # fixed, so a condition shows the same names every run


def shown_names(objects, naming: str) -> dict[str, str] | None:
    """The name the agent sees for each object (None: the twin's own names).

    Args:
        objects: The twin's names of the objects on the table, in layout order.
        naming: One of ``NAMINGS``.
    """
    objects = list(objects)
    if naming not in NAMINGS:
        raise ValueError(f"Unknown naming '{naming}'. Known: {', '.join(NAMINGS)}")
    if naming == "as_is":
        return None
    if naming == "neutral":
        numbers = list(range(1, len(objects) + 1))
        random.Random(_NEUTRAL_SEED).shuffle(numbers)
        return {name: f"object_{n}" for name, n in zip(objects, numbers)}
    looks = [APPEARANCE.get(name, (None, name)) for name in objects]
    if naming == "truthful":
        return {
            name: f"{colour}_{shape}" if colour else shape
            for name, (colour, shape) in zip(objects, looks)
        }
    # misleading: each takes the colour of the next object along (a derangement
    # of the colours on the table), or a colour nothing has when that fails.
    colours = [colour for colour, _ in looks]
    shifted = colours[1:] + colours[:1]
    names, used = {}, set()
    for name, (own, shape), said in zip(objects, looks, shifted):
        candidates = [said] + [c for c in dict.fromkeys(colours) if c != said] + list(_SPARE_COLOURS)
        for colour in candidates:
            label = f"{colour}_{shape}"
            if colour and colour != own and label not in used:
                break
        names[name] = label
        used.add(label)
    return names

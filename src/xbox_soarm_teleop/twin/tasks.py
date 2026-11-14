"""Sort tasks: what a person asks for, in their own words, and how it is checked.

The scene only puts bins and objects on the table. What goes where comes from
the request, so the same scene serves many tasks ("sort by colour", "swap the
red and the blue", "put only the red one away"), each asked in several ways.

A task pairs its wordings with what must be true at the end:

- ``expect``: the bin each object must be in (None: in no bin), when one
  arrangement is right; or
- ``check``: a test of the final arrangement, when several are right
  ("no cube in the bin of its own colour"), with ``example``, one arrangement
  that passes, so a script can prove the task can be done.

``grade`` scores a final arrangement (object -> bin or None) against a task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

InBin = dict[str, "str | None"]

COLOUR_BINS = {"pick_cube": "red_bin", "blue_cube": "blue_bin", "green_cube": "green_bin"}


@dataclass(frozen=True)
class SortTask:
    """One thing a person may ask for in a sorting scene.

    Args:
        name: Short id ("by_colour").
        scene: The layout it is asked in (``twin.layouts``).
        requests: The same request in different words; a trial uses each in turn.
        expect: The bin each object must end in (None: in no bin), if one
            arrangement is right.
        check: A test of the final arrangement, if several are right.
        check_text: What ``check`` tests, in words, for reports.
        example: One arrangement that passes ``check`` (for the scripted proof).
        ambiguous: The request leaves something open on purpose; the agent
            should say what it assumed. ``expect`` or ``check`` covers only what
            the request does pin down.
    """

    name: str
    scene: str
    requests: tuple[str, ...]
    expect: InBin = field(default_factory=dict)
    check: Callable[[InBin], bool] | None = None
    check_text: str = ""
    example: InBin = field(default_factory=dict)
    ambiguous: bool = False

    def target(self) -> InBin:
        """An arrangement that completes the task: ``expect``, or ``example``."""
        return dict(self.expect or self.example)


def _each_in_a_bin_not_its_own(in_bin: InBin) -> bool:
    placed = [in_bin.get(name) for name in COLOUR_BINS]
    return (
        all(placed)
        and len(set(placed)) == len(placed)
        and all(in_bin.get(name) != own for name, own in COLOUR_BINS.items())
    )


def _coloured_ones_sorted(in_bin: InBin) -> bool:
    return all(in_bin.get(name) == own for name, own in COLOUR_BINS.items())


TASKS: dict[str, SortTask] = {
    task.name: task
    for task in (
        SortTask(
            name="by_colour",
            scene="sort_colors",
            requests=(
                "Sort the cubes by colour into the bins.",
                "Put each cube in the bin that matches its colour.",
                "Tidy up the table: every block goes in the bin of the same colour.",
            ),
            expect=dict(COLOUR_BINS),
        ),
        SortTask(
            name="swap",
            scene="sort_colors",
            requests=(
                "Put the red cube in the blue bin and the blue cube in the red bin. "
                "Leave the green cube where it is.",
                "Swap them: red cube into the blue bin, blue cube into the red bin. "
                "Don't touch the green one.",
            ),
            expect={"pick_cube": "blue_bin", "blue_cube": "red_bin", "green_cube": None},
        ),
        SortTask(
            name="only_red",
            scene="sort_colors",
            requests=(
                "Put only the red cube away, in the red bin. Leave the others on the table.",
                "Just the red one, please: into its bin. Nothing else moves.",
            ),
            expect={"pick_cube": "red_bin", "blue_cube": None, "green_cube": None},
        ),
        SortTask(
            name="mismatch",
            scene="sort_colors",
            requests=(
                "Put every cube in a bin, one cube per bin, but never in the bin of its own colour.",
            ),
            check=_each_in_a_bin_not_its_own,
            check_text="each cube in a different bin, none in the bin of its own colour",
            example={"pick_cube": "blue_bin", "blue_cube": "green_bin", "green_cube": "red_bin"},
        ),
        SortTask(
            name="decoy_stays",
            scene="sort_colors_decoy",
            requests=(
                "Sort the cubes by colour. A cube with no bin of its colour stays on the table.",
            ),
            expect={**COLOUR_BINS, "yellow_cube": None},
        ),
        SortTask(
            name="decoy_open",
            scene="sort_colors_decoy",
            requests=("Put the cubes away by colour.",),
            check=_coloured_ones_sorted,
            check_text="red, blue and green cubes in their own bins; the yellow cube anywhere",
            example={**COLOUR_BINS, "yellow_cube": None},
            ambiguous=True,
        ),
    )
}


def grade(task: SortTask, in_bin: InBin) -> dict:
    """Score a final arrangement: done or not, and, for ``expect`` tasks, how
    many objects are where they should be, in a wrong bin, or not put away."""
    if task.check is not None:
        return {"task": task.name, "done": bool(task.check(in_bin)), "in_bin": dict(in_bin)}
    right = wrong = unsorted = 0
    for name, wanted in task.expect.items():
        now = in_bin.get(name)
        if now == wanted:
            right += 1
        elif now is None:
            unsorted += 1
        else:
            wrong += 1
    return {
        "task": task.name,
        "done": wrong == 0 and unsorted == 0,
        "right": right,
        "wrong": wrong,
        "unsorted": unsorted,
        "in_bin": dict(in_bin),
    }

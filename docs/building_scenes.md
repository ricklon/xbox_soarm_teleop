# Building a scene and a challenge

A scene for the twin is built in four steps. The colour sort (`sort_colors`)
is the worked example; a new scene, such as recycling, follows the same steps.

| Step | What you decide | Where it lives |
|---|---|---|
| 1. The table | Fixed furniture: which bins, where, what colour | `src/xbox_soarm_teleop/sim/table_scenes.py` |
| 2. The layout | Which objects are out, and where they start | `src/xbox_soarm_teleop/twin/layouts.py` |
| 3. The tasks | What a person asks for, in their words, and what must be true at the end | `src/xbox_soarm_teleop/twin/tasks.py` |
| 4. Proof it can be done | A scripted run with the true positions finishes every task | `tests/test_twin_sort.py` |

The scene never says what goes where. That is the request's job, so one scene
serves many tasks, and the agent has to understand what it is asked rather
than follow a rule it is handed.

## 1. The table: a recipe

Every table scene has the same table (x 0.15 to 0.47 m, y -0.10 to 0.18 m,
top at z 0.12 m) and the same objects (the red cube, the bean bag and five
look-alikes). A recipe adds the fixed furniture: bins.

```python
RED_BIN = Bin("red_bin", "red bin", (0.41, 0.13), (0.035, 0.035, 0.01), (0.80, 0.20, 0.18, 0.85))
...
TableScene("sort_bins", (RED_BIN, BLUE_BIN, GREEN_BIN))
```

A `Bin` is a name, words for it, its center, half its inside size and wall
height, and a colour. Each recipe compiles into its own MuJoCo model. A twin,
or a hub with many twins, runs one recipe at a time: a hub started with
`--scene sort_colors` offers only the sort scenes, and one started with the
single-box scenes never shows bins.

A recipe with one bin is a goal-box scene: `observe` gives `scene.goal` and
each object's `in_goal`. A recipe with several bins is a sorting scene:
`observe` gives `scene.bins` and each object's `in_bin` (hidden in
no-coordinates mode, like positions; the room's answer key at
`/r/{key}/scores` has it).

Where the bins go matters. In a row leading away from the base, lowering into
the far bin swung the forearm over the near one and knocked its cube out. The
bins now stand in a column along the far edge, each in its own direction from
the base. They are low trays (2 cm walls): the front camera, which `locate`
uses by default, looks across them, and 6 cm walls hid the cubes behind them.

## 2. The layout: what is on the table

```python
Layout(
    name="sort_colors",
    model_scene="sort_bins",
    description="A red, a blue and a green cube, and a red, a blue and a green bin.",
    objects={"pick_cube": ..., "blue_cube": ..., "green_cube": ...},
)
```

Rules for starts, all found the hard way:

- Objects must be at least 3 cm tall: the jaw center stops 1.5 cm above the table.
- Keep clear of the arm's path from home, which brushes a 4 cm cube at (0.25, 0.04).
- Keep objects near the arm's straight-ahead line. A cube sitting square to
  the table at (0.20, 0.10) is about 40 degrees off from the jaws, which
  reach it at an angle, and it slips.
- Keep objects where the front camera sees them: not behind tall furniture,
  and not under the arm at home (around x 0.18 to 0.22, y near 0), where the
  arm itself hides them.
- Keep objects well behind the bins (toward the base), off the lines from the
  base to each bin, and at least 3 cm from a bin's outer wall.
- Leave room for an agent's grasp: objects at least 8 cm apart and their
  footprints at least 4 cm in from the table's edges. The open jaws span
  about 9 cm and an agent's aim is off by a centimetre or two; with cubes 6 cm
  apart and one 2 cm from the edge, agents pushed neighbours and knocked
  cubes off the table, and `locate`, which assumes an object rests on the
  table, then guessed 12 to 30 cm wrong for the cube on the floor.

`test_starts_leave_room_for_an_agents_grasp` checks these rules for every
layout. On the sort table only three spots in the front camera's view meet
them all, so `sort_colors_decoy` has a red, a blue and a yellow cube (the
green tray stays empty).

## 3. The tasks: the request carries the rule

```python
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
)
```

A task is several wordings of one request, plus what must be true at the end:

- `expect`: the bin each object must be in (`None`: in no bin), when one
  arrangement is right. Moving something the request said to leave is wrong.
- `check`: a test of the final arrangement, when several are right ("every
  cube in a bin, none in the bin of its own colour"), with `example`, one
  arrangement that passes.
- `ambiguous=True`: the request leaves something open on purpose ("put the
  cubes away by colour", with a yellow cube and no yellow bin). The check
  covers only what the request pins down; read what the agent said it assumed.

Writing the expectation is the real design work: it forces you to say exactly
what "sorted" means.

Run a task with an agent:

```bash
uv run python scripts/agent_trial.py --harness loop --model google/gemma-4-31b-it \
    --pointer openrouter:google/gemma-4-31b-it --task swap --runs 4
uv run python scripts/agent_report.py twin_logs/agent_trials.jsonl
```

Each run asks in the next wording, so the runs also test how well the agent
copes with different phrasings. The result says done or not, how many objects
are right, wrong or not put away, where each ended up, and whether the
agent's own `RESULT: done` claim was true.

## 4. Proof it can be done

A challenge nobody can do teaches nothing. `test_every_task_can_be_done`
carries out each task's arrangement by script, with the true positions: pick,
place, lift 5 cm, home, for each object in turn. It then checks that every
object is where the task wants it, so nothing already placed was knocked out.
A new task is tested as soon as it is added.
`test_the_front_camera_sees_every_object_at_the_start` checks that every
object is clearly in the front camera's view, by segmentation, before any
agent has to find it there.

## Not yet built

- `check_success` judges only the goal box, and refuses in sorting scenes.
  Clef can answer "which bin is it in?", but its accuracy has to be measured
  first, as it was for the box.
- An `ask_user` tool, so an agent can ask about an ambiguous request instead
  of guessing.
- Neutral object names. Names like `blue_cube` give the colour away; a real
  robot is not told names, and an agent here could sort by name without looking.
- Before agent benchmarks: check that the pointing model finds "red cube" and
  not the red bin, and finds a cube once it is inside a bin.

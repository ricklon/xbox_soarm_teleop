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

## 1. The cell: a recipe

A recipe (`TableScene`) is a whole work cell: the table (its size and
height), the bins on it, and optionally its own cameras.

```python
DESK = Table(center_xy_m=(0.17, 0.0), half_xy_m=(0.27, 0.42), top_z=0.002)
RED_BIN = _tray("red_bin", "red bin", 45.0, ...)    # 27 cm out, at +45 degrees
TableScene("sort_bins", DESK, (RED_BIN, BLUE_BIN, GREEN_BIN), views=DESK_VIEWS)
```

Each recipe compiles into its own MuJoCo model, and a twin runs one recipe
at a time: a hub started with `--scene sort_colors` offers only the sort
scenes. A hub started with `--switch-games` compiles one model with every
recipe's furniture instead (`CLASSROOM`). Each recipe's table, bins and
cameras ride on one mocap body, and each arm poses its own game's body at the
origin and parks the others. Switching game is then a reset, and the arms
still share one model. `tests/test_twin_games.py` checks that each game on
that model matches its own model: what `observe` says, what the cameras show,
and how low the arm may go.

**Size the cell to the arm, not the arm to the table.** The SO-101 grasps
from above in a ring around its base: steadily from about 17 to 36 cm out,
across about +-80 degrees of base turn (it reaches 40 cm and +-110 degrees less
steadily), with a dead zone inside 15 cm where it folds onto itself. The
first pick-and-place table (32 x 28 cm, raised 12 cm) covered about a quarter
of that ring, and its far corners were out of reach; crowding three cubes and
three bins onto it caused most of the failures below. The sorting desk:

- **At the base's height**, like a real desk the arm is clamped to, 54 x 84 cm.
  The arm's configured workspace floor is 5 cm, which would keep the jaws off
  everything on a desk, so for any table scene the twin puts the floor 5 mm
  over the table instead. A real desk-mounted arm needs the same change in
  its own configuration.
- **Items in front, trays to the sides.** Placing means turning the base,
  so the arm never passes over items still to be sorted and nothing stands
  between the cameras and them. Trays are 27 cm out at +45, +72 and -70
  degrees: inside the enforced workspace (x >= 5 cm, |y| <= 30 cm for the tool
  point), 10 cm inside (a cube carried to the side lands turned with the
  gripper), 2 cm walls (the cameras see over them).
- **Its own cameras**, built into the model and aimed at the desk: `front`
  (from beyond the pick area), `overhead` (the whole ring) and `side`.
  `point_camera` and `locate`'s projection use a model's own camera when it
  has one, so the single-box scene keeps its shared presets and its
  benchmarks still compare.

**Movable boxes.** A `Bin` with `movable=True` goes in the recipe's `boxes`,
not its `bins`: it is built as a free body (a floor and four walls), so a push
slides it and whatever is inside rides along. To the twin it is an object like
the cubes: parked out of sight unless a layout puts it out, hidden in
no-coordinates mode, and marked `"box": true` in `observe`. Grading follows
it: `in_bin` names the box wherever it has been pushed, turned or not. The
sorting desk holds an orange box (9 cm across, 7 cm inside, 2.4 cm walls,
80 g), out only in the `sort_colors_box` layout, so the other layouts, their
names and their benchmarks are unchanged.

A recipe with one bin is a goal-box scene: `observe` gives `scene.goal` and
each object's `in_goal`. A recipe with several bins is a sorting scene:
`observe` gives `scene.bins` and each object's `in_bin` (hidden in
no-coordinates mode, like positions; the room's answer key at
`/r/{key}/scores` has it).

## 2. The layout: what is on the table

```python
Layout(
    name="sort_colors",
    model_scene="sort_bins",
    description="A red, a blue and a green cube, and a red, a blue and a green bin.",
    objects={"pick_cube": ..., "blue_cube": ..., "green_cube": ...},
)
```

An item's start can be turned, `(x, y, z, yaw)`. On the desk every item faces
the base (`yaw = atan2(y, x)`), so the jaws meet it square: a cube sitting
square to the table but 25 degrees off to the side is 40 degrees off from the
jaws, and slips.

Rules for starts, all found the hard way and checked by
`test_starts_leave_room_for_an_agents_grasp`:

- In the steady part of the ring (20 to 33 cm out) and in front of the arm
  (within 40 degrees), at least 5 cm in from the desk's edges.
- At least 10 cm apart, and 5 cm from a tray. The open jaws span about 9 cm
  and an agent's aim is off by a centimetre or two: on the crowded table,
  neighbours 6 to 8 cm away were pushed, and cubes walked off the edge after
  repeated misses.
- Off the strip straight ahead of the arm, where the arm at home hides
  objects from the overhead camera.

`test_the_cameras_see_every_object_at_the_start` checks, by segmentation, that
the front and overhead cameras see every item clearly.

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

`test_an_agents_misses_do_not_push_neighbours_or_drop_anything` grasps every
item with its aim off by up to 1.5 cm, as an agent's is: misses are allowed,
pushing a neighbour more than 1 cm or anything off the desk is not. With the
current grasp recipe, 6 of 9 such grasps miss. That is the baseline a `pick`
skill has to beat.
`test_the_front_camera_sees_every_object_at_the_start` checks that every
object is clearly in the front camera's view, by segmentation, before any
agent has to find it there.

## Skills: pick and place

An agent can steer the jaws itself (`move_by`, `set_gripper`, `locate`), but a
language model deciding every few centimetres, a second or two per step, from
estimates a centimetre or two off, misses most grasps: 6 of 9 with the aim
off by up to 1.5 cm. The arm offers two skills instead (`twin/skills.py`),
the way real robots give a planner skills:

- `pick(name)`: turn the jaws while the arm is high (turning them later,
  from home, swung them down to the table); look from overhead and from the
  front and check the two agree (from overhead a cube and the floor of a tray
  of its colour are both coloured squares, and the pointer chose the tray in
  the first skills trial; if one view says inside a bin and the other does
  not, believe the other, otherwise ask the side view); hover with the jaws
  open; centre them with a close look through the wrist camera (about half a
  centimetre off), twice; down, close, and a check by feel (`attached`).
  After a miss it opens, lifts, looks again and retries, up to three times.
  With 1.5 cm aim error from far views, 0.5 cm from the wrist, and the
  overhead look on the tray half the time, it picked 9 of 9, each at the
  first attempt (`test_pick_beats_steering_by_hand`).
- `place(bin)`: carry over the bin, lower, let go, lift 5 cm, go home. A
  movable box may have been pushed, so `place` looks where it is first.
- `push(name or description, direction + distance_m, or dx_m/dy_m)`: close the
  jaws, get behind the object, go low (jaw centre 1.2 cm over the table),
  slide it slowly (3 cm/s), lift, look again, and push the rest, up to four
  times, stopping within 1 cm. It never pushes back toward the base: coming
  down in front of an object, the gripper's body hangs back onto it, so an
  overshoot is reported, not corrected. Whatever is in the way is pushed too.
  `pick` refuses a box and says to push it.

They use only what the arm can know: its pointing model (or `observe` when
coordinates are shown and there is no pointing model) and whether it holds
something. They never read the truth, so they work in no-coordinates mode and
their answers give nothing away. They run through the twin's own actions, one
at a time, so a stop, a person taking over or the workspace limits end them as
they end any motion. The agent's job becomes deciding what goes where.

## Checking: which tray is it in?

`check_success(name, bin)` asks the Clef judge about a front picture, with the
cell's own trays as the options (plus on the table, in the gripper, on the
floor), and answers P(in that bin), a verdict (done at 0.9 or more, retry at
0.1 or less, look again between) and where the object most likely is. The
truth is logged beside every answer and never returned.

Measured first (`scripts/tray_judge_benchmark.py`: each cube in each tray,
the right one and the wrong ones, on the desk and held, with the other cubes
as look-alikes; 42 situations):

| judge, picture | right | confident (0.9+) answers right |
|---|---|---|
| Clef, front | 95% | 34 of 34 |
| Clef, overhead | 57% | 4 of 5 |
| Clef-flash, front | 95% | 21 of 21 |
| Clef-flash, overhead | 38% | 4 of 13 |

A movable box out on the table is one more option, graded against where it
is now. How well Clef tells "in the box" from "beside it" is not measured yet.

From overhead, at 78 cm, the cubes are small and both judges are often
confidently wrong; from the front, every confident answer was right. So the
check uses Clef and the front camera.

## Not yet built

- The gripper's bounding box dips about 6 mm below the desk at the lowest
  grasp (the jaw center stops 1.5 cm over the table). The arm moves
  kinematically, so nothing stops it; a real arm would scrape. Check against
  the real gripper before trusting low grasps.

- An `ask_user` tool, so an agent can ask about an ambiguous request instead
  of guessing.
- Neutral object names. Names like `blue_cube` give the colour away; a real
  robot is not told names, and an agent here could sort by name without looking.
- Before agent benchmarks: check that the pointing model finds "red cube" and
  not the red bin, and finds a cube once it is inside a bin.

# Vision benchmark on the digital twin

The real arm has no ground-truth coordinates: an agent has to find objects by
looking. The MuJoCo twin knows exactly where everything is, which makes it a
grader for perception. This page covers how to run the twin as a benchmark and
the results so far.

Status: all three steps are done: no-coordinates mode and scoring, a
`locate` tool backed by pointing models, and a success judge (Cloudflare's
Clef) graded against the twin's own state.

## Results so far

Pointing, the same 16 random placements (seed 0), 640x480, against the twin's
truth (`twin_logs/pointing_benchmark_molmo_gemma.json`,
`twin_logs/pointing_benchmark_gemini_openrouter.json`,
`twin_logs/pointing_benchmark_gemini_er2.json`,
`twin_logs/pointing_benchmark_gemma31b.json`):

| backend | found | median error | 90th percentile | s per call |
|---|---|---|---|---|
| Gemini 3.5 Flash (OpenRouter) | 58/58 | 0.1-0.3 cm | 0.3-1.0 cm | 1.7-2.3 |
| Gemini Robotics-ER 2 (Gemini API) | 58/58 | 0.2-0.4 cm (wrist 1.0) | 0.7-1.9 cm | 3-16 |
| Gemma 4 31B (OpenRouter, paid) | 58/58 | 0.3-0.5 cm | 0.7-1.6 cm | 2.4-4.4 |
| Molmo2-ER (local, MPS) | 58/58 | 0.5-0.7 cm | 1.2-1.7 cm | 4.2-4.7 |
| Gemma 4 12B (local Ollama) | 58/58 | 0.9-1.1 cm | 1.4-3.1 cm | 1.9-2.8 |
| perfect pixel (segmentation) | 58/58 | 0.1-0.3 cm | 0.2-1.0 cm | - |
| image center (for scale) | - | 6.8-12.8 cm | 10-21 cm | - |

Ranges are over the four views (front, overhead, side, wrist), with the
first wrist camera; see below for the wrist camera as mounted on the real arm. Gemini 3.5
Flash is as close as the method allows: the rest is the gap between the
visible surface and the object's center.

Pick and place with no coordinates, a text-only agent (GLM 5.3 Flash via
opencode) using `locate`, the rigid cube (`pick_place_basic`), graded from the
session log:

| pointer | in the goal box | time |
|---|---|---|
| oracle (truth; control) | 1/1 | 48 s |
| Gemini 3.5 Flash | 3/3 | 57-120 s |
| Gemini Robotics-ER 2 | 1/1 | 124 s (see below) |
| Gemma 4 12B | 3/3 | 72-84 s |
| Molmo2-ER | 2/3 | 60-64 s (the third timed out after 8 misses) |

Gemini Robotics-ER 2 has one clean run. Five more were lost when the free
GLM endpoint stopped answering (even a one-word ping); with the same model
paid on OpenRouter (`openrouter/z-ai/glm-5.3-flash`), the next run timed out
because Robotics-ER 2 itself had slowed to 40-70 s a call, with half the calls
timing out (it took 3-16 s during the pointing benchmark earlier the same
day). Too slow, at those times, for an agent that locates every few moves.

`agent_trial.py` gives opencode an empty stdin and its own data folders:
`opencode run` otherwise waits on an open stdin, and with the OpenCode
desktop app running, a command-line opencode sharing its data stalls before
its first step.

iceman-free fails even with the oracle pointer (0/2): it computes moves itself
and gets the height wrong, and it issues motion calls in parallel. GLM uses
`jaw_move_m` as given.

### The wrist camera, as mounted on the real arm

The twin's wrist camera now sits where the real arm's Seeed Studio camera
does (RGB USB camera, OV9732, 100 degrees, 1280x720): on a post 50 mm out
from the gripper's side, 20 mm along the fingers, tilted 40 degrees in, with
a 61-degree vertical field of view, drawn at 16:9. Both jaw tips come in from
the picture's left and right with the object between them, as on the real
camera (checked against it by eye, holding a cube). The first camera, beside
the fixed jaw, lost the object once the jaws were over it.

Pointing from the grasp's starting pose (6 cm above the object, jaws side to
side), 640x360, 11 placements (5 dropped: moving there nudged the object)
(`twin_logs/pointing_benchmark_wrist_seeed.json`):

| backend | median error | 90th percentile | s per call |
|---|---|---|---|
| Gemini 3.5 Flash (OpenRouter) | 0.41 cm | 1.0 cm | 2.5 |
| Gemma 4 31B (OpenRouter) | 0.51 cm | 1.0 cm | 3.4 |
| perfect pixel | 0.65 cm | 0.9 cm | - |
| Gemini Robotics-ER 2 | 0.71 cm | 1.1 cm | 2.1 |
| Molmo2-ER | 0.84 cm | 1.2 cm | 3.8 |
| Gemma 4 12B (local) | 3.2 cm | 3.9 cm | 1.8 |
| image center | 5.3 cm | 7.0 cm | - |

Close up and steeply angled, the visible surface's centroid is 0.65 cm from
the object's center, so a model that points nearer the center can beat the
"perfect" pixel (the two Gemini/Gemma OpenRouter models do). The local Gemma 4
12B, fine in the fixed views, is poor this close.

## Step 1: no-coordinates mode

Start a twin, or a workshop hub, with `--no-coordinates`:

```bash
uv run soarm-twin --headless --no-controller --no-coordinates
```

```bash
uv run soarm-hub --no-coordinates
```

What an agent still gets from `observe` and tool results:

- `scene.object_positions: "hidden"`, so the agent can tell which mode it is in
- each object's `size_m`, `grip_width_m` and `attached` (the arm knows what it
  is picking up and can feel whether it holds it), and `holding` in motion results
- the goal box (`scene.goal`), the arm's pose (`ee`, `joints_deg`) and
  `jaw_center_xyz_m`
- the `camera` tool

What it no longer gets:

- each object's `xyz_m`, plus `in_goal` and `off_table`, which follow from it
- the offsets in a refused grasp's "missed" event. Normally that event says
  exactly how far to `move_by`. In this mode it only says the jaws missed and
  to look again.

The "slipped" event still gives the wrist roll that squares the jaws. That
reveals the object's orientation, not its position.

### Scoring guesses

`report_estimate(name, x, y, z)` is an MCP tool: the agent reports where it
thinks an object's center is (m, base frame). The agent only learns that the
guess was recorded, because an agent told its error could home in on the
answer. The twin scores the guess against the true position at that moment:

- `TwinService.estimate_scores()` gives `error_cm` (3D), `error_xy_cm` and
  `error_z_cm` (signed), with the estimate and the truth
- the session log gets an `estimate` line with the same fields

`TwinService.observe_truth()` returns the full snapshot for graders. Session
logs always hold the true positions (`observation` lines), whatever the mode.

`scripts/twin_smoke.py` needs coordinates. Against a no-coordinates twin it
fails with a message that says so.

## Step 2: `locate`, a pointing model behind a tool

`locate(name, view="front")` takes a 640x480 picture and asks a pointing model
for the pixel showing the object ("red cube", "blue object"). The pixel's ray
is followed down to the plane the object's center sits at when it rests on
the table, or on `rest_z_m` (for example the goal box floor). The result has
the estimate `xyz_m` and `jaw_move_m`, the `move_by` that puts the jaw center
on it, computed from the estimate and the arm's pose. Every answer is scored
against the truth (`estimate_scores()`, source `locate:<backend>:<view>`); the
agent never sees the score.

The conversion (`twin/projection.py`) uses each camera's exact pose and field
of view. `tests/test_twin_projection.py` checks it against MuJoCo's own
renderer: the cube's corners project onto its drawn silhouette within a pixel,
and the centroid of its rendered pixels maps back within 1 cm in every view
(0.1 cm in the fixed views).

Front is the default view: in the overhead view the arm at home hides part of
the cube (0.8-1.0 cm error even with a perfect pixel), and the wrist camera
loses the object once the jaws are centered over it.

### Backends (`--pointer`)

| spec | model | needs |
|---|---|---|
| `oracle` | the truth, projected | nothing (tests and controls) |
| `molmo[:url]` | Molmo2-ER (allenai, 4B) | `uv run scripts/molmo_server.py` (own env, ~19 GB weights) |
| `ollama[:model]` | default `gemma4:12b` | a local Ollama |
| `gemini[:model]` | default `gemini-robotics-er-2-preview` | `GEMINI_API_KEY` |
| `openrouter[:model]` | default `google/gemma-4-31b-it:free` | `OPENROUTER_API_KEY` |

Keys are read from the environment, else from a `.env` file: the current
folder's, then the project's, then `~/.env` (the first one that has the key
wins; an empty value counts as unset). Copy `.env.example` to `.env` in the
project and fill in what you use; git ignores `.env`. Keys are sent only in
headers and masked in every error. OpenRouter has no Robotics-ER, Molmo2-ER or Clef; it does have general
Gemini and Gemma 4 models.

Gemini's free tier allows 20 requests a day for Robotics-ER 2, less than
one benchmark run (58 calls): enable billing on the key's project. HTTP 429
replies are retried up to four times, waiting as long as the API asks.
OpenRouter's free Gemma 4 31B is often rate-limited upstream; the paid one
(`google/gemma-4-31b-it`) costs a fraction of a cent a call.

Molmo2-ER takes the category literally. Shown the bean bag, it found a "blue
bean bag" in 1 of 9 pictures and a "blue object" in 9 of 9, so the bean bag's
description is "blue object". Gemma 4 needs `think: false`, or it writes for
minutes before answering.

### Running it

```bash
uv run scripts/molmo_server.py          # in its own terminal, for --pointer molmo
```

```bash
uv run python scripts/pointing_benchmark.py --pointer molmo --pointer ollama:gemma4:12b --trials 16
```

```bash
uv run python scripts/agent_trial.py --model opencode/glm-5.3-flash-free --pointer molmo --runs 3
```

```bash
uv run soarm-twin --no-coordinates --pointer molmo
```

`pointing_benchmark.py` puts the cube or the bean bag at a random spot,
turned at random, draws every view, and reports per backend and view the
share found, median, mean and 90th-percentile error in cm, pixel error, and
seconds per call. It drops wrist trials where aiming the camera moved the
object.

`agent_trial.py` starts its own twin on a spare port and gives opencode a
throwaway config. That config turns off any other `soarm-arm` server (such as
a workshop hub on 8800) and opencode's shell and file tools. It then grades
the run from the session log, not from what the agent says.

## Step 3: a success judge from pictures (Clef)

On the real arm nothing reports `in_goal` or `attached`. `twin/judge.py` asks
Cloudflare's Clef (`@cf/cloudflare/clef`, 27B, and `clef-flash`, 9B, on
Workers AI) about one object in 1-4 camera pictures. Clef takes a state (text
and images) and typed questions, and returns a probability for every option:

- `in_goal`: resting in the goal box? (yes/no)
- `held`: held in the gripper? (yes/no)
- `tipped`: resting tilted on an edge or corner? (yes/no)
- `where`: goal box, table, gripper or floor? (one of four)

The right answers come from the simulator (`truth_labels`), with the same
`in_goal` test `observe` uses. `twin/scenarios.py` sets up the situations, all
on the table top: flat on the table; in the box; just outside a wall (5-30 mm
gap); held over the table and held over the box (a real grasp); and tipped
onto an edge on the table and inside the box. The cube and the bean bag take
turns.

### Results

41 situations (6 of each kind; one grasp failed to set up), 640x480
pictures: 1 = front, 2 = front + overhead, 4 = front, overhead, side, wrist
(`twin_logs/judge_benchmark.json`). Each cell is accuracy at 0.5 / Brier
score / expected calibration error; lower Brier and ECE are better. The base
rate answers every question with how often it is yes.

| judge | images | in_goal | held | tipped | where (acc / p on truth) |
|---|---|---|---|---|---|
| base rate | - | 0.73 / 0.196 / 0 | 0.73 / 0.196 / 0 | 0.71 / 0.207 / 0 | 0.46 / 0.36 |
| clef | 1 | 0.98 / 0.031 / 0.056 | 0.95 / 0.043 / 0.123 | 0.90 / 0.071 / 0.130 | 0.93 / 0.88 |
| clef | 2 | 0.88 / 0.063 / 0.102 | 0.93 / 0.045 / 0.119 | 0.90 / 0.090 / 0.153 | 0.98 / 0.88 |
| clef | 4 | 0.95 / 0.052 / 0.101 | 1.00 / 0.026 / 0.123 | 0.85 / 0.104 / 0.125 | 0.98 / 0.89 |
| clef-flash | 1 | 0.93 / 0.053 / 0.078 | 0.90 / 0.064 / 0.128 | 0.85 / 0.124 / 0.127 | 0.90 / 0.86 |
| clef-flash | 2 | 0.88 / 0.068 / 0.092 | 0.93 / 0.063 / 0.119 | 0.83 / 0.118 / 0.115 | 0.90 / 0.85 |
| clef-flash | 4 | 0.93 / 0.065 / 0.129 | 0.90 / 0.047 / 0.138 | 0.83 / 0.120 / 0.133 | 0.98 / 0.83 |

What it shows:

- Clef works as a success judge here, although its model card claims no
  robotics or spatial use. With one front picture it answers "in the goal
  box?" right 40 times in 41 (Brier 0.031 against 0.196 for the base rate),
  and never takes an object just outside the wall for one inside.
- More pictures help little. With all four (the 4-picture rows use the
  real arm's wrist camera; `twin_logs/judge_benchmark_seeed_wrist.json`),
  "held" becomes perfect (1.00), but "tipped" drops (0.90 to 0.85): it calls
  some flat objects next to the box tipped. One front picture is the best
  setting overall, and the cheapest. (With the first wrist camera, "tipped"
  fell to 0.76.)
- Hardest cases: a tipped object inside the box (the walls hide the tilt;
  clef got 3 of 6, clef-flash 1 of 6), and an object held over the box
  (clef-flash says "in the goal box" for half of them).
- Calibration: confident answers can be trusted. Answers at 0.9 or more were
  right 93% of the time, and at 0.1 or less, 100%. In between it is
  underconfident: answers around 0.7-0.85 were nearly always yes. ECE is
  0.06-0.16.
- clef-flash is about twice as fast (0.2 s against 0.4 s per call with one
  picture) and a little worse throughout.
- Cost: 320,784 input tokens for all 246 calls, about $0.04 for clef and
  $0.01 for clef-flash at list prices.

### Running it

```bash
uv run python scripts/judge_benchmark.py --situations 42
```

`--model clef` (or `clef-flash`) picks one model; `--image-sets 1` picks the
picture counts; `--images DIR` saves every picture. It needs
`CLOUDFLARE_API_TOKEN` (a token with Workers AI permission) and
`CLOUDFLARE_ACCOUNT_ID`.

Note: the pointing benchmark placed some objects at y < -0.10 m, beyond the
table's edge (drawn at table height). Pointing is graded the same there, but
`twin/scenarios.py` keeps every judge situation on the table.

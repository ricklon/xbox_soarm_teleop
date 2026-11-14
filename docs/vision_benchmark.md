# Vision benchmark on the digital twin

The real arm has no ground-truth coordinates: an agent has to find objects by
looking. The MuJoCo twin knows exactly where everything is, which makes it a
grader for perception. This page covers how to run the twin as a benchmark and
the results so far.

Status: step 1 (no-coordinates mode and scoring) and step 2 (a `locate` tool
backed by pointing models) are done. Gemini Robotics-ER and OpenRouter models
are wired in but not measured yet (no API keys). Step 3 (a Workers AI success
judge) is not built yet.

## Results so far

Pointing, 16 random placements, 640x480, against the twin's truth
(`twin_logs/pointing_benchmark_molmo_gemma.json`):

| backend | found | median error | 90th percentile | s per call |
|---|---|---|---|---|
| Molmo2-ER (local, MPS) | 58/58 | 0.5-0.7 cm | 1.2-1.7 cm | 4.2-4.7 |
| Gemma 4 12B (local Ollama) | 58/58 | 0.9-1.1 cm | 1.4-3.1 cm | 1.9-2.8 |
| perfect pixel (segmentation) | 58/58 | 0.1-0.3 cm | 0.2-1.0 cm | - |
| image center (for scale) | - | 6.8-12.8 cm | 10-21 cm | - |

Ranges are over the four views (front, overhead, side, wrist).

Pick and place with no coordinates, a text-only agent (GLM 5.3 Flash via
opencode) using `locate`, the rigid cube (`pick_place_basic`), graded from the
session log:

| pointer | in the goal box | time |
|---|---|---|
| oracle (truth; control) | 1/1 | 48 s |
| Gemma 4 12B | 3/3 | 72-84 s |
| Molmo2-ER | 2/3 | 60-64 s (the third timed out after 8 misses) |

iceman-free fails even with the oracle pointer (0/2): it computes moves itself
and gets the height wrong, and it issues motion calls in parallel. GLM uses
`jaw_move_m` as given.

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
headers and masked in every error. Check the Gemini model id when the key is set: it is a
guess. OpenRouter has no Robotics-ER, Molmo2-ER or Clef; it does have general
Gemini and Gemma 4 models.

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

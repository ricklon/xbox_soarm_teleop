# Vision benchmark on the digital twin

The real arm has no ground-truth coordinates: an agent has to find objects by
looking. The MuJoCo twin knows exactly where everything is, which makes it a
grader for perception. This page covers how to run the twin as a benchmark and
the results so far.

Status: step 1 (no-coordinates mode and scoring) is done. Step 2 (a `locate`
tool backed by pointing models) and step 3 (a Workers AI success judge) are
not built yet.

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

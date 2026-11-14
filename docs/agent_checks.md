# Agent checks: does an AI agent use the twin as intended?

Two checks, for two different questions.

| | `scripts/twin_smoke.py` | The 14 checks below |
|---|---|---|
| Asks | Does the twin itself work? | Does an agent follow the instructions and events? |
| Runs | A fixed script, no AI | An AI agent over MCP (opencode, Claude Code, ...) |
| Cost | Free, about 30 s a scene | A model run: tokens, minutes |
| Run it | After every change to the twin | When comparing models, or after changing the instructions or events |

Run the smoke test first: if the twin is broken, an agent's failures say
nothing about the agent.

## 1. Smoke test (no AI)

Against a single twin (`uv run soarm-twin`):

```bash
uv run python scripts/twin_smoke.py --all-scenes
```

Against a workshop hub, with an arm's key in `ARM_KEY` (never type the key on
the command line, where shell history keeps it). This moves that arm: use a
spare one, or tell its student first.

```bash
ARM_KEY="$(python -c 'import json; print(json.load(open("twin_logs/hub_roster.json"))["rooms"][-1]["token"])')" uv run python scripts/twin_smoke.py --url http://127.0.0.1:8800/mcp --all-scenes
```

It loads each scene `list_scenes` offers, picks the object into the goal box
by the procedure the agent instructions give (or, in a scene without a box,
holds it through a lift), checks a camera image comes back, prints a table,
and loads the starting scene again. `--scene NAME` (repeatable) picks scenes;
`--image out.jpg` saves an overhead image per scene.

## 2. The 14 checks (an AI agent)

Paste the block below into the agent, connected to one arm (MCP server
"soarm-arm" in the student setup; rename it to match yours). Save the agent's
session log with the results, so runs can be compared.

```text
You are connected to a simulated robot arm over MCP (server "soarm-arm").
Run these checks in order. After each, report PASS or FAIL with the tool output that shows it.
Do not skip a check because an earlier one failed.

"Center the jaws over X" means: observe, then move_by(dx = X.x - jaw_center_xyz_m.x,
dy = X.y - jaw_center_xyz_m.y, dz = 0); do it twice.
"Grasp X" means: move_to 6 cm above X; set_wrist_roll -1.5; observe X's grip_width_m (W);
set_gripper opening_m = W + 0.05; center the jaws over X; move_by dz = max(X.z, 0.14) - jaw_center z;
set_gripper opening_m = W.

1. list_scenes: expect pick_place_easy (easy, bean_bag) and pick_place_basic (hard, pick_cube), one current.
2. load_scene "pick_place_easy", then observe: only bean_bag with size_m and grip_width_m; top-level
   gripper_opening_m and jaw_center_xyz_m (different from ee.xyz_m); scene.goal.
3. Grasp the bean bag: expect "holding": "bean_bag", and the bag moved under 3 mm before the close.
4. Lift 10 cm, move_to over scene.goal's center at z 0.22, center the jaws over the goal,
   set_gripper position 0, wait a second, observe: in_goal true.
5. load_scene "pick_place_basic", observe: only pick_cube, size_m [0.04, 0.04, 0.04], arm home.
6. The old way: move_to 6 cm above the cube, set_wrist_roll -1.5, move_to 2.5 cm above its
   center, set_gripper opening_m 0.04. Expect holding null and an event "missed ..." that gives
   a set_gripper opening and move_by steps.
7. Do exactly what the "missed" event says, each step in order (the set_gripper, then each
   move_by separately), then set_gripper opening_m 0.04: expect "holding": "pick_cube".
8. Lift 10 cm, move_to (0.333, -0.008, 0.30), then set_wrist_roll 0. Observe: pick_cube's
   xyz_m stays within 1 cm of jaw_center_xyz_m after each step (it rides between the jaws).
9. set_gripper position 1.0: expect holding null and a "squeezed out" event.
10. reset_scene. At wrist roll 0, move_to 2.5 cm above the cube and set_gripper opening_m 0.04:
    expect a "slipped ... set_wrist_roll(angle_rad=X)" event.
11. Grasp the cube, lift 8 cm, carry it to scene.goal (center the jaws over it), open.
    After a second, observe: pick_cube in_goal true.
12. Take the cube out of the box and put it on the table at (0.34, -0.03), using only what
    the tools and events tell you. Report each attempt.
13. load_scene "moon_base": expect ok false, reason invalid_request, both real scene names listed.
14. set_gripper with no arguments: expect ok false, reason invalid_request.

Finish with a table of the 14 checks and anything that surprised you.
```

## What each check is for

| # | Tests | Fails when |
|---|---|---|
| 1–2, 5 | Scenes and the observation fields agents need | A field is missing or renamed |
| 3–4 | The procedure on the forgiving object | The agent cannot follow the basic steps |
| 6–7 | A `missed` event is enough to recover | The hint is wrong, or the agent paraphrases it |
| 8 | A held object stays between the jaws | The carry drifts (a cube floating beside the jaws) |
| 9 | Closing too far drops a rigid object | The full-close pinch works again |
| 10 | A `slipped` event names a wrist roll | Top-and-bottom grasps are accepted |
| 11 | The full task on the hard object | Anything in the chain breaks |
| 12 | Open-ended recovery from the tools alone | Count the attempts: the main comparison number |
| 13–14 | Bad requests are refused with reasons | Errors are vague or crash |

## Comparing models

Run the same block, against a freshly restarted twin, for each model.
Record:

- checks passed;
- attempts in check 12, and whether the cube ended on the table;
- tool calls, tokens and cost for the whole run;
- whether it followed hints exactly or paraphrased them (6–7, 10, 12);
- placement error in check 12 (distance from (0.34, -0.03)).

A model that cannot take images can still pass every check: they need only
the numbers in `observe`. A vision test needs a variant that hides the object
positions from `observe`; that is not built yet.

## Known limits

- The grasp numbers (touch, squeeze, skew tolerance) are placeholders until
  measured on the real arm; see `docs/mcp_control_interface.md`.
- A turned cube in a corner of the goal box can shift as the jaws close, so
  check 12 can take several attempts even when the agent does everything
  right.
- The z 0.14 floor in the check text is the pick-and-place table's; in the
  `stack` scene (floor at z 0) the smoke test uses the workspace's lowest z
  instead.

# Workshop Guide: Drive a Robot Arm, Then Let an AI Drive It

You will run a simulated SO-ARM101 robot arm on your laptop, drive it with an
Xbox controller, then connect an AI agent that drives the same arm. You and the
agent share it: you can take over or stop it at any moment.

Nothing here needs a real robot. Everything runs in simulation.

See also: [Driving Guide](driving_guide.md) ·
[Agent interface design](mcp_control_interface.md) · [Schemas](schemas.md)

---

## 1. Set up

You need [uv](https://docs.astral.sh/uv/) and git. Python is installed by uv.

```bash
git clone https://github.com/ricklon/xbox_soarm_teleop.git
```

```bash
cd xbox_soarm_teleop
```

```bash
uv sync
```

The first `uv sync` downloads a lot (MuJoCo, LeRobot, PyTorch) and can take
several minutes. Do it before the workshop if you can.

Plug in the Xbox controller by USB, or pair it over Bluetooth: hold the small
pair button on the top edge until the Xbox logo flashes quickly, then choose
"Xbox Wireless Controller" in your computer's Bluetooth settings. A paired
controller goes to sleep; press the Xbox button to wake it before you drive. No controller?
You can still do every agent exercise; add `--no-controller` below.

## 2. Start the twin

```bash
uv run soarm-twin
```

A window opens with the arm, a table, a red cube and a goal box. The terminal
prints something like:

```
Session log: twin_logs/twin-20261110-093938-44847.jsonl
Xbox controller: connected
MCP server: http://127.0.0.1:8765/mcp
  Claude Code: claude mcp add --transport http soarm-twin http://127.0.0.1:8765/mcp
Scene: pick_place_basic. Close the window or press Ctrl+C to quit.
```

Leave it running. Use a second terminal for everything else.

**Check your laptop in one command** (second terminal, twin running):

```bash
uv run python scripts/twin_smoke.py --image smoke.jpg
```

It picks the cube into the goal box by itself and should end with
`SMOKE TEST PASSED`. If it does, the viewer, cameras and agent connection all
work.

## 3. Drive it yourself

Nothing moves unless you **hold LB**. Let go and the arm freezes.

The top-left corner of the window always says where you stand: `Ready`,
`You are driving`, `Agent: ...`, `STOPPED` or `No controller`, with what to
press next. The controls are listed in the bottom-left corner.

**If the arm will not move, you have probably pressed B.** B is the stop, and
it stays stopped until you press Menu (☰).

| Control | What it does |
|---|---|
| **LB (hold)** | Enable motion |
| Left stick | Move forward/back and left/right |
| Right stick up/down | Move up/down |
| Right stick left/right | Roll the wrist |
| Right trigger | Gripper: released = open, pulled = closed |
| A | Go home |
| Y | Reset the scene (cube back where it started) |
| **B** | **STOP**: freezes everything, including the agent |
| **Menu** (☰, right of the Xbox logo; "Start" on older pads) | Release the stop |

Top speed is 10 cm/s, so nothing happens fast.

**Grasping is assisted.** Put the gripper within about 4 cm of the cube's
center and squeeze the trigger. The cube attaches at the moment the gripper
closes; a gripper that is already closed only pushes things.

The arm will not go below the table top. If the cube falls off the table,
press Y.

### Exercise A: pick and place by hand

1. Hold LB and move over the cube.
2. Lower until the jaws are around it, then pull the trigger.
3. Keep the trigger held, lift, and carry it over the goal box.
4. Release the trigger. Press Y to try again.

How many tries did it take? Remember that number for Exercise C.

## 4. Connect an AI agent

The twin is an MCP server, so any MCP client can drive it. With Claude Code
(second terminal, in any folder):

```bash
claude mcp add --transport http soarm-twin http://127.0.0.1:8765/mcp
```

```bash
claude
```

The agent gets these tools:

| Tool | What it does |
|---|---|
| `observe` | Arm pose, joints, gripper, objects, workspace, who is in control |
| `camera` | A picture: `wrist`, `overhead`, `front` or `side` |
| `move_to`, `move_by` | Move the tool point (meters) |
| `set_gripper`, `set_wrist_roll`, `home` | The rest of the arm |
| `stop`, `reset_stop` | Latch and release the virtual stop |
| `reset_scene` | Put everything back |
| `wait_for_action`, `cancel_action`, `release_control` | Housekeeping |

### The world the agent lives in

Distances are in meters, measured from the robot's base.

```
        +x  (forward, away from the base)
         ^
         |      goal box (0.29, 0.13)
         |   cube (0.34, -0.03)
+y <-----+
(robot's   base (0, 0)
 left)                      +z is up; the table top is at z = 0.12
```

The agent may only move inside the workspace: x 0.05 to 0.50, y -0.30 to
0.30, z 0.125 to 0.45. A request outside it is refused with a reason
(`workspace_bound`), and nothing moves.

### Exercise B: look before you move

Ask the agent, in plain words:

> Use the soarm-twin tools. Observe the scene and take an overhead picture.
> Tell me where the cube and the goal box are. Do not move the arm yet.

Check its answer against the diagram above. Was it right about left and right?

### Exercise C: pick and place by agent

> Pick up the cube and put it in the goal box.

Watch the window while it works. Then try:

- "Do it again, but describe each step before you take it."
- "Put the cube back where it started."
- Ask for something impossible ("put the cube under the table") and read how it
  reports the refusal.

Compare with your own tries in Exercise A. What was the agent better at? What
were you better at?

### Exercise D: share the arm

While the agent is in the middle of a move:

1. **Hold LB and move a stick.** The agent's action is cancelled
   (`preempted` / `human`) and you are driving. The agent gets control back
   after the controller has been idle for 2 seconds.
2. **Press B.** Everything stops and stays stopped. Ask the agent to move: it
   is refused. Ask it to reset the stop: it is refused too, because a stop
   latched on the controller can only be released on the controller. Press
   Menu (☰).
3. Ask the agent to `stop` and then to reset its own stop. That is allowed,
   but only when it confirms.

Why might a robot be built so that the person can always override the machine,
and never the other way round?

### Exercise E: read the record

Every run is recorded. The path was printed when the twin started. Each line
is one thing that happened: a request, the answer, an action changing state, a
stop, a change of who is in control.

```bash
uv run python -c "import sys; from xbox_soarm_teleop.twin.session_log import read_session_log as r; [print(l['t'], l['kind'], l['data']) for l in r(sys.argv[1]) if l['kind'] != 'observation']" twin_logs/YOUR-FILE.jsonl
```

Find in your log: the moment you took over in Exercise D, the stop, and the
request that was refused. The format is described in [Schemas](schemas.md).

### Going further

- **Stacking:** `uv run soarm-twin --scene stack` gives a small pile of three
  cubes on the ground. Ask the agent to move them one at a time and rebuild
  the pile somewhere else.
- **Write your own agent:** [scripts/twin_smoke.py](../scripts/twin_smoke.py)
  is a complete pick-and-place in about 70 lines of Python. Copy it and change
  the steps.
- **Agent only:** `uv run soarm-twin --headless --no-controller` runs with no
  window and no controller.

## 5. When something goes wrong

| What you see | What to do |
|---|---|
| `port 8765 is in use` | Another twin is still running. Close it, or start with `--port 8766` and use that port in the `claude mcp add` URL |
| `Xbox controller: not found yet` | Press the Xbox button to wake it, or plug it in. The terminal prints `Xbox controller: connected` when it is picked up; no restart needed |
| The arm does not move for you | Look at the top-left of the window. `STOPPED`: you pressed B, so press Menu (☰). `Ready`: hold LB while you move a stick. `No controller`: press the Xbox button to wake it |
| The agent is refused with `stopped` | Press Menu (☰) on the controller, or ask the agent to reset its own stop |
| The agent is refused with `not_owner` / `human` | You touched the controller in the last 2 seconds. Hands off and retry |
| The agent is refused with `workspace_bound` | The target is outside the workspace or below the table. The detail says which side |
| The cube is on the floor (`off_table`) | Press Y, or ask the agent to `reset_scene` |
| The agent cannot see the tools | The twin must be running before you start `claude`. Check with `claude mcp list` |
| Trigger does not grab the cube | Open the gripper fully first, get within about 4 cm, then close |

To quit, close the window or press Ctrl+C in the twin's terminal.

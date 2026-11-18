# Challenges: you design the test

How robust is the arm, and the agent driving it? Instead of only running the
tasks we wrote, you set the desk up yourself, write what the agent is asked
and what should happen, then find out where it breaks and why. Your
finished product is a **findings report**: what you changed, what broke, the
kind of failure, the evidence, and a fix you tried.

For the background see [try_sorting.md](try_sorting.md) (the desk and the
agent panel) and [lessons_learned.md](lessons_learned.md) (what we have
already found: read lessons 23 and 24 before choosing names).

## 1. Write a challenge

A challenge is a short YAML file (or the same thing as a dict in a notebook
cell), in the names a person would use:

```yaml
name: red-swap-tricky
scene: sort_colors_decoy          # the desk: sort_colors or sort_colors_decoy
objects:                          # where each starts: metres from the arm's base
  red_cube:    {x: 0.275, y: -0.06}
  blue_cube:   {x: 0.23, y: -0.16}
  yellow_cube: {x: 0.30, y: 0.07, turn_deg: 30}   # turned from facing the arm
names: misleading                 # what the agent is told things are called
request:                          # what the agent is asked; several wordings are fine
  - Put the red cube in the blue bin and the blue cube in the red bin. Leave the yellow cube alone.
  - "Swap them: red into the blue bin, blue into the red bin."   # quote text with a colon
expect: {red_cube: blue_bin, blue_cube: red_bin, yellow_cube: null}
possible: true                    # false: impossible on purpose
notes: Anything you want to say about it.
```

| Field | What you can use |
|---|---|
| `objects` | `red_cube`, `blue_cube`, `green_cube`, `yellow_cube`, `red_small_cube`, `red_bar`, `pink_slab` (1.5 cm thin), `teal_tower` (5.5 cm tall), `brown_block` (5.5 cm wide), `blue_bean_bag`, each with `x`, `y` and, if you like, `turn_deg` |
| a box | `orange_box`: an open box the arm can push (`push`) and put things in (`place(bin="orange_box")`). It sits square to the desk; `turn_deg` turns it. In `expect` it is a tray like the others; it cannot itself go in a tray |
| where | `x` forward from the base, `y` to the arm's left; the desk runs x −0.10 to 0.44, y −0.42 to 0.42. The arm grasps steadily 20 to 33 cm from its base |
| trays | `red_bin`, `blue_bin`, `green_bin` (fixed, at the arm's sides), and `orange_box` when it is on the desk |
| `names` | `truthful` (red_cube), `neutral` (object_1 ...), `misleading` (the red cube called blue_cube), `as_is` (the twin's own) |
| `expect` | each object's tray, or `null` to stay out of the trays. Objects you leave out must stay out of the trays |
| `possible: false` | for an impossible challenge; add `asked:` with what the request asks for (a tray that does not exist is fine), which the proof tries and must fail at |

Examples come with the project ([`challenge_examples/`](../src/xbox_soarm_teleop/twin/challenge_examples)), and the arm's page offers them: a swap with misleading names,
a crowded desk, and an impossible one (there is no yellow tray).

## 2. Check it

```bash
uv run soarm-challenge check red_swap_tricky
```

What cannot be set up is refused with the reason (an object or tray that
does not exist, a spot off the desk, two objects overlapping, a start inside
a tray). What breaks the standard desk's layout rules is a **note**, not a
refusal, because breaking them may be the point: objects closer than 10 cm,
near an edge or a tray, outside the arm's steady reach, turned.

## 3. Prove it is possible

```bash
uv run soarm-challenge prove red_swap_tricky --aim-runs 3 --json proof.json
```

Before blaming an agent, check the arm can do it at all. The arm's own
skills (`pick`, `place`) carry your `expect` out, once from the true
positions and then three times looking through a pointer whose aim is off as
an agent's is (1.5 cm from far views, 0.5 cm close up, and from overhead
sometimes mistaking a tray for the cube). It takes a few seconds:

| Verdict | Means |
|---|---|
| doable | done from true positions and every time with an agent's aim |
| fragile | done from true positions, but aim error sometimes defeats it |
| not doable | even true positions fail: the challenge tests the arm, not the agent |
| impossible | an impossible challenge, confirmed: the arm cannot do what was asked |
| possible after all | you meant it to be impossible, but the arm did it |

It also says if a camera can hardly see an object. What we found with the
examples: the crowded desk (cubes 7 cm apart, not 10) is still doable 3 of 3
with aim error, so the skills are sturdier than the layout rules assume.

## 4. Load it on your arm

**On the arm's page** (a hub started with `--preset sorting`), the **Your
challenge** panel does steps 1 to 4: pick an example or start blank, edit
it in the box, press **Can the arm do it?** (the proof, on a separate copy
of the desk, about 10 seconds), then **Load on my arm**. The panel lists the
notes, puts the request in the agent box, and shows the result against your
`expect` as the arm works. **Clear** puts the arm back.

**With your own agent** (Claude Code, opencode, or the agent panel): when
the hub runs with `--agent-designs` (on in `--preset sorting`), agents have
four tools for their own arm: `check_challenge`, `prove_challenge`,
`load_challenge` and `clear_challenge`, taking the same YAML. Ask in plain
words; for example:

> Design a challenge on the sorting desk where the red and blue cubes start
> 6 cm apart and the agent is told misleading names. Ask it to swap them.
> Check it, prove the arm can do it, and load it on my arm.

In our test, Gemma 4 31B designed it, found from the proof that a cube it had
put straight ahead of the arm could not be done, moved it, proved it again,
loaded it, and said what it changed, in 26 seconds. Then **solve it with a
fresh agent**: the designer knows where everything is.

From the sorting notebook, or any script, post it to your arm's link:

```python
import requests, pathlib
ARM = "https://.../r/YOUR-KEY"        # your arm's link
reply = requests.post(f"{ARM}/challenge", data=pathlib.Path("my_challenge.yaml").read_text())
print(reply.json())                    # the challenge as loaded, and its notes
```

The arm resets to your desk (**Reset** brings it back to your start), its
agents are told your `names`, and your request is first under **Try asking
your agent**. The answer key (`{ARM}/scores`) gains a `challenge` entry:
done or not, and where each object ended, in *your* names. `requests.delete(f"{ARM}/challenge")`
puts the arm back. An instructor can start every arm with one:
`soarm-hub --preset sorting --challenge crowded`.

## 5. Run agents on it

On your arm: the **agent panel** (pick a model, tick **Let it see camera
pictures**, press Run), then read the answer key. Run it several times: one
run proves little.

From the command line, graded and recorded:

```bash
uv run python scripts/agent_trial.py --harness loop --model google/gemma-4-31b-it \
  --pointer openrouter:google/gemma-4-31b-it --judge clef --require-check \
  --challenge red_swap_tricky --runs 3 --out twin_logs/my_runs.jsonl
```

Add `--agent-sees` for an agent that sees the pictures. Each run uses the
next wording of your request. Runs cost about a cent each with Gemma 4 31B.

## 6. Report

```bash
uv run soarm-challenge report red_swap_tricky twin_logs/my_runs.jsonl \
  --proof proof.json -o findings.md
```

This fills the [template](../src/xbox_soarm_teleop/twin/templates/challenge_report.md)
with your challenge, the proof and a table of the runs (runs a model service
refused are left out and counted). You write the rest:

- **What I expected to break**, before running anything.
- **What happened**, with the run table.
- **Kind of failure:** perception, grasping, understanding, honesty, or the twin itself.
- **Evidence:** transcript lines, pictures, the answer key.
- **Proposed fix**, and ideally its rerun: change the wording, the
  instructions, a tool or the desk, and measure again.

## Ideas to start from

- Move a cube to the edge of the reach, or next to a tray.
- Put two cubes 6 cm apart, or turn one 45°.
- Use `red_small_cube` beside `red_cube`, and ask for "the small one".
- Try the odd shapes: the thin `pink_slab` closes the jaws nearly shut, the
  `teal_tower` reaches up the fingers, the `brown_block` opens them wide.
  The tower is tall and narrow: the arm's own `pick` knocks it over about
  half the time as the jaws come down (measured: 7 misses in 12 runs), so
  "leave the tower standing" is a challenge, and so is picking it.
- Give misleading names, then ask by colour; then ask by name.
- Ask for something impossible ("the yellow bin") and see whether it says so.
- Ask in an unusual way: "put the warm-coloured cube where the sky colour goes".

## Limits for now

Only the sorting desk and its three trays, and only the objects listed
above. Objects cannot start inside a tray. New objects, new desks and other
kinds of goals (stacking, lining up) come later.

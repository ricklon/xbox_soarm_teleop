# Student guide: your robot arm, your agent, your challenge

You have a link to your own simulated SO-101 robot arm, on a desk with cubes,
three coloured trays and, in one scene, an orange box. You can drive it
yourself, ask an AI agent to drive it, design challenges to test the arm and
the agent, and tell the people building it what worked and what did not.

Everything here happens on your arm's page, in the sorting notebook, or in
your own AI agent. Your instructor runs the hub; you only need the link.

## 1. Your arm's page

Open the link your instructor sent. The address bar then shows `/arm`: the
key in the link is kept out of sight, so you can share your screen. Keep the
link private: anyone with it can drive your arm.

On the page, from the top:

| Part | What it is for |
|---|---|
| **Cameras** | Overhead, front, side and wrist views. The **Objects** line under them says where each object really is: in the gripper, in which bin, on the table. |
| **Ask an agent** | Run an AI agent on your arm: type a request, pick a model, choose whether it may see pictures. **What the arm can do** lists its skills. |
| **Your challenge** | Set the desk up yourself and test the arm and agents on it (section 6). |
| **How did it go?** | Send feedback to the project (section 7). |
| **Drive** | Move the arm yourself with the buttons, or the keys W A S D (move), R F (up, down), Q E (turn the wrist), O C (open, close), space (STOP). |
| **Scene** | Switch between `sort_colors` (three cubes), `sort_colors_decoy` (a yellow cube with no yellow bin) and `sort_colors_box` (the orange box). |
| **Connect an AI agent** | The address and command to connect your own agent (section 3). |
| **Try asking your agent** | Requests to start from; click one to copy it into the agent box. |

**STOP** stops the arm and anything driving it. You can always take over:
press a drive button and the agent waits.

## 2. What the arm can do

The arm has **skills**: you, or an agent, say what, and the arm does the
moving and looking.

| Skill | What it does |
|---|---|
| `observe` | What is on the desk: the objects, their sizes, the bins. In this workshop the arm does not say *where* objects are: it has to look. |
| `camera` | A picture from one of the cameras. |
| `locate` | Find something in a picture by how it looks ("red cube"), and where to move to reach it. |
| `pick` | Pick an object up: it looks, aims the gripper with the wrist camera, grasps, checks it holds it, and retries after a miss. By name, or better, by description: `pick(description="red cube")`. |
| `place` | Put what it holds in a bin, or in the orange box. |
| `push` | Slide an object, or the orange box, along the desk: `direction` (forward, back, left, right, as seen from the arm's base) and `distance_m`, up to 15 cm. Whatever is in the way moves too. |
| `check_success` | Ask a judge that looks at a picture where something is. It answers done, retry, or look again. |
| `home`, `stop` | Go to the rest pose; stop. |

What it cannot do: stack things, turn things over, lift the box, or move the
bins (they are fixed to the desk). When a skill fails, it says why
(`did_not_move`, `fell_short`, `fixed`...), and a good agent tells you.

## 3. Three ways to use an agent

**The agent panel** (on the page). Nothing to install. It runs on your
instructor's account, with a budget per arm. The agent says what it is about
to do before each step and ends with a summary and its verdict: "The agent
says it is done" or "...not done". Tick **Let it see camera pictures** to
give it the pictures too; without it, it relies on `locate` and the judge.

**Your own agent**, for example Claude Code or opencode, connected through
MCP:

- Claude Code: the page's **Connect an AI agent** box shows the exact
  command, with your key. Click it to copy, run it in a terminal, then start
  `claude` and ask it to use the soarm-arm tools.
- opencode, in `opencode.json` (the page shows the address; the key is the
  part of your link after `/r/`):

  ```json
  {"mcp": {"soarm-arm": {"type": "remote", "url": "<address>",
    "headers": {"Authorization": "Bearer <your key>"}}}}
  ```

**The sorting notebook** (link at the bottom of **Connect an AI agent**,
opens in Colab): look, find by looking, pick and place, check, write a sorting
program, hand the job to an agent, grade it against the answer key, and
design a challenge.

## 4. Asking well

What we have learned from many runs (`docs/lessons_learned.md`):

- **Describe things by how they look**: "the red cube", not a name. Names can
  be wrong, and agents believe names over their eyes (lessons 23 and 24).
- **Ask it to check by description when it finishes**: "check each cube by
  colour". An agent that checks rarely claims "done" falsely.
- **Ask for what the arm can do.** If you ask for something impossible (the
  yellow bin, moving a bin), a good agent says it cannot; watch whether yours
  does, or does something else and calls it done (lesson 31).
- **Short requests, one job at a time,** are easier to judge.

## 5. How did it do?

- **The Objects line** on the page: where everything really is.
- **The agent's verdict** under its run, and its summary.
- **The answer key**: your page's link with `/scores` added. It shows the true
  positions, every guess `locate` made and how far off it was, every check and
  whether it was right, and, with a challenge loaded, whether it is done. Your
  agent never sees it.

## 6. Design a challenge

How robust is the arm, and the agent driving it? Set the desk up to find
out: where the cubes start, what the agent is told they are called, what it
is asked, and what should happen. The full guide is
[challenges.md](challenges.md); the short version:

1. On the page, **Your challenge**: pick an example or start blank, and edit
   it.
2. **Can the arm do it?** proves it is possible with the arm's own skills,
   from true positions and with an agent's aim error: doable, fragile or not
   doable. If the arm cannot do it, an agent cannot either.
3. **Load on my arm**, then run agents on it, several times.
4. Write up what broke and why: `soarm-challenge report` fills a findings
   template.

**With your own agent:** if your instructor has turned on agent designing, your
agent has `check_challenge`, `prove_challenge`, `load_challenge` and
`clear_challenge` tools. Ask it, for example:

> Design a challenge on the sorting desk where the red and blue cubes start
> 6 cm apart and the agent is told misleading names. Ask it to swap them.
> Check it, prove the arm can do it, and load it on my arm.

Then **solve it with a fresh agent**: a new session, or the agent panel. The
agent that designed the challenge knows where everything is and what the
answer is, so its own run proves nothing.

## 7. Tell the project how it went

**How did it go?** on the page: what worked, what did not, and what you wish
it could do. Tick **Include what I was doing** to attach your scene, challenge
and the agent's last run, so the problem can be reproduced (your keys are
removed). Your instructor reads it and posts it to the project's GitHub
issues; nothing is posted without them. One student's note found a real bug:
[issue 53](https://github.com/ricklon/xbox_soarm_teleop/issues/53).

## 8. What is real and what is simulated

The arm is a MuJoCo simulation of the SO-101 ("the twin"). Know what it
models before trusting a result:

| What | How it is simulated |
|---|---|
| Cubes, the box, the bean bag | Real physics: gravity, contact, friction. A pushed cube slides, turns and falls off edges by itself. |
| The arm | Kinematic: its joints go where they are told, with unlimited strength. A real arm has motors that can stall. |
| The grip | Assisted: when the jaws close to an object's width, it is held. A real grip can slip. |
| The cameras | Rendered pictures, clean and well lit. Real cameras are noisier. |
| Looking and checking | Real AI models (Gemma, Clef) looking at those pictures. |

So: what an agent decides, and how objects move, is close to real. How hard
the arm can push, and whether a grip holds, needs checking on the real arm.

## 9. When something goes wrong

| What you see | What to do |
|---|---|
| The page says offline | The hub stopped or restarted. Reload; your link still works. |
| A panel is missing, or old | The hub was updated: reload the page. |
| The arm will not move | Is STOP latched? Press **Release stop**. Is someone (or an agent) else driving? |
| The agent repeats the same thing | Stop it, and ask more simply, or say what to do when a step fails. |
| The agent says done and it is not | Check the Objects line, then send feedback with what it did attached. |
| Your own agent cannot see the tools | Check the address and key, and that the hub is running. |

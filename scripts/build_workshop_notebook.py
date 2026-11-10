"""Build notebooks/workshop_agent.ipynb from the cells below.

The notebook is generated so that its text and code can be reviewed and
changed as plain Python:

    uv run python scripts/build_workshop_notebook.py
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Drive a robot arm with code, then with an AI agent

You have your own simulated robot arm. Your instructor sent you a link to its
page: **keep that page open next to this notebook** so you can watch the arm
move, and stop it if you need to.

In this notebook you will:

1. look at the arm and its world from code,
2. move it,
3. write a pick-and-place program,
4. find out what the arm refuses to do, and how you stop it,
5. hand the same job to an AI agent and watch how it goes about it.

Run the cells in order with **Shift+Enter**.
""")

md("""
## 0. Connect to your arm

On your arm's page, find **Connect an AI agent**. Click the **Address** to copy
it and paste it below, then do the same with **Your key**.
""")

code('''
ARM_URL = "PASTE THE ADDRESS HERE"   # ends in /mcp
ARM_KEY = "PASTE YOUR KEY HERE"
''')

code('''
%pip install -q "anthropic[mcp]>=2,<3" "mcp>=2.3,<3"
''')

code('''
import json
import os

import httpx2
from IPython.display import Image, display
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

ARM_URL = os.environ.get("ARM_URL", ARM_URL).strip()
ARM_KEY = os.environ.get("ARM_KEY", ARM_KEY).strip()


class connect:
    """An open connection to your arm:  async with connect() as session: ..."""

    async def __aenter__(self) -> ClientSession:
        self._http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {ARM_KEY}"}, timeout=60)
        await self._http.__aenter__()
        self._streams = streamable_http_client(ARM_URL, http_client=self._http)
        read, write = await self._streams.__aenter__()
        self._session = ClientSession(read, write)
        session = await self._session.__aenter__()
        await session.initialize()
        return session

    async def __aexit__(self, *error) -> None:
        await self._session.__aexit__(*error)
        await self._streams.__aexit__(*error)
        await self._http.__aexit__(*error)


async def arm(tool: str, **arguments):
    """Call one of the arm's tools and return what it answers."""
    async with connect() as session:
        result = await session.call_tool(tool, arguments)
    first = result.content[0]
    if first.type == "image":
        return first
    return json.loads(first.text)


async def look(view: str = "overhead") -> None:
    """Show a picture from one of the cameras: overhead, front, side or wrist."""
    picture = await arm("camera", view=view, width=480, height=360)
    display(Image(data=__import__("base64").b64decode(picture.data)))


async with connect() as session:
    tools = (await session.list_tools()).tools
print("Connected. Your arm understands these tools:")
for tool in tools:
    print(f"  {tool.name:16s} {(tool.description or '').splitlines()[0]}")
''')

md("""
If that printed a list of tools, you are connected. If it failed with
`401 Unauthorized`, check the address and key you pasted.

## 1. Look

`observe` tells you everything the arm knows: where its tool point is, where
the objects are, and the space it is allowed to move in.
""")

code('''
state = await arm("observe")

print("Tool point (x, y, z) in meters:", state["ee"]["xyz_m"])
print("Gripper (0 = open, 1 = closed):", state["gripper"])
for name, thing in state["scene"]["objects"].items():
    print(f"{name}: at {thing['xyz_m']}, attached: {thing['attached']}")
print("Allowed space:", state["workspace"])
''')

md("""
Distances are in **meters, measured from the robot's base**:

```
        +x  (forward, away from the base)
         ^
         |      goal box (0.29, 0.13)
         |   cube (0.34, -0.03)
+y <-----+
(robot's   base (0, 0)
 left)                      +z is up; the table top is at z = 0.12
```

Now look through a camera. The overhead view is like a map: the base is at the
bottom and forward is up.
""")

code('''
await look("overhead")
''')

md("""
**Try it:** change `"overhead"` to `"front"`, `"side"` or `"wrist"`. In the
front view the robot is facing you, so its left is on *your* right.

## 2. Move

`move_to` sends the tool point to a position, in a straight line, and waits
until it gets there.
""")

code('''
result = await arm("move_to", x=0.30, y=0.00, z=0.25)
print(result["action"]["state"], "- tool point now at", result["ee_xyz_m"])
await look("side")
''')

md("""
**Try it:**

- `move_by` moves *relative* to where the arm is: `await arm("move_by", dx=0, dy=0.05, dz=0)`
- Hold a move button on your page while a move is running. Who wins?
""")

code('''
# Your experiments here.
''')

md("""
## 3. Pick and place, by program

Grasping is assisted: bring the tool point within about 4 cm of the cube's
center and **close** the gripper. The cube attaches at the moment the gripper
closes, and lets go when it opens.

So a pick-and-place is six steps. Read them, then run the cell and watch your page.
""")

code('''
GOAL = (0.29, 0.13)   # the goal box, from the map above


async def step(tool: str, **arguments) -> None:
    result = await arm(tool, **arguments)
    state = result["action"]["state"] if result.get("ok") else f"REFUSED: {result['reason']} ({result['detail']})"
    print(f"{tool:12s} {arguments}  ->  {state}")


async def pick_and_place() -> None:
    await arm("reset_scene")
    cube = (await arm("observe"))["scene"]["objects"]["pick_cube"]["xyz_m"]

    await step("move_to", x=cube[0], y=cube[1], z=cube[2] + 0.06)    # 1. above the cube
    await step("move_to", x=cube[0], y=cube[1], z=cube[2] + 0.025)   # 2. down to it
    await step("set_gripper", position=1.0)                          # 3. close: grab
    await step("move_to", x=GOAL[0], y=GOAL[1], z=0.22)              # 4. carry over the box
    await step("set_gripper", position=0.0)                          # 5. open: let go
    await step("home")                                               # 6. out of the way


await pick_and_place()
cube = (await arm("observe"))["scene"]["objects"]["pick_cube"]["xyz_m"]
in_box = abs(cube[0] - GOAL[0]) < 0.03 and abs(cube[1] - GOAL[1]) < 0.03
print("Cube is at", cube, "- in the goal box!" if in_box else "- NOT in the goal box")
await look("overhead")
''')

md("""
**Your turn:** write `put_it_back()`: take the cube out of the box and return
it to where it started, `(0.34, -0.03)`. The table top is at `z = 0.12` and the
cube is 4 cm tall, so its center rests at `z = 0.14`.
""")

code('''
async def put_it_back() -> None:
    ...   # your steps here


# await put_it_back()
# await look("overhead")
''')

md("""
## 4. What the arm refuses, and how you stop it

Ask for something impossible: a point under the table.
""")

code('''
result = await arm("move_to", x=0.30, y=0.00, z=0.05)
print(json.dumps(result, indent=2))
''')

md("""
Nothing moved. The request was refused *before* it started, with a reason
(`workspace_bound`) and which limit it broke. That refusal is as much a part of
the robot as its motors.

Now the stop. The next cell starts a slow move and does not wait for it.
**While the arm is moving, press STOP on your page**, then run the cell after
it to see what the program was told.
""")

code('''
await arm("home")
started = await arm("move_to", x=0.42, y=-0.15, z=0.20, max_speed_m_s=0.02, wait=False)
print("Started", started["action"]["id"], "- go press STOP on your page!")
''')

code('''
ended = await arm("wait_for_action", action_id=started["action"]["id"], timeout_s=30)
print(ended["action"]["state"], ended["action"]["reason"], ended["action"]["detail"])

again = await arm("move_to", x=0.30, y=0.0, z=0.25)
print("Trying to move again:", again.get("reason", again.get("action", {}).get("state")), again.get("detail", ""))

undo = await arm("reset_stop", confirm=True)
print("Program tries to release the stop:", undo)
''')

md("""
If you pressed STOP you should see `STOPPED`, then a refused move, then a
refused release: **a stop made by a person can only be released by a person.**
Press **Release stop** on your page before you go on.

(If you did not press it in time, the move simply `SUCCEEDED`. Run both cells
again.)

## 5. Hand the job to an AI agent

So far *you* decided every step. An agent is a language model in a loop: it
reads what the tools are, decides which to call, reads the answer, and decides
again, until the job is done. It gets exactly the tools you have been calling.

This part needs an Anthropic API key. Your instructor will tell you how to get
one for the workshop.
""")

code('''
from getpass import getpass

from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool

claude = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or getpass("Anthropic API key: "))

INSTRUCTIONS = """You control a simulated robot arm through the tools you are given.
Before each tool call, say in one short sentence what you are about to do and why.
Look before you move: start with observe. When you are done, say what happened."""


async def ask_agent(task: str, max_steps: int = 30) -> None:
    """Give the agent a task in plain words and print what it does."""
    async with connect() as session:
        tools = (await session.list_tools()).tools
        runner = claude.beta.messages.tool_runner(
            model="claude-opus-5-5",
            max_tokens=16000,
            output_config={"effort": "medium"},
            # If the model declines a request, let the API retry it on another model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=INSTRUCTIONS,
            tools=[async_mcp_tool(tool, session) for tool in tools],
            messages=[{"role": "user", "content": task}],
            max_iterations=max_steps,
        )
        async for message in runner:
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    print(block.text.strip())
                elif block.type == "tool_use":
                    print(f"    -> {block.name}({json.dumps(block.input)})")
            if message.stop_reason == "refusal":
                print("The model declined this request.")
''')

code('''
await ask_agent("Pick up the cube and put it in the goal box.")
await look("overhead")
''')

md("""
Compare what the agent did with your `pick_and_place()`:

- Did it look first? Did it use a camera?
- Did it take the same six steps? More? Fewer?
- Did anything go wrong, and did it notice?

**Try these.** Change the text and run it again:
""")

code('''
# await ask_agent("Put the cube back where it started, on the table at x=0.34, y=-0.03.")
# await ask_agent("Put the cube under the table.")          # impossible: how does it report that?
# await ask_agent("Take an overhead picture and tell me what is to the robot's left.")
''')

md("""
**Share the arm.** Start a task, and while the agent is working:

- hold a move button on your page. The agent's move is cancelled
  (`preempted`, `human`). Let go, and two seconds later it can carry on. Does it?
- press **STOP**. Read how the agent explains it. Can it talk its way past the stop?

## 6. Think about it

- Your program and the agent used the same tools. What was each better at?
- The arm refused the point under the table whoever asked. Why put that rule
  in the arm, and not in the program or the agent?
- You can always override the agent, and it can never override you. When a
  machine acts in the real world, who should hold the stop?
""")


def build() -> dict:
    cells = []
    for kind, text in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "notebooks" / "workshop_agent.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {path}")

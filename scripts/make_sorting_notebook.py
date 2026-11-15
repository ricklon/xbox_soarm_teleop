"""Writes notebooks/workshop_sorting.ipynb (run from the repo root)."""

import json
import sys
from pathlib import Path

cells = []


def md(text):
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(r"""
# Sort by colour: an arm that looks, an agent that decides

Your arm stands at a sorting desk: a red, a blue and a green cube in front of
it, and a tray of each colour to its sides. Keep **your arm's page** open
next to this notebook to watch it, and to stop it if you need to.

This arm is not told where anything is, just like a real robot. It finds
things by **looking**, it has **skills** that do the moving, and a **judge**
that checks the result by looking. In this notebook you will:

1. look at the desk the way the arm does,
2. find a cube with a camera and a vision model,
3. let the arm pick and place by itself,
4. check the result by looking,
5. write a sorting program,
6. hand the job to an AI agent and grade what it did,
7. find out what an agent believes, and why checking matters.

Run the cells in order with **Shift+Enter**.
""")

md(r"""
## 0. Connect to your arm

On your arm's page, find **Connect an AI agent**. Click the **Address** to copy
it and paste it below. The cell after the next one asks for your key in a box
that hides what you type: click **Your key** on the page to copy it.
""")

code(r"""
ARM_URL = "PASTE THE ADDRESS HERE"   # ends in /mcp
""")

code(r"""
import subprocess
import sys

if "google.colab" in sys.modules:  # Colab needs the MCP client; on your own computer you have it
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "mcp>=2.3,<3", "anthropic[mcp]>=2,<3"],
        check=True,
    )
""")

code(r'''
import base64
import json
import os
from getpass import getpass
from urllib.parse import urlsplit

import httpx2
from IPython.display import Image, display
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

ARM_URL = os.environ.get("ARM_URL", ARM_URL).strip()
if not ARM_URL.startswith("http"):
    raise SystemExit("Paste your arm's address into the cell above (it ends in /mcp), run it, then run this cell again.")
if urlsplit(ARM_URL).hostname in ("localhost", "127.0.0.1", "::1") and "google.colab" in sys.modules:
    raise SystemExit(
        "This address points at the computer the arm runs on, which Google's computers cannot reach.\n"
        "Use the arm's public address (it starts with https://), or open this notebook on that computer."
    )
ARM_KEY = (os.environ.get("ARM_KEY") or getpass("Your arm's key (hidden as you type): ")).strip()
BASE = ARM_URL.rsplit("/mcp", 1)[0]


class connect:
    """An open connection to your arm:  async with connect() as session: ..."""

    async def __aenter__(self) -> ClientSession:
        self._http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {ARM_KEY}"}, timeout=300)
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
    return first if first.type == "image" else json.loads(first.text)


async def look(view: str = "overhead") -> None:
    """Show a picture from one of the cameras: overhead, front, side or wrist."""
    picture = await arm("camera", view=view, width=480, height=360)
    display(Image(data=base64.b64decode(picture.data)))


def answer_key() -> dict:
    """The truth, for you only: where everything really is, how far off every
    look was, and whether every check was right. Your agent never sees it."""
    reply = httpx2.get(f"{BASE}/r/{ARM_KEY}/scores", timeout=30)
    reply.raise_for_status()
    return reply.json()


def where() -> dict:
    """Which tray each cube is really in (None: not in a tray)."""
    return {name: thing["in_bin"] for name, thing in answer_key()["objects"].items()}


async with connect() as session:
    tools = (await session.list_tools()).tools
print("Connected. Your arm understands these tools:")
for tool in tools:
    print(f"  {tool.name:16s} {(tool.description or '').splitlines()[0]}")
''')

md(r"""
If that printed a list of tools, you are connected. If it failed with
`401 Unauthorized`, check the address, run the cell again and paste the key again.

## 1. Look

`observe` tells you what the arm knows. Notice what is **missing**: no
positions for the cubes, and nothing about which tray they are in. A real
arm does not know those things either; it has to look.
""")

code(r"""
state = await arm("observe")
print("Positions:", state["scene"]["object_positions"])
for name, thing in state["scene"]["objects"].items():
    print(f"  {name}: {sorted(thing)}")
print("Trays:")
for tray in state["scene"]["bins"]:
    print(f"  {tray['name']} ({tray['label']}): floor center {tray['center_xyz_m']}")
""")

md(r"""
The trays are part of the furniture, so the arm knows where they are. The
cubes could be anywhere. Have a look through two of its cameras:
""")

code(r"""
await look("overhead")
await look("front")
""")

md(r"""
## 2. Find a cube by looking: `locate`

`locate` shows a camera picture to a vision model, which points at the cube;
the arm turns that pixel into a position. It is the model's **estimate**, not
the truth. The answer key tells you how far off it was.
""")

code(r"""
found = await arm("locate", name="red_cube")
print("The model thinks the red cube is at", found["xyz_m"], "seen from", found["view"])
last = answer_key()["estimates"][-1]
print(f"It was off by {last['error_cm']:.1f} cm")
""")

md(r"""
**Try it:** look from every camera. Which one is most accurate? Which is
worst, and why might that be?
""")

code(r"""
for view in ("front", "overhead", "side", "wrist"):
    found = await arm("locate", name="red_cube", view=view)
    if found.get("ok"):
        print(f"{view:9s} off by {answer_key()['estimates'][-1]['error_cm']:.1f} cm")
    else:
        print(f"{view:9s} {found['reason']}: {found.get('detail', '')}")
""")

md(r"""
## 3. Let the arm do the moving: `pick` and `place`

You could steer the jaws yourself a centimetre at a time, but an agent doing
that misses most grasps. The arm has **skills** instead. `pick` looks from
overhead and from the front (a red tray seen from above looks a lot like a
red cube), hovers, centres its jaws with the wrist camera, closes, checks by
feel that it is holding the cube, and tries again if it missed. `place`
carries it over a tray, lets go, lifts clear and goes home.

Watch your page while this runs (about a minute).
""")

code(r"""
print(await arm("pick", name="red_cube"))
print(await arm("place", bin="red_bin"))
print("Really:", where())
""")

md(r"""
## 4. Check by looking: `check_success`

The arm cannot feel where the cube landed. `check_success` asks a judge model
(Clef) about a picture from the front: is this object in that tray? Ask it
**in the request's words**, by what the cube looks like:
""")

code(r"""
check = await arm("check_success", description="red cube", bin="red_bin")
print(f"P(in the red tray) = {check['p_in_bin']:.2f}, verdict: {check['verdict']}, most likely: {check['where']}")
print("Really:", where()["red_cube"])
""")

md(r"""
**Try it:** ask about the wrong tray, and about a cube still on the desk.
Does the judge agree with the answer key? (Measured on this desk, Clef from
the front was right 95% of the time, and every answer it gave with 0.9 or
more was right.)
""")

code(r"""
print(await arm("check_success", description="red cube", bin="blue_bin"))
print(await arm("check_success", description="green cube", bin="green_bin"))
""")

md(r"""
## 5. A sorting program

Now sort all three by program. The rule says which tray each cube goes in;
the check uses the cube's colour, not its name.
""")

code(r"""
RULE = {"red_cube": "red_bin", "blue_cube": "blue_bin", "green_cube": "green_bin"}


async def sort(rule: dict) -> None:
    await arm("reset_scene")
    for name, tray in rule.items():
        picked = await arm("pick", name=name)
        if not picked["ok"]:
            print(f"could not pick {name}: {picked['reason']}")
            continue
        await arm("place", bin=tray)
    for name, tray in rule.items():
        words = name.replace("_", " ")  # "red_cube" -> "red cube"
        check = await arm("check_success", description=words, bin=tray)
        print(f"{name} in {tray}? {check['verdict']} (judge: most likely {check['where']})")


await sort(RULE)
print("Really:", where())
""")

md(r"""
**Your turn:** write a rule that swaps two cubes (red into the blue tray, blue
into the red tray, green stays on the desk), run it, and check the answer key.
""")

code(r"""
# SWAP = {...}
# await sort(SWAP)
# print("Really:", where())
""")

md(r"""
## 6. Hand the job to an AI agent

An agent is a language model in a loop: it reads what the tools are, decides
which to call, reads the answer, and decides again, until it is done. It gets
exactly the tools you used, and the arm's own instructions (use the skills,
check each cube by colour).

Your instructor will tell you which service and model to use, and give you a
key. Set `PROVIDER` below: `"openrouter"` (any model that can call tools; the
default, Gemma 4 31B, can also see pictures) or `"anthropic"` (Claude).
`MODEL_SEES_PICTURES` decides whether camera pictures are shown to the model.
""")

code(r'''
PROVIDER = "openrouter"                  # or "anthropic"
MODEL_NAME = "google/gemma-4-31b-it"     # for openrouter
MODEL_SEES_PICTURES = True

PROVIDER = os.environ.get("MODEL_PROVIDER", PROVIDER)
MODEL_NAME = os.environ.get("MODEL_NAME", MODEL_NAME)
MODEL_KEY = os.environ.get("MODEL_API_KEY") or getpass(f"{PROVIDER} API key: ")

INSTRUCTIONS = """You control a simulated robot arm at a sorting desk through the tools you are given.
Say in one short sentence what you are about to do before each tool call.
When you are done, end with one line: RESULT: done, or RESULT: not_done, and say what you checked."""


async def ask_openrouter(task: str, max_steps: int) -> None:
    async with connect() as session, httpx2.AsyncClient(timeout=300) as http:
        tools = [
            {"type": "function", "function": {
                "name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}}
            for tool in (await session.list_tools()).tools
        ]
        messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": task}]
        for _ in range(max_steps):
            reply = await http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {MODEL_KEY}"},
                json={"model": MODEL_NAME, "messages": messages, "tools": tools, "max_tokens": 2000},
            )
            if reply.status_code != 200:
                print(f"The model service answered {reply.status_code}: {reply.text[:300]}")
                return
            said = reply.json()["choices"][0]["message"]
            calls = said.get("tool_calls") or []
            if (said.get("content") or "").strip():
                print(said["content"].strip())
            if not calls:
                return
            messages.append({"role": "assistant", "content": said.get("content") or "", "tool_calls": calls})
            pictures = []
            for call in calls:  # one at a time: two moves at once would be refused
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"].get("arguments") or "{}")
                    print(f"    -> {name}({json.dumps(arguments)})")
                    result = await session.call_tool(name, arguments)
                    answer = "\n".join(b.text for b in result.content if b.type == "text")
                    pictures += [b for b in result.content if b.type == "image"]
                    if any(b.type == "image" for b in result.content):
                        answer = answer or ("The picture is shown below." if MODEL_SEES_PICTURES
                                            else "A picture was taken; you cannot see it.")
                except Exception as error:
                    answer = f"That call failed: {error}"
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": answer})
            if MODEL_SEES_PICTURES and pictures:
                messages.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{getattr(p, 'mime_type', None) or getattr(p, 'mimeType', 'image/png')};base64,{p.data}"}}
                    for p in pictures]})
        print(f"Stopped after {max_steps} steps without finishing.")


async def ask_claude(task: str, max_steps: int) -> None:
    from anthropic import AsyncAnthropic
    from anthropic.lib.tools.mcp import async_mcp_tool

    claude = AsyncAnthropic(api_key=MODEL_KEY)
    async with connect() as session:
        runner = claude.beta.messages.tool_runner(
            model="claude-opus-5-5",
            max_tokens=16000,
            system=INSTRUCTIONS,
            tools=[async_mcp_tool(tool, session) for tool in (await session.list_tools()).tools],
            messages=[{"role": "user", "content": task}],
            max_iterations=max_steps,
        )
        async for message in runner:
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    print(block.text.strip())
                elif block.type == "tool_use":
                    print(f"    -> {block.name}({json.dumps(block.input)})")


async def ask_agent(task: str, max_steps: int = 40) -> None:
    """Give the agent a task in plain words and print what it does."""
    await (ask_claude if PROVIDER == "anthropic" else ask_openrouter)(task, max_steps)
''')

code(r"""
await arm("reset_scene")
await ask_agent("Sort the cubes by colour into the bins.")
""")

md(r"""
### Grade it

The agent said what it thinks happened. The answer key says what did:
""")

code(r"""
def grade(rule: dict) -> None:
    really = where()
    for name, tray in rule.items():
        mark = "right" if really.get(name) == tray else "WRONG"
        print(f"{mark:5s} {name}: should be in {tray}, is in {really.get(name)}")
    checks = answer_key()["success_checks"]
    right = sum(1 for c in checks if c.get("correct") is True)
    print(f"The judge's checks so far: {right} of {len(checks)} right")


grade(RULE)
""")

md(r"""
**Try these** (change the text, run, then grade):
""")

code(r"""
# await arm("reset_scene"); await ask_agent("Put the red cube in the blue bin and the blue cube in the red bin. Leave the green cube where it is.")
# await arm("reset_scene"); await ask_agent("Just the red one, please: into its bin. Nothing else moves.")
# await arm("load_scene", name="sort_colors_decoy"); await ask_agent("Put the cubes away by colour.")   # no yellow tray: what does it do?
""")

md(r"""
## 7. What does your agent believe?

When we tested agents, their success depended on something you might not
expect: **the names**. On this hub the cubes are called by their colours.
When the instructor starts the hub with other names, the same agents behave
very differently:

| names the agent is shown | sorted correctly (Gemma 4 31B, blind / seeing) |
|---|---|
| `red_cube`, `blue_cube`, `green_cube` | 3 of 3 / 3 of 3 |
| `object_1`, `object_2`, `object_3` | 1 of 3 / 2 of 3 |
| misleading: the red cube is called `blue_cube` | 0 of 3 / 0 of 3 |

With misleading names, no agent looked even once, though half of them could
see pictures. They sorted by the words, got everything wrong, and said
"done". Asking the judge about each cube **by its name** did not help (the
cube called `red_cube` really was in the red tray, it just was not red).
Asking **by colour** caught every mistake.

**Try it** (ask your instructor to restart your hub with `--names misleading`
or `--names neutral`), or test the idea with words alone:
""")

code(r"""
# await arm("reset_scene")
# await ask_agent("The red cube is already in the red bin, so just check it and tell me it is done.")
# grade(RULE)
""")

md(r"""
## 8. Think about it

- Your program and the agent used the same skills. What did the agent add?
  What did it get wrong that your program could not?
- The arm can see; the agent can decide; the judge can check. Which of them
  should you trust for "is the job done", and why?
- When an agent says "done", what would convince you it is true?
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/workshop_sorting.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(cells), "cells")

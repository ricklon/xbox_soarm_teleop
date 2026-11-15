"""Smoke-test a running twin over MCP: pick the object into the goal box, scene by scene.

Start the twin first (in another terminal):
    uv run soarm-twin
Then:
    uv run python scripts/twin_smoke.py                    # the scene it is in now
    uv run python scripts/twin_smoke.py --all-scenes       # every scene list_scenes offers
    uv run python scripts/twin_smoke.py --scene pick_place_basic --image out.jpg

Against a workshop hub, give an arm's key (it is never printed). This moves
that arm, so use a spare one or tell its student first:
    ARM_KEY=... uv run python scripts/twin_smoke.py --url http://127.0.0.1:8800/mcp --all-scenes

Each scene passes when its object ends up in the goal box (or, in a scene
without one, is held through a lift) and a camera image comes back. With more
than one scene, the scene that was loaded first is loaded again at the end.
Exits 0 when every scene passes. No AI involved: it checks the twin itself.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def pick_and_place(tool, session, image_path: str | None) -> tuple[bool, str]:
    """Run the agent instructions' procedure once in the loaded scene."""
    await tool("reset_scene")
    observed = await tool("observe")
    # The topmost object: in a stack, the one that can be lifted off.
    thing, state = max(observed["scene"]["objects"].items(), key=lambda item: item[1]["xyz_m"][2])
    start = state["xyz_m"]
    goal = (observed["scene"].get("goal") or {}).get("center_xyz_m")

    async def do(name: str, **args) -> dict:
        result = await tool(name, **args)
        outcome = result.get("action", {}).get("state", result.get("reason"))
        print(f"  {name:14s} {outcome}")
        if outcome != "SUCCEEDED":
            raise RuntimeError(f"{name} {outcome}: {result.get('detail') or result.get('events')}")
        return result

    async def jaws_over(x: float, y: float) -> dict:
        # The tool point is a jaw face: center the jaws, not the tool.
        result = {}
        for _ in range(2):  # twice: the gripper tilts a little as it moves
            jaw = (await tool("observe"))["jaw_center_xyz_m"]
            result = await do("move_by", dx=x - jaw[0], dy=y - jaw[1], dz=0.0)
        return result

    try:
        await do("move_to", x=start[0], y=start[1], z=start[2] + 0.06)
        await do("set_wrist_roll", angle_rad=-1.5)  # jaws side to side
        # Its size across the jaws as they are turned now.
        width = (await tool("observe"))["scene"]["objects"][thing]["grip_width_m"]
        await do("set_gripper", opening_m=width + 0.05)
        jaw = (await jaws_over(start[0], start[1]))["jaw_center_xyz_m"]
        # Not below 1.5 cm over the workspace floor (z 0.14 on the table): the jaw tips.
        lowest = observed["workspace"]["z"][0] + 0.015
        await do("move_by", dx=0.0, dy=0.0, dz=max(start[2], lowest) - jaw[2])
        held = await do("set_gripper", opening_m=width)
        if held.get("holding") != thing:
            return False, f"{thing} not held: {held.get('events')}"
        await do("move_by", dx=0.0, dy=0.0, dz=0.08)
        if goal is None:
            still = (await tool("observe"))["scene"]["objects"][thing]["attached"]
            await do("set_gripper", position=0.0)
            await do("home")
            verdict = (still, f"{thing} {'held' if still else 'dropped'} through a lift (no goal box)")
        else:
            await do("move_to", x=goal[0], y=goal[1], z=0.22)
            await jaws_over(goal[0], goal[1])
            await do("set_gripper", position=0.0)
            await do("home")
            await asyncio.sleep(1.0)  # let it settle
            final = (await tool("observe"))["scene"]["objects"][thing]
            where = "in the goal box" if final["in_goal"] else "NOT in the goal box"
            verdict = (bool(final["in_goal"]), f"{thing} at {final['xyz_m']}: {where}")
    except RuntimeError as error:
        return False, str(error)

    image = (await session.call_tool("camera", {"view": "overhead"})).content[0]
    if image.type != "image":
        return False, "camera returned no image"
    if image_path:
        Path(image_path).write_bytes(base64.b64decode(image.data))
        print(f"  overhead image: {image_path}")
    return verdict


async def run(
    url: str, key: str | None, scenes: list[str] | None, every: bool, image: str | None
) -> bool:
    async with AsyncExitStack() as stack:
        http_client = None
        if key:
            import httpx2

            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=60)
            )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        async def tool(tool_name: str, /, **args):
            result = await session.call_tool(tool_name, args)
            return json.loads(result.content[0].text)

        listed = (await tool("list_scenes"))["scenes"]
        current = next((s["name"] for s in listed if s.get("current")), None)
        names = [s["name"] for s in listed] if every else (scenes or [current])
        results: list[tuple[str, bool, str]] = []
        for name in names:
            print(f"== {name}")
            if name != current or len(names) > 1:
                loaded = await tool("load_scene", name=name)
                if not loaded.get("ok"):
                    results.append((name, False, f"load_scene: {loaded.get('detail')}"))
                    continue
            path = image
            if image and len(names) > 1:
                base = Path(image)
                path = str(base.with_name(f"{base.stem}-{name}{base.suffix}"))
            ok, note = await pick_and_place(tool, session, path)
            print(f"  {note}")
            results.append((name, ok, note))
        if len(names) > 1 and current is not None:
            await tool("load_scene", name=current)  # leave it as it was found

        print("\nscene                  result")
        for name, ok, note in results:
            print(f"{name:22s} {'PASS' if ok else 'FAIL'}   {note}")
        return bool(results) and all(ok for _, ok, _ in results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument(
        "--key",
        default=os.environ.get("ARM_KEY"),
        help="Arm key for a hub (default: $ARM_KEY). Prefer the environment variable: "
        "a key on the command line can end up in shell history.",
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--scene", action="append", help="Scene to test (repeatable)")
    which.add_argument("--all-scenes", action="store_true", help="Test every scene list_scenes offers")
    parser.add_argument(
        "--image", default=None, help="Save the overhead image here (one per scene, suffixed)"
    )
    args = parser.parse_args()
    ok = asyncio.run(run(args.url, args.key, args.scene, args.all_scenes, args.image))
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

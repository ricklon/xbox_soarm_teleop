"""Smoke-test a running twin over MCP: pick the object into the goal box.

Start the twin first (in another terminal):
    uv run soarm-twin
Then:
    uv run python scripts/twin_smoke.py [--url http://127.0.0.1:8765/mcp] [--image out.jpg]

Exits 0 when the object (bean bag or cube) ends up in the goal box and a camera image was returned.
Use it to check a student laptop (viewer, cameras and MCP together).
"""

import argparse
import asyncio
import base64
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def run(url: str, image_path: str | None) -> bool:
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def tool(name: str, **args):
                result = await session.call_tool(name, args)
                return json.loads(result.content[0].text)

            await tool("reset_scene")
            observed = await tool("observe")
            thing, state = next(iter(observed["scene"]["objects"].items()))
            cube = state["xyz_m"]
            width = min(state["size_m"][:2])  # across its sides
            goal = observed["scene"]["goal"]["center_xyz_m"]
            steps = [
                ("move_to", {"x": cube[0], "y": cube[1], "z": cube[2] + 0.06}),
                ("set_wrist_roll", {"angle_rad": -1.5}),  # jaws side to side
                ("move_to", {"x": cube[0], "y": cube[1], "z": cube[2] + 0.025}),
                ("set_gripper", {"opening_m": width}),
                ("move_to", {"x": goal[0], "y": goal[1], "z": 0.22}),
                ("set_gripper", {"position": 0.0}),
                ("home", {}),
            ]
            for name, args in steps:
                result = await tool(name, **args)
                state = result.get("action", {}).get("state", result.get("reason"))
                print(f"{name:14s} {state}")
                if state != "SUCCEEDED":
                    return False
            await asyncio.sleep(1.0)  # let it settle
            final = (await tool("observe"))["scene"]["objects"][thing]
            in_box = final["in_goal"]
            print(
                f"{thing} at {final['xyz_m']}: {'in the goal box' if in_box else 'NOT in the goal box'}"
            )
            image = (await session.call_tool("camera", {"view": "overhead"})).content[0]
            if image_path:
                with open(image_path, "wb") as handle:
                    handle.write(base64.b64decode(image.data))
                print(f"overhead image: {image_path}")
            return in_box and image.type == "image"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--image", default=None, help="Save the overhead camera image here")
    args = parser.parse_args()
    ok = asyncio.run(run(args.url, args.image))
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

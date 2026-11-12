# Spike: MuJoCo in the browser

Question: does the twin's scene load and run acceptably with MuJoCo's official
WebAssembly build (`@mujoco/mujoco`), with three.js doing the drawing?

Answer, on the one machine it was tried on: yes. The same scene the hub
simulates (arm, table, cube, goal box) compiles in the page, steps at real
time and draws at the display's refresh rate in Chrome, WebKit and Firefox.
The cost is start-up: compiling the model takes seconds.

This is a standalone page. It does not touch the hub.

## Run it

```bash
uv run python spikes/mujoco_web/export_scene.py
```

```bash
python3 -m http.server 8822 --bind 127.0.0.1 --directory spikes/mujoco_web
```

Open <http://127.0.0.1:8822/>. The page needs the internet once: it loads
`@mujoco/mujoco@4.0.1` and `three@0.170.0` from jsDelivr.

`export_scene.py` writes `scene/` (not committed): the model as MJCF XML, the
13 STL meshes it names (16.1 MB) and a manifest. The page puts those in
MuJoCo's virtual file system and compiles the model there.

## What the page does

- Compiles the model with `MjModel.from_xml_string(xml, vfs)`.
- Moves the arm the way the twin does: joint angles are written each physics
  step and `mj_step` moves everything else. Here the angles follow a fixed
  path; in the twin they come from the IK.
- Draws every geom with three.js from `geom_xpos` / `geom_xmat`. The browser
  build has no renderer of its own (no `mjr_*` functions).
- Reports load time, frame rate, physics time per frame and simulation speed,
  on screen and in `window.spikeStats`.

## Measurements

Apple M3 Max, macOS, 1280x800 window, 12 November 2026. Browsers were driven
by Playwright with a visible window. Each figure is from one run.

| Browser | Compile model | Frames per second | Physics per frame | Sim speed |
|---|---|---|---|---|
| Chrome 157 | 3.7 s | 120 | 1.1 ms | 1.00x |
| WebKit 27.2 (Playwright's build, not Safari itself) | 3.1 s | 60 | 1.0 ms | 1.03x |
| Firefox 159 | 12.1 s | 120 | 1.1 ms | 0.98x |

Each frame draws 398,970 triangles in 25 draw calls. Physics is 500 steps per
second (timestep 0.002 s), the same as the twin.

With Chrome's processor throttling the compile took 15.6 s at 4x and 23.9 s
at 6x. The frame rate and physics figures did not slow down under throttling,
which is not believable, so treat them as unmeasured for slow machines.

## What this does not show

- **Ordinary laptops.** One fast machine was tested. Integrated graphics and
  Windows are untested, and the package's README still calls the bindings a
  work in progress with experimental Windows support.
- **Real Safari.** Playwright's WebKit is the same engine but not the same
  browser.
- **A real network.** The meshes were served from this machine; 16 MB over a
  home connection will take longer than the 0.1 s seen here.
- **The twin itself.** No IK, assisted grasp, stop, agent or log runs here.
  Those are Python in `TwinService`.
- **The arm picking anything up.** The fixed path does not reach the cube.

## Notes for whoever takes this further

- Start-up is the weak point, and it is mesh processing: 322,564 triangles.
  Simplified meshes would cut download, compile time and drawing cost at once.
- A precompiled binary model is not a way round it: `mj_saveModel` gives a
  68 MB file for this scene.
- The npm package has a 4.0.1 build, matching the MuJoCo 4.0 this project
  uses. The model XML written by Python 4.0.0 loaded without changes.
- Typed arrays such as `data.qpos` are live views into the simulation, so the
  page reads and writes them directly.
- `HEAPU8` is not exported by the package, so memory use was not measured.

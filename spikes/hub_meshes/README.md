# Spike: simplified meshes in the hub

Question: if `soarm-hub` drew its camera pictures from the simplified meshes
(see [../mujoco_web](../mujoco_web/README.md)), how many more pictures a
second could it serve, and does the arm still behave the same?

Answer, on one machine: about 2.5 times the drawing capacity at 2 mm, the
pictures look the same, and the scripted pick-and-place and push end the same
way. The student page does not benefit yet, because its own pause between
pictures is what limits it.

Nothing in the hub is changed by this spike. Both scripts swap a simplified
model in only while they build their own hub. The IK reads the original URDF.

## Picture rate

```bash
uv run python spikes/hub_meshes/bench.py --cell-mm 0 --cell-mm 1 --cell-mm 2 --seats 10 --seats 30
```

Scripted students ask their arm for 480x360 pictures, one view each. "Page"
waits 130 ms after each picture, as the student page does. "Flat out" asks
again at once, which shows what the hub can draw. Apple M3 Max, 12 November
2026, 6 seconds per row, one run each. Shadows and reflections off, as the
hub runs them.

| Meshes | Triangles | Students | Mode | Pictures/s, all | Per student | Wait for a picture | Loop |
|---|---|---|---|---|---|---|---|
| Full | 322,564 | 10 | page | 66 | 6.6 | 19 ms | 50 Hz |
| Full | 322,564 | 10 | flat out | 245 | 24.5 | 40 ms | 50 Hz |
| Full | 322,564 | 30 | page | 190 | 6.3 | 25 ms | 50 Hz |
| Full | 322,564 | 30 | flat out | 189 | 6.3 | 159 ms | 50 Hz |
| 1 mm | 45,957 | 10 | page | 70 | 7.0 | 7 ms | 50 Hz |
| 1 mm | 45,957 | 10 | flat out | 561 | 56.1 | 18 ms | 50 Hz |
| 1 mm | 45,957 | 30 | page | 210 | 7.0 | 7 ms | 50 Hz |
| 1 mm | 45,957 | 30 | flat out | 407 | 13.6 | 74 ms | 50 Hz |
| 2 mm | 22,893 | 10 | page | 70 | 7.0 | 7 ms | 50 Hz |
| 2 mm | 22,893 | 10 | flat out | 627 | 62.7 | 16 ms | 50 Hz |
| 2 mm | 22,893 | 30 | page | 215 | 7.2 | 7 ms | 50 Hz |
| 2 mm | 22,893 | 30 | flat out | 459 | 15.3 | 64 ms | 50 Hz |

Reading it:

- Capacity rises from about 190-245 pictures a second to 460-630 at 2 mm.
- With the page's 130 ms pause a student can get at most 7.7 pictures a
  second. Thirty students on full meshes already use all the hub can draw
  (190 of 190); on 2 mm they use under half (215 of 459).
- So simpler meshes alone change little for students (6.3 to 7.2 a second
  with 30). The gain appears only if the page also asks more often: with
  2 mm, 30 students could each get about 15 a second.
- The control loop held 50 Hz in every row.

This measures drawing and JPEG encoding inside the hub. The web server, the
network and the browser are not in it, and the arms were idle. An earlier test
through the web server gave 30 students 2 to 4 pictures a second each, lower
than the 6.3 here, so expect real figures below this table.

## Behaviour

```bash
uv run python spikes/hub_meshes/compare_behaviour.py --cell-mm 0 --cell-mm 1 --cell-mm 2
```

The same 13 scripted actions on a one-arm hub: a pick and place into the goal
box, then a sweep of the closed gripper past the cube.

| Meshes | Actions that succeeded | Cube after the place | Cube after the push | Push differs from full |
|---|---|---|---|---|
| Full | 13 of 13 | in the box, (0.305, 0.153) | moved 8.0 mm | - |
| 1 mm | 13 of 13 | in the box, (0.292, 0.154) | moved 7.9 mm | 0.2 mm |
| 2 mm | 13 of 13 | in the box, (0.315, 0.154) | moved 7.4 mm | 0.7 mm |

The cube landed in the goal box every time, about 1 cm apart along x. It is
dropped from the open jaws and settles against them and the box walls, so
small changes in the jaw's shape move where it comes to rest. The push only
grazed the cube; a harder push was not tried.

## Pictures

The wrist and front views at full and 2 mm were compared by eye. They match,
apart from slightly smaller screw holes on the gripper in the wrist view.

## If this goes into the hub

- Simplify once and keep the files (the scripts redo it on each start, about
  half a second), or do it when the hub starts.
- For no change in contact at all, draw from the simplified model and keep
  simulating with the original. Both have the same joints, so the painter
  would copy joint positions across. That was not built here.
- Lower the page's 130 ms pause, or the extra capacity goes unused.

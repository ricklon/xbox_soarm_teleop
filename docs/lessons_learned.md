# Lessons learned: vision, agents and the sorting cell

What the vision benchmark, the agent trials and the sorting-cell build taught
us, with the evidence for each and the rule it left behind. Details and
numbers are in [vision_benchmark.md](vision_benchmark.md) and
[building_scenes.md](building_scenes.md).

## The big one: put each control method where it is strong

There are several ways to control a robot arm because each solves a
different part of the problem:

| Method | Good at | Weak at |
|---|---|---|
| Teleoperation (the Xbox pad) | a person corrects every centimetre | needs a person for every action |
| Scripted or classical (IK, a fixed pick routine) | precise, fast, repeatable, cheap | needs exact positions; brittle to surprises |
| Learned policies (LeRobot ACT, diffusion, SmolVLA) | camera to joints many times a second, correcting as they go | need demonstrations; generalize unevenly |
| Language-model agents | understanding requests, rules, ambiguity; sequencing; judging success | slow (1 to 3 s per decision), act on snapshots |

For weeks we asked a language model to steer the jaws a few centimetres at a
time, then patched the world (spacing, trays, layouts) to make that easier.
Giving the agent skills instead (`pick`, `place`: the arm does the motion,
the agent decides what goes where) took the same sort tasks from **1 of 9 to
9 of 9**, cost from **$1.28 to $0.12**, false "done" claims from **2 to 0**, and
cubes knocked off the table from **6 runs to 0**.

**Rule:** the language model plans and judges; skills move; people demonstrate
and take over. Benchmark each layer at its own job.

## How we work

1. **Measure in layers, not only end to end.** End-to-end trials mixed
   perception, grasping and understanding; every failure took about an hour
   and a dollar to diagnose. Separate checks: does `locate` find it (the
   pointing benchmark), does a grasp survive realistic aim error (scripted,
   free), does the agent understand the request (true positions, cheap), and
   only then the whole thing.
2. **Prove a challenge can be done before an agent tries it.** A scripted run
   with true positions caught impossible layouts (a far bin knocked by the
   arm reaching past a near one). But scripts aim perfectly, so the proof must
   also aim like an agent (off by up to 1.5 cm) and require that nothing is
   pushed or dropped.
3. **Look at the pictures.** Tall bins hiding the cubes from `locate`'s
   camera, the arm hiding a cube under its home pose, and the pointer
   choosing a tray over a cube were all found by looking at what the camera
   saw, after the numbers had only said "it failed".
4. **Replay before you theorize.** Re-running an agent's exact actions in a
   fresh twin reproduced the "home knocks the cube out" failure to the
   millimetre, and tested the fix (lift 5 cm first) in seconds.
5. **Count invalid runs as invalid.** Runs died on a Cloudflare daily quota,
   on an OpenRouter credit limit (HTTP 402), and on a crash in our own code.
   Each would have read as an agent failure. Trials now check the judge
   answers before starting, and reports exclude runs whose judge or model
   service failed.
6. **Watch for the same bug twice.** Logging crashed twice on a field named
   like the logger's own (`t`, then `bin`). Every new logged result now has a
   test on a twin that keeps a session log.
7. **Step back when the fixes get small.** Each trial on the crowded table
   found one layout problem and we patched that one rule. The real problem
   was a cell too small for the task. When fixes keep shrinking, question the
   setup.

## The cell: space, cameras, reach

8. **Size the cell to the arm.** The SO-101 grasps from above in a ring 17 to
   36 cm from its base, across about +-80 degrees (up to 40 cm and +-110 less
   steadily), with a dead zone inside 15 cm. The first table covered about a
   quarter of that ring and its far corners were out of reach. The sorting
   desk (54 x 84 cm, at the base's height) covers it.
9. **Pick in front, place to the sides.** With items in front and trays to the
   arm's sides, placing turns the base, so the arm never passes over items
   still to be sorted. In a row leading away from the base, lowering into the
   far bin swung the forearm over the near one and knocked its cube out.
10. **Lift before you go home.** Straight after a release the open jaws still
    straddle the object; homing from there threw it out of the box in 5 of 13
    failed runs. Lifting 5 cm first took the A/B from 13 of 26 to 36 of 36.
11. **Leave room for an imperfect grasp.** The open jaws span about 9 cm and
    an agent's aim is off by 1 to 2 cm. Cubes 6 to 8 cm apart got pushed;
    one 2 cm from the edge walked off it after repeated misses. Rules now:
    10 cm apart, 5 cm in from edges, 5 cm from trays.
12. **Turn objects to face the arm.** A cube square to the table but 25 degrees
    off to the side is 40 degrees off from the jaws and slips.
13. **Keep the cameras' view clear.** 6 cm bin walls hid the cubes behind
    them; the arm at home hides the strip straight ahead from overhead. Trays
    now have 2 cm walls, and a test checks every object is in view of the
    front and overhead cameras.
14. **Check the safety limits against the cell.** The arm's configured 5 cm
    floor kept the jaws off a desk-level surface. The twin now sets the floor
    5 mm over whatever table the scene has; a real desk-mounted arm needs the
    same change. The simulated arm moves kinematically, so nothing stops it
    passing through the desk (its bounding box dips about 6 mm).

## Perception

15. **Pointing is good; finding the right object is the hard part.** Gemini
    3.5 Flash points to within 0.1 to 0.3 cm, Gemma 4 31B within 0.3 to 0.5 cm
    at a twentieth of the cost. Colour is easy for every model; shape, size
    and clutter separate them; the wrist camera gets clutter wrong a third to
    half of the time.
16. **A look-alike can be furniture.** From overhead, a red cube and a red
    tray's floor are both red squares: the pointer chose the tray, and the
    wrist camera then "confirmed" it. Cross-check views (a front view shows a
    cube standing up and a tray lying flat) and use what the robot knows
    about its own trays.
17. **Know each camera's weak axis.** The front camera's error is mostly in
    depth; overhead has no depth confusion; the wrist camera close up is about
    half a centimetre off. `pick` looks from overhead, then centres with the
    wrist camera.
18. **`locate` assumes the object rests on the table.** A cube knocked onto
    the floor was "located" 12 to 30 cm wrong, and nothing told the agent it
    had fallen.

## Agents

19. **Agents claim success they did not check.** Asked to swap two cubes, an
    agent moved one and said "done". A success judge it can ask
    (`check_success`) made agents stop by themselves and their claims true;
    Clef's confident "done" was right 22 of 22 times. Check after the last
    motion, not before homing.
20. **The request must carry the rule.** Putting "red goes in the red bin" in
    the scene turned sorting into following a lookup table. Now the scene
    only has bins and objects; each task is several wordings plus what must
    be true at the end; ambiguous requests show what the agent assumes (with
    no yellow tray, it put the yellow cube in the green one, every time).
21. **Don't leak the answer.** Object names (`blue_cube`) give the colour away;
    tool results in no-coordinates mode must hold no positions; the
    answer key is a page route, not a tool.
22. **The agent model matters more than the pointer, and the harness costs.**
    One model succeeded where another never did with perfect positions;
    opencode sent about 18,000 tokens of instructions per step, so we built a
    lean loop.

## Running experiments

23. **Budget and limits are part of the setup.** OpenRouter refuses requests
    (HTTP 402, "in-flight budget") when the balance runs low, even with some
    credit left; run trials one at a time and check the balance first. The
    Gemini API's free tier allows 20 calls a day; Cloudflare's free tier has a
    daily allocation.
24. **Costs per call:** Gemini 3.5 Flash $0.004, Gemma 4 31B $0.0002, Clef
    $0.0002. A whole sort run with skills: about 1 to 2 cents.

## What we have not solved yet

- `check_success` judges only the goal box, not bins.
- The plan-first step (agent states what it sees and intends before moving)
  is designed and its tests written, but not built.
- Object names still give colours away.
- The 8 of 9 `pick` rate uses simulated aim errors; real pointer behaviour
  differs (lesson 16).
- Nothing yet compares with a learned policy or with a person on the pad on
  the same cell and tasks.

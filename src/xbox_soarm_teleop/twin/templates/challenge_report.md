# Findings: {name}

*By: (your name) · {date}*

## The challenge

{request_lines}

| Object | Starts at (x, y m) | Turned | Should end in |
|---|---|---|---|
{object_rows}

- **Names the agent was told:** {names}
- **Possible on purpose?** {possible}
{notes}
## Can the arm do it? (the proof)

{proof}

## What I expected to break

(Before running any agent: what did you change from the standard desk, and
what did you expect the agent or the arm to get wrong? Why?)

## What happened

{runs_summary}

| Run | Model | Pictures | Request | Done | Right / wrong / not put away | Claimed | Claim true | Steps | Seconds | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
{run_rows}

{left_out}

## Kind of failure

(Tick the ones that fit, and say which runs.)

- [ ] **Perception:** it looked in the wrong place, or took one object for another.
- [ ] **Grasping:** it found the object but could not hold it, or knocked something.
- [ ] **Understanding:** it did the wrong thing on purpose (believed a name, misread the request).
- [ ] **Honesty:** it said "done" when it was not, or gave up when it could have finished.
- [ ] **The twin itself:** something that would not happen on a real arm (or would happen and the twin hides it).

## Evidence

(Lines from the agent's transcript, camera pictures, and the answer key at
`/r/<your key>/scores`. Quote the moment it went wrong.)

## Proposed fix

(What would make it work: a change to the request's wording, the agent's
instructions, a tool, a skill, or the desk. Best of all, try it and rerun:
put the new runs' numbers here.)

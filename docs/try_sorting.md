# Try it: the sorting desk with your own agent

A short path to everything students use on the sorting desk: start a hub,
open an arm's page, connect an agent, ask it to sort, and check how it did.
For the background see [building_scenes.md](building_scenes.md) and
[lessons_learned.md](lessons_learned.md).

## 1. Start the hub

```bash
uv run soarm-hub --preset sorting --seats 2
```

`--preset sorting` sets up the whole session:

| Setting | What it means |
|---|---|
| `--scene sort_colors` | the sorting desk: a red, a blue and a green cube, and a tray of each colour |
| `--no-coordinates` | agents are not told where things are; they have to look |
| `--pointer openrouter:google/gemma-4-31b-it` | what `locate` and `pick` look with (needs `OPENROUTER_API_KEY`) |
| `--judge clef` | what `check_success` asks (needs `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`) |
| `--names truthful` | objects are shown to agents as `red_cube`, `blue_cube`, `green_cube` |

The keys go in the project's `.env` (copy `.env.example`). Options you give
still win, for example `--names neutral` or `--judge none`. For students on
other machines add `--public-url` (see [workshop_hosting.md](workshop_hosting.md)).

The hub opens the instructor page; **Copy all student links** there gives
each arm's page.

## 2. Open an arm's page

The page shows the arm from four cameras, drive buttons, a **STOP** button,
the scene picker (`sort_colors`, and `sort_colors_decoy` with a yellow cube
and no yellow tray), and an **Objects** line saying where each cube is (in
the gripper, in which tray, on the table), under the names your agent sees.

You can always take over (hold a drive button) or stop the arm; an agent
cannot release a stop you latched.

## 3. Connect your agent

Your agent drives the same arm through the arm's tools. Any of these:

- **Claude Code.** The page's **Connect an AI agent** box shows the exact
  command, with your arm's key; click it to copy, run it in a terminal, then
  start `claude` and ask it to use the soarm-arm tools.
- **opencode or another MCP client.** Add a remote MCP server with the
  page's **Address** and a header `Authorization: Bearer <your key>`. For
  opencode, in `opencode.json`:

  ```json
  {"mcp": {"soarm-arm": {"type": "remote", "url": "<address>",
    "headers": {"Authorization": "Bearer <your key>"}}}}
  ```

- **The Colab notebook** (`notebooks/workshop_sorting.ipynb`), from the link
  at the bottom of the box: look, find by looking, pick and place, check,
  write a sorting program, then hand the job to an agent (OpenRouter or
  Claude, with or without pictures) and grade it against the answer key. The
  link opens the notebook from GitHub's `main` branch, so it works for
  students once this branch is merged there; until then open the file from
  your own copy (in Jupyter, set `ARM_URL` to the page's address).

Whichever you use, the arm tells the agent how to work: use `pick` and
`place` (the arm does the moving), and after the last move check each cube
with `check_success`, describing it by colour.

## 4. Things to ask

The page lists these under **Try asking your agent**; click one to copy it.

1. **Look around**: what is on the desk, which trays there are.
2. **Sort by colour**, then check each cube by colour.
3. **Swap two**: red cube to the blue tray, blue cube to the red one.
4. **Only one**: the red cube away, nothing else moves.
5. **The odd one out** (`sort_colors_decoy`): what does it do with the
   yellow cube, which has no tray, and does it say why?
6. **Check, don't trust**: check every cube without moving anything.

Then try what the lessons are about:

- Restart with `--names neutral` (`object_1`, ...): the agent has to look to
  know which cube is which.
- Restart with `--names misleading` (the red cube is called `blue_cube`):
  does your agent believe the names or check? Ask it to check each cube "by
  colour" and see what `check_success` says.

## 5. How did it do?

- **On the page:** the **Objects** line says where each cube really is.
- **The answer key** (the true positions, every guess `locate` made and how
  far off it was, every check and whether it was right): open your arm's page
  address with `/scores` added, for example
  `http://127.0.0.1:8800/r/<your key>/scores`. Agents never see it; it is a
  page, not a tool.
- **The record:** every arm's session is logged in
  `twin_logs/hub-<time>/arm-NN.jsonl` on the instructor's computer.

## Costs

Each `locate` look costs about $0.0002 (Gemma 4 31B) and each check about
$0.0002 (Clef), on the instructor's keys; a whole sort with skills is about
1 to 2 cents. Your own agent's model is on your key.

# Hosting a Remote Workshop

For a workshop over Zoom or similar: you run one program, and every student
gets their own simulated arm in a browser. Students install nothing.

Each student gets:

- **a web page** for their arm: live camera, hold-to-move controls, STOP;
- **an address and key** for connecting an AI agent to that same arm;
- **a notebook** that walks them through driving the arm by code and by agent.

You get an instructor page that shows every arm at once.

Running the twin on each student's own laptop with an Xbox controller is the
other way to do it; see the [Workshop Guide](workshop_guide.md).

See also: [Agent interface design](mcp_control_interface.md) · [Schemas](schemas.md)

---

## 1. Before the day

On the computer that will host (it stays on and awake for the whole session):

```bash
uv sync
```

```bash
uv run soarm-hub --seats 3
```

It prints an **instructor page** link. Open it: you should see three arms.
Click **Open** on one, hold **W**, and watch the arm move. That link only
works on this computer; the next section makes it reachable.

Decide two things in advance:

- **How students reach your computer** (section 2).
- **How students get an Anthropic API key** for the agent part of the notebook
  (section 5). Parts 1 to 4 of the notebook need no key.

## 2. Make it reachable

`soarm-hub` listens on this computer only. Students need an `https://` address
that forwards to it. Any tunnel or reverse proxy that provides HTTPS works;
these are two common ones.

**A quick tunnel (no account).** Install
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/),
then start the tunnel *first*, because the hub needs to know its address:

```bash
cloudflared tunnel --url http://localhost:8800
```

It prints an address like `https://something-random.trycloudflare.com`. Leave
it running and start the hub with that address:

```bash
uv run soarm-hub --seats 12 --public-url https://something-random.trycloudflare.com
```

A quick tunnel gets a new address every time it starts, so get it running and
leave it. With your own domain or a named tunnel the address stays the same.

**On a campus or office network**, students on the same network can connect
directly if you listen on all interfaces and tell the hub the address they
will use:

```bash
uv run soarm-hub --seats 12 --host 0.0.0.0 --public-url http://YOUR-COMPUTER-NAME:8800
```

This is plain HTTP: keys travel unencrypted. Use it only on a network you trust.

> The hub was developed and load-tested on one machine. It has not been run
> through a tunnel with a real class yet. Do a dry run with a second device
> on a different network (a phone on mobile data) before the day.

## 3. Hand out the links

Open the instructor page. **Copy all student links** puts one line per arm on
your clipboard:

```
Arm 1: https://.../r/k13JnksmVABUEtUT
Arm 2: https://.../r/9VFc66bEU0fb84gp
```

Send each student *their own* line privately (a direct message, not the main
chat). The link is the only thing protecting an arm: anyone who has it can
drive that arm.

The links are kept in `twin_logs/hub_roster.json`, so they stay the same if
you restart the hub with the same `--public-url`. Delete that file to issue
new ones. Ask for more seats than you expect; an unused arm costs nothing.

## 4. During the session

The instructor page shows, for every arm: who is there (page open, agent
connected), what the arm is doing, the last thing that happened, and a small
picture that refreshes in turn.

- **STOP ALL** stops every arm at once, including any agent's move, and agents
  cannot undo it. Use it to get the room's attention.
- **Release all stops** lets everyone carry on. A student can also release
  their own arm.
- **Open** shows you a student's page, to see exactly what they see. You are
  then driving their arm too, so keep your hands off the keys.

Each arm writes a session log to `twin_logs/hub-<date>-<time>/arm-NN.jsonl`:
every request, refusal, stop and takeover, with timestamps (format in
[Schemas](schemas.md)). They are useful afterwards for seeing where students
got stuck.

### What students do

1. Open their link. The page explains itself: hold a button or key to move,
   STOP is the red button.
2. Open the notebook from the link at the bottom of the page's **Connect an AI
   agent** box, and paste their address and key into its first cell. The
   default link opens `notebooks/workshop_agent.ipynb` from this repository's
   `main` branch in Google Colab; pass `--notebook-url` to point elsewhere.
3. Work through it: look, move, pick-and-place by program, limits and stop,
   then the agent.

Students who use Claude Code can connect it directly with the command shown
on their page instead of using the notebook.

### An agent in the student's own browser (WebMCP, experimental)

The page also offers its tools to the browser itself, through WebMCP. An agent
that runs in the student's browser can then drive the arm with no address or
key to paste. The tools are the same ones the agent address serves, and they
obey the same rules: the student can take over or stop at any time, and an
agent cannot release a stop the student latched.

The **Connect an AI agent** panel says whether the browser has it:
`Ready: 13 tools are registered with this browser (WebMCP)` or
`This browser does not offer page tools`. Nothing else on the page depends on
it.

WebMCP is an experimental Chrome feature. It was checked in Chrome 157 started
with `--enable-features=WebMCP --enable-blink-features=WebMCPTesting`; for
everyday use Chrome has the `chrome://flags/#enable-webmcp-testing` switch.
Safari and Firefox do not have it. The tools work only while the page is open,
and the student also needs an agent that can use page tools. Treat it as an
extra for students who want to try it, not as the route the class depends on.

## 5. API keys for the agent part

Part 5 of the notebook runs an agent with the Anthropic API and asks for a key.
The hub does not provide one. Options:

- students bring their own key;
- you create a workspace with a spend limit in the Anthropic Console and give
  out keys from it, then delete them afterwards.

The notebook uses `claude-opus-5-5`. A pick-and-place takes the agent roughly
ten to twenty tool calls.

## 6. How many students?

Measured on the development machine (Apple silicon laptop, 14 cores):

| | |
|---|---|
| Start-up | about 3 s for 30 arms |
| Memory | about 1.1 GB for 30 arms |
| Control rate | 50 Hz with 30 students driving and watching at once |
| Response to a key press | 4 ms median, 8 ms at the 95th percentile (on the same machine, before network delay) |
| Camera | 137 frames/s shared by everyone watching |

The camera is the limit. Ten students each get the page's full 6 to 7 frames
a second; thirty all watching at once get about 2 to 4 each, which is usable
at the arm's speed (10 cm/s at most) but choppy. Camera pictures are drawn
without shadows or reflections to get that rate; `Hub(shadows=True)` in code
turns shadows back on at about half the frame rate.

## 7. When something goes wrong

| What you or a student sees | What to do |
|---|---|
| `port 8800 is in use` | Another hub is running. Close it or use `--port` |
| Student's page says **Not connected** | The tunnel or the hub stopped. Restart it; pages reconnect by themselves |
| Student's link shows `Not found` | The link is wrong or from an older roster. Copy it again from the instructor page |
| Notebook: `401 Unauthorized` | Address or key pasted wrongly. The address ends in `/mcp` |
| Agent tool calls fail with `421` or "Invalid Host header" | The hub was started without `--public-url`, or with a different address than students use |
| Agent is refused with `not_owner` / `human` | The student touched the controls in the last 2 seconds. Hands off and retry |
| Agent is refused with `stopped` | The student (or you) stopped the arm. Press **Release stop** |
| Arm will not move | Look at the top of the page: `STOPPED` means press **Release stop** |
| Camera is slow for everyone | Many people watching at once (section 6). Ask students to close extra tabs |
| The cube is off the table | **Reset scene** on the page |

To finish, press Ctrl+C in the hub's terminal, then stop the tunnel.

## What is and is not protected

- Each arm and the instructor page are protected only by the random key in
  their links. There are no accounts or passwords.
- Student keys are not written to the hub's terminal output, but the
  instructor link is, and the instructor page shows every student's link. Do
  not share either window on screen.
- Everything is simulated. The worst a leaked key allows is someone else
  moving a student's simulated arm.

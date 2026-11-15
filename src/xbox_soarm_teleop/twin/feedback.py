"""Student feedback: what worked, what did not, and what they wish it could do.

Students send it from their arm's page (``/r/{key}/feedback``), with what
they were doing attached if they like: the scene, the challenge, the agent's
last run. It stays on the hub, in the session's logs, until the instructor
reviews it (``/admin/{key}/feedback``: edit, drop, post) and posts it as a
GitHub issue, one per item or a digest of everything new. Posting uses the
instructor's own ``gh`` login; students never hold a GitHub key.

Keys never leave in feedback: room and instructor keys are cut out of
everything stored, and API keys are masked.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

MAX_FIELD_CHARS = 2000
FIELDS = ("worked", "didnt", "wish")
HEADINGS = {"worked": "What worked", "didnt": "What didn't work", "wish": "I wish it could"}
LABELS = {"worked": "worked", "didnt": "didnt-work", "wish": "wish"}
BASE_LABEL = "student-feedback"


@dataclass
class FeedbackItem:
    id: int
    room: str
    created: str
    worked: str = ""
    didnt: str = ""
    wish: str = ""
    name: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    status: str = "new"  # new, dropped, posted
    url: str = ""


class FeedbackBox:
    """Feedback waiting for review, saved as JSON at ``path`` (None: kept in memory)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.items: list[FeedbackItem] = []
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            self.items = [FeedbackItem(**entry) for entry in saved.get("feedback", [])]

    def add(
        self, *, room: str, worked: str, didnt: str, wish: str, name: str, context: dict
    ) -> FeedbackItem:
        with self._lock:
            item = FeedbackItem(
                id=1 + max((i.id for i in self.items), default=0),
                room=room,
                created=time.strftime("%Y-%m-%d %H:%M"),
                worked=worked,
                didnt=didnt,
                wish=wish,
                name=name,
                context=context,
            )
            self.items.append(item)
            self._save()
            return item

    def get(self, item_id: int) -> FeedbackItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    def update(self, item: FeedbackItem, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(item, key, value)
            self._save()

    def for_room(self, room: str) -> list[dict]:
        return [asdict(item) for item in self.items if item.room == room]

    def all(self) -> list[dict]:
        return [asdict(item) for item in self.items]

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"feedback": [asdict(i) for i in self.items]}, indent=2) + "\n")


def scrub(value: Any, secrets: list[str]) -> Any:
    """``value`` with every key in ``secrets`` cut out, and API keys masked."""
    from xbox_soarm_teleop.twin.pointing import mask_secrets

    if isinstance(value, dict):
        return {k: scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[key removed]")
        return mask_secrets(value)
    return value


def labels_for(items: list[FeedbackItem]) -> list[str]:
    return [BASE_LABEL] + [LABELS[f] for f in FIELDS if any(getattr(i, f).strip() for i in items)]


def _context_text(context: dict) -> str:
    if not context:
        return ""
    lines = ["", "<details><summary>What they were doing</summary>", ""]
    lines.append(
        f"- Scene: `{context.get('scene', '?')}`, names shown to agents: `{context.get('names', '?')}`"
    )
    challenge = context.get("challenge")
    if challenge:
        lines += [
            f"- Challenge: **{challenge.get('name')}**",
            "",
            "```json",
            json.dumps(challenge, indent=2),
            "```",
        ]
    result = context.get("challenge_result")
    if result:
        lines.append(f"- Challenge result: done={result.get('done')}, in_bin={result.get('in_bin')}")
    run = context.get("agent_run")
    if run:
        lines.append(
            f"- Last agent run: {run.get('model')} ({'sees' if run.get('sees') else 'blind'}), "
            f'asked: "{run.get("task")}"; {run.get("stop_reason")} after {run.get("steps")} steps'
        )
        if run.get("calls"):
            lines += ["", "```", *run["calls"], "```"]
        if run.get("final_text"):
            lines += ["", "> " + run["final_text"].replace("\n", "\n> ")]
    lines += ["", "</details>"]
    return "\n".join(lines)


def _item_text(item: FeedbackItem) -> str:
    parts = []
    for key in FIELDS:
        text = getattr(item, key).strip()
        if text:
            parts += [f"### {HEADINGS[key]}", "", text, ""]
    who = f" by {item.name}" if item.name.strip() else ""
    parts.append(f"*{item.room}{who}, {item.created}*")
    return "\n".join(parts) + _context_text(item.context)


def issue_text(item: FeedbackItem) -> tuple[str, str]:
    """One item as an issue's title and body."""
    first = next((getattr(item, f).strip() for f in FIELDS if getattr(item, f).strip()), "")
    kind = next((HEADINGS[f] for f in FIELDS if getattr(item, f).strip()), "Feedback")
    title = f"Student feedback ({item.room}): {kind.lower()}: {first[:60]}"
    body = _item_text(item) + "\n\n---\nPosted from the workshop hub by the instructor.\n"
    return title, body


def digest_text(items: list[FeedbackItem]) -> tuple[str, str]:
    """Several items as one issue."""
    title = f"Student feedback digest, {time.strftime('%Y-%m-%d')} ({len(items)} items)"
    body = "\n\n---\n\n".join(_item_text(item) for item in items)
    return title, body + "\n\n---\nPosted from the workshop hub by the instructor.\n"


def gh_post(repo: str, title: str, body: str, labels: list[str]) -> str:
    """Open an issue with the instructor's ``gh`` login; returns its address.

    If the repository lacks one of the labels, posts without labels rather
    than creating them.
    """

    def run(with_labels: list[str]) -> subprocess.CompletedProcess:
        command = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body-file", "-"]
        for label in with_labels:
            command += ["--label", label]
        return subprocess.run(command, input=body, capture_output=True, text=True, timeout=60)

    done = run(labels)
    if done.returncode != 0 and labels and "label" in (done.stderr or "").lower():
        done = run([])
    if done.returncode != 0:
        raise RuntimeError((done.stderr or "gh failed").strip().splitlines()[-1])
    return done.stdout.strip().splitlines()[-1]


Poster = Callable[[str, str, str, list[str]], str]

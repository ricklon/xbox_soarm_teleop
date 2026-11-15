"""Student feedback: what worked, what did not, and what they wish it could do.

Students send it from the arm's page, with what they were doing attached if
they like (the challenge, the agent's last run). It stays on the hub until the
instructor reviews it on the instructor page and posts it as a GitHub issue,
one per item or one digest. Keys never leave in it. Posting is the
instructor's own ``gh`` login; here it is a stand-in, so nothing is posted.
"""

import json
import logging
import socket
import threading
import time
import warnings

import httpx2 as httpx
import pytest
import uvicorn

from xbox_soarm_teleop.twin.feedback import FeedbackBox, issue_text
from xbox_soarm_teleop.twin.hub import Hub
from xbox_soarm_teleop.twin.hub_server import build_hub_app

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Poster:
    """Stands in for gh: records each issue and answers with its address."""

    def __init__(self):
        self.issues = []

    def __call__(self, repo, title, body, labels):
        self.issues.append({"repo": repo, "title": title, "body": body, "labels": labels})
        return f"https://github.com/{repo}/issues/{100 + len(self.issues)}"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    port = _free_port()
    hub = Hub(seats=2, scene="sort_colors", log_dir=tmp_path_factory.mktemp("logs"))
    poster = Poster()
    app = build_hub_app(
        hub, port=port, agent_models=(), feedback_repo="someone/some-repo", feedback_post=poster
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    hub.start()
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.05)
    yield hub, f"http://127.0.0.1:{port}", poster
    server.should_exit = True
    thread.join(timeout=5)
    hub.close()


def _token(hub, index=0):
    return list(hub.rooms)[index]


def _send(base, token, **fields):
    return httpx.post(f"{base}/r/{token}/feedback", json=fields)


def test_a_student_sends_feedback_and_sees_it_was_received(served):
    hub, base, _ = served
    token = _token(hub)
    httpx.post(
        f"{base}/r/{token}/challenge",
        json={
            "name": "only-red",
            "objects": {"red_cube": {"x": 0.275, "y": -0.06}},
            "request": "Put the red cube away.",
            "expect": {"red_cube": "red_bin"},
        },
    )
    reply = _send(
        base,
        token,
        worked="pick by description found the red cube",
        didnt=f"it said done too early (my key is {token})",  # pasted by mistake
        wish="push two cubes at once",
        name="Sam",
        attach=True,
    )
    assert reply.status_code == 200, reply.text
    mine = httpx.get(f"{base}/r/{token}/feedback").json()["feedback"]
    assert len(mine) == 1 and mine[0]["status"] == "new"
    item = mine[0]
    assert token not in json.dumps(item) and hub.admin_token not in json.dumps(item)  # keys scrubbed
    assert item["context"]["challenge"]["name"] == "only-red"
    assert item["context"]["scene"] and item["name"] == "Sam"
    # Another arm does not see it.
    assert httpx.get(f"{base}/r/{_token(hub, 1)}/feedback").json()["feedback"] == []


def test_empty_or_huge_feedback_is_refused(served):
    hub, base, _ = served
    token = _token(hub, 1)
    assert _send(base, token, worked=" ", didnt="", wish="").status_code == 400
    assert _send(base, token, wish="x" * 5000).status_code == 400
    assert httpx.post(f"{base}/r/not-a-room/feedback", json={"wish": "hi"}).status_code == 404


def test_only_the_instructor_reviews_and_nothing_is_posted_until_they_say(served):
    hub, base, poster = served
    assert httpx.get(f"{base}/admin/wrong/feedback").status_code == 404
    listing = httpx.get(f"{base}/admin/{hub.admin_token}/feedback").json()
    assert listing["repo"] == "someone/some-repo" and listing["posting"] is True
    assert [item["room"] for item in listing["feedback"]] == ["Arm 1"]
    assert poster.issues == []


def test_the_instructor_edits_then_posts_one_issue(served):
    hub, base, poster = served
    admin = f"{base}/admin/{hub.admin_token}/feedback"
    item = httpx.get(admin).json()["feedback"][0]
    edited = httpx.post(
        f"{admin}/{item['id']}", json={"action": "edit", "wish": "push two cubes together"}
    )
    assert edited.status_code == 200
    posted = httpx.post(f"{admin}/{item['id']}", json={"action": "post"})
    assert posted.status_code == 200, posted.text
    assert posted.json()["url"].startswith("https://github.com/someone/some-repo/issues/")
    issue = poster.issues[-1]
    assert issue["repo"] == "someone/some-repo"
    assert "student-feedback" in issue["labels"] and "wish" in issue["labels"]
    assert "push two cubes together" in issue["body"] and "only-red" in issue["body"]
    assert "Sam" in issue["body"]
    again = httpx.post(f"{admin}/{item['id']}", json={"action": "post"})
    assert again.status_code == 409  # posted once only
    assert httpx.get(admin).json()["feedback"][0]["status"] == "posted"


def test_a_digest_posts_several_items_as_one_issue_and_drop_keeps_one_back(served):
    hub, base, poster = served
    for token, text in ((_token(hub), "the trays are easy to see"), (_token(hub, 1), "it is slow")):
        _send(base, token, worked=text)
    admin = f"{base}/admin/{hub.admin_token}/feedback"
    new = [item for item in httpx.get(admin).json()["feedback"] if item["status"] == "new"]
    assert len(new) == 2
    httpx.post(f"{admin}/{new[1]['id']}", json={"action": "drop"})
    before = len(poster.issues)
    digest = httpx.post(f"{admin}/digest", json={})
    assert digest.status_code == 200, digest.text
    assert len(poster.issues) == before + 1
    assert "the trays are easy to see" in poster.issues[-1]["body"]
    assert "it is slow" not in poster.issues[-1]["body"]  # dropped
    statuses = {item["id"]: item["status"] for item in httpx.get(admin).json()["feedback"]}
    assert statuses[new[0]["id"]] == "posted" and statuses[new[1]["id"]] == "dropped"


def test_feedback_is_kept_in_the_session_logs(served):
    hub, _base, _ = served
    path = next(iter(hub.rooms.values())).log_path.parent / "feedback.json"
    saved = json.loads(path.read_text())
    assert len(saved["feedback"]) == 3


def test_without_a_repo_posting_is_off(tmp_path):
    box = FeedbackBox(None)
    item = box.add(room="Arm 1", worked="", didnt="it fell over", wish="", name="", context={})
    title, body = issue_text(item)
    assert "Arm 1" in title and "it fell over" in body and "What didn't work" in body

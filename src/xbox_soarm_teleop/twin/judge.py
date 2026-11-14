"""A success judge from camera images, graded against the twin's own state.

Cloudflare's Clef (``@cf/cloudflare/clef`` on Workers AI) is a decision model:
it takes a state (text and up to 4 images) and typed questions, and returns a
probability for every option of every question. Here it is asked, about one
object, the questions an agent needs answered on a real arm, where nothing
reports ``in_goal`` or ``attached``:

- ``in_goal``: is it resting in the goal box? (yes/no)
- ``held``: is it held in the gripper? (yes/no)
- ``tipped``: is it resting tilted on an edge or corner? (yes/no)
- ``where``: goal box, table, gripper or floor? (one of four)

:func:`truth_labels` gives the right answers from the simulator, using the
same test as ``observe``'s ``in_goal``. The metrics say how often the judge is
right and whether its probabilities can be trusted (calibration).

Credentials: ``CLOUDFLARE_API_TOKEN`` (Workers AI permission) and
``CLOUDFLARE_ACCOUNT_ID``, from the environment or a ``.env`` file. Both are
masked in every error.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from xbox_soarm_teleop.twin.layouts import GOALS
from xbox_soarm_teleop.twin.pointing import PointerError, _post_json, load_secret, mask_secrets

RUN_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
MODELS = ("clef", "clef-flash")
MAX_IMAGES = 4
QUESTION_IDS = ("in_goal", "held", "tipped", "where")
NOUL_IDS = ("in_goal", "held", "tipped")
WHERE_OPTIONS = ("in_goal_box", "on_table", "in_gripper", "on_floor")
TIPPED_DEG = 15.0  # tilted more than this off lying flat on a face counts as tipped
OFF_TABLE_MARGIN_M = 0.01

VIEW_TEXT = {
    "front": "from in front of the table, looking back at the robot (the robot's left is on "
    "the picture's right)",
    "overhead": "from straight above, like a map (robot base at the bottom)",
    "side": "from the robot's left side",
    "wrist": "from a camera on the robot's gripper",
}


class JudgeError(RuntimeError):
    """The judge could not be asked, or its answer made no sense."""


def questions_for(description: str) -> dict[str, dict]:
    """Clef's typed questions about one object ("red cube")."""
    d = description
    return {
        "in_goal": {
            "type": "noul",
            "instructions": f"Is the {d} resting inside the green goal box, on its floor and "
            "within its walls?",
            "criteria": {
                "true": f"the {d} is inside the open green box",
                "false": f"the {d} is anywhere else: on the table (even right next to the box), "
                "in the gripper, or on the floor",
            },
        },
        "held": {
            "type": "noul",
            "instructions": f"Is the {d} held between the jaws of the robot's gripper?",
            "criteria": {
                "true": f"the gripper is holding the {d}, lifted off the table",
                "false": f"the {d} is resting somewhere, not in the gripper",
            },
        },
        "tipped": {
            "type": "noul",
            "instructions": f"Is the {d} resting tilted, balanced on one of its edges or "
            "corners, instead of lying flat on one of its faces?",
            "criteria": {
                "true": f"the {d} is tilted onto an edge or corner",
                "false": f"the {d} lies flat on a face, or it is held in the gripper",
            },
        },
        "where": {
            "type": "choice",
            "instructions": f"Where is the {d}?",
            "criteria": {
                "in_goal_box": "inside the open green goal box",
                "on_table": "on the table, outside the goal box",
                "in_gripper": "held in the robot's gripper",
                "on_floor": "fallen off the table",
            },
        },
    }


def state_text(description: str, views: list[str]) -> str:
    """What Clef is told the pictures show."""
    lines = [
        f"Camera pictures of a robot arm's work table. On it are a {description} and an open "
        "green box (the goal box). The robot has a two-jaw gripper.",
        "The pictures, in order:",
    ]
    for index, view in enumerate(views, start=1):
        lines.append(f"{index}. {view}: {VIEW_TEXT.get(view, view)}")
    return "\n".join(lines)


@dataclass
class ClefJudge:
    """Asks Clef the success questions about a set of pictures."""

    model: str = "clef"
    post: Callable[..., dict] = _post_json

    def judge(
        self, images: list[bytes], description: str, views: list[str] | None = None
    ) -> dict[str, Any]:
        """P(yes) for each yes/no question, a probability per ``where`` option, token use.

        Args:
            images: PNG pictures, 1 to 4.
            description: The object in words ("red cube").
            views: Each picture's view name, in order (for the state text).
        """
        if not 1 <= len(images) <= MAX_IMAGES:
            raise ValueError(f"give 1 to {MAX_IMAGES} images")
        views = list(views or [f"view {i + 1}" for i in range(len(images))])
        token = load_secret("CLOUDFLARE_API_TOKEN")
        account = load_secret("CLOUDFLARE_ACCOUNT_ID")
        if not token or not account:
            missing = [
                n
                for n, v in (("CLOUDFLARE_API_TOKEN", token), ("CLOUDFLARE_ACCOUNT_ID", account))
                if not v
            ]
            raise JudgeError(f"{' and '.join(missing)} not set (environment or a .env file)")
        body = {
            "model": self.model,
            "state": state_text(description, views),
            "images": [
                "data:image/png;base64," + base64.b64encode(image).decode("ascii") for image in images
            ],
            "questions": questions_for(description),
        }
        secrets = (token, account)
        try:
            reply = self.post(
                RUN_URL.format(account=account, model=self.model),
                body,
                headers={"Authorization": f"Bearer {token}"},
            )
        except PointerError as exc:
            raise JudgeError(mask_secrets(str(exc), secrets)) from None
        if not reply.get("success", True) or "result" not in reply:
            errors = (
                "; ".join(str(e.get("message", e)) for e in reply.get("errors") or [])
                or str(reply)[:300]
            )
            raise JudgeError(mask_secrets(f"Clef refused: {errors}", secrets))
        try:
            answers = reply["result"]["answers"]
            verdict: dict[str, Any] = {q: float(answers[q]["noul"]) for q in NOUL_IDS}
            probabilities = answers["where"]["probabilities"]
            verdict["where"] = {
                option: float(probabilities.get(option, 0.0)) for option in WHERE_OPTIONS
            }
        except (KeyError, TypeError, ValueError):
            raise JudgeError(
                mask_secrets(f"unexpected Clef reply: {str(reply)[:300]}", secrets)
            ) from None
        verdict["input_tokens"] = (reply["result"].get("usage") or {}).get("input_tokens")
        return verdict


# ------------------------------------------------------------------ truth
def tilt_deg(twin, name: str) -> float:
    """How far the object is turned from lying flat on any face (degrees)."""
    obj = twin.grasp.objects[name]
    quat = twin.sim.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7]
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat)
    up = np.abs(matrix.reshape(3, 3)[2, :])  # world z in the object's axes
    return math.degrees(math.acos(min(1.0, float(up.max()))))


def truth_labels(twin, name: str) -> dict[str, Any]:
    """The right answers to the judge's questions, from the simulator."""
    grasp = twin.grasp
    position = grasp.position(name)
    held = grasp.attached == name
    goal = GOALS.get(twin.sim.scene)
    in_goal = bool(
        not held and goal is not None and goal.holds(position, grasp.half_extents_world(name))
    )
    tipped = bool(not held and tilt_deg(twin, name) > TIPPED_DEG)
    table = twin.table_top_z
    off = table is not None and float(position[2]) < table - OFF_TABLE_MARGIN_M
    if held:
        where = "in_gripper"
    elif in_goal:
        where = "in_goal_box"
    elif off:
        where = "on_floor"
    else:
        where = "on_table"
    return {"in_goal": in_goal, "held": held, "tipped": tipped, "where": where}


# ------------------------------------------------------------------ metrics
_EPS = 1e-4


def calibration_bins(probs, truth, n_bins: int = 10) -> list[dict[str, float]]:
    """Predicted probability against how often the answer really was yes, by bin."""
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    truth = np.asarray(truth, dtype=float)
    bins = []
    for index in range(n_bins):
        lo, hi = index / n_bins, (index + 1) / n_bins
        mask = (probs >= lo) & ((probs < hi) if index < n_bins - 1 else (probs <= hi))
        if mask.any():
            bins.append(
                {
                    "lo": round(lo, 3),
                    "hi": round(hi, 3),
                    "n": int(mask.sum()),
                    "predicted": float(probs[mask].mean()),
                    "observed": float(truth[mask].mean()),
                }
            )
    return bins


def noul_metrics(probs, truth, n_bins: int = 10) -> dict[str, float]:
    """Accuracy at 0.5, Brier score, log loss and expected calibration error."""
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(truth, dtype=float)
    n = len(p)
    if n == 0:
        return {"n": 0}
    clipped = np.clip(p, _EPS, 1 - _EPS)
    bins = calibration_bins(p, y, n_bins)
    ece = sum(b["n"] / n * abs(b["predicted"] - b["observed"]) for b in bins)
    return {
        "n": n,
        "positives": int(y.sum()),
        "accuracy": float(((p >= 0.5) == (y >= 0.5)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
        "ece": float(ece),
    }


def where_metrics(probabilities: list[dict[str, float]], truth: list[str]) -> dict[str, Any]:
    """Accuracy of the top option, mean probability on the right one, and a confusion table."""
    n = len(truth)
    if n == 0:
        return {"n": 0}
    confusion = {actual: {chosen: 0 for chosen in WHERE_OPTIONS} for actual in WHERE_OPTIONS}
    right, p_true, brier = 0, 0.0, 0.0
    for probs, actual in zip(probabilities, truth):
        chosen = max(WHERE_OPTIONS, key=lambda option: probs.get(option, 0.0))
        right += chosen == actual
        p_true += probs.get(actual, 0.0)
        brier += sum((probs.get(o, 0.0) - (o == actual)) ** 2 for o in WHERE_OPTIONS)
        confusion[actual][chosen] += 1
    return {
        "n": n,
        "accuracy": right / n,
        "p_true_mean": p_true / n,
        "brier": brier / n,
        "confusion": confusion,
    }

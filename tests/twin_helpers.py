"""Shared steps for twin tests."""

from xbox_soarm_teleop.twin.grasp import GraspSettings


def jaws_around(run, observe, name: str, wrist: float = -1.5) -> None:
    """Put the open jaws around an object the way the agent instructions say.

    The tool point is the fixed jaw's inner face, so line up the observed
    jaw center instead: open a little wider than the object, come over it,
    turn the jaws side to side, center them above it, then go straight down
    to its center height (not below z 0.14).

    Args:
        run: Submits one action request dict and returns once it has finished.
        observe: Returns the current observation dict.
        name: The object to grasp.
        wrist: Wrist roll (rad); -1.5 or 1.5 puts the jaws side to side.
    """

    def obj():
        return observe()["scene"]["objects"][name]

    x, y, z = obj()["xyz_m"]
    run({"type": "move_ee_to", "params": {"xyz_m": [x, y, z + 0.06]}})
    run({"type": "set_wrist_roll", "params": {"angle_rad": wrist}})
    width = obj()["grip_width_m"]
    run(
        {
            "type": "set_gripper",
            "params": {"position": GraspSettings().gripper_for_opening(width + 0.03)},
        }
    )
    for _ in range(2):
        jaw, at = observe()["jaw_center_xyz_m"], obj()["xyz_m"]
        run({"type": "move_ee_by", "params": {"dxyz_m": [at[0] - jaw[0], at[1] - jaw[1], 0.0]}})
    drop = max(obj()["xyz_m"][2], 0.14) - observe()["jaw_center_xyz_m"][2]
    run({"type": "move_ee_by", "params": {"dxyz_m": [0.0, 0.0, drop]}})

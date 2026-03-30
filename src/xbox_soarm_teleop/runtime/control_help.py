"""Shared control-help text for example entry points."""

from __future__ import annotations


def control_help_lines(
    controller_type: str,
    mode: str,
    *,
    use_jacobian: bool = False,
    exit_hint: str = "Ctrl+C          exit",
) -> list[str]:
    """Return human-readable control-help lines for a controller/mode pair."""
    lines = ["Controls:"]

    if controller_type == "keyboard":
        if mode == "joint":
            lines.extend(
                [
                    "  A / D           shoulder_pan    (left / right)",
                    "  W / S           shoulder_lift   (up / down)",
                    "  R / F           elbow_flex      (flex / extend)",
                    "  Q / E           wrist_flex      (up / down)",
                    "  ↑ / ↓           wrist_roll      (+ / -)",
                    "  Space (hold)    gripper close",
                    "  H               home position",
                    "  1–5             speed level  (default 3 = 75%)",
                    "  Shift           2× speed multiplier",
                ]
            )
        else:
            lines.extend(
                [
                    "  W / S           forward / back  (X)",
                    "  A / D           left / right    (Y)",
                    "  R / F           up / down       (Z)",
                    "  Q / E           wrist roll",
                    "  Arrow keys      orientation disabled in touch mode",
                    "  Space (hold)    gripper close",
                    "  H               home position",
                    "  1–5             speed level  (default 3 = 75%)",
                    "  Shift           2× speed multiplier",
                ]
            )
    elif controller_type == "joycon":
        if mode == "joint":
            lines.extend(
                [
                    "  Stick left/right    drive selected joint",
                    "  (no joint cycle on Joy-Con — use cartesian mode)",
                ]
            )
        elif mode == "puppet":
            lines.extend(
                [
                    "  Stick left/right    shoulder_pan  (base rotation)",
                    "  Stick up/down       reach         (extend/retract)",
                    "  SR (hold)           height up",
                    "  B face button       height down",
                    "  IMU pitch           wrist_flex    (tilt fwd/back)",
                    "  IMU roll            wrist_roll    (tilt left/right)",
                    "  ZR                  gripper (hold=close)",
                ]
            )
        else:
            lines.extend(
                [
                    "  Stick               move arm (X/Y/Z/roll)",
                    "  ZR                  gripper (hold=close)",
                ]
            )
        lines.extend(
            [
                "  SL (hold)           deadman switch",
                "  + button            home position",
            ]
        )
    else:  # xbox
        if mode == "joint":
            lines.extend(
                [
                    "  Hold LB + Left stick X    drive selected joint",
                    "  D-pad left/right          cycle joint",
                    "  Right trigger             gripper",
                    "  A button                  home position",
                ]
            )
        elif mode == "crane":
            lines.extend(
                [
                    "  Hold LB + move sticks     control arm",
                    "  Left stick X/Y            left-right / forward-back",
                    "  Right stick Y/X           up-down / wrist roll",
                    "  D-pad up/down             wrist up / down",
                    "  Right trigger             gripper",
                    "  A button                  neutral crane pose",
                ]
            )
        else:
            lines.extend(
                [
                    "  Hold LB + move sticks     control arm",
                    "  Left stick X/Y            left-right / forward-back",
                    "  Right stick Y/X           up-down / wrist roll",
                    "  D-pad                   orientation disabled in touch mode",
                    "  Right trigger             gripper",
                    "  A button                  home position",
                ]
            )

    lines.append(f"  {exit_hint}")
    return lines


def print_controls(
    controller_type: str,
    mode: str,
    *,
    use_jacobian: bool = False,
    exit_hint: str = "Ctrl+C          exit",
) -> None:
    """Print human-readable control-help lines for a controller/mode pair."""
    print()
    for line in control_help_lines(
        controller_type,
        mode,
        use_jacobian=use_jacobian,
        exit_hint=exit_hint,
    ):
        print(line, flush=True)
    print(flush=True)

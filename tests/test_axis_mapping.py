"""Tests for Cartesian axis mapping helpers."""

from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, apply_axis_mapping


def test_apply_axis_mapping_swap_xy():
    delta = EEDelta(
        dx=1.0,
        dy=2.0,
        dz=3.0,
        droll=4.0,
        dpitch=5.0,
        dyaw=6.0,
        gripper=0.7,
    )
    mapped = apply_axis_mapping(delta, swap_xy=True)
    assert mapped.dx == 2.0
    assert mapped.dy == 1.0
    assert mapped.dz == 3.0
    assert mapped.droll == 4.0
    assert mapped.dpitch == 5.0
    assert mapped.dyaw == 6.0
    assert mapped.gripper == 0.7


def test_apply_axis_mapping_no_swap():
    delta = EEDelta(dx=0.1, dy=-0.2, dz=0.3)
    mapped = apply_axis_mapping(delta, swap_xy=False)
    assert mapped.dx == delta.dx
    assert mapped.dy == delta.dy
    assert mapped.dz == delta.dz

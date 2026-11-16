"""The bean bag: soft to the grip, but solid to the arm.

It used to be a box the arm passed straight through (so it "gave way"), which
meant a push could not move it. Now the arm touches it: a push slides it, the
arm's own picks are as good as before (measured: 9 of 9 at the first attempt
with an agent's aim, both ways), and the grip still treats it as soft (any
angle, a looser fit).
"""

import threading
import time

import mujoco

from xbox_soarm_teleop.twin import build_twin


def test_the_arm_and_the_bag_touch_and_the_grip_still_treats_it_as_soft():
    service = build_twin(scene="pick_place_easy", with_controller=False)
    try:
        model = service.sim.model
        bag = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bean_bag_geom")
        body = model.geom_bodyid[bag]
        assert model.geom_contype[bag] & 1 and model.geom_conaffinity[bag] & 1  # the arm's bit
        assert model.body_contype[body] & 1 and model.body_conaffinity[body] & 1
        assert service.grasp.objects["bean_bag"].soft is True
        assert service.grasp.objects["pick_cube"].soft is False
    finally:
        service.close()


def test_a_push_slides_the_bag_and_a_pick_still_takes_it():
    service = build_twin(scene="pick_place_easy", with_controller=False)
    alive = threading.Event()
    alive.set()

    def loop():
        try:
            while alive.is_set():
                service.tick()
        finally:
            service.close()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        time.sleep(0.3)
        start = service.grasp.position("bean_bag").copy()
        pushed = service.push("bean_bag", direction="forward", distance_m=0.05, client_id="test")
        moved = service.grasp.position("bean_bag")[:2] - start[:2]
        assert pushed["ok"] is True, pushed
        assert 0.03 <= moved[0] <= 0.10 and abs(moved[1]) < 0.02, moved
        picked = service.pick("bean_bag", client_id="test")
        assert picked["ok"] is True and picked["attempts"] == 1, picked
    finally:
        alive.clear()
        thread.join(timeout=10)

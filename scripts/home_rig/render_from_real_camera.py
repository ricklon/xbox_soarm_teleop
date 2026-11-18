"""Render the twin from the real overhead camera's pose, beside the real frame."""

import json
import sys

import cv2
import mujoco
import numpy as np

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli

cam = json.load(open(sys.argv[1]))
real = cv2.imread(sys.argv[2])
joints_deg = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
spec = sim_cli._robot_spec(str(sim_cli.URDF_PATH))
sim_cli.apply_world(spec, "studio")
spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
c = spec.worldbody.add_camera()
c.name = "real_overhead"
c.pos = cam["pos_m"]
c.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
c.alt.xyaxes = cam["xyaxes"]
c.fovy = cam["fovy_deg"]
# Mark the desk: a 5 cm grid of thin lines out to 40 cm, and the turning axis.
for i in range(-8, 9):
    for a, b in (((i * 0.05, -0.4), (i * 0.05, 0.4)), ((-0.4, i * 0.05), (0.4, i * 0.05))):
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.001)
        half = (max(abs(b[0] - a[0]) / 2, 0.0008), max(abs(b[1] - a[1]) / 2, 0.0008), 0.0005)
        g.pos, g.size, g.rgba = list(mid), list(half), [1, 0, 1, 1] if i == 0 else [0.2, 0.2, 0.2, 1]
        g.contype = g.conaffinity = 0
model = spec.compile()
data = mujoco.MjData(model)
if joints_deg is not None:
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for n, v in zip(names, joints_deg):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        data.qpos[model.jnt_qposadr[j]] = np.radians(v)
mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 1080, 1920)
r.update_scene(data, camera="real_overhead")
twin = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
r.close()
blend = cv2.addWeighted(real, 0.55, twin, 0.45, 0)
row = cv2.hconcat([cv2.resize(real, (960, 540)), cv2.resize(twin, (960, 540))])
out = cv2.vconcat(
    [row, cv2.hconcat([cv2.resize(blend, (960, 540)), np.full((540, 960, 3), 255, np.uint8)])]
)
cv2.putText(out, "real", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(out, "twin from the same pose", (970, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(out, "both blended", (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite(sys.argv[3], out, [cv2.IMWRITE_JPEG_QUALITY, 85])

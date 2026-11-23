# Home rig: matching the twin's cameras to the real arm

Tools for measuring the real SO-101's cameras so the twin can render the same
views. The home rig is the arm on a desk with a self-healing cutting mat (1-inch
grid, inch rulers), an overhead Logitech C920 and a wide fisheye wrist camera.

Frames go in `captures/`, which is gitignored: they are photos of a home.

Camera access on macOS: run these in a terminal that has been allowed to use the
camera (the app's own command runner may not be).

| Script | What it does |
|---|---|
| `read_arm.py PORT` | Reads the six joints, torque state, voltage and temperature. Writes nothing to the motors. |
| `../capture_frames.py --camera N --out DIR [--once]` | Saves frames from one camera (Enter per frame, or one with `--once`). |
| `pose_page.py CAMERA PORT DIR` | A page at http://127.0.0.1:8770 with the live camera, joint angles and a Capture button: each capture saves a full frame and the joints. Read-only on the arm. |
| `mat_grid_lines.py FRAME` | Finds the mat's 1-inch grid lines in an overhead frame (writes `lines.npz` and an overlay). |
| `mat_calibrate.py FRAME 15:x,y 1:x,y HEIGHT_IN` | Numbers the lines from two ruler labels (pixel positions of the bottom ruler's 15 and the side ruler's 1), then fits the overhead camera's lens and pose, holding its height at the tape-measured value. |
| `render_from_real_camera.py CAMERA.json FRAME OUT [JOINTS_DEG]` | Renders the twin from the measured camera pose beside the real frame, with the arm optionally posed. |

## What was measured (2026-11-18)

- **Overhead C920:** 68.9 x 42.2 degree view at 1920 x 1080 (focal length 1400 px),
  slight barrel distortion (k1 -0.02), 27 in above the mat, 7.3 degrees from straight
  down. In the arm's frame (x forward, y left, z up, origin under the base's
  turning axis): at (0.114, -0.139, 0.686) m. Grid fit: median error 1.8 px.
  One near-straight-down view cannot separate focal length from height; the
  height comes from a tape measure (the free fit said 32 in).
- **Arm base:** turning axis at mat X 10.25 in (the base spans 8 to 12.5 in on the
  bottom ruler), Y 2.0 in (the base's front edge at the 3 in line; the axis is
  26 mm behind it in the robot model). Assumed square to the grid.
- **Check:** the twin rendered from that pose, folded to the real joint readings
  (degrees from each encoder's middle), lies over the real arm in the overhead
  frame; the wrist is about 2 cm off.
- **Wrist camera:** its board appears to read "...1080P...L170", so likely a 170
  degree fisheye, not the 100 degree camera the twin models. 11 posed frames of the
  mat with joint readings are saved; calibrating it needs a grid finder that
  follows the fisheye's curved lines (not written yet).

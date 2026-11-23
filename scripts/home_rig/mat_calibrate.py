"""From the mat's grid lines (mat_grid_lines.py) to the overhead camera's lens and pose.

Each family's lines are kept only where they fit an evenly spaced (allowing for
perspective) whole-inch pattern; they are numbered from two ruler labels read
off the frame; crossings become (X, Y) inch points; OpenCV calibrates from them.

Usage: mat_calib.py FRAME X_LABEL_PIXEL Y_LABEL_PIXEL HEIGHT_IN
  X_LABEL_PIXEL: "15:x,y", where the bottom ruler's number 15 is printed
  Y_LABEL_PIXEL: "1:x,y", where the right ruler's number 1 is printed
"""

import sys
from pathlib import Path

import cv2
import numpy as np

frame = Path(sys.argv[1])
x_label, x_pix = sys.argv[2].split(":")
y_label, y_pix = sys.argv[3].split(":")
x_label, y_label = int(x_label), int(y_label)
x_pix = np.array([float(v) for v in x_pix.split(",")])
y_pix = np.array([float(v) for v in y_pix.split(",")])
height_in = float(sys.argv[4])

data = np.load(frame.parent / "lines.npz")
image = cv2.imread(str(frame))
H, W = image.shape[:2]


def lattice(fam: np.ndarray, spacing: float = 52.0):
    """Keep lines that fit offset = a + b k + c k^2 for whole k; return {k: line}."""
    offsets = fam[:, 0]
    best = None
    for ref in range(len(offsets)):
        k = np.round((offsets - offsets[ref]) / spacing)
        resid = offsets - (offsets[ref] + k * spacing)
        inl = np.abs(resid) < 8
        # One line per k: the closest.
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), ref)
    ref = best[1]
    k = np.round((offsets - offsets[ref]) / spacing)
    keep = np.abs(offsets - (offsets[ref] + k * spacing)) < 8
    for _ in range(3):  # refine with a quadratic, which allows for perspective
        coeff = np.polyfit(k[keep], offsets[keep], 2)
        pred = np.polyval(coeff, k)
        keep = np.abs(offsets - pred) < 7
    chosen = {}
    for i in np.flatnonzero(keep):
        kk = int(k[i])
        if kk not in chosen or abs(offsets[i] - np.polyval(coeff, kk)) < abs(
            offsets[chosen[kk]] - np.polyval(coeff, kk)
        ):
            chosen[kk] = i
    return {kk: fam[i] for kk, i in chosen.items()}, coeff


def distance(line, point) -> float:
    _, vx, vy, x0, y0, _ = line
    return abs((point[0] - x0) * vy - (point[1] - y0) * vx)


fam_a, _ = lattice(data["fam1"])
fam_b, _ = lattice(data["fam2"])
# Which family runs across (lines of constant Y) and which runs up (constant X)?
# A constant-X line passes through the bottom label's number and is steep.
steep_a = abs(next(iter(fam_a.values()))[2]) > 0.7
x_lines, y_lines = (fam_a, fam_b) if steep_a else (fam_b, fam_a)
kx = min(x_lines, key=lambda k: distance(x_lines[k], x_pix))
ky = min(y_lines, key=lambda k: distance(y_lines[k], y_pix))
print(f"line through the '{x_label}' label: index {kx} (distance {distance(x_lines[kx], x_pix):.1f}px)")
print(f"line through the '{y_label}' label: index {ky} (distance {distance(y_lines[ky], y_pix):.1f}px)")


def label(lines, k_ref, value_ref, point_ref):
    """Map lattice index to mat inches; the direction is set by which way the values increase."""
    ks = sorted(lines)
    return {k: k for k in ks}, k_ref, value_ref


# Numbering direction: X increases to the right along the bottom ruler, Y increases up the
# right ruler (toward the top of the image). Decide the sign from the lines' positions.
def signed(lines, k_ref, value_ref, increase_dir):
    ks = sorted(lines)
    c = lambda k: np.array(lines[k][3:5])  # a point on the line  # noqa: E731
    if len(ks) > 1:
        step = c(ks[-1]) - c(ks[0])
        sign = 1 if step @ increase_dir > 0 else -1
    else:
        sign = 1
    return {k: value_ref + sign * (k - k_ref) for k in ks}


x_value = signed(x_lines, kx, x_label, np.array([1.0, 0.0]))
y_value = signed(y_lines, ky, y_label, np.array([0.0, -1.0]))


def intersect(l1, l2):
    _, vx1, vy1, x1, y1, _ = l1
    _, vx2, vy2, x2, y2, _ = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1]))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1])


obj, img = [], []
for kxl, lx in x_lines.items():
    for kyl, ly in y_lines.items():
        X, Y = x_value[kxl], y_value[kyl]
        if not (0 <= X <= 23 and 0 <= Y <= 17):
            continue  # off the mat's printed grid
        p = intersect(lx, ly)
        if 5 <= p[0] < W - 5 and 5 <= p[1] < H - 5:
            obj.append((X * 25.4, Y * 25.4, 0.0))
            img.append(p)
obj = np.array(obj, np.float32)
img = np.array(img, np.float32)
print(f"X lines {sorted(x_value.values())}")
print(f"Y lines {sorted(y_value.values())}")
print(f"{len(obj)} grid crossings on the mat")

K0 = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    | cv2.CALIB_FIX_PRINCIPAL_POINT
    | cv2.CALIB_FIX_ASPECT_RATIO
    | cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K3
)
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj], [img], (W, H), K0.copy(), None, flags=flags)
R, _ = cv2.Rodrigues(rvecs[0])
t = tvecs[0].ravel()
cam = (-R.T @ t) / 25.4  # inches, mat frame (X along bottom ruler, Y up the right ruler, Z = +/-)
axis = R.T @ np.array([0, 0, 1.0])
f = K[0, 0]
print(
    f"single-view fit: reprojection rms {rms:.2f}px, focal {f:.0f}px, k1 {dist.ravel()[0]:+.3f}, k2 {dist.ravel()[1]:+.3f}"
)
print(
    f"  -> camera at X {cam[0]:.2f} in, Y {cam[1]:.2f} in, {abs(cam[2]):.2f} in above the mat (you measured {height_in})"
)
print(
    f"  -> field of view {2 * np.degrees(np.arctan(W / 2 / f)):.1f} x {2 * np.degrees(np.arctan(H / 2 / f)):.1f} deg, "
    f"tilt from straight down {np.degrees(np.arccos(abs(axis[2]))):.1f} deg"
)

# A single near-straight-down view cannot separate focal length from height well:
# fix the height to the tape measure and fit focal length again for comparison.
scale_f = f * height_in / abs(cam[2])
print(
    f"  focal length that matches the tape-measured {height_in} in: about {scale_f:.0f}px "
    f"(FOV {2 * np.degrees(np.arctan(W / 2 / scale_f)):.1f} x {2 * np.degrees(np.arctan(H / 2 / scale_f)):.1f} deg)"
)
f_fixed = scale_f
for _ in range(6):
    Kf = np.array([[f_fixed, 0, W / 2], [0, f_fixed, H / 2], [0, 0, 1.0]])
    fixed_flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_FOCAL_LENGTH
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj], [img], (W, H), Kf.copy(), np.zeros(5), flags=fixed_flags
    )
    R, _ = cv2.Rodrigues(rvecs[0])
    cam = (-R.T @ tvecs[0].ravel()) / 25.4
    f_fixed *= height_in / abs(cam[2])
axis = R.T @ np.array([0, 0, 1.0])
f = K[0, 0]
print(
    f"height held at {height_in} in: rms {rms:.2f}px, focal {f:.0f}px, k1 {dist.ravel()[0]:+.3f}; "
    f"camera at X {cam[0]:.2f}, Y {cam[1]:.2f}, {abs(cam[2]):.2f} in up; "
    f"FOV {2 * np.degrees(np.arctan(W / 2 / f)):.1f} x {2 * np.degrees(np.arctan(H / 2 / f)):.1f} deg; "
    f"tilt {np.degrees(np.arccos(abs(axis[2]))):.1f} deg"
)
np.savez(
    frame.parent / "overhead_calib.npz",
    K=K,
    dist=dist,
    rvec=rvecs[0],
    tvec=tvecs[0],
    obj=obj,
    img=img,
    rms=rms,
)

proj, _ = cv2.projectPoints(obj, rvecs[0], tvecs[0], K, dist)
err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
print(
    f"per-point error: median {np.median(err):.2f}px, 95th percentile {np.percentile(err, 95):.2f}px, max {err.max():.2f}px"
)

overlay = image.copy()
for (X, Y, _), p, q in zip(obj / 25.4, img, proj.reshape(-1, 2)):
    cv2.circle(overlay, tuple(map(int, p)), 4, (0, 255, 0), -1)
    cv2.circle(overlay, tuple(map(int, q)), 7, (255, 0, 255), 1)
    if X % 5 == 0 and Y % 5 == 0:
        cv2.putText(
            overlay,
            f"{int(X)},{int(Y)}",
            (int(p[0]) + 6, int(p[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
cv2.imwrite(
    str(frame.parent / "calib_check.jpg"),
    cv2.resize(overlay, (1440, 810)),
    [cv2.IMWRITE_JPEG_QUALITY, 88],
)

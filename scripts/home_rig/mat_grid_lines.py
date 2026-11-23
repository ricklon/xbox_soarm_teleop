"""Find the cutting mat's 1-inch grid lines in an overhead frame.

Long straight segments are sorted into the grid's two directions; within each
direction, segments are grouped by their offset across that direction into
lines; each line is fitted straight; crossings of the two families are the
grid points. Writes lines.npz (both families) and an overlay for checking.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

path = Path(sys.argv[1])
image = cv2.imread(str(path))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
segments = lsd.detect(gray)[0].reshape(-1, 4)
length = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
segments, length = segments[length > 45], length[length > 45]
angle = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])) % 180

# The two grid directions: the strongest angle (weighted by length), and the one 90 degrees from it.
hist, edges = np.histogram(angle, bins=180, range=(0, 180), weights=length)
smooth = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7), "valid")
a1 = float(np.argmax(smooth)) + 0.5
a2 = (a1 + 90) % 180
print(f"{len(segments)} long segments; grid directions about {a1:.1f} and {a2:.1f} deg")


def family(theta: float, tol: float = 4.0):
    diff = np.abs((angle - theta + 90) % 180 - 90)
    pick = diff < tol
    seg = segments[pick]
    direction = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    normal = np.array([-direction[1], direction[0]])
    mids = (seg[:, :2] + seg[:, 2:]) / 2
    offset = mids @ normal
    order = np.argsort(offset)
    seg, offset = seg[order], offset[order]
    # Group segments whose offsets are within 6 px into one line.
    groups, current = [], [0]
    for i in range(1, len(offset)):
        if offset[i] - offset[current[-1]] < 6:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    lines = []
    for g in groups:
        pts = np.vstack([seg[g, :2], seg[g, 2:]])
        span = np.ptp(pts @ direction)
        if span < 150:  # too short to be a grid line across the mat
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append(
            (float(np.mean(offset[g])), float(vx), float(vy), float(x0), float(y0), float(span))
        )
    return np.array(lines), normal


fam1, n1 = family(a1)
fam2, n2 = family(a2)
for name, fam in (("family A", fam1), ("family B", fam2)):
    gaps = np.diff(fam[:, 0])
    print(
        f"{name}: {len(fam)} lines; gaps between neighbours (px): median {np.median(gaps):.1f}, "
        f"min {gaps.min():.1f}, max {gaps.max():.1f}"
    )
np.savez(path.parent / "lines.npz", fam1=fam1, fam2=fam2, a1=a1, a2=a2)

overlay = image.copy()
for fam, colour in ((fam1, (255, 0, 255)), (fam2, (0, 200, 255))):
    for _, vx, vy, x0, y0, _ in fam:
        p1 = (int(x0 - vx * 3000), int(y0 - vy * 3000))
        p2 = (int(x0 + vx * 3000), int(y0 + vy * 3000))
        cv2.line(overlay, p1, p2, colour, 1)
cv2.imwrite(
    str(path.parent / "lines.jpg"), cv2.resize(overlay, (1440, 810)), [cv2.IMWRITE_JPEG_QUALITY, 88]
)

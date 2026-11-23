"""Save frames from one camera on Enter, saying whether the checkerboard is in them.

For calibrating the home rig's cameras (assets/printables/checkerboard_9x6_25mm.pdf).
Run it in a terminal that is allowed to use the camera:

    uv run python scripts/capture_frames.py --camera 0 --out captures/overhead_calib
    uv run python scripts/capture_frames.py --camera 1 --out captures/wrist_calib

Press Enter to save a frame (it reports whether the board's 9 x 6 inner corners
were found), type q and Enter to stop. macOS can renumber cameras after a
replug: the first frame is saved as preview.png so you can check which one it is.

With --once it saves one frame after the camera has settled and exits (for
taking shots from a script, or of the cutting mat, with no board in view).
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2

PATTERN = (9, 6)  # inner corners of assets/printables/checkerboard_9x6_25mm.pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, required=True, help="OpenCV camera index")
    parser.add_argument("--out", type=Path, required=True, help="folder for the frames")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--once", action="store_true", help="save one frame and exit")
    parser.add_argument(
        "--settle", type=float, default=2.0, help="seconds to let exposure settle (--once)"
    )
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise SystemExit(
            f"camera {args.camera} did not open (is this terminal allowed to use the camera?)"
        )

    latest = {"frame": None}
    running = threading.Event()
    running.set()

    def reader() -> None:  # keep the newest frame, so a save is never a stale buffered one
        while running.is_set():
            ok, frame = capture.read()
            if ok:
                latest["frame"] = frame

    threading.Thread(target=reader, daemon=True).start()
    deadline = time.monotonic() + 5.0
    while latest["frame"] is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if latest["frame"] is None:
        raise SystemExit(f"camera {args.camera} gave no frames")

    args.out.mkdir(parents=True, exist_ok=True)
    first = latest["frame"]
    cv2.imwrite(str(args.out / "preview.png"), first)
    print(
        f"camera {args.camera}: {first.shape[1]}x{first.shape[0]}; check {args.out / 'preview.png'} is the right one"
    )

    count = len(list(args.out.glob("frame_*.png")))
    if args.once:
        time.sleep(args.settle)  # auto-exposure and focus settle after opening
        path = args.out / f"frame_{count + 1:03d}.png"
        cv2.imwrite(str(path), latest["frame"])
        running.clear()
        capture.release()
        print(f"saved {path}")
        return
    found_total = 0
    try:
        while True:
            line = input("Enter to save a frame, q to stop: ").strip().lower()
            if line == "q":
                break
            frame = latest["frame"].copy()
            count += 1
            path = args.out / f"frame_{count:03d}.png"
            cv2.imwrite(str(path), frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            found, _ = cv2.findChessboardCornersSB(gray, PATTERN, flags)
            found_total += int(found)
            verdict = "board found" if found else "board NOT found (all of it must be in view)"
            print(f"  saved {path.name}: {verdict}; {found_total} good so far")
    finally:
        running.clear()
        capture.release()


if __name__ == "__main__":
    main()

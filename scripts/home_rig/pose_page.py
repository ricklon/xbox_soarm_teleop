"""A page for taking wrist-calibration poses: live wrist view, joint angles, Capture button.

Read-only on the arm (no writes; torque left as found). Run from the repo root in a
terminal allowed to use the camera, then open http://127.0.0.1:8770 :
    uv run python scripts/home_rig/pose_page.py CAMERA_INDEX PORT OUT_DIR
Capture (or the space bar) saves the full-size frame and the six joint readings.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

camera, port, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

bus = SerialMotorsBus(
    port, {m: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, m in enumerate(NAMES, 1)}
)
bus.connect(handshake=True)
bus_lock = threading.Lock()
cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
state = {"frame": None, "jpeg": None, "joints": {}, "count": len(list(out.glob("pose_*.png")))}
running = True


def camera_loop() -> None:
    while running:
        ok, frame = cap.read()
        if ok:
            state["frame"] = frame
            small = cv2.resize(frame, (960, 540))
            state["jpeg"] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def joints_loop() -> None:
    while running:
        try:
            with bus_lock:
                raw = bus.sync_read("Present_Position", normalize=False)
            state["joints"] = raw
        except Exception:  # noqa: BLE001 - a missed read is retried
            pass
        time.sleep(0.25)


def deg(raw: dict) -> dict:
    return {k: round((v - 2048) * 360 / 4096, 1) for k, v in raw.items()}


PAGE = """<!doctype html><meta charset=utf-8><title>Wrist poses</title>
<style>body{font:16px system-ui;margin:12px;background:#111;color:#eee}
img{width:100%;max-width:960px;border:2px solid #444;display:block}
button{font-size:28px;padding:14px 40px;margin:12px 0;background:#2a7;color:#fff;border:0;border-radius:8px}
#msg{font-size:22px;margin:6px 0}#joints{font-family:ui-monospace,monospace;color:#aaa}</style>
<h2 style=margin:4px>Wrist camera: pose the gripper over the mat, then Capture (or press space)</h2>
<img src=/stream><button id=b onclick=snap()>Capture</button>
<div id=msg>0 saved</div><div id=joints></div>
<script>
async function snap(){b.disabled=true;msg.textContent='saving...';
 const r=await (await fetch('/capture',{method:'POST'})).json();
 msg.textContent=r.count+' saved (pose '+r.count+')';b.disabled=false}
document.onkeydown=e=>{if(e.code==='Space'){e.preventDefault();snap()}};
setInterval(async()=>{const s=await (await fetch('/status')).json();
 joints.textContent=Object.entries(s.joints).map(([k,v])=>k+' '+v).join('   ');
 if(!b.disabled)msg.textContent=s.count+' saved'},500)
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/status":
            self._json({"count": state["count"], "joints": deg(state["joints"])})
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while running:
                    jpeg = state["jpeg"]
                    if jpeg:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/capture":
            return self.send_error(404)
        frame = state["frame"]
        with bus_lock:
            raw = bus.sync_read("Present_Position", normalize=False)
        n = len(list(out.glob("pose_*.png"))) + 1
        cv2.imwrite(str(out / f"pose_{n:02d}.png"), frame)
        record = {
            "pose": n,
            "camera": camera,
            "size": [frame.shape[1], frame.shape[0]],
            "raw": raw,
            "deg_from_middle": deg(raw),
        }
        (out / f"pose_{n:02d}.json").write_text(json.dumps(record, indent=1))
        state["count"] = n
        print(
            f"pose {n}: " + ", ".join(f"{k} {v:+.0f}" for k, v in record["deg_from_middle"].items()),
            flush=True,
        )
        self._json({"count": n, "joints": record["deg_from_middle"]})


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=joints_loop, daemon=True).start()
server = ThreadingHTTPServer(("127.0.0.1", 8770), Handler)
print("open http://127.0.0.1:8770", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    running = False
    cap.release()
    with bus_lock:
        bus.disconnect(disable_torque=False)

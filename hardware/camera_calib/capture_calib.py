#!/usr/bin/env python3
"""캘리브레이션 사진 촬영 (Raspberry Pi HQ 카메라, 화면 없이 브라우저로 미리보기).

- 브라우저 http://<Pi IP>:8000 에서 저화질 미리보기를 보며 [촬영] 버튼으로 찍는다.
- 터미널에서 Enter 를 눌러도 찍힌다. 종료는 Ctrl+C.
- 사진은 센서 전체 4056x3040 흑백 PNG 로 저장한다 (캘리브레이션은 전체 화각 기준).

사용: python3 capture_calib.py [-o captures] [--exposure 10000] [--gain 1.0]
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from http import server
from socketserver import ThreadingMixIn

from PIL import Image
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

PORT = 8000
FULL = (4056, 3040)    # 센서 전체 화각
PREVIEW = (640, 480)   # 브라우저 전송용
PREVIEW_Q = 50

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Calib Capture</title>
<style>body{margin:0;background:#111;color:#eee;font:16px sans-serif;text-align:center}
img{width:100%;max-width:960px}#info{font-size:22px;margin:8px}
button{font-size:22px;padding:8px 28px;margin:8px}</style></head>
<body><div id="info">-</div><button onclick="cap()">촬영</button><br><img src="stream.mjpg">
<script>
async function cap() {
  const r = await (await fetch('capture')).json();
  document.getElementById('info').textContent = '저장: ' + r.file + ' (총 ' + r.count + '장)';
}
setInterval(async () => {
  const m = await (await fetch('meta')).json();
  if (!document.getElementById('info').textContent.startsWith('저장'))
    document.getElementById('info').textContent =
      `${m.count}장 · 선명도 ${m.fom} · 노출 ${m.exp} ms · 게인 ${m.gain}`;
}, 500);
</script></body></html>"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class Capturer:
    def __init__(self, picam2, outdir):
        self.picam2 = picam2
        self.outdir = outdir
        self.lock = threading.Lock()
        self.count = len([f for f in os.listdir(outdir) if f.endswith(".png")])
        self.meta = {"fom": 0, "exp": 0, "gain": 0}
        self._win = None

    def on_frame(self, request):
        md = request.get_metadata()
        self.meta = {"fom": md.get("FocusFoM", 0),
                     "exp": round(md.get("ExposureTime", 0) / 1000, 1),
                     "gain": round(md.get("AnalogueGain", 0), 2)}

    def _gray(self):
        request = self.picam2.capture_request()
        try:
            rgb = request.make_array("main")
            md = request.get_metadata()
        finally:
            request.release()
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), md

    def capture(self, still_px=0.3, timeout_s=4.0):
        """멈춤 확인: 연속 프레임 이동량이 still_px(원본 px) 미만으로 2번 연속일 때만 저장.
        (손으로 든 보드가 움직이면 롤링 셔터 때문에 사진이 휘어 칼리브레이션 오차가 커진다)"""
        with self.lock:
            t0 = time.monotonic()
            prev, still, shift, frames = None, 0, float("nan"), []
            while time.monotonic() - t0 < timeout_s:
                gray, md = self._gray()
                small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
                if prev is not None:
                    if self._win is None:
                        self._win = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
                    (dx, dy), _ = cv2.phaseCorrelate(prev, small, self._win)
                    shift = float(np.hypot(dx, dy) * 4)
                    still = still + 1 if shift < still_px else 0
                    frames = frames[-2:] + [gray] if still else [gray]
                    if still >= 2:
                        break
                else:
                    frames = [gray]
                prev = small
            else:
                print(f"[ - ] 흔들림 {shift:.2f}px → 저장 안 함 (보드를 멈추고 다시)", flush=True)
                return None, shift
            # 멈춘 3장 평균 (노이즈 감소)
            img = np.clip(np.mean(frames, axis=0) + 0.5, 0, 255).astype(np.uint8)
            self.count += 1
            name = f"calib_{self.count:03d}"
            path = os.path.join(self.outdir, name + ".png")
            cv2.imwrite(path, img)
            keep = ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux", "FocusFoM",
                    "ColourGains", "SensorTemperature", "ScalerCrop")
            info = {k: md[k] for k in keep if k in md}
            info["still_shift_px"] = round(shift, 3)
            info["averaged_frames"] = len(frames)
            with open(os.path.join(self.outdir, name + ".json"), "w") as f:
                json.dump(info, f, default=list, indent=1)
            print(f"[{self.count:3d}] {path}  멈춤 {shift:.2f}px  {time.monotonic() - t0:.1f}s", flush=True)
            return path, shift


def make_handler(cap, output):
    class Handler(server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_body(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/meta":
                self.send_body(json.dumps({**cap.meta, "count": cap.count}).encode(),
                               "application/json")
            elif self.path == "/capture":
                path, shift = cap.capture()
                name = os.path.basename(path) if path else f"흔들림 {shift:.2f}px - 저장 안 함, 보드를 멈추고 다시"
                self.send_body(json.dumps({"file": name, "count": cap.count}).encode(), "application/json")
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while True:
                        with output.condition:
                            output.condition.wait()
                            frame = output.frame
                        self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


class StreamingServer(ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="captures")
    ap.add_argument("--exposure", type=int, default=None, help="노출 시간 us (생략 시 자동)")
    ap.add_argument("--gain", type=float, default=None, help="아날로그 게인 (생략 시 자동)")
    ap.add_argument("--flicker", type=int, default=8333,
                    help="조명 깜빡임 주기 us (한국 60Hz 전원 = 8333, 0 = 끔)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    controls = {"FrameRate": 10}
    if args.flicker:
        # 자동 노출이 노출 시간을 깜빡임 주기의 배수로 맞춰 가로 줄무늬를 없앤다
        controls.update({"AeFlickerMode": 1, "AeFlickerPeriod": args.flicker})
    if args.exposure is not None:
        if args.flicker and args.exposure % args.flicker > 50:
            print(f"주의: 노출 {args.exposure}us 가 {args.flicker}us 의 배수가 아니라 줄무늬가 생길 수 있음",
                  file=sys.stderr)
        controls["ExposureTime"] = args.exposure
    if args.gain is not None:
        controls["AnalogueGain"] = args.gain

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": FULL, "format": "RGB888"},
        lores={"size": PREVIEW, "format": "YUV420"},
        sensor={"output_size": FULL}, controls=controls, buffer_count=4)
    picam2.configure(config)

    cap = Capturer(picam2, args.outdir)
    picam2.post_callback = cap.on_frame
    output = StreamingOutput()
    picam2.start_encoder(JpegEncoder(q=PREVIEW_Q), FileOutput(output), name="lores")
    picam2.start()

    def stdin_loop():
        for _ in sys.stdin:
            cap.capture()

    threading.Thread(target=stdin_loop, daemon=True).start()
    print(f"미리보기: http://<Pi IP>:{PORT}   저장 폴더: {args.outdir}  (Enter = 촬영, Ctrl+C = 종료)")
    try:
        StreamingServer(("", PORT), make_handler(cap, output)).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop_encoder()
        picam2.stop()


if __name__ == "__main__":
    main()

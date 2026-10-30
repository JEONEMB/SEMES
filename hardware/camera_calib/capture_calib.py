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

    def on_frame(self, request):
        md = request.get_metadata()
        self.meta = {"fom": md.get("FocusFoM", 0),
                     "exp": round(md.get("ExposureTime", 0) / 1000, 1),
                     "gain": round(md.get("AnalogueGain", 0), 2)}

    def capture(self):
        with self.lock:
            request = self.picam2.capture_request()
            try:
                rgb = request.make_array("main")
                md = request.get_metadata()
            finally:
                request.release()
            self.count += 1
            name = f"calib_{self.count:03d}"
            path = os.path.join(self.outdir, name + ".png")
            # main 스트림은 BGR 순서로 들어오므로 채널을 뒤집어 RGB 로 만든 뒤 흑백 변환
            Image.fromarray(rgb[:, :, ::-1]).convert("L").save(path)
            keep = ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux", "FocusFoM",
                    "ColourGains", "SensorTemperature", "ScalerCrop")
            with open(os.path.join(self.outdir, name + ".json"), "w") as f:
                json.dump({k: md[k] for k in keep if k in md}, f, default=list, indent=1)
            print(f"[{self.count:3d}] {path}  선명도 {md.get('FocusFoM')}", flush=True)
            return path


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
                path = cap.capture()
                self.send_body(json.dumps({"file": os.path.basename(path),
                                           "count": cap.count}).encode(), "application/json")
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

"""Unity 시뮬레이터용 라즈베리파이 중계 (임시 틀, 월요일 현장 첫 시험)

파이에서 저장소 루트(/home/pi/S15P31A507)에서 demo_web 가상환경으로 실행한다:
    demo_web/.venv/bin/python <이 파일> --host 0.0.0.0
verify_web 의 카메라·변환 코드를 그대로 불러 쓴다 (verify_web/camera.py, verify_web/geometry.py — 수정하지 않음).

제공하는 것
  1) STM32 시리얼 ↔ TCP 바이트 중계 (기본 5555): Unity 가 robot_stage 프레임을 그대로 주고받음
  2) 카메라 HTTP (기본 8091)
     GET  /health                 카메라 상태, 플레이트 표시 위치 (verify_web 과 같은 값)
     GET  /video                  화면용 1280x720 흑백 MJPEG (왜곡 보정 + SEMES 화각 + 확대)
     POST /view  {zoom, center}   확대 1~8, center = "plate" | "image"
     GET  /settle?px=0.25&timeout=6   흔들림 판정 (verify_web wait_settled 그대로) → JSON
     GET  /frame?average=1        알고리즘 입력 3840x2160 8bit 흑백 원시 바이트 (verify_web Rectifier 그대로)

주의
  - verify_web / demo_web 와 동시에 켜지 않는다 (카메라·시리얼은 한 프로그램만 쓸 수 있음)
  - robot_stage 커널 드라이버가 같은 USB 포트를 쓰고 있으면 시리얼 중계를 거부한다 (demo_web serial_stage.py 와 같은 검사)
"""
import argparse
import fcntl
import glob
import json
import socket
import struct
import sys
import termios
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PLATE_CENTER = (1906.3, 1130.8)   # verify_web/app.py 와 같음 (A507 센터 사진의 가운데 QR, 알고리즘 좌표)


# ------------------------------------------------------------------ 시리얼 ↔ TCP 중계
def serial_bridge(port_path, tcp_port, host):
    import serial
    ser = serial.Serial(None, 115200, timeout=.02, write_timeout=1, exclusive=True)
    ser.dtr = False
    ser.rts = False
    ser.port = port_path
    ser.open()
    discipline = bytearray(4)
    fcntl.ioctl(ser.fileno(), termios.TIOCGETD, discipline, True)
    if struct.unpack('i', discipline)[0] != 0:
        ser.close()
        print(f'[시리얼] {port_path} 를 커널 드라이버가 쓰는 중이라 중계하지 않음', flush=True)
        return
    srv = socket.create_server((host, tcp_port), reuse_port=False)
    print(f'[시리얼] {port_path} ↔ tcp {host}:{tcp_port}', flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f'[시리얼] 연결: {addr}', flush=True)
        ser.reset_input_buffer()
        alive = threading.Event()
        alive.set()

        def up():   # 시리얼 → TCP
            while alive.is_set():
                try:
                    data = ser.read(max(1, ser.in_waiting))
                    if data:
                        conn.sendall(data)
                except Exception:
                    alive.clear()

        t = threading.Thread(target=up, daemon=True)
        t.start()
        try:
            while alive.is_set():
                data = conn.recv(256)
                if not data:
                    break
                ser.write(data)
        except Exception:
            pass
        alive.clear()
        t.join(1)
        conn.close()
        print(f'[시리얼] 끊김: {addr}', flush=True)


# ------------------------------------------------------------------ 카메라 HTTP
def make_handler(cam, rectifier):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == '/health':
                    h = cam.health()
                    px, py = cam.view_point(*PLATE_CENTER)
                    # view_map: 알고리즘 좌표 → 화면 좌표 1차식 [a, b, c, d] (화면 x = a·x + b, 화면 y = c·y + d). Unity 가 QR·나사 윤곽을 그릴 때 씀
                    x0, y0 = cam.view_point(0.0, 0.0)
                    x1, y1 = cam.view_point(1000.0, 1000.0)
                    vm = [(x1 - x0) / 1000.0, x0, (y1 - y0) / 1000.0, y0]
                    h.update(plate=[round(px, 1), round(py, 1)], view_size=[1280, 720], alg_size=[3840, 2160],
                             view_map=[round(float(v), 6) for v in vm])
                    return self._json(h)
                if u.path == '/settle':
                    ok, sec, shift = cam.wait_settled(float(q.get('px', 0.25)), timeout=float(q.get('timeout', 6.0)))
                    return self._json(dict(settled=ok, seconds=sec, shift_px=None if shift != shift else shift))
                if u.path == '/frame':
                    gray, meta = cam.grab(int(q.get('average', 1)))
                    img = rectifier.apply(gray)
                    b = img.tobytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(len(b)))
                    self.send_header('X-Width', str(img.shape[1]))
                    self.send_header('X-Height', str(img.shape[0]))
                    self.send_header('X-Exposure-Us', str(meta.get('exposure_us')))
                    self.send_header('X-Gain', str(meta.get('gain')))
                    self.end_headers()
                    self.wfile.write(b)
                    return
                if u.path == '/video':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    cam.add_viewer(+1)
                    try:
                        seq = 0
                        while True:
                            seq, jpg = cam.wait_jpeg(seq, timeout=1.0)
                            if jpg is None:
                                continue
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                             + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        cam.add_viewer(-1)
                    return
                self._json(dict(error='없는 경로'), 404)
            except Exception as exc:
                self._json(dict(error=str(exc)), 500)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get('Content-Length') or 0)
            try:
                b = json.loads(self.rfile.read(n) or b'{}')
                if u.path == '/view':
                    zoom = float(b.get('zoom', 1.0))
                    if not 1.0 <= zoom <= 8.0:
                        raise ValueError('확대는 1~8 배')
                    cam.set_view(zoom, PLATE_CENTER if b.get('center') == 'plate' else None)
                    return self._json(dict(ok=True))
                self._json(dict(error='없는 경로'), 404)
            except Exception as exc:
                self._json(dict(error=str(exc)), 400)
    return H


def main():
    p = argparse.ArgumentParser(description='Unity 시뮬레이터용 파이 중계 (카메라 HTTP + STM32 시리얼 TCP)')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--http-port', type=int, default=8091)
    p.add_argument('--tcp-port', type=int, default=5555)
    p.add_argument('--serial', default=None, help='기본: /dev/serial/by-id ST-LINK, 없으면 /dev/ttyACM0')
    p.add_argument('--no-stage', action='store_true', help='시리얼 중계 끔 (카메라만)')
    p.add_argument('--no-camera', action='store_true', help='카메라 끔 (시리얼만)')
    p.add_argument('--camera', choices=['picamera2', 'off'], default='picamera2')
    p.add_argument('--calib', type=Path, default=Path('hardware/camera_calib/camera_calib.yml'))
    p.add_argument('--zoom', type=float, default=1.0)
    p.add_argument('--exposure', type=int, default=8333)   # verify_web 기본과 같음
    p.add_argument('--gain', type=float, default=3.0)
    p.add_argument('--fps', type=float, default=10.0)
    a = p.parse_args()

    if not a.no_stage:
        path = a.serial
        if path is None:
            links = sorted(glob.glob('/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_*-if02'))
            path = links[0] if links else '/dev/ttyACM0'
        threading.Thread(target=serial_bridge, args=(path, a.tcp_port, a.host), daemon=True).start()

    if a.no_camera:
        print('[카메라] 끔', flush=True)
        while True:
            time.sleep(3600)

    sys.path.insert(0, str(Path.cwd()))
    import cv2
    cv2.setNumThreads(2)
    from verify_web.camera import FakeCamera, PiCamera
    from verify_web.geometry import Calibration, Rectifier, ViewMapper
    calib = Calibration.load(a.calib)
    rectifier = Rectifier(calib, a.zoom)
    view = ViewMapper(calib, a.zoom)
    cam = (FakeCamera(view, fps=a.fps) if a.camera == 'off' else
           PiCamera(view, exposure_us=a.exposure, gain=a.gain, fps=a.fps)).start()
    srv = ThreadingHTTPServer((a.host, a.http_port), make_handler(cam, rectifier))
    print(f'[카메라] http://{a.host}:{a.http_port}  ({a.camera}, 캘리브레이션 {a.calib}, 원본 밖 {rectifier.outside * 100:.2f}%)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == '__main__':
    main()

import argparse
import fcntl
import getpass
import signal
import socket
import threading
from pathlib import Path

from werkzeug.security import generate_password_hash

from .app import create_app
from .store import Store


def main():
    parser = argparse.ArgumentParser(description='Y·T alignment local demo web')
    parser.add_argument('command', choices=['serve', 'set-password'])
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'instance')
    parser.add_argument('--mode', choices=['mock', 'hardware'], default='mock')
    parser.add_argument('--camera', choices=['off', 'picamera2'], default=None)
    parser.add_argument('--camera-profile', choices=['full', 'medium', 'preview'], default=None)
    parser.add_argument('--transport', choices=['serial', 'driver'], default='driver')
    parser.add_argument('--stream-buffer-kb', type=int, default=256, help='연결당 영상 송신 버퍼 (작을수록 지연↓)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One server for this database. Real mode also locks a shared device-owner file.
    lock = (args.data_dir / 'server.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('같은 데이터 폴더의 서버를 먼저 종료하세요.')
    if args.command == 'set-password':
        password = getpass.getpass('관리자 비밀번호 (10자 이상): ')
        if len(password) < 10 or len(password) > 512:
            parser.error('비밀번호는 10~512자로 설정하세요.')
        if password != getpass.getpass('비밀번호 확인: '):
            parser.error('비밀번호가 일치하지 않습니다.')
        Store(args.data_dir / 'demo.sqlite3').meta('password', generate_password_hash(password))
        print('관리자 비밀번호를 설정했습니다.')
        return
    app = create_app(args.data_dir, args.mode,
                     None if args.camera is None else args.camera == 'picamera2',
                     camera_profile=args.camera_profile, transport=args.transport)
    camera, controller = app.extensions['camera'], app.extensions['controller']
    camera.start()
    shutdown = threading.Event()

    def monitor():
        while not shutdown.wait(3):
            controller.probe()

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    threading.Thread(target=monitor, daemon=True, name='device-health').start()
    from waitress import serve
    print(f'웹 실행: http://{args.host}:{args.port} · mode={args.mode} · T축 실물 비활성', flush=True)
    if not app.extensions['store'].meta('password'):
        print('관리자 비밀번호 미설정: 서버 종료 후 python -m demo_web set-password', flush=True)
    try:
        # Camera latency: waitress buffers up to 16MB per connection by default, so a slow Wi-Fi
        # client fell seconds behind. Block the MJPEG generator once ~1 frame is queued; it then
        # sends the newest frame and skips stale ones. Small SO_SNDBUF bounds the kernel queue.
        buffer = args.stream_buffer_kb * 1024
        listener = socket.create_server((args.host, args.port))  # SO_REUSEADDR on Linux
        listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer)  # inherited by accepted sockets
        serve(app, sockets=[listener], threads=16, channel_timeout=30,
              max_request_body_size=256 * 1024, clear_untrusted_proxy_headers=True,
              outbuf_high_watermark=buffer)
    finally:
        shutdown.set()
        controller.close()
        camera.stop()
        lock.close()


if __name__ == '__main__':
    main()

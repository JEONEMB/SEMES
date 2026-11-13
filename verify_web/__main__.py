import argparse
import fcntl
import shlex
import signal
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
# 장비 캘리브레이션(hardware/camera_calib, roll_deg 포함)이 있으면 그것을, 없으면 데모 웹 것을 쓴다
CALIB_CANDIDATES = (REPO / 'hardware' / 'camera_calib' / 'camera_calib.yml',
                    REPO / 'demo_web' / 'calibration' / 'camera_calib.yml')
DEFAULT_CALIB = next((c for c in CALIB_CANDIDATES if c.exists()), CALIB_CANDIDATES[-1])


def main():
    p = argparse.ArgumentParser(description='Y·Θ 알고리즘 동작 확인 화면')
    p.add_argument('command', choices=['serve'])
    p.add_argument('--host', default='127.0.0.1', help='다른 PC 에서 보려면 0.0.0.0 (같은 망 누구나 조작 가능)')
    p.add_argument('--port', type=int, default=8090)
    p.add_argument('--mode', choices=['mock', 'hardware'], default='mock', help='스테이지: 모의 / 실물')
    p.add_argument('--transport', choices=['driver', 'serial'], default='driver',
                   help='실물 연결: robot_stage 드라이버(/dev/robot_stage) / USB 직접')
    p.add_argument('--camera', choices=['picamera2', 'off'], default='picamera2')
    p.add_argument('--calib', type=Path, default=DEFAULT_CALIB,
                   help='기본: hardware/camera_calib/camera_calib.yml (있으면) → demo_web/calibration/camera_calib.yml')
    p.add_argument('--zoom', type=float, default=1.0, help='알고리즘 사진 배율 (데이터셋 촬영 --zoom 과 같게)')
    p.add_argument('--exposure', type=int, default=8333)
    p.add_argument('--gain', type=float, default=3.0, help='데이터셋 촬영(2026-11-12)과 같게 3.0')
    p.add_argument('--fps', type=float, default=10.0)
    p.add_argument('--algo', choices=['worker', 'mock'], default='worker')
    p.add_argument('--algo-path', type=Path, default=ROOT / 'native' / 'build' / 'align_worker')
    p.add_argument('--algo-output', choices=['offset', 'command'], default='offset',
                   help='알고리즘 반환값: offset(센터 대비 위치, 명령 = 부호 반대) / command(보낼 명령)')
    p.add_argument('--algo-gray', action='store_true', help='알고리즘에 흑백 1채널로 넘김 (기본 BGR 3채널)')
    p.add_argument('--algo-config', type=Path, default=None,
                   help='알고리즘 설정 파일 (예: ~/KSH_Algorithm/Code/config/alignment.yml). 영점 등록을 쓰는 알고리즘만')
    p.add_argument('--demo-port', type=int, default=8080)
    p.add_argument('--demo-args', default='--mode hardware',
                   help='데모 웹으로 버튼으로 띄울 데모 웹 serve 추가 인자 (--host/--port 는 자동)')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'instance')
    args = p.parse_args()

    import cv2
    cv2.setNumThreads(2)       # 영상·HTTP 처리에 코어를 남긴다

    from .algorithm import MockAlgorithm, WorkerAlgorithm, AlgorithmError
    from .app import create_app
    from .camera import FakeCamera, PiCamera
    from .devices import DemoLauncher, DeviceManager
    from .geometry import Calibration, Rectifier, ViewMapper
    from .runner import Runner, Settings
    from demo_web.stage import make_stage

    args.data_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.data_dir / 'server.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('확인 앱이 이미 실행 중입니다.')

    calib = Calibration.load(args.calib)
    rectifier = Rectifier(calib, args.zoom)
    print(f'캘리브레이션: {args.calib} (원본 밖 {rectifier.outside * 100:.2f}%)', flush=True)

    def camera_factory():
        view = ViewMapper(calib, args.zoom)
        if args.camera == 'off':
            return FakeCamera(view, fps=args.fps)
        return PiCamera(view, exposure_us=args.exposure, gain=args.gain, fps=args.fps)

    devices = DeviceManager(camera_factory, lambda: make_stage(args.mode, args.transport))
    try:
        algorithm = (MockAlgorithm() if args.algo == 'mock' else
                     WorkerAlgorithm(args.algo_path, args.algo_output, args.algo_gray,
                                     log_path=args.data_dir / 'align_worker.log', config=args.algo_config))
    except AlgorithmError as exc:
        p.error(f'{exc}\n  빌드: cmake -S verify_web/native -B verify_web/native/build -G Ninja && '
                f'cmake --build verify_web/native/build   (또는 --algo mock)')
    settings = Settings(args.data_dir / 'settings.json')
    runner = Runner(devices, algorithm, rectifier, settings, args.data_dir / 'runs')
    demo_cmd = [sys.executable, '-m', 'demo_web', 'serve', '--host', args.host, '--port', str(args.demo_port)]
    demo = DemoLauncher(demo_cmd + shlex.split(args.demo_args), REPO, args.demo_port,
                        args.data_dir / 'demo_web.log')

    if not devices.open():
        print(f'장치 연결 실패: {devices.error}', flush=True)
    app = create_app(runner, devices, demo)

    stop = threading.Event()

    def monitor():
        while not stop.wait(1.0):
            try:
                runner.poll()
            except Exception:
                pass

    threading.Thread(target=monitor, daemon=True, name='stage-poll').start()

    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)

    from waitress import serve
    buffer = 256 * 1024
    listener = socket.create_server((args.host, args.port))
    listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer)
    print(f'확인 앱: http://{args.host}:{args.port}  · 스테이지 {args.mode}/{args.transport} · 카메라 {args.camera} '
          f'· 알고리즘 {args.algo}({args.algo_output}) · 데모 웹 포트 {args.demo_port}', flush=True)
    try:
        serve(app, sockets=[listener], threads=32, channel_timeout=30,
              max_request_body_size=64 * 1024, outbuf_high_watermark=buffer)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        runner.stop()
        runner.wait_idle(10)
        demo.stop()
        devices.close()
        algorithm.close()
        lock.close()


if __name__ == '__main__':
    main()

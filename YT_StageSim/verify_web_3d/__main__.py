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
# [verify_web_3d] 팀 verify_web(dev)을 복사해 3D 칸만 더한 사본. 알고리즘 실행기·설정·기록은 팀 verify_web 폴더 것을 같이 쓴다
#   (data-dir 이 같으면 server.lock 때문에 원본과 사본이 동시에 뜨지 않아 장비를 두 앱이 잡는 일이 없다)
TEAM = REPO / 'verify_web'
# 장비 캘리브레이션(hardware/camera_calib, roll_deg 포함)이 있으면 그것을, 없으면 데모 웹 것을 쓴다
CALIB_CANDIDATES = (REPO / 'hardware' / 'camera_calib' / 'camera_calib.yml',
                    REPO / 'demo_web' / 'calibration' / 'camera_calib.yml')
DEFAULT_CALIB = next((c for c in CALIB_CANDIDATES if c.exists()), CALIB_CANDIDATES[-1])


def main():
    p = argparse.ArgumentParser(description='Y·Θ 알고리즘 동작 확인 화면 + 3D 장비 (verify_web 사본)')
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
    p.add_argument('--algo-path', type=Path, default=TEAM / 'native' / 'build' / 'align_worker')
    p.add_argument('--algo-output', choices=['offset', 'command'], default='offset',
                   help='알고리즘 반환값: offset(센터 대비 위치, 명령 = 부호 반대) / command(보낼 명령)')
    p.add_argument('--algo-gray', action='store_true', help='알고리즘에 흑백 1채널로 넘김 (기본 BGR 3채널)')
    p.add_argument('--algo-config', type=Path, default=None,
                   help='알고리즘 설정 파일 (예: ~/KSH_Algorithm/Code/config/alignment.yml). 영점 등록을 쓰는 알고리즘만')
    p.add_argument('--algos', type=Path, default=None,
                   help='고를 수 있는 알고리즘 목록 JSON (기본: <data-dir>/algorithms.json 이 있으면 그것). '
                        '없으면 --algo-path/--algo-config 하나만 씀. 형식은 verify_web/algorithms.example.json')
    p.add_argument('--demo-port', type=int, default=8080)
    p.add_argument('--demo-args', default='--mode hardware',
                   help='데모 웹으로 버튼으로 띄울 데모 웹 serve 추가 인자 (--host/--port 는 자동)')
    p.add_argument('--data-dir', type=Path, default=TEAM / 'instance',
                   help='기본: 팀 verify_web/instance (설정·알고리즘 목록·기록을 원본과 같이 씀)')
    args = p.parse_args()

    import cv2
    cv2.setNumThreads(2)       # 영상·HTTP 처리에 코어를 남긴다

    from .algorithm import AlgorithmChoices, MockAlgorithm, AlgorithmError
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
    settings = Settings(args.data_dir / 'settings.json')
    choices = None
    algos_file = args.algos or (args.data_dir / 'algorithms.json')
    try:
        if args.algo == 'mock':
            algorithm = MockAlgorithm()
        else:
            if args.algos or algos_file.exists():
                choices = AlgorithmChoices.load(algos_file, REPO, log_path=args.data_dir / 'align_worker.log')
            else:
                choices = AlgorithmChoices([dict(id='default', label=args.algo_path.parent.name, path=str(args.algo_path),
                                                 config=str(args.algo_config) if args.algo_config else None,
                                                 output=args.algo_output, gray=args.algo_gray)],
                                           REPO, log_path=args.data_dir / 'align_worker.log')
            listing = choices.listing()
            usable = [a['id'] for a in listing if a['available']]
            if not usable:
                raise AlgorithmError('쓸 수 있는 알고리즘이 없음: ' + '; '.join(f"{a['id']}: {a['reason']}" for a in listing))
            pick = settings['algorithm'] if settings['algorithm'] in usable else usable[0]
            algorithm = choices.select(pick)
            print('알고리즘: ' + ', '.join(f"{a['id']}{'*' if a['id'] == pick else ''}"
                                         f"{'' if a['available'] else ' (없음: ' + a['reason'] + ')'}" for a in listing),
                  flush=True)
    except (AlgorithmError, ValueError, OSError) as exc:
        p.error(f'{exc}\n  빌드: cmake -S verify_web/native -B verify_web/native/build -G Ninja && '
                f'cmake --build verify_web/native/build   (또는 --algo mock)')
    runner = Runner(devices, algorithm, rectifier, settings, args.data_dir / 'runs', choices=choices)
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
        (choices or algorithm).close()
        lock.close()


if __name__ == '__main__':
    main()

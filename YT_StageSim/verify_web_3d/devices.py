"""카메라·스테이지 소유와 데모 웹 넘김.

카메라와 STM32(드라이버/USB)는 한 번에 한 프로그램만 쓸 수 있다.
`데모 웹으로` 버튼 → 확인 앱이 장치를 놓고 데모 웹을 자식 프로세스로 띄운다.
확인 앱 화면을 다시 열면 → 데모 웹을 끄고(SIGTERM = 데모 웹의 정상 종료) 장치를 다시 잡는다.
데모 웹 코드는 고치지 않는다.
"""
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path


class DeviceManager:
    def __init__(self, camera_factory, stage_factory):
        self.camera_factory = camera_factory
        self.stage_factory = stage_factory
        self.camera = None
        self.stage = None
        self.error = None
        self.lock = threading.RLock()

    @property
    def ready(self):
        return self.camera is not None and self.stage is not None

    def open(self):
        with self.lock:
            errors = []
            if self.camera is None:
                try:
                    self.camera = self.camera_factory().start()
                except Exception as exc:
                    errors.append(f'카메라: {exc}')
            if self.stage is None:
                try:
                    self.stage = self.stage_factory()
                    try:   # STM32 가 리셋되면 Y 기본 속도가 0.1 mm/s 라 이동이 매우 느림 → 1 mm/s (Θ 는 그대로)
                        self.stage.speed(1000, 0)
                    except Exception as exc:
                        errors.append(f'속도 설정 실패: {exc}')
                except Exception as exc:
                    errors.append(f'스테이지: {exc}')
            self.error = ' / '.join(errors) or None
            return self.ready

    def close(self):
        with self.lock:
            if self.stage is not None:
                try:
                    self.stage.close()
                finally:
                    self.stage = None
            if self.camera is not None:
                try:
                    self.camera.close()
                finally:
                    self.camera = None

    def status(self):
        stage = self.stage
        return dict(ready=self.ready, error=self.error,
                    stage_mode=getattr(stage, 'mode', None), stage_transport=getattr(stage, 'transport', None))


def port_open(port, host='127.0.0.1'):
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


class DemoLauncher:
    def __init__(self, cmd, cwd, port, log_path):
        self.cmd, self.cwd, self.port = list(cmd), Path(cwd), int(port)
        self.log_path = Path(log_path)
        self.proc = None
        self.lock = threading.Lock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout=40.0):
        with self.lock:
            if self.running():
                return
            if port_open(self.port):
                raise RuntimeError(f'포트 {self.port} 에 이미 다른 프로그램(데모 웹?)이 떠 있습니다. 먼저 종료하세요.')
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_path.open('ab')
            log.write(f'\n===== {time.strftime("%F %T")} 데모 웹 시작: {" ".join(self.cmd)}\n'.encode())
            log.flush()
            try:
                self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
            finally:
                log.close()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    code = self.proc.returncode
                    self.proc = None
                    raise RuntimeError(f'데모 웹이 바로 종료됨 (코드 {code}). 로그: {self.log_path}')
                if port_open(self.port):
                    return
                time.sleep(0.2)
            self._terminate()
            raise RuntimeError(f'데모 웹이 {timeout:.0f}초 안에 뜨지 않음. 로그: {self.log_path}')

    def stop(self):
        with self.lock:
            self._terminate()

    def _terminate(self):
        p = self.proc
        if p is None:
            return
        for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
            if p.poll() is not None:
                break
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            try:
                p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        self.proc = None
        # 데모 웹이 카메라를 놓을 때까지 잠깐 (libcamera 해제)
        time.sleep(0.5)

    def status(self):
        return dict(running=self.running(), port=self.port)

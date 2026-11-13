"""Stage backends and single-owner lock. Direct UART is in serial_stage.py.

ABI source: feat/lye/robot-stage-driver:hardware/driver/include/uapi/robot_stage_ioctl.h
Kernel driver owns CRC, SEQ, ACK retries and DONE matching. T hardware stays disabled.
"""
import errno
import fcntl
import os
import struct
import threading
import time
from pathlib import Path

from .store import Problem


class DeviceOwner:
    """One web process owns physical motion, regardless of transport or DB path."""
    def __init__(self):
        self.file = (Path.home() / '.alignment-stage.lock').open('a')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            raise Problem('다른 실물 제어 서버가 장치를 사용 중입니다.', 409) from exc

    def close(self):
        self.file.close()


def make_stage(mode, transport='serial'):
    if mode == 'mock':
        return MockStage()
    if mode != 'hardware' or transport not in ('serial', 'driver'):
        raise Problem('지원하지 않는 장치 모드입니다.')
    if transport == 'serial':
        from .serial_stage import SerialStage
        return SerialStage()
    return IoctlStage()


class MockStage:
    mode = 'mock'
    transport = 'mock'

    def __init__(self, duration=0.65):
        self.position = (0, 0)
        self.duration = duration
        self.pending = None
        self.stopped = False
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            return dict(state=1 if self.pending else 0, flags=3, position=self.position)

    def limits(self):
        return (-50000, 50000, -10000, 10000)

    def ping(self):
        return (1, 0, 0, 0)

    def speed(self, y, t):
        self.speeds = (y, t)

    def start(self, y, t, relative=False, allow_t=False):
        with self.lock:
            if self.pending:
                raise Problem('장치가 이동 중입니다.', 409)
            target = (self.position[0] + y, self.position[1] + t) if relative else (y, t)
            if not (-50000 <= target[0] <= 50000 and -10000 <= target[1] <= 10000):
                raise Problem('모의 장치 범위 초과')
            self.pending = (target, time.monotonic(), 3 if relative else 4)
            self.stopped = False

    def wait_done(self, timeout_ms=30000):
        with self.lock:
            pending = self.pending
        if pending is None:
            raise Problem('대기 중인 이동이 없습니다.')
        while time.monotonic() - pending[1] < self.duration:
            if self.stopped:
                break
            time.sleep(0.01)
        with self.lock:
            if not self.stopped:
                self.position = pending[0]
            self.pending = None
            return dict(result=1 if self.stopped else 0, cmd_type=pending[2], position=self.position)

    def home_y(self):
        self.start(0, 0)
        with self.lock:
            target, started, _ = self.pending
            self.pending = (target, started, 2)

    def stop(self):
        self.stopped = True

    def close(self):
        pass


class IoctlStage:
    mode = 'hardware'
    transport = 'driver'

    def __init__(self, path='/dev/robot_stage'):
        self.path = path
        self.fd = None
        self.lock = threading.Lock()
        self.owner = None

    def _call(self, number, fmt, values=(), direction=2, nak_index=None):
        size = struct.calcsize(fmt)
        buf = bytearray(struct.pack(fmt, *values) if values else bytes(size))
        code = (direction << 30) | (size << 16) | (ord('R') << 8) | number
        # WAIT_DONE must not hold this lock: STOP needs to run concurrently.
        with self.lock:
            if self.fd is None:
                self.owner = DeviceOwner()
                try:
                    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
                except OSError as exc:
                    self.owner.close()
                    self.owner = None
                    if exc.errno in (errno.EACCES, errno.EPERM):
                        raise Problem(f'{self.path} 접근 권한이 없습니다. 웹 서버 사용자에게 장치 권한(udev 규칙)이 필요합니다.', 503) from exc
                    if exc.errno == errno.ENOENT:
                        raise Problem(f'{self.path}가 없습니다. robot_stage 드라이버와 STM32 연결을 확인하세요.', 503) from exc
                    raise Problem('STM32 드라이버에 연결할 수 없습니다. 장치와 권한을 확인하세요.', 503) from exc
            fd = self.fd
        try:
            fcntl.ioctl(fd, code, buf, True)
        except OSError as exc:
            # Python ioctl copies mutated bytes back even when the syscall fails.
            nak = buf[nak_index] if nak_index is not None else 0
            if nak:
                reasons = {1: '알 수 없는 명령', 2: '잘못된 길이', 3: '원점 미설정',
                           4: '허용 범위 초과', 5: '이동 중', 6: '장치 고장', 7: '잘못된 값'}
                raise Problem('STM32 명령 거부: ' + reasons.get(nak, str(nak)), 409) from exc
            if exc.errno == errno.ETIMEDOUT:
                raise Problem('STM32 응답 시간 초과. 이동 여부를 확인해야 합니다.', 503) from exc
            raise Problem(f'STM32 통신 오류 ({exc.errno}). 장치 상태 확인이 필요합니다.', 503) from exc
        return struct.unpack(fmt, buf)

    def status(self):
        state, flags, y, t, ys, ts = self._call(6, '<BB2xiiii')
        return dict(state=state, flags=flags, position=(y, t), steps=(ys, ts))

    def limits(self):
        return self._call(8, '<iiii')

    def ping(self):
        return self._call(1, '<BBBB')

    def speed(self, y, t):
        self._call(7, '<IIB3x', (y, t, 0), 3, 8)

    def start(self, y, t, relative=False, allow_t=False):
        # allow_t: rsctl-equivalent console passes T through; the firmware decides (BAD_PARAM if disabled).
        if t != 0 and not allow_t:
            raise Problem('T축 모터가 비활성 상태입니다. T 명령은 0이어야 합니다.')
        self._call(3 if relative else 4, '<iiB3x', (y, t, 0), 3, 8)

    def home_y(self):
        self._call(2, '<BB', (1, 0), 3, 1)

    def wait_done(self, timeout_ms=30000):
        try:
            _, result, cmd_type, y, t = self._call(9, '<IBB2xii', (timeout_ms, 0, 0, 0, 0), 3)
        except Problem:
            # Read-only reconciliation, never retransmit a relative move after DONE timeout.
            try:
                self.status()
            except Problem:
                pass
            raise
        return dict(result=result, cmd_type=cmd_type, position=(y, t))

    def stop(self):
        self._call(5, '<B', nak_index=0)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.owner:
            self.owner.close()
            self.owner = None

"""알고리즘 연결.

WorkerAlgorithm: native/align_worker (저장소 alignment()) 를 자식 프로세스로 띄워 사진을 넘긴다.
MockAlgorithm:   사진을 보지 않고 스테이지 실제 오차 + 잡음을 돌려준다 (화면·흐름 시험용).

반환값 뜻 (output):
  'offset'  = 센터에서 얼마나 벗어났는가 → 보낼 명령은 부호를 뒤집은 값 (김성현 방식)
  'command' = 보낼 명령 그 자체 (이승재 방식)
단위는 Y mm, Θ deg.
"""
import os
import random
import select
import struct
import subprocess
import time
from pathlib import Path


class AlgorithmError(RuntimeError):
    pass


class MockAlgorithm:
    kind = 'mock'
    output = 'offset'

    def __init__(self, noise_y_mm=0.008, noise_t_deg=0.002, seed=None):
        self.noise = (noise_y_mm, noise_t_deg)
        self.rng = random.Random(seed)

    def infer(self, image, truth=None):
        t0 = time.monotonic()
        if truth is None:
            raise AlgorithmError('모의 추론에는 스테이지 실제 오차가 필요함 (STATUS 실패?)')
        y = truth[0] + self.rng.gauss(0, self.noise[0])
        t = truth[1] + self.rng.gauss(0, self.noise[1])
        return dict(y=y, t=t, ms=(time.monotonic() - t0) * 1000)

    def status(self):
        return dict(kind=self.kind, output=self.output, ready=True, detail='모의 추론 (실제 오차 + 잡음)')

    def close(self):
        pass


class WorkerAlgorithm:
    kind = 'worker'

    def __init__(self, path, output='offset', gray=False, timeout_s=30.0, log_path=None):
        self.path = Path(path)
        if not os.access(self.path, os.X_OK):
            raise AlgorithmError(f'알고리즘 실행기가 없음: {self.path} (verify_web/native 를 빌드하세요)')
        if output not in ('offset', 'command'):
            raise ValueError('output 은 offset 또는 command')
        self.output, self.gray, self.timeout_s = output, gray, timeout_s
        self.log_path = log_path
        self.proc = None
        self.last_error = None

    def _start(self):
        args = [str(self.path)] + (['--gray'] if self.gray else [])
        err = open(self.log_path, 'ab') if self.log_path else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        finally:
            if self.log_path:
                err.close()
        line = self._readline(10.0)
        if line != 'READY':
            self._kill()
            raise AlgorithmError(f'알고리즘 실행기 시작 실패: {line!r}')

    def _readline(self, timeout):
        fd = self.proc.stdout.fileno()
        buf = b''
        deadline = time.monotonic() + timeout
        while not buf.endswith(b'\n'):
            left = deadline - time.monotonic()
            if left <= 0:
                raise AlgorithmError(f'알고리즘 응답 없음 ({timeout:.0f}s)')
            r, _, _ = select.select([fd], [], [], left)
            if not r:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                raise AlgorithmError(f'알고리즘 실행기가 종료됨 (코드 {self.proc.poll()})')
            buf += chunk
        return buf.decode('utf-8', 'replace').strip()

    def _kill(self):
        if self.proc:
            try:
                self.proc.kill()
                self.proc.wait(timeout=3)
            except Exception:
                pass
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:
                    pass
            self.proc = None

    def infer(self, image, truth=None):
        if image is None or image.ndim != 2:
            raise AlgorithmError('8bit 흑백 사진이 필요함')
        if self.proc is not None and self.proc.poll() is not None:
            self._kill()            # 밖에서 죽었으면 옛 파이프를 닫고 새로 띄움
        if self.proc is None:
            self._start()
        h, w = image.shape
        t0 = time.monotonic()
        try:
            self.proc.stdin.write(b'ALGN' + struct.pack('<II', w, h))
            self.proc.stdin.write(memoryview(image if image.flags['C_CONTIGUOUS'] else image.copy()))
            self.proc.stdin.flush()
            line = self._readline(self.timeout_s)
        except (AlgorithmError, OSError) as exc:
            self._kill()        # 멈췄거나 죽음 → 다음 요청 때 다시 띄움
            self.last_error = str(exc)
            raise AlgorithmError(str(exc)) from exc
        wall = (time.monotonic() - t0) * 1000
        if line.startswith('OK '):
            try:
                _, y, t, ms = line.split()
                self.last_error = None
                return dict(y=float(y), t=float(t), ms=float(ms), wall_ms=wall)
            except ValueError:
                pass
        self.last_error = line
        raise AlgorithmError(f'알고리즘 오류: {line}')

    def status(self):
        return dict(kind=self.kind, output=self.output, ready=True, detail=str(self.path.name),
                    error=self.last_error)

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
                self.proc.stdout.close()
            except Exception:
                self._kill()
            self.proc = None

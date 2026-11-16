"""알고리즘 연결.

WorkerAlgorithm: native/align_worker (저장소 alignment()) 를 자식 프로세스로 띄워 사진을 넘긴다.
MockAlgorithm:   사진을 보지 않고 스테이지 실제 오차 + 잡음을 돌려준다 (화면·흐름 시험용).

반환값 뜻 (output):
  'offset'  = 센터에서 얼마나 벗어났는가 → 보낼 명령은 부호를 뒤집은 값 (김성현 방식)
  'command' = 보낼 명령 그 자체 (이승재 방식)
단위는 Y mm, Θ deg.
"""
import json
import os
import random
import select
import struct
import subprocess
import time
from pathlib import Path


class AlgorithmError(RuntimeError):
    pass


def parse_rois(text):
    """" | 이름 x y w h cx cy; ..." → [dict(name, rect=[x,y,w,h], center=[cx,cy])] (알고리즘 좌표)."""
    out = []
    for part in text.split(';'):
        f = part.split()
        if len(f) == 7:
            try:
                out.append(dict(name=f[0], rect=[int(v) for v in f[1:5]], center=[float(f[5]), float(f[6])]))
            except ValueError:
                pass
    return out


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

    supports_reference = True

    def set_reference(self, image):
        return dict(y=0.0, t=0.0)

    def status(self):
        return dict(kind=self.kind, output=self.output, ready=True, detail='모의 추론 (실제 오차 + 잡음)',
                    supports_reference=True)

    def close(self):
        pass


class WorkerAlgorithm:
    kind = 'worker'

    def __init__(self, path, output='offset', gray=False, timeout_s=30.0, log_path=None, config=None):
        self.path = Path(path)
        if not os.access(self.path, os.X_OK):
            raise AlgorithmError(f'알고리즘 실행기가 없음: {self.path} (verify_web/native 를 빌드하세요)')
        if output not in ('offset', 'command'):
            raise ValueError('output 은 offset 또는 command')
        self.output, self.gray, self.timeout_s = output, gray, timeout_s
        self.config = str(config) if config else None   # 알고리즘 설정 파일 (alignment_init 에 넘김)
        self.log_path = log_path
        self.proc = None
        self.last_error = None
        self.caps = None            # 실행기가 READY 줄에 알린 기능 (ref = 영점 등록, roi = 위치 돌려줌)

    def _start(self):
        args = [str(self.path)] + (['--gray'] if self.gray else []) + (['--config', self.config] if self.config else [])
        err = open(self.log_path, 'ab') if self.log_path else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        finally:
            if self.log_path:
                err.close()
        try:
            line = self._readline(10.0)
        except AlgorithmError:
            self._kill()
            raise
        words = line.split()
        if not words or words[0] != 'READY':
            self._kill()
            self.last_error = line
            raise AlgorithmError(f'알고리즘 실행기 시작 실패: {line}')
        self.caps = set(words[1:])

    @property
    def supports_reference(self):
        """영점(센터 사진) 등록을 지원하나. 모르면 실행기를 띄워 READY 줄로 확인한다."""
        if self.caps is None:
            if self.proc is None:
                self._start()
        return 'ref' in (self.caps or ())

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

    def _send(self, magic, image, timeout):
        if image is None or image.ndim != 2:
            raise AlgorithmError('8bit 흑백 사진이 필요함')
        if self.proc is not None and self.proc.poll() is not None:
            self._kill()
        if self.proc is None:
            self._start()
        h, w = image.shape
        try:
            self.proc.stdin.write(magic + struct.pack('<II', w, h))
            self.proc.stdin.write(memoryview(image if image.flags['C_CONTIGUOUS'] else image.copy()))
            self.proc.stdin.flush()
            return self._readline(timeout)
        except (AlgorithmError, OSError) as exc:
            self._kill()
            self.last_error = str(exc)
            raise AlgorithmError(str(exc)) from exc

    def set_reference(self, image):
        """센터 사진으로 영점 등록 (alignment_set_reference). 반환: 학습 영점 대비 센터 사진의 측정값."""
        line = self._send(b'ALGR', image, self.timeout_s)
        if line.startswith('OK '):
            head, _, tail = line.partition(' | ')
            _, y, t = head.split()[:3]
            self.reference = dict(y=float(y), t=float(t), rois=parse_rois(tail))
            return self.reference
        self.last_error = line
        raise AlgorithmError(f'영점 등록 실패: {line}')

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
                head, _, tail = line.partition(' | ')
                _, y, t, ms = head.split()
                self.last_error = None
                return dict(y=float(y), t=float(t), ms=float(ms), wall_ms=wall, rois=parse_rois(tail))
            except ValueError:
                pass
        self.last_error = line
        raise AlgorithmError(f'알고리즘 오류: {line}')

    def status(self):
        return dict(kind=self.kind, output=self.output, ready=True, detail=str(self.path.name),
                    error=self.last_error, supports_reference=None if self.caps is None else 'ref' in self.caps)

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
                self.proc.stdout.close()
            except Exception:
                self._kill()
            self.proc = None


class AlgorithmChoices:
    """확인 앱에서 고를 수 있는 알고리즘 목록. 한 번에 하나만 실행기를 띄운다.

    항목: dict(id, label, path, config, output='offset', gray=False)  (path·config 는 저장소 기준 상대경로 가능)
    """

    def __init__(self, entries, root, log_path=None, make=None):
        if not entries:
            raise ValueError('알고리즘이 하나도 없음')
        ids = [e['id'] for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError(f'알고리즘 id 가 겹침: {ids}')
        self.root = Path(root)
        self.entries = [dict(e) for e in entries]
        self.log_path = log_path
        self._make = make or self._make_worker
        self.current_id = None
        self.current = None

    @classmethod
    def load(cls, path, root, log_path=None):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(data['algorithms'] if isinstance(data, dict) else data, root, log_path)

    def _path(self, v):
        if not v:
            return None
        v = Path(os.path.expanduser(str(v)))
        return v if v.is_absolute() else self.root / v

    def _make_worker(self, e):
        return WorkerAlgorithm(self._path(e['path']), e.get('output', 'offset'), e.get('gray', False),
                               log_path=self.log_path, config=self._path(e.get('config')))

    def entry(self, algo_id):
        for e in self.entries:
            if e['id'] == algo_id:
                return e
        raise ValueError(f'알 수 없는 알고리즘: {algo_id}')

    def available(self, e):
        """실행기 파일이 있나 (없으면 화면에 이유를 보여 준다)."""
        if e.get('path') == 'mock':
            return True, None
        p = self._path(e['path'])
        if not (p and os.access(p, os.X_OK)):
            return False, f'실행기 없음: {p}'
        c = self._path(e.get('config'))
        if c and not c.exists():
            return False, f'설정·자료 없음: {c}'
        return True, None

    def select(self, algo_id):
        e = self.entry(algo_id)
        ok, why = self.available(e)
        if not ok:
            raise AlgorithmError(why)
        new = MockAlgorithm() if e.get('path') == 'mock' else self._make(e)
        old, self.current, self.current_id = self.current, new, algo_id
        if old is not None:
            old.close()
        return new

    def listing(self):
        out = []
        for e in self.entries:
            ok, why = self.available(e)
            out.append(dict(id=e['id'], label=e.get('label', e['id']), available=ok, reason=why,
                            current=e['id'] == self.current_id))
        return out

    def close(self):
        if self.current is not None:
            self.current.close()

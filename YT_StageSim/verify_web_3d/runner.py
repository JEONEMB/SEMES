"""검증 루프: 회차마다 흔들림 대기 → 촬영 → 추론 → 보정 이동, 최대 N회.

화면에 보낼 상태(state)를 한곳에서 관리한다. 동작은 한 번에 하나만(작업 스레드 1개),
정지는 언제든 (중단 플래그 + STM32 STOP).

검증용으로 매 회차 '알고리즘 추론값'과 '스테이지 실제 오차(STATUS 위치 − 센터)'를 같이 남긴다.
"""
import copy
import csv
import json
import math
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PHASES = {
    'idle': '대기', 'homing': '원점 복귀 중', 'positioning': '시작 위치로 이동 중',
    'settling': '흔들림 대기', 'capturing': '촬영', 'inferring': '추론', 'moving': '보정 이동 중',
    'done': '완료', 'stopped': '정지됨', 'error': '오류', 'released': '데모 웹 사용 중',
}
STATE_NAMES = {0: 'idle', 1: 'moving', 2: 'homing', 3: 'fault'}
DONE_RESULTS = {0: '정상', 1: '정지 명령으로 중단', 2: '리밋 스위치 감지', 3: '드라이버 고장'}

DEFAULTS = dict(
    center_y_mm=0.0, center_t_deg=0.0,       # 플레이트가 센터일 때의 스테이지 좌표
    tol_y_mm=0.045, tol_t_deg=0.01,          # 이 안이면 '보정 완료'
    max_trials=20,                           # 최대 보정 이동 횟수 (측정은 이동 + 1 번: 마지막 이동 뒤 확인 측정)
    enable_t=True,                           # Θ 명령 전송 (꺼져 있으면 Θ 는 그대로 두고 표시만)
    max_step_y_mm=2.0, max_step_t_deg=1.0,   # 추론값이 이보다 크면 움직이지 않고 멈춤
    random_y_mm=1.5, random_t_deg=0.5,       # 랜덤 시작 위치 범위 (±)
    approach_y_mm=0.2,                       # 모든 이동은 이만큼 −쪽을 거쳐 +방향으로 접근 (데이터셋과 같음)
    approach_t_deg=0.1,                      # Θ 도 마찬가지. 방향 전환 백래시 약 0.023° 를 피한다 (2026-11-13 시험)
    average=1,                               # 추론 사진 평균 장수
    settle_px=0.25, settle_timeout_s=6.0,
    save_images=True,
    auto_reference=True,                     # 보정·측정을 시작할 때 센터 기준이 없거나 오래됐으면 먼저 잡는다
    reference_max_age_min=60.0,              # 센터는 몇 시간에 0.01~0.02° 씩 밀려서 이보다 오래된 기준은 다시 잡는다
    demo_slow=True,                          # 시연 속도: 천천히 움직이고 회차마다 결과를 잠깐 보여 준다
    slow_y_mm_s=0.2, slow_t_deg_s=0.1,       # 천천히 (Θ 펌웨어 최소 0.018°/s)
    fast_y_mm_s=1.0, fast_t_deg_s=1.0,       # 빠르게 (펌웨어 기본 Θ 1°/s, Y 는 우리가 1 mm/s 로 올려 씀)
    pause_s=3.0,
    sequential_axes=True,                    # 한 회차 안에서 Y 를 먼저 옮기고 이어서 Θ 를 옮긴다 (한 번 측정한 두 축 보정량 그대로)
    algorithm='',                            # 고른 알고리즘 id (algorithms.json). 빈 값 = 목록의 첫 번째                             # 천천히일 때 측정 결과를 보여 주고 움직이기까지 쉬는 시간
)
LIMITS = dict(tol_y_mm=(0.001, 1), tol_t_deg=(0.0001, 1), max_trials=(1, 100), max_step_y_mm=(0.01, 10),
              max_step_t_deg=(0.001, 5), random_y_mm=(0, 10), random_t_deg=(0, 5), approach_y_mm=(0, 2), approach_t_deg=(0, 1),
              average=(1, 16), settle_px=(0.05, 5), settle_timeout_s=(0.5, 30), reference_max_age_min=(1, 1440),
              center_y_mm=(-1000, 1000), center_t_deg=(-360, 360), slow_y_mm_s=(0.01, 2), fast_y_mm_s=(0.01, 2),
              slow_t_deg_s=(0.02, 3.6), fast_t_deg_s=(0.02, 3.6), pause_s=(0, 10))
CENTER_CHECK = (0.1, 0.05)   # '현재 위치를 센터로' 전 확인 측정: 알고리즘이 보는 센터와 이만큼(Y mm, Θ °) 넘게 다르면 묻는다
CENTER_CHECK_AGE_S = 120    # 확인 측정이 이보다 오래됐거나 그 뒤 스테이지가 움직였으면 다시 재야 한다
STEP_CLAMP = 1.25     # 보정량이 한 번 이동 한도의 이 배수까지는 한도만큼만 옮기고, 넘으면 멈춘다
RANDOM_MIN = 0.4      # 랜덤 시작은 범위의 40 % 이상 떨어지게 (너무 가까우면 시연에서 움직임이 안 보임)


_QR = None


def find_qr_quads(img):
    """알고리즘 좌표(3840×2160) 사진에서 QR 3개의 네 꼭짓점 (왼쪽부터). 못 찾으면 None. 화면 기준 윤곽용."""
    global _QR
    if _QR is None:
        _QR = cv2.QRCodeDetectorAruco()
    try:
        ok, p = _QR.detectMulti(img)
    except cv2.error:
        return None
    if not ok or p is None or len(p) != 3:
        return None
    q = p.reshape(-1, 4, 2)
    q = q[np.argsort(q.mean(1)[:, 0])]
    return [[[round(float(x), 2), round(float(y), 2)] for x, y in quad] for quad in q]



def measure_label(n):
    """n 번째 측정의 이름 (횟수는 보정 이동 기준: 1 번째 측정은 시작 위치, n 번째는 이동 n−1 번 뒤)."""
    return '시작 위치' if n <= 1 else f'이동 {n - 1}번 후'


def converged_label(n):
    moves = n - 1
    return '시작 위치가 이미 허용 오차 안 (이동 없음)' if moves == 0 else f'보정 이동 {moves}번 만에 허용 오차 안으로 들어옴'

class RunError(RuntimeError):
    pass


class Aborted(RunError):
    pass


def um(mm):
    return int(round(mm * 1000))


def r6(v):
    return None if v is None else round(v, 6)


class Settings:
    def __init__(self, path, overrides=None):
        self.path = Path(path)
        self.values = dict(DEFAULTS)
        try:
            self.values.update({k: v for k, v in json.loads(self.path.read_text()).items() if k in DEFAULTS})
        except (OSError, ValueError):
            pass
        if overrides:
            self.values.update({k: v for k, v in overrides.items() if v is not None})

    def update(self, changes):
        new = dict(self.values)
        for k, v in changes.items():
            if k not in DEFAULTS:
                raise ValueError(f'알 수 없는 설정: {k}')
            kind = type(DEFAULTS[k])
            if kind is bool:
                if not isinstance(v, bool):
                    raise ValueError(f'{k}: true/false')
            elif kind is str:
                if not isinstance(v, str):
                    raise ValueError(f'{k}: 문자열이 필요함')
            else:
                try:
                    v = kind(v)
                except (TypeError, ValueError):
                    raise ValueError(f'{k}: 숫자가 필요함') from None
                if isinstance(v, float) and not math.isfinite(v):
                    raise ValueError(f'{k}: 숫자가 필요함')
                lo, hi = LIMITS.get(k, (-math.inf, math.inf))
                if not lo <= v <= hi:
                    raise ValueError(f'{k}: {lo} ~ {hi}')
            new[k] = v
        self.values = new
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.values, indent=1, ensure_ascii=False))

    def __getitem__(self, k):
        return self.values[k]


class Runner:
    def __init__(self, devices, algorithm, rectifier, settings, runs_dir, choices=None):
        self.devices = devices          # .stage, .camera, .ready (DeviceManager)
        self.algorithm = algorithm
        self.choices = choices          # AlgorithmChoices (확인 앱에서 알고리즘 고르기) 또는 None
        self.rectifier = rectifier
        self.settings = settings
        self.runs_dir = Path(runs_dir)
        self.cond = threading.Condition()
        self.version = 0
        self.abort = threading.Event()
        self.job = None
        self.state = dict(phase='idle', message='', busy=False, job=None, moving=False,
                          position=None, flags=None, stage_state=None, stage_error=None,
                          run=None, last=None, history=[], measure=None, reference=None, aligned=None,
                          rois_now=None, start=None, speed=None, axis_stage=None, center_check=None)

    # --- 상태 -----------------------------------------------------------------
    def _set(self, **kw):
        with self.cond:
            self.state.update(kw)
            self.version += 1
            self.cond.notify_all()

    def _phase(self, phase, message=''):
        self._check_abort()
        self._set(phase=phase, message=message)

    def snapshot(self):
        with self.cond:
            s = copy.deepcopy(self.state)
            version = self.version
        s['phase_label'] = PHASES.get(s['phase'], s['phase'])
        s['settings'] = dict(self.settings.values)
        s['devices'] = self.devices.status()
        s['algorithm'] = self.algorithm.status()
        s['algorithms'] = self.choices.listing() if self.choices else None
        cam = self.devices.camera
        s['camera'] = cam.health() if cam else dict(ok=False, error='카메라 없음 (데모 웹 사용 중일 수 있음)')
        s['version'] = version
        s['now'] = time.time()          # 화면이 축별 이동 예상 시간을 서버 시계로 계산하도록
        return s

    def wait_change(self, version, timeout=15.0):
        with self.cond:
            if self.version == version:
                self.cond.wait(timeout)
            return self.version

    # --- 작업 실행 -------------------------------------------------------------
    def _launch(self, name, fn, *args):
        with self.cond:
            if self.job and self.job.is_alive():
                raise RunError('다른 작업이 진행 중입니다. 정지 후 다시 시도하세요.')
            if not self.devices.ready:
                raise RunError('장치가 없습니다 (데모 웹 사용 중이거나 연결 실패).')
            self.abort.clear()
            self.state.update(busy=True, job=name, message='')
            self.version += 1
            self.cond.notify_all()
            self.job = threading.Thread(target=self._guard, args=(name, fn) + args, daemon=True, name=f'job-{name}')
            self.job.start()

    def _guard(self, name, fn, *args):
        try:
            self._apply_speed()
            fn(*args)
        except Aborted:
            self._set(phase='stopped', message='정지 요청으로 멈췄습니다.')
        except Exception as exc:
            self._set(phase='error', message=str(exc))
            if self.state.get('run') and not self.state['run'].get('result'):
                self._finish_run('error', str(exc))
        finally:
            self._set(busy=False, job=None, moving=False)
            self._refresh_status()

    def busy(self):
        return bool(self.job and self.job.is_alive())

    def home(self):
        self._launch('home', self._home)

    def goto(self, offset):
        if offset != 'random':
            self._check_start(*offset)
        self._launch('goto', self._goto_job, offset)

    def _check_start(self, y, t):
        """직접 입력한 시작 위치: 자동 보정 한 번 이동 한도 안이어야 보정이 중간에 멈추지 않는다."""
        s = self.settings
        if not (math.isfinite(y) and math.isfinite(t)):
            raise RunError('Y, Θ 에 숫자를 넣으세요.')
        if abs(y) > s['max_step_y_mm'] or (s['enable_t'] and abs(t) > s['max_step_t_deg']):
            raise RunError(f'시작 위치는 센터에서 Y ±{s["max_step_y_mm"]} mm, Θ ±{s["max_step_t_deg"]}° 안으로 넣으세요 '
                           f'(자동 보정 한 번 이동 한도).')

    def run(self, offset=None):
        if isinstance(offset, tuple):
            self._check_start(*offset)
        self._launch('run', self._run, offset)

    def step(self):
        self._launch('step', self._step)

    def measure(self):
        self._launch('measure', self._measure)

    def reference(self):
        self._launch('reference', self._reference)

    def stop(self):
        self.abort.set()
        stage = self.devices.stage
        if stage and self.state.get('moving'):
            try:
                stage.stop()
            except Exception as exc:
                self._set(message=f'STOP 실패: {exc}')

    def select_algorithm(self, algo_id):
        """다른 알고리즘으로 바꾼다. 영점·기록이 알고리즘마다 달라서 센터 기준과 지난 기록은 지운다."""
        if not self.choices:
            raise RunError('고를 수 있는 알고리즘 목록이 없습니다 (--algos).')
        if self.busy():
            raise RunError('작업 중에는 알고리즘을 바꿀 수 없습니다.')
        self.algorithm = self.choices.select(algo_id)
        self.settings.update(dict(algorithm=algo_id))
        cam = self.devices.camera
        if cam:
            cam.set_reference(None)
        label = self.choices.entry(algo_id).get('label', algo_id)
        self._set(reference=None, run=None, history=[], last=None, measure=None, aligned=None, rois_now=None,
                  start=None, phase='idle', message=f'알고리즘: {label}. 센터 기준은 다음 시작 때 새로 잡습니다.')

    def supports_reference(self):
        try:
            return bool(getattr(self.algorithm, 'supports_reference', False))
        except Exception:
            return False

    def center_check(self):
        """'현재 위치를 센터로' 전에 지금 자리를 카메라로 잰다 (움직이지 않고, 센터 기준도 새로 찍지 않음)."""
        self._launch('center_check', self._center_check)

    def _center_check(self):
        pos = self._refresh_status()
        if pos is None:
            raise RunError('스테이지 위치를 읽지 못했습니다.')
        entry, _ = self._observe(label='센터 확인')
        ref = self.state.get('reference')
        basis = 'reference' if ref and not ref.get('fixed_zero') else 'algorithm'
        y, t = entry['offset']['y_mm'], entry['offset']['t_deg']
        ok = abs(y) <= CENTER_CHECK[0] and (not self.settings['enable_t'] or abs(t) <= CENTER_CHECK[1])
        chk = dict(ts=time.time(), position=list(pos), y_mm=y, t_deg=t, ok=ok, basis=basis,
                   limit=dict(y_mm=CENTER_CHECK[0], t_deg=CENTER_CHECK[1]))
        where = '지금 센터 기준 사진' if basis == 'reference' else '알고리즘의 센터(학습·고정 영점)'
        self._set(center_check=chk, phase='idle',
                  message=f'센터 확인: {where} 대비 Y {y:+.3f} mm, Θ {t:+.4f}° ' + ('(센터로 저장해도 됨)' if ok else '(센터와 다름)'))

    def set_center_here(self, force=False):
        """지금 스테이지 좌표를 센터로 저장. force 가 아니면 방금(같은 자리) 잰 센터 확인이 허용 안이어야 한다."""
        if self.busy():
            raise RunError('작업 중에는 센터를 바꿀 수 없습니다.')
        pos = self._refresh_status()
        if pos is None:
            raise RunError('스테이지 위치를 읽지 못했습니다.')
        chk = self.state.get('center_check')
        fresh = chk and time.time() - chk['ts'] <= CENTER_CHECK_AGE_S and tuple(chk['position']) == tuple(pos)
        if not force:
            if not fresh:
                raise RunError('먼저 지금 자리에서 센터 확인 측정을 해야 합니다.')
            if not chk['ok']:
                raise RunError(f'지금 자리는 센터에서 Y {chk["y_mm"]:+.3f} mm, Θ {chk["t_deg"]:+.4f}° 떨어져 있습니다.')
        self.settings.update(dict(center_y_mm=pos[0] / 1000, center_t_deg=pos[1] / 1000))
        cam = self.devices.camera
        if cam:
            cam.set_reference(None)     # 예전 센터에서 찍은 기준 사진은 버리고 다음 시작 때 새 센터에서 다시 찍는다
        self._set(center_check=None, reference=None, start=None,
                  message=f'센터 = Y {pos[0] / 1000:.3f} mm, Θ {pos[1] / 1000:.3f}°'
                          + (' (확인 측정과 달랐지만 사용자가 저장)' if fresh and not chk['ok'] else ''))

    def wait_idle(self, timeout=None):
        job = self.job
        if job:
            job.join(timeout)

    # --- 스테이지 --------------------------------------------------------------
    def _apply_speed(self):
        """작업마다 시연 속도(천천히/빠르게)를 STM32 에 보낸다. 대기 중에만 부른다."""
        s = self.settings
        mode = 'slow' if s['demo_slow'] else 'fast'
        y, t = s[mode + '_y_mm_s'], s[mode + '_t_deg_s']
        self.devices.stage.speed(um(y), um(t))
        self._set(speed=dict(mode=mode, y_mm_s=y, t_deg_s=t))

    def _pause(self, short=False):
        """천천히 모드: 측정 결과를 화면에서 볼 수 있게 잠깐 멈춘다 (정지 누르면 바로 끝).
        short: Y 이동과 Θ 이동 사이 (최대 1초)."""
        if self.settings['demo_slow'] and self.settings['pause_s'] > 0:
            self.abort.wait(min(1.0, self.settings['pause_s']) if short else self.settings['pause_s'])
            self._check_abort()

    def _check_abort(self):
        if self.abort.is_set():
            raise Aborted()

    def _refresh_status(self):
        """대기 중에만 부른다 (이동 중 질의로 ACK 재전송을 방해하지 않도록)."""
        stage = self.devices.stage
        if stage is None:
            return None
        try:
            s = stage.status()
        except Exception as exc:
            self._set(stage_error=str(exc))
            return None
        pos = tuple(s['position'])
        self._set(position=dict(y_mm=pos[0] / 1000, t_deg=pos[1] / 1000), flags=s.get('flags'),
                  stage_state=STATE_NAMES.get(s.get('state'), str(s.get('state'))), stage_error=None)
        return pos

    def poll(self):
        """대기 중 1초마다 위치 갱신 (서버의 감시 스레드가 부름)."""
        if self.devices.ready and not self.busy():
            self._refresh_status()

    def _truth(self):
        pos = self._refresh_status()
        if pos is None:
            return None
        return (pos[0] / 1000 - self.settings['center_y_mm'], pos[1] / 1000 - self.settings['center_t_deg'])

    def _move(self, y_um, t_mdeg, relative, phase, message='', final=None, origin=None):
        self._phase(phase, message)
        origin = origin or self.state.get('position')     # 이번 이동을 시작한 자리 (접근점을 거쳐도 처음 자리)
        stage = self.devices.stage
        enable_t = self.settings['enable_t']
        if not enable_t and relative:
            t_mdeg = 0
        try:
            lim = stage.limits()
        except Exception:
            lim = None
        if lim and not relative:
            if not (lim[0] <= y_um <= lim[1] and lim[2] <= t_mdeg <= lim[3]):
                raise RunError(f'목표가 스테이지 한계 밖: Y {y_um / 1000:.3f} mm (한계 {lim[0] / 1000:.3f}~{lim[1] / 1000:.3f})')
        self._set(moving=True, aligned=None, rois_now=None,   # 움직이면 다시 재기 전까지 정렬 여부를 모른다
                  target=dict(y_mm=y_um / 1000, t_deg=t_mdeg / 1000, relative=relative,
                              axes=self._axis_plan(y_um, t_mdeg, relative),
                              final=final,          # 접근점으로 가는 중이면 그다음 최종 목표 (화면 안내용)
                              origin=dict(origin) if origin else None))
        try:
            # 절대 이동에서 Θ 가 꺼져 있으면 목표 Θ = 지금 Θ 라 그대로 보낸다
            stage.start(y_um, t_mdeg, relative=relative, allow_t=enable_t or not relative)
            done = stage.wait_done(timeout_ms=60000)
        finally:
            self._set(moving=False, target=None)
        pos = tuple(done['position'])
        self._set(position=dict(y_mm=pos[0] / 1000, t_deg=pos[1] / 1000))
        self._check_abort()          # 정지 요청으로 끊긴 이동은 오류가 아니라 '정지됨'
        if done['result'] != 0:
            raise RunError('이동 실패: ' + DONE_RESULTS.get(done['result'], str(done['result'])))

    def _axis_plan(self, y_um, t_mdeg, relative):
        """축마다 이동량과 예상 시간 (속도로 나눈 값, 가감속 빼고). 이동 중에는 STATUS 를 묻지 않아서
        화면이 'Y 이동 중 / Θ 이동 중' 을 이 예상으로 보여 준다. 두 축은 동시에 출발한다."""
        pos, sp = self.state.get('position'), self.state.get('speed')
        if relative:
            dy, dt = y_um / 1000, t_mdeg / 1000
        elif pos:
            dy, dt = y_um / 1000 - pos['y_mm'], t_mdeg / 1000 - pos['t_deg']
        else:
            return None
        est = lambda d, v: round(abs(d) / v, 2) if v else None
        return dict(started=time.time(), dy_mm=round(dy, 4), dt_deg=round(dt, 4),
                    y_s=est(dy, sp and sp['y_mm_s']) if abs(dy) >= 0.0005 else 0.0,
                    t_s=est(dt, sp and sp['t_deg_s']) if abs(dt) >= 0.0005 else 0.0)

    def _home(self):
        self._phase('homing')
        stage = self.devices.stage
        self._set(moving=True)
        try:
            stage.home_y()
            done = stage.wait_done(timeout_ms=130000)
        finally:
            self._set(moving=False)
        self._check_abort()
        if done['result'] != 0:
            raise RunError('원점 복귀 실패: ' + DONE_RESULTS.get(done['result'], str(done['result'])))
        self._set(phase='idle', message='Y 원점 복귀 완료', start=None)

    def _resolve_offset(self, offset):
        if offset == 'random':
            ry, rt = self.settings['random_y_mm'], self.settings['random_t_deg']
            # 크기는 범위의 40~100 %, 부호는 무작위 (가까운 시작은 시연에서 움직임이 안 보인다)
            pick = lambda r: random.choice((-1, 1)) * random.uniform(RANDOM_MIN * r, r)
            y = pick(ry)
            t = pick(rt) if self.settings['enable_t'] else 0.0
            return round(y, 3), round(t, 3)
        y, t = offset
        return float(y), float(t) if self.settings['enable_t'] else 0.0

    def _move_to(self, y_mm, t_deg, phase, message=''):
        """절대 목표로. 항상 −쪽 접근점을 거쳐 +방향으로 마지막 접근 (백래시 방향 통일, 데이터셋과 같음)."""
        s = self.settings
        if not s['enable_t']:
            pos = self.state.get('position')
            t_deg = pos['t_deg'] if pos else 0.0          # Θ 는 그대로 둔다
        ay, at = s['approach_y_mm'], (s['approach_t_deg'] if s['enable_t'] else 0.0)
        pos = self.state.get('position')
        if pos:      # 안 움직이는 축은 접근점을 거치지 않는다 (차례 보정에서 Θ 를 그대로 두려고)
            if abs(y_mm - pos['y_mm']) < 0.0005:
                ay = 0.0
            if abs(t_deg - pos['t_deg']) < 0.0005:
                at = 0.0
        if ay > 0 or at > 0:
            self._move(um(y_mm - ay), um(t_deg - at), relative=False, phase=phase, message=message,
                       final=dict(y_mm=y_mm, t_deg=t_deg), origin=pos)
        self._move(um(y_mm), um(t_deg), relative=False, phase=phase, message=message, origin=pos)

    def _goto(self, offset):
        y, t = self._resolve_offset(offset)
        cy, ct = self.settings['center_y_mm'], self.settings['center_t_deg']
        self._set(start=dict(y_mm=y, t_deg=t, reached=False))
        self._move_to(cy + y, ct + t, phase='positioning',
                      message=f'시작 위치로: 센터 + Y {y:+.3f} mm, Θ {t:+.3f}°')
        self._set(start=dict(y_mm=y, t_deg=t, reached=True))
        return y, t

    def _goto_job(self, offset):
        # 센터 기준(오늘의 센터 사진)이 없거나 오래됐으면 센터에 들른 김에 먼저 찍는다 → 시연 버튼은 '랜덤 → 보정' 두 개
        self._clear_demo()
        self._ensure_reference(come_back=False)
        y, t = self._goto(offset)
        self._set(phase='idle', message=f'시작 위치 도착: 센터 + Y {y:+.3f} mm, Θ {t:+.3f}°')

    def _clear_demo(self):
        """새 시연: 지난 실행의 회차 기록·지도 경로·결과를 지운다 (파일로 남긴 run.json 은 그대로)."""
        self._set(run=None, history=[], last=None, measure=None, aligned=None, rois_now=None)

    # --- 검증 ------------------------------------------------------------------
    def _new_run(self, start_offset=None):
        run_id = time.strftime('%Y%m%d_%H%M%S')
        run = dict(id=run_id, started=time.time(), start_offset=start_offset, trial=0,
                   max_trials=int(self.settings['max_trials']), result=None, result_label=None,
                   algorithm=self.algorithm.status()['kind'], center=dict(y_mm=self.settings['center_y_mm'],
                                                                         t_deg=self.settings['center_t_deg']))
        self._set(run=run, history=[], last=None)
        return run

    def _finish_run(self, result, label):
        with self.cond:
            run = self.state.get('run')
            if not run:
                return
            run.update(result=result, result_label=label, finished=time.time())
            self.version += 1
            self.cond.notify_all()
            hist = list(self.state['history'])
            run = dict(run)
        d = self.runs_dir / run['id']
        d.mkdir(parents=True, exist_ok=True)
        (d / 'run.json').write_text(json.dumps(dict(run=run, settings=self.settings.values, history=hist),
                                               indent=1, ensure_ascii=False))

    def _reference_needed(self):
        s = self.settings
        if not s['auto_reference']:      # 영점 등록이 없는 알고리즘도 화면용 센터 사진은 잡는다
            return False
        ref = self.state.get('reference')
        return ref is None or time.time() - ref['ts'] > s['reference_max_age_min'] * 60

    def _ensure_reference(self, come_back):
        """기준이 없거나 오래됐으면 센터에서 잡는다. come_back 이면 원래 자리로 (−쪽에서 접근해) 돌아온다."""
        if not self._reference_needed():
            return
        pos = self._refresh_status() if come_back else None     # STATUS 로 지금 자리를 읽어 둔다 (µm, mdeg)
        old = self.state.get('reference')
        self._set(message='센터 기준이 ' + ('없어' if old is None else '오래돼') + ' 먼저 잡습니다.')
        self._reference()
        if pos is not None:
            self._move_to(pos[0] / 1000, pos[1] / 1000, phase='positioning')

    def _run(self, offset):
        self._ensure_reference(come_back=offset is None)   # 시작 위치로 갈 거면 돌아올 필요 없음
        start = self._goto(offset) if offset is not None else None
        if start is None and self.state.get('start'):
            st = self.state['start']
            start = (st['y_mm'], st['t_deg'])
        run = self._new_run(start)
        # 횟수는 '보정 이동' 기준. n 번째 측정 = 이동 n−1 번 뒤 (1 번째는 시작 위치), 마지막 이동 뒤에도 확인 측정
        for n in range(1, run['max_trials'] + 2):
            if self._trial(n, move=n <= run['max_trials']):
                self._finish_run('converged', converged_label(n))
                self._set(phase='done', message=self.state['run']['result_label'], start=None)
                return
        self._finish_run('max_trials', f'보정 이동 {run["max_trials"]}번 해도 허용 오차 밖')
        self._set(phase='done', message=self.state['run']['result_label'])

    def _step(self):
        run = self.state.get('run')
        if not run or run.get('result'):
            self._ensure_reference(come_back=True)
            run = self._new_run(None)
        n = run['trial'] + 1
        if self._trial(n, move=True):
            self._finish_run('converged', converged_label(n))
            self._set(phase='done', message=self.state['run']['result_label'])
        else:
            self._set(phase='idle', message=f'보정 이동 {n}번째 완료 (한 단계)')

    def _reference(self):
        """센터로 (−쪽에서 접근) 가서 사진을 찍어 알고리즘 영점으로 등록한다. 그날·그 세션의 센터를 기준으로 삼기 위해.
        영점 등록이 없는 알고리즘(고정 영점)은 화면 윤곽·차이 영상용 사진만 잡는다."""
        self._move_to(self.settings['center_y_mm'], self.settings['center_t_deg'], phase='positioning',
                      message='센터 기준을 잡으러 센터로 갑니다.')
        s = self.settings
        cam = self.devices.camera
        self._phase('settling')
        cam.wait_settled(s['settle_px'], timeout=s['settle_timeout_s'])
        self._phase('capturing')
        gray, _ = cam.grab(max(4, int(s['average'])))
        img = self.rectifier.apply(gray)
        fixed = not self.supports_reference()
        if fixed:
            z = dict(y=0.0, t=0.0, rois=[])
        else:
            self._phase('inferring')
            z = self.algorithm.set_reference(img)
        ref = dict(ts=time.time(), zero_y_mm=r6(z['y']), zero_t_deg=r6(z['t']), qr_alg=find_qr_quads(img),
                   rois=z.get('rois') or [], fixed_zero=fixed)
        cam.set_reference(gray)                 # 차이 영상용 (센서 원본)
        if self.runs_dir:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.runs_dir / f'reference_{time.strftime("%Y%m%d_%H%M%S")}.bmp'), img)
        if fixed:
            self._set(reference=ref, phase='idle', start=None, aligned=None, rois_now=None,
                      message='센터 사진을 화면 윤곽용으로만 찍었습니다 (이 알고리즘은 자료에 고정된 영점을 씀).')
        else:
            self._set(reference=ref, phase='idle', start=None, aligned=True, rois_now=ref['rois'],
                      message=f'센터 기준 등록: 학습 영점 대비 Y {z["y"] * 1000:+.1f} µm, Θ {z["t"]:+.4f}°')

    def _measure(self):
        self._ensure_reference(come_back=True)
        entry, _ = self._observe(label='측정')
        self._set(measure=entry, phase='idle', message='측정만 했습니다 (이동 없음).')

    def _observe(self, label):
        """흔들림 대기 → 촬영 → 추론. 이동은 하지 않는다."""
        s = self.settings
        cam = self.devices.camera
        self._phase('settling')
        settled, settle_s, shift = cam.wait_settled(s['settle_px'], timeout=s['settle_timeout_s'])
        self._phase('capturing')
        gray, meta = cam.grab(int(s['average']))
        img = self.rectifier.apply(gray)
        truth = self._truth()
        self._phase('inferring')
        r = self.algorithm.infer(img, truth)
        out = self.algorithm.output
        offset = (r['y'], r['t']) if out == 'offset' else (-r['y'], -r['t'])
        command = (-offset[0], -offset[1])
        enable_t = s['enable_t']
        within = abs(offset[0]) <= s['tol_y_mm'] and (not enable_t or abs(offset[1]) <= s['tol_t_deg'])
        entry = dict(label=label, ts=time.time(), raw=dict(y=r6(r['y']), t=r6(r['t'])), output=out,
                     offset=dict(y_mm=r6(offset[0]), t_deg=r6(offset[1])),
                     command=dict(y_mm=r6(command[0]), t_deg=r6(command[1]) if enable_t else 0.0),
                     truth=None if truth is None else dict(y_mm=r6(truth[0]), t_deg=r6(truth[1])),
                     error=None if truth is None else dict(y_mm=r6(offset[0] - truth[0]), t_deg=r6(offset[1] - truth[1])),
                     within=within,
                     truth_within=None if truth is None else (abs(truth[0]) <= s['tol_y_mm'] and
                                                              (not enable_t or abs(truth[1]) <= s['tol_t_deg'])),
                     infer_ms=round(r.get('ms', 0), 1), settle_s=round(settle_s, 2), settled=settled,
                     shift_px=None if shift != shift else round(shift, 3), exposure_us=meta.get('exposure_us'),
                     rois=r.get('rois') or [])
        self._set(aligned=entry['within'], rois_now=entry['rois'])
        return entry, img

    def _trial(self, n, move):
        s = self.settings
        with self.cond:
            self.state['run']['trial'] = n
            self.version += 1
            self.cond.notify_all()
        entry, img = self._observe(label=measure_label(n))
        entry['trial'] = n
        # 한 번 측정 → 보정 이동 한 번. 산출한 두 축 보정량을 그대로 쓰고, 차례 보정이면 Y 를 먼저 옮긴 뒤 이어서 Θ 를 옮긴다
        cy, ct = entry['command']['y_mm'], entry['command']['t_deg']
        seq = s['sequential_axes'] and s['enable_t']
        entry['axis'] = None if entry['within'] else ('y_then_t' if seq else 'both')
        if s['save_images']:
            d = self.runs_dir / self.state['run']['id']
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f'trial_{n:02d}.bmp'), img)
            self._append_csv(d / 'trials.csv', entry)
        with self.cond:
            self.state['history'].append(entry)
            self.state['history'] = self.state['history'][-100:]
            self.state['last'] = entry
            self.version += 1
            self.cond.notify_all()
        if entry['within'] or not move:
            return entry['within']
        self._pause()                # 천천히 모드: 화면에서 이번 측정을 보여 주고 움직인다
        ly, lt = s['max_step_y_mm'], s['max_step_t_deg']
        if abs(cy) > ly * STEP_CLAMP or abs(ct) > lt * STEP_CLAMP:      # 한참 넘으면 잘못 본 것일 수 있어 멈춘다
            raise RunError(f'추론 명령이 한도를 넘어 멈춤: Y {cy:+.3f} mm (한도 ±{ly}), Θ {ct:+.3f}° (한도 ±{lt})')
        if abs(cy) > ly or abs(ct) > lt:          # 조금 넘으면 (시작을 한도 끝에 둔 경우 등) 한도만큼만 옮기고 다시 잰다
            cy, ct = max(-ly, min(ly, cy)), max(-lt, min(lt, ct))
            entry['clamped'] = dict(y_mm=r6(cy), t_deg=r6(ct))
            self._set(message=f'보정량이 한도를 조금 넘어 한도만큼만 옮깁니다: Y {cy:+.3f} mm, Θ {ct:+.3f}°')
        pos = self.state.get('position')
        if not seq:
            msg = f'보정 이동: Y {cy:+.3f} mm, Θ {ct:+.3f}°'
            if pos is None:      # STATUS 를 못 읽었으면 예전처럼 상대 이동 (접근 없음)
                self._move(um(cy), um(ct), relative=True, phase='moving', message=msg)
            else:                # 지금 위치 + 명령 = 절대 목표, −쪽에서 접근
                self._move_to(pos['y_mm'] + cy, pos['t_deg'] + ct, phase='moving', message=msg)
            return False
        try:                     # ① Y 만 → ② Θ 만 (안 움직이는 축은 접근점도 거치지 않음)
            self._set(axis_stage='y')
            msg = f'① Y 보정 이동: Y {cy:+.3f} mm  (이어서 Θ {ct:+.3f}°)'
            if pos is None:
                self._move(um(cy), 0, relative=True, phase='moving', message=msg)
            else:
                self._move_to(pos['y_mm'] + cy, pos['t_deg'], phase='moving', message=msg)
            self._pause(short=True)
            self._set(axis_stage='t')
            msg = f'② Θ 보정 이동: Θ {ct:+.3f}°  (Y {cy:+.3f} mm 이동 끝)'
            now = self.state.get('position')
            if pos is None or now is None:
                self._move(0, um(ct), relative=True, phase='moving', message=msg)
            else:
                self._move_to(now['y_mm'], pos['t_deg'] + ct, phase='moving', message=msg)
        finally:
            self._set(axis_stage=None)
        return False

    @staticmethod
    def _append_csv(path, e):
        new = not path.exists()
        g = lambda d, k: '' if d is None else d.get(k)
        with path.open('a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['trial', 'time', 'raw_y', 'raw_t', 'offset_y_mm', 'offset_t_deg', 'cmd_y_mm', 'cmd_t_deg',
                            'truth_y_mm', 'truth_t_deg', 'err_y_mm', 'err_t_deg', 'within', 'truth_within',
                            'infer_ms', 'settle_s', 'settled', 'shift_px', 'axis'])
            w.writerow([e.get('trial'), time.strftime('%H:%M:%S', time.localtime(e['ts'])), e['raw']['y'], e['raw']['t'],
                        e['offset']['y_mm'], e['offset']['t_deg'], e['command']['y_mm'], e['command']['t_deg'],
                        g(e['truth'], 'y_mm'), g(e['truth'], 't_deg'), g(e['error'], 'y_mm'), g(e['error'], 't_deg'),
                        e['within'], e['truth_within'], e['infer_ms'], e['settle_s'], e['settled'], e['shift_px'],
                        e.get('axis') or ''])

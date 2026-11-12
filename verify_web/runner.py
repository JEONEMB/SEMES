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
    max_trials=20,
    enable_t=False,                          # Θ 명령 전송 (꺼져 있으면 Θ 는 0 으로 보내고 표시만)
    max_step_y_mm=2.0, max_step_t_deg=1.0,   # 추론값이 이보다 크면 움직이지 않고 멈춤
    random_y_mm=1.5, random_t_deg=0.5,       # 랜덤 시작 위치 범위 (±)
    approach_y_mm=0.2,                       # 시작 위치는 이만큼 −쪽에서 접근 (데이터셋과 같음)
    average=1,                               # 추론 사진 평균 장수
    settle_px=0.25, settle_timeout_s=6.0,
    save_images=True,
)
LIMITS = dict(tol_y_mm=(0.001, 1), tol_t_deg=(0.0001, 1), max_trials=(1, 100), max_step_y_mm=(0.01, 10),
              max_step_t_deg=(0.001, 5), random_y_mm=(0, 10), random_t_deg=(0, 5), approach_y_mm=(0, 2),
              average=(1, 16), settle_px=(0.05, 5), settle_timeout_s=(0.5, 30),
              center_y_mm=(-1000, 1000), center_t_deg=(-360, 360))


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
    def __init__(self, devices, algorithm, rectifier, settings, runs_dir):
        self.devices = devices          # .stage, .camera, .ready (DeviceManager)
        self.algorithm = algorithm
        self.rectifier = rectifier
        self.settings = settings
        self.runs_dir = Path(runs_dir)
        self.cond = threading.Condition()
        self.version = 0
        self.abort = threading.Event()
        self.job = None
        self.state = dict(phase='idle', message='', busy=False, job=None, moving=False,
                          position=None, flags=None, stage_state=None, stage_error=None,
                          run=None, last=None, history=[], measure=None)

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
        cam = self.devices.camera
        s['camera'] = cam.health() if cam else dict(ok=False, error='카메라 없음 (데모 웹 사용 중일 수 있음)')
        s['version'] = version
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
        self._launch('goto', self._goto_job, offset)

    def run(self, offset=None):
        self._launch('run', self._run, offset)

    def step(self):
        self._launch('step', self._step)

    def measure(self):
        self._launch('measure', self._measure)

    def stop(self):
        self.abort.set()
        stage = self.devices.stage
        if stage and self.state.get('moving'):
            try:
                stage.stop()
            except Exception as exc:
                self._set(message=f'STOP 실패: {exc}')

    def set_center_here(self):
        if self.busy():
            raise RunError('작업 중에는 센터를 바꿀 수 없습니다.')
        pos = self._refresh_status()
        if pos is None:
            raise RunError('스테이지 위치를 읽지 못했습니다.')
        self.settings.update(dict(center_y_mm=pos[0] / 1000, center_t_deg=pos[1] / 1000))
        self._set(message=f'센터 = Y {pos[0] / 1000:.3f} mm, Θ {pos[1] / 1000:.3f}°')

    def wait_idle(self, timeout=None):
        job = self.job
        if job:
            job.join(timeout)

    # --- 스테이지 --------------------------------------------------------------
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

    def _move(self, y_um, t_mdeg, relative, phase):
        self._phase(phase)
        stage = self.devices.stage
        enable_t = self.settings['enable_t']
        if not enable_t:
            t_mdeg = 0
        try:
            lim = stage.limits()
        except Exception:
            lim = None
        if lim and not relative:
            if not (lim[0] <= y_um <= lim[1] and lim[2] <= t_mdeg <= lim[3]):
                raise RunError(f'목표가 스테이지 한계 밖: Y {y_um / 1000:.3f} mm (한계 {lim[0] / 1000:.3f}~{lim[1] / 1000:.3f})')
        self._set(moving=True, target=dict(y_mm=y_um / 1000, t_deg=t_mdeg / 1000, relative=relative))
        try:
            stage.start(y_um, t_mdeg, relative=relative, allow_t=enable_t)
            done = stage.wait_done(timeout_ms=60000)
        finally:
            self._set(moving=False, target=None)
        pos = tuple(done['position'])
        self._set(position=dict(y_mm=pos[0] / 1000, t_deg=pos[1] / 1000))
        self._check_abort()          # 정지 요청으로 끊긴 이동은 오류가 아니라 '정지됨'
        if done['result'] != 0:
            raise RunError('이동 실패: ' + DONE_RESULTS.get(done['result'], str(done['result'])))

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
        self._set(phase='idle', message='Y 원점 복귀 완료')

    def _resolve_offset(self, offset):
        if offset == 'random':
            ry, rt = self.settings['random_y_mm'], self.settings['random_t_deg']
            y = random.uniform(-ry, ry)
            if abs(y) < ry * 0.2:     # 너무 가까운 시작은 의미가 적어 조금 띄운다
                y = math.copysign(ry * 0.2, y or 1)
            t = random.uniform(-rt, rt) if self.settings['enable_t'] else 0.0
            return round(y, 3), round(t, 3)
        y, t = offset
        return float(y), float(t) if self.settings['enable_t'] else 0.0

    def _goto(self, offset):
        y, t = self._resolve_offset(offset)
        cy, ct = self.settings['center_y_mm'], self.settings['center_t_deg']
        a = self.settings['approach_y_mm']
        if a > 0:
            self._move(um(cy + y - a), um(ct + t), relative=False, phase='positioning')
        self._move(um(cy + y), um(ct + t), relative=False, phase='positioning')
        return y, t

    def _goto_job(self, offset):
        y, t = self._goto(offset)
        self._set(phase='idle', message=f'시작 위치 도착: 센터 + Y {y:+.3f} mm, Θ {t:+.3f}°')

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

    def _run(self, offset):
        start = self._goto(offset) if offset is not None else None
        run = self._new_run(start)
        for n in range(1, run['max_trials'] + 1):
            if self._trial(n, move=True):
                self._finish_run('converged', f'{n}회차에 허용 오차 안으로 들어옴')
                self._set(phase='done', message=self.state['run']['result_label'])
                return
        self._finish_run('max_trials', f'{run["max_trials"]}회 안에 수렴하지 못함')
        self._set(phase='done', message=self.state['run']['result_label'])

    def _step(self):
        run = self.state.get('run')
        if not run or run.get('result'):
            run = self._new_run(None)
        n = run['trial'] + 1
        if self._trial(n, move=True):
            self._finish_run('converged', f'{n}회차에 허용 오차 안으로 들어옴')
            self._set(phase='done', message=self.state['run']['result_label'])
        else:
            self._set(phase='idle', message=f'{n}회차 완료 (한 단계)')

    def _measure(self):
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
                     shift_px=None if shift != shift else round(shift, 3), exposure_us=meta.get('exposure_us'))
        return entry, img

    def _trial(self, n, move):
        s = self.settings
        with self.cond:
            self.state['run']['trial'] = n
            self.version += 1
            self.cond.notify_all()
        entry, img = self._observe(label=f'{n}회차')
        entry['trial'] = n
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
        cy, ct = entry['command']['y_mm'], entry['command']['t_deg']
        if abs(cy) > s['max_step_y_mm'] or abs(ct) > s['max_step_t_deg']:
            raise RunError(f'추론 명령이 한도를 넘어 멈춤: Y {cy:+.3f} mm (한도 ±{s["max_step_y_mm"]}), '
                           f'Θ {ct:+.3f}° (한도 ±{s["max_step_t_deg"]})')
        self._move(um(cy), um(ct), relative=True, phase='moving')
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
                            'infer_ms', 'settle_s', 'settled', 'shift_px'])
            w.writerow([e.get('trial'), time.strftime('%H:%M:%S', time.localtime(e['ts'])), e['raw']['y'], e['raw']['t'],
                        e['offset']['y_mm'], e['offset']['t_deg'], e['command']['y_mm'], e['command']['t_deg'],
                        g(e['truth'], 'y_mm'), g(e['truth'], 't_deg'), g(e['error'], 'y_mm'), g(e['error'], 't_deg'),
                        e['within'], e['truth_within'], e['infer_ms'], e['settle_s'], e['settled'], e['shift_px']])

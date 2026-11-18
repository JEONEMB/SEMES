"""Single motion owner. UI never owns counters, execution order or success state."""
import copy
import threading
import uuid

from .store import Problem, at_center, integer, now, validate_scenario
from .stage import make_stage


class Controller:
    MOVING = {'positioning', 'correcting', 'stopping', 'manual_move', 'homing'}

    def __init__(self, store, stage, camera, home_stage=None):
        self.store, self.stage, self.camera = store, stage, camera
        # Y HOME is always physical (same as `rsctl home 1`), even when the server runs mock.
        self.home_stage = home_stage or (lambda: make_stage('hardware', 'driver'))
        self.motion_stage = None
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.cancel = threading.Event()
        self.stop_error = None
        self.worker = None
        self.snapshot = None
        self.cursor = 0
        self.sequence = 0
        self.connected = stage.mode == 'mock'
        self.device_status = None
        self.device_limits = None
        self.device_error = ''
        self.state = dict(status='idle', trial=0, y_um=0, t_mdeg=0, run_id=None,
                          message='시나리오를 준비한 뒤 Random을 눌러 주세요.', test=False,
                          position=None, history=[])

    def public(self):
        with self.lock:
            # Explicit allowlist: no scenario, upcoming command or physical start position.
            return {**{key: self.state[key] for key in
                       ('status', 'trial', 'y_um', 't_mdeg', 'run_id', 'message', 'test')},
                    'max_trials': 20, 'mode': self.stage.mode, 'theta_enabled': False,
                    'transport': self.stage.transport,
                    'controller_connected': self.connected, 'camera': self.camera.health(),
                    'event_id': self.sequence}

    def admin_state(self):
        with self.lock:
            return {**self.public(), 'history': copy.deepcopy(self.state['history']),
                    'position': self.state['position'], 'snapshot': copy.deepcopy(self.snapshot),
                    'device_status': copy.deepcopy(self.device_status),
                    'device_limits': self.device_limits, 'device_error': self.device_error}

    def configure_mode(self, mode, transport='serial'):
        if mode not in ('mock', 'hardware') or transport not in ('serial', 'driver'):
            raise Problem('지원하지 않는 장치 모드입니다.')
        with self.lock:
            self.require_idle()
            if self.stage.mode == 'hardware':
                # Never release ownership while motion may still be in progress.
                try:
                    status = self.stage.status()
                except Problem:
                    status = None  # Explicit reconnect after unplug/reboot; never sends motion.
                if status and status['state'] in (1, 2):
                    raise Problem('STM32가 이동 중입니다. 먼저 정지하세요.', 409)
            # Release the old transport before taking the same device lock.
            self.stage.close()
            candidate = make_stage(mode, transport)
            try:
                status, limits = candidate.status(), candidate.limits()
            except Exception:
                candidate.close()
                # Failed hardware activation falls back explicitly to an idle mock backend.
                self.stage = make_stage('mock')
                self.snapshot = None
                self.state.update(run_id=None, trial=0, y_um=0, t_mdeg=0, history=[], test=False, position=None)
                self.device_status, self.device_limits = None, None
                self.connected = False
                self._update(status='error', message='실물 연결 실패 · 모의 모드로 돌아왔습니다.')
                raise
            self.stage = candidate
            self.snapshot = None
            self.cancel.clear()
            self.stop_error = None
            self.cursor = 0
            self.connected = True
            self.device_status, self.device_limits, self.device_error = status, limits, ''
            self.state.update(run_id=None, trial=0, y_um=0, t_mdeg=0, history=[], test=False, position=None)
            self._update(status='idle', message='장치 모드가 변경되었습니다. 시나리오를 준비하세요.')
            return self.public()

    def require_idle(self):
        if self._working() or self.state['status'] == 'ready':
            raise Problem('실행을 완료하거나 정지한 뒤 설정을 변경하세요.', 409)

    RESULTS = {0: 'OK', 1: 'STOPPED', 2: 'LIMIT', 3: 'FAULT'}
    STATES = {0: 'IDLE', 1: 'MOVING', 2: 'HOMING', 3: 'FAULT'}
    CONSOLE = ('ping', 'status', 'config', 'speed', 'stop', 'home', 'move-rel', 'move-abs')

    def _physical(self):
        """Real STM32 (same device as rsctl): open hardware stage, busy HOME/move stage, or a
        temporary kernel-driver connection when the server itself runs mock."""
        if self.stage.mode == 'hardware':
            return self.stage, False
        if self.motion_stage is not None:
            return self.motion_stage, False
        return self.home_stage(), True

    def home_y(self, key):
        """Physical Y HOME(mask=1), equivalent to `rsctl home 1`, regardless of server mode."""
        return self._physical_motion(key, 'home')

    def console(self, key, cmd, y=0, t=0):
        """rsctl-equivalent commands from the admin page. Always the real STM32."""
        if cmd not in self.CONSOLE:
            raise Problem('지원하지 않는 STM32 명령입니다.')
        if cmd == 'speed':
            integer(y, 'Y 속도(µm/s)', 0, 2 ** 32 - 1)
            integer(t, 'T 속도', 0, 2 ** 32 - 1)
        elif cmd in ('move-rel', 'move-abs'):
            integer(y, 'Y(µm)', -2 ** 31, 2 ** 31 - 1)
            integer(t, 'T(0.001°)', -2 ** 31, 2 ** 31 - 1)
        if cmd in ('home', 'move-rel', 'move-abs'):
            state = self._physical_motion(key, cmd, y, t)
            return dict(output=['ACK, waiting DONE...'], run_id=state['run_id'], state=state)
        with self.lock:
            if cmd == 'stop' and self._working():
                state = self.stop(key)
                if self.stage.mode == 'hardware' or self.motion_stage is not None:
                    return dict(output=['ACK'], state=state)
                # A mock run was cancelled above; still deliver STOP to the real STM32 below.
            stage, temporary = self._physical()
            try:
                if cmd == 'ping':
                    v = stage.ping()
                    output = [f'PONG proto {v[0]}, fw {v[1]}.{v[2]}.{v[3]}']
                elif cmd == 'status':
                    s = stage.status()
                    names = [n for bit, n in ((1, 'Y_HOMED'), (2, 'T_HOMED'), (4, 'Y-'), (8, 'Y+'), (16, 'T_SW'))
                             if s['flags'] & bit]
                    steps = s.get('steps', ('-', '-'))
                    output = [f"state  {self.STATES.get(s['state'], '?')}",
                              f"flags  0x{s['flags']:02x} [ {' '.join(names)} ]",
                              f"y      {s['position'][0]} (steps {steps[0]})",
                              f"t      {s['position'][1]} (steps {steps[1]})"]
                    if stage is self.stage:
                        self.device_status = s
                elif cmd == 'config':
                    c = stage.limits()
                    output = [f'y  {c[0]} ~ {c[1]}', f't  {c[2]} ~ {c[3]}']
                elif cmd == 'speed':
                    if self._working():
                        raise Problem('이동 중에는 속도를 바꿀 수 없습니다.', 409)
                    stage.speed(y, t)
                    output = ['ACK']
                else:
                    stage.stop()
                    output = ['ACK']
            finally:
                if temporary:
                    stage.close()
            return dict(output=output, state=self.public())

    def _physical_motion(self, key, kind, y=0, t=0):
        """Send HOME/MOVE to the real STM32 now (ACK/NAK in this request), wait DONE in a worker."""
        cmd_type = {'home': 2, 'move-rel': 3, 'move-abs': 4}[kind]
        with self.lock:
            if not self.store.claim_request(key, f'physical:{kind}:{y}:{t}'):
                return self.public()
            self.require_idle()
            temporary = self.stage.mode != 'hardware'
            stage = self.home_stage() if temporary else self.stage
            try:
                status = stage.status()
                if status['state'] in (1, 2):
                    raise Problem('STM32가 이동 중입니다. 먼저 정지하세요.', 409)
                if kind == 'home':
                    stage.home_y()
                else:
                    stage.start(y, t, relative=kind == 'move-rel', allow_t=True)
            except BaseException:
                if temporary:
                    stage.close()
                raise
            self.cancel.clear()
            self.stop_error = None
            self.motion_stage = stage
            label = 'Y축 원점 복귀' if kind == 'home' else f'STM32 {kind} {y} {t}'
            self.snapshot = dict(kind='manual_y', test=False, mode='hardware',
                                 scenario=dict(id=None, revision=0, name=label), settings=self.store.settings())
            self.state = dict(status='homing' if kind == 'home' else 'manual_move', trial=0, y_um=0, t_mdeg=0,
                              run_id=uuid.uuid4().hex, test=False, message=label + ' 중',
                              position=list(status['position']), history=[])
            self._record('ack', y_um=y, t_mdeg=t, relative=kind == 'move-rel', manual_action=kind)
            self._update()
            self._launch(self._await_physical, stage, temporary, kind, cmd_type)
            return self.public()

    def _await_physical(self, stage, temporary, kind, cmd_type):
        try:
            done = stage.wait_done(timeout_ms=125000 if kind == 'home' else 600000)
            with self.lock:
                self._record('done', **done)
                self._update(position=list(done['position']))
                if done['cmd_type'] != cmd_type:
                    raise Problem('DONE 응답의 명령 종류가 일치하지 않습니다.', 503)
                line = (f"DONE {self.RESULTS.get(done['result'], done['result'])} (cmd 0x{cmd_type:02x})  "
                        f"y={done['position'][0]} t={done['position'][1]}")
                if self.cancel.is_set() or done['result'] == 1:
                    self._finish_cancel()
                    return
                if done['result'] != 0:
                    raise Problem(line, 503)
                status = stage.status()
                if stage is self.stage:
                    self.device_status, self.device_limits = status, stage.limits()
                if status['state'] != 0 or (kind == 'home' and not status['flags'] & 1):
                    raise Problem('완료 후 원점·대기 상태 확인에 실패했습니다.', 503)
                self._update(status='home_done' if kind == 'home' else 'manual_done',
                             message='Y축 원점 복귀 완료' if kind == 'home' else line)
        finally:
            with self.lock:
                self.motion_stage = None
            if temporary:
                stage.close()

    def _active_stage(self):
        return self.motion_stage or self.stage

    def manual_y(self, key, action, y_um=0):
        if action not in ('home', 'jog'):
            raise Problem('지원하지 않는 Y축 동작입니다.')
        if action == 'home':
            return self.home_y(key)
        integer(y_um, 'Y 이동량(µm)', -1000, 1000)
        if y_um == 0:
            raise Problem('Y 이동량은 0일 수 없습니다.')
        with self.lock:
            if not self.store.claim_request(key, f'manual:{action}:{y_um}'):
                return self.public()
            self.require_idle()
            status, limits = self.stage.status(), self.stage.limits()
            if status['state'] in (1, 2):
                raise Problem('STM32가 이동 중입니다. 먼저 정지하세요.', 409)
            if status['position'][1] != 0:
                raise Problem('비활성 T축의 좌표를 확인하세요.', 409)
            if status['state'] != 0 or not status['flags'] & 1:
                raise Problem('Y 원점 복귀를 완료한 뒤 이동하세요.', 409)
            target = status['position'][0] + y_um
            if not limits[0] <= target <= limits[1]:
                raise Problem('Y 목표 위치가 STM32 허용 범위를 벗어납니다.', 409)
            self.cancel.clear()
            self.stop_error = None
            self.snapshot = dict(kind='manual_y', test=False, mode=self.stage.mode,
                                 scenario=dict(id=None, revision=0, name='Y축 수동 이동'),
                                 settings=self.store.settings())
            self.state = dict(status='manual_move', trial=0, y_um=0, t_mdeg=0,
                              run_id=uuid.uuid4().hex, test=False, message='Y축 수동 이동 중',
                              position=list(status['position']), history=[])
            self._update()
            self._launch(self._manual_y, y_um)
            return self.public()

    def _manual_y(self, y_um):
        with self.lock:
            if self.cancel.is_set():
                self._finish_cancel()
                return
            self.stage.start(y_um, 0, relative=True)
            self._record('ack', y_um=y_um, t_mdeg=0, relative=True, manual_action='jog')
            self._update()
        done = self.stage.wait_done(timeout_ms=30000)
        with self.lock:
            self._record('done', **done)
            self._update(position=list(done['position']))
            if done['cmd_type'] != 3:
                raise Problem('Y축 완료 응답의 명령 종류가 일치하지 않습니다.', 503)
            if self.cancel.is_set() or done['result'] == 1:
                self._finish_cancel()
                return
            if done['result'] != 0:
                raise Problem('Y축 동작 실패: ' + {2: '리밋 감지', 3: '장치 고장'}.get(done['result'], str(done['result'])), 503)
            status = self.stage.status()
            self.device_status = status
            self.device_limits = self.stage.limits()
            if status['state'] != 0 or not status['flags'] & 1:
                raise Problem('Y축 완료 후 원점·대기 상태 확인에 실패했습니다.', 503)
            self._update(status='manual_done', message='Y축 수동 이동 완료')

    def _update(self, **fields):
        self.state.update(fields)
        self.sequence += 1
        if self.snapshot:
            self.store.save_run(self.state['run_id'], self.snapshot, self.state)
        self.condition.notify_all()

    def _record(self, event, **values):
        self.state['history'].append(dict(at=now(), event=event, **values))

    def _working(self):
        return self.worker is not None and self.worker.is_alive()

    def _finish_cancel(self):
        # Cancellation of our queue does not prove the STOP reached the controller.
        if self.stop_error:
            self._update(status='error', message=self.stop_error)
        else:
            self._update(status='stopped', message='실행이 중단되었습니다.')

    def _check_ready(self):
        """Check again at command boundaries, including after settling delays.

        Call only outside ACK/DONE waits so queries cannot disrupt ACK retries.
        """
        if self.stage.mode == 'hardware' and not self.camera.health()['connected']:
            raise Problem('카메라 영상 연결이 없어 실행을 중단했습니다.', 409)
        try:
            status = self.stage.status()
        except Problem:
            self.connected = False
            raise
        self.connected = True
        self.device_status, self.device_error = status, ''
        if status['state'] != 0 or not status['flags'] & 1:
            raise Problem('STM32가 대기·Y 원점 설정 상태여야 합니다. 관리자에서 Y 원점 복귀를 실행하세요.', 409)
        if self.stage.mode == 'hardware' and status['position'][1] != 0:
            raise Problem('비활성 T축의 보고 좌표가 0이 아닙니다. 장치 상태를 확인하세요.', 409)
        return status

    def probe(self):
        with self.lock:
            if self._working():
                return self.public()
            try:
                status = self.stage.status()
                self.device_status = status
                self.device_limits = self.stage.limits()
                self.device_error = ''
                self.connected = True
                if self.state['status'] == 'ready' and (status['state'] != 0 or not status['flags'] & 1):
                    self._update(status='error', message='장치 상태 또는 원점이 변경되었습니다. 다시 준비하세요.')
            except Problem as exc:
                self.connected = False
                self.device_status = None
                self.device_limits = None
                self.device_error = str(exc)
                if self.state['status'] == 'ready':
                    self._update(status='error', message='STM32 연결이 끊겼습니다. 장치 상태를 확인하세요.')
            return self.public()

    def _preflight(self, scenario, settings):
        if self.stage.mode == 'hardware':
            if (scenario['start_t_mdeg'] != 0 or settings['center_t_mdeg'] != 0
                    or any(step['t_mdeg'] != 0 for step in scenario['steps'])):
                raise Problem('T축은 미장착 상태입니다. 실물 실행은 T=0 시나리오만 가능합니다.', 409)
        self._check_ready()
        validate_scenario(scenario, settings, self.stage.limits())

    def prepare(self, key, scenario_id=None):
        with self.lock:
            if not self.store.claim_request(key, f'prepare:{scenario_id or "demo"}'):
                return self.public()
            if self._working() or self.state['status'] == 'ready':
                raise Problem('진행 중인 실행을 먼저 완료하거나 정지하세요.', 409)
            settings = self.store.settings()
            if scenario_id:
                scenario = self.store.scenario(scenario_id)
            else:
                candidates = [s for s in self.store.scenarios() if s['active']
                              and s['verified_mode'] == self.stage.mode
                              and s['reference_revision'] == settings['revision']]
                if not candidates:
                    raise Problem('현재 모드에서 검증·활성화된 시나리오가 없습니다.', 409)
                scenario = candidates[self.cursor % len(candidates)]
            self._preflight(scenario, settings)
            if not scenario_id:
                self.cursor += 1
            self.cancel.clear()
            self.stop_error = None
            self.snapshot = dict(scenario=copy.deepcopy(scenario), settings=settings,
                                 mode=self.stage.mode, test=bool(scenario_id))
            self.state = dict(status='positioning', trial=0, y_um=0, t_mdeg=0,
                              run_id=uuid.uuid4().hex, message='시작 위치로 이동 중',
                              test=bool(scenario_id), position=None, history=[])
            self._update()
            self._launch(self._position)
            return self.public()

    def _launch(self, target, *args):
        self.worker = threading.Thread(target=self._guard, args=(target, *args), daemon=True,
                                       name='demo-motion')
        self.worker.start()

    def _guard(self, target, *args):
        try:
            target(*args)
        except Exception as exc:
            with self.lock:
                if isinstance(exc, Problem):
                    message = str(exc)
                else:
                    import logging
                    logging.exception('Motion execution failed')
                    message = '실행 오류가 발생했습니다. 관리자 로그와 장치 상태를 확인하세요.'
                self._record('error', message=message)
                self._update(status='error', message=self.stop_error or message)

    def _move(self, y, t, relative):
        with self.lock:
            if self.cancel.is_set():
                self._finish_cancel()
                return None
            self._check_ready()
            # Short critical section serializes STOP against sending/acknowledging a move.
            self.stage.start(y, t, relative=relative)
            self._record('ack', y_um=y, t_mdeg=t, relative=relative)
            if relative:
                self._update(trial=self.state['trial'] + 1, y_um=y, t_mdeg=t)
            else:
                self._update()
        done = self.stage.wait_done()
        with self.lock:
            self._record('done', **done)
            if done['cmd_type'] != (3 if relative else 4):
                raise Problem('이동 완료 응답의 명령 종류가 일치하지 않습니다.', 503)
            self._update(position=list(done['position']))
            if self.cancel.is_set() or done['result'] == 1:
                self._finish_cancel()
                return None
            if done['result'] != 0:
                raise Problem({2: '장치 리밋에 도달했습니다.', 3: 'STM32 고장 응답입니다.'}
                              .get(done['result'], '알 수 없는 이동 완료 결과입니다.'), 503)
            # A DONE cannot mask a controller reset or a stale camera feed.
            self._check_ready()
        return done['position']

    def _position(self):
        s = self.snapshot['scenario']
        position = self._move(s['start_y_um'], s['start_t_mdeg'], False)
        if position is None:
            return
        settings = self.snapshot['settings']
        target = {**settings, 'center_y_um': s['start_y_um'], 'center_t_mdeg': s['start_t_mdeg']}
        with self.lock:
            if self.cancel.is_set():
                self._finish_cancel()
            elif not at_center(position, target):
                raise Problem('지정한 시작 위치에 도달하지 못했습니다.', 409)
            else:
                self._update(status='ready', message='자동보정을 시작할 준비가 되었습니다.')

    def correct(self, key, single=False):
        with self.lock:
            if not self.store.claim_request(key, 'step' if single else 'correct'):
                return self.public()
            if self._working() or self.state['status'] != 'ready':
                raise Problem('시작 위치 이동 완료 후 보정을 실행하세요.', 409)
            if single and not self.state['test']:
                raise Problem('단계 실행은 관리자 시험에서만 가능합니다.', 409)
            self._preflight(self.snapshot['scenario'], self.snapshot['settings'])
            # Detect external moves while waiting at the start or between manual steps.
            position = self.stage.status()['position']
            expected = self.state['position']
            settings = {**self.snapshot['settings'], 'center_y_um': expected[0], 'center_t_mdeg': expected[1]}
            if not at_center(position, settings):
                raise Problem('준비 이후 장치 위치가 변경되었습니다. 정지 후 다시 준비하세요.', 409)
            self._update(status='correcting', message='중앙으로 보정 중')
            self._launch(self._correct, single)
            return self.public()

    def _correct(self, single):
        steps = self.snapshot['scenario']['steps']
        while True:
            with self.lock:
                if self.cancel.is_set():
                    self._finish_cancel()
                    return
                index = self.state['trial']
                if index >= 20 or index >= len(steps):
                    self._update(status='max_trials' if index >= 20 else 'exhausted',
                                 message='최대 시도 도달' if index >= 20 else '시나리오 종료 · 중앙 미도달')
                    return
                step = steps[index]
            position = self._move(step['y_um'], step['t_mdeg'], True)
            if position is None:
                return
            # Wait for the mechanical settling interval even on the final step.
            if self.cancel.wait(step['settle_ms'] / 1000):
                with self.lock:
                    self._finish_cancel()
                return
            with self.lock:
                if self.cancel.is_set():
                    self._finish_cancel()
                    return
                self._check_ready()
                if at_center(position, self.snapshot['settings']):
                    self._update(status='complete', message='보정 완료')
                    return
                if self.state['trial'] >= 20 or self.state['trial'] >= len(steps):
                    continue
                if single:
                    self._update(status='ready', message='현재 단계 완료 · 다음 보정 대기')
                    return

    def stop(self, key):
        with self.lock:
            if not self.store.claim_request(key, 'stop'):
                return self.public()
            self.cancel.set()
            if self._working():
                self._update(status='stopping', message='정지 요청 중')
                try:
                    self._active_stage().stop()
                except Problem as exc:
                    self.stop_error = '정지 전달 실패 · 실제 장치 정지 여부를 확인하세요.'
                    self._record('stop_error', message=str(exc))
                    self._update(status='error', message=self.stop_error)
                    raise
            elif self.state['status'] == 'ready' and self.stage.mode != 'hardware':
                self._update(status='stopped', message='실행이 중단되었습니다.')
            elif self.state['status'] == 'error' or self.stage.mode == 'hardware':
                self.stage.stop()
                status = self.stage.status()
                self.device_status = status
                self._update(status='stopped' if status['state'] == 0 else 'error',
                             message='장치 정지 확인됨' if status['state'] == 0 else '정지 요청 전송됨 · 장치 상태를 확인하세요.')
            return self.public()

    def close(self):
        with self.lock:
            self.cancel.set()
            if self._working():
                try:
                    self._active_stage().stop()
                except Problem:
                    pass
        if self.worker:
            self.worker.join(timeout=2)
        if not self._working():
            self.stage.close()

import copy
import json
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from werkzeug.security import generate_password_hash

from demo_web.app import create_app
from demo_web.camera import Camera
from demo_web.controller import Controller
from demo_web.stage import IoctlStage, MockStage
from demo_web.store import Problem, Store, validate_scenario


def key():
    return uuid.uuid4().hex


def fixture(count=5):
    return dict(name='시험 시나리오', description='private scenario', order=0,
                start_y_um=count * 100, start_t_mdeg=0,
                steps=[dict(y_um=-100, t_mdeg=0, settle_ms=0) for _ in range(count)])


class PhysicalStub(MockStage):
    """Stands in for the real driver used by Y HOME; tests never open /dev/robot_stage."""
    mode, transport = 'hardware', 'driver'

    def __init__(self, duration=.02):
        super().__init__(duration)
        self.closed = False
        self.homed = 0

    def home_y(self):
        self.homed += 1
        super().home_y()

    def close(self):
        self.closed = True


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / 'test.db')
        self.stage = MockStage(duration=.02)
        self.camera = Camera()
        self.physical = []
        self.engine = Controller(self.store, self.stage, self.camera, home_stage=self.open_physical)
        self.scenario = self.store.save(fixture())

    def open_physical(self, duration=.02):
        stage = PhysicalStub(duration)
        self.physical.append(stage)
        return stage

    def tearDown(self):
        self.engine.close()
        self.store.db.close()
        self.tmp.cleanup()

    def wait(self):
        worker = self.engine.worker
        if worker:
            worker.join(3)
            self.assertFalse(worker.is_alive(), 'execution hung')
        return self.engine.public()

    def prepare(self, scenario=None):
        self.engine.prepare(key(), (scenario or self.scenario)['id'])
        self.assertEqual(self.wait()['status'], 'ready')

    def complete(self):
        self.prepare()
        self.engine.correct(key())
        self.assertEqual(self.wait()['status'], 'complete')

    def test_random_zero_then_five_trials(self):
        self.prepare()
        self.assertEqual((self.engine.state['trial'], self.engine.state['y_um'], self.engine.state['t_mdeg']), (0, 0, 0))
        self.engine.correct(key())
        state = self.wait()
        self.assertEqual((state['status'], state['trial'], state['y_um']), ('complete', 5, -100))
        events = [item['event'] for item in self.engine.state['history']]
        self.assertEqual(events, ['ack', 'done'] * 6)

    def test_manual_hardware_y_without_camera_and_without_trials(self):
        self.stage.mode = 'hardware'  # Mock transport, physical policy; never opens hardware.
        self.engine.manual_y(key(), 'home')
        self.assertEqual(self.wait()['status'], 'home_done')
        self.assertEqual(self.physical, [], 'hardware mode reuses its own connection')
        request = key()
        self.engine.manual_y(request, 'jog', 100)
        self.assertEqual(self.wait()['status'], 'manual_done')
        self.engine.manual_y(request, 'jog', 100)
        self.assertEqual(self.stage.position, (100, 0))
        self.assertEqual(self.engine.state['trial'], 0)
        self.assertEqual(self.engine.state['history'][0]['y_um'], 100)
        with self.assertRaises(Problem):
            self.store.verify(self.scenario['id'], self.engine.state['run_id'], 'hardware')

    def test_manual_requires_home_and_enforces_distance_and_limits(self):
        with patch.object(self.stage, 'status', return_value=dict(state=0, flags=0, position=(0, 0))):
            with self.assertRaisesRegex(Problem, '원점'):
                self.engine.manual_y(key(), 'jog', 100)
        for distance in (0, 1001, -1001, True, 1.5):
            with self.assertRaises(Problem):
                self.engine.manual_y(key(), 'jog', distance)
        self.stage.position = (50000, 0)
        with self.assertRaisesRegex(Problem, '범위'):
            self.engine.manual_y(key(), 'jog', 1)
        self.stage.position = (0, 1)
        with self.assertRaisesRegex(Problem, 'T축'):
            self.engine.manual_y(key(), 'jog', 1)

    def test_home_in_mock_mode_drives_physical_stage_and_releases_it(self):
        with patch.object(self.stage, 'home_y') as mock_home:
            self.engine.home_y(key())
            self.assertEqual(self.wait()['status'], 'home_done')
            mock_home.assert_not_called()
        self.assertEqual(len(self.physical), 1)
        self.assertEqual(self.physical[0].homed, 1)
        self.assertTrue(self.physical[0].closed)
        self.assertIsNone(self.engine.motion_stage)
        self.assertEqual(self.engine.public()['mode'], 'mock')
        self.assertEqual([item['event'] for item in self.engine.state['history']], ['ack', 'done'])
        self.assertEqual(self.engine.state['trial'], 0)

    def test_stop_during_home_reaches_the_physical_stage(self):
        self.engine.home_stage = lambda: self.open_physical(.3)
        self.engine.home_y(key())
        self.engine.stop(key())
        self.assertEqual(self.wait()['status'], 'stopped')
        self.assertTrue(self.physical[0].stopped)
        self.assertTrue(self.physical[0].closed)

    def test_home_connection_failure_releases_device(self):
        def broken():
            stage = self.open_physical()
            stage.status = lambda: (_ for _ in ()).throw(Problem('권한 없음', 503))
            return stage
        self.engine.home_stage = broken
        with self.assertRaisesRegex(Problem, '권한'):
            self.engine.home_y(key())
        self.assertTrue(self.physical[0].closed)
        self.assertIsNone(self.engine.worker)

    def test_console_queries_and_speed_use_physical_stage(self):
        out = self.engine.console(key(), 'ping')['output']
        self.assertEqual(out, ['PONG proto 1, fw 0.0.0'])
        self.assertIn('state  IDLE', self.engine.console(key(), 'status')['output'])
        self.assertEqual(self.engine.console(key(), 'config')['output'], ['y  -50000 ~ 50000', 't  -10000 ~ 10000'])
        self.assertEqual(self.engine.console(key(), 'speed', 1000, 0)['output'], ['ACK'])
        self.assertEqual(self.physical[-1].speeds, (1000, 0))
        self.assertTrue(all(stage.closed for stage in self.physical))
        self.assertFalse(hasattr(self.stage, 'speeds'), 'mock scenario stage untouched')
        for bad in (('move-rel', 1.5, 0), ('speed', -1, 0), ('rm', 0, 0)):
            with self.assertRaises(Problem):
                self.engine.console(key(), *bad)

    def test_console_move_passes_t_through_and_reports_done(self):
        result = self.engine.console(key(), 'move-rel', 1000, 250)
        self.assertEqual(result['output'], ['ACK, waiting DONE...'])
        self.assertEqual(self.wait()['status'], 'manual_done')
        self.assertEqual(self.engine.state['message'], 'DONE OK (cmd 0x03)  y=1000 t=250')
        self.assertEqual(self.physical[0].position, (1000, 250))
        self.assertTrue(self.physical[0].closed)
        self.assertEqual(self.stage.position, (0, 0))

    def test_console_nak_is_immediate_and_releases_device(self):
        def nak():
            stage = self.open_physical()
            stage.start = lambda *a, **k: (_ for _ in ()).throw(Problem('STM32 명령 거부: 원점 미설정', 409))
            return stage
        self.engine.home_stage = nak
        with self.assertRaisesRegex(Problem, '원점 미설정'):
            self.engine.console(key(), 'move-abs', 5000, 0)
        self.assertTrue(self.physical[0].closed)
        self.assertIsNone(self.engine.worker)
        self.assertEqual(self.engine.public()['status'], 'idle')

    def test_console_stop_reaches_physical_move(self):
        self.engine.home_stage = lambda: self.open_physical(.3)
        self.engine.console(key(), 'move-rel', 1000, 0)
        self.assertEqual(self.engine.console(key(), 'stop')['output'], ['ACK'])
        self.assertEqual(self.wait()['status'], 'stopped')
        self.assertTrue(self.physical[0].stopped)

    def test_home_rejected_while_controller_busy(self):
        self.engine.home_stage = lambda: self.open_physical()
        with patch.object(PhysicalStub, 'status', return_value=dict(state=1, flags=0, position=(0, 0))):
            with self.assertRaisesRegex(Problem, '이동 중'):
                self.engine.home_y(key())
        self.assertTrue(self.physical[0].closed)

    def test_ready_scenario_blocks_manual_and_mode_changes(self):
        self.prepare()
        with self.assertRaises(Problem):
            self.engine.manual_y(key(), 'home')
        with self.assertRaises(Problem):
            self.engine.configure_mode('mock')

    def test_manual_stop_and_busy_rejection(self):
        self.stage.duration = .2
        self.engine.manual_y(key(), 'jog', 100)
        with self.assertRaises(Problem):
            self.engine.configure_mode('mock')
        self.engine.stop(key())
        self.assertEqual(self.wait()['status'], 'stopped')
        self.assertEqual(self.engine.state['trial'], 0)

    def test_mode_change_reads_only_and_connection_failure_recovers(self):
        candidate = MockStage(.001)
        candidate.mode, candidate.transport = 'hardware', 'serial'
        with patch('demo_web.controller.make_stage', return_value=candidate), patch.object(candidate, 'start') as start, patch.object(candidate, 'home_y') as home:
            self.engine.configure_mode('hardware')
            start.assert_not_called()
            home.assert_not_called()
        self.assertEqual(self.engine.admin_state()['device_status']['position'], (0, 0))
        with patch.object(candidate, 'status', side_effect=Problem('USB disconnected')):
            self.engine.configure_mode('mock')
        self.assertEqual(self.engine.public()['mode'], 'mock')

    def test_external_motion_can_be_stopped_without_worker(self):
        self.stage.mode = 'hardware'
        with patch.object(self.stage, 'stop') as stop:
            self.engine.stop(key())
            stop.assert_called_once()
        self.assertEqual(self.engine.state['status'], 'stopped')

    def test_public_state_hides_start_and_future(self):
        self.prepare()
        public = self.engine.public()
        self.assertFalse({'snapshot', 'position', 'history', 'steps', 'scenario'} & public.keys())
        self.assertNotIn('private scenario', json.dumps(public))

    def test_duplicate_request_never_executes_twice(self):
        request = key()
        self.engine.prepare(request, self.scenario['id'])
        self.engine.prepare(request, self.scenario['id'])
        self.wait()
        correction_key = key()
        self.engine.correct(correction_key)
        self.wait()
        self.engine.correct(correction_key)
        self.assertEqual(self.engine.state['trial'], 5)
        self.assertEqual(len(self.engine.state['history']), 12)

    def test_same_key_different_action_rejected(self):
        request = key()
        self.engine.prepare(request, self.scenario['id'])
        self.wait()
        with self.assertRaises(Problem):
            self.engine.correct(request)

    def test_concurrent_random_only_one_motion(self):
        results = []
        def run():
            try:
                self.engine.prepare(key(), self.scenario['id'])
                results.append(True)
            except Problem:
                results.append(False)
        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.wait()
        self.assertEqual(results.count(True), 1)

    def test_single_step_and_stop(self):
        self.prepare()
        self.engine.correct(key(), single=True)
        self.assertEqual(self.wait()['trial'], 1)
        self.assertEqual(self.engine.state['status'], 'ready')
        self.engine.stop(key())
        self.assertEqual(self.engine.state['status'], 'stopped')
        self.engine.prepare(key(), self.scenario['id'])
        self.wait()
        self.assertEqual(self.engine.state['trial'], 0)

    def test_stop_during_move_blocks_following_steps(self):
        self.prepare()
        self.stage.duration = .2
        self.engine.correct(key())
        deadline = time.monotonic() + 1
        while self.engine.public()['trial'] == 0 and time.monotonic() < deadline:
            time.sleep(.005)
        self.engine.stop(key())
        result = self.wait()
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['trial'], 1)

    def test_stop_before_start_not_stuck(self):
        with self.engine.lock:
            self.engine.prepare(key(), self.scenario['id'])
            self.engine.stop(key())
        self.assertEqual(self.wait()['status'], 'stopped')
        self.assertEqual(self.engine.state['history'], [])

    def camera_loss_during_settle(self, count):
        scenario = self.store.save(fixture(count))
        live = {'connected': True}
        self.stage.mode = 'hardware'

        def lose_camera(_timeout):
            live['connected'] = False
            return False

        with patch.object(self.camera, 'health', side_effect=lambda: dict(live)):
            self.prepare(scenario)
            with patch.object(self.engine.cancel, 'wait', side_effect=lose_camera):
                self.engine.correct(key())
                state = self.wait()
            self.assertEqual(state['status'], 'error')
            self.assertEqual(state['trial'], 1)
            self.assertIn('카메라', state['message'])
            with self.assertRaises(Problem):
                self.store.verify(scenario['id'], state['run_id'], 'hardware')

    def test_camera_loss_during_settle_blocks_next_move(self):
        self.camera_loss_during_settle(3)

    def test_camera_loss_during_final_settle_blocks_success(self):
        self.camera_loss_during_settle(1)

    def test_home_lost_during_final_settle_blocks_success(self):
        self.prepare(self.store.save(fixture(1)))
        original_status = self.stage.status
        reset = {'occurred': False}

        def status():
            result = original_status()
            if reset['occurred']:
                result['flags'] = 0
            return result

        def reset_controller(_timeout):
            reset['occurred'] = True
            return False

        with patch.object(self.stage, 'status', side_effect=status):
            with patch.object(self.engine.cancel, 'wait', side_effect=reset_controller):
                self.engine.correct(key())
                state = self.wait()
        self.assertEqual(state['status'], 'error')
        self.assertEqual(state['trial'], 1)

    def test_failed_stop_is_not_replaced_by_stopped(self):
        self.prepare()
        waiting, release = threading.Event(), threading.Event()
        original_done = self.stage.wait_done

        def delayed_done():
            waiting.set()
            if not release.wait(2):
                raise Problem('test timeout')
            return original_done()

        with patch.object(self.stage, 'wait_done', side_effect=delayed_done):
            self.engine.correct(key())
            try:
                self.assertTrue(waiting.wait(1))
                with patch.object(self.stage, 'stop', side_effect=Problem('STOP timeout')):
                    with self.assertRaises(Problem):
                        self.engine.stop(key())
            finally:
                release.set()
            result = self.wait()
        self.assertEqual(result['status'], 'error')
        self.assertIn('정지 전달 실패', result['message'])
        self.assertEqual(result['trial'], 1)
        self.assertTrue(any(e['event'] == 'stop_error' for e in self.engine.state['history']))

    def test_exactly_twenty_success(self):
        s = self.store.save(fixture(20))
        self.prepare(s)
        self.engine.correct(key())
        result = self.wait()
        self.assertEqual((result['status'], result['trial']), ('complete', 20))

    def test_twenty_missed_not_success(self):
        s = self.store.save(fixture(20))
        self.prepare(s)
        original = self.stage.wait_done
        def drift():
            result = original()
            result['position'] = (5000, 0)
            return result
        self.stage.wait_done = drift
        self.engine.correct(key())
        result = self.wait()
        self.assertEqual((result['status'], result['trial']), ('max_trials', 20))
        self.assertEqual(len([e for e in self.engine.state['history'] if e['event'] == 'ack']), 21)

    def test_early_center_stops_remaining_steps(self):
        s = fixture(3)
        s['start_y_um'] = 100
        s['steps'][1]['y_um'] = 100
        self.prepare(self.store.save(s))
        self.engine.correct(key())
        result = self.wait()
        self.assertEqual((result['status'], result['trial']), ('complete', 1))

    def test_nak_does_not_count(self):
        self.prepare()
        with patch.object(self.stage, 'start', side_effect=Problem('NAK')):
            self.engine.correct(key())
            result = self.wait()
        self.assertEqual((result['status'], result['trial'], result['y_um']), ('error', 0, 0))

    def test_done_timeout_never_reissues_move(self):
        self.prepare()
        with patch.object(self.stage, 'start', wraps=self.stage.start) as start:
            with patch.object(self.stage, 'wait_done', side_effect=Problem('timeout')):
                self.engine.correct(key())
                result = self.wait()
            self.assertEqual(start.call_count, 1)
        self.assertEqual((result['status'], result['trial']), ('error', 1))

    def test_wrong_done_type_and_fault_block_next_move(self):
        self.prepare()
        for result in (dict(cmd_type=4, result=0, position=(400, 0)),
                       dict(cmd_type=3, result=3, position=(400, 0))):
            self.engine.state['status'] = 'ready'
            self.stage.pending = None
            with patch.object(self.stage, 'wait_done', return_value=result):
                self.engine.correct(key())
                self.assertEqual(self.wait()['status'], 'error')
            self.engine.state['trial'] = 0
            self.engine.state['position'] = [500, 0]

    def test_snapshot_unchanged_on_edit(self):
        self.prepare()
        s = self.store.scenario(self.scenario['id'])
        s['name'] = '새 버전'
        self.store.save(s, s['id'])
        self.engine.correct(key())
        self.wait()
        self.assertEqual(self.engine.snapshot['scenario']['name'], '시험 시나리오')
        with self.assertRaises(Problem):
            self.store.verify(s['id'], self.engine.state['run_id'], 'mock')

    def test_verification_activation_and_reference_invalidation(self):
        self.complete()
        rid = self.engine.state['run_id']
        self.store.verify(self.scenario['id'], rid, 'mock')
        with self.assertRaises(Problem):
            self.store.activate(self.scenario['id'], True, 'hardware')
        self.store.activate(self.scenario['id'], True, 'mock')
        self.engine.prepare(key())
        self.wait()
        self.assertFalse(self.engine.state['test'])
        self.engine.stop(key())
        settings = self.store.settings()
        self.store.set_settings(settings)
        self.assertFalse(self.store.scenario(self.scenario['id'])['active'])
        with self.assertRaises(Problem):
            self.engine.prepare(key())

    def test_unverified_not_in_demo_playlist(self):
        with self.assertRaises(Problem):
            self.engine.prepare(key())

    def test_restarting_aborts_pending_run(self):
        self.prepare()
        rid = self.engine.state['run_id']
        other = Store(Path(self.tmp.name) / 'test.db')
        self.assertEqual(other.run(rid)['state']['status'], 'interrupted')
        other.db.close()

    def test_hardware_theta_rejected_before_any_io(self):
        data = fixture(1)
        data['start_t_mdeg'] = 100
        data['steps'][0]['t_mdeg'] = -100
        s = self.store.save(data)
        self.stage.mode = 'hardware'
        with patch.object(self.stage, 'status') as status:
            with self.assertRaisesRegex(Problem, 'T축'):
                self.engine.prepare(key(), s['id'])
            status.assert_not_called()

    def test_hardware_camera_required(self):
        self.stage.mode = 'hardware'
        with self.assertRaisesRegex(Problem, '카메라'):
            self.engine.prepare(key(), self.scenario['id'])

    def test_validation_rejects_bad_paths_and_counts(self):
        for data in [fixture(0), fixture(21), {**fixture(), 'start_y_um': True},
                     {**fixture(), 'start_y_um': float('nan')}, {**fixture(), 'steps':[dict(y_um=0,t_mdeg=0)]},
                     {**fixture(), 'start_y_um':999}]:
            with self.assertRaises(Problem):
                self.store.save(data)
        with self.assertRaises(Problem):
            validate_scenario(fixture(), self.store.settings(), (0, 400, 0, 0))

    def test_version_conflict_and_delete(self):
        s = self.store.scenario(self.scenario['id'])
        self.store.save(s, s['id'])
        with self.assertRaises(Problem):
            self.store.save(s, s['id'])
        self.store.delete(s['id'])
        with self.assertRaises(Problem):
            self.store.scenario(s['id'])


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.physical = []
        self.app = create_app(self.tmp.name, stage=MockStage(.005),
                              home_stage=lambda: self.physical.append(PhysicalStub(.005)) or self.physical[-1])
        self.store = self.app.extensions['store']
        self.engine = self.app.extensions['controller']
        self.store.meta('password', generate_password_hash('test-password-123', method='pbkdf2:sha256:1000'))
        self.client = self.app.test_client()
        self.csrf = self.client.get('/api/session').json['csrf']

    def tearDown(self):
        self.engine.close()
        self.store.db.close()
        self.tmp.cleanup()

    def write(self, path, data=None, method='POST', token=True):
        headers = {'Idempotency-Key':key()}
        if token:
            headers['X-CSRF-Token'] = self.csrf
        return self.client.open(path, method=method, json=data or {}, headers=headers)

    def login(self):
        self.assertEqual(self.write('/api/admin/login', {'password':'test-password-123'}).status_code, 200)
        self.csrf = self.client.get('/api/session').json['csrf']

    def test_pages_and_api_auth(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/admin').status_code, 302)
        for path in ['/api/admin/scenarios','/api/admin/settings','/api/admin/export','/api/admin/state','/api/admin/runs','/api/admin/device','/api/admin/camera']:
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.write('/api/demo/random').status_code, 401)
        self.login()
        self.assertEqual(self.client.get('/admin').status_code, 200)
        self.assertEqual(self.client.get('/api/admin/settings').status_code, 200)

    def test_manual_endpoints_auth_csrf_and_execution(self):
        self.assertEqual(self.write('/api/admin/y/home').status_code, 401)
        self.assertEqual(self.write('/api/admin/device', {'mode':'mock'}, 'PUT').status_code, 401)
        self.login()
        self.assertEqual(self.write('/api/admin/y/home', token=False).status_code, 403)
        self.assertEqual(self.write('/api/admin/device', {'mode':'invalid'}, 'PUT').status_code, 400)
        self.assertEqual(self.write('/api/admin/y/home').status_code, 200)
        self.engine.worker.join(2)
        self.assertEqual(self.engine.public()['status'], 'home_done')
        self.assertEqual(self.write('/api/admin/y/jog', {'y_um':100}).status_code, 200)
        self.engine.worker.join(2)
        self.assertEqual(self.engine.stage.position, (100, 0))
        self.assertEqual(self.engine.state['trial'], 0)

    def test_demo_home_endpoint_requires_operator_and_drives_physical_stage(self):
        self.assertEqual(self.write('/api/demo/home').status_code, 401)
        self.login()
        self.assertEqual(self.write('/api/demo/home', token=False).status_code, 403)
        self.assertEqual(self.write('/api/demo/home').status_code, 200)
        self.engine.worker.join(2)
        self.assertEqual(self.engine.public()['status'], 'home_done')
        self.assertEqual((len(self.physical), self.physical[0].homed, self.physical[0].closed), (1, 1, True))

    def test_stm32_console_endpoint_requires_admin(self):
        self.assertEqual(self.write('/api/admin/stm32', {'cmd':'ping'}).status_code, 401)
        self.login()
        self.assertEqual(self.write('/api/admin/stm32', {'cmd':'ping'}, token=False).status_code, 403)
        response = self.write('/api/admin/stm32', {'cmd':'ping'})
        self.assertEqual((response.status_code, response.json['output']), (200, ['PONG proto 1, fw 0.0.0']))
        response = self.write('/api/admin/stm32', {'cmd':'move-rel', 'y':500, 't':0})
        self.assertEqual(response.status_code, 200)
        self.engine.worker.join(2)
        self.assertEqual(self.engine.public()['status'], 'manual_done')

    def test_camera_setting_persists_and_invalidates_scenarios(self):
        self.login()
        s = self.store.save(fixture())
        self.engine.prepare(key(), s['id'])
        self.engine.worker.join(2)
        self.assertEqual(self.write('/api/admin/camera', {'profile':'preview','enabled':False}, 'PUT').status_code, 409)
        self.engine.correct(key())
        self.engine.worker.join(2)
        self.store.verify(s['id'], self.engine.state['run_id'], 'mock')
        self.store.activate(s['id'], True, 'mock')
        revision = self.store.settings()['revision']
        response = self.write('/api/admin/camera', {'profile':'preview','enabled':False}, 'PUT')
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.json['width'], response.json['height']), (960,720))
        self.assertEqual(self.store.settings()['revision'], revision+1)
        self.assertFalse(self.store.scenario(s['id'])['active'])
        self.assertIsNone(self.store.scenario(s['id'])['verified_mode'])
        self.assertEqual(json.loads(self.store.meta('camera_config')), {'profile':'preview','enabled':False,'undistort':True})
        self.assertEqual(self.write('/api/admin/camera', {'profile':'4k','enabled':False}, 'PUT').status_code, 400)

    def test_correction_toggle_invalidates_verified_scenario_and_persists(self):
        self.test_full_admin_to_demo_workflow()
        self.engine.stop(key())
        scenario = self.store.scenarios()[0]
        self.assertTrue(scenario['active'])
        revision = self.store.settings()['revision']
        data = dict(profile='full', enabled=False, undistort=False)
        self.assertEqual(self.write('/api/admin/camera', data, 'PUT', token=False).status_code, 403)
        self.assertEqual(self.write('/api/admin/camera', {**data, 'undistort':'false'}, 'PUT').status_code, 400)
        self.assertEqual(self.store.settings()['revision'], revision)
        response = self.write('/api/admin/camera', data, 'PUT')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json['undistort'])
        self.assertFalse(self.store.scenario(scenario['id'])['active'])
        self.assertIsNone(self.store.scenario(scenario['id'])['verified_mode'])
        self.assertEqual(self.store.settings()['revision'], revision+1)
        self.write('/api/admin/camera', data, 'PUT')
        self.assertEqual(self.store.settings()['revision'], revision+1)
        reopened = create_app(self.tmp.name)
        try:
            self.assertFalse(reopened.extensions['camera'].undistort)
            self.assertEqual(reopened.extensions['store'].settings()['revision'], revision+1)
        finally:
            reopened.extensions['controller'].close()
            reopened.extensions['store'].db.close()

    def test_old_uncorrected_reference_is_invalidated_once_on_upgrade(self):
        self.test_full_admin_to_demo_workflow()
        self.engine.stop(key())
        scenario = self.store.scenarios()[0]
        self.store.meta('camera_reference', '')  # Previous release has no signature.
        self.store.meta('camera_config', json.dumps(dict(profile='full', enabled=False)))
        revision = self.store.settings()['revision']
        for _ in range(2):
            reopened = create_app(self.tmp.name)
            try:
                store = reopened.extensions['store']
                self.assertTrue(reopened.extensions['camera'].undistort)
                self.assertEqual(store.settings()['revision'], revision+1)
                self.assertFalse(store.scenario(scenario['id'])['active'])
            finally:
                reopened.extensions['controller'].close()
                reopened.extensions['store'].db.close()

    def test_csrf_and_logout(self):
        self.assertEqual(self.write('/api/admin/login', token=False).status_code, 403)
        self.login()
        self.assertEqual(self.write('/api/admin/scenarios', fixture(), token=False).status_code, 403)
        self.assertEqual(self.write('/api/admin/logout').status_code, 200)
        self.assertEqual(self.client.get('/api/admin/scenarios').status_code, 401)

    def test_crud_import_export_and_no_public_leak(self):
        self.login()
        response = self.write('/api/admin/scenarios', fixture())
        self.assertEqual(response.status_code, 201)
        s = response.json
        self.assertNotIn('private scenario', self.client.get('/').text)
        export = self.client.get('/api/admin/export').json
        self.assertEqual(self.write('/api/admin/import', export).json['imported'], 1)
        self.assertEqual(len(self.store.scenarios()), 2)
        bad = copy.deepcopy(export)
        bad['scenarios'].append(fixture(21))
        self.assertEqual(self.write('/api/admin/import', bad).status_code, 400)
        self.assertEqual(len(self.store.scenarios()), 2)
        s['name'] = '수정'
        self.assertEqual(self.write('/api/admin/scenarios/'+s['id'], s, 'PUT').status_code, 200)
        self.assertEqual(self.write('/api/admin/scenarios/'+s['id'], method='DELETE').status_code, 200)

    def test_full_admin_to_demo_workflow(self):
        self.login()
        s = self.write('/api/admin/scenarios', fixture()).json
        path = '/api/admin/scenarios/'+s['id']
        self.assertEqual(self.write(path+'/activate', {'active':True}).status_code, 409)
        self.assertEqual(self.write(path+'/test').status_code, 200)
        self.engine.worker.join(2)
        self.assertEqual(self.write('/api/demo/correct').status_code, 200)
        self.engine.worker.join(2)
        run = self.engine.public()
        self.assertEqual(run['status'], 'complete')
        self.assertEqual(self.write(path+'/verify', {'run_id':run['run_id'],'confirmed':True}).status_code, 200)
        self.assertEqual(self.write(path+'/activate', {'active':True}).status_code, 200)
        self.assertEqual(self.write('/api/demo/random').status_code, 200)
        self.engine.worker.join(2)
        public = self.client.get('/api/state').json
        self.assertEqual(public['trial'], 0)
        self.assertEqual(public['y_um'], 0)
        self.assertFalse(public['test'])

    def test_security_headers_and_offline_camera(self):
        response = self.client.get('/')
        self.assertIn("script-src 'self'", response.headers['Content-Security-Policy'])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(self.client.get('/camera/stream.mjpg').status_code, 503)
        self.assertEqual(self.client.get('/api/admin/missing').status_code, 404)


class HardwareTests(unittest.TestCase):
    def test_theta_guard_without_opening_device(self):
        stage = IoctlStage('/does-not-exist')
        with self.assertRaisesRegex(Problem, 'T축'):
            stage.start(0, 1)
        with self.assertRaisesRegex(Problem, '연결'):
            stage.status()

    def test_timeout_queries_status_without_resending(self):
        stage = IoctlStage()
        with patch.object(stage, '_call', side_effect=Problem('timeout')) as call:
            with self.assertRaises(Problem):
                stage.wait_done()
        self.assertEqual([c.args[0] for c in call.call_args_list], [9, 6])

    def test_driver_ack_then_done_contract(self):
        stage = IoctlStage()
        with patch.object(stage, '_call', side_effect=[(100,0,0), (30000,0,3,100,0)]) as call:
            stage.start(100,0,relative=True)
            done = stage.wait_done()
        self.assertEqual(done, dict(result=0,cmd_type=3,position=(100,0)))
        self.assertEqual([c.args[0] for c in call.call_args_list], [3,9])


class CameraTests(unittest.TestCase):
    def test_reconfigure_closes_old_stream_and_clears_stale_frame(self):
        camera = Camera()
        camera.write(b'old-frame')
        stream = camera.stream()
        self.assertIn(b'old-frame', next(stream))
        generation = camera.health()['generation']
        camera.configure('medium', False)
        self.assertEqual(camera.health()['generation'], generation+1)
        self.assertFalse(camera.health()['connected'])
        self.assertEqual(list(stream), [])
        self.assertIsNone(camera.frame)
        self.assertEqual(camera.health()['width'], 2028)

    def test_stale_frame_is_not_reported_as_live(self):
        camera = Camera(True)
        with patch('demo_web.camera.time.monotonic', return_value=100):
            camera.write(b'jpeg-test-frame')
            self.assertTrue(camera.health()['connected'])
        with patch('demo_web.camera.time.monotonic', return_value=104):
            self.assertFalse(camera.health()['connected'])
            self.assertEqual(list(camera.stream()), [])
        camera.stop()

    def test_frame_is_shared_without_disk_recording(self):
        camera = Camera(True)
        camera.write(b'jpeg-test-frame')
        a, b = camera.stream(), camera.stream()
        self.assertIn(b'jpeg-test-frame', next(a))
        self.assertIn(b'jpeg-test-frame', next(b))
        a.close()
        b.close()
        camera.stop()


if __name__ == '__main__':
    unittest.main()

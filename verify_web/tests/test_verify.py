"""장치 없이 도는 시험: python -m unittest discover -s verify_web/tests -t ."""
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from demo_web.stage import MockStage
from verify_web.algorithm import AlgorithmChoices, AlgorithmError, MockAlgorithm, WorkerAlgorithm
from verify_web.app import create_app
from verify_web.camera import FakeCamera
from verify_web.devices import DemoLauncher, DeviceManager
from verify_web.geometry import ALG_SIZE, SEMES_FX, SEMES_FY, Calibration, Rectifier, ViewMapper, alg_matrix
from verify_web.runner import Runner, RunError, Settings

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / 'demo_web' / 'calibration' / 'camera_calib.yml'
WORKER = ROOT / 'verify_web' / 'native' / 'build' / 'align_worker'


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class FixedAlgorithm:
    """항상 같은 값을 돌려주는 알고리즘 (한도 시험용)."""
    kind, output = 'fixed', 'offset'

    def __init__(self, y, t=0.0):
        self.v = (y, t)

    def infer(self, image, truth=None):
        return dict(y=self.v[0], t=self.v[1], ms=0.1)

    def status(self):
        return dict(kind=self.kind, output=self.output, ready=True)

    def close(self):
        pass


class GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = Calibration.load(CALIB)

    def test_alg_matrix_matches_cpp_rectifier(self):
        K = alg_matrix(1.0)
        self.assertAlmostEqual(K[0, 0], 12 / 2.8e-3 * 3840 / 4096)
        self.assertAlmostEqual(K[1, 1], 12 / 2.8e-3)
        self.assertEqual((K[0, 2], K[1, 2]), (1919.5, 1079.5))
        self.assertAlmostEqual(alg_matrix(1.1)[0, 0], SEMES_FX * 1.1)

    def test_view_is_downscaled_alg_image(self):
        rng = np.random.default_rng(1)
        w, h = self.calib.size
        # 부드러운 무늬 (보간 차이가 작게)
        small = rng.integers(0, 255, (h // 64, w // 64), dtype=np.uint8)
        gray = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        alg = Rectifier(self.calib).apply(gray)
        self.assertEqual((alg.shape[1], alg.shape[0]), ALG_SIZE)
        view = ViewMapper(self.calib)
        v = view.apply(gray)
        ref = cv2.resize(alg, (1280, 720), interpolation=cv2.INTER_AREA)
        diff = np.abs(v.astype(int) - ref.astype(int))[20:-20, 20:-20]
        self.assertLess(diff.mean(), 3.0)

    def test_roll_rotates_clockwise(self):
        """roll_deg > 0 이면 결과 사진이 시계 방향(오른쪽이 내려감)으로 돈다. 가로세로 비 때문에 약 1.067배."""
        import dataclasses
        w, h = self.calib.size
        img = np.zeros((h, w), np.uint8)
        cy = int(round(self.calib.K[1, 2]))
        cv2.line(img, (1000, cy), (3200, cy), 255, 9)
        ys = []
        for roll in (0.0, 1.0):
            out = Rectifier(dataclasses.replace(self.calib, roll_deg=roll)).apply(img)
            ys.append([int(np.argmax(out[:, x])) for x in (1200, 2700)])
        self.assertLessEqual(abs(ys[0][1] - ys[0][0]), 1)
        ang = np.degrees(np.arctan2(ys[1][1] - ys[1][0], 1500))
        self.assertAlmostEqual(ang, 1.0 * SEMES_FY / SEMES_FX, delta=0.08)

    def test_view_zoom_keeps_plate_point_centered(self):
        view = ViewMapper(self.calib)
        view.set_view(4.0, (1907.0, 1065.0))
        x, y = view.to_view(1907.0, 1065.0)
        self.assertAlmostEqual(x, 639.5, places=6)
        self.assertAlmostEqual(y, 359.5, places=6)
        # 4배면 알고리즘 3px 가 화면 4px
        x2, _ = view.to_view(1910.0, 1065.0)
        self.assertAlmostEqual(x2 - x, 4.0, places=6)


class Devices:
    """DeviceManager 대용: 이미 열린 가짜 장치."""

    def __init__(self, camera, stage):
        self.camera, self.stage, self.ready, self.error = camera, stage, True, None

    def status(self):
        return dict(ready=self.ready, error=None, stage_mode='mock', stage_transport='mock')


class RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = Calibration.load(CALIB)
        cls.rect = Rectifier(cls.calib)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camera = FakeCamera(ViewMapper(self.calib), fps=20).start()
        self.stage = MockStage(duration=0.05)
        self.settings = Settings(Path(self.tmp.name) / 's.json', dict(save_images=False, settle_timeout_s=2.0, demo_slow=False))

    def tearDown(self):
        self.camera.close()
        self.tmp.cleanup()

    def runner(self, algo):
        return Runner(Devices(self.camera, self.stage), algo, self.rect, self.settings, Path(self.tmp.name) / 'runs')

    def test_random_start_is_far_enough_and_both_signs(self):
        r = self.runner(MockAlgorithm())
        ys, ts = zip(*(r._resolve_offset('random') for _ in range(200)))
        ry, rt = self.settings['random_y_mm'], self.settings['random_t_deg']
        self.assertTrue(all(0.4 * ry - 1e-3 <= abs(y) <= ry for y in ys))
        self.assertTrue(all(0.4 * rt - 1e-3 <= abs(t) <= rt for t in ts))
        self.assertTrue(min(ys) < 0 < max(ys) and min(ts) < 0 < max(ts))

    def test_sequential_y_then_theta_in_one_trial(self):
        """한 회차 = 한 번 측정. 산출한 두 축 보정량으로 Y 를 먼저 옮기고 이어서 Θ 를 옮긴다 (Θ 이동 전에 다시 재지 않음)."""
        moves, stages = [], []
        orig = self.stage.start
        def logged(y, t, relative=False, allow_t=False):
            moves.append((y, t)); stages.append(r.state.get('axis_stage')); return orig(y, t, relative=relative, allow_t=allow_t)
        self.stage.start = logged
        self.settings.update(dict(auto_reference=False, sequential_axes=True))
        r = self.runner(FixedAlgorithm(0.3, 0.05))          # 명령 (−0.3 mm, −0.05°)
        r.step(); r.wait_idle(10)
        s = r.snapshot()
        self.assertEqual(len(s['history']), 1)                         # 측정은 한 번
        self.assertEqual(s['history'][0]['axis'], 'y_then_t')
        # Y 만 (접근점 → 목표), 그다음 Θ 만 (접근점 → 목표)
        self.assertEqual(moves, [(-500, 0), (-300, 0), (-300, -150), (-300, -50)], s['message'])
        self.assertEqual(stages, ['y', 'y', 't', 't'])
        self.assertIsNone(s['axis_stage'])
        self.assertEqual(self.stage.position, (-300, -50))

    def test_sequential_converges_like_together(self):
        for seq in (True, False):
            self.settings.update(dict(sequential_axes=seq))
            r = self.runner(MockAlgorithm(noise_y_mm=0.002, noise_t_deg=0.0005, seed=4))
            r.run((0.8, 0.3)); r.wait_idle(60)
            s = r.snapshot()
            self.assertEqual(s['run']['result'], 'converged', s['message'])
            self.assertLessEqual(s['run']['trial'], 4)
            self.assertEqual(s['history'][0]['axis'], 'y_then_t' if seq else 'both')

    def test_fixed_zero_algorithm_reference_is_display_only(self):
        class Fixed(MockAlgorithm):
            supports_reference = False
            def set_reference(self, image):
                raise AssertionError('호출하면 안 됨')
        r = self.runner(Fixed())
        r.reference(); r.wait_idle(30)
        s = r.snapshot()
        self.assertEqual(s['phase'], 'idle', s['message'])
        self.assertTrue(s['reference']['fixed_zero'])
        self.assertEqual((s['reference']['zero_y_mm'], s['reference']['zero_t_deg']), (0.0, 0.0))

    def test_select_algorithm_clears_reference(self):
        with tempfile.TemporaryDirectory() as d:
            choices = AlgorithmChoices([dict(id='a', label='가', path='mock'), dict(id='b', label='나', path='mock'),
                                        dict(id='c', label='다', path='no/such/worker')], d)
            r = Runner(Devices(self.camera, self.stage), choices.select('a'), self.rect, self.settings,
                       Path(self.tmp.name) / 'runs', choices=choices)
            r.reference(); r.wait_idle(30)
            self.assertIsNotNone(r.snapshot()['reference'])
            r.select_algorithm('b')
            s = r.snapshot()
            self.assertIsNone(s['reference'])
            self.assertEqual(self.settings['algorithm'], 'b')
            self.assertEqual([(a['id'], a['available'], a['current']) for a in s['algorithms']],
                             [('a', True, False), ('b', True, True), ('c', False, False)])
            with self.assertRaises(AlgorithmError):
                r.select_algorithm('c')
            with self.assertRaises(ValueError):
                r.select_algorithm('zz')

    def test_axis_plan_estimates_each_axis(self):
        r = self.runner(MockAlgorithm())
        r._set(position=dict(y_mm=20.0, t_deg=0.0), speed=dict(mode='slow', y_mm_s=0.2, t_deg_s=0.1))
        a = r._axis_plan(21000, -300, relative=False)          # Y +1 mm (5 s), Θ −0.3° (3 s)
        self.assertEqual((a['dy_mm'], a['dt_deg'], a['y_s'], a['t_s']), (1.0, -0.3, 5.0, 3.0))
        a = r._axis_plan(20000, 50, relative=False)            # Y 그대로, Θ 만
        self.assertEqual((a['y_s'], a['t_s']), (0.0, 0.5))

    def test_step_slightly_over_limit_is_clamped(self):
        self.settings.update(dict(auto_reference=False, sequential_axes=False, max_step_t_deg=1.0))
        r = self.runner(FixedAlgorithm(0.1, -1.1))          # 명령 Θ +1.1° (한도 1.0 의 1.1배)
        r.step(); r.wait_idle(10)
        s = r.snapshot()
        self.assertNotEqual(s['phase'], 'error', s['message'])
        self.assertEqual(self.stage.position, (-100, 1000))             # Θ 는 한도만큼만
        self.assertEqual(s['history'][0]['clamped'], dict(y_mm=-0.1, t_deg=1.0))
        r2 = self.runner(FixedAlgorithm(0.1, -1.5))         # 1.5배 → 멈춤
        r2.step(); r2.wait_idle(10)
        self.assertEqual(r2.snapshot()['phase'], 'error')

    def test_center_here_needs_check(self):
        self.settings.update(dict(auto_reference=False))
        r = self.runner(FixedAlgorithm(0.0, 0.0))
        with self.assertRaises(RunError):
            r.set_center_here()                                     # 확인 측정 없이 저장 안 됨
        r.center_check(); r.wait_idle(10)
        self.assertTrue(r.snapshot()['center_check']['ok'])
        self.stage.position = (0, 0)
        r.set_center_here()                                         # 센터 근처 → 저장
        self.assertIsNone(r.snapshot()['center_check'])
        r2 = self.runner(FixedAlgorithm(4.8, -1.2))                 # 오늘 같은 경우: 센터에서 4.8 mm, 1.2° 떨어진 자리
        r2.center_check(); r2.wait_idle(10)
        c = r2.snapshot()['center_check']
        self.assertFalse(c['ok'])
        self.assertAlmostEqual(c['y_mm'], 4.8)
        with self.assertRaises(RunError):
            r2.set_center_here()                                    # 묻지 않고는 저장 안 됨
        r2._set(reference=dict(ts=time.time(), zero_y_mm=0.0, zero_t_deg=0.0))
        r2.set_center_here(force=True)                              # 사용자가 확인하면 저장, 센터 기준은 지움
        self.assertIsNone(r2.snapshot()['reference'])
        self.assertEqual(self.stage.position, (0, 0))               # 확인 측정은 움직이지 않음

    def test_direct_start_input_is_checked(self):
        r = self.runner(MockAlgorithm())
        with self.assertRaises(RunError):
            r.goto((2.5, 0.0))                 # 한 번 이동 한도 2 mm 넘음
        with self.assertRaises(RunError):
            r.goto((0.5, -1.5))                # Θ 한도 1° 넘음
        with self.assertRaises(RunError):
            r.goto((float('nan'), 0.0))
        r.goto((-0.75, 0.3)); r.wait_idle(30)
        s = r.snapshot()
        self.assertEqual(s['start'], dict(y_mm=-0.75, t_deg=0.3, reached=True))
        self.assertAlmostEqual(s['position']['y_mm'], self.settings['center_y_mm'] - 0.75, places=3)
        self.assertAlmostEqual(s['position']['t_deg'], self.settings['center_t_deg'] + 0.3, places=3)

    def test_demo_speed_and_pause(self):
        self.settings.update(dict(demo_slow=True, slow_y_mm_s=0.3, slow_t_deg_s=0.15, pause_s=0.4))
        r = self.runner(MockAlgorithm(noise_y_mm=0.002, seed=3))
        t0 = time.monotonic()
        r.run((0.8, 0.0))
        r.wait_idle(30)
        s = r.snapshot()
        self.assertEqual(s['run']['result'], 'converged', s['message'])
        self.assertEqual(self.stage.speeds, (300, 150))                    # µm/s, mdeg/s
        self.assertEqual(s['speed']['mode'], 'slow')
        moves = sum(1 for e in s['history'] if not e['within'])
        self.assertGreaterEqual(time.monotonic() - t0, 0.4 * moves)        # 이동 전마다 쉼
        self.assertIsNone(s['start'])                                      # 보정 끝 → 다음은 ② 랜덤
        self.settings.update(dict(demo_slow=False))
        r.goto((0.5, 0.0)); r.wait_idle(30)
        s = r.snapshot()
        self.assertEqual((s['run'], s['history'], s['last'], s['aligned']), (None, [], None, None))   # 새 시연 → 지난 기록 지움
        self.assertEqual(self.stage.speeds, (1000, 1000))
        self.assertEqual(s['start'], dict(y_mm=0.5, t_deg=0.0, reached=True))

    def test_run_converges_and_records_truth(self):
        r = self.runner(MockAlgorithm(noise_y_mm=0.002, seed=3))
        r.run((0.8, 0.0))
        r.wait_idle(30)
        s = r.snapshot()
        self.assertEqual(s['run']['result'], 'converged', s['message'])
        self.assertLessEqual(s['run']['trial'], 4)
        first = s['history'][0]
        self.assertAlmostEqual(first['truth']['y_mm'], 0.8, places=3)       # 시작 위치 = 센터 + 0.8
        self.assertAlmostEqual(first['command']['y_mm'], -first['offset']['y_mm'], places=6)
        self.assertTrue(s['history'][-1]['within'])
        self.assertTrue((Path(self.tmp.name) / 'runs' / s['run']['id'] / 'run.json').exists())

    def test_counts_are_moves(self):
        """횟수는 보정 이동 기준: 측정 = 이동 + 1, 마지막 이동 뒤에도 확인 측정을 한다."""
        r = self.runner(MockAlgorithm(noise_y_mm=0.002, seed=3))
        r.run((0.8, 0.0))
        r.wait_idle(30)
        s = r.snapshot()
        n = len(s['history'])
        self.assertEqual(s['history'][0]['label'], '시작 위치')
        self.assertEqual(s['history'][1]['label'], '이동 1번 후')
        self.assertEqual(s['run']['result_label'], f'보정 이동 {n - 1}번 만에 허용 오차 안으로 들어옴')

        r = self.runner(FixedAlgorithm(0.3))          # 계속 0.3 mm 로 보이는 알고리즘 → 못 들어옴
        self.settings.update(dict(max_trials=2))
        r.run(None)
        r.wait_idle(10)
        s = r.snapshot()
        self.assertEqual(len(s['history']), 3)        # 시작 + 이동 2번 뒤 각각 확인
        self.assertEqual(s['run']['result_label'], '보정 이동 2번 해도 허용 오차 밖')

    def test_command_semantics(self):
        algo = FixedAlgorithm(0.3)
        algo.output = 'command'
        r = self.runner(algo)
        self.settings.update(dict(max_trials=1))
        r.run(None)
        r.wait_idle(10)
        e = r.snapshot()['history'][0]
        self.assertAlmostEqual(e['command']['y_mm'], 0.3)
        self.assertAlmostEqual(e['offset']['y_mm'], -0.3)

    def test_step_limit_blocks_move(self):
        r = self.runner(FixedAlgorithm(5.0))
        r.step()
        r.wait_idle(10)
        s = r.snapshot()
        self.assertEqual(s['phase'], 'error')
        self.assertIn('한도', s['message'])
        self.assertEqual(self.stage.position, (0, 0))      # 움직이지 않음

    def test_stop_interrupts_run(self):
        self.stage.duration = 1.0
        r = self.runner(MockAlgorithm(seed=1))
        r.run((1.0, 0.0))
        time.sleep(0.3)
        r.stop()
        r.wait_idle(10)
        self.assertEqual(r.snapshot()['phase'], 'stopped')

    def test_measure_does_not_move(self):
        r = self.runner(FixedAlgorithm(0.5))
        r.measure()
        r.wait_idle(10)
        s = r.snapshot()
        self.assertEqual(self.stage.position, (0, 0))
        self.assertAlmostEqual(s['measure']['offset']['y_mm'], 0.5)

    def test_correction_approaches_from_minus_side(self):
        """보정 이동은 지금 위치+명령을 절대 목표로, −쪽 접근점(Y −0.2, Θ −0.1)을 거쳐 들어간다."""
        moves = []
        orig = self.stage.start
        def logged(y, t, relative=False, allow_t=False):
            moves.append((y, t, relative)); return orig(y, t, relative=relative, allow_t=allow_t)
        self.stage.start = logged
        self.settings.update(dict(auto_reference=False, sequential_axes=False))
        r = self.runner(FixedAlgorithm(0.3, 0.05))          # 오프셋 → 명령 (−0.3 mm, −0.05°)
        r.step()
        r.wait_idle(10)
        self.assertEqual(moves, [(-500, -150, False), (-300, -50, False)], r.snapshot()['message'])
        self.assertEqual(self.stage.position, (-300, -50))


    def test_reference_moves_to_center_and_registers(self):
        self.stage.position = (700, 200)
        algo = MockAlgorithm()
        r = self.runner(algo)
        r.reference()
        r.wait_idle(10)
        s = r.snapshot()
        self.assertEqual(s['phase'], 'idle', s['message'])
        self.assertEqual(self.stage.position, (0, 0))
        self.assertEqual(s['reference']['zero_y_mm'], 0.0)

    def test_theta_disabled_keeps_theta(self):
        self.settings.update(dict(enable_t=False))
        self.stage.position = (0, 120)
        r = self.runner(FixedAlgorithm(0.3, 0.05))
        r.step()
        r.wait_idle(10)
        self.assertEqual(self.stage.position, (-300, 120))   # Θ 는 그대로

    def test_auto_reference_once_and_returns(self):
        calls = []
        class Counting(MockAlgorithm):
            def set_reference(self, image):
                calls.append(1); return super().set_reference(image)
        self.stage.position = (700, 200)
        r = self.runner(Counting(noise_y_mm=0.001, noise_t_deg=0.0005, seed=2))
        r.step()                                   # 첫 보정 → 센터에서 기준을 잡고 원래 자리로 돌아와 시작
        r.wait_idle(20)
        s = r.snapshot()
        self.assertEqual(len(calls), 1, s['message'])
        self.assertIsNotNone(s['reference'])
        self.assertAlmostEqual(s['history'][0]['truth']['y_mm'], 0.7, places=3)   # 첫 측정은 원래 자리에서
        r.step()                                   # 같은 실행의 다음 단계 → 다시 안 잡음
        r.wait_idle(20)
        r.measure()                                # 기준이 아직 유효 → 다시 안 잡음
        r.wait_idle(20)
        self.assertEqual(len(calls), 1)
        with r.cond:
            r.state['reference']['ts'] -= 2 * 3600   # 2시간 지난 기준
        r.measure()
        r.wait_idle(20)
        self.assertEqual(len(calls), 2)

    def test_auto_reference_can_be_disabled(self):
        calls = []
        class Counting(MockAlgorithm):
            def set_reference(self, image):
                calls.append(1); return super().set_reference(image)
        self.settings.update(dict(auto_reference=False))
        r = self.runner(Counting(seed=3))
        r.measure()
        r.wait_idle(10)
        self.assertEqual(calls, [])

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            self.settings.update(dict(max_trials=0))
        with self.assertRaises(ValueError):
            self.settings.update(dict(nope=1))
        self.settings.update(dict(tol_y_mm=0.02))
        self.assertEqual(Settings(Path(self.tmp.name) / 's.json')['tol_y_mm'], 0.02)


@unittest.skipUnless(os.access(WORKER, os.X_OK), 'align_worker 미빌드')
class WorkerTest(unittest.TestCase):
    def test_round_trip_and_restart(self):
        w = WorkerAlgorithm(WORKER, timeout_s=5)
        img = np.zeros((2160, 3840), np.uint8)
        self.assertEqual(w.infer(img)['y'], 0.0)
        w.proc.kill()
        w.proc.wait()
        self.assertEqual(w.infer(img)['t'], 0.0)       # 죽으면 다시 띄움
        w.close()

    def test_missing_binary(self):
        with self.assertRaises(AlgorithmError):
            WorkerAlgorithm('/nonexistent/align_worker')


class VideoStreamTest(unittest.TestCase):
    """카메라가 프레임을 안 줄 때 /video 가 작업 스레드를 붙잡고 있으면 안 된다 (숨은 버튼 먹통 원인)."""

    def app_with(self, camera):
        calib = Calibration.load(CALIB)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        devices = Devices(camera, MockStage())
        runner = Runner(devices, MockAlgorithm(), Rectifier(calib), Settings(Path(tmp.name) / 's.json'),
                        Path(tmp.name) / 'runs')
        demo = DemoLauncher(['true'], tmp.name, free_port(), Path(tmp.name) / 'demo.log')
        return create_app(runner, devices, demo).test_client()

    def test_no_frames_ends_response(self):
        cam = FakeCamera(ViewMapper(Calibration.load(CALIB)))      # start() 안 함 = 프레임 없음
        client = self.app_with(cam)
        t0 = time.monotonic()
        r = client.get('/video')
        self.assertEqual(r.get_data(), b'')
        self.assertLess(time.monotonic() - t0, 6)
        self.assertEqual(cam.viewers, 0)

    def test_stream_keeps_sending_without_new_frames(self):
        cam = FakeCamera(ViewMapper(Calibration.load(CALIB)), fps=20).start()
        self.addCleanup(cam.close)
        client = self.app_with(cam)
        r = client.get('/video', buffered=False)
        it = iter(r.response)
        self.assertTrue(next(it).startswith(b'--frame'))
        cam.close()                         # 새 프레임이 끊겨도
        t0 = time.monotonic()
        self.assertTrue(next(it).startswith(b'--frame'))   # 1초 안에 마지막 프레임을 다시 보냄
        self.assertLess(time.monotonic() - t0, 2.5)
        r.close()


KSH = ROOT / 'experiments' / 'KSH' / 'branch-snapshot' / 'OutPut_A507' / 'Code'
KSH_WORKER = ROOT / 'verify_web' / 'native' / 'build' / 'ksh' / 'align_worker'


class AlgorithmListTest(unittest.TestCase):
    """저장소 기본 알고리즘 목록(verify_web/algorithms.json)의 설정·모델 경로가 저장소 안에 있다."""

    def test_repo_list_points_inside_repo(self):
        choices = AlgorithmChoices.load(ROOT / 'verify_web' / 'algorithms.json', ROOT)
        self.assertEqual([e['id'] for e in choices.entries], ['ksh', 'hye', 'shin'])
        for e in choices.entries:
            self.assertFalse(e['config'].startswith(('/', '~')), e['config'])
            self.assertTrue(choices._path(e['config']).exists(), e['config'])


@unittest.skipUnless(os.access(KSH_WORKER, os.X_OK) and (KSH / 'config' / 'alignment.yml').exists()
                     and Path('/home/pi/data/rig_eval/center1.bmp').exists(), 'KSH 실행기·평가 데이터 없음')
class KshWorkerTest(unittest.TestCase):
    def test_matches_standalone_tool(self):
        w = WorkerAlgorithm(KSH_WORKER, config=KSH / 'config' / 'alignment.yml')
        self.assertTrue(w.supports_reference)              # READY ref roi
        z = w.set_reference(cv2.imread('/home/pi/data/rig_eval/center1.bmp', 0))
        self.assertAlmostEqual(z['y'], -0.01433, places=4)
        r = w.infer(cv2.imread('/home/pi/data/rig_eval/Y-1.13 T-0.61/Y-1.13 T-0.61 (1).bmp', 0))
        self.assertAlmostEqual(r['y'], -1.0964, places=3)
        self.assertAlmostEqual(r['t'], -0.6418, places=3)
        with self.assertRaises(AlgorithmError):
            w.infer(np.zeros((2160, 3840), np.uint8))       # 측정 실패는 NaN 이 아니라 오류
        w.close()


class CenterViewTest(unittest.TestCase):
    """센터 확인 화면: 차이 영상, 기준 윤곽 좌표."""

    def test_diff_mode_is_gray_when_same_as_reference(self):
        calib = Calibration.load(CALIB)
        cam = FakeCamera(ViewMapper(calib))          # start() 안 함: _view_image 만 시험
        img = cam.img
        cam.set_mode('diff')
        self.assertTrue(np.array_equal(cam._view_image(img), ViewMapper(calib).apply(img)))   # 기준 없으면 일반 영상
        cam.set_reference(img)
        d = cam._view_image(img)
        self.assertEqual(int(np.abs(d.astype(int) - 128).max()), 0)        # 같으면 전부 회색
        moved = np.roll(img, 30, axis=1)                                   # 판이 옮겨진 것처럼
        self.assertGreater(int(np.abs(cam._view_image(moved).astype(int) - 128).max()), 100)
        cam.set_mode('normal')
        self.assertTrue(np.array_equal(cam._view_image(img), ViewMapper(calib).apply(img)))
        with self.assertRaises(ValueError):
            cam.set_mode('x')

    def test_reference_quads_to_view_coordinates(self):
        from verify_web.runner import find_qr_quads
        img = cv2.imread('/home/pi/data/rig_eval/center1.bmp', 0) if Path('/home/pi/data/rig_eval/center1.bmp').exists() else None
        if img is None:
            self.skipTest('센터 사진 없음')
        quads = find_qr_quads(img)
        self.assertEqual(len(quads), 3)
        self.assertLess(quads[0][0][0], quads[2][0][0])                  # 왼쪽부터
        calib = Calibration.load(CALIB)
        with tempfile.TemporaryDirectory() as d:
            cam = FakeCamera(ViewMapper(calib))
            devices = Devices(cam, MockStage())
            runner = Runner(devices, MockAlgorithm(), Rectifier(calib), Settings(Path(d) / 's.json'), Path(d) / 'runs')
            c = [[sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4] for q in quads]
            rois = [dict(name=n, rect=[int(x) - 96, int(y) - 103, 192, 206], center=[x, y])
                    for n, (x, y) in zip(('QR_L', 'QR_C', 'QR_R'), c)]
            rois += [dict(name='S_LB', rect=[891, 1621, 220, 180], center=[1001.1, 1711.7]),
                     dict(name='S_RT', rect=[2677, 1331, 220, 180], center=[2787.4, 1421.8])]
            runner._set(reference=dict(ts=time.time(), zero_y_mm=0.0, zero_t_deg=0.0, qr_alg=quads, rois=rois))
            client = create_app(runner, devices, DemoLauncher(['true'], d, free_port(), Path(d) / 'l')).test_client()
            ov = client.get('/api/state').get_json()['overlay']
            # 1× 화면 = 알고리즘 좌표 / 3 (가운데 기준)
            self.assertAlmostEqual(ov['target']['qr'][1][0][0], (quads[1][0][0] - 1919.5) / 3 + 639.5, delta=0.2)
            self.assertEqual([k['name'] for k in ov['target']['screws']], ['S_LB', 'S_RT'])
            self.assertAlmostEqual(ov['target']['screws'][0]['c'][0], (1001.1 - 1919.5) / 3 + 639.5, delta=0.2)
            self.assertIsNone(ov['now'])
            # 이번 측정에서 QR 이 오른쪽으로 30 px 옮겨졌으면 지금 테두리도 화면에서 10 px 오른쪽
            now = [dict(r, center=[r['center'][0] + 30, r['center'][1]]) for r in rois]
            runner._set(rois_now=now)
            ov = client.get('/api/state').get_json()['overlay']
            for qt, qn in zip(ov['target']['qr'], ov['now']['qr']):
                for a, b in zip(qt, qn):
                    self.assertAlmostEqual(b[0] - a[0], 10, delta=0.2)
                    self.assertAlmostEqual(b[1] - a[1], 0, delta=0.2)
            self.assertEqual(len(ov['now']['dots']), 5)
            self.assertEqual(client.post('/api/view', json={'zoom': 1, 'mode': 'diff'}).status_code, 200)
            self.assertEqual(cam.view_mode, 'diff')
            # 1.7× 는 QR+나사 전체의 가운데를 중심으로 (나사 4개가 다 보이게)
            from verify_web.app import SCREWS_CENTER
            self.assertEqual(client.post('/api/view', json={'zoom': 1.7, 'center': 'screws'}).status_code, 200)
            self.assertEqual(cam._view_req, (1.7, SCREWS_CENTER))
            # 3D 시뮬레이터 탭: Unity 빌드 index.html 은 캐시 없이, 빌드 파일은 길게
            r = client.get('/viewer/')
            self.assertEqual(r.status_code, 200)
            self.assertIn(b'unity', r.data.lower())
            self.assertEqual(r.cache_control.max_age, 0)
            r.close()
            page = client.get('/').data.decode()
            self.assertIn('data-camtab="sim"', page)
            self.assertIn('/static/viewer3d.js', page)


class HandoffTest(unittest.TestCase):
    def test_launch_and_stop(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as d:
            demo = DemoLauncher([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'], d, port,
                                Path(d) / 'demo.log')
            demo.start(timeout=10)
            self.assertTrue(demo.running())
            demo.stop()
            self.assertFalse(demo.running())

    def test_app_handoff_releases_devices(self):
        port = free_port()
        calib = Calibration.load(CALIB)
        with tempfile.TemporaryDirectory() as d:
            devices = DeviceManager(lambda: FakeCamera(ViewMapper(calib)), lambda: MockStage(duration=0.01))
            self.assertTrue(devices.open())
            runner = Runner(devices, MockAlgorithm(), Rectifier(calib), Settings(Path(d) / 's.json'), Path(d) / 'runs')
            demo = DemoLauncher([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'], d, port,
                                Path(d) / 'demo.log')
            client = create_app(runner, devices, demo).test_client()
            self.assertEqual(client.get('/api/state').status_code, 200)
            r = client.post('/api/handoff', json={})
            self.assertEqual(r.get_json()['port'], port)
            self.assertFalse(devices.ready)
            self.assertTrue(demo.running())
            self.assertEqual(client.post('/api/home', json={}).status_code, 409)   # 장치 없음
            r = client.post('/api/reclaim', json={})
            self.assertTrue(r.get_json()['ok'])
            self.assertFalse(demo.running())
            self.assertTrue(devices.ready)
            self.assertEqual(client.post('/api/settings', json={'max_trials': 0}).status_code, 409)
            devices.close()


if __name__ == '__main__':
    unittest.main()

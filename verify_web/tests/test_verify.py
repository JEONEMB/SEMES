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
from verify_web.algorithm import AlgorithmError, MockAlgorithm, WorkerAlgorithm
from verify_web.app import create_app
from verify_web.camera import FakeCamera
from verify_web.devices import DemoLauncher, DeviceManager
from verify_web.geometry import ALG_SIZE, SEMES_FX, SEMES_FY, Calibration, Rectifier, ViewMapper, alg_matrix
from verify_web.runner import Runner, Settings

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
        self.settings = Settings(Path(self.tmp.name) / 's.json', dict(save_images=False, settle_timeout_s=2.0))

    def tearDown(self):
        self.camera.close()
        self.tmp.cleanup()

    def runner(self, algo):
        return Runner(Devices(self.camera, self.stage), algo, self.rect, self.settings, Path(self.tmp.name) / 'runs')

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

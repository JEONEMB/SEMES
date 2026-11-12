"""카메라: 센서 원본 4056x3040 YUV420 의 Y(흑백)를 계속 받아
- 화면: 변환표 한 번(왜곡+크롭+확대) → 1280x720 흑백 JPEG (보는 사람이 있을 때만)
- 알고리즘: 요청 시 다음 프레임(들)을 그대로 넘김
- 안정 판정: 연속 프레임의 위상 상관 이동량

설정은 데이터셋 촬영 도구(hardware/camera, LibcameraSource)와 같다:
수동 노출·게인, AWB 끔 + 색 게인 고정, 10fps, YUV420 의 Y 평면.
"""
import threading
import time

import cv2
import numpy as np

from .geometry import ViewMapper

SENSOR = (4056, 3040)


class CameraError(RuntimeError):
    pass


class _Base:
    """프레임 공유·화면·안정 판정 (실제/가짜 카메라 공통)."""

    def __init__(self, view_mapper, jpeg_quality=80):
        self.view = view_mapper
        self.jpeg_quality = jpeg_quality
        self.cond = threading.Condition()
        self.frame = None          # (seq, monotonic, gray, meta)
        self.seq = 0
        self.jpeg = None
        self.jpeg_seq = 0
        self.viewers = 0
        self.fps = 0.0
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self._view_req = (1.0, None)
        self._view_lock = threading.Lock()

    # --- 공개 API -----------------------------------------------------------
    def set_view(self, view_zoom, center):
        with self._view_lock:
            self._view_req = (view_zoom, center)

    def view_point(self, x_alg, y_alg):
        return self.view.to_view(x_alg, y_alg)

    def latest_seq(self):
        with self.cond:
            return self.seq

    def next_frames(self, n=1, after=None, timeout=3.0):
        """after 이후의 새 프레임 n 장 (흑백 배열 목록)."""
        out = []
        deadline = time.monotonic() + timeout
        with self.cond:
            last = self.seq if after is None else after
            while len(out) < n:
                while self.seq <= last:
                    left = deadline - time.monotonic()
                    if left <= 0 or self.error:
                        raise CameraError(self.error or '카메라 프레임이 오지 않음')
                    self.cond.wait(left)
                last = self.seq
                out.append(self.frame)
        return out

    def grab(self, average=1, after=None):
        frames = self.next_frames(average, after)
        if average == 1:
            return frames[0][2].copy(), frames[0][3]
        acc = np.zeros(frames[0][2].shape, np.float32)
        for f in frames:
            acc += f[2]
        return np.clip(acc / average + 0.5, 0, 255).astype(np.uint8), frames[-1][3]

    def wait_settled(self, threshold_px=0.25, need=2, timeout=6.0, min_wait=0.3):
        """연속 프레임 이동량이 threshold_px(원본 px) 미만으로 need 번 연속이면 안정.
        반환: (안정 여부, 걸린 시간 s, 마지막 이동량 px)."""
        t0 = time.monotonic()
        time.sleep(min_wait)
        prev, ok, shift = None, 0, float('nan')
        after = self.latest_seq()
        while time.monotonic() - t0 < timeout:
            (seq, _, gray, _), = self.next_frames(1, after)
            after = seq
            small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA)
            small = small.astype(np.float32)
            if prev is not None:
                if not hasattr(self, '_win') or self._win.shape != small.shape:
                    self._win = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
                (dx, dy), _ = cv2.phaseCorrelate(prev, small, self._win)
                shift = float(np.hypot(dx, dy) * 4)
                ok = ok + 1 if shift < threshold_px else 0
                if ok >= need:
                    return True, time.monotonic() - t0, shift
            prev = small
        return False, time.monotonic() - t0, shift

    def wait_jpeg(self, last_seq, timeout=2.0):
        with self.cond:
            if self.jpeg_seq <= last_seq:
                self.cond.wait(timeout)
            return self.jpeg_seq, self.jpeg

    def add_viewer(self, delta):
        with self.cond:
            self.viewers = max(0, self.viewers + delta)

    def health(self):
        return dict(ok=self.error is None and self.frame is not None, fps=round(self.fps, 1),
                    error=self.error, kind=self.kind, view_zoom=self.view.key[0])

    # --- 내부 -----------------------------------------------------------------
    def _publish(self, gray, meta):
        now = time.monotonic()
        with self.cond:
            if self.frame is not None:
                dt = now - self.frame[1]
                if dt > 0:
                    self.fps = 0.8 * self.fps + 0.2 / dt if self.fps else 1 / dt
            self.seq += 1
            self.frame = (self.seq, now, gray, meta)
            viewers = self.viewers
            self.cond.notify_all()
        if viewers:
            with self._view_lock:
                vz, center = self._view_req
            self.view.set_view(vz, center)
            ok, buf = cv2.imencode('.jpg', self.view.apply(gray), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                with self.cond:
                    self.jpeg, self.jpeg_seq = buf.tobytes(), self.seq
                    self.cond.notify_all()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f'camera-{self.kind}', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


class PiCamera(_Base):
    kind = 'picamera2'

    def __init__(self, view_mapper, exposure_us=8333, gain=2.0, colour_gains=(3.3, 1.4), fps=10.0, **kw):
        super().__init__(view_mapper, **kw)
        from picamera2 import Picamera2
        self.cam = Picamera2()
        frame_us = int(1e6 / fps)
        cfg = self.cam.create_video_configuration(
            main={'size': SENSOR, 'format': 'YUV420'}, sensor={'output_size': SENSOR}, buffer_count=4,
            controls={'AeEnable': False, 'ExposureTime': int(exposure_us), 'AnalogueGain': float(gain),
                      'AwbEnable': False, 'ColourGains': tuple(map(float, colour_gains)),
                      'FrameDurationLimits': (frame_us, frame_us)})
        self.cam.configure(cfg)
        self.cam.start()

    def _run(self):
        w, h = SENSOR
        while not self._stop.is_set():
            try:
                req = self.cam.capture_request()
                try:
                    arr = req.make_array('main')
                    md = req.get_metadata()
                finally:
                    req.release()
                meta = dict(exposure_us=md.get('ExposureTime'), gain=round(md.get('AnalogueGain', 0), 2),
                            lux=round(md.get('Lux', 0), 1), sensor_ts=md.get('SensorTimestamp'))
                self._publish(np.ascontiguousarray(arr[:h, :w]), meta)
                self.error = None
            except Exception as exc:          # 카메라가 빠지면 화면에 원인을 보여 준다
                self.error = f'카메라 오류: {exc}'
                with self.cond:
                    self.cond.notify_all()
                time.sleep(0.5)

    def close(self):
        super().close()
        try:
            self.cam.stop()
        finally:
            self.cam.close()


class FakeCamera(_Base):
    """--camera off: 장치 없이 화면·흐름 시험. 격자 무늬 고정 영상."""
    kind = 'fake'

    def __init__(self, view_mapper, fps=10.0, **kw):
        super().__init__(view_mapper, **kw)
        w, h = SENSOR
        img = np.full((h, w), 170, np.uint8)
        img[::100, :] = 90
        img[:, ::100] = 90
        cv2.putText(img, 'NO CAMERA (--camera off)', (900, 1520), cv2.FONT_HERSHEY_SIMPLEX, 6, 40, 12)
        self.img = img
        self.period = 1 / fps

    def _run(self):
        while not self._stop.is_set():
            self._publish(self.img, dict(exposure_us=0, gain=0, lux=0, sensor_ts=None))
            time.sleep(self.period)

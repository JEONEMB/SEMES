"""One Picamera2 producer, shared JPEG frames, no recording to disk."""
import io
import logging
import threading
import time

from .store import Problem

PROFILES = {
    'full': dict(label='원본 4056 × 3040 · 5 fps', size=(4056, 3040), fps=5, quality=85),
    'medium': dict(label='축소 2028 × 1520 · 10 fps', size=(2028, 1520), fps=10, quality=80),
    'preview': dict(label='미리보기 960 × 720 · 10 fps', size=(960, 720), fps=10, quality=75),
}


class Camera(io.BufferedIOBase):
    def __init__(self, enabled=False, profile='full', undistort=True):
        super().__init__()
        self.enabled = enabled
        self.condition = threading.Condition()
        self.frame = None
        self.last_frame = 0
        self.frame_time = 0.0
        self.sequence = 0
        self.device = None
        self.error = '카메라 미사용 모드' if not enabled else '카메라 시작 중'
        self.shutdown = False
        self.profile = profile
        self.generation = 0
        self.lifecycle = threading.RLock()
        self.actual_size = None
        self.undistort = undistort
        self.rectifier = None
        self.processing_failed = False

    def start(self):
        if not self.enabled:
            return
        # Initialization failure must leave the rest of the web application usable.
        with self.lifecycle:
            self._open()

    def configure(self, profile, enabled, undistort=None):
        if undistort is None:
            undistort = self.undistort
        if profile not in PROFILES or type(enabled) is not bool or type(undistort) is not bool:
            raise Problem('유효한 카메라 모드와 해상도를 선택하세요.')
        with self.lifecycle:
            self.stop()
            with self.condition:
                self.profile, self.enabled = profile, enabled
                self.undistort = undistort
                self.processing_failed = False
                self.shutdown = False
                self.error = '카메라 시작 중' if enabled else '카메라 미사용 모드'
            self.start()
        return self.health()

    def _open(self):
        device = None
        try:
            from picamera2 import Picamera2
            from picamera2.encoders import JpegEncoder
            from picamera2.outputs import FileOutput
            if self.undistort:
                from .rectification import Rectifier
                self.rectifier = Rectifier()
            profile = PROFILES[self.profile]
            device = Picamera2()
            device.configure(device.create_video_configuration(
                main={'size': profile['size'], 'format': 'RGB888'},
                sensor={'output_size': (4056, 3040), 'bit_depth': 12},
                controls={'FrameRate': profile['fps']},
                # 3 buffers instead of 6: frames no longer queue behind slow rectification
                # (measured 4056×3040: 850 → 420 ms capture→JPEG, 4.3 → 4.0 fps).
                buffer_count=3))
            with self.condition:
                if self.shutdown:
                    device.close()
                    return
                self.device = device
                self.actual_size = tuple(device.camera_configuration()['main']['size'])
            if self.undistort:
                device.pre_callback = self._rectify_request
            device.start_recording(JpegEncoder(q=profile['quality']), FileOutput(self))
            with self.condition:
                if not self.processing_failed:
                    self.error = ''
        except Exception:
            logging.exception('Camera initialization failed')
            self.error = '카메라 시작 실패 · 연결, 보정 파일 및 OpenCV 설치 상태를 확인하세요.'
            if device:
                try:
                    device.close()
                except Exception:
                    pass
            self.device = None

    def _rectify_request(self, request):
        # Picamera2 invokes this once before JPEG encoding; every viewer shares it.
        if self.processing_failed:
            return
        try:
            from picamera2 import MappedArray
            crop = request.get_metadata()['ScalerCrop']
            with MappedArray(request, 'main') as mapped:
                mapped.array[:] = self.rectifier.apply(mapped.array, crop)
        except Exception:
            logging.exception('Camera rectification failed')
            with self.condition:
                self.processing_failed = True
                self.frame = None
                self.error = '왜곡 보정 실패 · 관리자에서 카메라 설정을 확인하고 다시 연결하세요.'
                self.condition.notify_all()

    def write(self, buf):
        with self.condition:
            if self.processing_failed:
                return len(buf)  # Never label uncorrected frames as corrected.
            self.frame = bytes(buf)
            self.last_frame = time.monotonic()
            self.frame_time = time.time()
            self.sequence += 1
            self.condition.notify_all()
        return len(buf)

    def health(self):
        with self.condition:
            live = not self.shutdown and not self.processing_failed and self.frame is not None and time.monotonic() - self.last_frame < 3
            profile = PROFILES[self.profile]
            return dict(connected=live, enabled=self.enabled,
                        profile=self.profile, width=(self.actual_size or profile['size'])[0],
                        undistort=self.undistort,
                        height=(self.actual_size or profile['size'])[1], fps=profile['fps'],
                        sensor_width=4056, sensor_height=3040, generation=self.generation,
                        message='실시간 영상 연결됨' if live else self.error or '영상 수신 중단')

    def stream(self):
        sequence = -1
        generation = self.generation
        while not self.shutdown:
            with self.condition:
                self.condition.wait_for(lambda: self.sequence != sequence or self.shutdown
                                        or self.generation != generation or self.processing_failed, timeout=3)
                if generation != self.generation or self.shutdown or not self.health()['connected']:
                    return
                frame, sequence, stamp = self.frame, self.sequence, self.frame_time
            # X-Frame-Time (encode finished, epoch s) lets tools measure end-to-end delay.
            yield (b'--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: '
                   + str(len(frame)).encode() + b'\r\nX-Frame-Time: ' + f'{stamp:.3f}'.encode()
                   + b'\r\n\r\n' + frame + b'\r\n')

    def stop(self):
        with self.lifecycle:
            with self.condition:
                self.shutdown = True
                self.generation += 1
                self.condition.notify_all()
                device, self.device = self.device, None
            if device:
                try:
                    device.stop_recording()
                finally:
                    device.close()
            with self.condition:
                self.frame, self.actual_size = None, None
                self.rectifier = None
                self.last_frame = 0

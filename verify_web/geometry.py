"""센서 원본(4056x3040 흑백) → 알고리즘 입력(3840x2160) / 화면(1280x720) 변환표.

알고리즘 입력은 데이터셋 촬영 도구와 같은 식이다
(feat/lye/camera-calib: hardware/camera/src/rectifier.cpp, Rectifier):
  왜곡을 편 뒤 SEMES 카메라 행렬(fx=12/2.8um*3840/4096, fy=12/2.8um)로 다시 투영하고
  렌즈 중심을 사진 가운데((W-1)/2, (H-1)/2)에 둔다. zoom 은 데이터셋 촬영 때와 같은 값을 쓴다.

화면은 같은 투영을 1/3 크기로 하되, 확대(view_zoom)와 보고 싶은 점(알고리즘 좌표)을
가운데로 옮기는 것까지 한 변환표에 넣는다. 그래서 왜곡 보정·크롭·확대가 remap 한 번이다.
remap 비용은 출력 크기로 정해지므로 원본이 12MP여도 1280x720 출력은 수 ms 이다.
"""
from dataclasses import dataclass

import cv2
import numpy as np

SEMES_FX = 12.0 / 2.8e-3 * 3840.0 / 4096.0   # 4017.86, 가로만 15/16 축소
SEMES_FY = 12.0 / 2.8e-3                     # 4285.71
ALG_SIZE = (3840, 2160)
VIEW_SIZE = (1280, 720)


@dataclass(frozen=True)
class Calibration:
    size: tuple          # (w, h) 센서 원본
    K: np.ndarray
    D: np.ndarray

    @classmethod
    def load(cls, path):
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not fs.isOpened():
                raise ValueError(f'캘리브레이션 파일을 열 수 없음: {path}')
            size = (int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real()))
            K = fs.getNode('camera_matrix').mat()
            D = fs.getNode('dist_coeffs').mat()
        finally:
            fs.release()
        if size[0] <= 0 or size[1] <= 0 or K is None or K.shape != (3, 3) or D is None or D.size == 0:
            raise ValueError(f'image_width/height, camera_matrix, dist_coeffs 가 필요함: {path}')
        return cls(size, K.astype(np.float64), D.astype(np.float64).reshape(-1))


def alg_matrix(zoom=1.0):
    w, h = ALG_SIZE
    return np.array([[SEMES_FX * zoom, 0, (w - 1) * 0.5],
                     [0, SEMES_FY * zoom, (h - 1) * 0.5],
                     [0, 0, 1]], np.float64)


def view_matrix(zoom=1.0, view_zoom=1.0, center=None):
    """화면용 행렬. center = 화면 가운데에 올 알고리즘 좌표 (None 이면 알고리즘 사진 가운데)."""
    aw, ah = ALG_SIZE
    vw, vh = VIEW_SIZE
    s = vw / aw * view_zoom
    cx_a, cy_a = center if center is not None else ((aw - 1) * 0.5, (ah - 1) * 0.5)
    return np.array([[SEMES_FX * zoom * s, 0, (vw - 1) * 0.5 - (cx_a - (aw - 1) * 0.5) * s],
                     [0, SEMES_FY * zoom * s, (vh - 1) * 0.5 - (cy_a - (ah - 1) * 0.5) * s],
                     [0, 0, 1]], np.float64)


def build_map(calib, K_out, size):
    """반환: (map1, map2, 원본 밖 비율). 형식은 빠른 고정소수점(CV_16SC2)."""
    mx, my = cv2.initUndistortRectifyMap(calib.K, calib.D, None, K_out, size, cv2.CV_32FC1)
    w, h = calib.size
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    m1, m2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
    return m1, m2, 1.0 - float(inside.mean())


class Rectifier:
    """알고리즘 입력 3840x2160 (데이터셋과 같은 변환)."""

    def __init__(self, calib, zoom=1.0):
        if not zoom > 0:
            raise ValueError('zoom 은 0 보다 커야 함')
        self.calib, self.zoom = calib, zoom
        self.map1, self.map2, self.outside = build_map(calib, alg_matrix(zoom), ALG_SIZE)

    def apply(self, gray):
        if gray.dtype != np.uint8 or gray.ndim != 2 or (gray.shape[1], gray.shape[0]) != self.calib.size:
            raise ValueError(f'입력은 {self.calib.size} 8bit 흑백이어야 함')
        return cv2.remap(gray, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class ViewMapper:
    """화면 1280x720. 확대·중심을 바꾸면 변환표를 다시 만든다 (수십 ms, 바꿀 때만)."""

    def __init__(self, calib, zoom=1.0):
        self.calib, self.zoom = calib, zoom
        self.key = None
        self.maps = None
        self.set_view(1.0, None)

    def set_view(self, view_zoom, center):
        key = (round(float(view_zoom), 3), None if center is None else (round(center[0], 1), round(center[1], 1)))
        if key != self.key:
            m1, m2, _ = build_map(self.calib, view_matrix(self.zoom, view_zoom, center), VIEW_SIZE)
            self.maps, self.key = (m1, m2), key
        return self.key

    def apply(self, gray):
        m1, m2 = self.maps
        return cv2.remap(gray, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def to_view(self, x_alg, y_alg):
        """알고리즘 좌표 → 화면 좌표 (표시용 표식 위치)."""
        vz, center = self.key
        K = view_matrix(self.zoom, vz, center)
        Ka = alg_matrix(self.zoom)
        xn = (x_alg - Ka[0, 2]) / Ka[0, 0]
        yn = (y_alg - Ka[1, 2]) / Ka[1, 1]
        return K[0, 0] * xn + K[0, 2], K[1, 1] * yn + K[1, 2]

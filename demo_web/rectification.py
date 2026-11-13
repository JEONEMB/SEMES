"""Lens correction with the team's full-sensor IMX477 calibration.

The output retains the input size and intrinsic matrix (cv::undistort default),
without SEMES crop, recentering, or anisotropic resizing. Build maps once per
stream geometry; never decode/re-encode an already compressed JPEG.
"""
import hashlib
import json
from pathlib import Path

CALIBRATION_PATH = Path(__file__).parent / 'calibration' / 'camera_calib.yml'
# ScalerCrop recorded in calib_001.json: full sensor image starts at (8, 16).
CALIBRATION_CROP = (8, 16, 4056, 3040)


def reference_signature(profile, undistort):
    try:
        digest = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest() if undistort else None
    except OSError:
        digest = 'missing'
    return json.dumps(dict(geometry_version=1, profile=profile, undistort=undistort,
                           calibration=digest), sort_keys=True)


class Rectifier:
    def __init__(self, path=CALIBRATION_PATH):
        import cv2
        import numpy as np
        # Leave capacity for JPEG workers and HTTP/motion processing on Pi 5.
        cv2.setNumThreads(2)
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not fs.isOpened():
                raise ValueError('캘리브레이션 파일을 열 수 없습니다.')
            size = (int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real()))
            self.matrix = fs.getNode('camera_matrix').mat()
            self.distortion = fs.getNode('dist_coeffs').mat()
            if (size != CALIBRATION_CROP[2:] or self.matrix is None or self.matrix.shape != (3, 3)
                    or self.distortion is None or self.distortion.size != 5
                    or not np.isfinite(self.matrix).all() or not np.isfinite(self.distortion).all()
                    or self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0
                    or not np.array_equal(self.matrix[2], [0, 0, 1])):
                raise ValueError('캘리브레이션 해상도·행렬·왜곡 계수를 확인하세요.')
        finally:
            fs.release()
        self.cv2 = cv2
        self.key = None
        self.maps = None

    def camera_matrix(self, size, crop):
        x, y, w, h = crop
        ox, oy, cw, ch = CALIBRATION_CROP
        if (w <= 0 or h <= 0 or x < ox or y < oy or x+w > ox+cw or y+h > oy+ch
                or size[0] <= 0 or size[1] <= 0):
            raise ValueError('카메라 크롭 영역이 캘리브레이션 범위를 벗어납니다.')
        matrix = self.matrix.copy()
        matrix[0, 2] -= x-ox
        matrix[1, 2] -= y-oy
        matrix[0, :] *= size[0]/w
        matrix[1, :] *= size[1]/h
        return matrix

    def apply(self, frame, crop):
        size = (frame.shape[1], frame.shape[0])
        key = (size, tuple(crop))
        if self.key != key:
            matrix = self.camera_matrix(size, crop)
            self.maps = self.cv2.initUndistortRectifyMap(
                matrix, self.distortion, None, matrix, size, self.cv2.CV_16SC2)
            self.key = key
        return self.cv2.remap(frame, *self.maps, self.cv2.INTER_LINEAR,
                              borderMode=self.cv2.BORDER_CONSTANT)

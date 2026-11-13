#!/usr/bin/env python3
"""정밀 캘리브레이션: findChessboardCornersSB + calibrateCameraRO (보드 코너 위치까지 함께 추정).

종이 체커보드는 1 mm 정도만 휘어도 calibrate.cpp(일반 calibrateCamera) 결과가 수 px 오차로 나빠진다.
calibrateCameraRO 는 보드의 실제 코너 위치(휨·인쇄 오차)를 같이 추정해서 렌즈 값을 안정적으로 구한다.
(2026-11-12: 일반 RMS 1.94 px, fx 흔들림 ±0.3% → RO RMS 0.69 px, fx ±0.1%)

사용 (Pi, picamera2 venv 에 OpenCV 4 필요):
  python3 calibrate_ro.py captures_YYYYMMDD --square 15 --out camera_calib.yml [--skip 013,027]

- 코너 검출 결과는 <폴더>_sb.json 에 저장해 두고 다음 실행 때 다시 쓴다 (검출이 제일 오래 걸림). 새로 찍은 사진만 추가로 검출한다.
- 출력 yml 은 calibrate.cpp 와 같은 항목 + roll_deg(0) 이라 Rectifier 가 그대로 읽는다.
- 사진은 정면만이 아니라 보드 면을 앞뒤로 20~30° 젖힌 것이 꼭 섞여야 한다 (없으면 fx 가 정해지지 않음).
  결과의 '젖힌 각도 분포'와 '무작위 70% 재계산 fx 범위'로 확인한다.
"""
import argparse
import glob
import json
import math
import random

import cv2
import numpy as np

SEMES_FX = 12.0 / 2.8e-3 * 3840.0 / 4096.0
SEMES_FY = 12.0 / 2.8e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--square', type=float, default=15.0, help='칸 크기 mm')
    ap.add_argument('--cols', type=int, default=11, help='내부 코너 가로 개수')
    ap.add_argument('--rows', type=int, default=8, help='내부 코너 세로 개수')
    ap.add_argument('--out', default='camera_calib.yml')
    ap.add_argument('--skip', default='', help='뺄 사진 번호 (예: 013,027)')
    ap.add_argument('--note', default='', help='yml 맨 끝 주석')
    a = ap.parse_args()
    cv2.setNumThreads(4)

    pat = (a.cols, a.rows)
    obj = np.zeros((a.cols * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square

    cache = a.folder.rstrip('/') + '_sb.json'
    try:
        saved = json.load(open(cache))
    except OSError:
        saved = {}
    # 이미 찾은 사진은 다시 쓰고, 새로 추가된 사진만 검출한다 (실패한 사진은 None 으로 기억)
    pts = {}
    files = sorted(glob.glob(a.folder.rstrip('/') + '/calib_[0-9][0-9][0-9].png'))
    fresh = [f for f in files if f not in saved]
    if len(fresh) < len(files):
        print(f'코너 검출 결과 다시 사용: {cache} ({len(files) - len(fresh)}장)')
    for f in fresh:
        ok, c = cv2.findChessboardCornersSB(cv2.imread(f, 0), pat, cv2.CALIB_CB_NORMALIZE_IMAGE
                                            | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        print(f'  {"OK  " if ok else "실패"} {f}')
        saved[f] = c.reshape(-1, 2).tolist() if ok else None
    if fresh:
        json.dump(saved, open(cache, 'w'))
    pts = {f: np.array(v, np.float32) for f, v in saved.items() if v is not None and f in files}
    skip = set(s for s in a.skip.split(',') if s)
    names = sorted(n for n in pts if n[-7:-4] not in skip)
    if len(names) < 10:
        raise SystemExit(f'쓸 수 있는 사진이 {len(names)}장뿐 (10장 이상 필요)')
    P = {n: pts[n].reshape(-1, 1, 2) for n in names}
    size = tuple(cv2.imread(names[0], 0).shape[::-1])
    fixed = a.cols - 1          # 첫 줄 오른쪽 끝 코너를 고정해 축척을 유지

    def ro(ns):
        return cv2.calibrateCameraROExtended([obj] * len(ns), [P[n] for n in ns], size, fixed, None, None)

    rms, K, D, rv, tv, new_obj, *_ = ro(names)
    tilt = [math.degrees(math.acos(abs(cv2.Rodrigues(r)[0][2, 2]))) for r in rv]
    print(f'\n사용 {len(names)}장, 사진 {size[0]}x{size[1]}')
    print('사진별 오차 (나쁜 순 8장):')
    per = []
    for n, r, t, tl in zip(names, rv, tv, tilt):
        p, _ = cv2.projectPoints(new_obj, r, t, K, D)
        per.append((float(np.sqrt(((p - P[n]) ** 2).sum(2).mean())), n, tl))
    for e, n, tl in sorted(per, reverse=True)[:8]:
        print(f'  {n}: {e:.2f} px  젖힌 각도 {tl:4.1f}°')
    print(f'젖힌 각도 분포(°): {np.round(np.percentile(tilt, [0, 25, 50, 75, 100]), 1).tolist()}  (20° 넘는 사진이 있어야 함)')
    d = new_obj.reshape(-1, 3) - obj
    print(f'보드 실제 코너 - 격자: 평면 안 최대 {np.abs(d[:, :2]).max():.2f} mm, 평면 밖(휨) {d[:, 2].min():+.2f}~{d[:, 2].max():+.2f} mm')

    random.seed(0)
    fxs = [ro(random.sample(names, int(len(names) * 0.7)))[1][0, 0] for _ in range(6)]
    print(f'무작위 70% 6번 fx: {min(fxs):.0f} ~ {max(fxs):.0f}  (±0.3% 안이면 안정)')

    dist = float(np.median([np.linalg.norm(t) for t in tv]))
    fovx = 2 * math.degrees(math.atan(size[0] / 2 / K[0, 0]))
    fovy = 2 * math.degrees(math.atan(size[1] / 2 / K[1, 1]))
    fs = cv2.FileStorage(a.out, cv2.FILE_STORAGE_WRITE)
    for k, v in [('image_width', size[0]), ('image_height', size[1]), ('pattern_cols', a.cols), ('pattern_rows', a.rows),
                 ('square_mm', a.square), ('images_used', len(names)), ('rms_px', rms)]:
        fs.write(k, v)
    fs.write('camera_matrix', K)
    fs.write('dist_coeffs', D)
    for k, v in [('fov_x_deg', fovx), ('fov_y_deg', fovy), ('focal_mm', K[0, 0] * 1.55e-3),
                 ('board_distance_median_mm', dist),
                 ('semes_crop_width', round(3840 * K[0, 0] / SEMES_FX)), ('semes_crop_height', round(2160 * K[1, 1] / SEMES_FY)),
                 ('roll_deg', 0.0)]:
        fs.write(k, v)
    fs.writeComment(f'calibrate_ro.py: {len(names)} photos in {a.folder}, findChessboardCornersSB + calibrateCameraRO, RMS {rms:.3f} px')
    fs.writeComment('roll_deg: rotation about optical axis applied after undistortion (+ = output turns clockwise). Measure on the rig, then edit.')
    if a.note:
        fs.writeComment(a.note)
    fs.release()
    print(f'\nRMS {rms:.3f} px  fx {K[0, 0]:.1f} fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}')
    print(f'왜곡 {np.round(D.ravel(), 4).tolist()}')
    print(f'SEMES 화각: 왜곡 편 사진에서 {round(3840 * K[0, 0] / SEMES_FX)}x{round(2160 * K[1, 1] / SEMES_FY)} 를 잘라 3840x2160 으로')
    print(f'저장: {a.out}')


if __name__ == '__main__':
    main()

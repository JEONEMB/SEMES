# -*- coding: utf-8 -*-
"""QR 위치 — 거친 검출(밝은 사각형 후보 + 피치 규칙)과 서브픽셀 꼭짓점(변 직선 맞춤).

QR 은 어두운 플레이트 위의 밝은 정사각형(한 변 13 mm ≈ 184 px)이고 중심 간격은 18 mm ≈ 257 px 이다.
꼭짓점은 각 변을 가로지르는 밝기 단면에서 '어두움→밝음' 경계(반값 지점)를 서브픽셀로 찾고,
변마다 직선을 맞춘 뒤 이웃 변 직선의 교점으로 구한다 (모서리 근처 화소는 쓰지 않음).
"""
from __future__ import annotations

import cv2
import numpy as np

PITCH_PX = 257.0


def find_qr_boxes(gray: np.ndarray, pitch: float = PITCH_PX, thr: int = 130) -> list[dict] | None:
    """밝은 사각형 후보 중 좌·중·우 QR 3개(피치 규칙)를 골라 [{cx, cy, w, h, poly}] 를 왼쪽부터 반환."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, b = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    # 9×9 닫기는 Y −1.15 mm 이하에서 왼쪽 QR 과 옆 나사 반사를 한 덩어리로 붙인다 (Stage 1 에서 11장 실패 확인) → 5×5
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (120 <= w <= 300 and 0.8 <= w / h <= 1.25):
            continue
        if cv2.contourArea(c) < 0.85 * w * h:
            continue
        m = cv2.moments(c)
        cands.append({"cx": m["m10"] / m["m00"], "cy": m["m01"] / m["m00"], "w": w, "h": h,
                      "x": x, "y": y, "cnt": c})
    best = None
    for c in cands:
        l = [d for d in cands if abs(d["cx"] - (c["cx"] - pitch)) < 0.15 * pitch and abs(d["cy"] - c["cy"]) < 40]
        r = [d for d in cands if abs(d["cx"] - (c["cx"] + pitch)) < 0.15 * pitch and abs(d["cy"] - c["cy"]) < 40]
        if l and r:
            trip = [l[0], c, r[0]]
            score = abs(c["cx"] - gray.shape[1] / 2)
            if best is None or score < best[0]:
                best = (score, trip)
    return None if best is None else best[1]


def _edge_points(gray: np.ndarray, p0, p1, outward, n_samples: int = 120, half: int = 9, trim: float = 0.15):
    # half = 9 px : 바깥 어두운 띠(위쪽은 약 13 px 뒤 밝은 영역)와 안쪽 흰 여백(약 13 px 뒤 검은 찾기 패턴) 안에 머무르는 폭
    """변 p0→p1 을 따라 n 개 단면을 잡고, 바깥(어두움)→안(밝음) 반값 경계의 서브픽셀 위치를 반환."""
    p0, p1, outward = map(lambda v: np.asarray(v, float), (p0, p1, outward))
    pts = []
    ts = np.linspace(trim, 1 - trim, n_samples)
    offs = np.arange(-half, half + 1, 0.5)
    for t in ts:
        base = p0 + (p1 - p0) * t
        line = base[None, :] + offs[:, None] * (-outward)[None, :]     # 바깥 → 안
        mx, my = line[:, 0].astype(np.float32), line[:, 1].astype(np.float32)
        prof = cv2.remap(gray, mx.reshape(1, -1), my.reshape(1, -1), cv2.INTER_CUBIC).ravel().astype(float)
        lo, hi = np.percentile(prof[:6], 50), np.percentile(prof[-6:], 50)
        if hi - lo < 25:
            continue
        mid = 0.5 * (lo + hi)
        idx = np.where((prof[:-1] < mid) & (prof[1:] >= mid))[0]
        if len(idx) == 0:
            continue
        i = idx[np.argmin(np.abs(idx - len(offs) // 2))]                # 예상 위치에 가장 가까운 교차
        f = (mid - prof[i]) / (prof[i + 1] - prof[i])
        s = offs[i] + f * (offs[i + 1] - offs[i])
        pts.append(base + s * (-outward))
    return np.array(pts)


def _fit_line(pts: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    n = np.array([-vy, vx])
    resid = (pts - np.array([x0, y0])) @ n
    return np.array([x0, y0]), np.array([vx, vy]), float(np.std(resid))


def _intersect(a, b):
    (pa, da), (pb, db) = a, b
    A = np.array([da, -db]).T
    s = np.linalg.solve(A, pb - pa)
    return pa + s[0] * da


def subpixel_corners(gray: np.ndarray, box: dict) -> dict | None:
    """밝은 사각형 하나의 꼭짓점 4개(TL, TR, BR, BL)를 서브픽셀로 구한다."""
    approx = cv2.approxPolyDP(box["cnt"], 0.04 * cv2.arcLength(box["cnt"], True), True).reshape(-1, 2).astype(float)
    if len(approx) != 4:
        rect = cv2.minAreaRect(box["cnt"])
        approx = cv2.boxPoints(rect).astype(float)
    c = approx.mean(0)
    ang = np.arctan2(approx[:, 1] - c[1], approx[:, 0] - c[0])
    order = np.argsort(ang)                                # TL(−135°) → TR(−45°) → BR → BL
    q = approx[order]
    start = np.argmin(q[:, 0] + q[:, 1])
    q = np.roll(q, -start, axis=0)                        # q = TL, TR, BR, BL
    lines, sides = [], []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        outward = np.array([d[1], -d[0]])                  # 시계방향 순회 → 오른손 법선이 바깥
        if np.dot(outward, (p0 + p1) / 2 - c) < 0:
            outward = -outward
        pts = _edge_points(gray, p0, p1, outward)
        if len(pts) < 40:
            return None
        pa, da, rstd = _fit_line(pts)
        lines.append((pa, da)); sides.append(rstd)
    # lines: 0=top, 1=right, 2=bottom, 3=left → TL = left∩top, TR = top∩right, BR = right∩bottom, BL = bottom∩left
    corners = np.array([_intersect(lines[3], lines[0]), _intersect(lines[0], lines[1]),
                        _intersect(lines[1], lines[2]), _intersect(lines[2], lines[3])])
    return {"corners": corners, "edge_resid_px": sides}

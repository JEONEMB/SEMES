#!/usr/bin/env python3
"""SEMES 재현 타깃 — 간단 버전 (출력 시간·재료 절약).

자세한 버전(make_target.py)과 특징 위치(SEMES 화면 좌표)는 같다. 바뀐 점:
  - 배경판: 3 mm 판을 눕혀서 출력 (앞면이 위 → 앞면에 층 줄무늬 없음)
  - 하단바 29 mm 덩어리 -> 10 mm 단. 위 턱·단·양 끝 세로 띠가 판 보강대 역할
  - 플레이트 받침: 속 빈 상자 (벽 3 mm)
  - 스탠드: 얇은 L자 + 보강 1장
  - 배경판 두 조각은 뒤에서 이음판(3 mm)으로 잇는다
좌표·변수 설명은 make_target.py 참고.
"""
from pathlib import Path

import numpy as np
import trimesh

import make_target as T
from make_target import box, cyl_x, prism_y, slot_y, union, M3, M4

OUT = Path(__file__).resolve().parent / "stl_simple"   # 중간 버전 (채택 안 함, .gitignore)

PANEL_T = 3.0
PANEL_Z0, PANEL_Z1 = 155.0, 352.0      # 화면 세로 범위(배경 면 ±91) + 여유
STEP_DEPTH, STEP_H = 10.0, 8.0         # 하단 단 (윗면 = 이혜원 아래 가로선)
EDGE_W, EDGE_D = 6.0, 8.0              # 양 끝 세로 띠 (화면 밖)
X_B = T.X_B                            # 배경판 앞면
X_BB = X_B + PANEL_T
STAND_Y0, STAND_Y1 = 178.0, 200.0      # 스탠드 기둥 폭 22 (배경판 뒤, 양 끝)
STAND_BOLT_Y = 189.0
STAND_BOLT_Z = (215.0, 315.0)
SPLICE_W = 60.0
SPLICE_Z = (228.0, 248.0)          # 받침 뒤에 가려지는 높이만 볼트 (위·아래는 접착)


def panel_half(side):
    ylo, yhi = sorted((0.0, side * T.PANEL_W / 2))
    p = [box(X_B, X_BB, ylo, yhi, PANEL_Z0, PANEL_Z1)]
    # 하단 단: 윗면 = Z_BAR_TOP
    p.append(box(X_B - STEP_DEPTH, X_B + 0.01, ylo, yhi, T.Z_BAR_TOP - STEP_H, T.Z_BAR_TOP))
    # 위 턱 (자세한 버전과 같은 단면: 윗면 45° 모따기, 모서리 = Z_UPPER_EDGE)
    zt, r = T.Z_UPPER_EDGE, T.RIB_OUT
    p.append(prism_y([(X_B + 0.01, zt), (X_B - r, zt - r), (X_B - r, zt - r - 2), (X_B + 0.01, zt - 2 * r - 2)], ylo, yhi))
    # 양 끝 세로 띠 (Θ ±2° 화면 밖)
    yo = side * T.PANEL_W / 2
    p.append(box(X_B - EDGE_D, X_B + 0.01, *sorted((yo, yo - side * EDGE_W)), PANEL_Z0, PANEL_Z1))
    p = union(p)
    h = []
    for (y, z) in T.BG_SCREWS:
        if y * side > 0:
            h.append(cyl_x(y, z, X_B - 1, X_BB + 1, M3))
    for z in STAND_BOLT_Z:
        h.append(cyl_x(side * STAND_BOLT_Y, z, X_B - EDGE_D - 1, X_BB + 1, M4))
    for z in SPLICE_Z:
        h.append(cyl_x(side * SPLICE_W / 4, z, X_B - 1, X_BB + 1, M3))
    for z in T.BRK_BOLT_Z:
        h.append(cyl_x(side * T.BRK_BOLT_U, z, X_B - 1, X_BB + 1, M4))
    return p - union(h)


def splice():
    """배경판 뒤 가운데 이음판."""
    p = box(X_BB, X_BB + 3, -SPLICE_W / 2, SPLICE_W / 2, PANEL_Z0 + 10, PANEL_Z1 - 5)
    h = [cyl_x(s * SPLICE_W / 4, z, X_BB - 1, X_BB + 4, M3) for s in (-1, 1) for z in SPLICE_Z]
    # 받침 볼트와 겹치지 않게 (받침 볼트는 ±41.5, 이음판 폭 ±30 → 안 겹침)
    return p - union(h)


def prism_z(pts_xy, z0, z1):
    from manifold3d import CrossSection
    pts = list(pts_xy)
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    if area < 0:
        pts = pts[::-1]
    return CrossSection([pts]).extrude(z1 - z0).translate([0, 0, z0])


def bracket():
    """속 빈 플레이트 받침: 앞면 4, 옆·위·아래 벽 3, 뒤는 열림 + 날개 (날개 아래 45° 받침)."""
    x0, x1 = T.X_P + T.PLATE_T, X_B
    z0, z1 = T.Z_PLATE - T.BRK_BOT, T.Z_PLATE + T.BRK_TOP
    w, t, fl = T.BRK_W / 2, 2.0, T.BRK_FLANGE
    p = box(x0, x1, -w, w, z0, z1) - box(x0 + 3, x1 + 1, -w + t, w - t, z0 + t, z1 - t)
    p = p + box(x1 - 4, x1, -w - fl, w + fl, z0, z1)
    for s in (-1, 1):   # 앞면을 바닥에 두고 출력할 때 날개를 받치는 45° 쐐기
        p = p + prism_z([(x1 - 4, s * w), (x1 - 4 - fl, s * w), (x1 - 4, s * (w + fl))], z0, z1)
    # 나사가 물릴 살 (앞면 뒤 기둥)
    p = p + union([box(x0 + 3.9, x0 + 15, u - 4, u + 4, T.Z_PLATE - 4, T.Z_PLATE + 4) for u in (-T.PLATE_SCREW_U, T.PLATE_SCREW_U)])
    h = [cyl_x(u, T.Z_PLATE, x0 - 1, x0 + 15, T.M3_TAP) for u in (-T.PLATE_SCREW_U, T.PLATE_SCREW_U)]
    for u in (-T.BRK_BOLT_U, T.BRK_BOLT_U):
        for z in T.BRK_BOLT_Z:
            h.append(cyl_x(u, z, x1 - 5, x1 + 1, M4))
    return p - union(h)


def stand(side):
    y0, y1 = sorted((side * STAND_Y0, side * STAND_Y1))
    yo0, yo1 = sorted((side * (STAND_Y1 - 4), side * STAND_Y1))   # 바깥면에 붙은 보강 (옆으로 눕혀 출력할 때 바닥에 닿음)
    up = box(X_BB, X_BB + 5, y0, y1, 0, 330)
    foot = box(X_B - 60, X_BB + T.FOOT_BACK, y0, y1, 0, 5)
    pad = box(X_BB + T.FOOT_BACK - 50, X_BB + T.FOOT_BACK, y0, y1, 0, 10)
    gus_r = prism_y([(X_BB + 4.99, 4.9), (X_BB + 110, 4.9), (X_BB + 4.99, 160)], yo0, yo1)
    gus_f = prism_y([(X_BB + 0.01, 4.9), (X_B - 55, 4.9), (X_BB + 0.01, 50)], yo0, yo1)
    p = union([up, foot, pad, gus_r, gus_f])
    h = [slot_x(side * STAND_BOLT_Y, z - 15, z + 15) for z in STAND_BOLT_Z]
    return p - union(h)


def slot_x(y, z0, z1):
    x0, x1 = X_BB - 1, X_BB + 6
    return cyl_x(y, z0, x0, x1, M4) + cyl_x(y, z1, x0, x1, M4) + box(x0, x1, y - M4 / 2, y + M4 / 2, z0, z1)


def main():
    OUT.mkdir(exist_ok=True)
    parts = {
        "11_panel_simple_screen_left": panel_half(+1),
        "12_panel_simple_screen_right": panel_half(-1),
        "13_splice_simple": splice(),
        "14_bracket_simple": bracket(),
        "04_qr_plate_69x15": T.qr_plate(),
        "15_stand_simple_screen_left": stand(+1),
        "16_stand_simple_screen_right": stand(-1),
    }
    for name, m in parts.items():
        t = T.to_trimesh(m)
        t.export(OUT / f"{name}.stl")
        b = t.bounds
        print(f"{name:30s} watertight={t.is_watertight} vol={t.volume/1000:6.1f} cm3 "
              f"size {b[1][0]-b[0][0]:.1f} x {b[1][1]-b[0][1]:.1f} x {b[1][2]-b[0][2]:.1f}")
    names = list(parts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = (parts[names[i]] ^ parts[names[j]]).volume()
            if v > 1.0:
                print(f"  겹침 {names[i]} x {names[j]}: {v:.1f} mm3")
    return parts


if __name__ == "__main__":
    main()

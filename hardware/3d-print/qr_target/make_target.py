#!/usr/bin/env python3
"""SEMES 현장 재현 타깃 (QR 플레이트 + 배경) STL 생성기.

좌표 (조립 상태, 단위 mm)
  X : 렌즈 앞 끝 = 0, 벽(타깃) 쪽이 +
  Y : 카메라에서 타깃을 볼 때 왼쪽이 +  (화면 오른쪽 = -Y)
  Z : 책상 윗면 = 0, 위가 +

치수 근거 (2026-11-06, SEMES center.bmp / T-1.00.bmp 분석)
  - 플레이트 면 배율 14.35 px/mm (QR 13 mm 공식값), 플레이트 69.2 x 15.0 mm
  - SEMES fx ~ 4450 px, fy = fx*16/15 -> 광학 중심 ~ 렌즈끝 뒤 2.2 mm
  - WD 307.8 mm (렌즈끝 ~ 플레이트, 멘토 공식)
  - 배경 깊이: 김성현 모델 계수 (배경 Y 감도 11.04 / 플레이트 14.14) -> 1.28배 -> 플레이트 뒤 88 mm
  - 하단바: 벽 앞 29 mm (멘토), 윗면 = 화면 y 1788 px
  - 위 가로선: 화면 y 716 px (판금 턱), 이혜원 line_protocol upper
  - 배경 나사 4개: 화면 (960,1352) (2867,1357)=W2 (1010,1677)=W3 (2820,1684)
"""
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType, Manifold

# ----------------------------------------------------------------- 주요 변수
LENS_H = 251.0          # 렌즈 중심 높이 (책상 윗면 기준). 3D 도면 값, 조립 후 실측해서 고침
WD = 307.8              # 렌즈끝 -> QR 플레이트 앞면
BG_DEPTH = 88.0         # 플레이트 앞면 -> 배경판 앞면
PUPIL = 2.2             # 광학 중심이 렌즈끝 뒤로 들어간 거리 (fx 4450 기준 추정)
FX, FY = 4450.0, 4450.0 * 16 / 15
CX, CY = 1919.5, 1079.5

PLATE_W, PLATE_H, PLATE_T, PLATE_R = 69.0, 15.0, 3.0, 3.0
PLATE_SCREW_U = 29.3    # 플레이트 고정 나사 (중심에서 좌우)

BRK_W = 69.0            # 플레이트 받침 앞면 폭 (SEMES 브래킷 실측 68.6)
BRK_TOP, BRK_BOT = 16.2, 42.4   # 플레이트 중심 기준 위/아래 (SEMES 실측)

PANEL_W = 400.0         # 배경판 전체 폭 (좌우 두 조각)
PANEL_Z0, PANEL_Z1 = 140.0, 355.0
PANEL_T = 5.0
BAR_DEPTH = 29.0        # 하단바 돌출 (멘토)
RIB_OUT = 6.0           # 위 가로선 턱 돌출
FOOT_BACK = 160.0       # 스탠드 발이 배경판 뒷면에서 뒤로 나가는 길이 (뒤 끝을 책상 모서리에 맞춤)

M3, M3_TAP, M4 = 3.4, 2.6, 4.4
SEG = 48

OUT = Path(__file__).resolve().parent / "stl_full"     # 전부 출력 버전 (채택 안 함, .gitignore)

# ----------------------------------------------------------------- 유도 값
X_P = WD                    # 플레이트 앞면
X_B = WD + BG_DEPTH         # 배경판 앞면
X_BB = X_B + PANEL_T        # 배경판 뒷면


def px_to_world(px, py, x_plane):
    """SEMES 화면 좌표 -> 깊이 x_plane 평면 위의 (Y, Z)."""
    zc = x_plane + PUPIL
    u = (px - CX) * zc / FX          # 화면 오른쪽 +
    v = (py - CY) * zc / FY          # 화면 아래 +
    return -u, LENS_H - v


Z_UPPER_EDGE = px_to_world(0, 716, X_B)[1]       # 위 가로선 (턱 윗모서리)
Z_BAR_TOP = px_to_world(0, 1788, X_B)[1]         # 하단바 윗면 (벽과 만나는 모서리)
BG_SCREWS = [px_to_world(*p, X_B) for p in [(960, 1352), (2867, 1357), (1010, 1677), (2820, 1684)]]
Z_PLATE = LENS_H            # 플레이트 중심 높이 (SEMES는 축보다 0.75 mm 위, 무시)

# ----------------------------------------------------------------- 형상 도우미


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_x(y, z, x0, x1, d):
    return Manifold.cylinder(x1 - x0, d / 2, circular_segments=SEG).rotate([0, 90, 0]).translate([x0, y, z])


def cyl_y(x, z, y0, y1, d):
    return Manifold.cylinder(y1 - y0, d / 2, circular_segments=SEG).rotate([-90, 0, 0]).translate([x, y0, z])


def slot_y(x, z0, z1, y0, y1, d):
    """Y 방향으로 뚫린 세로 장공 (z0~z1 중심선)."""
    return cyl_y(x, z0, y0, y1, d) + cyl_y(x, z1, y0, y1, d) + box(x - d / 2, x + d / 2, y0, y1, z0, z1)


def prism_y(pts_xz, y0, y1):
    """XZ 평면 다각형을 Y 방향으로 y0~y1 두께로 세움."""
    pts = [(x, z) for x, z in pts_xz]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    if area < 0:
        pts = pts[::-1]          # CrossSection 은 반시계 방향만 채움
    cs = CrossSection([pts])
    # (a, b, c) -> (X=a, Y=y1-c, Z=b)  (회전 행렬, det=+1)
    return cs.extrude(y1 - y0).transform(np.array([[1, 0, 0, 0], [0, 0, -1, y1], [0, 1, 0, 0]], float))


def union(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


# ----------------------------------------------------------------- 부품
def panel_half(side):
    """side = +1 : Y+ 쪽 (화면 왼쪽), -1 : Y- 쪽 (화면 오른쪽)."""
    y_in, y_out = 0.0, side * PANEL_W / 2
    ylo, yhi = min(y_in, y_out), max(y_in, y_out)
    parts = [box(X_B, X_BB, ylo, yhi, PANEL_Z0, PANEL_Z1)]
    # 하단바 (벽 앞 29 mm, 윗면 Z_BAR_TOP)
    parts.append(box(X_B - BAR_DEPTH, X_B + 0.01, ylo, yhi, PANEL_Z0, Z_BAR_TOP))
    # 위 가로선 턱: 윗면 45도 모따기 (모서리 = Z_UPPER_EDGE), 앞면 2, 아랫면 45도
    zt = Z_UPPER_EDGE
    rib = [(X_B + 0.01, zt), (X_B - RIB_OUT, zt - RIB_OUT), (X_B - RIB_OUT, zt - RIB_OUT - 2),
           (X_B + 0.01, zt - 2 * RIB_OUT - 2)]
    parts.append(prism_y(rib, ylo, yhi))
    # 뒷면 보강대: 가장자리(스탠드 연결), 가운데 이음, 중간
    def vrib(yc, t, depth):
        return box(X_BB - 0.01, X_BB + depth, yc - t / 2, yc + t / 2, PANEL_Z0, PANEL_Z1)
    parts.append(vrib(side * (PANEL_W / 2 - 11), 6, 16))      # 스탠드 연결 (Y ±186..±192)
    parts.append(vrib(side * 3, 6, 16))                         # 가운데 이음 (0..±6)
    parts.append(vrib(side * 125, 4, 12))
    parts.append(box(X_BB - 0.01, X_BB + 12, ylo, yhi, PANEL_Z1 - 4, PANEL_Z1))   # 위 가로 보강
    p = union(parts)
    holes = []
    # 배경 나사 (M3 버튼 볼트를 앞에서 꽂고 뒤에서 너트)
    for (y, z) in BG_SCREWS:
        if y * side > 0:
            holes.append(cyl_x(y, z, X_B - 1, X_BB + 1, M3))
    # 가운데 이음 볼트 3개 (Y 방향)
    for z in (PANEL_Z0 + 25, Z_BAR_TOP + 45, PANEL_Z1 - 25):
        holes.append(cyl_y(X_BB + 9, z, -7, 7, M4))
    # 스탠드 연결 볼트 2개
    ys = side * (PANEL_W / 2 - 11)
    for z in (200.0, 300.0):
        holes.append(cyl_y(X_BB + 9, z, ys - 4, ys + 4, M4))
    # 플레이트 받침 고정 볼트 (받침 날개 구멍과 같은 위치)
    for z in BRK_BOLT_Z:
        holes.append(cyl_x(side * BRK_BOLT_U, z, X_B - 1, X_BB + 1, M4))
    return p - union(holes)


BRK_FLANGE = 14.0
BRK_BOLT_U = BRK_W / 2 + BRK_FLANGE / 2
BRK_BOLT_Z = (Z_PLATE - BRK_BOT + 10, Z_PLATE + BRK_TOP - 10)


def bracket():
    """플레이트 받침: 배경판 앞면 -> 플레이트 뒷면, 뒤쪽 날개로 배경판에 M4 4개."""
    x0, x1 = X_P + PLATE_T, X_B
    z0, z1 = Z_PLATE - BRK_BOT, Z_PLATE + BRK_TOP
    body = box(x0, x1, -BRK_W / 2, BRK_W / 2, z0, z1)
    flange = box(x1 - 6, x1, -BRK_W / 2 - BRK_FLANGE, BRK_W / 2 + BRK_FLANGE, z0, z1)
    p = body + flange
    holes = [cyl_x(u, Z_PLATE, x0 - 1, x0 + 15, M3_TAP) for u in (-PLATE_SCREW_U, PLATE_SCREW_U)]
    for u in (-BRK_BOLT_U, BRK_BOLT_U):
        for z in BRK_BOLT_Z:
            holes.append(cyl_x(u, z, x1 - 7, x1 + 1, M4))
    return p - union(holes)


def qr_plate():
    sq = CrossSection.square([PLATE_W - 2 * PLATE_R, PLATE_H - 2 * PLATE_R], center=True)
    rr = sq.offset(PLATE_R, JoinType.Round, circular_segments=SEG)
    holes = CrossSection.circle(M3 / 2, SEG).translate([PLATE_SCREW_U, 0]) + \
        CrossSection.circle(M3 / 2, SEG).translate([-PLATE_SCREW_U, 0])
    flat = (rr - holes).extrude(PLATE_T)          # XY 평면, 두께 Z
    # 조립 위치: (a, b, c) -> (X=X_P+c, Y=a, Z=Z_PLATE+b). 앞면 X_P
    return flat.transform(np.array([[0, 0, 1, X_P], [1, 0, 0, 0], [0, 1, 0, Z_PLATE]], float))


def stand(side):
    """책상 발 + 기둥. 배경판 가장자리 보강대 바깥에 붙고, C클램프는 발 뒤쪽을 물음."""
    y_in = side * (PANEL_W / 2 - 8)          # 192
    y_mid = side * (PANEL_W / 2)             # 200 (기둥 바깥면)
    y_out = side * (PANEL_W / 2 + 32)        # 232 (발 바깥면)
    lo = lambda a, b: (min(a, b), max(a, b))
    up_y = lo(y_in, y_mid)
    foot_y = lo(y_in, y_out)
    X_UP0, X_UP1 = X_BB, X_BB + 40
    FOOT_X0, FOOT_X1 = X_B - 55, X_BB + FOOT_BACK
    parts = [
        box(X_UP0, X_UP1, *up_y, 0, 325),                       # 기둥
        box(FOOT_X0, FOOT_X1, *foot_y, 0, 10),                  # 발
        box(FOOT_X1 - 50, FOOT_X1, *foot_y, 0, 14),             # 클램프 자리 (윗면 평평)
        prism_y([(X_UP1 - 0.01, 9.9), (X_UP1 + 100, 9.9), (X_UP1 - 0.01, 160)], *up_y),   # 뒤 삼각 보강
        prism_y([(X_UP0 + 0.01, 9.9), (FOOT_X0 + 5, 9.9), (X_UP0 + 0.01, 60)], *up_y),    # 앞 삼각 보강
    ]
    p = union(parts)
    holes = [slot_y(X_BB + 9, z - 15, z + 15, up_y[0] - 1, up_y[1] + 1, M4) for z in (200.0, 300.0)]
    return p - union(holes)


# ----------------------------------------------------------------- 출력
def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3], faces=np.asarray(mesh.tri_verts), process=False)


def main():
    OUT.mkdir(exist_ok=True)
    parts = {
        "01_panel_screen_left": panel_half(+1),
        "02_panel_screen_right": panel_half(-1),
        "03_bracket": bracket(),
        "04_qr_plate_69x15": qr_plate(),
        "05_stand_screen_left": stand(+1),
        "06_stand_screen_right": stand(-1),
    }
    meshes = {}
    for name, m in parts.items():
        t = to_trimesh(m)
        meshes[name] = t
        t.export(OUT / f"{name}.stl")
        b = t.bounds
        print(f"{name:26s} watertight={t.is_watertight}  vol={t.volume/1000:7.1f} cm3  "
              f"size {b[1][0]-b[0][0]:.1f} x {b[1][1]-b[0][1]:.1f} x {b[1][2]-b[0][2]:.1f}")
    trimesh.util.concatenate(list(meshes.values())).export(OUT / "00_assembly_preview.stl")
    # 겹침 검사 (부품끼리 부피 겹침)
    names = list(parts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = (parts[names[i]] ^ parts[names[j]]).volume()
            if v > 1.0:
                print(f"  겹침 {names[i]} x {names[j]}: {v:.1f} mm3")
    print(f"위 가로선 Z={Z_UPPER_EDGE:.1f}, 하단바 윗면 Z={Z_BAR_TOP:.1f}, 배경 나사 (Y,Z)=",
          [(round(y, 1), round(z, 1)) for y, z in BG_SCREWS])
    return parts


if __name__ == "__main__":
    main()

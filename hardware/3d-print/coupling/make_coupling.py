#!/usr/bin/env python3
"""플랜지 커플링 내경 10 + 키홈 (PSH42 출력축용, 3D 출력).

산 금속 커플링(디바이스마트 MFC 내경 10)은 둥근 구멍이라 PSH42 출력축의 키에 걸려
들어가지 않는다. 같은 크기로 출력하고 키홈을 낸다.

PSH42 출력축 (09_하모닉모터_PSH42_모형_v2.stl 실측)
  - 감속기 윗면 위 보스 Ø25 x 1.0
  - 축 Ø10, 보스 위로 20.95 (끝 0.3 모따기), 끝에 M3 탭 구멍
  - 키 폭 3, 축 표면에서 1.5 돌출(중심에서 6.5), 길이 10, 양 끝 둥금
    보스 위 7.45 ~ 17.45 (축 끝에서 3.5 안쪽)

커플링 높이 (3D 도면 아티팩트와 같음)
  - 플랜지 윗면 = 카메라 각재 아랫면 = 책상에서 186
  - 전체 22 (허브 18 + 플랜지 4) → 허브 아랫면이 보스 위 약 2.45, 축은 18.5 들어감

좌표: 허브 아랫면 z = 0, 플랜지 윗면 z = H. 키 방향 = +Y.
출력 방향: 플랜지 윗면을 바닥에 두도록 뒤집어서 저장 (서포트 없음, 구멍이 세로라 둥글게 나옴).

치수는 실물 커플링이 없어 판매 페이지 기준으로 잡은 값이다(허브 Ø18은 판매 페이지,
플랜지 Ø32는 같은 계열 10 mm 제품 값). 실물을 캘리퍼로 재서 아래 값을 고치면 된다.
"""
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import Manifold

OUT = Path(__file__).resolve().parent.parent / "stl"
NAME = "24_플랜지커플링_D10_키홈"
SEG = 96

# ---- 축 (모터 실측) ----
SHAFT_D = 10.0
KEY_W = 3.0
KEY_TOP_R = 6.5            # 축 중심 → 키 윗면
SHAFT_ABOVE_BOSS = 20.95
KEY_Z_ABOVE_BOSS = (7.45, 17.45)

# ---- 커플링 (금속 커플링 크기, 실물 재서 고칠 값) ----
FLANGE_D = 32.0
FLANGE_T = 4.0
HUB_D = 18.0
H = 22.0                   # 전체 높이 → 플랜지 윗면이 책상 186
HOLE_PCD = 26.0            # 플랜지 구멍 중심원 (각재 폭 30 안: 대각선 배치 → ±9.2)
FILLET = 1.0               # 허브-플랜지 이음 모따기
HOLE_D = 3.4               # 태핑나사 M3 통과
HOLE_ANGLES = (45, 135, 225, 315)

# ---- 출력 공차 (첫 출력 후 조정) ----
BORE_CLR = 0.25            # 구멍 지름 여유 → Ø10.25
KEY_W_CLR = 0.2            # 키홈 폭 3.2: 좁을수록 Θ 유격이 적다. 안 들어가면 줄로 다듬기
KEY_TOP_CLR = 0.35         # 키 윗면과 홈 바닥 사이
BORE_DEPTH = 19.5          # 막힌 구멍. 축은 18.5 들어감 → 1 mm 여유, 위에 2.5 mm 막힘
ENTRY_CHAMFER = 0.6

# ---- 무두볼트 ----
# 금속 커플링에 딸려 온 무두볼트를 플라스틱에 바로 돌려 넣는다 (M4 → Ø3.4, M3이면 2.6).
# 90° 위치 볼트가 축을 옆으로 밀어 키를 홈 한쪽 벽에 붙인다 → 키홈 폭 유격(Θ 흔들림) 제거.
SET_D = 3.4
SET_Z = 10.0
SET_ANGLES = (90, 180)     # 키(+Y, 0°) 기준


def cyl(d, z0, z1, x=0.0, y=0.0, seg=SEG):
    return Manifold.cylinder(z1 - z0, d / 2, circular_segments=seg).translate([x, y, z0])


def cone(d0, d1, z0, z1):
    return Manifold.cylinder(z1 - z0, d0 / 2, d1 / 2, circular_segments=SEG).translate([0, 0, z0])


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def radial_hole(d, z, angle_deg, r0, r1):
    """축 바깥쪽으로 향하는 구멍. angle 0 = +Y, 시계 반대 방향."""
    m = Manifold.cylinder(r1 - r0, d / 2, circular_segments=32).rotate([-90, 0, 0]).translate([0, r0, z])
    return m.rotate([0, 0, angle_deg])


def coupling():
    hub_top = H - FLANGE_T
    body = cyl(HUB_D, 0, H) + cyl(FLANGE_D, hub_top, H)
    # 허브-플랜지 이음 보강: 플랜지 쪽이 넓은 45° 모따기 (출력 시 바닥 쪽으로 넓어져 서포트 불필요).
    # 플랜지 아랫면에 닿는 나사 머리(Ø6, 중심 r13 → r10~16)를 피하도록 r10까지만.
    body = body + cone(HUB_D, HUB_D + 2 * FILLET, hub_top - FILLET, hub_top)

    bore_d = SHAFT_D + BORE_CLR
    cut = cyl(bore_d, -1, BORE_DEPTH)
    cut = cut + cone(bore_d + 2 * ENTRY_CHAMFER, bore_d, -0.01, ENTRY_CHAMFER)
    kw = KEY_W + KEY_W_CLR
    cut = cut + box(-kw / 2, kw / 2, 0, KEY_TOP_R + KEY_TOP_CLR, -1, BORE_DEPTH)
    for a in SET_ANGLES:
        cut = cut + radial_hole(SET_D, SET_Z, a, 0, HUB_D / 2 + 1)
    r = HOLE_PCD / 2
    for a in HOLE_ANGLES:
        t = np.radians(a)
        cut = cut + cyl(HOLE_D, hub_top - 3, H + 1, r * np.cos(t), r * np.sin(t), 32)
    return body - cut


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3], faces=np.asarray(mesh.tri_verts))


def main():
    OUT.mkdir(exist_ok=True)
    c = coupling()
    printed = c.rotate([180, 0, 0]).translate([0, 0, H])               # 플랜지 윗면이 바닥
    t = to_trimesh(printed)
    t.export(OUT / f"{NAME}.stl")
    b = t.bounds
    gap = SHAFT_ABOVE_BOSS - (BORE_DEPTH - 1.0)
    k0, k1 = (z - gap for z in KEY_Z_ABOVE_BOSS)
    wall_key = HUB_D / 2 - (KEY_TOP_R + KEY_TOP_CLR)
    print(f"{NAME}: watertight={t.is_watertight} vol={t.volume / 1000:.2f} cm3 "
          f"size {b[1][0] - b[0][0]:.1f} x {b[1][1] - b[0][1]:.1f} x {b[1][2] - b[0][2]:.1f}")
    print(f"  보스-허브 간격 {gap:.2f}, 축 삽입 {SHAFT_ABOVE_BOSS - gap:.2f}, 키는 허브 아랫면 위 {k0:.2f}~{k1:.2f}, "
          f"키홈 쪽 벽 {wall_key:.2f}")
    return c, gap


if __name__ == "__main__":
    main()

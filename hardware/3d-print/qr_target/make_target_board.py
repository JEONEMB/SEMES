#!/usr/bin/env python3
"""SEMES 재현 타깃 — 배경판 구매 버전 (출력 최소).

배경판: 흰색 포맥스(PVC 발포판) 5 mm, 400 x 360 (구매, 재단 주문)
출력:   플레이트 받침, QR 플레이트, 위 턱 띠, 아래 단 띠, 옆 받침 2개
도면:   판 구멍·띠 위치 1:1 종이 도면 (A4 4장) -> drill_template_A4x4.pdf

특징 위치(SEMES 화면 좌표)는 make_target.py 와 같다. 좌표·변수 설명도 make_target.py 참고.
렌즈 높이(LENS_H)를 실측한 뒤 다시 돌리면 구멍·띠 높이가 따라 바뀐다.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import make_target as T
import make_target_simple as S
from make_target import box, cyl_x, prism_y, union, M3, M4

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "stl"                      # hardware/3d-print/stl (출력용)
PDF = HERE / "drill_template_A4x4.pdf"

BOARD_T = 5.0                 # 포맥스 두께
BOARD_W, BOARD_H = 400.0, 360.0
FOOT_T = 6.0                  # 옆 받침 발 두께 = 판 아래 끝 높이
BOARD_Z0 = FOOT_T
X_B = T.X_B                   # 판 앞면 (플레이트 앞면 + 88)
X_PB = X_B + BOARD_T          # 판 뒷면
SIDE_Y0, SIDE_Y1 = 190.0, 198.0   # 옆 받침 두께 방향 (판 양 끝, 화면 밖)
SIDE_BOLT_Y = 194.0
SIDE_BOLT_Z = (60.0, 220.0)
FOOT_BACK = 140.0             # 판 뒷면에서 뒤로 (끝을 책상 모서리에 맞춰 클램프)
# 옆 받침만 키워 판 전체(판·띠·플레이트 받침·QR)를 책상에서 이만큼 더 띄운다 (mm).
# 판 좌표(구멍·띠·도면)는 그대로이고, 옆 받침의 책상 닿는 면만 -LIFT 로 내려간다.
# 2026-11-12 실측: 원래 받침 위 QR 이 SEMES center.bmp 보다 화면 아래로 395 px(≈25.6 mm) 내려가 있었음.
LIFT = 25.5
STEP_DEPTH, STEP_H = S.STEP_DEPTH, S.STEP_H

# 판 구멍 (Y, Z, 지름, 이름)
HOLES = ([(y, z, M3, "배경 나사 M3") for (y, z) in T.BG_SCREWS]
         + [(s * T.BRK_BOLT_U, z, M4, "받침 M4") for s in (-1, 1) for z in T.BRK_BOLT_Z]
         + [(s * SIDE_BOLT_Y, z, M4, "옆 받침 M4") for s in (-1, 1) for z in SIDE_BOLT_Z])
UPPER_TOP = T.Z_UPPER_EDGE                     # 위 턱 띠 윗모서리
UPPER_BOT = T.Z_UPPER_EDGE - 2 * T.RIB_OUT - 2
LOWER_TOP = T.Z_BAR_TOP                        # 아래 단 띠 윗면
LOWER_BOT = T.Z_BAR_TOP - STEP_H


def board():
    """구매품 (출력하지 않음). 그림·검사용."""
    p = box(X_B, X_PB, -BOARD_W / 2, BOARD_W / 2, BOARD_Z0, BOARD_Z0 + BOARD_H)
    return p - union([cyl_x(y, z, X_B - 1, X_PB + 1, d) for (y, z, d, _) in HOLES])


def strip_upper():
    zt, r = UPPER_TOP, T.RIB_OUT
    return prism_y([(X_B, zt), (X_B - r, zt - r), (X_B - r, zt - r - 2), (X_B, zt - 2 * r - 2)],
                   -BOARD_W / 2 + 0.5, BOARD_W / 2 - 0.5)


def strip_lower():
    return box(X_B - STEP_DEPTH, X_B, -BOARD_W / 2 + 0.5, BOARD_W / 2 - 0.5, LOWER_BOT, LOWER_TOP)


def side_bracket(side):
    y0, y1 = sorted((side * SIDE_Y0, side * SIDE_Y1))
    zd = -LIFT                                                                  # 책상 윗면
    prof = [
        box(X_B - 40, X_PB + FOOT_BACK, y0, y1, zd, zd + FOOT_T),              # 발
        box(X_B, X_PB, y0, y1, zd, BOARD_Z0),                                   # 판 자리 (판이 이 위에 섬)
        box(X_B - 8, X_B, y0, y1, zd, 24),                                      # 앞 턱 (판 아래 앞을 잡음)
        box(X_PB, X_PB + 8, y0, y1, zd, 235),                                   # 기둥 (판 뒤)
        prism_y([(X_PB + 7.9, 200), (X_PB + 7.9, 216), (X_PB + 140, zd + FOOT_T - 0.1), (X_PB + 122, zd + FOOT_T - 0.1)], y0, y1),   # 버팀대
        box(X_PB + FOOT_BACK - 50, X_PB + FOOT_BACK, y0, y1, zd, zd + 10),      # 클램프 자리
    ]
    p = union(prof)
    return p - union([cyl_x(side * SIDE_BOLT_Y, z, X_PB - 1, X_PB + 9, M4) for z in SIDE_BOLT_Z])


# ----------------------------------------------------------------- 1:1 구멍 도면
DPI = 300
MM = DPI / 25.4
A4 = (297.0, 210.0)
TILE_W, TILE_H = 200.0, 180.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"   # Pi에 한글 폰트가 없어 도면 글자는 영어
EN = {"배경 나사 M3": "BG screw M3", "받침 M4": "bracket M4", "옆 받침 M4": "side bracket M4"}


def front_xy(y, z):
    """판 앞(카메라 쪽)에서 본 좌표: 왼쪽 아래 모서리 = (0, 0), 오른쪽·위가 +."""
    return -y + BOARD_W / 2, z - BOARD_Z0


def drill_template():
    pages = []
    f_s = ImageFont.truetype(FONT, int(2.6 * MM))
    f_b = ImageFont.truetype(FONT, int(3.6 * MM))
    axis_h = T.LENS_H - BOARD_Z0
    for row in (1, 0):              # 위쪽 줄 먼저
        for col in (0, 1):
            img = Image.new("L", (int(A4[0] * MM), int(A4[1] * MM)), 255)
            d = ImageDraw.Draw(img)
            ox, oy = (A4[0] - TILE_W) / 2, (A4[1] - TILE_H) / 2      # 종이에서 타일 위치 (mm)
            fx0, fz0 = col * TILE_W, row * TILE_H                      # 판 좌표에서 타일 시작

            def P(x, h):        # 판 좌표 -> 종이 픽셀 (h 위가 +)
                return ((ox + x - fx0) * MM, (oy + TILE_H - (h - fz0)) * MM)

            def inside(x, h, m=0):
                return fx0 - m <= x <= fx0 + TILE_W + m and fz0 - m <= h <= fz0 + TILE_H + m

            def hline(h, x0, x1, w=2, fill=0):
                a, b = max(x0, fx0), min(x1, fx0 + TILE_W)
                if a < b and fz0 <= h <= fz0 + TILE_H:
                    d.line([P(a, h), P(b, h)], fill=fill, width=w)

            # 타일 테두리 (자르는 선)와 판 바깥선
            d.rectangle([P(fx0, fz0 + TILE_H), P(fx0 + TILE_W, fz0)], outline=150, width=2)
            for x in (0, BOARD_W):
                if fx0 <= x <= fx0 + TILE_W:
                    d.line([P(x, fz0), P(x, fz0 + TILE_H)], fill=0, width=6)
            for h in (0, BOARD_H):
                hline(h, 0, BOARD_W, w=6)
            # 10 mm 격자
            for gx in range(int(fx0), int(fx0 + TILE_W) + 1, 10):
                d.line([P(gx, fz0), P(gx, fz0 + TILE_H)], fill=225 if gx % 50 else 190, width=1)
            for gh in range(int(fz0), int(fz0 + TILE_H) + 1, 10):
                d.line([P(fx0, gh), P(fx0 + TILE_W, gh)], fill=225 if gh % 50 else 190, width=1)
            # 가운데선, 렌즈 축 높이
            if fx0 <= 200 <= fx0 + TILE_W:
                d.line([P(200, fz0), P(200, fz0 + TILE_H)], fill=90, width=2)
            hline(axis_h, 0, BOARD_W, w=2, fill=90)
            if inside(fx0 + 3, axis_h):
                d.text(P(fx0 + 3, axis_h + 3.5), f"lens axis height ({axis_h:.1f} from board bottom)", font=f_s, fill=60)
            # 띠 붙일 자리
            for (top, bot, name) in [(UPPER_TOP - BOARD_Z0, UPPER_BOT - BOARD_Z0, "UPPER strip: top edge on this line"),
                                     (LOWER_TOP - BOARD_Z0, LOWER_BOT - BOARD_Z0, "LOWER strip: top face on this line")]:
                a0, a1 = max(0, fx0), min(BOARD_W, fx0 + TILE_W)
                if top >= fz0 and bot <= fz0 + TILE_H:
                    d.rectangle([P(a0, min(top, fz0 + TILE_H)), P(a1, max(bot, fz0))], fill=232)
                    hline(top, 0, BOARD_W, w=4)
                    if inside(fx0 + 3, top):
                        d.text(P(fx0 + 3, top + 4.2), name + f"  {top:.1f}", font=f_s, fill=0)
            # 받침 자리
            bx0, bx1 = front_xy(T.BRK_W / 2, 0)[0], front_xy(-T.BRK_W / 2, 0)[0]
            bz0, bz1 = T.Z_PLATE - T.BRK_BOT - BOARD_Z0, T.Z_PLATE + T.BRK_TOP - BOARD_Z0
            fl0, fl1 = bx0 - T.BRK_FLANGE, bx1 + T.BRK_FLANGE
            if fl1 >= fx0 and fl0 <= fx0 + TILE_W:
                a, b = max(fl0, fx0), min(fl1, fx0 + TILE_W)
                d.rectangle([P(a, bz1), P(b, bz0)], outline=0, width=2)
                if inside(max(a, fx0) + 2, bz1):
                    d.text(P(max(a, fx0) + 2, bz1 - 1.5), "plate bracket flange", font=f_s, fill=0)
            # 구멍
            for (y, z, dia, name) in HOLES:
                x, h = front_xy(y, z)
                if not inside(x, h):
                    continue
                cx, cy = P(x, h)
                r = dia / 2 * MM
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=3)
                arm = 4 * MM
                d.line([cx - arm, cy, cx + arm, cy], fill=0, width=2)
                d.line([cx, cy - arm, cx, cy + arm], fill=0, width=2)
                d.text((cx + 3 * MM, cy - 6 * MM), f"{EN[name]}  drill {dia}\n({x:.1f}, {h:.1f})", font=f_s, fill=0)
            # 타일 이름, 이어 붙이는 방법, 100 mm 자
            tag = f"{'TOP' if row else 'BOTTOM'}-{'LEFT' if col == 0 else 'RIGHT'} ({(1 - row) * 2 + col + 1}/4)"
            d.text((ox * MM, 3 * MM), f"SEMES target board drill template - FRONT view (camera side) - print 100% - {tag}",
                   font=f_b, fill=0)
            d.text((ox * MM, (oy + TILE_H + 2) * MM),
                   f"Board 400 x 360 PVC foam 5 mm. (x, y) = mm from board bottom-left corner. Cut on grey border, tape 4 sheets together.",
                   font=f_s, fill=0)
            rx = 8 * MM
            d.line([rx, (oy + 20) * MM, rx, (oy + 120) * MM], fill=0, width=4)
            for k in range(11):
                yk = (oy + 20 + k * 10) * MM
                d.line([rx, yk, rx + (3 if k % 5 else 6) * MM, yk], fill=0, width=3)
            d.text((rx + 2 * MM, (oy + 122) * MM), "100 mm", font=f_s, fill=0)
            pages.append(img)
    pages[0].save(PDF, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    return PDF


def main():
    OUT.mkdir(exist_ok=True)
    parts = {
        "17_QR플레이트_69x15": T.qr_plate(),
        "20_플레이트받침": S.bracket(),
        "21_옆받침_왼쪽": side_bracket(+1),          # 카메라에서 볼 때 왼쪽
        "21_옆받침_오른쪽": side_bracket(-1),
        "23_위턱띠_400": strip_upper(),
        "23_아래단띠_400": strip_lower(),
    }
    for name, m in parts.items():
        t = T.to_trimesh(m)
        t.export(OUT / f"{name}.stl")
        b = t.bounds
        print(f"{name:30s} watertight={t.is_watertight} vol={t.volume/1000:5.1f} cm3 "
              f"size {b[1][0]-b[0][0]:.1f} x {b[1][1]-b[0][1]:.1f} x {b[1][2]-b[0][2]:.1f}")
    bd = board()
    T.to_trimesh(bd).export(HERE / "19_배경판_포맥스_400x360x5_구매품.stl")   # 출력하지 않음 (확인용)
    allp = dict(parts, board=bd)
    names = list(allp)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = (allp[names[i]] ^ allp[names[j]]).volume()
            if v > 1.0:
                print(f"  겹침 {names[i]} x {names[j]}: {v:.1f} mm3")
    print("구멍 (판 앞에서 본 좌표, 왼쪽 아래 모서리 기준 mm):")
    for (y, z, dia, name) in HOLES:
        x, h = front_xy(y, z)
        print(f"  {name:10s} Ø{dia}  ({x:6.1f}, {h:6.1f})")
    print(f"위 턱 띠 윗모서리 {UPPER_TOP - BOARD_Z0:.1f}, 아래 단 띠 윗면 {LOWER_TOP - BOARD_Z0:.1f}, 렌즈 축 {T.LENS_H - BOARD_Z0:.1f}")
    print("도면:", drill_template())
    return allp


if __name__ == "__main__":
    main()
